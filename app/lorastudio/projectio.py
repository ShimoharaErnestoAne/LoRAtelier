"""Deleting, exporting (zip) and importing (zip) projects, deleting runs, and clearing the app's logs.

Deletion first renames the folder into data/trash (instant, and the project disappears from the list at
once), then removes it in the background.  Leftovers (a file was still open) are retried at the next start.
Exports are streamed while they are written, so the download starts at once even for several GB.
"""
from __future__ import annotations

import io
import json
import os
import queue
import shutil
import sqlite3
import tempfile
import threading
import time
import zipfile
from pathlib import Path

from . import config
from .store import Project, ProjectManager, _slug

TRASH_DIR = config.DATA_DIR / "trash"
STAGED_LORAS = config.DATA_DIR / "comfy_models" / "loras" / "lorastudio"   # copies made for the comparison images
MANIFEST = "lorastudio_project.json"
FORMAT = 1
# already-compressed files are stored as they are (deflating them only costs time)
STORED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".safetensors", ".ckpt", ".pt", ".bin", ".npz", ".zip"}
APP_LOGS = ("analysis_worker.log", "comfyui.log", "i2t_worker.log", "prefetch_qwen.log", "server_errors.log")


# ====================================================================== deletion
def _trash_dir(path: Path) -> Path:
    """data/trash, or a .trash folder next to the path when it is on another drive (a folder cannot be moved
    across drives in one step; copying a whole project there would take long and need the space twice)."""
    from .linking import same_drive
    return TRASH_DIR if same_drive(path, TRASH_DIR) else path.parent / ".trash"


def _to_trash(path: Path) -> Path:
    trash = _trash_dir(path)
    trash.mkdir(parents=True, exist_ok=True)
    dest = trash / f"{path.name}_{time.strftime('%Y%m%d-%H%M%S')}"
    last = None
    for _ in range(10):   # a thumbnail being sent or a virus scan can hold a file for a moment
        try:
            os.replace(path, dest)
            return dest
        except PermissionError as e:
            last = e
            time.sleep(0.3)
    raise OSError(f"使用中のファイルがあるため移動できません: {last}")


def _remove_later(path: Path):
    threading.Thread(target=shutil.rmtree, args=(path,), kwargs={"ignore_errors": True}, daemon=True).start()


def empty_trash():
    """Remove what an earlier deletion could not (called at start-up, in the background)."""
    for trash in [TRASH_DIR, *(r / ".trash" for r in config.storage_dirs("projects"))]:
        try:
            items = list(trash.iterdir()) if trash.is_dir() else []
        except OSError:
            continue   # e.g. a drive that is not connected
        for d in items:
            shutil.rmtree(d, ignore_errors=True) if d.is_dir() else d.unlink(missing_ok=True)
    # copies left by a move or an import the app was closed during (the original is removed only after its copy
    # is complete and in place, so these are never the only copy)
    for r in config.storage_dirs("projects"):
        try:
            left = [d for d in r.iterdir() if d.is_dir() and (d.name.startswith(".moving_") or d.name == ".importing")]
        except OSError:
            continue
        for d in left:
            shutil.rmtree(d, ignore_errors=True)


def delete_project(projects: ProjectManager, pid: str):
    with projects._lock:
        p = projects._open.pop(pid, None)
        if p is not None:
            p.close()
        path = projects.path_of(pid)
        if path is None:
            raise KeyError(pid)
        try:
            trash = _to_trash(path)
        except OSError:
            if p is not None:   # keep the project usable when it could not be removed
                projects._open[pid] = Project(path)
            raise
    _remove_later(trash)
    staged = STAGED_LORAS / pid
    if staged.exists():
        try:
            _remove_later(_to_trash(staged))
        except OSError:
            pass   # e.g. ComfyUI still holds a staged LoRA: removed with the trash at the next start or left as a copy


def delete_run(project: Project, run_id: int):
    run = project.run(run_id)
    with project.tx() as db:
        db.execute("DELETE FROM checkpoints WHERE run_id=?", (run_id,))
        db.execute("DELETE FROM evals WHERE run_id=?", (run_id,))
        db.execute("DELETE FROM runs WHERE id=?", (run_id,))
        # exports rows stay: the exported LoRA folders are outside the project and are not deleted
    run_dir = Path(run["run_dir"]) if run["run_dir"] else None
    if run_dir and not _inside(run_dir, project.path):
        run_dir = None   # recorded for another folder (an old import): that folder is not this project's to remove
    for d in (run_dir, STAGED_LORAS / project.id / f"r{run_id}"):
        if d and d.is_dir():
            try:
                _remove_later(_to_trash(d))
            except OSError:
                shutil.rmtree(d, ignore_errors=True)


def training_leftovers(project: Project, skip_runs=()) -> dict:
    """What "学習データを破棄" (offered after an export) removes: every run's saved LoRAs except the exported
    ones, the resume states and the training caches (with ComfyUI's copies of those LoRAs).  The material,
    captions, crops, comparison images and the exported LoRAs stay, so the project can be trained again (also for
    another model).  skip_runs: runs still in use (training, comparison images being made)."""
    from . import traincache
    keep = {r["ckpt_path"] for r in project.q("SELECT ckpt_path FROM exports")}
    ckpts, files = [], []
    for run in project.runs():
        run_dir = Path(run["run_dir"]) if run["run_dir"] else None
        if run["id"] in skip_runs or not run_dir or not _inside(run_dir, project.path):
            continue
        for c in project.checkpoints(run["id"]):
            path = Path(c["path"])
            if (c["kind"] == "lora" and c["path"] in keep) or not _inside(path, project.path):
                continue
            ckpts.append(c["id"])
            files += [f for f in path.rglob("*") if f.is_file()] if path.is_dir() else [path]
            if c["kind"] == "lora":
                files.append(STAGED_LORAS / project.id / f"r{run['id']}" / path.name)
        files += traincache.run_files(run_dir)
    if not skip_runs:   # the shared store: a run in use may still be copying from it
        files += traincache.store_files(project.path)
    size = 0
    for f in files:
        try:
            size += f.stat().st_size
        except OSError:
            pass
    return {"checkpoints": ckpts, "files": files, "bytes": size}


def discard_training_leftovers(project: Project, skip_runs=()) -> int:
    """Remove training_leftovers().  A checkpoint row goes only when its file is gone (a file held open stays
    listed and can be removed again later).  Returns the bytes freed."""
    lo = training_leftovers(project, skip_runs)
    paths = {r["id"]: Path(r["path"]) for r in project.q("SELECT id, path FROM checkpoints")}
    freed = 0
    for f in lo["files"]:
        try:
            n = f.stat().st_size
            f.unlink()
            freed += n
        except OSError:
            pass
    gone = []
    for cid in lo["checkpoints"]:
        path = paths.get(cid)
        if path is None:
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            gone.append(cid)
    if not skip_runs:
        shutil.rmtree(project.path / "traincache", ignore_errors=True)
    if gone:
        with project.tx() as db:
            db.executemany("DELETE FROM checkpoints WHERE id=?", [(c,) for c in gone])
    return freed


def _inside(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except OSError:
        return False


def dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


# ====================================================================== app logs
def app_logs() -> list[dict]:
    out = []
    for name in APP_LOGS:
        f = config.DATA_DIR / name
        if f.exists():
            out.append({"name": name, "size": f.stat().st_size})
    return out


def app_log_tail(name: str, lines: int = 300) -> dict:
    """The last lines of one app log, for the settings screen (only names in APP_LOGS can be read)."""
    if name not in APP_LOGS:
        raise ValueError(name)
    f = config.DATA_DIR / name
    if not f.exists():
        return {"file": str(f), "text": ""}
    with open(f, "rb") as fh:
        fh.seek(max(0, f.stat().st_size - 300_000))
        data = fh.read().decode("utf-8", "replace")
    rows = [l for l in data.replace("\r", "\n").split("\n") if l.strip()]
    return {"file": str(f), "text": "\n".join(rows[-lines:])}


def app_logs_zip() -> bytes:
    """All app logs (the last 5MB of each) plus a short environment note, to hand to someone for a bug report."""
    import platform
    import sys
    from . import __version__
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        info = [f"作成日時: {time.strftime('%F %T')}", f"LoRAtelier: {__version__}",
                f"OS: {platform.platform()}", f"Python: {sys.version.split()[0]}"]
        try:
            from .gpu import query_gpus
            info += [f"GPU: {g['name']} {g['vram_total_mb']}MB（ドライバー {g['driver']}）" for g in query_gpus()] or ["GPU: 見つかりません"]
        except Exception as e:   # the note is still useful without the GPU line
            info.append(f"GPU: 取得できませんでした（{type(e).__name__}）")
        z.writestr("環境.txt", "\n".join(info) + "\n")
        for it in app_logs():
            f = config.DATA_DIR / it["name"]
            try:
                with open(f, "rb") as fh:
                    fh.seek(max(0, it["size"] - 5_000_000))
                    z.writestr(it["name"], fh.read())
            except OSError as e:
                z.writestr(it["name"] + ".読み込み失敗.txt", str(e))
    return buf.getvalue()


def clear_app_logs() -> dict:
    """Empty the log files (truncate rather than delete: a running worker may still be writing to them)."""
    done, failed = [], []
    for it in app_logs():
        try:
            with open(config.DATA_DIR / it["name"], "r+b") as f:
                f.truncate(0)
            done.append(it["name"])
        except OSError as e:
            failed.append(f"{it['name']}: {e}")
    return {"cleared": done, "failed": failed}


# ====================================================================== export
class _Pipe(io.RawIOBase):
    """Write end of a zip being streamed: chunks go to a queue read by the HTTP response."""

    def __init__(self):
        self.q: queue.Queue = queue.Queue(maxsize=32)
        self.aborted = False

    def writable(self):
        return True

    def write(self, b):
        if self.aborted:
            raise OSError("ダウンロードが中断されました")
        data = bytes(b)
        while True:
            try:
                self.q.put(data, timeout=1)
                return len(data)
            except queue.Full:
                if self.aborted:
                    raise OSError("ダウンロードが中断されました")


def _db_snapshot(project: Project, include_runs: bool) -> Path:
    """A consistent copy of project.db (the live one may have uncommitted WAL pages)."""
    fd, tmp = tempfile.mkstemp(suffix=".db", dir=config.DATA_DIR)
    os.close(fd)
    dst = sqlite3.connect(tmp)
    with project._lock:
        project.db.backup(dst)
    if not include_runs:
        for t in ("checkpoints", "evals", "exports", "runs"):
            dst.execute(f"DELETE FROM {t}")
        dst.commit()
    dst.execute("PRAGMA journal_mode=DELETE")
    dst.close()
    return Path(tmp)


def _export_files(project: Project, include_runs: bool):
    skip_top = {"project.db", "project.db-wal", "project.db-shm", "incoming"}
    if not include_runs:
        skip_top |= {"runs", "exports"}
    for root, dirs, files in os.walk(project.path):
        rel_root = Path(root).relative_to(project.path)
        if rel_root == Path("."):
            dirs[:] = [d for d in dirs if d not in skip_top]
            files = [f for f in files if f not in skip_top]
        for f in files:
            if f.endswith((".part", ".tmp")):
                continue
            yield Path(root) / f, (rel_root / f).as_posix()


def export_size(project: Project, include_runs: bool) -> int:
    return sum(os.path.getsize(p) for p, _ in _export_files(project, include_runs))


def export_stream(project: Project, include_runs: bool):
    """Generator of zip bytes.  The zip is written in a thread; closing the generator aborts it."""
    pipe = _Pipe()
    DONE = object()
    err: list[Exception] = []

    def produce():
        snap = None
        try:
            snap = _db_snapshot(project, include_runs)
            with zipfile.ZipFile(pipe, "w", allowZip64=True) as zf:
                manifest = {"format": FORMAT, "id": project.id, "name": project.meta("name"), "path": str(project.path),
                            "include_runs": include_runs, "exported": time.strftime("%Y-%m-%d %H:%M:%S")}
                zf.writestr(MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2))
                zf.write(snap, "project.db", zipfile.ZIP_DEFLATED)
                for path, arc in _export_files(project, include_runs):
                    try:
                        zf.write(path, arc, zipfile.ZIP_STORED if path.suffix.lower() in STORED_EXT else zipfile.ZIP_DEFLATED)
                    except FileNotFoundError:
                        pass   # removed while exporting (e.g. a rotated training state)
        except Exception as e:
            err.append(e)
        finally:
            if snap:
                snap.unlink(missing_ok=True)
            while not pipe.aborted:
                try:
                    pipe.q.put(DONE, timeout=1)
                    break
                except queue.Full:
                    pass

    threading.Thread(target=produce, daemon=True, name=f"export-{project.id}").start()
    try:
        while True:
            chunk = pipe.q.get()
            if chunk is DONE:
                break
            yield chunk
        if err:
            raise err[0]   # breaks the download instead of handing over a truncated zip as complete
    finally:
        pipe.aborted = True


# ====================================================================== import
def _rebase(obj, old: str, new: str):
    """Replace the old project folder at the start of every path string (paths are stored absolute)."""
    if isinstance(obj, str):
        # the folder itself or a path inside it, not a sibling folder whose name starts the same ("<old>_2")
        inside = len(obj) == len(old) or obj[len(old):len(old) + 1] in ("\\", "/")
        return new + obj[len(old):] if inside and os.path.normcase(obj[:len(old)]) == os.path.normcase(old) else obj
    if isinstance(obj, list):
        return [_rebase(x, old, new) for x in obj]
    if isinstance(obj, dict):
        return {k: _rebase(v, old, new) for k, v in obj.items()}
    return obj


def _members(zf: zipfile.ZipFile) -> tuple[str, list[zipfile.ZipInfo]]:
    """The folder inside the zip that holds project.db (the root, or one folder when the zip was re-packed)."""
    dbs = [i.filename for i in zf.infolist() if i.filename.rsplit("/", 1)[-1] == "project.db"]
    if not dbs:
        raise ValueError("LoRAtelier のプロジェクトの zip ではありません（project.db がありません）。")
    prefix = min(dbs, key=len)[: -len("project.db")]
    out = []
    for i in zf.infolist():
        if not i.filename.startswith(prefix) or i.is_dir():
            continue
        rel = i.filename[len(prefix):]
        p = Path(rel)
        if (not rel or rel.startswith(("/", "\\")) or p.is_absolute() or p.drive or p.root or ".." in p.parts
                or ":" in rel):
            raise ValueError(f"不正なパスを含む zip です: {i.filename}")
        out.append(i)
    return prefix, out


def import_zip(projects: ProjectManager, fileobj) -> Project:
    with zipfile.ZipFile(fileobj) as zf:
        prefix, members = _members(zf)
        manifest = {}
        if prefix + MANIFEST in zf.namelist():
            try:
                manifest = json.loads(zf.read(prefix + MANIFEST).decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as e:
                raise ValueError(f"zip の中の {MANIFEST} を読めません: {e}")
            if not isinstance(manifest, dict):
                raise ValueError(f"zip の中の {MANIFEST} の形式が違います。")
        need = sum(i.file_size for i in members)
        root = projects.root
        try:
            root.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(root).free
        except OSError as e:
            raise ValueError(f"プロジェクトの置き場所 {root} に書き込めません（{e}）。設定画面の「データの置き場所」を確認してください。") from e
        if need > free - 1_000_000_000:
            raise ValueError(f"空き容量が足りません（必要 {need / 1e9:.1f}GB、空き {free / 1e9:.1f}GB）。")
        name = manifest.get("name") or "imported"
        base = time.strftime("%Y%m%d-%H%M%S") + "_" + _slug(name)
        dest = root / base
        work = root / ".importing" / base   # same drive as dest: moved into place in one step
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        try:
            for i in members:
                rel = i.filename[len(prefix):]
                if rel == MANIFEST:
                    continue
                target = work / rel
                if not _inside(target, work):
                    raise ValueError(f"不正なパスを含む zip です: {i.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(i) as src, open(target, "wb") as dst:
                    _copy_capped(src, dst, i.file_size, i.filename)
            _fix_db(work / "project.db", manifest.get("path"), dest)
            os.replace(work, dest)
        except BaseException:
            shutil.rmtree(work, ignore_errors=True)
            raise
    return projects.get(dest.name)


def _copy_capped(src, dst, size: int, name: str):
    """Copy at most the size the zip declares (the free-space check was made with the declared sizes)."""
    done = 0
    while True:
        b = src.read(4 << 20)
        if not b:
            return
        done += len(b)
        if done > size:
            raise ValueError(f"zip の中のファイルの大きさが記録と違います: {name}")
        dst.write(b)


def _guess_root(db) -> str | None:
    """The project folder the stored paths point into, for a zip without the manifest (a project folder
    zipped by hand): run folders are recorded as <project folder>/runs/<n>."""
    for (run_dir,) in db.execute("SELECT run_dir FROM runs WHERE run_dir IS NOT NULL AND run_dir != ''").fetchall():
        parent = os.path.dirname(os.path.normpath(run_dir))
        if os.path.basename(parent).lower() == "runs":
            return os.path.dirname(parent)
    return None


def _fix_db(db_path: Path, old_root: str | None, new_root: Path):
    db = sqlite3.connect(db_path)
    try:
        old_root = _fix_db_rows(db, old_root, str(new_root))
    except (sqlite3.Error, json.JSONDecodeError) as e:   # a broken or partial project.db (e.g. tables of an old version missing)
        raise ValueError(f"zip の中の project.db を読めません: {e}")
    finally:
        db.close()   # open files cannot be removed on Windows (the import folder is cleaned up after a failure)
    # run.json is the engine-side copy of the run snapshot; keep it in step with the database
    if old_root:
        for rj in (db_path.parent / "runs").glob("*/run.json"):
            try:
                config.atomic_write_json(rj, _rebase(json.loads(rj.read_text(encoding="utf-8")), old_root, str(new_root)))
            except (OSError, json.JSONDecodeError):
                pass


def _fix_db_rows(db, old_root: str | None, new: str) -> str | None:
    db.execute("SELECT COUNT(*) FROM meta").fetchone()
    # runs that were in progress when exported cannot continue as they were
    db.execute("UPDATE runs SET status='failed', error=? WHERE status IN ('queued','preparing','training','evaluating')",
               (json.dumps({"code": "interrupted", "message": "書き出し時に実行中だったため中断扱いにしました。",
                            "fixes": ["再開用の保存地点があれば「再開」できます"]}, ensure_ascii=False),))
    db.execute("UPDATE evals SET status='failed', error='書き出し時に実行中だったため中断扱いにしました' WHERE status IN ('queued','running')")
    db.execute("UPDATE gen_batches SET status='paused', message='取り込み前に止まっていました。「再開」で続きを生成できます。' "
               "WHERE status IN ('queued','running')")
    old_root = old_root or _guess_root(db)
    if old_root:
        for rid, cfg, run_dir in db.execute("SELECT id, config, run_dir FROM runs").fetchall():
            db.execute("UPDATE runs SET config=?, run_dir=? WHERE id=?",
                       (json.dumps(_rebase(json.loads(cfg), old_root, new), ensure_ascii=False), _rebase(run_dir, old_root, new), rid))
        for cid, path in db.execute("SELECT id, path FROM checkpoints").fetchall():
            db.execute("UPDATE checkpoints SET path=? WHERE id=?", (_rebase(path, old_root, new), cid))
        for eid, ck in db.execute("SELECT id, ckpt_path FROM exports").fetchall():
            db.execute("UPDATE exports SET ckpt_path=? WHERE id=?", (_rebase(ck, old_root, new), eid))
    db.commit()
    return old_root
