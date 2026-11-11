"""Moving data between the places of the settings page (e.g. finished projects from the SSD to an HDD).

One item at a time: within one drive a folder or file is renamed (instant); across drives it is copied into a
temporary name next to its final place, each file flushed to disk and its size checked, then renamed into place,
and only then is the original removed.  A cancel or an error during the copy removes the half-made copy and leaves
the original as it was.  Project databases store absolute paths, so those are rewritten in the moved copy.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
import time
from pathlib import Path

from . import config
from .linking import same_drive

CHUNK = 16 << 20   # large reads and writes: fewer seeks on an HDD
REWRITE_EXT = {".json", ".toml", ".txt", ".yaml", ".yml"}   # run files that may name the project folder
REWRITE_MAX = 20 << 20


class Cancelled(Exception):
    pass


class MoveError(Exception):
    pass


_lock = threading.Lock()
_cancel = threading.Event()
state: dict = {"running": False}


def status() -> dict:
    with _lock:
        return dict(state, errors=list(state.get("errors", [])), notes=list(state.get("notes", [])))


def cancel() -> bool:
    with _lock:
        if not state.get("running"):
            return False
        _cancel.set()
        state["cancelling"] = True
    return True


def _progress(**kw):
    with _lock:
        state.update(kw)


def _place(kind: str, dest: str) -> Path:
    for d in config.storage_dirs(kind):
        if config._same(d, Path(dest)):
            return d
    raise MoveError("移し先は設定画面に出ている置き場所から選んでください。")


def start(kind: str, items: list[str], dest: str, projects=None, busy=None) -> dict:
    """kind "projects": items are project ids; "imagegen" / "i2t": items are places (folders) whose images move.
    busy(pid) -> reason or None: work that still uses a project (checked again right before it moves)."""
    if kind not in config.STORAGE:
        raise MoveError(f"unknown kind: {kind}")
    target = _place(kind, dest)
    items = [x for x in dict.fromkeys(items or []) if isinstance(x, str) and x]
    if not items:
        raise MoveError("移すものを選んでください。")
    if kind != "projects":
        items = [str(_place(kind, x)) for x in items if not config._same(_place(kind, x), target)]
        if not items:
            raise MoveError("移し先と同じ置き場所しか選ばれていません。")
    with _lock:
        if state.get("running"):
            raise MoveError("ほかの移動が進行中です。終わってからもう一度操作してください。")
        _cancel.clear()
        state.clear()
        state.update(running=True, kind=kind, dest=str(target), total=len(items), done=0, current=None,
                     bytes_total=0, bytes_done=0, errors=[], notes=[], started=time.time(), cancelling=False)
    threading.Thread(target=_run, args=(kind, items, target, projects, busy), daemon=True, name="relocate").start()
    return status()


def _run(kind, items, target, projects, busy):
    try:
        for n, item in enumerate(items):
            if _cancel.is_set():
                break
            try:
                if kind == "projects":
                    _progress(current=item)
                    move_project(projects, item, target, busy)
                else:
                    _progress(current=item)
                    move_images(kind, Path(item), target)
            except Cancelled:
                break
            except (MoveError, OSError, sqlite3.Error, KeyError) as e:
                with _lock:
                    state["errors"].append({"item": item, "message": _message(e)})
            _progress(done=n + 1)
        _forget_empty_places(kind)
    finally:
        _progress(running=False, current=None, finished=time.time(), cancelled=_cancel.is_set())


def _message(e) -> str:
    if isinstance(e, KeyError):
        return "見つかりません（削除されたか、ドライブが外れている可能性があります）"
    if isinstance(e, OSError) and (getattr(e, "winerror", None) == 112 or e.errno == 28):
        return "移し先の空き容量が足りません。元のデータはそのまま残っています。"
    return str(e)


# ====================================================================== copying
def _need_space(total: int, target: Path):
    target.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target).free
    if total > free - 1_000_000_000:
        raise MoveError(f"移し先の空き容量が足りません（必要 {total / 1e9:.1f}GB、空き {free / 1e9:.1f}GB）。元のデータはそのまま残っています。")


def _copy_file(src: Path, dst: Path):
    """Copy, flush to disk, and check the size; dst is a temporary name the caller renames afterwards."""
    size = src.stat().st_size
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while True:
            if _cancel.is_set():
                raise Cancelled()
            b = fi.read(CHUNK)
            if not b:
                break
            fo.write(b)
            with _lock:
                state["bytes_done"] += len(b)
        fo.flush()
        os.fsync(fo.fileno())
    if dst.stat().st_size != size:
        raise MoveError(f"コピーした {src.name} の大きさが元と違います。元のデータはそのまま残っています。")
    shutil.copystat(src, dst)


def _tree_size(root: Path) -> int:
    total = 0
    for d, _dirs, files in os.walk(root):
        for f in files:
            try:
                total += os.stat(os.path.join(d, f)).st_size
            except OSError:
                pass
    return total


def _copy_tree(src: Path, dst: Path):
    for d, dirs, files in os.walk(src):
        rel = Path(d).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            _copy_file(Path(d) / f, dst / rel / f)


def _replace(src: Path, dst: Path):
    last = None
    for _ in range(10):   # a thumbnail being sent or a virus scan can hold a file for a moment
        try:
            os.replace(src, dst)
            return
        except PermissionError as e:
            last = e
            time.sleep(0.3)
    raise MoveError(f"使用中のファイルがあるため移動できません: {last}")


# ====================================================================== projects
def move_project(projects, pid: str, target: Path, busy=None):
    src = projects.path_of(pid)
    if src is None:
        raise KeyError(pid)
    if config._same(src.parent, target):
        return
    final = target / pid
    if final.exists():
        raise MoveError(f"移し先に同じ名前のフォルダーがあります: {final}")
    try:
        name = projects.get(pid).meta("name") or pid
    except Exception:
        name = pid
    with projects._lock:
        if pid in projects._moving:
            raise MoveError("移動中です。")
        projects._moving[pid] = name
    try:
        # checked after the project is marked moving, so no new work can start for it in between
        why = busy(pid) if busy else None
        if why:
            raise MoveError(why)
        with projects._lock:
            p = projects._open.pop(pid, None)
        if p is not None:
            p.close()
        target.mkdir(parents=True, exist_ok=True)
        if same_drive(src, target):
            _replace(src, final)
            try:
                _rebase_project(final, str(src), str(final))
            except BaseException:
                _replace(final, src)   # back where it was
                # ...and its paths with it.  The database rolls back when its own rewrite fails, but a run file
                # that could not be rewritten fails after the database was committed with the new paths
                try:
                    _rebase_db(src, str(final), str(src))
                    _rebase_run_files(src, str(final), str(src), strict=False)
                except Exception as e:
                    with _lock:
                        state.setdefault("notes", []).append(
                            f"{name}: 元の場所に戻しましたが、保存してあるパスを元に戻せませんでした（{e}）")
                raise
            return
        total = _tree_size(src)
        _progress(bytes_total=state.get("bytes_total", 0) + total)
        _need_space(total, target)
        tmp = target / f".moving_{pid}"
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            _copy_tree(src, tmp)
            _rebase_project(tmp, str(src), str(final))
            _replace(tmp, final)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        _retire(src, final)
    finally:
        with projects._lock:
            projects._moving.pop(pid, None)   # the next get() opens the project at its new place


def _retire(src: Path, final: Path):
    """The copy is complete and in place: the original goes.  Its project.db is renamed first, so even a folder
    that cannot be removed at once (a file held open) is no longer listed next to the moved one."""
    try:
        os.replace(src / "project.db", src / "project.db.moved")
    except OSError as e:
        with _lock:
            state["notes"].append(f"移動は完了しましたが、元のフォルダーを片付けられませんでした（{e}）: {src}")
        return
    from .projectio import _remove_later, _to_trash
    try:
        _remove_later(_to_trash(src))
    except OSError:
        with _lock:
            state["notes"].append(f"移動は完了しましたが、元のフォルダーが使用中で消せませんでした。あとで手動で消してください: {src}")


def _forms(old: str, new: str) -> list[tuple[str, str]]:
    """A path as stored: plain, inside JSON (backslashes doubled), and with forward slashes."""
    return [(old, new), (old.replace("\\", "\\\\"), new.replace("\\", "\\\\")), (old.replace("\\", "/"), new.replace("\\", "/"))]


def _rebase_text(s: str, forms) -> str:
    for o, n in forms:
        if s == o:
            return n
        for sep in ("\\\\", "\\", "/"):
            if o + sep in s:
                s = s.replace(o + sep, n + sep)
        for q in ('"', "'"):   # a path that is the whole value inside JSON or TOML
            s = s.replace(q + o + q, q + n + q)
    return s


def _rebase_obj(obj, old: str, new: str):
    """JSON values: a path string that is the old folder or inside it (Japanese may be stored as unicode escapes, so the
    text cannot simply be searched)."""
    if isinstance(obj, str):
        n = os.path.normcase
        if n(obj) == n(old):
            return new
        for sep in ("\\", "/"):
            if n(obj[:len(old) + 1]) == n(old + sep):
                return new + obj[len(old):]
        return obj
    if isinstance(obj, list):
        return [_rebase_obj(x, old, new) for x in obj]
    if isinstance(obj, dict):
        return {k: _rebase_obj(v, old, new) for k, v in obj.items()}
    return obj


def _rebase_value(v: str, old: str, new: str, forms) -> str:
    if v[:1] in "{[":
        try:
            obj = json.loads(v)
        except ValueError:
            obj = None
        if isinstance(obj, (dict, list)):
            nobj = _rebase_obj(obj, old, new)
            return v if nobj == obj else json.dumps(nobj, ensure_ascii=False)
    return _rebase_text(v, forms) if any(o in v for o, _ in forms) else v


def _rebase_project(folder: Path, old: str, new: str):
    """Rewrite the project folder's old path in every text value of project.db (one transaction) and in the
    run files that name it."""
    _rebase_db(folder, old, new)
    _rebase_run_files(folder, old, new)


def _rebase_db(folder: Path, old: str, new: str):
    forms = _forms(old, new)
    db = sqlite3.connect(folder / "project.db")
    try:
        db.execute("BEGIN IMMEDIATE")
        tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        for t in tables:
            cols = [r[1] for r in db.execute(f'PRAGMA table_info("{t}")')]
            for row in db.execute(f'SELECT rowid, * FROM "{t}"').fetchall():
                changed = {}
                for c, v in zip(cols, row[1:]):
                    if isinstance(v, str):
                        nv = _rebase_value(v, old, new, forms)
                        if nv != v:
                            changed[c] = nv
                if changed:
                    db.execute(f'UPDATE "{t}" SET ' + ", ".join(f'"{c}"=?' for c in changed) + " WHERE rowid=?",
                               (*changed.values(), row[0]))
        db.execute("COMMIT")
    except BaseException:
        db.execute("ROLLBACK") if db.in_transaction else None
        raise
    finally:
        db.close()


def _rebase_run_files(folder: Path, old: str, new: str, strict: bool = True):
    """strict=False (putting a failed move back): a file that cannot be written is passed over - it is the one
    the move failed on, and still names the old place."""
    forms = _forms(old, new)
    for d, _dirs, files in os.walk(folder / "runs"):
        for f in files:
            fp = Path(d) / f
            if fp.suffix.lower() not in REWRITE_EXT:
                continue
            try:
                if fp.stat().st_size > REWRITE_MAX:
                    continue
                text = fp.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            nt = _rebase_value(text.strip(), old, new, forms) if fp.suffix.lower() == ".json" else _rebase_text(text, forms)
            if nt != text and nt != text.strip():
                try:
                    config.atomic_write_text(fp, nt)
                except OSError:
                    if strict:
                        raise


# ====================================================================== generated / added images
def _image_files(kind: str, place: Path) -> list[tuple[Path, str]]:
    """(file, path relative to the place) of what moves: imagegen = the images and their thumbnails;
    i2t = the added images (its thumbnails are a cache named by the image path, made again where needed)."""
    roots = [place, place / "thumbs"] if kind == "imagegen" else [place / "images"]
    out = []
    for r in roots:
        try:
            with os.scandir(r) as it:
                out += [(Path(e.path), str(Path(e.path).relative_to(place))) for e in it if e.is_file()
                        and not e.name.endswith((".part", ".tmp", ".moving"))]
        except OSError:
            continue
    return out


def place_summary(kind: str, place: Path) -> dict:
    files = _image_files(kind, place)
    size = 0
    for f, _ in files:
        try:
            size += f.stat().st_size
        except OSError:
            pass
    return {"files": len(files), "bytes": size}


def move_images(kind: str, place: Path, target: Path):
    files = _image_files(kind, place)
    total = sum(f.stat().st_size for f, _ in files)
    _progress(bytes_total=state.get("bytes_total", 0) + total)
    fast = same_drive(place, target)
    if not fast:
        _need_space(total, target)
    for src, rel in files:
        if _cancel.is_set():
            raise Cancelled()
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if dst.stat().st_size == src.stat().st_size:
                src.unlink()   # copied by an earlier move that stopped before removing the original
            else:
                with _lock:
                    state["notes"].append(f"移し先に同じ名前の別のファイルがあるため移していません: {src}")
            continue
        if fast:
            _replace(src, dst)
            with _lock:
                state["bytes_done"] += dst.stat().st_size
            continue
        tmp = dst.with_name(dst.name + ".moving")
        try:
            _copy_file(src, tmp)
            _replace(tmp, dst)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        src.unlink()


def _forget_empty_places(kind: str):
    """A place used before that holds nothing of this kind any more is no longer listed (the folder stays)."""
    from .store import ProjectManager
    with config.Settings._lock:
        s = config.settings()
        old = {k: list(v) for k, v in (s.get("storage_dirs_old") or {}).items()}
        keep = []
        for d in old.get(kind, []):
            p = Path(d)
            if not p.is_dir():
                keep.append(d)   # maybe a drive that is not connected: keep looking there
            elif kind == "projects":
                if ProjectManager(p).folders():
                    keep.append(d)
            elif place_summary(kind, p)["files"]:
                keep.append(d)
        if keep != old.get(kind, []):
            old[kind] = keep
            s.data["storage_dirs_old"] = old
            s.save()
