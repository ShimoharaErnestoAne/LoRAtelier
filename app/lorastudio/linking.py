"""One copy on disk for files LoRAtelier and ComfyUI both use.

Each program keeps reading its own folders; identical files are joined with hard links, so both paths
stay and the data is stored once.  Hard links need no administrator rights but only work within one
drive; files on another drive are listed and left alone.

* Base models: a file under LoRAtelier's models/ whose exact copy (same size and SHA-256) is in the
  ComfyUI folders is replaced by a hard link to ComfyUI's file, after the user saw the list.
* Exported LoRAs: also placed in ComfyUI's models/loras/LoRAStudio/ (only that folder, so they never mix
  with the user's own LoRAs), as a hard link.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from pathlib import Path

from . import config
from .models import comfy_model_dirs, sample_digest

COMFY_SUBDIRS = ("checkpoints", "diffusion_models", "unet", "loras", "text_encoders", "clip", "vae")
EXPORT_SUBDIR = "LoRAStudio"
EXTS = (".safetensors", ".gguf")

_lock = threading.Lock()
_state: dict = {"state": "idle"}
_hash_cache: dict[str, str] = {}


def comfy_dir() -> Path | None:
    p = config.settings().get("comfy_dir")
    return Path(p) if p else None


def set_comfy_dir(path: str | None) -> Path | None:
    """The ComfyUI installation to share files with (its folder, a parent of it, or its models folder)."""
    if not path or not path.strip():
        config.settings().set("comfy_dir", None)
        return None
    dirs = comfy_model_dirs(Path(path.strip().strip('"')))
    if not dirs:
        raise ValueError("ComfyUI のフォルダーが見つかりません（main.py と models フォルダーがある場所を指定してください）")
    install = dirs[0].parent
    config.settings().set("comfy_dir", str(install))
    return install


def same_drive(a: Path, b: Path) -> bool:
    return os.path.splitdrive(os.path.abspath(a))[0].lower() == os.path.splitdrive(os.path.abspath(b))[0].lower()


_link_ok: dict[str, bool] = {}
NO_LINK_MESSAGE = "このドライブ（exFAT・FAT32 など）はハードリンクに対応していません"


def links_supported(folder: Path) -> bool:
    """True when hard links can be made on the drive of folder (NTFS yes; exFAT, FAT32 no).  Tried once per drive
    with two small files in folder."""
    folder = Path(folder)
    drive = os.path.splitdrive(os.path.abspath(folder))[0].lower()
    if drive in _link_ok:
        return _link_ok[drive]
    a = folder / f".lslinktest_{os.getpid()}"
    b = a.with_name(a.name + "_b")
    try:
        folder.mkdir(parents=True, exist_ok=True)
        a.write_bytes(b"")
        os.link(a, b)
        ok = True
    except OSError:
        ok = False
    finally:
        for f in (b, a):
            try:
                f.unlink()
            except OSError:
                pass
    _link_ok[drive] = ok
    return ok


def _sha256(p: Path, fresh: bool = False) -> str:
    """fresh: read the file again (the check right before a file is replaced must not trust the cache).
    Otherwise the hash kept in settings["hash_cache"] (the same key as models.Hasher and loraconf) is used, so
    a same-size checkpoint is not read in full again on every start of the app."""
    st = p.stat()
    key = f"{p}|{st.st_size}|{st.st_mtime_ns}"
    saved_key = f"{p}|{st.st_size}|{int(st.st_mtime)}"   # models.Hasher._key
    if not fresh and key not in _hash_cache:
        saved = config.settings().get("hash_cache", {}).get(saved_key)
        if saved:
            _hash_cache[key] = saved
    if fresh or key not in _hash_cache:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(8 << 20), b""):
                h.update(chunk)
        _hash_cache[key] = h.hexdigest()
        with config.Settings._lock:
            cache = dict(config.settings().get("hash_cache", {}))
            if cache.get(saved_key) != _hash_cache[key]:
                cache[saved_key] = _hash_cache[key]
                config.settings().set("hash_cache", cache)
    return _hash_cache[key]


def _files(dirs) -> list[Path]:
    out = []
    for d in dirs:
        if d.is_dir():
            out += [f for f in d.rglob("*") if f.suffix.lower() in EXTS and f.is_file()]
    return out


def find_duplicates(progress=None) -> list[dict]:
    """Files under LoRAtelier's models/ that ComfyUI also has.  Candidates are matched by exact size, sieved by
    three 1MB pieces (SDXL fp16 checkpoints all have the same size: most pairs end here) and confirmed by SHA-256
    (the whole file is read once per file; cached while the app runs)."""
    cd = comfy_dir()
    if not cd:
        return []
    comfy = [d / s for d in comfy_model_dirs(cd) for s in COMFY_SUBDIRS]
    by_size: dict[int, list[Path]] = {}
    for f in _files(comfy):
        by_size.setdefault(f.stat().st_size, []).append(f)
    tool_files = [f for f in _files([config.MODELS_DIR]) if f.stat().st_size in by_size]
    out = []
    for i, f in enumerate(tool_files):
        if progress:
            progress(i, len(tool_files), f.name)
        size = f.stat().st_size
        for c in by_size[size]:
            item = {"tool": str(f), "comfy": str(c), "size": size}
            try:
                if os.path.samefile(f, c):
                    item["status"] = "linked"          # already one file
                elif sample_digest(f) != sample_digest(c):
                    continue                           # three 1MB pieces differ: not read in full (below)
                elif _sha256(f) != _sha256(c):
                    continue                           # same size, different contents (all SDXL fp16 files are the same size)
                elif not same_drive(f, c):
                    item["status"] = "other_drive"
                else:
                    item["status"] = "same" if links_supported(f.parent) else "no_link"
            except OSError as e:
                item.update(status="error", error=str(e))
            out.append(item)
            if item["status"] in ("same", "linked", "other_drive", "no_link"):
                break
    return out


def _replace_with_link(target: Path, master: Path):
    """Make target a hard link to master, atomically: the old file is replaced only after the link exists."""
    tmp = target.with_name(target.name + ".lslink")
    if tmp.exists():
        tmp.unlink()
    os.link(master, tmp)
    os.replace(tmp, target)


def link_duplicates(items: list[dict]) -> dict:
    """Replace each confirmed copy under models/ by a hard link to ComfyUI's file.  The links this tool
    made for its own ComfyUI (engines/ComfyUI/models) to the old file are replaced too, otherwise they
    would keep the old data on disk."""
    from .evaluate import COMFY_MODELS
    staged = _files([COMFY_MODELS])
    done, freed, errors = 0, 0, []
    for it in items:
        if it.get("status") != "same":
            continue
        t, m = Path(it["tool"]), Path(it["comfy"])
        try:
            if t.stat().st_size != m.stat().st_size or _sha256(t, fresh=True) != _sha256(m, fresh=True):
                errors.append(f"{t.name}: 確認後に内容が変わったため、そのままにしました")
                continue
            old = t.stat()
            others = [s for s in staged if s.stat().st_ino == old.st_ino and s.stat().st_dev == old.st_dev and s != t]
            _replace_with_link(t, m)
            for s in others:
                _replace_with_link(s, m)
            done += 1
            freed += it["size"]
        except OSError as e:
            if not links_supported(t.parent):
                errors.append(f"{t.name}: {NO_LINK_MESSAGE}。そのままにしました")
            else:
                errors.append(f"{t.name}: {e}（使用中の可能性があります。学習や生成が終わってからもう一度試してください）")
    return {"linked": done, "freed_bytes": freed, "errors": errors}


# ---------------------------------------------------------------- background scan (hashing takes a while)
def scan_state() -> dict:
    return dict(_state)


def start_scan():
    with _lock:
        if _state.get("state") == "running":
            return
        _state.clear()
        _state.update(state="running", message="調べています…", started=time.time())

    def work():
        def prog(i, n, name):
            _state["message"] = f"中身を比べています（{i + 1} / {n}）: {name}"
        try:
            items = find_duplicates(prog)
            _state.update(state="done", items=items, message="")
        except Exception as e:
            _state.update(state="failed", message=f"調べられませんでした: {e}")
    threading.Thread(target=work, daemon=True).start()


# ---------------------------------------------------------------- exported LoRAs
def link_export(lora: Path) -> dict | None:
    """Also place an exported LoRA in ComfyUI's models/loras/LoRAStudio/.  Never overwrites; a hard link
    within one drive, else a symbolic link if Windows allows it, else nothing (no second copy is made); on a drive
    without links at all (exFAT, FAT32) a copy."""
    cd = comfy_dir()
    if not cd:
        return None
    dest_dir = cd / "models" / "loras" / EXPORT_SUBDIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / lora.name
    n = 2
    while dest.exists():
        dest = dest_dir / f"{lora.stem}_{n}{lora.suffix}"
        n += 1
    if same_drive(lora, dest):
        try:
            os.link(lora, dest)
            return {"path": str(dest), "kind": "hardlink"}
        except OSError:
            pass   # a drive without hard links (exFAT, FAT32): a symbolic link is tried as for another drive
    try:
        os.symlink(lora, dest)
        return {"path": str(dest), "kind": "symlink"}
    except OSError:
        pass
    if same_drive(lora, dest):
        # no link of any kind on this drive (exFAT, FAT32): a LoRA is small, so a copy puts it in ComfyUI all the same
        import shutil
        tmp = dest.with_name(dest.name + ".part")
        try:
            shutil.copyfile(lora, tmp)
            os.replace(tmp, dest)
            return {"path": str(dest), "kind": "copy"}
        except OSError as e:
            tmp.unlink(missing_ok=True)
            return {"path": None, "kind": "none", "message": f"ComfyUI の loras フォルダーに置けませんでした: {e}"}
    return {"path": None, "kind": "none",
            "message": "ComfyUI が別のドライブにあり、リンクを作れなかったため、ComfyUI の loras フォルダーには置いていません"}
