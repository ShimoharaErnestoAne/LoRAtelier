"""Latent and text-encoder caches kept across training runs of a project.

Every run trains from its own frozen dataset folder, and the engines write their caches next to it (sd-scripts:
.npz files beside the images; musubi-tuner: the run's cache folder).  Without this, each run - also every run of
a batch reservation that changes only the learning rate - encodes every image with the VAE and every caption
with the text encoder again.

Cache files are stored per project under a group key (model files, engine and its version, and the settings the
encoders depend on) and per image under a content key: the image hash and crop for latents, plus the caption for
text-encoder outputs.  Before a run, matching files are copied into the run's folders under the run's file names;
the engines then skip them (sd-scripts checks the existing file fits the bucket; musubi-tuner is given
--skip_existing).  After a run that got through caching, new files are added to the store.  Copies, not hard
links: an engine that rewrites a cache file it considers out of date must never change the stored one.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

CACHE_VERSION = 1   # bump when the dataset settings written by training.py change
_TE_SUFFIXES = ("_te.npz", "_te_outputs.npz", "_te.safetensors")
_SD_ENGINES = ("sd-scripts", "sd-scripts-moss")


def _file_id(path: str) -> list:
    try:
        st = os.stat(path)
        return [str(path).lower(), st.st_size, int(st.st_mtime)]
    except OSError:
        return [str(path).lower(), None, None]


def group_key(snap: dict) -> str | None:
    """None: this run does not share caches (texture LoRA runs build their own image pairs)."""
    if snap.get("enhance"):
        return None
    m = snap["model"]
    v = snap.get("values") or {}
    # the encoders' inputs and precision; training-only settings (learning rate, rank, steps...) are left out so
    # runs that differ only in those share the caches
    enc = {k: v[k] for k in sorted(v) if k == "resolution" or "text_encoder" in k or "vae" in k}
    raw = [CACHE_VERSION, m.get("family"), m.get("variant"), snap.get("engine"), snap.get("engine_commit"),
           {k: _file_id(p) for k, p in sorted((m.get("files") or {}).items())}, enc]
    return hashlib.sha1(json.dumps(raw, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def _keys(item: dict, masked_run: bool = False) -> tuple[str, str]:
    base = f"{item['sha']}|{json.dumps(item.get('crop'))}" + (f"|prep{item['prep']}" if item.get("prep") else "")
    # in a masked run every latent file holds an alpha mask (all ones for an image without one; sd-scripts stores it
    # in the .npz and encodes again when it is missing or unexpected), so those files are kept apart
    extra = f"|{item['mask']['key']}" if item.get("mask") else "|alpha" if masked_run else ""
    lat = hashlib.sha1((base + extra).encode()).hexdigest()[:24]
    te = hashlib.sha1((base + "|" + (item.get("caption") or "")).encode("utf-8")).hexdigest()[:24]
    return lat, te


def is_te(name: str) -> bool:
    return name.endswith(_TE_SUFFIXES)


def run_cache_dir(run_dir: Path, snap: dict) -> Path:
    return run_dir / "dataset" if snap.get("engine") in _SD_ENGINES else run_dir / "cache"


def store_dir(project_path: Path, key: str) -> Path:
    return Path(project_path) / "traincache" / key


def _copy(src: Path, dest: Path) -> bool:
    tmp = dest.with_name(dest.name + ".part")
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dest)
        return True
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def prime(project_path: Path, run_dir: Path, snap: dict) -> int:
    """Copy stored caches into the run's folders.  Returns the number of files put in place."""
    key = group_key(snap)
    if not key:
        return 0
    st = store_dir(project_path, key)
    if not st.exists():
        return 0
    dest_dir = run_cache_dir(run_dir, snap)
    dest_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for it in snap.get("items") or []:
        stem = Path(it["file"]).stem
        lat, te = _keys(it, bool(snap.get("mask")))
        for sub, k in (("L", lat), ("T", te)):
            for f in (st / sub).glob(f"{k}_*"):
                dest = dest_dir / (stem + f.name[len(k):])
                if not dest.exists() and _copy(f, dest):
                    n += 1
    return n


def harvest(project_path: Path, run_dir: Path, snap: dict) -> int:
    """Add the run's cache files to the store (only after caching finished: a file the engine was writing when it
    was stopped would otherwise be kept).  Returns the number of files added."""
    key = group_key(snap)
    if not key:
        return 0
    src_dir = run_cache_dir(run_dir, snap)
    if not src_dir.exists():
        return 0
    st = store_dir(project_path, key)
    n = 0
    for it in snap.get("items") or []:
        stem = Path(it["file"]).stem
        lat, te = _keys(it, bool(snap.get("mask")))
        for f in src_dir.glob(f"{stem}_*"):
            if f.name.endswith(".part"):
                continue
            sub, k = ("T", te) if is_te(f.name) else ("L", lat)
            dest = st / sub / (k + f.name[len(stem):])
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            if _copy(f, dest):
                n += 1
    return n


def keep() -> bool:
    """Settings "学習用キャッシュを残す" (default on).  Off: nothing is stored, and a run's own caches are deleted when
    it ends (a resumed sd-scripts run encodes them again)."""
    from . import config
    return config.settings().get("keep_train_cache", True) is not False


def _size(paths) -> int:
    n = 0
    for p in paths:
        try:
            n += p.stat().st_size
        except OSError:
            pass
    return n


def run_files(run_dir: Path) -> list[Path]:
    """The engines' cache files of one run (the musubi-tuner cache folder, the .npz files sd-scripts writes beside
    the images)."""
    run_dir = Path(run_dir)
    files = [f for f in (run_dir / "dataset").rglob("*") if f.is_file() and (f.suffix == ".npz" or is_te(f.name))]
    return files + [f for f in (run_dir / "cache").rglob("*") if f.is_file()]


def store_files(project_path: Path) -> list[Path]:
    return [f for f in (Path(project_path) / "traincache").rglob("*") if f.is_file()]


def clear_run(run_dir: Path) -> int:
    """Delete the engines' cache files of one run.  The images, captions and LoRA files stay.  Returns the bytes
    freed."""
    run_dir = Path(run_dir)
    files = run_files(run_dir)
    cache = run_dir / "cache"
    freed = _size(files)
    for f in files:
        try:
            f.unlink()
        except OSError:
            freed -= _size([f])
    shutil.rmtree(cache, ignore_errors=True)
    return freed


def clear_project(project_path: Path, skip_runs: tuple = ()) -> int:
    """Delete the project's store and the caches of its runs, except the runs named in skip_runs (the one training
    now).  Returns the bytes freed."""
    store = Path(project_path) / "traincache"
    freed = _size(store_files(project_path))
    shutil.rmtree(store, ignore_errors=True)
    runs = Path(project_path) / "runs"
    for d in (runs.iterdir() if runs.is_dir() else []):
        if d.is_dir() and d.name not in skip_runs:
            freed += clear_run(d)
    return freed


def complete(run_dir: Path, snap: dict) -> set[str]:
    """musubi-tuner runs: the kinds of cache ("latents", "te") that prime() put in place for every image.  Its cache
    scripts skip existing files only after loading their encoder (the text encoder of Qwen-Image is 7B), so a step
    whose files are all there is left out (without_complete).  sd-scripts encodes inside the training process."""
    if snap.get("engine") in _SD_ENGINES or not group_key(snap):
        return set()
    d = run_cache_dir(run_dir, snap)
    items = snap.get("items") or []
    if not items or not d.exists():
        return set()
    names = [f.name for f in d.iterdir() if f.is_file() and not f.name.endswith(".part")]
    have_l, have_t = set(), set()
    for it in items:
        stem = Path(it["file"]).stem + "_"
        for n in names:
            if n.startswith(stem):
                (have_t if is_te(n) else have_l).add(stem)
    stems = {Path(it["file"]).stem + "_" for it in items}
    return {k for k, have in (("latents", have_l), ("te", have_t)) if have >= stems}


def without_complete(cmds: list, done: set[str]) -> list:
    """Leave out the musubi-tuner cache steps whose files are all in place (complete())."""
    def kind(cmd):
        scripts = [Path(a).name for a in cmd[1][1:3]]
        if any("cache_latents" in x for x in scripts):
            return "latents"
        if any("cache_text_encoder_outputs" in x for x in scripts):
            return "te"
        return None
    return [c for c in cmds if kind(c) not in done]


def with_skip_existing(cmd: tuple) -> tuple:
    """musubi-tuner cache scripts: use the files put in place by prime() instead of encoding again."""
    phase, argv = cmd[0], cmd[1]
    # the script is argv[1], or argv[2] behind a bridge (Qwen-Image-2.1 INT8: [py, bridge, script, ...])
    scripts = [Path(a).name for a in argv[1:3]]
    if any("cache_latents" in s or "cache_text_encoder_outputs" in s for s in scripts) and "--skip_existing" not in argv:
        return (phase, list(argv) + ["--skip_existing"], *cmd[2:])
    return cmd
