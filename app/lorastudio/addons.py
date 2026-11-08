"""Add-on installation: model files from the catalog, plus what some models need besides their files
(tokenizer files for the training engine, optional engines such as a ComfyUI custom node).

Nothing here runs at setup: a model and its extras are fetched when the user adds that model, so the
first setup stays small.  Optional engines are pinned in engines.lock.json like the main engines.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import config
from .events import bus
from .models import CATALOG, catalog_entry, registry

LOCK_FILE = config.ROOT / "engines.lock.json"
_state: dict[str, dict] = {}
_lock = threading.Lock()


def state() -> dict:
    return _state


def _lockfile() -> dict:
    return json.loads(LOCK_FILE.read_text(encoding="utf-8"))


def engine_dir(name: str) -> Path:
    e = _lockfile()["engines"][name]
    return config.ROOT / e.get("path", f"engines/{name}")


_ready: dict[str, tuple] = {}   # name -> (time, result): the catalog asks for each entry on every redraw


def engine_ready(name: str) -> bool:
    import time
    hit = _ready.get(name)
    if hit and time.time() - hit[0] < 30:
        return hit[1]
    ok = _engine_ready(name)
    _ready[name] = (time.time(), ok)
    return ok


def _engine_ready(name: str) -> bool:
    e = _lockfile()["engines"].get(name)
    if not e:
        return False
    d = engine_dir(name)
    if not (d / ".git").exists():   # copied from the zip's bundled copy (ensure_ppm): the commit it was taken at
        try:
            return (d / BUNDLE_MARK).read_text(encoding="utf-8").strip() == e["commit"]
        except OSError:
            return False
    return _git(d, "rev-parse", "HEAD").strip() == e["commit"] and         (not e.get("patch") or _patch(e).applied(d))


def _patch(e: dict):
    """Local fix applied on top of a pinned commit (the patch file says why)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("ls_patch", config.ROOT / e["patch"])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git(d: Path, *args) -> str:
    # safe.directory: git refuses folders it thinks belong to another user (exFAT drives, copied folders)
    r = subprocess.run(["git", "-c", f"safe.directory={d.as_posix()}", "-C", str(d), *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return r.stdout


def _run(argv: list[str], cwd: Path | None = None):
    r = subprocess.run([str(a) for a in argv], cwd=str(cwd) if cwd else None, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"{Path(str(argv[0])).name} が失敗しました: {(r.stderr or r.stdout)[-600:]}")
    return r.stdout


def install_engine(name: str) -> None:
    """Clone an optional engine at its pinned commit and install its pinned pip packages into the env
    it belongs to.  Idempotent."""
    _ready.pop(name, None)
    e = _lockfile()["engines"][name]
    d = engine_dir(name)
    local = config.ROOT / "tools" / "git" / "cmd"   # the portable git setup_env.py fetches when the PC has none
    if (local / "git.exe").exists() and str(local) not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] = str(local) + os.pathsep + os.environ["PATH"]
    if not shutil.which("git"):
        raise RuntimeError("git が見つかりません。setup.bat をもう一度実行すると、このフォルダーの中に git を用意します。")
    if not (d / ".git").exists():
        d.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", e["url"], str(d)])
    git = ["git", "-c", f"safe.directory={d.as_posix()}", "-C", str(d)]
    if _git(d, "rev-parse", "HEAD").strip() != e["commit"]:
        if e.get("fetch"):   # e.g. a pull request head that is not on any branch
            _run([*git, "fetch", "origin", e["fetch"]])
        else:
            _run([*git, "fetch", "--depth", "200", "origin"])
            if subprocess.run([*git, "cat-file", "-e", e["commit"] + "^{commit}"], capture_output=True).returncode:
                _run([*git, "fetch", "origin", e["commit"]])
        _run([*git, "checkout", "--force", e["commit"]])
    if e.get("patch"):
        _patch(e).apply(d)
    if e.get("pip"):
        py = config.ENVS_DIR / e["env"] / "Scripts" / "python.exe"
        uv = shutil.which("uv") or (config.ROOT / "tools" / "uv" / "uv.exe")
        if Path(uv).exists() or shutil.which("uv"):
            _run([uv, "pip", "install", "--python", str(py), *e["pip"]])
        else:   # the envs are created with pip seeded
            _run([py, "-m", "pip", "install", *e["pip"]])
    _ready.pop(name, None)   # a check made during the install is out of date


PPM = "ComfyUI-ppm"
MOSS_EDIT = "ComfyUI-Moss-Image-Edit-Native"   # MOSS-Image edit LoRAs (instruction editing from a source image)
RGTHREE = "rgthree-comfy"   # rgthree's nodes in workflows read from images (SDXL Empty Latent Image ...)
BUNDLE_MARK = ".lorastudio_commit"   # in a copy made without git: the commit the files are from
_tried: set[str] = set()


def ppm_ready() -> bool:
    return engine_ready(PPM)


def bundled_dir(name: str) -> Path:
    """An engine shipped in the zip (engines/bundled/<name>, with its LICENSE and BUNDLE_MARK).  It cannot sit at
    its place under engines/ComfyUI: setup clones ComfyUI into that folder, which must then be empty."""
    return config.ROOT / "engines" / "bundled" / name


def ensure_ppm() -> bool:
    return ensure_node(PPM)


def ensure_node(name: str) -> bool:
    """A custom node pack (ComfyUI-ppm: NegPip, Attention Couple, ...; ComfyUI-Moss-Image-Edit-Native) is put in
    place the first time ComfyUI starts: copied from the zip's bundled copy when it is the pinned commit, else
    fetched with git (small, no pip packages).  Tried once per run: when neither works, generation goes on without it."""
    if engine_ready(name):
        return True
    if name not in _tried:
        _tried.add(name)
        e, d, src = _lockfile()["engines"][name], engine_dir(name), bundled_dir(name)
        try:
            mark = src / BUNDLE_MARK
            bundled = mark.read_text(encoding="utf-8").strip() if mark.is_file() else None   # no bundled copy: git
            if not (d / ".git").exists() and bundled == e["commit"]:
                if d.exists():
                    shutil.rmtree(d)   # an older bundled copy (a git checkout is updated by git instead)
                shutil.copytree(src, d)
            else:
                install_engine(name)
        except Exception as ex:
            print(f"LoRAtelier: {name} を用意できませんでした（{ex}）", flush=True)
        _ready.pop(name, None)
    return engine_ready(name)


def fetch_tokenizers(repos: list) -> None:
    """Tokenizer / config files the training engine loads from Hugging Face on first use, fetched now so
    that training later runs offline."""
    code = ("import sys, json\nfrom huggingface_hub import snapshot_download as d\n"
            "for repo, pats in json.loads(sys.argv[1]):\n    d(repo, allow_patterns=pats)\n")
    _run([config.TRAIN_PYTHON, "-c", code, json.dumps(repos)])


# ======================================================================== model files
def _existing(names: list[str], gb: float) -> Path | None:
    """A file of the same name (or a known other name of the same file) and about the same size anywhere
    under models/ (e.g. fetched by an older version into models/sdxl), so it is not downloaded twice."""
    for f in (f for n in names for f in config.MODELS_DIR.rglob(n)):
        try:
            if f.is_file() and abs(f.stat().st_size / 1e9 - gb) <= max(0.02 * gb, 0.05):
                return f
        except OSError:
            pass
    return None


def _comfy_existing(names: list[str], gb: float) -> Path | None:
    """The same file in the ComfyUI folders set in the settings (name and about the same size)."""
    from .linking import COMFY_SUBDIRS, comfy_dir
    from .models import comfy_model_dirs
    cd = comfy_dir()
    if not cd:
        return None
    for d in (d / s for d in comfy_model_dirs(cd) for s in COMFY_SUBDIRS):
        if not d.is_dir():
            continue
        for f in (f for n in names for f in d.rglob(n)):
            try:
                if f.is_file() and abs(f.stat().st_size / 1e9 - gb) <= max(0.02 * gb, 0.05):
                    return f
            except OSError:
                pass
    return None


def installed(entry: dict) -> dict | None:
    """The registered model that holds this catalog entry's main file, if any."""
    names = {_save_name(f).lower() for f in entry["files"].values() if f} | \
        {_save_name(ch["file"]).lower() for chs in entry.get("choices", {}).values() for ch in chs} | \
        {n.lower() for ns in entry.get("aliases", {}).values() for n in ns}
    for m in registry.list():
        if m.get("variant") != entry["variant"] or m["missing"]:
            continue
        if entry.get("merge"):   # an add-on to a model: installed when all of its files are there
            have = {Path(p).name.lower() for p in m["files"].values() if p}
            if all(_save_name(f).lower() in have for f in entry["files"].values() if f) and \
                    all(any(_save_name(ch["file"]).lower() in have for ch in chs) for chs in entry.get("choices", {}).values()):
                return m
        elif any(Path(p).name.lower() in names for p in m["files"].values()):
            return m
    return None


def _save_name(f: tuple) -> str:
    """File name under models/: the 5th item when the source name is generic (e.g. a Diffusers VAE)."""
    return f[4] if len(f) > 4 else Path(f[1]).name


def resolve_files(entry: dict, choices: dict | None = None) -> dict:
    """The entry's files with the user's (or the recommended) choice filled in."""
    files = dict(entry["files"])
    for key, opts in entry.get("choices", {}).items():
        want = (choices or {}).get(key) or recommend(entry, key)
        ch = next((o for o in opts if o["id"] == want), None)
        if not ch:
            raise ValueError(f"選択肢がありません: {want}")
        files[key] = ch["file"]
    return files


def recommend(entry: dict, key: str) -> str:
    """Choice that suits the GPU: the first whose VRAM bracket contains this GPU's VRAM."""
    opts = entry["choices"][key]
    vram = _vram_gb()
    if vram is None:
        return opts[0]["id"]
    return next((o["id"] for o in opts if vram < o["max_vram"]), opts[-1]["id"])


_vram: list = []


def _vram_gb() -> float | None:
    """The GPU's total VRAM, asked once (nvidia-smi per choice made the catalog slow to open)."""
    if not _vram:
        from .gpu import query_gpus
        gpus = query_gpus()
        if not gpus:
            return None   # asked again next time (e.g. the driver was busy)
        _vram.append(gpus[0]["vram_total_mb"] / 1024)
    return _vram[0]


class _NoAuthOffHost(urllib.request.HTTPRedirectHandler):
    """The token goes to huggingface.co only: the signed CDN URL it redirects to rejects a second
    credential, and should not receive the token anyway."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).hostname != urllib.parse.urlsplit(req.full_url).hostname:
            new.remove_header("Authorization")
        return new


class Cancelled(Exception):
    """The user pressed キャンセル: the half-fetched file is removed (nothing is left to continue)."""


_opener = urllib.request.build_opener(_NoAuthOffHost)
_in_flight: dict[str, threading.Lock] = {}   # destination file -> lock (two catalog entries can share a file)


def _download(repo: str, path: str, dest: Path, token: str | None, st: dict) -> None:
    """Resumable: an interrupted download continues from the .part file.  The .part is named after the
    source URL, so a different file saved under the same name never continues it."""
    url = f"https://huggingface.co/{repo}/resolve/main/{path}"
    with _lock:
        guard = _in_flight.setdefault(str(dest).lower(), threading.Lock())
    with guard:
        if dest.exists():   # fetched meanwhile by another entry that shares this file
            st["done_bytes"] += dest.stat().st_size
            return
        tmp = dest.with_name(f"{dest.name}.{hashlib.sha1(url.encode()).hexdigest()[:12]}.part")
        have = tmp.stat().st_size if tmp.exists() else 0
        req = urllib.request.Request(url)
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        if have:
            req.add_header("Range", f"bytes={have}-")
        expected = None
        try:
            with _opener.open(req, timeout=60) as r:
                if have and r.status != 206:   # server ignored the range: start over
                    have = 0
                length = r.headers.get("Content-Length")
                expected = have + int(length) if length and length.isdigit() else None
                st["done_bytes"] += have
                with open(tmp, "ab" if have else "wb") as f:
                    while chunk := r.read(8 << 20):
                        if st.get("cancel"):
                            raise Cancelled
                        f.write(chunk)
                        st["done_bytes"] += len(chunk)
        except Cancelled:
            tmp.unlink(missing_ok=True)
            raise
        except urllib.error.HTTPError as e:
            if not (e.code == 416 and have):   # 416: the .part is already whole (the last try ended after the last byte)
                raise
        except OSError as e:   # timeout, reset: the .part is kept
            raise RuntimeError(f"{dest.name} のダウンロード中に接続が切れました。もう一度追加すると続きから取得します。") from e
        # read() returns b"" when the connection drops early: a short file must not be installed.  Without a length
        # from the server, a safetensors file is checked against its own header
        from .remote import _incomplete
        short = _incomplete(tmp, dest.name, expected)
        if short:
            raise RuntimeError(f"{dest.name} のダウンロードが途中で切れました（{short}）。"
                               "もう一度追加すると続きから取得します。")
        tmp.replace(dest)


def cancel(cid: str) -> bool:
    """Stops at the next 8MB chunk, or before the next step (an engine install already running finishes)."""
    st = _state.get(cid)
    if not st or st.get("state") != "running":
        return False
    st.update(cancel=True, message="キャンセル中")
    return True


def start(cid: str, token: str | None = None, choices: dict | None = None) -> dict:
    entry = catalog_entry(cid)
    if not entry:
        raise KeyError(cid)
    if entry.get("deferred"):
        raise ValueError(entry.get("note") or "このモデルは現在追加できません。")
    if entry.get("needs_model") and not any(m.get("variant") == entry["variant"] for m in registry.list()):
        raise ValueError(entry["needs_model"])   # an add-on with nothing to add it to: do not download it in vain
    with _lock:
        if _state.get(cid, {}).get("state") == "running":
            return _state[cid]
        files = resolve_files(entry, choices)
        total = int(sum(f[2] for f in files.values()) * 1e9)
        st = {"state": "running", "done_bytes": 0, "total_bytes": total, "error": None, "message": "準備中", "started": time.time()}
        _state[cid] = st
    threading.Thread(target=_work, args=(entry, files, token, st), daemon=True, name=f"addon-{cid}").start()
    bus.publish("download", id=cid, state="running")
    return st


def _work(entry: dict, sources: dict, token: str | None, st: dict) -> None:
    try:
        files = {}
        for key, f in sources.items():
            if st.get("cancel"):
                raise Cancelled
            repo, path, gb, sub = f[:4]
            name = _save_name(f)
            hit = _existing([name, *entry.get("aliases", {}).get(key, [])], gb)
            if hit:
                files[key] = str(hit)
                st["done_bytes"] += hit.stat().st_size
                continue
            dest = config.MODELS_DIR / sub / name
            hit = _comfy_existing([name, *entry.get("aliases", {}).get(key, [])], gb)
            if hit:   # ComfyUI has it: one copy on disk (hard link), or used where it is on another drive
                from .linking import same_drive
                files[key] = str(hit)
                if same_drive(hit, dest):
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(hit, dest)
                        files[key] = str(dest)
                    except OSError:
                        pass   # a drive without hard links (exFAT, FAT32): used where it is, as on another drive
                st["done_bytes"] += hit.stat().st_size
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(dest.parent).free / 1e9
            if free < gb + 2:
                raise RuntimeError(f"空き容量が不足しています（{name} に約 {gb:.1f}GB 必要 / 空き {free:.1f}GB）。")
            st["message"] = f"ダウンロード中: {name}"
            _download(repo, path, dest, token, st)
            files[key] = str(dest)
        needs = entry.get("needs") or {}
        for eng in needs.get("engines", []):
            if st.get("cancel"):
                raise Cancelled
            st["message"] = f"追加の部品を準備中: {eng}"
            install_engine(eng)
        if st.get("cancel"):
            raise Cancelled
        if needs.get("tokenizers"):
            st["message"] = "学習用のトークナイザーを取得中"
            fetch_tokenizers(needs["tokenizers"])
        if "te" not in files and files.get("train_te"):   # training files alone: generation uses the bf16 encoder too
            files["te"] = files["train_te"]
        m = installed(entry)
        if not m and entry.get("merge"):
            # add the files to a model of the same kind the user already has (e.g. the generation-only one)
            m = next((x for x in registry.list() if x.get("variant") == entry["variant"]), None)
            if m:
                keep = {k: p for k, p in m["files"].items() if k == "te" and k in files and "train_te" in files}
                registry.update(m["id"], files={**m["files"], **files, **keep})
        if not m:
            m = registry.add(entry["name"], entry["variant"], files)
        st.update(state="done", model_id=m["id"], message="完了")
    except Cancelled:
        st.update(state="cancelled", message="キャンセルしました")
    except Exception as e:
        msg = str(e)
        if "401" in msg or "403" in msg:
            msg = "このモデルは Hugging Face でのログインと利用条件への同意が必要です。モデルのページで同意し、アクセストークンを入力してください。"
        st.update(state="failed", error=msg, message="失敗")
    bus.publish("download", id=entry["id"], state=st["state"])


def catalog_view() -> list[dict]:
    out = []
    for c in CATALOG:
        m = installed(c)
        if m and not all(engine_ready(e) for e in (c.get("needs") or {}).get("engines", [])):
            m = None   # an engine pin moved on (e.g. the Qwen-Image-2.1 trainer): "追加する" again updates it
        rec = {k: recommend(c, k) for k in c.get("choices", {})}
        files = resolve_files(c, rec)
        out.append({**c, "files": files, "size_gb": round(sum(f[2] for f in files.values()), 1), "recommended": rec,
                    "installed_model": m["id"] if m else None, "dl": _state.get(c["id"]),
                    # what worked with it on this PC (an add-on to a model has no record of its own)
                    "checked": (m.get("checked") or {}) if m and not c.get("merge") else None})
    return out
