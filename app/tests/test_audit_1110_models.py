"""2026-11-10 review of the model / download / storage code.  Risks covered: generation that stops after a restart
on a PC whose git is the bundled one, an official model that cannot be added because another model of its family
is registered, model files hidden from the lists (same name and size), model folders walked once per missing file,
hashing of every model started by opening a list, a download that cannot be continued although there is room for
the rest, errors hidden behind "log in", a token sent to a look-alike host, removed runs left inside the project,
and a failed move that leaves the database naming a folder that does not exist."""
from __future__ import annotations

import errno
import io
import json
import os
import shutil
import sqlite3
import struct
import sys
import time
import urllib.error
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SDXL = ["conditioner.embedders.1.model.x", "model.diffusion_model.y"]
LORA = ["lora_unet_input_blocks_4_1_proj_in.lora_down.weight"]


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "データ 空白"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config
    config._settings = None
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    return tmp_path


def _st(path: Path, keys=SDXL, data: bytes = b"\0" * 64, meta: dict | None = None) -> Path:
    hdr = {k: {"dtype": "F16", "shape": [1], "data_offsets": [0, len(data)]} for k in keys}
    if meta:
        hdr["__metadata__"] = meta
    raw = json.dumps(hdr).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)
    return path


class _Body(io.BytesIO):
    status = 200
    headers: dict = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ------------------------------------------------------------------ an engine git fetched, on a PC without git
def test_an_engine_git_fetched_is_asked_for_its_commit_without_starting_git(env, monkeypatch):
    from lorastudio import addons, config
    monkeypatch.setattr(config, "ROOT", env / "app root")
    monkeypatch.setenv("PATH", r"C:\somewhere")

    def run(argv, **kw):
        raise FileNotFoundError("git")
    monkeypatch.setattr(addons.subprocess, "run", run)
    eng = env / "app root" / "engines" / "node"
    (eng / ".git").mkdir(parents=True)
    (eng / ".git" / "HEAD").write_text("abc\n", encoding="utf-8")
    lock = {"engines": {"node": {"commit": "abc", "path": "engines/node"}}}
    monkeypatch.setattr(addons, "_lockfile", lambda: lock)
    monkeypatch.setattr(addons, "_ready", {})
    # asked right after a start (nothing was installed in this run): read from the folder, so the engine is ready
    # on a PC without git too (it used to fail the request, then to answer "not ready")
    assert addons.engine_ready("node") is True
    lock["engines"]["node"]["commit"] = "newer"
    addons._ready.clear()
    assert addons.engine_ready("node") is False


def test_a_bundled_node_cut_off_while_copying_does_not_look_ready(env, monkeypatch):
    from lorastudio import addons, config
    monkeypatch.setattr(config, "ROOT", env / "root")
    monkeypatch.setattr(addons, "_lockfile", lambda: {"engines": {
        n: {"commit": "abc", "path": f"engines/ComfyUI/custom_nodes/{n}"} for n in ("good", "cut")}})
    for n in ("good", "cut"):
        src = addons.bundled_dir(n)
        src.mkdir(parents=True)
        (src / addons.BUNDLE_MARK).write_text("abc", encoding="utf-8")
        (src / "nodes.py").write_text("x = 1", encoding="utf-8")
    assert addons.ensure_node("good") is True
    nodes = env / "root" / "engines" / "ComfyUI" / "custom_nodes"
    assert sorted(p.name for p in nodes.iterdir()) == ["good"] and (nodes / "good" / "nodes.py").is_file()
    real = shutil.copytree

    def cut(src, dst, **kw):
        Path(dst).mkdir(parents=True)
        shutil.copy(Path(src) / addons.BUNDLE_MARK, Path(dst) / addons.BUNDLE_MARK)   # the mark first, then the app closes
        raise OSError("stopped")
    monkeypatch.setattr(addons.shutil, "copytree", cut)
    assert addons.ensure_node("cut") is False
    assert not (nodes / "cut").exists()                      # nothing that looks like the finished node
    left = [p.name for p in nodes.iterdir() if p.name != "good"]
    assert all(name.endswith(".disabled") for name in left)  # ComfyUI skips such a folder
    monkeypatch.setattr(addons.shutil, "copytree", real)
    addons._tried.clear()
    assert addons.ensure_node("cut") is True and (nodes / "cut" / "nodes.py").is_file()   # the next start finishes it
    assert sorted(p.name for p in nodes.iterdir()) == ["cut", "good"]


# ------------------------------------------------------------------ catalog: "added" only for the model itself
def test_another_model_of_the_family_does_not_make_the_official_one_added(env, monkeypatch):
    from lorastudio import addons, config
    m = env / "models"
    te, vae = _st(m / "text_encoders" / "qwen_3_06b_base.safetensors"), _st(m / "vae" / "qwen_image_vae.safetensors")
    mine = _st(m / "diffusion_models" / "myAnimaMix.safetensors")
    config.settings().set("models", [{"id": "mix", "name": "mix", "variant": "anima", "family": "anima",
                                      "files": {"ckpt": str(mine), "te": str(te), "vae": str(vae)}}])
    entry = addons.catalog_entry("anima_base")
    assert addons.installed(entry) is None                   # shared text encoder / VAE: not the official model
    base = _st(m / "diffusion_models" / "anima-base-v1.0.safetensors")
    config.settings().set("models", config.settings().get("models") + [
        {"id": "base", "name": "Anima", "variant": "anima", "family": "anima",
         "files": {"ckpt": str(base), "te": str(te), "vae": str(vae)}}])
    assert addons.installed(entry)["id"] == "base"
    # an add-on to a model is still "added" when all of its files are in one model
    adapter = _st(m / "loras" / "zimage_turbo_training_adapter_v2.safetensors", LORA)
    z = _st(m / "diffusion_models" / "z.safetensors")
    config.settings().set("models", [{"id": "z", "name": "z", "variant": "zimage_turbo", "family": "zimage",
                                      "files": {"ckpt": str(z), "adapter": str(adapter)}}])
    assert addons.installed(addons.catalog_entry("zimage_adapter"))["id"] == "z"


def test_catalog_asks_for_the_models_once(env, monkeypatch):
    from lorastudio import addons
    calls = []
    real = addons.registry.list
    monkeypatch.setattr(addons.registry, "list", lambda: calls.append(1) or real())
    monkeypatch.setattr(addons, "_vram_gb", lambda: 16.0)
    monkeypatch.setattr(addons, "engine_ready", lambda name: True)
    view = addons.catalog_view()
    assert len(view) == len(addons.CATALOG) and len(calls) == 1


def test_a_cut_off_copy_is_not_taken_for_the_catalog_file(env):
    from lorastudio import addons
    whole = _st(env / "models" / "old place" / "v6.safetensors", data=b"\0" * 4000)
    gb = whole.stat().st_size / 1e9
    assert addons._existing(["v6.safetensors"], gb) == whole
    whole.write_bytes(whole.read_bytes()[:-2000])            # a manual copy that stopped halfway (about the same size)
    assert addons._existing(["v6.safetensors"], gb) is None


# ------------------------------------------------------------------ downloads
def test_login_message_only_for_a_refused_download(env, monkeypatch):
    from lorastudio import addons
    entry = {"id": "t", "variant": "illustrious", "name": "T", "files": {"ckpt": ("a/b", "t.safetensors", 0.001, "checkpoints")}}

    def refused(*a, **k):
        raise urllib.error.HTTPError("https://huggingface.co/x", 401, "Unauthorized", {}, None)
    monkeypatch.setattr(addons._opener, "open", refused)
    st = {"state": "running", "done_bytes": 0, "total_bytes": 0, "error": None}
    addons._work(entry, dict(entry["files"]), None, st)
    assert st["state"] == "failed" and "ログイン" in st["error"]
    # an engine that could not be installed: the real reason is shown, also when its text has those digits
    reason = "git が失敗しました: fatal: reference is not a tree: 9702b25aa9b57b32020ce513ff79b7d2143ec664 (line 1403)"
    monkeypatch.setattr(addons, "_download", lambda repo, path, dest, token, st, gb=None: _st(dest))
    monkeypatch.setattr(addons, "install_engine", lambda name: (_ for _ in ()).throw(RuntimeError(reason)))
    st = {"state": "running", "done_bytes": 0, "total_bytes": 0, "error": None}
    addons._work({**entry, "needs": {"engines": ["x"]}}, dict(entry["files"]), None, st)
    assert st["state"] == "failed" and st["error"] == reason


def test_a_full_drive_is_not_reported_as_a_dropped_connection(env, monkeypatch):
    from lorastudio import addons, remote
    assert remote.disk_full(OSError(errno.ENOSPC, "No space left on device"))
    assert not remote.disk_full(TimeoutError("timed out")) and not remote.disk_full(ConnectionResetError())

    class Full(_Body):
        headers = {"Content-Length": "100"}

        def read(self, n=-1):
            raise OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(addons._opener, "open", lambda *a, **k: Full())
    dest = env / "models" / "checkpoints" / "m.safetensors"
    dest.parent.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="空き容量"):
        addons._download("a/b", "m.safetensors", dest, None, {"done_bytes": 0})
    monkeypatch.setattr(remote, "probe", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(remote, "_open", lambda *a, **k: Full())
    st = {"id": "abc", "name": "x.safetensors", "state": "running", "done_bytes": 0, "total_bytes": 0, "error": None}
    remote._download(st, "https://civitai.com/x", "x.safetensors", {}, None)
    assert st["state"] == "failed" and "空き容量" in st["error"] and "接続" not in st["error"]


def test_continuing_a_download_needs_room_for_the_rest_only(env, monkeypatch):
    from lorastudio import addons, remote
    usage = type("U", (), {"free": (2 << 30) + 600})()
    monkeypatch.setattr(remote.shutil, "disk_usage", lambda p: usage)
    monkeypatch.setattr(remote, "probe", lambda *a, **k: {"ok": True})
    placed = []
    monkeypatch.setattr(remote, "_place", lambda st, tmp, name, hint: placed.append(tmp.read_bytes()))
    part = env / "models" / "_incoming" / "abc_x.gguf.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(b"a" * 500)

    class Rest(_Body):
        status = 206
        headers = {"Content-Range": "bytes 500-999/1000"}
    asked = []
    monkeypatch.setattr(remote, "_open", lambda url, h, timeout=60: asked.append(h.get("Range")) or Rest(b"b" * 500))
    st = {"id": "abc", "name": "x.gguf", "state": "running", "done_bytes": 0, "total_bytes": 1000, "error": None}
    remote._download(st, "https://huggingface.co/x", "x.gguf", {}, None)   # 500 left, room for 600: it goes on
    assert st["error"] is None and asked == ["bytes=500-"] and placed == [b"a" * 500 + b"b" * 500]
    part.unlink()
    st = {"id": "abc", "name": "x.gguf", "state": "running", "done_bytes": 0, "total_bytes": 1000, "error": None}
    remote._download(st, "https://huggingface.co/x", "x.gguf", {}, None)   # from the start: 1000 do not fit
    assert st["state"] == "failed" and "空き容量" in st["error"]
    # the catalog: the same for its own .part file
    dest = env / "models" / "checkpoints" / "t.safetensors"
    dest.parent.mkdir(parents=True)
    assert addons._part_gb("a/b", "t.safetensors", dest) == 0.0
    addons._part_file(addons._url("a/b", "t.safetensors"), dest).write_bytes(b"x" * 800_000)
    assert addons._part_gb("a/b", "t.safetensors", dest) == pytest.approx(0.0008)
    entry = {"id": "t", "variant": "illustrious", "name": "T", "files": {"ckpt": ("a/b", "t.safetensors", 0.001, "checkpoints")}}
    monkeypatch.setattr(addons.shutil, "disk_usage", lambda p: type("U", (), {"free": 2.0005e9})())
    monkeypatch.setattr(addons, "_download", lambda repo, path, dest, token, st, gb=None: _st(dest))
    st = {"state": "running", "done_bytes": 0, "total_bytes": 0, "error": None}
    addons._work(entry, dict(entry["files"]), None, st)
    assert st["state"] == "done", st


def test_token_goes_to_the_real_site_only_and_the_environment_one_is_used(env, monkeypatch):
    from lorastudio import remote
    assert remote._site_of("https://huggingface.co/a/b/resolve/main/m.safetensors") == "huggingface"
    assert remote._site_of("https://cdn.civitai.com/x") == "civitai"
    for bad in ("https://evilhuggingface.co/m.safetensors", "https://huggingface.co.evil.example/m.safetensors",
                "http://huggingface.co/a/b/resolve/main/m.safetensors", "https://notcivitai.com/x"):
        assert remote._site_of(bad) is None
        with pytest.raises(ValueError):
            remote.start_download(bad, "m.safetensors", "huggingface", "secret")
    started = []

    class T:
        def __init__(self, target=None, args=(), **kw):
            started.append(args)

        def start(self):
            pass
    monkeypatch.setattr(remote.threading, "Thread", T)
    monkeypatch.setenv("HF_TOKEN", "from-env")
    url = "https://huggingface.co/a/b/resolve/main/m.safetensors"
    remote.start_download(url, "m.safetensors", "civitai", None)   # the site named by the page is not trusted
    _st0, real, _name, headers, _hint = started[-1]
    assert real == url and headers == {"Authorization": "Bearer from-env"}
    remote.start_download(url + "?x=1", "m.safetensors", "huggingface", "typed")
    assert started[-1][3] == {"Authorization": "Bearer typed"}


def test_a_file_shorter_than_a_header_is_a_value_error(env):
    from lorastudio import models
    for data in (b"", b"abc"):
        f = env / "cut.safetensors"
        f.write_bytes(data)
        with pytest.raises(ValueError):
            models.read_safetensors_header(f)


# ------------------------------------------------------------------ model lists and folder walks
def test_same_name_and_size_but_other_contents_are_both_listed(env):
    from lorastudio import models
    m = env / "models"
    _st(m / "loras" / "v1" / "chr.safetensors", LORA, b"1" * 64)
    _st(m / "loras" / "v2" / "chr.safetensors", LORA, b"2" * 64)          # trained again with the same settings
    (m / "loras" / "copy of v1").mkdir()
    shutil.copy(m / "loras" / "v1" / "chr.safetensors", m / "loras" / "copy of v1" / "chr.safetensors")
    _st(m / "checkpoints" / "a" / "m.safetensors", SDXL, b"a" * 64)
    _st(m / "checkpoints" / "b" / "m.safetensors", SDXL, b"b" * 64)
    os.link(m / "checkpoints" / "a" / "m.safetensors", m / "checkpoints" / "m_link.safetensors")
    (m / "checkpoints" / "c").mkdir()
    shutil.copy(m / "checkpoints" / "a" / "m.safetensors", m / "checkpoints" / "c" / "m.safetensors")
    scan = models.library.scan()
    loras = sorted(Path(x["path"]).parent.name for x in scan["loras"])
    assert loras == ["copy of v1", "v2"] or loras == ["v1", "v2"]          # the copy once, the other LoRA too
    ck = sorted(str(Path(x["path"]).relative_to(m / "checkpoints")) for x in scan["checkpoints"])
    assert ck == [str(Path("a") / "m.safetensors"), str(Path("b") / "m.safetensors"), "m_link.safetensors"]
    # both different files are registered as models, the copy of the first is not (a file under another name is
    # its own entry, as before)
    reg = models.registry.list()
    names = sorted(Path(x["files"]["ckpt"]).parent.name + "/" + Path(x["files"]["ckpt"]).name for x in reg)
    assert names == ["a/m.safetensors", "b/m.safetensors", "checkpoints/m_link.safetensors"]


def test_each_folder_is_walked_once_however_many_files_are_missing(env, monkeypatch):
    from lorastudio import config, models
    m = env / "models"
    lib = models.library
    lib.scan()                                                              # makes the standard folders
    kept = _st(m / "checkpoints" / "new place" / "moved.safetensors")
    config.settings().set("models", [
        {"id": "a", "name": "a", "variant": "illustrious", "family": "sdxl", "files": {"ckpt": str(m / "old" / "moved.safetensors")}},
        *({"id": f"g{i}", "name": "gone", "variant": "illustrious", "family": "sdxl",
           "files": {"ckpt": str(m / "old" / f"deleted{i}.safetensors")}} for i in range(4))])
    walks = []
    real = os.walk
    monkeypatch.setattr(models.os, "walk", lambda top, *a, **k: walks.append(str(top)) or real(top, *a, **k))
    lib._repair_paths()
    assert models.registry.get("a")["files"]["ckpt"] == str(kept)           # found again by its name
    assert len(walks) == len(set(walks)) > 0                                # no folder twice, for five missing files
    # the four deleted files stay missing: a page that asks again at once does not walk the folders again...
    walks.clear()
    lib._repair_paths()
    assert walks == []
    # ...a scan always looks, and so does the next ask after a moment
    lib._repair_paths(fresh=True)
    assert walks
    back = _st(m / "checkpoints" / "deleted0.safetensors")
    monkeypatch.setattr(lib, "REPAIR_RETRY_SEC", 0)
    lib._repair_paths()
    assert models.registry.get("g0")["files"]["ckpt"] == str(back)


def test_scan_lists_each_folder_once(env, monkeypatch):
    from lorastudio import models
    m = env / "models"
    for name in ("qwen_3_06b_base.safetensors",):
        _st(m / "text_encoders" / "sub" / name, ["model.layers.0.self_attn.q_proj.weight"])
    _st(m / "vae" / "qwen_image_vae.safetensors", ["decoder.middle.0.residual.0.gamma"])
    _st(m / "checkpoints" / "one.safetensors")
    _st(m / "loras" / "l.safetensors", LORA)
    walks = []
    real = os.walk
    monkeypatch.setattr(models.os, "walk", lambda top, *a, **k: walks.append(str(top).lower()) or real(top, *a, **k))
    scan = models.library.scan()
    assert [x["name"] for x in scan["checkpoints"]] == ["one"] and [x["name"] for x in scan["loras"]] == ["l"]
    assert len(walks) == len(set(walks))
    # a file put in place after the scan is seen by the next one (the listing is not kept between scans)
    _st(m / "checkpoints" / "two.safetensors", data=b"2" * 64)
    assert sorted(x["name"] for x in models.library.scan()["checkpoints"]) == ["one", "two"]
    # a companion in a subfolder is found by its name, whatever the letter case
    assert models.library.find_companion("anima", "te") == str(m / "text_encoders" / "sub" / "qwen_3_06b_base.safetensors")


def test_one_model_is_read_without_looking_at_the_other_models_files(env, monkeypatch):
    from lorastudio import config, models
    config.settings().set("models", [
        {"id": "a", "name": "a", "variant": "illustrious", "family": "sdxl", "files": {"ckpt": r"Z:\slow drive\a.safetensors"}},
        {"id": "b", "name": "b", "variant": "pony", "family": "sdxl", "files": {"ckpt": str(_st(env / "b.safetensors"))}}])
    looked = []
    real = Path.exists
    monkeypatch.setattr(Path, "exists", lambda self: looked.append(self.name) or real(self))
    b = models.registry.get("b")
    assert b["missing"] == [] and b["variant_info"]["family"] == "sdxl" and looked == ["b.safetensors"]
    assert models.registry.get("nope") is None
    assert [x["missing"] for x in models.registry.list()] == [["ckpt"], []]


# ------------------------------------------------------------------ hashing
def test_a_list_does_not_start_hashing_and_an_asked_file_goes_first(env):
    from lorastudio import models
    f = env / "m.bin"
    f.write_bytes(b"abc")
    h = models.hasher
    assert h.peek(str(f)) is None and h.peek(str(env / "no such file")) is None
    assert not h._running and not h._q._items                                # nothing was queued
    assert h.get(str(env / "no such file")) is None
    assert h.get(str(f)) is None
    t0 = time.time()
    while h.peek(str(f)) is None and time.time() - t0 < 5:
        time.sleep(0.02)
    assert h.peek(str(f)) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    q = models.Hasher._Queue()
    q.put(("a", "ka"))
    q.put(("b", "kb"))
    q.put(("c", "kc"), front=True)                                           # asked for now: ahead of the others
    assert q.get() == ("c", "kc")
    q.to_front("kb")                                                         # asked for again while it waits
    assert [q.get(), q.get()] == [("b", "kb"), ("a", "ka")]


def test_same_size_files_are_not_read_in_full_when_a_piece_differs(env, monkeypatch):
    from lorastudio import config, evaluate, linking as ln, models
    monkeypatch.setattr(evaluate, "COMFY_MODELS", env / "engine comfy models")
    inst = env / "Comfy UI"
    (inst / "models" / "checkpoints").mkdir(parents=True)
    (inst / "main.py").write_text("")
    ln.set_comfy_dir(str(inst))
    big = 3 * (1 << 20) + 5000
    (inst / "models" / "checkpoints" / "theirs.safetensors").write_bytes(b"A" * big)
    tool = config.MODELS_DIR / "checkpoints"
    tool.mkdir(parents=True)
    (tool / "other.safetensors").write_bytes(b"A" * (big // 2) + b"B" + b"A" * (big - big // 2 - 1))   # one byte, in the middle
    monkeypatch.setattr(ln, "_sha256", lambda *a, **k: pytest.fail("read in full"))
    assert ln.find_duplicates() == []
    assert models.sample_digest(tool / "other.safetensors") != models.sample_digest(inst / "models" / "checkpoints" / "theirs.safetensors")


# ------------------------------------------------------------------ trash, export, move
def test_a_removed_run_on_another_drive_leaves_nothing_inside_the_project(env, monkeypatch):
    from lorastudio import config, linking, projectio as pio, store
    hdd = env / "HDD 置き場"
    config.set_storage_base("projects", str(hdd))
    monkeypatch.setattr(linking, "same_drive", lambda a, b: False)           # the HDD is another drive than data/
    p = store.ProjectManager().create("消す", "character", None)
    run = p.runs_dir / "0001_x"
    (run / "output").mkdir(parents=True)
    (run / "output" / "a.safetensors").write_bytes(b"w" * 100)
    trash = hdd / "projects" / ".trash"
    assert pio._trash_dir(run) == trash                                      # not <project>/runs/.trash
    first = pio._to_trash(run)
    (run / "output").mkdir(parents=True)                                     # the same run number again, the same second
    second = pio._to_trash(run)
    assert first.parent == second.parent == trash and first != second and not run.exists()
    # what an older version left inside the project is removed at the next start, with the rest of the trash
    old = p.path / "runs" / ".trash" / "0002_y_20261101-000000"
    (old / "output").mkdir(parents=True)
    (old / "output" / "big.safetensors").write_bytes(b"w" * 100)
    pio.empty_trash()
    assert not old.exists() and not any(trash.iterdir())
    p.close()


def test_a_zip_without_training_results_leaves_the_training_caches_out(env):
    from lorastudio import projectio as pio, store
    p = store.ProjectManager().create("出す", "character", None)
    cache = p.path / "traincache" / "0123abcd" / "x.npz"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"latents")
    (p.path / "sources").mkdir(exist_ok=True)
    (p.path / "sources" / "a.png").write_bytes(b"png")

    def names(include_runs):
        with zipfile.ZipFile(io.BytesIO(b"".join(pio.export_stream(p, include_runs)))) as z:
            return z.namelist()
    assert not any(n.startswith("traincache/") for n in names(False)) and "sources/a.png" in names(False)
    assert "traincache/0123abcd/x.npz" in names(True)                        # a whole copy keeps them
    assert pio.export_size(p, False) < pio.export_size(p, True)
    p.close()


def test_a_move_that_fails_after_the_database_was_rewritten_puts_the_paths_back(env, monkeypatch):
    from lorastudio import config, relocate as rel, store
    hdd = env / "HDD 置き場"
    config.set_storage_base("projects", str(hdd))
    pm = store.ProjectManager()
    p = store.ProjectManager(config.PROJECTS_DIR).create("動かす", "character", None)
    pid, src = p.id, p.path
    run_dir = p.runs_dir / "0001_x"
    run_dir.mkdir(parents=True)
    cfg = {"dir": str(src), "items": [{"src": str(src / "sources" / "x.png")}]}
    (run_dir / "run.json").write_text(json.dumps(cfg), encoding="utf-8")
    with p.tx() as db:
        db.execute("INSERT INTO runs(created, status, name, config, run_dir) VALUES(?,?,?,?,?)",
                   (time.time(), "done", "#1", json.dumps(cfg), str(run_dir)))
    p.close()
    monkeypatch.setattr(rel, "same_drive", lambda a, b: True)

    def locked(path, text):
        raise PermissionError("in use")
    monkeypatch.setattr(rel.config, "atomic_write_text", locked)             # a run file is held open
    with pytest.raises(PermissionError):
        rel.move_project(pm, pid, hdd / "projects")
    assert src.is_dir() and not (hdd / "projects" / pid).exists()            # back where it was...
    db = sqlite3.connect(src / "project.db")
    try:
        (stored, conf), = db.execute("SELECT run_dir, config FROM runs").fetchall()
    finally:
        db.close()
    assert stored == str(run_dir) and json.loads(conf) == cfg                # ...and so are the paths it stores
    assert json.loads((run_dir / "run.json").read_text(encoding="utf-8")) == cfg
