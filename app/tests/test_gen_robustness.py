"""Failure paths of generation, evaluation staging, model scanning and downloads that used to leave the app
stuck or wrong: a bad file breaking the whole library, a cut download installed as complete, a same-size
file taken for "already here", a replaced model still used through an old link, a background thread dying."""
from __future__ import annotations

import io
import json
import struct
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "data dir"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config
    config._settings = None
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    return tmp_path


def _st(path: Path, hdr, data=b""):
    raw = json.dumps(hdr).encode() if not isinstance(hdr, bytes) else hdr
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)
    return path


def test_one_malformed_file_does_not_break_the_library(env):
    from lorastudio.models import library
    loras = env / "models" / "loras"
    _st(loras / "list.safetensors", [1, 2])                                    # header is a JSON list
    _st(loras / "odd.safetensors", {"cap_embedder.1.weight": 5})               # tensor entry is not a dict
    _st(loras / "meta.safetensors", {"__metadata__": {"ss_tag_frequency": json.dumps({"a": {"zxq": "many"}, "b": [1]})},
                                     "lora_unet_x.lora_down.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 0]}})
    (loras / "bad.gguf").write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 1) + b"k" + struct.pack("<I", 99))
    scan = library.scan()                                                      # used to raise -> /api/library 500
    from lorastudio import loraconf
    assert loraconf.get(str(loras / "meta.safetensors"))["triggers"] == []      # odd tag counts are skipped
    assert {x["name"] for x in scan["loras"]} >= {"meta"}


def test_hasher_survives_a_file_that_disappears(env):
    from lorastudio.models import hasher
    gone = env / "gone.bin"
    hasher._q.put((gone, "x|1|1"))
    ok = env / "ok.bin"
    ok.write_bytes(b"abc")
    hasher.get(str(ok))
    t0 = time.time()
    while hasher.get(str(ok)) is None and time.time() - t0 < 5:
        time.sleep(0.05)
    assert hasher.get(str(ok)) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_same_size_is_already_here_only_with_the_same_header(env):
    from lorastudio import remote
    ck = env / "models" / "checkpoints"
    a = _st(ck / "mine.safetensors", {"a": {"dtype": "F16", "shape": [4], "data_offsets": [0, 8]}}, b"\0" * 8)
    hdr_b = json.dumps({"b": {"dtype": "F16", "shape": [4], "data_offsets": [0, 8]}}).encode()
    size = a.stat().st_size
    assert len(hdr_b) + 16 == size                                             # another model of exactly the same size
    import hashlib
    sha_a = hashlib.sha1(a.read_bytes()[8:8 + struct.unpack("<Q", a.read_bytes()[:8])[0]]).hexdigest()
    assert remote._have("renamed.safetensors", size, hashlib.sha1(hdr_b).hexdigest()) is None
    assert remote._have("renamed.safetensors", size, sha_a) == str(a)          # same bytes, renamed by the site
    assert remote._have("mine.safetensors", size) == str(a)
    assert remote._have("mine.safetensors", size + 1) is None                  # same name, different file


def test_download_cut_short_is_kept_as_part_and_not_installed(env, monkeypatch):
    from lorastudio import remote
    hdr = json.dumps({"lora_unet_x.lora_down.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    body = struct.pack("<Q", len(hdr)) + hdr + b"\0\0"

    class Resp(io.BytesIO):
        status = 200
        headers = {"Content-Length": str(len(body) + 1000)}                   # the server meant to send more

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(remote, "probe", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(remote, "_open", lambda *a, **k: Resp(body))
    st = {"id": "abc", "name": "x.safetensors", "state": "running", "done_bytes": 0, "total_bytes": 0, "error": None}
    remote._download(st, "https://civitai.com/x", "x.safetensors", {}, None)
    assert st["state"] == "failed" and "途中で切れました" in st["error"]
    assert (env / "models" / "_incoming" / "abc_x.safetensors.part").exists()
    assert not list((env / "models").glob("loras/*.safetensors"))


def test_download_names_are_plain_file_names(env):
    from lorastudio.remote import safe_name
    assert safe_name("../../evil.safetensors") == "evil.safetensors"
    assert safe_name("..\\..\\a:b?.safetensors") == "a_b_.safetensors"
    with pytest.raises(ValueError):
        safe_name("..")


def test_replaced_model_is_linked_again(env):
    from lorastudio.evaluate import comfy
    src = env / "models" / "checkpoints" / "m.safetensors"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"old")
    name = comfy.stage_model("checkpoints", str(src))
    other = env / "other" / "m.safetensors"                                    # same name, another folder
    other.parent.mkdir()
    other.write_bytes(b"xyz")
    assert comfy.stage_model("checkpoints", str(other)) != name
    src.unlink()
    src.write_bytes(b"new")                                                    # downloaded again, same size
    from lorastudio.evaluate import COMFY_MODELS
    comfy.stage_model("checkpoints", str(src))
    assert (COMFY_MODELS / "checkpoints" / name).read_bytes() == b"new"


def test_seed_must_fit_comfyui(env):
    from lorastudio.generate import MAX_SEED, check_seed
    assert check_seed("12", 4) == 12
    for bad in (-1, MAX_SEED, "x"):
        with pytest.raises(ValueError):
            check_seed(bad, 4)
    assert check_seed(MAX_SEED - 3, 4) == MAX_SEED - 3


def test_comfyui_port_taken_by_another_program_is_reported(env, monkeypatch):
    import socket
    from lorastudio import config, evaluate
    srv = socket.socket()
    srv.bind((config.HOST, 0))
    srv.listen(8)
    monkeypatch.setattr(config, "COMFY_PORT", srv.getsockname()[1])
    try:
        assert evaluate.comfy._port_released(wait=0.2) is False
    finally:
        srv.close()
    assert evaluate.comfy._port_released(wait=0.2) is True


def test_comfyui_failing_to_start_marks_the_batch_failed(env, monkeypatch):
    """Used to escape run_batch (and end the job runner): the batch stayed "running" for ever."""
    from lorastudio import config, generate, models, store
    ck = _st(env / "models" / "checkpoints" / "m.safetensors",
             {"conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 0]},
              "model.diffusion_model.input_blocks.0.0.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 0]}})
    model = models.registry.add("m", "illustrious", {"ckpt": str(ck)})
    p = store.ProjectManager(config.PROJECTS_DIR).create("x", "character", None)
    bid = generate.new_batch(p, {"gen_model_id": model["id"], "positive": "1girl"}, 3)

    def boom(*a, **k):
        raise RuntimeError("ComfyUI の起動に失敗しました")
    monkeypatch.setattr(generate.comfy, "start", boom)
    monkeypatch.setattr(generate.comfy, "extra_dirs", {})
    generate.run_batch(p, bid)
    row = p.q("SELECT status, message FROM gen_batches WHERE id=?", (bid,))[0]
    assert row["status"] == "failed" and "起動に失敗" in row["message"]


class _Endless(io.RawIOBase):
    """A response that keeps sending until the test presses キャンセル (after the second chunk)."""
    status = 200

    def __init__(self, st, total):
        self.st, self.n, self.headers = st, 0, {"Content-Length": str(total)}

    def read(self, size=-1):
        self.n += 1
        if self.n == 2:
            self.st["cancel"] = True
        return b"x" * 16

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_link_download_cancel_removes_part(env, monkeypatch):
    from lorastudio import remote
    st = {"id": "abc", "name": "x.safetensors", "state": "running", "done_bytes": 0, "total_bytes": 0, "error": None}
    remote._state["abc"] = st
    monkeypatch.setattr(remote, "probe", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(remote, "_open", lambda *a, **k: _Endless(st, 10 ** 9))
    remote._download(st, "https://civitai.com/x", "x.safetensors", {}, None)
    assert st["state"] == "cancelled" and not st["error"]
    assert not list((env / "models").rglob("*.part")) and not list((env / "models").rglob("*.safetensors"))
    assert remote.cancel("abc") is False                                           # nothing running any more


def test_catalog_download_cancel_stops_before_engines(env, monkeypatch):
    from lorastudio import addons
    entry = {"id": "m", "name": "M", "variant": "sdxl_base", "needs": {"engines": ["never"]}}
    st = {"state": "running", "done_bytes": 0, "total_bytes": 0, "error": None}
    addons._state["m"] = st
    monkeypatch.setattr(addons._opener, "open", lambda *a, **k: _Endless(st, 10 ** 9))
    monkeypatch.setattr(addons, "install_engine", lambda name: pytest.fail("installed after cancel"))
    addons._work(entry, {"ckpt": ("r/x", "a.safetensors", 0.001, "checkpoints")}, None, st)
    assert st["state"] == "cancelled" and not st["error"]
    assert not list((env / "models").rglob("*.part")) and not (env / "models" / "checkpoints" / "a.safetensors").exists()


def test_image_to_image_material_goes_through_the_source_images(env, monkeypatch):
    """I2I material: each shot starts from the source images in turn, at the chosen strength; a missing source image
    is an error before the start, and a source image a batch used is kept when the recipe drops it."""
    from PIL import Image
    from lorastudio import config, generate, models, store
    ck = _st(env / "models" / "checkpoints" / "m.safetensors",
             {"conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 0]},
              "model.diffusion_model.input_blocks.0.0.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 0]}})
    model = models.registry.add("m", "illustrious", {"ckpt": str(ck)})
    p = store.ProjectManager(config.PROJECTS_DIR).create("x", "character", None)
    names = []
    for c in ("red", "blue"):
        buf = io.BytesIO()
        Image.new("RGB", (64, 96), c).save(buf, "PNG")
        names.append(generate.save_source_image(buf.getvalue(), generate.gen_source_dir(p))["name"])
    recipe = {"gen_model_id": model["id"], "positive": "1girl", "init_images": names, "denoise": 2}
    assert not generate.check_recipe(recipe, "character", p)["errors"]
    assert generate.check_recipe({**recipe, "init_images": ["gone.png"]}, "character", p)["errors"]
    bid = generate.new_batch(p, recipe, 3)

    graphs = []
    out = config.DATA_DIR / "comfy_output"
    out.mkdir(parents=True, exist_ok=True)

    def run_prompt(g, **k):
        graphs.append(g)
        f = out / f"o{len(graphs)}.png"
        Image.new("RGB", (8, 8)).save(f)
        return [{"filename": f.name, "subfolder": ""}]
    for name, fn in (("ensure_started", lambda *a, **k: None), ("stage_model", lambda kind, src: Path(src).name),
                     ("run_prompt", run_prompt), ("log_size", lambda: 0), ("park", lambda: None)):
        monkeypatch.setattr(generate.comfy, name, fn)
    monkeypatch.setattr(generate, "check_consistency", lambda project: None, raising=False)
    generate.run_batch(p, bid)
    assert [g["5a"]["inputs"]["image"] for g in graphs] == [names[0], names[1], names[0]]
    assert all(g["6"]["inputs"]["denoise"] == 1.0 and g["5"]["class_type"] == "VAEEncode" for g in graphs)   # clamped to 1
    assert (generate.__dict__["config"].DATA_DIR / "comfy_input" / names[0]).exists()
