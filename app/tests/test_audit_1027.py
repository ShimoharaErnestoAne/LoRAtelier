"""Fixes of the 2026-10-27 review.  Failure risks: images trained other than the app shows them (phone photos
sideways, transparent parts black, TIFF / GIF skipped by the engines), a setting that makes SDXL training stop at
once (caption dropout), task lists lost by a task added right after a restart, a task cancelled while its next
step starts, ComfyUI graphs that sample a v-prediction model as eps or drop a LoRA's text encoder part, an edit
LoRA used as a plain LoRA, and a moved file that restarts ComfyUI instead of saying it is missing."""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lorastudio import presets, traincache  # noqa: E402
from lorastudio.evaluate import Comfy, build_t2i  # noqa: E402
from lorastudio.models import EDIT_HQ, detect_lora  # noqa: E402
from lorastudio.training import _export_image, _sdxl_te_cache, needs_prep  # noqa: E402


# ---------------------------------------------------------------- images as the engines read them
def test_exported_images_are_upright_on_white_and_in_a_format_the_engines_read(tmp_path):
    rot = Image.new("RGB", (300, 100), (200, 0, 0))
    ex = rot.getexif()
    ex[0x0112] = 6   # a phone photo held upright
    rot.save(tmp_path / "rot.jpg", exif=ex)
    Image.new("RGBA", (40, 40), (0, 0, 0, 0)).save(tmp_path / "clear.png")
    Image.new("RGB", (40, 40), (1, 2, 3)).save(tmp_path / "scan.tif")
    Image.new("RGB", (40, 40), (9, 9, 9)).save(tmp_path / "plain.jpg")
    for name, size, px in (("rot.jpg", (100, 300), None), ("clear.png", (40, 40), (255, 255, 255)), ("scan.tif", (40, 40), (1, 2, 3))):
        src = tmp_path / name
        assert needs_prep(src), name
        dest = tmp_path / f"out_{src.stem}.png"
        _export_image(src, dest, None, True)
        with Image.open(dest) as im:
            assert im.size == size and im.mode == "RGB"
            if px:
                assert im.getpixel((0, 0)) == px
    assert not needs_prep(tmp_path / "plain.jpg")   # linked as it is
    # a cropped transparent picture is on white too (it was saved with its alpha, which the engines drop)
    _export_image(tmp_path / "clear.png", tmp_path / "crop.png", [0, 0, 20, 20], True)
    with Image.open(tmp_path / "crop.png") as im:
        assert im.mode == "RGB" and im.getpixel((5, 5)) == (255, 255, 255)
    # pair runs: made at about the training size
    Image.new("RGB", (4000, 3000)).save(tmp_path / "big.png")
    _export_image(tmp_path / "big.png", tmp_path / "fit.png", None, False, 1024)
    with Image.open(tmp_path / "fit.png") as im:
        assert im.width * im.height <= 1024 * 1024 * 1.01 and im.width > 1024
    # the latent cache keeps re-encoded images apart from files cached before as they were
    a = traincache._keys({"sha": "s", "crop": None, "caption": "x"})
    b = traincache._keys({"sha": "s", "crop": None, "caption": "x", "prep": 1})
    assert a[0] != b[0]


def test_caption_dropout_does_not_stop_sdxl_training():
    assert _sdxl_te_cache({"caption_dropout_rate": 0.0}) == {}
    assert _sdxl_te_cache({"caption_dropout_rate": 0.1}) == {"cache_text_encoder_outputs": False,
                                                             "cache_text_encoder_outputs_to_disk": False}
    # musubi-tuner families have no caption dropout: not offered
    assert "caption_dropout_rate" not in presets.defaults("zimage", "character", "", 20, 16000)
    assert "caption_dropout_rate" in presets.defaults("sdxl", "character", "illustrious", 20, 16000)


def test_cached_text_encoder_outputs_are_reused_behind_the_qwen21_bridge():
    cmd = ("preparing", ["py", "bridges/qwen21_int8.py", "qwen_image_21_cache_text_encoder_outputs.py", "--x"], "cwd")
    assert "--skip_existing" in traincache.with_skip_existing(cmd)[1]
    cmd = ("training", ["py", "-m", "accelerate.commands.launch", "x_train_network.py"], "cwd")
    assert "--skip_existing" not in traincache.with_skip_existing(cmd)[1]


# ---------------------------------------------------------------- tasks
@pytest.fixture()
def pipe(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "データ"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config, pipeline
    config._settings = None
    return pipeline


def test_tasks_saved_before_a_restart_survive_a_task_added_right_after_it(pipe):
    pipe.FILE.parent.mkdir(parents=True, exist_ok=True)
    pipe.FILE.write_text(json.dumps([{"id": "old1", "project": "a", "status": "waiting", "gen_count": 0}]), encoding="utf-8")
    pl = pipe.Pipeline(None)   # recover() runs later, after the startup scan
    assert [t["id"] for t in pl.list()] == ["old1"]


def test_a_task_cancelled_while_its_run_is_reserved_stops_that_run(pipe, monkeypatch):
    from types import SimpleNamespace
    log = []
    pl = pipe.Pipeline(None)
    t = {"id": "t", "project": "a", "status": "analyzing", "message": "", "gen_count": 0}
    pl.tasks.append(t)
    proj = SimpleNamespace(id="a", q=lambda sql, args=(): [{"n": 0}])

    def start_training(p, name):
        t["status"] = "cancelled"   # the click lands while the run is being made
        return 7
    monkeypatch.setattr(pipe.hooks, "start_training", start_training, raising=False)
    monkeypatch.setattr(pipe.hooks, "stop_run", lambda p, rid: log.append(rid), raising=False)
    monkeypatch.setattr(pipe.hooks, "analysis_busy", lambda: False, raising=False)
    pl._analyzing(t, proj)
    assert log == [7] and t["status"] == "cancelled"


# ---------------------------------------------------------------- generation graphs
S = {"steps": 20, "cfg": 5.0, "sampler": "euler", "scheduler": "normal"}


def _types(g):
    return [n["class_type"] for n in g.values()]


def test_v_prediction_models_are_sampled_as_trained():
    g = build_t2i({"family": "sdxl", "variant": "noobai_vpred"}, {"ckpt": "m.safetensors", "ppm": False}, [], "a", "", 1,
                  1024, 1024, S, "x")
    assert "ModelSamplingDiscrete" in _types(g)
    n = next(n for n in g.values() if n["class_type"] == "ModelSamplingDiscrete")["inputs"]
    assert n["sampling"] == "v_prediction" and n["zsnr"] is True
    g = build_t2i({"family": "sdxl", "variant": "illustrious"}, {"ckpt": "m.safetensors", "ppm": False}, [], "a", "", 1,
                  1024, 1024, S, "x")
    assert "ModelSamplingDiscrete" not in _types(g)


def test_loras_reach_the_text_encoder_for_every_family_when_asked():
    staged = {"dit": "d.safetensors", "te": "t.safetensors", "vae": "v.safetensors", "ppm": False}
    g = build_t2i({"family": "zimage", "variant": "zimage_turbo"}, staged, [("l.safetensors", 1.0)], "a", "", 1, 1024, 1024,
                  S, "x", lora_clip=True)
    assert "LoraLoader" in _types(g) and "LoraLoaderModelOnly" not in _types(g)
    g = build_t2i({"family": "zimage", "variant": "zimage_turbo"}, staged, [("l.safetensors", 1.0)], "a", "", 1, 1024, 1024,
                  S, "x")   # comparisons: the trained LoRA only
    assert "LoraLoaderModelOnly" in _types(g)


def test_an_edit_lora_is_known_by_its_metadata_alone(tmp_path):
    hdr = {"layers.0.proj.q.lora_A": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
           "__metadata__": {"adapter_config": json.dumps({"format": "moss-edit-lora", "metadata": {}})}}
    raw = json.dumps(hdr).encode()
    p = tmp_path / "attn_only_edit.safetensors"
    p.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 4)
    d = detect_lora(p)
    assert d["family"] == "moss_image" and d["edit_format"] == EDIT_HQ


def test_a_missing_file_is_reported_without_adding_a_search_folder(tmp_path):
    c = Comfy()
    with pytest.raises(RuntimeError, match="見つかりません"):
        c.stage_model("loras", str(tmp_path / "gone" / "x.safetensors"))
    assert not c.extra_dirs
