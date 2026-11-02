"""MOSS-Image edit LoRAs (ComfyUI-Moss-Image-Edit-Native): the file says which reference layout it was trained with, the
image page edits the source image with it (MOSSImageEditTextEncode, the image's own size, the pack's recommended
settings), and nothing changes without an edit LoRA or a source image.  Graph and settings only: the edit itself is
unverified on a GPU (a CPU run of the pack on this ComfyUI with a tiny random model is in docs/VERIFICATION.md)."""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lorastudio import imagegen  # noqa: E402
from lorastudio.evaluate import build_t2i  # noqa: E402
from lorastudio.models import (EDIT_CHINOLL, EDIT_HQ, EDIT_HQ512, detect_lora, is_lora_keys,  # noqa: E402
                               moss_edit_format)

CHINOLL_META = {"adapter_config": json.dumps({"format": "moss-edit-lora", "format_version": 1,
                                              "metadata": {"model_config": {"dim": 3072}, "edit_contract": "x"}})}
HQ_META = {"adapter_config": json.dumps({"format": "moss-edit-lora", "format_version": 1, "metadata": {"edit_contract": "x"}})}


def _lora(path: Path, meta: dict | None) -> Path:
    """A safetensors file with one MOSS edit LoRA tensor as published (layers.N.proj.q.lora_A) and the given metadata."""
    hdr = {"layers.0.proj.q.lora_A": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
           "layers.0.proj.ffn_up_gate.lora_A": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}}
    if meta:
        hdr["__metadata__"] = meta
    raw = json.dumps(hdr).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 8)
    return path


def test_the_layout_comes_from_the_file():
    assert moss_edit_format(CHINOLL_META, "moss_image_edit_lora_rank_64_bf16.safetensors") == EDIT_CHINOLL
    assert moss_edit_format(HQ_META, "armV3-moss-edit-lora.safetensors") == EDIT_HQ
    assert moss_edit_format(HQ_META, "moss_image_edit_hq_armV5.safetensors") == EDIT_HQ512
    assert moss_edit_format(HQ_META, "armV6.safetensors") == EDIT_HQ512
    assert moss_edit_format({}, "edit.safetensors") is None
    assert moss_edit_format({"adapter_config": "not json"}) is None


def test_edit_loras_as_published_are_listed_as_moss_loras(tmp_path):
    assert is_lora_keys(["layers.0.proj.q.lora_A", "layers.0.proj.q.lora_B"])
    d = detect_lora(_lora(tmp_path / "armV5-moss-edit-lora.safetensors", HQ_META))
    assert d["family"] == "moss_image" and d["edit_format"] == EDIT_HQ512
    d = detect_lora(_lora(tmp_path / "style.safetensors", None))   # a MOSS LoRA that is not an edit LoRA
    assert d["family"] == "moss_image" and "edit_format" not in d


MOSS = {"family": "moss_image", "variant": "moss_image_sft"}
STAGED = {"dit": "d.safetensors", "te": "t.safetensors", "vae": "v.safetensors", "ppm": True}
S = {"steps": 30, "cfg": 3.0, "sampler": "euler", "scheduler": "simple"}


def _by_type(g, t):
    return [n for n in g.values() if n["class_type"] == t]


def test_edit_graph():
    g = build_t2i(MOSS, STAGED, [("armV5.safetensors", 1.0)], "Change the car to (blue:1.2).", "", 7, 1024, 1024, S, "x",
                  lora_clip=True, init={"image": "src.png", "denoise": 0.6, "edit": EDIT_HQ512})
    enc = _by_type(g, "MOSSImageEditTextEncode")[0]["inputs"]
    assert enc["edit_format"] == EDIT_HQ512 and g[enc["images.image_1"][0]]["class_type"] == "LoadImage"
    assert enc["prompt"] == "Change the car to (blue:1.2)." and enc["reference_max_edge"] == 0 and enc["output_max_edge"] == 0
    ks = _by_type(g, "KSampler")[0]["inputs"]
    assert ks["positive"] == ["3", 0] and ks["negative"] == ["3", 1] and ks["latent_image"] == ["3", 2] and ks["denoise"] == 1.0
    slg = _by_type(g, "SkipLayerGuidanceDiTSimple")[0]["inputs"]
    assert slg["single_layers"] == "22" and slg["end_percent"] == 1.0
    # NegPip would hand the edit node a text encoder that cannot see images: left out while editing
    assert not _by_type(g, "CLIPNegPip") and not _by_type(g, "VAEEncode") and not _by_type(g, "CLIPTextEncode")


def test_without_edit_the_graph_is_as_before():
    init = {"image": "src.png", "denoise": 0.6}
    for model in (MOSS, {"family": "sdxl", "variant": "illustrious"}):
        g = build_t2i(model, {**STAGED, "ckpt": "m.safetensors"}, [], "a cat", "", 1, 1024, 1024, S, "x", lora_clip=True,
                      init=dict(init))
        assert not _by_type(g, "MOSSImageEditTextEncode") and _by_type(g, "VAEEncode")
        assert _by_type(g, "KSampler")[0]["inputs"]["denoise"] == 0.6
    # an edit asked for a model that is not MOSS-Image is ignored
    g = build_t2i({"family": "sdxl", "variant": "illustrious"}, {**STAGED, "ckpt": "m.safetensors"}, [], "a cat", "", 1, 1024,
                  1024, S, "x", lora_clip=True, init={**init, "edit": EDIT_CHINOLL})
    assert not _by_type(g, "MOSSImageEditTextEncode")


def test_request_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(imagegen, "default_sampling", lambda model, fast=None: {
        "steps": 25, "cfg": 2.5, "sampler": "er_sde", "scheduler": "simple", "fast": fast or "on"})
    chinoll = str(_lora(tmp_path / "moss_image_edit_lora_bf16.safetensors", CHINOLL_META))
    hq = str(_lora(tmp_path / "armV3.safetensors", HQ_META))
    plain = str(_lora(tmp_path / "style.safetensors", None))
    form = {"init_image": "a.png", "loras": [{"path": plain}, {"path": chinoll}]}
    ds, edit = imagegen.request_sampling(MOSS, form)
    assert edit == EDIT_CHINOLL and ds["cfg"] == 5.0 and ds["steps"] == 30 and ds["sampler"] == "euler" and ds["fast"] == "off"
    ds, edit = imagegen.request_sampling(MOSS, {**form, "loras": [{"path": hq}]})
    assert edit == EDIT_HQ and ds["cfg"] == 3.0
    # the 8-step LoRA switched on in 詳細: its own settings stay
    ds, edit = imagegen.request_sampling(MOSS, {**form, "sampling": {"fast": "on"}})
    assert edit == EDIT_CHINOLL and ds["steps"] == 25
    # no source image, no edit LoRA, or another model: not editing
    assert imagegen.request_sampling(MOSS, {"loras": [{"path": chinoll}]})[1] is None
    assert imagegen.request_sampling(MOSS, {**form, "loras": [{"path": plain}]})[1] is None
    assert imagegen.request_sampling({"family": "sdxl", "variant": "illustrious"}, form)[1] is None
    assert imagegen.request_sampling(MOSS, {**form, "loras": [{"path": str(tmp_path / "gone.safetensors")}]})[1] is None


def test_the_bundled_pack_is_the_pinned_commit():
    from lorastudio import addons
    e = addons._lockfile()["engines"][addons.MOSS_EDIT]
    src = addons.bundled_dir(addons.MOSS_EDIT)
    assert (src / addons.BUNDLE_MARK).read_text(encoding="utf-8").strip() == e["commit"]
    assert (src / "LICENSE").is_file() and (src / "__init__.py").is_file() and (src / "moss_edit" / "model.py").is_file()
