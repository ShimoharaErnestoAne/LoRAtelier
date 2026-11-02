"""Adding models from links / dropped files: the file's own header decides what it is and where it goes,
and anything unusable or unsafe is refused before (or instead of) being placed in models/."""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def remote(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "data"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config, remote
    config._settings = None
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    return remote


def _hdr(keys, meta=None):
    h = {k: {"dtype": "F16", "shape": [1], "data_offsets": [0, 0]} for k in keys}
    if meta:
        h["__metadata__"] = meta
    return h


def test_header_decides_kind_and_folder(remote):
    c = remote.classify_header
    sdxl = c(_hdr(["conditioner.embedders.1.model.x", "model.diffusion_model.y"]), "a.safetensors")
    assert (sdxl["kind"], sdxl["dest"], sdxl["ok"]) == ("model", "checkpoints", True)
    lora = c(_hdr(["lora_unet_input_blocks_4_1_proj_in.lora_down.weight"]), "l.safetensors")
    assert (lora["kind"], lora["dest"], lora["family"], lora["ok"]) == ("lora", "loras", "sdxl", True)
    sd15 = c(_hdr(["lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight"]), "o.safetensors")
    assert sd15["ok"] is False and sd15["reason"]
    assert c(_hdr(["conv1.weight", "decoder.head.0.gamma", "encoder.conv1.weight"]), "v.safetensors")["dest"] == "vae"
    assert c(_hdr(["model.layers.0.self_attn.q_proj.weight"]), "t.safetensors")["dest"] == "text_encoders"
    sd15ckpt = c(_hdr(["model.diffusion_model.input_blocks.0.0.weight", "cond_stage_model.x"]), "s.safetensors")
    assert sd15ckpt["ok"] is False


def test_only_known_sites_and_safe_formats(remote):
    with pytest.raises(ValueError, match="PixAI"):
        remote.resolve("https://pixai.art/model/123")
    with pytest.raises(ValueError):
        remote.resolve("https://example.com/model.safetensors")
    with pytest.raises(ValueError):
        remote.start_download("https://huggingface.co/a/b/resolve/main/m.ckpt", "m.ckpt", "huggingface")
    with pytest.raises(ValueError):
        remote.start_download("https://evil.example/m.safetensors", "m.safetensors", "huggingface")
    with pytest.raises(ValueError):
        remote.begin_upload("model.pt")


def test_civitai_token_goes_in_query(remote):
    assert remote._civitai_url("https://civitai.com/api/download/models/1?type=Model", "k y") == \
        "https://civitai.com/api/download/models/1?type=Model&token=k%20y"
    assert remote._civitai_url("https://civitai.com/api/download/models/1", None) == "https://civitai.com/api/download/models/1"


def test_dropped_file_is_checked_then_placed(remote, tmp_path):
    def raw(keys):
        b = json.dumps(_hdr(keys)).encode()
        return struct.pack("<Q", len(b)) + b
    st, tmp = remote.begin_upload("キャラ 1.safetensors")
    tmp.write_bytes(raw(["lora_unet_input_blocks_4_1_proj_in.lora_down.weight"]))
    st = remote.finish_upload(st, tmp, True)
    assert st["state"] == "done" and Path(st["path"]).parent.name == "loras" and Path(st["path"]).exists()
    # same name again: kept side by side, never overwritten
    st2, tmp2 = remote.begin_upload("キャラ 1.safetensors")
    tmp2.write_bytes(raw(["lora_unet_input_blocks_4_1_proj_in.lora_down.weight"]))
    st2 = remote.finish_upload(st2, tmp2, True)
    assert st2["path"] != st["path"] and Path(st["path"]).exists()
    bad, tmp3 = remote.begin_upload("sd15.safetensors")
    tmp3.write_bytes(raw(["lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight"]))
    bad = remote.finish_upload(bad, tmp3, True)
    assert bad["state"] == "failed" and not tmp3.exists()
    assert not any((tmp_path / "models" / "loras").glob("sd15*"))
