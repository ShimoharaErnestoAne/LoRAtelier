"""（対応）／（非対応） in the model lists: a VAE / text encoder is judged by its contents (the safetensors header) for
each model family, and 学習 対応 / 非対応 by the checks made before training starts.  A wrong mark sends the user to a
file that fails in ComfyUI (a Wan VAE for MOSS-Image: "expected 64 channels, but got 128") or hides one that works."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _t(*shape):
    return {"dtype": "BF16", "shape": list(shape), "data_offsets": [0, 0]}


SD_VAE = {"decoder.conv_in.weight": _t(512, 4, 3, 3)}
FLUX1_AE = {"decoder.conv_in.weight": _t(512, 16, 3, 3)}
FLUX2_VAE = {"decoder.conv_in.weight": _t(512, 32, 3, 3)}
WAN21_VAE = {"decoder.middle.0.residual.0.gamma": _t(384), "decoder.head.2.weight": _t(12, 96, 3, 3, 3)}
QWEN35 = {"model.layers.0.linear_attn.in_proj.weight": _t(1, 1), "model.embed_tokens.weight": _t(248320, 4096)}
QWEN3_4B = {"model.embed_tokens.weight": _t(151936, 2560)}
QWEN3VL_4B = {"model.embed_tokens.weight": _t(151936, 2560), "visual.blocks.0.attn.qkv.weight": _t(1, 1)}
QWEN3VL_8B = {"model.embed_tokens.weight": _t(151936, 4096)}


def test_vae_by_the_latent_it_decodes():
    from lorastudio.models import companion_fit as fit
    assert (fit("sdxl", "vae", SD_VAE), fit("sdxl", "vae", FLUX2_VAE), fit("sdxl", "vae", WAN21_VAE)) == (True, False, False)
    assert (fit("moss_image", "vae", FLUX2_VAE), fit("moss_image", "vae", SD_VAE), fit("moss_image", "vae", WAN21_VAE)) == (True, False, False)
    assert (fit("zimage", "vae", FLUX1_AE), fit("zimage", "vae", FLUX2_VAE)) == (True, False)
    assert (fit("qwen_image", "vae", WAN21_VAE), fit("qwen_image", "vae", FLUX2_VAE)) == (True, False)
    assert fit("moss_image", "vae", {}) is None                      # nothing to tell by: not judged


def test_text_encoder_by_its_model():
    from lorastudio.models import companion_fit as fit
    assert (fit("moss_image", "te", QWEN35), fit("moss_image", "te", QWEN3VL_8B)) == (True, False)
    assert (fit("qwen_image21", "te", QWEN3VL_8B), fit("qwen_image21", "te", QWEN35)) == (True, False)
    assert (fit("zimage", "te", QWEN3_4B), fit("zimage", "te", QWEN3VL_4B), fit("zimage", "te", QWEN3VL_8B)) == (True, False, False)
    assert (fit("krea2", "te", QWEN3VL_4B), fit("krea2", "te", QWEN3_4B), fit("krea2", "te", QWEN3VL_8B)) == (True, None, False)
    assert fit("anima", "te", {"model.embed_tokens.weight": _t(151936, 1024)}) is True


def test_train_errors_are_the_preflight_model_checks(tmp_path, monkeypatch):
    from lorastudio import training
    ck = tmp_path / "m.safetensors"
    ck.write_bytes(b"x")
    monkeypatch.setattr("lorastudio.models.detect", lambda p: {"family": "sdxl", "fp8": False})
    ok = {"variant": "illustrious", "family": "sdxl", "missing": [], "files": {"ckpt": str(ck)}}
    assert training.model_train_errors(ok) == []
    monkeypatch.setattr("lorastudio.models.detect", lambda p: {"family": "sdxl", "fp8": True})
    ck.write_bytes(b"xy")   # another size: read again
    assert "fp8" in training.model_train_errors(ok)[0]
    assert training.model_train_errors({**ok, "variant": None})
