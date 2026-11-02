"""Model files put in the folders by hand are listed for generation even when their text encoder / VAE is not found
by the usual name (other distributors name them differently): the file is set up when chosen, with the text
encoder / VAE picked from the files in those folders.  A text encoder / VAE that does not fit the model (e.g. the
older Qwen-Image VAE for Qwen-Image-2.1, which made ComfyUI fail with "too many values to unpack") is never paired,
and is replaced in a model set up with it.  Fake headers only."""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_model_versions import _st, env  # noqa: E402,F401

Q21 = {"txt_in.text_norm.weight": "BF16", "modulation.1.weight": "BF16", "img_in.weight": "BF16"}
# the parts of real headers that tell text encoders / VAEs apart (Comfy-Org, unsloth files, 2026-10-26)
TE_Q21 = {"model.language_model.embed_tokens.weight": [151936, 4096], "visual.blocks.0.attn.qkv.weight": [1, 1]}
TE_Q25 = {"model.embed_tokens.weight": [152064, 3584], "visual.blocks.0.attn.qkv.weight": [1, 1]}
VAE_Q21 = {"decoder.middle.0.residual.0.gamma": [1], "decoder.upsamples.0.upsamples.0.residual.2.weight": [1],
           "decoder.head.2.weight": [4, 144, 1, 3, 3]}
VAE_WAN21 = {"decoder.middle.0.residual.0.gamma": [1], "decoder.head.2.weight": [3, 96, 3, 3, 3]}   # qwen_image_vae


def _sh(path: Path, shapes: dict) -> Path:
    raw = json.dumps({k: {"dtype": "BF16", "shape": v, "data_offsets": [0, 0]} for k, v in shapes.items()}).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    return path


def test_a_file_without_a_found_pair_is_listed_and_set_up_when_chosen(env, monkeypatch):
    from lorastudio import config, models, server
    m_dir = config.MODELS_DIR
    dit = _st(m_dir / "diffusion_models" / "qwen-image-2.1-turbo-int8-convrot.safetensors", Q21)
    te = _sh(m_dir / "text_encoders" / "odd" / "my_encoder.safetensors", TE_Q21)
    vae = _sh(m_dir / "vae" / "my_vae.safetensors", VAE_Q21)
    wan = _sh(m_dir / "vae" / "qwen_image_vae.safetensors", VAE_WAN21)
    _sh(m_dir / "text_encoders" / "qwen_2.5_vl_7b.safetensors", TE_Q25)
    monkeypatch.setattr(models.Library, "find_companion", lambda self, fam, kind: None)   # nothing found by itself

    lib = server.get_library()
    assert lib["models"] == []
    c = next(x for x in lib["checkpoints"] if x["path"] == str(dit))
    assert c["family"] == "qwen_image21" and c["model_id"] is None         # but listed, so the screen shows it

    r = server.models_from_file({"path": str(dit)})
    assert r["needs"] == ["te", "vae"]
    assert r["choices"]["vae"] == [{"path": str(vae), "fits": True}, {"path": str(wan), "fits": False}]
    assert r["choices"]["te"][0] == {"path": str(te), "fits": True} and r["choices"]["te"][1]["fits"] is False
    with pytest.raises(Exception):                                         # the older Qwen-Image VAE is refused
        server.models_from_file({"path": str(dit), "te": str(te), "vae": str(wan)})

    r = server.models_from_file({"path": str(dit), "te": str(te), "vae": str(vae)})
    lib = server.get_library()
    m = next(x for x in lib["models"] if x["id"] == r["id"])
    assert m["family"] == "qwen_image21" and m["files"] == {"ckpt": str(dit), "te": str(te), "vae": str(vae)}
    assert next(x for x in lib["checkpoints"] if x["path"] == str(dit))["model_id"] == r["id"]
    assert server.models_from_file({"path": str(dit)})["id"] == r["id"]   # chosen again: the same model


def test_pairs_are_found_by_contents_and_a_wrong_vae_is_replaced(env):
    from lorastudio import config, generate, models, server
    m_dir = config.MODELS_DIR
    dit = _st(m_dir / "diffusion_models" / "q21.safetensors", Q21)
    te = _sh(m_dir / "text_encoders" / "Qwen-Image-2.1-text_encoder-FP8.safetensors", TE_Q21)
    wan = _sh(m_dir / "vae" / "qwen_image_vae.safetensors", VAE_WAN21)
    _sh(m_dir / "text_encoders" / "qwen_2.5_vl_7b.safetensors", TE_Q25)
    # a model set up before with the older Qwen-Image VAE
    m = models.registry.add("q21", "qwen_image21", {"ckpt": str(dit), "te": str(te), "vae": str(wan)})
    errors, _ = generate.check_model_loras(m["id"], [])
    assert any("VAE（qwen_image_vae.safetensors）" in e for e in errors)
    good = _sh(m_dir / "vae" / "q21_vae_any_name.safetensors", VAE_Q21)
    server.get_library()                                                   # the scan puts the fitting VAE in
    m = models.registry.get(m["id"])
    assert m["files"]["vae"] == str(good) and m["files"]["te"] == str(te)
    assert generate.check_model_loras(m["id"], [])[0] == []
    # a new file (in ComfyUI's unet folder) is set up by the scan alone, its pair found by contents
    dit2 = _st(m_dir / "unet" / "q21b.safetensors", Q21)
    m2 = next(x for x in server.get_library()["models"] if x["files"].get("ckpt") == str(dit2))
    assert m2["files"]["te"] == str(te) and m2["files"]["vae"] == str(good)


def test_files_outside_the_folders_are_refused(env, tmp_path):
    from lorastudio import models
    outside = _st(tmp_path / "elsewhere" / "q21.safetensors", Q21)
    with pytest.raises(ValueError):
        models.library.register_file(str(outside))
