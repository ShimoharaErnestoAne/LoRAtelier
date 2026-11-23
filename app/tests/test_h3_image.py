"""MiniMax H3-Image (ComfyUI v0.45.0): generation, editing and (since 2026-11-23, experimental) LoRA training with
ai-toolkit (test_h3_train_1123.py).  The file is told apart from the MiniMax H3 video model and from the other DiT
families, its text encoder / VAE are known by their contents, the ComfyUI graph is the one of the official templates,
and nothing offers it for material generation (its license does not let its images train another model).  The key names and shapes are those of
the published files (headers read 2026-11-11).  Nothing here runs ComfyUI (unverified on the real files)."""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

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


def _t(dtype="BF16", shape=(1,)):
    return {"dtype": dtype, "shape": list(shape), "data_offsets": [0, 0]}


def _st(path: Path, hdr: dict) -> Path:
    raw = json.dumps(hdr).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    return path


def _dit(audio=False, int8=False, prefix=""):
    """minimax_h3_image_pruned_*.safetensors (audio=True: minimax_h3_fl2va_*, the video model)."""
    w = "I8" if int8 else "BF16"
    h = {prefix + "video_patch_proj.weight": _t("F32", (5376, 96)), prefix + "condition_proj.weight": _t("BF16", (5376, 5120)),
         prefix + "blocks.0.attn.qkv_proj.weight": _t(w, (21504, 5376)), prefix + "blocks.0.mlp.fc1.weight": _t(w, (28672, 5376)),
         prefix + "token_refiner.blocks.0.attn.qkv_proj.weight": _t("BF16", (21504, 5376)),
         prefix + "final_layer.video_out.weight": _t("F32", (96 if audio else 576, 5376)), prefix + "adaln_t_table": _t("F32", (1025, 8))}
    if int8:
        h[prefix + "blocks.0.attn.qkv_proj.weight_scale"] = _t("F32", (21504, 1))
        h[prefix + "blocks.0.attn.qkv_proj.comfy_quant"] = _t("U8", (72,))
    if audio:
        h[prefix + "audio_patch_proj.weight"] = _t("F32", (5376, 32))
        h[prefix + "final_layer.audio_out.weight"] = _t("F32", (32, 5376))
    else:
        h["__metadata__"] = {"config": json.dumps({"transformer": {"pdd_steps": 6, "pdd_sigmas": [1.0, 0.97, 0.89, 0.75, 0.53, 0.23, 0.0]}})}
    return h


def _te(nvfp4=False):
    """qwen3vl_32b_minimax_h3_int8_convrot / _nvfp4_awq: Qwen3-VL-32B (width 5120) with its vision tower."""
    return {"model.embed_tokens.weight": _t("I8" if nvfp4 else "BF16", (151936, 5120)), "model.layers.0.self_attn.q_proj.weight": _t("I8"),
            "visual.blocks.0.attn.qkv.weight": _t()}


def _vae(image=True):
    """minimax_h3_image_vae_fp16 (image=False: minimax_h3_video_vae_fp16)."""
    w = 1024 if image else 2048
    return {"decoder.transformer_blocks.0.scale1": _t("F16", (w,)), "decoder.x_embedder.weight": _t("F16", (w, 24)),
            "decoder.proj_out.weight": _t("F16", (768 if image else 3072, w)), "decoder.register_tokens": _t("F16", (1, 4, w)),
            "encoder.conv_in.weight": _t("F16", (128, 3, 3, 3, 3)), "post_quant_conv.weight": _t("F16", (24, 24, 1, 1))}


def _files(env):
    from lorastudio import config
    return (_st(config.MODELS_DIR / "diffusion_models" / "minimax_h3_image_pruned_int8_convrot.safetensors", _dit(int8=True)),
            _st(config.MODELS_DIR / "text_encoders" / "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", _te()),
            _st(config.MODELS_DIR / "vae" / "minimax_h3_image_vae_fp16.safetensors", _vae()))


def test_the_image_model_is_told_apart_from_the_video_model_and_its_loras_are_known(env):
    from lorastudio import models
    for hdr in (_dit(), _dit(int8=True), _dit(prefix="model.diffusion_model.")):
        d = models.detect_header(hdr, "renamed.safetensors")
        assert (d["family"], d["variant"], d["confidence"]) == ("h3_image", "h3_image", "高")
    d = models.detect_header(_dit(audio=True), "minimax_h3_fl2va_pruned_int8_convrot.safetensors")
    assert d["family"] is None and "動画用" in d["evidence"]
    # ComfyUI's generic layout (Comfy-Org's turbo LoRAs) and the lora_unet_ one; the other families keep theirs
    comfy = {"diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": _t(), "diffusion_model.token_refiner.blocks.0.mlp.fc1.lora_B.weight": _t()}
    assert models.is_lora_keys(comfy) and models.detect_lora_header(comfy)["family"] == "h3_image"
    assert models.detect_lora_header({"lora_unet_blocks_12_attn_qkv_proj.lora_down.weight": _t()})["family"] == "h3_image"
    assert models.detect_lora_header({"lora_unet_layers_0_proj_q.lora_down.weight": _t()})["family"] == "moss_image"
    assert models.detect_lora_header({"lora_unet_layers_0_attention_to_q.lora_down.weight": _t()})["family"] == "zimage"
    assert models.detect_lora_header({"diffusion_model.blocks.0.self_attn.q.lora_A.weight": _t()})["family"] is None
    assert models.FAMILY_LABEL["h3_image"] == "MiniMax H3-Image"


def test_text_encoder_and_vae_are_known_by_their_contents():
    from lorastudio.models import companion_fit
    for te in (_te(), _te(nvfp4=True)):
        assert companion_fit("h3_image", "te", te) is True
        for fam in ("anima", "qwen_image", "qwen_image21", "krea2", "zimage", "moss_image"):
            assert companion_fit(fam, "te", te) is False, fam
    # another model's text encoder is not this one: Qwen3-VL 8B (Qwen-Image-2.1), Qwen3.5 (MOSS-Image)
    assert companion_fit("h3_image", "te", {"model.embed_tokens.weight": _t("BF16", (151936, 4096)), "visual.x": _t()}) is False
    assert companion_fit("h3_image", "te", {"model.layers.0.linear_attn.in_proj_qkv.weight": _t()}) is False
    assert companion_fit("h3_image", "vae", _vae()) is True
    assert companion_fit("h3_image", "vae", _vae(image=False)) is None        # the video VAE: not known to work, not refused
    assert companion_fit("h3_image", "vae", {"decoder.conv_in.weight": _t("F16", (512, 16, 3, 3))}) is False
    for fam in ("sdxl", "anima", "qwen_image", "qwen_image21", "krea2", "zimage", "moss_image"):
        assert companion_fit(fam, "vae", _vae()) is False, fam
        assert companion_fit(fam, "vae", _vae(image=False)) is False, fam


def test_scan_registers_it_and_training_asks_for_the_training_files(env, monkeypatch):
    from lorastudio import addons, models, presets, training
    _files(env)
    models.library.scan()
    m = next(x for x in models.registry.list() if x["family"] == "h3_image")
    assert m["variant"] == "h3_image" and m["variant_confirmed"] and not m["missing"]
    assert Path(m["files"]["te"]).name == "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
    assert Path(m["files"]["vae"]).name == "minimax_h3_image_vae_fp16.safetensors"
    # offered for making LoRAs (experimental); without the adapter and the engine it says what to add
    assert models.trainable("h3_image") and m["variant_info"].get("train", True)
    monkeypatch.setattr(addons, "engine_ready", lambda name: False)
    errs = training.model_train_errors(m)
    assert errs and all("MiniMax H3-Image 学習用ファイル" in e for e in errs)
    assert presets.auto_steps("h3_image", "character", 20) > 0   # its training settings exist now


def test_its_training_settings_are_those_of_ai_toolkit(env):
    from lorastudio import presets
    d = presets.defaults("h3_image", "style", "h3_image", 30, 32000)
    assert d["resolution"] == 2048 and d["optimizer_type"] == "adamw8bit" and "blocks_to_swap" not in d
    assert presets.validate("h3_image", d) == []


@pytest.mark.parametrize("vram, dit, te", [(12, "w4a8", "nvfp4"), (16, "w4a8", "nvfp4"), (24, "int8", "int8"), (32, "int8", "int8"),
                                           (48, "bf16", "int8"), (None, "w4a8", "nvfp4")])
def test_catalog_picks_files_for_the_gpu(env, monkeypatch, vram, dit, te):
    from lorastudio import addons, models
    monkeypatch.setattr(addons, "_vram_gb", lambda: vram)
    entry = models.catalog_entry("h3_image")
    assert (addons.recommend(entry, "ckpt"), addons.recommend(entry, "te")) == (dit, te)
    files = addons.resolve_files(entry)
    assert dit in files["ckpt"][1] and te in files["te"][1] and files["vae"][1].endswith("minimax_h3_image_vae_fp16.safetensors")
    assert all(f[0] == "Comfy-Org/MiniMax-H3-Image" for f in files.values())
    # found again by the names COMPANIONS looks for; no engine or tokenizer to fetch for generation (ComfyUI carries
    # its own; the training engine comes with 「MiniMax H3-Image 学習用ファイル」)
    assert Path(files["te"][1]).name in models.COMPANIONS["h3_image"]["te"] and "needs" not in entry
    assert "学習用ファイル" in entry["note"] and "ライセンス" in entry["note"]
    # its total size for this GPU
    view = {c["id"]: c for c in addons.catalog_view()}
    assert view["h3_image"]["train"] is True and view["illustrious_v2"]["train"] is True
    assert view["h3_image"]["size_gb"] == round(sum(f[2] for f in files.values()), 1)


def test_comfy_graph_follows_the_templates(env):
    from lorastudio import evaluate
    model = {"family": "h3_image", "variant": "h3_image", "files": {"ckpt": "x"}}
    staged = {"dit": "h3.safetensors", "te": "te.safetensors", "vae": "vae.safetensors", "on": "ckpt", "gguf": False}
    s = evaluate.sampling_for(model, staged)
    assert s == {"steps": 6, "cfg": 1.0, "sampler": "euler", "scheduler": "simple"}
    g = evaluate.build_t2i(model, staged, [("a.safetensors", 0.8)], "a cat", "blurry", 7, 2050, 1530, s, "p", lora_clip=True)
    assert g["1"] == {"class_type": "UNETLoader", "inputs": {"unet_name": "h3.safetensors", "weight_dtype": "default"}}
    assert g["1t"]["inputs"] == {"clip_name": "te.safetensors", "type": "minimax"}
    assert g["L0"]["class_type"] == "LoraLoaderModelOnly"            # the DiT only: ComfyUI maps no text encoder keys
    c = g["3"]
    assert c["class_type"] == "MiniMaxH3ImageConditioning" and c["inputs"] == {
        "clip": ["1t", 0], "vae": ["1v", 0], "prompt": "a cat", "width": 2048, "height": 1536}   # multiples of 32
    assert g["4"] == {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["3", 0]}}
    k = g["6"]["inputs"]
    assert (k["model"], k["positive"], k["negative"], k["latent_image"]) == (["L0", 0], ["3", 0], ["4", 0], ["3", 1])
    assert (k["steps"], k["cfg"], k["sampler_name"], k["scheduler"], k["denoise"], k["seed"]) == (6, 1.0, "euler", "simple", 1.0, 7)
    assert "blurry" not in json.dumps(g)                              # no negative prompt in this model
    assert not any(n["class_type"].startswith("Empty") or n["class_type"] == "CLIPTextEncode" for n in g.values())
    # a source image is edited as the prompt says: <Picture 1>, at the picture's own size, whatever the size fields say
    e = evaluate.build_t2i(model, staged, [], "Make <Picture 1> a watercolor.", "", 7, 1024, 1024, s, "p",
                           init={"image": "src.png", "denoise": 0.4, "edit": evaluate.EDIT_H3})
    assert e["5a"] == {"class_type": "LoadImage", "inputs": {"image": "src.png"}}
    assert e["3"]["inputs"]["images.image_1"] == ["5a", 0] and (e["3"]["inputs"]["width"], e["3"]["inputs"]["height"]) == (0, 0)
    assert e["6"]["inputs"]["denoise"] == 1.0 and "VAEEncode" not in {n["class_type"] for n in e.values()}


def test_image_page_takes_it_and_material_generation_does_not(env):
    from lorastudio import generate, imagegen, models, server
    _files(env)
    models.library.scan()
    m = next(x for x in models.registry.list() if x["family"] == "h3_image")
    assert imagegen.native_size(m) == 2048 and imagegen.size_limits(m) == {"step": 32, "min": 512, "max": 6272}
    assert imagegen.size_limits(None) == {"step": 16, "min": 512, "max": 2048}

    class Runner:
        def enqueue_task(self, owner, label, fn, on_cancel=None):
            return SimpleNamespace()
    form = {"model_id": m["id"], "positive": "a cat on a sofa", "negative": "", "loras": [], "width": 2725, "height": 1530,
            "count": 1, "seed": 5, "sampling": {}}
    assert imagegen.check(form)["errors"] == []
    s = imagegen.submit(form, Runner())["settings"]
    assert (s["width"], s["height"]) == (2720, 1536) and s["sampling"]["steps"] == 6 and "edit" not in s
    assert imagegen._size(9000, 2048, m) == 6272 and imagegen._size(2725, 1024, None) == 2048   # the other models: as before
    # with a source image: the model edits it itself (no edit LoRA, no edit pack), at its own 6 steps
    ds, edit = imagegen.request_sampling(m, {**form, "init_image": "x.png"})
    assert edit == "h3" and ds["steps"] == 6 and ds["cfg"] == 1.0
    assert imagegen.request_sampling(m, form)[1] is None
    # no negative prompt at any CFG (the page says so; the prompt editor is told)
    assert not server.negative_used(m, {"cfg": 4})
    # material generation: said why, not "生成には対応していません"
    errors, _ = generate.check_model_loras(m["id"], [])
    assert len(errors) == 1 and "素材の生成には使えません" in errors[0] and "ライセンス" in errors[0]
    assert "h3_image" in generate.IMAGE_FAMILIES and "h3_image" not in generate.GEN_FAMILIES


def test_the_web_page_lists_it_for_image_generation_only():
    js = (ROOT / "lorastudio" / "static" / "app.js").read_text(encoding="utf-8")
    assert 'const IMAGE_FAMILIES = [...GEN_FAMILIES, "h3_image"];' in js
    assert "genModelOptions(S.lib, f.model_id, IMAGE_FAMILIES)" in js           # 画像生成
    assert "genModelOptions(S.lib, r.gen_model_id)}" in js                      # 素材の生成: the default list, without it
    assert '2048: [["正方形", 2048, 2048]' in js
    sizes = json.loads(js.split("  2048: ", 1)[1].split(",\n};", 1)[0])
    assert all(w % 32 == 0 and h % 32 == 0 and 3_900_000 < w * h < 4_400_000 for _label, w, h in sizes)
