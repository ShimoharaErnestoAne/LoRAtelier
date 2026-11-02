"""MOSS-Image (sd-scripts v0.12.1): the file is told apart from the other DiT families, the settings follow the
GPU (24 / 16 / 12 / 8GB), the training config uses the engine's own options, and the ComfyUI graph is the one
of the official templates.  Nothing here runs the engine or ComfyUI (unverified on real files)."""
from __future__ import annotations

import json
import struct
import sys
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


def _t(dtype="BF16"):
    return {"dtype": dtype, "shape": [1], "data_offsets": [0, 0]}


def _st(path: Path, hdr: dict) -> Path:
    raw = json.dumps(hdr).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    return path


def _dit(int8=False, prefix="model.diffusion_model."):
    h = {prefix + "shared_block_modulation.weight": _t(), prefix + "layers.0.proj.ffn_up_gate.weight": _t("I8" if int8 else "BF16"),
         prefix + "layers.0.proj.qkv.weight": _t("I8" if int8 else "BF16")}
    if int8:
        h[prefix + "layers.0.proj.qkv.weight_scale"] = _t("F32")
        h[prefix + "layers.0.proj.qkv.comfy_quant"] = _t("U8")
    return h


def test_dit_lora_and_text_encoder_are_recognised(env):
    from lorastudio import models, remote
    d = models.detect_header(_dit(int8=True), "moss_image_sft_int8_convrot.safetensors")
    assert (d["family"], d["variant"], d["int8"]) == ("moss_image", "moss_image_sft", True)
    d = models.detect_header(_dit(prefix=""), "moss_image_base_bf16.safetensors")
    assert (d["family"], d["variant"], d["int8"]) == ("moss_image", "moss_image_base", False)
    assert models.detect_header(_dit(), "renamed.safetensors")["variant"] is None   # the user picks SFT / Base
    # sd-scripts lora_generic keys share prefixes with Z-Image (lora_unet_layers_) and SD1.5 (lora_te_)
    moss = {"lora_unet_layers_0_proj_q.lora_down.weight": _t(), "lora_te_layers_0_self_attn_q_proj.lora_down.weight": _t()}
    assert models.detect_lora_header(moss)["family"] == "moss_image"
    assert models.detect_lora_header({"__metadata__": {"modelspec.architecture": "moss-image-base/lora"}})["family"] == "moss_image"
    zimage = {"lora_unet_layers_0_attention_to_q.lora_down.weight": _t()}
    assert models.detect_lora_header(zimage)["family"] == "zimage"
    assert models.detect_lora_header({"lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": _t()})["family"] == "sd15"
    # Qwen3.5 begins with linear-attention layers (no self_attn in layer 0)
    te = {"model.language_model.layers.0.linear_attn.in_proj_qkv.weight": _t(), "model.language_model.layers.0.linear_attn.A_log": _t()}
    assert remote.classify_header(te, "qwen3.5_9b_int8_convrot.safetensors")["dest"] == "text_encoders"


def test_scan_registers_the_model_with_its_companions(env):
    from lorastudio import config, models
    _st(config.MODELS_DIR / "diffusion_models" / "moss_image_sft_int8_convrot.safetensors", _dit(int8=True))
    _st(config.MODELS_DIR / "text_encoders" / "qwen3.5_9b_int8_convrot.safetensors", {"x": _t()})
    _st(config.MODELS_DIR / "vae" / "flux2-vae_small.safetensors", {"x": _t()})
    models.library.scan()
    m = next(x for x in models.registry.list() if x["family"] == "moss_image")
    assert m["variant"] == "moss_image_sft" and m["variant_confirmed"]
    assert Path(m["files"]["te"]).name == "qwen3.5_9b_int8_convrot.safetensors"
    assert Path(m["files"]["vae"]).name == "flux2-vae_small.safetensors"


@pytest.mark.parametrize("vram, dit, te, swap", [(24564, 0, 0, 0), (20000, 0, 1, 0), (16376, 1, 1, 0), (12282, 1, 1, 16), (10240, 1, 1, 28), (None, 1, 1, 0)])
def test_settings_follow_the_gpu(env, vram, dit, te, swap):
    from lorastudio import presets
    d = presets.defaults("moss_image", "character", "moss_image_sft", 20, vram)
    assert (d["convrot_int8"], d["convrot_int8_text_encoder"], d["blocks_to_swap"]) == (dit, te, swap)
    assert (d["learning_rate"], d["network_dim"], d["network_alpha"]) == (1e-4, 16, 1.0)
    assert presets.validate("moss_image", d) == []
    assert presets.validate("moss_image", {**d, "convrot_int8": 2})
    assert presets.validate("moss_image", {**d, "blocks_to_swap": 41})


@pytest.mark.parametrize("vram, dit, te", [(16376, "int8", "int8"), (24564, "bf16", "bf16"), (20470, "bf16", "int8"), (12282, "int8", "int8")])
def test_catalog_picks_files_for_the_gpu(env, monkeypatch, vram, dit, te):
    from lorastudio import addons, gpu
    monkeypatch.setattr(gpu, "query_gpus", lambda: [{"name": "x", "vram_total_mb": vram, "vram_used_mb": 0, "driver": "x"}])
    entry = addons.catalog_entry("moss_image_sft")
    assert (addons.recommend(entry, "ckpt"), addons.recommend(entry, "te")) == (dit, te)
    files = addons.resolve_files(entry)
    assert files["vae"][1] == "vae/flux2-vae_small.safetensors"   # Apache-2.0; flux2-vae.safetensors is not offered
    assert "moss_image_sft_" in files["ckpt"][1] and "sd-scripts-moss" in entry["needs"]["engines"]


def _snap(tmp_path, ckpt, te, values):
    return {"model": {"family": "moss_image", "variant": "moss_image_sft", "files": {"ckpt": str(ckpt), "te": str(te),
                                                                                       "vae": str(tmp_path / "vae.safetensors")}},
            "values": values, "output_name": "テスト lora", "engine": "sd-scripts-moss", "items": []}


def test_training_config_uses_the_engine_options(env, monkeypatch):
    import toml
    from lorastudio import presets, training
    hub = env / "hf hub"   # never the real Hugging Face cache of this PC
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    run_dir = env / "run 1"
    run_dir.mkdir()
    bf16 = _st(env / "m" / "moss_image_sft_bf16.safetensors", _dit())
    te16 = _st(env / "m" / "qwen3.5_9b_bf16.safetensors", {"x": _t()})
    v = presets.defaults("moss_image", "character", "moss_image_sft", 20, 12282)
    cmds = training.build_commands(_snap(env, bf16, te16, v), run_dir, None)
    assert cmds[0][0] == "preparing" and "OpenMOSS-Team/MOSS-Image-Base" in cmds[0][1][-1]
    # the tokenizer already in the Hub cache: no Python process just to fetch it
    repo = hub / "models--OpenMOSS-Team--MOSS-Image-Base"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text("abc123")
    tok = repo / "snapshots" / "abc123" / "tokenizer"
    tok.mkdir(parents=True)
    (tok / "tokenizer_config.json").write_text("{}")
    assert training.build_commands(_snap(env, bf16, te16, v), run_dir, None)[0][0] == "preparing"   # tokenizer.json missing
    (tok / "tokenizer.json").write_text("{}")
    assert [c[0] for c in training.build_commands(_snap(env, bf16, te16, v), run_dir, None)] == ["training"]
    phase, argv, cwd = cmds[-1]
    assert phase == "training" and "moss_image_train_network.py" in argv and cwd == training.engine_cwd("sd-scripts-moss")
    t = toml.load(argv[argv.index("--config_file") + 1])
    assert t["network_module"] == "networks.lora_generic" and t["network_train_unet_only"] is True
    assert (t["timestep_sampling"], t["weighting_scheme"], t["logit_mean"], t["discrete_flow_shift"]) == ("sigma", "logit_normal", -1.09, 1.0)
    assert t["convrot_int8"] is True and t["convrot_int8_text_encoder"] is True and t["blocks_to_swap"] == 16
    assert not {"sdpa", "xformers", "fp8_base"} & set(t)
    # INT8 files are loaded as INT8 by the engine itself: not quantized again
    i8 = _st(env / "m" / "moss_image_sft_int8_convrot.safetensors", _dit(int8=True))
    te8 = _st(env / "m" / "qwen3.5_9b_int8_convrot.safetensors", {"x": _t()})
    t = toml.load(training.build_commands(_snap(env, i8, te8, v), run_dir, None)[-1][1][-1])
    assert "convrot_int8" not in t and "convrot_int8_text_encoder" not in t
    assert training.engine_env("sd-scripts-moss")["PYTHONPATH"] == str(training.engine_cwd("sd-scripts-moss"))


def test_preflight_rejects_w4a8_and_8gb(env, monkeypatch):
    from lorastudio import addons, training
    monkeypatch.setattr(addons, "engine_ready", lambda name: True)
    errs = training._family_file_errors({"family": "moss_image", "files": {"te": "x/qwen3.5_9b_w4a8_gptq.safetensors"}})
    assert any("W4A8" in e for e in errs)
    assert training._family_file_errors({"family": "moss_image", "files": {"te": "x/qwen3.5_9b_int8_convrot.safetensors"}}) == []


def test_comfy_graph_follows_the_templates(env):
    from lorastudio import evaluate
    staged = {"dit": "d.safetensors", "te": "t.safetensors", "vae": "v.safetensors", "on": "ckpt", "gguf": False}
    for variant, slg in (("moss_image_sft", False), ("moss_image_base", True)):
        model = {"family": "moss_image", "variant": variant, "files": {}}
        s = evaluate.sampling_for(model, staged)
        assert (s["steps"], s["cfg"], s["sampler"], s["scheduler"]) == (25, 2.5, "er_sde", "simple")
        g = evaluate.build_t2i(model, staged, [("l.safetensors", 1.0)], "a cat", "nsfw", 1, 1024, 1024, s, "p")
        assert g["1t"]["inputs"]["type"] == "moss_image" and g["5"]["class_type"] == "EmptyFlux2LatentImage"
        assert not any(n["class_type"].startswith("ModelSampling") for n in g.values())
        assert g["L0"]["class_type"] == "LoraLoaderModelOnly"
        assert ("1g" in g) == slg
        if slg:
            assert g["1g"]["inputs"]["single_layers"] == "22" and g["1g"]["inputs"]["end_percent"] == 0.5
            assert g["6"]["inputs"]["model"] == ["1g", 0]


def test_8step_lora_serves_generation_only(env, monkeypatch):
    """The DMD LoRA (lodestones) is found next to the other LoRAs, used on SFT with 8 steps and no CFG unless the
    form turns it off, never on Base, never in comparisons or in the training snapshot, and its loss does not
    make the model unusable."""
    from lorastudio import addons, config, evaluate, generate, models
    _st(config.MODELS_DIR / "diffusion_models" / "moss_image_sft_bf16.safetensors", _dit())
    _st(config.MODELS_DIR / "text_encoders" / "qwen3.5_9b_int8_convrot.safetensors", {"x": _t()})
    _st(config.MODELS_DIR / "vae" / "flux2-vae_small.safetensors", {"x": _t()})
    with pytest.raises(ValueError):   # nothing to add it to: refused before the 1.2GB download
        addons.start("moss_image_sft_dmd")
    models.library.scan()
    m = next(x for x in models.registry.list() if x["family"] == "moss_image")
    assert "fast" not in generate.default_sampling(m)
    fast = _st(config.MODELS_DIR / "loras" / "moss_image_sft_8steps_dmd.safetensors",
               {"diffusion_model.layers.0.proj.ffn_up_gate.lora_down.weight": _t("F32")})   # key layout of the real file
    models.library.scan()
    m = models.registry.get(m["id"])
    assert Path(m["files"]["fast"]) == fast
    assert models.detect_lora_header(models.read_safetensors_header(fast))["family"] == "moss_image"

    s = generate.default_sampling(m)
    assert (s["fast"], s["steps"], s["cfg"], s["sampler"]) == ("on", 8, 1.0, "euler")
    off = generate.default_sampling(m, "off")
    assert (off["fast"], off["steps"], off["cfg"]) == ("off", 25, 2.5)
    assert "fast" not in generate.default_sampling({**m, "variant": "moss_image_base"})

    monkeypatch.setattr(evaluate.comfy, "stage_model", lambda kind, src: Path(src).name)
    staged = generate.stage_model_files(m)
    g = generate.build_graph(m, staged, [("mine.safetensors", 0.8)], "a cat", "", 1, 1024, 1024, s, "p")
    assert g["L0"]["inputs"]["lora_name"] == fast.name and g["L0"]["inputs"]["strength_model"] == 1.0
    assert g["L1"]["inputs"]["lora_name"] == "mine.safetensors"
    assert g["6"]["inputs"]["steps"] == 8
    g = generate.build_graph(m, staged, [], "a cat", "", 1, 1024, 1024, off, "p")
    assert not any(n["class_type"].startswith("Lora") for n in g.values())
    # comparisons: the normal sampling, no 8-step LoRA
    es = evaluate.sampling_for(m, evaluate.stage_files(m))
    assert "fast" not in es and es["steps"] == 25

    # newer files are preferred when several are there: DMD2 (2026-10-31) at 1, then lora_latest at 0.7
    key = {"diffusion_model.layers.0.proj.ffn_up_gate.lora_down.weight": _t("F32")}
    latest = _st(config.MODELS_DIR / "loras" / "moss_image_sft_8steps_dmd_latest.safetensors", key)
    assert models.library.find_companion("moss_image", "fast") == str(latest)

    def strength(path):
        mm = {**m, "files": {**m["files"], "fast": str(path)}}
        g = generate.build_graph(mm, generate.stage_model_files(mm), [], "a cat", "", 1, 1024, 1024, s, "p")
        assert g["L0"]["inputs"]["lora_name"] == Path(path).name
        return g["L0"]["inputs"]["strength_model"]
    assert strength(latest) == 0.7
    dmd2 = _st(config.MODELS_DIR / "loras" / "moss_image_8steps_dmd2.safetensors", key)
    assert models.library.find_companion("moss_image", "fast") == str(dmd2) and strength(dmd2) == 1.0
    entry = next(e for e in models.CATALOG if e["id"] == "moss_image_sft_dmd")
    assert entry["files"]["fast"][:2] == ("lodestones/MOSS-Image-Base-8-steps-dmd", "lora_dmd2.safetensors")
    assert addons._save_name(entry["files"]["fast"]) == dmd2.name

    fast.unlink()
    m = models.registry.get(m["id"])
    assert not m["missing"] and "fast" not in generate.default_sampling(m)
