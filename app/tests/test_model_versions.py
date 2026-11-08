"""Qwen-Image-2512 / 2.1 / Z-Image files chosen for the PC: the light generation file and the training files end up
in one model (the fp8 text encoder for generation, the bf16 one for training), the choice follows the GPU, and
training refuses files the engine cannot use.  Nothing here downloads or runs the engines (unverified on real files)."""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QWEN = {"txt_norm.weight": "BF16", "img_in.weight": "BF16", "transformer_blocks.0.attn.to_q.weight": "BF16"}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "data dir"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config
    config._settings = None
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    return tmp_path


def _st(path: Path, tensors: dict) -> Path:
    hdr = {k: {"dtype": dt, "shape": [1], "data_offsets": [0, 0]} for k, dt in tensors.items()}
    raw = json.dumps(hdr).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    return path


def _gpu(monkeypatch, vram_mb):
    from lorastudio import gpu
    monkeypatch.setattr(gpu, "query_gpus", lambda: [{"name": "x", "vram_total_mb": vram_mb, "vram_used_mb": 0, "driver": "x"}])


@pytest.mark.parametrize("vram, quant", [(8188, "Q3_K_M"), (12282, "Q3_K_M"), (16376, "Q4_K_M"), (20470, "Q6_K"), (24564, "Q8_0")])
def test_qwen2512_light_file_follows_the_gpu(env, monkeypatch, vram, quant):
    from lorastudio import addons
    _gpu(monkeypatch, vram)
    entry = addons.catalog_entry("qwen_image_2512_gen")
    assert addons.recommend(entry, "gen") == quant
    files = addons.resolve_files(entry)
    assert files["gen"][1] == f"qwen-image-2512-{quant}.gguf" and "fp8_scaled" in files["te"][1]


@pytest.mark.parametrize("vram, te", [(8188, "w4a8"), (12282, "int8"), (16376, "int8")])
def test_qwen21_text_encoder_follows_the_gpu(env, monkeypatch, vram, te):
    from lorastudio import addons
    _gpu(monkeypatch, vram)
    assert addons.recommend(addons.catalog_entry("qwen_image21_gen"), "te") == te


def _fake_fetch(monkeypatch):
    from lorastudio import addons

    def dl(repo, path, dest, token, st, gb=None):
        if not dest.exists():
            _st(dest, QWEN if "diffusion_models" in str(dest) and dest.suffix == ".safetensors" else {"x": "BF16"})
        return dest
    monkeypatch.setattr(addons, "_download", dl)
    monkeypatch.setattr(addons, "install_engine", lambda name: None)
    monkeypatch.setattr(addons, "fetch_tokenizers", lambda repos: None)


@pytest.mark.parametrize("order", [("qwen_image_2512_gen", "qwen_image_2512"), ("qwen_image_2512", "qwen_image_2512_gen")])
def test_light_and_training_files_become_one_model(env, monkeypatch, order):
    from lorastudio import addons, models
    _gpu(monkeypatch, 16376)
    _fake_fetch(monkeypatch)
    for cid in order:
        entry = addons.catalog_entry(cid)
        st = {"state": "running", "done_bytes": 0, "total_bytes": 0, "error": None}
        addons._work(entry, addons.resolve_files(entry), None, st)
        assert st["state"] == "done", st
    reg = models.registry.list()
    assert len(reg) == 1
    f = reg[0]["files"]
    assert f["gen"].endswith("qwen-image-2512-Q4_K_M.gguf") and f["ckpt"].endswith("qwen_image_2512_bf16.safetensors")
    assert f["te"].endswith("qwen_2.5_vl_7b_fp8_scaled.safetensors")       # generation: the light encoder
    assert f["train_te"].endswith("qwen_2.5_vl_7b.safetensors")            # training: bf16
    assert all(addons.installed(addons.catalog_entry(c)) for c in order)


def test_training_files_alone_can_generate_and_light_alone_is_not_training(env, monkeypatch):
    from lorastudio import addons, models, training
    _gpu(monkeypatch, 16376)
    _fake_fetch(monkeypatch)
    entry = addons.catalog_entry("qwen_image_2512")
    addons._work(entry, addons.resolve_files(entry), None, {"state": "running", "done_bytes": 0, "total_bytes": 0, "error": None})
    m = models.registry.list()[0]
    assert m["files"]["te"] == m["files"]["train_te"]                      # the bf16 encoder serves generation too
    assert not addons.installed(addons.catalog_entry("qwen_image_2512_gen"))
    assert training._family_file_errors(m) == []
    light = {"family": "qwen_image", "files": {"gen": "x/qwen-image-2512-Q4_K_M.gguf", "te": "x/qwen_2.5_vl_7b_fp8_scaled.safetensors"}}
    assert any("学習用ファイル" in e for e in training._family_file_errors(light))


def test_qwen_training_uses_the_bf16_encoder(env):
    import toml
    from lorastudio import presets, training
    run_dir = env / "run 1"
    run_dir.mkdir()
    files = {"ckpt": "m/qwen_image_2512_bf16.safetensors", "gen": "m/qwen-image-2512-Q4_K_M.gguf",
             "te": "m/qwen_2.5_vl_7b_fp8_scaled.safetensors", "train_te": "m/qwen_2.5_vl_7b.safetensors", "vae": "m/vae.safetensors"}
    v = presets.defaults("qwen_image", "character", "qwen_image", 20, 16376)
    snap = {"model": {"family": "qwen_image", "variant": "qwen_image", "files": files}, "values": v, "output_name": "x", "items": []}
    cmds = training.build_commands(snap, run_dir, None)
    te_cache = cmds[1][1]
    assert te_cache[te_cache.index("--text_encoder") + 1] == files["train_te"]
    t = toml.load(cmds[-1][1][-1])
    assert t["dit"] == files["ckpt"] and t["text_encoder"] == files["train_te"] and t["fp8_base"] and t["fp8_scaled"]


def test_generation_uses_the_light_file(env, monkeypatch):
    from lorastudio import evaluate
    monkeypatch.setattr(evaluate.comfy, "stage_model", lambda kind, p: Path(p).name)
    files = {"ckpt": "m/qwen_image_2512_bf16.safetensors", "gen": "m/qwen-image-2512-Q4_K_M.gguf",
             "te": "m/qwen_2.5_vl_7b_fp8_scaled.safetensors", "vae": "m/vae.safetensors"}
    s = evaluate.stage_files({"family": "qwen_image", "files": files})
    assert s["on"] == "gen" and s["gguf"] and "dtype" not in s
    g = evaluate.build_t2i({"family": "qwen_image", "files": files}, s, [], "a", "", 1, 1328, 1328,
                           evaluate.SAMPLING["qwen_image"], "p")
    assert g["1"]["class_type"] == "UnetLoaderGGUF"


def test_no_experimental_marks_left(env):
    from lorastudio import models
    # 2026-11-07: no model is marked 試験導入・動作未確認; the lists show 検証済み once it worked on the user's PC
    for v in models.VARIANTS:
        assert not models.VARIANTS[v].get("experimental") and "試験" not in models.VARIANTS[v]["label"]
        assert "未確認" not in models.VARIANTS[v]["label"]
        assert models.experimental_note(v) is None
    for c in models.CATALOG:
        assert not c.get("experimental") and "試験" not in c.get("note", "") and "確認していません" not in c.get("note", "")
        assert "動作を確認していない" not in c.get("note", "")


def test_worked_once_is_recorded(env):
    from lorastudio import config, models
    config.settings().set("models", [{"id": "a", "name": "A", "variant": "moss_image_sft", "family": "moss_image", "files": {}}])
    models.registry.mark_checked("a", "gen")
    first = models.registry.get("a")["checked"]["gen"]
    models.registry.mark_checked("a", "gen", "lora")
    c = models.registry.get("a")["checked"]
    assert c["gen"] == first and "lora" in c
    models.registry.mark_checked(None, "gen")   # a workflow with no registered model: nothing


def test_system_ram_is_read():
    from lorastudio.gpu import system_ram_gb
    ram = system_ram_gb()
    assert ram is None or ram > 1


@pytest.mark.parametrize("vram, pick", [(8188, "int8"), (16376, "int8"), (24564, "bf16")])
def test_qwen21_training_files_follow_the_gpu(env, monkeypatch, vram, pick):
    """Up to 20GB the ComfyUI INT8 files are fetched for training (half the size); the INT8 text encoder is the very
    file the generation entry fetches above 10GB, so it is not downloaded twice."""
    from lorastudio import addons
    _gpu(monkeypatch, vram)
    train = addons.resolve_files(addons.catalog_entry("qwen_image21_train"))
    assert (pick in train["ckpt"][1]) and (pick in train["train_te"][1])
    gen = addons.resolve_files(addons.catalog_entry("qwen_image21_gen"))
    if vram >= 10240 and pick == "int8":
        assert addons._save_name(gen["te"]) == addons._save_name(train["train_te"])


def test_qwen21_training_reads_bf16_or_int8_text_encoders_only(env):
    from lorastudio import training
    te = training.qwen21_train_te
    assert te({"te": "x/qwen3vl_8b_int8_convrot.safetensors"}).endswith("int8_convrot.safetensors")
    assert te({"te": "x/qwen3vl_8b_bf16.safetensors"}).endswith("bf16.safetensors")
    assert te({"te": "x/qwen3vl_8b_w4a8.safetensors"}) is None   # generation only
    assert te({"te": "x/qwen3vl_8b_w4a8.safetensors", "train_te": "y/t.safetensors"}) == "y/t.safetensors"
