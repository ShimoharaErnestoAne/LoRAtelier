"""Preflight / failure handling: the start button must be blocked (with a reason) instead of
launching a run that is known to fail."""
from __future__ import annotations

import json
import struct
import sys
from collections import namedtuple
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "data dir"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config, models, store, training
    config._settings = None
    monkeypatch.setattr(training, "query_gpus", lambda: [{"name": "fake", "vram_total_mb": 16000, "vram_used_mb": 0, "driver": "x"}])
    return {"tmp": tmp_path, "models": models, "store": store, "training": training, "config": config}


def _st(path: Path, tensors: dict):
    """Write a header-only safetensors file with the given tensor names (no data needed for detect)."""
    hdr = {k: {"dtype": dt, "shape": [1], "data_offsets": [0, 0]} for k, dt in tensors.items()}
    raw = json.dumps(hdr).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    return path


def _project_with_images(env, model_id, n=3):
    from PIL import Image
    import hashlib
    p = env["store"].ProjectManager(env["config"].PROJECTS_DIR).create("pf", "character", model_id)
    imgs = []
    for i in range(n):
        f = env["tmp"] / f"{i}.png"
        Image.new("RGB", (1024, 1024), (i * 40, 0, 0)).save(f)
        iid, _ = p.add_image_file(f, f.name, hashlib.sha256(f.read_bytes()).hexdigest())
        p.set_auto_caption(iid, "x, solo", "k")
        imgs.append(p.image(iid))
    return p, imgs


VALUES = {"max_train_steps": 400, "save_every_n_steps": 100, "seed": 1, "resolution": 1024, "learning_rate": 1e-4,
          "network_dim": 16, "network_alpha": 8, "train_batch_size": 2, "optimizer_type": "AdamW8bit",
          "lr_scheduler": "constant", "caption_dropout_rate": 0.0}


def test_detect_families(env, tmp_path):
    m = env["models"]
    sdxl = _st(tmp_path / "a b/モデル.safetensors", {"model.diffusion_model.x": "F16", "conditioner.embedders.1.model.y": "F16"})
    vpred = _st(tmp_path / "v.safetensors", {"model.diffusion_model.x": "F16", "conditioner.embedders.1.model.y": "F16", "v_pred": "F32"})
    anima = _st(tmp_path / "an.safetensors", {"net.llm_adapter.blocks.0.w": "BF16"})
    krea8 = _st(tmp_path / "k.safetensors", {"blocks.0.attn.wq.weight": "F8_E4M3", "txtfusion.layerwise_blocks.0.w": "BF16"})
    sd15 = _st(tmp_path / "s.safetensors", {"model.diffusion_model.input_blocks.0.weight": "F16"})
    assert m.detect(sdxl)["family"] == "sdxl"
    assert m.detect(vpred)["variant"] == "noobai_vpred"
    assert m.detect(anima)["family"] == "anima"
    d = m.detect(krea8)
    assert d["family"] == "krea2" and d["quantized"] and "RAW" in d["note"]   # generation only: training needs RAW
    assert m.detect(sd15)["family"] is None


def test_preflight_blocks_disk_full(env, monkeypatch):
    ck = _st(env["tmp"] / "m.safetensors", {"model.diffusion_model.x": "F16", "conditioner.embedders.1.model.y": "F16"})
    model = env["models"].registry.add("m", "illustrious", {"ckpt": str(ck)})
    p, imgs = _project_with_images(env, model["id"])
    DU = namedtuple("DU", "total used free")
    monkeypatch.setattr(env["training"].shutil, "disk_usage", lambda _p: DU(10**12, 10**12 - 10**9, 10**9))
    pf = env["training"].preflight(p, imgs, VALUES)
    assert any("空き容量" in e for e in pf["errors"])


def test_krea_trains_only_on_raw_and_missing_files_block(env):
    k = _st(env["tmp"] / "turbo fp8.safetensors", {"blocks.0.attn.wq.weight": "F8_E4M3", "txtfusion.a.w": "BF16"})
    te = _st(env["tmp"] / "te.safetensors", {"x": "BF16"})
    vae = _st(env["tmp"] / "vae.safetensors", {"x": "BF16"})
    krea = env["models"].registry.add("k", "krea2", {"turbo": str(k), "te": str(te), "vae": str(vae)})
    p, imgs = _project_with_images(env, krea["id"])
    assert any("RAW 版の本体" in e for e in env["training"].preflight(p, imgs, VALUES)["errors"])   # Turbo fp8: generation only
    raw = _st(env["tmp"] / "raw.safetensors", {"blocks.0.attn.wq.weight": "BF16", "txtfusion.a.w": "BF16"})
    env["models"].registry.update(krea["id"], files={"turbo": str(k), "ckpt": str(raw), "te": str(te), "vae": str(vae)})
    errs = env["training"].preflight(p, imgs, VALUES)["errors"]
    assert not any("RAW" in e or "対応していません" in e for e in errs), errs
    dit = _st(env["tmp"] / "anima dit.safetensors", {"net.llm_adapter.a": "BF16"})
    model = env["models"].registry.add("a", "anima", {"ckpt": str(dit), "te": str(te), "vae": str(vae)})
    p, imgs = _project_with_images(env, model["id"])
    assert not any("見つかりません" in e for e in env["training"].preflight(p, imgs, VALUES)["errors"])
    vae.unlink()
    assert any("見つかりません" in e for e in env["training"].preflight(p, imgs, VALUES)["errors"])


def test_preflight_rejects_inconsistent_settings(env):
    from lorastudio import presets
    errs = presets.validate("sdxl", dict(VALUES, save_every_n_steps=1000, resolution=1000, learning_rate=1.0))
    assert len(errs) == 3
    assert presets.validate("anima", dict(VALUES, blocks_to_swap=40))


def test_preflight_needs_images(env):
    ck = _st(env["tmp"] / "m.safetensors", {"model.diffusion_model.x": "F16", "conditioner.embedders.1.model.y": "F16"})
    model = env["models"].registry.add("m", "illustrious", {"ckpt": str(ck)})
    p, imgs = _project_with_images(env, model["id"], n=2)
    assert any("3 枚以上" in e for e in env["training"].preflight(p, imgs, VALUES)["errors"])


def test_pending_or_empty_captions_do_not_block_start(env):
    # Captions are optional: the start is allowed with a warning, and the snapshot falls back to the trigger.
    ck = _st(env["tmp"] / "m.safetensors", {"model.diffusion_model.x": "F16", "conditioner.embedders.1.model.y": "F16"})
    model = env["models"].registry.add("m", "illustrious", {"ckpt": str(ck)})
    p, imgs = _project_with_images(env, model["id"])
    assert not env["training"].preflight(p, imgs, VALUES)["errors"]
    p.set_fields(imgs[0]["id"], caption_status="pending", caption="")
    imgs = [p.image(i["id"]) for i in imgs]
    pf = env["training"].preflight(p, imgs, VALUES)
    assert not any("説明文" in e for e in pf["errors"]) and any("まだ終わっていない" in w for w in pf["warnings"])
    p.set_fields(imgs[0]["id"], caption_status="failed")
    imgs = [p.image(i["id"]) for i in imgs]
    pf = env["training"].preflight(p, imgs, VALUES)
    assert not any("説明文" in e for e in pf["errors"]) and any("説明文が空" in w for w in pf["warnings"])
    rid = env["training"].create_run(p, imgs, VALUES, [])
    snap = json.loads((Path(p.run(rid)["run_dir"]) / "run.json").read_text(encoding="utf-8"))
    caps = {it["id"]: it["caption"] for it in snap["items"]}
    assert caps[imgs[0]["id"]] == p.get_meta()["trigger"] and caps[imgs[1]["id"]].endswith("x, solo")


def test_truncated_state_is_not_resumable(env, tmp_path):
    import io as _io
    import zipfile
    tr = env["training"]
    st = tmp_path / "x-step00000400-state"
    st.mkdir()
    (st / "train_state.json").write_text("{}")
    (st / "model.safetensors").write_bytes(b"0")
    buf = _io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("data.pkl", b"x" * 5000)
    (st / "optimizer.bin").write_bytes(buf.getvalue())
    assert tr.state_is_complete(st)
    (st / "optimizer.bin").write_bytes(buf.getvalue()[:3000])   # killed while writing
    assert not tr.state_is_complete(st)
    (st / "optimizer.bin").unlink()
    assert not tr.state_is_complete(st)


def test_preflight_blocks_wrong_file_for_family(env):
    lora = _st(env["tmp"] / "a lora.safetensors", {"lora_unet_down_blocks_0.lora_down.weight": "F16"})
    model = env["models"].registry.add("bad", "illustrious", {"ckpt": str(lora)})
    p, imgs = _project_with_images(env, model["id"])
    errs = env["training"].preflight(p, imgs, VALUES)["errors"]
    assert any("LoRA ファイル" in e for e in errs)
