"""Texture / enhancement LoRA: partner images must differ from the material in the intended direction only,
and the engine commands must chain copy -> merge -> training (pair) or call LECO (words) with real files."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import toml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "data dir"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config, enhance, presets, training
    config._settings = None
    return {"tmp": tmp_path, "enhance": enhance, "presets": presets, "training": training}


def _textured(size=512):
    """Skin-like flat tone with fine noise and one strong edge."""
    rng = np.random.default_rng(0)
    a = np.full((size, size, 3), (200, 160, 140), np.float32) + rng.normal(0, 12, (size, size, 1))
    a[:, size // 2:] -= 120   # edge
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _hf(im):
    g = np.asarray(im.convert("L"), np.float32)
    return np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]).mean()


def test_partner_ops_change_the_intended_quality(env):
    en = env["enhance"]
    im = _textured()
    for op in ("smooth", "soften", "blur"):
        out = en.process(im, op)
        assert out.size == im.size and _hf(out) < _hf(im) * 0.7, op
    # smoothing keeps the strong edge (eyes / outlines), unlike a plain blur of the whole image
    sm = np.asarray(en.process(im, "smooth").convert("L"), np.float32)
    col = sm[:, 256 - 3:256 + 3].mean(0)
    assert col[0] - col[-1] > 80
    assert np.asarray(en.process(im, "flatten").convert("L"), np.float32).std() < np.asarray(im.convert("L"), np.float32).std()
    s0 = np.asarray(im.convert("HSV"))[..., 1].mean()
    assert np.asarray(en.process(im, "desaturate").convert("HSV"))[..., 1].mean() < s0
    flat = Image.new("RGB", (256, 256), (128, 128, 128))
    assert _hf(en.process(flat, "grain")) > _hf(flat) + 1
    assert np.array_equal(np.asarray(en.process(flat, "grain")), np.asarray(en.process(flat, "grain")))  # reproducible


def test_clean_keeps_user_words_and_rejects_custom_for_pairs(env):
    en = env["enhance"]
    assert en.clean({"method": "pair", "effect": "custom"})["effect"] == "detail"
    c = en.clean({"method": "words", "effect": "skin", "positive": " a ,b\nc ", "negative": "", "subject": None})
    assert c["positive"] == "a, b, c" and c["negative"] == "" and c["subject"] == en.EFFECTS["skin"]["subject"]


def _snap(method, effect="skin", items=None):
    return {"kind": "enhance", "enhance": {"method": method, "effect": effect, "positive": "detailed skin texture",
                                           "negative": "smooth skin", "subject": "person"},
            "model": {"family": "sdxl", "variant": "illustrious", "files": {"ckpt": "C:/m/base model.safetensors"}},
            "output_name": "enh_r1", "items": items or []}


def test_pair_commands_chain_copy_merge_training(env, tmp_path):
    tr, pr = env["training"], env["presets"]
    run_dir = tmp_path / "run 日本語"
    (run_dir / "output").mkdir(parents=True)
    snap = _snap("pair")
    snap["values"] = pr.defaults("sdxl", "enhance", "illustrious", 20, 16000, "pair")
    cmds = tr.build_commands(snap, run_dir, None)
    assert [c[0] for c in cmds] == ["copy", "merge", "training"]
    copy_cfg = toml.loads((run_dir / "train_copy.toml").read_text(encoding="utf-8"))
    final_cfg = toml.loads((run_dir / "train.toml").read_text(encoding="utf-8"))
    merged = str(run_dir / "copy" / "merged.safetensors")
    assert copy_cfg["pretrained_model_name_or_path"] == "C:/m/base model.safetensors"
    assert final_cfg["pretrained_model_name_or_path"] == merged and merged in cmds[1][1]
    assert str(run_dir / "copy" / "copy.safetensors") in cmds[1][1]
    assert not copy_cfg["save_state"] and not final_cfg["save_state"]
    # "remove" effect: the copy LoRA learns the processed (smoothed) side, the result learns the material
    ds_copy = toml.loads((run_dir / "dataset_copy.toml").read_text(encoding="utf-8"))
    ds = toml.loads((run_dir / "dataset.toml").read_text(encoding="utf-8"))
    assert ds_copy["datasets"][0]["subsets"][0]["image_dir"].endswith("dataset_partner")
    assert ds["datasets"][0]["subsets"][0]["image_dir"].endswith("dataset")
    assert cmds[0][3] == snap["values"]["copy_steps"] and cmds[2][3] == snap["values"]["max_train_steps"]
    # "add" effect (grain): the other way round
    snap = dict(_snap("pair", "grain"), values=snap["values"])
    tr.build_commands(snap, run_dir, None)
    ds_copy = toml.loads((run_dir / "dataset_copy.toml").read_text(encoding="utf-8"))
    assert ds_copy["datasets"][0]["subsets"][0]["image_dir"].endswith("dataset")


def test_pair_materialize_writes_partners_with_captions(env, tmp_path):
    tr = env["training"]
    src = tmp_path / "a.png"
    _textured(256).save(src)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    snap = _snap("pair", items=[{"id": 1, "file": "00001.png", "src": str(src), "caption": "enh_x, 1girl", "crop": None}])
    tr.materialize(run_dir, snap)
    p = run_dir / "dataset_partner" / "00001.png"
    assert p.exists() and (run_dir / "dataset_partner" / "00001.txt").read_text(encoding="utf-8") == "enh_x, 1girl"
    with Image.open(p) as a, Image.open(run_dir / "dataset" / "00001.png") as b:
        assert a.size == b.size and _hf(a) < _hf(b)


def test_words_commands_and_prompt_file(env, tmp_path):
    tr, pr = env["training"], env["presets"]
    run_dir = tmp_path / "run"
    (run_dir / "output").mkdir(parents=True)
    snap = _snap("words")
    snap["values"] = pr.defaults("sdxl", "enhance", "illustrious", 0, 16000, "words")
    assert not pr.validate("sdxl", snap["values"])
    cmds = tr.build_commands(snap, run_dir, None)
    assert len(cmds) == 1 and "sdxl_train_leco.py" in cmds[0][1]
    cfg = toml.loads((run_dir / "train.toml").read_text(encoding="utf-8"))
    pf = toml.loads(Path(cfg["prompts_file"]).read_text(encoding="utf-8"))
    t = pf["targets"][0]
    assert t["target_class"] == "person" and "detailed skin texture" in t["positive"] and "smooth skin" in t["negative"]
    assert pf["guidance_scale"] == snap["values"]["leco_guidance"] and "" in [n.split(", ")[-1] for n in pf["neutrals"]] or pf["neutrals"]
    # sd-scripts' own parser must accept the file (expands to bidirectional pairs)
    sys.path.insert(0, str(ROOT.parent / "engines" / "sd-scripts"))
    try:
        from library.leco_train_util import load_prompt_settings
    except ImportError:
        pytest.skip("sd-scripts dependencies are not importable from this environment")
    settings = load_prompt_settings(cfg["prompts_file"])
    assert len(settings) >= 4


def test_effect_measure_and_recommendation(env, tmp_path):
    en = env["enhance"]
    from lorastudio.evaluate import recommend
    im = _textured(256)
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    en.process(im, "smooth").save(a)
    im.save(b)
    assert en.measure(b, "detail") > en.measure(a, "detail")
    imgs = [{"ckpt": "base", "strength": 0.0}, {"ckpt": "final", "strength": 0.6}, {"ckpt": "final", "strength": 1.0}]
    rec = recommend("enhance", imgs, {"effect": [0.010, 0.019, 0.020], "effect_metric": "detail"})
    assert rec["ckpt"] == "final" and rec["strength"] == 0.6 and not rec["weak"]   # weakest one near the top gain
    rec = recommend("enhance", imgs, {"effect": [0.010, 0.0101, 0.0102], "effect_metric": "detail"})
    assert rec["weak"]
