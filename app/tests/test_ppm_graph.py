"""ComfyUI-ppm in the generation graph: NegPip only where it changes something (a negative weight, any weight on
MOSS-Image), the left / right prompts (Attention Couple) on SDXL / Anima only, RescaleCFGPost when asked, and
nothing of it when ComfyUI-ppm is not loaded.  Graph shape only (the effect is unverified on a GPU)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lorastudio.evaluate import build_t2i, wants_negpip  # noqa: E402

S = {"steps": 20, "cfg": 5.0, "sampler": "euler", "scheduler": "normal"}


def _g(fam, ppm=True, prompt="1girl", negative="lowres", regions=None, **samp):
    model = {"family": fam, "variant": {"sdxl": "illustrious"}.get(fam, fam)}
    staged = {"ckpt": "m.safetensors", "dit": "d.safetensors", "te": "t.safetensors", "vae": "v.safetensors", "ppm": ppm}
    return build_t2i(model, staged, [("l.safetensors", 0.8)], prompt, negative, 1, 1024, 1024, {**S, **samp}, "x",
                     lora_clip=True, regions=regions)


def types(g):
    return [n["class_type"] for n in g.values()]


def test_negpip_only_where_it_changes_something():
    assert wants_negpip("sdxl", "1girl, (hat:-1)", "") and wants_negpip("anima", "a", "(blur:- 0.5)")
    assert not wants_negpip("sdxl", "1girl, (hat:1.2)", "lowres")      # usual weights: ComfyUI's own encoding
    assert wants_negpip("moss_image", "a girl (smiling:1.3)", "")       # MOSS-Image ignores weights without it
    assert not wants_negpip("qwen_image21", "(hat:-1)", "")
    g = _g("sdxl", prompt="1girl, (hat:-1)")
    n = next(k for k, v in g.items() if v["class_type"] == "CLIPNegPip")
    assert g[n]["inputs"]["model"] == ["L0", 0] and g[n]["inputs"]["clip"] == ["L0", 1]   # after the LoRA
    assert g["3"]["inputs"]["clip"] == [n, 1] and g["6"]["inputs"]["model"] == [n, 0]
    assert "CLIPNegPip" not in types(_g("sdxl", prompt="1girl, (hat:-1)", ppm=False))


def test_left_and_right_prompts():
    g = _g("sdxl", prompt="2girls, park", regions=["red hair", "blue hair"], rescale="on")
    c = next(v for v in g.values() if v["class_type"] == "AttentionCouplePPM")["inputs"]
    assert c["base_cond"] == ["3", 0] and set(c) == {"model", "base_cond", "base_mask", "cond_1", "mask_1", "cond_2", "mask_2"}
    assert g[c["cond_1"][0]]["inputs"]["text"] == "2girls, park, red hair"
    masks = [g[c[k][0]]["inputs"] for k in ("base_mask", "mask_1", "mask_2")]
    assert [(m["x"], m["w"], m["value"]) for m in masks] == [(0.0, 1.0, 0.6), (0.0, 0.5, 0.4), (0.5, 0.5, 0.4)]
    assert g["6"]["inputs"]["model"][0] == next(k for k, v in g.items() if v["class_type"] == "RescaleCFGPost")
    one = _g("anima", regions=["", "blue hair"])                          # one side only: that half alone
    c = next(v for v in one.values() if v["class_type"] == "AttentionCouplePPM")["inputs"]
    assert set(c) == {"model", "base_cond", "base_mask", "cond_1", "mask_1"} and one[c["mask_1"][0]]["inputs"]["x"] == 0.5
    for fam in ("qwen_image21", "moss_image"):                            # not supported there
        assert "AttentionCouplePPM" not in types(_g(fam, regions=["a", "b"]))
    assert not {"AttentionCouplePPM", "RescaleCFGPost"} & set(types(_g("sdxl", ppm=False, regions=["a", "b"], rescale="on")))


def test_bundled_copy_is_put_in_place_without_git(tmp_path, monkeypatch):
    """The zip carries ComfyUI-ppm in engines/bundled (not under engines/ComfyUI, which setup clones into)."""
    from lorastudio import addons, config
    monkeypatch.setattr(config, "ROOT", tmp_path)
    lock = {"engines": {"ComfyUI-ppm": {"url": "x", "commit": "abc", "path": "engines/ComfyUI/custom_nodes/ComfyUI-ppm"}}}
    monkeypatch.setattr(addons, "_lockfile", lambda: lock)
    monkeypatch.setattr(addons, "install_engine", lambda name: (_ for _ in ()).throw(AssertionError("no git")))
    monkeypatch.setattr(addons, "_tried", set())
    addons._ready.pop("ComfyUI-ppm", None)
    src = tmp_path / "engines" / "bundled" / "ComfyUI-ppm"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / addons.BUNDLE_MARK).write_text("abc\n", encoding="utf-8")
    assert not addons.ppm_ready()
    assert addons.ensure_ppm()
    assert (tmp_path / "engines" / "ComfyUI" / "custom_nodes" / "ComfyUI-ppm" / "__init__.py").is_file()
    lock["engines"]["ComfyUI-ppm"]["commit"] = "newer"                  # the lock moved on: not ready any more
    addons._ready.pop("ComfyUI-ppm", None)
    assert not addons.ppm_ready()
    addons._ready.pop("ComfyUI-ppm", None)


def test_a_bf16_model_larger_than_the_gpu_is_loaded_as_fp8(tmp_path, monkeypatch):
    """Krea 2 Turbo bf16 (26GB) on a 16GB card was streamed from main memory every step; it is loaded as fp8."""
    import json
    import struct
    from lorastudio import evaluate, gpu

    def st(name, dtype):
        raw = json.dumps({"a.weight": {"dtype": dtype, "shape": [1], "data_offsets": [0, 0]}}).encode()
        p = tmp_path / name
        p.write_bytes(struct.pack("<Q", len(raw)) + raw)
        return str(p)
    bf16, fp8 = st("krea2_turbo_bf16.safetensors", "BF16"), st("krea2_turbo_fp8_scaled.safetensors", "F8_E4M3")
    monkeypatch.setattr(gpu, "query_gpus", lambda: [{"vram_total_mb": 2048}])   # the file is "larger" than this GPU
    assert evaluate.load_as_fp8("krea2", bf16)
    assert not evaluate.load_as_fp8("krea2", fp8)                                # already small: as it is
    assert not evaluate.load_as_fp8("sdxl", bf16) and not evaluate.load_as_fp8("anima", bf16)
    monkeypatch.setattr(gpu, "query_gpus", lambda: [{"vram_total_mb": 98304}])  # fits a 96GB card: full precision
    assert not evaluate.load_as_fp8("krea2", bf16)
    assert evaluate.load_as_fp8("qwen_image", bf16)                              # Qwen-Image bf16: always, as before


def test_krea2_generation_uses_the_fp8_text_encoder_when_it_is_there(tmp_path, monkeypatch):
    from lorastudio import config, evaluate
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    te_dir = tmp_path / "models" / "text_encoders"
    te_dir.mkdir(parents=True)
    bf16 = te_dir / "qwen3vl_4b_bf16.safetensors"
    bf16.write_bytes(b"x")
    assert evaluate.gen_text_encoder("krea2", str(bf16)) == str(bf16)          # no fp8 one: as registered
    fp8 = te_dir / "sub" / "qwen3vl_4b_fp8_scaled.safetensors"
    fp8.parent.mkdir()
    fp8.write_bytes(b"x")
    assert evaluate.gen_text_encoder("krea2", str(bf16)) == str(fp8)
    assert evaluate.gen_text_encoder("zimage", str(bf16)) == str(bf16)         # other families: unchanged
