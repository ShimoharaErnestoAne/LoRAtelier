"""MOSS-Image の生成設定 (2026-11-02): 高速生成の 8 / 4 ステップ LoRA と、Batman MOSS の専用設定。
ComfyUI へ送るグラフの中身 (ステップ・CFG・サンプラー・重ねる LoRA) を偽のファイルで確かめる。GPU・ComfyUI は使わない。"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _st(p: Path, keys):
    h, off = {}, 0
    for k, dt in keys:
        h[k] = {"dtype": dt, "shape": [1], "data_offsets": [off, off + 1]}
        off += 1
    b = json.dumps(h).encode()
    p.write_bytes(struct.pack("<Q", len(b)) + b + b"\0" * off)


@pytest.fixture()
def gen(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "data"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import evaluate, generate
    monkeypatch.setattr(generate, "_stage", lambda kind, src: Path(src).name)   # no hard links into ComfyUI
    monkeypatch.setattr(evaluate.comfy, "ppm", False)
    dit = [("diffusion_model.layers.0.proj.q.weight", "I8"), ("diffusion_model.layers.0.proj.q.weight_scale", "F32")]
    lora = [("diffusion_model.layers.0.proj.q.lora_down.weight", "F32"), ("diffusion_model.layers.0.proj.q.lora_up.weight", "F32")]
    for n in ("batmanMOSS_v30.safetensors", "moss_image_sft_int8_convrot.safetensors"):
        _st(tmp_path / n, dit)
    for n in ("te.safetensors", "vae.safetensors"):
        _st(tmp_path / n, [("x.weight", "BF16")])
    for n in ("moss_image_8steps_dmd2.safetensors", "moss_image_sft_4step_v2.0.safetensors"):
        _st(tmp_path / n, lora)

    def model(ckpt, fast=("fast", "fast4")):
        f = {"ckpt": str(tmp_path / ckpt), "te": str(tmp_path / "te.safetensors"), "vae": str(tmp_path / "vae.safetensors")}
        names = {"fast": "moss_image_8steps_dmd2.safetensors", "fast4": "moss_image_sft_4step_v2.0.safetensors"}
        f.update({slot: str(tmp_path / names[slot]) for slot in fast})
        return {"id": "x", "family": "moss_image", "variant": "moss_image_sft", "files": f}

    def graph(mo, fast=None):
        s = generate.default_sampling(mo, fast)
        g = generate.build_graph(mo, generate.stage_model_files(mo), [], "a girl", "", 1, 1024, 1024, s, "t")
        ks = g["6"]["inputs"]
        loras = [(n["inputs"]["lora_name"], n["inputs"]["strength_model"]) for n in g.values() if n["class_type"].startswith("Lora")]
        return s, (ks["steps"], ks["cfg"], ks["sampler_name"], ks["scheduler"]), loras

    return model, graph


def test_fast_choices_put_the_chosen_lora_and_its_sampling(gen):
    model, graph = gen
    sft = model("moss_image_sft_int8_convrot.safetensors")
    s, ks, loras = graph(sft)
    assert s["fast"] == "on" and s["fast_choices"] == ["on", "4"]           # unset: the 8-step one first
    assert ks == (8, 1.0, "euler", "simple") and loras == [("moss_image_8steps_dmd2.safetensors", 1.0)]
    _, ks, loras = graph(sft, "4")
    assert ks == (4, 1.0, "euler", "simple") and loras == [("moss_image_sft_4step_v2.0.safetensors", 1.0)]
    _, ks, loras = graph(sft, "off")
    assert ks == (25, 2.5, "er_sde", "simple") and loras == []
    s, ks, loras = graph(model("moss_image_sft_int8_convrot.safetensors", fast=("fast4",)))   # only the 4-step file
    assert s["fast_choices"] == ["4"] and ks[0] == 4 and loras == [("moss_image_sft_4step_v2.0.safetensors", 1.0)]


def test_batman_moss_has_its_own_sampling_and_no_fast_lora_on_top(gen):
    model, graph = gen
    s, ks, loras = graph(model("batmanMOSS_v30.safetensors"), "on")
    assert ks == (8, 2.5, "er_sde", "beta")          # the turbo LoRA is merged in; CFG 2.5 is the user's choice
    assert loras == [] and "fast_choices" not in s
