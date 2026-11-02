"""Speed fixes of 2026-11-01.  Risks: a musubi-tuner cache step left out while some image still lacks its cache
(the run would train without it or fail), and image generation's ComfyUI stopped (or no longer freed from VRAM)
when material analysis starts."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "データ 空白"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config
    config._settings = None
    return tmp_path


def _snap():
    return {"engine": "musubi-tuner", "model": {"family": "qwen_image", "variant": "x", "files": {}},
            "values": {"resolution": 1024}, "items": [{"file": "0001.png", "sha": "a"}, {"file": "0002.png", "sha": "b"}]}


def test_cache_steps_are_left_out_only_when_every_image_has_its_files(env):
    from lorastudio import traincache
    run_dir = env / "run"
    cache = run_dir / "cache"
    cache.mkdir(parents=True)
    cmds = [("preparing", ["py", "qwen_image_cache_latents.py", "--x"], "."),
            ("preparing", ["py", "bridge.py", "qwen_image_cache_text_encoder_outputs.py"], "."),
            ("training", ["py", "train.py"], ".")]
    (cache / "0001_1024x1024_qi.safetensors").write_bytes(b"l")
    (cache / "0001_qi_te.safetensors").write_bytes(b"t")
    (cache / "0002_qi_te.safetensors").write_bytes(b"t")
    (cache / "0002_1024x1024_qi.safetensors.part").write_bytes(b"half")   # an unfinished copy does not count
    done = traincache.complete(run_dir, _snap())
    assert done == {"te"}
    assert [c[1][1] for c in traincache.without_complete(cmds, done)] == ["qwen_image_cache_latents.py", "train.py"]
    (cache / "0002_1024x1024_qi.safetensors").write_bytes(b"l")
    done = traincache.complete(run_dir, _snap())
    assert done == {"latents", "te"} and [c[0] for c in traincache.without_complete(cmds, done)] == ["training"]
    # sd-scripts encodes inside its training process: nothing to leave out
    assert traincache.complete(run_dir, {**_snap(), "engine": "sd-scripts"}) == set()


def test_analysis_parks_the_comfyui_kept_after_image_generation(env, monkeypatch):
    from lorastudio import imagegen
    from lorastudio.gpu import gpu
    calls = []
    monkeypatch.setattr(imagegen.comfy, "park", lambda *a, **k: calls.append("park"))
    monkeypatch.setattr(imagegen.comfy, "stop", lambda *a, **k: calls.append("stop"))
    monkeypatch.setattr(imagegen, "_kept", True)
    assert gpu.begin_analysis(timeout=1)
    gpu.end_analysis()
    assert calls == ["park"] and imagegen._kept is False
    assert gpu.begin_analysis(timeout=1)   # nothing kept any more: nothing to do
    gpu.end_analysis()
    assert calls == ["park"]
