"""Speed-ups that must not change what is trained or generated.  Failure risks: a stored encoder cache used for a
different image, crop, caption or model (silently training on the wrong data), a cache file the engine was still
writing kept in the store, a stale converted LoRA used after the run wrote a newer one, and ComfyUI / the analysis
worker kept in VRAM while training runs."""
from __future__ import annotations

import os
import sys
import time
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


def _snap(tmp, engine="sd-scripts", caption="chr, smile", crop=None, model_file="m.safetensors", lr=4e-4):
    mf = tmp / model_file
    if not mf.exists():
        mf.write_bytes(b"model")
    return {"model": {"family": "sdxl", "variant": "illustrious", "files": {"ckpt": str(mf)}}, "engine": engine,
            "engine_commit": "abc", "values": {"resolution": 1024, "learning_rate": lr},
            "items": [{"id": 12, "sha": "s" * 40, "crop": crop, "caption": caption, "file": "00012.png"}]}


def test_caches_are_reused_only_for_the_same_image_caption_and_model(env):
    from lorastudio import traincache
    proj = env / "proj"
    run1, run2 = env / "r1", env / "r2"
    s1 = _snap(env)
    k1 = traincache.group_key(s1)
    d1 = traincache.run_cache_dir(run1, s1)
    d1.mkdir(parents=True)
    (d1 / "00012_1024x1024_sdxl.npz").write_bytes(b"latent")
    (d1 / "00012_te_outputs.npz").write_bytes(b"te")
    (d1 / "00012_1024x1024_sdxl.npz.part").write_bytes(b"half")      # never stored
    (d1 / "00012.txt").write_text("chr, smile", encoding="utf-8")       # not a cache file
    assert traincache.harvest(proj, run1, s1) == 2
    # another learning rate: same caches
    s2 = _snap(env, lr=1e-4)
    assert traincache.group_key(s2) == traincache.group_key(s1)
    assert traincache.prime(proj, run2, s2) == 2
    d2 = traincache.run_cache_dir(run2, s2)
    assert (d2 / "00012_1024x1024_sdxl.npz").read_bytes() == b"latent" and (d2 / "00012_te_outputs.npz").read_bytes() == b"te"
    # a copy, not a link: the engine rewriting the run's file leaves the store alone
    (d2 / "00012_te_outputs.npz").write_bytes(b"rewritten")
    run3 = env / "r3"
    traincache.prime(proj, run3, s2)
    assert (traincache.run_cache_dir(run3, s2) / "00012_te_outputs.npz").read_bytes() == b"te"
    # changed caption: the latent is reused, the text-encoder output is not
    run4 = env / "r4"
    assert traincache.prime(proj, run4, _snap(env, caption="chr, sad")) == 1
    assert not (traincache.run_cache_dir(run4, s1) / "00012_te_outputs.npz").exists()
    # changed crop: nothing
    assert traincache.prime(proj, env / "r5", _snap(env, crop=[0, 0, 10, 10])) == 0
    # another model file or another resolution: another group
    assert traincache.group_key(_snap(env, model_file="other.safetensors")) != traincache.group_key(s1)
    s6 = _snap(env)
    s6["values"]["resolution"] = 768
    assert traincache.group_key(s6) != traincache.group_key(s1)
    # the model file replaced in place (same path, new content): another group
    time.sleep(0.01)
    (env / "m.safetensors").write_bytes(b"model v2!")
    os.utime(env / "m.safetensors", (time.time() + 5, time.time() + 5))
    assert traincache.group_key(_snap(env)) != k1
    # texture LoRA runs never share
    assert traincache.group_key(dict(s1, enhance={"method": "pair"})) is None


def test_musubi_caches_go_through_the_cache_folder_and_skip_existing(env):
    from lorastudio import traincache
    s = _snap(env, engine="musubi-tuner")
    assert traincache.run_cache_dir(env / "r", s) == env / "r" / "cache"
    cmd = ("preparing", ["py", "zimage_cache_text_encoder_outputs.py", "--dataset_config", "x"], Path("."))
    assert traincache.with_skip_existing(cmd)[1][-1] == "--skip_existing"
    train = ("training", ["py", "-m", "accelerate.commands.launch", "zimage_train_network.py"], Path("."))
    assert traincache.with_skip_existing(train) == train
    d = traincache.run_cache_dir(env / "r", s)
    d.mkdir(parents=True)
    (d / "00012_1024x1024_zimage.safetensors").write_bytes(b"l")
    (d / "00012_zimage_te.safetensors").write_bytes(b"t")
    assert traincache.harvest(env / "p", env / "r", s) == 2
    st = traincache.store_dir(env / "p", traincache.group_key(s))
    assert len(list((st / "T").iterdir())) == 1 and len(list((st / "L").iterdir())) == 1


def test_converted_lora_is_reused_until_the_run_writes_a_newer_one(env, monkeypatch):
    from lorastudio import evaluate
    calls = []
    real = evaluate._convert_lora
    monkeypatch.setattr(evaluate, "_convert_lora", lambda fam, s, d: calls.append(d) or real(fam, s, d))
    src = env / "run" / "x.safetensors"
    src.parent.mkdir()
    src.write_bytes(b"v1")
    dest = env / "loras" / "x.safetensors"
    evaluate.comfy_lora_file("sdxl", src, dest)
    evaluate.comfy_lora_file("sdxl", src, dest)
    assert len(calls) == 1 and dest.read_bytes() == b"v1" and not list(dest.parent.glob("*.part*"))
    src.write_bytes(b"v2")
    os.utime(src, (time.time() + 5, time.time() + 5))
    evaluate.comfy_lora_file("sdxl", src, dest)
    assert len(calls) == 2 and dest.read_bytes() == b"v2"


def test_training_frees_comfyui_and_the_worker_other_jobs_keep_them(env, monkeypatch):
    from lorastudio import analysis, evaluate
    from lorastudio.gpu import gpu
    got = []
    monkeypatch.setattr(evaluate.comfy, "stop", lambda: got.append("comfy.stop"))
    monkeypatch.setattr(analysis.worker, "release_vram", lambda: got.append("worker.quit"))
    monkeypatch.setattr(analysis.worker, "unload", lambda: got.append("worker.unload"))
    for owner, want in (("学習 #3", ["comfy.stop", "worker.quit"]), ("比較 #3", ["worker.unload"]),
                        ("素材生成 #2", ["worker.unload"]), ("生成候補のそろい具合の確認", ["comfy.stop"])):
        got.clear()
        gpu.owner = owner
        evaluate._stop_comfy_for_exclusive()
        analysis._release_worker_vram()
        assert got == want, owner
    gpu.owner = None
