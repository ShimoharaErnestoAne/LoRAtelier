"""Follow-ups of the 2026-10-27 review.  Failure risks: a person mask made for the largest person of the whole
picture while the crop shows another one, images added to a member of a multi-character LoRA trained without
masks, Krea 2's RAW model sampled with the Turbo model's 8 steps without CFG (blurry), and a training blocked by a
missing file that only generation reads."""
from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ANALYSIS_PY = ROOT.parent / "envs" / "analysis" / "Scripts" / "python.exe"
VALUES = {"max_train_steps": 400, "save_every_n_steps": 100, "seed": 1, "resolution": 1024, "learning_rate": 1e-4,
          "network_dim": 16, "network_alpha": 8, "train_batch_size": 1, "optimizer_type": "AdamW8bit",
          "lr_scheduler": "constant"}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "データ 空白"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config, models
    config._settings = None
    return {"tmp": tmp_path, "models": models}


def _st(path: Path, tensors: dict) -> Path:
    hdr = {k: {"dtype": dt, "shape": [1], "data_offsets": [0, 0]} for k, dt in tensors.items()}
    raw = json.dumps(hdr).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    return path


def _sdxl(env):
    f = env["tmp"] / "モデル" / "sdxl.safetensors"
    f.parent.mkdir(exist_ok=True)
    f.write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{}      ")
    return env["models"].registry.add("sdxl", "illustrious", {"ckpt": str(f)})


def _add_image(env, p, i, size=(120, 80)):
    f = env["tmp"] / "素材" / f"{p.id}_{i}.png"
    f.parent.mkdir(exist_ok=True)
    Image.fromarray((np.random.default_rng(i).random((size[1], size[0], 3)) * 255).astype("uint8")).save(f)
    iid, _ = p.add_image_file(f, f.name, hashlib.sha256(f.read_bytes()).hexdigest())
    p.set_fields(iid, ai={"domain": {"label": "illustration"}}, ai_status="done", fast_status="done")
    return iid


# ---------------------------------------------------------------- masks
def test_a_person_mask_follows_the_crop(env, monkeypatch):
    from lorastudio import analysis, masks, server
    p = server.projects.create("キャラ", "character", _sdxl(env)["id"])
    p.set_meta(mask={"on": True, "floor": 0.1})
    iid = _add_image(env, p, 0)
    calls = []

    def fake_call(op, timeout=900, **kw):
        calls.append(kw)
        Image.new("L", (120, 80), 255).save(kw["out"])
        return {"ok": True, "result": {"status": "done", "people": 1, "method": "isnetis", "coverage": 0.4}}
    monkeypatch.setattr(analysis.worker, "call", fake_call)
    queued = []
    monkeypatch.setattr(server.analyzer, "_enqueue", lambda prio, item: queued.append(item))
    server.analyzer._do_mask(p, iid, "person")
    whole = masks.usable(p.image(iid), "person")
    assert calls[-1]["crop"] is None and whole and whole.exists()
    p.edit("crop", [(iid, "crop", [60, 0, 120, 80])])
    assert masks.state(p.image(iid), "person")["status"] == "missing" and masks.usable(p.image(iid), "person") is None
    server.analyzer.ensure_mask(p, iid)   # what the analysis of the new crop asks for
    assert queued == [("mask:person", p.id, iid)]
    server.analyzer._do_mask(p, iid, "person")
    assert calls[-1]["crop"] == [60, 0, 120, 80]
    cropped = masks.usable(p.image(iid), "person")
    assert cropped and cropped != whole and cropped.exists()
    # the background mode masks every person: the crop does not matter
    assert masks.mask_file("ab" * 12, "background", None) == masks.image_mask_file({"sha": "ab" * 12, "crop": [1, 2, 3, 4]},
                                                                                   "background")
    # undoing the crop finds the first mask again (cached), without the worker
    n = len(calls)
    p.undo()
    server.analyzer._do_mask(p, iid, "person")
    assert len(calls) == n and masks.usable(p.image(iid), "person") == whole


def test_images_added_to_a_member_of_a_masked_multi_character_project_get_masks(env, monkeypatch):
    from lorastudio import masks, server
    model = _sdxl(env)
    a = server.projects.create("A", "character", model["id"])
    b = server.projects.create("B", "character", model["id"])
    queued = []
    monkeypatch.setattr(server.analyzer, "_enqueue", lambda prio, item: queued.append(item))
    iid = _add_image(env, a, 1)
    server.analyzer.ensure_mask(a, iid)
    assert queued == []   # no project trains A's images with masks
    multi = server.projects.create("AB", "character", model["id"], source="multi")
    multi.set_meta(members=[a.id, b.id], mask={"on": True, "floor": 0.1})
    server.analyzer._multi_cache = (0.0, [])
    server.analyzer.ensure_mask(a, iid)
    assert queued == [("mask:person", a.id, iid)] and masks.state(a.image(iid), "person")["status"] == "pending"
    assert server.analyzer.mask_mode(b) == "person" and server.analyzer.mask_mode(multi) == "person"
    multi.set_meta(mask={"on": False, "floor": 0.1})
    server.analyzer._multi_cache = (0.0, [])
    assert server.analyzer.mask_mode(a) is None


@pytest.mark.skipif(not ANALYSIS_PY.exists(), reason="analysis environment not installed")
def test_worker_picks_the_person_in_the_crop(tmp_path):
    script = tmp_path / "check.py"
    script.write_text(r'''
import json, sys
import numpy as np
from PIL import Image
sys.path.insert(0, sys.argv[1])
import analysis_worker as w
import imgutils.detect, imgutils.segment
src = sys.argv[2] + "/src.png"
Image.new("RGB", (200, 100), (128, 128, 128)).save(src)
seg = np.zeros((100, 200), np.float32); seg[20:80, 10:50] = 1; seg[10:90, 100:180] = 1
imgutils.segment.get_isnetis_mask = lambda img: seg
imgutils.detect.detect_person = lambda img: [((10, 20, 50, 80), "person", 0.9), ((100, 10, 180, 90), "person", 0.9)]
out = {}
r = w.make_mask(src, "person", "illustration", sys.argv[2] + "/ill.png", [0, 0, 70, 100])
m = np.asarray(Image.open(sys.argv[2] + "/ill.png"))
out["ill"] = [r["status"], r["people"], int(m[50, 30]), int(m[50, 140])]
p1 = np.zeros((100, 200), np.float32); p1[0:50, 0:50] = 1
p2 = np.zeros((100, 200), np.float32); p2[:, 100:200] = 1
w.M.person_masks_photo = lambda img: [p1, p2]
r = w.make_mask(src, "person", "photo", sys.argv[2] + "/photo.png", [0, 0, 60, 60])
m = np.asarray(Image.open(sys.argv[2] + "/photo.png"))
out["photo"] = [r["status"], r["people"], int(m[25, 25]), int(m[50, 150])]
out["empty"] = w.make_mask(src, "person", "photo", sys.argv[2] + "/e.png", [55, 60, 95, 100])["status"]
print(json.dumps(out))
''', encoding="utf-8")
    r = subprocess.run([str(ANALYSIS_PY), str(script), str(ROOT / "worker"), str(tmp_path)], capture_output=True, text=True,
                       encoding="utf-8", timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["ill"] == ["done", 1, 255, 0]      # the smaller person, the one in the crop
    assert out["photo"] == ["done", 1, 255, 0]
    assert out["empty"] == "none"                  # nobody in the crop


# ---------------------------------------------------------------- Krea 2 RAW
def test_krea2_raw_is_sampled_with_full_steps_and_cfg(env):
    from lorastudio import evaluate, generate
    raw = _st(env["tmp"] / "m" / "raw.safetensors", {"txtfusion.a.w": "BF16"})
    m = {"family": "krea2", "variant": "krea2", "files": {"ckpt": str(raw)}}
    s = generate.default_sampling(m)
    assert (s["steps"], s["cfg"]) == (28, 5.5)
    assert evaluate.sampling_for(m, {"on": "ckpt"})["cfg"] == 5.5
    turbo = _st(env["tmp"] / "m" / "k2.safetensors", {"txtfusion.a.w": "F8_E4M3"})
    m2 = {**m, "files": {"ckpt": str(raw), "turbo": str(turbo)}}
    assert generate.default_sampling(m2)["steps"] == 8 and evaluate.gen_slot(m2) == "turbo"
    # a Turbo file in the RAW slot is known by its name
    t2 = _st(env["tmp"] / "m" / "krea2_turbo_bf16.safetensors", {"txtfusion.a.w": "BF16"})
    assert generate.default_sampling({**m, "files": {"ckpt": str(t2)}})["steps"] == 8
    # the Turbo file moved away: RAW, with RAW's settings
    turbo.unlink()
    assert evaluate.gen_slot(m2) == "ckpt" and generate.default_sampling(m2)["cfg"] == 5.5


# ---------------------------------------------------------------- generation-only files
def test_a_missing_generation_file_does_not_block_training(env, monkeypatch):
    from lorastudio import config, evaluate, store, training
    monkeypatch.setattr(training, "query_gpus", lambda: [{"name": "fake", "vram_total_mb": 24000, "vram_used_mb": 0, "driver": "x"}])
    t = {"transformer_blocks.0.attn.w": "BF16", "txt_norm.weight": "BF16", "img_in.weight": "BF16"}
    dit = _st(env["tmp"] / "m" / "qwen_image_bf16.safetensors", t)
    gen = _st(env["tmp"] / "m" / "qwen_image_Q4.gguf", {})
    te = _st(env["tmp"] / "m" / "qwen_2.5_vl_7b.safetensors", {"x": "BF16"})
    vae = _st(env["tmp"] / "m" / "qwen_image_vae.safetensors", {"x": "BF16"})
    model = env["models"].registry.add("q", "qwen_image", {"ckpt": str(dit), "gen": str(gen), "te": str(te), "vae": str(vae)})
    p = store.ProjectManager(config.PROJECTS_DIR).create("pf", "character", model["id"])
    imgs = []
    for i in range(3):
        f = env["tmp"] / f"{i}.png"
        Image.new("RGB", (1024, 1024), (i * 40, 0, 0)).save(f)
        iid, _ = p.add_image_file(f, f.name, hashlib.sha256(f.read_bytes()).hexdigest())
        p.set_auto_caption(iid, "x, solo", "k")
        imgs.append(p.image(iid))
    gen.unlink()
    pf = training.preflight(p, imgs, VALUES)
    assert not any("見つかりません" in e for e in pf["errors"]), pf["errors"]
    assert any("学習には使わない生成用のファイル" in w and "qwen_image_Q4.gguf" in w for w in pf["warnings"])
    assert evaluate.gen_slot(env["models"].registry.get(model["id"])) == "ckpt"   # the comparison uses the DiT
    te.unlink()
    assert any("見つかりません" in e for e in training.preflight(p, imgs, VALUES)["errors"])


# ---------------------------------------------------------------- a broken CUDA state
@pytest.mark.skipif(not ANALYSIS_PY.exists(), reason="analysis environment not installed")
def test_the_worker_ends_after_a_cuda_error_so_a_retry_gets_a_new_process(tmp_path):
    script = tmp_path / "run.py"
    script.write_text(r'''
import sys
sys.path.insert(0, sys.argv[1])
import analysis_worker as w
def broken(*a, **k):
    raise RuntimeError("CUDA error: unknown error")
w.analyze = broken
w.main()
''', encoding="utf-8")
    req = '{"op": "analyze", "rid": 1, "path": "x.png"}\n{"op": "analyze", "rid": 2, "path": "x.png"}\n'
    r = subprocess.run([str(ANALYSIS_PY), str(script), str(ROOT / "worker")], input=req, capture_output=True, text=True,
                       encoding="utf-8", timeout=300, env={**os.environ, "PYTHONUTF8": "1"})   # as the app starts it
    msgs = [json.loads(s) for s in r.stdout.splitlines() if s.startswith("{")]
    answers = [m for m in msgs if "rid" in m]
    assert len(answers) == 1 and not answers[0]["ok"] and "GPU が使えない" in answers[0]["error"]   # rid 2 never read


@pytest.mark.skipif(not ANALYSIS_PY.exists(), reason="analysis environment not installed")
def test_a_gpu_that_fails_to_start_leaves_the_worker_on_the_cpu(tmp_path):
    script = tmp_path / "run.py"
    script.write_text(r'''
import os, sys
sys.path.insert(0, sys.argv[1])
import torch
torch.cuda.is_available = lambda: True
def zeros(*a, **k):
    raise RuntimeError("CUDA error: unknown error")
torch.zeros = zeros
import analysis_worker as w
w.main()
print('{"onnx": "%s", "device": "%s"}' % (os.environ.get("ONNX_MODE"), w.M.device))
''', encoding="utf-8")
    req = ('{"op": "caption", "rid": 1, "path": "%s", "instruction": "x"}\n{"op": "quit", "rid": 2}\n'
           % str(tmp_path / "a.png").replace("\\", "/"))
    Image.new("RGB", (8, 8)).save(tmp_path / "a.png")
    r = subprocess.run([str(ANALYSIS_PY), str(script), str(ROOT / "worker")], input=req, capture_output=True, text=True,
                       encoding="utf-8", timeout=300, env={**os.environ, "PYTHONUTF8": "1"})
    msgs = [json.loads(s) for s in r.stdout.splitlines() if s.startswith("{")]
    assert "CUDA error" in msgs[0]["gpu_error"]
    assert not msgs[1]["ok"] and "説明文の生成を止めました" in msgs[1]["error"]   # no 4B VLM crawling on the CPU
    assert msgs[-1] == {"onnx": "cpu", "device": "cpu"}


def test_the_app_tries_the_gpu_again_after_a_while(env, monkeypatch):
    import io
    from types import SimpleNamespace
    from lorastudio import analysis
    w = analysis.WorkerClient()

    def closed(b):
        raise OSError("pipe closed")
    proc = SimpleNamespace(stdout=io.BytesIO(b'{"event": "ready", "gpu_error": "CUDA error: unknown error"}\n'),
                           poll=lambda: None, stdin=SimpleNamespace(write=closed))
    w.proc = proc
    w._reader(proc)
    assert w.gpu_error and w.cpu_since
    quits, starts = [], []
    monkeypatch.setattr(w, "_quit_locked", lambda: (quits.append(1), setattr(w, "proc", None)))
    monkeypatch.setattr(w, "_start", lambda: (starts.append(1), (_ for _ in ()).throw(RuntimeError("stop here"))))
    assert not w.call("analyze")["ok"] and quits == [] and starts == []   # just fell back: this process is kept
    w.proc, w.cpu_since = proc, w.cpu_since - analysis.GPU_RETRY_S - 1
    with pytest.raises(RuntimeError):
        w.call("analyze")
    assert quits == [1] and starts == [1]
