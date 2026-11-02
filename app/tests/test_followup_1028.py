"""Follow-ups of 2026-10-28.  Failure risks: turning training caches off deleting images, captions or LoRA files, or
the caches of the run that is training now; caches still stored with the setting off; Krea 2 offered for training
on a Turbo / quantized file it cannot train on."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

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


def _write(p: Path, data=b"x" * 10) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def test_turning_training_caches_off_deletes_only_caches(env, monkeypatch):
    from lorastudio import server, traincache
    assert traincache.keep()   # kept unless the user turns it off
    root = server.projects.root
    pa = root / "a"
    _write(pa / "project.db")
    img, cap = _write(pa / "runs" / "0001_x" / "dataset" / "1_k" / "a.png"), _write(pa / "runs" / "0001_x" / "dataset" / "1_k" / "a.txt")
    npz = _write(pa / "runs" / "0001_x" / "dataset" / "1_k" / "a_1024x1024_sdxl.npz")
    te = _write(pa / "runs" / "0001_x" / "dataset" / "1_k" / "a_te.npz")
    lora = _write(pa / "runs" / "0001_x" / "output" / "x.safetensors")
    store = _write(pa / "traincache" / "k" / "L" / "abc_1024x1024_sdxl.npz")
    busy = _write(pa / "runs" / "0002_y" / "cache" / "a_krea2.safetensors")   # the run training now
    _write(root / "not a project" / "runs" / "0001_x" / "cache" / "z.safetensors")
    monkeypatch.setattr(server.runner, "current", SimpleNamespace(run_id=2, project=SimpleNamespace(id="a")))
    r = server.ui_conf({"keep_train_cache": False})
    assert r["keep_train_cache"] is False and r["freed"] == 30
    assert not npz.exists() and not te.exists() and not store.exists() and not (pa / "traincache").exists()
    assert img.exists() and cap.exists() and lora.exists() and busy.exists()
    assert (root / "not a project" / "runs" / "0001_x" / "cache" / "z.safetensors").exists()
    monkeypatch.setattr(server.runner, "current", None)
    assert server.status()["keep_train_cache"] is False
    # the run training now: its own caches go when it ends
    assert traincache.clear_run(pa / "runs" / "0002_y") == 10 and not (pa / "runs" / "0002_y" / "cache").exists()
    assert server.ui_conf({"keep_train_cache": True})["keep_train_cache"] is True


def test_krea2_is_trainable_and_offered_with_its_training_files():
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import models
    from lorastudio.remote import classify_header
    assert models.trainable("krea2") and "生成のみ" not in models.VARIANTS["krea2"]["label"]
    raw = models.catalog_entry("krea2_raw")
    assert not raw.get("deferred") and raw["merge"] and set(raw["files"]) == {"ckpt", "te", "vae"}
    # a Krea 2 file from a link is accepted (generation files too)
    hdr = {"blocks.0.attn.wq.weight": {"dtype": "F8_E4M3", "shape": [1], "data_offsets": [0, 1]},
           "txtfusion.a.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [1, 3]}}
    c = classify_header(hdr, "krea2_turbo_fp8.safetensors")
    assert c["ok"] and c["dest"] == "diffusion_models" and c["family"] == "krea2"


# ---------------------------------------------------------------- downloads from a link
def _st_bytes(n_data: int) -> bytes:
    import json
    import struct
    hdr = json.dumps({"w": {"dtype": "U8", "shape": [n_data], "data_offsets": [0, n_data]}}).encode()
    return struct.pack("<Q", len(hdr)) + hdr + bytes(n_data)


class _Resp:
    def __init__(self, data: bytes, status=200, headers=None, fail_after=None):
        self.data, self.status, self.headers, self.pos, self.fail_after = data, status, headers or {}, 0, fail_after

    def read(self, n):
        if self.fail_after is not None and self.pos >= self.fail_after:
            raise TimeoutError("timed out")
        out = self.data[self.pos:self.pos + min(n, 1000)]
        self.pos += len(out)
        return out

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_a_download_cut_without_a_length_is_not_placed_and_continues(env, monkeypatch):
    import urllib.error
    from lorastudio import remote
    whole = _st_bytes(5000)
    placed, answers = [], []
    monkeypatch.setattr(remote, "probe", lambda url, name, headers=None: {"ok": True})
    monkeypatch.setattr(remote, "_place", lambda st, tmp, name, hint: (placed.append(tmp.read_bytes()), st.update(state="done")))
    monkeypatch.setattr(remote, "_open", lambda url, headers=None, timeout=30: answers.pop(0)(headers or {}))

    def go():
        st = {"id": "d1", "name": "x.safetensors", "state": "running", "done_bytes": 0, "total_bytes": 0}
        remote._download(st, "https://huggingface.co/x", "x.safetensors", {}, None)
        return st
    # no Content-Length, and the connection ends early: the file is not placed (its header alone looks fine)
    answers.append(lambda h: _Resp(whole[:3000]))
    st = go()
    assert st["state"] == "failed" and "途中で切れました" in st["error"] and not placed
    # pressing again continues from the .part
    answers.append(lambda h: (_Resp(whole[3000:], 206, {"Content-Range": f"bytes 3000-{len(whole) - 1}/{len(whole)}"})
                              if h.get("Range") == "bytes=3000-" else pytest.fail(h)))
    st = go()
    assert st["state"] == "done" and placed == [whole]
    # a timeout: a message in Japanese, the .part kept
    answers.append(lambda h: _Resp(whole, 200, {"Content-Length": str(len(whole))}, fail_after=2000))
    st = go()
    assert st["state"] == "failed" and "接続が途中で切れました" in st["error"]
    # the last try got every byte but stopped before checking: the server answers 416, and the file is placed
    tmp = remote.config.MODELS_DIR / "_incoming" / "d1_x.safetensors.part"
    tmp.write_bytes(whole)

    def r416(h):
        raise urllib.error.HTTPError("u", 416, "Range Not Satisfiable", {}, None)
    answers.append(r416)
    st = go()
    assert st["state"] == "done" and placed[-1] == whole
    # Civitai lists sizes rounded to KB: a whole file a little off that size is not called cut
    assert remote._incomplete(tmp, "x.safetensors", None, len(whole) + 500) is None
    assert remote._incomplete(tmp, "x.safetensors", None, len(whole) + 100000)


# ---------------------------------------------------------------- i2t on a GPU that cannot start
ANALYSIS_PY = ROOT.parent / "envs" / "analysis" / "Scripts" / "python.exe"


@pytest.mark.skipif(not ANALYSIS_PY.exists(), reason="analysis environment not installed")
def test_i2t_models_run_on_the_cpu_when_cuda_fails(tmp_path):
    import json
    import os
    import shutil
    import subprocess
    shutil.copy(ROOT.parent / "engines" / "i2t" / "i2t.py", tmp_path / "i2t.py")   # a copy: nothing is written next to the real one
    script = tmp_path / "run.py"
    script.write_text(r'''
import json, sys
import onnxruntime as ort
sys.path.insert(0, sys.argv[1])
import i2t
calls = []
class Session:
    def __init__(self, path, options, providers):
        calls.append(providers)
        if "CUDAExecutionProvider" in providers:
            raise RuntimeError("CUDA failure 999: unknown error")
ort.InferenceSession = Session
ort.get_available_providers = lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
i2t.make_onnx_session("m.onnx")
print(json.dumps(calls))
''', encoding="utf-8")
    r = subprocess.run([str(ANALYSIS_PY), str(script), str(tmp_path)], capture_output=True, text=True, encoding="utf-8",
                       timeout=300, cwd=str(tmp_path), env={**os.environ, "PYTHONUTF8": "1"})
    assert r.returncode == 0, r.stderr[-2000:]
    assert json.loads(r.stdout.strip().splitlines()[-1]) == [["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"]]
