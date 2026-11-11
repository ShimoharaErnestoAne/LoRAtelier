"""2026-10-30 speed / memory changes.  Failure risks: a run slowed down by VRAM spilling into system memory with no
word about it (or a warning for an ordinary checkpoint save), the next start repeating the same settings silently,
a parked ComfyUI holding the models in RAM when memory is short (or dropping them when it is not), the i2t VLM left
in VRAM while training runs, and loader processes the engines no longer get."""
from __future__ import annotations

import importlib.util
import io
import json
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


# ------------------------------------------------------------------ 1. VRAM shortage seen in the step time
class _Clock:
    t = 1000.0


class _Stream:
    """Engine output, one progress line per read; the clock moves by that step's duration."""

    def __init__(self, durations, total):
        self.lines = []
        for i, d in enumerate(durations, start=1):
            self.lines.append((d, f"steps:  {i}%|#| {i}/{total} [00:10<01:30,  1.00s/it, avr_loss=0.1]\r".encode()))

    def read1(self, _n):
        if not self.lines:
            return b""
        d, line = self.lines.pop(0)
        _Clock.t += d
        return line


def _job(durations, family="sdxl", values=None):
    from lorastudio import training
    values = values or {"train_batch_size": 2, "resolution": 1024}
    job = training.Job(SimpleNamespace(id="p", q=lambda *a: None, checkpoints=lambda _r: []), 1)
    job.vram_ctx = (family, values)
    job.proc = SimpleNamespace(stdout=_Stream(durations, len(durations)), wait=lambda: 0, pid=0)
    return job


def _read(env, monkeypatch, job):
    from lorastudio import training
    monkeypatch.setattr(training.time, "time", lambda: _Clock.t)
    r = training.Runner(None)
    r._publish = lambda j: None
    r._set = lambda *a, **k: None
    with open(env / "log.txt", "ab") as logf:
        r._read_engine(job, "training", len(job.proc.stdout.lines), env / "no_output", logf, "state")
    return job.progress


def test_a_lasting_slowdown_warns_with_advice_for_the_family(env, monkeypatch):
    p = _read(env, monkeypatch, _job([1.0] * 60 + [10.0] * 40))
    assert p["slow_x"] == 10.0
    assert "10 分の 1" in p["warning"] and "バッチサイズを 1 にする" in p["warning"]
    assert "ブロックスワップ" not in p["warning"]                  # SDXL training has no block swap
    assert p["vram_short"] == {"train_batch_size": 2, "resolution": 1024}
    assert p["speed"] == 10.0                                   # the remaining time follows the slow speed


def test_one_slow_window_such_as_a_save_is_not_a_warning(env, monkeypatch):
    p = _read(env, monkeypatch, _job([1.0] * 30 + [30.0] + [1.0] * 40))
    assert "warning" not in p and "vram_short" not in p and "slow_x" not in p


def test_block_swap_comes_first_where_the_engine_has_it(env):
    from lorastudio import training
    tip = training.vram_advice("anima", {"blocks_to_swap": 10, "train_batch_size": 1, "resolution": 1024})
    assert tip.startswith("ブロックスワップ数を増やす（今は 10、最大 26") and "バッチサイズ" not in tip
    assert "ブロックスワップ" not in training.vram_advice("anima", {"blocks_to_swap": 26})


def _add_run(proj, name, model, progress):
    with proj.tx() as db:
        db.execute("INSERT INTO runs(created, status, name, config, run_dir, progress) VALUES(?,?,?,?,?,?)",
                   (0, "done", name, json.dumps({"model": {"id": model}, "items": []}), "", json.dumps(progress)))


def test_next_start_warns_until_the_settings_need_less_vram(env):
    from lorastudio import store, training
    short = {"train_batch_size": 2, "resolution": 1024}
    proj = store.ProjectManager().create("vram", "character", None)
    _add_run(proj, "#1", "m", {"step": 900})
    _add_run(proj, "#2", "m", {"step": 40, "vram_short": short})
    _add_run(proj, "#3", "m", {"step": 0})   # not started: skipped
    assert training.last_vram_short(proj, "m")["name"] == "#2"
    assert training.last_vram_short(proj, "other") is None
    assert not training._vram_lighter({"train_batch_size": 2, "resolution": 1024}, short)
    assert training._vram_lighter({"train_batch_size": 1, "resolution": 1024}, short)
    assert training._vram_lighter({"train_batch_size": 2, "resolution": 896}, short)
    assert training._vram_lighter({"blocks_to_swap": 4}, {"blocks_to_swap": 0})
    # a later run of the same model that went well: the shortage was dealt with
    _add_run(proj, "#4", "m", {"step": 900})
    assert training.last_vram_short(proj, "m") is None


# ------------------------------------------------------------------ 4. no loader processes
def test_runs_read_their_caches_without_loader_processes(env):
    from lorastudio import training
    v = {"max_train_steps": 100, "save_every_n_steps": 50, "seed": 1, "learning_rate": 1e-4, "network_dim": 8,
         "network_alpha": 8, "optimizer_type": "AdamW8bit", "lr_scheduler": "constant"}
    cfg = training._common_save(v, env, "x")
    assert cfg["max_data_loader_n_workers"] == 0
    assert "persistent_data_loader_workers" not in cfg   # torch refuses persistent workers without workers


# ------------------------------------------------------------------ 2. parked ComfyUI keeps its models in RAM
def _comfy(monkeypatch):
    from lorastudio import evaluate
    c = evaluate.Comfy()
    posts = []
    monkeypatch.setattr(c, "running", lambda: True)
    monkeypatch.setattr(c, "_post", lambda path, data, timeout=30: posts.append((path, data)))
    monkeypatch.setattr(c, "_arm_idle", lambda keep_s: None)
    return evaluate, c, posts


def test_park_frees_vram_but_keeps_the_loaded_models(env, monkeypatch):
    evaluate, c, posts = _comfy(monkeypatch)
    trims = []
    monkeypatch.setattr(c, "_trim_ram", lambda uses, *a: trims.append(uses))
    c.park()
    assert posts == [("/free", {"unload_models": True})] and len(trims) == 1


def test_models_are_dropped_only_when_system_memory_is_short(env, monkeypatch):
    evaluate, c, posts = _comfy(monkeypatch)
    monkeypatch.setattr(evaluate, "free_ram_gb", lambda: 20.0)
    c._trim_ram(c._uses, wait_s=0)
    assert posts == []
    monkeypatch.setattr(evaluate, "free_ram_gb", lambda: 3.0)
    c._trim_ram(c._uses - 1, wait_s=0)                          # a job used it again meanwhile: its models now
    assert posts == []
    c._trim_ram(c._uses, wait_s=0)
    assert posts == [("/free", {"unload_models": True, "free_memory": True})]
    posts.clear()
    monkeypatch.setattr(evaluate, "free_ram_gb", lambda: None)   # unreadable: kept, as before a reading existed
    c._trim_ram(c._uses, wait_s=0)
    assert posts == []


# ------------------------------------------------------------------ 3. i2t VLM kept between images
def _worker(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["w", str(tmp_path), str(tmp_path / "d"), str(tmp_path / "m")])
    spec = importlib.util.spec_from_file_location("i2t_worker_test", ROOT / "worker" / "i2t_worker.py")
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    return w


def test_worker_keeps_the_vlm_only_while_low_vram_is_on(env, monkeypatch):
    w = _worker(monkeypatch, env)
    sent = []

    class VLM:
        host, model, keep_alive = "http://127.0.0.1:11434", "qwen3-vl:8b", 0

        def _chat(self, payload):
            sent.append(dict(payload))
            return {}

    low = {"on": True}
    vlm = VLM()
    eng = SimpleNamespace(_vlm=None, _low_vram=lambda: low["on"])
    eng.vlm = lambda: vlm
    w.engine = eng
    w._wrap_engine_vlm(eng)
    eng.vlm()._chat({"keep_alive": 0})              # i2t's main request
    eng.vlm()._chat({})                             # a detail request (i2t sends no keep_alive there)
    assert [p["keep_alive"] for p in sent] == ["2m", "2m"]
    eng._vlm = vlm
    assert w._kept_vlm() == [{"host": "http://127.0.0.1:11434", "model": "qwen3-vl:8b", "low_vram": True}]
    low["on"] = False                               # setting turned off: i2t's own keep_alive is sent untouched
    eng.vlm()._chat({"x": 1})
    # ...and the model Ollama then keeps by its own rule is still reported (unloaded for training / generation only)
    assert "keep_alive" not in sent[-1]
    assert w._kept_vlm() == [{"host": "http://127.0.0.1:11434", "model": "qwen3-vl:8b", "low_vram": False}]


def test_server_unloads_the_kept_vlm_when_the_gpu_is_taken(env, monkeypatch):
    from lorastudio import i2tlink
    from lorastudio.gpu import gpu
    calls = []

    class R:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"{}"

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=10: calls.append((req.full_url, json.loads(req.data))) or R())
    i2tlink.client.kept_vlm = [{"host": "http://127.0.0.1:11434", "model": "qwen3-vl:8b", "low_vram": True}]
    gpu.acquire("学習 #1")
    try:
        assert calls == [("http://127.0.0.1:11434/api/generate", {"model": "qwen3-vl:8b", "keep_alive": 0})]
        assert i2tlink.client.kept_vlm == []
    finally:
        gpu.release("学習 #1")
    gpu.begin_analysis()                            # nothing kept any more: no request
    gpu.end_analysis()
    assert len(calls) == 1


def test_an_analysis_that_ends_during_training_unloads_at_once(env, monkeypatch):
    from lorastudio import i2tlink
    from lorastudio.gpu import gpu
    done = []
    monkeypatch.setattr(i2tlink, "unload_kept_vlm", lambda: done.append(1))
    monkeypatch.setattr(i2tlink.bus, "publish", lambda *a, **k: None)
    c = i2tlink.I2tClient()
    c.job = {"rid": 5, "op": "analyze"}
    msg = {"rid": 5, "ok": True, "vlm_kept": {"host": "http://h", "model": "m"}}
    proc = SimpleNamespace(stdout=io.BytesIO((json.dumps(msg) + "\n").encode()))
    c.proc = proc
    gpu.owner = "学習 #2"
    try:
        c._reader(proc)
    finally:
        gpu.owner = None
    import time
    for _ in range(50):
        if done:
            break
        time.sleep(0.02)
    assert done == [1] and c.kept_vlm == [{"host": "http://h", "model": "m"}]
