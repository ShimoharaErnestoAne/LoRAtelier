"""Job control failure risks: a failing task must not stop the queue, a resume must not retrain or
overwrite earlier saves, a stop at the last save point must not mark a finished run 中止, and an export
must not leave half-written versions or reuse a name."""
from __future__ import annotations

import json
import struct
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "data dir"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config, training
    config._settings = None
    return {"tmp": tmp_path, "training": training, "config": config}


def _lora(path: Path, n_bytes: int, cut: int = 0) -> Path:
    hdr = json.dumps({"a.lora_down.weight": {"dtype": "F16", "shape": [n_bytes // 2], "data_offsets": [0, n_bytes]}}).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((struct.pack("<Q", len(hdr)) + hdr + b"\0" * n_bytes)[:-cut or None])
    return path


def test_a_failing_gpu_task_does_not_stop_the_queue(env):
    training = env["training"]
    r = training.Runner(None)

    class P:
        id = "p"
    ran = threading.Event()

    def boom():
        raise RuntimeError("ComfyUI の起動に失敗しました")
    r.enqueue_task(P(), "比較 #1", boom)
    r.enqueue_task(P(), "比較 #2", ran.set)
    assert ran.wait(5), "the task after a failing one never ran"
    assert "比較 #1" in (env["config"].DATA_DIR / "server_errors.log").read_text(encoding="utf-8")


def _state(dir_: Path, step: int) -> Path:
    dir_.mkdir(parents=True)
    (dir_ / "train_state.json").write_text(json.dumps({"current_epoch": 1, "current_step": step}), encoding="utf-8")
    return dir_


def test_resume_only_where_the_engine_continues_at_the_saved_step(env, tmp_path):
    training = env["training"]
    snap = {"engine": "sd-scripts", "values": {"train_batch_size": 1, "max_train_steps": 1600}, "items": [{}] * 20}
    assert training.resume_block_reason(snap, _state(tmp_path / "s10", 10)) is None
    # past one epoch: bridges/sd_resume.py keeps the saved step (test_sd_resume.py)
    assert training.resume_block_reason(snap, _state(tmp_path / "s600", 600)) is None
    assert training.resume_block_reason(dict(snap, engine="sd-scripts-moss"), _state(tmp_path / "s700", 700)) is None
    assert training.resume_block_reason(snap, _state(tmp_path / "s1600", 1600))   # nothing left to train
    # musubi-tuner restarts at step 0 and trains the full run again
    musubi = dict(snap, engine="musubi-tuner")
    assert "musubi-tuner" in training.resume_block_reason(musubi, tmp_path / "s10")
    assert training.resume_block_reason(dict(snap, enhance={"method": "pair"}), tmp_path / "s10")


def test_cut_off_lora_files_are_not_registered(env, tmp_path):
    training = env["training"]
    assert training.lora_file_complete(_lora(tmp_path / "ok.safetensors", 64))
    assert not training.lora_file_complete(_lora(tmp_path / "cut.safetensors", 64, cut=10))
    (tmp_path / "junk.safetensors").write_bytes(b"\x01")
    assert not training.lora_file_complete(tmp_path / "junk.safetensors")


def test_stop_whose_save_point_is_the_end_lets_the_run_finish(env):
    training = env["training"]
    r = training.Runner(None)

    class P:
        id = "p"

        def run(self, rid):
            return {"config": {"values": {"save_every_n_steps": 300, "max_train_steps": 1000}}}
    job = training.Job(P(), 1)
    job.progress.update(phase="training", step=950, total=1000)
    r.current = job
    res = r.stop(P(), 1)
    assert res["accepted"] is False and not job.stop_requested_at and not job.stop_after_step
    job.progress["step"] = 250
    res = r.stop(P(), 1)
    assert res["accepted"] and job.stop_after_step == 300
    # during the comparison images a stop cancels them, not the finished training
    job2 = training.Job(P(), 2)
    job2.progress.update(phase="evaluating")
    r.current = job2
    assert r.stop(P(), 2)["accepted"] and job2.eval_cancel and not job2.stop_requested_at
    r.current = None


def test_save_interval_zero_is_a_message_not_a_crash(env):
    from lorastudio import presets
    v = {"max_train_steps": 400, "save_every_n_steps": 0, "learning_rate": 1e-4, "network_dim": 16}
    assert presets.estimate_disk_gb("sdxl", v) > 0
    assert any("保存間隔" in e for e in presets.validate("sdxl", v))


def test_export_with_brackets_in_the_name_and_no_half_written_version(env, tmp_path, monkeypatch):
    from lorastudio import export, linking
    root = tmp_path / "loras"
    root.mkdir()
    (root / "[キャラ]_v1").mkdir()
    assert export._next_version_dir(root, "[キャラ]").name == "[キャラ]_v2"

    class P:
        id = "p"

        def run(self, rid):
            return {"config": {"model": {"family": "sdxl", "name": "m", "variant": "illustrious", "files": {}}, "trigger": "zxq",
                               "kind": "character", "engine": "sd-scripts", "engine_commit": "x", "preset_version": 1,
                               "values": {}, "changed": [], "items": []}}

        def meta(self, k):
            return "[キャラ]"

        def checkpoints(self, rid):
            return []

        def tx(self):
            import contextlib
            import sqlite3
            db = sqlite3.connect(":memory:")
            db.execute("CREATE TABLE exports(run_id, created, ckpt_path, dest, info)")
            return contextlib.nullcontext(db)
    src = _lora(tmp_path / "src.safetensors", 64)
    monkeypatch.setattr(linking, "link_export", lambda f: None)

    def broken(family, s, dest):
        dest.write_bytes(b"partial")
        raise OSError("ディスクがいっぱいです")
    monkeypatch.setattr(export, "comfy_lora_file", broken)
    with pytest.raises(OSError):
        export.export(P(), 1, str(src), 1.0, str(root), None)
    assert sorted(p.name for p in root.iterdir()) == ["[キャラ]_v1"]   # nothing half-written is left in loras

    def ok(family, s, dest):
        dest.write_bytes(Path(s).read_bytes())
        return dest
    monkeypatch.setattr(export, "comfy_lora_file", ok)
    res = export.export(P(), 1, str(src), 1.0, str(root), None)
    assert Path(res["file"]).is_file() and Path(res["dir"]).name == "[キャラ]_v2"
    assert (Path(res["dir"]) / "lora_info.json").is_file() and (Path(res["dir"]) / "使い方.txt").is_file()
    res = export.export(P(), 1, str(src), 1.0, str(root), None)
    assert Path(res["dir"]).name == "[キャラ]_v3"   # the second export of the same name no longer fails
