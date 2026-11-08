"""2026-11-08 review fixes in the training queue and the project store (fake data, no GPU)."""
from __future__ import annotations

import sqlite3
import sys
import threading
import time

import pytest


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "データ 空白"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config
    config._settings = None
    return tmp_path


def _project(name="p"):
    from lorastudio import store
    return store.ProjectManager().create(name, "character", None)


def test_a_transaction_that_cannot_begin_frees_the_project_lock(env):
    p = _project()
    real = p.db
    p.db = sqlite3.connect(":memory:", check_same_thread=False)
    p.db.execute("BEGIN")   # a second BEGIN fails like a locked or vanished database would
    with pytest.raises(sqlite3.Error):
        with p.tx():
            pass
    got = []

    def other():
        got.append(p._lock.acquire(timeout=2))
        p._lock.release()
    t = threading.Thread(target=other)
    t.start()
    t.join()
    assert got == [True]       # another thread can still use the project
    p.db = real


def test_stopping_a_run_that_waits_for_the_gpu_cancels_it_at_once(env, monkeypatch):
    from lorastudio import training
    p = _project()
    with p.tx() as db:
        rid = db.execute("INSERT INTO runs(created, status, name, config, run_dir) VALUES(0,'queued','#1','{}','')").lastrowid
    monkeypatch.setattr(training.Runner, "_loop", lambda self: None)   # no runner thread
    runner = training.Runner(None)
    job = training.Job(p, rid)
    runner.current = job                    # taken from the queue, still waiting in gpu.acquire
    r = runner.stop(p, rid)
    assert r["accepted"] and "開始前" in r["message"]
    assert p.run(rid)["status"] == "cancelled" and job.stop_requested_at
    runner._waiting_for_analysis(job)       # a late wait message does not overwrite the cancel
    assert p.run(rid)["progress"]["phase"] == "cancelled"


def test_the_runner_skips_a_job_stopped_while_waiting_for_the_gpu(env, monkeypatch):
    from lorastudio import training
    from lorastudio.gpu import gpu
    p = _project()
    with p.tx() as db:
        rid = db.execute("INSERT INTO runs(created, status, name, config, run_dir) VALUES(0,'queued','#1','{}','')").lastrowid
    ran = []
    monkeypatch.setattr(training.Runner, "_materialize_async", lambda self, job: {})
    monkeypatch.setattr(training.Runner, "_run", lambda self, job, prep=None: ran.append(job.run_id))

    def acquire(owner, on_wait=None):     # the stop arrives while the job waits for the GPU
        runner.stop(p, rid)
    monkeypatch.setattr(gpu, "acquire", acquire)
    monkeypatch.setattr(gpu, "release", lambda owner: None)
    runner = training.Runner(None)
    runner.enqueue(p, rid)
    for _ in range(100):
        if runner.current is None and not runner.jobs and p.run(rid)["status"] == "cancelled":
            break
        time.sleep(0.02)
    assert ran == [] and p.run(rid)["status"] == "cancelled"


def test_enqueue_writes_the_queued_status_before_the_runner_can_take_the_job(env, monkeypatch):
    from lorastudio import training
    p = _project()
    with p.tx() as db:
        rid = db.execute("INSERT INTO runs(created, status, name, config, run_dir) VALUES(0,'failed','#1','{}','')").lastrowid
    monkeypatch.setattr(training.Runner, "_loop", lambda self: None)
    runner = training.Runner(None)
    seen = []
    real_append = list.append

    class Jobs(list):
        def append(self, job):
            seen.append(p.run(rid)["status"])   # what the runner would see when it takes the job
            real_append(self, job)
    runner.jobs = Jobs()
    runner.enqueue(p, rid)
    assert seen == ["queued"]


def test_captions_are_written_whole(env, tmp_path, monkeypatch):
    from lorastudio import training
    run_dir = tmp_path / "run"
    (run_dir / "dataset").mkdir(parents=True)
    (run_dir / "dataset" / "1.png").write_bytes(b"x")    # the image is already there
    training.materialize(run_dir, {"items": [{"file": "1.png", "caption": "chr, 1girl", "src": "-"}]})
    assert (run_dir / "dataset" / "1.txt").read_text(encoding="utf-8") == "chr, 1girl"
    assert not list((run_dir / "dataset").glob("*.tmp"))


def test_a_failed_run_snapshot_leaves_no_folder_behind(env, monkeypatch):
    from lorastudio import training
    p = _project()

    def broken(project, run_id, *a, **k):
        d = project.runs_dir / f"{run_id:04d}_20261108-000000"
        (d / "dataset").mkdir(parents=True)
        raise OSError("disk full")
    monkeypatch.setattr(training, "_write_run", broken)
    monkeypatch.setattr(training.registry, "get", lambda mid: {"id": "m", "family": "sdxl"})
    monkeypatch.setattr(p, "get_meta", lambda: {"model_id": "m"})
    with pytest.raises(OSError):
        training.create_run(p, [], {}, [], "x")
    assert not list(p.runs_dir.glob("*_*"))
    assert p.q("SELECT status FROM runs")[0]["status"] == "failed"


def test_import_rebase_does_not_touch_a_sibling_folder(env):
    from lorastudio import projectio
    old, new = r"D:\proj\abc", r"E:\new\abc"
    assert projectio._rebase(old + r"\sources\1.png", old, new) == new + r"\sources\1.png"
    assert projectio._rebase(old, old, new) == new
    assert projectio._rebase(old + r"_2\sources\1.png", old, new) == old + r"_2\sources\1.png"
