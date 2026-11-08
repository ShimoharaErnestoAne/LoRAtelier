"""Analysis queue fixes (2026-11-08 review).  Failure risks: a request waiting behind a long caption batch
killing the worker, the VLM writing the same caption twice, a caption batch lost while its project is moved,
a deliberately re-cropped copy judged an exact duplicate, identity recomputed for every image of an import,
a finished analysis thrown away, a worker of another version going unnoticed, and a mask cache entry
trusting a file that was rewritten for another domain."""
from __future__ import annotations

import io
import itertools
import json
import queue
import sys
import threading
import time
from concurrent.futures import Future
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

AI = {"tags": {"general": {"1girl": 0.9, "solo": 0.9, "red hair": 0.9, "smile": 0.8}, "character": {}}}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "データ 空白"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import analysis, config, models, store
    config._settings = None
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    ck = tmp_path / "base.safetensors"
    ck.write_bytes(b"0")
    model = models.registry.add("自然文", "sdxl_base", {"ckpt": str(ck)})
    pm = store.ProjectManager()
    an = analysis.Analyzer.__new__(analysis.Analyzer)   # no dispatch thread: the test drives the queue
    an.projects, an.q, an._seq, an._queued, an._qlock, an.current = pm, queue.PriorityQueue(), itertools.count(), set(), threading.Lock(), None
    an._forced = set()
    monkeypatch.setattr(analysis.worker, "available", lambda: True)
    monkeypatch.setattr(analysis.gpu, "begin_analysis", lambda timeout=None: True)
    monkeypatch.setattr(analysis.gpu, "end_analysis", lambda: None)
    return {"pm": pm, "an": an, "model": model, "analysis": analysis, "store": store, "tmp": tmp_path}


def _project(env, name="キャラ"):
    return env["pm"].create(name, "character", env["model"]["id"], "include")


def _image(env, p, n):
    f = env["tmp"] / f"{p.id}_{n}.png"
    f.write_bytes(f"img{p.id}{n}".encode())
    iid, _ = p.add_image_file(f, f.name, f"{p.id}{n:04d}".ljust(40, "0"))
    p.set_fields(iid, fast_status="done", fast={"ok": True, "colorfulness": 0.2}, ai_status="done", ai=AI)
    return iid


class _Stdin:
    def write(self, b):
        pass

    def flush(self):
        pass


class _Proc:
    stdin = _Stdin()
    pid = 0

    def poll(self):
        return None


def test_a_request_queued_behind_a_long_one_is_not_timed_out(env, monkeypatch):
    an = env["analysis"]
    w = an.WorkerClient()
    w.proc, w.status = _Proc(), "ready"
    killed = []
    monkeypatch.setattr(w, "kill", lambda: killed.append(1))
    w._pending[0] = Future()           # an earlier request (e.g. a caption batch) the worker is still answering

    def answer():
        while 1 not in w._pending:
            time.sleep(0.01)
        time.sleep(0.6)                # longer than this request's own time limit
        w._pending.pop(1).set_result({"ok": True, "matrix": []})
    threading.Thread(target=answer, daemon=True).start()
    assert w.call("ccip_diff", timeout=0.2)["ok"] and not killed
    # alone and past its limit, a request still ends the stuck worker
    w._pending.clear()
    assert w.call("ccip_diff", timeout=0.2)["error"] == "解析がタイムアウトしました" and killed


def test_a_caption_being_written_is_not_queued_again_and_a_cached_one_needs_no_vlm(env, monkeypatch):
    an, p = env["an"], _project(env)
    iid = _image(env, p, 1)
    an.ensure_caption(p, iid)
    assert ("caption", p.id, iid) in an._queued
    an._queued.clear()
    calls = []

    def call(op, **k):
        calls.append(k["paths"])
        # while the VLM writes, a settings change that keeps the caption policy asks again for the caption
        an.ensure_caption(p, iid)
        assert ("caption", p.id, iid) not in an._queued
        return {"ok": True, "texts": ["a girl smiling"]}
    monkeypatch.setattr(env["analysis"].worker, "call", call)
    an._dispatch_one("caption", p.id, [iid], ("caption", p.id, iid))
    assert len(calls) == 1 and p.image(iid)["caption_status"] == "done"
    first = p.image(iid)["caption"]
    # queued again anyway (e.g. undo of a removal): the cached text is used without the VLM
    p.set_fields(iid, caption_status="pending")
    an._dispatch_one("caption", p.id, [iid], ("caption", p.id, iid))
    assert len(calls) == 1 and p.image(iid)["caption"] == first and p.image(iid)["caption_status"] == "done"


def test_a_caption_batch_is_requeued_whole_while_its_project_moves(env, monkeypatch):
    an, store = env["an"], env["store"]
    p = _project(env)

    def moving(pid):
        raise store.ProjectMoving(pid)
    monkeypatch.setattr(an.projects, "get", moving)

    class NowTimer:
        def __init__(self, _t, fn, args=()):
            self.fn, self.args = fn, args

        def start(self):
            self.fn(*self.args)
    monkeypatch.setattr(env["analysis"].threading, "Timer", NowTimer)
    an._dispatch_one("caption", p.id, [3, 4, 5], ("caption", p.id, 3))
    assert {("caption", p.id, i) for i in (3, 4, 5)} <= an._queued


def test_two_crops_of_one_file_are_not_duplicates_but_the_same_crop_is(env):
    from lorastudio import judge
    base = {"sha": "a" * 40, "fast": {}, "ai": {}}
    imgs = [{**base, "id": 1, "crop": None}, {**base, "id": 2, "crop": [0, 0, 100, 100]},
            {**base, "id": 3, "crop": [0, 0, 100, 100]}]
    ctx = judge.build_context(imgs, {"kind": "character"})
    assert ctx["dup_of"] == {3: 2}


def test_identity_waits_for_the_rest_of_the_analyses(env, monkeypatch):
    an, p = env["an"], _project(env)
    asked = []
    monkeypatch.setattr(p, "images", lambda: asked.append(1) or [])
    an._queued.add(("analyze", p.id, 7))
    an.compute_identity(p)
    assert not asked
    an._queued.clear()
    an.compute_identity(p)
    assert asked


def test_an_analysis_finished_after_a_crop_change_is_kept_in_the_cache(env, monkeypatch):
    an, a, p = env["an"], env["analysis"], _project(env)
    iid = _image(env, p, 1)
    p.set_fields(iid, ai_status="pending", ai=None, ai_key=None)
    img = p.image(iid)
    key = a.ai_key(img["sha"], img.get("crop"))

    def call(op, **k):
        p.set_fields(iid, crop=[0, 0, 10, 10])   # the user crops while the worker analyses
        return {"ok": True, "result": {"tags": {"general": {}, "character": {}}}}
    monkeypatch.setattr(a.worker, "call", call)
    an._do_analyze(p, iid)
    assert a.cache_get(key) is not None               # undoing the crop does not analyse again


def test_a_worker_of_another_version_is_reported(env):
    a = env["analysis"]
    w = a.WorkerClient()

    class P:
        stdout = io.BytesIO((json.dumps({"event": "ready", "version": "ai-0"}) + "\n").encode())
    w.proc = P
    w._reader(P)
    assert "版" in (w.last_error or "")
    w2 = a.WorkerClient()

    class Q:
        stdout = io.BytesIO((json.dumps({"event": "ready", "version": a.ANALYZER_VERSION}) + "\n").encode())
    w2._reader(Q)
    assert w2.last_error is None


def test_a_mask_cache_entry_stops_trusting_a_rewritten_file(env, tmp_path):
    a = env["analysis"]
    f = tmp_path / "m.png"
    f.write_bytes(b"x")
    entry = {"status": "done", "file": a._stamp(f)}
    assert a._stamp_ok(entry, f) and a._stamp_ok({"status": "done"}, f)   # older entries: trusted as before
    time.sleep(0.01)
    f.write_bytes(b"yy")                                                   # made again for the other domain
    assert not a._stamp_ok(entry, f)
