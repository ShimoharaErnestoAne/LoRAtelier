"""LoRA creation tasks (generate -> analyse -> train, one task after another) and cancelling waiting GPU jobs.
Failure risks: training started before the captions are written, a stalled analysis blocking every later task
forever, a cancelled task still starting its next step, a task order lost or changed by a restart, a task that
was generating left paused after a restart, and a cancelled GPU job still running or leaving its record queued."""
from __future__ import annotations

import json
import sys
import threading
import time
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


class FakeProject:
    def __init__(self, pid):
        self.id = pid
        self.batches = {}      # id -> [status, message]
        self.imgs = []
        self.runs = {}

    def meta(self, k):
        return {"gen_recipe": {"gen_model_id": "m"}, "kind": "character"}.get(k)

    def q(self, sql, args=()):
        if "gen_batches" in sql:
            b = self.batches.get(args[0])
            return [{"status": b[0], "message": b[1]}] if b else []
        if "FROM images" in sql:   # the count of images still being analysed
            return [{"n": sum(1 for im in self.imgs if im["fast_status"] == "pending" or im["ai_status"] in ("pending", "running")
                              or im["caption_status"] in ("pending", "running"))}]
        if "FROM runs" in sql:     # status columns as stored (progress / error are JSON text)
            r = self.runs.get(args[0])
            return [{"status": r["status"], "progress": json.dumps(r.get("progress")) if r.get("progress") else None,
                     "error": json.dumps(r.get("error")) if r.get("error") else None}] if r else []
        raise AssertionError(sql)

    def images(self):
        return self.imgs

    def run(self, rid):
        return self.runs.get(rid)


def _img(fast="done", ai="done", cap="done"):
    return {"fast_status": fast, "ai_status": ai, "caption_status": cap}


@pytest.fixture()
def pipe(env, monkeypatch):
    from lorastudio import generate, models, pipeline
    projs = {"a": FakeProject("a"), "b": FakeProject("b")}
    log = []
    monkeypatch.setattr(generate, "check_recipe", lambda r, k, p=None: {"errors": []})
    monkeypatch.setattr(models.registry, "get", lambda mid: {"id": mid})

    def new_batch(p, recipe, n):
        bid = len(p.batches) + 1
        p.batches[bid] = ["queued", "順番待ち"]
        return bid
    monkeypatch.setattr(generate, "new_batch", new_batch)
    monkeypatch.setattr(generate, "_set_status", lambda p, bid, st, msg: p.batches[bid].__setitem__(0, st))
    h = pipeline.hooks
    monkeypatch.setattr(h, "queue_batch", lambda p, bid: log.append(("queue", p.id, bid)), raising=False)
    monkeypatch.setattr(h, "stop_batch", lambda p, bid: log.append(("stop_batch", p.id, bid)), raising=False)
    monkeypatch.setattr(h, "import_batch", lambda p, bid: (log.append(("import", p.id, bid)), p.imgs.extend([_img("pending", "pending", "pending")] * 2), 2)[2], raising=False)

    def start_training(p, name):
        rid = len(p.runs) + 1
        p.runs[rid] = {"status": "queued", "progress": {"message": "待機中"}}
        log.append(("train", p.id, rid))
        return rid
    monkeypatch.setattr(h, "start_training", start_training, raising=False)
    monkeypatch.setattr(h, "stop_run", lambda p, rid: log.append(("stop_run", p.id, rid)), raising=False)
    busy = {"v": False}
    monkeypatch.setattr(h, "analysis_busy", lambda: busy["v"], raising=False)
    pl = pipeline.Pipeline(SimpleNamespace(get=lambda pid: projs[pid]))
    return SimpleNamespace(pl=pl, projs=projs, log=log, busy=busy, mod=pipeline)


def test_a_task_generates_waits_for_captions_then_trains_and_the_next_waits(pipe):
    pl, a, b = pipe.pl, pipe.projs["a"], pipe.projs["b"]
    b.imgs = [_img()]
    ta = pl.add(a, 3)
    tb = pl.add(b, 0)
    pl.step()
    assert pipe.log == [("queue", "a", 1)] and pl.list()[0]["status"] == "generating"
    a.batches[1] = ["running", "生成中 1 / 3 枚"]
    pl.step()
    assert pl.list()[0]["status"] == "generating" and "1 / 3" in pl.list()[0]["message"]
    a.batches[1] = ["done", "完了"]
    pl.step()
    assert pipe.log[-1] == ("import", "a", 1) and pl.list()[0]["status"] == "analyzing"
    pl.step()   # the imported images are not analysed yet: no training
    assert not any(x[0] == "train" for x in pipe.log) and "残り 2 枚" in pl.list()[0]["message"]
    a.imgs[:] = [_img(), _img(cap="running")]
    pl.step()
    assert not any(x[0] == "train" for x in pipe.log)
    a.imgs[:] = [_img(), _img()]
    pl.step()
    assert pipe.log[-1] == ("train", "a", 1) and pl.list()[0]["status"] == "training"
    pl.step()   # b waits while a trains
    assert pl.list()[1]["status"] == "waiting"
    a.runs[1]["status"] = "evaluating"
    pl.step()
    assert pl.list()[0]["status"] == "training"
    a.runs[1]["status"] = "done"
    pl.step()
    assert pl.list()[0]["status"] == "done"
    pl.step()   # b: no generation, its material is analysed: straight to training
    pl.step()
    assert pipe.log[-1] == ("train", "b", 1)
    assert {t["id"] for t in pl.list()} == {ta["id"], tb["id"]}


def test_a_stalled_analysis_does_not_block_forever(pipe, monkeypatch):
    pl, a = pipe.pl, pipe.projs["a"]
    a.imgs = [_img(), _img(ai="pending")]
    pl.add(a, 0)
    pl.step()
    pl.step()
    assert not any(x[0] == "train" for x in pipe.log)
    monkeypatch.setattr(pipe.mod, "ANALYSIS_STALL_S", 0)
    pipe.busy["v"] = True        # still queued for analysis (e.g. another GPU job runs): keep waiting
    pl.step()
    assert not any(x[0] == "train" for x in pipe.log)
    pipe.busy["v"] = False
    pl.step()
    assert pipe.log[-1] == ("train", "a", 1) and "1 枚" in pl.list()[0]["message"]


def test_failures_and_cancel(pipe):
    pl, a, b = pipe.pl, pipe.projs["a"], pipe.projs["b"]
    t1 = pl.add(a, 3)
    pl.step()
    a.batches[1] = ["failed", "生成に失敗しました: x"]
    pl.step()
    assert pl.list()[0]["status"] == "failed" and "生成に失敗" in pl.list()[0]["message"]
    # cancelling a generating task stops its batch, and a step already running does not move it on
    t2 = pl.add(a, 3)
    pl.step()
    assert pl.cancel(t2["id"]).startswith("取り消しました") and ("stop_batch", "a", 2) in pipe.log
    a.batches[2] = ["done", "完了"]
    pl.step()
    assert not any(x[0] == "import" for x in pipe.log)
    # training: the run is stopped; training that cannot start fails the task with the reason
    b.imgs = [_img()]
    t3 = pl.add(b, 0)
    pl.step()
    pl.step()
    pl.cancel(t3["id"])
    assert ("stop_run", "b", 1) in pipe.log
    pipe.mod.hooks.start_training = lambda p, n: (_ for _ in ()).throw(ValueError("採用できる素材がありません。"))
    t4 = pl.add(b, 0)
    pl.step()
    pl.step()
    t = next(x for x in pl.list() if x["id"] == t4["id"])
    assert t["status"] == "failed" and "採用できる素材" in t["message"]
    with pytest.raises(KeyError):
        pl.cancel(t1["id"])     # already ended
    assert pl.remove_ended() == 4 and pl.list() == []


def test_order_survives_a_restart_and_generation_continues(pipe):
    pl, a, b = pipe.pl, pipe.projs["a"], pipe.projs["b"]
    t1 = pl.add(a, 3)
    pl.step()                                    # a generating batch 1
    t2, t3 = pl.add(b, 0), pl.add(a, 0)
    assert pl.move(t3["id"], "top") and not pl.move(t1["id"], "top")   # only waiting tasks move
    # restart: the startup recovery paused the batch
    a.batches[1] = ["paused", "アプリの終了で止まりました"]
    pl2 = pipe.mod.Pipeline(pl.projects)
    pl2.start = lambda: None
    pl2.recover()
    assert [t["id"] for t in pl2.list()] == [t1["id"], t3["id"], t2["id"]]
    assert a.batches[1][0] == "queued" and pipe.log.count(("queue", "a", 1)) == 2
    # a run that the restart interrupted ends the task as failed
    a.batches[1] = ["done", "完了"]
    pl2.step()
    a.imgs[:] = [_img()]
    pl2.step()
    a.runs[1] = {"status": "failed", "error": {"message": "アプリの終了により中断されました。"}}
    pl2.step()
    assert pl2.list()[0]["status"] == "failed" and "中断" in pl2.list()[0]["message"]


def test_waiting_gpu_jobs_can_be_cancelled_and_never_run(env):
    from lorastudio import training
    runner = training.Runner(SimpleNamespace())
    gate, ran, cancelled = threading.Event(), [], []
    p = SimpleNamespace(id="x")
    runner.enqueue_task(p, "占有", gate.wait, on_cancel=lambda: None)
    for _ in range(100):
        if runner.current:
            break
        time.sleep(0.01)
    j1 = runner.enqueue_task(p, "比較 #1", lambda: ran.append(1), on_cancel=lambda: cancelled.append(1))
    j2 = runner.enqueue_task(p, "比較 #2", lambda: ran.append(2))
    assert not runner.cancel_task(j2.uid)          # no way to mark its record: stays
    assert runner.cancel_task(j1.uid) and cancelled == [1]
    assert not runner.cancel_task(j1.uid)
    gate.set()
    for _ in range(200):
        if ran:
            break
        time.sleep(0.01)
    assert ran == [2]


def test_paused_queue_finishes_nothing_new_until_resumed(env):
    from lorastudio import config, training
    runner = training.Runner(SimpleNamespace())
    runner.set_paused(True)
    ran = []
    runner.enqueue_task(SimpleNamespace(id="x"), "比較 #1", lambda: ran.append(1), on_cancel=lambda: None)
    time.sleep(0.3)
    assert ran == [] and len(runner.waiting()) == 1
    assert config.settings().get("queue_paused") is True      # kept across a restart
    runner.set_paused(False)
    for _ in range(200):
        if ran:
            break
        time.sleep(0.01)
    assert ran == [1]


def test_paused_tasks_hold_before_their_next_stage(pipe):
    pl, a = pipe.pl, pipe.projs["a"]
    pl.add(a, 3)
    pl.set_paused(True)
    pl.step()
    assert pl.list()[0]["status"] == "waiting" and not pipe.log          # a waiting task does not start
    pl.set_paused(False)
    pl.step()
    assert pl.list()[0]["status"] == "generating"
    pl.set_paused(True)
    a.batches[1] = ["done", "完了"]                                         # the stage running goes on to its end
    pl.step()
    assert pl.list()[0]["status"] == "generating" and "一時停止中" in pl.list()[0]["message"]
    assert not any(x[0] == "import" for x in pipe.log)
    pl.set_paused(False)
    pl.step()
    assert pipe.log[-1] == ("import", "a", 1)


def test_the_running_job_is_stopped_only_by_a_click_meant_for_it(env):
    from fastapi.testclient import TestClient
    from lorastudio import generate, server, training
    server.runner.set_paused(True)                 # nothing else picks jobs meanwhile
    p = server.projects.create("ジョブ", "character", None)
    with p.tx() as db:
        bid = db.execute("INSERT INTO gen_batches(created, status, message, spec) VALUES(?,?,?,?)",
                         (time.time(), "running", "生成中 3 / 24 枚", "{}")).lastrowid
    j = training.Job(p, None)
    j.task, j.label = (lambda: None), f"素材生成 #{bid}"
    server.runner.current = j
    c = TestClient(server.app)
    q = c.get("/api/queue").json()
    assert q["current"]["kind"] == "gen" and q["current"]["fraction"] == 3 / 24 and q["paused"] is True
    assert {x["action"] for x in q["current"]["controls"]} == {"pause", "stop"}
    assert c.post("/api/queue/current/pause", json={"uid": j.uid + 1}).status_code == 409
    assert c.post("/api/queue/current/force", json={"uid": j.uid}).status_code == 400
    assert c.post("/api/queue/current/pause", json={"uid": j.uid}).status_code == 200
    assert generate.control.get(p.id, bid) == "pause"
    j2 = training.Job(p, None)
    j2.task, j2.label = (lambda: None), "比較 #4"
    server.runner.current = j2
    assert c.post("/api/queue/current/stop", json={"uid": j2.uid}).status_code == 200 and j2.eval_cancel
    server.runner.current = None
    generate.control.clear(p.id, bid)


def test_imported_project_puts_its_exported_loras_back_in_the_lora_list(env, monkeypatch, tmp_path):
    import json as _json
    from lorastudio import server
    p = server.projects.create("移した", "character", None)
    ck = p.path / "runs" / "1" / "output" / "x-000400.safetensors"
    ck.parent.mkdir(parents=True)
    ck.write_bytes(b"lora")
    here = tmp_path / "loras" / "移した_v1"
    here.mkdir(parents=True)
    with p.tx() as db:
        db.execute("INSERT INTO checkpoints(run_id, step, path, kind, created) VALUES(1, 400, ?, 'lora', 0)", (str(ck),))
        info = _json.dumps({"tested_strength": 0.8, "evaluation": {"seeds": [1]}})
        # exported twice on the old PC (both folders missing here), once from a save the zip does not have,
        # and once to a folder that exists on this PC
        db.executemany("INSERT INTO exports(run_id, created, ckpt_path, dest, info) VALUES(1, 0, ?, ?, ?)",
                       [(str(ck), r"D:\old\loras\移した_v1", info), (str(ck), r"D:\old\loras\移した_v2", info),
                        (str(p.path / "runs" / "1" / "output" / "gone.safetensors"), r"D:\old\loras\移した_v3", info)])
        db.execute("INSERT INTO checkpoints(run_id, step, path, kind, created) VALUES(1, 800, ?, 'lora', 0)",
                   (str(tmp_path / "kept.safetensors"),))
        (tmp_path / "kept.safetensors").write_bytes(b"k")
        db.execute("INSERT INTO exports(run_id, created, ckpt_path, dest, info) VALUES(1, 0, ?, ?, '{}')",
                   (str(tmp_path / "kept.safetensors"), str(here)))
    calls = []

    def fake_export(proj, rid, ckpt, strength, dest_root, summary):
        calls.append((rid, ckpt, strength, dest_root, summary))
        (tmp_path / "loras" / "移した_v2").mkdir()
        with proj.tx() as db:
            db.execute("INSERT INTO exports(run_id, created, ckpt_path, dest, info) VALUES(?,?,?,?,?)",
                       (rid, 1, ckpt, str(tmp_path / "loras" / "移した_v2"), "{}"))
        return {"dir": str(tmp_path / "loras" / "移した_v2")}
    monkeypatch.setattr(server, "do_export", fake_export)
    out = server._restore_exports(p)
    assert out == {"added": ["移した_v2"], "failed": []}
    assert calls == [(1, str(ck), 0.8, None, {"seeds": [1]})]      # once per saved LoRA, to the LoRA folder
    dests = [r["dest"] for r in p.q("SELECT dest FROM exports")]
    assert r"D:\old\loras\移した_v1" not in dests and str(here) in dests
    assert server._restore_exports(p) == {"added": [], "failed": []}   # already there: nothing twice
