"""2026-11-10 review fixes in the material checks, the analysis queue, training, the server and the i2t link
(fake data, no GPU, no real models).  Failure risks: camera JPEGs refused as an unknown format, white-background
drawings flagged as overexposed, clothing and expressions bound to the trigger word, work that waits for the GPU
counted as finished, a second run of a batch trained with the first one's caption dropout, a run queued twice,
a crashed engine reported as a clean stop, a finished run marked failed by a restart, the start-up recovery
stopping half way, pages of other web sites driving the API, and an i2t job whose start fails answering 500."""
from __future__ import annotations

import importlib.util
import io
import itertools
import json
import queue
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

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


# ------------------------------------------------------------------ material checks
def test_a_camera_jpeg_with_an_embedded_preview_is_a_jpeg(env):
    from lorastudio import fastcheck
    rng = np.random.default_rng(0)
    big = Image.fromarray((rng.random((600, 800, 3)) * 255).astype("uint8"))
    f = env / "camera.jpg"
    big.save(f, "MPO", save_all=True, append_images=[big.resize((160, 120))])
    with Image.open(f) as im:
        assert im.format == "MPO"            # what Pillow calls such a file
    facts = fastcheck.analyze(f, env / "t.webp")
    assert facts["ok"] and facts["format"] == "JPEG" and not facts["animated"] and facts["jpeg_quality"]


def _img(iid, **kw):
    fast = {"ok": True, "width": 1024, "height": 1024, "phash": format(iid * 0x0123456789ABCDEF % 2**64, "016x"),
            "sharpness_p90": 50, **kw.pop("fast", {})}
    return {"id": iid, "sha": f"{iid:040d}", "crop": None, "fast": fast, "fast_status": "done", "ai_status": "done",
            "ai": {"tags": {"general": {"1girl": 0.9}, "character": {}}, "domain": {"label": "illustration"},
                   "persons": [{"score": 0.9, "bbox": [100, 50, 700, 1000]}], "heads": [{"score": 0.9}],
                   "subject": {"bbox": [100, 50, 700, 1000], "area_ratio": 0.55, "eff_res": 600}}, **kw}


def test_a_white_background_drawing_is_not_an_exposure_warning(env):
    from lorastudio import judge
    drawing = _img(1, fast={"clip_bright": 0.75})
    photo = _img(2, fast={"clip_bright": 0.75})
    photo["ai"] = {**photo["ai"], "domain": {"label": "photo"}}
    ctx = judge.build_context([drawing, photo], {"kind": "character"})
    d, p = judge.check_image(drawing, ctx), judge.check_image(photo, ctx)
    assert [c["status"] for c in d["checks"] if c["key"] == "exposure"] == ["info"] and d["verdict"] == "ok"
    assert [c["status"] for c in p["checks"] if c["key"] == "exposure"] == ["warn"]


def test_two_crops_of_one_file_are_not_near_duplicates(env):
    from lorastudio import judge
    a, b = _img(1, crop=[0, 0, 500, 900]), _img(2, crop=[300, 0, 800, 400])
    b["sha"], b["fast"] = a["sha"], dict(a["fast"])          # the same file, cut out twice
    c, d = _img(3), _img(4)
    d["fast"] = dict(c["fast"])                                 # two whole files that look the same
    ctx = judge.build_context([a, b, c, d], {"kind": "character"})
    assert 1 not in ctx["near"] and 2 not in ctx["near"] and ctx["near"] == {3: {4}, 4: {3}}


def test_two_analysed_images_do_not_wait_for_the_analysis_forever(env):
    from lorastudio import judge
    imgs = [_img(1), _img(2)]
    ctx = judge.build_context(imgs, {"kind": "character"})
    res = {im["id"]: judge.check_image(im, ctx) for im in imgs}
    notes = [n["text"] for n in judge.dataset_report(imgs, res, {"kind": "character"})["notes"]]
    assert not any("解析の完了後" in t for t in notes)
    imgs[1]["ai_status"], imgs[1]["ai"] = "running", None      # ...but it is said while one is still being analysed
    res = {im["id"]: judge.check_image(im, ctx) for im in imgs}
    assert any("解析の完了後" in n["text"] for n in judge.dataset_report(imgs, res, {"kind": "character"})["notes"])


def test_clothing_and_expressions_are_not_taken_for_the_characters_features(env):
    from lorastudio import captions
    ai = {"tags": {"general": {"1girl": 0.9, "scarf": 0.9, "closed eyes": 0.9, "blue eyes": 0.9, "scar on cheek": 0.9,
                               "wet hair": 0.8, "t-shirt": 0.8, "high heels": 0.8, "long hair": 0.9}, "character": {}}}
    vary = captions.tag_caption(ai, "character", "change", "chr_x").split(", ")
    assert {"scarf", "closed eyes", "wet hair", "t-shirt", "high heels"} <= set(vary)        # stay controllable
    assert not {"blue eyes", "scar on cheek", "long hair"} & set(vary)                       # bound to the trigger
    fixed = captions.tag_caption(ai, "character", "include", "chr_x").split(", ")
    assert not {"scarf", "t-shirt", "high heels"} & set(fixed) and "closed eyes" in fixed


def test_a_short_trigger_inside_another_word_is_not_the_trigger(env):
    from lorastudio import captions
    assert captions.finalize_nl("A girl drinking tea, wearing earrings.", "character", "rin").startswith("rin, ")
    assert captions.finalize_nl("rin drinks tea.", "character", "rin") == "rin drinks tea."


# ------------------------------------------------------------------ analysis queue
@pytest.fixture()
def an(env, monkeypatch):
    from lorastudio import analysis, models, store
    ck = env / "base.safetensors"
    ck.write_bytes(b"0")
    model = models.registry.add("タグ", "illustrious", {"ckpt": str(ck)})
    pm = store.ProjectManager()
    a = analysis.Analyzer.__new__(analysis.Analyzer)   # no dispatch thread: the test drives the queue
    a.projects, a.q, a._seq, a._queued, a._qlock, a.current = pm, queue.PriorityQueue(), itertools.count(), set(), threading.Lock(), None
    a._forced = set()
    monkeypatch.setattr(analysis.worker, "available", lambda: True)
    return SimpleNamespace(a=a, pm=pm, model=model, analysis=analysis, tmp=env)


def _add(an, p, n, **fields):
    f = an.tmp / f"{p.id}_{n}.png"
    f.write_bytes(f"img{n}".encode())
    iid, _ = p.add_image_file(f, f.name, f"{n:040d}")
    p.set_fields(iid, fast_status="done", fast={"ok": True, "colorfulness": 0.2}, **fields)
    return iid


def test_an_item_waiting_for_the_gpu_still_counts_as_work(an, monkeypatch):
    analysis, a = an.analysis, an.a
    p = an.pm.create("待ち", "character", an.model["id"])
    iid = _add(an, p, 1, ai_status="pending")
    waiting, go = threading.Event(), threading.Event()

    def begin(timeout=None):
        if timeout == 0:
            return False              # training holds the GPU
        waiting.set()
        go.wait(5)
        return True
    monkeypatch.setattr(analysis.gpu, "begin_analysis", begin)
    monkeypatch.setattr(analysis.gpu, "end_analysis", lambda: None)
    seen = []
    monkeypatch.setattr(a, "_do_analyze", lambda project, image_id: seen.append(image_id))
    a._enqueue(5, ("analyze", p.id, iid))
    threading.Thread(target=a._dispatch_loop, daemon=True).start()
    assert waiting.wait(5)
    # taken out of the queue, not being analysed yet: a LoRA task must not think the analysis is over
    assert a.queue_size() == 1 and a.has_work(p.id) and a.current is None
    go.set()
    for _ in range(100):
        if seen:
            break
        time.sleep(0.02)
    assert seen == [iid]
    for _ in range(100):
        if not a.queue_size() and a.current is None:
            break
        time.sleep(0.02)
    assert a.queue_size() == 0 and not a.has_work(p.id)


def test_a_failed_caption_batch_leaves_the_finished_captions_alone(an):
    a = an.a
    p = an.pm.create("説明文", "character", an.model["id"])
    done = _add(an, p, 1, ai_status="done", caption_status="done")
    waiting = _add(an, p, 2, ai_status="done", caption_status="running")
    a._mark_failed("caption", p.id, [done, waiting], "解析ワーカーが繰り返し停止しました")
    assert p.image(done)["caption_status"] == "done" and p.image(waiting)["caption_status"] == "failed"


def test_identity_is_tried_again_when_the_gpu_is_busy(an, monkeypatch):
    analysis, a = an.analysis, an.a
    p = an.pm.create("同一性", "character", an.model["id"])
    for n in (1, 2):
        _add(an, p, n, ai_status="done", ai_key=f"k{n}", ai={"ccip": [0.1, 0.2], "tags": {"general": {}, "character": {}}})
    monkeypatch.setattr(analysis.gpu, "begin_analysis", lambda timeout=None: False)
    later = []
    monkeypatch.setattr(a, "_identity_later", lambda project, seconds: later.append((project.id, seconds)))
    a.compute_identity(p)
    assert later == [(p.id, 60)] and not p.meta("identity")


def test_tag_captions_follow_a_change_of_the_tag_rules(an):
    from lorastudio import captions
    a = an.a
    p = an.pm.create("規則", "character", an.model["id"])
    iid = _add(an, p, 1, ai_status="done", ai={"tags": {"general": {"1girl": 0.9, "scarf": 0.9}, "character": {}}})
    how, key, text = a._caption_plan(p.get_meta(), p.image(iid))
    assert how == "now" and key.endswith(captions.TAG_RULES)   # captions made under the older rules are made again


# ------------------------------------------------------------------ training
def test_text_encoder_caches_are_kept_apart_by_the_caption_dropout_rate(env):
    from lorastudio import traincache
    item = {"sha": "a" * 40, "crop": None, "caption": "chr_x, smile", "file": "00001.png"}

    def te(family, engine, rate):
        snap = {"engine": engine, "model": {"family": family}, "values": {"caption_dropout_rate": rate}}
        return traincache._keys(item, False, traincache._te_extra(snap))
    # Anima / MOSS-Image: sd-scripts stores the rate in the cache file and trains with the stored one
    assert te("anima", "sd-scripts", 0)[1] != te("anima", "sd-scripts", 0.1)[1]
    assert te("moss_image", "sd-scripts-moss", 0.05)[1] != te("moss_image", "sd-scripts-moss", 0.1)[1]
    assert te("anima", "sd-scripts", 0)[0] == te("anima", "sd-scripts", 0.1)[0]          # the latents are shared
    assert te("sdxl", "sd-scripts", 0)[1] == te("sdxl", "sd-scripts", 0.1)[1]            # not stored there
    assert te("qwen_image", "musubi-tuner", 0)[1] == te("qwen_image", "musubi-tuner", 0.1)[1]


def _runner(monkeypatch):
    from lorastudio import store, training
    monkeypatch.setattr(training.Runner, "_loop", lambda self: None)   # no runner thread
    p = store.ProjectManager().create("学習", "character", None)
    with p.tx() as db:
        rid = db.execute("INSERT INTO runs(created, status, name, config, run_dir) VALUES(0,'cancelled','#1','{}','')").lastrowid
    return training, training.Runner(None), p, rid


def test_the_same_run_is_not_queued_twice(env, monkeypatch):
    training, runner, p, rid = _runner(monkeypatch)
    assert runner.enqueue(p, rid, resume="state") is True
    assert runner.enqueue(p, rid, resume="state") is False          # 再開 pressed twice
    assert len(runner.jobs) == 1 and p.run(rid)["status"] == "queued"


def test_waiting_tasks_can_be_taken_out_of_the_queue(env, monkeypatch):
    training, runner, p, rid = _runner(monkeypatch)
    job = runner.enqueue_task(p, "画像生成", lambda: None)
    job.entry = "e1"
    runner.enqueue(p, rid)
    assert runner.drop_tasks(lambda j: getattr(j, "entry", None) == "e1") == 1
    assert [j.run_id for j in runner.jobs] == [rid]                 # the training run stays


def test_an_engine_that_crashes_after_a_stop_request_is_a_failure(env, monkeypatch):
    training, runner, p, rid = _runner(monkeypatch)
    out = env / "out"
    out.mkdir()
    job = training.Job(p, rid)
    job.output_name = "x"
    snap = {"output_name": "x"}

    def crash(job, phase, argv, cwd, log_path, total, out_dir, snap=None):
        job.stop_requested_at = time.time()      # the safe stop was accepted, then the engine ran out of memory
        job.stop_after_step = 200
        return 1, "torch.OutOfMemoryError: CUDA out of memory"
    monkeypatch.setattr(runner, "_exec", crash)
    runner._run_cmds(job, snap, env, out, env, [("training", ["py", "train.py"], str(env))], 1000)
    r = p.run(rid)
    assert r["status"] == "failed" and json.loads(r["error"])["code"] != "interrupted"

    def at_save(job, phase, argv, cwd, log_path, total, out_dir, snap=None):
        job.stop_requested_at, job.stopped_at_save = time.time(), True   # ended by the app at the save point
        return 1, ""
    job2 = training.Job(p, rid)
    job2.output_name = "x"
    monkeypatch.setattr(runner, "_exec", at_save)
    runner._run_cmds(job2, snap, env, out, env, [("training", ["py", "train.py"], str(env))], 1000)
    assert p.run(rid)["status"] == "cancelled" and "保存地点で停止" in p.run(rid)["progress"]["message"]


def test_caches_made_by_their_own_commands_count_as_finished(env, monkeypatch):
    training, runner, p, rid = _runner(monkeypatch)
    out = env / "out"
    out.mkdir()
    job = training.Job(p, rid)
    job.output_name = "x"
    cmds = [("preparing", ["py", "qwen_image_cache_latents.py"], str(env)),
            ("preparing", ["py", "qwen_image_cache_text_encoder_outputs.py"], str(env)),
            ("training", ["py", "qwen_image_train_network.py"], str(env))]
    monkeypatch.setattr(runner, "_exec", lambda job, phase, argv, *a: (1, "out of memory") if "train" in argv[1] else (0, ""))
    runner._run_cmds(job, {"output_name": "x"}, env, out, env, cmds, 1000)
    assert job.caches_done and p.run(rid)["status"] == "failed"     # the next run does not encode everything again


def test_the_weights_of_the_last_step_are_listed_once(env, monkeypatch):
    import struct
    training, runner, p, rid = _runner(monkeypatch)
    out = env / "out"
    out.mkdir()
    for name in ("x-step00000400.safetensors", "x-step00000800.safetensors", "x.safetensors"):
        (out / name).write_bytes(struct.pack("<Q", 2) + b"{}")
    job = training.Job(p, rid)
    job.output_name = "x"
    runner._register_checkpoints(job, out, final=True)                  # as the 2 s poll saw them while training
    assert [c["step"] for c in p.checkpoints(rid)] == [None, 400, 800]
    runner._register_checkpoints(job, out, final=True, last_step=800)   # the run ended: step 800 is the final file
    assert sorted(c["step"] or 0 for c in p.checkpoints(rid)) == [0, 400]
    assert (out / "x-step00000800.safetensors").exists()                # the file is not deleted


def test_the_loss_of_a_leco_run_is_read(env):
    from lorastudio import training
    line = "steps:  12%|█▏        | 60/500 [01:10<08:30,  1.16s/it, loss=0.0123]"
    assert training.STEP_RE.search(line).group(7) == "0.0123"
    line = "steps:  12%|█▏        | 60/500 [01:10<08:30,  1.16s/it, avr_loss=0.0456]"
    assert training.STEP_RE.search(line).group(7) == "0.0456"


# ------------------------------------------------------------------ server
def test_a_run_cut_off_while_comparing_is_done_not_failed(env, monkeypatch):
    from lorastudio import server
    p = server.projects.create("比較中に終了", "character", None)
    cfg = json.dumps({"values": {"max_train_steps": 800}, "output_name": "x"})
    prog = json.dumps({"phase": "evaluating", "loss_hist": [[10, 0.1]], "vram_short": {"resolution": 1024}})
    with p.tx() as db:
        rid = db.execute("INSERT INTO runs(created, status, name, config, run_dir, progress) VALUES(0,'evaluating','#1',?,'',?)",
                         (cfg, prog)).lastrowid
        other = db.execute("INSERT INTO runs(created, status, name, config, run_dir) VALUES(0,'training','#2',?,'')", (cfg,)).lastrowid
    monkeypatch.setattr(server, "_resume_analysis", lambda p: None)
    server._recover_project(p, {}, [])
    r = p.run(rid)
    assert r["status"] == "done" and r["error"] is None and r["progress"]["step"] == 800
    assert r["progress"]["loss_hist"] == [[10, 0.1]] and r["progress"]["vram_short"] == {"resolution": 1024}
    assert server._resume_info(r, [])["can_resume"] is False
    assert p.run(other)["status"] == "failed"                       # a run cut off while training stays interrupted


def test_startup_recovery_reaches_the_tasks_after_an_early_error(env, monkeypatch):
    from lorastudio import server
    seen = []
    monkeypatch.setattr(server.imagegen, "recover", lambda: (_ for _ in ()).throw(OSError("history unreadable")))
    monkeypatch.setattr(server.projects, "list", lambda: (_ for _ in ()).throw(ValueError("broken meta")))
    monkeypatch.setattr(server.tasks, "recover", lambda: seen.append("tasks"))
    server.resume_pending_on_startup()
    assert seen == ["tasks"]
    assert "history unreadable" in (server.config.DATA_DIR / "server_errors.log").read_text(encoding="utf-8")


def test_a_lost_active_run_can_be_stopped(env):
    from fastapi.testclient import TestClient
    from lorastudio import server
    p = server.projects.create("取り残し", "character", None)
    with p.tx() as db:
        rid = db.execute("INSERT INTO runs(created, status, name, config, run_dir) VALUES(0,'queued','#1','{}','')").lastrowid
    r = TestClient(server.app).post(f"/api/projects/{p.id}/runs/{rid}/stop", json={})
    assert r.status_code == 200 and r.json()["accepted"] and p.run(rid)["status"] == "failed"


def test_an_unhandled_error_is_logged_with_its_traceback(env):
    from fastapi.testclient import TestClient
    from lorastudio import server

    @server.app.get("/api/_boom_for_test")
    def boom():
        raise RuntimeError("the cause to find")
    r = TestClient(server.app, raise_server_exceptions=False).get("/api/_boom_for_test")
    assert r.status_code == 500
    log = (server.config.DATA_DIR / "server_errors.log").read_text(encoding="utf-8")
    assert "the cause to find" in log and "in boom" in log and "NoneType: None" not in log


def test_pages_of_other_sites_cannot_change_anything(env):
    from fastapi.testclient import TestClient
    from lorastudio import server
    c = TestClient(server.app)
    cross = {"Sec-Fetch-Site": "cross-site", "Origin": "https://evil.example"}
    assert c.post("/api/queue/pause", json={"on": True}, headers=cross).status_code == 403
    assert server.runner.paused is False
    assert c.post("/api/queue/pause", json={"on": False}, headers={"Origin": "https://evil.example"}).status_code == 403
    own = {"Sec-Fetch-Site": "same-origin", "Origin": "http://testserver"}
    assert c.post("/api/queue/pause", json={"on": False}, headers=own).status_code == 200
    assert c.post("/api/queue/pause", json={"on": False}).status_code == 200            # a script, not a browser
    assert c.get("/api/queue", headers=cross).status_code == 200                           # reading is not changing
    # another name pointed at this PC (DNS rebinding) while only this PC is listened to
    assert c.get("/api/queue", headers={"Host": "evil.example:7865"}).status_code == 403
    assert c.get("/api/queue", headers={"Host": "localhost:7865"}).status_code == 200


def test_a_member_project_is_in_use_while_its_multi_character_run_waits(env, monkeypatch):
    from lorastudio import server, training
    multi = server.projects.create("まとめ", "character", None)
    member = server.projects.create("メンバー", "character", None)
    cfg = json.dumps({"members": [{"pid": member.id, "trigger": "chr_a"}]})
    with multi.tx() as db:
        rid = db.execute("INSERT INTO runs(created, status, name, config, run_dir) VALUES(0,'queued','#1',?,'')", (cfg,)).lastrowid
    assert server._busy_reason(member.id) is None
    monkeypatch.setattr(server.runner, "jobs", [training.Job(multi, rid)])
    assert "複数キャラ" in server._busy_reason(member.id) and server._busy_reason(multi.id)


def test_resume_states_are_checked_once_while_their_files_stay(env, monkeypatch):
    from lorastudio import server
    st = env / "x-step000200-state"
    st.mkdir()
    (st / "train_state.json").write_text("{}")
    calls = []
    monkeypatch.setattr(server, "state_is_complete", lambda p: calls.append(1) or True)
    assert server._state_complete(str(st)) and server._state_complete(str(st)) and len(calls) == 1
    (st / "optimizer.bin").write_bytes(b"new")                      # the state changed: looked at again
    assert server._state_complete(str(st)) and len(calls) == 2
    assert server._state_complete(str(env / "gone")) is False


def test_the_size_of_a_sideways_phone_photo_is_the_upright_one(env):
    from lorastudio import server
    im = Image.new("RGB", (400, 300))
    ex = im.getexif()
    ex[0x0112] = 6
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=ex)
    with Image.open(io.BytesIO(buf.getvalue())) as x:
        assert server._upright_size(x) == [300, 400]
    with Image.open(io.BytesIO(buf.getvalue())) as x:
        assert list(x.size) == [400, 300]                           # as stored


def test_startup_resume_does_not_load_every_analysis(env, monkeypatch):
    from lorastudio import server
    p = server.projects.create("起動", "character", None)
    f = env / "a.png"
    f.write_bytes(b"x")
    iid, _ = p.add_image_file(f, "a.png", "a" * 40)
    p.set_fields(iid, fast_status="done", ai_status="done", ai_key=server.ANALYZER_VERSION + ":a:full",
                 ai={"embed": [0.0] * 1000}, caption_status="done")
    p._parsed.clear()
    masked = []
    monkeypatch.setattr(server.analyzer, "mask_mode", lambda project: None)
    monkeypatch.setattr(server.analyzer, "ensure_mask", lambda *a, **k: masked.append(a))
    server._resume_analysis(p)
    assert not p._parsed and not masked          # nothing to do: nothing parsed, nothing kept in memory


# ------------------------------------------------------------------ i2t link
def _client(monkeypatch):
    from lorastudio import i2tlink
    monkeypatch.setattr(i2tlink.bus, "publish", lambda *a, **k: None)
    return i2tlink, i2tlink.I2tClient()


def test_an_i2t_job_that_cannot_start_says_why(env, monkeypatch):
    i2tlink, c = _client(monkeypatch)
    monkeypatch.setattr(c, "_send", lambda obj: (_ for _ in ()).throw(RuntimeError("i2t を動かす環境がありません")))
    with pytest.raises(RuntimeError, match="環境がありません"):
        c.start_job("translate", words=["a"])
    assert c.job is None and c.last_done["ok"] is False


def test_a_second_i2t_job_is_refused_and_the_first_one_kept(env, monkeypatch):
    i2tlink, c = _client(monkeypatch)
    sent = []
    monkeypatch.setattr(c, "_send", lambda obj: sent.append(obj))
    first = c.start_job("analyze", path="a.png", cfg={"low_vram": True})
    with pytest.raises(RuntimeError, match="ほかの処理"):
        c.start_job("analyze", path="b.png", cfg={"low_vram": True})
    assert c.job["rid"] == first["rid"] and len(sent) == 1


def test_cancelling_an_i2t_analysis_that_waits_for_the_gpu(env, monkeypatch):
    i2tlink, c = _client(monkeypatch)
    hold = threading.Event()
    monkeypatch.setattr(i2tlink.gpu, "begin_analysis", lambda timeout=None: hold.wait(5))
    ended = []
    monkeypatch.setattr(i2tlink.gpu, "end_analysis", lambda: ended.append(1))
    sent = []
    monkeypatch.setattr(c, "_send", lambda obj: sent.append(obj))
    c.start_job("analyze", path="a.png", cfg={"low_vram": False})   # GPU free at the start: no "waiting" message
    assert c.job["message"] == "準備中..." and c.job["waiting_gpu"]
    c.control("cancel")
    assert c.job is None and c.last_done["cancelled"]
    hold.set()
    for _ in range(100):
        if ended:
            break
        time.sleep(0.02)
    assert ended == [1] and not sent and not c._gpu_held             # never sent, and the GPU is given back


def test_a_refused_worker_restart_does_not_extend_the_lockout(env, monkeypatch):
    i2tlink, c = _client(monkeypatch)
    monkeypatch.setattr(c, "available", lambda: True)
    c.proc = SimpleNamespace(poll=lambda: 1)       # the worker ended by itself
    old = time.time() - 590
    c._restarts = [old, old, old]
    for _ in range(5):                              # pages opened while it is locked out
        with pytest.raises(RuntimeError, match="繰り返し停止"):
            c._start()
    assert c._restarts == [old, old, old]           # ends 10 minutes after the last real restart


def test_what_downloads_change_is_asked_in_one_call(env, monkeypatch):
    i2tlink, c = _client(monkeypatch)
    calls = []

    def call(fn, *a, **k):
        calls.append(fn)
        if fn == "consts":
            return {"tagger_models": {"wd": {"label": "WD", "ready": False}, "camie": {"label": "C", "ready": False}},
                    "moss_catalog_ready": False}
        return {"tagger_ready": {"wd": True, "camie": False}, "moss_catalog_ready": True}
    monkeypatch.setattr(c, "call", call)
    c.consts()
    got = c.consts()
    assert calls == ["consts", "consts_now"]
    assert got["tagger_models"]["wd"]["ready"] and not got["tagger_models"]["camie"]["ready"] and got["moss_catalog_ready"]


def test_models_kept_with_low_vram_off_are_unloaded_for_training_only(env, monkeypatch):
    i2tlink, c = _client(monkeypatch)
    monkeypatch.setattr(i2tlink, "client", c)
    calls = []

    class R:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"{}"
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=10: calls.append(json.loads(req.data)["model"]) or R())
    c.kept_vlm = [{"host": "http://h", "model": "kept-2m", "low_vram": True},
                  {"host": "http://h", "model": "ollama-rule", "low_vram": False}]
    i2tlink.unload_kept_for_analysis()              # material analysis (or an i2t analysis itself) begins
    assert calls == ["kept-2m"] and [k["model"] for k in c.kept_vlm] == ["ollama-rule"]
    i2tlink.unload_kept_vlm()                       # training / generation takes the GPU
    assert calls == ["kept-2m", "ollama-rule"] and c.kept_vlm == []


# ------------------------------------------------------------------ i2t worker
def _worker(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["w", str(tmp_path), str(tmp_path / "d"), str(tmp_path / "m")])
    spec = importlib.util.spec_from_file_location("i2t_worker_1110", ROOT / "worker" / "i2t_worker.py")
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    (tmp_path / "d").mkdir(exist_ok=True)
    return w


def test_the_pages_old_copy_does_not_overwrite_the_measured_vlm_times(env, monkeypatch):
    w = _worker(monkeypatch, env)
    saved = {"vlm_seconds": {"qwen": 41.5}, "target": "sdxl"}
    written, out = [], []
    w.i2t = SimpleNamespace(load_config=lambda: dict(saved), save_config=lambda cfg: written.append(cfg),
                            describe_error=str)
    monkeypatch.setattr(w, "emit", out.append)
    w._call(1, {"fn": "save_config", "args": [{"target": "anima", "vlm_seconds": {}}]})   # the page was opened earlier
    assert written == [{"target": "anima", "vlm_seconds": {"qwen": 41.5}}] and out[0]["ok"]


def test_the_page_can_name_files_of_i2ts_own_folders_only(env, monkeypatch):
    w = _worker(monkeypatch, env)
    hist = env / "d" / "history"
    hist.mkdir()
    deleted = []
    w.i2t = SimpleNamespace(HISTORY_DIR=hist, history_delete=deleted.append,
                            history_update_prompts=lambda path, pos, neg: True,
                            render_openpose=lambda points, size: Image.new("RGB", (4, 4)))
    with pytest.raises(ValueError):
        w.history_delete("..\\..\\settings")
    with pytest.raises(ValueError):
        w.history_update_prompts(str(env / "settings.json"), "a", "b")
    with pytest.raises(ValueError):
        w.save_openpose([], [4, 4], str(env / "outside.png"))
    w.history_delete("20261110-120000")
    assert deleted == ["20261110-120000"]
    assert w.history_update_prompts(str(hist / "x.json"), "a", "b") is True
    assert Path(w.save_openpose([], [4, 4], str(env / "d" / "openpose.png"))).is_file()


def test_words_the_model_only_copied_are_not_asked_about_after_every_analysis(env, monkeypatch):
    w = _worker(monkeypatch, env)
    w.i2t = SimpleNamespace(chip_core=lambda t: (t, 1.0), lookup_ja=lambda table, core: table.get(core.lower(), ""),
                            dedup=lambda xs: list(dict.fromkeys(xs)), load_config=lambda: {"translate_model": "plamo"})
    monkeypatch.setattr(w, "chip_table", lambda: {"smile": "笑顔"})
    assert w.missing_ja(["smile", "1girl", "cowboy shot"], []) == ["1girl", "cowboy shot"]
    w._note_untranslated("plamo", ["1girl"], {})   # what a translation job left: it answered "1girl" for "1girl"
    assert w.missing_ja(["smile", "1girl", "cowboy shot"], []) == ["1girl", "cowboy shot"]   # once may be a hiccup
    w._note_untranslated("plamo", ["1girl"], {})   # ...twice: left out of the automatic translation
    assert w.missing_ja(["smile", "1girl", "cowboy shot"], []) == ["cowboy shot"]


def test_the_dictionary_is_read_again_only_when_a_file_changed(env, monkeypatch):
    w = _worker(monkeypatch, env)
    bundled = env / "tags_ja.json"
    bundled.write_text(json.dumps({"smile": "笑顔"}), encoding="utf-8")
    loads = []

    def load_tag_dictionary():
        loads.append(1)
        return json.loads(bundled.read_text(encoding="utf-8"))
    w.i2t = SimpleNamespace(USER_DICT_DIR=env / "dict", BUNDLED_DICT_PATH=bundled, load_tag_dictionary=load_tag_dictionary,
                            load_translations=lambda: {"cat ears": "猫耳"})
    a = w.chip_table()
    a["smile"] = "changed by the caller"
    b = w.chip_table()
    assert len(loads) == 1 and b == {"smile": "笑顔", "cat ears": "猫耳"}
    bundled.write_text(json.dumps({"smile": "笑顔", "blush": "赤面"}), encoding="utf-8")
    assert w.chip_table()["blush"] == "赤面" and len(loads) == 2
