"""Comparing runs, the reservation queue and whole-tag caption edits.  Failure risks: a renamed trigger hiding the
real caption changes, images of different conditions shown as comparable, a reservation lost or reordered by a
restart, a resumed run retrained from the start over its saves, a batch reservation sending invalid settings."""
from __future__ import annotations

import json
import sys
import threading
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


def _snap(trigger="chr_a", values=None, items=None, model="m1"):
    return {"kind": "character", "trigger": trigger, "outfit_mode": "include", "model": {"id": model, "name": model},
            "engine_commit": "abcdef123", "preset_version": "p1", "values": values or {"learning_rate": 4e-4, "max_train_steps": 1600},
            "items": items if items is not None else [{"id": 1, "caption": f"{trigger}, 1girl, red hair", "crop": None},
                                                      {"id": 2, "caption": f"{trigger}, 1girl, smile", "crop": None}]}


def test_caption_changes_ignore_a_renamed_trigger(env):
    from lorastudio import runcompare
    a = _snap("chr_a")
    b = _snap("chr_b", values={"learning_rate": 2e-4, "max_train_steps": 1600},
              items=[{"id": 1, "caption": "chr_b, 1girl, red hair", "crop": None},
                     {"id": 2, "caption": "chr_b, 1girl, open mouth", "crop": [0, 0, 10, 10]},
                     {"id": 3, "caption": "chr_b, 1girl", "crop": None}])
    m = runcompare.material_diff(a, b)
    assert m["added"] == [3] and m["removed"] == [] and m["crop"] == [2]
    assert [c["id"] for c in m["caption"]] == [2]          # image 1 only had the trigger renamed
    assert m["caption"][0]["added"] == ["open mouth"] and m["caption"][0]["removed"] == ["smile"]
    s = runcompare.change_summary(a, b)
    assert any("呼び出し語" in x for x in s) and any("学習率" in x and "0.0004" in x and "0.0002" in x for x in s)
    assert any("1 枚増やし" in x for x in s) and any("1 枚の説明文" in x for x in s)
    assert runcompare.change_summary(None, b) == []
    rows = {r["key"]: r for r in runcompare.settings_rows([a, b])}
    assert rows["learning_rate"]["differs"] and not rows["max_train_steps"]["differs"]


def _eval(eid, trigger, imgs, cond=("euler", 1024), status="done", scores=None):
    prompts = [f"q, {trigger}, solo, city", f"q, {trigger}, solo, cafe"]
    return {"id": eid, "status": status, "spec": {"prompts": prompts, "width": cond[1], "height": cond[1]},
            "results": {"images": imgs, "scores": scores or {}, "conditions": {"sampling": {"sampler": cond[0]}, "negative": "n"}}}


def test_images_match_across_runs_only_under_the_same_conditions(env):
    from lorastudio import runcompare
    im = lambda p, seed, ck, s: {"prompt": p, "seed": seed, "ckpt": ck, "strength": s, "file": f"p{p}_{ck}_{s}.png"}
    r1 = {"id": 1, "name": "#1", "snap": _snap("chr_a"),
          "evals": [_eval(10, "chr_a", [im(0, 1234, "base", 0.0), im(0, 1234, "final", 1.0), im(1, 1234, "final", 1.0)],
                          scores={"ccip_diff": [0.4, 0.2, 0.3]})]}
    # trigger renamed: same prompts otherwise, so the images still pair up
    r2 = {"id": 2, "name": "#2", "snap": _snap("chr_b"),
          "evals": [_eval(20, "chr_b", [im(0, 1234, "final", 1.0), im(0, 1234, "final", 0.6)], scores={"ccip_diff": [0.1, 0.2]})]}
    # another sampler: never put next to the others
    r3 = {"id": 3, "name": "#3", "snap": _snap("chr_a"),
          "evals": [_eval(30, "chr_a", [im(0, 1234, "final", 1.0)], cond=("dpm", 1024))]}
    res = runcompare.compare_images([r1, r2, r3], "final", 1.0)
    assert len(res["rows"]) == 1 and res["conds"] == 2
    row = res["rows"][0]
    assert set(row["cells"]) == {1, 2} and row["base"]["file"] == "p0_base_0.0.png"
    assert row["cells"][2]["scores"]["ccip_diff"] == 0.1 and row["seed"] == 1234
    assert res["strengths"] == [0.6, 1.0]
    # an image of one run alone (prompt 1 of run 1, no base) is no comparison
    assert all(r["prompt"].endswith("city") for r in res["rows"])


def test_aligned_generation_skips_what_exists_and_makes_no_lora_once(env):
    from lorastudio import runcompare
    have = [{"prompt": pi, "seed": 1234, "ckpt": ck, "strength": s, "file": "x.png"}
            for pi in (0, 1) for ck, s in (("base", 0.0), ("final", 1.0))]
    r1 = {"id": 1, "name": "#1", "snap": _snap("chr_a"), "evals": [_eval(10, "chr_a", have)]}
    r2 = {"id": 2, "name": "#2", "snap": _snap("chr_b"), "evals": []}
    prompts = lambda r: [f"q, {r['snap']['trigger']}, solo, city", f"q, {r['snap']['trigger']}, solo, cafe"]
    specs = runcompare.align_specs([r1, r2], "final", 1.0, [1234], prompts, lambda r: (1024, 1024))
    assert [rid for rid, _ in specs] == [2]                 # run 1 already has every image
    sp = specs[0][1]
    assert sp["include_base"] is False and sp["checkpoints"] == ["final"] and sp["prompts"][0] == "q, chr_b, solo, city"
    specs = runcompare.align_specs([r2, r1], "final", 1.0, [1234], prompts, lambda r: (1024, 1024))
    assert specs[0][1]["include_base"] is True             # "no LoRA" with the first run only


def test_whole_tag_replace(env):
    from lorastudio import captions
    assert captions.replace_tag("chr_a, red hair, smile", "red hair", "blue hair") == "chr_a, blue hair, smile"
    assert captions.replace_tag("chr_a, red hair, smile", "red", "blue") == "chr_a, red hair, smile"   # no partial match
    assert captions.replace_tag("chr_a, red hair, blue hair", "red hair", "blue hair") == "chr_a, blue hair"
    assert captions.replace_tag("chr_a, red hair", "red hair", "") == "chr_a"


class _P:
    def __init__(self, pid="p"):
        self.id = pid


def test_queue_reorder_and_file_follow_the_waiting_runs(env):
    from lorastudio import training
    r = training.Runner(None)
    r._cond.acquire()   # keep the runner thread from taking jobs while the test looks at the queue
    try:
        r._set = lambda *a, **k: None
        for rid in (1, 2, 3):
            r.jobs.append(training.Job(_P(), rid))
        r._save_queue()
        uids = [j.uid for j in r.jobs]
    finally:
        r._cond.release()
    with r._cond:
        pass
    # the runner may have taken the first job by now; work on what is waiting
    waiting = r.waiting()
    last = waiting[-1]
    assert r.move(last.uid, "top")
    assert r.waiting()[0].uid == last.uid
    saved = json.loads(training.QUEUE_FILE.read_text(encoding="utf-8"))
    assert saved[0]["run"] == last.run_id and len(saved) == len(r.waiting())
    assert not r.move(10 ** 9, "up")
    assert uids


def test_reservations_survive_a_restart_in_their_order(env, monkeypatch):
    from lorastudio import server, training
    monkeypatch.setattr(server.analyzer, "submit_fast", lambda *a, **k: None)
    p = server.projects.create("予約 テスト", "character", None)
    rows = []
    with p.tx() as db:
        for i in range(4):
            rid = db.execute("INSERT INTO runs(created, status, name, config, run_dir, progress) VALUES(?,?,?,?,?,?)",
                             (100 + i, "queued", f"#{i}", json.dumps({"items": [], "values": {}}), str(p.runs_dir / f"r{i}"),
                              json.dumps({"phase": "queued"}))).lastrowid
            rows.append(rid)
        # run 3 had trained and was waiting to resume from a save that no longer exists
        db.execute("INSERT INTO checkpoints(run_id, step, path, kind, created) VALUES(?,?,?,?,?)", (rows[3], 100, "x", "lora", 1))
    state = p.runs_dir / "state-100"
    state.mkdir()
    training.QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    training.QUEUE_FILE.write_text(json.dumps([
        {"project": p.id, "run": rows[2], "resume": None},
        {"project": p.id, "run": rows[0], "resume": str(state)},
        {"project": p.id, "run": rows[3], "resume": str(p.runs_dir / "gone")}]), encoding="utf-8")
    got = []
    monkeypatch.setattr(server.runner, "enqueue", lambda proj, rid, resume=None: got.append((rid, resume)))
    server.resume_pending_on_startup()
    # file order first, then a reserved run missing from the file (never trained) by creation time
    assert got == [(rows[2], None), (rows[0], str(state)), (rows[1], None)]
    r3 = p.run(rows[3])
    assert r3["status"] == "failed" and "保存地点" in json.loads(r3["error"])["message"]


def test_batch_reservation_varies_one_setting_and_rejects_bad_values(env, monkeypatch):
    from fastapi.testclient import TestClient
    from lorastudio import server
    p = server.projects.create("まとめ", "character", None)
    vals = {"max_train_steps": 1600, "save_every_n_steps": 200, "learning_rate": 4e-4, "network_dim": 16, "network_alpha": 8,
            "resolution": 1024, "seed": 42, "train_batch_size": 2}
    plan = {"preflight": {"errors": [], "warnings": [], "info": {}}, "values": dict(vals), "defaults": dict(vals), "changed": [],
            "model": {"family": "sdxl"}}
    monkeypatch.setattr(server, "_plan_with", lambda proj: (plan, [], None))
    made = []
    monkeypatch.setattr(server, "create_run", lambda proj, used, v, changed, name, members=None: made.append((v, changed, name)) or len(made))
    monkeypatch.setattr(server.runner, "enqueue", lambda *a, **k: None)
    c = TestClient(server.app)
    r = c.post(f"/api/projects/{p.id}/train/batch", json={"field": "network_dim", "values": [8, 32, "32"]})
    assert r.status_code == 200 and r.json()["run_ids"] == [1, 2]
    assert [m[0]["network_alpha"] for m in made] == [4, 16]          # alpha / rank ratio kept
    assert "network_dim" in made[0][1] and made[0][2] == "rank (network_dim) 8"
    made.clear()
    r = c.post(f"/api/projects/{p.id}/train/batch", json={"field": "max_train_steps", "values": [100]})
    assert r.status_code == 200 and made[0][0]["save_every_n_steps"] == 100   # save interval clamped to the run
    made.clear()
    assert c.post(f"/api/projects/{p.id}/train/batch", json={"field": "learning_rate", "values": [1]}).status_code == 400
    assert c.post(f"/api/projects/{p.id}/train/batch", json={"field": "optimizer_type", "values": ["x"]}).status_code == 400
    assert c.post(f"/api/projects/{p.id}/train/batch", json={"field": "seed", "values": list(range(9))}).status_code == 400
    assert not made                                                  # nothing reserved by a rejected request
