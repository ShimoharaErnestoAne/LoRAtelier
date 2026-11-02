"""Failures that used to stop work for the rest of the session or touch files outside the project:
a zip member that resolves to the drive root, a hand-zipped project whose stored paths point into the
original, the analysis queue dying on one bad item, overlapping GPU users, and undo bringing back an
auto caption made for an old trigger."""
from __future__ import annotations

import io
import json
import sqlite3
import sys
import threading
import time
import zipfile
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "データ 空白"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config, projectio, store
    config._settings = None
    return {"store": store, "pio": projectio, "config": config, "tmp": tmp_path}


def _image(env, name="a.png", color="red"):
    f = env["tmp"] / name
    Image.new("RGB", (64, 64), color).save(f)
    return f


def test_zip_member_at_the_drive_root_is_rejected(env):
    pm = env["store"].ProjectManager()
    for evil in ("/evil_root.txt", "\\evil_root.txt", "C:/evil.txt", "sub/../../evil.txt"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("project.db", b"")
            zf.writestr(zipfile.ZipInfo(evil), b"x")   # ZipInfo keeps the name as given
        with zipfile.ZipFile(buf) as zf, pytest.raises(ValueError, match="不正なパス"):
            env["pio"]._members(zf)   # rejected before anything is extracted
        with pytest.raises(ValueError):
            env["pio"].import_zip(pm, buf)
    assert pm.list() == []


def test_hand_zipped_project_does_not_point_into_the_original(env):
    """No manifest: the stored absolute paths are rebased from the run folders, and deleting a run of the
    copy never moves the original's files."""
    pm = env["store"].ProjectManager()
    p = pm.create("元", "character", None)
    p.add_image_file(_image(env), "a.png", "ab" * 20)
    run_dir = p.runs_dir / "0001_x"
    (run_dir / "output").mkdir(parents=True)
    ck = run_dir / "output" / "x-step100.safetensors"
    ck.write_bytes(b"x" * 100)
    with p.tx() as db:
        rid = db.execute("INSERT INTO runs(created, status, name, config, run_dir) VALUES(?,?,?,?,?)",
                         (time.time(), "done", "#1", json.dumps({"out": str(run_dir / "output")}), str(run_dir))).lastrowid
        db.execute("INSERT INTO checkpoints(run_id, step, path, kind, created) VALUES(?,?,?,?,?)",
                   (rid, 100, str(ck), "lora", time.time()))
    p.db.execute("PRAGMA wal_checkpoint(FULL)")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:   # the folder zipped by hand: no lorastudio_project.json
        for f in p.path.rglob("*"):
            if f.is_file() and not f.name.endswith(("-wal", "-shm")):
                zf.write(f, f"{p.path.name}/{f.relative_to(p.path).as_posix()}")
    q = env["pio"].import_zip(pm, buf)
    r = q.run(rid)
    assert Path(r["run_dir"]).resolve().is_relative_to(q.path.resolve())
    assert r["config"]["out"].startswith(str(q.path))
    assert Path(q.checkpoints(rid)[0]["path"]).resolve().is_relative_to(q.path.resolve())
    env["pio"].delete_run(q, rid)
    assert ck.exists()   # the original is untouched


def test_run_folder_outside_the_project_is_never_removed(env):
    pm = env["store"].ProjectManager()
    p = pm.create("x", "style", None)
    other = env["tmp"] / "よその run"
    other.mkdir()
    with p.tx() as db:
        rid = db.execute("INSERT INTO runs(created, status, name, config, run_dir) VALUES(?,?,?,?,?)",
                         (time.time(), "done", "#1", "{}", str(other))).lastrowid
    env["pio"].delete_run(p, rid)
    assert other.exists() and p.run(rid) is None


def test_broken_project_db_in_zip_is_a_message_and_leaves_nothing(env):
    pm = env["store"].ProjectManager()
    db_file = env["tmp"] / "old.db"
    db = sqlite3.connect(db_file)
    db.execute("CREATE TABLE meta(key TEXT, value TEXT)")   # an old/partial database: no runs table
    db.commit()
    db.close()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.write(db_file, "project.db")
    with pytest.raises(ValueError, match="project.db"):
        env["pio"].import_zip(pm, buf)
    assert not any((pm.root / ".importing").iterdir())   # built next to the projects (same drive)


def test_analysis_queue_keeps_running_after_a_failing_item(env, monkeypatch):
    from lorastudio import analysis
    pm = env["store"].ProjectManager()
    p = pm.create("x", "style", None)
    a, _ = p.add_image_file(_image(env, "a.png"), "a.png", "aa" * 20)
    b, _ = p.add_image_file(_image(env, "b.png", "blue"), "b.png", "bb" * 20)
    calls = []

    def call(op, **kw):
        calls.append(kw["path"])
        if len(calls) == 1:
            raise RuntimeError("解析ワーカーが繰り返し停止したため、自動再起動を止めました。")
        return {"ok": False, "error": "読めません"}
    monkeypatch.setattr(analysis.worker, "available", lambda: True)
    monkeypatch.setattr(analysis.worker, "call", call)
    an = analysis.Analyzer(pm)
    an._enqueue(5, ("analyze", p.id, a))
    an._enqueue(5, ("analyze", p.id, b))
    end = time.time() + 5
    while time.time() < end and (p.image(b)["ai_status"] != "failed" or p.image(a)["ai_status"] != "failed"):
        time.sleep(0.05)
    assert p.image(a)["ai_status"] == "failed" and "自動再起動" in p.image(a)["ai_error"]
    assert p.image(b)["ai_status"] == "failed" and len(calls) == 2   # the second image was still processed
    from lorastudio.gpu import gpu
    assert not gpu.analysis_active


def test_gpu_waits_until_every_analysis_has_ended():
    from lorastudio.gpu import GPU
    g = GPU()
    assert g.begin_analysis() and g.begin_analysis()   # e.g. material analysis and i2t at once
    got = threading.Event()
    threading.Thread(target=lambda: (g.acquire("学習"), got.set()), daemon=True).start()
    g.end_analysis()
    assert not got.wait(0.3)   # the other one still uses the GPU
    g.end_analysis()
    assert got.wait(2)
    g.release("学習")


def test_undo_shows_the_current_auto_caption(env):
    pm = env["store"].ProjectManager()
    p = pm.create("x", "character", None)
    iid, _ = p.add_image_file(_image(env), "a.png", "ab" * 20)
    p.set_auto_caption(iid, "old_trigger, red", "k-old")
    p.edit("手動", [(iid, "caption", "手で直した"), (iid, "caption_edited", 1)])
    p.set_auto_caption(iid, "new_trigger, red", "k-new")   # the trigger changed while the caption was edited
    p.undo()
    im = p.image(iid)
    assert im["caption"] == "new_trigger, red" and not im["caption_edited"]
    p.redo()
    assert p.image(iid)["caption"] == "手で直した"


def test_settings_that_are_not_an_object_are_set_aside(env):
    cfg = env["config"]
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg.SETTINGS_FILE.write_text("[1, 2]", encoding="utf-8")
    s = cfg.Settings()
    assert s.get("models") == [] and not cfg.SETTINGS_FILE.exists()
    assert list(cfg.DATA_DIR.glob("settings.broken-*.json"))


def test_unreadable_gpu_memory_is_skipped(monkeypatch):
    from lorastudio import gpu as gpu_mod

    class R:
        stdout = "Laptop GPU, [N/A], [N/A], 560.1\nRTX, 16384, 100, 581.0\n"
    monkeypatch.setattr(gpu_mod.shutil, "which", lambda _: "nvidia-smi")
    monkeypatch.setattr(gpu_mod.subprocess, "run", lambda *a, **k: R())
    assert [g["name"] for g in gpu_mod.query_gpus()] == ["RTX"]


def test_reused_parsed_analysis_follows_every_change(env):
    """The parsed ai/fast columns are reused for speed: a new analysis result must show at once, and a
    caller changing its copy must not change what the next read returns."""
    pm = env["store"].ProjectManager()
    p = pm.create("キャッシュ", "character", None)
    f = _image(env)
    iid, _ = p.add_image_file(f, "a.png", "ab" * 32)
    p.set_fields(iid, ai={"domain": "anime", "embed": [0.1, 0.2]})
    first = p.image(iid)
    first["ai"]["domain"] = "changed by caller"
    assert p.image(iid)["ai"]["domain"] == "anime"
    p.set_fields(iid, ai={"domain": "photo", "embed": [0.3]})
    assert p.images()[0]["ai"] == {"domain": "photo", "embed": [0.3]}
    assert p.run(12345) is None
