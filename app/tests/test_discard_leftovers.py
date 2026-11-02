"""「学習データを破棄」 after an export.  Risks: the exported version, the material or the captions removed (the
project can no longer be trained again), and files of a run that is still training or being compared removed."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "データ 空白"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config, projectio, store
    config._settings = None
    return {"pm": store.ProjectManager(), "io": projectio}


def _run(p, n: int, status: str = "done"):
    run_dir = p.runs_dir / f"{n:04d}_x"
    out, ds = run_dir / "output", run_dir / "dataset"
    out.mkdir(parents=True)
    ds.mkdir()
    (ds / "a.png").write_bytes(b"img")
    (ds / "a.txt").write_text("caption", encoding="utf-8")
    (ds / "a_1024x1024_sdxl.npz").write_bytes(b"c" * 500)
    with p.tx() as db:
        rid = db.execute("INSERT INTO runs(created, status, name, config, run_dir) VALUES(?,?,?,?,?)",
                         (time.time(), status, f"#{n}", json.dumps({"items": []}), str(run_dir))).lastrowid
        paths = {}
        for step in (100, 200):
            f = out / f"x-step{step}.safetensors"
            f.write_bytes(b"w" * 1000)
            paths[step] = str(f)
            db.execute("INSERT INTO checkpoints(run_id, step, path, kind, created) VALUES(?,?,?,?,?)",
                       (rid, step, str(f), "lora", time.time()))
        st = out / "x-step200-state"
        st.mkdir()
        (st / "optimizer.bin").write_bytes(b"o" * 3000)
        db.execute("INSERT INTO checkpoints(run_id, step, path, kind, created) VALUES(?,?,?,?,?)",
                   (rid, 200, str(st), "state", time.time()))
    return rid, paths, run_dir


def test_discard_keeps_the_exported_version_and_the_material(env):
    p = env["pm"].create("破棄", "character", None)
    (p.sources / "a.png").write_bytes(b"source")
    rid, paths, run_dir = _run(p, 1)
    busy, busy_paths, busy_dir = _run(p, 2, "training")
    with p.tx() as db:
        db.execute("INSERT INTO exports(run_id, created, ckpt_path, dest) VALUES(?,?,?,?)",
                   (rid, time.time(), paths[200], "E:/elsewhere"))
    (p.path / "traincache" / "k" / "L").mkdir(parents=True)
    (p.path / "traincache" / "k" / "L" / "x.npz").write_bytes(b"s" * 700)

    lo = env["io"].training_leftovers(p, {busy})
    assert lo["bytes"] == 1000 + 3000 + 500   # step 100, the state, the run's cache (not the store: a run is busy)
    freed = env["io"].discard_training_leftovers(p, {busy})
    assert freed == lo["bytes"]
    assert [c["path"] for c in p.checkpoints(rid)] == [paths[200]] and Path(paths[200]).exists()
    assert not Path(paths[100]).exists() and not (run_dir / "output" / "x-step200-state").exists()
    assert not (run_dir / "dataset" / "a_1024x1024_sdxl.npz").exists()
    assert (run_dir / "dataset" / "a.txt").exists() and (p.sources / "a.png").read_bytes() == b"source"
    # the run still training is left alone, and so is the shared cache store it may copy from
    assert len(p.checkpoints(busy)) == 3 and (busy_dir / "dataset" / "a_1024x1024_sdxl.npz").exists()
    assert (p.path / "traincache" / "k" / "L" / "x.npz").exists()

    # nothing in use: the store goes too
    assert env["io"].training_leftovers(p)["bytes"] == 1000 * 2 + 3000 + 500 + 700
    env["io"].discard_training_leftovers(p)
    assert not (p.path / "traincache").exists() and env["io"].training_leftovers(p)["bytes"] == 0


def test_settings_button_counts_and_discards_every_project(env, monkeypatch):
    from lorastudio import server
    a = server.projects.create("一つ目", "character", None)
    b = server.projects.create("二つ目", "style", None)
    ra, pa, _ = _run(a, 1)
    _run(b, 1)
    with a.tx() as db:
        db.execute("INSERT INTO exports(run_id, created, ckpt_path, dest) VALUES(?,?,?,?)", (ra, time.time(), pa[200], "x"))
    n = server.all_leftovers()
    assert n == {"bytes": (1000 + 3000 + 500) + (2000 + 3000 + 500), "projects": 2, "skipped": False}
    assert server.discard_all_leftovers() == {"freed": n["bytes"], "skipped": False}
    assert server.all_leftovers()["bytes"] == 0 and Path(pa[200]).exists()
