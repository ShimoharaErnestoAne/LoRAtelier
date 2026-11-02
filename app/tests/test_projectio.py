"""Project deletion / zip export and import / run deletion.  Risks: an import that still points at the old
project's files (resume or export would read the wrong folder), a zip that writes outside the project
folder, and a deletion that leaves the project half-open."""
from __future__ import annotations

import io
import json
import sys
import time
import zipfile
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    data = tmp_path / "データ 空白"
    monkeypatch.setenv("LORASTUDIO_DATA", str(data))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config, projectio, store
    config._settings = None
    return {"store": store, "pio": projectio, "tmp": tmp_path}


def _project_with_run(env):
    pm = env["store"].ProjectManager()
    p = pm.create("テスト キャラ", "character", None)
    img = env["tmp"] / "a.png"
    Image.new("RGB", (64, 64), "red").save(img)
    iid, _ = p.add_image_file(img, "a.png", "ab" * 20)
    p.edit("説明文", [(iid, "caption", "chr_test, red")])
    run_dir = p.runs_dir / "0001_x"
    (run_dir / "output").mkdir(parents=True)
    ck = run_dir / "output" / "chr_r1-step100.safetensors"
    ck.write_bytes(b"x" * 1000)
    cfg = {"items": [{"src": str(p.path / "sources" / "x.png")}], "values": {}}
    (run_dir / "run.json").write_text(json.dumps(cfg), encoding="utf-8")
    with p.tx() as db:
        rid = db.execute("INSERT INTO runs(created, status, name, config, run_dir) VALUES(?,?,?,?,?)",
                         (time.time(), "done", "#1", json.dumps(cfg), str(run_dir))).lastrowid
        db.execute("INSERT INTO checkpoints(run_id, step, path, kind, created) VALUES(?,?,?,?,?)",
                   (rid, 100, str(ck), "lora", time.time()))
    return pm, p, rid


def _zip(env, p, runs):
    return io.BytesIO(b"".join(env["pio"].export_stream(p, runs)))


def test_export_import_rebases_paths(env):
    pm, p, rid = _project_with_run(env)
    buf = _zip(env, p, True)
    q = env["pio"].import_zip(pm, buf)
    assert q.id != p.id and q.meta("name") == "テスト キャラ"
    im = q.images()[0]
    assert im["caption"] == "chr_test, red" and q.source_path(im).exists()
    assert q.history_state()["undo"] == "説明文"
    run = q.run(rid)
    assert run["run_dir"].startswith(str(q.path)) and Path(run["run_dir"]).is_dir()
    assert run["config"]["items"][0]["src"].startswith(str(q.path))
    ck = q.checkpoints(rid)[0]["path"]
    assert ck.startswith(str(q.path)) and Path(ck).read_bytes() == b"x" * 1000
    rj = json.loads((Path(run["run_dir"]) / "run.json").read_text(encoding="utf-8"))
    assert rj["items"][0]["src"].startswith(str(q.path))
    assert {x["id"] for x in pm.list()} == {p.id, q.id}


def test_export_without_runs_drops_run_rows_and_files(env):
    pm, p, rid = _project_with_run(env)
    buf = _zip(env, p, False)
    names = zipfile.ZipFile(buf).namelist()
    assert not any(n.startswith("runs/") for n in names)
    buf.seek(0)
    q = env["pio"].import_zip(pm, buf)
    assert q.runs() == [] and len(q.images()) == 1


def test_import_rejects_unsafe_and_foreign_zips(env):
    pm = env["store"].ProjectManager()
    bad = io.BytesIO()
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("project.db", b"")
        zf.writestr("../evil.txt", b"x")
    with pytest.raises(ValueError):
        env["pio"].import_zip(pm, bad)
    other = io.BytesIO()
    with zipfile.ZipFile(other, "w") as zf:
        zf.writestr("photo.jpg", b"x")
    with pytest.raises(ValueError):
        env["pio"].import_zip(pm, other)
    assert pm.list() == []
    assert not (env["tmp"] / "evil.txt").exists()


def test_delete_project_and_run(env):
    pm, p, rid = _project_with_run(env)
    run_dir = Path(p.run(rid)["run_dir"])
    env["pio"].delete_run(p, rid)
    assert p.run(rid) is None and p.checkpoints(rid) == []
    assert not run_dir.exists()
    path = p.path
    env["pio"].delete_project(pm, p.id)
    assert not path.exists() and pm.list() == []
    with pytest.raises(KeyError):
        pm.get(p.id)
