"""Moving data between places (settings page).  Risks: an original removed before its copy is complete, a moved
project whose stored paths still point at the old folder (results, export and resume break), a cancel or a failed
copy that loses data, a project moved while work still uses it, and images that can no longer be opened."""
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
    from lorastudio import config, i2tlink, imagegen, relocate, store
    config._settings = None
    config.set_storage_base("projects", str(tmp_path / "HDD 置き場"))
    config.set_storage_base("imagegen", str(tmp_path / "HDD 置き場"))
    config.set_storage_base("i2t", str(tmp_path / "HDD 置き場"))
    return {"config": config, "store": store, "rel": relocate, "ig": imagegen, "i2t": i2tlink,
            "hdd": tmp_path / "HDD 置き場" / "projects", "ssd": config.PROJECTS_DIR}


def _project(env, root: Path, name: str):
    p = env["store"].ProjectManager(root).create(name, "character", None)
    run_dir = p.runs_dir / "0001_x"
    (run_dir / "output").mkdir(parents=True)
    ck = run_dir / "output" / "a-step100.safetensors"
    ck.write_bytes(b"w" * 3000)
    cfg = {"items": [{"src": str(p.path / "sources" / "x.png")}], "dir": str(p.path)}
    (run_dir / "run.json").write_text(json.dumps(cfg), encoding="utf-8")
    (run_dir / "dataset.toml").write_text(f"image_dir = '{run_dir / 'img'}'\n", encoding="utf-8")
    with p.tx() as db:
        rid = db.execute("INSERT INTO runs(created, status, name, config, run_dir) VALUES(?,?,?,?,?)",
                         (time.time(), "done", "#1", json.dumps(cfg), str(run_dir))).lastrowid
        db.execute("INSERT INTO checkpoints(run_id, step, path, kind, created) VALUES(?,?,?,?,?)",
                   (rid, 100, str(ck), "lora", time.time()))
    p.close()
    return p.id, p.path


def _wait(rel):
    for _ in range(200):
        if not rel.status()["running"]:
            return rel.status()
        time.sleep(0.05)
    raise AssertionError("move did not end")


@pytest.mark.parametrize("other_drive", [True, False])
def test_project_moves_with_its_paths(env, monkeypatch, other_drive):
    rel = env["rel"]
    monkeypatch.setattr(rel, "same_drive", lambda a, b: not other_drive)
    pid, old = _project(env, env["ssd"], "移す")
    sib, sib_path = _project(env, env["ssd"], "移す")   # same name: the folder name starts the same (+ "_2")
    pm = env["store"].ProjectManager()
    pm.get(pid)   # open: the move closes it first
    rel.start("projects", [pid], str(env["hdd"]), pm, lambda _pid: None)
    s = _wait(rel)
    assert s["errors"] == [] and s["done"] == 1
    new = env["hdd"] / pid
    assert pm.path_of(pid) == new and not (old / "project.db").exists()
    assert {x["id"] for x in pm.list()} == {pid, sib}
    p = pm.get(pid)
    run = p.runs()[0]
    assert run["run_dir"] == str(new / "runs" / "0001_x")
    assert run["config"]["dir"] == str(new) and run["config"]["items"][0]["src"] == str(new / "sources" / "x.png")
    assert Path(p.checkpoints(run["id"])[0]["path"]).read_bytes() == b"w" * 3000
    rj = json.loads((new / "runs" / "0001_x" / "run.json").read_text(encoding="utf-8"))
    assert rj["dir"] == str(new)
    assert str(new) in (new / "runs" / "0001_x" / "dataset.toml").read_text(encoding="utf-8")
    assert not list(env["hdd"].glob(".moving_*"))
    # the other project, whose folder name only starts the same, is untouched
    q = pm.get(sib)
    assert q.path == sib_path and q.runs()[0]["run_dir"].startswith(str(sib_path))


def test_cancel_and_failed_copy_keep_the_original(env, monkeypatch):
    rel = env["rel"]
    monkeypatch.setattr(rel, "same_drive", lambda a, b: False)
    pid, old = _project(env, env["ssd"], "残す")
    pm = env["store"].ProjectManager()
    real = rel._copy_file

    def cancel_midway(src, dst):
        if src.suffix == ".safetensors":
            rel._cancel.set()
        return real(src, dst)
    monkeypatch.setattr(rel, "_copy_file", cancel_midway)
    rel.start("projects", [pid], str(env["hdd"]), pm, None)
    assert _wait(rel)["cancelled"]
    assert pm.get(pid).path == old and (old / "project.db").exists() and not list(env["hdd"].glob(".moving_*"))

    def broken(src, dst):
        real(src, dst)
        if src.suffix == ".safetensors":
            raise OSError(112, "ディスクがいっぱい")
    monkeypatch.setattr(rel, "_copy_file", broken)
    rel.start("projects", [pid], str(env["hdd"]), pm, None)
    s = _wait(rel)
    assert s["errors"] and pm.get(pid).path == old and pm.get(pid).runs()[0]["run_dir"].startswith(str(old))
    assert not list(env["hdd"].glob(".moving_*"))


def test_busy_project_is_not_moved_and_moving_one_is_not_opened(env, monkeypatch):
    rel = env["rel"]
    pid, old = _project(env, env["ssd"], "学習中")
    pm = env["store"].ProjectManager()
    rel.start("projects", [pid], str(env["hdd"]), pm, lambda _pid: "学習 #1 が実行中です。")
    s = _wait(rel)
    assert s["errors"][0]["message"] == "学習 #1 が実行中です。" and pm.get(pid).path == old
    seen = {}

    def slow_tree(src, dst):   # while the copy runs, the project cannot be opened
        try:
            pm.get(pid)
        except env["store"].ProjectMoving:
            seen["moving"] = True
        r = orig(src, dst)
        # the list is shown meanwhile: the moving project is marked, the half-made copy is never opened
        seen["list"] = pm.list()
        return r
    orig = rel._copy_tree
    monkeypatch.setattr(rel, "same_drive", lambda a, b: False)
    monkeypatch.setattr(rel, "_copy_tree", slow_tree)
    rel.start("projects", [pid], str(env["hdd"]), pm, None)
    assert _wait(rel)["errors"] == [] and seen.get("moving")
    assert [(x["id"], x.get("moving"), x["name"]) for x in seen["list"]] == [(pid, True, "学習中")]
    assert not any(k.startswith(".") for k in pm._open)
    assert pm.get(pid).path == env["hdd"] / pid


def test_leftover_copy_is_cleaned_at_start(env):
    pid, old = _project(env, env["ssd"], "残り")
    left = env["hdd"] / f".moving_{pid}"
    left.mkdir(parents=True)
    (left / "project.db").write_bytes(b"half")
    pm = env["store"].ProjectManager()
    assert [x["id"] for x in pm.list()] == [pid]
    from lorastudio import projectio
    projectio.empty_trash()
    assert not left.exists() and pm.get(pid).path == old


def test_images_move_one_by_one_and_stay_reachable(env, monkeypatch):
    rel, c = env["rel"], env["config"]
    monkeypatch.setattr(rel, "same_drive", lambda a, b: False)
    old = env["ig"].IMG_DIR
    (old / "thumbs").mkdir(parents=True)
    (old / "a_0_s1.png").write_bytes(b"png a")
    (old / "thumbs" / "a_0_s1.webp").write_bytes(b"t")
    i2t_old = env["i2t"].IMAGES_DIR
    i2t_old.mkdir(parents=True)
    (i2t_old / "x.png").write_bytes(b"png x")
    for kind, src in (("imagegen", old), ("i2t", c.STORAGE["i2t"][0])):
        rel.start(kind, [str(src)], str(c.storage_dir(kind)), None, None)
        assert _wait(rel)["errors"] == []
    new = c.storage_dir("imagegen")
    assert env["ig"].img_file("a_0_s1.png") == new / "a_0_s1.png" and (new / "thumbs" / "a_0_s1.webp").exists()
    assert not (old / "a_0_s1.png").exists()
    moved = c.storage_dir("i2t") / "images" / "x.png"
    assert moved.read_bytes() == b"png x" and not (i2t_old / "x.png").exists()
    # i2t history keeps the old path: it is found at the new place
    assert env["i2t"].allowed_file(str(i2t_old / "x.png")) == moved.resolve()
