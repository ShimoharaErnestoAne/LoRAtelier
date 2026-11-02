"""Data places set on the settings page (projects / imagegen images / i2t images on another drive).  Risks: projects
made before the change disappear from the list, a deletion or an import fails because a folder cannot be moved
across drives, a generated image made in the old place is not found, and a path that cannot be used is accepted."""
from __future__ import annotations

import io
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
    from lorastudio import config, i2tlink, imagegen, projectio, store
    config._settings = None
    return {"config": config, "store": store, "pio": projectio, "ig": imagegen, "i2t": i2tlink,
            "hdd": tmp_path / "HDD 置き場"}


def test_new_projects_go_to_the_set_place_and_old_ones_stay_listed(env):
    c, pm = env["config"], env["store"].ProjectManager()
    old = pm.create("前の", "character", None)
    assert old.path.parent == c.PROJECTS_DIR
    c.set_storage_base("projects", str(env["hdd"]))
    new = pm.create("新しい", "style", None)
    assert new.path.parent == env["hdd"] / "projects"
    assert {p["id"] for p in pm.list()} == {old.id, new.id}
    # a fresh manager (= after a restart) finds both
    pm2 = env["store"].ProjectManager()
    assert pm2.get(old.id).path == old.path and pm2.get(new.id).path == new.path
    # back to the default: the HDD place stays among the places looked in
    c.set_storage_base("projects", "")
    assert c.storage_dir("projects") == c.PROJECTS_DIR
    assert {p["id"] for p in env["store"].ProjectManager().list()} == {old.id, new.id}
    with pytest.raises(KeyError):
        pm2.get("..\\" + old.id)


def test_unusable_paths_are_refused(env, tmp_path):
    c = env["config"]
    for bad in ("relative\\dir", "\\no_drive"):
        with pytest.raises(ValueError):
            c.set_storage_base("projects", bad)
    afile = tmp_path / "file.txt"
    afile.write_text("x")
    with pytest.raises(ValueError):
        c.set_storage_base("projects", str(afile))   # "projects" cannot be made inside a file
    assert c.storage_base("projects") is None


def test_place_on_a_missing_drive_does_not_break_the_list(env, monkeypatch):
    c, pm = env["config"], env["store"].ProjectManager()
    p = pm.create("a", "character", None)
    c.set_storage_base("projects", str(env["hdd"]))
    import shutil
    shutil.rmtree(env["hdd"])   # as if the HDD were unplugged
    assert [x["id"] for x in env["store"].ProjectManager().list()] == [p.id]
    with pytest.raises(ValueError, match="書き込めません"):
        monkeypatch.setattr(Path, "mkdir", lambda *a, **k: (_ for _ in ()).throw(OSError("no drive")))
        pm.create("b", "character", None)


def test_delete_and_import_on_another_drive(env, monkeypatch):
    c, pio = env["config"], env["pio"]
    c.set_storage_base("projects", str(env["hdd"]))
    pm = env["store"].ProjectManager()
    p = pm.create("消す", "character", None)
    import lorastudio.linking as linking
    monkeypatch.setattr(linking, "same_drive", lambda a, b: False)   # the HDD is another drive than data/
    assert pio._trash_dir(p.path) == env["hdd"] / "projects" / ".trash"
    pio.delete_project(pm, p.id)
    assert not p.path.exists() and p.id not in {x["id"] for x in pm.list()}
    # import: built next to its final place, so it is moved into place in one step
    q = pm.create("出す", "character", None)
    buf = io.BytesIO(b"".join(pio.export_stream(q, False)))
    time.sleep(1.1)   # the imported folder is named by the time
    r = pio.import_zip(pm, buf)
    assert r.path.parent == env["hdd"] / "projects"
    assert not any((env["hdd"] / "projects" / ".importing").iterdir())


def test_generated_images_found_in_old_and_new_place(env):
    c, ig = env["config"], env["ig"]
    old = ig.IMG_DIR / "a_0_s1.png"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"old")
    c.set_storage_base("imagegen", str(env["hdd"]))
    assert c.storage_dir("imagegen") == env["hdd"] / "imagegen"
    assert ig.img_file("a_0_s1.png") == old
    assert ig.img_file("b_0_s1.png") == env["hdd"] / "imagegen" / "b_0_s1.png"   # new images go here


def test_i2t_images_go_to_the_set_place_and_stay_allowed(env, tmp_path):
    c, link = env["config"], env["i2t"]
    before = link.save_upload("a.png", b"\x89PNG a")
    c.set_storage_base("i2t", str(env["hdd"]))
    after = link.save_upload("b.png", b"\x89PNG b")
    assert after.parent == env["hdd"] / "i2t" / "images"
    assert link.allowed_file(str(before)) == before.resolve()
    assert link.allowed_file(str(after)) == after.resolve()
    other = env["hdd"] / "secret.txt"   # beside the i2t folder, not in it
    other.write_text("x")
    assert link.allowed_file(str(other)) is None
