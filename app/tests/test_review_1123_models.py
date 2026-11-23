"""2026-11-23 review of the model / download / storage code.  Risks covered: every project shown as "updated just
now" after the app was closed normally, GBs of a dropped file left for good after a crash, all settings silently gone
after settings.json broke, a continued download that mixes the old and the new file of the source, an old project
zip that cannot be imported, a finished download that cannot be placed in a models folder on another drive, the
start-up cleaning that removes an import the user has just started, a .lslink left in the models folder, a hash read
again after two copies became one file, an original image deleted because another one has its name and size, a
folder link with a space that asks for "%2520", an engine settings file left pointing at the old drive letter, a
database copy left in data after a failed export, and a tuple list shown as the reason an engine could not be
updated.  Nothing here goes out to the network or reads real data."""
from __future__ import annotations

import errno
import io
import json
import os
import shutil
import sqlite3
import sys
import threading
import time
import urllib.parse
import zipfile
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
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    return tmp_path


class _Body(io.BytesIO):
    status = 200
    headers: dict = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ------------------------------------------------------------------ the project list
def test_opening_the_list_after_a_normal_exit_does_not_make_every_project_updated_now(env):
    from lorastudio import store
    pm = store.ProjectManager()
    old = pm.create("古い方", "character", None)
    new = pm.create("新しい方", "style", None)
    ids = (old.id, new.id)
    for p in (old, new):
        p.close()   # the app closed normally: SQLite removes -wal / -shm
    three_days, one_day = time.time() - 3 * 86400, time.time() - 86400
    for pid, when in zip(ids, (three_days, one_day)):
        db = store.ProjectManager().path_of(pid) / "project.db"
        assert not db.with_name("project.db-wal").exists()
        os.utime(db, (when, when))
    pm = store.ProjectManager()   # the next start
    for _ in range(2):            # the second look finds the -wal the first one made (empty)
        rows = pm.list()
        assert [r["id"] for r in rows] == [ids[1], ids[0]]   # newest first
        assert abs(rows[0]["updated"] - one_day) < 5 and abs(rows[1]["updated"] - three_days) < 5
    pm.get(ids[0]).set_meta(name="直した")   # an edit (written to the WAL) is an update
    rows = pm.list()
    assert rows[0]["id"] == ids[0] and time.time() - rows[0]["updated"] < 60


# ------------------------------------------------------------------ what a crash leaves
def test_a_dropped_file_cut_off_by_a_crash_is_removed_at_start_and_downloads_are_kept(env):
    from lorastudio import remote
    st, tmp = remote.begin_upload("絵 柄.safetensors")
    tmp.write_bytes(b"x" * 3000)   # the app ended while the browser was still sending
    link = tmp.with_name("0123456789ab_big.safetensors.part")   # a link download: adding again continues it
    link.write_bytes(b"y" * 500)
    remote._etag_file(link).write_text('"v1"', encoding="utf-8")
    assert remote.clean_uploads() == 3000
    assert not tmp.exists() and link.read_bytes() == b"y" * 500 and remote._etag_file(link).exists()
    assert remote.clean_uploads() == 0


def test_broken_settings_come_back_from_the_save_before(env):
    from lorastudio import config
    s = config.settings()
    s.set("models", [{"id": "m1", "name": "登録したモデル"}])
    hdd = str(env / "hdd")   # only stored, never opened
    s.set("storage_dirs", {"projects": hdd})
    s.set("ui", {"blur": True})   # the last save: only this one is not in the copy
    bak = json.loads(s.backup_path.read_text(encoding="utf-8"))
    assert bak["storage_dirs"] == {"projects": hdd} and "ui" not in bak
    assert s.recovered is None
    s.path.write_bytes(b'{"models": [{"id": "m1"')   # cut off
    config._settings = None
    s = config.settings()
    assert s.get("models") == [{"id": "m1", "name": "登録したモデル"}] and s.get("storage_dirs") == {"projects": hdd}
    assert s.recovered["restored"] is True and Path(s.recovered["broken"]).exists()
    assert json.loads(s.path.read_text(encoding="utf-8"))["models"][0]["id"] == "m1"   # whole again on disk
    # the copy is unreadable too: defaults as before, and where the broken file went is known
    s.path.write_bytes(b"{")
    s.backup_path.write_bytes(b"\xff\xfe not json")
    config._settings = None
    s = config.settings()
    assert s.get("models") == [] and s.recovered["restored"] is False and Path(s.recovered["broken"]).exists()
    assert not s.path.exists()   # nothing written over until something is saved


# ------------------------------------------------------------------ continued downloads
def _link_download(remote, monkeypatch, answer):
    """One try of a link download of x.gguf (id 'abc'); returns (state, what was placed, the request headers)."""
    monkeypatch.setattr(remote, "probe", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(remote, "RETRY_TRIES", 0)   # one try each: the automatic continuation has its own tests
    placed, asked = [], []
    monkeypatch.setattr(remote, "_place", lambda st, tmp, name, hint: placed.append(tmp.read_bytes()))
    monkeypatch.setattr(remote, "_open", lambda url, h, timeout=60: asked.append(dict(h)) or answer(h))
    st = {"id": "abc", "name": "x.gguf", "state": "running", "done_bytes": 0, "total_bytes": 0, "error": None}
    remote._download(st, "https://huggingface.co/a/b/resolve/main/x.gguf", "x.gguf", {}, None)
    return st, placed, asked


def test_a_link_download_is_not_continued_with_another_file_of_the_source(env, monkeypatch):
    from lorastudio import remote
    part = env / "models" / "_incoming" / "abc_x.gguf.part"

    class Cut(_Body):   # 1000 bytes announced, the connection drops after 500
        headers = {"Content-Length": "1000", "ETag": '"v1"'}
    st, placed, asked = _link_download(remote, monkeypatch, lambda h: Cut(b"a" * 500))
    assert st["state"] == "failed" and "続きから" in st["error"] and not placed
    assert part.read_bytes() == b"a" * 500 and remote._etag_file(part).read_text(encoding="utf-8") == '"v1"'

    # the file at the source was replaced meanwhile: the server answers If-Range with the whole new file
    class New(_Body):
        headers = {"Content-Length": "800", "ETag": '"v2"'}
    st, placed, asked = _link_download(remote, monkeypatch, lambda h: New(b"n" * 800))
    assert asked[0]["Range"] == "bytes=500-" and asked[0]["If-Range"] == '"v1"'
    assert st["error"] is None and placed == [b"n" * 800]   # not 500 old bytes followed by new ones

    # still the same file: the rest only
    part.write_bytes(b"a" * 500)
    remote._etag_file(part).write_text('"v1"', encoding="utf-8")

    class Rest(_Body):
        status = 206
        headers = {"Content-Range": "bytes 500-999/1000", "ETag": '"v1"'}
    st, placed, asked = _link_download(remote, monkeypatch, lambda h: Rest(b"b" * 500))
    assert st["error"] is None and placed == [b"a" * 500 + b"b" * 500]

    # a server that ignores If-Range and continues another file: the .part is dropped and the user is told
    part.write_bytes(b"a" * 500)
    remote._etag_file(part).write_text('"v1"', encoding="utf-8")

    class Other(_Body):
        status = 206
        headers = {"Content-Range": "bytes 500-999/1000", "ETag": '"v9"'}
    st, placed, asked = _link_download(remote, monkeypatch, lambda h: Other(b"z" * 500))
    assert st["state"] == "failed" and "差し替わ" in st["error"] and not placed
    assert not part.exists() and not remote._etag_file(part).exists()


def test_a_catalog_download_is_not_continued_with_another_file_of_the_source(env, monkeypatch):
    from lorastudio import addons, remote
    dest = env / "models" / "diffusion_models" / "big.gguf"
    dest.parent.mkdir(parents=True)
    part = addons._part_file(addons._url("a/b", "big.gguf"), dest)
    part.write_bytes(b"a" * 500)                                   # begun earlier, when the file was "v1"
    remote._etag_file(part).write_text('"v1"', encoding="utf-8")
    asked = []

    class New(_Body):
        headers = {"Content-Length": "800", "ETag": '"v2"'}

    def answer(req, timeout=60):
        asked.append({k.lower(): v for k, v in req.header_items()})
        return New(b"n" * 800)
    monkeypatch.setattr(addons._opener, "open", answer)
    st = {"done_bytes": 0}
    assert addons._download("a/b", "big.gguf", dest, None, st) == dest
    assert asked[0]["range"] == "bytes=500-" and asked[0]["if-range"] == '"v1"'
    assert dest.read_bytes() == b"n" * 800 and st["done_bytes"] == 800
    assert not part.exists() and not remote._etag_file(part).exists()


# ------------------------------------------------------------------ placing a finished download
def test_a_finished_download_reaches_a_models_folder_on_another_drive(env, monkeypatch):
    from lorastudio import remote
    tmp = env / "models" / "_incoming" / "abc_x.safetensors.part"
    final = env / "other drive" / "checkpoints" / "x.safetensors"
    for f in (tmp, final):
        f.parent.mkdir(parents=True)
    tmp.write_bytes(b"model" * 1000)
    real = os.replace

    def replace(src, dst, *a, **k):   # a junction to another drive: no rename out of models/_incoming
        if Path(src).parent.name == "_incoming":
            raise OSError(errno.EXDEV, "システムはファイルを別のディスク ドライブに移動できません。")
        return real(src, dst, *a, **k)
    monkeypatch.setattr(remote.os, "replace", replace)
    remote._move_into(tmp, final)
    assert final.read_bytes() == b"model" * 1000 and not tmp.exists()
    assert [f.name for f in final.parent.iterdir()] == ["x.safetensors"]
    # no room there: said so, nothing half-copied under any name, and the received file is still there
    tmp.write_bytes(b"model" * 1000)
    final.unlink()

    def full(src, dst, n=0):
        dst.write(b"half")
        raise OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(remote.shutil, "copyfileobj", full)
    with pytest.raises(RuntimeError, match="空き容量"):
        remote._move_into(tmp, final)
    assert list(final.parent.iterdir()) == [] and tmp.exists()


# ------------------------------------------------------------------ project zips and the start-up cleaning
def test_a_project_zip_of_an_older_version_is_imported(env):
    from lorastudio import projectio, store
    pm = store.ProjectManager()
    p = pm.create("昔のプロジェクト", "character", None)
    buf = io.BytesIO(b"".join(projectio.export_stream(p, True)))
    # as an older version wrote it: without the tables added later
    db_file = env / "old.db"
    out = io.BytesIO()
    with zipfile.ZipFile(buf) as zin, zipfile.ZipFile(out, "w") as zout:
        for i in zin.infolist():
            data = zin.read(i)
            if i.filename.endswith("project.db"):
                db_file.write_bytes(data)
                db = sqlite3.connect(db_file)
                for t in ("gen_batches", "gen_images", "exports", "evals"):
                    db.execute(f"DROP TABLE IF EXISTS {t}")
                db.commit()
                db.close()
                data = db_file.read_bytes()
            zout.writestr(i, data)
    out.seek(0)
    q = projectio.import_zip(pm, out)
    assert q.id != p.id and q.meta("name") == "昔のプロジェクト"
    assert q.q("SELECT COUNT(*) c FROM gen_batches")[0]["c"] == 0


def test_cleaning_at_start_never_removes_an_import_begun_meanwhile(env, monkeypatch):
    from lorastudio import config, projectio
    root = config.storage_dirs("projects")[0]
    for d in (root / ".moving_abc", root / ".importing" / "20261101-000000_old", projectio.TRASH_DIR / "deleted_project"):
        d.mkdir(parents=True)
        (d / "f.bin").write_bytes(b"x")
    stale = config.DATA_DIR / "export_stale.db.tmp"
    stale.write_bytes(b"x")
    begun = root / ".importing" / "20261123-120000_new"
    real = shutil.rmtree

    def slow_rmtree(path, *a, **k):   # deleting takes a while: the user starts an import during it
        if not begun.exists():
            begun.mkdir(parents=True)
            (begun / "project.db").write_bytes(b"being unpacked")
        return real(path, *a, **k)
    monkeypatch.setattr(projectio.shutil, "rmtree", slow_rmtree)
    projectio.empty_trash()
    assert (begun / "project.db").read_bytes() == b"being unpacked"
    assert not (root / ".moving_abc").exists() and not (root / ".importing" / "20261101-000000_old").exists()
    assert not stale.exists() and list(projectio.TRASH_DIR.iterdir()) == []


def test_a_failed_export_leaves_no_database_copy_in_data(env):
    from lorastudio import config, projectio

    class NoRoom:
        def backup(self, dst):
            raise sqlite3.OperationalError("database or disk is full")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with pytest.raises(sqlite3.OperationalError):
        projectio._db_snapshot(SimpleNamespace(_lock=threading.Lock(), db=NoRoom()), True)
    assert not [f for f in config.DATA_DIR.iterdir() if f.suffix in (".db", ".tmp") or ".db" in f.name]


# ------------------------------------------------------------------ one file for two copies
def test_a_copy_that_cannot_be_replaced_leaves_no_lslink_file(env, monkeypatch):
    from lorastudio import linking
    d = env / "models" / "checkpoints"
    d.mkdir(parents=True)
    master, target = d / "master.safetensors", d / "copy.safetensors"
    master.write_bytes(b"same")
    target.write_bytes(b"same")
    if not linking.links_supported(d):
        pytest.skip("this drive has no hard links")

    def in_use(src, dst, *a, **k):
        raise PermissionError(13, "プロセスはファイルにアクセスできません。別のプロセスが使用中です。")
    monkeypatch.setattr(linking.os, "replace", in_use)
    with pytest.raises(PermissionError):
        linking._replace_with_link(target, master)
    assert sorted(f.name for f in d.iterdir()) == ["copy.safetensors", "master.safetensors"]


def test_the_hash_of_a_copy_is_still_known_after_it_became_a_link(env, monkeypatch):
    from lorastudio import linking, models
    tool, comfy = env / "models" / "checkpoints" / "m.safetensors", env / "comfy" / "m.safetensors"
    for f in (tool, comfy):
        f.parent.mkdir(parents=True)
        f.write_bytes(b"weights" * 500)
    if not linking.links_supported(tool.parent):
        pytest.skip("this drive has no hard links")
    yesterday = time.time() - 86400
    os.utime(comfy, (yesterday, yesterday))   # the two copies were made at different times
    res = linking.link_duplicates([{"status": "same", "tool": str(tool), "comfy": str(comfy), "size": tool.stat().st_size}])
    assert res["linked"] == 1 and not res["errors"] and os.path.samefile(tool, comfy)
    import hashlib
    assert models.hasher.peek(str(tool)) == hashlib.sha256(b"weights" * 500).hexdigest()   # not read in full again


# ------------------------------------------------------------------ moving the data
def test_moving_images_keeps_an_original_whose_name_and_size_another_image_has(env):
    from lorastudio import relocate
    place, target = env / "from", env / "to"
    for d in (place, target):
        d.mkdir()
    (place / "copied.png").write_bytes(b"A" * 100)
    (target / "copied.png").write_bytes(b"A" * 100)   # an earlier move copied it and stopped before removing it
    (place / "other.png").write_bytes(b"B" * 100)
    (target / "other.png").write_bytes(b"C" * 100)    # another image: same name, same size
    relocate.state.setdefault("notes", [])
    relocate._cancel.clear()
    relocate.move_images("imagegen", place, target)
    assert not (place / "copied.png").exists()
    assert (place / "other.png").read_bytes() == b"B" * 100 and (target / "other.png").read_bytes() == b"C" * 100
    assert any("other.png" in n for n in relocate.state["notes"])


def test_paths_written_in_another_case_follow_the_project(env):
    from lorastudio import relocate
    forms = relocate._forms("D:\\Data\\projects\\p1", "E:\\LoRA\\projects\\p1")
    toml = 'train_data_dir = "d:\\\\data\\\\projects\\\\p1\\\\runs\\\\0001\\\\img"\noutput = "d:/data/projects/p1/runs/0001/out"\n'
    out = relocate._rebase_text(toml, forms)
    assert out == 'train_data_dir = "E:\\\\LoRA\\\\projects\\\\p1\\\\runs\\\\0001\\\\img"\noutput = "E:/LoRA/projects/p1/runs/0001/out"\n'
    assert relocate._rebase_text("d:\\data\\projects\\p1", forms) == "E:\\LoRA\\projects\\p1"
    # another project whose name only begins the same stays
    assert relocate._rebase_text('x = "D:\\\\Data\\\\projects\\\\p10\\\\a"', forms) == 'x = "D:\\\\Data\\\\projects\\\\p10\\\\a"'


# ------------------------------------------------------------------ links and engines
def test_a_hugging_face_folder_link_with_a_space_is_asked_for_once_encoded(env, monkeypatch):
    from lorastudio import remote
    asked = []

    def fake_json(url, headers=None):
        asked.append(url)
        return {} if "/tree/" not in url else [{"type": "file", "path": "my folder/絵柄/a.safetensors", "size": 10}]
    monkeypatch.setattr(remote, "_json", fake_json)
    monkeypatch.setattr(remote, "_finish", lambda d, h, tag: d)
    d = remote._resolve_hf(urllib.parse.urlsplit("https://huggingface.co/a/b/tree/main/my%20folder/%E7%B5%B5%E6%9F%84"), None)
    assert asked[1] == "https://huggingface.co/api/models/a/b/tree/main/my%20folder/%E7%B5%B5%E6%9F%84?recursive=1"
    assert d["files"][0]["path"] == "my folder/絵柄/a.safetensors"


def test_an_engine_that_cannot_be_overwritten_says_so_in_words(env, monkeypatch, tmp_path):
    from lorastudio import enginefetch
    commit = "b" * 40

    def download(url, dest, timeout=60):
        with zipfile.ZipFile(dest, "w") as z:
            z.writestr(f"repo-{commit}/node.py", "new")
    monkeypatch.setattr(enginefetch, "_download", download)
    dest = tmp_path / "engines" / "some-node"
    dest.mkdir(parents=True)
    (dest / "node.py").write_text("old", encoding="utf-8")   # an older version is there (and in use)

    def in_use(src, dst, **k):
        raise shutil.Error([(str(Path(src) / "node.py"), str(Path(dst) / "node.py"), "[WinError 32] プロセスはファイルにアクセスできません。")])
    monkeypatch.setattr(enginefetch.shutil, "copytree", in_use)
    with pytest.raises(RuntimeError) as e:
        enginefetch.fetch_archive("https://github.com/someone/repo.git", commit, dest, log=lambda m: None)
    msg = str(e.value)
    assert "some-node" in msg and "WinError 32" in msg and "もう一度" in msg and "[(" not in msg
