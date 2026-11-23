"""Project zip import through import_raw (2026-11-23).  Risks: a zip of tens of GB written first to the system's
temporary folder on C: (no room there while the projects are kept on another drive), the received zip left behind,
and a drive without room for the zip and the project together.  Made-up project in a temp folder only."""
from __future__ import annotations

import io
import sys
from pathlib import Path
from types import SimpleNamespace

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
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    from fastapi.testclient import TestClient
    from lorastudio import projectio, server
    return SimpleNamespace(s=server, pio=projectio, c=TestClient(server.app, base_url="http://127.0.0.1"), tmp=tmp_path)


def _zip(env) -> bytes:
    p = env.s.projects.create("取り込み テスト", "character", None)
    img = env.tmp / "a.png"
    Image.new("RGB", (64, 64), "blue").save(img)
    iid, _ = p.add_image_file(img, "a.png", "cd" * 20)
    p.edit("説明文", [(iid, "caption", "chr_x, blue")])
    return b"".join(env.pio.export_stream(p, False))


def test_the_zip_is_received_in_the_projects_folder_and_removed_after(env, monkeypatch):
    data = _zip(env)
    seen = []
    real = env.pio.import_zip

    def watch(pm, f):
        seen.append(Path(f.name))
        return real(pm, f)
    monkeypatch.setattr(env.pio, "import_zip", watch)
    r = env.c.post("/api/projects/import_raw", content=data)
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "取り込み テスト"
    root = env.s.projects.root
    assert seen and seen[0].parent == root / ".importing"          # on the projects' drive, not in C:'s TEMP
    assert not seen[0].exists() and not list((root / ".importing").glob("upload_*"))   # removed afterwards
    q = env.s.projects.get(r.json()["id"])
    assert q.images()[0]["caption"] == "chr_x, blue"


def test_a_broken_zip_is_refused_and_not_left_behind(env):
    r = env.c.post("/api/projects/import_raw", content=b"PK not really a zip" * 100)
    assert r.status_code == 400 and "zip" in r.json()["detail"]
    assert not list((env.s.projects.root / ".importing").glob("upload_*"))


def test_no_room_for_the_zip_and_the_project_is_said_before_receiving(env, monkeypatch):
    data = _zip(env)
    monkeypatch.setattr(env.s.shutil, "disk_usage", lambda p: SimpleNamespace(total=1, used=0, free=1_000_000_000 + len(data)))
    r = env.c.post("/api/projects/import_raw", content=data)
    assert r.status_code == 400 and "空き容量" in r.json()["detail"]
    assert not list((env.s.projects.root / ".importing").glob("upload_*"))
