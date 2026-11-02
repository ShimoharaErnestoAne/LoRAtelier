"""i2t page: the server shows / analyses only files under its own folders.  A mistake here would let the
page read any file on the PC through /api/i2t/file."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def link(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "data"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config, i2tlink
    config._settings = None
    return i2tlink


def test_uploaded_and_generated_images_only(link, tmp_path):
    p = link.save_upload("../../evil name.png", b"\x89PNG data")
    assert p.parent == link.IMAGES_DIR and ".." not in p.name
    assert link.save_upload("a.png", b"\x89PNG data") .read_bytes() == b"\x89PNG data"
    assert link.allowed_file(str(p)) == p.resolve()
    outside = tmp_path / "secret.txt"
    outside.write_text("x")
    assert link.allowed_file(str(outside)) is None
    assert link.allowed_file(str(link.IMAGES_DIR / ".." / ".." / ".." / "secret.txt")) is None
    gen = link.config.DATA_DIR / "imagegen" / "images" / "g.png"
    gen.parent.mkdir(parents=True)
    gen.write_bytes(b"x")
    assert link.allowed_file(str(gen)) == gen.resolve()
