"""似ている組 on the material page (2026-11-23): the image list carries the groups of images the checks call the same
file or very similar.  Risks: a picture similar to two others split over two groups (keeping "one of each" would
then keep two), a group of one shown, and the list failing for a project without any similar images.  Made-up
images in a temp folder only."""
from __future__ import annotations

import sys
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
    from lorastudio import config
    config._settings = None
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    from lorastudio import server
    return server, tmp_path


def test_groups_join_through_a_shared_image_and_skip_singles(env):
    s, _ = env
    ctx = {"dup_of": {7: 3}, "near": {1: {2}, 2: {1, 5}, 5: {2}, 9: set()}}
    assert s._similar_groups(ctx) == [[1, 2, 5], [3, 7]]   # 1-2 and 2-5 are one group; 9 alone is no group
    assert s._similar_groups({"dup_of": {}, "near": {}}) == []


def test_the_image_list_carries_the_groups(env):
    s, tmp = env
    p = s.projects.create("似ている", "character", None)
    f = tmp / "a.png"
    Image.new("RGB", (640, 640), "red").save(f)
    a, _ = p.add_image_file(f, "a.png", "aa" * 20)
    b, _ = p.add_image_file(f, "b.png", "aa" * 20)   # the same file twice: a duplicate
    g = tmp / "c.png"
    Image.new("RGB", (640, 640), "blue").save(g)
    c, _ = p.add_image_file(g, "c.png", "cc" * 20)
    out = s.list_images(p.id)
    assert out["similar_groups"] == [sorted([a, b])]
    assert c not in out["similar_groups"][0]
