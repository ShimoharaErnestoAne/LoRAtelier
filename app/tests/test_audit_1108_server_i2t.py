"""2026-11-08 review fixes in server.py and i2t.  Failure risks: the keep-cache setting crashing while a comparison
runs, one broken project stopping the startup recovery of all the others, a stale dataset check reused after the
images changed, a huge drop read into memory before it is refused, sideways phone photos analysed sideways, i2t's
settings reset by a half-written config.json, a cancel not reaching JoyCaption, and JoyCaption shown the numbered
person boxes instead of the picture."""
from __future__ import annotations

import sys
import threading
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
    return tmp_path


def test_cache_setting_off_while_a_task_runs(env, monkeypatch):
    from fastapi.testclient import TestClient
    from lorastudio import server, traincache
    p = server.projects.create("比較中", "character", None)
    monkeypatch.setattr(server.runner, "current", SimpleNamespace(project=p, run_id=None, task=lambda: None))
    cleared = []
    monkeypatch.setattr(traincache, "clear_project", lambda d, skip=(): cleared.append((d.name, skip)) or 0)
    r = TestClient(server.app).put("/api/ui", json={"keep_train_cache": False})
    assert r.status_code == 200 and (p.id, ()) in cleared


def test_startup_recovery_goes_on_after_a_broken_project(env, monkeypatch):
    from lorastudio import server
    a = server.projects.create("壊れた", "character", None)
    b = server.projects.create("ふつう", "character", None)
    seen = []

    def recover(p, order, reserved):
        seen.append(p.id)
        if p.id == a.id:
            raise OSError("disk gone")
    monkeypatch.setattr(server, "_recover_project", recover)
    monkeypatch.setattr(server.tasks, "recover", lambda: seen.append("tasks"))
    server.resume_pending_on_startup()
    assert b.id in seen and seen[-1] == "tasks"


def test_dataset_context_is_reused_until_an_image_changes(env, monkeypatch):
    from lorastudio import judge, server
    p = server.projects.create("文脈", "character", None)
    built = []
    real = judge.build_context
    monkeypatch.setattr(judge, "build_context", lambda *a: built.append(1) or real(*a))
    imgs = [{"id": 1, "sha": "a" * 40, "crop": None, "ai_key": None, "fast": {"ok": True, "phash": "0" * 16, "sharpness_p90": 5}},
            {"id": 2, "sha": "b" * 40, "crop": None, "ai_key": None, "fast": {"ok": True, "phash": "f" * 16, "sharpness_p90": 5}}]
    meta = p.get_meta()
    c1 = server._ctx(p, imgs, meta)
    assert server._ctx(p, [dict(i) for i in imgs], meta) is c1 and len(built) == 1
    imgs[1]["sha"] = "a" * 40   # now an exact duplicate: the checks must see it
    c2 = server._ctx(p, imgs, meta)
    assert len(built) == 2 and c2["dup_of"] == {2: 1}


def test_too_large_drop_is_refused_by_its_length(env):
    from fastapi.testclient import TestClient
    from lorastudio import server
    r = TestClient(server.app).post("/api/i2t/upload?name=a.png", content=b"x",
                                    headers={"content-length": str(300 * 2**20)})
    assert r.status_code == 400 and "200MB" in r.json()["detail"]


def _i2t():
    sys.path.insert(0, str(ROOT.parent / "engines" / "i2t"))
    import i2t
    return i2t


def test_i2t_turns_phone_photos_upright(tmp_path):
    i2t = _i2t()
    im = Image.new("RGB", (40, 20), (255, 0, 0))
    exif = im.getexif()
    exif[0x0112] = 6   # rotate 90 CW to view
    f = tmp_path / "縦 写真.jpg"
    im.save(f, exif=exif)
    assert i2t.Engine._open(f).size == (20, 40)
    plain = tmp_path / "plain.png"
    Image.new("RGB", (40, 20)).save(plain)
    assert i2t.Engine._open(plain).size == (40, 20)


def test_i2t_config_save_leaves_no_half_file(tmp_path, monkeypatch):
    i2t = _i2t()
    monkeypatch.setattr(i2t, "CONFIG_PATH", tmp_path / "config.json")
    i2t.save_config({**i2t.DEFAULT_CONFIG, "describer_model": "x"})
    assert i2t.load_config()["describer_model"] == "x" and not list(tmp_path.glob("*.tmp"))


def test_i2t_cancel_reaches_the_describer_and_it_sees_the_clean_image(monkeypatch):
    i2t = _i2t()
    monkeypatch.setattr(i2t.OllamaVLM, "capabilities", lambda self: {"vision"})
    eng = i2t.Engine({**i2t.DEFAULT_CONFIG, "describer_model": i2t.JOYCAPTION_MODEL})
    aborted, seen = [], {}
    started = threading.Event()

    def describe(self, image):
        seen["size"] = image.size
        started.set()
        assert eng._writer is self
        eng.cancel()
        assert aborted == [self.model]
        raise i2t.OllamaError("connection closed")
    monkeypatch.setattr(i2t.OllamaVLM, "describe", describe)
    monkeypatch.setattr(i2t.OllamaVLM, "abort", lambda self: aborted.append(self.model))
    tags = i2t.TagResult("general", 0.9, [("1girl", 0.9)], [], {"1girl": 0.9})
    boxed, clean = Image.new("RGB", (64, 64)), Image.new("RGB", (32, 32))
    with pytest.raises(i2t.OperationCancelled):
        eng._run_vlm(boxed, tags, "anime", [], "sdxl", (64, 64), [], 2, ["2girls"], describe_image=clean)
    assert seen["size"] == (32, 32) and eng._writer is None


def test_i2t_people_hint_kept_with_other_guidance(monkeypatch):
    i2t = _i2t()
    monkeypatch.setattr(i2t.OllamaVLM, "capabilities", lambda self: {"vision"})
    got = {}
    monkeypatch.setattr(i2t.OllamaVLM, "analyze", lambda self, image, tags, medium, negative, g="", refs=None, separate=(): got.update(g=g) or {})
    tags = i2t.TagResult("general", 0.9, [("2girls", 0.9)], [], {"2girls": 0.9})
    img = Image.new("RGB", (32, 32))
    eng = i2t.Engine({**i2t.DEFAULT_CONFIG})
    eng._run_vlm(img, tags, "anime", [], "sdxl", (32, 32), [], 2, ["2girls"], extra_guidance="gender rule")
    assert "gender rule" in got["g"] and "The tagger suggests 2 people" in got["g"]
    eng._run_vlm(img, tags, "anime", [], "sdxl", (32, 32), [], 2, ["2girls"], extra_guidance="split rule", persons_guided=True)
    assert "The tagger suggests" not in got["g"]
