"""Items left from the 2026-11-08 review.  Failure risks: a comparison stop that waits for the image being drawn (or,
once it interrupts it, shows the stop as a failure and restarts ComfyUI), the training cache store mixing up images
whose names start alike, the image-generation page sending every result ever made on each update, start.bat lines
that cmd misreads, and Camie Tagger used with the WD tagger's threshold."""
from __future__ import annotations

import json
import sys
import time
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
    return tmp_path


class Stopped(Exception):
    pass


def test_a_comparison_stop_interrupts_the_image_and_counts_as_a_stop(env, monkeypatch):
    from lorastudio import evaluate as ev
    from lorastudio import server
    stopped = {"flag": False}

    def run_prompt(_g):
        # the stop arrives while the image is drawn: ComfyUI reports the interrupted prompt as an error
        server.runner.stop(SimpleNamespace(id="p1"), 1)
        raise RuntimeError("ComfyUI 実行エラー: ")
    cancels = []
    monkeypatch.setattr(ev.comfy, "cancel_current", lambda: cancels.append(1))
    monkeypatch.setattr(ev, "graph_for", lambda *a: {})
    for name, fn in {"ensure_started": lambda: None, "run_prompt": run_prompt, "log_size": lambda: 0,
                     "park": lambda *a: None}.items():
        monkeypatch.setattr(ev.comfy, name, fn)
    job = SimpleNamespace(project=SimpleNamespace(id="p1"), run_id=1, progress={"phase": "evaluating"})
    monkeypatch.setattr(server.runner, "current", job)

    def progress(*_a):
        if getattr(job, "eval_cancel", False):
            stopped["flag"] = True
            raise Stopped()
    model = {"id": "m", "variant": "illustrious", "family": "sdxl", "files": {}}
    spec = {"checkpoints": ["final"], "strengths": [1.0], "seeds": [1], "include_base": False, "prompts": ["a"],
            "width": 64, "height": 64}
    with pytest.raises(Stopped):
        ev._run_eval_images(SimpleNamespace(id="p1", q=lambda *a: []), 1, 1, progress, {"config": {}}, {}, spec, model,
                            "sdxl", env, {"ckpt": "a.safetensors"}, {"final": "l.safetensors"}, None)
    assert cancels == [1] and stopped["flag"]


def test_the_cache_store_keeps_images_with_similar_names_apart(env):
    from lorastudio import traincache
    proj, run = env / "proj", env / "proj" / "runs" / "1"
    items = [{"file": "img_1.png", "sha": "a" * 40, "caption": "x"}, {"file": "img_10.png", "sha": "b" * 40, "caption": "y"}]
    snap = {"engine": "musubi-tuner", "model": {"family": "qwen", "files": {}}, "values": {}, "items": items}
    cache = traincache.run_cache_dir(run, snap)
    cache.mkdir(parents=True)
    for stem in ("img_1", "img_10"):
        (cache / f"{stem}_0512x0512_qi.safetensors").write_text(stem)
        (cache / f"{stem}_te.safetensors").write_text(stem + "te")
    (cache / "img_1_0512x0512_qi.safetensors.part").write_text("half")
    assert traincache.harvest(proj, run, snap) == 4
    run2 = env / "proj" / "runs" / "2"
    assert traincache.prime(proj, run2, snap) == 4
    got = {f.name: f.read_text() for f in traincache.run_cache_dir(run2, snap).iterdir()}
    assert got == {"img_1_0512x0512_qi.safetensors": "img_1", "img_1_te.safetensors": "img_1te",
                   "img_10_0512x0512_qi.safetensors": "img_10", "img_10_te.safetensors": "img_10te"}
    assert traincache.complete(run2, snap) == {"latents", "te"}
    assert traincache.prime(proj, run2, snap) == 0   # already in place


def test_the_image_page_gets_the_newest_results_and_the_running_ones(env):
    from fastapi.testclient import TestClient
    from lorastudio import imagegen, server
    items = [{"id": f"e{i}", "created": time.time() - i, "status": "done", "message": "", "settings": {},
              "images": [], "warnings": []} for i in range(40)]
    items[35]["status"] = "queued"
    imagegen.HISTORY.parent.mkdir(parents=True, exist_ok=True)
    imagegen.HISTORY.write_text(json.dumps(items), encoding="utf-8")
    c = TestClient(server.app)
    r = c.get("/api/imagegen").json()
    assert [e["id"] for e in r["history"]] == [f"e{i}" for i in range(30)] + ["e35"] and r["history_more"] == 9
    r = c.get("/api/imagegen?limit=60").json()
    assert len(r["history"]) == 40 and r["history_more"] == 0
    assert len(imagegen.history()) == 40   # nothing is dropped from the file


def test_start_bat_has_no_japanese_where_cmd_reads():
    b = (ROOT.parent / "start.bat").read_bytes()
    head, _, tail = b.partition(b"#MESSAGES#")
    assert tail and all(c < 128 for c in head)
    msgs = tail.decode("utf-8").split("\r\n")
    assert "setup.bat を実行します" in msgs[1] and "起動できません" in msgs[2]


def _i2t():
    sys.path.insert(0, str(ROOT.parent / "engines" / "i2t"))
    import i2t
    return i2t


@pytest.mark.parametrize("camie,general,want", [
    (True, 0.35, {"long_hair"}),                     # the WD default: Camie's own 0.49
    (True, 0.30, {"long_hair", "smile", "angry"}),   # set by the user: kept (no WD per-tag thresholds)
    (False, 0.35, {"long_hair", "smile", "angry"}),  # WD: the default and its per-tag thresholds (angry 0.20)
])
def test_camie_uses_its_own_threshold_unless_changed(camie, general, want):
    import numpy as np
    i2t = _i2t()
    t = object.__new__(i2t.WD14Tagger)
    t.camie, t.rating_session = camie, None
    t.names, t.categories = ["long_hair", "smile", "angry", "general"], [0, 0, 0, 9]
    t._probs = lambda _im: np.array([0.6, 0.4, 0.31, 0.9], dtype=np.float32)
    got = {n for n, _p in t.predict(None, general, 0.85).general}
    assert got == want
