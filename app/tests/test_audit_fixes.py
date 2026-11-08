"""Fixes from the 2026-10-26 audit.  Failure risks: a crop change leaving the caption of the uncropped picture,
a failed analysis leaving the caption "being made" forever, masks never made after a restart, one unreadable
image failing a whole caption batch, a trigger change writing every VLM caption again, and a dying worker
failing requests sent to its replacement."""
from __future__ import annotations

import sys
from concurrent.futures import Future
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from test_caption_modes import AI, _image, _project, env  # noqa: F401  (fixture)


def test_crop_change_makes_the_caption_again(env):
    an, analysis = env["an"], env["analysis"]
    p = _project(env)
    p.set_meta(caption_mode="tags")
    iid = _image(env, p, 1)
    im = p.image(iid)
    p.set_fields(iid, ai_key=analysis.ai_key(im["sha"], None))
    an.ensure_caption(p, iid)
    assert "smile" in p.image(iid)["caption"]
    cropped = {"tags": {"general": {"1girl": 0.9, "solo": 0.9, "frown": 0.9}, "character": {}}}
    p.set_fields(iid, crop=[0, 0, 100, 100])
    analysis.cache_put(analysis.ai_key(im["sha"], [0, 0, 100, 100]), cropped)
    an.submit_ai(p, iid)   # the cropped picture was analysed before: cached
    cap = p.image(iid)["caption"]
    assert "frown" in cap and "smile" not in cap


def test_failed_analysis_does_not_leave_the_caption_pending(env, monkeypatch):
    an, analysis = env["an"], env["analysis"]
    p = _project(env)
    iid = _image(env, p, 1)
    p.set_fields(iid, ai_status="pending", ai=None, caption_status="pending")
    monkeypatch.setattr(analysis.worker, "call", lambda *a, **k: {"ok": False, "error": "壊れた画像"})
    an._do_analyze(p, iid)
    im = p.image(iid)
    assert im["ai_status"] == "failed" and im["caption_status"] == "skipped"


def test_one_unreadable_image_fails_only_itself(env, monkeypatch):
    an, analysis = env["an"], env["analysis"]
    p = _project(env)
    ids = [_image(env, p, n) for n in (1, 2, 3)]
    monkeypatch.setattr(analysis.Analyzer, "_caption_plan", staticmethod(lambda meta, img: ("vlm", analysis._policy(meta), None)))
    monkeypatch.setattr(analysis.worker, "call", lambda *a, **k: {"ok": True, "texts": ["a cat", None, "a dog"]})
    an._do_captions(p, ids)
    st = [p.image(i)["caption_status"] for i in ids]
    assert st == ["done", "failed", "done"]


def test_trigger_change_keeps_vlm_captions_with_the_new_word(env, monkeypatch):
    an, analysis = env["an"], env["analysis"]
    from lorastudio import captions, models
    p = env["pm"].create("画風", "style", env["model"]["id"], "include")
    iid = _image(env, p, 1)
    monkeypatch.setitem(models.VARIANTS, "sdxl_base", {**models.VARIANTS["sdxl_base"], "caption": "nl"})
    old = p.get_meta()
    im = p.image(iid)
    k_old = captions.policy_key("style", "include", "nl", old["trigger"], [], "body")
    analysis.cache_put(f"cap:{k_old}:{im['sha'][:24]}:None", {"text": f"{old['trigger']} style. a cat on a sofa"})
    p.set_meta(trigger="newword")
    an.carry_trigger(p, old)
    an.ensure_caption(p, iid)
    assert p.image(iid)["caption"] == "newword style. a cat on a sofa" and an.q.empty()


def test_dying_worker_does_not_fail_requests_of_the_next_one(env):
    analysis = env["analysis"]
    w = analysis.WorkerClient()

    class Proc:
        stdout = iter([])
    old, new = Proc(), Proc()
    f_old, f_new = Future(), Future()
    f_old.proc, f_new.proc = old, new
    w._pending = {1: f_old, 2: f_new}
    w.proc = new
    w._reader(old)
    assert f_old.done() and not f_new.done() and 2 in w._pending
