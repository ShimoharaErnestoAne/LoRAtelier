"""Faster captions for natural-language models.  Failure risks: a caption mode overwriting a caption the user
edited, a queued VLM caption overwriting the instant caption after the mode changed, a batch taking another
project's images, and a failed batch leaving images "running" forever."""
from __future__ import annotations

import itertools
import queue
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

AI = {"tags": {"general": {"1girl": 0.9, "solo": 0.9, "red hair": 0.9, "smile": 0.8, "school uniform": 0.8,
                           "running": 0.7, "white background": 0.9}, "character": {}}}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "データ 空白"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import analysis, config, models, store
    config._settings = None
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    ck = tmp_path / "base.safetensors"
    ck.write_bytes(b"0")
    model = models.registry.add("自然文", "sdxl_base", {"ckpt": str(ck)})
    pm = store.ProjectManager()
    an = analysis.Analyzer.__new__(analysis.Analyzer)   # no dispatch thread: the test drives the queue
    an.projects, an.q, an._seq, an._queued, an._qlock, an.current = pm, queue.PriorityQueue(), itertools.count(), set(), threading.Lock(), None
    an._forced = set()
    return {"pm": pm, "an": an, "model": model, "analysis": analysis, "tmp": tmp_path}


def _project(env, outfit="include", name="キャラ"):
    p = env["pm"].create(name, "character", env["model"]["id"], outfit)
    return p


def _image(env, p, n, gen=None):
    f = env["tmp"] / f"{p.id}_{n}.png"
    f.write_bytes(f"img{p.id}{n}".encode())
    iid, _ = p.add_image_file(f, f.name, f"{p.id}{n:04d}".ljust(40, "0"))
    p.set_fields(iid, fast_status="done", fast={"ok": True, "colorfulness": 0.2}, ai_status="done", ai=AI)
    if gen:
        p.set_fields(iid, gen=gen)
    return iid


SHOT = {"framing": "full_body", "angle": "side", "expr": "laugh", "bg": "simple", "pose": "running"}


def test_tag_mode_makes_every_caption_at_once_without_bound_features(env):
    p = _project(env)
    p.set_meta(caption_mode="tags")
    iid = _image(env, p, 1)
    env["an"].ensure_caption(p, iid)
    im = p.image(iid)
    assert im["caption_status"] == "done" and env["an"].q.empty()
    assert im["caption"].startswith(p.meta("trigger") + ",") and "smile" in im["caption"]
    assert "red hair" not in im["caption"] and "school uniform" not in im["caption"]   # bound to the trigger


def test_gen_mode_uses_the_shot_for_generated_material_only(env):
    from lorastudio import generate
    p = _project(env, outfit="change")
    p.set_meta(caption_mode="gen")
    shot = dict(SHOT, framing=generate._B) if hasattr(generate, "_B") else SHOT
    g = _image(env, p, 1, gen={"shot": shot})
    other = _image(env, p, 2)
    env["an"].ensure_caption(p, g)
    env["an"].ensure_caption(p, other)
    cap = p.image(g)["caption"]
    assert p.image(g)["caption_status"] == "done" and "running" in cap and "side profile" in cap
    assert "wearing school uniform" in cap and "red hair" not in cap
    assert p.image(other)["caption_status"] == "pending" and ("caption", p.id, other) in env["an"]._queued


def test_mode_change_keeps_edited_captions_and_queued_vlm_work_is_skipped(env, monkeypatch):
    p = _project(env)
    a, b = _image(env, p, 1), _image(env, p, 2)
    env["an"].ensure_caption(p, a)
    env["an"].ensure_caption(p, b)
    p.edit("手で直す", [(b, "caption", "my own caption"), (b, "caption_edited", 1)])
    p.set_meta(caption_mode="tags")
    called = []
    monkeypatch.setattr(env["analysis"].worker, "available", lambda: True)
    monkeypatch.setattr(env["analysis"].worker, "call", lambda *a, **k: called.append(a) or {"ok": True, "texts": []})
    monkeypatch.setattr(env["analysis"].gpu, "begin_analysis", lambda timeout=None: True)
    monkeypatch.setattr(env["analysis"].gpu, "end_analysis", lambda: None)
    env["an"]._dispatch_one("caption", p.id, [a, b], ("caption", p.id, a))
    assert not called                                         # the VLM is not run for the queued images
    assert p.image(a)["caption_status"] == "done" and p.image(a)["caption"].endswith(".")
    assert p.image(b)["caption"] == "my own caption"


def test_captions_are_batched_per_project_and_a_failed_batch_is_marked(env, monkeypatch):
    an = env["an"]
    p, q = _project(env), _project(env, name="別")
    ids = [_image(env, p, n) for n in range(10)]
    other = _image(env, q, 1)
    for i in ids:
        an.ensure_caption(p, i)
    an.ensure_caption(q, other)
    prio, _s, first = an.q.get()
    an._queued.discard(first)
    more = an._more_captions(p.id)
    assert len(more) == env["analysis"].CAPTION_BATCH - 1 and first[2] not in more
    assert ("caption", q.id, other) in an._queued                 # another project's image stays queued
    monkeypatch.setattr(env["analysis"].worker, "available", lambda: True)
    monkeypatch.setattr(env["analysis"].worker, "call", lambda *a, **k: {"ok": False, "error": "x"})
    monkeypatch.setattr(env["analysis"].gpu, "begin_analysis", lambda timeout=None: True)
    monkeypatch.setattr(env["analysis"].gpu, "end_analysis", lambda: None)
    batch = [first[2]] + more
    an._dispatch_one("caption", p.id, batch, first)
    assert all(p.image(i)["caption_status"] == "failed" for i in batch)
    monkeypatch.setattr(env["analysis"].worker, "call",
                        lambda op, **k: {"ok": True, "texts": [f"a person number {n}" for n in range(len(k["paths"]))]})
    an._dispatch_one("caption", p.id, batch, first)
    caps = [p.image(i)["caption"] for i in batch]
    assert all(c.startswith(p.meta("trigger")) for c in caps) and len(set(caps)) == len(batch)


def test_anima_writes_tags_by_default_and_sentences_when_chosen(env):
    """Anima reads tags and sentences: tags unless the project picked the VLM; a sentence choice must
    reach the VLM queue instead of silently writing tags again."""
    from lorastudio import captions, models
    ck = env["tmp"] / "anima.safetensors"
    ck.write_bytes(b"0")
    anima = models.registry.add("Anima", "anima", {"ckpt": str(ck), "te": str(ck), "vae": str(ck)})
    p = env["pm"].create("アニマ", "character", anima["id"], "include")
    iid = _image(env, p, 1)
    assert captions.project_style(p.get_meta()) == "tags"
    env["an"].ensure_caption(p, iid)
    im = p.image(iid)
    assert im["caption_status"] == "done" and im["caption"].startswith(p.meta("trigger") + ", ") and "." not in im["caption"]
    p.set_meta(caption_mode="ai")
    assert captions.project_style(p.get_meta()) == "natural"
    env["an"].ensure_caption(p, iid)
    assert p.image(iid)["caption_status"] == "pending" and ("caption", p.id, iid) in env["an"]._queued
