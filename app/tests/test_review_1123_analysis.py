"""2026-11-23 review fixes in the material analysis, the captions, the checks and the i2t link (fake data, no GPU, no
real models; the workers are replaced).  Failure risks: a caption that fails without a word about why, an empty
answer of the VLM saved (and cached for good) as a caption that is only the trigger, captions that failed because
the GPU was unusable never tried again, 16-bit grey images read as white, a huge image refused by the worker after
the mechanical check passed it, an older reading of an image kept after the check changed, a trigger with a
backslash breaking the change of the trigger, excluded images deciding who "the character" is, a low-confidence
photo / illustration guess told as a fact, the resolution check ignoring the project's training resolution, a stray
line of a worker ending its reader thread, the GPU kept when an item fails after waiting for it, an i2t job paused
while it waited for the GPU running on, and a finished i2t analysis reported as failed because a side note could not
be saved."""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import io
import itertools
import json
import queue
import sys
import threading
import warnings
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")   # never the real models folder
    return tmp_path


@pytest.fixture()
def an(env, monkeypatch):
    """An analyzer without its dispatch thread (the test drives the queue), a tag model and a sentence model."""
    from lorastudio import analysis, models, store
    ck = env / "base.safetensors"
    ck.write_bytes(b"0")
    tag_model = models.registry.add("タグ", "illustrious", {"ckpt": str(ck)})
    nl_model = models.registry.add("文章", "sdxl_base", {"ckpt": str(ck)})
    pm = store.ProjectManager()
    a = analysis.Analyzer.__new__(analysis.Analyzer)
    a.projects, a.q, a._seq, a._queued, a._qlock, a.current = pm, queue.PriorityQueue(), itertools.count(), set(), threading.Lock(), None
    a._forced = set()
    monkeypatch.setattr(analysis.worker, "available", lambda: True)
    monkeypatch.setattr(analysis.bus, "publish", lambda *x, **k: None)
    return SimpleNamespace(a=a, pm=pm, model=tag_model, nl=nl_model, analysis=analysis, tmp=env)


AI = {"tags": {"general": {"1girl": 0.9, "smile": 0.9}, "character": {}}, "domain": {"label": "illustration"}}


def _add(an, p, n, **fields):
    f = an.tmp / f"{p.id}_{n}.png"
    f.write_bytes(f"img{n}".encode())
    iid, _ = p.add_image_file(f, f.name, hashlib.sha256(f"img{n}".encode()).hexdigest())   # the caches key on its start
    fast = {"ok": True, "colorfulness": 0.2, "width": 1024, "height": 1024, "sharpness_p90": 50,
            "phash": format(n * 0x0123456789ABCDEF % 2**64, "016x")}
    p.set_fields(iid, fast_status="done", fast=fast, **fields)
    return iid


def _queued(a):
    return sorted(a._queued)


# ------------------------------------------------------------------ captions
def test_a_trigger_with_a_backslash_can_replace_the_old_one(env):
    from lorastudio import captions
    assert captions.replace_trigger("chr_a, smile", "chr_a", r"new\d") == r"new\d, smile"
    assert captions.replace_trigger("chr_a standing", "chr_a", r"x\1y") == r"x\1y standing"
    assert captions.replace_trigger("earrings, rin", "rin", "kai") == "earrings, kai"       # still whole words only


def test_an_empty_answer_of_the_vlm_is_a_failure_not_a_caption(an, monkeypatch):
    analysis, a = an.analysis, an.a
    p = an.pm.create("空", "character", an.nl["id"])
    trig = p.meta("trigger")
    ok, empty, unread = (_add(an, p, n, ai_status="done", ai=AI, caption_status="pending") for n in (1, 2, 3))
    monkeypatch.setattr(analysis.worker, "call",
                        lambda op, **kw: {"ok": True, "texts": [f"{trig} waves at the camera.", "  \n ", None]})
    a._do_captions(p, [ok, empty, unread])
    good, bad, none = p.image(ok), p.image(empty), p.image(unread)
    assert good["caption_status"] == "done" and good["caption"] == f"{trig} waves at the camera."
    # the answer with nothing in it: not a caption of only the trigger, and the reason is kept
    assert bad["caption_status"] == "failed" and bad["caption"] == "" and "中身のない文" in bad["ai_error"]
    assert none["caption_status"] == "failed" and "読み込めませんでした" in none["ai_error"]
    # ...and nothing was cached for it: asked again, the VLM is called again
    key = analysis._policy(p.get_meta())
    assert analysis.cache_get(analysis._cap_ckey(key, good)) and analysis.cache_get(analysis._cap_ckey(key, bad)) is None
    a.ensure_caption(p, empty, force=True)
    assert p.image(empty)["caption_status"] == "pending" and ("caption", p.id, empty) in a._queued


def test_a_blank_caption_cached_by_an_older_version_is_written_again(an):
    analysis, a = an.analysis, an.a
    p = an.pm.create("古いキャッシュ", "character", an.nl["id"])
    trig = p.meta("trigger")
    iid = _add(an, p, 1, ai_status="done", ai=AI, caption_status="pending")
    key = analysis._policy(p.get_meta())
    analysis.cache_put(analysis._cap_ckey(key, p.image(iid)), {"text": f"{trig}, "})     # what the old code cached
    a.ensure_caption(p, iid)
    assert p.image(iid)["caption"] == "" and ("caption", p.id, iid) in a._queued          # the VLM writes it again
    analysis.cache_put(analysis._cap_ckey(key, p.image(iid)), {"text": f"{trig}, sitting on a bench"})
    a.ensure_caption(p, iid, force=True)
    assert p.image(iid)["caption"] == f"{trig}, sitting on a bench"                       # a real one is still used


def test_why_a_caption_failed_is_kept_and_shown_with_the_image(an, monkeypatch):
    from lorastudio import judge
    analysis, a = an.analysis, an.a
    p = an.pm.create("理由", "character", an.nl["id"])
    trig = p.meta("trigger")
    iid = _add(an, p, 1, ai_status="done", ai=AI, caption_status="pending")
    monkeypatch.setattr(analysis.worker, "call", lambda op, **kw: {"ok": False, "error": "GPU メモリ不足: OutOfMemoryError"})
    a._do_captions(p, [iid])
    im = p.image(iid)
    assert im["caption_status"] == "failed" and im["ai_status"] == "done" and im["ai_error"] == "GPU メモリ不足: OutOfMemoryError"
    res = judge.check_image(im, judge.build_context([im], p.get_meta()))
    note = [c for c in res["checks"] if c["key"] == "caption"]
    assert note and "GPU メモリ不足" in note[0]["message"] and note[0]["status"] == "info"
    made = judge.check_image({**im, "caption_status": "done"}, judge.build_context([im], p.get_meta()))
    assert (res["verdict"], res["use"]) == (made["verdict"], made["use"])       # a note: the verdict of the image stays
    # the batch as a whole failing (the worker stopped restarting) keeps its message too
    other = _add(an, p, 2, ai_status="done", ai=AI, caption_status="running")
    a._mark_failed("caption", p.id, [other], "解析ワーカーが繰り返し停止しました")
    assert p.image(other)["ai_error"] == "解析ワーカーが繰り返し停止しました"
    # made after all: the reason goes
    monkeypatch.setattr(analysis.worker, "call", lambda op, **kw: {"ok": True, "texts": [f"{trig} smiles."]})
    a.ensure_caption(p, iid, force=True)
    a._do_captions(p, [iid])
    im = p.image(iid)
    assert im["caption_status"] == "done" and im["ai_error"] is None
    assert not [c for c in judge.check_image(im, judge.build_context([im], p.get_meta()))["checks"] if c["key"] == "caption"]
    # a caption edited by hand is the user's: no note about the automatic one
    p.edit("手で直す", [(other, "caption", "written by hand"), (other, "caption_edited", 1)])
    im = p.image(other)
    assert not [c for c in judge.check_image(im, judge.build_context([im], p.get_meta()))["checks"] if c["key"] == "caption"]


def test_captions_that_failed_without_a_gpu_are_tried_again_a_few_times(an, monkeypatch):
    analysis, a = an.analysis, an.a
    p = an.pm.create("GPU", "character", an.nl["id"])
    iid = _add(an, p, 1, ai_status="done", ai=AI, caption_status="pending")
    timers = []

    class FakeTimer:
        def __init__(self, seconds, fn):
            self.seconds, self.fn, self.daemon = seconds, fn, False

        def start(self):
            timers.append(self)
    monkeypatch.setattr(analysis.threading, "Timer", FakeTimer)
    calls = []

    def call(op, **kw):
        calls.append(op)
        return {"ok": False, "error": "GPU が使えないため、説明文の生成を止めました。", "reason": "gpu"}
    monkeypatch.setattr(analysis.worker, "call", call)
    for attempt in range(analysis.GPU_CAPTION_RETRIES):
        a._do_captions(p, [iid])
        im = p.image(iid)
        assert im["caption_status"] == "failed" and "自動でもう一度試します" in im["ai_error"]
        assert len(timers) == attempt + 1 and timers[-1].seconds > analysis.GPU_RETRY_S   # after the worker's own retry
        timers[-1].fn()                                                                    # the time has come
        assert p.image(iid)["caption_status"] == "pending" and ("caption", p.id, iid) in a._queued
        a._queued.clear()
    a._do_captions(p, [iid])                       # still no GPU: not for ever
    im = p.image(iid)
    assert len(timers) == analysis.GPU_CAPTION_RETRIES and im["caption_status"] == "failed"
    assert "止めました" in im["ai_error"] and "説明文を作り直す" in im["ai_error"]
    assert len(calls) == analysis.GPU_CAPTION_RETRIES + 1
    # another failure (not the GPU) is not queued again by itself
    other = _add(an, p, 2, ai_status="done", ai=AI, caption_status="pending")
    monkeypatch.setattr(analysis.worker, "call", lambda op, **kw: {"ok": False, "error": "解析がタイムアウトしました"})
    a._do_captions(p, [other])
    assert len(timers) == analysis.GPU_CAPTION_RETRIES and p.image(other)["ai_error"] == "解析がタイムアウトしました"


def test_the_worker_names_the_missing_gpu_as_the_reason(env, monkeypatch):
    w = _analysis_worker()
    out = []
    monkeypatch.setattr(w, "emit", out.append)
    monkeypatch.setattr(w.M, "_torch", lambda: None)
    w.M.gpu_error = "RuntimeError: CUDA unknown error"       # CUDA failed to start in this process
    monkeypatch.setattr(w, "load_image", lambda p, c=None: Image.new("RGB", (8, 8)))
    monkeypatch.setattr(w.sys, "stdin", io.StringIO(json.dumps(
        {"op": "captions", "rid": 3, "paths": ["a.png"], "crops": [None], "instruction": "x"}) + "\n"))
    w.main()
    assert out[0]["event"] == "ready" and out[1] == {"rid": 3, "ok": False, "reason": "gpu",
                                                     "error": "GPU が使えないため、説明文の生成を止めました。"}


# ------------------------------------------------------------------ 16-bit images, huge images, older readings
def _analysis_worker():
    spec = importlib.util.spec_from_file_location("analysis_worker_1123", ROOT / "worker" / "analysis_worker.py")
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    return w


def _grey16(path: Path, fmt_mode="I;16"):
    rng = np.random.default_rng(1)
    a = (rng.random((300, 400)) * 20000 + 5000).astype(np.uint16)       # dark to middle grey, far from white
    if fmt_mode == "I":
        a = a.astype(np.int32)
    elif fmt_mode == "F":
        a = (a / 65535).astype(np.float32)
    Image.fromarray(a).save(path)
    return path


@pytest.mark.parametrize("name, mode", [("scan.png", "I;16"), ("scan.tif", "I;16"), ("scan32.tif", "I"), ("scanf.tif", "F")])
def test_a_16_bit_grey_image_is_not_read_as_white(env, name, mode):
    from lorastudio import fastcheck
    f = _grey16(env / name, mode)
    with Image.open(f) as im:
        assert im.mode == mode                       # what Pillow calls such a file
    facts = fastcheck.analyze(f, env / "t.webp")
    assert facts["ok"] and facts["mode"] == mode
    assert 0.18 < facts["brightness"] < 0.28 and facts["clip_bright"] < 0.01      # was 0.998 / 99.6% white
    with Image.open(env / "t.webp") as t:
        assert np.asarray(t.convert("L")).mean() < 100                            # the thumbnail too
    # the AI worker reads the same picture
    img = _analysis_worker().load_image(str(f))
    assert img.mode == "RGB" and 45 < np.asarray(img).mean() < 72


def test_an_8_bit_image_is_read_as_before(env):
    from lorastudio import fastcheck
    rng = np.random.default_rng(2)
    a = (rng.random((200, 300, 3)) * 255).astype(np.uint8)
    f = env / "plain.png"
    Image.fromarray(a).save(f)
    with Image.open(f) as im:
        assert fastcheck.to_8bit(im) is im
    assert np.array_equal(np.asarray(_analysis_worker().load_image(str(f))), a)


def test_an_image_over_the_size_limit_is_refused_with_its_reason(env, monkeypatch):
    from lorastudio import fastcheck
    f = env / "big.png"
    Image.new("RGB", (40, 30), (9, 9, 9)).save(f)
    monkeypatch.setattr(fastcheck, "MAX_PIXELS", 1000)
    before = list(warnings.filters)
    facts = fastcheck.analyze(f, env / "t.webp")
    assert not facts["ok"] and "大きすぎる" in facts["error"] and "40×30" in facts["error"]
    assert warnings.filters == before                # the warning filters of the process are left alone (threads)
    monkeypatch.setattr(fastcheck, "MAX_PIXELS", 200_000_000)
    assert fastcheck.analyze(f, env / "t.webp")["ok"]
    # an image the mechanical check lets through is not refused by the AI worker's (smaller) Pillow default
    w = _analysis_worker()
    assert w.Image.MAX_IMAGE_PIXELS == 200_000_000 == fastcheck.Image.MAX_IMAGE_PIXELS


def test_only_the_images_an_older_check_read_wrongly_are_checked_again(an, monkeypatch):
    from lorastudio import fastcheck
    analysis, a = an.analysis, an.a
    assert fastcheck.outdated({"ok": True, "version": 2, "mode": "I;16"})
    assert not fastcheck.outdated({"ok": True, "version": 2, "mode": "RGB"})               # read the same way: kept
    assert not fastcheck.outdated({"ok": True, "version": fastcheck.FASTCHECK_VERSION, "mode": "I;16"})
    assert not fastcheck.outdated({"ok": False, "version": 2, "mode": "I;16"}) and not fastcheck.outdated(None)
    p = an.pm.create("16bit", "character", an.model["id"])
    old = _add(an, p, 1, ai_status="done", ai=AI)
    plain = _add(an, p, 2, ai_status="done", ai=AI)
    p.set_fields(old, fast={"ok": True, "version": 2, "mode": "I;16", "colorfulness": 0.0})
    p.set_fields(plain, fast={"ok": True, "version": 2, "mode": "RGB", "colorfulness": 0.2})
    sent = []
    monkeypatch.setattr(a, "submit_fast", lambda project, image_id, redo_ai=False: sent.append((image_id, redo_ai)))
    assert a.refresh_outdated(p) == 1 and sent == [(old, True)]


def test_an_image_read_anew_is_analysed_and_captioned_again(an, monkeypatch):
    analysis, a = an.analysis, an.a
    p = an.pm.create("読み直し", "character", an.nl["id"])
    trig = p.meta("trigger")
    f = _grey16(an.tmp / "scan.png")
    iid, _ = p.add_image_file(f, f.name, "7" * 40)
    key = analysis.ai_key("7" * 40, None)
    p.set_fields(iid, fast_status="done", fast={"ok": True, "version": 2, "mode": "I;16"}, ai=AI, ai_key=key,
                 ai_status="done", caption_status="done")
    analysis.cache_put(key, AI)                                                    # what the white reading gave
    ckey = analysis._cap_ckey(analysis._policy(p.get_meta()), p.image(iid))
    analysis.cache_put(ckey, {"text": f"{trig}, a blank white image"})
    p.set_auto_caption(iid, f"{trig}, a blank white image", analysis._policy(p.get_meta()))
    a._fast(p, iid, True)                                                          # what refresh_outdated queues
    im = p.image(iid)
    assert im["fast"]["version"] == 3 and im["fast"]["brightness"] < 0.3 and im["ai_status"] == "pending"
    assert _queued(a) == [("analyze", p.id, iid)]                                  # not answered from the cache
    new_ai = {**AI, "tags": {"general": {"1girl": 0.9, "monochrome": 0.9}, "character": {}}}
    monkeypatch.setattr(analysis.worker, "call", lambda op, **kw: {"ok": True, "result": new_ai})
    a._queued.clear()
    a._do_analyze(p, iid)
    im = p.image(iid)
    assert im["ai"] == new_ai and analysis.cache_get(ckey) is None                 # the old caption is not reused
    assert im["caption_status"] == "pending" and ("caption", p.id, iid) in a._queued


# ------------------------------------------------------------------ checks
def _img(iid, **kw):
    fast = {"ok": True, "width": 1024, "height": 1024, "phash": format(iid * 0x0123456789ABCDEF % 2**64, "016x"),
            "sharpness_p90": 50, **kw.pop("fast", {})}
    return {"id": iid, "sha": f"{iid:040d}", "crop": None, "fast": fast, "fast_status": "done", "ai_status": "done",
            "ai": {"tags": {"general": {"1girl": 0.9}, "character": {}}, "domain": {"label": "illustration", "confidence": "高"},
                   "persons": [{"score": 0.9, "bbox": [100, 50, 700, 1000]}], "heads": [{"score": 0.9}],
                   "subject": {"bbox": [100, 50, 700, 1000], "area_ratio": 0.55, "eff_res": 600}}, **kw}


def _notes(imgs, meta):
    from lorastudio import judge
    ctx = judge.build_context(imgs, meta)
    res = {im["id"]: judge.check_image(im, ctx) for im in imgs}
    return [n["text"] for n in judge.dataset_report(imgs, res, meta)["notes"]]


def test_the_photo_and_illustration_note_counts_only_what_it_is_sure_of(env):
    imgs = [_img(n) for n in range(1, 11)]
    for im in imgs[:3]:
        im["ai"] = {**im["ai"], "domain": {"label": "photo", "confidence": "中"}}
    mixed = [t for t in _notes(imgs, {"kind": "style"}) if "写真とイラスト" in t]
    assert len(mixed) == 1 and "写真 3 枚・イラスト 7 枚" in mixed[0] and "可能性があります" in mixed[0]
    assert "{" not in mixed[0] and "混在しています" not in mixed[0]                 # no Python dict, no flat claim
    # the same three, but the analysis is not sure they are photographs: not counted, so nothing is claimed
    for im in imgs[:3]:
        im["ai"] = {**im["ai"], "domain": {"label": "photo", "confidence": "低"}}
    assert not [t for t in _notes(imgs, {"kind": "style"}) if "写真とイラスト" in t]
    # sure of two, unsure of one more: the unsure one is told apart
    for im in imgs[:2]:
        im["ai"] = {**im["ai"], "domain": {"label": "photo", "confidence": "高"}}
    mixed = [t for t in _notes(imgs, {"kind": "style"}) if "写真とイラスト" in t]
    assert len(mixed) == 1 and "写真 2 枚・イラスト 7 枚" in mixed[0] and "自信のない 1 枚" in mixed[0]


def test_the_resolution_check_follows_the_projects_training_resolution(env):
    from lorastudio import judge
    small = _img(1, fast={"width": 400, "height": 600})

    def res_check(meta):
        r = judge.check_image(small, judge.build_context([small], meta))
        return next((c for c in r["checks"] if c["key"] == "res"), None), r["verdict"]
    c, verdict = res_check({"kind": "style"})                                      # default 1024: too small, as before
    assert c["status"] == "bad" and "1024px" in c["message"] and verdict == "exclude"
    c, verdict = res_check({"kind": "style", "train_overrides": {"resolution": 768}})
    assert c["status"] == "warn" and "768px" in c["message"] and verdict == "review"
    c, verdict = res_check({"kind": "style", "train_overrides": {"resolution": 512}})
    assert c is None and verdict == "ok"                                           # 400 px is fine for 512
    # a higher resolution names itself but does not make the limits stricter (images in use would drop out)
    mid = _img(2, fast={"width": 800, "height": 900})
    r = judge.check_image(mid, judge.build_context([mid], {"kind": "style", "train_overrides": {"resolution": 1536}}))
    assert not [c for c in r["checks"] if c["key"] == "res"]
    c, _ = res_check({"kind": "style", "train_overrides": {"resolution": 1536}})
    assert c["status"] == "bad" and "1536px" in c["message"]
    assert judge.train_resolution({"train_overrides": {"resolution": "abc"}}) == 1024


# ------------------------------------------------------------------ identity
def _ccip_project(an, monkeypatch, n_a, n_b):
    """n_a images of the character and n_b of another one; the fake CCIP difference is 0 inside a group, 1 across."""
    analysis, a = an.analysis, an.a
    p = an.pm.create("同一性", "character", an.model["id"])
    ids = [_add(an, p, n, ai_status="done", ai_key=f"k{n}", ai={"ccip": [0.0 if n <= n_a else 1.0], "tags": {"general": {}, "character": {}}})
           for n in range(1, n_a + n_b + 1)]
    monkeypatch.setattr(analysis.gpu, "begin_analysis", lambda timeout=None: True)
    monkeypatch.setattr(analysis.gpu, "end_analysis", lambda: None)
    calls = []

    def call(op, feats=None, **kw):
        calls.append(op)
        return {"ok": True, "matrix": [[abs(x[0] - y[0]) for y in feats] for x in feats]}
    monkeypatch.setattr(analysis.worker, "call", call)
    return p, ids, calls


def test_excluded_images_do_not_decide_who_the_character_is(an, monkeypatch):
    a = an.a
    p, ids, calls = _ccip_project(an, monkeypatch, n_a=2, n_b=3)
    mine, others = ids[:2], ids[2:]
    a.compute_identity(p)
    assert p.meta("auto_reference")[0] in others                 # the majority: the other character got in three times
    p.edit("別キャラを除外", [(i, "decision", "exclude") for i in others])
    a.compute_identity(p)                                        # the exclusion changed: judged again
    assert len(calls) == 2 and p.meta("auto_reference")[0] in mine
    m = p.meta("identity")["map"]
    assert all(m[str(i)]["diff"] == 0 for i in mine) and all(m[str(i)]["diff"] == 1 for i in others)
    a.compute_identity(p)                                        # nothing changed: not computed again
    assert len(calls) == 2


def test_with_everything_excluded_the_reference_is_chosen_as_before(an, monkeypatch):
    a = an.a
    p, ids, calls = _ccip_project(an, monkeypatch, n_a=1, n_b=2)
    p.edit("全部除外", [(i, "decision", "exclude") for i in ids])
    a.compute_identity(p)
    assert p.meta("auto_reference")[0] in ids[1:]


# ------------------------------------------------------------------ worker connections
def test_a_stray_line_does_not_end_the_reader_of_the_analysis_worker(env, monkeypatch):
    from lorastudio import analysis
    monkeypatch.setattr(analysis.bus, "publish", lambda *a, **k: None)
    c = analysis.WorkerClient()
    fut = Future()
    c._pending[7] = fut
    proc = SimpleNamespace(stdout=[b"0\n", b"[]\n", b'"text"\n', b"null\n", b"not json\n", b'{"rid": 7, "ok": true}\n'])
    c.proc = proc
    c._reader(proc)
    assert fut.result(timeout=0) == {"rid": 7, "ok": True}       # the answer after the stray lines still arrives


def _client(monkeypatch):
    from lorastudio import i2tlink
    monkeypatch.setattr(i2tlink.bus, "publish", lambda *a, **k: None)
    return i2tlink, i2tlink.I2tClient()


def test_a_stray_line_does_not_end_the_reader_of_the_i2t_worker(env, monkeypatch):
    i2tlink, c = _client(monkeypatch)
    fut = Future()
    c._pending[4] = fut
    c.job = {"rid": 9, "op": "translate", "message": "", "fraction": None}
    proc = SimpleNamespace(stdout=[b"3\n", b"[1, 2]\n", b'{"event": "progress", "rid": 9, "message": "half", "fraction": 0.5}\n',
                                   b'{"rid": 4, "ok": true, "value": 1}\n', b'{"rid": 9, "ok": true, "value": {}}\n'])
    c.proc = proc
    c._reader(proc)
    assert fut.result(timeout=0)["value"] == 1 and c.job is None and c.last_done["ok"] and c.last_done["op"] == "translate"


def test_a_job_cancelled_between_two_messages_does_not_end_the_reader(env, monkeypatch):
    i2tlink, c = _client(monkeypatch)
    job = {"rid": 9, "op": "analyze", "message": "", "fraction": None}

    class Lines:
        """The job ends (a cancel on another thread) while the reader handles the worker's lines."""
        def __iter__(self):
            c.job = job
            yield b'{"event": "progress", "rid": 9, "message": "a"}\n'
            c.job = None
            yield b'{"event": "progress", "rid": 9, "message": "b"}\n'
            yield b'{"rid": 9, "ok": false}\n'
            yield b'{"rid": 5, "ok": true, "value": "still read"}\n'
    fut = Future()
    c._pending[5] = fut
    proc = SimpleNamespace(stdout=Lines())
    c.proc = proc
    c._reader(proc)
    assert fut.result(timeout=0)["value"] == "still read" and job["message"] == "a"


def test_the_gpu_is_given_back_when_an_item_fails_after_waiting_for_it(an, monkeypatch):
    analysis, a = an.analysis, an.a
    p = an.pm.create("待ち", "character", an.model["id"])
    iid = _add(an, p, 1, ai_status="pending")
    held = []

    def begin(timeout=None):
        if timeout == 0:
            return False                      # training holds the GPU
        held.append(1)                        # ...and now the analysis has it
        return True
    monkeypatch.setattr(analysis.gpu, "begin_analysis", begin)
    monkeypatch.setattr(analysis.gpu, "end_analysis", lambda: held.pop())
    item = ("analyze", p.id, iid)
    a._taken = [item]
    opened = []

    def project(op, pid, image_id, it):
        opened.append(1)
        if len(opened) == 2:                  # opened again after the wait: the drive is gone
            raise OSError("disk I/O error")
        return p
    monkeypatch.setattr(a, "_project", project)
    with pytest.raises(OSError):
        a._dispatch_one("analyze", p.id, iid, item)
    assert held == [] and a.current is None   # training and generation do not wait for an analysis that is over


def test_an_i2t_job_paused_while_it_waits_for_the_gpu_stays_paused(env, monkeypatch):
    i2tlink, c = _client(monkeypatch)
    hold = threading.Event()
    monkeypatch.setattr(i2tlink.gpu, "begin_analysis", lambda timeout=None: hold.wait(5))
    monkeypatch.setattr(i2tlink.gpu, "end_analysis", lambda: None)
    sent, done = [], threading.Event()

    def send(obj):
        sent.append(obj)
        if obj.get("op") == "pause":
            done.set()
    monkeypatch.setattr(c, "_send", send)
    c.start_job("analyze", path="a.png", cfg={"low_vram": False})
    c.control("pause")                        # pressed while training still holds the GPU (no worker runs yet)
    assert c.job["paused"] and not sent
    hold.set()                                # the GPU is free: the job goes to the worker...
    assert done.wait(5)
    # ...and the pause after it: the worker lifts a pause when a job starts
    assert [o.get("op") for o in sent] == ["analyze", "pause"] and c.job["paused"]


def test_an_i2t_job_not_paused_is_sent_alone(env, monkeypatch):
    i2tlink, c = _client(monkeypatch)
    monkeypatch.setattr(i2tlink.gpu, "begin_analysis", lambda timeout=None: True)
    monkeypatch.setattr(i2tlink.gpu, "end_analysis", lambda: None)
    sent = []
    monkeypatch.setattr(c, "_send", sent.append)
    c.job ={"rid": 1, "op": "analyze", "waiting_gpu": True}
    c._send_after_gpu(1, "analyze", {"path": "a.png"})
    assert [o.get("op") for o in sent] == ["analyze"]


# ------------------------------------------------------------------ i2t worker
def _worker(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["w", str(tmp_path), str(tmp_path / "d"), str(tmp_path / "m")])
    spec = importlib.util.spec_from_file_location("i2t_worker_1123", ROOT / "worker" / "i2t_worker.py")
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    (tmp_path / "d").mkdir(exist_ok=True)
    return w


@dataclasses.dataclass
class _Result:
    prompt: str = "1girl, smile"


def test_a_finished_i2t_analysis_is_not_failed_by_the_time_estimate_it_could_not_save(env, monkeypatch):
    w = _worker(monkeypatch, env)

    class Cancelled(Exception):
        pass

    def save_config(cfg):
        raise PermissionError("config.json is being read by the page")
    w.i2t = SimpleNamespace(load_config=lambda: {"vlm_seconds": {}}, save_config=save_config, OperationCancelled=Cancelled,
                            describe_error=str, upright=lambda im: im, history_add=lambda *a, **k: None)
    w.engine = SimpleNamespace(cfg=None, progress=None, fraction=None, last_vlm=("qwen", 12.5), _vlm=None,
                               run=lambda path, refs, controls: _Result())
    out = []
    monkeypatch.setattr(w, "emit", out.append)
    w._analyze(1, {"path": str(env / "none.png")})
    assert out[-1]["ok"] is True and out[-1]["result"] == {"prompt": "1girl, smile"}       # was: ok False
    assert w.engine.last_vlm is None


def test_a_translation_that_could_not_be_saved_is_not_reported_as_done(env, monkeypatch):
    w = _worker(monkeypatch, env)

    class Cancelled(Exception):
        pass
    vlm = SimpleNamespace(cancel_event=None)
    w.i2t = SimpleNamespace(load_config=lambda: {"ollama_host": "h", "translate_model": "m"}, DEFAULT_CONFIG={"translate_model": "m"},
                            VLM_TIMEOUT=60, OllamaVLM=lambda *a: vlm, OperationCancelled=Cancelled,
                            translate_chips_ja=lambda v, words, keep_alive=None, checkpoint=None: {"smile": "笑顔"},
                            load_translations=lambda: {}, TRANSLATIONS_PATH=env / "missing dir" / "translations.json")
    monkeypatch.setattr(w, "_unload_used", lambda cfg, *vlms: [])
    monkeypatch.setattr(w, "emit", lambda obj: None)
    w._job.update(rid=1, cancel=threading.Event(), gate=SimpleNamespace(wait=lambda *a: None), vlm=None)
    with pytest.raises(ValueError, match="保存できません"):          # the folder is gone: nothing was kept
        w._translate(1, {"words": ["smile"]}, w._job["cancel"])
