"""Tests for the failure risks that matter (spec 17): lost/overwritten edits, run snapshots
changing under a running job, undo, restart persistence, honest verdicts for failed analysis,
odd paths.  They exercise the real store / judge / snapshot code, not re-implementations."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    data = tmp_path / "データ 空白"           # Japanese + space in every path
    monkeypatch.setenv("LORASTUDIO_DATA", str(data))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config, store
    config._settings = None
    return {"data": data, "config": config, "store": store, "tmp": tmp_path}


def _img(path: Path, size=(1200, 900), seed=0):
    rng = np.random.default_rng(seed)
    a = (rng.random((size[1], size[0], 3)) * 255).astype("uint8")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(path)
    return path


def _add(project, path: Path):
    import hashlib
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    iid, _ = project.add_image_file(path, path.name, sha)
    return iid


def test_auto_caption_never_overwrites_manual_edit(env):
    pm = env["store"].ProjectManager(env["config"].PROJECTS_DIR)
    p = pm.create("テスト", "character", None)
    iid = _add(p, _img(env["tmp"] / "画像 1.png"))
    assert p.set_auto_caption(iid, "auto v1", "k1") is True
    p.edit("手動", [(iid, "caption", "手で直した説明"), (iid, "caption_edited", 1)])
    # analysis finishing later (e.g. after trigger change) must not overwrite
    assert p.set_auto_caption(iid, "auto v2", "k2") is False
    im = p.image(iid)
    assert im["caption"] == "手で直した説明"
    assert im["caption_auto"] == "auto v2"          # kept for "reset to auto"


def test_undo_redo_bulk_and_restart_persistence(env):
    S = env["store"]
    pm = S.ProjectManager(env["config"].PROJECTS_DIR)
    p = pm.create("x", "style", None)
    ids = [_add(p, _img(env["tmp"] / f"{i}.png", seed=i)) for i in range(3)]
    p.edit("除外", [(i, "decision", "exclude") for i in ids])
    p.edit("切り抜き", [(ids[0], "crop", [10, 10, 500, 500])])
    assert p.image(ids[0])["crop"] == [10, 10, 500, 500]
    assert p.undo()["label"] == "切り抜き"
    assert p.image(ids[0])["crop"] is None
    assert p.undo()["label"] == "除外"
    assert all(p.image(i)["decision"] is None for i in ids)
    assert p.redo()["label"] == "除外"
    assert all(p.image(i)["decision"] == "exclude" for i in ids)
    # a new edit drops the redo branch
    p.edit("採用", [(ids[1], "decision", "use")])
    assert p.history_state()["redo"] is None
    # restart: reopen from disk
    path = p.path
    p.close()
    p2 = S.Project(path)
    assert p2.image(ids[1])["decision"] == "use"
    assert p2.image(ids[2])["decision"] == "exclude"
    assert p2.history_state()["undo"] == "採用"


def test_failed_or_pending_analysis_is_never_ok(env):
    from lorastudio import judge
    base = {"id": 1, "sha": "a", "fast_status": "done", "crop": None, "decision": None, "ai": None,
            "fast": {"ok": True, "width": 1024, "height": 1024, "phash": "0" * 16, "sharpness_p90": 100.0}}
    ctx = judge.build_context([base], {"kind": "character"})
    pend = judge.check_image({**base, "ai_status": "pending"}, ctx)
    assert pend["verdict"] == "pending"
    failed = judge.check_image({**base, "ai_status": "failed", "ai_error": "OOM"}, ctx)
    assert failed["verdict"] == "hold"
    broken = judge.check_image({**base, "fast": {"ok": False, "error": "破損"}, "ai_status": "skipped"}, ctx)
    assert broken["verdict"] == "unusable" and broken["use"] is False


def test_style_blur_is_not_quality_failure(env):
    from lorastudio import judge
    imgs = []
    for i, sp in enumerate([300.0, 280.0, 5.0]):
        imgs.append({"id": i + 1, "sha": str(i), "fast_status": "done", "crop": None, "decision": None, "ai": None,
                     "ai_status": "done", "fast": {"ok": True, "width": 1024, "height": 1024,
                                                   "phash": format(i * 0x0F0F0F0F0F0F0F0F % (1 << 64), "016x"),
                                                   "sharpness_p90": sp}})
    st = judge.build_context(imgs, {"kind": "style"})
    ch = judge.build_context(imgs, {"kind": "character"})
    assert judge.check_image(imgs[2], st)["verdict"] == "ok"        # painterly softness is info only
    assert judge.check_image(imgs[2], ch)["verdict"] == "review"


def test_exact_duplicate_is_exclusion_candidate_not_deleted(env):
    pm = env["store"].ProjectManager(env["config"].PROJECTS_DIR)
    p = pm.create("dup", "character", None)
    f = _img(env["tmp"] / "a.png")
    a = _add(p, f)
    b = _add(p, f)
    from lorastudio import fastcheck, judge
    for i in (a, b):
        im = p.image(i)
        p.set_fields(i, fast=fastcheck.analyze(p.source_path(im), p.thumbs / f"{i}.webp"), fast_status="done")
    imgs = p.images()
    ctx = judge.build_context(imgs, p.get_meta())
    va, vb = judge.check_image(imgs[0], ctx), judge.check_image(imgs[1], ctx)
    assert va["verdict"] != "exclude" and vb["verdict"] == "exclude"
    assert p.source_path(imgs[1]).exists()                          # original never deleted


def test_fastcheck_corrupt_and_lowres(env, tmp_path):
    from lorastudio import fastcheck
    bad = tmp_path / "壊れた 画像.jpg"
    good = _img(tmp_path / "ok.png")
    bad.write_bytes(good.read_bytes()[:400])
    r = fastcheck.analyze(bad, tmp_path / "t1.webp")
    assert r["ok"] is False and "error" in r
    small = _img(tmp_path / "small.png", size=(300, 300))
    r2 = fastcheck.analyze(small, tmp_path / "t2.webp")
    assert r2["ok"] and r2["width"] == 300
    jpg = tmp_path / "q.jpg"
    Image.open(good).save(jpg, quality=30)
    assert fastcheck.analyze(jpg, tmp_path / "t3.webp")["jpeg_quality"] in range(25, 36)


def test_run_snapshot_is_frozen_against_later_edits(env, monkeypatch):
    S = env["store"]
    cfg = env["config"]
    ckpt = env["tmp"] / "モデル" / "fake sdxl.safetensors"
    ckpt.parent.mkdir()
    ckpt.write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{}      ")
    from lorastudio import models, training
    m = models.registry.add("fake", "illustrious", {"ckpt": str(ckpt)})
    pm = S.ProjectManager(cfg.PROJECTS_DIR)
    p = pm.create("snap", "character", m["id"])
    ids = [_add(p, _img(env["tmp"] / f"s{i}.png", seed=i)) for i in range(3)]
    for i in ids:
        p.set_auto_caption(i, f"{p.meta('trigger')}, solo, pose{i}", "k")
    p.edit("crop", [(ids[0], "crop", [0, 0, 800, 800])])
    imgs = p.images()
    values = {"max_train_steps": 100, "save_every_n_steps": 50, "seed": 1, "resolution": 1024, "learning_rate": 1e-4,
              "network_dim": 4, "network_alpha": 2, "train_batch_size": 1, "optimizer_type": "AdamW8bit",
              "lr_scheduler": "constant", "caption_dropout_rate": 0.0}
    rid = training.create_run(p, imgs, values, [])
    run = p.run(rid)
    # edits made between the click and the (deferred) file export must not leak into the run
    p.edit("edit before export", [(ids[2], "caption", "edited-too-early"), (ids[2], "caption_edited", 1)])
    training.materialize(Path(run["run_dir"]), run["config"])
    ds = Path(run["run_dir"]) / "dataset"
    assert (ds / f"{ids[2]:05d}.txt").read_text(encoding="utf-8") == f"{p.meta('trigger')}, solo, pose{ids[2]}"
    before = {f.name: f.read_bytes() for f in ds.iterdir()}
    snap_before = (Path(run["run_dir"]) / "run.json").read_text(encoding="utf-8")
    assert Image.open(ds / f"{ids[0]:05d}.png").size == (800, 800)   # crop applied in the snapshot
    # user keeps editing for the next run
    p.edit("edit", [(ids[1], "caption", "changed"), (ids[1], "caption_edited", 1), (ids[2], "crop", [0, 0, 300, 300]),
                    (ids[0], "decision", "exclude")])
    after = {f.name: f.read_bytes() for f in ds.iterdir()}
    assert before == after
    assert (Path(run["run_dir"]) / "run.json").read_text(encoding="utf-8") == snap_before
    # engine configs are TOML files with paths containing spaces / Japanese
    cmds = training.build_commands(json.loads(snap_before), Path(run["run_dir"]), None)
    phase, argv, cwd = cmds[-1]
    assert "--config_file" in argv and all(isinstance(a, str) for a in argv)
    import toml
    t = toml.load(argv[argv.index("--config_file") + 1])
    assert t["pretrained_model_name_or_path"] == str(ckpt)
    assert t["save_state"] is True and t["max_train_steps"] == 100


def test_progress_and_error_parsing():
    from lorastudio import training
    line = "steps:  12%|█▏        | 120/1000 [02:10<15:50,  1.08s/it, avr_loss=0.0912]"
    m = training.STEP_RE.search(line)
    assert m and int(m.group(1)) == 120 and int(m.group(2)) == 1000 and float(m.group(7)) == pytest.approx(0.0912)
    e = training.translate_error("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate", 1)
    assert e["code"] == "oom" and e["fixes"]
    assert training.translate_error("OSError: [Errno 28] No space left on device", 1)["code"] == "disk"
    assert training.translate_error("something odd", 3)["code"] == "unknown"


def test_trigger_change_updates_edited_captions_undoably(env):
    S = env["store"]
    pm = S.ProjectManager(env["config"].PROJECTS_DIR)
    p = pm.create("t", "character", None)
    iid = _add(p, _img(env["tmp"] / "t.png"))
    old = p.meta("trigger")
    p.edit("e", [(iid, "caption", f"{old}, smiling, {old}x"), (iid, "caption_edited", 1)])
    from lorastudio import captions
    new = captions.replace_trigger(p.image(iid)["caption"], old, "newname")
    assert new == f"newname, smiling, {old}x"         # whole-word only


def test_tag_caption_policy():
    from lorastudio import captions
    ai = {"tags": {"general": {"1girl": .99, "solo": .98, "red hair": .95, "blue eyes": .9, "witch hat": .9, "dress": .8,
                               "standing": .7, "forest": .6, "watercolor (medium)": .5}, "character": {"pepper (pepper&carrot)": .95}}}
    inc = captions.tag_caption(ai, "character", "include", "chr_ab12")
    chg = captions.tag_caption(ai, "character", "change", "chr_ab12")
    sty = captions.tag_caption(ai, "style", "include", "sty_x")
    assert inc.startswith("chr_ab12") and "red hair" not in inc and "witch hat" not in inc and "forest" in inc
    assert "witch hat" in chg and "red hair" not in chg and "pepper" not in chg
    assert sty.startswith("sty_x style") and "watercolor (medium)" not in sty and "red hair" in sty


def test_style_outlier_flagged_only_in_style_projects(env):
    from lorastudio import judge
    rng = np.random.default_rng(1)
    base = rng.standard_normal(64)
    imgs = []
    for i in range(12):
        e = base + 0.45 * rng.standard_normal(64) if i < 11 else rng.standard_normal(64)   # last one: different look
        e = (e / np.linalg.norm(e)).tolist()
        imgs.append({"id": i + 1, "sha": str(i), "fast_status": "done", "crop": None, "decision": None, "ai_status": "done",
                     "ai": {"embed": e, "tags": {"general": {}, "character": {}}},
                     "fast": {"ok": True, "width": 1024, "height": 1024, "phash": format(i * 0x1111111111111111 % (1 << 64), "016x"),
                              "sharpness_p90": 100.0}})
    st = judge.build_context(imgs, {"kind": "style"})
    assert set(st["style_out"]) == {12}
    assert judge.check_image(imgs[11], st)["verdict"] == "review"
    assert judge.check_image(imgs[0], st)["verdict"] == "ok"
    ch = judge.build_context(imgs, {"kind": "character"})
    assert all(c["key"] != "styleout" for c in judge.check_image(imgs[11], ch)["checks"])


def test_export_version_never_reuses_a_name_in_loras(env, tmp_path):
    from lorastudio import export
    root = tmp_path / "loras"
    (root / "美月凛_v1").mkdir(parents=True)
    (root / "美月凛_v2.safetensors").write_bytes(b"x")       # a file the user put there by hand
    assert export._next_version_dir(root, "美月凛").name == "美月凛_v3"
    assert export._next_version_dir(root, "別名").name == "別名_v1"
