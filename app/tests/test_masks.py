"""Masked training: which kinds and models offer it, masks made in the analysis queue (the segmenters replaced by
fakes), the frozen training images carrying the mask in their alpha channel with the floor and the crop, the
dataset config asking sd-scripts for alpha masks, separate latent caches, and the settings / per-image API."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ANALYSIS_PY = ROOT.parent / "envs" / "analysis" / "Scripts" / "python.exe"

VALUES = {"max_train_steps": 100, "save_every_n_steps": 50, "seed": 1, "resolution": 1024, "learning_rate": 1e-4,
          "network_dim": 4, "network_alpha": 2, "train_batch_size": 1, "optimizer_type": "AdamW8bit",
          "lr_scheduler": "constant", "caption_dropout_rate": 0.0}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "データ 空白"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config, masks, models, store, training, traincache
    config._settings = None

    def add(name, variant):
        files = {}
        for key, _label, required in models.FAMILY_FILES[models.VARIANTS[variant]["family"]]:
            if required or key == "ckpt":
                f = tmp_path / "モデル" / f"{name}_{key}.safetensors"
                f.parent.mkdir(exist_ok=True)
                f.write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{}      ")
                files[key] = str(f)
        return models.registry.add(name, variant, files)
    return {"tmp": tmp_path, "config": config, "masks": masks, "store": store, "training": training,
            "traincache": traincache, "add": add}


def _project(env, kind="character", n=2, **kw):
    pm = env["store"].ProjectManager(env["config"].PROJECTS_DIR)
    model = env["add"]("sdxl", "illustrious")
    p = pm.create("テスト", kind, model["id"], **kw)
    for i in range(n):
        f = env["tmp"] / "素材" / f"{i}.png"
        f.parent.mkdir(exist_ok=True)
        Image.fromarray((np.random.default_rng(i).random((80, 120, 3)) * 255).astype("uint8")).save(f)
        iid, _ = p.add_image_file(f, f.name, hashlib.sha256(f.read_bytes()).hexdigest())
        p.set_auto_caption(iid, f"{p.meta('trigger')}, solo", "k")
        p.set_fields(iid, ai={"domain": {"label": "illustration"}}, ai_status="done")
    return p


def _done_mask(env, p, iid, mode="person", value=255):
    """A finished mask as the analysis queue leaves it: left half learnt."""
    im = p.image(iid)
    f = env["masks"].image_mask_file(im, mode)   # a person mask is made for the image's crop
    f.parent.mkdir(parents=True, exist_ok=True)
    m = np.zeros((80, 120), np.uint8)
    m[:, :60] = value
    Image.fromarray(m, "L").save(f)
    p.set_fields(iid, mask={mode: {"status": "done", "coverage": 0.5, "people": 1, "version": env["masks"].MASK_VERSION,
                                   "crop": env["masks"].crop_of(im, mode)}})


def test_kinds_and_models_that_offer_masked_training(env):
    M = env["masks"]
    assert M.mode_for({"kind": "character"}) == "person"
    assert M.mode_for({"kind": "concept", "concept_target": "outfit"}) == "person"
    assert M.mode_for({"kind": "concept", "concept_target": "place"}) == "background"
    assert M.mode_for({"kind": "concept", "concept_target": "object"}) is None
    assert M.mode_for({"kind": "style"}) is None and M.mode_for({"kind": "enhance"}) is None
    on = {"kind": "character", "mask": {"on": True, "floor": 0.2}}
    assert M.active(on, "sdxl") == "person" and M.active(on, "anima") == "person"
    assert M.active(on, "zimage") is None and "SDXL" in M.unsupported_reason(on, "zimage")   # musubi-tuner: no masked loss
    assert M.active({"kind": "character"}, "sdxl") is None   # off unless turned on


def test_run_images_carry_the_mask_in_alpha_with_floor_and_crop(env):
    tr, M = env["training"], env["masks"]
    p = _project(env, n=3)
    a, b, c = [im["id"] for im in p.images()]
    p.edit("crop", [(a, "crop", [40, 0, 120, 80])])
    _done_mask(env, p, a)
    _done_mask(env, p, b)
    p.edit("off", [(b, "mask_off", 1)])          # the user's "この画像はマスクを使わない"
    p.set_meta(mask={"on": True, "floor": 0.2})
    rid = tr.create_run(p, p.images(), VALUES, [], None)
    run = p.run(rid)
    snap = run["config"]
    assert snap["mask"] == {"mode": "person", "floor": 0.2, "n_masked": 1}
    items = {it["id"]: it for it in snap["items"]}
    assert items[a].get("mask") and not items[b].get("mask") and not items[c].get("mask")   # c: no mask yet
    rd = Path(run["run_dir"])
    tr.materialize(rd, snap)
    im = Image.open(rd / "dataset" / items[a]["file"])
    assert im.mode == "RGBA" and im.size == (80, 80)
    alpha = np.asarray(im.getchannel("A"))
    assert alpha[:, :20].min() == 255 and alpha[:, 30:].max() == round(0.2 * 255)   # crop starts at x=40 of the mask
    assert Image.open(rd / "dataset" / items[c]["file"]).mode == "RGB"
    cmds = tr.build_commands(snap, rd, None)
    ds = (rd / "dataset.toml").read_text(encoding="utf-8")
    assert "alpha_mask = true" in ds and cmds
    # latents of a masked run are kept apart from unmasked ones (the .npz holds the mask)
    k = env["traincache"]._keys
    assert k(items[a], True)[0] != k({**items[a], "mask": None}, False)[0]
    assert k(items[c], True)[0] != k(items[c], False)[0] and k(items[c], True)[1] == k(items[c], False)[1]


def test_without_the_setting_runs_are_unchanged(env):
    tr = env["training"]
    p = _project(env, n=1)
    _done_mask(env, p, p.images()[0]["id"])
    rid = tr.create_run(p, p.images(), VALUES, [], None)
    snap = p.run(rid)["config"]
    assert snap["mask"] is None and "mask" not in snap["items"][0]
    tr.build_commands(snap, Path(p.run(rid)["run_dir"]), None)
    assert "alpha_mask" not in (Path(p.run(rid)["run_dir"]) / "dataset.toml").read_text(encoding="utf-8")


def test_analysis_queue_makes_masks_and_api(env, monkeypatch):
    from fastapi.testclient import TestClient
    from lorastudio import analysis, server
    M = env["masks"]
    pm = server.projects
    model = env["add"]("sdxl", "illustrious")
    p = pm.create("キャラ", "character", model["id"])
    f = env["tmp"] / "a.png"
    Image.new("RGB", (64, 64), (200, 100, 100)).save(f)
    iid, _ = p.add_image_file(f, f.name, hashlib.sha256(f.read_bytes()).hexdigest())
    p.set_fields(iid, ai={"domain": {"label": "photo"}}, ai_status="done", fast_status="done", width=64, height=64)
    monkeypatch.setattr(server, "_used_images", lambda pr: pr.images())   # the material checks are not under test here
    queued = []
    monkeypatch.setattr(server.analyzer, "_enqueue", lambda prio, item: queued.append(item))
    calls = []

    def fake_call(op, timeout=900, **kw):
        calls.append((op, kw))
        Image.new("L", (64, 64), 255).save(kw["out"])
        return {"ok": True, "result": {"status": "done", "people": 2, "method": "maskrcnn", "coverage": 0.4}}
    monkeypatch.setattr(analysis.worker, "call", fake_call)
    cl = TestClient(server.app)
    assert cl.patch(f"/api/projects/{p.id}", json={"mask": {"on": True, "floor": 2}}).status_code == 400
    assert cl.patch(f"/api/projects/{p.id}", json={"mask": {"on": True, "floor": 0.1}}).status_code == 200
    server._queue_masks(p)   # what the PATCH starts in the background
    assert ("mask:person", p.id, iid) in queued and M.state(p.image(iid), "person")["status"] == "pending"
    plan = cl.get(f"/api/projects/{p.id}/train/plan").json()
    assert plan["mask"]["on"] and any("マスクを作っています" in w for w in plan["preflight"]["warnings"])
    server.analyzer._do_mask(p, iid, "person")
    assert calls[0][1]["mode"] == "person" and calls[0][1]["domain"] == "photo"
    d = cl.get(f"/api/projects/{p.id}/images/{iid}").json()["mask"]
    assert d["status"] == "done" and d["people"] == 2 and not d["off"]
    assert cl.get(f"/api/projects/{p.id}/images/{iid}/mask_preview").status_code == 200
    r = cl.post(f"/api/projects/{p.id}/edit", json={"label": "off", "changes": [{"id": iid, "field": "mask_off", "value": 1}]})
    assert r.status_code == 200 and cl.get(f"/api/projects/{p.id}/images/{iid}").json()["mask"]["off"]
    plan = cl.get(f"/api/projects/{p.id}/train/plan").json()
    assert plan["mask"]["summary"]["off"] == 1
    # a style project is not offered masks; neither is a model family whose engine has no masked loss
    s = pm.create("画風", "style", model["id"])
    assert cl.get(f"/api/projects/{s.id}/train/plan").json()["mask"]["mode"] is None


@pytest.mark.skipif(not ANALYSIS_PY.exists(), reason="analysis environment not installed")
def test_worker_mask_logic_with_fake_segmenters(tmp_path):
    """make_mask in the analysis environment, with the segmenters replaced (no model download): the largest person
    is kept in person mode, background mode inverts all people, no person gives status none."""
    script = tmp_path / "check.py"
    script.write_text(r'''
import json, sys
import numpy as np
from PIL import Image
sys.path.insert(0, sys.argv[1])
import analysis_worker as w
import imgutils.detect, imgutils.segment
src = sys.argv[2] + "/src.png"
Image.new("RGB", (200, 100), (128, 128, 128)).save(src)
seg = np.zeros((100, 200), np.float32); seg[20:80, 10:50] = 1; seg[10:90, 100:180] = 1
imgutils.segment.get_isnetis_mask = lambda img: seg
imgutils.detect.detect_person = lambda img: [((10, 20, 50, 80), "person", 0.9), ((100, 10, 180, 90), "person", 0.9)]
out = {}
for mode in ("person", "background"):
    r = w.make_mask(src, mode, "illustration", sys.argv[2] + f"/{mode}.png")
    m = np.asarray(Image.open(sys.argv[2] + f"/{mode}.png"))
    out[mode] = [r["status"], r["people"], int(m[50, 30]), int(m[50, 140]), int(m[5, 70])]
p1 = np.zeros((100, 200), np.float32); p1[0:50, 0:50] = 1
p2 = np.zeros((100, 200), np.float32); p2[:, 100:200] = 1
w.M.person_masks_photo = lambda img: [p1, p2]
r = w.make_mask(src, "person", "photo", sys.argv[2] + "/photo.png")
m = np.asarray(Image.open(sys.argv[2] + "/photo.png"))
out["photo"] = [r["status"], r["method"], int(m[25, 25]), int(m[50, 150])]
w.M.person_masks_photo = lambda img: []
out["none"] = w.make_mask(src, "person", "photo", sys.argv[2] + "/none.png")["status"]
print(json.dumps(out))
''', encoding="utf-8")
    r = subprocess.run([str(ANALYSIS_PY), str(script), str(ROOT / "worker"), str(tmp_path)], capture_output=True, text=True,
                       encoding="utf-8", timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["person"][:2] == ["done", 2] and out["person"][2] == 0 and out["person"][3] == 255   # only the larger one
    assert out["background"][2] == 0 and out["background"][3] == 0 and out["background"][4] == 255   # people out
    assert out["photo"] == ["done", "maskrcnn", 0, 255]
    assert out["none"] == "none"
