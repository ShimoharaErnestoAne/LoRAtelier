"""Multi-character LoRA (character project with source "multi"): the member projects' adopted images and
captions go into one run under ids that cannot collide, each character keeps its own trigger, the checks stop
a run whose characters cannot be told apart, and the comparison asks for every character by its trigger."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VALUES = {"max_train_steps": 100, "save_every_n_steps": 50, "seed": 1, "resolution": 1024, "learning_rate": 1e-4,
          "network_dim": 4, "network_alpha": 2, "train_batch_size": 1, "optimizer_type": "AdamW8bit",
          "lr_scheduler": "constant", "caption_dropout_rate": 0.0}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "データ 空白"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config, evaluate, models, multichar, store, training
    config._settings = None
    ckpt = tmp_path / "モデル" / "fake sdxl.safetensors"
    ckpt.parent.mkdir()
    ckpt.write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{}      ")
    model = models.registry.add("fake", "illustrious", {"ckpt": str(ckpt)})
    return {"tmp": tmp_path, "config": config, "store": store, "training": training, "evaluate": evaluate,
            "multichar": multichar, "model": model}


def _char(env, pm, name, n, seed):
    p = pm.create(name, "character", env["model"]["id"])
    for i in range(n):
        path = env["tmp"] / name / f"{i}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((np.random.default_rng(seed + i).random((64, 64, 3)) * 255).astype("uint8")).save(path)
        iid, _ = p.add_image_file(path, path.name, hashlib.sha256(path.read_bytes()).hexdigest())
        p.set_auto_caption(iid, f"{p.meta('trigger')}, solo, pose{i}", "k")
    return p


def test_members_go_into_one_run_with_their_own_triggers(env):
    S, mc, tr = env["store"], env["multichar"], env["training"]
    pm = S.ProjectManager(env["config"].PROJECTS_DIR)
    a, b = _char(env, pm, "アリス", 3, 0), _char(env, pm, "ボブ", 4, 10)
    a.set_meta(reference_ids=[a.images()[0]["id"]])
    multi = pm.create("ふたり", "character", env["model"]["id"], source="multi")
    multi.set_meta(members=[a.id, b.id])
    used, members = mc.gather(pm, multi, lambda mp: mp.images())
    assert len(used) == 7 and len({im["id"] for im in used}) == 7     # same image ids in both projects, no collision
    assert mc.check(members, lambda mid: "tags") == ([], [])
    rid = tr.create_run(multi, used, VALUES, [], None, members)
    run = multi.run(rid)
    tr.materialize(Path(run["run_dir"]), run["config"])
    caps = sorted(f.read_text(encoding="utf-8") for f in (Path(run["run_dir"]) / "dataset").glob("*.txt"))
    assert sum(a.meta("trigger") in c for c in caps) == 3 and sum(b.meta("trigger") in c for c in caps) == 4
    snap = run["config"]
    assert [m["trigger"] for m in snap["members"]] == [a.meta("trigger"), b.meta("trigger")]
    assert snap["members"][0]["refs"] == [mc.MEMBER_ID_STEP + a.images()[0]["id"]]
    # the first comparison calls every character, and each prompt can be traced back to its character
    prompts = env["evaluate"].default_spec(multi, run)["prompts"]
    assert [mc.member_of_prompt(snap["members"], p) for p in prompts] == [0, 0, 1, 1]
    assert len(env["evaluate"].run_prompts(run, ["standard", "pose"])) == 16


def test_checks_stop_runs_whose_characters_cannot_be_told_apart(env):
    mc = env["multichar"]
    m = lambda name, trig, n=5, mid="x": {"pid": name, "name": name, "kind": "character", "multi": False, "trigger": trig,
                                          "outfit_mode": "include", "model_id": mid, "n_used": n, "refs": []}
    errs, _ = mc.check([m("A", "chr_ab12"), m("B", "CHR_AB12")], lambda mid: "tags")
    assert any("呼び出し語が同じ" in e for e in errs)
    errs, _ = mc.check([m("A", "chr_a"), m("B", "chr_b", 2)], lambda mid: "tags")
    assert any("3 枚以上" in e for e in errs)
    errs, _ = mc.check([m("A", "chr_a", mid="tagmodel"), m("B", "chr_b", mid="nlmodel")],
                       lambda mid: "tags" if mid == "tagmodel" else "natural")
    assert any("説明文の形式" in e for e in errs)
    errs, warns = mc.check([m("A", "chr_a", 40), m("B", "chr_b", 5)], lambda mid: "tags")
    assert not errs and any("枚数" in w for w in warns)
    assert mc.check([m("A", "chr_a")], lambda mid: "tags")[0]   # one character is not a multi-character LoRA


def test_api_creates_multi_projects_only_from_character_projects(env):
    from fastapi.testclient import TestClient
    from lorastudio import server
    c = TestClient(server.app)
    a = c.post("/api/projects", json={"kind": "character", "model_id": env["model"]["id"]}).json()
    b = c.post("/api/projects", json={"kind": "character", "model_id": env["model"]["id"]}).json()
    st = c.post("/api/projects", json={"kind": "style"}).json()
    assert c.post("/api/projects", json={"kind": "character", "source": "multi", "members": [a["id"], st["id"]]}).status_code == 400
    r = c.post("/api/projects", json={"kind": "character", "source": "multi", "members": [a["id"], b["id"], a["id"]]})
    assert r.status_code == 200
    mp = r.json()
    assert mp["source"] == "multi" and mp["members"] == [a["id"], b["id"]] and mp["model_id"] == env["model"]["id"]
    assert [m["pid"] for m in mp["member_info"]] == [a["id"], b["id"]]
    up = c.post(f"/api/projects/{mp['id']}/upload", files=[("files", ("x.png", b"\x89PNG", "image/png"))])
    assert up.status_code == 400   # material goes into the member projects
    r = c.patch(f"/api/projects/{mp['id']}", json={"members": [b["id"], mp["id"]]})
    assert r.status_code == 200 and r.json()["members"] == [b["id"]]   # never itself


def test_each_image_is_scored_against_the_character_its_prompt_calls(env):
    ev = env["evaluate"]
    snap = {"items": [{"id": 1_000_001, "file": "a.png"}, {"id": 2_000_001, "file": "b.png"}],
            "members": [{"trigger": "chr_a", "refs": [1_000_001], "id_range": [1_000_000, 2_000_000]},
                        {"trigger": "chr_b", "refs": [], "id_range": [2_000_000, 3_000_000]}]}

    class W:   # CCIP stand-in: matrix rows = generated images, columns = reference images (a, b)
        def __init__(self, matrix):
            self.matrix = matrix

        def call(self, op, **kw):
            if op == "ccip_features":
                return {"ok": True, "feats": kw["paths"]}
            return {"ok": True, "matrix": self.matrix}
    results = [{"prompt": 0, "ckpt": "final", "strength": 1.0}, {"prompt": 1, "ckpt": "final", "strength": 1.0}]
    prompts = ["chr_a, solo", "chr_b, solo"]
    s = ev._member_ccip(snap, Path("ds"), ["g0", "g1"], results, prompts, W([[0.1, 0.5], [0.45, 0.12]]))
    assert s == {"ccip_diff": [0.1, 0.12], "ccip_other": [0.5, 0.45]}
    rec = ev.recommend("character", results, {**s, "copy_sim": [0.5, 0.5]})
    assert not any("混ざっている" in r for r in rec["reasons"])
    # each character's images are about as close to the other character: they are mixing
    s = ev._member_ccip(snap, Path("ds"), ["g0", "g1"], results, prompts, W([[0.3, 0.32], [0.31, 0.3]]))
    rec = ev.recommend("character", results, {**s, "copy_sim": [0.5, 0.5]})
    assert any("混ざっている" in r for r in rec["reasons"])
