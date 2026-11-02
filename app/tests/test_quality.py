"""LoRA quality aids: which saved version is suggested, and which generated candidates are unpicked because
they do not look like the rest.  A wrong call here steers the user to a weaker LoRA, so the rules are
tested on small hand-made score tables."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def mods(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "data"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config, evaluate, generate, store
    config._settings = None
    return {"evaluate": evaluate, "generate": generate, "store": store, "config": config, "tmp": tmp_path}


def _imgs(cols, n=4):
    return [{"prompt": i, "seed": 1, "ckpt": c, "strength": s} for c, s in cols for i in range(n)]


def test_character_pick_is_the_closest_identity(mods):
    rec = mods["evaluate"].recommend
    cols = [("base", 0.0), (600, 1.0), (1200, 1.0), ("final", 1.0)]
    ccip = [0.40] * 4 + [0.25] * 4 + [0.14] * 4 + [0.16] * 4
    r = rec("character", _imgs(cols), {"ccip_diff": ccip, "copy_sim": [0.5] * 16})
    assert (r["ckpt"], r["strength"], r["weak"]) == ("1200", 1.0, False)
    assert any("同じキャラ" in x for x in r["reasons"])


def test_near_tie_prefers_less_copying_then_earlier(mods):
    rec = mods["evaluate"].recommend
    cols = [("base", 0.0), (1200, 1.0), ("final", 1.0)]
    ccip = [0.40] * 4 + [0.150] * 4 + [0.145] * 4
    # final is marginally closer but copies the training images more: the earlier, cleaner one wins
    r = rec("character", _imgs(cols), {"ccip_diff": ccip, "copy_sim": [0.5] * 4 + [0.60] * 4 + [0.80] * 4})
    assert r["ckpt"] == "1200"
    r2 = rec("character", _imgs(cols), {"ccip_diff": ccip, "copy_sim": [0.5] * 12})
    assert r2["ckpt"] == "1200"   # same copying: the earlier step


def test_weak_effect_is_reported(mods):
    rec = mods["evaluate"].recommend
    cols = [("base", 0.0), ("final", 1.0)]
    r = rec("character", _imgs(cols), {"ccip_diff": [0.30] * 4 + [0.29] * 4})
    assert r["weak"] is True


def test_style_pick_uses_style_similarity(mods):
    rec = mods["evaluate"].recommend
    cols = [("base", 0.0), (800, 0.8), (800, 1.0)]
    r = rec("style", _imgs(cols), {"style_sim": [0.50] * 4 + [0.60] * 4 + [0.70] * 4, "copy_sim": [0.4] * 12})
    assert (r["ckpt"], r["strength"], r["weak"]) == ("800", 1.0, False)


def test_no_scores_no_pick(mods):
    assert mods["evaluate"].recommend("character", _imgs([("final", 1.0)]), {"error": "x"}) is None


def test_consistency_unpicks_outliers_but_not_manual_picks(mods, monkeypatch):
    gen = mods["generate"]
    p = mods["store"].ProjectManager(mods["config"].PROJECTS_DIR).create("g", "character", None)
    bid = p.q("INSERT INTO gen_batches(created, status, message, spec) VALUES(0,'done','', '{}')") or 1
    with p.tx() as db:
        for i in range(5):
            db.execute("INSERT INTO gen_images(batch_id, idx, file, seed, prompt, shot, picked, created, pick_manual) "
                       "VALUES(1,?,?,0,'','{}',1,0,?)", (i, f"{i}.png", 1 if i == 4 else 0))
    # images 0-2 are the same character, 3 and 4 differ; 4 was picked by hand
    feats = {f"{i}.png": [float(i)] for i in range(5)}
    group = {0: "a", 1: "a", 2: "a", 3: "b", 4: "c"}

    class FakeWorker:
        def call(self, op, **kw):
            if op == "ccip_features":
                return {"ok": True, "feats": [feats[Path(x).name] for x in kw["paths"]]}
            fs = [int(f[0]) for f in kw["feats"]]
            return {"ok": True, "matrix": [[0.05 if group[a] == group[b] else 0.30 for b in fs] for a in fs]}
    import lorastudio.analysis as an
    monkeypatch.setattr(an, "worker", FakeWorker())
    s = gen.check_consistency(p)
    rows = {r["idx"]: r for r in p.q("SELECT * FROM gen_images")}
    assert s == {"n": 5, "outliers": 2, "unpicked": 1}
    assert [rows[i]["picked"] for i in range(5)] == [1, 1, 1, 0, 1]
    assert rows[3]["ccip_diff"] > gen.SAME_CHAR and rows[0]["ccip_diff"] <= gen.SAME_CHAR
    assert "1 枚の選択を外しました" in gen.consistency_message(s)
