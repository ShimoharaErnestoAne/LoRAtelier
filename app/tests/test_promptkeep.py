"""Material generation keeps what the user's prompt says about the character (pose, expression and
camera were followed, but hair, outfit, age, gender and body were changed).  A wrong call here
either lets a wrong-looking image into the training set or throws good candidates away, so the reading of the
prompt, the negative words and the unpick rule are tested on hand-made tagger outputs."""
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
    from lorastudio import config, generate, promptkeep, store
    config._settings = None
    return {"generate": generate, "pk": promptkeep, "store": store, "config": config}


FULL = {"framing": "full", "angle": "front", "expr": "smile", "bg": "simple", "pose": "standing"}
FACE = {"framing": "face", "angle": "front", "expr": "smile", "bg": "simple"}
BACK = {"framing": "upper", "angle": "back_full", "expr": "calm", "bg": "simple"}
PROMPT = "1girl, mature female, (Long_Hair:1.2), blonde hair, blue eyes, large breasts, ponytail, school uniform, pleated skirt, kirisame marisa"


def test_the_prompt_is_read_for_tags_and_sentences(mods):
    pk = mods["pk"]
    r = pk.requested(PROMPT, "tags")
    assert (r["gender"], r["age"], r["hair_length"], r["breasts"]) == ("female", "adult", 3, 3)
    assert r["hair_colour"] == ["blonde"] and r["eye_colour"] == ["blue"]
    assert "kirisame marisa" not in r["tags"] and {"ponytail", "school uniform", "pleated skirt"} <= set(r["tags"])
    n = pk.requested("An old man in his 60s with short white hair and a grey beard, wearing a black suit.", "natural")
    assert (n["gender"], n["age"], n["hair_length"]) == ("male", "adult", 1) and n["hair_colour"] == ["white"]
    assert "gender" not in pk.requested("1girl, 1boy, holding hands", "tags")   # two people: no gender to keep


def test_opposites_go_to_the_negative_of_tag_models_only(mods):
    g = mods["generate"]
    neg = g.compose_negative("lowres", FULL, "tags", "illustrious", PROMPT)
    for t in ("1boy", "male focus", "child", "loli", "shota", "very short hair", "short hair", "flat chest", "small breasts"):
        assert t in neg.split(", "), t
    assert "medium hair" not in neg.split(", ") and "huge breasts" not in neg.split(", ")   # one step away is fine
    assert "child" not in g.compose_negative("", FULL, "tags", "illustrious", "1girl, child, chibi").split(", ")
    assert g.compose_negative("lowres", FULL, "natural", "qwen_image21", PROMPT) == "lowres"
    assert "1boy" not in g.compose_negative("", FULL, "tags", "illustrious")   # no prompt given: as before


def test_the_users_tags_come_before_the_angle_expression_and_pose(mods):
    g = mods["generate"]
    p = [t.strip() for t in g.compose_prompt("score_9, 1girl, long hair, school uniform", FULL, "tags", "pony").split(",")]
    assert p.index("full body") < p.index("1girl") < p.index("long hair") < p.index("school uniform") < p.index("smile") \
        < p.index("standing") < p.index("white background")
    assert p[0] == "score_9"


def test_tagger_reading_against_the_prompt(mods):
    pk = mods["pk"]
    req = pk.requested(PROMPT, "tags")
    good = {"1girl": .99, "mature_female": .6, "long_hair": .9, "blonde_hair": .8, "blue_eyes": .7, "large_breasts": .7,
            "ponytail": .5, "school_uniform": .8, "pleated_skirt": .6}
    assert pk.check(req, good, FULL) == {"conflicts": [], "missing": []}
    bad = {**good, "1girl": .1, "1boy": .9, "loli": .7, "long_hair": .0, "short_hair": .8, "blonde_hair": .0,
           "black_hair": .9, "large_breasts": 0, "flat_chest": .8}
    c = pk.check(req, bad, FULL)["conflicts"]
    assert any("性別" in x for x in c) and any("年齢" in x for x in c) and any("髪の長さ" in x for x in c) \
        and any("髪の色" in x for x in c) and any("胸" in x for x in c)
    near = {**good, "blonde_hair": 0.0, "light_brown_hair": .7}
    assert pk.check(req, near, FULL)["conflicts"] == []   # colours a tagger swaps are not a contradiction
    lost = {k: v for k, v in good.items() if k not in ("ponytail", "school_uniform", "pleated_skirt")}
    assert pk.check(req, lost, FULL)["missing"] == ["ponytail", "school uniform", "pleated skirt"]
    face = pk.check(req, lost, FACE)["missing"]
    assert "pleated skirt" not in face and "school uniform" in face   # the skirt is out of a close-up's frame
    assert "blue eyes" not in pk.check(req, {**good, "blue_eyes": 0}, BACK)["missing"]   # eyes not seen from behind
    assert not pk.unpick({"conflicts": [], "missing": ["ponytail"]})   # one small item: shown only
    assert pk.unpick({"conflicts": [], "missing": ["ponytail", "school uniform"]})
    assert pk.unpick({"conflicts": ["性別（男性に見える）"], "missing": []})


def test_candidates_that_differ_are_unpicked_but_manual_picks_stay(mods, monkeypatch):
    g = mods["generate"]
    p = mods["store"].ProjectManager(mods["config"].PROJECTS_DIR).create("g", "character", None)
    spec = {"positive": "1girl, long hair, blonde hair, school uniform, ponytail", "prompt_style": "tags", "shots": []}
    p.q("INSERT INTO gen_batches(created, status, message, spec) VALUES(0,'done','',?)", (json.dumps(spec),))
    with p.tx() as db:
        for i in range(4):
            db.execute("INSERT INTO gen_images(batch_id, idx, file, seed, prompt, shot, picked, created, pick_manual) "
                       "VALUES(1,?,?,0,'',?,1,0,?)", (i, f"{i}.png", json.dumps(FULL), 1 if i == 3 else 0))
    ok = {"1girl": .99, "long_hair": .9, "blonde_hair": .9, "school_uniform": .9, "ponytail": .6}
    readings = {"0.png": ok, "1.png": {**ok, "1girl": .1, "1boy": .9}, "2.png": {**ok, "ponytail": 0},
                "3.png": {**ok, "long_hair": 0, "very_short_hair": .9}}
    calls = []

    class FakeWorker:
        def call(self, op, **kw):
            calls.append(op)
            assert op == "tag_images"
            return {"ok": True, "tags": [readings[Path(x).name] for x in kw["paths"]]}
    import lorastudio.analysis as an
    monkeypatch.setattr(an, "worker", FakeWorker())
    s = g.check_prompt(p)
    rows = {r["idx"]: r for r in p.q("SELECT * FROM gen_images")}
    assert s == {"n": 4, "mismatch": 3, "unpicked": 1}
    assert [rows[i]["picked"] for i in range(4)] == [1, 0, 1, 1]   # 2: one item only, 3: picked by hand
    assert "性別" in json.loads(rows[1]["prompt_check"])["conflicts"][0]
    assert "1 枚の選択を外しました" in g.prompt_check_message(s)
    assert g.check_prompt(p) == {"n": 0, "mismatch": 0, "unpicked": 0} and calls == ["tag_images"]   # read once
