"""Body-type / body-part LoRA (kind "concept"): the feature words are the only thing left out of the
captions, the material checks look for the feature instead of one identity, and the suggested version is
the one whose images get the feature tag most often.  A mistake in any of these binds the feature to the
caption words instead of the trigger, or points the user at a version that does not show it."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def mods(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "データ"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import captions, config, evaluate, judge, store
    config._settings = None
    return {"captions": captions, "evaluate": evaluate, "judge": judge, "store": store, "config": config}


AI = {"tags": {"general": {"1boy": .99, "solo": .98, "muscular": .95, "muscular male": .9, "abs": .9, "short hair": .9,
                           "black hair": .85, "tank top": .8, "standing": .7, "gym": .6, "pectorals": .3},
               "character": {}}}


def test_only_the_feature_words_leave_the_caption(mods):
    c = mods["captions"]
    cap = c.tag_caption(AI, "concept", "include", "cpt_ab12", c.clean_concept_words("Muscular, abs"))
    tags = c.split_tags(cap)
    assert tags[0] == "cpt_ab12"
    assert not {"muscular", "muscular male", "abs"} & set(tags)
    # everything else about the person stays, so the trigger does not absorb hair or clothes
    assert {"black hair", "short hair", "tank top", "1boy", "gym"} <= set(tags)
    assert c.finalize_nl("A man in a gym.", "concept", "cpt_ab12") == "cpt_ab12, A man in a gym."
    assert "muscular" in c.nl_instruction("concept", "include", "cpt_ab12", ["muscular"])


def test_changing_feature_words_recaptions_only_concept_projects(mods):
    c = mods["captions"]
    assert c.policy_key("concept", "include", "tags", "t", ["abs"]) != c.policy_key("concept", "include", "tags", "t", ["muscular"])
    # existing character/style projects keep their keys (no needless re-captioning after the update)
    assert c.policy_key("character", "include", "tags", "t", ["abs"]) == c.policy_key("character", "include", "tags", "t")


def test_project_keeps_feature_words_and_trigger_prefix(mods):
    pm = mods["store"].ProjectManager(mods["config"].PROJECTS_DIR)
    p = pm.create("筋肉", "concept", None, concept_words="muscular、 abs ,abs")
    m = p.get_meta()
    assert m["kind"] == "concept" and m["trigger"].startswith("cpt_") and m["concept_words"] == ["muscular", "abs"]


def _im(i, tags, chars=None):
    return {"id": i, "sha": str(i), "fast_status": "done", "crop": None, "decision": None, "ai_status": "done",
            "ai": {"tags": {"general": tags, "character": chars or {}}, "persons": [{"bbox": [0, 0, 900, 900], "score": .9}],
                   "subject": {"bbox": [0, 0, 900, 900], "area_ratio": .6, "eff_res": 900}},
            "fast": {"ok": True, "width": 1024, "height": 1024, "phash": format(i * 0x1111111111111111 % (1 << 64), "016x"),
                     "sharpness_p90": 100.0}}


def test_material_checks_look_for_the_feature(mods):
    j = mods["judge"]
    imgs = [_im(i, {"1girl": .9, "muscular": .8, "abs": .7, "red hair": .6}) for i in range(1, 6)]
    imgs.append(_im(6, {"1girl": .9, "red hair": .6}))
    meta = {"kind": "concept", "concept_words": ["muscular"]}
    ctx = j.build_context(imgs, meta)
    res = {im["id"]: j.check_image(im, ctx) for im in imgs}
    assert res[1]["verdict"] == "ok"
    assert any(c["key"] == "concept" for c in res[6]["checks"]) and res[6]["verdict"] == "review"
    assert not any(c["key"] == "ident" for r in res.values() for c in r["checks"])   # no identity check
    rep = j.dataset_report(imgs, res, meta)
    sug = [x["tag"] for x in rep["coverage"]["concept_suggest"]]
    assert "abs" in sug and "muscular" not in sug and "1girl" not in sug
    rep0 = j.dataset_report(imgs, res, {"kind": "concept", "concept_words": []})
    assert "muscular" in [x["tag"] for x in rep0["coverage"]["concept_suggest"]]
    assert "覚えさせたい体の特徴の語が決まっていません" in rep0["notes"][0]["text"]


def test_pick_is_the_version_that_shows_the_feature(mods):
    rec = mods["evaluate"].recommend
    imgs = [{"prompt": i, "seed": 1, "ckpt": c, "strength": 1.0} for c in ("base", 800, "final") for i in range(4)]
    r = rec("concept", imgs, {"concept_prob": [0.1] * 4 + [0.7] * 4 + [0.5] * 4, "copy_sim": [0.4] * 12})
    assert (r["ckpt"], r["weak"]) == ("800", False)
    r = rec("concept", imgs, {"concept_prob": [0.30] * 4 + [0.33] * 4 + [0.32] * 4})
    assert r["weak"] is True
    ps = mods["evaluate"].build_prompts("concept", "include", "illustrious", "cpt_ab12")
    assert all("cpt_ab12" in p for p in ps) and any("1boy" in p for p in ps) and any("1girl" in p for p in ps)


# ---- the other targets (衣装・物・場所・ポーズ・表情): same method, different instructions, checks and prompts
def test_targets_keep_body_keys_and_change_what_is_not_described(mods):
    c = mods["captions"]
    # body projects made before targets existed keep their caption key (no re-captioning after the update)
    assert c.policy_key("concept", "include", "natural", "t", ["abs"]) == c.policy_key("concept", "include", "natural", "t", ["abs"], "body")
    assert c.policy_key("concept", "include", "natural", "t", ["abs"]) != c.policy_key("concept", "include", "natural", "t", ["abs"], "outfit")
    outfit = c.nl_instruction("concept", "include", "t", ["sailor uniform"], "outfit")
    assert "do not describe the outfit" in outfit and "clothing)" not in outfit
    pose = c.nl_instruction("concept", "include", "t", ["jumping"], "pose")
    assert "do not describe the pose" in pose and "pose and action" not in pose
    assert c.concept_target({"concept_target": "nonsense"}) == "body" and c.concept_target({}) == "body"


def test_target_projects_get_their_own_trigger_prefix(mods):
    pm = mods["store"].ProjectManager(mods["config"].PROJECTS_DIR)
    m = pm.create("制服", "concept", None, concept_words="sailor uniform", concept_target="outfit").get_meta()
    assert m["concept_target"] == "outfit" and m["trigger"].startswith("out_")
    with pytest.raises(ValueError):
        pm.create("x", "concept", None, concept_target="nonsense")


def test_object_and_place_material_needs_no_person(mods):
    j = mods["judge"]
    im = _im(1, {"no humans": .9, "sword": .9, "scenery": .5})
    im["ai"]["persons"], im["ai"]["subject"] = [], None
    for target in ("object", "place"):
        ctx = j.build_context([im], {"kind": "concept", "concept_words": ["sword"], "concept_target": target})
        assert j.check_image(im, ctx)["verdict"] == "ok"
    ctx = j.build_context([im], {"kind": "concept", "concept_words": ["sword"], "concept_target": "outfit"})
    assert any(c["key"] == "nosubj" for c in j.check_image(im, ctx)["checks"])


def test_suggestions_keep_tags_that_are_the_feature_for_the_target(mods):
    j = mods["judge"]
    pose = [_im(i, {"1girl": .9, "standing": .9, "arms up": .8}) for i in range(1, 5)]
    assert "standing" in [x["tag"] for x in j.concept_suggest(pose, [], target="pose")]
    assert "standing" not in [x["tag"] for x in j.concept_suggest(pose, [], target="body")]
    place = [_im(i, {"no humans": .9, "scenery": .9, "indoors": .8, "library": .8}) for i in range(1, 5)]
    tags = [x["tag"] for x in j.concept_suggest(place, [], target="place")]
    assert "library" in tags and "indoors" in tags and "no humans" not in tags and "scenery" not in tags


def test_each_target_has_its_own_comparison_prompts(mods):
    e, c = mods["evaluate"], mods["captions"]
    for target in c.CONCEPT_TARGETS:
        sets = e.prompt_sets("concept", target)
        assert sets[0]["key"] == "standard" and sets[0]["n"] >= 4
        for variant in ("illustrious", "sdxl_base"):
            for s in sets:
                ps = e.build_prompts("concept", "include", variant, "out_ab12", None, s["key"], target)
                assert ps and all("out_ab12" in p for p in ps)
    assert e.build_prompts("concept", "include", "illustrious", "t", target="place") != e.build_prompts("concept", "include", "illustrious", "t")
