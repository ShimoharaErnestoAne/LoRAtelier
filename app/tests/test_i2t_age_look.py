"""i2t: 年齢の見た目を書く - the apparent age of every person is read from the image and written into the prompt, so
the generated picture does not drift to the age the model likes (an adult in her twenties for the prose models, a
younger look for the anime tag models).  Fake tags and a fake VLM answer; no model is loaded.  Failure risks: a child
written as "a woman" (and the other way round), one age forced on everybody in a picture of several people, the age
written twice or against what the text already says, text copied from the image being changed, and an age under 18
written into a sexual prompt."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def i2t(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT.parent / "engines" / "i2t"))
    import i2t as mod
    monkeypatch.setattr(mod, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(mod, "MOSS_CATALOG_PATH", tmp_path / "moss_catalog" / "index.json")
    return mod


def _engine(i2t, tmp_path, general, cfg, natural="", age="", rating="general"):
    """An Engine whose tagger step returns the given tags and whose VLM step (when natural is given) returns the given
    sentence and age group."""
    tags = i2t.TagResult(rating, 0.9, list(general), [], dict(general))
    img = Image.new("RGB", (64, 96), (200, 180, 170))
    f = tmp_path / "メイン 画像.png"
    img.save(f)
    inputs = i2t.TagInputs(image=img, vlm_image=img, ref_images={}, size=(832, 1216), tags=tags, person_raw=tags,
                           ref_tags={}, tag_source={}, main_person=None)
    eng = i2t.Engine({**i2t.DEFAULT_CONFIG, "engine": "hybrid" if natural else "tagger", "era_tags": False, **cfg})
    eng.tagger = lambda: SimpleNamespace(name="fake")
    eng._tag_inputs = lambda *a, **k: inputs
    eng._era_step = lambda *a, **k: None
    if natural:
        out = i2t.VlmOutcome()
        out.out, out.natural_prompt, out.model, out.age = {"natural_prompt": natural, "age": age}, natural, "fake-vlm", age
        eng.seen = {}
        eng._run_vlm = lambda *a, **k: eng.seen.update(k) or out
        eng.vlm = lambda: SimpleNamespace(model="fake-vlm", keep_alive=0)
    return eng, f


WOMAN = [("1girl", 0.99), ("solo", 0.9), ("long_hair", 0.95), ("black_hair", 0.9), ("dress", 0.9), ("standing", 0.8),
         ("outdoors", 0.7)]
KID = [("1girl", 0.99), ("solo", 0.9), ("child", 0.8), ("female_child", 0.7), ("short_hair", 0.9), ("brown_hair", 0.9),
       ("dress", 0.9), ("standing", 0.8), ("outdoors", 0.7)]


# ------------------------------------------------------------------ the words
def test_a_person_is_named_by_their_age(i2t):
    noun = lambda subject, age, **k: i2t.age_noun_en({"subject": subject, "age": age}, **k)  # noqa: E731
    assert noun("girl", "child") == "a little girl of elementary-school age"
    assert noun("boy", "preteen") == "a preteen boy" and noun("girl", "teen") == "a teenage girl"
    assert noun("girl", "young adult") == "a woman in her twenties" and noun("boy", "forties") == "a man in his forties"
    assert noun("girl", "elderly") == "an elderly woman" and noun("other", "sixties") == "a person in their sixties"
    assert noun("other", "toddler") == "a toddler" and noun("other", "teen") == "a teenager" and noun("boy", "baby") == "a baby boy"
    assert noun("girl", "") == "a woman" and noun("boy", "") == "a man" and noun("other", "") == "a person"   # as before
    assert noun("girl", "teen", article=False) == "teenage girl"
    assert set(i2t.AGE_BAND_JA) >= set(i2t.AGE_BANDS) and "unknown" in i2t.AGE_ENUM
    assert i2t.clean_age("Young_Adult") == "young adult" and i2t.clean_age("unknown") == "" and i2t.clean_age(None) == ""


def test_the_tagger_and_the_vlm_are_put_together(i2t):
    assert i2t.age_from_tags(["1girl", "female child", "long hair"]) == "child"
    assert i2t.age_from_tags(["(old man:1.2)"]) == "elderly" and i2t.age_from_tags(["mature female"]) == "middle-aged"
    assert i2t.age_from_tags(["1girl", "long hair"]) == ""
    assert i2t.merge_age("forties", "") == "forties" and i2t.merge_age("", "child") == "child"
    assert i2t.merge_age("preteen", "child") == "preteen"          # both a child: the finer reading
    assert i2t.merge_age("young adult", "child") == "child"        # the tagger's own word decides child or not
    assert i2t.merge_age("teen", "middle-aged") == "middle-aged"
    assert i2t.merge_age("thirties", "elderly") == "elderly" and i2t.merge_age("sixties", "elderly") == "sixties"
    assert i2t.merge_age("fifties", "middle-aged") == "fifties"


def test_every_person_gets_their_own_age(i2t):
    """Three people: read one by one (the numbers of the boxes), with the tagger's tag for the one the VLM left out."""
    tags = i2t.TagResult("general", 0.9, [("2girls", 0.9), ("1boy", 0.9)], [], {})
    people = [{"subject": "girl", "position": "left", "tags": ["long hair"], "box": [0, 0, 1, 1]},
              {"subject": "girl", "position": "center", "tags": ["short hair", "child"], "box": [1, 0, 2, 1]},
              {"subject": "boy", "position": "right", "tags": ["beard"], "box": [2, 0, 3, 1]}]
    numbered = [{"number": 3, "age": "elderly"}, {"number": 1, "age": "forties"}, {"number": 2, "age": ""}]
    out, main, note = i2t.assign_ages(people, "", numbered, tags, False, False)
    assert [p["age"] for p in out] == ["forties", "child", "elderly"] and main == ""
    assert "人物 1 = 40 代" in note and "人物 2 = 子ども" in note and "人物 3 = 70 歳以上" in note
    text = i2t.describe_persons_en(out)
    assert text.startswith("Three people appear in the scene")                 # not "three women"
    assert "On the left, a woman in her forties with long hair." in text
    assert "In the center, a little girl of elementary-school age with short hair." in text   # "child" is not said twice
    assert "On the right, an elderly man with beard." in text
    assert i2t.person_lead_en(out[2]) == "The elderly man on the right"
    # no ages read: the wording of before
    same = i2t.describe_persons_en([{k: v for k, v in p.items() if k != "age"} for p in out[:2]])
    assert same.startswith("Two women appear") and "On the left, a woman with long hair." in same


def test_the_first_naming_of_the_person_is_reworded(i2t):
    fix = i2t.apply_age_text
    out = fix("An illustration of a young woman with long hair. The young woman smiles at the viewer.", [], "child")
    assert out == "An illustration of a little girl of elementary-school age with long hair. The girl smiles at the viewer."
    out = fix("A photograph of a slender young girl in a red coat. The girl holds an umbrella.", [], "forties")
    assert out == "A photograph of a slender woman in her forties in a red coat. The woman holds an umbrella."
    assert fix("A man in his thirties reads on a bench.", [], "sixties") == "A man in his sixties reads on a bench."
    assert fix("An old man reads on a bench.", [], "teen") == "A teenage boy reads on a bench."
    assert fix("A cute anime girl waves.", [], "teen") == "A cute anime teenage girl waves."
    # already said: nothing is touched
    for text, age in (("A teenage girl waves.", "teen"), ("A woman in her forties waves.", "forties"),
                      ("A little boy waves.", "child"), ("A young woman waves.", "young adult"), ("An elderly man waves.", "elderly")):
        assert fix(text, [], age) == text
    assert fix("A woman waves.", [], "") == "A woman waves."
    # text copied from the image is not a naming of the person; without one a sentence is added
    out = fix('A poster that reads "A YOUNG WOMAN" in red letters.', [], "elderly", "girl")
    assert out == 'A poster that reads "A YOUNG WOMAN" in red letters. The person shown is an elderly woman.'
    # "a photograph of a woman": the words before the noun are not swallowed
    assert fix("A view of the back of a woman.", [], "fifties") == "A view of the back of a woman in her fifties."


def test_several_people_get_a_sentence_only_when_the_text_does_not_say(i2t):
    people = [{"subject": "girl", "age": "teen", "tags": []}, {"subject": "girl", "age": "forties", "tags": []}]
    said = "On the left, a teenage girl; on the right, a woman in her forties."
    assert i2t.apply_age_text(said, people, "") == said
    out = i2t.apply_age_text("Two women stand side by side.", people, "")
    assert out == "Two women stand side by side. From left to right, the people are a teenage girl and a woman in her forties."
    unread = [{"subject": "girl", "age": "", "tags": []}, {"subject": "boy", "age": "", "tags": []}]
    assert i2t.apply_age_text("Two people stand.", unread, "") == "Two people stand."


def test_age_tags_and_the_negatives_that_hold_the_age(i2t):
    assert i2t.age_tags_for("girl", "child") == ["child", "female child"] and i2t.age_tags_for("boy", "preteen") == ["child", "male child"]
    assert i2t.age_tags_for("girl", "forties") == ["mature female"] and i2t.age_tags_for("boy", "elderly") == ["old man"]
    assert i2t.age_tags_for("girl", "teen") == [] and i2t.age_tags_for("girl", "young adult") == []   # no Danbooru tag for these
    p = i2t.with_age_tags({"subject": "boy", "age": "elderly", "tags": ["beard"], "prompt": "1boy, beard"})
    assert p["tags"] == ["old man", "beard"] and p["prompt"] == "1boy, old man, beard"
    have = {"subject": "girl", "age": "child", "tags": ["child", "female child"], "prompt": "1girl, child, female child"}
    assert i2t.with_age_tags(have) == have
    assert i2t.age_negatives([("girl", "forties")]) == ["child", "aged down", "loli"]
    assert i2t.age_negatives([("girl", "young adult"), ("boy", "elderly")]) == ["child", "aged down", "loli", "shota"]
    assert i2t.age_negatives([("boy", "child")]) == ["aged up", "mature male"]
    assert i2t.age_negatives([("girl", "child"), ("girl", "forties")]) == []       # a child and an adult: neither is suppressed
    assert i2t.age_negatives([("girl", "forties"), ("girl", "")]) == [] and i2t.age_negatives([]) == []


# ------------------------------------------------------------------ what the VLM is asked and what is read back
def test_the_vlm_is_asked_for_the_age_of_each_person(i2t):
    props = i2t.VLM_SCHEMA["properties"]
    assert props["age"]["enum"] == i2t.AGE_ENUM and "age" in i2t.VLM_SCHEMA["required"]
    person = props["persons"]["items"]
    assert person["properties"]["age"]["enum"] == i2t.AGE_ENUM and "age" in person["required"]
    ref = i2t.vlm_ref_schema(("pose", "outfit", "style"))
    assert ref["properties"]["age"]["enum"] == i2t.AGE_ENUM and "age" in ref["required"]
    assert '"age"' in i2t.VLM_SYSTEM and "elderly (70 or older)" in i2t.VLM_SYSTEM
    assert '- "age": the apparent age group of the character of image 1' in i2t.vlm_reference_system(["person", "pose"])
    got = i2t.parse_persons([{"subject": "girl", "position": "left", "tags": ["long hair"], "age": "Teen"},
                             {"subject": "boy", "position": "right", "tags": [], "age": "about 40"}])
    assert [p["age"] for p in got] == ["teen", ""]


# ------------------------------------------------------------------ through the engine
def test_prose_names_the_age_the_vlm_read(i2t, tmp_path):
    natural = "A photograph of a young woman with long black hair in a dress, standing outdoors. The young woman looks at the camera."
    eng, f = _engine(i2t, tmp_path, WOMAN, {"target": "krea2"}, natural=natural, age="forties")
    r = eng.run(f)
    assert "a woman in her forties with long black hair" in r.positive_prompt and "young" not in r.positive_prompt
    assert "The woman looks at the camera." in r.positive_prompt
    assert "年齢の見た目" in r.analysis and "40 代" in r.analysis
    assert i2t.AGE_VLM in eng.seen["extra_guidance"]


def test_a_child_is_not_written_as_a_woman_without_a_vlm(i2t, tmp_path):
    eng, f = _engine(i2t, tmp_path, KID, {"target": "mossimage"})
    r = eng.run(f)
    low = r.positive_prompt.lower()
    assert "a little girl of elementary-school age" in low and "woman" not in low
    assert ", child" not in low and "female child" not in low          # said once, in the naming
    assert "子ども" in r.analysis


def test_tag_targets_get_the_age_tag_after_the_count_and_a_negative(i2t, tmp_path):
    natural = "An illustration of a woman with long black hair in a dress."
    eng, f = _engine(i2t, tmp_path, WOMAN, {"target": "sdxl"}, natural=natural, age="forties")
    r = eng.run(f)
    chips = [c.strip() for c in r.positive_prompt.split(",")]
    assert "mature female" in chips and chips.index("mature female") < chips.index("long hair")
    assert chips.index("mature female") > chips.index("1girl")
    neg = [c.strip() for c in r.negative_prompt.split(",")]
    assert {"child", "aged down", "loli"} <= set(neg)
    # a child: the tagger's tags are there already, and the negatives hold the age the other way
    eng, f = _engine(i2t, tmp_path, KID, {"target": "sdxl"})
    r = eng.run(f)
    chips = [c.strip() for c in r.positive_prompt.split(",")]
    assert chips.count("child") == 1 and chips.count("female child") == 1
    neg = [c.strip() for c in r.negative_prompt.split(",")]
    assert {"aged up", "mature female"} <= set(neg) and "child" not in neg


def test_anima_writes_the_age_in_both_parts(i2t, tmp_path):
    natural = "A young woman with long black hair stands outdoors in a dress."
    eng, f = _engine(i2t, tmp_path, WOMAN, {"target": "anima"}, natural=natural, age="elderly")
    r = eng.run(f)
    head, _, tail = r.positive_prompt.partition(". ")
    assert "old woman" in [c.strip() for c in head.split(",")]
    assert tail.startswith("An elderly woman with long black hair")


def test_off_and_no_identity_write_nothing(i2t, tmp_path):
    natural = "A photograph of a young woman with long black hair in a dress."
    for cfg in ({"age_look": False}, {"no_identity": True}):
        eng, f = _engine(i2t, tmp_path, WOMAN, {"target": "krea2", **cfg}, natural=natural, age="forties")
        r = eng.run(f)
        assert "forties" not in r.positive_prompt and "年齢の見た目" not in r.analysis
        assert i2t.AGE_VLM not in eng.seen["extra_guidance"]
        eng, f = _engine(i2t, tmp_path, WOMAN, {"target": "sdxl", **cfg}, natural=natural, age="forties")
        r = eng.run(f)
        assert "mature female" not in r.positive_prompt and "aged down" not in r.negative_prompt


def test_two_people_not_split_get_no_age(i2t, tmp_path):
    """Two people the tags name but nobody split: whose age it is cannot be told, so none is written."""
    pair = [("2girls", 0.99), ("multiple_girls", 0.9), ("child", 0.6), ("long_hair", 0.9), ("outdoors", 0.7)]
    eng, f = _engine(i2t, tmp_path, pair, {"target": "krea2", "split_persons": False},
                     natural="Two women stand outdoors, one of them with long hair.", age="forties")
    r = eng.run(f)
    assert "forties" not in r.positive_prompt and "elementary" not in r.positive_prompt and "年齢の見た目" not in r.analysis


def test_no_age_under_18_in_a_sexual_prompt(i2t, tmp_path):
    natural = "An illustration of a girl with short brown hair."
    for target in ("krea2", "sdxl", "anima"):
        eng, f = _engine(i2t, tmp_path, [t for t in KID if t[0] not in ("child", "female_child")] + [("nude", 0.9)],
                         {"target": target}, natural=natural, age="child", rating="explicit")
        r = eng.run(f)
        assert r.nsfw
        low = (r.positive_prompt + " " + r.negative_prompt).lower()
        assert "elementary" not in low and "little girl" not in low and "child" not in low and "aged up" not in low, target
        assert "18 歳未満に見える人物の年齢は書けません" in r.analysis
    # an adult in such an image is still written with her age
    eng, f = _engine(i2t, tmp_path, WOMAN + [("nude", 0.9)], {"target": "krea2"},
                     natural="A photograph of a young woman with long black hair.", age="forties", rating="explicit")
    r = eng.run(f)
    assert "a woman in her forties" in r.positive_prompt and "書けません" not in r.analysis


def test_the_setting_is_kept_and_the_old_age_setting_is_not_read(i2t):
    import json
    assert i2t.DEFAULT_CONFIG["age_look"] is True and "age" not in i2t.DEFAULT_CONFIG
    i2t.CONFIG_PATH.write_text(json.dumps({"age": "30s", "age_look": False}), "utf-8")
    cfg = i2t.load_config()
    assert cfg["age_look"] is False and "age" not in cfg
    i2t.save_config(cfg)
    assert json.loads(i2t.CONFIG_PATH.read_text("utf-8"))["age_look"] is False
    i2t.CONFIG_PATH.write_text(json.dumps({"age_look": "yes"}), "utf-8")       # edited by hand into a wrong type
    assert i2t.load_config()["age_look"] is True


# ------------------------------------------------------------------ MOSS-Image: the negative field as well
def test_moss_negative_words_hold_the_age(i2t):
    words = i2t.age_negatives_prose
    assert words([("girl", "young adult")]) == ["child", "childlike face", "teenager"]
    assert words([("girl", "forties"), ("boy", "elderly")]) == ["child", "childlike face", "teenager", "young adult", "youthful face"]
    assert words([("girl", "thirties"), ("boy", "sixties")]) == ["child", "childlike face", "teenager"]   # one of them is 30s
    assert words([("girl", "child")]) == ["adult", "mature face", "middle-aged", "teenager"]
    assert words([("boy", "teen")]) == ["adult", "mature face", "middle-aged"]
    assert words([("girl", "child"), ("girl", "forties")]) == [] and words([("girl", "")]) == [] and words([]) == []


def test_moss_gets_the_age_in_the_negative_field_too(i2t, tmp_path):
    natural = "A photograph of a young woman with long black hair in a dress, standing outdoors."
    eng, f = _engine(i2t, tmp_path, WOMAN, {"target": "mossimage"}, natural=natural, age="forties")
    r = eng.run(f)
    assert "a woman in her forties" in r.positive_prompt
    assert r.negative_prompt == "child, childlike face, teenager, young adult, youthful face"
    assert "MOSS-Image のネガティブ欄に" in r.analysis and "CFG 1 の高速 LoRA では効きません" in r.analysis
    # a child, read by the tagger alone
    eng, f = _engine(i2t, tmp_path, KID, {"target": "mossimage"})
    assert eng.run(f).negative_prompt == "adult, mature face, middle-aged, teenager"
    # with 画風を変える: its words stay first
    eng, f = _engine(i2t, tmp_path, WOMAN, {"target": "mossimage", "restyle": "photo"}, natural=natural, age="forties")
    neg = eng.run(f).negative_prompt
    assert neg.startswith(", ".join(i2t.RESTYLES["photo"]["negative"])) and neg.endswith("young adult, youthful face")


def test_moss_negative_stays_empty_when_no_age_is_written(i2t, tmp_path):
    natural = "A photograph of a young woman with long black hair in a dress."
    for cfg in ({"age_look": False}, {"no_identity": True}):
        eng, f = _engine(i2t, tmp_path, WOMAN, {"target": "mossimage", **cfg}, natural=natural, age="forties")
        assert eng.run(f).negative_prompt == ""
    eng, f = _engine(i2t, tmp_path, WOMAN, {"target": "mossimage"}, natural=natural, age="")         # not read
    assert eng.run(f).negative_prompt == ""
    eng, f = _engine(i2t, tmp_path, [t for t in KID if t[0] not in ("child", "female_child")] + [("nude", 0.9)],
                     {"target": "mossimage"}, natural="An illustration of a girl.", age="child", rating="explicit")
    r = eng.run(f)
    assert r.nsfw and r.negative_prompt == ""                    # under 18 in a sexual image: nothing about the age
    # the other prose targets are as before
    eng, f = _engine(i2t, tmp_path, WOMAN, {"target": "krea2"}, natural=natural, age="forties")
    assert "childlike" not in eng.run(f).negative_prompt
