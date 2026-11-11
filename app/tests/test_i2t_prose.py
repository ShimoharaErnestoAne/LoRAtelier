"""i2t: prompts written in sentences (MOSS-Image, Krea 2, Qwen-Image, Z-Image ...) and how much a reference image counts.
Fake tags and a fake VLM; no model is loaded.  Failure risks: Danbooru jargon handed to a model that reads sentences,
the look of a style reference written last (so the main picture's look wins), a rewrite by the VLM that drops or adds
content being used anyway, the rewrite running where it should not, and names losing their capitals."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

KEYS = ("subject", "pose", "outfit", "objects", "background", "camera", "lighting", "style", "motion", "face")


@pytest.fixture()
def i2t(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT.parent / "engines" / "i2t"))
    import i2t as mod
    monkeypatch.setattr(mod, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(mod, "MOSS_CATALOG_PATH", tmp_path / "moss_catalog" / "index.json")
    return mod


def _extras(**tags):
    out = {k: {"tags": [], "text": "", "prompt": ""} for k in KEYS}
    for k, v in tags.items():
        if isinstance(v, str):
            out[k]["text"] = v
        else:
            out[k]["tags"] = list(v)
    return out


# ------------------------------------------------------------------ Danbooru words in sentences
def test_danbooru_words_become_plain_english_in_sentences(i2t):
    assert i2t.plain_english("cowboy shot") == "framed from the thighs up"
    assert i2t.plain_english("zettai_ryouiki").startswith("a band of bare thigh")
    assert i2t.plain_english("(ahoge:1.2)") == "a single upright strand of hair"       # weights are taken off
    assert i2t.plain_english("long hair") == "long hair"                                # plain already
    assert i2t.plain_english("in the style of Akira Toriyama") == "in the style of Akira Toriyama"   # names keep capitals
    assert i2t.plain_english("jack-o' challenge \\(meme\\)") == "jack-o' challenge (meme)"
    assert i2t._plain_prose(["solo", "long hair", "looking at viewer", "highres"]) == "long hair, looking at the camera"


def test_a_prompt_put_together_from_tags_has_no_jargon(i2t):
    ex = _extras(subject=["1girl", "solo", "long hair", "ahoge"], pose=["wariza", "looking at viewer"],
                 outfit=["serafuku", "thighhighs"], background=["simple background"], camera=["cowboy shot"])
    tags = i2t.TagResult("general", 0.9, [("1girl", 0.9)], [], {})
    for target in ("krea2", "qwen", "mossimage", "flux2"):
        text = i2t.compose_prose_fallback(ex, "anime", (832, 1216), target, tags)
        low = text.lower()
        assert not any(w in low for w in ("ahoge", "wariza", "serafuku", "thighhighs", "cowboy shot", "looking at viewer", "solo")), target
        assert "a single upright strand of hair" in low and "sailor-style school uniform" in low
        assert "framed from the thighs up" in low and "looking at the camera" in low


def test_tag_prompts_keep_the_danbooru_words(i2t, tmp_path):
    """The models that read tags are given the tags as they are: only sentences are rewritten."""
    eng, f = _engine(i2t, tmp_path, [("1girl", 0.99), ("ahoge", 0.9), ("cowboy_shot", 0.9), ("thighhighs", 0.9)], {"target": "sdxl"})
    p = eng.run(f).positive_prompt
    assert "ahoge" in p and "cowboy shot" in p and "thighhighs" in p


# ------------------------------------------------------------------ the look of a style reference comes first
def test_style_tags_of_a_style_reference_lead_the_tag_list(i2t):
    general = [("1girl", 0.9), ("solo", 0.9), ("long_hair", 0.9), ("school_uniform", 0.9), ("standing", 0.8),
               ("watercolor_(medium)", 0.8), ("outdoors", 0.7)]
    tags = i2t.TagResult("general", 0.9, general, [], dict(general))
    src = {"long_hair": "person", "school_uniform": "outfit", "standing": "pose", "watercolor_(medium)": "style",
           "outdoors": "background"}
    out = i2t.order_reference_tags(tags, src, {})
    names = [t.replace("\\", "") for t in out]
    assert names.index("watercolor (medium)") < names.index("school uniform") < names.index("long hair")
    # no style reference: the order is as before (the look sits among the person's tags)
    plain = {k: v for k, v in src.items() if v != "style"}
    names = [t.replace("\\", "") for t in i2t.order_reference_tags(tags, plain, {})]
    assert names.index("school uniform") < names.index("long hair") < names.index("outdoors")
    assert names.index("long hair") < names.index("watercolor (medium)") or names.index("watercolor (medium)") > names.index("school uniform")


def test_the_look_is_written_first_in_sentences_when_a_style_reference_is_given(i2t):
    ex = _extras(subject="A woman with long silver hair", outfit=["white dress"], background=["beach"],
                 style="Shot on a film camera with soft window light and visible grain", lighting=["soft lighting"])
    last = i2t.compose_zimage(ex, "photorealistic photograph", "a woman", "white dress", "", "", "beach", True, False)
    first = i2t.compose_zimage(ex, "photorealistic photograph", "a woman", "white dress", "", "", "beach", True, False, True)
    assert last.index("Shot on a film camera") > last.index("A woman with long silver hair")
    assert first.startswith("Photorealistic photograph. Shot on a film camera with soft window light and visible grain.")
    assert first.index("A woman with long silver hair") > first.index("Shot on a film camera")
    assert first.count("hotorealistic photograph") == 1                       # the medium is not repeated at the end
    tags = i2t.TagResult("general", 0.9, [("1girl", 0.9)], [], {})
    ex2 = _extras(subject=["1girl", "long hair"], style=["watercolor (medium)", "soft shading"], background=["beach"])
    k = i2t.compose_prose_fallback(ex2, "painting", (832, 1216), "krea2", tags, style_first=True)
    assert k.index("Style: watercolor, soft shading.") < k.index("Subject:")      # "(medium)" is Danbooru's marker
    assert i2t.plain_english("1990s (style)") == "1990s style"
    k = i2t.compose_prose_fallback(ex2, "painting", (832, 1216), "krea2", tags)
    assert k.index("Style:") > k.index("Subject:")


# ------------------------------------------------------------------ strength of the look
def test_the_look_has_a_strength_of_its_own(i2t):
    assert ("style", "画風") in i2t.STRENGTH_GROUPS
    shared = ["1girl", "long hair", "anime coloring", "school uniform"]
    out, notes = i2t.apply_strength(shared, {"style": "strong"}, "sdxl", "tags")
    assert out == ["1girl", "long hair", "(anime coloring:1.2)", "school uniform"] and not notes
    # the person's strength no longer moves the look once the look has its own
    out, _ = i2t.apply_strength(shared, {"style": "weak", "person": "strong"}, "sdxl", "tags")
    assert "(anime coloring:0.8)" in out and "(long hair:1.2)" in out
    # nothing chosen for the look: as before, it belongs to the person
    out, _ = i2t.apply_strength(shared, {"person": "strong"}, "sdxl", "tags")
    assert "(anime coloring:1.2)" in out


# ------------------------------------------------------------------ the rewrite by the VLM
def _engine(i2t, tmp_path, general, cfg, natural="", rewrite=None, ref_tags=None, tag_source=None, characters=()):
    probs = {n: p for n, p in general}
    tags = i2t.TagResult("general", 0.9, list(general), list(characters), probs)
    img = Image.new("RGB", (64, 96), (200, 180, 170))
    f = tmp_path / "メイン 画像.png"
    img.save(f)
    inputs = i2t.TagInputs(image=img, vlm_image=img, ref_images={k: img for k in ref_tags or {}}, size=(832, 1216), tags=tags, person_raw=tags,
                           ref_tags=ref_tags or {}, tag_source=tag_source or {}, main_person=None)
    eng = i2t.Engine({**i2t.DEFAULT_CONFIG, "engine": "hybrid" if (natural or rewrite) else "tagger", "era_tags": False, **cfg})
    eng.tagger = lambda: SimpleNamespace(name="fake")
    eng._tag_inputs = lambda *a, **k: inputs
    eng._era_step = lambda *a, **k: None
    eng.calls = []
    if natural or rewrite:
        out = i2t.VlmOutcome()
        out.out, out.natural_prompt, out.model = {"natural_prompt": natural}, natural, "fake-vlm"
        seen = {}
        eng._run_vlm = lambda *a, **k: seen.update(k) or out
        eng.seen = seen

        def do_rewrite(draft, guidance, style_first=False):
            eng.calls.append({"draft": draft, "guidance": guidance, "style_first": style_first})
            return rewrite(draft) if callable(rewrite) else rewrite
        eng.vlm = lambda: SimpleNamespace(model="fake-vlm", rewrite=do_rewrite, keep_alive=0)
    return eng, f


GIRL = [("1girl", 0.99), ("solo", 0.9), ("long_hair", 0.95), ("pink_hair", 0.9), ("school_uniform", 0.9), ("standing", 0.8),
        ("outdoors", 0.9), ("umbrella", 0.92)]
NATURAL = "An illustration of a girl with long pink hair standing outdoors in a school uniform."


def test_a_prompt_put_together_from_parts_is_rewritten_as_one_text(i2t, tmp_path):
    good = "An illustration of a girl with long pink hair, standing outdoors in a school uniform and holding an umbrella."
    eng, f = _engine(i2t, tmp_path, GIRL, {"target": "krea2"}, natural=NATURAL, rewrite=good,
                     ref_tags={"outfit": i2t.TagResult("general", 0.9, [("school_uniform", 0.9)], [], {})},
                     tag_source={"school_uniform": "outfit"})
    r = eng.run(f)
    assert r.positive_prompt == good and "仕上げの書き直し" in r.analysis
    call = eng.calls[0]
    assert "Subject:" in call["draft"] or "school uniform" in call["draft"]      # the assembled draft, not the VLM's summary
    assert "Krea 2" in call["guidance"]
    assert eng.seen["polish_next"] is True                                        # the model stays loaded for the rewrite


def test_a_rewrite_that_loses_or_adds_content_is_not_used(i2t, tmp_path):
    refs = {"ref_tags": {"outfit": i2t.TagResult("general", 0.9, [("school_uniform", 0.9)], [], {})},
            "tag_source": {"school_uniform": "outfit"}}
    eng, f = _engine(i2t, tmp_path, GIRL, {"target": "krea2"}, natural=NATURAL, rewrite="A girl.", **refs)
    r = eng.run(f)
    assert r.positive_prompt != "A girl." and "school uniform" in r.positive_prompt
    assert any("仕上げの書き直し" in w and "使いませんでした" in w for w in r.warnings)
    # an explanation instead of the prompt
    eng, f = _engine(i2t, tmp_path, GIRL, {"target": "krea2"}, natural=NATURAL,
                     rewrite=lambda d: "Here is the rewritten prompt:\n" + d, **refs)
    assert "Here is" not in eng.run(f).positive_prompt
    # the VLM is down: the assembled text is kept, and it is said
    def boom(d):
        raise i2t.OllamaError("connection refused")
    eng, f = _engine(i2t, tmp_path, GIRL, {"target": "krea2"}, natural=NATURAL, rewrite=boom, **refs)
    r = eng.run(f)
    assert "school uniform" in r.positive_prompt and any("できませんでした" in w for w in r.warnings)


def test_what_was_taken_out_must_not_come_back_and_copied_text_must_stay(i2t):
    draft = 'A woman in a red coat holds a sign that says "OPEN". Setting: a street at night.'
    ok = 'A woman in a red coat holds up a sign that says "OPEN" on a street at night.'
    assert i2t.polish_problem(draft, ok, ["red coat"], ["blue hair"]) == ""
    assert "外した要素" in i2t.polish_problem(draft, ok.replace("A woman", "A woman with blue hair"), ["red coat"], ["blue hair"])
    assert "画像の文字" in i2t.polish_problem(draft, ok.replace('"OPEN"', '"Open"'), ["red coat"], [])
    assert "短すぎ" in i2t.polish_problem(draft, "A woman.", ["red coat"], [])


def test_one_image_is_not_rewritten(i2t, tmp_path):
    """The text the VLM wrote from the image itself is used as it is: a second call for every image made each
    analysis slower."""
    eng, f = _engine(i2t, tmp_path, GIRL, {"target": "mossimage"}, natural=NATURAL, rewrite="x")
    r = eng.run(f)
    assert r.positive_prompt == NATURAL and eng.calls == [] and "仕上げの書き直し" not in r.analysis
    assert eng.seen["polish_next"] is False


def test_the_rewrite_can_be_turned_off_and_is_only_for_sentences(i2t, tmp_path):
    refs = {"ref_tags": {"outfit": i2t.TagResult("general", 0.9, [("school_uniform", 0.9)], [], {})},
            "tag_source": {"school_uniform": "outfit"}}
    eng, f = _engine(i2t, tmp_path, GIRL, {"target": "mossimage", "polish_prose": False}, natural=NATURAL, rewrite="x", **refs)
    assert "school uniform" in eng.run(f).positive_prompt and eng.calls == []
    for target in ("sdxl", "anima", "midjourney"):
        eng, f = _engine(i2t, tmp_path, GIRL, {"target": target}, natural=NATURAL, rewrite="x", **refs)
        eng.run(f)
        assert eng.calls == [], target
    assert i2t.DEFAULT_CONFIG["polish_prose"] is True


def test_the_rewrite_is_told_to_name_the_look_first_with_a_style_reference(i2t, tmp_path):
    general = GIRL + [("watercolor_(medium)", 0.8)]
    good = "A watercolor painting of a girl with long pink hair standing outdoors in a school uniform, holding an umbrella."
    eng, f = _engine(i2t, tmp_path, general, {"target": "qwen"}, natural=NATURAL, rewrite=good,
                     ref_tags={"style": i2t.TagResult("general", 0.9, [("watercolor_(medium)", 0.8)], [], {"watercolor_(medium)": 0.8})},
                     tag_source={"watercolor_(medium)": "style"})
    eng.run(f)
    assert eng.calls and eng.calls[0]["style_first"] is True
    assert eng.calls[0]["draft"].index("watercolor") < eng.calls[0]["draft"].index("Subject")


# ------------------------------------------------------------------ the character's name must not get lost
def test_a_rewrite_that_drops_the_character_name_is_not_used(i2t, tmp_path):
    """The rewrite is free to reword tags, so it may treat a name as jargon and describe it away.  A name cannot be
    reworded: when one that was in the assembled text is gone, the rewrite is not used."""
    refs = {"ref_tags": {"outfit": i2t.TagResult("general", 0.9, [("school_uniform", 0.9)], [], {})},
            "tag_source": {"school_uniform": "outfit"}, "characters": [("hatsune_miku", 0.95)]}
    nameless = "An illustration of a girl with long pink hair, standing outdoors in a school uniform and holding an umbrella."
    eng, f = _engine(i2t, tmp_path, GIRL, {"target": "mossimage"}, natural=NATURAL, rewrite=nameless, **refs)
    r = eng.run(f)
    assert "hatsune miku" in eng.calls[0]["draft"].lower()                     # the name was in what was handed over
    assert "hatsune miku" in r.positive_prompt.lower() and r.positive_prompt != nameless
    assert any("キャラ名・作品名" in w and "hatsune miku" in w for w in r.warnings)
    named = "An illustration of Hatsune Miku with long pink hair, standing outdoors in a school uniform and holding an umbrella."
    eng, f = _engine(i2t, tmp_path, GIRL, {"target": "mossimage"}, natural=NATURAL, rewrite=named, **refs)
    r = eng.run(f)
    assert r.positive_prompt.startswith("An illustration of Hatsune Miku") and not any("キャラ名" in w for w in r.warnings)


def test_names_are_checked_one_by_one_and_the_instructions_keep_them(i2t):
    draft = "Subject: a girl, hatsune miku, kagamine rin, long hair. Outfit: school uniform. Setting: a street at night."
    both = "Hatsune Miku and Kagamine Rin, girls with long hair, wear school uniforms on a street at night."
    names = ["hatsune miku", "kagamine rin", "megurine luka"]                  # the third was never in the draft
    assert i2t.polish_problem(draft, both, ["long hair"], [], names) == ""
    one = both.replace("Hatsune Miku and Kagamine Rin", "Hatsune Miku and another girl")
    assert "キャラ名・作品名 (kagamine rin)" in i2t.polish_problem(draft, one, ["long hair"], [], names)
    assert i2t.polish_problem(draft, one, ["long hair"], []) == ""             # without names: as before
    assert "Keep every name of a character" in i2t.POLISH_SYSTEM
    assert "keep writing the name" in i2t.AGE_VLM and "name each person" not in i2t.AGE_VLM
