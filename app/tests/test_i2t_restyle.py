"""i2t: 画風を変える (MOSS-Image) - the image written as another kind of picture: 実写 / アニメ風 / 漫画風 / 水彩 /
厚塗り / 油絵 / 鉛筆画.  Fake tags, no tagger model, no VLM.  Failure risks: the main picture's look winning over the
look that was asked for (the look named last, the old medium still written), a character name calling up the drawn
look in a photograph, the switch acting on targets it is not meant for, an older setting lost, and - now that the age
option is gone - anything of it still changing a prompt."""
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


def _engine(i2t, tmp_path, general, cfg, characters=(), ref_tags=None, tag_source=None):
    """An Engine whose tagger step returns the given tags (no model is loaded)."""
    tags = i2t.TagResult("general", 0.9, list(general), list(characters), dict(general))
    img = Image.new("RGB", (64, 96), (200, 180, 170))
    f = tmp_path / "メイン 画像.png"
    img.save(f)
    inputs = i2t.TagInputs(image=img, vlm_image=img, ref_images={k: img for k in ref_tags or {}}, size=(832, 1216), tags=tags,
                           person_raw=tags, ref_tags=ref_tags or {}, tag_source=tag_source or {}, main_person=None)
    eng = i2t.Engine({**i2t.DEFAULT_CONFIG, "engine": "tagger", "era_tags": False, **cfg})
    eng.tagger = lambda: SimpleNamespace(name="fake")
    eng._tag_inputs = lambda *a, **k: inputs
    eng._era_step = lambda *a, **k: None
    return eng, f


ANIME_GIRL = [("1girl", 0.99), ("solo", 0.9), ("long_hair", 0.95), ("pink_hair", 0.9), ("blue_eyes", 0.9), ("school_uniform", 0.9),
              ("anime_coloring", 0.7), ("standing", 0.8), ("outdoors", 0.7)]
PHOTO_WOMAN = [("1girl", 0.99), ("solo", 0.9), ("long_hair", 0.95), ("black_hair", 0.9), ("dress", 0.9), ("realistic", 0.9),
               ("photorealistic", 0.8), ("standing", 0.8), ("outdoors", 0.7)]


# ------------------------------------------------------------------ the text
def test_a_drawing_is_rewritten_as_a_photograph_with_the_photograph_named_first(i2t):
    shot = "The image is shot on a full-frame camera with soft window light"
    text = ("An anime-style illustration of a woman with large blue eyes and long pink hair, drawn in a modern anime style. "
            f"She wears a white dress. {shot}. Photorealistic photograph, soft lighting.")
    out, changed = i2t.restyle_text(text, "photo", 1, shot)
    assert out.startswith("A real photograph of a real person, taken with a camera. " + shot + ".")
    low = out.lower()
    assert not any(w in low for w in ("anime", "illustration", "photorealistic", "drawn in"))
    assert "a photograph of a woman with large blue eyes and long pink hair" in low and "she wears a white dress" in low
    assert out.endswith(i2t.PHOTOREAL_PEOPLE) and changed


def test_every_kind_names_its_look_first_and_ends_with_how_it_is_made(i2t):
    photo = "A photograph of a woman in a red coat on a street, realistic skin. The image is rendered in a photorealistic style."
    for kind, spec in i2t.RESTYLES.items():
        out, _ = i2t.restyle_text(photo, kind, 1)
        assert out.startswith(spec["lead"]) and out.endswith(spec["tail"]), kind
        assert "a woman in a red coat on a street" in out, kind                       # the content stays
        low = out[len(spec["lead"]):-len(spec["tail"])].lower()
        if kind != "photo":
            assert "photograph" not in low and "realistic" not in low, kind         # the old medium is gone
            assert spec["noun"] in low, kind
    assert list(i2t.RESTYLE_MODES) == ["none", *i2t.RESTYLES]                         # every kind can be chosen, and none


def test_a_photograph_becomes_a_watercolour_and_an_anime_picture_a_pencil_drawing(i2t):
    out, changed = i2t.restyle_text("A candid photograph of a man reading on a bench, shallow depth of field.", "watercolor", 1)
    assert out.startswith("A watercolour painting on textured paper. A watercolour painting of a man reading on a bench")
    assert any("photograph" in c for c in changed)
    out, _ = i2t.restyle_text("An anime illustration of a girl with twin tails, cel-shaded, with clean line art.", "pencil", 1)
    low = out.lower()
    assert low.startswith("a pencil drawing on paper. a pencil drawing of a girl with twin tails")
    assert "anime" not in low and "cel-shaded" not in low


def test_articles_stay_right_and_copied_text_is_left_alone(i2t):
    out, _ = i2t.restyle_text('An anime girl in a uniform holds a sign reading "ANIME EXPO 3D".', "photo", 1)
    assert 'A girl in a uniform holds a sign reading "ANIME EXPO 3D".' in out
    out, _ = i2t.restyle_text("A stylized elf, a 3D rendered image with realistic skin.", "photo", 1)
    assert "An elf, a photograph with real skin." in out
    out, _ = i2t.restyle_text("A photograph of an elf.", "oil", 1)
    assert "An oil painting of an elf." in out                                        # a photograph -> an oil painting
    out, _ = i2t.restyle_text("A watercolor painting of a mountain lake.", "photo", 0)
    assert out.startswith("A real photograph, taken with a camera. A photograph of a mountain lake.")
    assert out.endswith(i2t.PHOTOREAL_SCENE)


def test_which_tags_are_taken_out(i2t):
    drawn = ["anime coloring", "cel shading", "photorealistic", "realistic", "3d", "1990s (style)", "sketch"]
    assert [t for t in drawn if i2t.restyle_removes(t, "photo")] == drawn
    kept = ["film grain", "depth of field", "holding camera", "long hair", "photo background"]
    assert [t for t in kept if not i2t.restyle_removes(t, "photo", keep_style=True)] == kept
    # turning a photograph into a drawing: the photograph's own words go too (the camera someone holds does not)
    assert i2t.restyle_removes("photo (medium)", "watercolor") and i2t.restyle_removes("realistic", "anime")
    assert not i2t.restyle_removes("holding camera", "watercolor") and not i2t.restyle_removes("long hair", "manga")


# ------------------------------------------------------------------ through the engine
def test_moss_prompt_as_a_photograph(i2t, tmp_path):
    eng, f = _engine(i2t, tmp_path, ANIME_GIRL, {"target": "mossimage", "restyle": "photo"},
                     characters=[("hatsune_miku", 0.95)])
    r = eng.run(f)
    p = r.positive_prompt
    assert p.startswith("A real photograph of a real person, taken with a camera.") and p.endswith(i2t.PHOTOREAL_PEOPLE)
    low = p.lower()
    assert "anime" not in low and "illustration" not in low and "photorealistic" not in low
    assert "hatsune miku" in low                                # whose photograph it is: the name stays
    assert "Do not name" not in i2t.restyle_vlm("photo") and "keep the character's name" in i2t.restyle_vlm("photo")
    assert "pink hair" in low and "school uniform" in low       # the looks stay
    assert r.negative_prompt == ", ".join(i2t.RESTYLES["photo"]["negative"])
    assert r.detected_medium == "photo" and "画風を変える (MOSS-Image): 実写" in r.analysis


def test_moss_prompt_of_a_photograph_as_a_watercolour(i2t, tmp_path):
    eng, f = _engine(i2t, tmp_path, PHOTO_WOMAN, {"target": "mossimage", "restyle": "watercolor"},
                     characters=[("hatsune_miku", 0.95)])
    r = eng.run(f)
    low = r.positive_prompt.lower()
    assert low.startswith("a watercolour painting on textured paper.") and r.positive_prompt.endswith(i2t.RESTYLES["watercolor"]["tail"])
    assert "photograph" not in low and "realistic" not in low
    assert "black hair" in low and "miku" in low                # the name stays for every kind
    assert r.detected_medium == "painting" and r.negative_prompt == ", ".join(i2t.RESTYLES["watercolor"]["negative"])


def test_a_style_reference_of_the_same_kind_is_used_and_another_kind_is_not(i2t, tmp_path):
    photo_ref = i2t.TagResult("general", 0.9, [("photorealistic", 0.9), ("film_grain", 0.8)], [],
                              {"photorealistic": 0.9, "film_grain": 0.8, "realistic": 0.9})
    general = [t for t in ANIME_GIRL if t[0] != "anime_coloring"] + [("photorealistic", 0.9), ("realistic", 0.9), ("film_grain", 0.8)]
    src = {"photorealistic": "style", "film_grain": "style", "realistic": "style"}
    eng, f = _engine(i2t, tmp_path, general, {"target": "mossimage", "restyle": "photo"}, ref_tags={"style": photo_ref}, tag_source=src)
    r = eng.run(f)
    assert "film grain" in r.positive_prompt.lower() and not any("同じ系統" in w for w in r.warnings)
    eng, f = _engine(i2t, tmp_path, general, {"target": "mossimage", "restyle": "watercolor"}, ref_tags={"style": photo_ref}, tag_source=src)
    r = eng.run(f)
    assert any("同じ系統と判定されなかった" in w for w in r.warnings) and r.positive_prompt.startswith("A watercolour painting")


def test_nothing_happens_for_other_targets_and_when_nothing_is_chosen(i2t, tmp_path):
    eng, f = _engine(i2t, tmp_path, ANIME_GIRL, {"target": "krea2", "restyle": "photo"})
    r = eng.run(f)
    assert "real photograph of a real person" not in r.positive_prompt and "画風を変える" not in r.analysis
    eng, f = _engine(i2t, tmp_path, ANIME_GIRL, {"target": "mossimage"})
    r = eng.run(f)
    assert "real photograph of a real person" not in r.positive_prompt and r.negative_prompt == ""
    assert r.detected_medium == "anime"


def test_the_vlm_is_told_what_the_picture_becomes(i2t, tmp_path):
    for kind, word in (("photo", "real photograph of real people"), ("manga", "a black-and-white manga illustration")):
        eng, f = _engine(i2t, tmp_path, ANIME_GIRL, {"target": "mossimage", "restyle": kind, "engine": "hybrid"})
        seen = {}

        def run_vlm(*a, **k):
            seen.update(k)
            return i2t.VlmOutcome()
        eng._run_vlm = run_vlm
        eng.run(f)
        assert word in seen["extra_guidance"], kind
    # the name is asked for in every kind, the photograph too: whose live-action picture it is hangs on the name
    assert "Do not name" not in i2t.restyle_vlm("photo") and "Do not name" not in i2t.restyle_vlm("anime")
    assert "keep the character's name" in i2t.restyle_vlm("photo")


def test_changing_the_look_wins_over_leaving_the_style_out(i2t, tmp_path):
    eng, f = _engine(i2t, tmp_path, ANIME_GIRL, {"target": "mossimage", "restyle": "photo", "no_style": True})
    r = eng.run(f)
    assert r.positive_prompt.startswith("A real photograph") and any("作風の特徴を入れない" in w for w in r.warnings)


def test_settings_are_saved_and_the_older_checkbox_is_carried_over(i2t):
    i2t.save_config({**i2t.DEFAULT_CONFIG, "restyle": "oil"})
    assert i2t.load_config()["restyle"] == "oil"
    i2t.save_config({**i2t.DEFAULT_CONFIG, "restyle": "crayon"})
    assert i2t.load_config()["restyle"] == "none"
    # a config.json saved while it was the 実写化 checkbox, with the age option that is gone
    i2t.CONFIG_PATH.write_text('{"photoreal": true, "age": "20s", "target": "mossimage"}', encoding="utf-8")
    cfg = i2t.load_config()
    assert cfg["restyle"] == "photo" and "photoreal" not in cfg and "age" not in cfg


def test_the_age_chosen_by_hand_is_gone(i2t, tmp_path):
    """The old option forced one chosen adult age on everybody.  It is gone: an old setting does nothing, and the age
    that is written is the one read from the image (test_i2t_age_look.py)."""
    assert "age" not in i2t.DEFAULT_CONFIG and not hasattr(i2t, "AGE_MODES") and not hasattr(i2t, "AGE_YEARS")
    young = ANIME_GIRL + [("loli", 0.6)]
    eng, f = _engine(i2t, tmp_path, young, {"target": "krea2", "age": "20s"})          # an old setting does nothing
    r = eng.run(f)
    assert "25-year-old" not in r.positive_prompt and "twenties" not in r.positive_prompt and "年齢を指定" not in r.analysis
    assert "a little girl of elementary-school age" in r.positive_prompt             # what the tagger saw in the image


def test_the_memo_says_where_the_time_went(i2t, tmp_path, monkeypatch):
    eng, f = _engine(i2t, tmp_path, ANIME_GIRL, {"target": "mossimage"})
    clock = iter(range(0, 1000, 3))                         # every look at the clock is 3 s later
    monkeypatch.setattr(i2t.time, "time", lambda: float(next(clock)))
    r = eng.run(f)
    line = r.analysis.splitlines()[-1]
    assert line.startswith("かかった時間: タガーと検出 ") and "合計" in line
