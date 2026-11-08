"""Material generation: images of the user's character are generated in ComfyUI from the user's own
checkpoint + LoRAs + prompt, with a fixed set of shots (framing / angle / expression / pose / background)
added by this tool.  The user picks which results become training material.

The generation model and the model the LoRA is trained on are chosen independently.  Captions of
imported images are made by the normal analysis from what is actually in the image (in the training
model's format), not from the requested shot: the model does not always follow the request, and a
caption naming a background or framing that is not there teaches the wrong thing.  The requested
shot is kept with the image for reference.

Jobs run on the same serial GPU queue as training.  Pause = stop after the current image and keep
the rest of the plan (resumable); stop = interrupt now.  Images already generated are always kept.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import time
from pathlib import Path

from . import config, loraconf, promptkeep
from .evaluate import COMFY_MODELS, SAMPLING, build_t2i, comfy, gen_slot, gen_text_encoder, krea2_raw, load_as_fp8
from .events import bus
from .models import FAMILY_LABEL, VARIANTS, detect_lora, library, registry

SHOT_SET_VERSION = "char_std_v4"

# Standard character shot set (initial proposal, to be checked by actual training).  A character LoRA has
# to draw the character in poses and moods the material does not show, so the set varies framing (face /
# upper body / thighs up / full body), angle, expression and pose together; captions come from the analysis
# of each image, so an expression the set asks for is captioned and not baked into the character.
# Expressions cover joy, anger, sorrow and fun plus calm, surprise and embarrassment.
# Every shot has a plain background: the generated material should
# show only the character, so a scene is never requested and background words in the user's prompt are
# removed (strip_background).  The background is always white (it replaced
# white/grey alternation): plan_shots overrides the table's bg column, kept for batches planned earlier.
# A shot is (framing, angle, expression, background, pose or None).
# char_std_v4 (more actions and angles): poses grew from 6 to 23 (running, jumping,
# kneeling, lying, hand gestures ...) and two angles were added (dutch angle, from above and the side).  Poses
# need no props or scenery, so the white background stays.  Order within each framing still puts joy / anger /
# sorrow / fun and the unusual angles in the first images of a batch.
_F, _U, _C, _B = "face", "upper", "cowboy", "full"
_S = "simple"
SHOTS = [
    (_F, "front", "calm", _S, None), (_F, "three_quarter", "smile", _S, "head_tilt"),
    (_F, "front", "angry", _S, None), (_F, "overhead", "laugh", _S, None),
    (_F, "side", "sad", _S, None), (_F, "below", "surprised", _S, None),
    (_F, "dutch", "smile", _S, "v_sign"), (_F, "three_quarter", "blush", _S, "hand_on_cheek"),
    (_F, "front", "laugh", _S, None), (_F, "above", "calm", _S, None),
    (_F, "back", "smile", _S, None), (_F, "front", "sad", _S, None),
    (_F, "three_quarter", "angry", _S, None), (_F, "side", "smile", _S, None),
    (_F, "above_side", "calm", _S, None), (_F, "front", "surprised", _S, None),
    (_F, "below", "blush", _S, None), (_F, "front", "smile", _S, None),

    (_U, "front", "smile", _S, "waving"), (_U, "back_full", "calm", _S, None),
    (_U, "three_quarter", "laugh", _S, "arms_up"), (_U, "side", "angry", _S, "crossed_arms"),
    (_U, "worm", "surprised", _S, None), (_U, "above", "sad", _S, "hand_on_chest"),
    (_U, "front", "smile", _S, "v_sign"), (_U, "three_quarter", "calm", _S, "pointing"),
    (_U, "dutch", "laugh", _S, None), (_U, "front", "blush", _S, "arms_behind_back"),
    (_U, "below", "angry", _S, "fighting_stance"), (_U, "side", "sad", _S, None),
    (_U, "back", "smile", _S, None), (_U, "overhead", "surprised", _S, "reaching"),
    (_U, "three_quarter", "smile", _S, "leaning_forward"), (_U, "above_side", "calm", _S, "stretching"),
    (_U, "front", "angry", _S, None), (_U, "three_quarter", "blush", _S, "head_tilt"),
    (_U, "side", "laugh", _S, None), (_U, "front", "calm", _S, None),

    (_C, "front", "smile", _S, "hands_on_hips"), (_C, "three_quarter", "angry", _S, "fighting_stance"),
    (_C, "back_full", "calm", _S, None), (_C, "below", "laugh", _S, "arms_up"),
    (_C, "side", "sad", _S, "arms_behind_back"), (_C, "above", "surprised", _S, "leaning_forward"),
    (_C, "dutch", "smile", _S, "v_sign"), (_C, "worm", "calm", _S, "crossed_arms"),
    (_C, "back", "smile", _S, None), (_C, "three_quarter", "blush", _S, "hand_on_chest"),
    (_C, "front", "laugh", _S, "waving"), (_C, "above_side", "calm", _S, "stretching"),
    (_C, "side", "smile", _S, "walking"), (_C, "front", "angry", _S, "pointing"),
    (_C, "three_quarter", "sad", _S, None), (_C, "overhead", "smile", _S, "reaching"),

    (_B, "front", "smile", _S, "standing"), (_B, "side", "laugh", _S, "running"),
    (_B, "back_full", "calm", _S, "walking"), (_B, "three_quarter", "surprised", _S, "jumping"),
    (_B, "overhead", "smile", _S, "lying"), (_B, "front", "sad", _S, "hugging_knees"),
    (_B, "worm", "angry", _S, "fighting_stance"), (_B, "three_quarter", "calm", _S, "kneeling"),
    (_B, "above", "smile", _S, "squatting"), (_B, "dutch", "laugh", _S, "jumping"),
    (_B, "side", "calm", _S, "walking"), (_B, "below", "smile", _S, "arms_up"),
    (_B, "front", "blush", _S, "sitting"), (_B, "back", "smile", _S, "standing"),
    (_B, "three_quarter", "angry", _S, "hands_on_hips"), (_B, "above_side", "calm", _S, "sitting"),
    (_B, "front", "laugh", _S, "waving"), (_B, "side", "surprised", _S, "running"),
    (_B, "three_quarter", "smile", _S, "leaning_forward"), (_B, "front", "calm", _S, "stretching"),
    (_B, "below", "sad", _S, "standing"), (_B, "three_quarter", "blush", _S, "arms_behind_back"),
]

FRAMING_LABEL = {_F: "顔アップ", _U: "上半身", _C: "太ももから上", _B: "全身"}
ANGLE_LABEL = {"front": "正面", "three_quarter": "斜め", "side": "横", "back": "振り返り", "back_full": "真後ろ",
               "above": "俯瞰", "overhead": "真上", "below": "あおり", "worm": "真下", "dutch": "傾けた構図",
               "above_side": "斜め上から横"}
POSE_LABEL = {"standing": "立つ", "walking": "歩く", "running": "走る", "jumping": "ジャンプ", "sitting": "座る",
              "hugging_knees": "体育座り", "kneeling": "ひざをつく", "squatting": "しゃがむ", "lying": "寝転ぶ",
              "waving": "手を振る", "hands_on_hips": "腰に手", "crossed_arms": "腕組み", "arms_up": "両手を上げる",
              "stretching": "伸び", "hand_on_chest": "胸に手", "v_sign": "ピース", "pointing": "指さし",
              "arms_behind_back": "後ろ手", "head_tilt": "首をかしげる", "fighting_stance": "構える",
              "reaching": "手を伸ばす", "leaning_forward": "前かがみ", "hand_on_cheek": "ほおに手"}
EXPR_LABEL = {"calm": "穏やか", "smile": "笑顔", "laugh": "大笑い", "angry": "怒り", "sad": "悲しみ",
              "surprised": "驚き", "blush": "照れ"}
# expressions of batches planned before char_std_v3 (labels only; they are not planned any more)
_OLD_EXPR_LABEL = {"closed": "口を閉じる", "open": "口を開ける"}

TAGS = {
    "framing": {_F: "portrait, close-up", _U: "upper body", _C: "cowboy shot", _B: "full body"},
    "angle": {"front": "looking at viewer", "three_quarter": "looking to the side", "side": "from side, profile",
              "back": "from behind, looking back", "above": "from above, looking at viewer", "below": "from below",
              "back_full": "from behind, facing away", "overhead": "from above, bird's eye view",
              "worm": "from below, foreshortening, looking down", "dutch": "dutch angle",
              "above_side": "from above, from side"},
    "expr": {"calm": "closed mouth", "smile": "smile", "laugh": "laughing, open mouth, happy",
             "angry": "angry, frown, v-shaped eyebrows", "sad": "sad, frown", "surprised": "wide-eyed, open mouth, :o",
             "blush": "blush, embarrassed", "closed": "closed mouth", "open": "open mouth"},
    # "detailed" is no longer planned; kept so batches planned before char_std_v2 can be resumed
    "bg": {"simple": "simple background, white background", "grey": "simple background, grey background",
           "detailed": ["outdoors, street", "indoors, room", "outdoors, park, tree"]},
    "pose": {"standing": "standing", "walking": "walking", "sitting": "sitting", "waving": "waving",
             "hands_on_hips": "hands on hips", "crossed_arms": "crossed arms", "running": "running", "jumping": "jumping",
             "hugging_knees": "sitting, hugging own legs", "kneeling": "kneeling", "squatting": "squatting",
             "lying": "lying, on back", "arms_up": "arms up", "stretching": "stretching, arms up",
             "hand_on_chest": "hand on own chest", "v_sign": "v", "pointing": "pointing at viewer",
             "arms_behind_back": "arms behind back", "head_tilt": "head tilt", "fighting_stance": "fighting stance",
             "reaching": "reaching towards viewer", "leaning_forward": "leaning forward", "hand_on_cheek": "hand on own cheek"},
}
NATURAL = {
    "framing": {_F: "close-up portrait", _U: "upper body shot", _C: "medium shot from the thighs up", _B: "full body shot"},
    "angle": {"front": "facing the viewer", "three_quarter": "three-quarter view, looking to the side",
              "side": "side profile view", "back": "seen from behind, looking back over the shoulder",
              "above": "seen from above", "below": "low angle shot from below",
              "back_full": "seen directly from behind, facing away from the viewer",
              "overhead": "bird's-eye view from directly above", "worm": "worm's-eye view from directly below, looking down",
              "dutch": "tilted dutch angle shot", "above_side": "seen from above and from the side"},
    "expr": {"calm": "with a calm, neutral expression", "smile": "smiling", "laugh": "laughing happily with an open mouth",
             "angry": "with an angry expression and furrowed brows", "sad": "with a sad expression",
             "surprised": "with a surprised expression, wide eyes and an open mouth",
             "blush": "blushing with an embarrassed expression",
             "closed": "with a calm expression", "open": "with an open mouth"},
    "bg": {"simple": "plain white background", "grey": "plain grey background",
           "detailed": ["on a city street outdoors", "in a room indoors", "in a park with trees"]},
    "pose": {"standing": "standing", "walking": "walking", "sitting": "sitting", "waving": "waving one hand",
             "hands_on_hips": "with hands on hips", "crossed_arms": "with arms crossed", "running": "running",
             "jumping": "jumping in the air", "hugging_knees": "sitting and hugging the knees", "kneeling": "kneeling",
             "squatting": "squatting", "lying": "lying on the back", "arms_up": "raising both arms",
             "stretching": "stretching with the arms raised", "hand_on_chest": "with one hand on the chest",
             "v_sign": "making a peace sign with one hand", "pointing": "pointing at the viewer",
             "arms_behind_back": "with the arms behind the back", "head_tilt": "tilting the head",
             "fighting_stance": "in a fighting stance", "reaching": "reaching out toward the viewer",
             "leaning_forward": "leaning forward", "hand_on_cheek": "with one hand on the cheek"},
}
# Negative words per shot (tag-style models only): the opposite expression, and a cropped body for a
# full-body shot, are the usual ways the model misses the request.  "calm" is not used as a tag: in a
# test (waiIllustriousSDXL v17, 2026-10-12) it changed the art style and was drawn as text; "surprised" drew
# "!" symbols even with them in the negative, so surprise is asked for by the face alone (wide-eyed, :o).
NEG = {
    "expr": {"calm": "open mouth, angry", "smile": "frown, angry, sad", "laugh": "frown, angry, sad, closed mouth",
             "angry": "smile, happy", "sad": "smile, happy", "surprised": "closed mouth, spoken exclamation mark, emphasis lines, !, notice lines",
             "blush": "angry"},
    "framing": {_B: "feet out of frame, close-up, bad feet"},
    "angle": {"back_full": "looking back, looking at viewer", "worm": "upskirt, panties"},
    "pose": {"waving": "bad hands, extra fingers", "hands_on_hips": "bad hands", "crossed_arms": "bad hands, extra arms",
             "v_sign": "bad hands, extra fingers", "pointing": "bad hands, extra fingers", "hand_on_chest": "bad hands",
             "hand_on_cheek": "bad hands", "reaching": "bad hands, extra fingers", "arms_up": "extra arms",
             "stretching": "extra arms", "running": "extra legs", "jumping": "extra legs", "kneeling": "extra legs",
             "squatting": "extra legs", "hugging_knees": "extra legs"},
}
# Minimal guard against broken images (broken ones can be unpicked, so this only
# needs to avoid the common failures).  The model's own quality words are added when the user's prompt has
# none; the negative gets the model's quality negative (without nsfw: the material is the user's choice)
# and, for tag models, the usual anatomy failures and several figures in one image.
QUALITY_WORDS = ("masterpiece", "best quality", "high quality", "amazing quality", "score_9", "score_8")
ANATOMY_NEG = ("bad anatomy, bad hands, extra fingers, missing fingers, fused fingers, extra arms, extra legs, "
               "deformed, multiple views, split screen, comic")


def _quality(variant: str | None) -> tuple[str, str]:
    from .evaluate import QUALITY
    pos, neg = QUALITY.get(variant or "", ("", ""))
    pos = ", ".join(t.strip() for t in pos.split(",") if t.strip() and t.strip() != "rating_safe")
    neg = ", ".join(t.strip() for t in neg.split(",") if t.strip() and t.strip() != "nsfw")
    return pos, neg
SIZE = {_F: (1024, 1024), _U: (896, 1152), _C: (896, 1152), _B: (832, 1216)}
# Qwen-Image-2512 card: 1328x1328 / 3:4 1104x1472 / 2:3 1056x1584
FAMILY_SIZE = {"qwen_image": {_F: (1328, 1328), _U: (1104, 1472), _C: (1104, 1472), _B: (1056, 1584)}}
GEN_FAMILIES = ("sdxl", "anima", "qwen_image", "qwen_image21", "krea2", "zimage", "moss_image")


# ======================================================================== style material
# For a style LoRA the user's prompt and LoRAs carry the style; the tool swaps the subject so the LoRA
# learns the drawing, not one character or scene (spec 7: people, animals, scenery and objects, different
# compositions).  Backgrounds stay: they are part of the style.  A subject is
# (key, group, size, tags, natural sentence).  Sizes: "sq" 1:1, "tall" 3:4, "wide" 4:3.
STYLE_SET_VERSION = "style_std_v1"
STYLE_SUBJECTS = [
    ("girl_portrait", "person", "sq", "1girl, solo, portrait, smile, looking at viewer", "a portrait of a smiling young woman looking at the viewer"),
    ("cat_window", "animal", "sq", "cat, sitting, windowsill, window, sunlight, no humans", "a cat sitting on a sunny windowsill"),
    ("mountain_lake", "scenery", "wide", "scenery, mountains, lake, sky, clouds, no humans", "a landscape of mountains and a lake under a cloudy sky"),
    ("still_life", "object", "sq", "still life, fruit bowl, apple, grapes, table, no humans", "a still life of a fruit bowl with apples and grapes on a table"),
    ("boy_full", "person", "tall", "1boy, solo, full body, standing, jacket, street", "a young man in a jacket standing on a street, full body"),
    ("city_night", "scenery", "wide", "cityscape, night, street lights, buildings, road, no humans", "a city street at night with street lights and buildings"),
    ("dog_field", "animal", "wide", "dog, running, grass field, outdoors, no humans", "a dog running across a grass field"),
    ("cafe_table", "object", "sq", "cup, coffee, cake, plate, cafe table, indoors, no humans", "a cup of coffee and a slice of cake on a cafe table"),
    ("old_man", "person", "sq", "old man, solo, portrait, beard, hat", "a portrait of an old man with a beard and a hat"),
    ("forest_path", "scenery", "tall", "forest, path, trees, sunlight, no humans", "a path through a sunlit forest"),
    ("bird_branch", "animal", "sq", "bird, perched, tree branch, leaves, no humans", "a small bird perched on a tree branch"),
    ("flower_vase", "object", "tall", "flowers, vase, table, indoors, no humans", "a vase of flowers on a table"),
    ("girl_sitting", "person", "tall", "1girl, solo, sitting, chair, reading, book, indoors", "a young woman sitting on a chair reading a book indoors"),
    ("seaside_town", "scenery", "wide", "seaside town, ocean, houses, harbor, boats, no humans", "a seaside town with houses, a harbor and boats"),
    ("horse_meadow", "animal", "wide", "horse, standing, meadow, outdoors, no humans", "a horse standing in a meadow"),
    ("room_interior", "scenery", "wide", "bedroom, interior, bed, window, curtains, no humans", "the interior of a bedroom with a bed and a window"),
    ("two_people", "person", "wide", "1girl, 1boy, walking, street, outdoors, from side", "a young woman and a young man walking side by side on a street"),
    ("old_car", "object", "wide", "car, old car, street, outdoors, no humans", "an old car parked on a street"),
    ("child_park", "person", "tall", "child, solo, running, park, trees, happy", "a happy child running in a park"),
    ("castle_sky", "scenery", "tall", "castle, sky, clouds, hill, no humans", "a castle on a hill under a sky with clouds"),
    ("fish_water", "animal", "wide", "fish, underwater, coral, water, no humans", "fish swimming among corals underwater"),
    ("food_table", "object", "wide", "food, bowl, ramen, chopsticks, table, no humans", "a bowl of ramen with chopsticks on a table"),
    ("woman_upper", "person", "tall", "1girl, solo, upper body, from side, wind, long hair, outdoors", "a young woman with long hair blowing in the wind, upper body, seen from the side"),
    ("rainy_street", "scenery", "tall", "rain, street, umbrella, puddle, buildings, no humans", "a rainy street with puddles and an umbrella"),
]
STYLE_GROUP_LABEL = {"person": "人物", "animal": "動物", "scenery": "風景・室内", "object": "物・食べ物"}
STYLE_SUBJECT_LABEL = {
    "girl_portrait": "女性の顔", "cat_window": "窓辺の猫", "mountain_lake": "山と湖", "still_life": "果物の静物",
    "boy_full": "男性の全身", "city_night": "夜の街", "dog_field": "走る犬", "cafe_table": "コーヒーとケーキ",
    "old_man": "老人の顔", "forest_path": "森の小道", "bird_branch": "枝の小鳥", "flower_vase": "花瓶の花",
    "girl_sitting": "読書する女性", "seaside_town": "海辺の町", "horse_meadow": "草原の馬", "room_interior": "部屋の中",
    "two_people": "歩く二人", "old_car": "古い車", "child_park": "公園の子ども", "castle_sky": "丘の城",
    "fish_water": "水中の魚", "food_table": "ラーメン", "woman_upper": "風になびく髪", "rainy_street": "雨の街",
}
STYLE_SIZE = {"sq": (1024, 1024), "tall": (896, 1152), "wide": (1152, 896)}
STYLE_FAMILY_SIZE = {"qwen_image": {"sq": (1328, 1328), "tall": (1104, 1472), "wide": (1472, 1104)}}
# subject words in the style prompt would put the same subject into every image; the tool picks subjects
SUBJECT_WORDS = {"1girl", "1boy", "2girls", "2boys", "multiple girls", "multiple boys", "solo", "girl", "boy",
                 "woman", "man", "no humans", "1other"}
_STYLE = {k: (g, z, t, n) for k, g, z, t, n in STYLE_SUBJECTS}


def plan_style(count: int, exclude_groups=()) -> list[dict]:
    kept = [x for x in STYLE_SUBJECTS if x[1] not in exclude_groups]
    if not kept:
        raise ValueError("すべての題材が外されています")
    return [{"subject": kept[i % len(kept)][0]} for i in range(count)]


def strip_subject_words(positive: str, caption_style: str) -> tuple[str, list[str]]:
    if caption_style != "tags":
        return positive, []
    kept, removed = [], []
    for t in positive.split(","):
        w = t.strip()
        if not w:
            continue
        bare = w.strip("()[]{} ").split(":")[0].strip().lower().replace("_", " ")
        (removed if bare in SUBJECT_WORDS else kept).append(w)
    return ", ".join(kept), removed


def style_size(model: dict, shot: dict) -> tuple[int, int]:
    return STYLE_FAMILY_SIZE.get(model["family"], STYLE_SIZE)[_STYLE[shot["subject"]][1]]


def compose_style_prompt(positive: str, shot: dict, caption_style: str, variant: str | None = None) -> str:
    """The user's style words + the subject of this image.  Quality words lead as for characters."""
    positive = strip_subject_words(positive, caption_style)[0].strip().rstrip(",.").strip()
    natural = _STYLE[shot["subject"]][3]
    natural = natural[0].upper() + natural[1:]
    if caption_style != "tags":
        return f"{natural}. {positive}" if positive else natural + "."
    tags = [t.strip() for t in positive.split(",") if t.strip()]
    low = {t.strip("()[]{} ").split(":")[0].strip().lower() for t in tags}
    added = 0
    if not any(q in low for q in QUALITY_WORDS):
        q = [t.strip() for t in _quality(variant)[0].split(",") if t.strip()]
        tags, added = q + tags, len(q)
    n = 0
    while n < len(tags) and tags[n].startswith(("score_", "source_", "rating_")):
        n += 1
    n = max(n, added)
    return ", ".join(tags[:n] + [t.strip() for t in _STYLE[shot["subject"]][2].split(",")] + tags[n:])


def compose_style_negative(negative: str, shot: dict, caption_style: str, variant: str | None = None) -> str:
    negative = (negative or "").strip().rstrip(",").strip()
    base = _quality(variant)[1]
    parts = [base]
    if caption_style == "tags" and (base or variant not in ("krea2", "qwen_image21")):
        parts.append("multiple views, split screen, comic")
        if _STYLE[shot["subject"]][0] == "person":
            parts.append(ANATOMY_NEG)
    have = {t.strip().lower() for t in negative.split(",") if t.strip()}
    extra = []
    for part in parts:
        for t in part.split(","):
            t = t.strip()
            if t and t.lower() not in have:
                have.add(t.lower())
                extra.append(t)
    return ", ".join([negative] * bool(negative) + extra)


def plan_shots(count: int, exclude_angles=(), exclude_framings=(), exclude_exprs=(), exclude_poses=()) -> list[dict]:
    """count images from the standard set: the set is repeated (with different seeds) as needed.  An excluded
    pose leaves its shots in the plan without the pose (the framing, angle and expression are still wanted)."""
    kept = [s for s in SHOTS if s[1] not in exclude_angles and s[0] not in exclude_framings and s[2] not in exclude_exprs]
    if not kept:
        raise ValueError("すべてのアングル・画角・表情が外されています")
    kept = [s[:4] + (None if s[4] in exclude_poses else s[4],) for s in kept]
    # interleave framings so the first few images already show face / upper body / thighs up / full body:
    # the user can judge early and pause before the whole set is generated
    groups = [[s for s in kept if s[0] == f] for f in (_F, _U, _C, _B)]
    base = [g[i] for i in range(max(map(len, groups))) for g in groups if i < len(g)]
    out = []
    for i in range(count):
        fr, an, ex, bg, pose = base[i % len(base)]
        shot = {"framing": fr, "angle": an, "expr": ex, "bg": "simple"}   # white background only
        if pose:
            shot["pose"] = pose
        out.append(shot)
    return out


def shot_tags(shot: dict) -> list[str]:
    parts = [TAGS["framing"][shot["framing"]], TAGS["angle"][shot["angle"]], TAGS["expr"][shot["expr"]]]
    if shot.get("pose"):
        parts.append(TAGS["pose"][shot["pose"]])
    bg = TAGS["bg"][shot["bg"]]
    parts.append(bg[shot.get("bg_i", 0)] if isinstance(bg, list) else bg)
    return [t.strip() for p in parts for t in p.split(",") if t.strip()]


def shot_sentence(shot: dict) -> str:
    n = NATURAL
    bg = n["bg"][shot["bg"]]
    bg = bg[shot.get("bg_i", 0)] if isinstance(bg, list) else bg
    pose = f", {n['pose'][shot['pose']]}" if shot.get("pose") else ""
    return f"{n['framing'][shot['framing']]}, {n['angle'][shot['angle']]}, {n['expr'][shot['expr']]}{pose}, {bg}"


def gen_caption(shot: dict, kind: str, outfit_mode: str, trigger: str, clothing=()) -> str | None:
    """Natural-language caption from the shot a material image was generated with ("gen" caption mode).
    Describes what was requested, which the model does not always follow; the user chose this for speed.
    The user's own prompt is not used: it describes the character, which must stay bound to the trigger."""
    if shot.get("subject"):
        if kind != "style" or shot["subject"] not in _STYLE:
            return None
        return f"{trigger} style. {_STYLE[shot['subject']][3]}."
    if kind != "character":
        return None
    n = NATURAL
    try:
        parts = [n["framing"][shot["framing"]], n["angle"][shot["angle"]]]
        if shot["angle"] != "back_full":   # the face is not visible from directly behind
            parts.append(n["expr"][shot["expr"]])
        if shot.get("pose"):
            parts.append(n["pose"][shot["pose"]])
        if outfit_mode == "change" and clothing:
            parts.append("wearing " + ", ".join(clothing))
        bg = n["bg"][shot["bg"]]
        parts.append(bg[shot.get("bg_i", 0)] if isinstance(bg, list) else bg)
    except KeyError:   # a shot of an unknown (newer or older) set
        return None
    return f"{trigger}, {', '.join(parts)}."


def shot_label(shot: dict) -> str:
    if shot.get("subject"):
        return f"{STYLE_GROUP_LABEL[_STYLE[shot['subject']][0]]}・{STYLE_SUBJECT_LABEL[shot['subject']]}"
    expr = EXPR_LABEL.get(shot.get("expr")) or _OLD_EXPR_LABEL.get(shot.get("expr"))
    return "・".join(x for x in (FRAMING_LABEL.get(shot["framing"]), ANGLE_LABEL.get(shot["angle"]), expr,
                                 POSE_LABEL.get(shot.get("pose"))) if x)


def compose_negative(negative: str, shot: dict, caption_style: str, variant: str | None = None, positive: str = "") -> str:
    """The user's negative plus the model's quality negative and, for tag-style models, the anatomy and
    shot negative words (natural-language models such as Qwen-Image follow the positive sentence; a cfg 1
    model has an empty quality negative and gets nothing added).  positive: the user's prompt; for tag-style
    models the opposite of the gender / age / hair length / breast size it asks for is added too (promptkeep),
    unless the prompt itself has that word."""
    negative = (negative or "").strip().rstrip(",").strip()
    base = _quality(variant)[1]
    if caption_style != "tags":
        parts = [base]
    elif not base and variant in ("krea2", "qwen_image21"):
        parts = []
    else:
        parts = [base, ANATOMY_NEG, NEG["expr"].get(shot.get("expr"), ""), NEG["framing"].get(shot.get("framing"), ""),
                 NEG["angle"].get(shot.get("angle"), ""), NEG["pose"].get(shot.get("pose"), "")]
        if positive:
            asked = {promptkeep._bare(t) for t in positive.split(",")}
            parts.append(", ".join(t for t in promptkeep.extra_negative(promptkeep.requested(positive, "tags"))
                                   if t not in asked))
    have = {t.strip().lower() for t in negative.split(",") if t.strip()}
    extra = []
    for part in parts:
        for t in part.split(","):
            t = t.strip()
            if t and t.lower() not in have:
                have.add(t.lower())
                extra.append(t)
    return ", ".join([negative] * bool(negative) + extra)


# Framing / angle / expression / pose words removed from the user's prompt (tag-style only): the shot
# set decides these, and e.g. "smile, full body" left in the character prompt turns every angry face
# into a smile and every close-up into a full body.
SHOT_WORDS = {
    "portrait", "close-up", "upper body", "cowboy shot", "full body", "lower body", "looking at viewer",
    "looking to the side", "looking away", "looking back", "from side", "profile", "from behind", "from above",
    "from below", "straight-on", "smile", "smiling", "grin", "light smile", "laughing", "open mouth", "closed mouth",
    "happy", "angry", "annoyed", "sad", "crying", "surprised", "wide-eyed", "blush", "embarrassed", "expressionless",
    "frown", "calm", "standing", "sitting", "walking", "waving", "hands on hips", "crossed arms",
    "facing away", "bird's eye view", "foreshortening", "looking down", "looking up", "dutch angle", "running", "jumping",
    "kneeling", "squatting", "lying", "on back", "arms up", "stretching", "hand on own chest", "v", "peace sign",
    "pointing", "pointing at viewer", "arms behind back", "head tilt", "fighting stance", "reaching",
    "reaching towards viewer", "leaning forward", "hand on own cheek", "hugging own legs",
}


def strip_shot_words(positive: str, caption_style: str) -> tuple[str, list[str]]:
    if caption_style != "tags":
        return positive, []
    kept, removed = [], []
    for t in positive.split(","):
        w = t.strip()
        if not w:
            continue
        bare = w.strip("()[]{} ").split(":")[0].strip().lower().replace("_", " ")
        (removed if bare in SHOT_WORDS else kept).append(w)
    return ", ".join(kept), removed


# Background / scene words removed from the user's prompt.  A tag is a background when it names a
# background ("night background") or a scene word, or ends with a place noun ("city street", "school
# classroom").  Other tags keep their place words ("street clothes", "sky blue hair", "school uniform").
BG_TAGS = {
    "scenery", "outdoors", "indoors", "landscape", "cityscape", "nature", "day", "daytime", "night", "sunset",
    "sunrise", "city lights", "neon lights", "bokeh", "depth of field", "horizon", "underwater", "snow", "rain",
    "water", "grass", "window", "bed", "couch", "chair", "table", "desk", "floor", "wall", "fence", "bench",
}
PLACE_NOUNS = {
    "city", "street", "road", "alley", "town", "building", "buildings", "room", "bedroom", "classroom", "kitchen",
    "bathroom", "office", "library", "cafe", "restaurant", "shop", "school", "park", "forest", "tree", "trees",
    "field", "garden", "beach", "ocean", "sea", "lake", "river", "mountain", "mountains", "sky", "cloud", "clouds",
    "stage", "shrine", "temple", "castle", "ruins", "space", "interior", "hallway", "station", "background",
}
_BG_PHRASES = ("background", "scenery", "outdoors", "indoors", "landscape", "cityscape")
_PLACE_RE = r"\b(in|at|on|inside|outside|under|by) (a|an|the) (\w+ ){0,2}(" + "|".join(sorted(PLACE_NOUNS)) + r")\b"


def strip_background(positive: str, caption_style: str) -> tuple[str, list[str]]:
    """Remove background / scene words from the user's prompt.  Returns (prompt, removed words)."""
    import re
    if caption_style == "tags":
        kept, removed = [], []
        for t in positive.split(","):
            w = t.strip()
            if not w:
                continue
            bare = w.strip("()[]{} ").split(":")[0].strip().lower().replace("_", " ")
            if bare in BG_TAGS or bare.split(" ")[-1] in PLACE_NOUNS:
                removed.append(w)
            else:
                kept.append(w)
        return ", ".join(kept), removed
    # natural language: drop comma / sentence chunks that describe the background or the place
    parts = [p for p in re.split(r"(?<=[,.;])\s*", positive) if p.strip()]
    kept, removed = [], []
    for p in parts:
        low = p.lower()
        if any(k in low for k in _BG_PHRASES) or re.search(_PLACE_RE, low):
            removed.append(p.strip(" ,.;"))
        else:
            kept.append(p)
    return " ".join(kept).strip(), removed


def compose_prompt(positive: str, shot: dict, caption_style: str, variant: str | None = None) -> str:
    positive = strip_background(positive, caption_style)[0]
    positive = strip_shot_words(positive, caption_style)[0]
    positive = positive.strip().rstrip(",.").strip()
    if caption_style == "tags":
        # framing first: earlier tags weigh more, and framing must win over e.g. "boots" in the character
        # prompt.  Then the user's tags, then the rest of the shot (angle, expression, pose, background): with the
        # whole shot in front the model kept the pose / expression / angle but changed the hair, outfit, age,
        # gender and body the user wrote.  Pony's score_/source_/rating_ tags only work
        # at the very start, so they stay in front.
        tags = [t.strip() for t in positive.split(",") if t.strip()]
        low = {t.strip("()[]{} ").split(":")[0].strip().lower() for t in tags}
        added = 0
        if not any(q in low for q in QUALITY_WORDS):
            q = [t.strip() for t in _quality(variant)[0].split(",") if t.strip()]
            tags, added = q + tags, len(q)
        if "solo" not in low:
            tags.append("solo")
        n = 0
        # the user's own leading quality words stay in front too (they are not about the character)
        while n < len(tags) and (tags[n].startswith(("score_", "source_", "rating_"))
                                 or tags[n].strip("()[]{} ").split(":")[0].strip().lower() in QUALITY_WORDS + ("absurdres", "very aesthetic", "newest")):
            n += 1
        n = max(n, added)   # added quality words also stay in front
        shot_t = shot_tags(shot)   # the framing tags come first in it
        k = len([t for t in TAGS["framing"][shot["framing"]].split(",") if t.strip()])
        return ", ".join(tags[:n] + shot_t[:k] + tags[n:] + shot_t[k:])
    text = f"{positive}. {shot_sentence(shot)}." if positive else shot_sentence(shot) + "."
    if (variant or "").startswith("moss_image") and shot.get("framing") == _B:
        text += " " + MOSS_FULL_BODY
    return text


# Full-body shots on MOSS-Image: the framing wording of hide1's MOSS-Image Character Catalog (its prompts for 33,935
# characters drawn by MOSS-Image SFT).  Whether it keeps the whole body in frame more often
# than "full body shot" alone has not been compared.
MOSS_FULL_BODY = ("Zoomed-out full-body view, centered, with the entire subject occupying about 65 percent of the canvas. "
                  "Keep the entire character visible within the frame, with space around the silhouette.")


# ======================================================================== checks
def check_recipe(recipe: dict, kind: str = "character", project=None) -> dict:
    """Errors that make generation impossible.  Only sure mismatches are errors; a LoRA whose model
    family cannot be read is allowed (the user can pause if the result looks wrong)."""
    errors, model = check_model_loras(recipe.get("gen_model_id"), recipe.get("loras"))
    if project is not None and any(not source_file(gen_source_dir(project), n) for n in recipe.get("init_images") or []):
        errors.append("元画像の一部が見つかりません。外してから、もう一度選んでください。")
    if kind == "style":
        if not (recipe.get("positive") or "").strip() and not recipe.get("loras"):
            errors.append("画風を表すプロンプトを入力するか、画風の LoRA を追加してください。")
    elif not (recipe.get("positive") or "").strip():
        errors.append("キャラクターのプロンプトを入力してください。")
    notes = []
    from .models import experimental_note
    if model and experimental_note(model.get("variant")):
        notes.append(experimental_note(model.get("variant")))
    if tuned_note(model):
        notes.append(tuned_note(model))
    style = prompt_style(model) if model else "tags"
    if kind == "style":
        removed = strip_subject_words(recipe.get("positive") or "", style)[1]
        if removed:
            notes.append("題材はツールが 1 枚ずつ入れ替えるため、プロンプトの次の指定は使いません: " + ", ".join(removed))
        return {"errors": errors, "notes": notes, "model": model}
    removed = strip_background(recipe.get("positive") or "", style)[1]
    if removed:
        notes.append("背景は単色で生成するため、プロンプトの次の指定は使いません: " + ", ".join(removed))
    removed = strip_shot_words(recipe.get("positive") or "", style)[1]
    if removed:
        notes.append("画角・アングル・表情・ポーズはツールが 1 枚ずつ指定するため、プロンプトの次の指定は使いません: " + ", ".join(removed))
    return {"errors": errors, "notes": notes, "model": model}


def check_model_loras(model_id: str | None, loras: list | None) -> tuple[list[str], dict | None]:
    """Errors in the generation model and LoRAs (shared with the plain image generation page)."""
    errors = []
    model = registry.get(model_id or "")
    if not model:
        errors.append("生成に使うモデルを選んでください。")
    elif model["missing"]:
        errors.append(f"生成モデルのファイルが見つかりません: {', '.join(model['missing'])}")
    elif model["family"] not in GEN_FAMILIES:
        errors.append(f"{FAMILY_LABEL.get(model['family'], model['family'])} での生成には対応していません。")
    elif bad := [k for k in ("te", "vae") if model["files"].get(k) and library.fits(model["family"], k, model["files"][k]) is False]:
        # a fitting one is put in automatically when the folders have it (Library._repair_companions)
        label = FAMILY_LABEL.get(model["family"], model["family"])
        errors.append("・".join(f"{'テキストエンコーダー' if k == 'te' else 'VAE'}（{Path(model['files'][k]).name}）" for k in bad)
                      + f" は {label} 用ではないため生成できません。{label} 用のファイルを text_encoders / vae フォルダーに置いてください。")
    elif model["family"] in ("qwen_image", "qwen_image21") and str(model["files"].get("gen", "")).lower().endswith(".gguf") \
            and not _gguf_ready():
        errors.append("GGUF 版の生成に必要な部品（ComfyUI-GGUF）が入っていません。設定画面の「おすすめのモデル」から"
                      "このモデルの軽量版（生成用）をもう一度追加すると導入されます。")
    for lo in loras or []:
        p = Path(lo.get("path") or "")
        if not p.is_file():
            errors.append(f"LoRA が見つかりません: {p.name or '(未選択)'}")
            continue
        try:
            fam = detect_lora(p)["family"]
        except (OSError, ValueError) as e:
            errors.append(f"LoRA を読めません: {p.name}（{e}）")
            continue
        if model and fam and fam != model["family"]:
            errors.append(f"LoRA「{p.stem}」は {FAMILY_LABEL.get(fam, fam)} 用で、生成モデル（{FAMILY_LABEL.get(model['family'])}）では使えません。")
    return errors, model


def _gguf_ready() -> bool:
    from .addons import engine_ready
    return engine_ready("ComfyUI-GGUF")


def is_turbo_gguf(path: str | None) -> bool:
    """Qwen-Image-2.1 light file: the Turbo GGUF (8 steps) or a GGUF of the normal model (25 steps).
    Only the file name tells them apart; an unnamed one gets the normal model's steps (slower, still works)."""
    return bool(path) and "turbo" in Path(path).name.lower()


def prompt_style(model: dict | None) -> str:
    from .models import VARIANTS
    return VARIANTS.get((model or {}).get("variant"), {}).get("caption", "tags")   # unknown SDXL variant: tags


# ======================================================================== ComfyUI graph
def _stage(kind: str, src: str) -> str:
    """Expose a file to ComfyUI without copying (evaluate.Comfy.stage_model: hard link in a subfolder per
    source folder)."""
    return comfy.stage_model(kind, src)


# ======================================================================== image-to-image source images
GEN_DENOISE_DEFAULT = 0.75   # material: the pose / angle / expression of each shot must still be able to change
MAX_SOURCES = 20


def save_source_image(data: bytes, folder: Path) -> dict:
    """Store a source image for image-to-image (RGB PNG, one file per content; transparency on white, as the
    material images)."""
    import hashlib
    import io
    from PIL import Image, ImageOps
    try:
        with Image.open(io.BytesIO(data)) as im0:
            im = ImageOps.exif_transpose(im0)   # photos taken sideways
            if im.mode in ("RGBA", "LA", "P") and (im.mode != "P" or "transparency" in im.info):
                rgba = im.convert("RGBA")
                rgb = Image.new("RGB", rgba.size, (255, 255, 255))
                rgb.paste(rgba, mask=rgba.getchannel("A"))
            else:
                rgb = im.convert("RGB")
    except Exception:
        raise ValueError("画像として開けません（PNG / JPEG / WebP などに対応）") from None
    name = hashlib.sha256(data).hexdigest()[:20] + ".png"
    f = folder / name
    if not f.exists():
        folder.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".part")
        rgb.save(tmp, "PNG")
        tmp.replace(f)
    return {"name": name, "size": list(rgb.size)}


def source_file(folder: Path, name) -> Path | None:
    f = folder / Path(str(name or "")).name
    return f if name and f.is_file() else None


def clamp_denoise(v, default: float) -> float:
    try:
        return max(0.05, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default


def stage_comfy_input(src: Path) -> str:
    """The name LoadImage finds the image by (a copy in ComfyUI's input folder; names are content hashes)."""
    from .evaluate import COMFY_INPUT
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    if not (COMFY_INPUT / src.name).exists():
        shutil.copyfile(src, COMFY_INPUT / src.name)
    return src.name


def gen_source_dir(project) -> Path:
    return project.path / "gen_sources"


def build_graph(model: dict, staged: dict, loras: list[tuple[str, float]], prompt: str, negative: str, seed: int,
                width: int, height: int, sampling: dict, prefix: str, init: dict | None = None,
                regions: list[str] | None = None) -> dict:
    return build_t2i(model, staged, loras, prompt, negative, seed, width, height, sampling, prefix, lora_clip=True, init=init,
                     regions=regions)


def stage_model_files(model: dict) -> dict:
    """Names ComfyUI knows the generation model's files by (same choice of file as the evaluation)."""
    f = model["files"]
    fam = model["family"]
    if fam == "sdxl":
        return {"ckpt": _stage("checkpoints", f["ckpt"]), "on": "ckpt"}
    on = gen_slot(model)
    dit = f[on]
    out = {"dit": _stage("diffusion_models", dit), "te": _stage("text_encoders", gen_text_encoder(fam, f["te"])), "vae": _stage("vae", f["vae"]),
           "on": on, "gguf": dit.lower().endswith(".gguf")}
    for choice, slot in FAST_SLOTS.items():
        if fast_lora(model, choice):
            out[slot] = _stage("loras", fast_lora(model, choice))
    if load_as_fp8(fam, dit):
        out["dtype"] = "fp8_e4m3fn"   # a bf16 model larger than the GPU: half the size, so it is not streamed every step
    return out


# 高速生成 of MOSS-Image SFT: the form's choice -> the model's file slot (8-step DMD LoRA / akhaliq's 4-step LoRA)
FAST_SLOTS = {"on": "fast", "4": "fast4"}
FAST_SAMPLING = {"on": "moss_image_fast", "4": "moss_image_fast4"}

# Fine-tuned MOSS-Image files that need their own sampling (these are
# not in the download catalog).  Found by the sha256 Civitai lists, or by the file name until the background hash is
# done.  Batman MOSS v3.0 (civitai.com/models/3028665, INT8): the 8-step turbo LoRA is merged in "at a gradient", so
# no fast LoRA goes on top; its page recommends 6-10 steps with er_sde + beta_57 and gives no CFG (2.5
# is used).  beta_57 is RES4LYF's, not in this ComfyUI: the core "beta" scheduler (alpha = beta = 0.6) is the nearest.
TUNED_MODELS = {
    "bedffa82a8da7547427aa6a0db28e50edf624146268e606d5a734e51a45500dc": {
        "label": "Batman MOSS v3.0", "names": ("batmanmoss_v30",),
        "sampling": {"steps": 8, "cfg": 2.5, "sampler": "er_sde", "scheduler": "beta"}},
}


def tuned_model(model: dict | None) -> dict | None:
    """The TUNED_MODELS entry of a registered MOSS-Image model, if its DiT is one of them."""
    if (model or {}).get("family") != "moss_image":
        return None
    p = (model.get("files") or {}).get("ckpt")
    if not p:
        return None
    from .models import hasher
    h = (hasher.get(p) or "").lower()   # never waits: None until the background hash is done
    stem = Path(p).stem.lower()
    return TUNED_MODELS.get(h) or next((t for t in TUNED_MODELS.values() if stem in t["names"]), None)


def tuned_note(model: dict | None) -> str | None:
    t = tuned_model(model)
    if not t:
        return None
    s = t["sampling"]
    return (f"{t['label']} 用の生成設定（{s['steps']} ステップ・CFG {s['cfg']}・{s['sampler']} / {s['scheduler']}）を初期値にしています。"
            "高速化 LoRA が組み込まれたモデルなので、高速生成の LoRA は重ねません。作者が勧めるスケジューラー beta_57 は"
            "この ComfyUI に無いため、近い beta を使います。この設定で生成できるかは確認していません。")


def fast_lora(model: dict | None, choice: str = "on") -> str | None:
    """The fast-generation LoRA of MOSS-Image SFT for a choice ("on": 8 steps, "4": 4 steps), when it is registered and
    its file is there (they do not fit Base, and a fine-tune with a turbo LoRA merged in gets none)."""
    p = ((model or {}).get("files") or {}).get(FAST_SLOTS.get(choice, ""))
    if not p or (model or {}).get("variant") != "moss_image_sft" or tuned_model(model):
        return None
    return p if Path(p).is_file() else None


def fast_choices(model: dict | None) -> list[str]:
    return [c for c in FAST_SLOTS if fast_lora(model, c)]


def default_sampling(model: dict | None, fast: str | None = None) -> dict:
    """fast: "on" (8 steps) / "4" (4 steps) / "off" from the form (unset: the first one there).  The "fast" key (and
    "fast_choices") is there only when a fast LoRA can be used."""
    fam = (model or {}).get("family", "sdxl")
    if fam == "qwen_image21" and gen_slot(model) == "gen" and is_turbo_gguf(model["files"].get("gen")):
        fam = "qwen_image21_turbo"
    elif fam == "zimage" and (model or {}).get("variant") == "zimage_base":
        fam = "zimage_base"
    elif fam == "anima" and (model or {}).get("variant") == "anima29":
        fam = "anima29"
    elif fam == "krea2" and krea2_raw(model, gen_slot(model)):
        fam = "krea2_raw"
    s = dict(SAMPLING.get(fam, SAMPLING["sdxl"]))
    tuned = tuned_model(model)
    if tuned:
        s.update(tuned["sampling"])
    choices = fast_choices(model)
    if choices:
        # unset (or a choice whose file is gone): the 8-step LoRA when it is there, else the 4-step one
        s["fast"] = fast if fast == "off" or fast in choices else choices[0]
        s["fast_choices"] = choices
        if s["fast"] != "off":
            s.update(SAMPLING[FAST_SAMPLING[s["fast"]]])
    s["clip_skip"] = 2 if (model or {}).get("variant") == "pony" else 1
    return s


MAX_SEED = 2**64 - 1   # ComfyUI's KSampler seed range


def check_seed(seed, count: int = 1) -> int:
    """A seed typed by the user; consecutive images use seed, seed+1, ... so the last one must fit too."""
    try:
        s = int(seed)
    except (TypeError, ValueError):
        raise ValueError("シードは整数で入れてください") from None
    if s < 0 or s + max(count, 1) - 1 > MAX_SEED:
        raise ValueError(f"シードは 0 から {MAX_SEED - max(count, 1) + 1} までの整数で入れてください")
    return s


# ======================================================================== job
class GenControl:
    """Pause / stop flags of running generation batches, keyed by (project id, batch id)."""

    def __init__(self):
        self.flags: dict[tuple[str, int], str] = {}

    def request(self, pid: str, bid: int, what: str):
        self.flags[(pid, bid)] = what
        if what == "stop":
            comfy.cancel_current()

    def get(self, pid: str, bid: int) -> str | None:
        return self.flags.get((pid, bid))

    def clear(self, pid: str, bid: int):
        self.flags.pop((pid, bid), None)


control = GenControl()


def batch_dir(project, bid: int) -> Path:
    return project.path / "generated" / f"{bid:03d}"


def run_batch(project, bid: int):
    """Generate the remaining images of a batch.  Called from the serial GPU queue."""
    row = project.q("SELECT * FROM gen_batches WHERE id=?", (bid,))[0]
    spec = json.loads(row["spec"])
    flag = control.get(project.id, bid)
    if flag in ("stop", "pause"):
        control.clear(project.id, bid)
        if flag == "stop":
            _set_status(project, bid, "cancelled", "開始前に中止しました")
        else:
            _set_status(project, bid, "paused", "一時停止中（開始前）")
        return
    done = {r["idx"] for r in project.q("SELECT idx FROM gen_images WHERE batch_id=?", (bid,))}
    todo = [i for i in range(len(spec["shots"])) if i not in done]
    if not todo:
        _set_status(project, bid, "done", "完了")
        return
    model = registry.get(spec["gen_model_id"])
    if not model or model["missing"]:
        _set_status(project, bid, "failed", "生成モデルのファイルが見つかりません")
        return
    _set_status(project, bid, "running", "準備中（モデルを読み込んでいます）")
    t0, n_done = time.time(), 0
    try:
        out_dir = batch_dir(project, bid)
        out_dir.mkdir(parents=True, exist_ok=True)
        staged = stage_model_files(model)
        sampling = spec["sampling"]
        if sampling.get("fast") in FAST_SLOTS and FAST_SLOTS[sampling["fast"]] not in staged:
            # the fast LoRA was removed after this batch was made: its few steps / CFG 1 need it
            sampling = {**sampling, **SAMPLING.get(model["family"], {}), "fast": "off"}
        loras = [(_stage("loras", lo["path"]), lo.get("strength", 1.0)) for lo in spec["loras"]]
        comfy.ensure_started()
        log0 = comfy.log_size()
        for idx in todo:
            flag = control.get(project.id, bid)
            if flag in ("pause", "stop"):
                break
            shot = spec["shots"][idx]
            seed = spec["seed"] + idx
            if shot.get("subject"):   # style material
                prompt = compose_style_prompt(spec["positive"], shot, spec["prompt_style"], model.get("variant"))
                w, h = style_size(model, shot)
                negative = compose_style_negative(spec["negative"], shot, spec["prompt_style"], model.get("variant"))
            else:
                prompt = compose_prompt(spec["positive"], shot, spec["prompt_style"], model.get("variant"))
                w, h = FAMILY_SIZE.get(model["family"], SIZE)[shot["framing"]]
                negative = compose_negative(spec["negative"], shot, spec["prompt_style"], model.get("variant"),
                                            spec["positive"])
            init = None
            if spec.get("init_images"):
                src = source_file(gen_source_dir(project), spec["init_images"][idx % len(spec["init_images"])])
                if not src:
                    raise RuntimeError("元画像が見つかりません（消された可能性があります）")
                init = {"image": stage_comfy_input(src), "denoise": spec["denoise"]}
            g = build_graph(model, staged, loras, prompt, negative, seed, w, h, sampling,
                            f"lsgen_{project.id[:20]}_b{bid}", init=init)
            try:
                imgs = comfy.run_prompt(g)
            except RuntimeError:
                if control.get(project.id, bid) == "stop":
                    break
                raise
            src = config.DATA_DIR / "comfy_output" / imgs[0].get("subfolder", "") / imgs[0]["filename"]
            name = f"{idx:03d}_s{seed}.png"
            shutil.move(str(src), out_dir / name)
            project.q("INSERT INTO gen_images(batch_id, idx, file, seed, prompt, shot, picked, created) VALUES(?,?,?,?,?,?,1,?)",
                      (bid, idx, name, seed, prompt, json.dumps(shot), time.time()))
            n_done += 1
            if n_done == 1 and loras:
                warn = comfy.lora_warnings(log0)
                if warn:
                    raise RuntimeError(f"LoRA の一部を読み込めませんでした（{len(warn)} 件）。この生成モデルに合わない LoRA の可能性があります。")
            el = time.time() - t0
            left = len(todo) - n_done
            _progress(project, bid, len(done) + n_done, len(spec["shots"]),
                      int(el / n_done * left) if n_done >= 2 else None)
    except Exception as e:
        _set_status(project, bid, "failed", f"生成に失敗しました: {e}")
        comfy.after_failure(e)
        control.clear(project.id, bid)
        return
    comfy.park()  # free VRAM for analysis / training; the process stays for the next generation
    cons = ""
    if spec.get("kind", "character") == "character":   # a style set has different subjects on purpose
        try:   # still on the GPU queue: the check runs before anything else takes the GPU
            cons = consistency_message(check_consistency(project))
        except Exception as e:   # an aid only; the generated images are kept either way
            cons = f"顔立ちのそろい具合を確認できませんでした（{e}）"
        try:
            kept = prompt_check_message(check_prompt(project))
        except Exception as e:
            kept = f"プロンプトの指定どおりかを確認できませんでした（{e}）"
        cons = "。".join(x for x in (cons, kept) if x)
    flag = control.get(project.id, bid)
    control.clear(project.id, bid)
    total_done = project.q("SELECT COUNT(*) c FROM gen_images WHERE batch_id=?", (bid,))[0]["c"]
    if total_done:   # the model worked on this PC (検証済み in the model lists)
        registry.mark_checked(spec.get("gen_model_id"), "gen")
    if flag == "stop":
        _set_status(project, bid, "cancelled", f"中止しました（{total_done} 枚は残っています）")
    elif flag == "pause" and total_done < len(spec["shots"]):
        _set_status(project, bid, "paused", f"一時停止中（{total_done} / {len(spec['shots'])} 枚）")
    else:
        _set_status(project, bid, "done", "完了" + (f"。{cons}" if cons else ""))


# ======================================================================== consistency of the candidates
# A character defined only by a prompt drifts between images (face, hair details, outfit).  Training on the
# mix teaches an average and the character comes out weak, so candidates that differ from the rest are
# found before import: CCIP (the same character-identity model the material check uses) is computed for
# every candidate, the candidate most similar to all others is the centre, and a candidate farther than
# CCIP's same-character threshold from it is unpicked.  Picks the user set by hand are never changed.
SAME_CHAR = 0.178   # imgutils' published CCIP threshold, also used by the material identity check


def _with_face(rows):
    """Candidates whose face can be compared: a shot from directly behind shows no face, so CCIP would
    call it a different character."""
    return [r for r in rows if json.loads(r["shot"] or "{}").get("angle") != "back_full"]


def check_consistency(project) -> dict:
    """Compute missing CCIP features and each candidate's difference to the centre.  Runs on the GPU
    queue (after a batch) or from the button; returns a summary for the message."""
    from .analysis import worker
    rows = _with_face(project.q("SELECT id, batch_id, file, ccip, picked, pick_manual, shot FROM gen_images WHERE deleted=0"))
    if len(rows) < 3:
        return {"n": len(rows), "outliers": 0, "unpicked": 0}
    need = [r for r in rows if not r["ccip"]]
    if need:
        paths = [str(batch_dir(project, r["batch_id"]) / r["file"]) for r in need]
        res = worker.call("ccip_features", paths=paths, timeout=1800)
        if not res.get("ok"):
            raise RuntimeError(res.get("error") or "CCIP を計算できませんでした")
        with project.tx() as db:
            for r, f in zip(need, res["feats"]):
                db.execute("UPDATE gen_images SET ccip=? WHERE id=?", (json.dumps(f), r["id"]))
        rows = _with_face(project.q("SELECT id, batch_id, file, ccip, picked, pick_manual, shot FROM gen_images WHERE deleted=0"))
    feats = [json.loads(r["ccip"]) for r in rows]
    res = worker.call("ccip_diff", feats=feats, refs=feats, timeout=600)
    if not res.get("ok"):
        raise RuntimeError(res.get("error") or "CCIP を比較できませんでした")
    mat = res["matrix"]
    n = len(rows)
    centre = min(range(n), key=lambda a: sorted(mat[a])[n // 2])
    outliers = unpicked = 0
    with project.tx() as db:
        for a, r in enumerate(rows):
            d = 0.0 if a == centre else round(float(mat[a][centre]), 4)
            out = d > SAME_CHAR
            outliers += out
            if out and r["picked"] and not r["pick_manual"]:
                db.execute("UPDATE gen_images SET ccip_diff=?, picked=0 WHERE id=?", (d, r["id"]))
                unpicked += 1
            else:
                db.execute("UPDATE gen_images SET ccip_diff=? WHERE id=?", (d, r["id"]))
    project.set_meta(gen_centre=rows[centre]["id"])
    return {"n": n, "outliers": outliers, "unpicked": unpicked}


def check_prompt(project) -> dict:
    """Whether each character candidate shows what the user's prompt asks for (gender, age, hair, eyes, body,
    outfit; promptkeep).  The WD tagger reads the candidates not checked yet; a candidate that contradicts the
    prompt, or misses two asked tags, is unpicked unless the user picked it by hand.  Runs on the GPU queue."""
    from .analysis import worker
    rows = project.q("SELECT g.id, g.batch_id, g.file, g.shot, g.picked, g.pick_manual, g.prompt_check, b.spec "
                     "FROM gen_images g JOIN gen_batches b ON b.id = g.batch_id WHERE g.deleted=0")
    rows = [r for r in rows if not json.loads(r["shot"] or "{}").get("subject")]   # style subjects differ on purpose
    need = [r for r in rows if r["prompt_check"] is None]
    if not need:
        return {"n": 0, "mismatch": 0, "unpicked": 0}
    res = worker.call("tag_images", paths=[str(batch_dir(project, r["batch_id"]) / r["file"]) for r in need], timeout=1800)
    if not res.get("ok"):
        raise RuntimeError(res.get("error") or "タグを読み取れませんでした")
    mismatch = unpicked = 0
    with project.tx() as db:
        for r, probs in zip(need, res["tags"]):
            spec = json.loads(r["spec"] or "{}")
            req = promptkeep.requested(spec.get("positive", ""), spec.get("prompt_style", "tags"))
            out = promptkeep.check(req, probs, json.loads(r["shot"] or "{}"))
            bad = bool(out["conflicts"] or out["missing"])
            mismatch += bad
            if promptkeep.unpick(out) and r["picked"] and not r["pick_manual"]:
                db.execute("UPDATE gen_images SET prompt_check=?, picked=0 WHERE id=?", (json.dumps(out, ensure_ascii=False), r["id"]))
                unpicked += 1
            else:
                db.execute("UPDATE gen_images SET prompt_check=? WHERE id=?", (json.dumps(out, ensure_ascii=False), r["id"]))
    return {"n": len(need), "mismatch": mismatch, "unpicked": unpicked}


def prompt_check_message(s: dict) -> str:
    if not s.get("n"):
        return ""
    if not s["mismatch"]:
        return f"{s['n']} 枚ともプロンプトの指定（髪・服・年齢・性別・体型など）と食い違いは見つかりませんでした（タグ判定 AI による推定）"
    msg = f"{s['n']} 枚のうち {s['mismatch']} 枚は、プロンプトの指定と違うところがある可能性があります（タグ判定 AI による推定）"
    return msg + (f"。そのうち {s['unpicked']} 枚の選択を外しました" if s["unpicked"] else "")


def consistency_message(s: dict) -> str:
    if not s.get("n") or s["n"] < 3:
        return ""
    if not s["outliers"]:
        return f"{s['n']} 枚の顔立ちはそろっています（キャラ判定 AI による推定）"
    msg = f"{s['n']} 枚のうち {s['outliers']} 枚は、ほかと顔立ちが違う可能性があります（キャラ判定 AI による推定）"
    return msg + (f"。そのうち {s['unpicked']} 枚の選択を外しました" if s["unpicked"] else "")


def _set_status(project, bid, status, message):
    project.q("UPDATE gen_batches SET status=?, message=? WHERE id=?", (status, message, bid))
    bus.publish("gen", project=project.id, batch=bid, status=status, message=message)


def _progress(project, bid, done, total, eta):
    msg = f"生成中 {done} / {total} 枚"
    project.q("UPDATE gen_batches SET message=? WHERE id=?", (msg, bid))
    bus.publish("gen", project=project.id, batch=bid, status="running", message=msg, done=done, total=total, eta_s=eta)


def new_batch(project, recipe: dict, count: int) -> int:
    model = registry.get(recipe["gen_model_id"])
    style = prompt_style(model)
    over = {k: v for k, v in (recipe.get("sampling") or {}).items() if v not in (None, "")}
    sampling = {**default_sampling(model, over.get("fast")), **over}
    kind = project.meta("kind") or "character"
    if kind == "style":
        shots, shot_set = plan_style(count, recipe.get("exclude_groups") or []), STYLE_SET_VERSION
    else:
        shots, shot_set = plan_shots(count, recipe.get("exclude_angles") or [], recipe.get("exclude_framings") or [],
                                     recipe.get("exclude_exprs") or [], recipe.get("exclude_poses") or []), SHOT_SET_VERSION
    positive, added = loraconf.apply(recipe.get("positive") or "", recipe.get("loras"))   # trigger words of the LoRAs (LoRA 設定)
    spec = {"gen_model_id": model["id"], "gen_model_name": model["name"], "positive": positive, "trigger_words": added,
            "negative": (recipe.get("negative") or "").strip(), "loras": recipe.get("loras") or [],
            "sampling": sampling, "seed": check_seed(recipe.get("seed"), len(shots)) if recipe.get("seed") not in (None, "", 0) else random.randint(1, 2**31 - 1000),
            "prompt_style": style, "kind": kind, "shot_set": shot_set, "shots": shots}
    sources = [n for n in recipe.get("init_images") or [] if source_file(gen_source_dir(project), n)]
    if sources:   # image-to-image: the shots go through the source images in turn
        spec.update(init_images=sources, denoise=clamp_denoise(recipe.get("denoise"), GEN_DENOISE_DEFAULT))
    with project.tx() as db:
        return db.execute("INSERT INTO gen_batches(created, status, message, spec) VALUES(?,?,?,?)",
                          (time.time(), "queued", "順番待ち", json.dumps(spec, ensure_ascii=False))).lastrowid
