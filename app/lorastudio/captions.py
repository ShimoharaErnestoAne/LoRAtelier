"""Caption policy.

Character LoRA: features that should be *bound to the trigger word* are removed from the
caption (if they were written out, the model would attribute them to those words instead of
the trigger).  Variable elements (pose, background, framing, and - in "outfit varies" mode -
clothing) are kept so that they stay controllable by prompt.  This is a policy, not a
guarantee that elements separate cleanly; the UI says so.

Style LoRA: content (subjects, objects, composition) is described, the style itself is not,
so the trigger takes over "how it is drawn".

Concept LoRA (body type / body parts, e.g. muscular): everything is described as usual - who the
person is, hair, clothes, pose, background - except the tags of the feature to learn, which the
user names (concept words).  The feature common to all images is then bound to the trigger.
"""
from __future__ import annotations

import hashlib
import json
import re

CAPTION_POLICY_VERSION = 3
TAG_THRESHOLD = 0.35
CHAR_TAG_THRESHOLD = 0.85

# physical / identity features -> bound to trigger in character mode
_FEATURE_PATTERNS = [
    r".* hair$", r"^hair (between eyes|intakes|over one eye|over eyes|flaps)$", r".*hair ornament$", r"^hairclip$",
    r"^ahoge$", r"^(blunt |swept |parted |asymmetrical |crossed )?bangs$", r"^sidelocks$", r"^twintails$", r"^ponytail$",
    r"^(side |high |low )?ponytail$", r".*braids?$", r".*hair bun$", r"^double bun$", r"^drill hair$", r"^twin drills$",
    r".* eyes$", r"^heterochromia$", r".*pupils$", r"^eyelashes$", r"^tsurime$", r"^tareme$", r"^jitome$",
    r".*(animal|cat|fox|dog|wolf|rabbit|horse|elf|pointy) ears$", r".* tail$", r"^horns?$", r".* horns?$", r"^wings$",
    r".* wings$", r"^halo$", r"^(dark|pale|tan|tanned|blue|green|grey|red|white) skin$", r"^freckles$", r"^mole.*$",
    r"^fang$", r"^fangs$", r"^skin fang$", r"^(flat|small|medium|large|huge|gigantic) breasts$", r"^facial mark$",
    r"^scar.*$", r"^tattoo$", r"^beard$", r"^mustache$", r"^facial hair$", r"^muscular.*$", r"^glasses$",
]
# clothing / accessories -> removed only when the outfit is part of the character
_CLOTHING_WORDS = {
    "dress", "shirt", "skirt", "jacket", "coat", "uniform", "gloves", "thighhighs", "socks", "kneehighs", "boots",
    "shoes", "hat", "cap", "ribbon", "bow", "bowtie", "necktie", "tie", "sleeves", "collar", "hood", "pants", "shorts",
    "kimono", "armor", "cape", "apron", "vest", "sweater", "hoodie", "bikini", "swimsuit", "leotard", "pantyhose",
    "legwear", "footwear", "headwear", "choker", "earrings", "jewelry", "necklace", "belt", "scarf", "serafuku",
    "blazer", "cardigan", "capelet", "headband", "hairband", "beret", "bodysuit", "corset", "frills", "lace",
    "neckerchief", "sailor collar", "pleated skirt", "miniskirt", "overalls", "jumpsuit", "robe", "cloak", "mask",
    "wristband", "bracelet", "armband", "gauntlets", "sash", "obi", "hakama", "garter straps", "suspenders",
    "raincoat", "parka", "poncho", "tunic", "blouse", "camisole", "pajamas", "tracksuit", "sneakers", "sandals",
}
_DROP_ALWAYS = {"general", "sensitive", "questionable", "explicit", "rating:general", "rating:sensitive"}
_STYLE_WORDS = re.compile(r"^(.*(style|art|artstyle|medium|parody)|sketch|lineart|watercolor \(medium\)|"
                          r"traditional media|official art|realistic|anime coloring|retro artstyle|monochrome|greyscale)$")

_feature_re = re.compile("|".join(f"(?:{p})" for p in _FEATURE_PATTERNS))


def norm_tag(t: str) -> str:
    return t.replace("_", " ").strip() if len(t) > 3 else t.strip()  # keep kaomoji like ^_^ intact


def is_feature(tag: str) -> bool:
    return bool(_feature_re.fullmatch(tag))


def is_clothing(tag: str) -> bool:
    if tag in _CLOTHING_WORDS:
        return True
    last = tag.split(" ")[-1]
    return last in _CLOTHING_WORDS or tag.endswith(" uniform") or tag.endswith(" outfit")


def concept_match(tag: str, words) -> bool:
    """Tag belongs to the feature being learned: "muscular" covers "muscular female", "abs" only "abs"."""
    t = tag.lower()
    for w in words or ():
        w = w.strip().lower().replace("_", " ")
        if w and (t == w or t.startswith(w + " ") or t.endswith(" " + w)):
            return True
    return False


# What a concept LoRA learns (project meta "concept_target", fixed at creation).  All targets share the method -
# the feature words are left out of the captions and everything else is described - and differ in what the
# VLM is told not to describe, which material checks apply and which comparison prompts are used.
# person: the material is expected to show people (person checks, "too many close-ups" etc.).
# keep_generic: tags that are usually too common to be the feature, but are the feature for this target.
CONCEPT_TARGETS = {
    "body": {"label": "体型・特徴", "short": "体型", "noun": "体の特徴", "trigger": "cpt_", "person": True,
             "example": "muscular, abs", "keep_generic": set(),
             "describe": "the people (gender, age, hair, face, clothing), their pose and action, expression, camera angle "
                         "and framing, objects, the background and the lighting",
             "avoid": "do not describe the body shape or build"},
    "outfit": {"label": "衣装", "short": "衣装", "noun": "衣装", "trigger": "out_", "person": True,
               "example": "sailor uniform, red neckerchief", "keep_generic": set(),
               "describe": "the people (gender, age, hair, face, body), their pose and action, expression, camera angle "
                           "and framing, objects, the background and the lighting",
               "avoid": "do not describe the outfit, the clothing or the accessories they wear"},
    "object": {"label": "物・小物", "short": "物", "noun": "物", "trigger": "obj_", "person": False,
               "example": "sword, ornate hilt", "keep_generic": set(),
               "describe": "where the object is and how it is placed, any people and what they are doing, other objects, "
                           "the camera angle and framing, the background and the lighting",
               "avoid": "do not describe the shape, colour, material or design of the object itself"},
    "place": {"label": "背景・場所", "short": "場所", "noun": "場所", "trigger": "plc_", "person": False,
              "example": "old library, bookshelves", "keep_generic": {"indoors", "outdoors"},
              "describe": "any people and what they are doing, the time of day, the weather, the lighting, "
                          "the camera angle and framing",
              "avoid": "do not describe the place itself (its buildings, rooms, furniture or landscape)"},
    "pose": {"label": "ポーズ・動作・構図", "short": "ポーズ", "noun": "ポーズ・構図", "trigger": "pos_", "person": True,
             "example": "jumping, arms up, from below",
             "keep_generic": {"standing", "full body", "upper body", "cowboy shot", "portrait", "looking at viewer"},
             "describe": "the people (gender, age, hair, face, clothing), objects, the background and the lighting",
             "avoid": "do not describe the pose, the action, the camera angle or the framing"},
    "expression": {"label": "表情", "short": "表情", "noun": "表情", "trigger": "exp_", "person": True,
                   "example": "smug, half-closed eyes",
                   "keep_generic": {"smile", "closed mouth", "open mouth", "teeth", "looking at viewer"},
                   "describe": "the people (gender, age, hair, clothing), their pose and action, camera angle and framing, "
                               "objects, the background and the lighting",
                   "avoid": "do not describe the facial expression or the mouth and eyes"},
}


def concept_target(meta_or_snap: dict) -> str:
    t = (meta_or_snap or {}).get("concept_target")
    return t if t in CONCEPT_TARGETS else "body"


def clean_concept_words(words) -> list[str]:
    if isinstance(words, str):
        words = words.replace("、", ",").split(",")
    out = []
    for w in words or []:
        w = " ".join(str(w).replace("_", " ").split()).lower()
        if w and w not in out:
            out.append(w)
    return out


def policy_key(kind: str, outfit_mode: str, style: str, trigger: str, concept=(), target: str = "body") -> str:
    key = [CAPTION_POLICY_VERSION, kind, outfit_mode, style, trigger]
    if kind == "concept":   # other kinds keep their keys (no needless re-captioning)
        key.append(sorted(clean_concept_words(concept)))
        if target != "body":   # body keeps the key it had before targets existed
            key.append(target)
    raw = json.dumps(key)
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def tags_from_ai(ai: dict) -> tuple[list[str], list[str]]:
    tags = (ai or {}).get("tags") or {}
    general = [norm_tag(t) for t, s in sorted(tags.get("general", {}).items(), key=lambda x: -x[1]) if s >= TAG_THRESHOLD]
    chars = [norm_tag(t) for t, s in tags.get("character", {}).items() if s >= CHAR_TAG_THRESHOLD]
    return general, chars


# "monochrome" alone is not used: the tagger also gives it to single-hue colour art ("blue theme" etc.)
MONO_TAGS = {"greyscale", "sepia"}
MONO_COLORFULNESS = 0.012   # mean channel spread of a grey image re-saved as RGB/JPEG stays below this


def is_monochrome(img: dict) -> bool:
    """Black-and-white (or sepia) image: near-zero colour spread, or tagged as such by the tagger."""
    f = img.get("fast") or {}
    if f.get("ok") and f.get("colorfulness") is not None and f["colorfulness"] < MONO_COLORFULNESS:
        return True
    general, _ = tags_from_ai(img.get("ai") or {})
    return bool(MONO_TAGS & set(general))


def mark_monochrome(text: str, kind: str, style: str) -> str:
    """Character LoRA: say explicitly that the image is black and white, so the lack of colour is
    attributed to these words instead of the trigger (otherwise the character tends to come out
    grey; a concept LoRA likewise).  Style LoRA: left alone, because black and white is part of the
    style being learned."""
    if kind == "style":
        return text
    if style == "tags":
        parts = [p.strip() for p in text.split(",")]
        add = [t for t in ("monochrome", "greyscale") if t not in parts]
        return ", ".join(parts[:1] + add + parts[1:]) if add else text
    return text if "monochrome" in text.lower() else f"{text} Monochrome, black and white image."


def tag_caption(ai: dict, kind: str, outfit_mode: str, trigger: str, concept=()) -> str:
    general, _chars = tags_from_ai(ai)   # character-name tags are always replaced by the trigger
    out = []
    for t in general:
        if t in _DROP_ALWAYS:
            continue
        if kind == "character":
            if is_feature(t):
                continue
            if outfit_mode == "include" and is_clothing(t):
                continue
        elif kind == "concept":
            if concept_match(t, concept):
                continue
        else:  # style: describe content, not the style
            if _STYLE_WORDS.match(t):
                continue
        out.append(t)
    head = f"{trigger} style" if kind == "style" else trigger
    return ", ".join([head] + out)


# How captions are made for natural-language models.  "ai": the local VLM writes each caption (detailed, slow).
# "gen": generated material gets its caption from the shot it was generated with, other images from the VLM.
# "tags": every caption is made from the tagger's tags at once (plainer).  Tag-style models always use the tags.
CAPTION_MODES = ("ai", "gen", "tags")


def project_style(meta: dict) -> str:
    """"tags" / "natural": how this project's captions are written.  Tag-style models that also read natural
    language (Anima, "natural_ok") write tags unless the project chose sentences ("ai" / "gen")."""
    from .models import VARIANTS, registry
    v = VARIANTS.get((registry.get(meta.get("model_id") or "") or {}).get("variant") or "") or {}
    style = v.get("caption", "tags")
    if style == "tags" and v.get("natural_ok") and meta.get("caption_mode") in ("ai", "gen"):
        return "natural"
    return style


def default_caption_mode(meta: dict) -> str:
    from .models import VARIANTS, registry
    v = VARIANTS.get((registry.get(meta.get("model_id") or "") or {}).get("variant") or "") or {}
    return "tags" if v.get("natural_ok") else "ai"


def tag_sentence(ai: dict, kind: str, outfit_mode: str, trigger: str, concept=()) -> str:
    """Natural-language model, made at once from the tags (same tag policy as tag_caption)."""
    tags = tag_caption(ai, kind, outfit_mode, trigger, concept).split(", ")[1:]
    body = ", ".join(tags)
    if kind == "style":
        return f"{trigger} style. {body}." if body else f"{trigger} style."
    return f"{trigger}, {body}." if body else f"{trigger}."


def clothing_tags(ai: dict) -> list[str]:
    general, _ = tags_from_ai(ai)
    return [t for t in general if t not in _DROP_ALWAYS and is_clothing(t)]


def nl_instruction(kind: str, outfit_mode: str, trigger: str, concept=(), target: str = "body") -> str:
    """Instruction for the local VLM captioner (Qwen3-VL)."""
    if kind == "character":
        base = (f"Write one concise English caption for this image for training an image model. "
                f"Refer to the main character only by the name \"{trigger}\" (never 'a girl', 'a man', etc.). "
                "Do not describe their hair, eyes, face, skin, body shape or other permanent physical features. ")
        if outfit_mode == "include":
            base += "Do not describe their usual outfit or accessories either. "
        else:
            base += "Describe what they are wearing in detail. "
        base += ("Describe the pose, action, expression, camera angle and framing, other people or objects, "
                 "the background and the lighting. Do not mention art style. Output only the caption.")
        return base
    if kind == "concept":
        t = CONCEPT_TARGETS.get(target) or CONCEPT_TARGETS["body"]
        words = ", ".join(clean_concept_words(concept)) or ("the body type" if target == "body" else "the feature")
        return ("Write one concise English caption for this image for training an image model. "
                f"Describe {t['describe']}. Do not mention these features at all: {words}; "
                f"{t['avoid']}. Output only the caption.")
    return ("Write one concise English caption for this image for training an image model. Describe only the content: "
            "the subjects, what they are doing, objects, composition, camera angle, background and lighting. "
            "Do not describe the art style, medium, line work, coloring, shading or rendering technique, "
            "and do not name any artist. Output only the caption.")


def finalize_nl(text: str, kind: str, trigger: str) -> str:
    text = " ".join((text or "").split())
    if kind in ("character", "concept"):
        return text if trigger in text else f"{trigger}, {text}"
    return f"{trigger} style. {text}"


def replace_trigger(caption: str, old: str, new: str) -> str:
    if not old:
        return caption
    return re.sub(rf"(?<![\w]){re.escape(old)}(?![\w])", new, caption)


# ---------------------------------------------------------------- bulk tag edits (comma separated captions)
def split_tags(caption: str) -> list[str]:
    return [t.strip() for t in caption.split(",") if t.strip()]


def add_tags(caption: str, tags: list[str], position: str = "end") -> str:
    cur = split_tags(caption)
    new = [t for t in tags if t not in cur]
    if position == "after_trigger" and cur:
        cur = cur[:1] + new + cur[1:]
    else:
        cur = cur + new
    return ", ".join(cur)


def remove_tags(caption: str, tags: list[str]) -> str:
    drop = {t.strip() for t in tags}
    return ", ".join(t for t in split_tags(caption) if t not in drop)


def replace_tag(caption: str, old: str, new: str) -> str:
    """Replace a whole tag (exact match); an empty `new` removes it, and a tag already present is not doubled."""
    old, new = old.strip(), new.strip()
    out = []
    for t in split_tags(caption):
        t = new if t == old else t
        if t and t not in out:
            out.append(t)
    return ", ".join(out)
