"""Keep what the user's prompt says about the character in generated material: pose,
expression and camera were followed, but hair, outfit, age, gender and body were changed.

Two parts:
- before generating: the opposite of the asked gender / age / hair length / breast size goes to the negative of
  tag-style models (``extra_negative``).  generate.compose_prompt also moves the user's tags in front of the shot
  tags (only the framing tag stays before them).
- after generating: the WD tagger (analysis worker ``tag_images``, the tagger the material check uses) reads each
  candidate, and ``check`` lists what contradicts the prompt (gender, age, hair length / colour, eye colour, breast
  size) and the asked tags it does not see.  The tagger's reading is an estimate, so the result only unpicks
  candidates (never deletes) and the user can pick them again.

Only tags the tagger knows are checked: the vocabulary is the i2t tag dictionary (engines/i2t/assets/tags_ja.json,
the general tags of the WD v3 taggers).  Names, artists, quality words and free text are never "missing".
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

from .config import ENGINES_DIR

HAIR_LENGTH = ["very short hair", "short hair", "medium hair", "long hair", "very long hair", "absurdly long hair"]
BREASTS = ["flat chest", "small breasts", "medium breasts", "large breasts", "huge breasts", "gigantic breasts"]
COLOURS = ["black", "brown", "light brown", "blonde", "orange", "red", "pink", "purple", "blue", "light blue", "aqua",
           "green", "white", "grey", "silver"]
# colours a tagger often swaps for each other (not a contradiction)
_NEAR = [{"white", "grey", "silver"}, {"blue", "light blue", "aqua"}, {"blonde", "light brown", "orange"},
         {"brown", "light brown"}, {"red", "orange"}, {"pink", "purple"}]
ADULT = {"mature female", "mature male", "old woman", "old man", "old", "aged up", "milf"}
YOUNG = {"child", "loli", "shota", "aged down"}
_ADULT_TEXT = re.compile(r"\b(adult|mature|middle-aged|elderly|old (woman|man|lady)|grandmother|grandfather|"
                         r"(2|3|4|5|6|7|8)0s|(1[89]|[2-9]\d)[- ]years?[- ]old)\b")
_YOUNG_TEXT = re.compile(r"\b(child|kid|little (girl|boy)|toddler|loli|shota)\b")
_FEMALE_TEXT = re.compile(r"\b(1girl|girl|woman|female|lady)\b")
_MALE_TEXT = re.compile(r"\b(1boy|boy|man|male|guy)\b")
# clothes that a close-up or an upper-body shot does not show (not "missing" there)
_LOWER = ("skirt", "shorts", "pants", "trousers", "jeans", "shoes", "boots", "sandals", "socks", "thighhighs",
          "kneehighs", "legwear", "pantyhose", "leggings", "stockings", "heels", "footwear", "barefoot", "garter")
SHOWS_LOWER = {"cowboy": ("skirt", "shorts", "pants", "trousers", "jeans", "garter"), "full": _LOWER}


@lru_cache(maxsize=1)
def vocabulary() -> frozenset[str]:
    try:
        return frozenset(json.loads((ENGINES_DIR / "i2t" / "assets" / "tags_ja.json").read_text("utf-8")))
    except (OSError, ValueError):
        return frozenset()


def _bare(tag: str) -> str:
    """"(Long_Hair:1.2)" -> "long hair"."""
    t = tag.strip().strip("()[]{} ")
    t = re.sub(r":[\d.]+$", "", t).strip()
    return re.sub(r"\s+", " ", t.replace("\\", "").replace("_", " ")).lower()


def _tags(positive: str, caption_style: str) -> list[str]:
    """The prompt's tags, or for natural-language prompts the known tag phrases found in the text."""
    low = (positive or "").lower().replace("_", " ")
    if caption_style == "tags":
        return [b for b in (_bare(t) for t in low.split(",")) if b]
    voc = vocabulary()
    found = []
    for size in (3, 2):   # "long blonde hair" is not a tag, "blonde hair" is
        for m in re.finditer(r"(?=\b(" + r"\s".join([r"[a-z0-9-]+"] * size) + r")\b)", low):
            if m.group(1) in voc and not any(m.group(1) in f for f in found):
                found.append(m.group(1))
    # sentences put several words before "hair" / "eyes" ("short white hair", "long wavy blonde hair")
    lengths = sorted((s.rsplit(" ", 1)[0] for s in HAIR_LENGTH), key=len, reverse=True)
    for m in re.finditer(r"\b((?:[a-z-]+ ){1,3})(hair|eyes)\b", low):
        words = m.group(1).strip()
        if m.group(2) == "hair":
            hit = next((x for x in lengths if re.search(rf"\b{x}\b", words)), None)
            if hit and f"{hit} hair" not in found:
                found.append(f"{hit} hair")
        for c in sorted(COLOURS, key=len, reverse=True):
            if re.search(rf"\b{c}\b", words):
                if f"{c} {m.group(2)}" not in found:
                    found.append(f"{c} {m.group(2)}")
                break
    return found


def requested(positive: str, caption_style: str) -> dict:
    """What the prompt asks about the character."""
    tags = _tags(positive, caption_style)
    low = (positive or "").lower().replace("_", " ")
    have = set(tags)
    out: dict = {"tags": [t for t in tags if t in vocabulary()]}
    female = bool(_FEMALE_TEXT.search(low)) or bool(have & {"1girl", "2girls", "multiple girls", "mature female", "old woman"})
    male = bool(_MALE_TEXT.search(low)) or bool(have & {"1boy", "2boys", "multiple boys", "mature male", "old man"})
    if female != male:
        out["gender"] = "female" if female else "male"
    adult = bool(have & ADULT) or bool(_ADULT_TEXT.search(low))
    young = bool(have & YOUNG) or bool(_YOUNG_TEXT.search(low))
    if adult != young:
        out["age"] = "adult" if adult else "young"
    for key, scale in (("hair_length", HAIR_LENGTH), ("breasts", BREASTS)):
        hit = [i for i, s in enumerate(scale) if s in have]
        if len(hit) == 1:
            out[key] = hit[0]
    for key, word in (("hair_colour", "hair"), ("eye_colour", "eyes")):
        cols = {c for c in COLOURS if f"{c} {word}" in have}
        if cols:
            out[key] = sorted(cols)
    return out


def extra_negative(req: dict) -> list[str]:
    """The opposite of what is asked, for the negative of tag-style models."""
    neg = []
    if req.get("gender") == "female":
        neg += ["1boy", "male focus"]
    elif req.get("gender") == "male":
        neg += ["1girl"]
    if req.get("age") == "adult":
        neg += ["child", "loli", "shota"]
    elif req.get("age") == "young":
        neg += ["mature female", "mature male", "old"]
    for key, scale in (("hair_length", HAIR_LENGTH), ("breasts", BREASTS)):
        i = req.get(key)
        if i is not None:
            neg += [s for j, s in enumerate(scale) if abs(j - i) >= 2]
    return neg


def _near(a: str, b: str) -> bool:
    return a == b or any(a in g and b in g for g in _NEAR)


def check(req: dict, probs: dict[str, float], shot: dict) -> dict:
    """The tagger's reading of one candidate against the request.  probs: tag -> probability (underscores or
    spaces).  Returns {"conflicts": [Japanese reasons], "missing": [asked tags not seen]}."""
    p = {k.replace("_", " ").lower(): float(v) for k, v in (probs or {}).items()}
    conflicts: list[str] = []
    back = shot.get("angle") in ("back", "back_full")
    if req.get("gender") == "female" and p.get("1boy", 0) >= 0.5 and p.get("1girl", 0) < 0.3:
        conflicts.append("性別（男性に見える）")
    if req.get("gender") == "male" and p.get("1girl", 0) >= 0.5 and p.get("1boy", 0) < 0.3:
        conflicts.append("性別（女性に見える）")
    if req.get("age") == "adult" and max((p.get(t, 0) for t in YOUNG), default=0) >= 0.5:
        conflicts.append("年齢（子どもに見える）")
    if req.get("age") == "young" and max((p.get(t, 0) for t in ADULT), default=0) >= 0.5:
        conflicts.append("年齢（大人に見える）")
    for key, scale, label in (("hair_length", HAIR_LENGTH, "髪の長さ"), ("breasts", BREASTS, "胸の大きさ")):
        if req.get(key) is None or (key == "breasts" and shot.get("angle") == "back_full"):
            continue
        seen = max(range(len(scale)), key=lambda j: p.get(scale[j], 0))
        if p.get(scale[seen], 0) >= 0.5 and abs(seen - req[key]) >= 2:
            conflicts.append(f"{label}（{scale[seen]} に見える）")
    for key, word, label in (("hair_colour", "hair", "髪の色"), ("eye_colour", "eyes", "目の色")):
        asked = req.get(key)
        if not asked or (word == "eyes" and (back or shot.get("expr") == "laugh")):
            continue
        if any(p.get(f"{c} {word}", 0) >= 0.2 for c in COLOURS if any(_near(c, a) for a in asked)):
            continue
        other = [c for c in COLOURS if p.get(f"{c} {word}", 0) >= 0.6 and not any(_near(c, a) for a in asked)]
        if other:
            conflicts.append(f"{label}（{other[0]} に見える）")
    framing = shot.get("framing")
    shows = SHOWS_LOWER.get(framing, ())   # generate's framing keys: face / upper / cowboy / full
    checked = {s for s in HAIR_LENGTH + BREASTS} | ADULT | YOUNG | {"1girl", "1boy", "solo"}
    missing = []
    for t in req.get("tags", []):
        if t in checked or t.endswith((" hair", " eyes")) and t.split(" ")[0] in COLOURS:
            continue   # judged above as a contradiction, not as a missing tag
        if t.endswith(" eyes") or "eye" in t.split():
            if back:
                continue
        if any(w in t for w in _LOWER) and not any(w in t for w in shows):
            continue   # not in the frame of this shot
        if p.get(t, 0) < 0.05:
            missing.append(t)
    return {"conflicts": conflicts, "missing": missing}


def unpick(result: dict) -> bool:
    """A contradiction, or two asked tags not seen, unpicks the candidate (one missing tag is only shown: a
    small item is easily missed by the tagger)."""
    return bool(result.get("conflicts")) or len(result.get("missing") or []) >= 2


def reason(result: dict) -> str:
    parts = list(result.get("conflicts") or [])
    if result.get("missing"):
        parts.append("見当たらない指定: " + ", ".join(result["missing"][:4]) + (" など" if len(result["missing"]) > 4 else ""))
    return " / ".join(parts)
