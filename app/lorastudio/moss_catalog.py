"""MOSS-Image character & style catalog (hide1/MOSS-Image-Character-Catalog).

The i2t worker fetches it (job "moss_catalog", only after the user agrees) and writes a small index
{characters: {key: [name, series, id]}, styles: [[name, prompt, id]]} to the i2t models folder; this module only
reads that file.  The catalog is 34,249 images of MOSS-Image SFT drawing names without a LoRA, not curated: a
character being listed means MOSS-Image may already draw it by name, never that it does.  Its data and images
are outside the catalog's MIT grant (its NOTICE.md), so nothing of it is bundled.
"""
from __future__ import annotations

import json
import re
from collections import Counter

from . import captions, i2tlink

FORMAT = 1   # i2t.MOSS_CATALOG_VERSION
IMAGES_REV = "f99f7a6d281e1710dea0975f147a596232d8fbbe"   # i2t.MOSS_CATALOG_IMAGES_REV
PAGE = "https://huggingface.co/spaces/hide1/MOSS-Image-Character-Catalog"
SIZE_MB = 7
SHARE = 0.4   # the character tag must be on this share of the analysed images (as the style check in judge)
_cache: dict = {"key": None, "data": None}


def path():
    return i2tlink.models_dir() / "moss_catalog" / "index.json"


def load() -> dict | None:
    p = path()
    try:
        st = p.stat()
    except OSError:
        return None
    key = (str(p), st.st_mtime_ns, st.st_size)
    if _cache["key"] != key:
        try:
            data = json.loads(p.read_text("utf-8"))
            ok = data.get("format") == FORMAT and isinstance(data.get("characters"), dict)
        except (OSError, ValueError, AttributeError):
            data, ok = None, False
        _cache.update(key=key, data=data if ok else None)
    return _cache["data"]


def name_key(name: str) -> str:
    """i2t.moss_name_key: case, "_" / " " and escaped brackets do not matter."""
    return re.sub(r"\s+", " ", str(name or "").replace("\\", "").replace("_", " ")).strip().lower()


def image_url(entry_id: str) -> str:
    return (f"https://huggingface.co/datasets/hide1/MOSS-Image-Character-Catalog/resolve/{IMAGES_REV}"
            f"/images/{entry_id[:2]}/{entry_id}.webp")


def character(name: str) -> dict | None:
    hit = ((load() or {}).get("characters") or {}).get(name_key(name))
    return {"name": hit[0], "series": hit[1], "image": image_url(hit[2])} if hit else None


def styles() -> list[dict]:
    """The 314 style entries (name + the prompt the catalog used), for tag completion on MOSS-Image models."""
    return [{"name": n, "prompt": pr, "image": image_url(i)} for n, pr, i in (load() or {}).get("styles") or []]


def character_hint(images: list[dict], results: dict[int, dict], meta: dict, family: str | None) -> dict | None:
    """Character LoRA on MOSS-Image: the character the tagger finds on most used images, and whether the catalog
    lists it.  Only informs; training is never held back.  None when nothing applies."""
    if meta.get("kind") != "character" or family != "moss_image":
        return None
    analysed = [im for im in images if results[im["id"]]["use"] and im.get("ai")]
    if len(analysed) < 3:
        return None
    chars = Counter()
    for im in analysed:
        chars.update(set(captions.tags_from_ai(im["ai"])[1]))
    if not chars:
        return None
    tag, k = chars.most_common(1)[0]
    if k / len(analysed) < SHARE:
        return None
    out = {"tag": tag, "count": k, "of": len(analysed)}
    if load() is None:
        return out | {"state": "missing", "size_mb": SIZE_MB, "page": PAGE}
    hit = character(tag)
    return out | ({"state": "listed", **hit} if hit else {"state": "unlisted"})
