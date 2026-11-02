"""Masked training: the loss is counted mainly on the part of each image that matters (after
Fizgig v8.9.6's masked training).

The engine does the work: sd-scripts multiplies the training loss by the alpha channel of the training image when
the dataset subset has alpha_mask = true (library/custom_train_functions.py apply_masked_loss), also with cached
latents.  This module only decides which part is learnt, makes the masks with the analysis worker, and writes them
into the alpha channel of the run's frozen images (training.materialize).

* What is learnt follows the kind: the main person for character LoRAs and the person-related concept targets,
  everything but the people for place LoRAs.  Objects, styles and texture LoRAs are not offered: the whole picture
  matters there, or no segmenter here finds the object.
* Outside the mask still counts a little (floor, default 0.1 as in Fizgig): with 0 the background would be learnt
  from nothing, and the LoRA could draw anything there.
* Only the sd-scripts families (SDXL, Anima).  musubi-tuner has no masked loss.
* Masks are cached by image content and kind of mask, so the same image is never segmented twice.  The main person
  is the largest one inside the image's crop, so a person mask belongs to its crop too (another crop: made again).
  An image whose mask could not be made (no person found, failure) trains without a mask; the train tab says how
  many.
* Character images of a multi-character project are masked when that project trains with masks, even when the
  member project itself does not (analysis.Analyzer.mask_mode).
"""
from __future__ import annotations

from pathlib import Path

from . import captions, config

MASK_VERSION = "m1"          # bump when make_mask in the worker changes
FAMILIES = ("sdxl", "anima")
DEFAULT_FLOOR = 0.1
MODE_LABEL = {"person": "人物", "background": "背景"}


def mode_for(meta: dict) -> str | None:
    """What part of the image is learnt for this project's kind (None: masked training not offered)."""
    kind = (meta or {}).get("kind")
    if kind == "character":
        return "person"
    if kind == "concept":
        return {"place": "background", "object": None}.get(captions.concept_target(meta), "person")
    return None


def unsupported_reason(meta: dict, family: str | None) -> str | None:
    if not mode_for(meta):
        return "この種類の LoRA では、画像全体が学習の対象なので使いません。"
    if family and family not in FAMILIES:
        return "マスク学習は SDXL 系と Anima のモデルで使えます。このモデルの学習エンジンにはマスク学習がありません。"
    return None


def setting(meta: dict) -> dict:
    s = (meta or {}).get("mask") or {}
    try:
        floor = min(1.0, max(0.0, float(s.get("floor", DEFAULT_FLOOR))))
    except (TypeError, ValueError):
        floor = DEFAULT_FLOOR
    return {"on": bool(s.get("on")), "floor": floor}


def active(meta: dict, family: str | None) -> str | None:
    """The mask mode a run of this project uses now (None: trains without masks)."""
    if not setting(meta)["on"] or unsupported_reason(meta, family):
        return None
    return mode_for(meta)


def crop_of(img: dict, mode: str):
    """The crop a mask of this mode depends on (the background mode masks every person: not on the crop)."""
    return (img.get("crop") or None) if mode == "person" else None


def crop_tag(crop) -> str:
    import hashlib
    return "" if not crop else "_c" + hashlib.sha1(repr([int(v) for v in crop]).encode()).hexdigest()[:10]


def mask_file(sha: str, mode: str, crop=None) -> Path:
    return config.ANALYSIS_CACHE_DIR / "masks" / f"{MASK_VERSION}_{mode}_{sha[:24]}{crop_tag(crop)}.png"


def image_mask_file(img: dict, mode: str) -> Path:
    return mask_file(img["sha"], mode, crop_of(img, mode))


def state(img: dict, mode: str) -> dict:
    """{"status": done / none / failed / pending / running / missing, ...facts} of an image's mask."""
    st = dict(((img.get("mask") or {}).get(mode)) or {})
    if st.get("version") != MASK_VERSION:
        st = {}
    if st.get("status") in ("done", "none", "failed") and (st.get("crop") or None) != crop_of(img, mode):
        st = {}   # made for another crop (or before masks followed the crop): made again
    if st.get("status") == "done" and not image_mask_file(img, mode).exists():
        st = {}
    st.setdefault("status", "missing")
    return st


def usable(img: dict, mode: str) -> Path | None:
    """The mask file a run uses for this image, or None (no mask: the image trains as it is)."""
    if img.get("mask_off"):
        return None
    f = image_mask_file(img, mode)
    return f if state(img, mode)["status"] == "done" and f.exists() else None


def summary(images: list[dict], mode: str) -> dict:
    out = {"done": 0, "none": 0, "failed": 0, "waiting": 0, "off": 0, "small": 0}
    for im in images:
        if im.get("mask_off"):
            out["off"] += 1
            continue
        st = state(im, mode)
        k = st["status"] if st["status"] in ("done", "none", "failed") else "waiting"
        out[k] += 1
        if k == "done" and mode == "person" and (st.get("coverage") or 0) < 0.03:
            out["small"] += 1
    return out


def on_white(im):
    """RGB picture with its transparent parts on white, as the analysis sees it (the engines drop the alpha
    channel, which leaves the colour stored under it: usually black)."""
    from PIL import Image
    if im.mode == "P" or "transparency" in im.info:
        im = im.convert("RGBA")
    if im.mode in ("RGBA", "LA", "PA", "RGBa", "La"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.getchannel("A"))
        return bg
    return im.convert("RGB")


def apply(src_img, mask_path: Path, crop, floor: float):
    """RGBA image for training: the (cropped) picture with the mask, lifted to `floor`, as alpha."""
    from PIL import Image
    with Image.open(mask_path) as m:
        m.load()
        m = m.convert("L")
    if m.size != src_img.size:
        m = m.resize(src_img.size, Image.BILINEAR)
    rgb = on_white(src_img)
    if crop:
        box = tuple(int(v) for v in crop)
        rgb, m = rgb.crop(box), m.crop(box)
    lo = int(round(floor * 255))
    m = m.point(lambda v: lo + (255 - lo) * v // 255)
    rgb.putalpha(m)
    return rgb
