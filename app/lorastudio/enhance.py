"""Texture / enhancement LoRAs (kind "enhance"): a LoRA that strengthens one quality such as skin texture,
detail, line sharpness, shading, colour or film grain, independent of the subject.

Two methods, both run by sd-scripts (nothing is trained by this app itself):

* "pair"  - difference LoRA (the "copy machine" method).  Each material image gets a partner made by image
  processing (e.g. skin smoothed).  A first LoRA learns the "without" side until it reproduces it, is merged
  into the base model, and the real LoRA is trained on the "with" side on top of that merged model, so it
  learns only the difference between the two sides.  sd-scripts: sdxl_train_network.py,
  networks/sdxl_merge_lora.py.
* "words" - slider LoRA from prompts only (LECO, sd-scripts sdxl_train_leco.py, docs/train_leco.md).  No
  images; it can only strengthen what the base model already knows.

Both are SDXL-only for now (the LECO script exists for SD1.x/SDXL only; the pair method was set up for SDXL).
Settings and effect strengths are starting points, not verified values (docs/VERIFICATION.md).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

FAMILIES = ("sdxl",)
METHODS = {"pair": "画像の組から作る", "words": "言葉から作る"}

# op: image processing that makes the partner image.  direction "remove": the partner lacks the quality
# (the material is the "with" side); "add": the partner has more of it (the material is the "without" side).
# metric: how the effect is measured on generated images (assistive number only).
EFFECTS = {
    "skin": {"label": "肌の質感", "desc": "肌のきめ・毛穴など、実写のような肌の細かさを強めます。",
             "op": "smooth", "direction": "remove", "metric": "detail",
             "positive": "detailed skin texture, skin pores, realistic skin", "negative": "smooth skin, airbrushed skin, plastic skin",
             "subject": "person, portrait",
             "eval": ["close-up portrait of a woman, face, natural light", "portrait of an old man, face, wrinkles, window light",
                      "upper body of a young man, outdoors, daylight", "close-up of a woman's face and shoulders, studio lighting"]},
    "detail": {"label": "細部の描き込み", "desc": "髪の毛先・布の織り目・小物など、細部の描き込みを増やします。",
               "op": "soften", "direction": "remove", "metric": "detail",
               "positive": "highly detailed, intricate details", "negative": "simple, flat, low detail", "subject": "",
               "eval": ["a woman in an ornate dress standing in a library", "a knight in armor in a forest",
                        "a busy market street with many stalls", "a cat sitting on a wooden table, still life"]},
    "sharp": {"label": "線・輪郭のくっきりさ", "desc": "線や輪郭をくっきりさせます。",
              "op": "blur", "direction": "remove", "metric": "detail",
              "positive": "sharp focus, crisp lineart, clear outlines", "negative": "blurry, soft focus, fuzzy lines", "subject": "",
              "eval": ["a girl standing in a city street, full body", "a red sports car parked on a road",
                       "a bouquet of flowers in a vase", "a castle on a hill under a blue sky"]},
    "contrast": {"label": "陰影・立体感", "desc": "影と光の差を強め、立体感を出します。",
                 "op": "flatten", "direction": "remove", "metric": "contrast",
                 "positive": "dramatic lighting, strong shadows, high contrast", "negative": "flat lighting, low contrast, no shadows",
                 "subject": "",
                 "eval": ["portrait of a woman, side lighting", "a man sitting on a chair in a room",
                          "a stone statue in a park", "a bowl of fruit on a table"]},
    "color": {"label": "色の鮮やかさ", "desc": "色を鮮やかにします。",
              "op": "desaturate", "direction": "remove", "metric": "saturation",
              "positive": "vivid colors, saturated colors, colorful", "negative": "muted colors, desaturated, dull colors", "subject": "",
              "eval": ["a girl in a flower field", "a city street at sunset", "a tropical beach with palm trees",
                       "a plate of fruit on a table"]},
    "grain": {"label": "フィルム写真の粒子", "desc": "フィルム写真のような粒子感を足します。",
              "op": "grain", "direction": "add", "metric": "detail",
              "positive": "film grain, analog photo, grainy", "negative": "clean digital photo, smooth, noiseless", "subject": "photo",
              "eval": ["photo of a woman walking on a street", "photo of a mountain landscape",
                       "photo of a cafe interior", "photo of a man reading a book"]},
    "custom": {"label": "自分で決める（言葉から作るときだけ）", "desc": "強めたいものと弱めたいものを英語のプロンプトで入れます。",
               "op": None, "direction": None, "metric": None, "positive": "", "negative": "", "subject": "",
               "eval": ["portrait of a woman, upper body", "a man standing in a city street", "a landscape with mountains and a lake",
                        "a cat sitting on a windowsill"]},
}


def effect(key: str) -> dict:
    return EFFECTS.get(key) or EFFECTS["detail"]


def clean(meta_enh: dict | None) -> dict:
    """Normalised settings stored in the project meta ("enhance")."""
    e = dict(meta_enh or {})
    method = e.get("method") if e.get("method") in METHODS else "pair"
    key = e.get("effect") if e.get("effect") in EFFECTS else ("detail" if method == "pair" else "custom")
    if method == "pair" and key == "custom":
        key = "detail"
    ef = EFFECTS[key]
    norm = lambda s: ", ".join(x.strip() for x in str(s or "").replace("\n", ",").split(",") if x.strip())
    return {"method": method, "effect": key,
            "positive": norm(e["positive"]) if e.get("positive") is not None else ef["positive"],
            "negative": norm(e["negative"]) if e.get("negative") is not None else ef["negative"],
            "subject": norm(e["subject"]) if e.get("subject") is not None else ef["subject"]}


# ======================================================================== pair images
def make_partner(src: Path, dest: Path, op: str) -> None:
    """Write the processed partner of one material image (same size, PNG)."""
    with Image.open(src) as im:
        im.load()
        im = ImageOps.exif_transpose(im).convert("RGB")
    out = process(im, op)
    tmp = dest.with_name("_tmp_" + dest.name)
    out.save(tmp, "PNG")
    tmp.replace(dest)


def process(im: Image.Image, op: str) -> Image.Image:
    long = max(im.size)
    if op == "smooth":
        # edge-preserving smoothing: blur flat areas (skin) and keep edges (eyes, hair outlines), like an
        # "airbrushed" photo.  The difference to the original is then mostly the fine texture.
        blurred = im.filter(ImageFilter.GaussianBlur(max(2.0, long / 380)))
        # edges are found on a pre-blurred copy, so the fine texture itself does not count as an edge
        g = np.asarray(im.convert("L").filter(ImageFilter.GaussianBlur(max(1.5, long / 500))), dtype=np.float32)
        gx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
        gy = np.abs(np.diff(g, axis=0, prepend=g[:1, :]))
        edge = Image.fromarray(np.clip((gx + gy - 6) * 12, 0, 255).astype(np.uint8))
        r = max(3, int(long / 300)) | 1
        edge = edge.filter(ImageFilter.MaxFilter(r)).filter(ImageFilter.GaussianBlur(r / 2))
        return Image.composite(im, blurred, edge)
    if op == "soften":
        w, h = im.size
        small = im.resize((max(1, w // 2), max(1, h // 2)), Image.LANCZOS)
        return small.resize((w, h), Image.BICUBIC)
    if op == "blur":
        return im.filter(ImageFilter.GaussianBlur(max(1.0, long / 900)))
    if op == "flatten":
        return ImageEnhance.Contrast(im).enhance(0.7)
    if op == "desaturate":
        return ImageEnhance.Color(im).enhance(0.55)
    if op == "grain":
        a = np.asarray(im, dtype=np.float32)
        import hashlib
        rng = np.random.default_rng(int.from_bytes(hashlib.md5(im.tobytes()[:65536]).digest()[:4], "little"))
        noise = rng.normal(0, 9.0, a.shape[:2]).astype(np.float32)
        noise = np.asarray(Image.fromarray(np.clip(noise + 128, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6)),
                           dtype=np.float32) - 128
        return Image.fromarray(np.clip(a + noise[..., None], 0, 255).astype(np.uint8))
    raise ValueError(f"unknown op {op}")


# ======================================================================== words (LECO)
def leco_prompts(enh: dict, variant: str, guidance: float, resolution: int = 768) -> dict:
    """Slider-target TOML for sd-scripts LECO (docs/train_leco.md 2.2 / 2.3).  Several neutral prompts help
    the effect generalise beyond one subject."""
    from .evaluate import QUALITY
    q = QUALITY.get(variant, ("", ""))[0].strip().rstrip(",")
    subject = enh.get("subject") or ""
    neutrals = ["", subject, "1girl, solo, upper body", "1boy, solo, portrait", "scenery, no humans", "still life, indoors"]
    neutrals = list(dict.fromkeys(n for n in neutrals))
    target = {"target_class": subject, "multiplier": 1.0, "weight": 1.0}
    if enh.get("positive"):
        target["positive"] = ", ".join(x for x in (subject, enh["positive"]) if x)
    if enh.get("negative"):
        target["negative"] = ", ".join(x for x in (subject, enh["negative"]) if x)
    if q:
        neutrals = [", ".join(x for x in (q, n) if x) for n in neutrals]
    return {"guidance_scale": float(guidance), "resolution": int(resolution), "dynamic_resolution": True, "dynamic_crops": True,
            "batch_size": 1, "neutrals": neutrals, "targets": [target]}


# ======================================================================== evaluation
def eval_prompts(enh: dict, variant: str) -> list[str]:
    from .evaluate import QUALITY, caption_style
    q = QUALITY.get(variant, ("", ""))[0]
    ps = list(effect(enh.get("effect")).get("eval"))
    if enh.get("method") == "words" and enh.get("subject") and enh.get("effect") == "custom":
        ps = [f"{enh['subject']}, {p}" for p in ps]
    if caption_style(variant) == "tags":
        ps = [p.replace("(", r"\(").replace(")", r"\)") for p in ps]
    return [q + p for p in ps]


def eval_strengths(enh: dict) -> list[float]:
    # sliders are made to be used in both directions
    return [-1.0, 1.0, 2.0] if enh.get("method") == "words" else [0.6, 1.0]


def measure(path: Path, metric: str) -> float:
    """Assistive measure of the effect on one generated image (not a quality score):
    detail     - mean absolute Laplacian of the luminance (fine texture / sharpness / grain)
    contrast   - standard deviation of the luminance
    saturation - mean saturation (HSV)"""
    with Image.open(path) as im:
        im = im.convert("RGB")
        if max(im.size) > 1024:
            im.thumbnail((1024, 1024))
        if metric == "saturation":
            return float(np.asarray(im.convert("HSV"), dtype=np.float32)[..., 1].mean() / 255)
        g = np.asarray(im.convert("L"), dtype=np.float32)
    if metric == "contrast":
        return float(g.std() / 255)
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(np.abs(lap).mean() / 255)


METRIC_LABEL = {"detail": "細かさ", "contrast": "明暗の差", "saturation": "彩度"}
