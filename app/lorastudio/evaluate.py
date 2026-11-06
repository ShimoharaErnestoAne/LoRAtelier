"""Effect check: generate comparison images in ComfyUI (the user's target environment) with
everything fixed (base model, prompt, seed, size, sampler, steps) except LoRA on/off,
checkpoint and strength.

ComfyUI (GPL-3.0) is run as a separate headless process on 127.0.0.1 and driven through its
HTTP API; it is not linked into this app.  ComfyUI keeps the base model resident between
requests, so switching LoRA checkpoint/strength does not reload the base model.
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import socket
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from . import config
from .events import bus
from .gpu import free_ram_gb, gpu

COMFY_MODELS = config.DATA_DIR / "comfy_models"
PARK_RAM_FREE_GB = 8.0   # a parked ComfyUI keeps its models in system memory only while at least this much stays free
PARK_RAM_CHECK_S = 3.0   # time for ComfyUI to move the models out of VRAM before free memory is looked at

# Prompt sets.  {T} = trigger.  Characters: different framing/pose/background than typical
# material.  Styles: subjects deliberately unlike typical style material.
PROMPTS = {
    ("character", "tags"): [
        "{T}, solo, standing, full body, city street, daytime",
        "{T}, solo, sitting, upper body, cafe, smiling, looking at viewer",
        "{T}, solo, running, from side, forest, outdoors",
        "{T}, solo, close-up, portrait, night, neon lights",
    ],
    ("character", "natural"): [
        "{T} standing on a busy city street in daylight, full body shot",
        "{T} sitting in a cafe, upper body, smiling at the viewer",
        "{T} running through a forest, seen from the side",
        "close-up portrait of {T} at night under neon lights",
    ],
    # people of both sexes, young and old, so a style that only fits one kind of face is visible
    ("style", "tags"): [
        "{T} style, 1girl, solo, young woman, smiling, upper body, city street",
        "{T} style, 1boy, solo, child, running, park, outdoors",
        "{T} style, 1boy, solo, old man, beard, reading newspaper, park bench",
        "{T} style, 1girl, solo, old woman, grey hair, knitting, sitting, living room",
        "{T} style, no humans, cat, sitting on windowsill, indoors",
        "{T} style, no humans, landscape, mountain, lake, scenery",
    ],
    ("style", "natural"): [
        "{T} style. A smiling young woman on a city street, upper body",
        "{T} style. A little boy running in a park",
        "{T} style. A bearded old man reading a newspaper on a park bench",
        "{T} style. A grey-haired old woman knitting in a living room",
        "{T} style. A cat sitting on a windowsill indoors",
        "{T} style. A landscape with mountains and a lake",
    ],
    # concept (body type / body parts): different people and clothes with the body in view; the same
    # prompt and seed with and without the LoRA show what it changes
    ("concept", "tags"): [
        "{T}, 1girl, solo, standing, full body, city street, daytime",
        "{T}, 1boy, solo, upper body, t-shirt, park, outdoors",
        "{T}, 1girl, solo, sitting, cowboy shot, cafe, indoors",
        "{T}, 1girl, solo, standing, full body, swimsuit, beach",
    ],
    ("concept", "natural"): [
        "{T}, a woman standing on a city street in daylight, full body shot",
        "{T}, a man in a t-shirt in a park, upper body",
        "{T}, a woman sitting in a cafe, shown from the thighs up",
        "{T}, a woman in a swimsuit standing on a beach, full body shot",
    ],
    # the other concept targets ("concept:<target>", captions.CONCEPT_TARGETS): the feature with people,
    # places and framings that the material probably does not have, so it shows whether it carries over
    ("concept:outfit", "tags"): [
        "{T}, 1girl, solo, standing, full body, city street, daytime",
        "{T}, 1girl, solo, sitting, cowboy shot, cafe, indoors",
        "{T}, 1girl, solo, from behind, full body, park, outdoors",
        "{T}, 1boy, solo, standing, full body, simple background",
    ],
    ("concept:outfit", "natural"): [
        "{T}, a woman standing on a city street in daylight, full body shot",
        "{T}, a woman sitting in a cafe, shown from the thighs up",
        "{T}, a woman seen from behind in a park, full body",
        "{T}, a man standing in front of a plain background, full body",
    ],
    ("concept:object", "tags"): [
        "{T}, no humans, on table, indoors, still life",
        "{T}, no humans, grass, outdoors, daytime",
        "{T}, 1girl, solo, holding, upper body, city street",
        "{T}, 1boy, solo, standing, full body, forest",
    ],
    ("concept:object", "natural"): [
        "{T} on a wooden table indoors, still life",
        "{T} lying on the grass outdoors in daylight",
        "a woman holding {T} on a city street, upper body",
        "a man standing in a forest with {T}, full body",
    ],
    ("concept:place", "tags"): [
        "{T}, no humans, scenery, daytime",
        "{T}, no humans, scenery, night",
        "{T}, 1girl, solo, standing, full body",
        "{T}, 1boy, solo, sitting, rain",
    ],
    ("concept:place", "natural"): [
        "{T}, no people, in daylight",
        "{T}, no people, at night",
        "a woman standing in {T}, full body",
        "a man sitting in {T} on a rainy day",
    ],
    ("concept:pose", "tags"): [
        "{T}, 1girl, solo, school uniform, city street",
        "{T}, 1boy, solo, t-shirt, jeans, park, outdoors",
        "{T}, 1girl, solo, dress, simple background",
        "{T}, 1girl, solo, swimsuit, beach",
    ],
    ("concept:pose", "natural"): [
        "{T}, a girl in a school uniform on a city street",
        "{T}, a man in a t-shirt and jeans in a park",
        "{T}, a woman in a dress in front of a plain background",
        "{T}, a woman in a swimsuit on a beach",
    ],
    ("concept:expression", "tags"): [
        "{T}, 1girl, solo, portrait, simple background",
        "{T}, 1boy, solo, upper body, outdoors",
        "{T}, 1girl, solo, from side, upper body, cafe",
        "{T}, 1girl, solo, old woman, grey hair, portrait",
    ],
    ("concept:expression", "natural"): [
        "{T}, close-up portrait of a young woman in front of a plain background",
        "{T}, a young man outdoors, upper body",
        "{T}, a woman in a cafe seen from the side, upper body",
        "{T}, close-up portrait of an old woman with grey hair",
    ],
}
OUTFIT_PROMPT = {"tags": "{T}, solo, swimsuit, beach, standing", "natural": "{T} wearing a swimsuit on a beach"}

# Extra prompt sets the user can add on the results screen ("standard" = PROMPTS above).  Each set changes one
# kind of condition, so it shows which conditions the LoRA keeps working under.  kind -> set -> (label, tags, natural)
PROMPT_SETS = {
    "character": {
        "pose": ("ポーズ・構図", [
            "{T}, solo, arms up, from below, full body, rooftop, blue sky",
            "{T}, solo, lying on back, from above, bed, indoors",
            "{T}, solo, from behind, looking back, upper body, street",
            "{T}, solo, squatting, cowboy shot, alley, outdoors"], [
            "{T} raising both arms on a rooftop under a blue sky, seen from below, full body",
            "{T} lying on a bed, seen from above",
            "{T} seen from behind, looking back over the shoulder on a street, upper body",
            "{T} squatting in an alley, shown from the thighs up"]),
        "place": ("場所・背景", [
            "{T}, solo, standing, classroom, indoors",
            "{T}, solo, standing, snow, winter, outdoors",
            "{T}, solo, sitting, train interior, window",
            "{T}, solo, standing, simple background, white background"], [
            "{T} standing in a classroom",
            "{T} standing outdoors in the snow in winter",
            "{T} sitting by the window inside a train",
            "{T} standing in front of a plain white background"]),
        "outfit": ("衣装", [
            "{T}, solo, school uniform, standing, upper body",
            "{T}, solo, kimono, standing, shrine, outdoors",
            "{T}, solo, business suit, office, indoors",
            "{T}, solo, pajamas, bedroom, sitting"], [
            "{T} in a school uniform, upper body",
            "{T} wearing a kimono at a shrine",
            "{T} in a business suit in an office",
            "{T} in pajamas sitting in a bedroom"]),
        "expression": ("表情・向き", [
            "{T}, solo, laughing, open mouth, portrait",
            "{T}, solo, crying, tears, portrait",
            "{T}, solo, angry, profile, from side, portrait",
            "{T}, solo, surprised, looking at viewer, close-up"], [
            "close-up portrait of {T} laughing with the mouth open",
            "close-up portrait of {T} crying",
            "{T} looking angry, seen in profile, portrait",
            "close-up of {T} looking surprised at the viewer"]),
    },
    "style": {
        "people": ("いろいろな人物", [
            "{T} style, 1girl, solo, office worker, suit, train station",
            "{T} style, 1boy, solo, teenage boy, hoodie, skateboard, street",
            "{T} style, 2girls, friends, laughing, cafe",
            "{T} style, 1boy, solo, muscular man, construction worker, helmet"], [
            "{T} style. A woman in a suit at a train station",
            "{T} style. A teenage boy in a hoodie riding a skateboard on a street",
            "{T} style. Two girls laughing together in a cafe",
            "{T} style. A muscular construction worker wearing a helmet"]),
        "things": ("物・動物", [
            "{T} style, no humans, still life, fruit, bowl, table",
            "{T} style, no humans, dog, running, grass",
            "{T} style, no humans, car, road, city",
            "{T} style, no humans, bird, tree branch, sky"], [
            "{T} style. A still life of fruit in a bowl on a table",
            "{T} style. A dog running on grass",
            "{T} style. A car on a city road",
            "{T} style. A bird on a tree branch against the sky"]),
        "scene": ("場面・光", [
            "{T} style, no humans, city, night, rain, neon lights",
            "{T} style, no humans, interior, kitchen, morning, sunlight",
            "{T} style, no humans, ocean, sunset, waves",
            "{T} style, no humans, forest, fog, dawn"], [
            "{T} style. A rainy city street at night with neon lights",
            "{T} style. A kitchen in the morning sunlight",
            "{T} style. The ocean at sunset with waves",
            "{T} style. A foggy forest at dawn"]),
    },
    "concept": {
        "pose": ("ポーズ・構図", [
            "{T}, 1girl, solo, from side, standing, full body, street",
            "{T}, 1girl, solo, from behind, full body, beach",
            "{T}, 1boy, solo, sitting, cowboy shot, bench, park",
            "{T}, 1girl, solo, lying, from above, bed"], [
            "{T}, a woman seen from the side standing on a street, full body",
            "{T}, a woman seen from behind on a beach, full body",
            "{T}, a man sitting on a park bench, shown from the thighs up",
            "{T}, a woman lying on a bed, seen from above"]),
        "outfit": ("服装", [
            "{T}, 1girl, solo, sweater, jeans, standing, full body",
            "{T}, 1girl, solo, dress, standing, full body, garden",
            "{T}, 1boy, solo, suit, standing, full body, office",
            "{T}, 1girl, solo, gym uniform, standing, gym"], [
            "{T}, a woman in a sweater and jeans, full body",
            "{T}, a woman in a dress in a garden, full body",
            "{T}, a man in a suit in an office, full body",
            "{T}, a woman in sportswear in a gym"]),
    },
    "concept:outfit": {
        "pose": ("ポーズ・構図", [
            "{T}, 1girl, solo, sitting, crossed legs, chair, full body",
            "{T}, 1girl, solo, running, from side, full body, street",
            "{T}, 1girl, solo, arms up, from below, full body, sky",
            "{T}, 1girl, solo, lying, from above, bed"], [
            "{T}, a woman sitting cross-legged on a chair, full body",
            "{T}, a woman running on a street, seen from the side, full body",
            "{T}, a woman raising both arms under the sky, seen from below, full body",
            "{T}, a woman lying on a bed, seen from above"]),
        "people": ("いろいろな人物", [
            "{T}, 1girl, solo, short hair, black hair, standing, full body",
            "{T}, 1girl, solo, long hair, blonde hair, standing, full body",
            "{T}, 1girl, solo, child, standing, full body",
            "{T}, 1girl, solo, mature female, standing, full body"], [
            "{T}, a woman with short black hair, full body",
            "{T}, a woman with long blonde hair, full body",
            "{T}, a little girl, full body",
            "{T}, a middle-aged woman, full body"]),
    },
    "concept:object": {
        "scene": ("場所・光", [
            "{T}, no humans, snow, outdoors, winter",
            "{T}, no humans, night, lamp light, indoors",
            "{T}, no humans, water, beach, sunset",
            "{T}, no humans, white background, simple background"], [
            "{T} in the snow outdoors in winter",
            "{T} indoors at night under lamp light",
            "{T} on a beach at sunset",
            "{T} in front of a plain white background"]),
    },
    "concept:place": {
        "light": ("時間・天気", [
            "{T}, no humans, scenery, sunset",
            "{T}, no humans, scenery, snow, winter",
            "{T}, no humans, scenery, fog, morning",
            "{T}, no humans, scenery, rain, overcast"], [
            "{T}, no people, at sunset",
            "{T}, no people, covered in snow in winter",
            "{T}, no people, on a foggy morning",
            "{T}, no people, on a rainy overcast day"]),
    },
    "concept:pose": {
        "people": ("いろいろな人物", [
            "{T}, 1boy, solo, old man, suit",
            "{T}, 1girl, solo, child, dress",
            "{T}, 1boy, solo, muscular, tank top",
            "{T}, 1girl, solo, kimono, shrine"], [
            "{T}, an old man in a suit",
            "{T}, a little girl in a dress",
            "{T}, a muscular man in a tank top",
            "{T}, a woman in a kimono at a shrine"]),
    },
    "concept:expression": {
        "angle": ("向き・構図", [
            "{T}, 1girl, solo, from side, profile, portrait",
            "{T}, 1girl, solo, from above, looking up, upper body",
            "{T}, 1girl, solo, from below, upper body",
            "{T}, 1boy, solo, full body, standing, street"], [
            "{T}, a woman seen in profile, portrait",
            "{T}, a woman looking up, seen from above, upper body",
            "{T}, a woman seen from below, upper body",
            "{T}, a man standing on a street, full body"]),
    },
}


def prompt_kind(kind: str, target: str | None = None) -> str:
    """Key of PROMPTS / PROMPT_SETS: a concept LoRA uses its target's prompts (body keeps plain "concept")."""
    return f"concept:{target}" if kind == "concept" and target and target != "body" else kind
PROMPT_SET_STANDARD = "standard"


def prompt_sets(kind: str, target: str | None = None) -> list[dict]:
    """The prompt sets offered for a kind (the standard set first)."""
    if kind == "enhance":
        return [{"key": PROMPT_SET_STANDARD, "label": "標準", "n": None}]
    kind = prompt_kind(kind, target)
    return [{"key": PROMPT_SET_STANDARD, "label": "標準", "n": len(PROMPTS.get((kind, "tags"), []))}] +         [{"key": k, "label": v[0], "n": len(v[1])} for k, v in PROMPT_SETS.get(kind, {}).items()]

QUALITY = {
    "illustrious": ("masterpiece, best quality, ", "nsfw, lowres, worst quality, low quality, bad anatomy, bad hands, jpeg artifacts, watermark"),
    "noobai_eps": ("masterpiece, best quality, ", "nsfw, lowres, worst quality, low quality, bad anatomy, bad hands, jpeg artifacts, watermark"),
    "noobai_vpred": ("masterpiece, best quality, ", "nsfw, lowres, worst quality, low quality, bad anatomy, bad hands, jpeg artifacts, watermark"),
    "pony": ("score_9, score_8_up, score_7_up, rating_safe, ", "score_4, score_5, score_6, lowres, bad anatomy, watermark"),
    "sdxl_base": ("", "blurry, low quality, deformed, watermark, text"),
    "anima": ("masterpiece, best quality, safe, ", "nsfw, worst quality, low quality, bad anatomy, watermark"),
    "anima29": ("masterpiece, best quality, safe, ", "nsfw, worst quality, low quality, bad anatomy, watermark"),
    "krea2": ("", ""),
    "qwen_image": ("", "blurry, low quality, deformed, watermark, text"),
    "qwen_image21": ("", ""),
    "zimage_turbo": ("", ""),
    "zimage_base": ("", "blurry, low quality, deformed, watermark, text"),
    # Comfy-Org templates image_moss_image_(base_)t2i: recommended negative prompt
    "moss_image_sft": ("", "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"),
    "moss_image_base": ("", "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"),
}
SAMPLING = {
    "sdxl": {"steps": 28, "cfg": 6.0, "sampler": "euler_ancestral", "scheduler": "normal"},
    "anima": {"steps": 30, "cfg": 4.5, "sampler": "euler_ancestral", "scheduler": "simple"},
    # Gazingstars123/Anima-2.9B model card: Euler + sgm_uniform (the author's setting), 28-50 steps, CFG 3.5-5
    "anima29": {"steps": 30, "cfg": 4.5, "sampler": "euler", "scheduler": "sgm_uniform"},
    "krea2": {"steps": 8, "cfg": 1.0, "sampler": "euler", "scheduler": "simple"},
    # the RAW model (not distilled): musubi-tuner docs/krea2.md "RAW model inference", 28 steps, guidance 5.5.
    # Unverified in this tool.
    "krea2_raw": {"steps": 28, "cfg": 5.5, "sampler": "euler", "scheduler": "simple"},
    # Comfy-Org template image_qwen_Image_2512: 50 steps, cfg 4, shift 3.1.  30 steps here to keep material
    # generation of 20B model practical on 16GB (unverified against 50).
    "qwen_image": {"steps": 30, "cfg": 4.0, "sampler": "euler", "scheduler": "simple", "shift": 3.1},
    # base model: Comfy-Org template image_qwen_image_2_1_t2i (25 steps, cfg 1).  Turbo (GGUF): 8 steps, cfg 1
    # (Comfy-Org / unsloth model cards).  Both unverified in this tool.
    "qwen_image21": {"steps": 25, "cfg": 1.0, "sampler": "euler", "scheduler": "simple"},
    "qwen_image21_turbo": {"steps": 8, "cfg": 1.0, "sampler": "euler", "scheduler": "simple"},
    # ComfyUI blueprints "Text to Image (Z-Image-Turbo / Z-Image-Base)": shift 3, res_multistep, simple;
    # Turbo 8 steps cfg 1, Base 25 steps cfg 4.
    "zimage": {"steps": 8, "cfg": 1.0, "sampler": "res_multistep", "scheduler": "simple", "shift": 3.0},
    "zimage_base": {"steps": 25, "cfg": 4.0, "sampler": "res_multistep", "scheduler": "simple", "shift": 3.0},
    # Comfy-Org templates image_moss_image_t2i / _base_t2i: 25 steps, cfg 2.5, er_sde / simple, the model's own shift
    # (5); Base adds skip-layer guidance on block 22 (build_t2i).  Official: 50 steps cfg 3.  Unverified here.
    "moss_image": {"steps": 25, "cfg": 2.5, "sampler": "er_sde", "scheduler": "simple"},
    # SFT with lodestones' 8-step DMD LoRA (generation only; model card: 8 steps, no CFG; demo Space: plain Euler on
    # the model's own shift-5 schedule).  Unverified here.
    "moss_image_fast": {"steps": 8, "cfg": 1.0, "sampler": "euler", "scheduler": "simple"},
    # akhaliq's 4-step LoRA (model card v2.0 and its ComfyUI workflow: euler / simple / 4 steps / CFG 1).  Unverified here.
    "moss_image_fast4": {"steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple"},
}
FAST_LORA_STRENGTH = 1.0   # lora.safetensors / lora_dmd2.safetensors at strength 1 (model card, commit note)
FAST_LATEST_STRENGTH = 0.7   # lora_latest.safetensors at 0.7 (model card: the full strength is "probably overcooked")


def fast_lora_strength(name: str) -> float:
    """Strength of the 8-step LoRA by its file: lora_latest of the SFT repo (saved as ..._latest) is meant for 0.7."""
    return FAST_LATEST_STRENGTH if Path(name).stem.lower().endswith("_latest") else FAST_LORA_STRENGTH
DEFAULT_STRENGTH = {"sdxl": 1.0, "anima": 1.0, "krea2": 1.0, "qwen_image": 1.0, "qwen_image21": 1.0, "zimage": 1.0, "moss_image": 1.0}
# Image sizes of the families whose native resolution is not ~1MP (Qwen-Image card: 1328x1328 etc.)
NATIVE_SIZE = {"qwen_image": 1328}
# Other models to also make the comparison with (a LoRA trained on the base model is
# usually used on merged / fine-tuned models).  Same family only; a few, because each one adds a model load.
MAX_EXTRA_MODELS = 3
EXTRA_MODELS_KEY = "eval_extra_models"


def extra_models_for(model: dict, ids=None) -> list[dict]:
    """Registered models of the training model's family to compare on as well.  ids: the chosen ones (default:
    the choice kept in the settings, used for every comparison after training).  Missing files, models of
    another family and the training model itself are left out."""
    from .models import VARIANTS, registry
    if ids is None:
        ids = config.settings().get(EXTRA_MODELS_KEY, [])
    files = lambda m: sorted((k, os.path.normcase(os.path.abspath(p))) for k, p in (m.get("files") or {}).items() if p)
    seen = [files(model)]
    out = []
    for mid in dict.fromkeys(x for x in ids or [] if isinstance(x, str)):
        m = registry.get(mid)
        # the same files registered under another name would give the very same graphs: ComfyUI answers a repeated
        # graph from its cache with the image already moved away, and the whole comparison fails
        if (m and m["id"] != model.get("id") and m.get("family") == model["family"] and not m["missing"]
                and m.get("variant") in VARIANTS and VARIANTS[m["variant"]]["family"] == model["family"]
                and files(m) not in seen):
            seen.append(files(m))
            out.append(m)
    return out[:MAX_EXTRA_MODELS]


def prompt_for_model(prompt: str, variant: str, other_variant: str) -> str:
    """A comparison prompt with the quality words of another model's variant (Pony's score tags and so on)."""
    q, _n = QUALITY.get(variant, ("", ""))
    q2, _n2 = QUALITY.get(other_variant, ("", ""))
    if q != q2 and prompt.startswith(q):
        return q2 + prompt[len(q):]
    return prompt


def caption_style(variant):
    from .models import VARIANTS
    return VARIANTS[variant]["caption"]


def subject_class(items) -> str | None:
    """Most common subject-class tag in the training captions (e.g. 1girl).  Tag-trained models
    keep it in captions, so leaving it out of evaluation prompts would understate the LoRA."""
    from collections import Counter
    c = Counter()
    for it in items or []:
        tags = {t.strip() for t in (it.get("caption") or "").split(",")}
        c.update(tags & {"1girl", "1boy", "1other"})
    return c.most_common(1)[0][0] if c else None


MONO_NEGATIVE = "monochrome, greyscale"


def trained_with_monochrome(snap: dict) -> bool:
    """Character run that included black-and-white material (e.g. manga pages).  Those captions
    carry "monochrome", so the colour output is requested by putting it in the negative prompt."""
    return snap.get("kind") in ("character", "concept") and any(
        "monochrome" in (it.get("caption") or "").lower() for it in snap.get("items") or [])


def build_prompts(kind, outfit_mode, variant, trigger, cls=None, prompt_set=PROMPT_SET_STANDARD, target=None):
    style = caption_style(variant)
    kind = prompt_kind(kind, target)
    if prompt_set == PROMPT_SET_STANDARD:
        ps = list(PROMPTS[(kind, "tags" if style == "tags" else "natural")])
        if kind == "character" and outfit_mode == "change":
            ps[1] = OUTFIT_PROMPT["tags" if style == "tags" else "natural"]
    else:
        _label, tags, natural = PROMPT_SETS[kind][prompt_set]
        ps = list(tags if style == "tags" else natural)
    if style == "tags" and kind == "character" and cls:
        ps = [p.replace("{T}, solo", "{T}, " + cls + ", solo") for p in ps]
    q, _neg = QUALITY.get(variant, ("", ""))
    out = []
    for p in ps:
        p = p.replace("{T}", trigger)
        if style == "tags":
            p = p.replace("(", r"\(").replace(")", r"\)")
        out.append(q + p)
    return out


# ======================================================================== ComfyUI server
def _link_or_copy(src: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if dest.stat().st_size == src.stat().st_size:
            return
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        shutil.copyfile(src, dest)


# 画像生成の詳細設定「アテンション」: ComfyUI's launch options (comfy/cli_args.py attn_group); they apply to the whole
# process, so a change restarts ComfyUI.  "": ComfyUI's own choice (PyTorch's SDPA with torch 2 on NVIDIA)
ATTENTION_FLAGS = {"pytorch": "--use-pytorch-cross-attention", "split": "--use-split-cross-attention",
                   "quad": "--use-quad-cross-attention", "sage": "--use-sage-attention", "flash": "--use-flash-attention"}
# the backends that need a package in ComfyUI's environment (not installed by setup)
ATTENTION_PACKAGES = {"sage": "sageattention", "flash": "flash_attn"}


def attention_installed() -> dict[str, bool]:
    """Whether each backend that needs a package has it in ComfyUI's environment (its folder in site-packages:
    no Python started, so the page does not wait)."""
    site = config.COMFY_PYTHON.parent.parent / "Lib" / "site-packages"
    return {k: (site / pkg).is_dir() for k, pkg in ATTENTION_PACKAGES.items()}


class Comfy:
    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.base = f"http://{config.HOST}:{config.COMFY_PORT}"
        self._lock = threading.Lock()
        self.extra_dirs: dict[str, set[str]] = {}
        self.started_extra: dict[str, set[str]] | None = None   # extra dirs the running process was started with
        self._idle: threading.Timer | None = None
        self._log = None   # comfyui.log, open while the process runs
        self.ppm = False   # ComfyUI-ppm's nodes loaded in the running ComfyUI (set at start)
        self.moss_edit = False   # ComfyUI-Moss-Image-Edit-Native's nodes loaded (set at start)
        self.attention = ""   # the attention backend asked for (ATTENTION_FLAGS key, "": ComfyUI's own choice)
        self.started_attention = ""   # ... and the one the running process was started with
        self._uses = 0     # ensure_started() calls: an idle stop armed before the latest use must not stop it
        self._cancels = 0  # cancel_current() calls: a run_prompt that sees this change stops its prompt

    def available(self):
        return config.COMFY_PYTHON.exists() and (config.COMFY_DIR / "main.py").exists()

    def _get(self, path, timeout=10):
        with urllib.request.urlopen(self.base + path, timeout=timeout) as r:
            return json.loads(r.read())

    def _post(self, path, data, timeout=30):
        req = urllib.request.Request(self.base + path, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            try:
                j = json.loads(detail)
                errs = [f"{nid}: {', '.join(x.get('message', '') + ' ' + str(x.get('details', '')) for x in v.get('errors', []))}"
                        for nid, v in (j.get("node_errors") or {}).items()]
                detail = (j.get("error", {}).get("message", "") + " " + " / ".join(errs)).strip()
            except (json.JSONDecodeError, AttributeError):
                pass
            raise RuntimeError(f"ComfyUI がリクエストを拒否しました ({e.code}): {detail[:600]}") from None

    def stage_model(self, kind: str, path: str) -> str:
        """Expose a model file to ComfyUI under data/comfy_models/<kind>/ (hard link, no copy on
        the same volume).  If linking is impossible, the file's own folder is added as an extra
        search path.  Returns the name ComfyUI knows it by."""
        import hashlib
        src = Path(path)
        if not src.is_file():
            raise RuntimeError(f"ファイルが見つかりません（移動・削除された可能性があります）: {src}")
        # each source folder gets its own subfolder, so two different files with the same name never collide
        sub = "lsg_" + hashlib.sha1(str(src.parent).lower().encode()).hexdigest()[:8]
        dest = COMFY_MODELS / kind / sub / src.name
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            # a file replaced in place (e.g. downloaded again under the same name) is a new file: link again
            if dest.exists() and not os.path.samefile(dest, src):
                dest.unlink()
            if not dest.exists():
                os.link(src, dest)
            return os.path.join(sub, src.name)
        except OSError:
            self.extra_dirs.setdefault(kind, set()).add(str(src.parent))
            return src.name

    def _write_extra_paths(self):
        lines = ["lorastudio:", f"  base_path: {json.dumps(str(COMFY_MODELS))}"]
        for k in ("checkpoints", "diffusion_models", "text_encoders", "vae", "loras"):
            dirs = [str(COMFY_MODELS / k)] + sorted(self.extra_dirs.get(k, []))
            lines.append(f"  {k}: |")
            lines += [f"    {d}" for d in dirs]
        p = config.DATA_DIR / "comfy_extra_model_paths.yaml"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return p

    def start(self, timeout=180):
        with self._lock:
            if self.proc is not None and self.proc.poll() is None:
                return
            for k in ("checkpoints", "diffusion_models", "text_encoders", "vae", "loras"):
                (COMFY_MODELS / k).mkdir(parents=True, exist_ok=True)
            extra = self._write_extra_paths()
            from .addons import MOSS_EDIT, PPM, ensure_node
            for name in (PPM, MOSS_EDIT):
                ensure_node(name)   # put in place on the first start (bundled copy, else a small git clone); without it, generation goes on as before
            self.started_nodes = self._custom_nodes()
            self.started_extra = {k: set(v) for k, v in self.extra_dirs.items()}
            self.started_attention = self.attention
            attn = [ATTENTION_FLAGS[self.attention]] if self.attention in ATTENTION_FLAGS else []
            out_dir = config.DATA_DIR / "comfy_output"
            out_dir.mkdir(parents=True, exist_ok=True)
            if not self._port_released():
                raise RuntimeError(f"ポート {config.COMFY_PORT} を別のプログラム（別に起動した ComfyUI や、前回終了し損ねた ComfyUI など）が"
                                   "使っているため、評価用の ComfyUI を起動できません。そのプログラムを終了してから、もう一度実行してください。")
            self._close_log()
            log = self._log = open(config.DATA_DIR / "comfyui.log", "ab")
            env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
            self.proc = subprocess.Popen(
                [str(config.COMFY_PYTHON), "main.py", "--listen", config.HOST, "--port", str(config.COMFY_PORT),
                 "--disable-auto-launch", "--extra-model-paths-config", str(extra), "--output-directory", str(out_dir),
                 "--input-directory", str(COMFY_INPUT),
                 "--disable-all-custom-nodes", *self._custom_nodes(), "--preview-method", "none", *attn],
                cwd=str(config.COMFY_DIR), stdout=log, stderr=subprocess.STDOUT, env=env, creationflags=flags)
            t0 = time.time()
            while time.time() - t0 < timeout:
                if self.proc.poll() is not None:
                    raise RuntimeError("ComfyUI の起動に失敗しました（data/comfyui.log を確認してください）")
                try:
                    self._get("/system_stats", timeout=2)
                except (urllib.error.URLError, OSError, ValueError):
                    time.sleep(1)
                    continue
                if self.proc.poll() is not None:   # what answered is not the process started here
                    raise RuntimeError("ComfyUI の起動に失敗しました（data/comfyui.log を確認してください）")
                try:   # ComfyUI-ppm's nodes, only when they loaded (a version mismatch fails its import)
                    self.ppm = "AttentionCouplePPM" in self._get("/object_info/AttentionCouplePPM", timeout=10)
                except (urllib.error.URLError, OSError, ValueError):
                    self.ppm = False
                try:
                    self.moss_edit = "MOSSImageEditTextEncode" in self._get("/object_info/MOSSImageEditTextEncode", timeout=10)
                except (urllib.error.URLError, OSError, ValueError):
                    self.moss_edit = False
                return
            raise RuntimeError("ComfyUI の起動がタイムアウトしました")

    @staticmethod
    def _port_released(wait=5.0) -> bool:
        """True when nothing accepts connections on ComfyUI's port.  A ComfyUI stopped just before (refresh)
        may take a moment to close its socket, so this waits a little before saying the port is taken."""
        t0 = time.time()
        while True:
            # Windows answers a closed local port only after retrying for ~2 s, so the wait is a few seconds;
            # no answer at all is taken as free too (a process that cannot bind is then caught by poll())
            try:
                socket.create_connection((config.HOST, config.COMFY_PORT), timeout=3).close()
            except OSError:
                return True
            if time.time() - t0 >= wait:
                return False
            time.sleep(0.5)

    @staticmethod
    def _custom_nodes() -> list[str]:
        """Custom nodes are off, except the optional ones this tool installed (e.g. ComfyUI-GGUF)."""
        names = [n for n in ("ComfyUI-GGUF", "ComfyUI-ppm", "ComfyUI-Moss-Image-Edit-Native") if (config.COMFY_DIR / "custom_nodes" / n / "__init__.py").exists()]
        return ["--whitelist-custom-nodes", *names] if names else []

    def refresh(self):
        """Restart if new extra dirs were added (ComfyUI reads extra paths at startup)."""
        self.stop()
        self.start()

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def ensure_started(self, attention: str = ""):
        """Start ComfyUI, or keep using the one left running by the previous job.  It is restarted only when
        files were staged from folders it does not know (extra paths and custom nodes are read at startup), or for
        another attention backend (画像生成 chooses one; every other job uses ComfyUI's own choice)."""
        with self._lock:
            self._uses += 1
            if self._idle:
                self._idle.cancel()
                self._idle = None
            self.attention = attention if attention in ATTENTION_FLAGS else ""
        if self.running() and ({k: set(v) for k, v in self.extra_dirs.items()} != self.started_extra
                               or getattr(self, "started_nodes", None) != self._custom_nodes()
                               or self.started_attention != self.attention):
            self.refresh()
        else:
            self.start()

    def park(self, keep_s: float = 300):
        """End of a job: move the models out of VRAM (for scoring, analysis and the next job) but keep the process,
        so the next ComfyUI job skips the start-up (about 10-30 s).  The loaded models stay in system memory, in
        ComfyUI's node cache, so the next job with the same model does not read and build it from disk again
        (free_memory would clear that cache); they are dropped only when system memory runs short (_trim_ram).
        Stopped when a job that does not use ComfyUI (training) takes the GPU, or after keep_s without use."""
        if not self.running():
            return
        try:
            self._post("/free", {"unload_models": True}, timeout=10)
        except Exception:
            self.stop()
            return
        self._arm_idle(keep_s)
        threading.Thread(target=self._trim_ram, args=(self._uses,), daemon=True, name="comfy-ram").start()

    def _trim_ram(self, uses, wait_s: float = PARK_RAM_CHECK_S):
        """After a park: ComfyUI moves the models from VRAM to system memory in its own thread; when that leaves too
        little free memory for the rest (analysis, the next job), it drops them too.  Not when a job uses it again."""
        time.sleep(wait_s)
        free = free_ram_gb()
        if free is None or free >= PARK_RAM_FREE_GB or self._uses != uses or not self.running():
            return
        try:
            self._post("/free", {"unload_models": True, "free_memory": True}, timeout=10)
        except Exception:
            pass

    def _arm_idle(self, keep_s):
        with self._lock:
            if self._idle:
                self._idle.cancel()
            self._idle = threading.Timer(keep_s, self._idle_stop, args=(keep_s, self._uses))
            self._idle.daemon = True
            self._idle.start()

    def _idle_stop(self, keep_s, uses):
        from .gpu import gpu
        with self._lock:
            if self._uses != uses:   # a job started using ComfyUI after this timer was armed: it is theirs now
                return
            if gpu.owner is None:
                self._idle = None
                self._stop_locked()
                return
        # in use by a ComfyUI job, or by another job (scoring after a comparison) that did not stop it: look again later
        self._arm_idle(keep_s)

    def after_failure(self, exc: Exception):
        """A job failed.  ComfyUI is started again only when it is the cause (no answer, ended, timed out);
        a rejected graph, a missing file or an unfitting LoRA keeps the process (its start-up is 10-30 s)."""
        if isinstance(exc, OSError) or "タイムアウト" in str(exc) or "応答しません" in str(exc) or not self.running():
            self.stop()
        else:
            self.park()

    def stop(self):
        with self._lock:
            if self._idle:
                self._idle.cancel()
                self._idle = None
            self._stop_locked()

    def _stop_locked(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                self._post("/free", {"unload_models": True, "free_memory": True}, timeout=5)
            except Exception:
                pass
            # the venv python.exe is a launcher with a child interpreter: end the whole tree
            from .training import kill_tree
            kill_tree(self.proc.pid)
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                pass
        self.proc = None
        self._close_log()

    def _close_log(self):
        if self._log is not None:
            try:
                self._log.close()
            except OSError:
                pass
            self._log = None

    def cancel_current(self) -> None:
        """Stop the prompt run_prompt is waiting for.  Seen by its wait loop, which takes a prompt that has not
        started out of ComfyUI's queue (an /interrupt sent then is lost: ComfyUI clears the flag when a prompt
        starts) and interrupts only that prompt once it runs."""
        self._cancels += 1

    def _stop_prompt(self, pid: str) -> bool:
        """True when the prompt is gone from the queue without running (nothing more will come for it)."""
        q = self._get("/queue", timeout=5)
        if any(item[1] == pid for item in q.get("queue_pending", [])):
            self._post("/queue", {"delete": [pid]}, timeout=5)
            q = self._get("/queue", timeout=5)   # it may have started meanwhile: then it is interrupted below
        if any(item[1] == pid for item in q.get("queue_running", [])):
            self._post("/interrupt", {"prompt_id": pid}, timeout=5)
            return False
        return not any(item[1] == pid for item in q.get("queue_pending", [])) and pid not in self._get(f"/history/{pid}", timeout=5)

    def run_prompt(self, graph: dict, timeout=900, on_progress=None) -> list[dict]:
        """Queue a graph and wait for its images.  on_progress(step, steps) is called with the sampler's steps
        as ComfyUI reports them (its websocket); without the socket the run works the same, just without steps."""
        client = uuid.uuid4().hex
        cancels = self._cancels
        sock = None
        if on_progress:
            try:
                sock = _ProgressSocket(config.HOST, config.COMFY_PORT, client, on_progress)
            except OSError:
                sock = None
        try:
            return self._run_prompt(graph, client, timeout, sock, cancels)
        finally:
            if sock:
                sock.close()

    def _run_prompt(self, graph: dict, client: str, timeout, sock, cancels: int | None = None) -> list[dict]:
        # a slow PC (a model read from an HDD, memory swapped out) can leave ComfyUI's server unanswered for a while:
        # one look that is not answered in time is not a failure while the process lives (the whole run has `timeout`)
        try:
            res = self._post("/prompt", {"prompt": graph, "client_id": client}, timeout=180)
        except (TimeoutError, socket.timeout):
            raise RuntimeError("ComfyUI が 3 分間応答しませんでした（PC の負荷が高いか、メモリーが足りない可能性があります）") from None
        if "error" in res:
            raise RuntimeError(f"ComfyUI がワークフローを拒否しました: {json.dumps(res)[:400]}")
        pid = res["prompt_id"]
        if sock:
            sock.done.clear()
            sock.prompt_id = pid
        t0 = time.time()
        interrupted = False
        while time.time() - t0 < timeout:
            try:
                h = self._get(f"/history/{pid}", timeout=30)
            except (TimeoutError, socket.timeout, urllib.error.URLError, ConnectionError) as e:
                if not self.running():
                    raise RuntimeError("生成の途中で ComfyUI が終了しました（data/comfyui.log を確認してください）") from None
                if isinstance(e, urllib.error.URLError) and not isinstance(e.reason, (TimeoutError, socket.timeout, ConnectionError)):
                    raise
                time.sleep(1)
                continue
            if pid not in h and cancels is not None and self._cancels != cancels and not interrupted:
                try:
                    if self._stop_prompt(pid):
                        raise RuntimeError("中止しました")
                    interrupted = not any(item[1] == pid for item in self._get("/queue", timeout=5).get("queue_pending", []))
                except (OSError, ValueError):
                    pass   # tried again on the next poll
            if pid in h:
                st = h[pid].get("status", {})
                if st.get("status_str") == "error":
                    msgs = [m for m in st.get("messages", []) if m[0] == "execution_error"]
                    detail = msgs[0][1].get("exception_message", "") if msgs else ""
                    raise RuntimeError(f"ComfyUI 実行エラー: {detail[:400]}")
                imgs = []
                for node in h[pid].get("outputs", {}).values():
                    imgs += node.get("images", [])
                if imgs:
                    return imgs
                if st.get("completed"):
                    raise RuntimeError("ComfyUI が画像を出力しませんでした")
            # with the progress socket the end is seen at once (polling stays as the fallback); without it, a short
            # interval: each image waited 0.25 s on average for the next look, and the GPU is idle meanwhile
            if sock and sock.done.is_set():
                time.sleep(0.03)   # ended: ComfyUI writes the history just after the message
            elif sock:
                sock.done.wait(0.5)
            else:
                time.sleep(0.15)
        raise RuntimeError(f"生成がタイムアウトしました（1 枚に {int(timeout // 60)} 分以上かかりました）")

    def lora_warnings(self, since_bytes: int) -> list[str]:
        p = config.DATA_DIR / "comfyui.log"
        try:
            with open(p, "rb") as f:
                f.seek(since_bytes)
                txt = f.read().decode("utf-8", "replace")
        except OSError:
            return []
        return [l for l in txt.splitlines() if "lora key not loaded" in l.lower()]

    def log_size(self) -> int:
        p = config.DATA_DIR / "comfyui.log"
        return p.stat().st_size if p.exists() else 0


comfy = Comfy()


class _ProgressSocket:
    """Receive-only websocket client for ComfyUI's /ws (stdlib only): calls on_progress(step, steps) for the
    "progress" messages of one prompt.  Runs in its own thread; any error just ends the progress reports."""

    def __init__(self, host: str, port: int, client_id: str, on_progress):
        self.prompt_id: str | None = None
        self.done = threading.Event()   # the prompt ended (ok, error or interrupted): its history is ready to read
        self._cb = on_progress
        self._s = socket.create_connection((host, port), timeout=5)
        key = base64.b64encode(os.urandom(16)).decode()
        self._s.sendall((f"GET /ws?clientId={client_id} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
                         f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = self._s.recv(4096)
            if not chunk:
                raise OSError("websocket handshake failed")
            head += chunk
        if b" 101 " not in head.split(b"\r\n", 1)[0]:
            raise OSError("websocket handshake refused")
        self._buf = head.split(b"\r\n\r\n", 1)[1]
        self._s.settimeout(None)
        self._closed = False
        threading.Thread(target=self._loop, daemon=True, name="comfy-ws").start()

    def _read(self, n: int) -> bytes:
        while len(self._buf) < n:
            chunk = self._s.recv(65536)
            if not chunk:
                raise OSError("closed")
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def _loop(self):
        msg, op = b"", 0
        try:
            while not self._closed:
                b0, b1 = self._read(2)
                n = b1 & 0x7F
                if n == 126:
                    n = struct.unpack(">H", self._read(2))[0]
                elif n == 127:
                    n = struct.unpack(">Q", self._read(8))[0]
                mask = self._read(4) if b1 & 0x80 else None
                data = self._read(n)
                if mask:
                    data = bytes(c ^ mask[i % 4] for i, c in enumerate(data))
                opcode = b0 & 0x0F
                if opcode == 8:
                    return
                if opcode in (1, 2):
                    msg, op = data, opcode
                elif opcode == 0:
                    msg += data
                else:
                    continue   # ping / pong: ComfyUI does not require answers
                if not b0 & 0x80 or op != 1:
                    continue   # unfinished fragment, or a binary preview image
                try:
                    m = json.loads(msg)
                except ValueError:
                    continue
                d = m.get("data") or {}
                if m.get("type") == "progress" and self.prompt_id and d.get("prompt_id") in (None, self.prompt_id):
                    self._cb(int(d.get("value") or 0), int(d.get("max") or 0))
                elif self.prompt_id and d.get("prompt_id") == self.prompt_id and (
                        m.get("type") in ("execution_success", "execution_error", "execution_interrupted")
                        or (m.get("type") == "executing" and d.get("node") is None)):
                    self.done.set()
        except (OSError, ValueError):
            pass

    def close(self):
        self._closed = True
        try:
            self._s.close()
        except OSError:
            pass


KEEP_COMFY_OWNERS: set[str] = set()   # GPU owners that reuse a ComfyUI left running for them (image generation)
COMFY_TASK_PREFIXES = ("素材生成", "比較")   # GPU task labels (server.py) of jobs that generate with ComfyUI


def comfy_user(owner: str | None) -> bool:
    return bool(owner) and (owner in KEEP_COMFY_OWNERS or owner.startswith(COMFY_TASK_PREFIXES))


def _stop_comfy_for_exclusive():
    if comfy_user(gpu.owner):
        return
    comfy.stop()


gpu.on_exclusive(_stop_comfy_for_exclusive)


# ======================================================================== workflows
def sampling_for(model: dict, staged: dict) -> dict:
    fam = model["family"]
    if fam == "qwen_image21" and staged.get("on") == "gen" and "turbo" in Path(model["files"]["gen"]).name.lower():
        return dict(SAMPLING["qwen_image21_turbo"])
    if fam == "zimage" and model.get("variant") == "zimage_base":
        return dict(SAMPLING["zimage_base"])
    if fam == "anima" and model.get("variant") == "anima29":
        return dict(SAMPLING["anima29"])
    if krea2_raw(model, staged.get("on")):
        return dict(SAMPLING["krea2_raw"])
    return dict(SAMPLING[fam])


NEGPIP_FAMILIES = ("sdxl", "anima", "moss_image")   # ComfyUI-ppm CLIPNegPip
COUPLE_FAMILIES = ("sdxl", "anima")                  # ComfyUI-ppm AttentionCouplePPM
_NEG_WEIGHT = re.compile(r":\s*-\s*\d")                                  # (word:-1)
_ANY_WEIGHT = re.compile(r"\([^()]*:\s*-?\d+(?:\.\d+)?\s*\)")           # (word:1.2)


def wants_negpip(fam: str, prompt: str, negative: str) -> bool:
    """NegPip only where it changes something: a negative weight, or any weight on MOSS-Image (whose own text
    encoding ignores weights).  Prompts without them stay exactly as ComfyUI encodes them."""
    text = f"{prompt}\n{negative}"
    return fam in NEGPIP_FAMILIES and bool(_NEG_WEIGHT.search(text) or (fam == "moss_image" and _ANY_WEIGHT.search(text)))


def build_t2i(model: dict, staged: dict, loras: list[tuple[str, float]], prompt: str, negative: str, seed: int,
              width: int, height: int, sampling: dict, prefix: str, lora_clip: bool = False, init: dict | None = None,
              regions: list[str] | None = None) -> dict:
    """Text-to-image graph for every supported family.  staged: names ComfyUI knows the files by
    (stage_files).  lora_clip: also apply the LoRAs to the text encoder (SDXL character LoRAs used
    for material generation); evaluation LoRAs are UNet-only.  init: {"image": name in ComfyUI's input
    folder, "denoise": 0-1} starts from that image (image-to-image) instead of empty noise.
    init["edit"] (MOSS-Image with an edit LoRA): the image is edited as the prompt says (ComfyUI-Moss-Image-Edit-Native's
    MOSSImageEditTextEncode in that LoRA's layout, its edit_format); the output takes the image's size (about 1 MP).
    regions: [left, right] prompts drawn on each half (ComfyUI-ppm Attention Couple, SDXL / Anima).
    sampling["rescale"] == "on": RescaleCFGPost against blown-out colours at high CFG (ComfyUI-ppm).
    The ComfyUI-ppm nodes are used only when they loaded in the running ComfyUI (staged["ppm"] overrides, for tests)."""
    fam = model["family"]
    edit = (init or {}).get("edit") if fam == "moss_image" else None
    g = {}
    if fam == "sdxl":
        g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": staged["ckpt"]}}
        model_ref, clip_ref, vae_ref = ["1", 0], ["1", 1], ["1", 2]
        if staged.get("vae_override"):   # 画像生成の詳細設定で選んだ VAE (e.g. sdxl_vae / a fixed fp16 one)
            g["1v"] = {"class_type": "VAELoader", "inputs": {"vae_name": staged["vae_override"]}}
            vae_ref = ["1v", 0]
        from .models import VARIANTS
        if VARIANTS.get(model.get("variant"), {}).get("prediction") == "v":
            # trained as v-prediction (training.py); a merge without ComfyUI's v_pred key would be sampled as eps
            g["1p"] = {"class_type": "ModelSamplingDiscrete", "inputs": {"model": model_ref, "sampling": "v_prediction", "zsnr": True}}
            model_ref = ["1p", 0]
    else:
        if staged.get("gguf"):
            g["1"] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": staged["dit"]}}
        else:
            g["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": staged["dit"], "weight_dtype": staged.get("dtype", "default")}}
        clip_type = {"krea2": "krea2", "qwen_image": "qwen_image", "qwen_image21": "qwen_image",
                     "zimage": "lumina2", "moss_image": "moss_image"}.get(fam, "stable_diffusion")
        g["1t"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": staged["te"], "type": clip_type}}
        g["1v"] = {"class_type": "VAELoader", "inputs": {"vae_name": staged["vae"]}}
        model_ref, clip_ref, vae_ref = ["1", 0], ["1t", 0], ["1v", 0]
    fast_slot = {"on": "fast", "4": "fast4"}.get(sampling.get("fast"))   # generate.FAST_SLOTS
    if fast_slot and staged.get(fast_slot):   # the fast LoRA first, under the user's LoRAs
        loras = [(staged[fast_slot], fast_lora_strength(staged[fast_slot])), *loras]
    for i, (name, strength) in enumerate(loras):
        nid = f"L{i}"
        if lora_clip:
            g[nid] = {"class_type": "LoraLoader", "inputs": {"model": model_ref, "clip": clip_ref, "lora_name": name,
                                                             "strength_model": float(strength), "strength_clip": float(strength)}}
            model_ref, clip_ref = [nid, 0], [nid, 1]
        else:
            g[nid] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model_ref, "lora_name": name,
                                                                      "strength_model": float(strength)}}
            model_ref = [nid, 0]
    if fam == "sdxl" and sampling.get("clip_skip", 1) > 1:
        g["1c"] = {"class_type": "CLIPSetLastLayer", "inputs": {"clip": clip_ref, "stop_at_clip_layer": -int(sampling["clip_skip"])}}
        clip_ref = ["1c", 0]
    ppm = staged.get("ppm", comfy.ppm and comfy.running())   # read when the graph is made (after ensure_started)
    if ppm and not edit and wants_negpip(fam, prompt + "\n" + "\n".join(regions or []), negative):
        g["1n"] = {"class_type": "CLIPNegPip", "inputs": {"model": model_ref, "clip": clip_ref}}
        model_ref, clip_ref = ["1n", 0], ["1n", 1]
    if sampling.get("shift"):
        g["1s"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": model_ref, "shift": float(sampling["shift"])}}
        model_ref = ["1s", 0]
    if fam == "moss_image" and (edit or model.get("variant") == "moss_image_base"):
        # skip-layer guidance as in the Base template (SkipLayerGuidanceDiTSimple, block 22, first half of the steps);
        # editing: all the steps, as in ComfyUI-Moss-Image-Edit-Native's workflows (SFT and Base)
        g["1g"] = {"class_type": "SkipLayerGuidanceDiTSimple", "inputs": {"model": model_ref, "double_layers": "",
                                                                         "single_layers": "22", "start_percent": 0.0,
                                                                         "end_percent": 1.0 if edit else 0.5}}
        model_ref = ["1g", 0]
    if edit:   # the instruction and the source image together; the empty latent comes out of it at the image's size
        g["5a"] = {"class_type": "LoadImage", "inputs": {"image": init["image"]}}
        g["3"] = {"class_type": "MOSSImageEditTextEncode", "inputs": {
            "clip": clip_ref, "vae": vae_ref, "prompt": prompt, "negative_prompt": negative, "edit_format": edit,
            "images.image_1": ["5a", 0], "reference_max_edge": 0, "output_max_edge": 0, "kv_cache": "auto", "image_strength": 1.0}}
        pos, neg = ["3", 0], ["3", 1]
    elif fam == "qwen_image21":
        g["3"] = {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": clip_ref, "prompt": prompt, "negative_prompt": negative,
                                                                   "resolution": 1024}}
        pos, neg = ["3", 0], ["3", 1]
    else:
        g["3"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": clip_ref}}
        g["4"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": clip_ref}}
        pos, neg = ["3", 0], ["4", 0]
    latent_cls = "EmptyLatentImage" if fam in ("sdxl", "qwen_image21") else \
        "EmptyFlux2LatentImage" if fam == "moss_image" else "EmptySD3LatentImage"
    if edit:
        pass   # the latent is MOSSImageEditTextEncode's third output
    elif init:   # the image, scaled and centre-cropped to the size, encoded by the model's own VAE
        g["5a"] = {"class_type": "LoadImage", "inputs": {"image": init["image"]}}
        g["5b"] = {"class_type": "ImageScale", "inputs": {"image": ["5a", 0], "upscale_method": "lanczos", "width": width,
                                                         "height": height, "crop": "center"}}
        g["5"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["5b", 0], "vae": vae_ref}}
    else:
        g["5"] = {"class_type": latent_cls, "inputs": {"width": width, "height": height, "batch_size": 1}}
    if ppm and fam in COUPLE_FAMILIES and regions and any(r.strip() for r in regions):
        # the whole prompt over the whole image (weight 0.6) and each half's own prompt on it (0.4), as in
        # ComfyUI-ppm's example workflow "attention_couple_with_global_prompt"
        couple = {"model": model_ref, "base_cond": pos}
        g["5m"] = {"class_type": "LatentToMaskBB", "inputs": {"latent": ["5", 0], "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "value": 0.6}}
        couple["base_mask"] = ["5m", 0]
        for k, text in enumerate(regions[:2], 1):
            if not text.strip():
                continue
            g[f"5r{k}"] = {"class_type": "CLIPTextEncode", "inputs": {"text": f"{prompt}, {text.strip()}", "clip": clip_ref}}
            g[f"5m{k}"] = {"class_type": "LatentToMaskBB", "inputs": {"latent": ["5", 0], "x": 0.5 * (k - 1), "y": 0.0,
                                                                      "w": 0.5, "h": 1.0, "value": 0.4}}
            n = len([key for key in couple if key.startswith("cond_")]) + 1
            couple[f"cond_{n}"], couple[f"mask_{n}"] = [f"5r{k}", 0], [f"5m{k}", 0]
        g["5c"] = {"class_type": "AttentionCouplePPM", "inputs": couple}
        model_ref = ["5c", 0]
    if ppm and sampling.get("rescale") == "on":
        g["5s"] = {"class_type": "RescaleCFGPost", "inputs": {"model": model_ref, "multiplier": 0.7, "alt_mode": False,
                                                              "sigma_start": -1.0, "sigma_end": -1.0}}
        model_ref = ["5s", 0]
    g["6"] = {"class_type": "KSampler", "inputs": {"model": model_ref, "seed": int(seed), "steps": int(sampling["steps"]),
                                                   "cfg": float(sampling["cfg"]), "sampler_name": sampling["sampler"],
                                                   "scheduler": sampling["scheduler"], "positive": pos, "negative": neg,
                                                   "latent_image": ["3", 2] if edit else ["5", 0],
                                                   "denoise": float(init["denoise"]) if init and not edit else 1.0}}
    g["7"] = {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": vae_ref}}
    g["8"] = {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": prefix}}
    return g


def graph_for(model: dict, lora_name: str | None, strength: float, prompt: str, negative: str, seed: int,
              width: int, height: int, prefix: str, eval_model: dict) -> dict:
    s = sampling_for(model, eval_model)
    if model["family"] == "sdxl" and model["variant"] == "pony":
        s["clip_skip"] = 2
    return build_t2i(model, eval_model, [(lora_name, strength)] if lora_name else [], prompt, negative, seed,
                     width, height, s, prefix)


def comfy_lora_file(family: str, src: Path, dest: Path) -> Path:
    """Write the LoRA in the key layout ComfyUI expects for this family.  A file already written from the same
    LoRA (not older than it) is reused: converting starts the training Python each time."""
    src, final = Path(src).resolve(), Path(dest).resolve()   # the converter runs with cwd=sd-scripts
    final.parent.mkdir(parents=True, exist_ok=True)
    try:
        if final.stat().st_size > 0 and final.stat().st_mtime >= src.stat().st_mtime:
            return final
    except OSError:
        pass
    dest = final.with_name(final.stem + ".part" + final.suffix)   # never leave a half-written file under the final name
    dest.unlink(missing_ok=True)
    _convert_lora(family, src, dest)
    os.replace(dest, final)
    return final


COMFY_INPUT = config.DATA_DIR / "comfy_input"   # images a graph loads (LoadImage): the image page's source images
CONVERTED_FAMILIES = ("anima", "zimage", "qwen_image21")   # their LoRA is rewritten for ComfyUI (the others are copied)


def _convert_lora(family: str, src: Path, dest: Path) -> Path:
    from .training import engine_env
    if family == "anima":
        r = subprocess.run([str(config.TRAIN_PYTHON), "networks/convert_anima_lora_to_comfy.py", str(src), str(dest)],
                           cwd=str(config.SD_SCRIPTS_DIR), env=engine_env("sd-scripts"), capture_output=True, text=True, encoding="utf-8", errors="replace",
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if r.returncode != 0 or not dest.exists():
            raise RuntimeError("Anima LoRA の ComfyUI 形式への変換に失敗しました: " + (r.stderr or r.stdout)[-400:])
        return dest
    if family == "zimage":
        # musubi-tuner docs/zimage.md: the Diffusers layout ("--target other") is what ComfyUI loads
        r = subprocess.run([str(config.TRAIN_PYTHON), "convert_lora.py", "--input", str(src), "--output", str(dest),
                            "--target", "other"],
                           cwd=str(config.MUSUBI_DIR), env=engine_env("musubi-tuner"), capture_output=True, text=True, encoding="utf-8", errors="replace",
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if r.returncode != 0 or not dest.exists():
            raise RuntimeError("Z-Image LoRA の ComfyUI 形式への変換に失敗しました: " + (r.stderr or r.stdout)[-400:])
        return dest
    if family == "qwen_image21":
        # PR #1145 output keeps img_mlp gate_layer/proj separate; ComfyUI fuses them, so the native file loads only
        # partly.  The PR author's conversion to the Diffusers layout loads fully (reported in the PR, unverified here).
        from .models import FAMILY_ENGINE
        from .training import engine_cwd
        eng = FAMILY_ENGINE[family]
        r = subprocess.run([str(config.TRAIN_PYTHON), "convert_lora.py", "--input", str(src), "--output", str(dest),
                            "--target", "other", "--diffusers_prefix", "transformer"],
                           cwd=str(engine_cwd(eng)), env=engine_env(eng), capture_output=True, text=True, encoding="utf-8",
                           errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if r.returncode != 0 or not dest.exists():
            raise RuntimeError("Qwen-Image-2.1 LoRA の ComfyUI 形式への変換に失敗しました: " + (r.stderr or r.stdout)[-400:])
        return dest
    shutil.copyfile(src, dest)
    return dest


# ======================================================================== evaluation job
def default_spec(project, run: dict) -> dict:
    """Initial comparison: limited on purpose (base + up to 3 checkpoints, strength 1.0, 2 seeds,
    4 prompts, 6 for styles).  More can be added from the results screen."""
    ckpts = [c for c in project.checkpoints(run["id"]) if c["kind"] == "lora"]
    steps = sorted({c["step"] for c in ckpts if c["step"]})
    picks = []
    if steps:
        picks = sorted({steps[len(steps) // 2 - 1] if len(steps) > 1 else steps[0], steps[(len(steps) * 3) // 4 - 1] if len(steps) > 2 else steps[-1]})
    picks.append("final")
    fam = run["config"]["model"]["family"]
    enh = run["config"].get("enhance")
    if enh:
        # texture LoRA: the same picture with and without the LoRA, at several strengths (sliders in both
        # directions).  One middle checkpoint and the final one keep the first comparison small.
        from . import enhance
        return {"checkpoints": (picks[-2:] if len(picks) > 1 else picks), "strengths": enhance.eval_strengths(enh),
                "seeds": [1234, 5678], "include_base": True, "prompts": enhance.eval_prompts(enh, run["config"]["model"]["variant"]),
                "width": 1024, "height": 1024, **_extra_spec(run)}
    if run["config"].get("members"):
        # multi-character: the first two standard prompts for each character keeps the first comparison small
        prompts = [p for m in run["config"]["members"] for p in _member_prompts(run["config"], m)[:2]]
    else:
        prompts = build_prompts(run["config"]["kind"], run["config"].get("outfit_mode"), run["config"]["model"]["variant"],
                                run["config"]["trigger"], subject_class(run["config"].get("items")),
                                target=run["config"].get("concept_target"))
    return {"checkpoints": picks, "strengths": [DEFAULT_STRENGTH[fam]], "seeds": [1234, 5678], "include_base": True,
            "prompts": prompts,
            "width": NATIVE_SIZE.get(fam, 1024), "height": NATIVE_SIZE.get(fam, 1024), **_extra_spec(run)}


def _extra_spec(run: dict, ids=None, checkpoints=("final",)) -> dict:
    """The other models of a comparison: on them only "no LoRA" and the final version (the first comparison stays
    small; more can be added from the results screen)."""
    ms = extra_models_for(run["config"]["model"], ids)
    return {"extra_models": [m["id"] for m in ms], "extra_checkpoints": list(checkpoints)} if ms else {}


def _member_prompts(snap: dict, m: dict, prompt_set: str = PROMPT_SET_STANDARD) -> list[str]:
    lo, hi = m["id_range"]
    items = [it for it in snap.get("items") or [] if lo <= it["id"] < hi]
    return build_prompts("character", m.get("outfit_mode"), snap["model"]["variant"], m["trigger"], subject_class(items), prompt_set)


def run_prompts(run: dict, sets: list[str]) -> list[str]:
    """Comparison prompts of a run for the chosen prompt sets, in the order of prompt_sets()."""
    snap = run["config"]
    if snap.get("enhance"):
        from . import enhance
        return enhance.eval_prompts(snap["enhance"], snap["model"]["variant"])
    offered = [s["key"] for s in prompt_sets(snap["kind"], snap.get("concept_target"))]
    out = []
    for key in offered:
        if key in sets and snap.get("members"):
            out += [p for m in snap["members"] for p in _member_prompts(snap, m, key)]
        elif key in sets:
            out += build_prompts(snap["kind"], snap.get("outfit_mode"), snap["model"]["variant"], snap["trigger"],
                                 subject_class(snap.get("items")), key, snap.get("concept_target"))
    return list(dict.fromkeys(out))


def stage_files(model: dict, prefer: str | None = None) -> dict:
    """Expose the model's files to ComfyUI and return the names to put in the graph.
    Krea2: Turbo if registered (train on RAW, use on Turbo).  Qwen-Image-2.1: the light generation file
    (GGUF Turbo) if registered, else the bf16 base.  Qwen-Image bf16 is loaded as fp8 so the 20B model
    fits a 16GB card with offloading."""
    f = model["files"]
    fam = model["family"]
    if fam == "sdxl":
        return {"ckpt": comfy.stage_model("checkpoints", f["ckpt"]), "on": "ckpt"}
    on = gen_slot(model, prefer)
    dit = f[on]
    out = {"dit": comfy.stage_model("diffusion_models", dit), "te": comfy.stage_model("text_encoders", gen_text_encoder(fam, f["te"])),
           "vae": comfy.stage_model("vae", f["vae"]), "on": on, "gguf": str(dit).lower().endswith(".gguf")}
    if load_as_fp8(fam, dit):
        out["dtype"] = "fp8_e4m3fn"
    return out


def gen_slot(model: dict, prefer: str | None = None) -> str:
    """Which file of a DiT model generation loads: the light generation file (Krea 2 Turbo, Qwen-Image GGUF) when it
    is registered and there, else the training file (a generation file moved away does not stop a comparison)."""
    f, fam = model["files"], model["family"]
    key = {"krea2": "turbo", "qwen_image": "gen", "qwen_image21": "gen"}.get(fam)
    ckpt_there = bool(f.get("ckpt")) and Path(f["ckpt"]).is_file()
    if key and f.get(key) and (Path(f[key]).is_file() or not ckpt_there) and (prefer != "ckpt" or not f.get("ckpt")):
        return key
    return "ckpt"


def krea2_raw(model: dict, on: str) -> bool:
    """Krea 2 generating on the RAW model (no Turbo file): full steps with CFG.  A Turbo file put in the RAW slot is
    known by its name only (as Qwen-Image-2.1's Turbo GGUF)."""
    ckpt = (model.get("files") or {}).get("ckpt")
    return model.get("family") == "krea2" and on == "ckpt" and bool(ckpt) and "turbo" not in Path(ckpt).name.lower()


LIGHT_TE = {"krea2": ("qwen3vl_4b_bf16.safetensors", "qwen3vl_4b_fp8_scaled.safetensors")}


def gen_text_encoder(fam: str, te: str) -> str:
    """The text encoder generation uses: for Krea 2, the fp8 one when it is in the text_encoders folders, also for a
    model set up earlier with the bf16 one (8.9GB vs 5.2GB; with the 13GB DiT on a 16GB card, less is swapped)."""
    heavy, light = LIGHT_TE.get(fam, ("", ""))
    if heavy and Path(te).name.lower() == heavy:
        from .models import library
        for d in library._companion_folders("te"):
            hit = next((x for x in d.rglob(light) if x.is_file()), None)
            if hit:
                return str(hit)
    return te


def load_as_fp8(fam: str, dit: str) -> bool:
    """A bf16 / fp16 DiT that does not fit the GPU is loaded with its weights in fp8 (half the size): left as it is,
    ComfyUI keeps most of it in main memory and every step copies it over, many times slower (e.g. Krea 2 Turbo
    bf16, 26GB, on a 16GB card).  Quantized files (fp8 / int8 / nvfp4 / GGUF) are used as they are.  Qwen-Image
    bf16 (41GB) is always loaded as fp8, as before."""
    from .models import read_safetensors_header
    p = Path(dit)
    if fam not in ("krea2", "qwen_image", "qwen_image21", "zimage", "moss_image") or p.suffix.lower() != ".safetensors":
        return False
    try:
        hdr = read_safetensors_header(p)
        size_gb = p.stat().st_size / 2**30
    except (OSError, ValueError):
        return False
    dtypes = {v.get("dtype") for k, v in hdr.items() if k != "__metadata__" and isinstance(v, dict)}
    if not dtypes or dtypes & {"F8_E4M3", "F8_E5M2", "I8", "U8"} or not dtypes & {"BF16", "F16"}:
        return False
    if fam == "qwen_image":
        return True
    from . import gpu
    try:
        vram_gb = max((g.get("vram_total_mb") or 0) for g in gpu.query_gpus()) / 1024
    except Exception:
        vram_gb = 0
    # the weights plus room for the text encoder's turn and the activations; unknown VRAM: 16GB assumed
    return size_gb > (vram_gb or 16) - 3


def eval_model_files(model: dict) -> dict:
    return stage_files(model)


def run_eval(project, run_id: int, eval_id: int, progress_cb=None):
    from .analysis import worker
    run = project.run(run_id)
    snap = run["config"]
    row = project.q("SELECT * FROM evals WHERE id=?", (eval_id,))[0]
    spec = json.loads(row["spec"])
    model = dict(snap["model"])
    fam = model["family"]
    if fam in ("krea2", "qwen_image", "qwen_image21"):   # light generation file registered after the run was created
        from .models import registry
        live = registry.get(model["id"])
        for k in ("turbo", "gen"):
            if live and live["files"].get(k):
                model["files"] = {**model["files"], k: live["files"][k]}
    out_dir = Path(run["run_dir"]) / "eval" / f"{eval_id:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    project.q("UPDATE evals SET status='running' WHERE id=?", (eval_id,))
    bus.publish("eval", project=project.id, run=run_id, eval=eval_id, status="running")

    if not comfy.available():
        raise RuntimeError("評価用 ComfyUI が未導入です（初回セットアップを実行してください）")
    em = eval_model_files(model)
    # other models (merged / fine-tuned) of the same family, looked up now: their files may have moved
    from .models import registry
    extras, notes = [], []
    for mid in spec.get("extra_models") or []:
        m = registry.get(mid)
        if not m or m["missing"] or m.get("family") != fam:
            notes.append(f"比較に使うモデル（{(m or {}).get('name') or mid}）が見つからないか、ファイルがないため、そのモデルでの比較は飛ばしました。")
            continue
        extras.append((m, stage_files(m)))
    # ComfyUI starts (10-30 s when it is not running) while the LoRAs are converted (a Python with torch per file for
    # some families): both are needed before the first image.  The model files are staged above, so the start-up
    # reads the same extra paths as the generation (no restart).  _run_eval_images waits for it (Comfy._lock); a failed
    # start here is tried again there.
    prestart = threading.Thread(target=_start_quietly, daemon=True, name="comfy-prestart")
    prestart.start()
    # LoRA files into ComfyUI's lora folder, converted when the family needs it
    ckpts = {c["step"] if c["step"] else "final": c for c in project.checkpoints(run_id) if c["kind"] == "lora"}
    lora_names = {}
    wanted = list(spec["checkpoints"]) + (list(spec.get("extra_checkpoints") or ["final"]) if extras else [])
    try:
        for key in dict.fromkeys(wanted):
            c = ckpts.get(key) if key == "final" else ckpts.get(int(key))
            if not c:
                continue
            lora_names[str(key)] = staged_lora(project.id, run_id, fam, Path(c["path"]))
    except BaseException:
        prestart.join()
        comfy.park()   # started above: kept for the next job, stopped after a while unused
        raise
    cancelled = []

    def cb(*a):
        # the progress callback is where a cancel arrives (server.EvalCancelled): remember it, so ComfyUI is parked
        # for the next job instead of being stopped as after a failure
        try:
            progress_cb(*a)
        except BaseException:
            cancelled.append(True)
            raise
    try:
        return _run_eval_images(project, run_id, eval_id, cb if progress_cb else None, run, snap, spec, model, fam, out_dir, em,
                                lora_names, worker, extras, notes)
    except BaseException as e:
        if not cancelled:
            comfy.after_failure(e)  # restarted when it may be in a bad state
        raise
    finally:
        comfy.park()  # free VRAM for scoring / training; the process stays for the next comparison


def staged_lora(project_id: str, run_id: int, family: str, src: Path) -> str:
    """The saved LoRA under ComfyUI's loras folder in the layout ComfyUI expects (converted once per saved file and
    reused, also by the export).  Returns the name ComfyUI knows it by."""
    rel = os.path.join("lorastudio", project_id, f"r{run_id}", Path(src).name)  # ComfyUI lists names with os.sep
    comfy_lora_file(family, Path(src), COMFY_MODELS / "loras" / rel)
    return rel


def _negative_for(variant: str, snap: dict) -> str:
    _q, negative = QUALITY.get(variant, ("", ""))
    if negative and trained_with_monochrome(snap):  # empty negative = cfg 1 model, where it has no effect
        negative = f"{negative}, {MONO_NEGATIVE}"
    return negative


class _CancelCheck:
    """The analysis worker, with the cancel check run before each request: scoring a comparison sends a few long
    requests (features of the training and generated images), and a cancel must not wait for all of them."""

    def __init__(self, worker, check):
        self._w, self._check, self.cancelled = worker, check, False

    def call(self, *a, **kw):
        if self._check:
            try:
                self._check()
            except BaseException:
                self.cancelled = True
                raise
        return self._w.call(*a, **kw)

    def __getattr__(self, name):
        return getattr(self._w, name)


def _start_quietly():
    try:
        comfy.ensure_started()
    except Exception:
        pass


def _run_eval_images(project, run_id, eval_id, progress_cb, run, snap, spec, model, fam, out_dir, em, lora_names, worker,
                     extras=(), notes=()):
    t_begin = time.time()
    comfy.ensure_started()

    negative = _negative_for(model["variant"], snap)
    base_col = [("base", None, 0.0)] if spec.get("include_base") else []
    cols = base_col + [(k, lora_names[str(k)], s) for k in spec["checkpoints"] if str(k) in lora_names for s in spec["strengths"]]
    # (model id or None for the training model, model, staged files, columns, negative)
    targets = [(None, model, em, cols, negative)]
    for m, mem in extras:
        mcols = base_col + [(k, lora_names[str(k)], s) for k in spec.get("extra_checkpoints") or ["final"]
                            if str(k) in lora_names for s in spec["strengths"]]
        targets.append((m["id"], m, mem, mcols, _negative_for(m["variant"], snap)))
    total = sum(len(t[3]) for t in targets) * len(spec["prompts"]) * len(spec["seeds"])
    results, warnings = [], []
    log0 = comfy.log_size()
    done = 0
    t0 = time.time()
    # model outermost (switching it loads another checkpoint), then the prompt (its text encoding is reused), then one
    # column after another: switching the LoRA or its strength patches the model again, which on an offloaded large
    # model moves weights each time
    for j, (mid, m, mem, mcols, neg) in enumerate(targets):
        for pi, prompt in enumerate(spec["prompts"]):
            text = prompt if mid is None else prompt_for_model(prompt, model["variant"], m["variant"])
            for ck, lora, strength in mcols:
                for seed in spec["seeds"]:
                    prefix = f"lorastudio_{project.id[:20]}_r{run_id}_e{eval_id}"
                    g = graph_for(m, lora, strength, text, neg, seed + pi, spec["width"], spec["height"], prefix, mem)
                    imgs = comfy.run_prompt(g)
                    src = config.DATA_DIR / "comfy_output" / imgs[0].get("subfolder", "") / imgs[0]["filename"]
                    name = f"p{pi}_s{seed}_{ck}_{strength:.2f}" + (f"_m{j}" if mid else "") + ".png"
                    shutil.move(str(src), out_dir / name)
                    results.append({"prompt": pi, "seed": seed, "ckpt": ck, "strength": strength, "file": name,
                                    **({"model": mid} if mid else {})})
                    done += 1
                    # recorded before the progress callback, which raises when the user cancelled: the finished image stays
                    project.q("UPDATE evals SET results=? WHERE id=?", (json.dumps({"images": results}), eval_id))
                    bus.publish("eval", project=project.id, run=run_id, eval=eval_id, status="running", done=done, total=total)
                    if progress_cb:
                        el = time.time() - t0
                        progress_cb(done, total, int(el / done * (total - done)) if done >= 2 else None)
    warnings = comfy.lora_warnings(log0)
    comfy.park()  # free VRAM for scoring / training

    scores = {}
    w = _CancelCheck(worker, (lambda: progress_cb(done, total, None)) if progress_cb else None)
    try:
        scores = score_eval(project, run, out_dir, results, w, spec["prompts"])
    except Exception as e:  # scores are only an aid; never fail the evaluation because of them
        if w.cancelled:
            raise
        scores = {"error": f"自動スコアを計算できませんでした: {e}"}
    finally:
        try:
            worker.unload()   # training ends the process when it takes the GPU
        except Exception:
            pass
    ok = "error" not in scores
    main = [i for i, r in enumerate(results) if not r.get("model")]
    info = {"images": results, "lora_key_warnings": warnings[:20], "n_lora_key_warnings": len(warnings), "scores": scores,
            "recommend": recommend(snap["kind"], [results[i] for i in main], subset_scores(scores, main, len(results))) if ok else None,
            "conditions": {"base": em, "sampling": sampling_for(model, em), "negative": negative, "width": spec["width"],
                           "height": spec["height"], "evaluated_on": em.get("on", "ckpt")},
            # how long it took (ComfyUI start, images and scores): the basis of the remaining-time estimate of the next
            # training run, whose comparison comes right after it
            "timing": {"seconds": round(time.time() - t_begin), "images": done}}
    if extras or notes:
        info["models"] = [{"id": m["id"], "name": m["name"], "variant": m["variant"],
                           "sampling": {**sampling_for(m, mem), **({"clip_skip": 2} if m["variant"] == "pony" else {})},
                           "negative": _negative_for(m["variant"], snap)} for m, mem in extras]
        info["other_models"] = other_models_summary(snap["kind"], results, scores, extras) if ok else None
        info["notes"] = list(notes)
    project.q("UPDATE evals SET status='done', results=? WHERE id=?", (json.dumps(info, ensure_ascii=False), eval_id))
    bus.publish("eval", project=project.id, run=run_id, eval=eval_id, status="done")
    return info


CCIP_SAME_CHAR = 0.178   # imgutils' published threshold: below it CCIP judges two images the same character
WEAK_GAIN = 0.03         # less improvement over "no LoRA" than this: the character barely shows


def subset_scores(scores: dict, idx: list[int], n: int) -> dict:
    """The per-image score lists cut down to the images idx (other entries are kept as they are)."""
    return {k: [v[i] for i in idx] if isinstance(v, list) and len(v) == n else v for k, v in scores.items()}


def _gain(kind: str, rows: list[dict], ckpt: str, strength: float) -> float | None:
    """How much the LoRA column moved the score away from "no LoRA" on the same model (positive = towards the
    material).  Comparing gains, not raw scores, keeps out how differently two models draw without a LoRA."""
    base = next((r for r in rows if r["ckpt"] == "base"), None)
    col = next((r for r in rows if r["ckpt"] == str(ckpt) and abs(r["strength"] - float(strength)) < 1e-6), None)
    if not base or not col:
        return None
    return base["score"] - col["score"] if kind == "character" else col["score"] - base["score"]


def other_models_summary(kind: str, images: list[dict], scores: dict, extras) -> list[dict]:
    """For each other model: its own recommendation and how strongly the LoRA works there compared with the
    training model, at the same saved version and strength.  Only an aid, like every score here."""
    n = len(images)
    main_idx = [i for i, r in enumerate(images) if not r.get("model")]
    main = recommend(kind, [images[i] for i in main_idx], subset_scores(scores, main_idx, n))
    out = []
    for m, _mem in extras:
        idx = [i for i, r in enumerate(images) if r.get("model") == m["id"]]
        rec = recommend(kind, [images[i] for i in idx], subset_scores(scores, idx, n)) if idx else None
        row = {"id": m["id"], "name": m["name"], "recommend": rec, "ratio": None, "reasons": []}
        if rec and main:
            for r in rec["table"]:
                if r["ckpt"] == "base":
                    continue
                g, g0 = _gain(kind, rec["table"], r["ckpt"], r["strength"]), _gain(kind, main["table"], r["ckpt"], r["strength"])
                if g is not None and g0 and g0 > 0:
                    row["ratio"] = round(g / g0, 2)
                    row["at"] = {"ckpt": r["ckpt"], "strength": r["strength"]}
                    break
        what = {"character": "キャラの特徴", "concept": "覚えさせた特徴", "enhance": "効果"}.get(kind, "画風")
        if row["ratio"] is not None:
            pct = round(row["ratio"] * 100)
            row["reasons"].append(f"LoRA なしからの変化（{what}の出方）は、学習したモデルの約 {pct}% です"
                                  f"（{'最終' if str(row['at']['ckpt']) == 'final' else 'ステップ ' + str(row['at']['ckpt'])}・強さ {row['at']['strength']}）")
            if row["ratio"] < 0.5:
                row["reasons"].append(f"このモデルでは{what}が弱く出ている可能性があります。強さを上げた比較も試してください")
            elif row["ratio"] > 1.5:
                row["reasons"].append(f"このモデルでは{what}が強く出ています。強さを下げると崩れにくくなる場合があります")
        elif rec and rec.get("weak"):
            row["reasons"].append(f"このモデルでは LoRA なしとの差が小さく、{what}があまり出ていない可能性があります")
        out.append(row)
    return out


def recommend(kind: str, images: list[dict], scores: dict) -> dict | None:
    """Which saved version and strength to try first, from the assistive scores (never a guarantee).

    Character: lowest mean CCIP difference to the reference images (the character shows most), among
    near-ties the one that copies the training images least, then the earlier step (less over-fitted).
    Style: highest gain in style similarity over "no LoRA", ties broken the same way.  Returns the pick,
    the reasons in plain words, and whether the effect looks weak compared with no LoRA."""
    if kind == "enhance":
        return _recommend_enhance(images, scores)
    key = {"character": "ccip_diff", "concept": "concept_prob"}.get(kind, "style_sim")
    vals = scores.get(key)
    if not vals or len(vals) != len(images):
        return None
    copy = scores.get("copy_sim") or [None] * len(images)
    groups: dict[tuple, dict] = {}
    for im, v, c in zip(images, vals, copy):
        g = groups.setdefault((str(im["ckpt"]), float(im["strength"])), {"v": [], "c": []})
        g["v"].append(v)
        if c is not None:
            g["c"].append(c)
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    rows = [{"ckpt": k[0], "strength": k[1], "score": round(mean(g["v"]), 4), "copy": round(mean(g["c"]), 4) if g["c"] else None,
             "n": len(g["v"])} for k, g in groups.items()]
    base = next((r for r in rows if r["ckpt"] == "base"), None)
    cands = [r for r in rows if r["ckpt"] != "base"]
    if not cands:
        return None
    better = (lambda r: r["score"]) if kind == "character" else (lambda r: -r["score"])
    best_v = min(better(r) for r in cands)
    near = [r for r in cands if better(r) - best_v <= 0.01]
    step = lambda r: float("inf") if r["ckpt"] == "final" else float(r["ckpt"])
    pick = min(near, key=lambda r: (r["copy"] if r["copy"] is not None else 0, step(r), abs(r["strength"] - 1.0)))
    reasons, weak = [], False
    if kind == "character":
        reasons.append(f"基準画像のキャラとの差（同一性差）の平均が {pick['score']} で、比べた中で最も小さい組み合わせです")
        if base:
            gain = base["score"] - pick["score"]
            weak = gain < WEAK_GAIN
            reasons.append(f"LoRA なし（{base['score']}）より {gain:.3f} 小さくなり、キャラに近づいています" if not weak else
                           f"LoRA なし（{base['score']}）との差が小さく、特徴があまり出ていない可能性があります")
        if pick["score"] <= CCIP_SAME_CHAR:
            reasons.append("キャラ判定 AI では、基準画像と同じキャラと判定される範囲です")
        other = [o for im, o in zip(images, scores.get("ccip_other") or []) if o is not None
                 and (str(im["ckpt"]), float(im["strength"])) == (pick["ckpt"], pick["strength"])]
        if other:   # multi-character run
            gap = sum(other) / len(other) - pick["score"]
            reasons.append(f"ほかのキャラの基準画像との差は平均 {sum(other) / len(other):.3f} です" +
                           ("。呼び出したキャラとの差とあまり変わらないため、キャラどうしが混ざっている可能性があります" if gap < 0.05 else ""))
    elif kind == "concept":
        reasons.append(f"生成画像に特徴のタグが付く確からしさ（タグ判定 AI）の平均が {pick['score']} で、比べた中で最も高い組み合わせです")
        if base:
            gain = pick["score"] - base["score"]
            weak = gain < 0.1
            reasons.append(f"LoRA なし（{base['score']}）より {gain:.3f} 高くなり、特徴が出ています" if not weak else
                           f"LoRA なし（{base['score']}）との差が小さく、特徴があまり出ていない可能性があります")
    else:
        reasons.append(f"素材の画風との類似の平均が {pick['score']} で、比べた中で最も高い組み合わせです")
        if base:
            gain = pick["score"] - base["score"]
            weak = gain < WEAK_GAIN
            reasons.append(f"LoRA なし（{base['score']}）より {gain:.3f} 大きくなり、素材の画風に近づいています" if not weak else
                           f"LoRA なし（{base['score']}）との差が小さく、画風があまり出ていない可能性があります")
    if len(near) > 1:
        reasons.append("差がほとんどない候補が複数あったため、素材の写し込みが少ない方・早い保存版を選びました")
    return {"ckpt": pick["ckpt"], "strength": pick["strength"], "weak": weak, "reasons": reasons,
            "table": sorted(rows, key=lambda r: (r["ckpt"] != "base", step(r) if r["ckpt"] != "base" else 0, r["strength"]))}


def _recommend_enhance(images: list[dict], scores: dict) -> dict | None:
    """Texture LoRA: the effect measure (enhance.measure) of each column against "no LoRA".  The pick is the
    weakest positive strength / earliest checkpoint that already reaches most of the largest gain, because
    pushing further mostly adds artefacts.  Only an aid: the user judges the images."""
    vals = scores.get("effect")
    if not vals or len(vals) != len(images):
        return None
    from .enhance import METRIC_LABEL
    label = METRIC_LABEL.get(scores.get("effect_metric"), "効果")
    groups: dict[tuple, list] = {}
    for im, v in zip(images, vals):
        groups.setdefault((str(im["ckpt"]), float(im["strength"])), []).append(v)
    rows = [{"ckpt": k[0], "strength": k[1], "score": round(sum(g) / len(g), 4), "copy": None, "n": len(g)} for k, g in groups.items()]
    base = next((r for r in rows if r["ckpt"] == "base"), None)
    cands = [r for r in rows if r["ckpt"] != "base" and r["strength"] > 0]
    if not base or not cands:
        return None
    step = lambda r: float("inf") if r["ckpt"] == "final" else float(r["ckpt"])
    gains = {id(r): r["score"] - base["score"] for r in cands}
    top = max(gains.values())
    weak = top < max(0.002, abs(base["score"]) * 0.05)
    ok = [r for r in cands if gains[id(r)] >= top * 0.7] if not weak else cands
    pick = min(ok, key=lambda r: (r["strength"], step(r))) if not weak else max(cands, key=lambda r: gains[id(r)])
    g = gains[id(pick)]
    reasons = [f"画像の{label}（参考の数値）が、LoRA なし（{base['score']}）より {g:+.4f} 変わりました",
               "効果が最も大きい組み合わせの 7 割以上に届く中で、強さが弱く、早い保存版を選んでいます（強くしすぎると崩れやすいため）"]
    if weak:
        reasons = [f"画像の{label}（参考の数値）が、LoRA なし（{base['score']}）からほとんど変わっていません。効果が弱い可能性があります"]
    return {"ckpt": pick["ckpt"], "strength": pick["strength"], "weak": weak, "reasons": reasons,
            "table": sorted(rows, key=lambda r: (r["ckpt"] != "base", step(r) if r["ckpt"] != "base" else 0, r["strength"]))}


FEATURE_CACHE = "eval_features.json"   # in the run folder: features of its frozen training images


def item_features(train_dir: Path, items: list[dict], op: str, worker, timeout: float, cache: Path | None = None) -> dict:
    """Worker features ("embed_images" -> embeds, "ccip_features" -> feats) of training images of a run, in the order
    of items.  The run's dataset is frozen, so with a cache file they are computed once per run and every later
    comparison reuses them; the key holds the image (hash, crop, mask) and the analysis model version."""
    field = "embeds" if op == "embed_images" else "feats"
    paths = [str(train_dir / it["file"]) for it in items]
    if cache is None:
        return worker.call(op, paths=paths, timeout=timeout)
    from .analysis import ANALYZER_VERSION
    key = lambda it: f"{op}|{ANALYZER_VERSION}|{it.get('sha')}|{json.dumps(it.get('crop'))}|{(it.get('mask') or {}).get('key', '')}"
    try:
        store = json.loads(cache.read_text(encoding="utf-8"))
        if not isinstance(store, dict):
            store = {}
    except (OSError, ValueError):
        store = {}
    need = [i for i, it in enumerate(items) if not it.get("sha") or key(it) not in store]
    got = {}
    if need:
        r = worker.call(op, paths=[paths[i] for i in need], timeout=timeout)
        if not r.get("ok"):
            return r
        got = dict(zip(need, r[field]))
        for i, v in got.items():
            if items[i].get("sha"):
                store[key(items[i])] = v
        try:
            config.atomic_write_json(cache, store)
        except OSError:
            pass   # only a speed-up
    return {"ok": True, field: [got[i] if i in got else store[key(it)] for i, it in enumerate(items)]}


def score_eval(project, run, out_dir: Path, results: list[dict], worker, prompts: list[str] | None = None) -> dict:
    """Assistive scores only (not success probabilities):
    * character (illustration): CCIP difference to the reference images (lower = closer identity)
    * all: max similarity to any training image (high = possibly copying the training images)
    * style: similarity to the training-set style centroid relative to the base image"""
    import numpy as np
    snap = run["config"]
    if snap.get("enhance"):
        # measured on the images themselves (no AI model); a custom slider has no fixed measure
        from .enhance import effect, measure
        metric = effect(snap["enhance"]["effect"])["metric"]
        if not metric:
            return {}
        return {"effect": [round(measure(out_dir / r["file"], metric), 4) for r in results], "effect_metric": metric}
    train_dir = Path(run["run_dir"]) / "dataset"
    cache = Path(run["run_dir"]) / FEATURE_CACHE
    train_items = snap["items"][:200]
    gen = [str(out_dir / r["file"]) for r in results]
    r1 = item_features(train_dir, train_items, "embed_images", worker, 1800, cache)
    r2 = worker.call("embed_images", paths=gen, timeout=1800)
    if not (r1.get("ok") and r2.get("ok")):
        return {"error": (r1.get("error") or r2.get("error"))}
    T = np.array(r1["embeds"])
    G = np.array(r2["embeds"])
    sim = G @ T.T
    out = {"copy_sim": [round(float(v), 4) for v in sim.max(1)]}
    if snap["kind"] == "concept":
        from .captions import concept_match, norm_tag
        words = snap.get("concept_words") or []
        if words:
            r = worker.call("tag_images", paths=gen, timeout=1800)
            if r.get("ok"):
                out["concept_prob"] = [round(max([v for t, v in tags.items() if concept_match(norm_tag(t), words)], default=0.0), 4)
                                       for tags in r["tags"]]
    elif snap["kind"] == "style":
        c = T.mean(0)
        c /= np.linalg.norm(c)
        out["style_sim"] = [round(float(v), 4) for v in G @ c]
    elif snap.get("members"):
        out.update(_member_ccip(snap, train_dir, gen, results, prompts or [], worker, cache))
    else:
        meta = project.get_meta()
        refs = meta.get("reference_ids") or meta.get("auto_reference") or []
        ref_items = [it for it in snap["items"] if it["id"] in refs] or train_items[:3]
        rf = item_features(train_dir, ref_items, "ccip_features", worker, 600, cache)
        gf = worker.call("ccip_features", paths=gen, timeout=1800)
        if rf.get("ok") and gf.get("ok"):
            d = worker.call("ccip_diff", feats=gf["feats"], refs=rf["feats"], timeout=600)
            if d.get("ok"):
                out["ccip_diff"] = [round(min(row), 4) for row in d["matrix"]]
    return out


def _member_ccip(snap: dict, train_dir: Path, gen: list[str], results: list[dict], prompts: list[str], worker,
                 cache: Path | None = None) -> dict:
    """Multi-character run: each image is compared with the reference images of the character its prompt calls
    (ccip_diff, lower = closer), and also with the other characters (ccip_other: when it is about as low, the
    characters are mixing)."""
    from .multichar import member_of_prompt
    members = snap["members"]
    ref_items, owner = [], []
    for k, m in enumerate(members):
        lo, hi = m["id_range"]
        mine = [it for it in snap["items"] if lo <= it["id"] < hi]
        picked = [it for it in mine if it["id"] in set(m.get("refs") or [])] or mine[:3]
        ref_items += picked
        owner += [k] * len(picked)
    if not ref_items:
        return {}
    rf = item_features(train_dir, ref_items, "ccip_features", worker, 600, cache)
    gf = worker.call("ccip_features", paths=gen, timeout=1800)
    if not (rf.get("ok") and gf.get("ok")):
        return {}
    d = worker.call("ccip_diff", feats=gf["feats"], refs=rf["feats"], timeout=600)
    if not d.get("ok"):
        return {}
    own, other = [], []
    for r, row in zip(results, d["matrix"]):
        k = member_of_prompt(members, prompts[r["prompt"]] if r["prompt"] < len(prompts) else "")
        mine = [v for v, o in zip(row, owner) if o == k] or list(row)
        rest = [v for v, o in zip(row, owner) if k is not None and o != k]
        own.append(round(min(mine), 4))
        other.append(round(min(rest), 4) if rest else None)
    return {"ccip_diff": own, "ccip_other": other}
