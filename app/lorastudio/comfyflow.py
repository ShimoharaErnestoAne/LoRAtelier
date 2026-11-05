"""画像生成の「workflow のまま生成」: a ComfyUI workflow read from an image (its API-format "prompt" graph) run as it is.

The form's fields cannot say everything a workflow does (a second sampler for hires, upscalers, detailers ...), so
the graph itself is queued; only what this page owns is changed:
  - the file names (checkpoint, diffusion model, VAE, text encoder, LoRA) are pointed at the same files on this PC,
    found by name in this app's model folders and exposed to ComfyUI as every other file (Comfy.stage_model);
    a name not found there is left as it is (ComfyUI's own models folder may have it)
  - the positive / negative text, when the user changed it on the page
  - the seed of the main sampler (count > 1: consecutive seeds), and where the images are saved
Before it is queued, the graph is checked against the running ComfyUI's /object_info: nodes it does not have
(custom nodes are off except the ones this app installs) and files it cannot find are said by name.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

from . import config

DIR = config.DATA_DIR / "imagegen" / "workflows"
# loader inputs naming a model file -> the ComfyUI folder it is staged in
FILE_INPUTS = {"ckpt_name": "checkpoints", "unet_name": "diffusion_models", "vae_name": "vae", "lora_name": "loras",
               "clip_name": "text_encoders", "clip_name1": "text_encoders", "clip_name2": "text_encoders",
               "clip_name3": "text_encoders", "clip_name4": "text_encoders"}
TEXT_KEYS = ("text", "text_g", "text_l", "prompt", "string", "value", "t5xxl", "clip_l")
_known: set[str] | None = None   # node classes of the last ComfyUI asked (for saying missing nodes when an image is read)


def _stem(name: str) -> str:
    base = re.split(r"[\\/]", str(name or ""))[-1]
    return re.sub(r"\.(safetensors|ckpt|pt|pth|bin|gguf|sft)$", "", base, flags=re.I).strip().lower()


def _nodes(graph: dict) -> list[tuple[str, dict]]:
    return [(k, n) for k, n in sorted(graph.items(), key=lambda kv: (len(str(kv[0])), str(kv[0]))) if isinstance(n, dict)]


# ------------------------------------------------------------------------------------- stored workflows
def save(graph: dict, source: str, summary: str) -> dict:
    """Keep a workflow read from an image (one file per content).  Returns its meta."""
    if not isinstance(graph, dict) or not graph:
        raise ValueError("ComfyUI の workflow ではありません")
    body = json.dumps(graph, ensure_ascii=False, sort_keys=True)
    wid = hashlib.sha1(body.encode("utf-8")).hexdigest()[:12]
    DIR.mkdir(parents=True, exist_ok=True)
    meta = {"id": wid, "source": source, "summary": summary, "nodes": len(graph)}
    p = DIR / f"{wid}.json"
    if not p.exists():
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"meta": meta, "graph": graph}, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)
    return meta


def _read(wid) -> dict | None:
    if not isinstance(wid, str) or not re.fullmatch(r"[0-9a-f]{12}", wid):
        return None
    try:
        return json.loads((DIR / f"{wid}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load(wid) -> dict | None:
    d = _read(wid)
    return d["graph"] if d else None


def meta(wid) -> dict | None:
    d = _read(wid)
    return d["meta"] if d else None


# ------------------------------------------------------------------------------------- prompt text
def _text_slots(graph: dict, ref, side: str, depth: int = 0) -> list[tuple[str, str, str]]:
    """(node id, input, text) of the text nodes a sampler's positive / negative comes from (pnginfo._comfy_texts)."""
    if depth > 20 or not (isinstance(ref, list) and ref):
        return []
    nid = str(ref[0])
    node = graph.get(nid)
    if not isinstance(node, dict):
        return []
    inputs = node.get("inputs") or {}
    out = []
    for key in TEXT_KEYS:
        v = inputs.get(key)
        if isinstance(v, str) and v.strip():
            out.append((nid, key, v.strip()))
        elif isinstance(v, list):
            out += _text_slots(graph, v, side, depth + 1)
    for key in (side, "conditioning", "conditioning_1", "conditioning_2", "conditioning_to", "conditioning_from"):
        if isinstance(inputs.get(key), list):
            out += _text_slots(graph, inputs[key], side, depth + 1)
    return out


def text_slots(graph: dict, side: str) -> list[tuple[str, str, str]]:
    seen, out = set(), []
    for _, node in _nodes(graph):
        if not re.search(r"Sampler|Guider", str(node.get("class_type", ""))):
            continue
        inputs = node.get("inputs") or {}
        ref = inputs.get("positive") or inputs.get("conditioning") if side == "positive" else inputs.get("negative")
        for slot in _text_slots(graph, ref, side):
            if slot[:2] not in seen:
                seen.add(slot[:2])
                out.append(slot)
    return out


def joined(slots) -> str:
    return "\n\n".join(dict.fromkeys(t for _, _, t in slots))


def apply_prompts(graph: dict, positive: str, negative: str) -> dict:
    """The page's prompt / negative put in the workflow's text nodes.  Unchanged text leaves the graph as it is; a
    changed one is written where the workflow has a single text (also when that text is in several inputs)."""
    g = copy.deepcopy(graph)
    for side, new, what in (("positive", positive, "プロンプト"), ("negative", negative, "ネガティブ")):
        slots = text_slots(g, side)
        new = (new or "").strip()
        if new == joined(slots):
            continue
        texts = list(dict.fromkeys(t for _, _, t in slots))
        if len(texts) > 1:
            raise ValueError(f"この workflow は{what}が {len(texts)} か所に分かれているため、この画面で変えた{what}を入れられません。"
                             f"{what}を読み込んだときのままに戻すか、workflow を使わずに生成してください。")
        if not slots:
            if new:
                raise ValueError(f"この workflow には{what}を書く場所が見つからないため、{what}を入れられません。")
            continue
        for nid, key, _ in slots:
            g[nid]["inputs"][key] = new
    return g


# ------------------------------------------------------------------------------------- seed / outputs
def _seed_slot(graph: dict) -> tuple[str, str] | None:
    """Where the main sampler's seed is: its own input, or the primitive node it is linked to."""
    for nid, node in _nodes(graph):
        if not re.search(r"Sampler|Noise", str(node.get("class_type", ""))):
            continue
        inputs = node.get("inputs") or {}
        for key in ("seed", "noise_seed"):
            if key not in inputs:
                continue
            v, at = inputs[key], (nid, key)
            for _ in range(5):
                if not isinstance(v, list):
                    return at
                src = graph.get(str(v[0])) or {}
                scalars = [(k, x) for k, x in (src.get("inputs") or {}).items() if not isinstance(x, list)]
                if len(scalars) != 1:
                    return None
                at, v = (str(v[0]), scalars[0][0]), scalars[0][1]
            return None
    return None


def seed_of(graph: dict) -> int | None:
    at = _seed_slot(graph)
    try:
        return int(graph[at[0]]["inputs"][at[1]]) if at else None
    except (TypeError, ValueError):
        return None


def for_run(graph: dict, seed: int | None, prefix: str) -> dict:
    """One run: the seed set, and every image saved by ComfyUI under this page's prefix (a workflow that only
    previews gets its preview nodes saved)."""
    g = copy.deepcopy(graph)
    if seed is not None and (at := _seed_slot(g)):
        g[at[0]]["inputs"][at[1]] = int(seed)
    saves = [n for _, n in _nodes(g) if n.get("class_type") == "SaveImage"]
    if not saves:
        for _, n in _nodes(g):
            if n.get("class_type") == "PreviewImage":
                n["class_type"] = "SaveImage"
                saves.append(n)
    for n in saves:
        n.setdefault("inputs", {})["filename_prefix"] = prefix
    return g


# ------------------------------------------------------------------------------------- files / checks
def local_files() -> dict[str, list[str]]:
    """The model files in this app's folders, by ComfyUI folder kind."""
    from .models import library
    scan = library.scan()
    comp = library.companion_files()
    ck = [c["path"] for c in scan["checkpoints"]]
    return {"checkpoints": ck, "diffusion_models": ck, "loras": [x["path"] for x in scan["loras"]],
            "vae": comp["vae"], "text_encoders": comp["te"]}


# the role of a loader input in a model set up here (its files[...] slot).  Text encoder / VAE: another version of the
# same one (bf16 / int8 of Qwen3.5-9B) works in its place.  LoRAs are not: only another name of the very same file
# (SAME_FILE) is taken, never the model's other fast LoRA (the SFT 8-step DMD is not Base's DMD2)
ROLE = {"clip_name": "te", "clip_name1": "te", "vae_name": "vae"}
# names other people / this app give the same file.  lodestones/MOSS-Image-Base-8-steps-dmd lora_dmd2 = lodestones/Kromoss
# 8-steps-dmd2-lora (the same sha256 7683fe1f..., checked 2026-11-05); this app downloads it as moss_image_8steps_dmd2
SAME_FILE = [
    ["moss_image_8steps_dmd2.safetensors", "lora_dmd2.safetensors", "8-steps-dmd2-lora.safetensors",
     "MOSS-Image-Base-8-steps-dmd2.safetensors"],
]


def resolve_files(graph: dict, files: dict[str, list[str]], stage, models: list[dict] = ()) -> tuple[dict, list[str], dict | None]:
    """File names pointed at this PC's files (stage(kind, path) -> the name ComfyUI knows it by).
    A name not here that is another version / name of a file the workflow's model is set up with here (e.g.
    qwen3.5_9b_bf16 for the int8 text encoder this PC has) takes that file, and is said in the notes.
    Returns (graph, notes, the set-up model the workflow uses or None)."""
    from .models import COMPANIONS
    g = copy.deepcopy(graph)
    model, todo, notes = None, [], []
    for _, node in _nodes(g):
        inputs = node.get("inputs") or {}
        for key, kind in FILE_INPUTS.items():
            v = inputs.get(key)
            if not isinstance(v, str) or not v.strip():
                continue
            hit = next((p for p in files.get(kind, []) if _stem(p) == _stem(v)), None)
            if hit:
                inputs[key] = stage(kind, hit)
                if key in ("ckpt_name", "unet_name") and model is None:
                    model = next((m for m in models if any(str(m["files"].get(s) or "").lower() == hit.lower()
                                                           for s in ("gen", "turbo", "ckpt"))), None)
            else:
                todo.append((inputs, key, kind, v))
    for inputs, key, kind, v in todo:
        name = Path(v.replace(chr(92), "/")).name
        group = next(({_stem(n) for n in grp} for grp in SAME_FILE if _stem(v) in {_stem(n) for n in grp}), None)
        if group:   # the same file under another name
            hit = next((p for p in files.get(kind, []) if _stem(p) in group), None)
            if hit:
                inputs[key] = stage(kind, hit)
                notes.append(f"workflow の「{name}」は、このパソコンの「{Path(hit).name}」と同じファイルなので、それを使いました。")
            continue
        role = ROLE.get(key)
        if not model or not role:
            continue
        names = {_stem(n) for n in COMPANIONS.get(model["family"], {}).get(role, [])}
        if _stem(v) not in names:
            continue
        own = model["files"].get(role)
        hit = own if own and Path(own).is_file() else next((p for p in files.get(kind, []) if _stem(p) in names), None)
        if hit:
            inputs[key] = stage(kind, hit)
            notes.append(f"workflow の「{name}」はこのパソコンにないため、同じものの別の版の「{Path(hit).name}」を使いました"
                         "（細部が元の画像と変わることがあります）。")
    return g, notes, model


# rgthree's 「SDXL Empty Latent Image」 (a custom node, not here): dimensions as "832 x 1216 (portrait)" and the size
# for CLIPTextEncodeSDXL as outputs 1 / 2; it is the empty latent of ComfyUI itself with a size list
RGTHREE_LATENT = "SDXL Empty Latent Image (rgthree)"
LATENT_CLASS = {"sdxl": "EmptyLatentImage", "qwen_image21": "EmptyLatentImage", "moss_image": "EmptyFlux2LatentImage"}


def substitute(graph: dict, family: str | None) -> tuple[dict, list[str]]:
    """Custom nodes that only wrap what ComfyUI has, put back as ComfyUI's own nodes."""
    g = copy.deepcopy(graph)
    notes = []
    for nid, node in _nodes(g):
        if node.get("class_type") != RGTHREE_LATENT:
            continue
        inputs = node.get("inputs") or {}
        m = re.search(r"(\d+)\s*x\s*(\d+)", str(inputs.get("dimensions") or ""))
        if not m:
            continue
        w, h = int(m.group(1)), int(m.group(2))
        scale = inputs.get("clip_scale", 1)
        scale = scale if isinstance(scale, (int, float)) else 1
        cls = LATENT_CLASS.get(family or "", "EmptySD3LatentImage" if family else "EmptyLatentImage")
        g[nid] = {"class_type": cls, "inputs": {"width": w, "height": h, "batch_size": inputs.get("batch_size", 1)}}
        for _, other in _nodes(g):   # CLIP_WIDTH / CLIP_HEIGHT outputs used elsewhere: the numbers themselves
            for k, v in (other.get("inputs") or {}).items():
                if isinstance(v, list) and len(v) == 2 and str(v[0]) == nid and v[1] in (1, 2):
                    other["inputs"][k] = int((w if v[1] == 1 else h) * scale)
        notes.append("rgthree の「SDXL Empty Latent Image」は、同じ働きをする ComfyUI の標準のノードに置き換えて生成しました。")
    return g, notes


def latent_size(graph: dict) -> tuple[int, int] | None:
    """The size of the empty image the workflow starts from."""
    for _, node in _nodes(graph):
        ct, inputs = str(node.get("class_type", "")), node.get("inputs") or {}
        if ct == RGTHREE_LATENT and (m := re.search(r"(\d+)\s*x\s*(\d+)", str(inputs.get("dimensions") or ""))):
            return int(m.group(1)), int(m.group(2))
        if re.search(r"Empty.*Latent", ct) and isinstance(inputs.get("width"), int) and isinstance(inputs.get("height"), int):
            return inputs["width"], inputs["height"]
    return None


def apply_size(graph: dict, width, height) -> dict:
    """The page's image size put in the workflow's empty latent (after substitute)."""
    g = copy.deepcopy(graph)
    if not width or not height:
        return g
    for _, node in _nodes(g):
        inputs = node.get("inputs") or {}
        if re.search(r"Empty.*Latent", str(node.get("class_type", ""))) and "width" in inputs and "height" in inputs:
            for key, v in (("width", width), ("height", height)):
                if not isinstance(inputs[key], list):
                    inputs[key] = int(v)
    return g


def remember_nodes(info: dict):
    global _known
    _known = set(info)


def missing_nodes(graph: dict, known: set[str] | None = None) -> list[str] | None:
    """Node classes the ComfyUI here does not have; None when not known yet (ComfyUI not asked since start)."""
    known = known if known is not None else _known
    if known is None:
        return None
    graph = substitute(graph, None)[0]   # what is put back as ComfyUI's own nodes is not missing
    return sorted({str(n.get("class_type")) for _, n in _nodes(graph)} - known)


def check(graph: dict, info: dict) -> list[str]:
    """What the running ComfyUI (its /object_info) would refuse, in words: missing nodes, files it cannot find."""
    remember_nodes(info)
    errors = []
    if miss := missing_nodes(graph, set(info)):
        errors.append("この workflow には、ここの ComfyUI に入っていないノードがあります: " + "、".join(miss) +
                      "。カスタムノードは、このアプリが入れたもの（ComfyUI-GGUF・ComfyUI-ppm など）しか使えません。")
    for nid, node in _nodes(graph):
        spec = info.get(str(node.get("class_type")))
        if not spec:
            continue
        req = {**((spec.get("input") or {}).get("required") or {}), **((spec.get("input") or {}).get("optional") or {})}
        for key, v in (node.get("inputs") or {}).items():
            opts = (req.get(key) or [None])[0]
            if isinstance(v, str) and isinstance(opts, list) and opts and v not in opts and key in FILE_INPUTS:
                errors.append(f"「{Path(v.replace(chr(92), '/')).name}」が見つかりません（{node.get('class_type')} の {key}）。"
                              "モデルのフォルダーに置いてください。")
    return errors


def summary(info: dict) -> str:
    """One line about a workflow from what pnginfo read: model and LoRAs (the size is the page's)."""
    parts = []
    if info.get("model_name"):
        parts.append("モデル " + Path(str(info["model_name"]).replace("\\", "/")).name)
    if info.get("loras"):
        parts.append("LoRA " + "、".join(f"{Path(str(lo['name']).replace(chr(92), '/')).name}（{lo['strength']:g}）" for lo in info["loras"]))
    return " ／ ".join(parts)
