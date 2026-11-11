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


_read_cache: dict[str, tuple] = {}   # wid -> ((mtime_ns, size), content): the page asks for the same workflow on every update


def _read(wid) -> dict | None:
    if not isinstance(wid, str) or not re.fullmatch(r"[0-9a-f]{12}", wid):
        return None
    p = DIR / f"{wid}.json"
    try:
        st = p.stat()
        sig = (st.st_mtime_ns, st.st_size)
        hit = _read_cache.get(wid)
        if hit and hit[0] == sig:
            return hit[1]
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        _read_cache.pop(wid, None)
        return None
    if len(_read_cache) >= 32:
        _read_cache.clear()
    _read_cache[wid] = (sig, d)
    return d


def load(wid) -> dict | None:
    d = _read(wid)
    return copy.deepcopy(d["graph"]) if d else None   # the caller's own: the kept one is never changed


def meta(wid) -> dict | None:
    d = _read(wid)
    return dict(d["meta"]) if d else None


# ------------------------------------------------------------------------------------- prompt text
def _text_slots(graph: dict, ref, side: str, depth: int = 0) -> list[tuple[str, str, str]]:
    """(node id, input, text) of the text nodes a sampler's positive / negative comes from (pnginfo._comfy_texts)."""
    if depth > 20 or not (isinstance(ref, list) and ref):
        return []
    nid = str(ref[0])
    node = graph.get(nid)
    if not isinstance(node, dict) or node.get("class_type") == "ConditioningZeroOut":   # no text: the zeroed one is not it
        return []
    inputs = node.get("inputs") or {}
    out = []
    for key in TEXT_KEYS:
        v = inputs.get(key)
        if isinstance(v, str):   # an empty one too: a workflow's blank negative is still where the negative goes
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
    return "\n\n".join(dict.fromkeys(t for _, _, t in slots if t))


def apply_prompts(graph: dict, positive: str, negative: str) -> dict:
    """The page's prompt / negative put in the workflow's text nodes.  Unchanged text leaves the graph as it is; a
    changed one is written where the workflow has a single text (also when that text is in several inputs)."""
    g = copy.deepcopy(graph)
    for side, new, what in (("positive", positive, "プロンプト"), ("negative", negative, "ネガティブ")):
        slots = text_slots(g, side)
        new = (new or "").strip()
        if new == joined(slots):
            continue
        texts = list(dict.fromkeys(t for _, _, t in slots if t))
        if len(texts) > 1:
            raise ValueError(f"この workflow は{what}が {len(texts)} か所に分かれているため、この画面で変えた{what}を入れられません。"
                             f"{what}を読み込んだときのままに戻すか、workflow を使わずに生成してください。")
        if not slots:
            # no negative text in the workflow (ConditioningZeroOut, CFG 1 ...): generated without it, as the workflow
            # was made (the page says so next to the field); a prompt with nowhere to go still stops it
            if new and side == "positive":
                raise ValueError(f"この workflow には{what}を書く場所が見つからないため、{what}を入れられません。")
            continue
        # where the text is (an SDXL encoder's empty text_l stays empty); all blank: every blank one
        for nid, key, t in slots:
            if t or not texts:
                g[nid]["inputs"][key] = new
    return g


def has_negative(graph: dict) -> bool:
    """Whether the workflow has a negative text to write in."""
    return bool(text_slots(graph, "negative"))


def cfg_of(graph: dict) -> float | None:
    """The CFG of the workflow's main sampler / guider (a linked primitive followed), None when not found."""
    for _, node in _nodes(graph):
        if not re.search(r"Sampler|Guider", str(node.get("class_type", ""))):
            continue
        v = (node.get("inputs") or {}).get("cfg")
        for _ in range(5):
            if not isinstance(v, list) or not v:
                break
            vals = [x for x in ((graph.get(str(v[0])) or {}).get("inputs") or {}).values() if not isinstance(x, list)]
            v = vals[0] if len(vals) == 1 else None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


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


def same_names(name: str) -> set[str]:
    """The stems a file is known by (SAME_FILE), itself included."""
    s = _stem(name)
    return next(({_stem(n) for n in grp} for grp in SAME_FILE if s in {_stem(n) for n in grp}), {s})


def nested_loras(inputs: dict) -> list[tuple[str, dict]]:
    """(input, value) of the LoRAs a node keeps as {"on", "lora", "strength"} values of its own (rgthree's Power Lora
    Loader: lora_1, lora_2, ...), not as a lora_name input."""
    return [(k, v) for k, v in (inputs or {}).items()
            if isinstance(v, dict) and isinstance(v.get("lora"), str) and v["lora"].strip() and v["lora"] != "None"]


def lora_in_use(v: dict) -> bool:
    """A nested LoRA that is switched on (rgthree skips the ones turned off or at strength 0)."""
    two = v.get("strengthTwo")
    return bool(v.get("on", True)) and (v.get("strength") != 0 or two not in (None, 0))


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
        for _key, v in nested_loras(inputs):   # the LoRA's own name, or another name of the very same file
            names = same_names(v["lora"])
            hit = next((p for p in files.get("loras", []) if _stem(p) == _stem(v["lora"])), None) \
                or next((p for p in files.get("loras", []) if _stem(p) in names), None)
            if hit:
                if _stem(hit) != _stem(v["lora"]):
                    notes.append(f"workflow の「{Path(v['lora'].replace(chr(92), '/')).name}」は、このパソコンの"
                                 f"「{Path(hit).name}」と同じファイルなので、それを使いました。")
                v["lora"] = stage("loras", hit)
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


# ------------------------------------------------------------------------------------- the empty image
# Which node makes the workflow's empty image is not decided by its name: a node that outputs a LATENT and takes
# no image / latent / model of its own (by ComfyUI's /object_info; a node ComfyUI does not have: "Empty...Latent"
# in its name) is the source.  Its size is its width / height, or a "832 x 1216" choice.
_TAKES = {"LATENT", "IMAGE", "MASK", "MODEL", "VAE", "CLIP", "CONDITIONING"}


def _spec_inputs(spec: dict) -> dict:
    return {**((spec.get("input") or {}).get("required") or {}), **((spec.get("input") or {}).get("optional") or {})}


def _latent_outputs(node: dict, info: dict | None) -> set[int] | None:
    """The output slots giving a LATENT when the node makes an empty image from nothing; None when it does not."""
    spec = (info or {}).get(str(node.get("class_type")))
    if spec:
        outs = [i for i, t in enumerate(spec.get("output") or []) if t == "LATENT"]
        takes = {v[0] for v in _spec_inputs(spec).values() if isinstance(v, (list, tuple)) and v and isinstance(v[0], str)}
        return set(outs) if outs and not (takes & _TAKES) else None
    return {0} if re.search(r"Empty.*Latent|Latent.*Empty", str(node.get("class_type", "")), re.I) else None


def _size_of(inputs: dict) -> tuple[int, int] | None:
    w, h = inputs.get("width"), inputs.get("height")
    if isinstance(w, int) and isinstance(h, int):
        return w, h
    for v in inputs.values():
        if isinstance(v, str) and (m := re.search(r"(\d+)\s*x\s*(\d+)", v)):
            return int(m.group(1)), int(m.group(2))
    return None


def _refs(graph: dict, nid: str) -> list[tuple[dict, str, int]]:
    """(inputs, key, output slot) of every link from node nid."""
    return [(n["inputs"], k, v[1]) for _, n in _nodes(graph) for k, v in (n.get("inputs") or {}).items()
            if isinstance(v, list) and len(v) == 2 and str(v[0]) == nid]


def substitute(graph: dict, family: str | None, info: dict | None = None) -> tuple[dict, list[str]]:
    """The workflow's empty image made in the shape the model takes.  A node making another shape (an SDXL latent
    for MOSS-Image: "expected ... 64 channels, but got 128", 2026-11-07) has its LATENT outputs taken by the model's
    own empty-image node of the same size; the node stays for its other outputs (sizes ...) and goes when nothing
    else uses it.  family None (model not known): left as it is."""
    g = copy.deepcopy(graph)
    notes = []
    if not family:
        return g, notes
    from .evaluate import latent_class
    want = latent_class(family)
    for nid, node in _nodes(graph):
        outs = _latent_outputs(node, info)
        size = _size_of(node.get("inputs") or {})
        if outs is None or not size or node.get("class_type") == want:
            continue
        bs = (node.get("inputs") or {}).get("batch_size", 1)
        new = f"{nid}_latent"
        g[new] = {"class_type": want, "inputs": {"width": size[0], "height": size[1], "batch_size": bs if isinstance(bs, int) else 1}}
        rest = False
        for inputs, key, slot in _refs(g, nid):
            if slot in outs:
                inputs[key] = [new, 0]
            else:
                rest = True
        if not rest:
            del g[nid]
        notes.append(f"workflow の「{node.get('class_type')}」の代わりに、このモデルに合った ComfyUI の「{want}」で、"
                     "同じ大きさの空の画像を作って生成しました。")
    return g, notes


def latent_size(graph: dict) -> tuple[int, int] | None:
    """The size of the empty image the workflow starts from."""
    for _, node in _nodes(graph):
        if _latent_outputs(node, None) is not None and (size := _size_of(node.get("inputs") or {})):
            return size
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
    # an empty-image node whose image only is used is replaced by the model's own (substitute): not missing
    gone = {nid for nid, n in _nodes(graph) if _latent_outputs(n, None) == {0} and _size_of(n.get("inputs") or {})
            and all(slot == 0 for _, _, slot in _refs(graph, nid))}
    return sorted({str(n.get("class_type")) for nid, n in _nodes(graph) if nid not in gone} - known)


def check(graph: dict, info: dict) -> list[str]:
    """What the running ComfyUI (its /object_info) would refuse, in words: missing nodes, files it cannot find."""
    remember_nodes(info)
    errors = []
    if miss := missing_nodes(graph, set(info)):
        errors.append("この workflow には、ここの ComfyUI に入っていないノードがあります: " + "、".join(miss) +
                      "。カスタムノードは、このアプリが入れたもの（ComfyUI-GGUF・ComfyUI-ppm・rgthree-comfy など）しか使えません。")
    loras = _lora_options(info)
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
        # a nested LoRA (rgthree's Power Lora Loader) the ComfyUI here does not have is skipped by that node without
        # a word, and the image comes out without it: said here, as for every other loader
        for _key, v in nested_loras(node.get("inputs") or {}):
            if loras is not None and lora_in_use(v) and v["lora"] not in loras and _stem(v["lora"]) not in {_stem(x) for x in loras}:
                errors.append(f"「{Path(v['lora'].replace(chr(92), '/')).name}」が見つかりません（{node.get('class_type')} の LoRA）。"
                              "モデルのフォルダーに置いてください。")
    return errors


def _lora_options(info: dict) -> list[str] | None:
    """The LoRA files the running ComfyUI lists (LoraLoader's lora_name choices), None when it does not say."""
    try:
        spec = info["LoraLoader"]["input"]["required"]["lora_name"]
    except (KeyError, TypeError):
        return None
    opts = spec[0] if isinstance(spec, (list, tuple)) and spec else None
    if isinstance(opts, list):
        return [str(x) for x in opts]
    more = spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], dict) else {}
    return [str(x) for x in more["options"]] if isinstance(more.get("options"), list) else None


def summary(info: dict) -> str:
    """One line about a workflow from what pnginfo read: model and LoRAs (the size is the page's)."""
    parts = []
    if info.get("model_name"):
        parts.append("モデル " + Path(str(info["model_name"]).replace("\\", "/")).name)
    if info.get("loras"):
        parts.append("LoRA " + "、".join(f"{Path(str(lo['name']).replace(chr(92), '/')).name}（{lo['strength']:g}）" for lo in info["loras"]))
    return " ／ ".join(parts)
