"""Generation settings read back from an image file (画像生成ページの「画像から設定を読み込む」).

What other tools write into their images, read the same way as としあきi2t (engines/i2t/i2t.py
generation_info_from_image), but split into the fields of this app's generation form:
  A1111 / Forge: PNG tEXt "parameters" (JPEG / WebP: EXIF UserComment), modules/processing.py create_infotext:
    "<prompt>\\nNegative prompt: <negative>\\nSteps: 20, Sampler: Euler a, ..."; LoRAs are <lora:name:weight> in the prompt
  ComfyUI: PNG "prompt" (the API-format node graph written by SaveImage); "workflow" (the screen layout) is not needed
  NovelAI: PNG "Comment" (JSON), Fooocus: "parameters" as JSON, InvokeAI: "invokeai_metadata"
Images whose metadata was stripped (posted to SNS, screenshots) have nothing to read.

The result names files (model, LoRAs) only by name; the server matches them with this PC's files.
"""
from __future__ import annotations

import io
import json
import re
from typing import Any

from PIL import Image

# A1111 sampler names -> ComfyUI sampler_name (sd_samplers_kdiffusion.py / ComfyUI comfy/samplers.py)
A1111_SAMPLERS = {
    "euler a": "euler_ancestral", "euler": "euler", "lms": "lms", "heun": "heun", "dpm2": "dpm_2",
    "dpm2 a": "dpm_2_ancestral", "dpm++ 2s a": "dpmpp_2s_ancestral", "dpm++ 2m": "dpmpp_2m", "dpm++ sde": "dpmpp_sde",
    "dpm++ 2m sde": "dpmpp_2m_sde", "dpm++ 2m sde gpu": "dpmpp_2m_sde_gpu", "dpm++ 3m sde": "dpmpp_3m_sde",
    "dpm fast": "dpm_fast", "dpm adaptive": "dpm_adaptive", "lcm": "lcm", "ddim": "ddim", "ddpm": "ddpm",
    "unipc": "uni_pc", "uni_pc": "uni_pc", "ipndm": "ipndm", "deis": "deis", "euler cfg++": "euler_cfg_pp",
    "euler a cfg++": "euler_ancestral_cfg_pp", "res_multistep": "res_multistep", "er sde": "er_sde",
}
# A1111 "Schedule type" (and the old " Karras" suffix of the sampler name) -> ComfyUI scheduler
A1111_SCHEDULERS = {
    "karras": "karras", "exponential": "exponential", "sgm uniform": "sgm_uniform", "simple": "simple",
    "normal": "normal", "uniform": "normal", "beta": "beta", "kl optimal": "kl_optimal", "ddim": "ddim_uniform",
    "align your steps": "ays",
}
# the parameter part of an infotext line (A1111 modules/infotext_utils.py re_param)
RE_PARAM = re.compile(r'\s*(\w[\w \-/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)')
RE_LORA = re.compile(r"<lora:([^:>]+)(?::([-+]?[0-9.]+))?[^>]*>", re.I)
# imagegen.run: the SaveImage prefix of this app's own images (+ the per-prompt tag of evaluate.fresh_save_prefixes)
RE_OWN = re.compile(r"^lsimg_([0-9a-f]{12})(?:_[0-9a-f]{8})?$")


def _num(v: Any, kind=float):
    try:
        return kind(v)
    except (TypeError, ValueError):
        return None


def _blank() -> dict:
    return {"source": "", "positive": "", "negative": "", "width": None, "height": None, "seed": None,
            "sampling": {}, "model_name": None, "loras": [], "notes": [], "vae_name": None, "te_name": None, "attention": None}


def _utf16(body: bytes) -> str:
    """UTF-16 without a stated byte order.  The NUL bytes of ASCII characters show it (high byte first: at even
    positions); text with none (all Japanese) is read as big-endian, which A1111 (piexif) writes."""
    if body[:2] == b"\xfe\xff":
        return body[2:].decode("utf-16-be", "replace")
    if body[:2] == b"\xff\xfe":
        return body[2:].decode("utf-16-le", "replace")
    return body.decode("utf-16-le" if body[1::2].count(0) > body[0::2].count(0) else "utf-16-be", "replace")


def _decode_user_comment(raw: Any) -> str:
    """EXIF UserComment (the first 8 bytes name the character code)."""
    if isinstance(raw, str):
        return raw.strip("\x00 ")
    if not isinstance(raw, bytes):
        return ""
    head, body = raw[:8], raw[8:]
    if head.startswith(b"UNICODE"):
        return _utf16(body).strip("\x00 ")
    if head.startswith(b"ASCII") or head == b"\x00" * 8:
        return body.decode("utf-8", "replace").strip("\x00 ")
    return raw.decode("utf-8", "replace").strip("\x00 ")


def _sampler_a1111(name: str, sched: str | None, out: dict):
    s = name.strip().lower()
    if not sched:   # the old style: "DPM++ 2M Karras"
        for suf in sorted(A1111_SCHEDULERS, key=len, reverse=True):
            if s.endswith(" " + suf) and s[: -len(suf) - 1] in A1111_SAMPLERS:
                s, sched = s[: -len(suf) - 1], suf
                break
    if s in A1111_SAMPLERS:
        out["sampling"]["sampler"] = A1111_SAMPLERS[s]
    else:
        out["notes"].append(f"サンプラー「{name}」に当たるものがないため、サンプラーは初期値のままです。")
    if sched and sched.strip().lower() != "automatic":
        if sched.strip().lower() in A1111_SCHEDULERS:
            out["sampling"]["scheduler"] = A1111_SCHEDULERS[sched.strip().lower()]
        else:
            out["notes"].append(f"スケジューラー「{sched}」に当たるものがないため、スケジューラーは初期値のままです。")


def parse_a1111(text: str) -> dict:
    out = _blank()
    out["source"] = "A1111 / Forge"
    lines = text.strip().split("\n")
    params = lines.pop().strip() if lines and re.match(r"\s*Steps: ", lines[-1]) else ""
    body = "\n".join(lines)
    if body.startswith("Negative prompt: "):
        positive, negative = "", body[len("Negative prompt: "):]
    else:
        positive, _, negative = body.partition("\nNegative prompt: ")
    # <lora:name:0.8> is how A1111 loads LoRAs: they go to the LoRA rows, not the prompt
    for m in RE_LORA.finditer(positive):
        w = _num(m.group(2))
        out["loras"].append({"name": m.group(1).strip(), "strength": 1.0 if w is None else w})
    positive = RE_LORA.sub("", positive)
    positive = re.sub(r"[ \t]*,[ \t]*(,[ \t]*)+", ", ", positive)
    positive = re.sub(r"^[\s,]+|[\s,]+$", "", positive)
    out["positive"], out["negative"] = positive.strip(), negative.strip()
    kv = {}
    for k, v in RE_PARAM.findall(params):
        v = v.strip()
        if v.startswith('"') and v.endswith('"'):
            try:
                v = json.loads(v)
            except ValueError:
                v = v[1:-1]
        kv[k.strip()] = v
    if (n := _num(kv.get("Steps"), int)) is not None:
        out["sampling"]["steps"] = n
    if (n := _num(kv.get("CFG scale"))) is not None:
        out["sampling"]["cfg"] = n
    if (n := _num(kv.get("Seed"), int)) is not None:
        out["seed"] = n
    if (n := _num(kv.get("Clip skip"), int)) is not None:
        out["sampling"]["clip_skip"] = n
    if m := re.match(r"(\d+)x(\d+)", kv.get("Size", "")):
        out["width"], out["height"] = int(m.group(1)), int(m.group(2))
    if kv.get("Sampler"):
        _sampler_a1111(kv["Sampler"], kv.get("Schedule type"), out)
    if kv.get("Model"):
        out["model_name"] = kv["Model"]
    if kv.get("VAE") and kv["VAE"].lower() not in ("none", "automatic"):
        out["vae_name"] = kv["VAE"]
    if kv.get("Hires upscale") or kv.get("Hires resize-1"):
        out["notes"].append("元の画像は高解像度化（Hires. fix）をしています。この画面では高解像度化をしないので、画像サイズは最初の生成の大きさにしました。")
    return out


# ------------------------------------------------------------------------------------------ ComfyUI
def _comfy_texts(graph: dict, ref: Any, side: str, depth: int = 0) -> list[str]:
    """From a sampler's positive / negative input back to the text nodes (through ControlNet apply,
    Conditioning combine ... on the same side).  As i2t's _comfy_texts."""
    if depth > 20 or not (isinstance(ref, list) and ref):
        return []
    node = graph.get(str(ref[0]))
    if not isinstance(node, dict) or node.get("class_type") == "ConditioningZeroOut":   # no text: the zeroed one is not it
        return []
    inputs = node.get("inputs") or {}
    texts: list[str] = []
    for key in ("text", "text_g", "text_l", "prompt", "string", "value", "t5xxl", "clip_l"):
        value = inputs.get(key)
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())
        elif isinstance(value, list):
            texts += _comfy_texts(graph, value, side, depth + 1)
    for key in (side, "conditioning", "conditioning_1", "conditioning_2", "conditioning_to", "conditioning_from"):
        if isinstance(inputs.get(key), list):
            texts += _comfy_texts(graph, inputs[key], side, depth + 1)
    return list(dict.fromkeys(texts))


def _scalar(graph: dict, v: Any, depth: int = 0):
    """An input value; a link to a primitive / seed node is followed to its value."""
    if not isinstance(v, list):
        return v
    if depth > 5 or not v:
        return None
    node = graph.get(str(v[0]))
    if not isinstance(node, dict):
        return None
    vals = [x for x in (node.get("inputs") or {}).values() if not isinstance(x, list)]
    links = [x for x in (node.get("inputs") or {}).values() if isinstance(x, list)]
    if len(vals) == 1:
        return vals[0]
    return _scalar(graph, links[0], depth + 1) if len(links) == 1 and not vals else None


def parse_comfy(prompt_json: str) -> dict:
    graph = json.loads(prompt_json)
    if not isinstance(graph, dict):
        raise ValueError("ComfyUI のグラフではありません")
    out = _blank()
    out["source"] = "ComfyUI"
    out["graph"] = graph   # for 「workflow のまま生成」 (comfyflow); not sent to the page
    nodes = [(k, n) for k, n in sorted(graph.items(), key=lambda kv: (len(str(kv[0])), str(kv[0]))) if isinstance(n, dict)]
    positives, negatives = [], []
    main = None
    for _, node in nodes:
        if not re.search(r"Sampler|Guider", str(node.get("class_type", ""))):
            continue
        inputs = node.get("inputs") or {}
        pos = _comfy_texts(graph, inputs.get("positive") or inputs.get("conditioning"), "positive")
        neg = _comfy_texts(graph, inputs.get("negative"), "negative")
        positives += [t for t in pos if t not in positives]
        negatives += [t for t in neg if t not in negatives]
        if main is None and (pos or any(k in inputs for k in ("seed", "noise_seed", "steps"))):
            main = inputs
    out["positive"], out["negative"] = "\n\n".join(positives), "\n\n".join(negatives)

    def first(key: str):
        """The value from the main sampler, else from any node that has it (SamplerCustomAdvanced splits the
        sampler into KSamplerSelect / BasicScheduler / RandomNoise / CFGGuider)."""
        if main and key in main and (v := _scalar(graph, main[key])) is not None:
            return v
        for _, n in nodes:
            inputs = n.get("inputs") or {}
            if key in inputs and (v := _scalar(graph, inputs[key])) is not None:
                return v
        return None

    seed = first("seed")
    seed = first("noise_seed") if seed is None else seed
    out["seed"] = _num(seed, int)
    for key, field, kind in (("steps", "steps", int), ("cfg", "cfg", float)):
        if (n := _num(first(key), kind)) is not None:
            out["sampling"][field] = n
    for key, field in (("sampler_name", "sampler"), ("scheduler", "scheduler")):
        if isinstance(v := first(key), str) and v:
            out["sampling"][field] = v
    for _, n in nodes:
        ct, inputs = str(n.get("class_type", "")), n.get("inputs") or {}
        if out["width"] is None and re.search(r"Empty.*Latent", ct):
            w, h = _num(_scalar(graph, inputs.get("width")), int), _num(_scalar(graph, inputs.get("height")), int)
            if m := re.search(r"(\d+)\s*x\s*(\d+)", str(inputs.get("dimensions") or "")):   # rgthree's size list
                w, h = int(m.group(1)), int(m.group(2))
            if w and h:
                out["width"], out["height"] = w, h
        if ct == "CLIPSetLastLayer" and (n2 := _num(_scalar(graph, inputs.get("stop_at_clip_layer")), int)):
            out["sampling"]["clip_skip"] = abs(n2)
        if ct == "VAELoader" and out["vae_name"] is None and isinstance(inputs.get("vae_name"), str):
            out["vae_name"] = inputs["vae_name"]
        if ct == "CLIPLoader" and out["te_name"] is None and isinstance(inputs.get("clip_name"), str):
            out["te_name"] = inputs["clip_name"]
        if re.match(r"(Dual|Triple|Quadruple)CLIPLoader", ct):
            out["notes"].append("元の画像は CLIP を複数のファイルから読み込んでいます。この画面では選べないため、CLIP はモデルのもののままです。")
        # KJNodes' 「Patch Sage Attention KJ」: SageAttention on the model ("disabled" = off)
        if re.search(r"SageAttention", ct) and str(_scalar(graph, inputs.get("sage_attention")) or "disabled") != "disabled":
            out["attention"] = "sage"
        for key in ("ckpt_name", "unet_name"):
            if out["model_name"] is None and isinstance(inputs.get(key), str):
                out["model_name"] = inputs[key]
        if isinstance(inputs.get("lora_name"), str):
            w = _num(_scalar(graph, inputs.get("strength_model", inputs.get("strength"))))
            out["loras"].append({"name": inputs["lora_name"], "strength": 1.0 if w is None else w})
        if ct == "SaveImage" and isinstance(inputs.get("filename_prefix"), str):
            if m := RE_OWN.match(inputs["filename_prefix"]):
                out["own_entry"] = m.group(1)
    return out


def parse_novelai(comment: str) -> dict:
    data = json.loads(comment)
    if not isinstance(data, dict):
        raise ValueError("NovelAI の Comment ではありません")
    out = _blank()
    out["source"] = "NovelAI"

    def v4(key: str) -> str:
        cap = ((data.get(key) or {}).get("caption") or {}) if isinstance(data.get(key), dict) else {}
        base = str(cap.get("base_caption") or "").strip()
        chars = [str(c.get("char_caption") or "").strip() for c in cap.get("char_captions") or [] if isinstance(c, dict)]
        return "\n".join(x for x in [base] + [c for c in chars if c] if x)

    out["positive"] = v4("v4_prompt") or str(data.get("prompt") or "").strip()
    out["negative"] = v4("v4_negative_prompt") or str(data.get("uc") or "").strip()
    out["seed"] = _num(data.get("seed"), int)
    out["width"], out["height"] = _num(data.get("width"), int), _num(data.get("height"), int)
    if (n := _num(data.get("steps"), int)) is not None:
        out["sampling"]["steps"] = n
    if (n := _num(data.get("scale"))) is not None:
        out["sampling"]["cfg"] = n
    if isinstance(data.get("sampler"), str):
        out["sampling"]["sampler"] = re.sub(r"^k_", "", data["sampler"])
    if data.get("noise_schedule") in ("karras", "exponential"):
        out["sampling"]["scheduler"] = data["noise_schedule"]
    out["notes"].append("NovelAI のモデルはこのパソコンにないため、モデルは今のままです。")
    return out


def parse_fooocus(text: str) -> dict:
    data = json.loads(text)
    out = _blank()
    out["source"] = "Fooocus"
    out["positive"], out["negative"] = str(data.get("prompt") or "").strip(), str(data.get("negative_prompt") or "").strip()
    out["seed"] = _num(data.get("seed"), int)
    if m := re.match(r"\(?\s*(\d+)\s*,\s*(\d+)", str(data.get("resolution") or "")):
        out["width"], out["height"] = int(m.group(1)), int(m.group(2))
    if (n := _num(data.get("steps"), int)) is not None:
        out["sampling"]["steps"] = n
    if (n := _num(data.get("guidance_scale"))) is not None:
        out["sampling"]["cfg"] = n
    for key, field in (("sampler", "sampler"), ("scheduler", "scheduler")):
        if isinstance(data.get(key), str):
            out["sampling"][field] = data[key]
    if isinstance(data.get("base_model"), str):
        out["model_name"] = data["base_model"]
    for lo in data.get("loras") or []:   # [[name, weight], ...] or [name, weight, hash]
        if isinstance(lo, (list, tuple)) and lo and isinstance(lo[0], str) and lo[0] != "None":
            out["loras"].append({"name": lo[0], "strength": _num(lo[1]) if len(lo) > 1 and _num(lo[1]) is not None else 1.0})
    return out


def parse_invoke(text: str) -> dict:
    data = json.loads(text)
    out = _blank()
    out["source"] = "InvokeAI"
    out["positive"], out["negative"] = str(data.get("positive_prompt") or "").strip(), str(data.get("negative_prompt") or "").strip()
    out["seed"] = _num(data.get("seed"), int)
    out["width"], out["height"] = _num(data.get("width"), int), _num(data.get("height"), int)
    if (n := _num(data.get("steps"), int)) is not None:
        out["sampling"]["steps"] = n
    if (n := _num(data.get("cfg_scale"))) is not None:
        out["sampling"]["cfg"] = n
    if isinstance(data.get("model"), dict) and data["model"].get("name"):
        out["model_name"] = data["model"]["name"]
    for lo in data.get("loras") or []:
        m = (lo.get("model") or lo.get("lora") or {}) if isinstance(lo, dict) else {}
        if isinstance(m, dict) and m.get("name"):
            w = _num(lo.get("weight"))
            out["loras"].append({"name": m["name"], "strength": 1.0 if w is None else w})
    if data.get("scheduler"):
        out["notes"].append(f"InvokeAI のスケジューラー「{data['scheduler']}」は読み替えないため、サンプラーは初期値のままです。")
    return out


def read(data: bytes) -> dict | None:
    """The settings in an image file, or None when it has none.  ValueError: not an image."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            info = {k: v for k, v in (im.info or {}).items() if isinstance(v, str)}
            if not any(k in info for k in ("parameters", "prompt", "Comment", "invokeai_metadata")):
                try:
                    comment = _decode_user_comment(im.getexif().get_ifd(0x8769).get(0x9286))
                except Exception:   # noqa: BLE001 - a broken EXIF is not read
                    comment = ""
                if comment:
                    info["parameters"] = comment
    except Exception as e:   # noqa: BLE001 - PIL.UnidentifiedImageError and friends
        raise ValueError("画像として読み込めませんでした") from e
    params = info.get("parameters", "").strip()
    candidates = []
    if params.startswith("{"):
        candidates.append((params, parse_fooocus))
    elif params:
        candidates.append((params, parse_a1111))
    if info.get("prompt", "").strip().startswith("{"):
        candidates.append((info["prompt"], parse_comfy))
    if info.get("Comment", "").strip().startswith("{"):
        candidates.append((info["Comment"], parse_novelai))
    if info.get("invokeai_metadata", "").strip().startswith("{"):
        candidates.append((info["invokeai_metadata"], parse_invoke))
    for raw, parse in candidates:
        try:
            out = parse(raw)
        except (ValueError, TypeError, AttributeError, KeyError):
            continue
        if out["positive"] or out["negative"] or out.get("own_entry"):
            return out
    return None
