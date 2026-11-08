"""Plain image generation ("画像生成"): model + LoRAs + prompt + size, nothing added by the tool.

Used to check a checkpoint / LoRA / prompt before using it for material generation; a result's settings
can be copied into a project's material-generation recipe.  The prompt is sent exactly as typed (no shot
words, no background removal): those are added only when the settings are used for LoRA material.

Requests run on the same serial GPU queue as training and evaluation.  ComfyUI is kept running for a
few minutes after a request so the next try does not reload the model; it is stopped as soon as any
other GPU job takes the GPU, or after the idle time.
"""
from __future__ import annotations

import json
import random
import shutil
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from . import comfyflow, config, loraconf
from .evaluate import ATTENTION_FLAGS, ATTENTION_PACKAGES, attention_installed
from .evaluate import COUPLE_FAMILIES, KEEP_COMFY_OWNERS, NATIVE_SIZE, comfy, comfy_user, wants_negpip
from .events import bus
from .generate import (_stage, build_graph, check_model_loras, check_seed, clamp_denoise, default_sampling, source_file,
                       stage_comfy_input, save_source_image, stage_model_files, tuned_note)
from .gpu import gpu
from .models import EDIT_CHINOLL, detect_lora, experimental_note, registry

LABEL = "画像生成"
OWNER = SimpleNamespace(id="__imagegen__")   # stands in for a project on the job queue
DIR = config.DATA_DIR / "imagegen"
IMG_DIR = DIR / "images"   # the default place; config.storage_dir("imagegen") is where new images go
MAX_COUNT = 100   # per request; images come one after another with consecutive seeds and can be stopped any time
HISTORY = DIR / "history.json"
FORM = DIR / "form.json"
PRESETS = DIR / "presets.json"
PRESET_THUMBS = DIR / "preset_thumbs"
PRESET_THUMB_PX = 512
INIT_DIR = DIR / "init"   # source images of image-to-image (one file per content, shared by the requests using it)
DENOISE_DEFAULT = 0.6
PRESET_KEYS = ("model_id", "loras", "positive", "negative")
KEEP_ALIVE_S = 300
SAMPLING_KEYS = ("steps", "cfg", "sampler", "scheduler", "clip_skip", "fast", "rescale")   # not fast_choices (computed)

_lock = threading.RLock()
_cancel: set[str] = set()
_progress: dict[str, dict] = {}   # running request -> {phase, image, count, step, steps, eta_s} (memory only)
_kept = False          # ComfyUI is running only because of this page (stopped when someone else needs the GPU)
_idle_timer: threading.Timer | None = None


def img_file(name: str) -> Path:
    """A generated image: in the current place, else in a place used before (the images stay where they were
    made when the place is changed on the settings page)."""
    dirs = config.storage_dirs("imagegen")
    return next((d / name for d in dirs if (d / name).exists()), dirs[0] / name)


def _load(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def history() -> list[dict]:
    with _lock:
        return _load(HISTORY, [])


def _save_history(items: list[dict]):
    config.atomic_write_json(HISTORY, items)


def _update(eid: str, **fields):
    with _lock:
        items = history()
        for e in items:
            if e["id"] == eid:
                e.update(fields)
                break
        _save_history(items)
    bus.publish("imagegen", id=eid, **{k: v for k, v in fields.items() if k in ("status", "message")})


def _set_progress(eid: str, **fields):
    p = _progress.setdefault(eid, {})
    p.update(fields)
    bus.publish("imagegen", id=eid, progress=dict(p))


def progress(eid: str) -> dict | None:
    return _progress.get(eid)


def get_form() -> dict:
    return _load(FORM, {})


def save_form(form: dict):
    config.atomic_write_json(FORM, form)


def presets() -> list[dict]:
    """Saved combinations of model, LoRAs and prompt (「保存した設定」), in the order they were saved."""
    with _lock:
        return _load(PRESETS, [])


def _write_presets(items: list[dict]):
    config.atomic_write_json(PRESETS, items)


def _update_preset(pid: str, **fields) -> bool:
    with _lock:
        items = presets()
        p = next((x for x in items if x["id"] == pid), None)
        if p:
            p.update(fields)
            _write_presets(items)
    if p:
        bus.publish("imagegen", preset=pid)
    return bool(p)


def save_preset(body: dict, runner=None) -> dict:
    """Save the model / LoRAs / prompt under a name.  A preset with the same name is replaced in place (its
    creation date and picture stay until the new picture is made).  With a runner, one image of the preset
    is queued as its thumbnail, at the size / detailed settings / seed in body["thumb"].
    body["restore"] (「元に戻す」 after a deletion) puts the deleted preset back as it was, at body["index"]."""
    name = (body.get("name") or "").strip()
    if not name:
        raise ValueError("名前を入れてください")
    if len(name) > 80:
        raise ValueError("名前は 80 文字以内にしてください")
    now = time.time()
    preset = {"id": body.get("id") or uuid.uuid4().hex[:12], "name": name, "created": now, "saved": now,
              **{k: body.get(k) for k in PRESET_KEYS}}
    if body.get("restore"):
        preset.update({k: body[k] for k in ("created", "saved", "thumb_file", "thumb", "model_name") if k in body})
    preset["loras"] = [{"path": lo["path"], "strength": float(lo["strength"] if lo.get("strength") is not None else 1.0),
                        **({"trigger": bool(lo["trigger"])} if lo.get("trigger") is not None else {}),
                        **({"words": loraconf.clean_words(lo["words"])} if isinstance(lo.get("words"), list) else {})}
                       for lo in preset["loras"] or [] if lo.get("path")]
    model = registry.get(preset.get("model_id") or "")
    if model:
        preset["model_name"] = model["name"]
    preset.setdefault("model_name", "")
    with _lock:
        items = presets()
        same = next((i for i, x in enumerate(items) if x["name"] == name), None)
        if same is None:
            at = body.get("index")
            items.insert(at if isinstance(at, int) and 0 <= at <= len(items) else len(items), preset)
        else:
            old = items[same]
            preset.update(id=old["id"], created=old.get("created", old.get("saved", now)))
            for k in ("thumb_file", "thumb"):
                if k in old and k not in preset:
                    preset[k] = old[k]
            items[same] = preset
        _write_presets(items)
    if runner is not None and not body.get("restore"):
        preset["thumb"] = make_preset_thumb(preset["id"], body.get("thumb") or {}, runner)
    return preset


def make_preset_thumb(pid: str, opts: dict, runner) -> dict:
    """Queue one image of the preset as its thumbnail.  opts: width / height / sampling / seed (the form's
    values when it was saved).  Returns the thumbnail state stored on the preset."""
    p = next((x for x in presets() if x["id"] == pid), None)
    if not p:
        raise ValueError("その設定は見つかりません")
    form = {**{k: p.get(k) for k in PRESET_KEYS}, **{k: opts.get(k) for k in ("width", "height", "sampling", "seed")},
            "count": 1, "thumb_for": pid, "thumb_name": p["name"]}
    try:
        e = submit(form, runner)
        state = {"status": "queued", "entry": e["id"]}
    except ValueError as ex:   # e.g. the model file is gone: the preset is saved, only without a picture
        state = {"status": "failed", "message": f"サムネイルを作れませんでした: {ex}"}
    _update_preset(pid, thumb=state)
    return state


def _preset_thumb_done(pid: str, eid: str, image: Path | None, message: str):
    """Called when a thumbnail request ends.  Only the newest request of the preset may set its picture."""
    p = next((x for x in presets() if x["id"] == pid), None)
    if not p or (p.get("thumb") or {}).get("entry") != eid:
        return
    if image is None:
        if p["thumb"].get("status") != "queued":   # e.g. recover() seeing a request that already set the picture
            return
        _update_preset(pid, thumb={"status": "failed", "entry": eid, "message": message})
        return
    try:
        from PIL import Image
        PRESET_THUMBS.mkdir(parents=True, exist_ok=True)
        name = f"{pid}_{eid}.webp"
        with Image.open(image) as x:
            x = x.convert("RGB")
            x.thumbnail((PRESET_THUMB_PX, PRESET_THUMB_PX))
            tmp = PRESET_THUMBS / f"{name}.{uuid.uuid4().hex[:8]}.tmp"
            x.save(tmp, "WEBP", quality=88)
            tmp.replace(PRESET_THUMBS / name)
    except Exception as ex:
        _update_preset(pid, thumb={"status": "failed", "entry": eid, "message": f"サムネイルを保存できませんでした: {ex}"})
        return
    _update_preset(pid, thumb_file=name, thumb={"status": "done", "entry": eid})
    prune_preset_thumbs()


def prune_preset_thumbs(min_age_s: float = 600):
    """Remove picture files no preset points to (replaced ones, and those of deleted presets once 「元に戻す」
    is long past)."""
    if not PRESET_THUMBS.is_dir():
        return
    keep = {p.get("thumb_file") for p in presets()}
    for f in PRESET_THUMBS.glob("*.webp"):
        try:
            if f.name not in keep and time.time() - f.stat().st_mtime > min_age_s:
                f.unlink()
        except OSError:
            pass


def delete_preset(pid: str) -> dict | None:
    """Remove a preset; returns it (so the page can offer 「元に戻す」), or None when there was none."""
    with _lock:
        items = presets()
        at = next((i for i, x in enumerate(items) if x["id"] == pid), None)
        if at is None:
            return None
        gone = dict(items.pop(at), index=at)
        config.atomic_write_json(PRESETS, items)
    return gone


def native_size(model: dict | None) -> int:
    return NATIVE_SIZE.get((model or {}).get("family"), 1024)


def _size(v, default: int) -> int:
    try:
        n = int(v)
    except (TypeError, ValueError):
        return default
    return max(512, min(2048, round(n / 16) * 16))


def save_init(data: bytes) -> dict:
    return save_source_image(data, INIT_DIR)


def init_file(name) -> Path | None:
    return source_file(INIT_DIR, name)


def _denoise(v) -> float:
    return clamp_denoise(v, DENOISE_DEFAULT)


# MOSS-Image editing (an edit LoRA and a source image): ComfyUI-Moss-Image-Edit-Native's recommended settings, 30 steps
# euler / simple, cfg 5 for chinoll's LoRA and 3 for MOSS-Image-Edit-HQ; what the user set in 詳細 is kept
EDIT_SAMPLING = {"steps": 30, "sampler": "euler", "scheduler": "simple"}


def edit_format(model: dict | None, form: dict) -> str | None:
    """The edit layout when the request edits its source image: MOSS-Image, a source image and an edit LoRA among the LoRAs."""
    if not model or model["family"] != "moss_image" or not form.get("init_image"):
        return None
    for lo in form.get("loras") or []:
        try:
            fmt = lo.get("path") and detect_lora(Path(lo["path"])).get("edit_format")
        except (OSError, ValueError):
            fmt = None
        if fmt:
            return fmt
    return None


def request_sampling(model: dict | None, form: dict) -> tuple[dict, str | None]:
    """The default sampling of a request and its edit layout (None: not editing)."""
    edit = edit_format(model, form)
    fast = (form.get("sampling") or {}).get("fast") or None
    # editing: the 8-step LoRA (made for text-to-image) only when switched on in 詳細
    ds = default_sampling(model, fast or ("off" if edit else None))
    if edit and ds.get("fast") in (None, "off"):
        ds.update(EDIT_SAMPLING, cfg=5.0 if edit == EDIT_CHINOLL else 3.0)
    return ds, edit


ATTENTION_LABEL = {"pytorch": "PyTorch（SDPA）", "split": "Split", "quad": "Sub-quadratic", "sage": "SageAttention",
                   "flash": "FlashAttention"}


def companions(form: dict, model: dict | None) -> dict:
    """The VAE / text encoder (CLIP) chosen in 詳細 instead of the model's own: {"vae": path, "te": path}.
    SDXL checkpoints carry two text encoders (CLIP-L and -G), which one file cannot replace: only the VAE there."""
    out = {}
    for kind in ("vae", "te"):
        p = str(form.get(kind) or "").strip()
        if p and not (kind == "te" and (model or {}).get("family") == "sdxl"):
            out[kind] = p
    return out


def check_extras(form: dict, model: dict | None) -> list[str]:
    from .models import library
    errors = []
    for kind, p in companions(form, model).items():
        what = "VAE" if kind == "vae" else "CLIP（テキストエンコーダー）"
        if not Path(p).is_file():
            errors.append(f"詳細設定で選んだ {what} のファイルが見つかりません: {Path(p).name}")
        elif model and library.fits(model["family"], kind, p) is False:
            errors.append(f"詳細設定で選んだ {what}（{Path(p).name}）は、このモデル用ではありません。")
    att = form.get("attention") or ""
    if att and att not in ATTENTION_FLAGS:
        errors.append("アテンションの種類が正しくありません。")
    elif att in ATTENTION_PACKAGES and not attention_installed()[att]:
        errors.append(f"{ATTENTION_LABEL[att]} は ComfyUI の環境に入っていないため使えません。詳細設定のアテンションを「標準」にしてください。")
    return errors


def check(form: dict) -> dict:
    if form.get("workflow"):   # workflow のまま生成: its own model / LoRAs / size / sampling
        errors = [] if comfyflow.meta(form["workflow"]) else ["読み込んだ workflow が見つかりません。画像をもう一度読み込んでください。"]
        errors += check_extras({"attention": form.get("attention")}, None)
        return {"errors": errors, "notes": [], "model": None}
    errors, model = check_model_loras(form.get("model_id"), form.get("loras"))
    errors += check_extras(form, model)
    if not (form.get("positive") or "").strip():
        errors.append("プロンプトを入力してください。")
    if form.get("init_image") and not init_file(form["init_image"]):
        errors.append("元画像が見つかりません。もう一度選んでください。")
    notes = [experimental_note(model.get("variant"))] if model and experimental_note(model.get("variant")) else []
    if tuned_note(model):
        notes.append(tuned_note(model))
    return {"errors": errors, "notes": notes, "model": model}


def submit(form: dict, runner) -> dict:
    """Queue one request (1-MAX_COUNT images with consecutive seeds) on the job runner.  Returns the history entry."""
    chk = check(form)
    if chk["errors"]:
        raise ValueError(" ".join(chk["errors"]))
    if form.get("workflow"):
        return _submit_workflow(form, runner)
    model = chk["model"]
    over = {k: v for k, v in (form.get("sampling") or {}).items() if k in SAMPLING_KEYS and v not in (None, "")}
    ds, edit = request_sampling(model, form)
    n = native_size(model)
    seed = form.get("seed")
    try:
        count = max(1, min(MAX_COUNT, int(form.get("count") or 1)))
    except (TypeError, ValueError):
        raise ValueError("枚数は整数で入れてください") from None
    loras = [{"path": lo["path"], "strength": float(lo.get("strength", 1.0)),
              **({"trigger": bool(lo["trigger"])} if lo.get("trigger") is not None else {}),
              **({"words": loraconf.clean_words(lo["words"])} if isinstance(lo.get("words"), list) else {})}
             for lo in form.get("loras") or [] if lo.get("path")]
    # 左右で描き分ける (ComfyUI-ppm Attention Couple, SDXL / Anima): [left, right] as typed
    reg = form.get("regions") or {}
    regions = [str(reg.get("left") or "").strip(), str(reg.get("right") or "").strip()]         if model["family"] in COUPLE_FAMILIES and isinstance(reg, dict) else []
    regions = regions if any(regions) else []
    # the typed prompt is kept as typed (restored by "この設定に戻す"); trigger words are added when sent
    sent, added = loraconf.apply(form["positive"].strip(), loras)
    settings = {
        "model_id": model["id"], "model_name": model["name"], "family": model["family"],
        "loras": loras,
        "positive": form["positive"].strip(), "prompt_sent": sent, "trigger_words": added,
        "negative": (form.get("negative") or "").strip(),
        "width": _size(form.get("width"), n), "height": _size(form.get("height"), n),
        "seed": check_seed(seed, count) if seed not in (None, "") else random.randint(1, 2**31 - 1000),
        "count": count,
        "sampling": {**ds, **over}, "sampling_overrides": over,
        **({"init_image": Path(form["init_image"]).name, "denoise": _denoise(form.get("denoise"))} if form.get("init_image") else {}),
        **({"edit": edit} if edit else {}),
        **({"thumb_for": form["thumb_for"], "thumb_name": form.get("thumb_name") or ""} if form.get("thumb_for") else {}),
        **({"regions": regions} if regions else {}),
        **companions(form, model),
        **({"attention": form["attention"]} if form.get("attention") in ATTENTION_FLAGS else {}),
    }
    entry = {"id": uuid.uuid4().hex[:12], "created": time.time(), "status": "queued", "message": "順番待ち",
             "settings": settings, "images": [], "warnings": []}
    with _lock:
        items = history()
        items.insert(0, entry)
        _save_history(items)
    job = runner.enqueue_task(OWNER, LABEL, lambda: run(entry["id"]), on_cancel=lambda: cancel(entry["id"]))
    job.detail = (form.get("positive") or "").strip()[:60] or None
    job.entry = entry["id"]
    bus.publish("imagegen", id=entry["id"], status="queued", message="順番待ち")
    return entry


def _count(form: dict) -> int:
    try:
        return max(1, min(MAX_COUNT, int(form.get("count") or 1)))
    except (TypeError, ValueError):
        raise ValueError("枚数は整数で入れてください") from None


def _queue(entry: dict, runner, detail: str) -> dict:
    with _lock:
        items = history()
        items.insert(0, entry)
        _save_history(items)
    job = runner.enqueue_task(OWNER, LABEL, lambda: run(entry["id"]), on_cancel=lambda: cancel(entry["id"]))
    job.detail = detail[:60] or None
    job.entry = entry["id"]
    bus.publish("imagegen", id=entry["id"], status="queued", message="順番待ち")
    return entry


def _wf_size(graph: dict, form: dict) -> dict:
    """The page's size (multiples of 16, as the form), or the workflow's own when the page has none."""
    own = comfyflow.latent_size(graph) or (None, None)
    if form.get("width") and form.get("height"):
        return {"width": _size(form["width"], own[0] or 1024), "height": _size(form["height"], own[1] or 1024)}
    return {"width": own[0], "height": own[1]}


def _submit_workflow(form: dict, runner) -> dict:
    """workflow のまま生成: the stored workflow with the page's prompt / negative (checked here, so a prompt that
    cannot go in is said before queueing), seed and count."""
    wid = form["workflow"]
    graph, m = comfyflow.load(wid), comfyflow.meta(wid)
    positive, negative = (form.get("positive") or "").strip(), (form.get("negative") or "").strip()
    comfyflow.apply_prompts(graph, positive, negative)   # ValueError: said on the page
    count = _count(form)
    seed = form.get("seed")
    seed = check_seed(seed, count) if seed not in (None, "") else random.randint(1, 2**31 - 1000)
    settings = {"workflow": wid, "workflow_source": m["source"], "workflow_summary": m["summary"],
                "model_name": "workflow", "family": None, "model_id": None, "loras": [],
                "positive": positive, "prompt_sent": positive, "trigger_words": [], "negative": negative,
                **_wf_size(graph, form), "seed": seed, "count": count, "sampling": {}, "sampling_overrides": {},
                **({"attention": form["attention"]} if form.get("attention") in ATTENTION_FLAGS else {})}
    entry = {"id": uuid.uuid4().hex[:12], "created": time.time(), "status": "queued", "message": "順番待ち",
             "settings": settings, "images": [], "warnings": []}
    return _queue(entry, runner, positive)


def cancel(eid: str) -> bool:
    e = next((x for x in history() if x["id"] == eid), None)
    if not e or e["status"] not in ("queued", "running"):
        return False
    _cancel.add(eid)
    if e["status"] == "queued":
        _update(eid, status="cancelled", message="中止しました")
        if e["settings"].get("thumb_for"):
            _preset_thumb_done(e["settings"]["thumb_for"], eid, None, "サムネイルの生成を中止しました")
    else:
        comfy.cancel_current()
    return True


def delete(eid: str) -> bool:
    with _lock:
        items = history()
        e = next((x for x in items if x["id"] == eid), None)
        if not e or e["status"] in ("queued", "running"):
            return False
        for im in e["images"]:
            f = img_file(im["file"])
            for f in (f, f.parent / "thumbs" / (f.stem + ".webp")):
                try:
                    f.unlink()
                except OSError:
                    pass
        rest = [x for x in items if x["id"] != eid]
        _save_history(rest)
        init = e["settings"].get("init_image")
        if init and init != get_form().get("init_image") and not any(x["settings"].get("init_image") == init for x in rest):
            from .evaluate import COMFY_INPUT
            for f in (INIT_DIR / init, COMFY_INPUT / init):   # no other request or the form uses it
                f.unlink(missing_ok=True)
    return True


def run(eid: str):
    """Generate one request.  Called from the serial GPU queue (the GPU is already ours)."""
    global _kept
    e = next((x for x in history() if x["id"] == eid), None)
    if not e or e["status"] != "queued" or eid in _cancel:
        _cancel.discard(eid)
        return
    s = e["settings"]
    if s.get("workflow"):
        return _run_workflow(eid, s)
    model = registry.get(s["model_id"])
    if not model or model["missing"]:
        _update(eid, status="failed", message="生成モデルのファイルが見つかりません")
        if s.get("thumb_for"):
            _preset_thumb_done(s["thumb_for"], eid, None, "生成モデルのファイルが見つかりません")
        return
    _update(eid, status="running", message="準備中（モデルを読み込んでいます）")
    _set_progress(eid, phase="loading", image=0, count=s["count"], step=0, steps=0, eta_s=None)
    img_dir = config.storage_dir("imagegen")
    img_dir.mkdir(parents=True, exist_ok=True)
    images, warnings = [], []
    durations: list[float] = []   # seconds per finished image, for the remaining-time estimate
    try:
        staged = stage_model_files(model)
        loras = [(_stage("loras", lo["path"]), lo["strength"]) for lo in s["loras"]]
        if s.get("vae"):   # the VAE / CLIP chosen in 詳細 instead of the model's own
            staged["vae_override" if model["family"] == "sdxl" else "vae"] = _stage("vae", s["vae"])
        if s.get("te") and model["family"] != "sdxl":
            staged["te"] = _stage("text_encoders", s["te"])
        init = None
        if s.get("init_image"):
            src = init_file(s["init_image"])
            if not src:
                raise RuntimeError("元画像が見つかりません（消された可能性があります）")
            init = {"image": stage_comfy_input(src), "denoise": s["denoise"]}
        comfy.ensure_started(s.get("attention") or "")   # restarts only for unknown folders / custom nodes, or another attention
        _kept = True
        log0 = comfy.log_size()
        if init and s.get("edit") and comfy.moss_edit:   # MOSS-Image edit LoRA: the source image edited as the prompt says
            init["edit"] = s["edit"]
        # what needs ComfyUI-ppm / ComfyUI-Moss-Image-Edit-Native is left out when it could not be fetched or loaded: said on the result
        no_ppm = [] if comfy.ppm or not (s.get("regions") or s["sampling"].get("rescale") == "on"
                                          or (not s.get("edit") and wants_negpip(model["family"], s["positive"], s["negative"]))) else \
            ["追加の部品（ComfyUI-ppm）を読み込めなかったため、左右の描き分け・白飛びの抑制・マイナスの強さの指定なしで生成しました。"]
        if init and s.get("edit") and not comfy.moss_edit:
            no_ppm.append("編集用の部品（ComfyUI-Moss-Image-Edit-Native）を読み込めなかったため、指示での編集ではなく、"
                          "元画像を下地にする方法で生成しました。")
        if no_ppm:
            _update(eid, warnings=no_ppm)
        for i in range(s["count"]):
            if eid in _cancel:
                break
            if i == 0:   # the count of images done is in the progress (memory only): the history file is written once per image
                _update(eid, message="生成中")
            t_img = time.time()
            first: list = []   # (time, step) of the first step report of this image
            _set_progress(eid, phase="loading" if i == 0 else "sampling", image=i + 1, step=0, steps=0,
                          eta_s=_eta(durations, s["count"] - i - 1, None))

            def on_progress(step, steps, i=i, t_img=t_img, first=first):
                now = time.time()
                if not first:
                    first.append((now, step))
                t0, s0 = first[0]
                rate = (now - t0) / (step - s0) if step > s0 else None   # measured seconds per step
                this = (steps - step) * rate if rate else None
                _set_progress(eid, phase="sampling", image=i + 1, step=step, steps=steps,
                              eta_s=_eta(durations, s["count"] - i - 1, this, steps * rate if rate else None))
            seed = s["seed"] + i
            g = build_graph(model, staged, loras, s.get("prompt_sent") or s["positive"], s["negative"], seed, s["width"], s["height"],
                            s["sampling"], f"lsimg_{eid}", init=init, regions=s.get("regions"))
            try:
                out = comfy.run_prompt(g, on_progress=on_progress)
            except RuntimeError:
                if eid in _cancel:
                    break
                raise
            src = config.DATA_DIR / "comfy_output" / out[0].get("subfolder", "") / out[0]["filename"]
            name = f"{eid}_{i}_s{seed}.png"
            shutil.move(str(src), img_dir / name)
            images.append({"file": name, "seed": seed})
            durations.append(time.time() - t_img)
            if i == 0 and loras:
                warnings = comfy.lora_warnings(log0)
            _update(eid, images=images, warnings=no_ppm + ([f"LoRA の一部を読み込めませんでした（{len(warnings)} 件）。この生成モデルに合わない LoRA の可能性があります。"] if warnings else []))
    except Exception as ex:
        _progress.pop(eid, None)
        _update(eid, status="failed", message=f"生成に失敗しました: {ex}", images=images)
        _kept = False
        comfy.after_failure(ex)
        _cancel.discard(eid)
        if s.get("thumb_for"):
            _preset_thumb_done(s["thumb_for"], eid, None, f"サムネイルの生成に失敗しました: {ex}")
        return
    _progress.pop(eid, None)
    if eid in _cancel:
        _cancel.discard(eid)
        _update(eid, status="cancelled", message=f"中止しました（{len(images)} 枚は残っています）", images=images)
    else:
        _update(eid, status="done", message="完了", images=images)
    if images:   # the model worked on this PC (検証済み in the model lists)
        registry.mark_checked(s.get("model_id"), "gen")
    if s.get("thumb_for"):
        _preset_thumb_done(s["thumb_for"], eid, img_dir / images[0]["file"] if images else None, "サムネイルの生成を中止しました")
    _arm_idle_stop()


def _run_workflow(eid: str, s: dict):
    """workflow のまま生成 (comfyflow): the graph with this PC's files, checked against the running ComfyUI, then
    once per seed.  Every image the workflow saves is kept."""
    global _kept
    _update(eid, status="running", message="準備中（workflow を確認しています）")
    _set_progress(eid, phase="loading", image=0, count=s["count"], step=0, steps=0, eta_s=None)
    img_dir = config.storage_dir("imagegen")
    img_dir.mkdir(parents=True, exist_ok=True)
    images, durations = [], []
    try:
        graph = comfyflow.load(s["workflow"])
        if graph is None:
            raise RuntimeError("読み込んだ workflow が見つかりません")
        graph = comfyflow.apply_prompts(graph, s["positive"], s["negative"])
        graph, notes, wf_model = comfyflow.resolve_files(graph, comfyflow.local_files(), comfy.stage_model, registry.list())
        comfy.ensure_started(s.get("attention") or "")
        _kept = True
        info = comfy._get("/object_info", timeout=60)   # which nodes make the empty image (substitute)
        graph, more = comfyflow.substitute(graph, (wf_model or {}).get("family"), info)
        graph = comfyflow.apply_size(graph, s.get("width"), s.get("height"))
        if notes + more:
            _update(eid, warnings=notes + more)
        errors = comfyflow.check(graph, info)
        if errors:
            raise RuntimeError(" ".join(errors))
        _update(eid, message="生成中")
        for i in range(s["count"]):
            if eid in _cancel:
                break
            t_img = time.time()
            first: list = []
            _set_progress(eid, phase="loading" if i == 0 else "sampling", image=i + 1, step=0, steps=0,
                          eta_s=_eta(durations, s["count"] - i - 1, None))

            def on_progress(step, steps, i=i, first=first):
                now = time.time()
                if not first or step < first[-1][1]:   # a workflow may sample more than once (hires): measured per pass
                    first.append((now, step))
                t0, s0 = first[-1]
                rate = (now - t0) / (step - s0) if step > s0 else None
                _set_progress(eid, phase="sampling", image=i + 1, step=step, steps=steps,
                              eta_s=_eta(durations, s["count"] - i - 1, (steps - step) * rate if rate else None))
            seed = s["seed"] + i
            try:
                out = comfy.run_prompt(comfyflow.for_run(graph, seed, f"lsimg_{eid}"), on_progress=on_progress)
            except RuntimeError:
                if eid in _cancel:
                    break
                raise
            outs = [o for o in out if o.get("type", "output") == "output"]
            if not outs:
                raise RuntimeError("workflow が画像を保存しませんでした")
            for k, o in enumerate(outs):
                src = config.DATA_DIR / "comfy_output" / o.get("subfolder", "") / o["filename"]
                name = f"{eid}_{i}{f'_{k}' if k else ''}_s{seed}.png"
                shutil.move(str(src), img_dir / name)
                images.append({"file": name, "seed": seed})
            durations.append(time.time() - t_img)
            _update(eid, images=images)
    except Exception as ex:
        _progress.pop(eid, None)
        _update(eid, status="failed", message=f"生成に失敗しました: {ex}", images=images)
        _kept = False
        comfy.after_failure(ex)
        _cancel.discard(eid)
        return
    _progress.pop(eid, None)
    if eid in _cancel:
        _cancel.discard(eid)
        _update(eid, status="cancelled", message=f"中止しました（{len(images)} 枚は残っています）", images=images)
    else:
        _update(eid, status="done", message="完了", images=images)
    if images and wf_model:   # the registered model the workflow used worked on this PC
        registry.mark_checked(wf_model.get("id"), "gen")
    _arm_idle_stop()


def _eta(durations: list[float], images_left: int, this_left: float | None, per_image: float | None = None) -> float | None:
    """Remaining seconds from measured times only: the rest of the current image (from its measured step
    rate) plus the images after it (average of finished images; before the first one finishes, this image's
    sampling time).  None until something has been measured (shown as 計測中)."""
    if durations:
        per = sum(durations) / len(durations)
    elif per_image:
        per = per_image
    else:
        return None if images_left or this_left is None else this_left
    return round((this_left or 0) + images_left * per)


def _arm_idle_stop():
    global _idle_timer
    if _idle_timer:
        _idle_timer.cancel()
    _idle_timer = threading.Timer(KEEP_ALIVE_S, _idle_stop, args=(comfy._uses,))
    _idle_timer.daemon = True
    _idle_timer.start()


def _idle_stop(uses):
    global _kept
    with comfy._lock:
        # another ComfyUI job (comparison, material generation) started since: the process is theirs now
        if _kept and gpu.owner is None and comfy._uses == uses:
            _kept = False
            comfy._stop_locked()


def _stop_kept_comfy():
    """GPU hook: another job (training, evaluation, material generation) takes the GPU.  A job that generates
    with ComfyUI too keeps using the running process (it loads its own model); anything else stops it."""
    global _kept
    if _kept and gpu.owner != LABEL:
        _kept = False
        # a ComfyUI job keeps the loaded model (often the same one, e.g. material made from these settings);
        # ComfyUI unloads it by itself when another needs the room, and that job parks at its end
        if not comfy_user(gpu.owner):
            comfy.stop()


def _park_kept_comfy():
    """GPU hook: material analysis starts.  The analysis needs the VRAM, not the process: the model moves to
    system memory (dropped when that runs short) and ComfyUI keeps running, so the next generation skips the
    start-up and the read from disk.  Before, ComfyUI was stopped here, after every generation that was followed
    by analysis."""
    global _kept
    if _kept and gpu.owner is None:
        _kept = False
        comfy.park()


gpu.on_exclusive(_stop_kept_comfy)
gpu.on_analysis(_park_kept_comfy)
KEEP_COMFY_OWNERS.add(LABEL)


def recover():
    """At startup: requests that were queued or running when the app ended cannot continue."""
    with _lock:
        items = history()
        changed = False
        for e in items:
            if e["status"] in ("queued", "running"):
                e["status"], e["message"] = "cancelled", "アプリの終了で止まりました"
                changed = True
        if changed:
            _save_history(items)
    for e in items:
        if e["status"] == "cancelled" and e["settings"].get("thumb_for"):
            _preset_thumb_done(e["settings"]["thumb_for"], e["id"], None, "アプリの終了で止まりました")
    prune_preset_thumbs()


def to_recipe(settings: dict) -> dict:
    """The fields of a material-generation recipe these settings fill in (size and seed are not copied:
    material generation chooses the size per shot and its own seeds)."""
    return {"gen_model_id": settings["model_id"], "loras": settings["loras"], "positive": settings["positive"],
            "negative": settings["negative"], "sampling": dict(settings.get("sampling_overrides") or {})}


def copy_to_project(projects, eid: str, project_id: str | None, name: str | None) -> dict:
    """Copy a result's settings into a character project's material-generation recipe (a new project if
    none is given).  The project's own choices (number of images, excluded angles) are kept; the model the
    LoRA is trained on follows the generation model unless the user chose another one."""
    from .generate import check_recipe
    e = next((x for x in history() if x["id"] == eid), None)
    if not e:
        raise KeyError(eid)
    if project_id:
        p = projects.get(project_id)
        if p.meta("kind") != "character":
            raise ValueError("キャラクターのプロジェクトを選んでください")
    else:
        p = projects.create((name or "").strip() or time.strftime("キャラ_%m%d_%H%M"), "character", None, "include", "generate")
    if p.meta("source") != "generate":
        p.set_meta(source="generate")
    old = p.meta("gen_recipe") or {}
    recipe = {**old, **to_recipe(e["settings"])}
    p.set_meta(gen_recipe=recipe)
    cur = p.meta("model_id")
    if registry.get(recipe["gen_model_id"]) and (not cur or cur == old.get("gen_model_id")):
        p.set_meta(model_id=recipe["gen_model_id"])
    chk = check_recipe(recipe)
    return {"project_id": p.id, "notes": chk["notes"], "check": chk["errors"]}
