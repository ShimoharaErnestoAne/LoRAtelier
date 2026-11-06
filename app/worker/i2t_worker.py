"""Resident i2t worker: runs the bundled copy of i2t (engines/i2t/i2t.py, unchanged) in the analysis
environment and keeps its Engine (tagger / person / pose / era models, Ollama connection) loaded between
requests.  Every feature of the i2t window is reached through the ops below; the screens are in the web UI.

Protocol: one JSON object per line on stdin; replies and events as JSON lines on stdout.
  {"rid": 1, "op": "analyze", "path": ..., "refs": {...}, "controls": {...}, "cfg": {...}, "mode": "single"}
  {"rid": 2, "op": "call", "fn": "history_entries", "args": [...]}          plain i2t functions (whitelist)
  {"rid": 3, "op": "<job>", ...}   long jobs: tagger_download / face_models / pull / translate / ollama_setup / moss_catalog
                                   / pose_estimate (one at a time; "cancel" stops them)
  {"op": "cancel"} / {"op": "pause"} / {"op": "resume"}                     act on the running job
Events: {"event": "ready"} / {"event": "progress", "rid", "message", "fraction"?} / {"event": "fraction", ...}
        / {"event": "stop_state", "rid", "cancellable"} (steps that must not be interrupted, e.g. installing)

i2t keeps its config, history, translations and user dictionaries next to i2t.py and its models in
models/; here they are moved to LoRAtelier's data folder (argv[2]) and model folder (argv[3]) by
rebinding the module's path constants before anything is loaded.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import shutil
import sys
import tempfile
import threading
import traceback
from pathlib import Path

I2T_DIR = Path(sys.argv[1])
DATA_DIR = Path(sys.argv[2])
MODELS_DIR = Path(sys.argv[3])
_out_lock = threading.Lock()


def emit(obj: dict) -> None:
    line = json.dumps(obj, ensure_ascii=False, default=_jsonable)
    with _out_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def _jsonable(o):
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    if hasattr(o, "tolist"):          # numpy
        return o.tolist()
    return repr(o)


def load_i2t():
    spec = importlib.util.spec_from_file_location("i2t", I2T_DIR / "i2t.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["i2t"] = mod
    spec.loader.exec_module(mod)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # every path constant i2t derives from its own folder (keep in sync with i2t.py when the copy is updated)
    mod.APP_DIR = DATA_DIR
    mod.RESOURCE_DIR = I2T_DIR
    mod.CONFIG_PATH = DATA_DIR / "config.json"
    mod.MODELS_DIR = MODELS_DIR
    mod.PERSON_MODEL_PATH = MODELS_DIR / "person" / "person_detect_v1.3_s.onnx"
    mod.HEAD_MODEL_PATH = MODELS_DIR / "head" / "head_detect_v2.0_s.onnx"
    # the detectors copy their path into a class attribute at import time
    mod.PersonDetector.model_path = mod.PERSON_MODEL_PATH
    mod.HeadDetector.model_path = mod.HEAD_MODEL_PATH
    mod.POSE_MODEL_PATH = MODELS_DIR / "pose" / "rtmpose-s_body7_256x192.onnx"
    mod.ERA_MODEL_PATH = MODELS_DIR / "style" / "anime_style_ages_mobilenetv3.onnx"
    mod.MOSS_CATALOG_PATH = MODELS_DIR / "moss_catalog" / "index.json"   # LoRAtelier reads the same file (moss_catalog.py)
    mod.TRANSLATIONS_PATH = DATA_DIR / "translations.json"
    mod.USER_DICT_DIR = DATA_DIR / "dict"
    mod.HISTORY_DIR = DATA_DIR / "history"
    mod.PASTE_DIR = DATA_DIR / "paste"
    mod.log = lambda msg: sys.stderr.write(str(msg) + "\n")
    mod.APP_NAME = "i2t"
    return mod


i2t = None
engine = None
_job: dict = {"rid": None, "cancel": None, "gate": None, "vlm": None}
_job_lock = threading.Lock()


# ------------------------------------------------------------------ helpers
def _progress(rid):
    def f(msg, frac=-1.0):
        emit({"event": "progress", "rid": rid, "message": str(msg), "fraction": frac})
    return f


def _result_from(d: dict):
    names = {f.name for f in dataclasses.fields(i2t.PromptResult)}
    return i2t.PromptResult(**{k: v for k, v in d.items() if k in names})


def texts(result: dict) -> dict:
    """The text forms the i2t window shows / saves for a result (after chips were deleted, too)."""
    r = _result_from(result)
    return {"to_text": r.to_text(), "extras_text": r.extras_text(), "tag_lines": r.tag_lines(), "person_lines": r.person_lines()}


def consts() -> dict:
    m = i2t
    return {
        "version": m.__version__, "app_name": m.APP_NAME, "translate_choices": m.TRANSLATE_CHOICES,
        "engines": m.ENGINES, "targets": m.TARGETS, "modes": m.MODES, "ref_slots": m.REF_SLOTS, "ref_what_ja": m.REF_WHAT_JA,
        "tagger_models": {k: {"repo": v[0], "label": v[1], "ready": m.tagger_ready(k)} for k, v in m.TAGGER_MODELS.items()},
        "mosaic_levels": {k: {kk: vv for kk, vv in v.items() if isinstance(vv, (str, int, float, bool))} for k, v in m.MOSAIC_LEVELS.items()},
        "camera_options": {d: [[v, lab] for v, lab, _tags in opts] for d, opts in m.CAMERA_OPTIONS.items()},
        "camera_labels_ja": m.CAMERA_LABELS_JA, "look_none": m.LOOK_NONE,
        "look_parts": {k: [[v, lab] for v, lab in opts] for k, opts in m.LOOK_PARTS.items()},
        "look_part_labels_ja": m.LOOK_PART_LABELS_JA, "look_parts_menu": m.LOOK_PARTS_MENU,
        "strength_groups": m.STRENGTH_GROUPS, "strength_weight": m.STRENGTH_WEIGHT,
        "extra_kinds": m.EXTRA_KINDS, "extra_short_ja": m.EXTRA_SHORT_JA, "extra_button_rows": m.EXTRA_BUTTON_ROWS,
        "help_sections": m.HELP_SECTIONS, "vlm_catalog": m.VLM_CATALOG, "default_config": m.DEFAULT_CONFIG,
        "keypoints": m.KEYPOINTS, "pose_bones": m.POSE_BONES, "pose_template": m.POSE_TEMPLATE,
        "pose_template_size": m.POSE_TEMPLATE_SIZE, "pose_min_conf": m.POSE_MIN_CONF,
        "face_models_ja": m.FACE_MODELS_JA, "pose_model_mb": m.POSE_MODEL_MB, "image_exts": m.IMAGE_EXTS,
        "installer_gb": m.OLLAMA_INSTALLER_GB, "app_gb": m.OLLAMA_APP_GB,
        "default_install_dir": str(m.default_ollama_install_dir()), "models_dir": str(MODELS_DIR), "data_dir": str(DATA_DIR),
        "history_dir": str(m.HISTORY_DIR), "history_limit": m.HISTORY_LIMIT,
        "moss_catalog_ready": m.moss_catalog_ready(), "moss_catalog_mb": m.MOSS_CATALOG_MB,
        "moss_catalog_page": m.MOSS_CATALOG_PAGE, "moss_catalog_dir": str(m.MOSS_CATALOG_PATH.parent),
    }


def probe_ollama(host: str, model: str) -> dict:
    """Models on the Ollama server and which of them take images (the i2t window's 「Ollama: …」)."""
    models = i2t.OllamaVLM(host, model).list_models()
    vision = []
    for name in models:
        try:
            if "vision" in i2t.OllamaVLM(host, name).capabilities():
                vision.append(name)
        except i2t.OllamaError:
            continue
    return {"models": models, "vision": vision}


def chip_table() -> dict:
    """Bundled dictionary + dict/*.csv + translations.json (later ones win), as the window loads it."""
    table = i2t.load_tag_dictionary()
    table.update(i2t.load_translations())
    return table


def chips_ja(chunks: list[str], table: dict | None = None) -> list[str]:
    table = table if table is not None else chip_table()
    return [i2t.chip_label_ja(c, table) for c in chunks]


def missing_ja(chunks: list[str], character_tags: list[str]) -> list[str]:
    """Chunks the dictionaries do not know (character names are never translated nor sent to Ollama)."""
    table = chip_table()
    names = {i2t.chip_core(t)[0].lower() for t in character_tags}
    return i2t.dedup([core for core, _w in map(i2t.chip_core, chunks)
                      if core and not i2t.lookup_ja(table, core) and core.lower() not in names])


def ja_table() -> dict:
    """The whole chip dictionary for the image generation page's 日本語 view, which turns edited Japanese back
    into English on the page: "table" is what chips_ja uses, "user" the registered / translated words
    (translations.json), which win when two English words share one Japanese word."""
    return {"table": chip_table(), "user": i2t.load_translations(), "readings": _load_readings()}


# readings (hiragana) of the words registered here, for the tag completion: kept beside translations.json in a
# file of their own, so translations.json stays in i2t's format
READINGS_PATH = DATA_DIR / "tag_readings_user.json"
_KANJI = __import__("re").compile(r"[\u3400-\u9fff々]")
_HIRA = __import__("re").compile(r"^[\u3041-\u3096ー・ 　]+$")


def _load_readings() -> dict:
    try:
        d = json.loads(READINGS_PATH.read_text("utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def register_ja(en: str, ja: str, reading: str = "") -> dict:
    """Register (or with an empty ja, remove) the Japanese for one English chip in translations.json, the
    file the i2t page's translations go to, with its reading in hiragana (required when the Japanese has kanji).
    Returns the saved {key: ja}."""
    key = i2t.chip_core(str(en or ""))[0].lower()
    if not key:
        raise ValueError("英語を入れてください")
    ja = str(ja or "").strip()
    reading = "".join(chr(ord(c) - 0x60) if "\u30a1" <= c <= "\u30f6" else c for c in str(reading or "").strip())
    if ja and _KANJI.search(ja) and not reading:
        raise ValueError(f"「{ja}」の読みを、ひらがなで入れてください")
    if reading and not _HIRA.match(reading):
        raise ValueError("読みは、ひらがなで入れてください")
    table = i2t.load_translations()
    if ja:
        table[key] = ja
    else:
        table.pop(key, None)
    if not i2t.save_translations(table):
        raise ValueError("辞書を保存できませんでした")
    readings = _load_readings()
    if ja and reading:
        readings[key] = reading
    else:
        readings.pop(key, None)
    if not _save_readings(readings):
        raise ValueError("読みを保存できませんでした（訳は登録しました）")
    return {key: ja}


def _save_readings(readings: dict) -> bool:
    try:
        tmp = READINGS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(readings, ensure_ascii=False, indent=1, sort_keys=True), "utf-8")
        tmp.replace(READINGS_PATH)
        return True
    except OSError:
        return False


def _to_hira(t: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "\u30a1" <= c <= "\u30f6" else c for c in str(t or "").strip())


def save_openpose(points, size, dest: str) -> str:
    i2t.render_openpose(points, tuple(size)).save(dest)
    return dest


def system_fit() -> dict:
    s = i2t.system_specs()
    return {"specs": s, "recommend": i2t.recommend_vlm(s.get("vram_gb"), s.get("ram_gb") or 0),
            "fit": {name: i2t.model_fit(s.get("vram_gb"), s.get("ram_gb") or 0, need, size)
                    for name, size, need, _desc in i2t.VLM_CATALOG}}


CALLS = {
    # results / chips / translation
    "texts": texts, "split_prompt_chips": None, "join_prompt_chips": None, "chips_ja": chips_ja, "missing_ja": missing_ja,
    "ja_table": ja_table, "register_ja": register_ja,
    "chip_core": None,
    # config
    "load_config": None, "save_config": None, "normalize_host": None, "find_look": None, "camera_tags": None,
    # history
    "history_entries": None, "history_update_prompts": None, "history_delete": None,
    # images
    "read_generation_info": None,
    # models
    "tagger_ready": None, "missing_face_models": None, "pose_model_ready": None, "face_models_ready": None,
    # pose
    "pose_tags_from_points": None, "save_openpose": save_openpose,
    # Ollama / HF
    "probe_ollama": probe_ollama, "parse_hf_model": None, "check_hf_model": None, "gather_ollama_info": None,
    "system_fit": system_fit, "free_gb": None, "ollama_models_dir": None,
    "consts": consts,
}


def history_result(path: str) -> dict:
    r, extra = i2t.history_result(Path(path))
    return {"result": dataclasses.asdict(r), "extra": extra}


CALLS["history_result"] = history_result


def _call(rid: int, req: dict) -> None:
    try:
        name = req["fn"]
        if name not in CALLS:
            raise ValueError(f"呼べない関数です: {name}")
        fn = CALLS[name] or getattr(i2t, name)
        args = list(req.get("args") or [])
        if name in ("free_gb", "history_update_prompts") and args:
            args[0] = Path(args[0])
        if name in ("read_generation_info",):
            args[0] = Path(args[0])
        if name == "save_config":           # the window saves its whole settings dict
            fn(args[0])
            value = i2t.load_config()
        else:
            value = fn(*args, **(req.get("kwargs") or {}))
        emit({"rid": rid, "ok": True, "value": value})
    except Exception as e:
        emit({"rid": rid, "ok": False, "error": i2t.describe_error(e) if i2t else str(e),
              "trace": traceback.format_exc()[-4000:]})


# ------------------------------------------------------------------ analysis
def _analyze(rid: int, req: dict) -> None:
    try:
        cfg = {**i2t.load_config(), **(req.get("cfg") or {})}
        engine.cfg = cfg
        engine.progress = lambda m: emit({"event": "progress", "rid": rid, "message": str(m)})
        engine.fraction = lambda f, est=None: emit({"event": "fraction", "rid": rid, "value": f, "est": est})
        engine.reset_cancel()
        path = Path(req["path"])
        refs = {k: Path(v) for k, v in (req.get("refs") or {}).items() if v}
        result = engine.run(path, refs, req.get("controls") or {})
        vlm_kept = _kept_vlm()
        if engine.last_vlm:   # remembered per model for the next estimate (as the window does)
            model, secs = engine.last_vlm
            saved = i2t.load_config()
            saved["vlm_seconds"] = {**(saved.get("vlm_seconds") or {}), model: secs}
            i2t.save_config(saved)
            engine.last_vlm = None
        hist = None
        try:
            from PIL import Image
            with Image.open(path) as im:
                thumb = im.convert("RGB")
                thumb.thumbnail((256, 256))
            hist = i2t.history_add(result, thumb, {"mosaic_strength": cfg.get("mosaic_strength"), "mode": req.get("mode", "single")})
        except Exception as e:   # the history is a convenience; the result is shown either way
            sys.stderr.write(f"history: {e}\n")
        emit({"rid": rid, "ok": True, "result": dataclasses.asdict(result), "history": str(hist) if hist else None,
              "vlm_kept": vlm_kept})
    except i2t.OperationCancelled:
        emit({"rid": rid, "ok": False, "cancelled": True, "error": "中止しました", "vlm_kept": _kept_vlm()})
    except Exception as e:
        emit({"rid": rid, "ok": False, "error": i2t.describe_error(e), "trace": traceback.format_exc()[-4000:],
              "vlm_kept": _kept_vlm()})
    finally:
        _end_job()


# i2t's 省 VRAM setting (low_vram) unloads the VLM after every request (keep_alive 0), so images analysed one after
# another load the model again each time.  Here it stays for VLM_KEEP; LoRAtelier unloads it at once when training,
# generation or material analysis takes the GPU (i2tlink.unload_kept_vlm), which is what the setting is for.
VLM_KEEP = "2m"


def _keep_vlm_loaded(vlm) -> None:
    """Every request of this VLM keeps the model for VLM_KEEP while low_vram is on (looked at per request)."""
    if getattr(vlm, "_ls_keep", False):
        return
    chat = vlm._chat

    def keep_chat(payload, *a, **kw):
        if engine._low_vram():
            payload["keep_alive"] = VLM_KEEP
        return chat(payload, *a, **kw)
    vlm._chat = keep_chat
    vlm._ls_keep = True


def _kept_vlm() -> dict | None:
    """The VLM this worker may have left loaded (sent with the reply so the server can unload it)."""
    v = engine._vlm
    if v is None or not engine._low_vram():
        return None
    return {"host": v.host, "model": v.model}


def _wrap_engine_vlm(eng) -> None:
    orig = eng.vlm

    def vlm():
        v = orig()
        _keep_vlm_loaded(v)
        return v
    eng.vlm = vlm


# ------------------------------------------------------------------ other long jobs
def _tagger_download(rid, req, cancel):
    i2t.download_tagger(req["name"], _progress(rid), cancel)
    return {"name": req["name"]}


def _face_models(rid, req, cancel):
    i2t.ensure_face_models(lambda m: _progress(rid)(m), cancel)
    return {"missing": i2t.missing_face_models()}


def _moss_catalog(rid, req, cancel):
    i2t.download_moss_catalog(_progress(rid), cancel)
    return {"ready": i2t.moss_catalog_ready()}


def _pull(rid, req, cancel):
    name = req["name"]
    i2t.OllamaVLM(req["host"], name).pull(name, progress=lambda m: _progress(rid)(m), cancel=cancel)
    return {"name": name}


def _translate(rid, req, cancel):
    cfg = i2t.load_config()
    model = str(cfg.get("translate_model") or i2t.DEFAULT_CONFIG["translate_model"])
    vlm = i2t.OllamaVLM(cfg["ollama_host"], model, int(cfg.get("ollama_timeout") or i2t.VLM_TIMEOUT))
    vlm.cancel_event = cancel
    _job["vlm"] = vlm
    keep = 0 if cfg.get("low_vram", True) else None
    gate = _job["gate"]

    def checkpoint():
        if cancel.is_set():
            raise i2t.OperationCancelled()
        gate.wait(cancel.is_set, lambda: _progress(rid)("一時停止中 (「▶ 再開」で続きから)"))

    _progress(rid)(f"日本語に訳しています ({len(req['words'])} 個、{model})...")
    found = i2t.translate_chips_ja(vlm, req["words"], keep_alive=keep, checkpoint=checkpoint)
    table = i2t.load_translations()
    table.update(found)
    i2t.save_translations(table)
    out = {"translated": found, "model": model}
    need = [(en, ja) for en, ja in found.items() if _KANJI.search(ja)]
    if need:   # readings for the tag completion; the translations are kept even when this fails
        try:
            _progress(rid)(f"読みを調べています ({len(need)} 語)...")
            out["readings"] = len(_store_readings(_text_llm(cfg, cancel), need, keep, checkpoint))
        except i2t.OperationCancelled:
            raise
        except Exception as e:
            out["readings_error"] = i2t.describe_error(e)
    return out


# ------------------------------------------------------------------ Japanese -> English, readings (Ollama)
_LLM_BATCH = 20


def _text_llm(cfg: dict, cancel):
    """A model that follows instructions: the translation model unless it is a translation-only one
    (Index-Translate), else i2t's VLM (it reads text as well)."""
    model = str(cfg.get("translate_model") or "")
    if not model or i2t.is_translation_model(model):
        model = str(cfg.get("ollama_model") or i2t.DEFAULT_CONFIG["ollama_model"])
    vlm = i2t.OllamaVLM(cfg["ollama_host"], model, int(cfg.get("ollama_timeout") or i2t.VLM_TIMEOUT))
    vlm.cancel_event = cancel
    _job["vlm"] = vlm
    return vlm


def _ask_list(vlm, system: str, items: list[str], keep, checkpoint, max_len: int = 120) -> list[str]:
    """One answer string per item, in order (a JSON schema fixes the count, as translate_chips_ja does)."""
    import re
    thinking = "thinking" in vlm.capabilities()
    out: list[str] = []
    for start in range(0, len(items), _LLM_BATCH):
        checkpoint()
        batch = items[start:start + _LLM_BATCH]
        payload = {
            "model": vlm.model, "stream": False,
            "format": {"type": "object", "required": ["out"], "properties": {"out": {
                "type": "array", "minItems": len(batch), "maxItems": len(batch),
                "items": {"type": "string", "maxLength": max_len}}}},
            "options": {"temperature": 0.1, "num_ctx": i2t.VLM_NUM_CTX,
                        "num_predict": (i2t.VLM_NUM_PREDICT_THINKING if thinking else 200) + 60 * len(batch)},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": "Items:\n" + "\n".join(f"{i + 1}. {w}" for i, w in enumerate(batch))}],
        }
        if thinking:
            payload["think"] = False
        if keep is not None:
            payload["keep_alive"] = keep
        data = vlm._chat(payload)
        content = re.sub(r"(?s)^\s*<think>.*?</think>", "", str((data.get("message") or {}).get("content") or "")).strip()
        try:
            got = json.loads(content).get("out")
        except (ValueError, AttributeError):
            raise i2t.OllamaError(f"{vlm.model} の応答を読めませんでした: {content[:200]}") from None
        if not isinstance(got, list) or len(got) != len(batch):
            raise i2t.OllamaError(f"{vlm.model} の応答の数が合いません。")
        out += [str(x or "").strip() for x in got]
    return out


READING_PROMPT = ("For each Japanese word or phrase, write how it is read, in hiragana only. Convert katakana to "
                  "hiragana and keep the long vowel mark ー (e.g. 猫耳フード -> ねこみみふーど). No kanji, no katakana, no "
                  "romaji, no spaces, no explanations. Keep the order and return exactly one reading per item.")
TO_ENGLISH_PROMPT = ("You turn Japanese words written by a user of an anime image generator into English prompt tags. For "
                     "each item, return the Danbooru tag with that meaning, in lowercase with spaces instead of underscores "
                     "(e.g. 猫耳 -> cat ears, 笑顔 -> smile, 座っている -> sitting). When no tag fits, return a short English "
                     "phrase. Keep the order and return exactly one English string per item, without explanations.")


def _store_readings(vlm, pairs: list[tuple[str, str]], keep, checkpoint) -> dict:
    """Ask the readings of the Japanese of (en, ja) pairs and save the ones written in hiragana only."""
    answers = _ask_list(vlm, READING_PROMPT, [ja for _en, ja in pairs], keep, checkpoint, 80)
    got = {}
    for (en, _ja), r in zip(pairs, answers):
        r = _to_hira(r)
        if r and _HIRA.match(r):
            got[i2t.chip_core(en)[0].lower()] = r
    if got:
        readings = _load_readings()
        readings.update(got)
        if not _save_readings(readings):
            raise ValueError("読みを保存できませんでした")
    return got


def _checkpoint(rid, cancel):
    gate = _job["gate"]

    def check():
        if cancel.is_set():
            raise i2t.OperationCancelled()
        gate.wait(cancel.is_set, lambda: _progress(rid)("一時停止中 (「▶ 再開」で続きから)"))
    return check


def _translate_en(rid, req, cancel):
    """Japanese typed on the image page that the dictionary does not know -> English, registered in
    translations.json with readings.  A word whose English is already in the dictionary under another Japanese is
    not registered over it ("known": the page writes the dictionary's Japanese instead)."""
    import re
    cfg = i2t.load_config()
    keep = 0 if cfg.get("low_vram", True) else None
    check = _checkpoint(rid, cancel)
    words = i2t.dedup([str(w).strip() for w in req.get("words") or [] if str(w).strip()])
    vlm = _text_llm(cfg, cancel)
    _progress(rid)(f"英語に訳しています ({len(words)} 個、{vlm.model})...")
    answers = _ask_list(vlm, TO_ENGLISH_PROMPT, words, keep, check)
    table, mine = chip_table(), i2t.load_translations()
    new, known, failed = {}, {}, []
    for ja, en in zip(words, answers):
        en = re.sub(r"\s+", " ", en.replace("_", " ")).strip().strip(".").lower()
        if not en or not re.search(r"[a-z]", en) or re.search(r"[\u3040-\u30ff\u3400-\u9fff]", en):
            failed.append(ja)
            continue
        key = i2t.chip_core(en)[0].lower()
        if key in table and table[key] != ja:
            known[ja] = [key, table[key]]
            continue
        mine[key] = ja
        new[ja] = key
    if new and not i2t.save_translations(mine):
        raise ValueError("辞書を保存できませんでした")
    out = {"translated": new, "known": known, "failed": failed, "model": vlm.model}
    need = [(en, ja) for ja, en in new.items() if _KANJI.search(ja)]
    if need:
        try:
            _progress(rid)(f"読みを調べています ({len(need)} 語)...")
            out["readings"] = len(_store_readings(vlm, need, keep, check))
        except i2t.OperationCancelled:
            raise
        except Exception as e:
            out["readings_error"] = i2t.describe_error(e)
    return out


def _readings(rid, req, cancel):
    """Readings of the registered words with kanji that have none (the dictionary page's button)."""
    cfg = i2t.load_config()
    have = _load_readings()
    pairs = [(en, ja) for en, ja in i2t.load_translations().items() if _KANJI.search(ja) and en not in have]
    if not pairs:
        return {"found": 0, "asked": 0}
    vlm = _text_llm(cfg, cancel)
    _progress(rid)(f"読みを調べています ({len(pairs)} 語、{vlm.model})...")
    got = _store_readings(vlm, pairs, 0 if cfg.get("low_vram", True) else None, _checkpoint(rid, cancel))
    return {"found": len(got), "asked": len(pairs), "model": vlm.model}


def _pose_estimate(rid, req, cancel):
    from PIL import Image
    missing = not i2t.pose_model_ready() or not i2t.model_file_ok(i2t.PERSON_MODEL_PATH)
    _progress(rid)("モデルを取得中... (初回のみ)" if missing else "骨格を推定中...")
    with Image.open(req["path"]) as im:
        image = im.convert("RGB")
    estimator = i2t.PoseEstimator()
    box = i2t.main_person_box(engine.person_detector().detect(image))
    points = [list(p) for p in estimator.estimate(image, box)]
    hidden = [i2t.KEYPOINTS[i][1] for i, p in enumerate(points) if p[2] < i2t.POSE_MIN_CONF]
    return {"points": points, "size": list(image.size), "hidden": hidden}


def _ollama_setup(rid, req, cancel):
    """The i2t setup dialog's OK: install / start / only change the models folder (see App._on_ollama_setup_ready)."""
    report = _progress(rid)
    action, host = req["action"], req["host"]
    models_dir, current = Path(req["models_dir"]), Path(req["current_models_dir"])
    exe = req.get("exe")
    tmp_dir = None
    try:
        if action == "install":
            tmp_dir = Path(tempfile.mkdtemp(prefix="i2t_ollama_"))
            installer = tmp_dir / "OllamaSetup.exe"
            report(f"Ollama のインストーラをダウンロード中 (約 {i2t.OLLAMA_INSTALLER_GB:.1f} GB)...")
            i2t.download_file(i2t.OLLAMA_INSTALLER_URL, installer, report, label="OllamaSetup.exe", cancel=cancel)
            if cancel.is_set():
                raise i2t.OperationCancelled()
            emit({"event": "stop_state", "rid": rid, "cancellable": False})   # installing must not be interrupted
            if models_dir != current:
                i2t.set_ollama_models_dir(models_dir)
            exe = i2t.install_ollama(installer, Path(req["install_dir"]), lambda m: report(m))
            report("Ollama を起動しています...")
            i2t.start_ollama(exe)
        elif action == "start":
            emit({"event": "stop_state", "rid": rid, "cancellable": False})
            if models_dir != current:
                i2t.set_ollama_models_dir(models_dir)
            report("Ollama を起動しています...")
            i2t.start_ollama(Path(exe))
        elif models_dir != current:
            i2t.set_ollama_models_dir(models_dir)
            report("モデル保存先を変更しました。Ollama を再起動すると反映されます")
        if action in ("install", "start") and not i2t.wait_ollama(host, timeout=90):
            raise i2t.OllamaError("Ollama を起動しましたが応答がありません。しばらく待ってから「再接続」を押してください。")
        return {"action": action}
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


JOBS = {"tagger_download": _tagger_download, "face_models": _face_models, "pull": _pull, "translate": _translate,
        "pose_estimate": _pose_estimate, "ollama_setup": _ollama_setup, "translate_en": _translate_en, "readings": _readings,
        "moss_catalog": _moss_catalog}
CANCEL_MESSAGES = {"tagger_download": "ダウンロードを中止しました", "face_models": "モデルの取得を中止しました",
                   "pull": "取得を中止しました (もう一度「追加 (pull)」を押すと続きから取得します)",
                   "translate": "日本語訳を中止しました", "pose_estimate": "骨格の推定を中止しました",
                   "translate_en": "英語への訳を中止しました", "readings": "読みの調べを中止しました",
                   "moss_catalog": "キャラ一覧の取得を中止しました (キャラ名はそのまま書きます)",
                   "ollama_setup": "Ollama の導入を中止しました (インストーラのダウンロード途中のファイルは削除しました)"}


def _run_job(rid: int, op: str, req: dict) -> None:
    cancel = _job["cancel"]
    try:
        value = JOBS[op](rid, req, cancel)
        emit({"rid": rid, "ok": True, "value": value})
    except i2t.OperationCancelled:
        emit({"rid": rid, "ok": False, "cancelled": True, "error": CANCEL_MESSAGES[op]})
    except Exception as e:
        emit({"rid": rid, "ok": False, "error": i2t.describe_error(e), "trace": traceback.format_exc()[-4000:]})
    finally:
        _end_job()


def _begin_job(rid: int) -> bool:
    with _job_lock:
        if _job["rid"] is not None:
            return False
        _job.update(rid=rid, cancel=threading.Event(), gate=i2t.PauseGate(), vlm=None)
        return True


def _end_job() -> None:
    with _job_lock:
        _job.update(rid=None, cancel=None, gate=None, vlm=None)


def main() -> None:
    global i2t, engine
    i2t = load_i2t()
    engine = i2t.Engine(i2t.load_config())
    _wrap_engine_vlm(engine)
    emit({"event": "ready", "version": i2t.__version__})
    for raw in sys.stdin:
        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            continue
        op, rid = req.get("op"), req.get("rid")
        if op == "cancel":
            engine.cancel()
            if _job["cancel"] is not None:
                _job["cancel"].set()
            if _job["vlm"] is not None:
                _job["vlm"].abort()
        elif op == "pause":
            engine.pause_gate.pause()
            if _job["gate"] is not None:
                _job["gate"].pause()
        elif op == "resume":
            engine.pause_gate.resume()
            if _job["gate"] is not None:
                _job["gate"].resume()
        elif op == "analyze" or op in JOBS:
            if not _begin_job(rid):
                emit({"rid": rid, "ok": False, "error": "ほかの処理の最中です。終わってからもう一度押してください。"})
            elif op == "analyze":
                threading.Thread(target=_analyze, args=(rid, req), daemon=True).start()
            else:
                threading.Thread(target=_run_job, args=(rid, op, req), daemon=True).start()
        elif op == "call":
            threading.Thread(target=_call, args=(rid, req), daemon=True).start()
        elif op == "ping":
            emit({"rid": rid, "ok": True})


if __name__ == "__main__":
    main()
