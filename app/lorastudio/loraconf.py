"""Per-LoRA settings (LoRA 設定): trigger words, whether image generation adds them to the prompt, and
the strength a newly added LoRA row starts with.

A LoRA is identified by file name + size, so its settings follow the file when it is moved and are shared
by hard-linked copies (e.g. the one placed in ComfyUI's loras folder).

Until the user saves settings for a LoRA, its trigger words come from what is known for certain:
  * lora_info.json next to a LoRA exported by this tool (the trigger the LoRA was trained with)
  * the trained words Civitai lists for the file, recorded when it is fetched from a Civitai link, or when
    the user presses "Civitai で調べる" (only the file's sha256 is sent; see civitai_lookup)
  * the trigger phrase some trainers write into the file's metadata (modelspec.trigger_phrase)
Words that appear in almost every training caption (sd-scripts' ss_tag_frequency) are only a guess and
are shown as candidates the user can add with one click; they are never added on their own.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path

from . import config

KEY = "lora_settings"
MAX_CANDIDATES = 6
_lock = threading.RLock()
_cache: dict[str, dict] = {}   # file key + mtime -> what the files say


def file_key(path: str | Path) -> str:
    p = Path(path)
    return f"{p.name.lower()}|{p.stat().st_size}"


def _all() -> dict:
    return config.settings().get(KEY, {}) or {}


def clean_words(words) -> list[str]:
    if isinstance(words, str):
        words = [words]
    out = []
    # one entry often holds several words ("abc, red cape" on Civitai): each is checked against the prompt
    for w in (part for item in words or [] for part in str(item).split(",")):
        w = re.sub(r"\s+", " ", str(w)).strip().strip(",")
        if w and w.lower() not in (x.lower() for x in out):
            out.append(w)
    return out


def _from_files(path: Path) -> dict:
    """Trigger words / strength the LoRA's own files state, and caption-frequency candidates."""
    st = path.stat()
    ck = f"{path}|{st.st_size}|{int(st.st_mtime)}"
    if ck in _cache:
        return _cache[ck]
    found = {"words": [], "source": None, "strength": None, "candidates": [], "made": False}
    info = path.parent / "lora_info.json"
    if info.exists():
        try:
            d = json.loads(info.read_text(encoding="utf-8"))
            found["made"] = True   # exported by this tool (lora_info.json is written next to every export)
            if d.get("trigger"):
                found.update(words=clean_words([d["trigger"]]), source="このツールで作った LoRA の呼び出し語")
            if isinstance(d.get("tested_strength"), (int, float)):
                found["strength"] = float(d["tested_strength"])
        except (OSError, json.JSONDecodeError):
            pass
    if path.suffix.lower() == ".safetensors":
        from .models import read_safetensors_header
        try:
            meta = read_safetensors_header(path).get("__metadata__") or {}
        except (OSError, ValueError, UnicodeDecodeError):
            meta = {}
        if not isinstance(meta, dict):
            meta = {}
        tp = meta.get("modelspec.trigger_phrase")
        if isinstance(tp, str) and tp and not found["words"]:
            found.update(words=clean_words(tp), source="LoRA ファイルに書かれた呼び出し語")
        try:
            freq = json.loads(meta.get("ss_tag_frequency") or "{}")
        except (json.JSONDecodeError, TypeError):
            freq = {}
        counts: dict[str, int] = {}
        # written by whatever trained the LoRA: anything that is not {folder: {tag: count}} is skipped
        for tags in freq.values() if isinstance(freq, dict) else []:
            for t, n in (tags.items() if isinstance(tags, dict) else []):
                try:
                    n = int(n or 0)
                except (TypeError, ValueError):
                    continue
                t = t.strip()
                if t:
                    counts[t] = counts.get(t, 0) + n
        if counts:
            top = max(counts.values())
            # a trigger is in (nearly) every caption; ordinary tags are not
            found["candidates"] = [t for t, n in sorted(counts.items(), key=lambda x: -x[1]) if n >= top * 0.9][:MAX_CANDIDATES]
    _cache[ck] = found
    return found


def get(path: str | Path) -> dict:
    """Settings for one LoRA file: {triggers, auto, strength, source, candidates, saved}."""
    p = Path(path)
    try:
        key = file_key(p)
    except OSError:
        return {"triggers": [], "auto": True, "strength": None, "source": None, "candidates": [], "saved": False, "made": False}
    saved = _all().get(key)
    f = _from_files(p)
    known = (saved or {}).get("found") or {}   # e.g. Civitai's trained words recorded at download
    words, source = f["words"], f["source"]
    if not words and known.get("words"):
        words, source = clean_words(known["words"]), known.get("source")
    cands = [c for c in f["candidates"] + clean_words(known.get("words")) if c.lower() not in (w.lower() for w in words)]
    if saved and saved.get("triggers") is not None:
        cur = [c for c in cands + words if c.lower() not in (w.lower() for w in saved["triggers"])]
        return {"triggers": saved["triggers"], "auto": saved.get("auto", True),
                "strength": saved.get("strength", f["strength"]), "source": "手入力", "candidates": clean_words(cur),
                "saved": True, "made": f["made"], "page": known.get("page")}
    return {"page": known.get("page"), "triggers": words, "auto": (saved or {}).get("auto", True), "strength": (saved or {}).get("strength", f["strength"]),
            "source": source, "candidates": clean_words(cands), "saved": False, "made": f["made"]}


def save(path: str | Path, triggers, auto: bool, strength) -> dict:
    p = Path(path)
    key = file_key(p)
    with _lock:
        data = dict(_all())
        cur = dict(data.get(key) or {})
        cur.update(name=p.name, triggers=clean_words(triggers), auto=bool(auto),
                   strength=float(strength) if strength not in (None, "") else None)
        data[key] = cur
        config.settings().set(KEY, data)
    return get(p)


def reset(path: str | Path) -> dict:
    """Back to what the files say (recorded Civitai words are kept)."""
    p = Path(path)
    key = file_key(p)
    with _lock:
        data = dict(_all())
        cur = data.get(key) or {}
        if cur.get("found"):
            data[key] = {"name": p.name, "found": cur["found"]}
        else:
            data.pop(key, None)
        config.settings().set(KEY, data)
    return get(p)


def record_found(path: str | Path, words, source: str, **extra) -> None:
    """Trigger words a download site lists for this file (not the user's choice; used until they save)."""
    words = clean_words(words)
    if not words:
        return
    p = Path(path)
    key = file_key(p)
    with _lock:
        data = dict(_all())
        cur = dict(data.get(key) or {})
        cur.update(name=p.name, found={"words": words, "source": source, **extra})
        data[key] = cur
        config.settings().set(KEY, data)


def sha256(path: Path) -> str:
    """sha256 of the file, shared with the model hash cache (computed here at full speed: LoRAs are small
    next to base models, and the user is waiting for the answer)."""
    st = path.stat()
    key = f"{path}|{st.st_size}|{int(st.st_mtime)}"   # same key as models.Hasher
    h = config.settings().get("hash_cache", {}).get(key)
    if h:
        return h
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            d.update(chunk)
    h = d.hexdigest()
    with config.settings()._lock:   # models.Hasher writes the same cache
        cache = dict(config.settings().get("hash_cache", {}))
        cache[key] = h
        config.settings().set("hash_cache", cache)
    return h


def civitai_lookup(path: str | Path, token: str | None = None) -> dict:
    """Find the file on Civitai by its sha256 and record the trigger words listed there.  Only the hash is
    sent.  Returns {found, title, page, words, conf}."""
    from . import remote
    p = Path(path)
    h = sha256(p)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        v = remote._json(f"https://civitai.com/api/v1/model-versions/by-hash/{h}", headers)
    except ValueError:   # 404
        return {"found": False, "conf": get(p)}
    title = f"{(v.get('model') or {}).get('name') or '名前不明'}（{v.get('name') or ''}）"
    page = f"https://civitai.com/models/{v.get('modelId')}?modelVersionId={v.get('id')}"
    words = clean_words(v.get("trainedWords") or [])
    record_found(p, words, "Civitai に書かれた呼び出し語", title=title, page=page)
    return {"found": True, "title": title, "page": page, "words": words, "conf": get(p)}


def _has(prompt: str, word: str) -> bool:
    return re.search(r"(?<![\w])" + re.escape(word.lower()) + r"(?![\w])", prompt.lower(), re.ASCII) is not None


def row_words(lo: dict, s: dict) -> list[str]:
    """The words a LoRA row puts in: the ones chosen in the row (a LoRA with several trigger words: the row's
    pull-down, 2026-10-27), else all of the LoRA's trigger words (rows saved before that)."""
    return clean_words(lo["words"]) if isinstance(lo.get("words"), list) else s["triggers"]


def apply(positive: str, loras: list | None) -> tuple[str, list[str]]:
    """The prompt with the trigger words of the LoRAs whose trigger switch is on put in front, skipping
    words the prompt already has.  A row without its own switch follows the LoRA's setting."""
    add: list[str] = []
    for lo in loras or []:
        if not lo.get("path"):
            continue
        s = get(lo["path"])
        if not (lo.get("trigger") if lo.get("trigger") is not None else s["auto"]):
            continue
        for w in row_words(lo, s):
            if not _has(positive, w) and w.lower() not in (x.lower() for x in add):
                add.append(w)
    if not add:
        return positive, []
    positive = positive.strip()
    return (", ".join(add) + (", " + positive if positive else "")), add
