"""Adding a model or LoRA from a link (Hugging Face / Civitai) or from a file the user drops in.

Before anything large is downloaded, the first bytes of each file are fetched (HTTP Range) and the
safetensors / GGUF header is read, so the tool can say what the file is (SDXL checkpoint, Anima DiT,
LoRA, VAE, ...) and where it goes - or that it cannot be used - without a multi-GB download.  The
site's own information (Civitai's base model, model type) is shown next to it as a hint only; the
header decides.

Only .safetensors and .gguf are fetched.  Pickle formats (.ckpt / .pt / .bin) can run code when
loaded and are never downloaded.  Only huggingface.co and civitai.com links are accepted; tokens are
used for the request and not stored.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import config
from .events import bus
from .models import (FAMILY_LABEL, detect_gguf, detect_header, detect_lora_header, is_lora_keys, library, registry)

UA = {"User-Agent": "LoRAtelier/1.0"}
SAFE_EXT = (".safetensors", ".gguf")
MAX_PROBE = 12                 # files probed per link; the rest are checked when added
GGUF_PROBE_LIMIT = 48 << 20    # GGUF tensor tables of large models can be tens of MB

# Civitai "baseModel" -> this tool's variant.  Only names that map to one variant; "SDXL 1.0" is used by
# anime and photo fine-tunes alike, so it is left to the user.
CIVITAI_BASE = {"pony": "pony", "illustrious": "illustrious", "noobai": "noobai_eps", "anima": "anima",
                "qwen": "qwen_image", "zimageturbo": "zimage_turbo", "zimagebase": "zimage_base"}

DEST_LABEL = {"checkpoints": "checkpoint", "diffusion_models": "モデル本体", "loras": "LoRA",
              "text_encoders": "テキストエンコーダー", "vae": "VAE"}

_state: dict[str, dict] = {}
_lock = threading.Lock()


class Cancelled(Exception):
    pass


def cancel(key: str) -> bool:
    """Stops at the next 8MB chunk; the half-fetched file is removed."""
    st = _state.get(key)
    if not st or st.get("state") != "running" or st.get("src") == "upload":
        return False
    st.update(cancel=True, message="キャンセル中")
    return True


class NeedToken(Exception):
    pass


def state() -> list[dict]:
    return sorted(_state.values(), key=lambda s: -s["started"])


# ======================================================================== HTTP
def _open(url: str, headers: dict | None = None, timeout=30):
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise NeedToken(str(e.code)) from None
        if e.code == 404:
            raise ValueError("ページまたはファイルが見つかりません（リンクを確認してください）") from None
        raise


def _json(url: str, headers: dict | None = None):
    with _open(url, headers) as r:
        return json.loads(r.read())


def _range(url: str, start: int, end: int, headers: dict | None = None) -> bytes:
    with _open(url, {**(headers or {}), "Range": f"bytes={start}-{end}"}) as r:
        data = r.read(end - start + 1)
        if r.status != 206 and start > 0:
            raise ValueError("このサーバーは部分取得に対応していません")
        return data


# ======================================================================== classification
def classify_header(hdr: dict, name: str) -> dict:
    """What a safetensors file is and where it goes under models/."""
    keys = [k for k in hdr if k != "__metadata__"]
    if is_lora_keys(keys):
        d = detect_lora_header(hdr, name)
        bad = d["family"] in ("sd15", "flux")
        return {"kind": "lora", "dest": "loras", "family": d["family"], "evidence": d["evidence"], "ok": not bad,
                "reason": d["evidence"] if bad else None}
    d = detect_header(hdr, name)
    if d.get("family"):
        dest = "checkpoints" if d["family"] == "sdxl" else "diffusion_models"
        return {"kind": "model", "dest": dest, "family": d["family"], "variant": d.get("variant"),
                "evidence": d["evidence"], "note": d.get("note"), "ok": True}
    ks = set(keys)
    if any(k.startswith(("decoder.", "first_stage_model.decoder.")) for k in ks) and             any(k.startswith(("encoder.", "first_stage_model.encoder.")) for k in ks) and             not any(k.startswith("model.diffusion_model.") for k in ks):
        return {"kind": "vae", "dest": "vae", "family": None, "evidence": "VAE", "ok": True}
    if any(re.search(r"(^|\.)layers\.0\.(self_attn|linear_attn)\.", k) for k in ks) or any("text_model.encoder" in k for k in ks):
        return {"kind": "te", "dest": "text_encoders", "family": None, "evidence": "テキストエンコーダー", "ok": True}
    return {"kind": None, "dest": None, "family": d.get("family"), "evidence": d.get("evidence"), "ok": False,
            "reason": d.get("evidence") or "このツールで使える種類のファイルではありません"}


def classify_gguf_bytes(data: bytes) -> dict:
    d = detect_gguf_bytes(data)
    if d.get("family") == "qwen_image21":
        return {"kind": "model", "dest": "diffusion_models", "family": d["family"], "variant": d["family"],
                "evidence": d["evidence"], "note": d.get("note"), "ok": True}
    return {"kind": None, "dest": None, "family": d.get("family"), "evidence": d["evidence"], "ok": False,
            "reason": d.get("note") or d["evidence"]}


def detect_gguf_bytes(data: bytes) -> dict:
    class _B(io.BytesIO):
        name = "remote.gguf"
    return detect_gguf(_B(data))


def probe(url: str, name: str, headers: dict | None = None) -> dict:
    """Read only the header of a remote file and classify it."""
    try:
        if name.lower().endswith(".gguf"):
            size = 1 << 20
            while True:
                data = _range(url, 0, size - 1, headers)
                try:
                    return classify_gguf_bytes(data)
                except (struct.error, UnicodeDecodeError, ValueError):
                    if size >= GGUF_PROBE_LIMIT or len(data) < size:
                        raise
                    size *= 4
        head = _range(url, 0, (1 << 20) - 1, headers)
        n = struct.unpack("<Q", head[:8])[0]
        if n > 100_000_000:
            raise ValueError("safetensors のヘッダーが読めません")
        if 8 + n > len(head):
            head += _range(url, len(head), 8 + n - 1, headers)
        hdr = json.loads(head[8:8 + n])
        out = classify_header(hdr, name)
        out["header_sha"] = hashlib.sha1(head[8:8 + n]).hexdigest()   # compared with files already here (_have)
        return out
    except NeedToken:
        raise
    except Exception as e:   # one unreadable file must not hide the others
        return {"kind": None, "dest": None, "ok": False, "evidence": None, "reason": f"中身を確認できませんでした（{e}）"}


# ======================================================================== links
def resolve(link: str, hf_token: str | None = None, civitai_token: str | None = None) -> dict:
    link = link.strip().strip('"')
    u = urllib.parse.urlparse(link if "://" in link else "https://" + link)
    host = u.netloc.lower().removeprefix("www.")
    if host == "huggingface.co":
        return _resolve_hf(u, hf_token)
    if host in ("civitai.com", "civitai.green"):
        return _resolve_civitai(u, civitai_token)
    if "pixai.art" in host:
        raise ValueError("PixAI のリンクからは取得できません（PixAI はファイルの取得方法を公開しておらず、ダウンロードできるかは投稿者の設定次第です）。"
                         "PixAI でダウンロードできたファイルは、下の欄にドロップすると追加できます。")
    raise ValueError("Hugging Face（huggingface.co）か Civitai（civitai.com）のリンクを入れてください。"
                     "ほかのサイトでダウンロードしたファイルは、下の欄にドロップすると追加できます。")


# file names Hugging Face repos often share (diffusers / PEFT defaults): saved under the repo's name instead, so two
# LoRAs do not end up as pytorch_lora_weights.safetensors and pytorch_lora_weights_2.safetensors
GENERIC_NAMES = {"pytorch_lora_weights.safetensors", "adapter_model.safetensors", "lora.safetensors", "model.safetensors",
                 "diffusion_pytorch_model.safetensors", "lora_weights.safetensors", "pytorch_model.safetensors"}


def hf_save_name(repo: str, path: str) -> str:
    """The name a Hugging Face file is saved as: its own, or for a generic one the repo (and folder) name.
    Example: Viggle/Qwen-Image-2.1-viggle-turbo, peft/adapter_model.safetensors -> Qwen-Image-2.1-viggle-turbo_peft.safetensors"""
    name = path.rsplit("/", 1)[-1]
    if name.lower() not in GENERIC_NAMES:
        return name
    folder = path.rsplit("/", 1)[0].replace("/", "_") if "/" in path else ""
    return f"{repo.split('/')[-1]}{'_' + folder if folder else ''}{Path(name).suffix}"


def _hf_headers(token):
    return {"Authorization": f"Bearer {token}"} if token else {}


def _resolve_hf(u, token) -> dict:
    parts = [p for p in u.path.split("/") if p]
    if parts and parts[0] in ("models",):
        parts = parts[1:]
    if len(parts) < 2 or parts[0] in ("datasets", "spaces"):
        raise ValueError("Hugging Face のモデルのリンクを入れてください（例: https://huggingface.co/作者/リポジトリ）")
    repo = "/".join(parts[:2])
    rev, sub, single = "main", "", None
    if len(parts) >= 4 and parts[2] in ("blob", "resolve", "tree"):
        rev = urllib.parse.unquote(parts[3])
        rest = "/".join(parts[4:])
        if parts[2] == "tree":
            sub = rest
        else:
            single = urllib.parse.unquote(rest)
    h = _hf_headers(token)
    try:
        info = _json(f"https://huggingface.co/api/models/{repo}", h)
    except NeedToken:
        raise NeedToken("hf")
    files = []
    if single:
        files.append({"path": single, "size": None})
    else:
        q = urllib.parse.quote(sub)
        try:
            tree = _json(f"https://huggingface.co/api/models/{repo}/tree/{urllib.parse.quote(rev)}/{q}?recursive=1", h)
        except NeedToken:
            raise NeedToken("hf")
        for t in tree:
            if t.get("type") == "file":
                files.append({"path": t["path"], "size": (t.get("lfs") or {}).get("size") or t.get("size")})
    out, skipped = [], []
    for f in files:
        name = f["path"].rsplit("/", 1)[-1]
        if not name.lower().endswith(SAFE_EXT):
            if name.lower().endswith((".ckpt", ".pt", ".pth", ".bin", ".pkl")):
                skipped.append(name)
            continue
        url = f"https://huggingface.co/{repo}/resolve/{urllib.parse.quote(rev)}/{urllib.parse.quote(f['path'])}"
        out.append({"name": hf_save_name(repo, f["path"]), "path": f["path"], "size": f["size"], "url": url})
    lic = (info.get("cardData") or {}).get("license") or next((t[8:] for t in info.get("tags", []) if t.startswith("license:")), None)
    return _finish({"source": "huggingface", "title": repo, "page": f"https://huggingface.co/{repo}", "license": lic,
                    "gated": bool(info.get("gated")), "files": out, "skipped": skipped}, h, "hf")


_trained_words: dict[str, list[str]] = {}   # Civitai download URL -> the trigger words the site lists


def _resolve_civitai(u, token) -> dict:
    qs = urllib.parse.parse_qs(u.query)
    parts = [p for p in u.path.split("/") if p]
    vid = (qs.get("modelVersionId") or [None])[0]
    mid = None
    if len(parts) >= 2 and parts[0] == "models" and parts[1].isdigit():
        mid = parts[1]
    if parts[:3] == ["api", "download", "models"] and len(parts) > 3:
        vid = parts[3]
    if not (vid or mid):
        raise ValueError("Civitai のモデルページのリンクを入れてください（例: https://civitai.com/models/12345）")
    h = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        if vid:
            v = _json(f"https://civitai.com/api/v1/model-versions/{vid}", h)
            m = v.get("model") or {}
            title, mtype = m.get("name"), m.get("type")
            mid = v.get("modelId")
        else:
            mm = _json(f"https://civitai.com/api/v1/models/{mid}", h)
            v = (mm.get("modelVersions") or [None])[0]
            if not v:
                raise ValueError("このモデルにはダウンロードできる版がありません")
            title, mtype = mm.get("name"), mm.get("type")
    except NeedToken:
        raise NeedToken("civitai")
    base = v.get("baseModel") or ""
    hint = next((var for k, var in CIVITAI_BASE.items() if base.lower().startswith(k)), None)
    out, skipped = [], []
    for f in v.get("files") or []:
        fmt = (f.get("metadata") or {}).get("format") or ""
        name = f.get("name") or ""
        if f.get("type") in ("Training Data", "Archive"):
            continue
        if fmt not in ("SafeTensor", "GGUF") and not name.lower().endswith(SAFE_EXT):
            skipped.append(name)
            continue
        if v.get("trainedWords"):
            _trained_words[f["downloadUrl"]] = list(v["trainedWords"])   # recorded for the LoRA when it is fetched
        out.append({"name": name, "size": int((f.get("sizeKB") or 0) * 1024) or None, "url": f["downloadUrl"],
                    "site_type": f.get("type"), "primary": bool(f.get("primary"))})
    out.sort(key=lambda x: not x["primary"])
    return _finish({"source": "civitai", "title": f"{title}（{v.get('name')}）", "page": f"https://civitai.com/models/{mid}?modelVersionId={v['id']}",
                    "site_info": f"Civitai の表記: {mtype or '種類不明'}・ベースモデル {base or '不明'}", "variant_hint": hint,
                    "license": None, "gated": False, "files": out, "skipped": skipped}, None, "civitai", token)


def _civitai_url(url: str, token: str | None) -> str:
    # the token goes in the query: the download redirects to a signed storage URL that rejects a second
    # Authorization header
    if not token:
        return url
    return url + ("&" if "?" in url else "?") + "token=" + urllib.parse.quote(token)


def _finish(res: dict, headers, site, civitai_token=None) -> dict:
    local: dict = {}
    for i, f in enumerate(res["files"]):
        if i >= MAX_PROBE:
            f["probe"] = {"ok": None, "reason": "追加するときに中身を確認します"}
            continue
        url = _civitai_url(f["url"], civitai_token) if site == "civitai" else f["url"]
        try:
            f["probe"] = probe(url, f["name"], headers)
        except NeedToken:
            # the page is public but the file needs a login (Civitai: many files; HF: gated repos)
            res["need_token"] = site
            f["probe"] = {"ok": None, "need_token": True,
                          "reason": "中身の確認と取得にはログインが必要です（下の欄にキーを入れて、もう一度「調べる」を押してください）"}
        p = f["probe"]
        p["dest_label"] = DEST_LABEL.get(p.get("dest") or "", "")
        p["family_label"] = FAMILY_LABEL.get(p.get("family") or "", "")
        f["have"] = _have(f["name"], f.get("size"), p.get("header_sha"), local)
    return res


def _local_files(ext: str, local: dict) -> list[tuple[Path, int]]:
    """(path, size) of the model files under models/ and the added folders, listed once per lookup."""
    if ext not in local:
        out = []
        for root in [config.MODELS_DIR] + [Path(r) for r in library.roots()]:
            if not root.is_dir():
                continue
            for f in root.rglob("*" + ext):
                try:
                    if "_incoming" not in f.parts and f.is_file():
                        out.append((f, f.stat().st_size))
                except OSError:
                    pass
        local[ext] = out
    return local[ext]


def _header_sha(path: Path) -> str | None:
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            return hashlib.sha1(f.read(n)).hexdigest() if n <= 100_000_000 else None
    except (OSError, struct.error):
        return None


def _have(name: str, size: int | None, header_sha: str | None = None, local: dict | None = None) -> str | None:
    """Path of the same file already under models/ or an added folder: the same name (and the same byte size
    when it is known), or - sites rename files (Civitai's Pony is ponyDiffusionV6XL_v6StartWithThisOne.safetensors,
    the same bytes as ponyDiffusionV6XL.safetensors) - the same byte size and the same safetensors header.
    Size alone is not enough: SDXL checkpoints with the same layout often have exactly the same size."""
    for f, n in _local_files(Path(name).suffix.lower(), {} if local is None else local):
        if f.name == name and (size is None or n == size):
            return str(f)
        if size is not None and n == size and header_sha and _header_sha(f) == header_sha:
            return str(f)
    return None


# ======================================================================== download / upload
_BAD_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_name(name: str) -> str:
    """A plain file name (no folders, no characters Windows does not allow) for a name a site or the
    browser gave."""
    name = _BAD_NAME_CHARS.sub("_", Path(str(name or "").replace("\\", "/")).name).strip().rstrip(".")
    if not name or name.startswith("."):
        raise ValueError("ファイル名が正しくありません")
    return name


_place_lock = threading.Lock()   # choosing a free final name and moving the file there is one step


def _final_path(dest: str, name: str) -> Path:
    d = config.MODELS_DIR / dest
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    i = 2
    while p.exists():
        p = d / f"{Path(name).stem}_{i}{Path(name).suffix}"
        i += 1
    return p


def start_download(url: str, name: str, site: str, token: str | None = None, variant_hint: str | None = None,
                   size: int | None = None) -> dict:
    name = safe_name(name)
    if not name.lower().endswith(SAFE_EXT):
        raise ValueError("safetensors か GGUF のファイルだけを取得します")
    host = urllib.parse.urlparse(url).netloc.lower()
    if not (host.endswith("huggingface.co") or host.endswith("civitai.com")):
        raise ValueError("Hugging Face か Civitai のファイルだけを取得できます")
    key = hashlib.sha1(url.encode()).hexdigest()[:12]
    with _lock:
        if _state.get(key, {}).get("state") == "running":
            return _state[key]
        st = {"id": key, "name": name, "state": "running", "done_bytes": 0, "total_bytes": size or 0, "error": None,
              "message": "準備中", "started": time.time(), "trained_words": _trained_words.get(url)}
        _state[key] = st
    real = _civitai_url(url, token) if site == "civitai" else url
    headers = _hf_headers(token) if site == "huggingface" else {}
    threading.Thread(target=_download, args=(st, real, name, headers, variant_hint), daemon=True).start()
    bus.publish("download", id=key, state="running")   # the job page shows it at once
    return st


def _download(st: dict, url: str, name: str, headers: dict, hint: str | None):
    # named by the download id (from the URL): a resumed download continues its own file, never another
    # file of the same name (another version, or a file two catalog entries share)
    tmp = config.MODELS_DIR / "_incoming" / f"{st['id']}_{name}.part"
    try:
        tmp.parent.mkdir(parents=True, exist_ok=True)
        st["message"] = "中身を確認中"
        cls = probe(url, name, headers)
        if not cls.get("ok"):
            raise ValueError(cls.get("reason") or "このツールで使える種類のファイルではありません")
        free = shutil.disk_usage(tmp.parent).free
        if st["total_bytes"] and free < st["total_bytes"] + (2 << 30):
            raise RuntimeError(f"空き容量が不足しています（必要 約{st['total_bytes'] / 1e9:.1f}GB / 空き {free / 1e9:.1f}GB）")
        have = tmp.stat().st_size if tmp.exists() else 0
        h = dict(headers)
        if have:
            h["Range"] = f"bytes={have}-"
        st["message"] = "ダウンロード中"
        expect, listed = None, st["total_bytes"] or None   # listed: the size the site gave (Civitai's is rounded to KB)
        try:
            with _open(url, h, timeout=60) as r:
                if have and r.status != 206:
                    have = 0
                total = r.headers.get("Content-Range", "").rsplit("/", 1)[-1] if have else r.headers.get("Content-Length")
                if total and total.isdigit():
                    st["total_bytes"] = expect = int(total) if have else have + int(total)
                st["done_bytes"] = have
                with open(tmp, "ab" if have else "wb") as f:
                    while chunk := r.read(8 << 20):
                        if st.get("cancel"):
                            raise Cancelled
                        f.write(chunk)
                        st["done_bytes"] += len(chunk)
        except urllib.error.HTTPError as e:
            if not (e.code == 416 and have):   # 416: the .part is already whole (the last try ended after the last byte)
                raise
        except OSError as e:   # timeout, reset: the .part is kept, and pressing again continues it
            raise RuntimeError("接続が途中で切れました。もう一度「追加する」を押すと続きから取得します"
                               f"（{tmp.stat().st_size / 1e9 if tmp.exists() else 0:.2f}GB まで取得済み）。") from e
        # read() also ends quietly when the connection drops: the .part is kept, and pressing again continues it
        short = _incomplete(tmp, name, expect, listed)
        if short:
            raise RuntimeError(f"ダウンロードが途中で切れました（{short}）。もう一度「追加する」を押すと続きから取得します")
        if st.get("cancel"):
            raise Cancelled
        _place(st, tmp, name, hint)
    except Cancelled:
        tmp.unlink(missing_ok=True)
        st.update(state="cancelled", message="キャンセルしました")
    except NeedToken:
        st.update(state="failed", message="失敗", error="このファイルの取得にはログイン（アクセストークン / API キー）が必要です。")
    except Exception as e:
        st.update(state="failed", message="失敗", error=str(e))
    bus.publish("download", id=st["id"], state=st["state"])


def _incomplete(tmp: Path, name: str, expect: int | None, listed: int | None = None) -> str | None:
    """Why the downloaded file is not whole, or None.  Without a length from the server, the size the site listed is
    a near check, and a safetensors file is checked against its own header: the tensor data must reach the end the
    header names."""
    size = tmp.stat().st_size
    if expect is not None and size != expect:
        return f"{size / 1e9:.2f} / {expect / 1e9:.2f}GB"
    if expect is None and listed and size < listed - 2048:
        return f"{size / 1e9:.2f} / {listed / 1e9:.2f}GB"
    if name.lower().endswith(".safetensors"):
        try:
            with open(tmp, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            end = 8 + n + max((v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__"), default=0)
        except (OSError, ValueError, KeyError, TypeError, struct.error):
            return "ファイルの先頭が読めません"
        if size < end:
            return f"{size / 1e9:.2f} / {end / 1e9:.2f}GB"
    return None


def _place(st: dict, tmp: Path, name: str, hint: str | None):
    """Check the finished file's own header, then move it to the folder its kind belongs to."""
    cls = classify_local(tmp, name)
    if not cls.get("ok"):
        tmp.unlink(missing_ok=True)
        raise ValueError(cls.get("reason") or "このツールで使える種類のファイルではありません")
    with _place_lock:
        final = _final_path(cls["dest"], name)
        tmp.replace(final)
    if cls["kind"] == "lora" and st.get("trained_words"):
        from . import loraconf
        loraconf.record_found(final, st["trained_words"], "Civitai に書かれた呼び出し語")
    library.scan()
    if hint and cls["kind"] == "model":
        for m in registry.list():
            if any(Path(p) == final for p in m["files"].values()) and m.get("auto") and not m.get("variant_confirmed"):
                if not (hint == "noobai_eps" and m.get("variant") == "noobai_vpred") and                         not (hint == "anima" and m.get("variant") == "anima29"):   # 2.9B is told by its block count
                    registry.update(m["id"], variant=hint, variant_confirmed=True)
    st.update(state="done", message="完了", path=str(final), kind=cls["kind"], dest=cls["dest"],
              dest_label=DEST_LABEL.get(cls["dest"], ""))


def classify_local(path: Path, name: str) -> dict:
    from .models import read_safetensors_header
    try:
        if name.lower().endswith(".gguf"):
            with open(path, "rb") as f:
                return classify_gguf_bytes(f.read(GGUF_PROBE_LIMIT))
        return classify_header(read_safetensors_header(path), name)
    except Exception as e:
        return {"ok": False, "reason": f"ファイルを読めません（{e}）"}


def begin_upload(name: str) -> tuple[dict, Path]:
    """A file dropped in the browser (e.g. downloaded from PixAI).  The server streams the request body
    into the returned path, then calls finish_upload()."""
    name = Path(name).name
    if not name.lower().endswith(SAFE_EXT):
        raise ValueError("safetensors か GGUF のファイルだけを追加できます（.ckpt / .pt などは、読み込むとプログラムが動く形式のため扱いません）")
    key = hashlib.sha1(f"upload:{name}:{time.time()}".encode()).hexdigest()[:12]
    st = {"id": key, "name": name, "state": "running", "done_bytes": 0, "total_bytes": 0, "error": None,
          "message": "受け取り中", "started": time.time(), "src": "upload"}
    _state[key] = st
    tmp = config.MODELS_DIR / "_incoming" / f"{key}_{name}.part"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    return st, tmp


def finish_upload(st: dict, tmp: Path, ok: bool, error: str | None = None) -> dict:
    try:
        if not ok:
            raise ValueError(error or "受け取りが途中で止まりました")
        _place(st, tmp, st["name"], None)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        st.update(state="failed", message="失敗", error=str(e))
    bus.publish("download", id=st["id"], state=st["state"])
    return st
