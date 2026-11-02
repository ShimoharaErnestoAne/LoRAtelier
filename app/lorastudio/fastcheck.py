"""Stage-1 mechanical checks: things that can be determined as facts without AI.

Runs in the UI process on a small thread pool (Pillow releases the GIL while decoding).
Results are facts (sizes, hashes, JPEG tables) plus simple signal measures (sharpness,
exposure) that are only *candidates* for problems; thresholds live in judge.py.
"""
from __future__ import annotations

import hashlib
import uuid
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = 200_000_000
SUPPORTED = {"JPEG", "PNG", "WEBP", "BMP", "TIFF", "GIF"}
FASTCHECK_VERSION = 2
THUMB_SIZE = 320


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dhash(gray: Image.Image, size: int = 8) -> int:
    small = np.asarray(gray.resize((size + 1, size), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _phash(gray: Image.Image) -> int:
    a = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    n = 32
    k = np.arange(n)
    dct = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))  # DCT-II basis
    d = dct @ a @ dct.T
    low = d[:8, :8].flatten()
    med = np.median(low[1:])
    return int("".join("1" if v > med else "0" for v in low), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _laplacian_var(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def _jpeg_quality(img: Image.Image) -> int | None:
    """Estimate IJG quality from the luminance quantization table (fact-based estimate)."""
    q = getattr(img, "quantization", None)
    if not q or 0 not in q:
        return None
    std = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=np.float64)
    table = np.array(q[0][:64], dtype=np.float64)
    # IJG: table = floor((std * scale + 50) / 100); invert per-coefficient and take the median
    scale = np.median((table * 100 - 50) / std)
    if scale <= 0:
        return 100
    quality = (200 - scale) / 2 if scale <= 100 else 5000 / scale
    return int(max(1, min(100, round(quality))))


def analyze(path: Path, thumb_path: Path) -> dict:
    """Return a dict of facts.  Never raises for bad images: returns {'ok': False, ...}."""
    out: dict = {"version": FASTCHECK_VERSION, "ok": False}
    try:
        out["file_size"] = path.stat().st_size
    except OSError as e:
        out["error"] = f"ファイルを読めません: {e}"
        out["missing"] = True
        return out
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as im:
                fmt = im.format
                out["format"] = fmt
                out["animated"] = bool(getattr(im, "is_animated", False))
                out["jpeg_quality"] = _jpeg_quality(im) if fmt == "JPEG" else None
                out["mode"] = im.mode
                im.load()  # full decode: detects truncated / corrupt data
                im = ImageOps.exif_transpose(im)
                has_alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
                # full-size copies of a large image are costly: RGBA only for images with transparency
                rgba = im.convert("RGBA") if has_alpha else im.convert("RGB")
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
        out["error"] = f"画像として読み込めません（破損または非対応形式）: {type(e).__name__}"
        return out
    if fmt not in SUPPORTED:
        out["error"] = f"非対応の形式です: {fmt}"
        return out

    w, h = rgba.size
    out.update(width=w, height=h, megapixels=round(w * h / 1e6, 2), aspect=round(w / h, 3))
    if has_alpha:
        out["transparent_ratio"] = float((np.asarray(rgba.getchannel("A")) < 250).mean())
        rgb = Image.new("RGB", rgba.size, (255, 255, 255))
        rgb.paste(rgba, mask=rgba.getchannel("A"))
    else:
        out["transparent_ratio"] = 0.0
        rgb = rgba
    del rgba

    # analysis on a bounded working copy (in place: the full-size image is not needed after this)
    work = rgb
    work.thumbnail((1024, 1024), Image.Resampling.BILINEAR)
    # thumbnail first so the grid can show it as early as possible
    thumb = work.copy()
    thumb.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.Resampling.LANCZOS)
    # a unique temporary name: two checks of the same image (start-up resume and 再解析) may overlap.
    # A thumbnail that cannot be written says nothing about the image, so it does not change the verdict.
    tmp = thumb_path.with_name(f"{thumb_path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        thumb_path.parent.mkdir(parents=True, exist_ok=True)
        thumb.save(tmp, "WEBP", quality=82)
        tmp.replace(thumb_path)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        out["thumb_error"] = str(e)

    gray = work.convert("L")
    g = np.asarray(gray)
    out["dhash"] = format(_dhash(gray), "016x")
    out["phash"] = format(_phash(gray), "016x")
    # sharpness: Laplacian variance at a fixed working scale; also the 90th percentile of local
    # variance so that images with a sharp subject and a deliberately blurred background are not penalised
    out["sharpness"] = round(_laplacian_var(g), 1)
    tiles = []
    th, tw = max(1, g.shape[0] // 4), max(1, g.shape[1] // 4)
    for y in range(4):
        for x in range(4):
            t = g[y * th:(y + 1) * th, x * tw:(x + 1) * tw]
            if t.shape[0] > 2 and t.shape[1] > 2:
                tiles.append(_laplacian_var(t))
    out["sharpness_p90"] = round(float(np.percentile(tiles, 90)), 1) if tiles else out["sharpness"]
    out["brightness"] = round(float(g.mean()) / 255, 3)
    out["clip_dark"] = round(float((g < 8).mean()), 3)
    out["clip_bright"] = round(float((g > 247).mean()), 3)
    a = np.asarray(work, dtype=np.int16)
    out["colorfulness"] = round(float(np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean()) / 255, 3)
    out["ok"] = True
    return out
