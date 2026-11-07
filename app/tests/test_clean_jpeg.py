"""「JPG で保存（情報なし）」: the JPEG keeps the picture but none of the prompt / ComfyUI workflow the PNG carries."""
from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_jpeg_has_the_pixels_and_no_generation_info(tmp_path):
    from PIL import Image, PngImagePlugin
    from lorastudio.server import _clean_jpeg
    meta = PngImagePlugin.PngInfo()
    meta.add_text("parameters", "secret prompt, sty_gqrx style")
    meta.add_text("prompt", '{"3": {"class_type": "KSampler"}}')
    meta.add_text("workflow", '{"nodes": [{"type": "KSampler"}]}')
    src = tmp_path / "a.png"
    im = Image.new("RGBA", (64, 48), (200, 30, 30, 255))
    im.putpixel((0, 0), (0, 0, 0, 0))   # a transparent pixel: white in the JPEG
    im.save(src, pnginfo=meta, exif=Image.Exif())
    data = _clean_jpeg(src)
    assert data[:2] == b"\xff\xd8"
    for word in (b"secret", b"sty_gqrx", b"KSampler", b"workflow", b"parameters", b"Exif"):
        assert word not in data
    out = Image.open(io.BytesIO(data))
    assert out.format == "JPEG" and out.size == (64, 48) and out.mode == "RGB"
    assert not {k for k in out.info if k not in ("jfif", "jfif_version", "jfif_unit", "jfif_density", "dpi", "progressive", "progression")}
    r, g, b = out.getpixel((32, 24))
    assert r > 180 and g < 60 and b < 60
    assert min(out.getpixel((0, 0))) > 150
