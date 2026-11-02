"""Make app/lorastudio/static/tag_readings.json: the hiragana reading of each Japanese translation in i2t's tag
dictionary (engines/i2t/assets/tags_ja.json), so the tag completion finds 笑顔 from えがお typed before the IME
conversion.  Run again when the dictionary changes.  Only for making the file: the app does not use Janome.

    python -m venv <scratch>  &&  <scratch>/Scripts/pip install janome==0.5.0
    <scratch>/Scripts/python app/scripts/make_tag_readings.py

Janome 0.5.0 (Apache-2.0) with its bundled IPADIC 2.7.0 (see LICENSES.md).  Readings are the dictionary's; words
it does not know keep their written form (as they are, or in hiragana for katakana).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from janome.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "engines" / "i2t" / "assets" / "tags_ja.json"
OUT = ROOT / "app" / "lorastudio" / "static" / "tag_readings.json"
KANJI = re.compile(r"[㐀-鿿々]")


def hira(t: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in t)


def main():
    table = json.loads(SRC.read_text("utf-8"))
    tok = Tokenizer()
    out = {}
    for en, ja in table.items():
        if not isinstance(ja, str) or not KANJI.search(ja):
            continue   # kana only: the page compares it in hiragana already
        r = "".join(t.reading if t.reading not in ("*", "") else t.surface for t in tok.tokenize(ja))
        out[en] = hira(r)
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":"), sort_keys=True), "utf-8")
    print(f"{len(out)} readings -> {OUT}")


if __name__ == "__main__":
    main()
