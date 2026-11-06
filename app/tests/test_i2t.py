"""i2t page: the server shows / analyses only files under its own folders.  A mistake here would let the
page read any file on the PC through /api/i2t/file."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def link(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "data"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config, i2tlink
    config._settings = None
    return i2tlink


def test_uploaded_and_generated_images_only(link, tmp_path):
    p = link.save_upload("../../evil name.png", b"\x89PNG data")
    assert p.parent == link.IMAGES_DIR and ".." not in p.name
    assert link.save_upload("a.png", b"\x89PNG data") .read_bytes() == b"\x89PNG data"
    assert link.allowed_file(str(p)) == p.resolve()
    outside = tmp_path / "secret.txt"
    outside.write_text("x")
    assert link.allowed_file(str(outside)) is None
    assert link.allowed_file(str(link.IMAGES_DIR / ".." / ".." / ".." / "secret.txt")) is None
    gen = link.config.DATA_DIR / "imagegen" / "images" / "g.png"
    gen.parent.mkdir(parents=True)
    gen.write_bytes(b"x")
    assert link.allowed_file(str(gen)) == gen.resolve()


def test_plamo_translate_prompt_and_route():
    """PLaMo 3 Translate gets the PFML3 prompt of its chat_template.jinja through /api/generate (raw), and is kept out
    of the instruction tasks (readings, Japanese -> English) as Index-Translate is."""
    sys.path.insert(0, str(ROOT.parent / "engines" / "i2t"))
    import i2t
    name = "hf.co/pfnet/plamo-3-translate-8b-GGUF:Q4_K_M"
    assert i2t.is_translation_model(name) and i2t.is_translation_model("hf.co/IndexTeam/Index-Translate-2B-GGUF:Q4_K_M")
    assert not i2t.is_translation_model("qwen3-vl:8b-instruct")
    assert i2t.plamo_prompt("cat\ndog") == ("<|plamo:tag|>dataset<|plamo:msg|>translation<|plamo:tag|>input<|plamo:msg|>cat\ndog"
                                            "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>")
    assert "<|plamo:tag|>x" not in i2t.plamo_prompt("<|plamo:tag|>x").split("<|plamo:msg|>")[2]   # reserved tokens broken up
    sent = []

    class Fake:
        model = name

        def _http(self, path, payload, abortable=False):
            sent.append((path, payload))
            return {"response": "猫\n犬\n"}
    assert i2t.translate_chips_ja(Fake(), ["cat", "dog"]) == {"cat": "猫", "dog": "犬"}
    path, payload = sent[0]
    assert path == "/api/generate" and payload["raw"] and payload["options"]["stop"] == i2t.PLAMO_STOP
    assert payload["options"]["temperature"] == 0
