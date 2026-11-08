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
    of the instruction tasks (readings) as Index-Translate is."""
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
    assert "<|plamo:tag|>context<|plamo:msg|>" + i2t.PLAMO_CONTEXT[("ja", "tags")] + "<|plamo:tag|>input" in payload["prompt"]


def test_plamo_gets_the_situation_and_a_second_try():
    """PLaMo is told what it translates (tags, or the sentences of Krea 2 / MOSS-Image prompts); a sentence goes alone,
    and a tag it only copied in the list is sent once more by itself."""
    sys.path.insert(0, str(ROOT.parent / "engines" / "i2t"))
    import i2t
    sent = []
    sentence = "A woman in a red coat stands on a rainy street at night."

    class Fake:
        model = "hf.co/pfnet/plamo-3-translate-8b-GGUF:Q4_K_M"

        def _http(self, path, payload, abortable=False):
            p = payload["prompt"]
            text = p.split("input<|plamo:msg|>")[1].split("<|plamo:tag|>")[0]
            ctx = p.split("context<|plamo:msg|>")[1].split("<|plamo:tag|>")[0]
            sent.append((text, ctx))
            if text == sentence:
                return {"response": "夜の雨の通りに、\n赤いコートの女性が立っている。"}
            if text == "thighhighs":
                return {"response": "ニーハイソックス"}
            return {"response": "1girl\nthighhighs\n"}   # the list: copied
    out = i2t.translate_chips_ja(Fake(), ["1girl", "thighhighs", sentence])
    assert out == {"thighhighs": "ニーハイソックス", sentence.lower(): "夜の雨の通りに、赤いコートの女性が立っている。"}
    tags, prose = i2t.PLAMO_CONTEXT[("ja", "tags")], i2t.PLAMO_CONTEXT[("ja", "prose")]
    assert sent == [("1girl\nthighhighs", tags), ("1girl", tags), ("thighhighs", tags), (sentence, prose)]
    assert not i2t.looks_like_sentence("long hair") and i2t.looks_like_sentence("夜の街を歩く赤いコートの女性。")


def test_plamo_translates_the_image_page_japanese_into_english(tmp_path, monkeypatch):
    """「Ollama で英語に訳す」 on the 画像生成 page with PLaMo 3 Translate as 訳のモデル: PLaMo translates (lang en);
    with another model the instruction model does, as before."""
    import importlib.util
    from types import SimpleNamespace
    monkeypatch.setattr(sys, "argv", ["w", str(ROOT.parent / "engines" / "i2t"), str(tmp_path / "d"), str(tmp_path / "m")])
    spec = importlib.util.spec_from_file_location("i2t_worker_plamo", ROOT / "worker" / "i2t_worker.py")
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    w.i2t = w.load_i2t()
    plamo = "hf.co/pfnet/plamo-3-translate-2b-GGUF:Q4_K_M"
    sent, asked = [], []

    class Fake:
        def __init__(self, host, model, timeout=0):
            self.model = model

        def _http(self, path, payload, abortable=False):
            sent.append(payload)
            return {"response": {"猫耳": "Cat ears", "笑顔": "smile."}[payload["prompt"].split("input<|plamo:msg|>")[1].split("<|plamo:tag|>")[0]]}
    cfg = {"translate_model": plamo, "ollama_host": "http://127.0.0.1:11434", "ollama_model": "qwen3-vl:8b", "low_vram": True}
    monkeypatch.setattr(w.i2t, "OllamaVLM", Fake)
    monkeypatch.setattr(w.i2t, "load_config", lambda: cfg)
    monkeypatch.setattr(w.i2t, "load_translations", lambda: {})
    monkeypatch.setattr(w.i2t, "save_translations", lambda t: True)
    monkeypatch.setattr(w, "chip_table", lambda: {})
    monkeypatch.setattr(w, "_text_llm", lambda c, cancel: SimpleNamespace(model="qwen3-vl:8b"))
    monkeypatch.setattr(w, "_ask_list", lambda vlm, system, items, keep, check, *a: asked.append(items) or ["cat ears"] * len(items))
    monkeypatch.setattr(w, "_store_readings", lambda *a: {})
    monkeypatch.setattr(w, "_progress", lambda rid: lambda m: None)
    monkeypatch.setattr(w, "_checkpoint", lambda rid, c: lambda: None)
    out = w._translate_en(1, {"words": ["猫耳", "笑顔"]}, None)
    assert out["translated"] == {"猫耳": "cat ears", "笑顔": "smile"} and out["model"] == plamo
    assert all(p["raw"] and p["prompt"].endswith("<|plamo:val|>en<|plamo:msg|>") for p in sent) and not asked
    assert all("context<|plamo:msg|>" + w.i2t.PLAMO_CONTEXT[("en", "tags")] + "<|plamo:tag|>" in p["prompt"] for p in sent)
    cfg["translate_model"] = "hf.co/IndexTeam/Index-Translate-2B-GGUF:Q4_K_M"   # translation into Japanese only
    sent.clear()
    out = w._translate_en(1, {"words": ["猫耳"]}, None)
    assert out["model"] == "qwen3-vl:8b" and asked == [["猫耳"]] and not sent
