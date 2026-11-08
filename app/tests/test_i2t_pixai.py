"""PixAI Tagger v0.9 in i2t: its input is what the distributor's preprocess.json says (RGB, NCHW, -1..1), only the
"prediction" output is read, and the rating it does not give comes from anime_rating (safe / r15 / r18 → WD names)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def test_pixai_tagger_input_outputs_and_rating(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT.parent / "engines" / "i2t"))
    import i2t
    monkeypatch.setattr(i2t, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(i2t, "RATING_MODEL_PATH", tmp_path / "rating" / "r.onnx")
    name = "pixai-tagger-v0.9"
    assert name in i2t.TAGGER_MODELS and i2t.tagger_is_pixai(name) and not i2t.tagger_is_pixai("wd-eva02-large-tagger-v3")
    model, csv = i2t.tagger_files(name)
    model.parent.mkdir(parents=True)
    model.write_bytes(b"x" * 1_000_001)
    csv.write_text("id,tag_id,name,category,count,ips\n0,1,1girl,0,9,[]\n1,2,solo,0,9,[]\n2,3,hatsune_miku,4,9,[]\n", "utf-8")
    i2t.RATING_MODEL_PATH.parent.mkdir(parents=True)
    i2t.RATING_MODEL_PATH.write_bytes(b"x" * 1_000_001)
    seen = {}

    class Tagger:
        def get_providers(self):
            return ["CPUExecutionProvider"]

        def get_inputs(self):
            return [SimpleNamespace(name="input", shape=[1, 3, 448, 448])]

        def get_outputs(self):
            return [SimpleNamespace(name=n) for n in ("embedding", "prediction", "logits")]

        def run(self, outputs, feed):
            seen["tagger"] = (outputs, feed["input"])
            return [np.array([[0.9, 0.2, 0.95]], dtype=np.float32)]

    class Rating:
        def get_inputs(self):
            return [SimpleNamespace(name="input", shape=[1, 3, 384, 384])]

        def run(self, outputs, feed):
            seen["rating"] = feed["input"]
            return [np.array([[0.1, 0.2, 0.7]], dtype=np.float32)]
    monkeypatch.setattr(i2t, "make_onnx_session", lambda path, cpu_only=False: Rating() if path == i2t.RATING_MODEL_PATH else Tagger())
    t = i2t.WD14Tagger(name)
    img = Image.new("RGBA", (300, 200), (255, 0, 0, 0))   # transparent: on white
    img.putpixel((0, 0), (0, 0, 255, 255))
    r = t.predict(img, 0.35, 0.85)
    outputs, x = seen["tagger"]
    assert outputs == ["prediction"] and x.shape == (1, 3, 448, 448) and x.dtype == np.float32
    assert np.isclose(x[0, :, 200, 200], 1.0).all()            # white → 1 in every channel
    assert x[0, 2, 0, 0] > x[0, 0, 0, 0]                        # RGB order: the blue pixel is in channel 2
    assert seen["rating"].shape == (1, 3, 384, 384)
    assert [n for n, _ in r.general] == ["1girl"] and [n for n, _ in r.characters] == ["hatsune_miku"]
    assert r.rating == "explicit" and abs(r.rating_conf - 0.7) < 1e-6


def test_camie_tagger_vocabulary_input_and_logits(tmp_path, monkeypatch):
    """Camie Tagger v2: the distributor's metadata.json becomes a WD-style selected_tags.csv (ratings renamed), the
    image keeps its aspect ratio on a 512 square padded with (124, 116, 104), ImageNet-normalized, and the
    refined_predictions logits go through a sigmoid."""
    import json
    sys.path.insert(0, str(ROOT.parent / "engines" / "i2t"))
    import i2t
    monkeypatch.setattr(i2t, "MODELS_DIR", tmp_path)
    name = "camie-tagger-v2"
    assert name in i2t.TAGGER_MODELS and i2t.tagger_is_camie(name) and not i2t.tagger_is_camie("pixai-tagger-v0.9")
    model, csv = i2t.tagger_files(name)
    model.parent.mkdir(parents=True)
    tags = ["year_2020", "1girl", "rating_explicit", "rating_general", "hatsune_miku", "vocaloid", "solo"]
    cats = {"year_2020": "year", "1girl": "general", "rating_explicit": "rating", "rating_general": "rating",
            "hatsune_miku": "character", "vocaloid": "copyright", "solo": "general"}
    meta = model.with_name("metadata.json")
    meta.write_text(json.dumps({"dataset_info": {"tag_mapping": {"idx_to_tag": {str(i): t for i, t in enumerate(tags)},
                                                                 "tag_to_category": cats}}}), "utf-8")
    meta.write_bytes(meta.read_bytes() + b" " * 1_000_001)   # model_file_ok: a finished download
    got = []
    monkeypatch.setattr(i2t, "download_file", lambda url, dest, *a, **k: (got.append(url), dest.write_bytes(b"x" * 1_000_001)))
    i2t.download_tagger(name)
    assert [u.rsplit("/", 1)[1] for u in got] == ["camie-tagger-v2.onnx"]   # metadata was there already
    assert i2t.tagger_ready(name)
    rows = csv.read_text("utf-8").splitlines()
    assert rows[0] == "tag_id,name,category,count" and rows[3] == "2,explicit,9,0" and rows[5] == "4,hatsune_miku,4,0"
    seen = {}

    class Tagger:
        def get_providers(self):
            return ["CPUExecutionProvider"]

        def get_inputs(self):
            return [SimpleNamespace(name="images", shape=["batch_size", 3, 512, 512])]

        def get_outputs(self):
            return [SimpleNamespace(name=n) for n in ("initial_predictions", "refined_predictions", "selected_candidates")]

        def run(self, outputs, feed):
            seen["run"] = (outputs, feed["images"])
            return [np.array([[0.0, 3.0, 2.0, -2.0, 4.0, 5.0, -3.0]], dtype=np.float32)]   # logits
    monkeypatch.setattr(i2t, "make_onnx_session", lambda path, cpu_only=False: Tagger())
    t = i2t.WD14Tagger(name)
    r = t.predict(Image.new("RGB", (400, 200), (255, 255, 255)), 0.35, 0.85)
    outputs, x = seen["run"]
    assert outputs == ["refined_predictions"] and x.shape == (1, 3, 512, 512) and x.dtype == np.float32
    white = (1.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    assert np.allclose(x[0, :, 256, 256], white, atol=1e-4)                  # the image (400x200 → 512x256, centred)
    pad = (np.array([124, 116, 104]) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    assert np.allclose(x[0, :, 10, 256], pad, atol=1e-4)                     # the padding above it
    assert [n for n, _ in r.general] == ["1girl"] and abs(r.general[0][1] - 1 / (1 + np.exp(-3))) < 1e-6
    assert [n for n, _ in r.characters] == ["hatsune_miku"]
    assert r.rating == "explicit" and "year_2020" not in r.general_probs and "vocaloid" not in r.general_probs


def test_joycaption_writes_first_and_the_vlm_gets_it_as_reference(monkeypatch):
    """先に説明を書かせる (describer_model): JoyCaption is asked for a description with its own prompt (no JSON
    format), the usual VLM gets that text in its guidance, and the text is kept in the analysis.  A failure of the
    describer only adds a warning; the usual VLM still runs."""
    sys.path.insert(0, str(ROOT.parent / "engines" / "i2t"))
    import i2t
    sent, guidance = [], {}
    monkeypatch.setattr(i2t.OllamaVLM, "capabilities", lambda self: {"vision", "completion"})

    def chat(self, payload):
        sent.append((self.model, payload))
        if fail:
            raise i2t.OllamaError("not pulled")
        return {"message": {"content": "A girl in a red coat with brass buttons stands by a lamp post."}}
    monkeypatch.setattr(i2t.OllamaVLM, "_chat", chat)

    def analyze(self, image, tags, medium, negative, g="", refs=None, separate=()):
        guidance["text"] = g
        return {"analysis": "main analysis"}
    monkeypatch.setattr(i2t.OllamaVLM, "analyze", analyze)
    tags = i2t.TagResult("general", 0.9, [("1girl", 0.9)], [], {"1girl": 0.9})
    img = Image.new("RGB", (64, 64), (255, 255, 255))
    for fail in (False, True):
        sent.clear()
        eng = i2t.Engine({**i2t.DEFAULT_CONFIG, "describer_model": i2t.JOYCAPTION_MODEL, "low_vram": True})
        r = eng._run_vlm(img, tags, "anime", [], "sdxl", (64, 64), [], 1, [])
        model, payload = sent[0]
        assert model == i2t.JOYCAPTION_MODEL and "format" not in payload and payload["keep_alive"] == 0
        assert payload["messages"][1]["content"] == i2t.DESCRIBER_PROMPT and payload["messages"][1]["images"]
        if not fail:
            assert "brass buttons" in guidance["text"] and "drop what you cannot see" in guidance["text"]
            assert r.analysis.startswith("main analysis") and "[JoyCaption の説明]" in r.analysis
        else:
            assert "brass buttons" not in guidance["text"] and r.analysis == "main analysis"
            assert any("JoyCaption の説明をスキップしました" in w for w in r.warnings)
    eng = i2t.Engine({**i2t.DEFAULT_CONFIG})   # not set: no extra call
    sent.clear()
    eng._run_vlm(img, tags, "anime", [], "sdxl", (64, 64), [], 1, [])
    assert sent == []
