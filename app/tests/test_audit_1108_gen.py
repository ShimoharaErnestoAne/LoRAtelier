"""Generation fixes of the 2026-11-08 review.  Failure risks: ComfyUI answering a repeated graph from its cache with an
image file this app already moved away (a workflow run several times, the same comparison image twice), export and
the comparison converting the same LoRA at once and removing each other's file, a copy cut short in ComfyUI's input
folder used for every later image-to-image run, a prompt starting with Japanese read back garbled from a JPEG, and a
failing progress report ending the signal that a prompt finished."""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "データ 空白"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config
    config._settings = None
    return tmp_path


def test_each_queued_prompt_saves_under_a_new_prefix_and_own_images_are_still_recognised(env):
    from lorastudio import evaluate, pnginfo
    graph = {"1": {"class_type": "KSampler", "inputs": {"seed": 1}},
             "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "lsimg_0123456789ab"}},
             "10": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": ["5", 0]}}}
    a, b = evaluate.fresh_save_prefixes(graph), evaluate.fresh_save_prefixes(graph)
    pa, pb = a["9"]["inputs"]["filename_prefix"], b["9"]["inputs"]["filename_prefix"]
    assert pa != pb and pa.startswith("lsimg_0123456789ab_")
    assert graph["9"]["inputs"]["filename_prefix"] == "lsimg_0123456789ab"   # the caller's graph is not changed
    assert a["1"] == graph["1"] and a["10"] == graph["10"]                   # other nodes / a linked prefix as they were
    for prefix in (pa, "lsimg_0123456789ab"):   # images made before and after the change open their history entry
        assert pnginfo.RE_OWN.match(prefix).group(1) == "0123456789ab"


def test_run_prompt_sends_the_fresh_prefix(env, monkeypatch):
    from lorastudio import evaluate
    sent = []

    def post(path, body, timeout=10):
        sent.append(body["prompt"])
        return {"prompt_id": "p1"}
    monkeypatch.setattr(evaluate.comfy, "_post", post)
    monkeypatch.setattr(evaluate.comfy, "_get", lambda path, timeout=10: {
        "p1": {"status": {"completed": True}, "outputs": {"9": {"images": [{"filename": "x.png"}]}}}})
    g = {"9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "lsgen_p_b1"}}}
    evaluate.comfy.run_prompt(g)
    evaluate.comfy.run_prompt(g)
    assert len({s["9"]["inputs"]["filename_prefix"] for s in sent}) == 2


def test_a_missing_output_file_is_reported_in_words(env):
    from lorastudio import config, evaluate
    out = config.DATA_DIR / "comfy_output"
    out.mkdir(parents=True)
    (out / "a.png").write_bytes(b"png")
    evaluate.take_output({"filename": "a.png"}, env / "a.png")
    assert (env / "a.png").read_bytes() == b"png" and not (out / "a.png").exists()
    with pytest.raises(RuntimeError, match="画像ファイルが見つかりません"):
        evaluate.take_output({"filename": "a.png"}, env / "b.png")


def test_export_and_comparison_converting_the_same_lora_at_once_convert_it_once(env, monkeypatch):
    from lorastudio import evaluate
    calls = []

    def convert(fam, s, d):
        calls.append(d)
        time.sleep(0.2)   # the other caller arrives while this one writes
        Path(d).write_bytes(Path(s).read_bytes())
    monkeypatch.setattr(evaluate, "_convert_lora", convert)
    src = env / "run" / "x.safetensors"
    src.parent.mkdir()
    src.write_bytes(b"v1")
    dest = env / "loras" / "x.safetensors"
    errors = []

    def go():
        try:
            evaluate.comfy_lora_file("anima", src, dest)
        except Exception as e:   # noqa: BLE001
            errors.append(e)
    ts = [threading.Thread(target=go) for _ in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors and len(calls) == 1 and dest.read_bytes() == b"v1"
    assert not list(dest.parent.glob("*.part*"))


def test_a_failed_conversion_leaves_no_part_file(env, monkeypatch):
    from lorastudio import evaluate

    def convert(fam, s, d):
        Path(d).write_bytes(b"half")
        raise RuntimeError("変換に失敗")
    monkeypatch.setattr(evaluate, "_convert_lora", convert)
    src = env / "x.safetensors"
    src.write_bytes(b"v1")
    with pytest.raises(RuntimeError):
        evaluate.comfy_lora_file("anima", src, env / "out" / "x.safetensors")
    assert not list((env / "out").iterdir())


def test_a_copy_cut_short_in_comfyuis_input_folder_is_made_again(env):
    from lorastudio import evaluate, generate
    src = env / "abc.png"
    src.write_bytes(b"0123456789")
    evaluate.COMFY_INPUT.mkdir(parents=True)
    (evaluate.COMFY_INPUT / "abc.png").write_bytes(b"0123")   # an earlier copy ended by a crash
    assert generate.stage_comfy_input(src) == "abc.png"
    assert (evaluate.COMFY_INPUT / "abc.png").read_bytes() == b"0123456789"
    assert [p.name for p in evaluate.COMFY_INPUT.iterdir()] == ["abc.png"]


@pytest.mark.parametrize("body,want", [
    ("夕焼け, 1girl".encode("utf-16-be"), "夕焼け, 1girl"),        # A1111 (piexif): big-endian, starts with Japanese
    ("1girl, 夕焼け".encode("utf-16-be"), "1girl, 夕焼け"),
    ("1girl, 夕焼け".encode("utf-16-le"), "1girl, 夕焼け"),
    ("夕焼け".encode("utf-16"), "夕焼け"),                          # with a byte order mark
])
def test_exif_user_comment_utf16_byte_order(env, body, want):
    from lorastudio import pnginfo
    assert pnginfo._decode_user_comment(b"UNICODE\x00" + body) == want


def test_a_failing_progress_report_does_not_hide_the_end_of_the_prompt(env):
    from lorastudio import evaluate

    def frame(obj):
        data = json.dumps(obj).encode()
        assert len(data) < 126
        return bytes([0x81, len(data)]) + data

    class Sock:
        def recv(self, n):
            return b""

    ws = evaluate._ProgressSocket.__new__(evaluate._ProgressSocket)
    ws.prompt_id, ws.done, ws._closed, ws._s = "p", threading.Event(), False, Sock()
    ws._cb = lambda step, steps: 1 / 0
    ws._buf = frame({"type": "progress", "data": {"prompt_id": "p", "value": 1, "max": 4}}) + \
        frame({"type": "execution_success", "data": {"prompt_id": "p"}})
    ws._loop()
    assert ws.done.is_set()
