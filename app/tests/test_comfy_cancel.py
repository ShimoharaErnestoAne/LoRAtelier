"""Stopping a generation: a prompt still waiting in ComfyUI's queue is taken out of it (an /interrupt sent then is
lost, ComfyUI clears the flag when the prompt starts), a running one is interrupted by its id only, and a run_prompt
started after the cancel is not affected.  ComfyUI itself is simulated."""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class FakeComfy:
    def __init__(self, state):
        self.state = state   # "pending" / "running" / "done"
        self.posts = []
        self.lock = threading.Lock()

    def get(self, path, timeout=10):
        if path == "/queue":
            item = [[0, "P1", {}, {}, []]]
            return {"queue_pending": item if self.state == "pending" else [],
                    "queue_running": item if self.state == "running" else []}
        if path.startswith("/history/"):
            if self.state == "interrupted":
                return {"P1": {"status": {"status_str": "error", "messages": [["execution_interrupted", {}]]}, "outputs": {}}}
            if self.state == "done":
                return {"P1": {"status": {"status_str": "success", "completed": True},
                               "outputs": {"8": {"images": [{"filename": "a.png"}]}}}}
            return {}
        raise AssertionError(path)

    def post(self, path, data, timeout=30):
        self.posts.append((path, data))
        if path == "/prompt":
            return {"prompt_id": "P1"}
        if path == "/queue":
            self.state = "deleted"
        if path == "/interrupt":
            assert data == {"prompt_id": "P1"}   # never the global interrupt
            self.state = "interrupted"
        return {}


@pytest.fixture()
def comfy(monkeypatch, tmp_path):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "data"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import evaluate
    monkeypatch.setattr(evaluate.time, "sleep", lambda s: None)
    return evaluate.comfy


def _run(comfy, monkeypatch, fake, cancel_before=False):
    monkeypatch.setattr(comfy, "_get", fake.get)
    monkeypatch.setattr(comfy, "_post", fake.post)
    if cancel_before:
        comfy.cancel_current()
    orig = fake.post

    def post(path, data, timeout=30):   # the cancel arrives right after the prompt was queued
        r = orig(path, data, timeout)
        if path == "/prompt" and not cancel_before:
            comfy.cancel_current()
        return r
    monkeypatch.setattr(comfy, "_post", post)
    return comfy.run_prompt({"1": {}})


def test_a_waiting_prompt_is_taken_out_of_the_queue(comfy, monkeypatch):
    fake = FakeComfy("pending")
    with pytest.raises(RuntimeError, match="中止"):
        _run(comfy, monkeypatch, fake)
    assert ("/queue", {"delete": ["P1"]}) in fake.posts and not any(p == "/interrupt" for p, _ in fake.posts)


def test_a_running_prompt_is_interrupted_by_its_id(comfy, monkeypatch):
    fake = FakeComfy("running")
    with pytest.raises(RuntimeError):
        _run(comfy, monkeypatch, fake)
    assert ("/interrupt", {"prompt_id": "P1"}) in fake.posts


def test_a_cancel_before_the_prompt_does_not_stop_it(comfy, monkeypatch):
    fake = FakeComfy("done")
    assert _run(comfy, monkeypatch, fake, cancel_before=True) == [{"filename": "a.png"}]
    assert [p for p, _ in fake.posts] == ["/prompt"]


def test_image_to_image_graph_and_source_files(comfy, monkeypatch, tmp_path):
    """I2I: the source image is stored once per content (transparency on white), encoded with the model's VAE at
    the chosen size, sampled with the chosen strength; an image the form or another result still uses is kept."""
    import io
    from PIL import Image
    from lorastudio import evaluate, imagegen
    buf = io.BytesIO()
    Image.new("RGBA", (300, 200), (255, 0, 0, 0)).save(buf, "PNG")
    r = imagegen.save_init(buf.getvalue())
    assert r["size"] == [300, 200] and imagegen.save_init(buf.getvalue())["name"] == r["name"]
    with Image.open(imagegen.init_file(r["name"])) as im:
        assert im.mode == "RGB" and im.getpixel((0, 0)) == (255, 255, 255)
    with pytest.raises(ValueError):
        imagegen.save_init(b"not an image")
    assert imagegen._denoise("x") == 0.6 and imagegen._denoise(3) == 1.0 and imagegen._denoise(0) == 0.05

    staged = {"dit": "d", "te": "t", "vae": "v", "on": "ckpt", "gguf": False}
    for fam in ("sdxl", "moss_image"):
        model = {"family": fam, "variant": "illustrious" if fam == "sdxl" else "moss_image_sft", "files": {}}
        st = {"ckpt": "c"} if fam == "sdxl" else staged
        s = {"steps": 20, "cfg": 5, "sampler": "euler", "scheduler": "normal"}
        g = evaluate.build_t2i(model, st, [], "a", "", 1, 832, 1216, s, "p", init={"image": r["name"], "denoise": 0.45})
        assert g["5a"]["inputs"]["image"] == r["name"] and g["5b"]["inputs"]["width"] == 832
        assert g["5"]["class_type"] == "VAEEncode" and g["6"]["inputs"]["denoise"] == 0.45
        assert evaluate.build_t2i(model, st, [], "a", "", 1, 832, 1216, s, "p")["6"]["inputs"]["denoise"] == 1.0

    imagegen._save_history([{"id": "a", "status": "done", "images": [], "settings": {"init_image": r["name"]}},
                            {"id": "b", "status": "done", "images": [], "settings": {"init_image": r["name"]}}])
    imagegen.delete("a")
    assert imagegen.init_file(r["name"])          # still used by b
    imagegen.delete("b")
    assert not imagegen.init_file(r["name"])
