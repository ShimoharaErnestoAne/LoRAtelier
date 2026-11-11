"""Generation fixes of the 2026-11-10 review.  Failure risks: the prompt check unpicking every generated candidate
because of quality words the tagger never reads, the background removal of a sentence prompt throwing the character
away with the place, a comparison stop pressed during the preparation waiting for the first image, a history /
saved-settings file that could not be read for a moment being saved over as empty, LoRAs of rgthree's Power Lora
Loader left pointing at another PC's files, a model the user deleted staying on the disk through this app's hard
link, and a replaced file used under the old name in a running ComfyUI."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

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


FULL = {"framing": "full", "angle": "front", "expr": "smile", "bg": "simple", "pose": "standing"}


# ------------------------------------------------------------------ 1. prompt check
def test_quality_words_and_words_taken_out_before_generating_are_not_asked_of_the_picture(env, monkeypatch):
    from lorastudio import config, generate as g, promptkeep as pk, store
    assert "masterpiece" in pk.vocabulary() and "best quality" in pk.vocabulary()   # why they were counted as missing
    asked = pk.requested("masterpiece, best quality, absurdres, very aesthetic, 1girl, long hair, school uniform", "tags")["tags"]
    assert set(asked) == {"1girl", "long hair", "school uniform"}
    p = store.ProjectManager(config.PROJECTS_DIR).create("g", "character", None)
    spec = {"positive": "masterpiece, best quality, 1girl, long hair, school uniform, outdoors, street, smile, standing, "
                        "looking at viewer", "prompt_style": "tags", "shots": []}
    p.q("INSERT INTO gen_batches(created, status, message, spec) VALUES(0,'done','',?)", (json.dumps(spec),))
    p.q("INSERT INTO gen_images(batch_id, idx, file, seed, prompt, shot, picked, created, pick_manual) "
        "VALUES(1,0,'0.png',0,'',?,1,0,0)", (json.dumps(FULL),))
    seen = {"1girl": .99, "long_hair": .9, "school_uniform": .9, "white_background": .9}   # a good candidate

    class FakeWorker:
        def call(self, op, **kw):
            return {"ok": True, "tags": [seen for _ in kw["paths"]]}
    import lorastudio.analysis as an
    monkeypatch.setattr(an, "worker", FakeWorker())
    assert g.check_prompt(p) == {"n": 1, "mismatch": 0, "unpicked": 0}
    row = p.q("SELECT picked, prompt_check FROM gen_images")[0]
    assert row["picked"] == 1 and json.loads(row["prompt_check"]) == {"conflicts": [], "missing": []}


# ------------------------------------------------------------------ 2. background removal of sentence prompts
@pytest.mark.parametrize("text", [
    "A girl with long silver hair in a school uniform.",
    "A girl in a sky blue dress and a forest green cloak.",
    "Shot at f/1.8, 1,000 lights. A girl.",
])
def test_words_that_only_look_like_a_place_stay_in_a_sentence_prompt(env, text):
    from lorastudio import generate as g
    assert g.strip_background(text, "natural") == (text, [])


def test_only_the_place_is_cut_out_of_a_sentence(env):
    from lorastudio import generate as g
    assert g.strip_background("A young woman with red hair sits in a cafe.", "natural") == \
        ("A young woman with red hair sits.", ["in a cafe"])
    p, removed = g.strip_background("A girl standing outdoors in a park, wearing a red dress, against a white background.", "natural")
    assert p.rstrip(",") == "A girl standing, wearing a red dress" and set(removed) == {"in a park", "outdoors", "against a white background"}
    p, removed = g.strip_background("A girl in a field of flowers, smiling. The background is a busy street.", "natural")
    assert p == "A girl, smiling." and removed == ["in a field of flowers", "The background is a busy street"]
    # the prompt sent for a shot still has the character
    sent = g.compose_prompt("A young woman with red hair sits in a cafe.", FULL, "natural")
    assert "red hair" in sent and "cafe" not in sent and "white background" in sent


# ------------------------------------------------------------------ 3. a comparison stopped while it is prepared
class Stopped(Exception):
    pass


def test_a_stop_pressed_while_comfyui_starts_ends_the_comparison_before_the_first_image(env, monkeypatch):
    from lorastudio import evaluate as ev
    drawn, seen = [], []
    monkeypatch.setattr(ev, "graph_for", lambda *a: {})
    for name, fn in {"ensure_started": lambda: None, "run_prompt": lambda g: drawn.append(g), "log_size": lambda: 0,
                     "park": lambda *a: None}.items():
        monkeypatch.setattr(ev.comfy, name, fn)

    def progress(done, total, eta):
        seen.append((done, total))
        raise Stopped()   # the stop was pressed during the preparation
    model = {"id": "m", "variant": "illustrious", "family": "sdxl", "files": {}}
    spec = {"checkpoints": ["final", 100], "strengths": [0.8, 1.0], "seeds": [1, 2], "include_base": True, "prompts": ["a", "b", "c"],
            "width": 64, "height": 64}
    with pytest.raises(Stopped):
        ev._run_eval_images(SimpleNamespace(id="p1", q=lambda *a: []), 1, 1, progress, {"config": {}}, {}, spec, model,
                            "sdxl", env, {"ckpt": "a.safetensors"}, {"final": "l.safetensors"}, None)
    assert not drawn and seen == [(0, (1 + 1 * 2) * 3 * 2)]   # the saved version 100 has no LoRA file: not a column


def test_a_stop_pressed_while_the_loras_are_converted_is_seen_between_them(env, monkeypatch):
    from lorastudio import evaluate as ev
    converted, parked = [], []
    monkeypatch.setattr(ev, "staged_lora", lambda pid, rid, fam, src: converted.append(src) or "x.safetensors")
    monkeypatch.setattr(ev, "_start_quietly", lambda: None)
    monkeypatch.setattr(ev, "eval_model_files", lambda m: {"ckpt": "a.safetensors"})
    monkeypatch.setattr(ev.comfy, "available", lambda: True)
    monkeypatch.setattr(ev.comfy, "park", lambda *a: parked.append(1))
    monkeypatch.setattr(ev.bus, "publish", lambda *a, **k: None)
    spec = {"checkpoints": [100, "final"], "strengths": [1.0], "seeds": [1], "include_base": False, "prompts": ["a"],
            "width": 64, "height": 64}
    run = {"run_dir": str(env / "run"), "config": {"model": {"id": "m", "variant": "illustrious", "family": "sdxl", "files": {}}}}
    project = SimpleNamespace(
        id="p1", run=lambda rid: run,
        q=lambda sql, args=(): [{"spec": json.dumps(spec)}] if sql.startswith("SELECT") else [],
        checkpoints=lambda rid: [{"kind": "lora", "step": 100, "path": "a"}, {"kind": "lora", "step": None, "path": "b"}])
    calls = []

    def progress(done, total, eta):
        calls.append((done, total))
        if len(calls) == 2:
            raise Stopped()
    with pytest.raises(Stopped):
        ev.run_eval(project, 1, 1, progress)
    assert len(converted) == 1 and calls == [(0, 2), (0, 2)] and parked   # ComfyUI is kept for the next job, not stopped


# ------------------------------------------------------------------ 5 / 6. history and saved settings
def _entry(i, status="done"):
    return {"id": f"e{i}", "created": 0, "status": status, "message": "", "settings": {"loras": []}, "images": [], "warnings": []}


def test_history_is_read_once_and_what_callers_change_does_not_reach_the_kept_one(env, monkeypatch):
    from lorastudio import imagegen as ig
    ig._save_history([_entry(i) for i in range(3)])
    reads = []
    real = Path.read_text

    def read_text(self, *a, **k):
        if self == ig.HISTORY:
            reads.append(1)
        return real(self, *a, **k)
    monkeypatch.setattr(Path, "read_text", read_text)
    h = ig.history()
    h[0]["progress"] = {"step": 3}                 # what the page's state adds to a running request
    h[0]["settings"]["seed"] = 99
    h[0]["images"].append({"file": "x.png"})
    h.pop()
    again = ig.history()
    assert len(again) == 3 and "progress" not in again[0] and "seed" not in again[0]["settings"] and again[0]["images"] == []
    images = [{"file": "a.png", "seed": 1}]
    ig._update("e1", images=images, status="running")
    images.append({"file": "b.png", "seed": 2})    # the running request goes on filling its own list
    assert ig.get_entry("e1")["images"] == [{"file": "a.png", "seed": 1}] and ig.get_entry("nope") is None
    assert not reads                               # served from memory all along
    assert [e["status"] for e in json.loads(real(ig.HISTORY, encoding="utf-8"))] == ["done", "running", "done"]   # and on the disk
    # a file changed from outside (another version of the app, a restored backup) is read again
    time.sleep(0.01)
    ig.HISTORY.write_text(json.dumps([_entry(7)]), encoding="utf-8")
    assert [e["id"] for e in ig.history()] == ["e7"] and reads == [1]


def test_a_history_that_cannot_be_read_is_never_saved_over_as_empty(env, monkeypatch):
    from lorastudio import imagegen as ig
    monkeypatch.setattr(ig, "_RETRY_S", 0)   # not time.sleep itself: that is every thread's in this process
    ig.HISTORY.parent.mkdir(parents=True, exist_ok=True)
    ig.HISTORY.write_text(json.dumps([_entry(i) for i in range(5)]), encoding="utf-8")
    ig.PRESETS.write_text(json.dumps([{"id": "p1", "name": "a"}]), encoding="utf-8")
    real = Path.read_text
    locked = {"on": True}

    def read_text(self, *a, **k):
        if locked["on"] and self in (ig.HISTORY, ig.PRESETS):
            raise PermissionError(13, "used by another process")
        return real(self, *a, **k)
    monkeypatch.setattr(Path, "read_text", read_text)
    with pytest.raises(RuntimeError, match="読めませんでした"):
        ig.history()
    with pytest.raises(RuntimeError):
        ig._update("e1", status="failed")           # before: an empty list was written here
    with pytest.raises(RuntimeError):
        ig.save_preset({"name": "b", "model_id": None, "loras": [], "positive": "x", "negative": ""})
    assert len(json.loads(real(ig.HISTORY, encoding="utf-8"))) == 5
    assert [p["name"] for p in json.loads(real(ig.PRESETS, encoding="utf-8"))] == ["a"]
    locked["on"] = False
    assert len(ig.history()) == 5
    # once read, a moment in which the file cannot be read does not matter: this app is its only writer
    locked["on"] = True
    os.utime(ig.HISTORY, ns=(1, 1))                 # looks changed, cannot be read
    ig._update("e1", status="failed")
    locked["on"] = False
    saved = json.loads(real(ig.HISTORY, encoding="utf-8"))
    assert len(saved) == 5 and saved[1]["status"] == "failed"


def test_a_broken_history_is_moved_aside_and_no_file_means_empty(env):
    from lorastudio import imagegen as ig
    assert ig.history() == [] and ig.get_form() == {} and ig.presets() == []
    ig.HISTORY.parent.mkdir(parents=True, exist_ok=True)
    ig.HISTORY.write_text('[{"id": "e1", "stat', encoding="utf-8")
    ig.FORM.write_text("[1, 2]", encoding="utf-8")   # not a form
    assert ig.history() == [] and ig.get_form() == {}
    assert len(list(ig.DIR.glob("history.broken-*.json"))) == 1 and len(list(ig.DIR.glob("form.broken-*.json"))) == 1
    ig.save_form({"positive": "a"})
    form = ig.get_form()
    form["positive"] = "changed by the caller"
    assert ig.get_form() == {"positive": "a"}


def test_a_workflow_file_is_read_once_and_callers_get_their_own_graph(env, monkeypatch):
    from lorastudio import comfyflow
    wid = comfyflow.save({"1": {"class_type": "KSampler", "inputs": {"seed": 1}}}, "ComfyUI", "s")["id"]
    reads = []
    real = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (reads.append(self.name), real(self, *a, **k))[1])
    g = comfyflow.load(wid)
    g["1"]["inputs"]["seed"] = 5
    assert comfyflow.load(wid)["1"]["inputs"]["seed"] == 1 and comfyflow.meta(wid)["nodes"] == 1
    assert reads == [f"{wid}.json"]
    (comfyflow.DIR / f"{wid}.json").unlink()
    assert comfyflow.load(wid) is None and comfyflow.meta(wid) is None


# ------------------------------------------------------------------ 8. rgthree's Power Lora Loader
def _power_graph():
    return {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sdxl\\animagine.safetensors"}},
            "2": {"class_type": "Power Lora Loader (rgthree)", "inputs": {
                "PowerLoraLoaderHeaderWidget": {"type": "PowerLoraLoaderHeaderWidget"},
                "lora_1": {"on": True, "lora": "style\\my style.safetensors", "strength": 0.8},
                "lora_2": {"on": False, "lora": "off one.safetensors", "strength": 1.0},
                "lora_3": {"on": True, "lora": "not here.safetensors", "strength": 1.0},
                "➕ Add Lora": "", "model": ["1", 0], "clip": ["1", 1]}},
            "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "1girl", "clip": ["2", 1]}},
            "4": {"class_type": "KSampler", "inputs": {"seed": 7, "steps": 20, "cfg": 5.0, "model": ["2", 0], "positive": ["3", 0]}}}


def test_loras_of_a_power_lora_loader_are_read_pointed_at_this_pc_and_missing_ones_are_said(env):
    from lorastudio import comfyflow, pnginfo
    info = pnginfo.parse_comfy(json.dumps(_power_graph()))
    assert info["loras"] == [{"name": "style\\my style.safetensors", "strength": 0.8},
                             {"name": "not here.safetensors", "strength": 1.0}]   # the one switched off is not used
    files = {"checkpoints": [], "loras": ["D:/loras/My Style.safetensors"], "vae": [], "text_encoders": [], "diffusion_models": []}
    staged = []
    g, notes, _ = comfyflow.resolve_files(_power_graph(), files, lambda kind, p: (staged.append((kind, p)), "lsg_x\\" + Path(p).name)[1])
    assert staged == [("loras", "D:/loras/My Style.safetensors")] and not notes
    assert g["2"]["inputs"]["lora_1"] == {"on": True, "lora": "lsg_x\\My Style.safetensors", "strength": 0.8}
    assert g["2"]["inputs"]["lora_3"]["lora"] == "not here.safetensors"
    spec = {c: {"input": {"required": {}}} for c in ("CheckpointLoaderSimple", "Power Lora Loader (rgthree)", "CLIPTextEncode", "KSampler")}
    spec["LoraLoader"] = {"input": {"required": {"lora_name": [["lsg_x\\My Style.safetensors", "other.safetensors"], {}]}}}
    errs = comfyflow.check(g, spec)
    assert len(errs) == 1 and "not here.safetensors" in errs[0]   # not the one switched off; the node would skip this one silently
    spec["LoraLoader"] = {"input": {"required": {"lora_name": ["COMBO", {"options": ["lsg_x\\My Style.safetensors", "sub\\not here.safetensors"]}]}}}
    assert comfyflow.check(g, spec) == []                         # the newer list form, found by its file name
    del spec["LoraLoader"]
    assert comfyflow.check(g, spec) == []                         # ComfyUI does not say which LoRAs it has: not guessed


# ------------------------------------------------------------------ 4 / 9. staged model files
def test_a_staged_file_whose_source_is_gone_or_replaced_is_removed_and_the_rest_stays(env, monkeypatch):
    from lorastudio import evaluate as ev
    comfy = ev.comfy
    keep = env / "m" / "keep.safetensors"
    gone = env / "m" / "gone.safetensors"
    keep.parent.mkdir()
    keep.write_bytes(b"k" * 10)
    gone.write_bytes(b"g" * 10)
    n_keep, n_gone = comfy.stage_model("checkpoints", str(keep)), comfy.stage_model("checkpoints", str(gone))
    old = ev.COMFY_MODELS / "loras" / "lsg_0123abcd" / "x.safetensors"   # a folder of the older naming: never used again
    old.parent.mkdir(parents=True)
    old.write_bytes(b"x")
    mine = ev.COMFY_MODELS / "loras" / "lorastudio" / "p1" / "r1" / "final.safetensors"   # not a staged folder: never touched
    mine.parent.mkdir(parents=True)
    mine.write_bytes(b"l")
    assert ev.clean_staged() == 1 and not old.parent.exists()            # everything else has its source
    gone.unlink()                                                        # the user deletes a model: the link held its data
    assert ev.clean_staged() == 1
    assert not (ev.COMFY_MODELS / "checkpoints" / n_gone).parent.exists()
    assert (ev.COMFY_MODELS / "checkpoints" / n_keep).read_bytes() == b"k" * 10 and mine.exists()
    assert comfy.stage_model("checkpoints", str(keep)) == n_keep          # staged once, used again
    # replaced under the same name: staged under a new name, the old one goes at the next start of ComfyUI
    later = keep.stat().st_mtime_ns + 5 * 10**9
    keep.unlink()
    keep.write_bytes(b"n" * 10)
    os.utime(keep, ns=(later, later))
    n_new = comfy.stage_model("checkpoints", str(keep))
    assert n_new != n_keep and (ev.COMFY_MODELS / "checkpoints" / n_new).read_bytes() == b"n" * 10
    assert ev.clean_staged() == 1 and not (ev.COMFY_MODELS / "checkpoints" / n_keep).exists()
    assert (ev.COMFY_MODELS / "checkpoints" / n_new).exists()


def test_a_lora_copied_on_a_drive_without_links_is_not_removed_while_its_source_is_there(env, monkeypatch):
    from lorastudio import evaluate as ev

    def no_link(a, b):
        raise OSError(1, "Incorrect function")
    monkeypatch.setattr(ev.os, "link", no_link)
    lora = env / "a" / "x.safetensors"
    lora.parent.mkdir()
    lora.write_bytes(b"lora")
    name = ev.comfy.stage_model("loras", str(lora))
    assert ev.clean_staged() == 0 and (ev.COMFY_MODELS / "loras" / name).read_bytes() == b"lora"
    copies = []
    monkeypatch.setattr(ev.shutil, "copyfile", lambda *a: copies.append(a))
    assert ev.comfy.stage_model("loras", str(lora)) == name and not copies   # not copied again after the clean-up
    lora.unlink()
    assert ev.clean_staged() == 1 and not (ev.COMFY_MODELS / "loras" / name).exists()


# ------------------------------------------------------------------ 13. the comparison's LoRA file
def test_the_comparisons_lora_is_a_link_where_no_conversion_is_needed_and_the_export_a_file_of_its_own(env, monkeypatch):
    from lorastudio import evaluate as ev
    src = env / "run" / "final.safetensors"
    src.parent.mkdir()
    src.write_bytes(b"lora")
    rel = ev.staged_lora("p1", 1, "sdxl", src)
    staged = ev.COMFY_MODELS / "loras" / rel
    assert os.path.samefile(staged, src) and not list(staged.parent.glob("*.part*"))
    assert ev.staged_lora("p1", 1, "sdxl", src) == rel                      # reused
    out = ev.comfy_lora_file("sdxl", src, env / "export" / "x.safetensors")
    assert out.read_bytes() == b"lora" and not os.path.samefile(out, src)   # the export: its own file
    # trained again under the same name (a new file): the comparison gets the new one
    later = src.stat().st_mtime_ns + 5 * 10**9
    src.unlink()
    src.write_bytes(b"new!")
    os.utime(src, ns=(later, later))
    ev.staged_lora("p1", 1, "sdxl", src)
    assert staged.read_bytes() == b"new!"
    # no links on this drive: copied as before; a family that is converted is never linked
    monkeypatch.setattr(ev.os, "link", lambda a, b: (_ for _ in ()).throw(OSError(1, "Incorrect function")))
    rel2 = ev.staged_lora("p1", 2, "sdxl", src)
    assert (ev.COMFY_MODELS / "loras" / rel2).read_bytes() == b"new!"
    monkeypatch.undo()
    converted = []
    monkeypatch.setattr(ev, "_convert_lora", lambda fam, s, d: (converted.append(fam), Path(d).write_bytes(b"conv"))[1])
    rel3 = ev.staged_lora("p1", 3, "anima", src)
    assert converted == ["anima"] and (ev.COMFY_MODELS / "loras" / rel3).read_bytes() == b"conv"
