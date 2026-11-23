"""Review of 2026-11-23, generation side.  Each test replays a failure found by reading the code:
a text node that holds the prompt and the negative together (Qwen-Image-2.1, MOSS-Image edit) read / written on the
wrong side, a MiniMax H3-Image result copied into a material project, a comparison after the model's files moved,
the ComfyUI kept after an image generation left in VRAM for the analysis that waited for it, a stop that waited out
the time limit while ComfyUI did not answer, the time limit of one image, the area of an H3-Image canvas, what a
comparison records (seed, clip skip, strengths that differ after the second decimal), a character tag with the name
of its work removed as a background, and comfyui.log growing for ever.
ComfyUI is simulated; nothing here runs a model (unverified on a GPU)."""
from __future__ import annotations

import json
import struct
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "データ 空白"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config
    config._settings = None
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    return tmp_path


class Runner:
    def enqueue_task(self, owner, label, fn, on_cancel=None):
        return SimpleNamespace()


def _sdxl(tmp_path, name="テスト モデル", variant="illustrious"):
    from lorastudio import models
    hdr = json.dumps({"conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight":
                      {"dtype": "F16", "shape": [1], "data_offsets": [0, 0]},
                      "model.diffusion_model.input_blocks.0.0.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 0]}}).encode()
    ck = tmp_path / "models" / "checkpoints" / f"{name}.safetensors"
    ck.parent.mkdir(parents=True, exist_ok=True)
    ck.write_bytes(struct.pack("<Q", len(hdr)) + hdr)
    return models.registry.add(name, variant, {"ckpt": str(ck)})


def _h3(env):
    from test_h3_image import _files   # the fake MiniMax H3-Image files of that test
    from lorastudio import models
    _files(env)
    models.library.scan()
    return next(x for x in models.registry.list() if x["family"] == "h3_image")


# ------------------------------------------------------------------ one node for the prompt and the negative
def _both_in_one(cls="TextEncodeQwenImage21", negative="ugly"):
    return {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen.safetensors"}},
            "3": {"class_type": cls, "inputs": {"clip": ["2", 0], "prompt": "a red apple", "negative_prompt": negative}},
            "5": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["3", 0], "negative": ["3", 1],
                                                       "latent_image": ["3", 2], "seed": 7, "steps": 20, "cfg": 4.0,
                                                       "sampler_name": "euler", "scheduler": "simple"}}}


@pytest.mark.parametrize("cls", ["TextEncodeQwenImage21", "MOSSImageEditTextEncode"])
def test_a_node_with_prompt_and_negative_is_read_by_the_output_taken(env, cls):
    from lorastudio import comfyflow, pnginfo
    g = _both_in_one(cls)
    out = pnginfo.parse_comfy(json.dumps(g))
    assert (out["positive"], out["negative"]) == ("a red apple", "ugly")   # was: the prompt on both sides
    assert comfyflow.joined(comfyflow.text_slots(g, "negative")) == "ugly" and comfyflow.has_negative(g)
    # nothing changed on the page: the graph stays as it is
    assert comfyflow.apply_prompts(g, "a red apple", "ugly") == g
    # the negative cleared or rewritten on the page goes to negative_prompt (was: written into the prompt)
    cleared = comfyflow.apply_prompts(g, "a red apple", "")["3"]["inputs"]
    assert (cleared["prompt"], cleared["negative_prompt"]) == ("a red apple", "")
    both = comfyflow.apply_prompts(g, "a green pear", "blurry")["3"]["inputs"]
    assert (both["prompt"], both["negative_prompt"]) == ("a green pear", "blurry")


def test_an_empty_negative_of_such_a_node_is_where_the_negative_goes(env):
    from lorastudio import comfyflow, pnginfo
    g = _both_in_one(negative="")
    assert pnginfo.parse_comfy(json.dumps(g))["negative"] == ""
    assert comfyflow.apply_prompts(g, "a red apple", "lowres")["3"]["inputs"] == {
        "clip": ["2", 0], "prompt": "a red apple", "negative_prompt": "lowres"}


def test_separate_text_nodes_are_read_as_before(env):
    from lorastudio import comfyflow, pnginfo
    g = {"3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "1girl"}},
         "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "lowres"}},
         "5": {"class_type": "KSampler", "inputs": {"positive": ["3", 0], "negative": ["4", 0], "seed": 1, "steps": 20}}}
    out = pnginfo.parse_comfy(json.dumps(g))
    assert (out["positive"], out["negative"]) == ("1girl", "lowres")
    new = comfyflow.apply_prompts(g, "1boy", "blurry")
    assert new["3"]["inputs"]["text"] == "1boy" and new["4"]["inputs"]["text"] == "blurry"


# ------------------------------------------------------------------ 「この設定で LoRA 素材を作る」
def test_an_h3_image_result_is_not_copied_into_a_material_project(env):
    from lorastudio import config, imagegen, store
    pm = store.ProjectManager(config.PROJECTS_DIR)
    m = _h3(env)
    e = imagegen.submit({"model_id": m["id"], "positive": "a cat", "negative": "", "loras": [], "width": 2048, "height": 2048,
                         "count": 1, "seed": 5, "sampling": {}}, Runner())
    before = {p["id"] for p in pm.list()}
    with pytest.raises(ValueError, match="素材の生成には使えません"):
        imagegen.copy_to_project(pm, e["id"], None, "新キャラ")
    assert {p["id"] for p in pm.list()} == before   # no project was made
    # an existing project keeps its recipe and its model
    p = pm.create("既存", "character", None, "include", "generate")
    p.set_meta(gen_recipe={"positive": "old", "count": 72})
    with pytest.raises(ValueError):
        imagegen.copy_to_project(pm, e["id"], p.id, None)
    assert p.meta("gen_recipe") == {"positive": "old", "count": 72} and not p.meta("model_id")


def test_copy_says_whether_the_training_model_followed(env):
    from lorastudio import config, imagegen, store
    pm = store.ProjectManager(config.PROJECTS_DIR)
    m = _sdxl(env)
    e = imagegen.submit({"model_id": m["id"], "positive": "1girl", "negative": "", "loras": [], "width": 1024, "height": 1024,
                         "count": 1, "seed": 5, "sampling": {}}, Runner())
    r = imagegen.copy_to_project(pm, e["id"], None, "新キャラ")
    assert r["model_changed"] is True and r["previous_model_id"] is None and pm.get(r["project_id"]).meta("model_id") == m["id"]
    again = imagegen.copy_to_project(pm, e["id"], r["project_id"], None)   # the same model again: nothing to reset
    assert again["model_changed"] is False
    chosen = pm.create("別モデル", "character", "other-model", "include", "images")
    kept = imagegen.copy_to_project(pm, e["id"], chosen.id, None)
    assert kept["model_changed"] is False and chosen.meta("model_id") == "other-model"


# ------------------------------------------------------------------ MiniMax H3-Image: the area of the canvas
def test_h3_image_canvas_is_limited_by_area_and_the_result_says_so(env):
    from lorastudio import imagegen
    m = _h3(env)
    form = {"model_id": m["id"], "positive": "a cat", "negative": "", "loras": [], "count": 1, "seed": 5, "sampling": {}}
    s = imagegen.submit({**form, "width": 6272, "height": 6272}, Runner())["settings"]
    assert s["width"] == s["height"] and s["width"] % 32 == 0 and s["width"] * s["height"] <= 6272 * 2688
    assert s["width"] >= 4064 and s["size_asked"] == [6272, 6272]            # as large as the limit allows, proportions kept
    wide = imagegen.submit({**form, "width": 6272, "height": 2688}, Runner())["settings"]
    assert (wide["width"], wide["height"]) == (6272, 2688) and "size_asked" not in wide   # the model card's own size
    assert imagegen.fit_area(2048, 2048, m) == (2048, 2048)
    assert imagegen.fit_area(2048, 2048, {"family": "sdxl"}) == (2048, 2048)   # the other models: sides only, as before
    tall = imagegen.fit_area(2688, 6272 + 32 * 20, m)
    assert tall[0] * tall[1] <= 6272 * 2688 and tall[0] % 32 == 0 and tall[1] % 32 == 0


# ------------------------------------------------------------------ comparison after the model's files moved
class RunProject:
    id = "p1"

    def __init__(self, model, spec, run_dir):
        self._run = {"id": 1, "run_dir": str(run_dir), "config": {"kind": "character", "outfit_mode": "include",
                                                                 "trigger": "chr_a", "items": [], "model": model}}
        self._spec = spec

    def run(self, _rid):
        return self._run

    def checkpoints(self, _rid):
        return []

    def q(self, sql, params=()):
        return [{"spec": json.dumps(self._spec)}] if sql.startswith("SELECT * FROM evals") else []


def _eval_catch(monkeypatch, ev):
    seen = {}

    def images(project, run_id, eval_id, cb, run, snap, spec, model, fam, out_dir, em, lora_names, worker, extras=(), notes=()):
        seen.update(model=model, em=em, notes=list(notes))
        return {}
    monkeypatch.setattr(ev, "_run_eval_images", images)
    monkeypatch.setattr(ev, "_start_quietly", lambda: None)
    monkeypatch.setattr(ev, "stage_files", lambda m: dict(m["files"]))
    for name, fn in {"available": lambda: True, "park": lambda *a: None, "after_failure": lambda e: None}.items():
        monkeypatch.setattr(ev.comfy, name, fn)
    return seen


def test_comparison_uses_the_registered_file_when_the_one_of_the_training_moved(env, monkeypatch):
    from lorastudio import evaluate as ev, models
    m = _sdxl(env)
    snap_model = json.loads(json.dumps(m))                      # the run's snapshot, as saved at training time
    old = Path(m["files"]["ckpt"])
    new = env / "別の場所" / old.name
    new.parent.mkdir()
    old.rename(new)
    models.registry.update(m["id"], files={"ckpt": str(new)})   # what models._repair_paths does for a moved file
    seen = _eval_catch(monkeypatch, ev)
    spec = {"checkpoints": ["final"], "strengths": [1.0], "seeds": [1], "prompts": ["x"], "width": 64, "height": 64}
    ev.run_eval(RunProject(snap_model, spec, env / "run"), 1, 1)
    assert seen["em"]["ckpt"] == str(new)                        # was: the old path, "ファイルが見つかりません"
    assert len(seen["notes"]) == 1 and old.name in seen["notes"][0] and "いまの登録" in seen["notes"][0]


def test_comparison_keeps_the_snapshot_files_while_they_are_there(env, monkeypatch):
    from lorastudio import evaluate as ev
    m = _sdxl(env)
    seen = _eval_catch(monkeypatch, ev)
    spec = {"checkpoints": ["final"], "strengths": [1.0], "seeds": [1], "prompts": ["x"], "width": 64, "height": 64}
    ev.run_eval(RunProject(json.loads(json.dumps(m)), spec, env / "run"), 1, 1)
    assert seen["em"]["ckpt"] == m["files"]["ckpt"] and seen["notes"] == []


# ------------------------------------------------------------------ analysis that waited for the GPU
def test_analysis_hooks_run_again_after_waiting_for_the_gpu(env):
    from lorastudio.gpu import GPU
    g = GPU()
    owners = []
    g.on_analysis(lambda: owners.append(g.owner))
    assert g.begin_analysis() and owners == [None]               # nothing to wait for: once, as before
    g.end_analysis()
    owners.clear()
    g.acquire("画像生成")
    t = threading.Thread(target=g.begin_analysis)
    t.start()
    for _ in range(200):
        if owners:
            break
        time.sleep(0.01)
    assert owners == ["画像生成"] and t.is_alive()                # the hook saw a running job and left its ComfyUI alone
    g.release("画像生成")
    t.join(5)
    assert owners == ["画像生成", None] and g.analysis_active      # asked again once the analysis has its turn
    g.end_analysis()


def test_a_timed_out_wait_for_the_gpu_takes_no_turn(env):
    from lorastudio.gpu import GPU
    g = GPU()
    calls = []
    g.on_analysis(lambda: calls.append(1))
    g.acquire("学習")
    assert g.begin_analysis(timeout=0.05) is False and calls == [1] and not g.analysis_active


# ------------------------------------------------------------------ ComfyUI start: partner nodes off, the log
def test_partner_nodes_are_switched_off_only_for_a_comfyui_that_knows_the_flag(env, monkeypatch):
    from lorastudio import config, evaluate as ev
    monkeypatch.setattr(config, "COMFY_DIR", env / "ComfyUI")
    assert ev._partner_flag() == []                               # not installed
    cli = env / "ComfyUI" / "comfy" / "cli_args.py"
    cli.parent.mkdir(parents=True)
    cli.write_text('parser.add_argument("--disable-api-nodes", action="store_true")\n', encoding="utf-8")
    assert ev._partner_flag() == []                               # an older ComfyUI would refuse to start with it
    cli.write_text('parser.add_argument("--disable-partner-nodes", action="store_true")\n', encoding="utf-8")
    assert ev._partner_flag() == ["--disable-partner-nodes"]


def test_the_comfyui_log_starts_again_once_it_is_large(env):
    from lorastudio import evaluate as ev
    log = env / "comfyui.log"
    log.write_bytes(b"x" * 100)
    ev._rotate_log(log, limit=1000)
    assert log.read_bytes() == b"x" * 100 and not (env / "comfyui.log.old").exists()
    (env / "comfyui.log.old").write_bytes(b"older")
    log.write_bytes(b"y" * 2000)
    ev._rotate_log(log, limit=1000)
    assert not log.exists() and (env / "comfyui.log.old").read_bytes() == b"y" * 2000   # one generation kept
    ev._rotate_log(env / "none.log", limit=1000)                  # no log yet: nothing happens
    assert ev.LOG_KEEP_BYTES == 20 * 1024 * 1024


# ------------------------------------------------------------------ waiting for one image
class Stuck:
    """A ComfyUI that takes the prompt and then answers nothing."""

    def __init__(self, comfy):
        self.comfy, self.stopped = comfy, 0

    def get(self, path, timeout=10):
        raise TimeoutError("timed out")

    def post(self, path, data, timeout=30):
        if path == "/prompt":
            self.comfy.cancel_current()   # 「中止」 pressed right after the image was queued
            return {"prompt_id": "P1"}
        return {}

    def stop(self):
        self.stopped += 1


def test_a_stop_is_seen_while_comfyui_does_not_answer(env, monkeypatch):
    from lorastudio import evaluate as ev
    comfy = ev.comfy
    fake = Stuck(comfy)
    monkeypatch.setattr(comfy, "_get", fake.get)
    monkeypatch.setattr(comfy, "_post", fake.post)
    monkeypatch.setattr(comfy, "running", lambda: True)
    monkeypatch.setattr(comfy, "stop", fake.stop)
    t0 = time.time()
    with pytest.raises(RuntimeError, match="中止しました"):       # was: waited out the whole time limit
        comfy.run_prompt({"1": {}}, timeout=20)
    assert time.time() - t0 < 10 and fake.stopped == 1            # the prompt cannot be taken out: the process is ended


class Quiet:
    """History never has the prompt; news of it keep coming over the socket for a while."""

    def __init__(self, sock=None, news_s=0.0):
        self.sock, self.until = sock, time.time() + news_s

    def get(self, path, timeout=10):
        if self.sock and time.time() < self.until:
            self.sock.last = time.time()
        time.sleep(0.02)
        return {}

    def post(self, path, data, timeout=30):
        return {"prompt_id": "P1"} if path == "/prompt" else {}


class FakeSock:
    def __init__(self):
        self.prompt_id, self.last = None, 0.0
        self.done = SimpleNamespace(clear=lambda: None, is_set=lambda: False, wait=lambda t: time.sleep(0.02))


def test_the_time_limit_counts_from_the_last_news_of_the_prompt(env, monkeypatch):
    from lorastudio import evaluate as ev
    comfy = ev.comfy
    monkeypatch.setattr(comfy, "running", lambda: True)
    # no socket (comparison, material generation): the whole image, as before
    fake = Quiet()
    monkeypatch.setattr(comfy, "_get", fake.get)
    monkeypatch.setattr(comfy, "_post", fake.post)
    t0 = time.time()
    with pytest.raises(RuntimeError, match="タイムアウト"):
        comfy._run_prompt({"1": {}}, "c", 0.4, None, None)
    assert 0.35 < time.time() - t0 < 2
    # with the socket: not while nodes and steps go on, then after the same time without news
    sock = FakeSock()
    fake = Quiet(sock, news_s=1.2)
    monkeypatch.setattr(comfy, "_get", fake.get)
    monkeypatch.setattr(comfy, "_post", fake.post)
    t0 = time.time()
    with pytest.raises(RuntimeError, match="タイムアウト.*処理が進みませんでした"):
        comfy._run_prompt({"1": {}}, "c", 0.4, sock, None)
    assert 1.5 < time.time() - t0 < 4


# ------------------------------------------------------------------ what a comparison records
class EvalProject:
    id = "p1"

    def __init__(self):
        self.saved = None

    def checkpoints(self, _rid):
        return []

    def q(self, sql, params=()):
        if sql.startswith("UPDATE evals SET status='done'"):
            self.saved = params[0]
        return []


def _compare(env, monkeypatch, variant, strengths):
    from lorastudio import config, evaluate as ev
    m = _sdxl(env, f"base_{variant}", variant)
    n = []

    def run_prompt(g):
        n.append(g)
        out = config.DATA_DIR / "comfy_output"
        out.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (len(n) * 20 % 255, 0, 0)).save(out / f"{len(n)}.png")
        return [{"filename": f"{len(n)}.png"}]
    for name, fn in {"ensure_started": lambda: None, "run_prompt": run_prompt, "log_size": lambda: 0,
                     "lora_warnings": lambda _n: [], "park": lambda *a: None}.items():
        monkeypatch.setattr(ev.comfy, name, fn)
    monkeypatch.setattr(ev, "score_eval", lambda _p, _r, _o, results, _w, _pr: {"ccip_diff": [0.1 for _ in results]})
    spec = {"checkpoints": ["final"], "strengths": strengths, "seeds": [100], "include_base": False,
            "prompts": ["chr_a, 1girl", "chr_a, 1girl, from behind"], "width": 64, "height": 64}
    run = {"id": 1, "run_dir": "", "config": {"kind": "character", "outfit_mode": "include", "trigger": "chr_a",
                                              "items": [{"id": 1, "caption": "chr_a, 1girl, solo"}], "model": m}}
    out = env / f"eval_{variant}"
    out.mkdir()
    info = ev._run_eval_images(EvalProject(), 1, 1, None, run, run["config"], spec, m, "sdxl", out, {"ckpt": "base.safetensors"},
                               {"final": "a.safetensors"}, type("W", (), {"unload": lambda self: None})(), [], [])
    return info, out


def test_strengths_that_differ_after_the_second_decimal_keep_their_own_images(env, monkeypatch):
    from lorastudio import evaluate as ev
    info, out = _compare(env, monkeypatch, "illustrious", [0.333, 0.334, 0.8])
    files = [r["file"] for r in info["images"]]
    assert len(set(files)) == len(files) == 6 and all((out / f).exists() for f in files)   # was: 0.334 over 0.333
    assert "p0_s100_final_0.80.png" in files                      # the usual strengths keep their names
    assert ev.strength_tag(1.0) == "1.00" and ev.strength_tag(0.333) != ev.strength_tag(0.334)


def test_the_conditions_say_the_seed_of_each_image_and_ponys_clip_skip(env, monkeypatch):
    from lorastudio import runcompare
    info, _ = _compare(env, monkeypatch, "pony", [1.0])
    assert [(r["prompt"], r["seed"], r["seed_used"]) for r in info["images"]] == [(0, 100, 100), (1, 100, 101)]
    c = info["conditions"]
    assert c["clip_skip"] == 2 and "seed_used" in c["seed_rule"]
    assert "clip_skip" not in c["sampling"]                       # runs are matched by "sampling": old comparisons still match
    plain, _ = _compare(env, monkeypatch, "illustrious", [1.0])
    assert "clip_skip" not in plain["conditions"]
    ev_row = {"id": 1, "status": "done", "spec": {"prompts": ["a", "b"], "width": 64, "height": 64}, "results": info}
    assert [im["seed"] for im in runcompare.run_images({"trigger": "chr_a"}, [ev_row])] == [100, 101]


# ------------------------------------------------------------------ background removal (tags)
def test_a_character_tag_with_the_name_of_its_work_is_not_a_background(env):
    from lorastudio.generate import strip_background
    kept, removed = strip_background("1girl, sophie (howl's moving castle), kyle broflovski (south park), "
                                     r"alice \(alice in wonderland\), castle, (forest:1.2), outdoors, red hair", "tags")
    assert "sophie (howl's moving castle)" in kept and "kyle broflovski (south park)" in kept   # were removed
    assert r"alice \(alice in wonderland\)" in kept and "red hair" in kept
    assert removed == ["castle", "(forest:1.2)", "outdoors"]      # places and weighted places go as before
