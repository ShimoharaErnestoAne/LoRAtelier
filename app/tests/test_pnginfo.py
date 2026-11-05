"""画像から読み込む: the settings other tools (and this page) write into their images, put into the generation
form.  A wrong read silently changes what the user generates (a LoRA left in the prompt as text, a sampler of
another name, the wrong model)."""
from __future__ import annotations

import io
import json
import struct
import sys
from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _png(**text) -> bytes:
    meta = PngImagePlugin.PngInfo()
    for k, v in text.items():
        meta.add_text(k, v)
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "PNG", pnginfo=meta)
    return buf.getvalue()


A1111 = ("1girl, <lora:my style:0.7>, silver hair,  <lora:detail_tweaker>\n"
         "Negative prompt: lowres, bad hands\n"
         'Steps: 28, Sampler: DPM++ 2M Karras, CFG scale: 5.5, Seed: 1234, Size: 832x1216, Model hash: abc, '
         'Model: animagine-xl-3.1, Clip skip: 2, Lora hashes: "my style: 1, detail_tweaker: 2", Version: v1.10')


def test_a1111_loras_leave_the_prompt_and_old_sampler_name_splits():
    from lorastudio import pnginfo
    r = pnginfo.read(_png(parameters=A1111))
    assert r["positive"] == "1girl, silver hair"
    assert r["negative"] == "lowres, bad hands"
    assert r["loras"] == [{"name": "my style", "strength": 0.7}, {"name": "detail_tweaker", "strength": 1.0}]
    assert r["sampling"] == {"steps": 28, "cfg": 5.5, "clip_skip": 2, "sampler": "dpmpp_2m", "scheduler": "karras"}
    assert (r["seed"], r["width"], r["height"], r["model_name"]) == (1234, 832, 1216, "animagine-xl-3.1")


def test_a1111_schedule_type_and_unknown_sampler_is_said():
    from lorastudio import pnginfo
    r = pnginfo.read(_png(parameters="cat\nSteps: 20, Sampler: Euler a, Schedule type: Automatic, Seed: 5"))
    assert r["sampling"] == {"steps": 20, "sampler": "euler_ancestral"} and r["negative"] == ""
    r = pnginfo.read(_png(parameters="cat\nSteps: 20, Sampler: Weird One, Schedule type: SGM Uniform"))
    assert "sampler" not in r["sampling"] and r["sampling"]["scheduler"] == "sgm_uniform"
    assert any("Weird One" in n for n in r["notes"])


def test_comfy_custom_advanced_graph_and_linked_seed():
    from lorastudio import pnginfo
    g = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "SDXL\\noob v-pred.safetensors"}},
        "2": {"class_type": "LoraLoader", "inputs": {"lora_name": "chars\\alice.safetensors", "strength_model": 0.8,
                                                     "model": ["1", 0], "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "alice, smile", "clip": ["2", 1]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "worst quality", "clip": ["2", 1]}},
        "5": {"class_type": "CFGGuider", "inputs": {"cfg": 4.0, "positive": ["3", 0], "negative": ["4", 0], "model": ["2", 0]}},
        "6": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
        "7": {"class_type": "BasicScheduler", "inputs": {"scheduler": "sgm_uniform", "steps": 24, "denoise": 1.0}},
        "8": {"class_type": "PrimitiveInt", "inputs": {"value": 777}},
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": ["8", 0]}},
        "10": {"class_type": "EmptyLatentImage", "inputs": {"width": 896, "height": 1152, "batch_size": 1}},
        "11": {"class_type": "CLIPSetLastLayer", "inputs": {"stop_at_clip_layer": -2, "clip": ["1", 1]}},
        "12": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["9", 0], "guider": ["5", 0], "sampler": ["6", 0],
                                                                 "sigmas": ["7", 0], "latent_image": ["10", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ComfyUI", "images": ["12", 0]}},
    }
    r = pnginfo.read(_png(prompt=json.dumps(g), workflow="{}"))
    assert (r["source"], r["positive"], r["negative"]) == ("ComfyUI", "alice, smile", "worst quality")
    assert r["sampling"] == {"steps": 24, "cfg": 4.0, "sampler": "euler_ancestral", "scheduler": "sgm_uniform", "clip_skip": 2}
    assert (r["seed"], r["width"], r["height"]) == (777, 896, 1152)
    assert r["model_name"] == "SDXL\\noob v-pred.safetensors"
    assert r["loras"] == [{"name": "chars\\alice.safetensors", "strength": 0.8}]
    assert "own_entry" not in r


def test_no_settings_and_not_an_image():
    from lorastudio import pnginfo
    assert pnginfo.read(_png()) is None
    with pytest.raises(ValueError):
        pnginfo.read(b"not an image")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "data dir"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config, models
    config._settings = None
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    hdr = json.dumps({"conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight":
                      {"dtype": "F16", "shape": [1], "data_offsets": [0, 0]},
                      "model.diffusion_model.input_blocks.0.0.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 0]}}).encode()
    ck = tmp_path / "models" / "checkpoints" / "animagine-xl-3.1.safetensors"
    ck.parent.mkdir(parents=True)
    ck.write_bytes(struct.pack("<Q", len(hdr)) + hdr)
    model = models.registry.add("テスト", "illustrious", {"ckpt": str(ck)})
    lhdr = json.dumps({"lora_unet_input_blocks_4_1_proj_in.lora_down.weight": {"dtype": "F16", "shape": [4, 4], "data_offsets": [0, 32]},
                       "lora_unet_input_blocks_4_1_proj_in.lora_up.weight": {"dtype": "F16", "shape": [4, 4], "data_offsets": [32, 64]}}).encode()
    lo = tmp_path / "models" / "loras" / "my style.safetensors"
    lo.parent.mkdir(parents=True)
    lo.write_bytes(struct.pack("<Q", len(lhdr)) + lhdr + b"\0" * 64)
    monkeypatch.setattr(models.library, "_folders", lambda: [ck.parent, lo.parent])
    from fastapi.testclient import TestClient
    from lorastudio import server
    return TestClient(server.app), model, lo


def test_endpoint_matches_model_and_loras_and_says_what_is_missing(client):
    cl, model, lo = client
    r = cl.post("/api/imagegen/pnginfo", content=_png(parameters=A1111))
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["form"]["model_id"] == model["id"] and j["model_file"] is None
    assert j["form"]["loras"] == [{"path": str(lo), "strength": 0.7}]
    assert any("detail_tweaker" in n for n in j["notes"])
    assert j["form"]["positive"] == "1girl, silver hair" and j["form"]["sampling"]["sampler"] == "dpmpp_2m"
    assert cl.post("/api/imagegen/pnginfo", content=_png()).status_code == 400


def test_own_image_restores_the_typed_settings(client, monkeypatch):
    cl, model, _ = client
    from lorastudio import imagegen
    entry = {"id": "0123456789ab", "status": "done", "settings": {
        "model_id": model["id"], "loras": [], "positive": "1girl", "prompt_sent": "trig, 1girl", "negative": "bad",
        "width": 832, "height": 1216, "seed": 10, "sampling_overrides": {"steps": 30}, "regions": ["red", "blue"]}}
    monkeypatch.setattr(imagegen, "history", lambda: [entry])
    g = {"1": {"class_type": "KSampler", "inputs": {"seed": 12, "steps": 30, "cfg": 5, "sampler_name": "euler", "scheduler": "normal",
                                                    "positive": ["2", 0], "negative": ["3", 0]}},
         "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "trig, 1girl"}},
         "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "bad"}},
         "4": {"class_type": "SaveImage", "inputs": {"filename_prefix": "lsimg_0123456789ab"}}}
    j = cl.post("/api/imagegen/pnginfo", content=_png(prompt=json.dumps(g))).json()
    f = j["form"]
    assert (f["positive"], f["seed"], f["sampling"], f["regions"]) == ("1girl", 12, {"steps": 30}, {"left": "red", "right": "blue"})


# ---------------------------------------------------------------- VAE / CLIP / attention (詳細設定)
def test_comfy_vae_clip_and_sage_patch_are_read():
    from lorastudio import pnginfo
    g = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "anima.safetensors"}},
         "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_06b_base.safetensors", "type": "stable_diffusion"}},
         "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
         "4": {"class_type": "PathchSageAttentionKJ", "inputs": {"model": ["1", 0], "sage_attention": "auto"}},
         "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "cat", "clip": ["2", 0]}},
         "6": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 20, "cfg": 4, "sampler_name": "euler", "scheduler": "simple",
                                                    "positive": ["5", 0], "negative": ["5", 0], "model": ["4", 0]}}}
    r = pnginfo.read(_png(prompt=json.dumps(g)))
    assert (r["vae_name"], r["te_name"], r["attention"]) == ("qwen_image_vae.safetensors", "qwen_3_06b_base.safetensors", "sage")
    g["4"]["inputs"]["sage_attention"] = "disabled"
    assert pnginfo.read(_png(prompt=json.dumps(g)))["attention"] is None
    r = pnginfo.read(_png(parameters="cat\nSteps: 20, Seed: 3, VAE: sdxl_vae.safetensors"))
    assert r["vae_name"] == "sdxl_vae.safetensors"


def test_endpoint_vae_found_and_sage_not_installed_is_said(client, monkeypatch, tmp_path):
    cl, model, _ = client
    from lorastudio import imagegen, models
    vae = tmp_path / "models" / "vae" / "sdxl_vae.safetensors"
    vae.parent.mkdir(parents=True)
    vae.write_bytes(b"x")
    monkeypatch.setattr(models.library, "companion_files", lambda: {"vae": [str(vae)], "te": []})
    monkeypatch.setattr(imagegen, "attention_installed", lambda: {"sage": False, "flash": False})
    g = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "animagine-xl-3.1.safetensors"}},
         "2": {"class_type": "VAELoader", "inputs": {"vae_name": "sdxl_vae.safetensors"}},
         "3": {"class_type": "PatchSageAttentionKJ", "inputs": {"sage_attention": "auto"}},
         "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "cat"}},
         "5": {"class_type": "KSampler", "inputs": {"seed": 1, "positive": ["4", 0], "negative": ["4", 0]}}}
    j = cl.post("/api/imagegen/pnginfo", content=_png(prompt=json.dumps(g))).json()
    assert j["form"]["vae"] == str(vae) and j["form"]["te"] == "" and j["form"]["attention"] == ""
    assert any("SageAttention" in n for n in j["notes"])


def test_submit_keeps_vae_and_refuses_attention_not_installed(env_ig, monkeypatch):
    ig, model, runner, vae = env_ig
    monkeypatch.setattr(ig, "attention_installed", lambda: {"sage": False, "flash": False})
    form = {"model_id": model["id"], "positive": "cat", "loras": [], "vae": str(vae), "te": str(vae), "attention": "split"}
    s = ig.submit(form, runner)["settings"]
    assert s["vae"] == str(vae) and "te" not in s and s["attention"] == "split"   # SDXL: one file cannot replace its CLIPs
    with pytest.raises(ValueError, match="SageAttention"):
        ig.submit({**form, "attention": "sage"}, runner)
    with pytest.raises(ValueError, match="VAE"):
        ig.submit({**form, "vae": str(vae) + ".gone"}, runner)


@pytest.fixture()
def env_ig(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "data dir"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config, imagegen, models
    config._settings = None
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    hdr = json.dumps({"conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight":
                      {"dtype": "F16", "shape": [1], "data_offsets": [0, 0]},
                      "model.diffusion_model.input_blocks.0.0.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 0]}}).encode()
    ck = tmp_path / "models" / "checkpoints" / "m.safetensors"
    ck.parent.mkdir(parents=True)
    ck.write_bytes(struct.pack("<Q", len(hdr)) + hdr)
    vae = tmp_path / "models" / "vae" / "sdxl_vae.safetensors"
    vae.parent.mkdir(parents=True)
    vae.write_bytes(b"x")

    class Runner:
        def enqueue_task(self, owner, label, fn, on_cancel=None):
            from types import SimpleNamespace
            return SimpleNamespace()
    return imagegen, models.registry.add("テスト", "illustrious", {"ckpt": str(ck)}), Runner(), vae


def test_sdxl_graph_uses_the_chosen_vae():
    from lorastudio import evaluate
    model = {"family": "sdxl", "variant": "illustrious"}
    s = dict(evaluate.SAMPLING["sdxl"]) if "sdxl" in evaluate.SAMPLING else {"steps": 20, "cfg": 5, "sampler": "euler", "scheduler": "normal"}
    g = evaluate.build_t2i(model, {"ckpt": "m.safetensors", "vae_override": "sub\sdxl_vae.safetensors", "ppm": False},
                           [], "cat", "", 1, 1024, 1024, s, "p")
    dec = next(n for n in g.values() if n["class_type"] == "VAEDecode")
    assert g[dec["inputs"]["vae"][0]] == {"class_type": "VAELoader", "inputs": {"vae_name": "sub\sdxl_vae.safetensors"}}
    g = evaluate.build_t2i(model, {"ckpt": "m.safetensors", "ppm": False}, [], "cat", "", 1, 1024, 1024, s, "p")
    assert next(n for n in g.values() if n["class_type"] == "VAEDecode")["inputs"]["vae"] == ["1", 2]


def test_attention_change_restarts_comfy_and_other_jobs_go_back_to_standard(monkeypatch):
    from lorastudio import evaluate
    c = evaluate.Comfy()
    calls = []
    monkeypatch.setattr(c, "running", lambda: True)
    monkeypatch.setattr(c, "refresh", lambda: (calls.append("restart"), setattr(c, "started_attention", c.attention)))
    monkeypatch.setattr(c, "start", lambda: calls.append("keep"))
    c.started_extra = {}
    c.started_nodes = c._custom_nodes()
    c.ensure_started()
    c.ensure_started("sage")
    c.ensure_started("sage")
    c.ensure_started()          # e.g. a comparison image: ComfyUI's own choice again
    assert calls == ["keep", "restart", "keep", "restart"]


# ---------------------------------------------------------------- workflow のまま生成 (comfyflow)
def _hires_graph():
    """txt2img then a hires pass with a second sampler: what the form cannot express."""
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "SDXL/animagine-xl-3.1.safetensors"}},
        "2": {"class_type": "LoraLoader", "inputs": {"lora_name": "my style.safetensors", "strength_model": 0.6, "strength_clip": 0.6,
                                                     "model": ["1", 0], "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "1girl, smile", "clip": ["2", 1]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "lowres", "clip": ["2", 1]}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 832, "height": 1216, "batch_size": 1}},
        "6": {"class_type": "PrimitiveInt", "inputs": {"value": 4242}},
        "7": {"class_type": "KSampler", "inputs": {"seed": ["6", 0], "steps": 25, "cfg": 5, "sampler_name": "euler", "scheduler": "normal",
                                                   "denoise": 1, "model": ["2", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["5", 0]}},
        "8": {"class_type": "LatentUpscaleBy", "inputs": {"samples": ["7", 0], "upscale_method": "nearest-exact", "scale_by": 1.5}},
        "9": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 15, "cfg": 5, "sampler_name": "euler", "scheduler": "normal",
                                                   "denoise": 0.5, "model": ["2", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["8", 0]}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
        "11": {"class_type": "PreviewImage", "inputs": {"images": ["10", 0]}},
    }


def test_workflow_prompt_seed_and_outputs():
    from lorastudio import comfyflow
    g = _hires_graph()
    assert comfyflow.apply_prompts(g, "1girl, smile", "lowres") == g          # unchanged: the graph as it is
    g2 = comfyflow.apply_prompts(g, "1girl, angry", "")
    assert g2["3"]["inputs"]["text"] == "1girl, angry" and g2["4"]["inputs"]["text"] == "" and g["3"]["inputs"]["text"] == "1girl, smile"
    g3 = dict(g, **{"12": {"class_type": "CLIPTextEncode", "inputs": {"text": "other"}},
                    "13": {"class_type": "ConditioningCombine", "inputs": {"conditioning_1": ["3", 0], "conditioning_2": ["12", 0]}}})
    g3["7"] = {**g3["7"], "inputs": {**g3["7"]["inputs"], "positive": ["13", 0]}}
    with pytest.raises(ValueError, match="2 か所"):
        comfyflow.apply_prompts(g3, "changed", "lowres")
    assert comfyflow.seed_of(g) == 4242
    r = comfyflow.for_run(g, 99, "lsimg_abc")
    assert r["6"]["inputs"]["value"] == 99 and r["9"]["inputs"]["seed"] == 1     # the linked seed of the first sampler
    assert r["11"] == {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": "lsimg_abc"}}
    assert g["11"]["class_type"] == "PreviewImage"


def test_workflow_files_and_check():
    from lorastudio import comfyflow
    files = {"checkpoints": ["D:/models/animagine-xl-3.1.safetensors"], "loras": [], "vae": [], "text_encoders": [], "diffusion_models": []}
    g, _, _ = comfyflow.resolve_files(_hires_graph(), files, lambda kind, p: f"lsg_x/{Path(p.replace(chr(92), '/')).name}")
    assert g["1"]["inputs"]["ckpt_name"] == "lsg_x/animagine-xl-3.1.safetensors"
    assert g["2"]["inputs"]["lora_name"] == "my style.safetensors"            # not here: left for ComfyUI to say
    info = {c: {"input": {"required": {}}} for c in ("CheckpointLoaderSimple", "CLIPTextEncode", "EmptyLatentImage", "PrimitiveInt",
                                                      "KSampler", "LatentUpscaleBy", "VAEDecode", "PreviewImage")}
    info["LoraLoader"] = {"input": {"required": {"lora_name": [["a.safetensors"], {}]}}}
    errs = comfyflow.check(g, info)
    assert len(errs) == 1 and "my style.safetensors" in errs[0]
    del info["LatentUpscaleBy"]
    assert "LatentUpscaleBy" in comfyflow.check(g, info)[0]
    assert comfyflow.missing_nodes(g) == ["LatentUpscaleBy"]                   # remembered for reading images


def test_endpoint_keeps_workflow_and_submit_runs_it(client, monkeypatch):
    cl, model, lo = client
    from lorastudio import comfyflow, evaluate, imagegen
    j = cl.post("/api/imagegen/pnginfo", content=_png(prompt=json.dumps(_hires_graph()))).json()
    wid = j["workflow"]["id"]
    assert j["form"]["workflow"] == wid and j["workflow"]["nodes"] == 11 and "animagine" in j["workflow"]["summary"]
    assert j["form"]["seed"] == 4242 and j["form"]["model_id"] == model["id"]     # the form is filled too (turn workflow off)
    st = cl.put("/api/imagegen/form", json={**j["form"], "count": 2}).json()
    assert st["check"] == []
    assert cl.get("/api/imagegen").json()["workflow"]["id"] == wid
    staged, posted = [], []
    monkeypatch.setattr(evaluate.comfy, "stage_model", lambda kind, p: (staged.append((kind, p)), "lsg/" + Path(p).name)[1])
    monkeypatch.setattr(evaluate.comfy, "ensure_started", lambda attention="": None)
    info = {c: {"input": {"required": {}}} for c in {n["class_type"] for n in _hires_graph().values()} | {"SaveImage"}}
    monkeypatch.setattr(evaluate.comfy, "_get", lambda path, timeout=10: info)
    out_dir = imagegen.config.DATA_DIR / "comfy_output"

    def run_prompt(g, on_progress=None):
        posted.append(g)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"x{len(posted)}.png").write_bytes(b"png")
        return [{"filename": f"x{len(posted)}.png", "subfolder": "", "type": "output"}, {"filename": "t.png", "type": "temp"}]
    monkeypatch.setattr(evaluate.comfy, "run_prompt", run_prompt)
    monkeypatch.setattr(imagegen, "_arm_idle_stop", lambda: None)
    e = imagegen.submit({**j["form"], "count": 2, "positive": "1girl, angry"}, type("R", (), {"enqueue_task": lambda *a, **k: type("J", (), {})()})())
    assert e["settings"]["workflow"] == wid and e["settings"]["seed"] == 4242
    imagegen.run(e["id"])
    done = next(x for x in imagegen.history() if x["id"] == e["id"])
    assert done["status"] == "done", done["message"]
    assert [im["seed"] for im in done["images"]] == [4242, 4243]
    assert [g["6"]["inputs"]["value"] for g in posted] == [4242, 4243]
    assert posted[0]["3"]["inputs"]["text"] == "1girl, angry" and posted[0]["1"]["inputs"]["ckpt_name"] == "lsg/animagine-xl-3.1.safetensors"
    assert posted[0]["2"]["inputs"]["lora_name"] == "lsg/my style.safetensors" and posted[0]["11"]["class_type"] == "SaveImage"
    # a workflow whose prompt is split cannot take a changed prompt: said before queueing
    g = _hires_graph()
    g["7"]["inputs"]["positive"] = ["13", 0]
    g["12"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "other"}}
    g["13"] = {"class_type": "ConditioningCombine", "inputs": {"conditioning_1": ["3", 0], "conditioning_2": ["12", 0]}}
    wid2 = comfyflow.save(g, "ComfyUI", "")["id"]
    with pytest.raises(ValueError, match="分かれている"):
        imagegen.submit({"workflow": wid2, "positive": "new", "negative": "lowres"}, None)


def _moss_graph():
    """A MOSS-Image workflow as other people share it: rgthree's size node, the bf16 text encoder, lodestones' LoRA name."""
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "moss_image_sft.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3.5_9b_bf16.safetensors", "type": "moss_image"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "4": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "MOSS-Image-Base-8-steps-dmd2.safetensors", "strength_model": 1,
                                                              "model": ["1", 0]}},
        "5": {"class_type": "SDXL Empty Latent Image (rgthree)", "inputs": {"dimensions": " 896 x 1152  (portrait)", "clip_scale": 2, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat", "clip": ["2", 0]}},
        "7": {"class_type": "KSampler", "inputs": {"seed": 5, "steps": 8, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1,
                                                   "model": ["4", 0], "positive": ["6", 0], "negative": ["6", 0], "latent_image": ["5", 0]}},
        "8": {"class_type": "CLIPTextEncodeSDXL", "inputs": {"width": ["5", 1], "height": ["5", 2], "text_g": "x", "text_l": "x", "clip": ["2", 0]}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "ComfyUI"}},
    }


def test_moss_workflow_takes_this_pcs_files_rgthree_size_and_page_size(tmp_path):
    from lorastudio import comfyflow, pnginfo
    te, vae, lora = (tmp_path / n for n in ("qwen3.5_9b_int8_convrot.safetensors", "flux2-vae_small.safetensors", "moss_image_8steps_dmd2.safetensors"))
    for f in (te, vae, lora):
        f.write_bytes(b"x")
    dit = str(tmp_path / "moss_image_sft.safetensors")
    model = {"family": "moss_image", "files": {"gen": dit, "te": str(te), "vae": str(vae), "fast": str(lora)}}
    files = {"diffusion_models": [dit], "checkpoints": [dit], "loras": [str(lora)], "vae": [str(vae)], "text_encoders": [str(te)]}
    g, notes, m = comfyflow.resolve_files(_moss_graph(), files, lambda kind, p: "lsg/" + Path(p).name, [model])
    assert m is model
    assert g["2"]["inputs"]["clip_name"] == "lsg/qwen3.5_9b_int8_convrot.safetensors"
    assert g["3"]["inputs"]["vae_name"] == "lsg/flux2-vae_small.safetensors"
    assert g["4"]["inputs"]["lora_name"] == "lsg/moss_image_8steps_dmd2.safetensors"
    assert len(notes) == 3 and "qwen3.5_9b_bf16.safetensors" in notes[0]
    assert comfyflow.latent_size(g) == (896, 1152)
    g, more = comfyflow.substitute(g, "moss_image")
    assert g["5"] == {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": 896, "height": 1152, "batch_size": 1}}
    assert (g["8"]["inputs"]["width"], g["8"]["inputs"]["height"]) == (1792, 2304) and more
    g = comfyflow.apply_size(g, 1024, 768)
    assert (g["5"]["inputs"]["width"], g["5"]["inputs"]["height"]) == (1024, 768)
    assert comfyflow.missing_nodes(_moss_graph(), {n["class_type"] for n in _moss_graph().values()} - {"SDXL Empty Latent Image (rgthree)"}
                                   | {"EmptyLatentImage"}) == []
    r = pnginfo.read(_png(prompt=json.dumps(_moss_graph())))
    assert (r["width"], r["height"]) == (896, 1152)
    # only the SFT 8-step DMD here (another LoRA): the workflow's DMD2 is not swapped for it
    sft = tmp_path / "moss_image_sft_8steps_dmd.safetensors"
    sft.write_bytes(b"x")
    g3, notes, _ = comfyflow.resolve_files(_moss_graph(), {**files, "loras": [str(sft)]}, lambda kind, p: "lsg/" + Path(p).name,
                                           [{**model, "files": {**model["files"], "fast": str(sft)}}])
    assert g3["4"]["inputs"]["lora_name"] == "MOSS-Image-Base-8-steps-dmd2.safetensors" and len(notes) == 2
    # a name of another role / another model is not swapped
    g2 = _moss_graph()
    g2["4"]["inputs"]["lora_name"] = "someone_else.safetensors"
    _, notes, _ = comfyflow.resolve_files(g2, files, lambda kind, p: p, [model])
    assert not any("someone_else" in n for n in notes)
