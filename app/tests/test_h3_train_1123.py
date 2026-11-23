"""MiniMax H3-Image LoRA training with ai-toolkit (2026-11-23, experimental).  Failure risks: the engine given settings of
another engine or files it would download again (the ComfyUI files must be used in place, with matching qtypes), its
preview images made at every save, saves of it not registered (its names are <name>_000000250.safetensors) and its
one optimizer.pt offered for a resume it cannot do, an env installed for everyone at setup or one that needs git, the
wrong CUDA build of torch for the driver, and a LoRA ComfyUI cannot load.  Nothing here starts the engine, uv,
nvidia-smi or ComfyUI, nor downloads anything: made-up files in a temp folder (unverified on a GPU)."""
from __future__ import annotations

import json
import os
import re
import struct
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "data dir"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config
    config._settings = None
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    return tmp_path


def _st(path: Path, hdr: dict, data: bytes = b"") -> Path:
    raw = json.dumps(hdr).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)
    return path


def _lora(path: Path) -> Path:
    """A LoRA as ai-toolkit saves it for this arch (PEFT lora_A / lora_B, ComfyUI's diffusion_model. prefix)."""
    hdr = {"diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": {"dtype": "F16", "shape": [16, 2], "data_offsets": [0, 64]},
           "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight": {"dtype": "F16", "shape": [2, 16], "data_offsets": [64, 128]}}
    return _st(path, hdr, b"\0" * 128)


def _files(env, dit="minimax_h3_image_pruned_int8_convrot", te="qwen3vl_32b_minimax_h3_nvfp4_awq"):
    m = env / "モデル 置き場"
    return {"ckpt": str(m / f"{dit}.safetensors"), "te": str(m / f"{te}.safetensors"),
            "vae": str(m / "minimax_h3_image_vae_fp16.safetensors"), "adapter": str(m / "minimax_h3_image_training_adapter_v1.safetensors")}


def _snap(env, files=None, **values):
    from lorastudio import presets
    v = presets.defaults("h3_image", "character", "h3_image", 20, 24000)
    v.update(values)
    return {"kind": "character", "trigger": "chr_ab12", "engine": "ai-toolkit", "output_name": "chr_ab12_r3", "values": v,
            "model": {"id": "h3", "name": "MiniMax H3-Image", "variant": "h3_image", "family": "h3_image",
                      "files": files or _files(env)}, "items": []}


# ------------------------------------------------------------------ catalog, engine, lock file
def test_h3_image_is_trainable_with_ai_toolkit_and_its_training_files_are_in_the_catalog(env):
    from lorastudio import models
    assert models.trainable("h3_image") and models.FAMILY_ENGINE["h3_image"] == "ai-toolkit"
    assert "ai-toolkit" in models.VARIANTS["h3_image"]["note"]
    e = models.catalog_entry("h3_image_train")
    assert e["variant"] == "h3_image" and e["merge"] and e["needs_model"]
    assert e["files"]["adapter"][:2] == ("ostris/minimax_h3_training_adapter", "minimax_h3_image_training_adapter_v1.safetensors")
    assert e["needs"]["engines"] == ["ai-toolkit"]
    assert e["needs"]["tokenizers"][0][0] == "MiniMaxAI/MiniMax-H3"
    assert not any(p.endswith(".safetensors") or p.endswith("*") and "text_encoder" in p for p in e["needs"]["tokenizers"][0][1])
    assert "記載なし" in e["license"] and "MIT" in e["license"]   # the adapter's page states no license (read 2026-11-23)
    assert "adapter" in dict((k, r) for k, _l, r in models.FAMILY_FILES["h3_image"])
    assert models.COMPANIONS["h3_image"]["adapter"] == ["minimax_h3_image_training_adapter_v1.safetensors"]


def test_the_engine_is_optional_and_its_env_is_pinned_without_git_or_torch():
    lock = json.loads((REPO / "engines.lock.json").read_text(encoding="utf-8"))["engines"]["ai-toolkit"]
    assert lock["optional"] and lock["env"] == "aitk" and lock["env_python"] == "3.12" and re.fullmatch(r"[0-9a-f]{40}", lock["commit"])
    assert lock["env_torch"] == {"torch": "2.13.0", "torchvision": "0.28.0", "torchaudio": "2.11.0", "torchcodec": "0.15.0"}
    text = (REPO / "envs" / lock["env_lock"]).read_text(encoding="utf-8")
    pins = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    assert pins and all("==" in ln or " @ https://" in ln for ln in pins)
    assert "git+" not in text and not any(re.match(r"(torch|torchvision|torchaudio|torchcodec)==", ln) for ln in pins)
    assert "diffusers @ https://github.com/huggingface/diffusers/archive/c943837899b16cbae2f619b8dd4f7bb6f07dd81a.zip" in pins
    assert any(ln.startswith("transformers==5.") for ln in pins) and any(ln.startswith("triton-windows==3.7.") for ln in pins)
    # setup.bat installs only the engines that are not optional: nobody gets this env without adding H3 training files
    setup = (REPO / "setup_env.py").read_text(encoding="utf-8")
    assert 'if not e.get("optional")' in setup and "aitk" not in setup


@pytest.mark.parametrize("cuda, flavor", [(None, "cu130"), ((13, 0), "cu130"), ((13, 2), "cu130"), ((12, 8), "cu126"), ((12, 6), "cu126")])
def test_the_cuda_build_follows_the_driver_as_the_engine_chooses_it(env, cuda, flavor):
    from lorastudio import addons
    assert addons.aitk_flavor(cuda) == flavor


def test_a_driver_too_old_for_the_pinned_torch_is_said(env):
    from lorastudio import addons
    with pytest.raises(RuntimeError, match="12.6 以上"):
        addons.aitk_flavor((12, 4))


def _env_setup(env, monkeypatch, cuda=(12, 8)):
    """envs/ in the temp folder with a copy of the lock file; uv / python calls recorded instead of run."""
    from lorastudio import addons, config
    envs = env / "envs"
    envs.mkdir()
    (envs / "aitk.lock.txt").write_text((REPO / "envs" / "aitk.lock.txt").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(config, "ENVS_DIR", envs)
    monkeypatch.setattr(addons, "_uv", lambda: "uv.exe")
    monkeypatch.setattr(addons, "_nvidia_cuda", lambda: cuda)
    calls = []

    def run(argv, **kw):
        argv = [str(a) for a in argv]
        calls.append(argv)
        if argv[:2] == ["uv.exe", "venv"]:
            py = Path(argv[2]) / "Scripts" / "python.exe"
            py.parent.mkdir(parents=True)
            py.write_text("", encoding="utf-8")
        out = f"2.13.0+cu{cuda[0]}{cuda[1]}" if cuda else "2.13.0+cu130"
        if "-c" in argv and "torch" in argv[-1]:
            out = "2.13.0+" + addons.aitk_flavor(cuda)
        return SimpleNamespace(returncode=0, stdout=out, stderr="")
    monkeypatch.setattr(addons.subprocess, "run", run)
    lock = json.loads((REPO / "engines.lock.json").read_text(encoding="utf-8"))["engines"]["ai-toolkit"]
    return addons, lock, envs, calls


def test_the_env_is_made_from_the_lock_file_with_torch_for_the_driver(env, monkeypatch):
    addons, lock, envs, calls = _env_setup(env, monkeypatch, cuda=(12, 8))
    assert not addons.env_ready(lock)
    py = addons.ensure_env(lock)
    assert py == envs / "aitk" / "Scripts" / "python.exe"
    venv = next(c for c in calls if c[1] == "venv")
    assert venv[venv.index("--python") + 1] == "3.12" and "only-managed" in venv
    inst = next(c for c in calls if c[1:3] == ["pip", "install"])
    assert inst[inst.index("-r") + 1] == str(envs / "aitk.lock.txt")
    assert {"torch==2.13.0+cu126", "torchvision==0.28.0+cu126", "torchaudio==2.11.0+cu126", "torchcodec==0.15.0+cu126"} <= set(inst)
    assert "https://download.pytorch.org/whl/cu126" in inst
    assert not any("git" in Path(c[0]).name.lower() for c in calls)
    assert addons.env_ready(lock)
    n = len(calls)
    addons.ensure_env(lock)
    assert len(calls) == n                                   # ready: nothing is run again
    # an update that changes the lock file makes the env out of date (installed into again, not made anew)
    (envs / "aitk.lock.txt").write_text((envs / "aitk.lock.txt").read_text(encoding="utf-8") + "extra-package==1.0\n", encoding="utf-8")
    assert not addons.env_ready(lock)
    addons.ensure_env(lock)
    assert not any(c[1] == "venv" for c in calls[n:]) and any(c[1:3] == ["pip", "install"] for c in calls[n:])


def test_a_torch_of_the_wrong_build_leaves_the_env_not_ready(env, monkeypatch):
    addons, lock, envs, calls = _env_setup(env, monkeypatch)
    real = addons.subprocess.run

    def cpu_torch(argv, **kw):
        r = real(argv, **kw)
        return SimpleNamespace(returncode=0, stdout="2.13.0+cpu", stderr="") if "-c" in [str(a) for a in argv] else r
    monkeypatch.setattr(addons.subprocess, "run", cpu_torch)
    with pytest.raises(RuntimeError, match="想定と違います"):
        addons.ensure_env(lock)
    assert not addons.env_ready(lock)


def _github(monkeypatch, tmp_path, commit):
    """GitHub's zip of ai-toolkit at its pinned commit, made here; nothing can be started (no git, no Python)."""
    from lorastudio import addons, enginefetch

    def download(url, dest, timeout=60):
        assert url.endswith(f"/{commit}.zip")
        with zipfile.ZipFile(dest, "w") as z:
            z.writestr(f"ai-toolkit-{commit}/run.py", "print('ai-toolkit')\n")
            z.writestr(f"ai-toolkit-{commit}/toolkit/__init__.py", "")
    for g in (enginefetch.__dict__, sys.modules["lorastudio.enginefetch"].__dict__, addons.fetch_archive.__globals__):
        monkeypatch.setitem(g, "_download", download)


def test_on_a_pc_without_git_the_engine_comes_as_a_zip_and_counts_as_ready_only_with_its_env(env, monkeypatch):
    from lorastudio import addons
    install = env / "install"
    (install / "envs").mkdir(parents=True)
    lockfile = json.loads((REPO / "engines.lock.json").read_text(encoding="utf-8"))
    e = lockfile["engines"]["ai-toolkit"]
    monkeypatch.setattr(addons.config, "ROOT", install)
    monkeypatch.setattr(addons.config, "ENVS_DIR", install / "envs")
    monkeypatch.setattr(addons, "_lockfile", lambda: lockfile)
    monkeypatch.setattr(addons, "_ready", {})
    monkeypatch.setenv("PATH", os.pathsep.join([r"C:\Windows\System32", r"C:\Windows"]))
    _github(monkeypatch, env, e["commit"])
    made = []
    monkeypatch.setattr(addons, "ensure_env", lambda entry, st=None: made.append((entry["env"], st)))
    st = {"id": "h3_image_train", "state": "running"}
    addons.install_engine("ai-toolkit", st)
    d = install / "engines" / "ai-toolkit"
    assert (d / "run.py").is_file() and addons.head_commit(d) == e["commit"]
    assert made == [("aitk", st)]
    assert not addons.engine_ready("ai-toolkit")             # the engine is there, its env is not (yet)
    monkeypatch.setattr(addons, "_ready", {})
    monkeypatch.setattr(addons, "env_ready", lambda entry: True)
    assert addons.engine_ready("ai-toolkit")                  # read from the folder after a restart: no git started


# ------------------------------------------------------------------ the job ai-toolkit is given
def test_the_job_uses_the_engine_options_and_the_registered_files_in_place(env):
    from lorastudio import training
    run_dir = env / "プロジェクト 1" / "runs" / "0003_x"
    snap = _snap(env)
    job = training.aitk_config(snap, run_dir)
    assert job["job"] == "extension" and job["config"]["name"] == "chr_ab12_r3"
    p = job["config"]["process"][0]
    assert p["type"] == "sd_trainer" and p["training_folder"] == str(run_dir / "output")
    m = p["model"]
    assert m["arch"] == "minimax_h3_image" and m["assistant_lora_path"] == snap["model"]["files"]["adapter"]
    assert m["model_kwargs"] == {"dit_path": snap["model"]["files"]["ckpt"], "text_encoder_path": snap["model"]["files"]["te"],
                                 "video_vae_path": snap["model"]["files"]["vae"]}
    assert (m["quantize"], m["qtype"], m["quantize_te"], m["qtype_te"]) == (True, "convrot8", True, "nvfp4")
    v = snap["values"]
    assert p["network"] == {"type": "lora", "linear": 16, "linear_alpha": 16.0, "network_kwargs": {"ignore_if_contains": ["adaln_proj"]}}
    assert p["save"]["save_every"] == v["save_every_n_steps"] and p["save"]["max_step_saves_to_keep"] > v["max_train_steps"] // v["save_every_n_steps"]
    t = p["train"]
    assert t["steps"] == v["max_train_steps"] and t["lr"] == 1e-4 and t["optimizer"] == "adamw8bit" and t["noise_scheduler"] == "flowmatch"
    assert t["disable_sampling"] and t["skip_first_sample"] and t["cache_text_embeddings"] and not t["train_text_encoder"]
    ds = p["datasets"][0]
    assert ds["folder_path"] == str(run_dir / "dataset") and ds["caption_ext"] == "txt" and ds["resolution"] == [2048]
    # no option of another engine leaks in
    flat = json.dumps(job)
    assert not any(k in flat for k in ("network_dim", "max_train_steps", "save_every_n_steps", "blocks_to_swap", "pretrained_model_name_or_path"))
    assert training.run_out_dir(run_dir, snap) == run_dir / "output" / "chr_ab12_r3"


@pytest.mark.parametrize("dit, te, qtype, qtype_te", [
    ("minimax_h3_image_pruned_w4a8", "qwen3vl_32b_minimax_h3_int8_convrot", "convrotw4a8", "convrot8"),
    ("minimax_h3_image_pruned_bf16", "qwen3vl_32b_minimax_h3_bf16", "qfloat8", "qfloat8")])
def test_the_qtypes_load_each_file_as_it_is(env, dit, te, qtype, qtype_te):
    from lorastudio import training
    q = training.h3_qtypes(_files(env, dit, te))
    assert (q["qtype"], q["qtype_te"]) == (qtype, qtype_te)


def test_the_command_runs_the_engine_entry_with_the_seed_in_its_own_env(env, monkeypatch):
    from lorastudio import training
    monkeypatch.setattr(training, "aitk_python", lambda: env / "envs" / "aitk" / "Scripts" / "python.exe")
    run_dir = env / "run 1"
    run_dir.mkdir()
    cmds = training.build_commands(_snap(env, seed=7), run_dir, None)
    assert len(cmds) == 1
    phase, argv, cwd = cmds[0]
    assert phase == "training" and cwd == training.engine_cwd("ai-toolkit")
    assert argv[0].endswith(str(Path("envs") / "aitk" / "Scripts" / "python.exe")) and argv[1] == "-c"
    assert "SEED" in argv[2] and "run.py" in argv[2] and argv[4] == "7"
    cfg = json.loads(Path(argv[3]).read_text(encoding="utf-8"))
    assert cfg["config"]["process"][0]["model"]["arch"] == "minimax_h3_image"
    # a resume is the same job: the engine itself continues from its newest save
    assert training.build_commands(_snap(env, seed=7), run_dir, str(run_dir / "output" / "x" / "optimizer.pt"))[0][1][:3] == argv[:3]
    env_vars = training.engine_env("ai-toolkit")
    from lorastudio import config
    assert env_vars["MODELS_PATH"] == str(config.MODELS_DIR)


def test_the_progress_line_of_the_engine_is_read():
    from lorastudio import training
    r = training.aitk_step_re("chr_ab12_r3")
    m = r.search("chr_ab12_r3:  12%|#2        | 120/1000 [01:20<09:40,  1.52s/it, lr: 1.0e-04 loss: 3.215e-01]")
    assert m and (m.group(1), m.group(2), m.group(6), m.group(7)) == ("120", "1000", "s/it", "3.215e-01")
    m = r.search("chr_ab12_r3:   1%|          | 5/1000 [00:10<30:00,  1.80s/it, lr: 1.0e-04 loss: nan]")
    assert m.group(7) == "nan"
    assert r.search("chr_ab12_r3:   0%|          | 0/1000 [00:00<?, ?it/s]").group(1) == "0"
    assert not r.search("Caching latents to disk:  50%|#####     | 10/20 [00:05<00:05,  2.00it/s]")


# ------------------------------------------------------------------ saves, the state, resume
def _project_run(env, snap):
    from lorastudio import store
    pm = store.ProjectManager()
    p = pm.create("H3 テスト", "character", None)
    run_dir = p.runs_dir / "0003_x"
    out = run_dir / "output" / snap["output_name"]
    out.mkdir(parents=True)
    with p.tx() as db:
        rid = db.execute("INSERT INTO runs(created, status, name, config, run_dir) VALUES(?,?,?,?,?)",
                         (time.time(), "training", "#3", json.dumps(snap), str(run_dir))).lastrowid
    return p, rid, out


def _optimizer(path: Path, t: float):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("archive/data.pkl", b"x")
    os.utime(path, (t, t))


def test_saves_and_the_one_optimizer_state_are_registered_and_resumed_from_the_newest_only(env):
    from lorastudio import training
    snap = _snap(env, max_train_steps=1000, save_every_n_steps=250)
    p, rid, out = _project_run(env, snap)
    runner = training.Runner.__new__(training.Runner)
    job = training.Job(p, rid)
    job.output_name = snap["output_name"]
    t0 = time.time() - 100
    s250 = _lora(out / "chr_ab12_r3_000000250.safetensors")
    os.utime(s250, (t0, t0))
    _optimizer(out / "optimizer.pt", t0 + 1)
    runner._register_checkpoints(job, out, final=True)
    ck = {(c["kind"], c["step"]) for c in p.checkpoints(rid)}
    assert ck == {("lora", 250), ("state", 250)}
    assert training.state_is_complete(out / "optimizer.pt") and training.saves_state(snap)
    assert training.resume_block_reason(snap, out / "optimizer.pt") is None
    # the next save: the optimizer file is written over, its row follows the newest save (one row, not two)
    s500 = _lora(out / "chr_ab12_r3_000000500.safetensors")
    os.utime(s500, (t0 + 10, t0 + 10))
    _optimizer(out / "optimizer.pt", t0 + 11)
    runner._register_checkpoints(job, out, final=True)
    states = [c for c in p.checkpoints(rid) if c["kind"] == "state"]
    assert [(c["step"], Path(c["path"]).name) for c in states] == [(500, "optimizer.pt")]
    # an optimizer older than the newest save (still being written when the LoRA appeared) is not taken for it
    s750 = _lora(out / "chr_ab12_r3_000000750.safetensors")
    os.utime(s750, (t0 + 20, t0 + 20))
    runner._register_checkpoints(job, out, final=True)
    assert [c["step"] for c in p.checkpoints(rid) if c["kind"] == "state"] == [500]
    # the end: the newest save is the last step, nothing to continue; the final file is the LoRA without a step
    _lora(out / "chr_ab12_r3_000001000.safetensors")
    _lora(out / "chr_ab12_r3.safetensors")
    runner._register_checkpoints(job, out, final=True, last_step=1000)
    loras = sorted((c["step"] or 0) for c in p.checkpoints(rid) if c["kind"] == "lora")
    assert loras == [0, 250, 500, 750]                        # the step-1000 copy of the final LoRA is listed once
    assert "最後" in training.resume_block_reason(snap, out / "optimizer.pt")
    # a finished run: its optimizer goes, the saved LoRAs stay
    dropped = runner._drop_states(job, out, snap["output_name"])
    assert dropped["n"] == 1 and not (out / "optimizer.pt").exists() and s250.exists() and s500.exists()
    assert not [c for c in p.checkpoints(rid) if c["kind"] == "state"]


def test_a_missing_adapter_or_engine_stops_the_start_with_where_to_get_them(env, monkeypatch):
    from lorastudio import addons, training
    monkeypatch.setattr(addons, "engine_ready", lambda name: False)
    model = {"family": "h3_image", "variant": "h3_image", "files": {k: v for k, v in _files(env).items() if k != "adapter"}}
    errs = training._family_file_errors(model)
    assert any("学習アダプター" in e and "MiniMax H3-Image 学習用ファイル" in e for e in errs)
    assert any("ai-toolkit" in e for e in errs)
    monkeypatch.setattr(addons, "engine_ready", lambda name: name == "ai-toolkit")
    assert training._family_file_errors({**model, "files": _files(env)}) == []


def test_the_vram_figure_is_said_to_be_a_guess(env, monkeypatch):
    from lorastudio import training
    monkeypatch.setattr(training, "_plan_gpus", lambda: [{"name": "x", "vram_total_mb": 16376, "vram_used_mb": 0, "driver": "x"}])
    w = []
    gpus = training._plan_gpus()
    # the same words preflight gives (the rest of preflight needs a project): a guess, not a guaranteed figure
    src = Path(training.__file__).read_text(encoding="utf-8")
    assert "確かめた値ではありません" in src and training.H3_VRAM_GUESS_MB > gpus[0]["vram_total_mb"]


def test_defaults_and_checks_are_those_of_the_engine(env):
    from lorastudio import presets
    d = presets.defaults("h3_image", "character", "h3_image", 20, 24000)
    assert (d["network_dim"], d["network_alpha"], d["resolution"], d["optimizer_type"], d["lr_scheduler"]) == (16, 16, 2048, "adamw8bit", "constant")
    assert presets.validate("h3_image", d) == [] and presets.setting_warnings("h3_image", d) == []
    assert any("オプティマイザー" in w for w in presets.setting_warnings("h3_image", {**d, "optimizer_type": "AdamW8bitt"}))
    assert any("スケジューラー" in w for w in presets.setting_warnings("h3_image", {**d, "lr_scheduler": "cosine_with_min_lr"}))
    assert presets.estimate_disk_gb("h3_image", d) > 4


def test_no_caches_of_another_layout_are_given_to_the_engine(env):
    from lorastudio import traincache
    assert traincache.group_key(_snap(env)) is None


# ------------------------------------------------------------------ the LoRA in ComfyUI
def test_the_lora_keys_are_ones_comfyui_maps_for_minimax_h3(env):
    from lorastudio import models
    lora = _lora(env / "x.safetensors")
    assert models.detect_lora(lora)["family"] == "h3_image"
    comfy = REPO / "engines" / "ComfyUI" / "comfy"
    if not (comfy / "lora.py").is_file():
        pytest.skip("ComfyUI is not installed here")
    src = (comfy / "lora.py").read_text(encoding="utf-8")
    adapter = (comfy / "weight_adapter" / "lora.py").read_text(encoding="utf-8")
    # model_lora_keys_unet: "diffusion_model.<module>" maps to the model weight of the same name (the generic layout),
    # and MiniMaxH3 models (H3-Image included) map "<module>" as well; lora_B / lora_A is the "diffusers2" layout
    assert 'key_map["{}".format(k[:-len(".weight")])] = k' in src and "isinstance(model, comfy.model_base.MiniMaxH3)" in src
    assert '"{}.lora_B.weight".format(x)' in adapter and '"{}.lora_A.weight".format(x)' in adapter
    base = (comfy / "model_base.py").read_text(encoding="utf-8")
    assert re.search(r"class MiniMaxH3Image\(MiniMaxH3\)", base)
    hdr = json.loads(lora.read_bytes()[8:8 + struct.unpack("<Q", lora.read_bytes()[:8])[0]])
    modules = {k.rsplit(".lora_", 1)[0] for k in hdr}
    assert modules == {"diffusion_model.blocks.0.attn.qkv_proj"}   # + ".weight" is the model's own weight name
