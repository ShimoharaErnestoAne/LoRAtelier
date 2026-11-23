"""sd-scripts v0.13.0 (2026-11-23).  Failure risks: the MOSS-Image runs still looking for the old separate checkout
(engines/sd-scripts-moss) after the two pins became one; a PC without git (engines from zips) taking the MOSS engine
for missing, or fetching it again, after a restart; a setting the app writes that the new engine no longer accepts;
the Anima block-count patch acting a second time on an engine that now does it itself; and, for finished runs, the
resume states (GBs) kept for ever although nothing can resume from them - or the saved LoRAs removed with them.
Nothing here starts an engine (made-up files in temporary folders only)."""
from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPO = ROOT.parent
V013 = "8c088eba9cd5d10a5d00dd58f0d83063fa9d57ca"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "データ 空白"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config
    config._settings = None
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")   # never the real models folder
    return tmp_path


def _lock() -> dict:
    return json.loads((REPO / "engines.lock.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ one pin for SDXL / Anima and MOSS-Image
def test_both_engine_names_are_the_one_v013_checkout(env):
    from lorastudio import config, training
    e = _lock()["engines"]
    assert e["sd-scripts"]["commit"] == e["sd-scripts-moss"]["commit"] == V013 and e["sd-scripts"]["version"] == "v0.13.0"
    assert e["sd-scripts-moss"]["path"] == "engines/sd-scripts"
    assert "triton-windows==3.4.0.post21" in e["sd-scripts-moss"]["pip"]   # still installed when MOSS-Image is added
    # runs recorded under either name (also before this update) run in, and resume from, the same folder
    assert training.engine_cwd("sd-scripts-moss") == training.engine_cwd("sd-scripts") == config.SD_SCRIPTS_DIR
    assert training.engine_env("sd-scripts-moss")["PYTHONPATH"] == str(config.SD_SCRIPTS_DIR)
    assert training.engine_cwd("musubi-tuner-qwen21") == config.ENGINES_DIR / "musubi-tuner-qwen21"   # others: as before


def test_moss_training_runs_in_the_pinned_sd_scripts(env, monkeypatch):
    import toml
    from lorastudio import config, presets, training
    monkeypatch.setattr(training, "hf_tokenizer_cached", lambda repo: True)
    f = {k: str(env / f"{k}.safetensors") for k in ("ckpt", "te", "vae")}
    v = presets.defaults("moss_image", "character", "moss_image_sft", 20, 12282)
    snap = {"model": {"family": "moss_image", "variant": "moss_image_sft", "files": f}, "values": v, "output_name": "o",
            "engine": "sd-scripts-moss", "items": []}
    run_dir = env / "run 1"
    (run_dir / "output").mkdir(parents=True)
    phase, argv, cwd = training.build_commands(snap, run_dir, None)[-1][:3]
    assert cwd == config.SD_SCRIPTS_DIR and "moss_image_train_network.py" in argv
    t = toml.load(argv[argv.index("--config_file") + 1])
    # the recipe the engine's own check of the script was made with (docs 7.2), not changed by the update
    assert (t["timestep_sampling"], t["weighting_scheme"], t["logit_mean"], t["logit_std"]) == ("sigma", "logit_normal", -1.09, 0.95)


def _fake_root(tmp: Path, main_commit: str, old_moss: str | None = None) -> Path:
    """An installation from the zip, no git: engines put in place from GitHub zips carry their commit in a mark."""
    from lorastudio.enginefetch import MARK
    root = tmp / "LoRAtelier 解凍"
    (root / "engines" / "sd-scripts").mkdir(parents=True)
    (root / "engines" / "sd-scripts" / MARK).write_text(main_commit, encoding="utf-8")
    if old_moss:   # the separate checkout of the earlier version, left on the disk
        (root / "engines" / "sd-scripts-moss").mkdir(parents=True)
        (root / "engines" / "sd-scripts-moss" / MARK).write_text(old_moss, encoding="utf-8")
    (root / "engines.lock.json").write_text((REPO / "engines.lock.json").read_text(encoding="utf-8"), encoding="utf-8")
    return root


def test_without_git_after_a_restart_the_moss_engine_is_ready_and_not_fetched_again(env, monkeypatch):
    from lorastudio import addons, config
    root = _fake_root(env, V013, old_moss="04ca541746b7667b27eab230db067579f682895e")
    monkeypatch.setattr(config, "ROOT", root)
    monkeypatch.setattr(addons, "LOCK_FILE", root / "engines.lock.json")
    monkeypatch.setenv("PATH", str(env / "no tools"))   # no git anywhere
    addons._ready.clear()
    assert addons.engine_ready("sd-scripts-moss")        # the main checkout is it; the old folder is not looked at
    fetched, ran = [], []
    monkeypatch.setattr(addons, "fetch_archive", lambda *a: fetched.append(a))
    monkeypatch.setattr(addons, "_run", lambda argv, cwd=None: ran.append([str(a) for a in argv]) or "")
    addons.install_engine("sd-scripts-moss")             # adding MOSS-Image: only the pip package goes in
    assert fetched == [] and any("triton-windows==3.4.0.post21" in a for a in ran)


def test_an_sd_scripts_not_yet_updated_is_not_taken_for_the_moss_engine(env, monkeypatch):
    from lorastudio import addons, config
    root = _fake_root(env, "690ea7f96c23182352ec63def76d431c6120bd2f")
    monkeypatch.setattr(config, "ROOT", root)
    monkeypatch.setattr(addons, "LOCK_FILE", root / "engines.lock.json")
    addons._ready.clear()
    assert not addons.engine_ready("sd-scripts-moss") and not addons.engine_ready("sd-scripts")


# ------------------------------------------------------------------ every setting the app writes is one the engine takes
def _engine_arguments(root: Path) -> set[str]:
    args = set()
    for f in [*root.glob("*.py"), *(root / "library").glob("*.py"), *(root / "networks").glob("*.py")]:
        args |= set(re.findall(r"""add_argument\(\s*["']--([A-Za-z0-9_\-]+)["']""", f.read_text(encoding="utf-8", errors="replace")))
    return args


def test_the_settings_written_for_sd_scripts_are_arguments_of_the_installed_engine(env, monkeypatch):
    """Read from the engine's source as text (nothing of it is imported): an engine update that drops or renames one
    of these would otherwise fail only at the start of a real run."""
    import toml
    from lorastudio import config, presets, training
    from lorastudio.enginefetch import head_commit
    if head_commit(config.SD_SCRIPTS_DIR) != _lock()["engines"]["sd-scripts"]["commit"]:
        pytest.skip("engines/sd-scripts is not at the pinned commit here (setup not run since the update)")
    args = _engine_arguments(config.SD_SCRIPTS_DIR)
    schema = (config.SD_SCRIPTS_DIR / "library" / "config_util.py").read_text(encoding="utf-8")
    monkeypatch.setattr(training, "hf_tokenizer_cached", lambda repo: True)
    f = {k: str(env / f"{k}.safetensors") for k in ("ckpt", "te", "vae")}
    for variant, fam in (("illustrious", "sdxl"), ("noobai_vpred", "sdxl"), ("anima", "anima"), ("moss_image_sft", "moss_image")):
        v = presets.defaults(fam, "character", variant, 20, 12282)
        for resume in (None, str(env / "o-step00000100-state")):
            snap = {"model": {"family": fam, "variant": variant, "files": f}, "values": v, "output_name": "o", "items": [],
                    "engine": "sd-scripts-moss" if fam == "moss_image" else "sd-scripts",
                    "mask": {"mode": "person", "floor": 0.1} if fam == "sdxl" else None}
            run_dir = env / f"{variant} {bool(resume)}"
            (run_dir / "output").mkdir(parents=True)
            argv = training.build_commands(snap, run_dir, resume)[-1][1]
            cfg = toml.load(argv[argv.index("--config_file") + 1])
            assert set(cfg) <= args, (variant, set(cfg) - args)
            ds = toml.load(cfg["dataset_config"])
            used = {*ds["general"], *(k for d in ds["datasets"] for k in d if k != "subsets"),
                    *(k for d in ds["datasets"] for s in d["subsets"] for k in s)}
            assert all(f'"{k}"' in schema for k in used), (variant, [k for k in used if f'"{k}"' not in schema])


# ------------------------------------------------------------------ Anima: the engine counts the blocks itself now
def _fake_library(tmp: Path, counts: bool) -> Path:
    lib = tmp / ("new" if counts else "old") / "library"
    lib.mkdir(parents=True)
    (lib / "__init__.py").write_text("", encoding="utf-8")
    (lib / "anima_models.py").write_text(
        "class Anima:\n    def __init__(self, num_blocks=28, **kw):\n        self.num_blocks = num_blocks\n", encoding="utf-8")
    body = ('    dit_config = {"num_blocks": 28}\n' if not counts else
            '    dit_config = {}\n    num_blocks = 40\n    dit_config["num_blocks"] = num_blocks\n')
    (lib / "anima_utils.py").write_text("from library import anima_models\n\n\ndef load_anima_model(device, dit_path, *a, **k):\n"
                                        + body + "    return anima_models.Anima(**dit_config)\n", encoding="utf-8")
    return lib.parent


@pytest.mark.parametrize("counts", [True, False])
def test_the_anima_block_patch_is_left_out_when_the_engine_does_it(env, monkeypatch, counts):
    for m in [m for m in sys.modules if m == "library" or m.startswith("library.")]:
        monkeypatch.delitem(sys.modules, m)
    monkeypatch.syspath_prepend(str(_fake_library(env, counts)))
    sys.path.insert(0, str(ROOT / "lorastudio" / "bridges"))
    try:
        import anima_blocks
        from library import anima_models, anima_utils
        original = anima_models.Anima.__init__
        anima_blocks.install()
        assert (anima_models.Anima.__init__ is original) is counts   # patched only for an engine fixed at 28 blocks
    finally:
        sys.path.remove(str(ROOT / "lorastudio" / "bridges"))
        for m in [m for m in sys.modules if m == "library" or m.startswith("library.") or m == "anima_blocks"]:
            sys.modules.pop(m, None)


# ------------------------------------------------------------------ resume states of a finished run
def _runner(monkeypatch):
    from lorastudio import store, training
    monkeypatch.setattr(training.Runner, "_loop", lambda self: None)   # no runner thread
    p = store.ProjectManager().create("学習", "character", None)
    with p.tx() as db:
        rid = db.execute("INSERT INTO runs(created, status, name, config, run_dir) VALUES(0,'training','#1','{}','')").lastrowid
    return training, training.Runner(None), p, rid


def _state(d: Path, step: int, mb: int = 1) -> Path:
    d.mkdir(parents=True)
    (d / "train_state.json").write_text(json.dumps({"current_step": step}), encoding="utf-8")
    (d / "model.safetensors").write_bytes(b"\0" * (mb << 20))
    import zipfile
    with zipfile.ZipFile(d / "optimizer.bin", "w") as z:
        z.writestr("x", "y")
    return d


def _lora(path: Path) -> Path:
    import struct
    hdr = json.dumps({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    path.write_bytes(struct.pack("<Q", len(hdr)) + hdr + b"\0\0\0\0")
    return path


def _out(env: Path) -> Path:
    out = env / "run 1" / "output"
    out.mkdir(parents=True)
    _state(out / "x-step00000400-state", 400)
    _state(out / "x-step00000800-state", 800)
    _state(out / "x-state", 1000)                       # the one the engine writes at the end
    _state(out / "y-step00000400-state", 400)           # another output name: not this run's
    for n in ("x-step00000400.safetensors", "x-step00000800.safetensors", "x.safetensors"):
        _lora(out / n)
    return out


def test_a_finished_run_keeps_its_loras_and_drops_its_resume_states(env, monkeypatch):
    training, runner, p, rid = _runner(monkeypatch)
    out = _out(env)
    monkeypatch.setattr(runner, "_exec", lambda job, *a, **k: (job.progress.update(step=1000), (0, ""))[1])
    job = training.Job(p, rid)
    job.output_name = "x"
    runner._run_cmds(job, {"output_name": "x"}, env / "run 1", out, env, [("training", ["py", "train.py"], str(env))], 1000)
    r = p.run(rid)
    assert r["status"] == "evaluating"
    assert sorted(f.name for f in out.iterdir()) == ["x-step00000400.safetensors", "x-step00000800.safetensors",
                                                     "x.safetensors", "y-step00000400-state"]
    kinds = [c["kind"] for c in p.checkpoints(rid)]
    assert kinds.count("state") == 0 and kinds.count("lora") == 3   # the list (and the 学習データ size) follows
    assert r["progress"]["states_removed"]["n"] == 3 and r["progress"]["states_removed"]["bytes"] >= 3 << 20


@pytest.mark.parametrize("how", ["stopped_at_save", "force", "failed"])
def test_a_stopped_or_failed_run_keeps_its_resume_states(env, monkeypatch, how):
    training, runner, p, rid = _runner(monkeypatch)
    out = _out(env)

    def ended(job, *a, **k):
        job.progress.update(step=800)
        if how == "failed":
            return 1, "RuntimeError: something"
        job.stop_requested_at = 1.0
        setattr(job, how, True)
        return 0, ""
    monkeypatch.setattr(runner, "_exec", ended)
    job = training.Job(p, rid)
    job.output_name = "x"
    runner._run_cmds(job, {"output_name": "x"}, env / "run 1", out, env, [("training", ["py", "train.py"], str(env))], 1000)
    assert p.run(rid)["status"] == ("failed" if how == "failed" else "cancelled")
    assert (out / "x-step00000800-state").is_dir() and (out / "x-step00000400-state").is_dir()   # the run can continue
    assert [c["kind"] for c in p.checkpoints(rid)].count("state") == 3


def test_a_state_folder_that_points_elsewhere_is_never_followed(env, monkeypatch):
    training, runner, p, rid = _runner(monkeypatch)
    out = env / "out"
    out.mkdir()
    elsewhere = _state(env / "ほかの 場所" / "keep", 5)
    link = out / "x-step00000005-state"
    try:
        import _winapi
        _winapi.CreateJunction(str(elsewhere), str(link))
    except (ImportError, OSError):
        pytest.skip("no junctions here")
    job = training.Job(p, rid)
    assert runner._drop_states(job, out, "x") == {"n": 0, "bytes": 0}
    assert (elsewhere / "model.safetensors").exists()
