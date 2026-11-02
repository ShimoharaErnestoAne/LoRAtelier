"""Work the jobs must not throw away or repeat.  Failure risks: cancelling a comparison stops ComfyUI (the next job
pays its start-up and model load again) and loses the last finished image; musubi-tuner runs, which cannot be
resumed, write resume states at every save (GBs for the large models) and a safe stop waits for a state that never
comes; an export converts the LoRA again although the comparison already converted that saved version; adding a
comparison computes the training images' features again."""
from __future__ import annotations

import json
import sys
from pathlib import Path

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
    return tmp_path


class Cancelled(Exception):   # stands for server.EvalCancelled
    pass


class FakeProject:
    id = "p1"

    def __init__(self, run, lora):
        self._run, self._lora, self.results = run, lora, None

    def run(self, _rid):
        return self._run

    def checkpoints(self, _rid):
        return [{"step": None, "path": str(self._lora), "kind": "lora"}]

    def get_meta(self):
        return {}

    def q(self, sql, params=()):
        if sql.startswith("SELECT * FROM evals"):
            return [{"spec": json.dumps({"checkpoints": ["final"], "strengths": [1.0], "seeds": [1, 2], "include_base": True,
                                         "prompts": ["chr_a, solo"], "width": 64, "height": 64})}]
        if sql.startswith("UPDATE evals SET results"):
            self.results = json.loads(params[0])
        return []


def test_cancelled_comparison_keeps_comfyui_and_the_last_image(env, monkeypatch):
    from lorastudio import config, evaluate as ev
    ckpt = env / "モデル" / "base.safetensors"
    ckpt.parent.mkdir()
    ckpt.write_bytes(b"x")
    lora = env / "run 1" / "output" / "chr_a_r1.safetensors"
    lora.parent.mkdir(parents=True)
    lora.write_bytes(b"lora")
    run = {"id": 1, "run_dir": str(env / "run 1"), "config": {"kind": "character", "trigger": "chr_a", "items": [],
           "model": {"id": "m", "name": "base", "family": "sdxl", "variant": "illustrious", "files": {"ckpt": str(ckpt)}}}}
    calls = []
    n = []

    def run_prompt(g, **_kw):
        n.append(1)
        out = config.DATA_DIR / "comfy_output"
        out.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(out / f"{len(n)}.png")
        return [{"filename": f"{len(n)}.png"}]
    for name, fn in {"available": lambda: True, "ensure_started": lambda: None, "run_prompt": run_prompt, "log_size": lambda: 0,
                     "lora_warnings": lambda _n: [], "park": lambda *a: calls.append("park"),
                     "stop": lambda: calls.append("stop")}.items():
        monkeypatch.setattr(ev.comfy, name, fn)

    def progress(done, total, eta):
        if done == 2:
            raise Cancelled()
    p = FakeProject(run, lora)
    with pytest.raises(Cancelled):
        ev.run_eval(p, 1, 1, progress)
    assert "stop" not in calls and calls[-1] == "park"   # the process stays for the next job
    assert len(p.results["images"]) == 2                 # the image finished before the cancel is recorded
    # a real failure still stops it (ComfyUI may be in a bad state)
    calls.clear()
    monkeypatch.setattr(ev.comfy, "run_prompt", lambda g, **_kw: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        ev.run_eval(p, 1, 1, lambda *a: None)
    assert "stop" in calls


def test_musubi_runs_save_no_resume_state_and_stop_at_a_lora(env):
    import toml
    from lorastudio import presets, training
    run_dir = env / "run"
    (run_dir / "output").mkdir(parents=True)
    v = {"max_train_steps": 100, "save_every_n_steps": 25, "seed": 1, "learning_rate": 1e-4, "network_dim": 16,
         "network_alpha": 16, "optimizer_type": "AdamW8bit", "lr_scheduler": "constant", "resolution": 1024,
         "train_batch_size": 1}
    files = {"ckpt": "dit.safetensors", "vae": "vae.safetensors", "te": "te.safetensors"}
    snap = {"model": {"family": "zimage", "variant": "zimage_base", "files": files}, "engine": "musubi-tuner",
            "values": v, "output_name": "x"}
    cfg = toml.load(training.build_commands(snap, run_dir, None)[-1][1][-1])
    assert cfg["save_state"] is False and "save_last_n_steps_state" not in cfg
    assert not training.saves_state(snap)
    sd = {"model": {"family": "sdxl", "variant": "illustrious", "files": {"ckpt": "m.safetensors"}}, "engine": "sd-scripts",
          "values": v, "output_name": "x"}
    cfg = toml.load(training.build_commands(sd, run_dir, None)[-1][1][-1])
    assert cfg["save_state"] is True and cfg["save_last_n_steps_state"] == 50 and training.saves_state(sd)
    # the disk estimate leaves the states out where none are written
    # (LoRA 9MB/rank x 16 x 5 saves + 1500MB caches) / 1024 + 1; the sdxl run keeps 2 states of 3x the LoRA size
    assert presets.estimate_disk_gb("zimage", v) == round((9.0 * 16 * 5 + 1500) / 1024 + 1.0, 1)
    assert presets.estimate_disk_gb("sdxl", v) == round((7.0 * 16 * 5 + 7.0 * 16 * 6 + 600) / 1024 + 1.0, 1)


def test_export_copies_the_lora_the_comparison_converted(env, monkeypatch):
    from lorastudio import evaluate as ev, export, linking
    calls = []
    monkeypatch.setattr(ev, "_convert_lora", lambda fam, s, d: calls.append(d) or Path(d).write_bytes(b"converted") or d)
    monkeypatch.setattr(linking, "link_export", lambda _f: None)   # never into a real ComfyUI folder
    monkeypatch.setattr(export, "read_safetensors_header", lambda _p: {"w": {}})
    src = env / "run" / "output" / "a_r1.safetensors"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"native")
    ev.staged_lora("p1", 1, "anima", src)   # what the comparison does
    assert len(calls) == 1

    class P:
        id = "p1"

        def __init__(self):
            self.n = 0

        def run(self, _rid):
            return {"config": {"kind": "character", "trigger": "chr_a", "items": [], "values": {}, "changed": [],
                               "engine": "sd-scripts", "engine_commit": "x", "preset_version": 1,
                               "model": {"name": "anima", "variant": "anima", "family": "anima", "files": {}}}}

        def meta(self, _k):
            return None

        def checkpoints(self, _rid):
            return []

        def tx(self):
            import contextlib
            return contextlib.nullcontext(type("DB", (), {"execute": lambda *a: None})())
    for _ in range(2):
        out = export.export(P(), 1, str(src), 1.0, str(env / "書き出し 先"), None)
        assert Path(out["file"]).read_bytes() == b"converted"
    assert len(calls) == 1   # converted once, copied for each export


def test_training_image_features_are_computed_once_per_run(env):
    from lorastudio import evaluate as ev
    run_dir = env / "run 1"
    (run_dir / "dataset").mkdir(parents=True)
    items = [{"id": i, "sha": f"s{i}", "crop": None, "file": f"{i:05d}.png"} for i in range(3)]
    sent = []

    class W:
        def call(self, op, paths, timeout):
            sent.append(list(paths))
            return {"ok": True, "embeds": [[float(len(p))] for p in paths]}
    cache = run_dir / ev.FEATURE_CACHE
    a = ev.item_features(run_dir / "dataset", items, "embed_images", W(), 10, cache)
    b = ev.item_features(run_dir / "dataset", items, "embed_images", W(), 10, cache)
    assert a == b and len(sent) == 1
    # another crop of the same image is another picture: computed, the rest reused
    items[1] = dict(items[1], crop=[0, 0, 4, 4])
    ev.item_features(run_dir / "dataset", items, "embed_images", W(), 10, cache)
    assert len(sent) == 2 and len(sent[1]) == 1
