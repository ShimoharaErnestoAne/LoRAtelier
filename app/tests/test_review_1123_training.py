"""2026-11-23 review fixes on the training side (fake data, no GPU, no engine started).  Failure risks: images cropped
without a word when the training resolution is lowered (one square bucket at 512), a caption dropout "rate" of 10
training a LoRA without its trigger word, the disk estimate counting half the resume states, a slow nvidia-smi
turning into "no GPU" (other initial settings), the run's caches stored only after the comparison images, a full
system memory or an unreachable Hub reported as "exit code N", a KeyError always blamed on the model, a loss that
stays nan going unnoticed, a finished run marked 中止, and a VRAM watcher left behind by an engine that never started."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ENGINES = ROOT.parent / "engines"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "データ 空白"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config
    config._settings = None
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")   # never the real models folder
    return tmp_path


# ------------------------------------------------------------------ 1. buckets of a run below 1024
def _engine_buckets():
    """make_bucket_resolutions of the pinned sd-scripts, taken from its source (the module itself needs torch)."""
    src = ENGINES / "sd-scripts" / "library" / "model_util.py"
    if not src.exists():
        pytest.skip("engines/sd-scripts is not installed here")
    tree = ast.parse(src.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_bucket_resolutions")
    ns = {"math": math}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(src), "exec"), ns)
    return ns["make_bucket_resolutions"]


def test_a_low_resolution_run_keeps_buckets_for_tall_and_wide_images(env):
    from lorastudio import presets
    make = _engine_buckets()

    def widest(reso, min_size):
        return max(max(w, h) / min(w, h) for w, h in make((reso, reso), min_size, 2048, 64))
    assert widest(512, 512) == 1.0 and widest(640, 512) <= 1.5      # the earlier fixed 512: square only at 512
    for reso in (512, 576, 640, 704, 768, 896):
        mb = presets.min_bucket_reso(reso)
        assert mb % 64 == 0 and 256 <= mb <= reso                  # what the engine accepts (dataset.py asserts)
        assert widest(reso, mb) >= 3.0, reso                        # tall and wide shapes have a bucket again
    # 1024 and above: the same buckets as before
    assert [presets.min_bucket_reso(r) for r in (1024, 1152, 1536, 2048)] == [512] * 4
    assert make((1024, 1024), presets.min_bucket_reso(1024), 2048, 64) == make((1024, 1024), 512, 2048, 64)


def test_the_dataset_file_carries_the_bucket_floor_of_its_resolution(env):
    import toml
    from lorastudio import training
    run = env / "run 1"
    run.mkdir()
    ds = toml.load(training._ds_sdscripts(run, {"resolution": 512, "train_batch_size": 1}))["datasets"][0]
    assert ds["min_bucket_reso"] == 256 and ds["max_bucket_reso"] == 2048 and ds["enable_bucket"] is True
    ds = toml.load(training._ds_sdscripts(run, {"resolution": 1024, "train_batch_size": 1}))["datasets"][0]
    assert ds["min_bucket_reso"] == 512


def _snap(engine, family, resolution):
    return {"model": {"family": family, "variant": "v", "files": {"ckpt": "m/none.safetensors"}}, "engine": engine,
            "engine_commit": "abc", "values": {"resolution": resolution, "learning_rate": 1e-4}}


def _key_before(snap):
    """The key as it was computed before this change (traincache.group_key without the bucket floor)."""
    from lorastudio import traincache
    m = snap["model"]
    raw = [traincache.CACHE_VERSION, m["family"], m["variant"], snap["engine"], snap["engine_commit"],
           {k: traincache._file_id(p) for k, p in sorted(m["files"].items())}, {"resolution": snap["values"]["resolution"]}]
    return hashlib.sha1(json.dumps(raw, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def test_caches_made_at_1024_keep_their_key_and_only_lower_resolutions_get_a_new_one(env):
    from lorastudio import traincache
    for reso in (1024, 1280, 2048):
        s = _snap("sd-scripts", "sdxl", reso)
        assert traincache.group_key(s) == _key_before(s), reso      # nothing is encoded again
    for reso in (512, 768, 896):
        s = _snap("sd-scripts-moss", "moss_image", reso)
        assert traincache.group_key(s) != _key_before(s), reso      # other buckets: not the old files
    # musubi-tuner makes its own buckets: its keys are as before at every resolution
    for reso in (512, 768, 1024):
        s = _snap("musubi-tuner", "zimage", reso)
        assert traincache.group_key(s) == _key_before(s), reso


# ------------------------------------------------------------------ 2 / 14. settings that are not what was meant
VALUES = {"max_train_steps": 1600, "save_every_n_steps": 200, "seed": 1, "learning_rate": 1e-4, "network_dim": 16,
          "network_alpha": 16, "optimizer_type": "AdamW8bit", "lr_scheduler": "constant", "resolution": 1024,
          "train_batch_size": 1, "caption_dropout_rate": 0.0}


def test_a_caption_dropout_meant_as_percent_is_refused(env):
    from lorastudio import presets
    assert presets.validate("sdxl", VALUES) == [] and presets.validate("sdxl", dict(VALUES, caption_dropout_rate=0.1)) == []
    for bad in (10, 1, 1.0, -0.1, "x"):
        errs = presets.validate("sdxl", dict(VALUES, caption_dropout_rate=bad))
        assert any("説明文ドロップアウト率" in e and "0.1" in e for e in errs), bad
    # a family without the setting (musubi-tuner) is not asked for it
    v = {k: x for k, x in VALUES.items() if k != "caption_dropout_rate"}
    assert presets.validate("zimage", dict(v, blocks_to_swap=0)) == []
    assert any("説明文ドロップアウト率が 0.6" in w for w in presets.setting_warnings("sdxl", dict(VALUES, caption_dropout_rate=0.6)))
    assert presets.setting_warnings("sdxl", dict(VALUES, caption_dropout_rate=0.1)) == []


def test_a_misspelt_optimizer_or_scheduler_is_a_warning_not_an_error(env):
    from lorastudio import presets
    w = presets.setting_warnings("sdxl", dict(VALUES, optimizer_type="AdamW8bitt", lr_scheduler="Cosine"))
    assert len(w) == 2 and "AdamW8bitt" in w[0] and "Cosine" in w[1]
    assert presets.validate("sdxl", dict(VALUES, optimizer_type="AdamW8bitt", lr_scheduler="Cosine")) == []   # the start is not refused
    # the names the engines take, as the presets and the engines' own help write them
    for opt in ("AdamW8bit", "adamw8bit", "AdamW", "Prodigy", "DAdaptAdam", "AdaFactor", "Lion", "RAdam", "SGD",
                "bitsandbytes.optim.AdEMAMix8bit", ""):
        assert presets.setting_warnings("sdxl", dict(VALUES, optimizer_type=opt)) == [], opt
    for sch in ("constant", "cosine", "cosine_with_restarts", "constant_with_warmup", "linear", "polynomial", "adafactor",
                "adafactor:0.0001", "piecewise_constant"):
        assert presets.setting_warnings("anima", dict(VALUES, lr_scheduler=sch)) == [], sch
    # musubi-tuner knows fewer optimizers by name (Prodigy would be looked up in torch.optim and fail), and has "rex"
    v = {k: x for k, x in VALUES.items() if k != "caption_dropout_rate"}
    assert presets.setting_warnings("zimage", dict(v, optimizer_type="adamw8bit", lr_scheduler="rex")) == []
    assert len(presets.setting_warnings("zimage", dict(v, optimizer_type="Prodigy"))) == 1
    assert len(presets.setting_warnings("sdxl", dict(VALUES, lr_scheduler="rex"))) == 1
    # every preset's own values pass
    for fam in ("sdxl", "anima", "krea2", "qwen_image", "qwen_image21", "zimage", "moss_image"):
        for kind in ("character", "style"):
            assert presets.setting_warnings(fam, presets.defaults(fam, kind, "", 20, 16000)) == [], (fam, kind)


def test_the_optimizer_names_are_those_of_the_pinned_engines(env):
    """The lists in presets.py against the engines' sources: a name added upstream must not be warned about."""
    import re
    from lorastudio import presets
    for engine, known in (("sd-scripts", presets._SD_OPTIMIZERS), ("sd-scripts-moss", presets._SD_OPTIMIZERS)):
        src = ENGINES / engine / "library" / "optimizer.py"
        if not src.exists():
            continue
        text = src.read_text(encoding="utf-8")
        names = {n.lower() for n in re.findall(r'optimizer_type == "([A-Za-z0-9]+)"\.lower\(\)', text)}
        assert names and names <= known, (engine, names - known)


# ------------------------------------------------------------------ 5. disk estimate: the resume states really kept
def test_the_disk_estimate_counts_the_states_the_engine_leaves(env):
    from lorastudio import presets
    lora = 7.0 * 16
    # 8 save steps: sd-scripts keeps the last three of them, and writes one more at the end of the run
    assert presets.estimate_disk_gb("sdxl", VALUES) == round((lora * 9 + lora * 3 * 4 + 600) / 1024 + 1.0, 1)
    # a run with a single save step cannot leave more than that one and the last
    assert presets.estimate_disk_gb("sdxl", dict(VALUES, max_train_steps=200)) == round((lora * 2 + lora * 3 * 2 + 600) / 1024 + 1.0, 1)
    # what training.py writes is unchanged: twice the save interval
    from lorastudio import training
    assert training._common_save(VALUES, env, "x")["save_last_n_steps_state"] == 400


def test_three_states_stay_with_twice_the_interval(env):
    """The engine's own arithmetic (library/checkpoint_io.py save_and_remove_state_stepwise) for the value above."""
    every, last_n, kept = 200, 400, set()
    for step in range(every, 1601, every):
        kept.add(step)
        remove = step - last_n - 1
        remove -= remove % every
        if remove > 0:
            kept.discard(remove)
    assert kept == {1200, 1400, 1600}
    src = ENGINES / "sd-scripts" / "library" / "checkpoint_io.py"
    if src.exists():   # the lines this test models are still what the pinned engine does
        text = src.read_text(encoding="utf-8")
        assert "remove_step_no = step_no - last_n_steps - 1" in text
        assert "remove_step_no = remove_step_no - (remove_step_no % args.save_every_n_steps)" in text


# ------------------------------------------------------------------ 6. nvidia-smi that does not answer
def test_a_query_without_an_answer_keeps_the_last_gpus(env, monkeypatch):
    from lorastudio import gpu
    answers = []

    def run(*a, **k):
        x = answers.pop(0)
        if isinstance(x, Exception):
            raise x
        return SimpleNamespace(stdout=x)
    monkeypatch.setattr(gpu.shutil, "which", lambda name: "nvidia-smi.exe")
    monkeypatch.setattr(gpu.subprocess, "run", run)
    answers[:] = [subprocess.TimeoutExpired("nvidia-smi", 5)]
    assert gpu.query_gpus() == []                                   # nothing known yet
    answers[:] = ["NVIDIA GeForce RTX 4080, 16376, 1200, 560.94\n", subprocess.TimeoutExpired("nvidia-smi", 5), OSError("busy")]
    first = gpu.query_gpus()
    assert first == [{"name": "NVIDIA GeForce RTX 4080", "vram_total_mb": 16376, "vram_used_mb": 1200, "driver": "560.94"}]
    assert gpu.query_gpus() == first and gpu.query_gpus() == first   # timed out / could not start: the last answer
    first[0]["vram_total_mb"] = 0                                    # a caller changing its copy
    answers[:] = [subprocess.TimeoutExpired("nvidia-smi", 5)]
    assert gpu.query_gpus()[0]["vram_total_mb"] == 16376
    # an answer that lists no GPU (driver lost) is "no GPU", also afterwards
    answers[:] = ["NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.\n", subprocess.TimeoutExpired("nvidia-smi", 5)]
    assert gpu.query_gpus() == [] and gpu.query_gpus() == []


def test_the_initial_settings_do_not_change_when_nvidia_smi_is_slow(env, monkeypatch):
    from lorastudio import gpu, training
    answers = ["RTX, 12282, 0, 1\n", subprocess.TimeoutExpired("nvidia-smi", 5)]

    def run(*a, **k):
        x = answers.pop(0)
        if isinstance(x, Exception):
            raise x
        return SimpleNamespace(stdout=x)
    monkeypatch.setattr(gpu.shutil, "which", lambda name: "nvidia-smi.exe")
    monkeypatch.setattr(gpu.subprocess, "run", run)
    assert training._plan_gpus(ttl=-1)[0]["vram_total_mb"] == 12282
    assert training._plan_gpus(ttl=-1)[0]["vram_total_mb"] == 12282   # the second query timed out


# ------------------------------------------------------------------ 7 / 10 / 11. the runner
def _runner(monkeypatch):
    from lorastudio import store, training
    monkeypatch.setattr(training.Runner, "_loop", lambda self: None)   # no runner thread
    p = store.ProjectManager().create("学習", "character", None)
    with p.tx() as db:
        rid = db.execute("INSERT INTO runs(created, status, name, config, run_dir) VALUES(0,'training','#1','{}','')").lastrowid
    return training, training.Runner(None), p, rid


def test_caches_are_stored_and_the_merged_model_deleted_before_the_comparison(env, monkeypatch):
    training, runner, p, rid = _runner(monkeypatch)
    run_dir = env / "run 1"
    out = run_dir / "output"
    out.mkdir(parents=True)
    (out / "x.safetensors").write_bytes(b"lora")
    merged = run_dir / "copy" / "merged.safetensors"
    merged.parent.mkdir()
    merged.write_bytes(b"7GB")
    order = []
    monkeypatch.setattr(training.traincache, "harvest", lambda *a: order.append("harvest") or 0)
    runner.after_train_hooks.append(lambda project, run_id: order.append(("comparison", merged.exists())))
    monkeypatch.setattr(runner, "_exec", lambda job, *a, **k: (job.progress.update(step=1000), (0, ""))[1])
    job = training.Job(p, rid)
    job.output_name = "x"
    snap = {"output_name": "x"}
    runner._run_cmds(job, snap, run_dir, out, env, [("training", ["py", "train.py"], str(env))], 1000)
    assert order == ["harvest", ("comparison", False)]
    runner._after_engine(job, run_dir, snap)          # what Runner._run does in its "finally": not a second time
    assert order == ["harvest", ("comparison", False)]


def test_a_failed_run_still_stores_its_finished_caches_once(env, monkeypatch):
    training, runner, p, rid = _runner(monkeypatch)
    calls = []
    monkeypatch.setattr(training.traincache, "harvest", lambda *a: calls.append(1) or 0)
    job = training.Job(p, rid)
    runner._after_engine(job, env, {})                # ended before any training step: nothing to store
    assert calls == []
    job2 = training.Job(p, rid)
    job2.progress["step"] = 12
    runner._after_engine(job2, env, {})
    runner._after_engine(job2, env, {})
    assert calls == [1]


def test_a_run_that_reached_its_end_after_a_stop_request_is_finished(env, monkeypatch):
    training, runner, p, rid = _runner(monkeypatch)
    out = env / "out"
    out.mkdir()
    (out / "x.safetensors").write_bytes(b"lora")
    hooks = []
    runner.after_train_hooks.append(lambda project, run_id: hooks.append(run_id))

    def ended_itself(job, phase, argv, cwd, log_path, total, out_dir, snap=None):
        job.stop_requested_at, job.stop_after_step = 1.0, 800    # 「停止」 accepted for step 800 ...
        job.progress.update(step=1000)
        return 0, ""                                             # ... but the engine ran to the end and exited with 0
    monkeypatch.setattr(runner, "_exec", ended_itself)
    job = training.Job(p, rid)
    job.output_name = "x"
    runner._run_cmds(job, {"output_name": "x"}, env, out, env, [("training", ["py", "train.py"], str(env))], 1000)
    r = p.run(rid)
    assert r["status"] == "evaluating" and hooks == [rid]        # finished: the comparison is made
    assert "停止" not in r["progress"]["message"]

    # ended by the app at the save point, or forced: still a stop
    for mark in ("stopped_at_save", "force"):
        def stopped(job, phase, argv, cwd, log_path, total, out_dir, snap=None, mark=mark):
            job.stop_requested_at = 1.0
            setattr(job, mark, True)
            return 0, ""
        monkeypatch.setattr(runner, "_exec", stopped)
        job = training.Job(p, rid)
        job.output_name = "x"
        runner._run_cmds(job, {"output_name": "x"}, env, out, env, [("training", ["py", "train.py"], str(env))], 1000)
        assert p.run(rid)["status"] == "cancelled", mark
    assert hooks == [rid]

    # a stop during an earlier command (caches) that ended with 0 is a stop: the training command never ran
    def cache_only(job, phase, argv, cwd, log_path, total, out_dir, snap=None):
        job.stop_requested_at = 1.0
        return 0, ""
    monkeypatch.setattr(runner, "_exec", cache_only)
    job = training.Job(p, rid)
    job.output_name = "x"
    cmds = [("preparing", ["py", "cache_latents.py"], str(env)), ("training", ["py", "train.py"], str(env))]
    runner._run_cmds(job, {"output_name": "x"}, env, out, env, cmds, 1000)
    assert p.run(rid)["status"] == "cancelled" and hooks == [rid]


def test_an_engine_that_cannot_be_started_leaves_no_vram_watcher(env, monkeypatch):
    training, runner, p, rid = _runner(monkeypatch)
    monkeypatch.setattr(training, "engine_cwd", lambda e: env / e)
    monkeypatch.setattr(training, "engine_env", lambda e: {})

    def no_start(*a, **k):
        raise FileNotFoundError("python.exe")
    monkeypatch.setattr(training.subprocess, "Popen", no_start)
    job = training.Job(p, rid)
    snap = {"model": {"family": "sdxl"}, "values": {}, "engine": "sd-scripts"}
    before = {t for t in threading.enumerate() if t.name == "vram-watch"}
    with pytest.raises(FileNotFoundError):
        runner._exec(job, "training", ["python.exe", "train.py"], env, env / "00.log", 100, env / "out", snap)
    assert {t for t in threading.enumerate() if t.name == "vram-watch"} == before


# ------------------------------------------------------------------ 8. error translation
def test_system_memory_and_the_hub_have_their_own_messages(env):
    from lorastudio import training
    t = training.translate_error
    ram = ["RuntimeError: [enforce fail at alloc_cpu.cpp:121] data. DefaultCPUAllocator: not enough memory: you tried to allocate 1073741824 bytes.",
           "MemoryError",
           "numpy._core._exceptions._ArrayMemoryError: Unable to allocate 2.10 GiB for an array with shape (3, 9000, 9000)",
           "OSError: [WinError 1455] ページング ファイルが小さすぎるため、この操作を完了できません。"]
    for line in ram:
        e = t("Traceback (most recent call last):\n" + line, 1)
        assert e["code"] == "ram" and "メインメモリ" in e["message"], line
    # VRAM stays VRAM (torch.OutOfMemoryError must not read as a system-memory MemoryError)
    assert t("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB", 1)["code"] == "oom"
    hub = ["huggingface_hub.errors.LocalEntryNotFoundError: Got: ConnectError: [Errno 11001] getaddrinfo failed\nAn error happened "
           "while trying to locate the files on the Hub, and we cannot find the appropriate snapshot folder",
           "OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files."]
    for line in hub:
        e = t(line, 1)
        assert e["code"] == "network" and "Hugging Face" in e["message"], line
    assert t("FileNotFoundError: [Errno 2] No such file or directory: 'x.safetensors'", 1)["code"] == "missing"


def test_a_keyerror_alone_is_not_blamed_on_the_model(env):
    from lorastudio import training
    e = training.translate_error("  File \"dataset.py\", line 10, in __getitem__\nKeyError: 'latents_64x64'", 1)
    assert e["code"] == "key" and "一致しません" not in e["message"] and "ことがあります" in e["message"]
    for line in ("RuntimeError: Error(s) in loading state_dict for UNet:\n\tsize mismatch for conv_in.weight",
                 "\tMissing key(s) in state_dict: \"a.weight\"", "\tUnexpected key(s) in state_dict: \"b.weight\""):
        assert training.translate_error(line, 1)["code"] == "incompatible", line


# ------------------------------------------------------------------ 9. a loss that stays nan
class _Stream:
    """Engine output, one progress line per read."""

    def __init__(self, losses):
        n = len(losses)
        self.lines = [f"steps:  {i}%|#| {i}/{n} [00:10<01:30,  1.00s/it, avr_loss={x}]\r".encode() for i, x in enumerate(losses, start=1)]

    def read1(self, _n):
        return self.lines.pop(0) if self.lines else b""


def _read(env, losses, repeat=1):
    from lorastudio import training
    job = training.Job(SimpleNamespace(id="p", q=lambda *a: None, checkpoints=lambda _r: []), 1)
    job.vram_ctx = ("sdxl", {"train_batch_size": 2, "resolution": 1024})
    stream = _Stream(losses)
    stream.lines = [x for line in stream.lines for x in [line] * repeat]
    job.proc = SimpleNamespace(stdout=stream, wait=lambda: 0, pid=0)
    r = training.Runner.__new__(training.Runner)
    r._publish = lambda j: None
    r._set = lambda *a, **k: None
    seen = []
    with open(env / "log.txt", "ab") as logf:
        orig = stream.read1

        def read1(n):
            seen.append(job.progress.get("warning"))
            return orig(n)
        stream.read1 = read1
        r._read_engine(job, "training", len(losses), env / "no_output", logf, "state")
    return job.progress, seen


def test_a_loss_that_stays_nan_is_said_on_the_training_screen(env):
    p, _ = _read(env, ["0.1"] * 30 + ["nan"] * 20)
    assert p["loss_nan"] == 31 and p["loss"] is None
    assert "ステップ 31 から" in p["warning"] and "nan" in p["warning"] and "可能性があります" in p["warning"]
    assert "vram_short" not in p                                   # not a VRAM warning: the next start says nothing
    assert all(step <= 30 for step, _ in p["loss_hist"])           # nan never enters the loss graph


def test_a_few_nan_lines_or_one_slow_step_printed_many_times_do_not_warn(env):
    p, _ = _read(env, ["0.1"] * 10 + ["nan"] * 5 + ["0.2"] * 10)
    assert "warning" not in p and "loss_nan" not in p
    p, _ = _read(env, ["0.1"] * 3 + ["nan"] * 3, repeat=8)          # 24 lines, but only 3 steps
    assert "warning" not in p and "loss_nan" not in p


def test_the_nan_warning_goes_when_the_loss_is_a_number_again_and_stands_beside_the_vram_one(env):
    from lorastudio import training
    p, seen = _read(env, ["0.1"] * 5 + ["nan"] * 12 + ["0.3"] * 5)
    assert any(w and "nan" in w for w in seen) and "warning" not in p and "loss_nan" not in p
    job = training.Job(SimpleNamespace(id="p"), 1)
    job.vram_ctx = ("sdxl", {"train_batch_size": 2, "resolution": 1024})
    job.progress.update(slow_x=10.0, loss_nan=40)
    training.update_vram_warning(job)
    assert "10 分の 1" in job.progress["warning"] and "ステップ 40 から" in job.progress["warning"]
    job.progress.pop("slow_x")
    training.update_vram_warning(job)
    assert "10 分の 1" not in job.progress["warning"] and "ステップ 40 から" in job.progress["warning"]


def test_the_start_check_shows_the_setting_warnings_and_refuses_a_percent_dropout(env, monkeypatch):
    import struct
    from PIL import Image
    from lorastudio import config, models, store, training
    monkeypatch.setattr(training, "query_gpus", lambda: [{"name": "fake", "vram_total_mb": 16000, "vram_used_mb": 0, "driver": "x"}])
    hdr = json.dumps({k: {"dtype": "F16", "shape": [1], "data_offsets": [0, 0]}
                      for k in ("model.diffusion_model.x", "conditioner.embedders.1.model.y")}).encode()
    ck = env / "m.safetensors"
    ck.write_bytes(struct.pack("<Q", len(hdr)) + hdr)
    model = models.registry.add("m", "illustrious", {"ckpt": str(ck)})
    p = store.ProjectManager(config.PROJECTS_DIR).create("pf", "character", model["id"])
    imgs = []
    for i in range(3):
        f = env / f"{i}.png"
        Image.new("RGB", (1024, 1024), (i * 40, 0, 0)).save(f)
        iid, _ = p.add_image_file(f, f.name, hashlib.sha256(f.read_bytes()).hexdigest())
        p.set_auto_caption(iid, "x, solo", "k")
        imgs.append(p.image(iid))
    pf = training.preflight(p, imgs, dict(VALUES, optimizer_type="Prodigyy", caption_dropout_rate=0.5))
    assert any("Prodigyy" in w for w in pf["warnings"]) and any("説明文ドロップアウト率が 0.5" in w for w in pf["warnings"])
    assert not any("オプティマイザー" in e or "ドロップアウト" in e for e in pf["errors"])    # warnings do not block the start
    pf = training.preflight(p, imgs, dict(VALUES, caption_dropout_rate=10))
    assert any("説明文ドロップアウト率" in e for e in pf["errors"])
    pf = training.preflight(p, imgs, VALUES)
    assert not any("オプティマイザー" in w or "スケジューラー" in w or "ドロップアウト" in w for w in pf["warnings"])
