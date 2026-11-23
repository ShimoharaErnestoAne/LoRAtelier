"""Review of 2026-11-23, server side.  Failure risks: a LoRA creation task that generates and analyses for hours and
only then finds out it cannot train; the training model following the generation model with the old family's
learning rate / rank still set; a generation-only model (MiniMax H3-Image) becoming a project's training model; a
finished run resumed over its final LoRA; a multi-character LoRA calling itself by a word no caption has; a restart
that does not say it stops a data move; a waiting generation batch of a cancelled task still taking the GPU.
Nothing here runs an engine, ComfyUI or the analysis AI (fake files in a temp folder)."""
from __future__ import annotations

import contextlib
import json
import sqlite3
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "データ 空白"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config, models
    config._settings = None
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    ck = tmp_path / "base.safetensors"
    ck.write_bytes(b"0")
    sdxl = models.registry.add("SDXL", "sdxl_base", {"ckpt": str(ck)})
    anima = models.registry.add("Anima", "anima", {"ckpt": str(ck), "te": str(ck), "vae": str(ck)})
    from lorastudio import server
    return SimpleNamespace(server=server, sdxl=sdxl, anima=anima, tmp=tmp_path, models=models)


def _h3(env):
    """A registered generation-only model (the record as the registry keeps it; no real file is read)."""
    return {"id": "h3x", "name": "H3", "variant": "h3_image", "family": "h3_image", "files": {}, "missing": []}


def test_training_model_following_the_generation_model_drops_the_old_family_settings(env):
    s = env.server
    p = s.projects.create("キャラ", "character", env.sdxl["id"], "include", "generate")
    ov = {"learning_rate": 5e-5, "network_dim": 64, "max_train_steps": 1200}
    p.set_meta(train_overrides=ov, gen_recipe={"gen_model_id": env.sdxl["id"]})
    r = s.gen_recipe(p.id, {"gen_model_id": env.anima["id"]})
    assert p.meta("model_id") == env.anima["id"]                       # it followed (it was the same model before)
    assert p.meta("train_overrides") == {"max_train_steps": 1200}       # SDXL's learning rate and rank are not Anima's
    assert len(r["project"]["reset_settings"]) == 2


def test_a_training_model_chosen_by_hand_does_not_follow(env):
    s = env.server
    p = s.projects.create("キャラ", "character", env.sdxl["id"], "include", "generate")
    ov = {"learning_rate": 5e-5}
    p.set_meta(train_overrides=ov, gen_recipe={"gen_model_id": env.anima["id"]})   # generation: Anima, training: SDXL
    s.gen_recipe(p.id, {"gen_model_id": env.anima["id"], "positive": "x"})
    assert p.meta("model_id") == env.sdxl["id"] and p.meta("train_overrides") == ov


def test_a_generation_only_model_never_becomes_the_training_model(env, monkeypatch):
    s = env.server
    h3 = _h3(env)
    # MiniMax H3-Image can be trained since 2026-11-23 (ai-toolkit): a family marked generation-only stands for it here
    monkeypatch.setitem(env.models.VARIANTS, "h3_image", {**env.models.VARIANTS["h3_image"], "train": False})
    real = env.models.registry.get
    monkeypatch.setattr(env.models.registry, "get", lambda mid: h3 if mid == "h3x" else real(mid))
    p = s.projects.create("キャラ", "character", env.sdxl["id"], "include", "generate")
    with pytest.raises(HTTPException) as e:
        s.update_project(p.id, {"model_id": "h3x"})
    assert e.value.status_code == 400 and "LoRA を作れません" in e.value.detail
    assert p.meta("model_id") == env.sdxl["id"]
    # as the generation model of the material recipe it is refused by the recipe check; the training model stays
    p.set_meta(gen_recipe={"gen_model_id": env.sdxl["id"]})
    s.gen_recipe(p.id, {"gen_model_id": "h3x"})
    assert p.meta("model_id") == env.sdxl["id"]
    # a project that already points at it (older data): the saved-settings list does not fail on it
    p.set_meta(model_id="h3x", train_overrides={"learning_rate": 1e-5})
    assert s._snapshot(p) is None


def test_a_task_is_not_registered_when_training_cannot_start(env, monkeypatch):
    s = env.server
    p = s.projects.create("キャラ", "character", env.sdxl["id"], "include", "generate")
    monkeypatch.setattr(s, "model_train_errors", lambda m: [])
    monkeypatch.setattr(s, "gpus_cached", lambda: [])                  # no GPU: the training at the end cannot run
    monkeypatch.setattr(s.generate, "check_recipe", lambda *a, **k: {"errors": [], "notes": []})
    with pytest.raises(HTTPException) as e:
        s.task_add(p.id, {"gen_count": 24})
    assert e.value.status_code == 400 and "GPU" in e.value.detail and "登録しませんでした" in e.value.detail
    assert s.tasks.list() == []
    # the model itself: e.g. only the fp8 file, which generates but cannot be trained on
    monkeypatch.setattr(s, "gpus_cached", lambda: [{"name": "x", "vram_total_mb": 24000}])
    monkeypatch.setattr(s, "model_train_errors", lambda m: ["学習には bf16 版のモデル本体が必要です"])
    with pytest.raises(HTTPException) as e:
        s.task_add(p.id, {"gen_count": 24})
    assert "bf16" in e.value.detail and s.tasks.list() == []


def test_a_waiting_task_whose_model_is_gone_fails_before_it_generates(env, monkeypatch):
    from lorastudio import pipeline
    made = []
    monkeypatch.setattr(pipeline.hooks, "train_blockers", lambda p: ["「LoRA を作るモデル」が選ばれていません"])
    monkeypatch.setattr(pipeline.hooks, "queue_batch", lambda p, bid: made.append(bid))
    pl = pipeline.Pipeline(SimpleNamespace(get=lambda pid: None))
    t = {"id": "t1", "status": "waiting", "gen_count": 24, "message": "", "project": "p"}
    pl.tasks.append(t)
    pl._waiting(t, SimpleNamespace(id="p", meta=lambda k: {}))
    assert t["status"] == "failed" and "学習を始められない" in t["message"] and made == []


def test_a_finished_run_is_not_resumed_over_its_lora(env, monkeypatch):
    s = env.server
    p = s.projects.create("キャラ", "character", env.sdxl["id"])
    monkeypatch.setattr(s, "_run_or_404", lambda proj, rid: {"status": "done", "config": {}})
    monkeypatch.setattr(s, "_resume_state", lambda *a: (_ for _ in ()).throw(AssertionError("looked for a state")))
    queued = []
    monkeypatch.setattr(s.runner, "enqueue", lambda *a, **k: queued.append(a) or True)
    with pytest.raises(HTTPException) as e:
        s.run_resume(p.id, 1)
    assert e.value.status_code == 400 and queued == []


def _lora(path: Path) -> Path:
    hdr = json.dumps({"lora_unet_x.lora_down.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(hdr)) + hdr + b"\0\0\0\0")
    return path


def test_multi_character_lora_is_called_by_its_characters_words(env, monkeypatch):
    from lorastudio import export, linking, loraconf
    members = [{"name": "アリス", "trigger": "chr_a", "pid": "a"}, {"name": "ボブ", "trigger": "chr_b", "pid": "b"}]

    class P:
        id = "p"

        def run(self, rid):
            return {"config": {"model": {"family": "sdxl", "name": "m", "variant": "illustrious", "files": {}}, "trigger": "multi_ab12",
                               "kind": "character", "engine": "sd-scripts", "engine_commit": "x", "preset_version": 1,
                               "values": {}, "changed": [], "items": [], "members": members}}

        def meta(self, k):
            return "ふたり"

        def checkpoints(self, rid):
            return []

        def tx(self):
            db = sqlite3.connect(":memory:")
            db.execute("CREATE TABLE exports(run_id, created, ckpt_path, dest, info)")
            return contextlib.nullcontext(db)
    monkeypatch.setattr(linking, "link_export", lambda f: None)
    monkeypatch.setattr(export, "comfy_lora_file", lambda fam, s, dest: (dest.write_bytes(Path(s).read_bytes()), dest)[1])
    res = export.export(P(), 1, str(_lora(env.tmp / "src.safetensors")), 0.8, str(env.tmp / "loras"), None)
    info = json.loads((Path(res["dir"]) / "lora_info.json").read_text(encoding="utf-8"))
    assert info["trigger"] is None and [c["trigger"] for c in info["characters"]] == ["chr_a", "chr_b"]
    conf = loraconf.get(res["file"])
    assert conf["triggers"] == ["chr_a", "chr_b"] and conf["made"] and conf["strength"] == 0.8

    # an export made before this fix recorded the gathering project's word: the characters' words are used all the same
    old = env.tmp / "old" / "ふたり_v1"
    _lora(old / "ふたり_v1.safetensors")
    (old / "lora_info.json").write_text(json.dumps({"trigger": "multi_ab12", "characters": members}), encoding="utf-8")
    assert loraconf.get(old / "ふたり_v1.safetensors")["triggers"] == ["chr_a", "chr_b"]


def test_another_lora_in_an_export_folder_does_not_take_its_trigger(env):
    from lorastudio import loraconf
    d = env.tmp / "loras" / "キャラ_v1"
    own = _lora(d / "キャラ_v1.safetensors")
    (d / "lora_info.json").write_text(json.dumps({"trigger": "chr_x", "tested_strength": 0.9}), encoding="utf-8")
    assert loraconf.get(own)["triggers"] == ["chr_x"]
    other = _lora(d / "downloaded style.safetensors")
    got = loraconf.get(other)
    assert got["triggers"] == [] and not got["made"] and got["strength"] is None
    # the exported file renamed by the user, alone in its folder, keeps its word
    solo = env.tmp / "loras" / "キャラ_v2"
    renamed = _lora(solo / "my character.safetensors")
    (solo / "lora_info.json").write_text(json.dumps({"trigger": "chr_y"}), encoding="utf-8")
    assert loraconf.get(renamed)["triggers"] == ["chr_y"]


def test_restart_question_names_a_data_move_and_a_folder_import(env, monkeypatch):
    s = env.server
    assert s._restart_losses() == []
    monkeypatch.setattr(s.relocate, "status", lambda: {"running": True})
    monkeypatch.setitem(s._folder_imports, "n", 1)
    out = s._restart_losses()
    assert any("移動" in x for x in out) and any("取り込み" in x for x in out)


def test_cancelled_task_takes_its_waiting_batch_out_of_the_queue(env, monkeypatch):
    s = env.server
    p = s.projects.create("キャラ", "character", env.sdxl["id"], "include", "generate")
    bid = s.generate.new_batch(p, {"gen_model_id": env.sdxl["id"], "positive": "x"}, 4)
    dropped = []
    monkeypatch.setattr(s.runner, "drop_tasks", lambda pred: dropped.append(pred) or 1)
    s._pipeline_stop_batch(p, bid)
    assert dropped and p.q("SELECT status FROM gen_batches WHERE id=?", (bid,))[0]["status"] == "cancelled"
    assert (p.id, bid) not in s.generate.control.flags


def test_relink_never_leaves_a_half_copied_original(env, monkeypatch):
    s = env.server
    p = s.projects.create("キャラ", "character", env.sdxl["id"])
    data = b"\x89PNG fake image bytes"
    import hashlib
    im = {"id": 1, "sha": hashlib.sha256(data).hexdigest(), "fast": {"file_size": len(data)}}
    dest = env.tmp / "originals" / "a.png"
    dest.parent.mkdir()
    monkeypatch.setattr(p, "images", lambda: [im])
    monkeypatch.setattr(p, "source_path", lambda i: dest)
    monkeypatch.setattr(s, "P", lambda pid: p)
    monkeypatch.setattr(s.analyzer, "submit_fast", lambda *a: None)
    src = env.tmp / "元 フォルダー"
    src.mkdir()
    (src / "a.png").write_bytes(data)

    real, full = s.shutil.copyfile, {"disk": True}

    def copyfile(a, b):
        if not full["disk"]:
            return real(a, b)
        Path(b).write_bytes(data[:5])
        raise OSError("ディスクがいっぱいです")
    monkeypatch.setattr(s.shutil, "copyfile", copyfile)
    with pytest.raises(HTTPException):
        s.relink(p.id, {"path": str(src)})
    assert not dest.exists() and list(dest.parent.iterdir()) == []      # no broken file stands as the original
    full["disk"] = False
    assert s.relink(p.id, {"path": str(src)}) == {"fixed": 1, "still_missing": 0} and dest.read_bytes() == data


def test_copying_a_generation_result_into_a_project_resets_the_old_family_settings(env, monkeypatch):
    s = env.server
    p = s.projects.create("キャラ", "character", env.sdxl["id"], "include", "generate")
    p.set_meta(train_overrides={"learning_rate": 5e-5, "max_train_steps": 900}, gen_recipe={"gen_model_id": env.sdxl["id"]})
    entry = {"settings": {"model_id": env.anima["id"], "loras": [], "positive": "1girl", "negative": "", "sampling_overrides": {}}}
    monkeypatch.setattr(s.imagegen, "get_entry", lambda eid: entry)
    r = s.imagegen_copy("e1", {"project_id": p.id})
    assert p.meta("model_id") == env.anima["id"] and p.meta("train_overrides") == {"max_train_steps": 900}
    assert len(r["reset_settings"]) == 1 and "model_changed" not in r and "previous_model_id" not in r
