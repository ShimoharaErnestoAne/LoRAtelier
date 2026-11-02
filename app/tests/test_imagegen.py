"""Plain image generation page: what is queued, and what is copied into a character project's material
recipe.  A wrong copy silently changes the user's material generation (model, prompt, kept choices)."""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "data dir"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config, imagegen, models, store
    config._settings = None
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    hdr = json.dumps({"conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight":
                      {"dtype": "F16", "shape": [1], "data_offsets": [0, 0]},
                      "model.diffusion_model.input_blocks.0.0.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 0]}}).encode()
    ck = tmp_path / "models" / "checkpoints" / "テスト モデル.safetensors"
    ck.parent.mkdir(parents=True)
    ck.write_bytes(struct.pack("<Q", len(hdr)) + hdr)
    model = models.registry.add("テスト", "illustrious", {"ckpt": str(ck)})

    class Runner:
        jobs = []

        def enqueue_task(self, owner, label, fn, on_cancel=None):
            self.jobs.append((owner.id, label))
            return SimpleNamespace()
    return {"ig": imagegen, "model": model, "runner": Runner(), "pm": store.ProjectManager(config.PROJECTS_DIR)}


def _form(env, **kw):
    return {"model_id": env["model"]["id"], "positive": "1girl, red hair, in a forest", "negative": "lowres",
            "loras": [], "width": 830, "height": 1220, "count": 9, "seed": 42, "sampling": {"steps": 20, "cfg": ""}, **kw}


def test_submit_queues_with_rounded_size_and_only_changed_sampling(env):
    e = env["ig"].submit(_form(env), env["runner"])
    s = e["settings"]
    assert env["runner"].jobs == [("__imagegen__", "画像生成")]
    assert (s["width"], s["height"], s["count"], s["seed"]) == (832, 1216, 9, 42)
    assert env["ig"].submit(_form(env, count=500), env["runner"])["settings"]["count"] == env["ig"].MAX_COUNT
    assert s["sampling_overrides"] == {"steps": 20} and s["sampling"]["steps"] == 20 and s["sampling"]["cfg"]
    assert s["positive"] == "1girl, red hair, in a forest"          # sent as typed: no background removal here
    assert env["ig"].history()[0]["status"] == "queued"


def test_submit_refuses_missing_prompt(env):
    with pytest.raises(ValueError):
        env["ig"].submit(_form(env, positive="  "), env["runner"])
    assert not env["runner"].jobs


def test_copy_keeps_project_choices_and_skips_size_and_seed(env):
    ig, pm = env["ig"], env["pm"]
    e = ig.submit(_form(env), env["runner"])
    p = pm.create("既存", "character", None, "include", "generate")
    p.set_meta(gen_recipe={"positive": "old", "count": 72, "exclude_angles": ["back"], "seed": 7})
    r = ig.copy_to_project(pm, e["id"], p.id, None)
    rec = p.meta("gen_recipe")
    assert rec["positive"] == "1girl, red hair, in a forest" and rec["gen_model_id"] == env["model"]["id"]
    assert rec["count"] == 72 and rec["exclude_angles"] == ["back"] and rec["seed"] == 7   # the project's own choices
    assert "width" not in rec and rec["sampling"] == {"steps": 20}
    assert p.meta("model_id") == env["model"]["id"]                   # the training model follows while unset
    assert any("in a forest" in n for n in r["notes"])                # the user is told the background goes away


def test_copy_does_not_override_a_chosen_training_model(env):
    ig, pm = env["ig"], env["pm"]
    e = ig.submit(_form(env), env["runner"])
    p = pm.create("別モデル", "character", "other-model", "include", "images")
    ig.copy_to_project(pm, e["id"], p.id, None)
    assert p.meta("model_id") == "other-model" and p.meta("source") == "generate"


def test_copy_to_new_project_and_style_refused(env):
    ig, pm = env["ig"], env["pm"]
    e = ig.submit(_form(env), env["runner"])
    r = ig.copy_to_project(pm, e["id"], None, "新キャラ")
    p = pm.get(r["project_id"])
    assert p.meta("name") == "新キャラ" and p.meta("kind") == "character" and p.meta("gen_recipe")["gen_model_id"]
    st = pm.create("画風", "style", None)
    with pytest.raises(ValueError):
        ig.copy_to_project(pm, e["id"], st.id, None)


def test_queued_requests_are_closed_after_restart(env):
    ig = env["ig"]
    ig.submit(_form(env), env["runner"])
    ig.recover()
    assert ig.history()[0]["status"] == "cancelled"


def _lora(path: Path, meta: dict | None = None) -> Path:
    hdr = {"lora_unet_input_blocks_4_1_proj_in.lora_down.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 0]}}
    if meta:
        hdr["__metadata__"] = meta
    raw = json.dumps(hdr).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    return path


def test_trigger_words_are_put_in_front_only_when_switched_on(env, tmp_path):
    from lorastudio import loraconf
    lo = _lora(tmp_path / "models" / "loras" / "chara_v1" / "chara_v1.safetensors")
    (lo.parent / "lora_info.json").write_text(json.dumps({"trigger": "zxq_chara", "tested_strength": 0.7}), encoding="utf-8")
    c = loraconf.get(lo)
    assert c["triggers"] == ["zxq_chara"] and c["auto"] and c["strength"] == 0.7 and not c["saved"]
    s = env["ig"].submit(_form(env, loras=[{"path": str(lo), "strength": 0.7}]), env["runner"])["settings"]
    assert s["prompt_sent"] == "zxq_chara, 1girl, red hair, in a forest" and s["trigger_words"] == ["zxq_chara"]
    assert s["positive"] == "1girl, red hair, in a forest"   # the typed prompt is kept for "この設定に戻す"
    off = env["ig"].submit(_form(env, loras=[{"path": str(lo), "strength": 0.7, "trigger": False}]), env["runner"])["settings"]
    assert off["prompt_sent"] == off["positive"] and off["trigger_words"] == []
    dup = env["ig"].submit(_form(env, positive="ZXQ_chara, smile", loras=[{"path": str(lo), "trigger": True}]), env["runner"])["settings"]
    assert dup["prompt_sent"] == "ZXQ_chara, smile"          # already in the prompt: not added twice
    # the user's own words replace what the file says, and follow the file when it is moved
    loraconf.save(lo, "abc, def ,abc", False, 0.9)
    moved = tmp_path / "models" / "loras" / "other" / lo.name
    moved.parent.mkdir()
    lo.replace(moved)
    c = loraconf.get(moved)   # same name and size in another folder: the same LoRA (no lora_info.json there)
    assert c["triggers"] == ["abc", "def"] and c["auto"] is False and c["strength"] == 0.9 and c["saved"]
    assert loraconf.apply("x", [{"path": str(moved)}]) == ("x", [])               # follows the LoRA's switch (off)
    assert loraconf.apply("x", [{"path": str(moved), "trigger": True}]) == ("abc, def, x", ["abc", "def"])
    moved.replace(lo)
    assert "zxq_chara" in loraconf.get(lo)["candidates"]       # what the file says stays offered
    assert loraconf.reset(lo)["triggers"] == ["zxq_chara"]


def test_caption_frequency_is_only_a_candidate_and_civitai_words_are_used(env, tmp_path):
    from lorastudio import loraconf
    freq = {"10_x": {"qqchar": 20, "1girl": 19, "smile": 5}}
    lo = _lora(tmp_path / "models" / "loras" / "a.safetensors", {"ss_tag_frequency": json.dumps(freq)})
    c = loraconf.get(lo)
    assert c["triggers"] == [] and c["candidates"] == ["qqchar", "1girl"]
    assert loraconf.apply("x", [{"path": str(lo), "trigger": True}]) == ("x", [])
    loraconf.record_found(lo, ["civ word", "civ word"], "Civitai に書かれた呼び出し語")
    c = loraconf.get(lo)
    assert c["triggers"] == ["civ word"] and c["source"].startswith("Civitai")


def test_civitai_lookup_sends_only_the_hash_and_records_words(env, tmp_path, monkeypatch):
    import hashlib
    from lorastudio import loraconf, remote
    lo = _lora(tmp_path / "models" / "loras" / "dl.safetensors")
    sent = []

    def fake(url, headers=None):
        sent.append(url)
        if url.endswith(hashlib.sha256(lo.read_bytes()).hexdigest()):
            return {"id": 7, "modelId": 3, "name": "v1", "model": {"name": "Some LoRA"}, "trainedWords": ["sk_char, red cape"]}
        raise ValueError("404")
    monkeypatch.setattr(remote, "_json", fake)
    r = loraconf.civitai_lookup(lo)
    assert sent == [f"https://civitai.com/api/v1/model-versions/by-hash/{hashlib.sha256(lo.read_bytes()).hexdigest()}"]
    assert r["found"] and r["conf"]["triggers"] == ["sk_char", "red cape"] and r["conf"]["page"].endswith("modelVersionId=7")
    other = _lora(tmp_path / "models" / "loras" / "x.safetensors", {"k": "v"})
    assert loraconf.civitai_lookup(other) == {"found": False, "conf": loraconf.get(other)}


def _fake_llm(monkeypatch, edit, write):
    """Ollama stand-in: step 1 (理解と編集) answers `edit`, step 2 (書き戻し) answers `write`."""
    from lorastudio import promptedit
    sent = []

    def fake(host, path, body=None, timeout=10):
        if path == "/api/tags":
            return {"models": [{"name": "big:27b", "size": 17e9, "details": {"family": "qwen35"}},
                               {"name": "vl:8b", "size": 6e9, "details": {"family": "qwen3vl"}},
                               {"name": "old:7b", "size": 5e9, "details": {"family": "qwen2"}},
                               {"name": "new:9b", "size": 8e9, "details": {"family": "qwen35"}}]}
        if "think" in body:   # the first try of each step: retried without the switch
            raise promptedit.LlmError('"think" is not supported by this model')
        sent.append(body)
        step1 = "変更後の絵" in body["format"]["properties"]
        return {"message": {"content": json.dumps(edit if step1 else write, ensure_ascii=False)}}
    monkeypatch.setattr(promptedit, "_req", fake)
    return promptedit, sent


EDIT = {"今の絵": "黒髪ロングの三つ編みの女の子", "変更後の絵": "銀髪ショートの女の子（三つ編みはない）",
        "出てほしくないもの": "低解像度、眼鏡", "変えたこと": "髪を銀髪のショートにしました。"}


def test_prompt_is_edited_in_japanese_then_written_back(env, monkeypatch):
    """Step 1 gets the prompt and the instruction and answers in Japanese; step 2 gets the Japanese
    descriptions (and the original prompt to reuse) and writes the prompt."""
    promptedit, sent = _fake_llm(monkeypatch, EDIT, {"プロンプト": "zxq, 1girl, short hair, silver hair", "ネガティブ": "lowres, glasses"})
    assert promptedit.models()["model"] == "new:9b"   # best family that fits; not too big, not qwen3-vl
    r = promptedit.rewrite("zxq, 1girl, long hair, black hair, twin braids", "lowres", "髪を銀色のショートに", "tags", ["zxq"])
    assert (r["positive"], r["negative"]) == ("zxq, 1girl, short hair, silver hair", "lowres, glasses")
    assert r["before"] == EDIT["今の絵"] and r["after"] == EDIT["変更後の絵"] and r["changes"] == [EDIT["変えたこと"]]
    assert r["lost_words"] == [] and r["model"] == "new:9b" and not r.get("unchanged")
    step1, step2 = (json.loads(b["messages"][-1]["content"]) for b in sent[-2:])
    assert step1 == {"プロンプト": "zxq, 1girl, long hair, black hair, twin braids", "指示": "髪を銀色のショートに",
                     "ネガティブプロンプト": "lowres"}
    assert step2["新しい絵"] == EDIT["変更後の絵"] and step2["元のプロンプト"].startswith("zxq") and step2["出てほしくないもの"]
    assert "zxq" in sent[-2]["messages"][0]["content"]   # the trigger word is named to the model
    assert "think" not in sent[-1]
    assert promptedit.get_conf()["model"] == ""        # using a model does not make it the user's choice
    promptedit.set_conf(model="old:7b")
    assert promptedit.models()["model"] == "old:7b"    # picked on the page: kept


def test_no_negative_prompt_at_cfg_1_and_the_conversation_is_sent(env, monkeypatch):
    promptedit, sent = _fake_llm(monkeypatch, EDIT, {"プロンプト": "1girl, smile", "ネガティブ": "hat, text"})
    hist = [{"instruction": "帽子をかぶせて", "changes": ["帽子をかぶせました。"]}, {"instruction": " ", "changes": []}]
    r = promptedit.rewrite("1girl, hat", "lowres", "帽子はなし。前のは違う", "tags", [], negative_used=False, history=hist)
    assert r["positive"] == "1girl, smile" and r["negative"] == "lowres"   # whatever step 2 answered
    step1 = json.loads(sent[-2]["messages"][-1]["content"])
    assert "ネガティブプロンプト" not in step1 and step1["これまでの指示"] == [{"指示": "帽子をかぶせて", "直し方": ["帽子をかぶせました。"]}]
    assert "ネガティブプロンプトを使いません" in sent[-2]["messages"][0]["content"] and "これまでの指示" in sent[-2]["messages"][0]["content"]
    assert json.loads(sent[-1]["messages"][-1]["content"])["出てほしくないもの"] == ""
    # nothing changed: said so; sentences use the sentence writer
    promptedit, sent = _fake_llm(monkeypatch, EDIT, {"プロンプト": "A girl.", "ネガティブ": ""})
    r = promptedit.rewrite("A girl.", "", "そのままで", "natural", [])
    assert r["unchanged"] and "何も変えませんでした" in r["note"] and "英語の文章" in sent[-1]["messages"][0]["content"]
    with pytest.raises(ValueError):
        promptedit.rewrite("x", "", "  ", "tags", [])


def test_negative_prompt_counts_only_above_cfg_1(env):
    from lorastudio import models, server
    krea = {"family": "krea2", "variant": "krea2", "files": {}}
    sdxl = {"family": "sdxl", "variant": "illustrious", "files": {}}
    assert not server.negative_used(krea, {}) and server.negative_used(sdxl, {})
    assert not server.negative_used(sdxl, {"cfg": 1}) and server.negative_used(krea, {"cfg": "3.5"})


def test_progress_socket_reads_comfyui_progress_frames(env):
    """The stdlib websocket client against a fake ComfyUI /ws: handshake, a binary preview frame (skipped),
    a fragmented text frame, other prompts' progress (ignored), then close."""
    import socket
    import threading
    import time
    from lorastudio.evaluate import _ProgressSocket
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def frame(op, data, fin=True):
        n = len(data)
        head = bytes([(0x80 if fin else 0) | op]) + (bytes([n]) if n < 126 else bytes([126]) + struct.pack(">H", n))
        return head + data

    def serve():
        c, _ = srv.accept()
        req = b""
        while b"\r\n\r\n" not in req:
            req += c.recv(4096)
        assert b"GET /ws?clientId=cid " in req
        c.sendall(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n")
        time.sleep(0.3)   # the prompt id is known once /prompt has answered
        msg = lambda pid, v: json.dumps({"type": "progress", "data": {"value": v, "max": 20, "prompt_id": pid}}).encode()
        c.sendall(frame(2, b"\x00\x00\x00\x01PNG..."))
        c.sendall(frame(1, msg("other", 9)))
        whole = msg("p1", 5) + b" " * 200   # > 125 bytes: 16-bit length
        c.sendall(frame(1, whole[:50], fin=False) + frame(0, whole[50:]))
        c.sendall(frame(1, msg("p1", 20)))
        c.sendall(frame(8, b""))
        c.close()
    threading.Thread(target=serve, daemon=True).start()
    got = []
    ws = _ProgressSocket("127.0.0.1", port, "cid", lambda step, steps: got.append((step, steps)))
    ws.prompt_id = "p1"
    for _ in range(50):
        if len(got) == 2:
            break
        time.sleep(0.05)
    ws.close()
    srv.close()
    assert got == [(5, 20), (20, 20)]


def test_remaining_time_is_only_from_measured_speed(env):
    eta = env["ig"]._eta
    assert eta([], 3, None) is None                       # nothing measured yet: shown as 計測中
    assert eta([], 0, 4.0) == 4                           # last image: the rest of its steps
    assert eta([], 2, 4.0, per_image=10.0) == 24          # before the first image finishes: its sampling time
    assert eta([12.0, 14.0], 2, 3.0, per_image=10.0) == 29  # finished images' average wins


def test_blur_setting_is_off_by_default_and_kept(env):
    """「画像をぼかして表示」: off until the user turns it on, and remembered after a restart."""
    from lorastudio import config, server
    assert server.ui_conf({})["blur_images"] is False
    assert server.ui_conf({"blur_images": True})["blur_images"] is True
    config._settings = None
    assert server.ui_conf({})["blur_images"] is True


def test_project_list_has_created_and_updated_for_sorting(env):
    """プロジェクト一覧の並べ替え: 作成 comes from the project, 更新 follows edits saved to project.db."""
    import os
    import time
    pm = env["pm"]
    a = pm.create("先に作った", "character", None)
    b = pm.create("後に作った", "style", None)
    a.set_meta(created=1000.0)
    b.set_meta(created=2000.0)
    later = time.time() + 100
    os.utime(a.path / "project.db", (later, later))      # the older project was edited last
    rows = {r["name"]: r for r in pm.list()}
    assert rows["先に作った"]["created"] == 1000.0 and rows["後に作った"]["created"] == 2000.0
    assert rows["先に作った"]["updated"] >= later > rows["後に作った"]["updated"]
