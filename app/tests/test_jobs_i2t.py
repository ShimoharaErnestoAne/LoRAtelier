"""画像からプロンプトの処理 (i2t のワーカー) をジョブ一覧に出す: /api/queue の i2t、一時停止の状態がどちらの画面からでも同じになること。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_running_i2t_job_is_shown_and_pause_is_kept(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "data"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import i2tlink, server
    from lorastudio.events import bus
    c = i2tlink.client
    assert server.queue_state()["i2t"] is None
    c.job = {"rid": 7, "op": "analyze", "message": "VLM で分析中", "fraction": 0.4, "est": None, "cancellable": True, "started": 1.0}
    v = server.queue_state()["i2t"]
    assert v["label"] == "画像の分析" and v["fraction"] == 0.4 and v["pausable"] and not v["paused"]
    seen = []
    monkeypatch.setattr(bus, "publish", lambda kind, **kw: seen.append((kind, kw)))
    c.control("pause")                                   # no worker process here: only the state and the event
    assert server.queue_state()["i2t"]["paused"] and seen[-1] == ("i2t", {"event": "paused", "rid": 7, "paused": True})
    c.control("resume")
    assert not server.queue_state()["i2t"]["paused"]
    c.job = {"rid": 8, "op": "pull", "message": "", "fraction": None, "cancellable": False}
    v = server.queue_state()["i2t"]
    assert v["label"] == "説明 AI（VLM）の取得" and not v["pausable"] and not v["cancellable"]
    c.job = None


def test_hugging_face_lora_names_and_the_older_diffusers_layout():
    from lorastudio import models, remote
    # generic diffusers / PEFT names are saved under the repo's name, so two LoRAs do not collide
    assert remote.hf_save_name("Viggle/Qwen-Image-2.1-viggle-turbo", "peft/adapter_model.safetensors") == "Qwen-Image-2.1-viggle-turbo_peft.safetensors"
    assert remote.hf_save_name("lightx2v/Qwen-Image-2.1-Lightning", "pytorch_lora_weights.safetensors") == "Qwen-Image-2.1-Lightning.safetensors"
    assert remote.hf_save_name("nerijs/pixel-art-xl", "pixel-art-xl.safetensors") == "pixel-art-xl.safetensors"
    # lightx2v's LoRA: transformer.transformer_blocks.0.attn.to_k.lora.down.weight (ComfyUI reads this layout)
    assert models.is_lora_keys(["transformer.transformer_blocks.0.attn.to_k.lora.down.weight",
                                "transformer.transformer_blocks.0.attn.to_k.lora.up.weight"])
    assert not models.is_lora_keys(["transformer_blocks.0.attn.to_k.weight"])


def test_a_row_puts_in_only_the_trigger_words_chosen(tmp_path, monkeypatch):
    from lorastudio import loraconf
    f = tmp_path / "multi.safetensors"
    f.write_bytes(b"x")
    monkeypatch.setattr(loraconf, "get", lambda p: {"triggers": ["alice", "bob", "carol"], "auto": True})
    assert loraconf.apply("1girl", [{"path": str(f)}])[1] == ["alice", "bob", "carol"]          # rows from before: all
    assert loraconf.apply("1girl", [{"path": str(f), "words": ["bob", "carol"]}]) == ("bob, carol, 1girl", ["bob", "carol"])
    assert loraconf.apply("1girl", [{"path": str(f), "words": []}]) == ("1girl", [])             # none chosen
    assert loraconf.apply("bob, 1girl", [{"path": str(f), "words": ["bob"]}])[1] == []           # already in the prompt
