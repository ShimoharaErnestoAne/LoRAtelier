"""Comparison on other models (merged / fine-tuned ones of the training model's family): the chosen models are kept
in the settings and used after every training, each one gets "no LoRA" and LoRA images with its own quality words,
the recommendation stays on the training model, and how strongly the LoRA works on each model is summarized."""
from __future__ import annotations

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
    from lorastudio import config, evaluate, models, runcompare
    config._settings = None

    def add(name, variant):
        files = {}
        for key, _label, required in models.FAMILY_FILES[models.VARIANTS[variant]["family"]]:
            if required or key == "ckpt":
                f = tmp_path / "モデル" / f"{name}_{key}.safetensors"
                f.parent.mkdir(exist_ok=True)
                f.write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{}      ")
                files[key] = str(f)
        return models.registry.add(name, variant, files)
    return {"tmp": tmp_path, "config": config, "evaluate": evaluate, "runcompare": runcompare, "models": models, "add": add}


class FakeProject:
    id = "p1"

    def __init__(self):
        self.saved = None

    def checkpoints(self, _rid):
        return []

    def q(self, sql, params=()):
        if sql.startswith("UPDATE evals SET status='done'"):
            self.saved = params[1]
        return []


def _run(model):
    return {"id": 1, "run_dir": "", "config": {"kind": "character", "outfit_mode": "include", "trigger": "chr_a",
                                               "items": [{"id": 1, "caption": "chr_a, 1girl, solo"}], "model": model}}


def test_chosen_models_of_the_same_family_go_into_every_comparison(env):
    ev, add = env["evaluate"], env["add"]
    main, merge, pony = add("base", "illustrious"), add("merge", "illustrious"), add("ポニー", "pony")
    other = add("anima", "anima")
    env["config"].settings().set(ev.EXTRA_MODELS_KEY, [merge["id"], other["id"], main["id"], "gone", pony["id"]])
    spec = ev.default_spec(FakeProject(), _run(main))
    # another family, the training model itself and unknown ids are left out; on them only the final version
    assert spec["extra_models"] == [merge["id"], pony["id"]] and spec["extra_checkpoints"] == ["final"]
    env["config"].settings().set(ev.EXTRA_MODELS_KEY, [])
    assert "extra_models" not in ev.default_spec(FakeProject(), _run(main))
    # the prompt gets the other model's quality words
    p = ev.build_prompts("character", "include", "illustrious", "chr_a", "1girl")[0]
    assert ev.prompt_for_model(p, "illustrious", "pony").startswith("score_9, ") and "masterpiece" not in ev.prompt_for_model(p, "illustrious", "pony")


def test_images_on_each_model_and_how_strongly_the_lora_works_there(env, monkeypatch):
    ev, add, cfg = env["evaluate"], env["add"], env["config"]
    main, pony = add("base", "illustrious"), add("ポニー", "pony")
    graphs = []

    def run_prompt(g):
        graphs.append(g)
        out = cfg.DATA_DIR / "comfy_output"
        out.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(out / f"{len(graphs)}.png")
        return [{"filename": f"{len(graphs)}.png"}]
    for name, fn in {"ensure_started": lambda: None, "run_prompt": run_prompt, "log_size": lambda: 0,
                     "lora_warnings": lambda _n: [], "park": lambda *a: None}.items():
        monkeypatch.setattr(ev.comfy, name, fn)

    def scores(_p, _r, _o, results, _w, _pr):
        # the LoRA brings the character much closer on the training model, only a little on the other model
        return {"ccip_diff": [0.30 if r["ckpt"] == "base" else 0.25 if r.get("model") else 0.10 for r in results]}
    monkeypatch.setattr(ev, "score_eval", scores)
    spec = {"checkpoints": [500, "final"], "strengths": [1.0], "seeds": [1], "include_base": True,
            "prompts": ev.build_prompts("character", "include", "illustrious", "chr_a", "1girl")[:2], "width": 64, "height": 64,
            "extra_models": [pony["id"]], "extra_checkpoints": ["final"]}
    out = env["tmp"] / "eval"
    out.mkdir()
    proj = FakeProject()
    info = ev._run_eval_images(proj, 1, 1, None, _run(main), _run(main)["config"], spec, main, "sdxl", out,
                               {"ckpt": "base.safetensors"}, {"500": "a500.safetensors", "final": "a.safetensors"},
                               type("W", (), {"unload": lambda self: None})(), [(pony, {"ckpt": "pony.safetensors"})], [])
    imgs = info["images"]
    on_pony = [r for r in imgs if r.get("model") == pony["id"]]
    assert len(imgs) == 2 * 3 + 2 * 2 and {r["ckpt"] for r in on_pony} == {"base", "final"}
    assert len({r["file"] for r in imgs}) == len(imgs) and all((out / r["file"]).exists() for r in imgs)
    pony_graphs = [g for g in graphs if g["1"]["inputs"]["ckpt_name"] == "pony.safetensors"]
    assert len(pony_graphs) == 4 and all(g["3"]["inputs"]["text"].startswith("score_9") for g in pony_graphs)
    assert all("score_4" in g["4"]["inputs"]["text"] for g in pony_graphs)   # the other model's negative too
    assert all(g["1c"]["inputs"]["stop_at_clip_layer"] == -2 for g in pony_graphs)   # Pony's sampling
    # the recommendation is made on the training model only
    assert {r["ckpt"] for r in info["recommend"]["table"]} == {"base", "500", "final"}
    o = info["other_models"][0]
    assert o["name"] == "ポニー" and o["ratio"] == 0.25 and any("弱く出ている" in x for x in o["reasons"])
    assert info["models"][0]["id"] == pony["id"] and proj.saved
    # comparing runs with each other uses the training model's images only
    ev_row = {"id": 1, "status": "done", "spec": spec, "results": info}
    assert len(env["runcompare"].run_images(_run(main)["config"], [ev_row])) == 6


def test_settings_api_keeps_the_choice(env):
    from fastapi.testclient import TestClient
    from lorastudio import server
    add = env["add"]
    a, b, c, d = add("a", "illustrious"), add("b", "illustrious"), add("c", "illustrious"), add("d", "illustrious")
    cl = TestClient(server.app)
    assert cl.put("/api/eval_models", json={"ids": ["nope"]}).status_code == 400
    assert cl.put("/api/eval_models", json={"ids": [a["id"], b["id"], c["id"], d["id"]]}).status_code == 400
    assert cl.put("/api/eval_models", json={"ids": [a["id"], b["id"]]}).json()["ids"] == [a["id"], b["id"]]
    assert cl.get("/api/eval_models").json()["ids"] == [a["id"], b["id"]]


def test_same_files_under_another_name_are_left_out(env):
    ev, models = env["evaluate"], env["models"]
    main = env["add"]("base", "illustrious")
    twin = models.registry.add("同じファイル", "pony", dict(main["files"]))
    env["config"].settings().set(ev.EXTRA_MODELS_KEY, [twin["id"]])
    assert ev.extra_models_for(main) == []


def test_from_the_api_to_the_export(env, monkeypatch):
    """「比較を追加」 with another model: queued, generated on both models, scored, shown and written to the export."""
    import hashlib
    import json

    import numpy as np
    from fastapi.testclient import TestClient
    from safetensors.numpy import save_file
    from lorastudio import linking, server, store, training
    ev, cfg, tmp = env["evaluate"], env["config"], env["tmp"]
    main, merge = env["add"]("base", "illustrious"), env["add"]("マージ", "illustrious")
    pm = server.projects
    p = pm.create("キャラ", "character", main["id"])
    for i in range(3):
        f = tmp / "素材" / f"{i}.png"
        f.parent.mkdir(exist_ok=True)
        Image.fromarray((np.random.default_rng(i).random((32, 32, 3)) * 255).astype("uint8")).save(f)
        iid, _ = p.add_image_file(f, f.name, hashlib.sha256(f.read_bytes()).hexdigest())
        p.set_auto_caption(iid, f"{p.meta('trigger')}, solo", "k")
    values = {"max_train_steps": 100, "save_every_n_steps": 50, "seed": 1, "resolution": 1024, "learning_rate": 1e-4,
              "network_dim": 4, "network_alpha": 2, "train_batch_size": 1, "optimizer_type": "AdamW8bit",
              "lr_scheduler": "constant", "caption_dropout_rate": 0.0}
    rid = training.create_run(p, p.images(), values, [], None)
    rd = Path(p.run(rid)["run_dir"])
    lora = rd / "a.safetensors"
    save_file({"lora_unet_x.lora_down.weight": np.zeros((4, 4), np.float32)}, str(lora))
    p.q("INSERT INTO checkpoints(run_id, step, path, kind, created) VALUES(?,?,?,?,?)", (rid, None, str(lora), "lora", 0.0))
    p.q("UPDATE runs SET status='done' WHERE id=?", (rid,))

    graphs = []

    def run_prompt(g):
        graphs.append(g)
        out = cfg.DATA_DIR / "comfy_output"
        out.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(out / f"{len(graphs)}.png")
        return [{"filename": f"{len(graphs)}.png"}]
    for name, fn in {"available": lambda: True, "ensure_started": lambda: None, "run_prompt": run_prompt,
                     "log_size": lambda: 0, "lora_warnings": lambda _n: [], "park": lambda *a: None, "stop": lambda: None}.items():
        monkeypatch.setattr(ev.comfy, name, fn)
    monkeypatch.setattr(ev, "score_eval", lambda _p, _r, _o, results, _w, _pr: {
        "ccip_diff": [0.3 if r["ckpt"] == "base" else 0.2 if r.get("model") else 0.1 for r in results]})
    monkeypatch.setattr(linking, "link_export", lambda _f: None)   # never into a real ComfyUI folder
    monkeypatch.setattr(server.runner, "enqueue_task", lambda _p, _l, fn, on_cancel=None: fn())   # run at once

    cl = TestClient(server.app)
    r = cl.post(f"/api/projects/{p.id}/runs/{rid}/eval", json={
        "checkpoints": ["final"], "strengths": [0.8, 1.0], "seeds": [1], "include_base": True,
        "prompt_sets": ["standard"], "extra_models": [merge["id"]]})
    assert r.status_code == 200 and r.json()["n_images"] == 2 * 3 * 4   # 2 models x (no LoRA + 2 strengths) x 4 prompts
    ev_row = cl.get(f"/api/projects/{p.id}/runs/{rid}/evals").json()[0]
    assert ev_row["status"] == "done", ev_row.get("error")
    res = ev_row["results"]
    assert len(res["images"]) == 24 and sum(1 for x in res["images"] if x.get("model") == merge["id"]) == 12
    assert {g["1"]["inputs"]["ckpt_name"].split("\\")[-1].split("/")[-1] for g in graphs} == {"base_ckpt.safetensors", "マージ_ckpt.safetensors"}
    assert res["recommend"]["ckpt"] == "final" and res["other_models"][0]["ratio"] == 0.5
    f = res["images"][-1]["file"]
    assert cl.get(f"/api/projects/{p.id}/runs/{rid}/evals/{ev_row['id']}/{f}").status_code == 200
    out = cl.post(f"/api/projects/{p.id}/runs/{rid}/export", json={"ckpt_path": str(lora), "strength": 1.0,
                                                                   "dest": str(tmp / "書き出し")})
    assert out.status_code == 200, out.text
    info = json.loads((Path(out.json()["dir"]) / "lora_info.json").read_text(encoding="utf-8"))
    assert info["evaluation"]["other_models"][0]["name"] == "マージ" and info["evaluation"]["other_models"][0]["ratio"] == 0.5
