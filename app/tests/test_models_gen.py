"""Model files placed in folders and prompt handling for material generation: a wrong guess here either
hides a usable model, picks the wrong engine, or puts a background into material that must not have one."""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "data dir"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config, models
    config._settings = None
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "モデル 置き場")
    return {"tmp": tmp_path, "models": models, "config": config}


def _st(path: Path, tensors: dict):
    hdr = {k: {"dtype": dt, "shape": [1], "data_offsets": [0, 0]} for k, dt in tensors.items()}
    raw = json.dumps(hdr).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    return path


def _gguf(path: Path, kv: dict, tensors: list[str]):
    """Header-only GGUF v3 with string values, one array value and the given tensor names."""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv) + 1)
    out += s("general.quantization_version_list") + struct.pack("<IIQ", 9, 4, 3) + struct.pack("<III", 1, 2, 3)
    for k, v in kv.items():
        out += s(k) + struct.pack("<I", 8) + s(v)
    for t in tensors:
        out += s(t) + struct.pack("<I", 2) + struct.pack("<QQ", 4, 4) + struct.pack("<I", 12) + struct.pack("<Q", 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(out)
    return path


QWEN = {"txt_norm.weight": "BF16", "img_in.weight": "BF16", "transformer_blocks.0.attn.to_q.weight": "BF16"}
QWEN21 = ["txt_in.text_norm.weight", "modulation.1.weight", "img_in.weight", "transformer_blocks.0.img_mlp.gate_up.weight"]


def test_qwen_detection_variants(env, tmp_path):
    m = env["models"]
    assert m.detect(_st(tmp_path / "a.safetensors", QWEN))["family"] == "qwen_image"
    fp8 = m.detect(_st(tmp_path / "b.safetensors", {k: "F8_E4M3" for k in QWEN}))
    assert fp8["family"] == "qwen_image" and fp8["fp8"]
    # the edit model has the same blocks plus the 2511 marker: not offered for text-to-image
    assert m.detect(_st(tmp_path / "c.safetensors", {**QWEN, "__index_timestep_zero__": "BF16"}))["family"] is None
    # ComfyUI-prefixed keys
    pre = {"model.diffusion_model." + k: v for k, v in QWEN.items()}
    assert m.detect(_st(tmp_path / "d.safetensors", pre))["family"] == "qwen_image"
    assert m.detect(_st(tmp_path / "e.safetensors", {k: "BF16" for k in QWEN21}))["family"] == "qwen_image21"


def test_gguf_family_from_tensors_not_arch(env, tmp_path):
    """Qwen-Image-2.1 Turbo GGUF files say "qwen_image" (or nothing) in general.architecture."""
    m = env["models"]
    f = _gguf(tmp_path / "t.gguf", {"general.architecture": "qwen_image"}, QWEN21)
    assert m.detect_gguf(f)["family"] == "qwen_image21"
    f2 = _gguf(tmp_path / "u.gguf", {}, ["txt_norm.weight", "img_in.weight"])
    assert m.detect_gguf(f2)["family"] == "qwen_image"


def test_dit_in_folder_registered_only_with_companions(env):
    m, cfg = env["models"], env["config"]
    md = cfg.MODELS_DIR
    _st(md / "diffusion_models" / "qwen_image_2512_bf16.safetensors", QWEN)
    scan = m.library.scan()
    c = scan["checkpoints"][0]
    assert c["family"] == "qwen_image" and "見つからない" in c["evidence"]
    assert not m.registry.list()
    _st(md / "text_encoders" / "qwen_2.5_vl_7b.safetensors", {"x": "BF16"})
    _st(md / "vae" / "qwen_image_vae.safetensors", {"y": "BF16"})
    m.library.scan()
    reg = m.registry.list()
    assert len(reg) == 1 and reg[0]["family"] == "qwen_image"
    assert set(reg[0]["files"]) == {"ckpt", "te", "vae", "train_te"} and not reg[0]["missing"]   # the bf16 encoder serves both
    m.library.scan()   # no duplicate on rescan
    assert len(m.registry.list()) == 1


def test_qwen21_gguf_and_bf16_become_one_model(env):
    m, cfg = env["models"], env["config"]
    md = cfg.MODELS_DIR
    _gguf(md / "diffusion_models" / "qwen-image-2.1-turbo-Q6_K.gguf", {"general.architecture": "qwen_image"}, QWEN21)
    _st(md / "diffusion_models" / "qwen_image_2.1_bf16.safetensors", {k: "BF16" for k in QWEN21})
    _st(md / "text_encoders" / "qwen3vl_8b_bf16.safetensors", {"x": "BF16"})
    _st(md / "vae" / "qwen_image_2.1_vae_bf16.safetensors", {"y": "BF16"})
    _st(md / "vae" / "qwen_image_2.1_vae_diffusers_bf16.safetensors", {"z": "BF16"})
    m.library.scan()
    reg = m.registry.list()
    assert len(reg) == 1
    f = reg[0]["files"]
    assert f["ckpt"].endswith("qwen_image_2.1_bf16.safetensors") and f["gen"].endswith(".gguf")
    assert f["train_vae"].endswith("diffusers_bf16.safetensors") and f["train_te"].endswith("qwen3vl_8b_bf16.safetensors")


def test_sdxl_files_in_checkpoints_each_selectable(env):
    m, cfg = env["models"], env["config"]
    sdxl = {"conditioner.embedders.1.model.x": "F16", "model.diffusion_model.y": "F16"}
    _st(cfg.MODELS_DIR / "checkpoints" / "waiIllustriousSDXL_v170.safetensors", sdxl)
    _st(cfg.MODELS_DIR / "checkpoints" / "sub" / "myMerge.safetensors", sdxl)
    m.library.scan()
    names = sorted(x["name"] for x in m.registry.list())
    assert names == ["myMerge", "waiIllustriousSDXL_v170"]


def test_comfyui_folder_is_used_in_place_with_its_extra_paths(env):
    """Pointing at a ComfyUI installation reads its models folder and the folders its extra_model_paths.yaml
    adds; text encoders kept in a subfolder are found; a quantized Krea 2 file is registered for generation."""
    m, tmp = env["models"], env["tmp"]
    inst = tmp / "Comfy UI" / "ComfyUI"
    (inst / "models").mkdir(parents=True)
    (inst / "main.py").write_text("")
    extra = tmp / "別ドライブ models"
    (inst / "extra_model_paths.yaml").write_text(
        f"ssd:\n  base_path: '{extra}\\'\n  text_encoders: text_encoders\n  vae: vae\n"
        f"audio:\n  base_path: '{tmp / 'audio'}'\n  mmaudio: mmaudio\n", encoding="utf-8")
    (tmp / "audio").mkdir()
    _st(inst / "models" / "diffusion_models" / "Krea2 test int8.safetensors",
        {"txtfusion.layerwise_blocks.0.w": "I8", "blocks.0.attn.wq.weight": "BF16"})
    _st(extra / "text_encoders" / "sub" / "qwen3vl_4b_fp8_scaled.safetensors", {"x": "F8_E4M3"})
    _st(extra / "vae" / "qwen_image_vae.safetensors", {"x": "BF16"})
    added = m.library.add_root(str(tmp / "Comfy UI"))
    assert added == [str(inst / "models"), str(extra)]          # the audio-only folder is not read
    m.library.scan()
    k = [x for x in m.registry.list() if x["family"] == "krea2"]
    assert len(k) == 1 and "turbo" in k[0]["files"] and "ckpt" not in k[0]["files"]
    assert k[0]["files"]["te"].endswith("qwen3vl_4b_fp8_scaled.safetensors")


def test_background_words_removed_but_not_lookalikes(env):
    from lorastudio import generate as g
    p, removed = g.strip_background(
        "score_9, 1girl, city street, outdoors, (scenery:1.2), night background, sky blue hair, street clothes, school uniform", "tags")
    assert p == "score_9, 1girl, sky blue hair, street clothes, school uniform"
    assert set(removed) == {"city street", "outdoors", "(scenery:1.2)", "night background"}
    p2, removed2 = g.strip_background("A girl with silver hair, standing in a busy city street. Blurry background.", "natural")
    assert "street" not in p2 and "silver hair" in p2 and len(removed2) == 2


def test_planned_shots_have_plain_backgrounds(env):
    from lorastudio import generate as g
    shots = g.plan_shots(72)
    assert {s["bg"] for s in shots} == {"simple"}   # white only
    for s in shots[:24]:
        tags = g.compose_prompt("1girl, forest, grey background, red eyes", s, "tags")
        assert "forest" not in tags and "grey background" not in tags and "white background" in tags
        assert "white background" in g.compose_prompt("A girl in a forest.", s, "natural")


def test_shot_set_covers_expressions_early(env):
    """The first 24 images (the smallest count) already show every framing and joy / anger / sorrow / fun,
    so a short run still teaches the character with more than one face."""
    from lorastudio import generate as g
    first = g.plan_shots(24)
    assert {s["framing"] for s in first} == set(g.FRAMING_LABEL)
    assert {"smile", "laugh", "angry", "sad", "surprised", "calm"} <= {s["expr"] for s in first}
    assert {s["expr"] for s in g.plan_shots(48)} == set(g.EXPR_LABEL)
    assert all(s["expr"] != "angry" for s in g.plan_shots(48, exclude_exprs=["angry"]))
    for s in g.plan_shots(48):   # every planned shot has words for both prompt styles and a label
        g.shot_tags(s), g.shot_sentence(s)
        assert g.shot_label(s).count("・") == (3 if s.get("pose") else 2)


def test_shot_words_in_prompt_do_not_override_the_shot(env):
    """A 'smile, full body' left in the character prompt would turn every angry face into a smile and every
    close-up into a full body: those words are removed for tag models, the shot's opposite goes negative."""
    from lorastudio import generate as g
    shot = {"framing": "face", "angle": "front", "expr": "angry", "bg": "simple"}
    p = g.compose_prompt("score_9, 1girl, (smile:1.2), full body, red eyes, smiley face hairclip", shot, "tags")
    assert "smile" in g.strip_shot_words("1girl, smile", "tags")[1]
    assert "(smile:1.2)" not in p and "full body" not in p and "smiley face hairclip" in p and "angry" in p
    assert p.startswith("score_9, portrait")
    neg = g.compose_negative("lowres, smile", shot, "tags", "illustrious")
    assert neg.startswith("lowres, smile, worst quality") and "happy" in neg and "bad hands" in neg and "nsfw" not in neg
    assert g.compose_negative("lowres", shot, "natural", "qwen_image21") == "lowres"   # cfg 1 model: nothing added
    # an old batch (char_std_v2 shots) still composes and is labelled
    old = {"framing": "full", "angle": "side", "expr": "closed", "bg": "grey", "pose": "walking"}
    assert "closed mouth" in g.compose_prompt("1girl", old, "tags") and g.shot_label(old) == "全身・横・口を閉じる・歩く"


def test_minimal_guard_against_broken_images(env):
    """Quality words of the model are added only when the user wrote none; one character per image."""
    from lorastudio import generate as g
    shot = {"framing": "full", "angle": "back_full", "expr": "calm", "bg": "grey", "pose": "standing"}
    p = g.compose_prompt("1girl, silver hair", shot, "tags", "illustrious")
    # framing, then the user's tags, then the rest of the shot (the shot in front changed the
    # character's hair / outfit / age / gender / body)
    assert p.startswith("masterpiece, best quality, full body, 1girl, silver hair, solo, from behind, facing away"), p
    assert g.compose_prompt("best quality, 1girl, solo", shot, "tags", "illustrious").count("quality") == 1
    pony = g.compose_prompt("1girl", shot, "tags", "pony")
    assert pony.startswith("score_9, score_8_up, score_7_up, full body") and "rating_safe" not in pony
    neg = g.compose_negative("", shot, "tags", "illustrious")
    assert "looking back" in neg and "multiple views" in neg and "bad feet" in neg


def test_new_angles_early_and_backs_left_out_of_face_check(env):
    from lorastudio import generate as g
    first = g.plan_shots(24)
    assert {"back_full", "overhead", "worm"} <= {s["angle"] for s in first}
    rows = [{"shot": '{"angle": "back_full"}'}, {"shot": '{"angle": "front"}'}, {"shot": None}]
    assert len(g._with_face(rows)) == 2


def test_style_material_varies_subjects_and_keeps_backgrounds(env):
    """Style material: the subject changes every image (people, animals, scenery, objects) so the LoRA learns
    the drawing, not one subject; the user's subject words would put the same figure everywhere."""
    from lorastudio import generate as g
    shots = g.plan_style(24)
    groups = [g._STYLE[s["subject"]][0] for s in shots]
    assert set(groups[:4]) == {"person", "animal", "scenery", "object"}      # mixed from the first images
    assert len({s["subject"] for s in shots}) == 24
    assert {g._STYLE[s["subject"]][0] for s in g.plan_style(24, ["person"])} == {"animal", "scenery", "object"}
    city = next(s for s in shots if s["subject"] == "city_night")
    p = g.compose_style_prompt("1girl, solo, watercolor, soft colors", city, "tags", "sdxl_illustrious")
    assert "1girl" not in p and "watercolor" in p and "cityscape" in p and "night" in p   # background kept
    assert "Night" in g.compose_style_prompt("ink wash painting", city, "natural") or "night" in g.compose_style_prompt("ink wash painting", city, "natural")
    assert "bad hands" not in g.compose_style_negative("", city, "tags", "sdxl_illustrious")
    person = next(s for s in shots if g._STYLE[s["subject"]][0] == "person")
    assert "bad hands" in g.compose_style_negative("", person, "tags", "sdxl_illustrious")
    assert g.check_recipe({"positive": "", "loras": []}, "style")["errors"][-1].startswith("画風")
    assert "1girl" in " ".join(g.check_recipe({"positive": "1girl, watercolor"}, "style")["notes"])
    assert g.shot_label(city) == "風景・室内・夜の街"


def test_old_anima_folder_move_keeps_model(env):
    """The 2026-10-08 version saved Anima (DiT, text encoder, VAE) in models/anima.  The current version tells
    which file to move from where to where; after the move the registered model keeps its id and finds the
    files again, and no second model is added for the same file."""
    m, md = env["models"], env["config"].MODELS_DIR
    old = md / "anima"
    dit = _st(old / "anima-base-v1.0.safetensors", {"net.llm_adapter.blocks.0.w": "BF16"})
    te = _st(old / "qwen_3_06b_base.safetensors", {"model.layers.0.w": "BF16"})
    vae = _st(old / "qwen_image_vae.safetensors", {"decoder.w": "BF16"})
    reg = m.registry.add("Anima base v1.0", "anima", {"ckpt": str(dit), "te": str(te), "vae": str(vae)})

    moves = {x["name"]: x for x in m.library.scan()["misplaced"]}
    assert moves["anima-base-v1.0.safetensors"]["to"] == str(md / "diffusion_models")
    assert moves["qwen_3_06b_base.safetensors"]["to"] == str(md / "text_encoders")
    assert moves["qwen_image_vae.safetensors"]["to"] == str(md / "vae")
    assert all(x["from"] == str(old) for x in moves.values())

    for f, sub in ((dit, "diffusion_models"), (te, "text_encoders"), (vae, "vae")):
        (md / sub).mkdir(parents=True, exist_ok=True)
        f.rename(md / sub / f.name)
    scan = m.library.scan()
    assert scan["misplaced"] == []
    models = m.registry.list()
    assert len(models) == 1 and models[0]["id"] == reg["id"] and not models[0]["missing"]
    assert models[0]["files"]["ckpt"] == str(md / "diffusion_models" / "anima-base-v1.0.safetensors")


def test_missing_model_message_says_where_to_move(env):
    """Before training, a model whose file is only in an old folder gets a 'move from - to' message."""
    m, md = env["models"], env["config"].MODELS_DIR
    from lorastudio import training
    dit = _st(md / "anima" / "anima-base-v1.0.safetensors", {"net.llm_adapter.blocks.0.w": "BF16"})
    te = _st(md / "text_encoders" / "qwen_3_06b_base.safetensors", {"model.layers.0.w": "BF16"})
    vae = _st(md / "vae" / "qwen_image_vae.safetensors", {"decoder.w": "BF16"})
    reg = m.registry.add("Anima", "anima", {"ckpt": str(dit), "te": str(te), "vae": str(vae)})
    # registered from another install folder that no longer exists
    m.registry.update(reg["id"], files={**reg["files"], "ckpt": str(env["tmp"] / "old install" / dit.name)})
    msg = training.missing_file_messages(m.registry.get(reg["id"]))
    assert msg == [f"「Anima」の Anima DiT (.safetensors)（anima-base-v1.0.safetensors）を、{md / 'anima'} から "
                   f"{md / 'diffusion_models'} に移してください。"]


def test_generated_material_varies_actions_and_angles(env):
    """char_std_v4: many actions and angles, not mostly standing figures.  A full
    batch uses every pose and angle, the first 24 images already show several actions, every pose has words in
    both prompt styles and is removed from the user's prompt, and an excluded pose only drops the pose."""
    from lorastudio import generate as g
    shots = g.plan_shots(72)
    assert {s.get("pose") for s in shots} - {None} == set(g.POSE_LABEL)
    assert {s["angle"] for s in shots} == set(g.ANGLE_LABEL)
    assert len({s.get("pose") for s in g.plan_shots(24)} - {None}) >= 12
    for pose in g.POSE_LABEL:
        shot = {"framing": "full", "angle": "front", "expr": "smile", "bg": "simple", "pose": pose}
        assert g.TAGS["pose"][pose].split(",")[0] in g.compose_prompt("1girl", shot, "tags")
        assert g.NATURAL["pose"][pose] in g.compose_prompt("A girl.", shot, "natural")
        for t in g.TAGS["pose"][pose].split(","):
            assert t.strip() in g.SHOT_WORDS
    no_jump = g.plan_shots(72, exclude_poses=["jumping", "lying"])
    assert len(no_jump) == 72 and not {"jumping", "lying"} & {s.get("pose") for s in no_jump}
    assert [s["angle"] for s in no_jump] == [s["angle"] for s in shots]
