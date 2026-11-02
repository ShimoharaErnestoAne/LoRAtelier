"""Export a chosen checkpoint as a LoRA package usable in ComfyUI.

Package (training images are never included), by default in models/loras/ of this tool, one folder named
after the LoRA (the project's exports folder was too deep and mixed with work files):
  <name>_v<N>/<name>_v<N>.safetensors   LoRA in the key layout ComfyUI expects
  <name>_v<N>/使い方.txt                 short usage guide
  <name>_v<N>/lora_info.json             base model id/hash, trigger, kind, settings, engine
                                         version, evaluation conditions, tested strengths
Existing files are never overwritten: a new version folder is created.  The export only
counts as done after the written file was re-read and its header validated.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path

from . import config
from .captions import CONCEPT_TARGETS, concept_target
from .evaluate import (COMFY_MODELS, CONVERTED_FAMILIES, MONO_NEGATIVE, comfy_lora_file, staged_lora,
                       trained_with_monochrome)
from .models import FAMILY_LABEL, VARIANTS, hasher, read_safetensors_header


def _next_version_dir(dest_root: Path, name: str) -> Path:
    n = 1
    for p in dest_root.iterdir():   # not glob: a name with [ ] would be read as a pattern
        m = re.fullmatch(re.escape(name) + r"_v(\d+)(\.safetensors)?", p.name)   # also loose files put there by hand
        if m:
            n = max(n, int(m.group(1)) + 1)
    return dest_root / f"{name}_v{n}"


def export(project, run_id: int, ckpt_path: str, strength: float | None, dest_root: str | None,
           eval_summary: dict | None) -> dict:
    run = project.run(run_id)
    snap = run["config"]
    src = Path(ckpt_path)
    if not src.exists():
        raise FileNotFoundError(f"保存版が見つかりません: {src}")
    root = Path(dest_root) if dest_root else config.MODELS_DIR / "loras"
    root.mkdir(parents=True, exist_ok=True)
    base_name = re.sub(r'[\\/:*?"<>|\s]+', "_", project.meta("name") or snap["trigger"]).strip("_") or snap["trigger"]
    final_dir = _next_version_dir(root, base_name)
    # built in a hidden sibling and renamed when complete: a failed export leaves no half-written LoRA in loras
    out_dir = root / f".{final_dir.name}.{uuid.uuid4().hex[:8]}.part"
    out_dir.mkdir(parents=False, exist_ok=False)
    try:
        return _write_package(project, run_id, run, snap, src, ckpt_path, strength, eval_summary, out_dir, final_dir)
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise


def _write_package(project, run_id, run, snap, src, ckpt_path, strength, eval_summary, out_dir: Path, final_dir: Path) -> dict:
    fname = final_dir.name + ".safetensors"
    fam = snap["model"]["family"]
    if fam in CONVERTED_FAMILIES:
        # the conversion starts the training Python: the file the comparison already converted (or one converted now
        # to the same place) is copied, so exporting the same saved version again does not convert again
        conv = COMFY_MODELS / "loras" / staged_lora(project.id, run_id, fam, src)
        lora = out_dir / fname
        shutil.copyfile(conv, lora)
    else:
        lora = comfy_lora_file(fam, src, out_dir / fname)
    # verify what was written
    hdr = read_safetensors_header(lora)
    n_tensors = len([k for k in hdr if k != "__metadata__"])
    if n_tensors == 0:
        raise RuntimeError("書き出したファイルに重みが含まれていません。")
    step = next((c["step"] for c in project.checkpoints(run_id) if c["path"] == ckpt_path), None)
    model = snap["model"]
    sha = dict(model.get("sha256") or {})
    for k, p in model["files"].items():  # fill hashes computed after the run started
        sha[k] = sha.get(k) or hasher.get(p)
    info = {
        "format": "LoRA (safetensors) for ComfyUI",
        "kind": {"character": "キャラクター", "style": "スタイル", "enhance": "質感・補強"}.get(snap["kind"])
                or CONCEPT_TARGETS[concept_target(snap)]["label"],
        "trigger": None if snap["kind"] == "enhance" else f"{snap['trigger']} style" if snap["kind"] == "style" else snap["trigger"],
        "characters": [{"name": m["name"], "trigger": m["trigger"]} for m in snap["members"]] if snap.get("members") else None,
        "base_model": {"name": model["name"], "variant": VARIANTS[model["variant"]]["label"], "family": model["family"],
                       "files": {k: Path(p).name for k, p in model["files"].items()},
                       "sha256": {k: v for k, v in sha.items() if v}},
        "checkpoint_step": step or "final",
        "tested_strength": strength,
        "strength_note": "比較画像で確認した強さです。どのプロンプトでも最適とは限りません。",
        "training": {"engine": snap["engine"], "engine_commit": snap["engine_commit"], "preset_version": snap["preset_version"],
                     "settings": snap["values"], "changed_from_default": snap["changed"], "n_images": len(snap["items"])},
        "evaluation": eval_summary or {},
        "exported": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tool": "LoRAtelier (local)",
    }
    if snap.get("members"):
        info["kind"] = "複数キャラクター"
    if snap.get("mask"):
        info["training"]["masked_training"] = {"learnt": {"person": "人物", "background": "背景"}[snap["mask"]["mode"]],
                                               "outside_weight": snap["mask"]["floor"], "n_masked": snap["mask"].get("n_masked")}
    config.atomic_write_json(out_dir / "lora_info.json", info)
    fam = model["family"]
    loader = "LoraLoaderModelOnly（または LoraLoader）"
    base_note = {
        "sdxl": f"ベースモデル: {model['name']}（SDXL 系）。別系列のモデル（SD1.5 / Anima など）では使えません。",
        "anima": f"ベースモデル: {model['name']}（Anima）。Anima 以外のモデルでは使えません。",
        "krea2": "ベースモデル: Krea 2。RAW で学習しており、Krea 2 Turbo での使用を想定しています。",
        "zimage": f"ベースモデル: {model['name']}（Z-Image）。Z-Image 以外のモデルでは使えません。"
                  + ("Turbo 用の学習アダプターを使って学習しています。使うときにアダプターは要りません。"
                     if model.get("variant") == "zimage_turbo" else ""),
    }.get(fam, f"ベースモデル: {model['name']}（{FAMILY_LABEL.get(fam, fam)}）。別系列のモデルでは使えません。")
    mono_note = ("" if not trained_with_monochrome(snap) else
                 f"5. 白黒の素材を含めて学習しています。カラーで出したいときは、ネガティブプロンプトに「{MONO_NEGATIVE}」を入れてください。\n"
                 "   白黒の素材からは色を学習できないため、髪や目などの色はプロンプトで指定すると安定します。\n")
    enh = snap.get("enhance")
    if enh:
        from .enhance import METHODS, effect
        info["enhance"] = {"method": METHODS[enh["method"]], "effect": effect(enh["effect"])["label"],
                           "positive": enh.get("positive"), "negative": enh.get("negative")}
        config.atomic_write_json(out_dir / "lora_info.json", info)
        s0 = strength if strength is not None else 1.0
        how = "3. 呼び出し語は要りません。LoRA を読み込むだけで効きます。\n"
        if enh["method"] == "words":
            how += (f"4. 強さ: {s0} から試してください。マイナスの強さにすると、逆向き"
                    f"（{enh.get('negative') or '効果を弱める方向'}）に働きます。\n")
        else:
            how += f"4. 強さ: {s0} から試してください。強くしすぎると画像が崩れやすくなります。\n"
        kind_line = f"種類: 質感・補強 LoRA（{info['enhance']['effect']}、{info['enhance']['method']}）"
    elif info.get("characters"):
        names = "\n".join(f"   - {c['name']}: {c['trigger']}" for c in info["characters"])
        how = (f"3. プロンプトに、出したいキャラの呼び出し語を入れます。\n{names}\n"
               f"4. 強さ: {strength if strength is not None else 1.0} から試してください（比較で確認した値。最適値はプロンプトにより変わります）。\n"
               "   1 枚の画像に複数のキャラを出すと、特徴が混ざりやすくなります。\n")
        kind_line = "種類: 複数キャラクター LoRA\n呼び出し語: " + " / ".join(f"{c['trigger']}（{c['name']}）" for c in info["characters"])
    else:
        how = (f"3. プロンプトに呼び出し語「{info['trigger']}」を入れます。\n"
               f"4. 強さ: {strength if strength is not None else 1.0} から試してください（比較で確認した値。最適値はプロンプトにより変わります）。\n")
        kind_line = f"種類: {info['kind']} LoRA\n呼び出し語: {info['trigger']}"
    guide = f"""LoRA: {fname}
{kind_line}
{base_note}

■ ComfyUI での使い方
1. {fname} を ComfyUI の models/loras フォルダーに入れます。
2. モデル読み込みノードの後に {loader} を追加し、このファイルを選びます。
{how}{mono_note}
■ 同梱ファイル
- {fname}: LoRA 本体
- lora_info.json: 学習条件・ベースモデル情報・評価条件
学習に使った画像は含まれていません。
"""
    (out_dir / "使い方.txt").write_text(guide, encoding="utf-8")
    os.rename(out_dir, final_dir)   # never replaces: fails if that version appeared meanwhile
    out_dir, lora = final_dir, final_dir / fname
    comfy_link = None
    try:   # a convenience: the export itself is complete either way
        from .linking import link_export
        comfy_link = link_export(lora)
    except OSError as e:
        comfy_link = {"path": None, "kind": "none", "message": f"ComfyUI の loras フォルダーに置けませんでした: {e}"}
    if comfy_link:
        info["comfy_link"] = comfy_link
        config.atomic_write_json(out_dir / "lora_info.json", info)
    with project.tx() as db:
        db.execute("INSERT INTO exports(run_id, created, ckpt_path, dest, info) VALUES(?,?,?,?,?)",
                   (run_id, time.time(), ckpt_path, str(out_dir), json.dumps(info, ensure_ascii=False)))
    return {"dir": str(out_dir), "file": str(lora), "info": info, "comfy_link": comfy_link}
