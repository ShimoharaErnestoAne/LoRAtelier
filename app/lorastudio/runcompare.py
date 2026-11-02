"""Comparing training runs with each other: what was changed (settings, material, captions) and the comparison
images made under the same conditions, side by side.

Runs are compared through their frozen snapshots (run.json / runs.config), so later edits of the project never
change what a run is shown to have been trained with.  Images are matched across runs only when everything but
the LoRA is the same: base model, sampling, negative prompt, size, seed and the prompt (with each run's own
trigger word put back to a placeholder, so a renamed trigger still matches).
"""
from __future__ import annotations

import json

from . import presets
from .captions import split_tags

# scores stored per image by evaluate.score_eval; shown as averages per run (assistive values only)
SCORE_KEYS = ("ccip_diff", "concept_prob", "style_sim", "copy_sim", "effect")
INFO_ROWS = (("model", "ベースモデル"), ("trigger", "呼び出し語"), ("outfit_mode", "衣装の扱い"), ("n_images", "採用画像の枚数"),
             ("engine_commit", "学習エンジンの版"), ("preset_version", "初期値の版"))
OUTFIT_LABEL = {"include": "衣装も含めて覚える", "change": "衣装を変えて使いたい"}


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _info(snap: dict) -> dict:
    return {"model": (snap.get("model") or {}).get("name"), "trigger": snap.get("trigger"),
            "outfit_mode": OUTFIT_LABEL.get(snap.get("outfit_mode"), snap.get("outfit_mode")) if snap.get("kind") == "character" else None,
            "n_images": len(snap.get("items") or []), "engine_commit": (snap.get("engine_commit") or "")[:7] or None,
            "preset_version": snap.get("preset_version")}


def settings_rows(snaps: list[dict]) -> list[dict]:
    """One row per setting or run property: the value in each run and whether they differ."""
    rows = []
    for key, label in INFO_ROWS:
        vals = [_info(s)[key] for s in snaps]
        if all(v is None for v in vals):
            continue
        rows.append({"key": key, "label": label, "values": [_fmt(v) for v in vals], "differs": len(set(map(str, vals))) > 1})
    keys = []
    for s in snaps:
        keys += [k for k in (s.get("values") or {}) if k not in keys]
    for k in keys:
        vals = [(s.get("values") or {}).get(k) for s in snaps]
        rows.append({"key": k, "label": presets.FIELDS[k][0] if k in presets.FIELDS else k, "values": [_fmt(v) for v in vals],
                     "differs": len(set(map(str, vals))) > 1, "setting": True})
    return rows


def _caption_change(before: str, after: str, b_norm: str, a_norm: str) -> dict:
    b, a = split_tags(b_norm), split_tags(a_norm)
    return {"added": [t for t in a if t not in b], "removed": [t for t in b if t not in a],
            "before": before[:400], "after": after[:400]}


def material_diff(base: dict, other: dict) -> dict:
    """Material of `other` compared with `base`: images added / left out, crops and captions changed."""
    bi = {it["id"]: it for it in base.get("items") or []}
    oi = {it["id"]: it for it in other.get("items") or []}
    crop, caption = [], []
    for iid in sorted(bi.keys() & oi.keys()):
        b, o = bi[iid], oi[iid]
        if (b.get("crop") or None) != (o.get("crop") or None):
            crop.append(iid)
        # the trigger is part of every caption: a renamed trigger alone is shown once in the settings, not per image
        bc = (b.get("caption") or "").replace(base.get("trigger") or "\0", "{T}")
        oc = (o.get("caption") or "").replace(other.get("trigger") or "\0", "{T}")
        if bc != oc:
            caption.append({"id": iid, **_caption_change(b.get("caption") or "", o.get("caption") or "", bc, oc)})
    return {"added": sorted(oi.keys() - bi.keys()), "removed": sorted(bi.keys() - oi.keys()), "crop": crop, "caption": caption}


def change_summary(prev: dict | None, snap: dict) -> list[str]:
    """Short sentences on what this run changed from the previous one (shown in the training history)."""
    if not prev or not prev.get("items") and not prev.get("values"):
        return []
    out = []
    pi, si = _info(prev), _info(snap)
    if pi["model"] != si["model"]:
        out.append(f"ベースモデルを {_fmt(pi['model'])} から {_fmt(si['model'])} に変えました")
    if pi["trigger"] != si["trigger"]:
        out.append(f"呼び出し語を {_fmt(pi['trigger'])} から {_fmt(si['trigger'])} に変えました")
    if pi["outfit_mode"] != si["outfit_mode"] and si["outfit_mode"]:
        out.append(f"衣装の扱いを「{si['outfit_mode']}」に変えました")
    pv, sv = prev.get("values") or {}, snap.get("values") or {}
    for k in sv:
        if k in pv and pv[k] != sv[k]:
            label = presets.FIELDS[k][0] if k in presets.FIELDS else k
            out.append(f"{label}を {_fmt(pv[k])} から {_fmt(sv[k])} に変えました")
    m = material_diff(prev, snap)
    if m["added"] or m["removed"]:
        parts = ([f"{len(m['added'])} 枚増やし"] if m["added"] else []) + ([f"{len(m['removed'])} 枚外し"] if m["removed"] else [])
        out.append("素材を " + "、".join(parts) + "ました")
    if m["crop"]:
        out.append(f"{len(m['crop'])} 枚の切り抜きを変えました")
    if m["caption"]:
        out.append(f"{len(m['caption'])} 枚の説明文を変えました")
    return out


# ------------------------------------------------------------------ comparison images across runs
def _norm_prompt(prompt: str, trigger: str | None) -> str:
    return prompt.replace(trigger, "{T}") if trigger else prompt


def _cond(snap: dict, ev: dict) -> str:
    c = (ev.get("results") or {}).get("conditions") or {}
    spec = ev["spec"]
    return json.dumps([(snap.get("model") or {}).get("id"), c.get("sampling"), c.get("negative"),
                       spec.get("width"), spec.get("height")], sort_keys=True, ensure_ascii=False)


def run_images(snap: dict, evals: list[dict]) -> list[dict]:
    """Every finished comparison image of a run with the conditions it was made under."""
    out = []
    for ev in evals:
        if ev["status"] != "done":
            continue
        res = ev.get("results") or {}
        imgs = res.get("images") or []
        scores = res.get("scores") or {}
        cond = _cond(snap, ev)
        for i, im in enumerate(imgs):
            if im.get("model"):   # made on another model: not under the same conditions as the other runs
                continue
            pi = im["prompt"]
            prompt = ev["spec"]["prompts"][pi] if pi < len(ev["spec"]["prompts"]) else ""
            sc = {k: scores[k][i] for k in SCORE_KEYS if isinstance(scores.get(k), list) and len(scores[k]) == len(imgs)}
            out.append({"eval": ev["id"], "file": im["file"], "ckpt": str(im["ckpt"]), "strength": float(im["strength"]),
                        "prompt": prompt, "key": _norm_prompt(prompt, snap.get("trigger")), "seed": im["seed"] + pi,
                        "cond": cond, "scores": sc})
    return out


def recommended(evals: list[dict]) -> dict | None:
    """The newest recommendation of the run's finished comparisons."""
    for ev in sorted(evals, key=lambda e: -e["id"]):
        rec = (ev.get("results") or {}).get("recommend") if ev["status"] == "done" else None
        if rec:
            return {"ckpt": str(rec["ckpt"]), "strength": float(rec["strength"])}
    return None


def pick_column(images: list[dict], mode: str, strength: float, rec: dict | None) -> tuple[str, float] | None:
    """Which saved version / strength stands for the run: the final one at `strength`, or the recommended one."""
    if mode == "recommend" and rec:
        return rec["ckpt"], rec["strength"]
    if any(im["ckpt"] == "final" and abs(im["strength"] - strength) < 1e-6 for im in images):
        return "final", strength
    return None


def compare_images(runs: list[dict], mode: str = "final", strength: float = 1.0) -> dict:
    """runs: [{"id", "name", "snap", "evals"}] in display order.  Rows are prompt/seed pairs made under the same
    conditions for at least two runs (or for one run and "no LoRA"); cells hold the image of each run."""
    cols, per_run = [], []
    for r in runs:
        imgs = run_images(r["snap"], r["evals"])
        rec = recommended(r["evals"])
        col = pick_column(imgs, mode, strength, rec)
        cols.append({"run": r["id"], "name": r["name"], "ckpt": col[0] if col else None, "strength": col[1] if col else None,
                     "recommended": rec})
        per_run.append((imgs, col))
    base_cells: dict[tuple, dict] = {}
    cells: dict[tuple, dict] = {}
    for r, (imgs, col) in zip(runs, per_run):
        for im in imgs:
            k = (im["cond"], im["key"], im["seed"])
            if im["ckpt"] == "base":
                base_cells.setdefault(k, {"run": r["id"], **im})
            elif col and im["ckpt"] == col[0] and abs(im["strength"] - col[1]) < 1e-6:
                cells.setdefault(k, {})[r["id"]] = im
    rows = []
    for k in cells:   # in the order of the first run's comparison (prompt order, then seed)
        found = cells[k]
        base = base_cells.get(k)
        if len(found) + (1 if base else 0) < 2:
            continue
        rows.append({"prompt": next(iter(found.values()))["prompt"], "seed": k[2], "base": base, "cells": found})
    strengths = sorted({im["strength"] for imgs, _c in per_run for im in imgs if im["ckpt"] == "final"})
    return {"cols": cols, "rows": rows, "strengths": strengths, "conds": len({k[0] for k in cells})}


def align_specs(runs: list[dict], mode: str, strength: float, seeds: list[int], prompts_for, size_for) -> list[tuple[int, dict]]:
    """Specs that give every run comparison images under the same conditions.  prompts_for(run) / size_for(run)
    return the run's prompts (same sets, its own trigger) and image size.  Runs that already have all of them are
    left out; "no LoRA" is generated with the first run only."""
    out = []
    for i, r in enumerate(runs):
        imgs = run_images(r["snap"], r["evals"])
        rec = recommended(r["evals"])
        ck, s = (rec["ckpt"], rec["strength"]) if mode == "recommend" and rec else ("final", float(strength))
        prompts = prompts_for(r)
        w, h = size_for(r)
        have = {(im["key"], im["seed"], im["ckpt"], im["strength"]) for im in imgs}
        need_base = i == 0
        want = [(_norm_prompt(p, r["snap"].get("trigger")), seed + pi) for pi, p in enumerate(prompts) for seed in seeds]
        if all((k, sd, ck, s) in have for k, sd in want) and (not need_base or all((k, sd, "base", 0.0) in have for k, sd in want)):
            continue
        out.append((r["id"], {"checkpoints": ["final" if ck == "final" else int(ck)], "strengths": [s], "seeds": list(seeds),
                              "include_base": need_base, "prompts": prompts, "width": w, "height": h}))
    return out
