"""Multi-character LoRA: one LoRA that holds several characters, each called by its own trigger word.

A multi-character project (kind "character", source "multi") has no material of its own.  It names character
projects ("members"); their adopted images, crops and captions are taken as they are when the training starts.
Each member keeps its own trigger in its captions, so the characters stay apart in one LoRA (not guaranteed:
characters that look alike can still mix, and the comparison images are there to check it).  Material, captions
and reference images are fixed in the member projects, which keeps all of the usual checks and editing.

Image ids of different members can be the same, so the images of a run get ids MEMBER_ID_STEP * (index + 1) + id.
"""
from __future__ import annotations

MEMBER_ID_STEP = 1_000_000
MAX_MEMBERS = 8


def is_multi(meta: dict) -> bool:
    return (meta or {}).get("source") == "multi"


def member_ids(meta: dict) -> list[str]:
    return [m for m in (meta or {}).get("members") or [] if isinstance(m, str)]


def gather(projects, p, used_of) -> tuple[list[dict], list[dict]]:
    """(images, members) of a multi-character project.  used_of(project) gives a member's adopted images.
    Each image is a copy with its run id, its source path (_src) and its member index; members carry what the
    snapshot, the checks and the comparison need."""
    images, members = [], []
    for k, pid in enumerate(member_ids(p.get_meta())):
        try:
            mp = projects.get(pid)
            mm = mp.get_meta()
        except (KeyError, OSError):
            members.append({"pid": pid, "name": pid, "missing": True})
            continue
        used = used_of(mp) if mm.get("kind") == "character" and not is_multi(mm) else []
        refs = set(mm.get("reference_ids") or mm.get("auto_reference") or [])
        off = MEMBER_ID_STEP * (k + 1)
        for im in used:
            images.append({**im, "id": off + im["id"], "_src": str(mp.source_path(im)), "_member": k})
        members.append({"pid": pid, "name": mm.get("name") or pid, "kind": mm.get("kind"), "multi": is_multi(mm),
                        "trigger": mm.get("trigger"), "outfit_mode": mm.get("outfit_mode", "include"),
                        "model_id": mm.get("model_id"), "n_used": len(used),
                        "refs": sorted(off + r for r in refs if any(im["id"] == r for im in used))})
    return images, members


def check(members: list[dict], caption_style_of) -> tuple[list[str], list[str]]:
    """Errors (block the start) and warnings of the member list.  caption_style_of(model_id) -> "tags" / "natural"."""
    errors, warnings = [], []
    if len(members) < 2:
        errors.append("複数キャラ LoRA には、キャラクター LoRA のプロジェクトを 2 つ以上選んでください。")
    if len(members) > MAX_MEMBERS:
        errors.append(f"一度にまとめられるのは {MAX_MEMBERS} キャラまでです。")
    for m in members:
        if m.get("missing"):
            errors.append(f"まとめるプロジェクト（{m['name']}）が見つかりません。削除されていたら、一覧から外してください。")
        elif m["kind"] != "character" or m["multi"]:
            errors.append(f"「{m['name']}」はキャラクター LoRA のプロジェクトではないため、まとめられません。")
        elif m["n_used"] < 3:
            errors.append(f"「{m['name']}」の採用画像が {m['n_used']} 枚です。キャラごとに 3 枚以上必要です。")
    ok = [m for m in members if not m.get("missing") and m["kind"] == "character" and not m["multi"]]
    seen = {}
    for m in ok:
        t = (m["trigger"] or "").strip().lower()
        if t in seen:
            errors.append(f"「{seen[t]}」と「{m['name']}」の呼び出し語が同じ（{m['trigger']}）です。キャラを呼び分けられないため、"
                          "どちらかのプロジェクトで呼び出し語を変えてください。")
        seen.setdefault(t, m["name"])
    styles = {}
    for m in ok:
        s = caption_style_of(m["model_id"])
        if s:
            styles.setdefault(s, []).append(m["name"])
    if len(styles) > 1:
        errors.append("説明文の形式（タグ形式と文章形式）がキャラによって違います: "
                      + " / ".join(f"{'タグ' if s == 'tags' else '文章'}: {', '.join(n)}" for s, n in styles.items())
                      + "。同じ系統のモデルを選んだプロジェクトどうしをまとめてください。")
    counts = [m["n_used"] for m in ok if m["n_used"]]
    if len(counts) > 1 and max(counts) > 3 * min(counts):
        warnings.append("キャラによって採用画像の枚数が大きく違います（"
                        + "、".join(f"{m['name']} {m['n_used']} 枚" for m in ok) + "）。枚数の少ないキャラは特徴が弱く出たり、"
                        "枚数の多いキャラに似たりすることがあります。")
    return errors, warnings


def snap_members(members: list[dict], items: list[dict]) -> list[dict]:
    """What a run keeps about its members (the comparison uses each member's trigger and reference images)."""
    out = []
    for k, m in enumerate(members):
        if m.get("missing"):
            continue
        lo, hi = MEMBER_ID_STEP * (k + 1), MEMBER_ID_STEP * (k + 2)
        mine = [it for it in items if lo <= it["id"] < hi]
        out.append({"pid": m["pid"], "name": m["name"], "trigger": m["trigger"], "outfit_mode": m["outfit_mode"],
                    "refs": m["refs"], "id_range": [lo, hi], "n_items": len(mine)})
    return out


def member_of_prompt(members: list[dict], prompt: str) -> int | None:
    """Index of the member whose trigger is in a comparison prompt (None for prompts without one)."""
    import re
    for k, m in enumerate(members):
        if re.search(rf"(?<![\w]){re.escape(m['trigger'])}(?![\w])", prompt or ""):
            return k
    return None
