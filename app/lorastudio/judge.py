"""Turn facts + AI results into per-image checks, a verdict, and a dataset-level bias report.

Principles (spec 8.x):
* image quality / fitness for the purpose / dataset bias are separate axes;
* every check says whether it is a *fact* or an *AI estimate*, with confidence 高/中/低;
* a failed or missing analysis is never shown as "no problem" (-> 判定保留 / 解析中);
* exclusion is only a suggestion; originals are never deleted.
"""
from __future__ import annotations

from collections import Counter
from statistics import median

import numpy as np

from . import captions

VERDICT_LABEL = {
    "unusable": "使用不能", "exclude": "除外候補", "review": "要確認", "hold": "判定保留",
    "pending": "解析中", "ok": "採用推奨",
}
TEXT_TAGS = {"watermark", "web address", "text", "english text", "logo", "speech bubble", "copyright name",
             "twitter username", "patreon username", "japanese text"}
SIGN_TAGS = {"signature", "artist name", "dated"}
MULTI_TAGS = {"2girls", "3girls", "4girls", "5girls", "6+girls", "2boys", "3boys", "4boys", "5boys", "6+boys",
              "multiple girls", "multiple boys", "multiple others"}
CCIP_SAME_THRESHOLD = 0.178   # imgutils' published default for "same character"
NEAR_DUP_HAMMING = 6
NEAR_DUP_EMBED = 0.965


def _c(key, label, status, source, conf, msg, fix=None):
    return {"key": key, "label": label, "status": status, "source": source, "confidence": conf, "message": msg, "fix": fix}


def build_context(images: list[dict], meta: dict, identity: dict | None = None) -> dict:
    """Dataset-wide context needed for per-image checks (duplicates, similarity, sharpness baseline)."""
    first_by_sha, dup_of = {}, {}
    for im in images:
        if im["sha"] in first_by_sha:
            dup_of[im["id"]] = first_by_sha[im["sha"]]
        else:
            first_by_sha[im["sha"]] = im["id"]
    near: dict[int, set[int]] = {}
    cand = [im for im in images if im.get("fast") and im["fast"].get("ok") and im["id"] not in dup_of]
    if len(cand) > 1:
        ids = np.array([im["id"] for im in cand])
        h = np.array([int(im["fast"]["phash"], 16) for im in cand], dtype=np.uint64)
        x = h[:, None] ^ h[None, :]
        ham = np.zeros(x.shape, dtype=np.int32)
        for _ in range(64):  # popcount
            ham += (x & np.uint64(1)).astype(np.int32)
            x >>= np.uint64(1)
        sim = ham <= NEAR_DUP_HAMMING
        emb = [(im.get("ai") or {}).get("embed") for im in cand]
        has = np.array([e is not None for e in emb])
        if has.sum() > 1:
            dim = len(next(e for e in emb if e is not None))
            E = np.array([e if e is not None else [0.0] * dim for e in emb], dtype=np.float32)
            sim |= ((E @ E.T) >= NEAR_DUP_EMBED) & has[:, None] & has[None, :]
        np.fill_diagonal(sim, False)
        for i, j in zip(*np.nonzero(sim)):
            near.setdefault(int(ids[i]), set()).add(int(ids[j]))
    # style outliers: images whose overall look is far from the rest of the set (e.g. another artist mixed in)
    style_out = {}
    embs = [(im["id"], (im.get("ai") or {}).get("embed")) for im in images if im["id"] not in dup_of]
    embs = [(i, e) for i, e in embs if e]
    if len(embs) >= 8:
        E = np.array([e for _, e in embs], dtype=np.float32)
        c = E.mean(0)
        c /= np.linalg.norm(c)
        sim = E @ c
        mu, sd = float(sim.mean()), float(sim.std()) + 1e-6
        for (iid, _), v in zip(embs, sim):
            if v < mu - 3 * sd and v < 0.8:
                style_out[iid] = round(float((v - mu) / sd), 1)
    sharp = [im["fast"]["sharpness_p90"] for im in images if im.get("fast") and im["fast"].get("ok")]
    return {"kind": meta.get("kind"), "outfit_mode": meta.get("outfit_mode", "include"), "dup_of": dup_of,
            "near": near, "median_sharp": median(sharp) if sharp else None, "identity": identity or {},
            "reference_ids": set(meta.get("reference_ids") or []), "train_res": 1024, "style_out": style_out,
            "concept": meta.get("concept_words") or [], "target": captions.concept_target(meta)}


def check_image(im: dict, ctx: dict) -> dict:
    checks = []
    f = im.get("fast") or {}
    ai = im.get("ai") or {}
    kind = ctx["kind"]

    if im["fast_status"] == "pending":
        return {"verdict": "pending", "checks": [], "use": False}
    if f.get("missing"):
        checks.append(_c("missing", "ファイル", "bad", "事実", "高", "元画像が見つかりません。場所を再指定してください。"))
        return {"verdict": "unusable", "checks": checks, "use": False}
    if not f.get("ok"):
        checks.append(_c("decode", "読み込み", "bad", "事実", "高", f.get("error", "読み込めません")))
        return {"verdict": "unusable", "checks": checks, "use": False}

    # ---- image state (facts)
    crop = im.get("crop")
    w, h = f["width"], f["height"]
    if crop:
        w, h = crop[2] - crop[0], crop[3] - crop[1]
    short = min(w, h)
    if short < 512:
        checks.append(_c("res", "解像度", "bad", "事実", "高", f"短辺 {short}px は学習解像度 {ctx['train_res']}px に対して小さすぎます。",
                         "より大きい元画像に差し替え"))
    elif short < 768:
        checks.append(_c("res", "解像度", "warn", "事実", "高", f"短辺 {short}px。拡大して学習されるため、細部がぼやけて学習される可能性があります。"))
    ar = max(w, h) / max(1, min(w, h))
    if ar > 3:
        checks.append(_c("aspect", "縦横比", "warn", "事実", "高", f"縦横比 {ar:.1f}:1 と極端です。切り抜きを検討してください。", "切り抜き"))
    if f.get("animated"):
        checks.append(_c("anim", "形式", "warn", "事実", "高", "アニメーション画像です。最初のフレームだけ使われます。"))
    if im["id"] in ctx["dup_of"]:
        checks.append(_c("dup", "重複", "bad", "事実", "高", f"画像 #{ctx['dup_of'][im['id']]} と完全に同じファイルです。"))
    near = ctx["near"].get(im["id"])
    if near:
        ids = ", ".join(f"#{i}" for i in sorted(near)[:5])
        checks.append(_c("near", "類似", "warn", "推定", "中",
                         f"{ids} とよく似ています。表情や角度が違うなら残して構いません（自動削除はしません）。"))
    jq = f.get("jpeg_quality")
    if jq is not None and jq < 60:
        checks.append(_c("jpeg", "圧縮", "warn", "事実(推定値)", "中", f"JPEG 品質が低め（推定 {jq}）。ブロックノイズも学習される可能性があります。"))

    is_style = kind == "style"
    ms = ctx["median_sharp"]
    sp = f.get("sharpness_p90", 0)
    if ms and sp < min(25.0, ms * 0.25):
        if is_style:
            checks.append(_c("blur", "鮮明さ", "info", "推定", "低", "全体が柔らかい画像です。画風としてのぼかしなら問題ありません。"))
        else:
            checks.append(_c("blur", "鮮明さ", "warn", "推定", "中", "ピンぼけ・手ぶれの可能性があります。"))
    if f.get("clip_dark", 0) > 0.5 or f.get("clip_bright", 0) > 0.6:
        st = "info" if is_style else "warn"
        checks.append(_c("exposure", "露出", st, "推定", "低", "白飛び・黒つぶれが多い画像です。" + ("画風の表現なら問題ありません。" if is_style else "")))
    if kind == "character" and captions.is_monochrome(im):
        checks.append(_c("mono", "白黒", "info", "推定", "中",
                         "白黒の画像です。髪や目などの色はこの画像からは学習されません。説明文に「monochrome」を付けて、"
                         "白黒であることをキャラの特徴として覚えにくくしています。"))
    if f.get("transparent_ratio", 0) > 0.3:
        checks.append(_c("alpha", "透明", "info", "事実", "高", "透過部分は白背景として学習されます。"))

    # ---- AI-based checks
    status = im.get("ai_status")
    if status in ("pending", "running"):
        pass
    elif status == "failed":
        checks.append(_c("ai", "AI解析", "hold", "—", "—", f"AI 解析に失敗しました: {im.get('ai_error') or '原因不明'}。再実行できます。"))
    elif ai:
        general, _ = captions.tags_from_ai(ai)
        gset = set(general)
        domain = (ai.get("domain") or {}).get("label")
        if "blurry" in gset and not any(c["key"] == "blur" for c in checks):
            if kind == "style":
                checks.append(_c("blur", "鮮明さ", "info", "推定", "低", "ぼけた画像と推定されました。画風としてのぼかしなら問題ありません。"))
            else:
                checks.append(_c("blur", "鮮明さ", "warn", "推定", "中", "画像全体がぼけていると推定されました（タグ推定）。"))
        txt = sorted(TEXT_TAGS & gset)
        if txt:
            checks.append(_c("text", "文字・透かし", "warn", "推定", "中", f"文字や透かしの可能性（{', '.join(txt)}）。学習されると生成画像にも出ることがあります。", "切り抜き / 除外"))
        sig = sorted(SIGN_TAGS & gset)
        if sig:
            checks.append(_c("sign", "署名", "info", "推定", "中", "署名・日付の可能性。説明文にタグとして残しているので、呼び出し語には結びつきにくくなっています。生成画像に出る場合は切り抜いてください。"))
        if kind == "style" and im["id"] in ctx.get("style_out", {}):
            checks.append(_c("styleout", "画風", "warn", "推定", "中",
                             "他の素材と見た目が大きく異なります。別の画風の絵が混ざっていないか確認してください。", "除外"))
        if kind == "concept" and captions.CONCEPT_TARGETS[ctx["target"]]["person"]:
            noun = captions.CONCEPT_TARGETS[ctx["target"]]["noun"]
            persons = [p for p in ai.get("persons", []) if p["score"] >= 0.5]
            if len(persons) > 1 or sorted(MULTI_TAGS & gset):
                checks.append(_c("multi", "人物混在", "info", "推定", "中",
                                 f"人物が複数写っています。全員にその{noun}がないなら、{noun}が弱まることがあります。", "切り抜き"))
            subj = ai.get("subject")
            if not persons and not subj:
                checks.append(_c("nosubj", "対象", "warn", "推定", "低", f"人物が検出できませんでした。{noun}が写っているか確認してください。"))
            elif subj and not crop and (subj.get("area_ratio", 1) < 0.06 or subj.get("eff_res", 9999) < 320):
                checks.append(_c("small", "対象の大きさ", "warn", "推定", "中",
                                 f"人物が小さく写っています（実効 {subj.get('eff_res', '?')}px）。人物の周りを切り抜くと{noun}を覚えやすくなります。", "自動切り抜き候補"))
        if kind == "concept":
            words = ctx.get("concept") or []
            raw = ((ai.get("tags") or {}).get("general") or {})
            if words and raw and not any(captions.concept_match(captions.norm_tag(t), words) for t in raw):
                checks.append(_c("concept", "特徴", "warn", "推定", "低",
                                 f"覚えさせたい{captions.CONCEPT_TARGETS[ctx['target']]['noun']}（{', '.join(words)}）のタグが検出されませんでした。写っていないか、"
                                 "小さい可能性があります。写っているなら採用のままで構いません。"))
        if kind == "character":
            persons = [p for p in ai.get("persons", []) if p["score"] >= 0.5]
            multi = sorted(MULTI_TAGS & gset)
            if len(persons) > 1 or multi:
                checks.append(_c("multi", "人物混在", "warn", "推定", "中",
                                 f"人物が複数写っています（検出 {len(persons)} 人{'、' + ', '.join(multi) if multi else ''}）。対象だけを切り抜くと特徴が混ざりにくくなります。", "切り抜き"))
            subj = ai.get("subject")
            if not persons and not subj:
                conf = "中" if domain == "photo" else "低"
                checks.append(_c("nosubj", "対象", "warn", "推定", conf,
                                 "人物が検出できませんでした。" + ("人外・メカ等のキャラクターなら問題ないことがあります。" if domain != "photo" else "")))
            if subj:
                eff = subj.get("eff_res")
                sx0, sy0, sx1, sy1 = subj["bbox"]
                if crop:  # subject size inside the crop
                    eff = None
                if subj.get("area_ratio", 1) < 0.06 or (eff is not None and eff < 320):
                    checks.append(_c("small", "対象の大きさ", "warn", "推定", "中",
                                     f"対象が小さく写っています（実効 {eff or '?'}px）。対象周辺の切り抜きを推奨します。", "自動切り抜き候補"))
                if domain != "photo" and not ai.get("heads") and persons:
                    checks.append(_c("head", "写り方", "warn", "推定", "低", "頭部が検出できません。顔が切れている・隠れている可能性があります。"))
            if gset & {"head out of frame", "out of frame", "cropped"}:
                checks.append(_c("cut", "写り方", "warn", "推定", "中", "対象の一部が画面外に切れている可能性があります。"))
            ident = ctx["identity"].get(im["id"])
            if im["id"] in ctx["reference_ids"]:
                checks.append(_c("ref", "同一性", "info", "—", "—", "基準画像です。"))
            elif ident:
                if ident.get("method") == "ccip":
                    subj = ai.get("subject") or {}
                    small = subj.get("area_ratio", 1) < 0.2 or subj.get("eff_res", 9999) < 800
                    if ident["diff"] > CCIP_SAME_THRESHOLD * 1.3 and not small:
                        checks.append(_c("ident", "同一性", "warn", "推定", "中", "基準画像と別のキャラクターの可能性があります。"))
                    elif ident["diff"] > CCIP_SAME_THRESHOLD:
                        msg = ("対象が小さく写っているため、基準画像と同じキャラクターか判定しきれません。" if small
                               else "基準画像と同じキャラクターか判定しきれません。")
                        checks.append(_c("ident", "同一性", "warn", "推定", "低", msg))
                elif ident.get("method") == "failed":
                    checks.append(_c("ident", "同一性", "warn", "推定", "低",
                                     "基準画像と同じキャラクターかを判定できませんでした（次の解析のときに判定し直します）。"))
                elif ident.get("method") == "embed" and ident.get("outlier"):
                    checks.append(_c("ident", "同一性", "warn", "推定", "低", "他の素材と傾向が大きく違います（写真の同一人物判定は参考値です）。"))

    statuses = {c["status"] for c in checks}
    if "bad" in statuses:
        verdict = "exclude"
    elif "hold" in statuses:
        verdict = "hold"
    elif "warn" in statuses:
        verdict = "review"
    elif status in ("pending", "running"):
        verdict = "pending"
    else:
        verdict = "ok"
    decision = im.get("decision")
    use = decision == "use" if decision else verdict in ("ok", "review", "hold", "pending")
    return {"verdict": verdict, "checks": checks, "use": use}


# ------------------------------------------------------------------ dataset level
FRAMING = {"portrait": "顔アップ", "close-up": "顔アップ", "upper body": "上半身", "cowboy shot": "膝上",
           "full body": "全身", "lower body": "下半身"}
ANGLES = {"from side": "横向き", "profile": "横向き", "from behind": "後ろ向き", "from above": "俯瞰", "from below": "あおり"}


# tags too generic to be "the feature" of a concept LoRA (shown nowhere as suggestions)
_GENERIC = {"solo", "1girl", "1boy", "1other", "multiple girls", "multiple boys", "2girls", "2boys", "looking at viewer",
            "simple background", "white background", "grey background", "standing", "full body", "upper body",
            "cowboy shot", "portrait", "closed mouth", "open mouth", "smile", "short hair", "long hair", "realistic",
            "photorealistic", "blurry", "blurry background", "outdoors", "indoors", "day", "night", "male focus", "teeth"}


def concept_suggest(analysed: list[dict], words: list[str], limit: int = 10, target: str = "body") -> list[dict]:
    """Tags shared by many material images: the feature to learn is usually among them.  Only a
    suggestion list; the user picks (hair colour and clothing can also be common by accident)."""
    if not analysed:
        return []
    c = Counter()
    for im in analysed:
        c.update(set(captions.tags_from_ai(im["ai"])[0]))
    m = len(analysed)
    t_ = captions.CONCEPT_TARGETS.get(target) or captions.CONCEPT_TARGETS["body"]
    generic = (_GENERIC - t_["keep_generic"]) | ({"no humans", "scenery"} if not t_["person"] else set())
    out = [{"tag": t, "share": round(k / m, 2)} for t, k in c.most_common()
           if k / m >= 0.3 and t not in generic and not captions.concept_match(t, words)
           and not captions._STYLE_WORDS.match(t)]
    return out[:limit]


def _concept_notes(analysed, tagsets, fr, meta) -> list[dict]:
    notes = []
    m = len(tagsets)
    target = captions.concept_target(meta)
    t_ = captions.CONCEPT_TARGETS[target]
    noun = t_["noun"]
    words = meta.get("concept_words") or []
    if words:
        hit = sum(1 for ts in tagsets if any(captions.concept_match(t, words) for t in ts))
        if hit / m < 0.5:
            notes.append({"level": "warn", "text": f"覚えさせたい{noun}（{', '.join(words)}）のタグが付いた画像は {hit}/{m} 枚です。"
                          f"{noun}の写っていない画像が混ざっているか、語がタグの表し方と違う可能性があります。"})
    chars = Counter()
    for im in analysed:
        chars.update(captions.tags_from_ai(im["ai"])[1])
    if chars:
        c, k = chars.most_common(1)[0]
        if k / m > 0.4:
            notes.append({"level": "warn", "text": f"特定のキャラクター（{c}）が {k}/{m} 枚に写っています。{noun}だけでなく、"
                          "そのキャラの顔や髪まで覚えてしまう可能性があります。いろいろな人物の素材を混ぜてください。"})
    if target in ("body", "outfit", "pose") and fr.get("顔アップ", 0) / m > 0.5:
        notes.append({"level": "warn", "text": f"顔のアップが多く、{'体' if target == 'body' else noun}が写っていない素材が多い可能性があります。"})
    if target == "expression" and fr.get("全身", 0) / m > 0.5:
        notes.append({"level": "info", "text": "全身の素材が多く、顔が小さく写っている可能性があります。顔の周りを切り抜くと表情を覚えやすくなります。"})
    if t_["person"]:
        gender = Counter(t for ts in tagsets for t in ts & {"1girl", "1boy"})
        if len(gender) == 1 and m >= 6:
            g = "女性" if "1girl" in gender else "男性"
            notes.append({"level": "info", "text": f"素材が{g}だけです。逆の性別に使うと、{noun}が出にくいことがあります。"})
    if target == "place":
        people = sum(1 for ts in tagsets if not ts & {"no humans"})
        if people / m > 0.7:
            notes.append({"level": "info", "text": f"人物の写った素材が {people}/{m} 枚あります。人物が大きく写っていると、場所より人物が学習されやすくなります。"
                          "人物のいない素材や、人物が小さい素材を混ぜてください。"})
    return notes


def dataset_report(images: list[dict], results: dict[int, dict], meta: dict) -> dict:
    kind = meta.get("kind")
    used = [im for im in images if results[im["id"]]["use"]]
    notes = []
    n = len(used)
    analysed = [im for im in used if im.get("ai")]
    counts = Counter(results[im["id"]]["verdict"] for im in images)
    if n == 0:
        return {"n_used": 0, "notes": [], "counts": counts, "coverage": {}}
    cov = {}
    if kind == "character":
        mono = sum(1 for im in used if captions.is_monochrome(im))
        if mono == n:
            notes.append({"level": "warn", "text": "採用画像がすべて白黒です。キャラの色（髪・目・服など）は学習できないため、"
                          "生成結果が白黒に寄ったり、色が毎回変わったりしやすくなります。カラー画像を数枚加えると色を覚えやすくなります。"
                          "カラー画像がない場合は、生成時のプロンプトで色を指定してください。"})
        elif mono / n > 0.5:
            notes.append({"level": "info", "text": f"白黒の画像が {mono}/{n} 枚あります。色はカラーの {n - mono} 枚から学習されるため、"
                          "色が弱めに出る可能性があります。説明文で白黒であることを区別しているので、白黒の素材はそのまま使えます。"})
    if len(analysed) >= max(3, n // 2):
        tagsets = [set(captions.tags_from_ai(im["ai"])[0]) for im in analysed]
        m = len(tagsets)
        domains = Counter((im["ai"].get("domain") or {}).get("label") for im in analysed)
        if len(domains) > 1 and min(domains.values()) / m > 0.15:
            notes.append({"level": "warn", "text": f"写真とイラストが混在しています（{dict(domains)}）。意図していなければ片方に揃えると安定します。"})
        fr = Counter()
        for ts in tagsets:
            hit = {FRAMING[t] for t in ts if t in FRAMING}
            fr.update(hit or {"不明"})
        an = Counter()
        for ts in tagsets:
            an.update({ANGLES[t] for t in ts if t in ANGLES} or {"正面付近"})
        cov = {"framing": dict(fr), "angle": dict(an)}
        bg_simple = sum(1 for ts in tagsets if ts & {"simple background", "white background", "grey background", "transparent background"})
        if kind == "character":
            if fr.get("全身", 0) / m < 0.1:
                notes.append({"level": "info", "text": "全身が写った素材が少ない可能性があります。全身の生成で体型や衣装が崩れる場合は全身の素材を加えてください。"})
            if (fr.get("顔アップ", 0)) / m > 0.7:
                notes.append({"level": "warn", "text": "顔のアップに偏っています。全身や別の構図で特徴が出にくくなる可能性があります。"})
            if (an.get("横向き", 0) + an.get("後ろ向き", 0)) / m < 0.1:
                notes.append({"level": "info", "text": "横向き・後ろ向きの素材が少ない可能性があります。その角度で生成したい場合に特徴が崩れやすくなります。"})
            if bg_simple / m > 0.7:
                notes.append({"level": "info", "text": "単色背景に集中しています。背景のある絵で特徴が弱まる場合は背景付きの素材を加えてください。"})
            if meta.get("outfit_mode") == "change":
                outfits = Counter(frozenset(t for t in ts if captions.is_clothing(t)) for ts in tagsets)
                top = outfits.most_common(1)[0][1] if outfits else 0
                if top / m > 0.8:
                    notes.append({"level": "info", "text": "ほぼ同じ衣装の素材です。「衣装を変えて使う」場合、衣装が特徴として固定されやすくなります。別衣装の素材があると分離しやすくなります（保証はできません）。"})
        elif kind == "concept":
            notes += _concept_notes(analysed, tagsets, fr, meta)
        elif kind == "enhance":
            # pair method: each image is compared with its own processed partner, so the content is not learned;
            # what matters is that the quality to strengthen is clearly visible
            pass
        else:
            chars = Counter()
            for im in analysed:
                _, ch = captions.tags_from_ai(im["ai"])
                chars.update(ch)
            if chars:
                c, k = chars.most_common(1)[0]
                if k / m > 0.4:
                    notes.append({"level": "warn", "text": f"特定のキャラクター（{c}）が {k}/{m} 枚に写っています。画風ではなくそのキャラが学習される可能性があります。"})
            solo = sum(1 for ts in tagsets if "solo" in ts and ts & {"1girl", "1boy", "1other"})
            if solo / m > 0.85:
                notes.append({"level": "info", "text": "人物1人の絵が大半です。背景・物・動物などを描いた素材があると、別の被写体にも画風が乗りやすくなります。"})
            if bg_simple / m > 0.7:
                notes.append({"level": "info", "text": "単色背景の絵に集中しています。背景の描き方は学習されにくくなります。"})
            known = {k: v for k, v in fr.items() if k != "不明"}
            if known and max(known.values()) / m > 0.8:
                notes.append({"level": "info", "text": f"構図が「{max(known, key=known.get)}」に集中しています。"})
    else:
        notes.append({"level": "info", "text": "AI 解析の完了後に、構図・角度・背景などの偏りを表示します。"})
    if kind == "concept":
        t_ = captions.CONCEPT_TARGETS[captions.concept_target(meta)]
        cov["concept_suggest"] = concept_suggest(analysed, meta.get("concept_words") or [], target=captions.concept_target(meta))
        if not meta.get("concept_words"):
            notes.insert(0, {"level": "warn", "text": f"覚えさせたい{t_['noun']}の語が決まっていません。上の欄に、素材に共通する{t_['noun']}のタグ"
                             f"（例: {t_['example']}）を入れてください。入れないと、{t_['noun']}が呼び出し語ではなく説明文のタグに結びつきます。"})
    if kind == "enhance":
        from .enhance import clean, effect
        enh = clean(meta.get("enhance"))
        if enh["method"] == "words":
            return {"n_used": 0, "notes": [{"level": "info", "text": "「言葉から作る」では素材の画像は使いません。"}],
                    "counts": dict(counts), "coverage": {}}
        ef = effect(enh["effect"])
        notes.insert(0, {"level": "info", "text": f"各画像と、それを加工した画像（{ef['label']}を"
                         f"{'弱めたもの' if ef['direction'] == 'remove' else '足したもの'}）の違いだけを学習します。写っている内容は学習されないので、"
                         f"被写体の偏りは気にしなくて構いません。{ef['label']}がはっきり写った、解像度の高い画像を選んでください。"})
        small = sum(1 for im in used if max(im.get("width") or 0, im.get("height") or 0) and max(im.get("width") or 0, im.get("height") or 0) < 1024)
        if small:
            notes.append({"level": "warn", "text": f"長い辺が 1024 ピクセル未満の画像が {small} 枚あります。細かい質感が写っていないと、効果が弱くなります。"})
    near_ids = {im["id"] for im in used if any(c["key"] == "near" for c in results[im["id"]]["checks"])}
    if n >= 6 and len(near_ids) / n > 0.3:
        notes.append({"level": "warn", "text": f"よく似た画像が {len(near_ids)}/{n} 枚あります。同じ構図に固執した LoRA になりやすくなります。"})
    if n < 5:
        notes.append({"level": "info", "text": f"採用画像は {n} 枚です。少ない枚数でも学習できますが、生成で特徴が出るかは結果画面で確認してください。"})
    return {"n_used": n, "notes": notes, "counts": dict(counts), "coverage": cov}
