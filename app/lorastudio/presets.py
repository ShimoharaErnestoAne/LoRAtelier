"""Training presets per (family, kind), expressed in each engine's own option names.

* Values come from the engine documentation (sd-scripts docs/train_SDXL-en.md,
  docs/anima_train_network.md, musubi-tuner docs/krea2.md, docs/zimage.md) and are adjusted by the
  verification log (docs/VERIFICATION.md).  Engine option names are never mixed.
* The user-facing "basic" fields are a small common set; everything else is per-engine
  "advanced" and shown in the detail screen with "changed from default" markers.
* PRESET_VERSION is recorded in every run so results can be traced after preset changes.
"""
from __future__ import annotations

import math

PRESET_VERSION = "2026.10.07-2"
# -2: SDXL lr 1e-4 / alpha=dim/2 -> lr 4e-4 / alpha=dim.  Verified on Illustrious-XL v2.0 (Pepper, 20 images):
#     the old values trained to completion but the character did not appear (CCIP diff 0.28 vs base 0.40);
#     the new values reproduced the character in new poses/backgrounds (CCIP diff 0.12 at step 1200).

# Common fields shown in the detail screen: key -> (label, type, help)
FIELDS = {
    "max_train_steps": ("学習ステップ数", "int", "多すぎると素材の構図に固執し、少なすぎると特徴が弱くなります。途中保存を比較して選べます。"),
    "save_every_n_steps": ("保存間隔（ステップ）", "int", "この間隔で途中の LoRA を保存し、結果画面で比較できます。"),
    "learning_rate": ("学習率", "float", ""),
    "network_dim": ("rank (network_dim)", "int", "大きいほど表現力が増えファイルも大きくなります。"),
    "network_alpha": ("alpha", "float", ""),
    "resolution": ("学習解像度", "int", "縦横比は保持し、エンジンの解像度別振り分け (bucket) を使います。"),
    "train_batch_size": ("バッチサイズ", "int", ""),
    "seed": ("seed", "int", ""),
    "optimizer_type": ("オプティマイザー", "str", ""),
    "lr_scheduler": ("学習率スケジューラー", "str", ""),
    "blocks_to_swap": ("ブロックスワップ数", "int", "VRAM が足りないときに増やします（遅くなります）。"),
    "caption_dropout_rate": ("説明文ドロップアウト率", "float", ""),
    # MOSS-Image only: chosen from the GPU's VRAM (sd-scripts docs/moss_image_train_network.md 7.1)
    "convrot_int8": ("本体を INT8 で読み込む（1 = はい、0 = いいえ）", "int",
                     "1 にすると、学習中の本体の VRAM が約半分になります。INT8 版のファイルを使っているときは、この値に関係なく INT8 で読み込みます。"),
    "convrot_int8_text_encoder": ("テキストエンコーダーを INT8 で読み込む（1 = はい、0 = いいえ）", "int",
                                  "1 にすると、説明文の前処理に使う VRAM が約半分（約 16GB → 8GB）になります。"),
    # enhance (texture) LoRA only
    "copy_steps": ("下準備の学習ステップ数", "int", "画像の組から作るとき、加工前の側を覚え込ませる下準備の長さです。短すぎると、質感以外の違いまで LoRA に入ります。"),
    "leco_guidance": ("効果の強さ（学習時）", "float", "言葉から作るとき、大きいほど効果の強い LoRA になります（目安 1〜3）。"),
}

_BASE_STEPS = {("sdxl", "character"): 1600, ("sdxl", "style"): 2400,
               ("anima", "character"): 1600, ("anima", "style"): 2400,
               ("krea2", "character"): 1200, ("krea2", "style"): 1600,
               ("qwen_image", "character"): 1200, ("qwen_image", "style"): 1600,
               ("qwen_image21", "character"): 1200, ("qwen_image21", "style"): 1600,
               ("zimage", "character"): 1200, ("zimage", "style"): 1600,
               ("moss_image", "character"): 1200, ("moss_image", "style"): 1600}


def auto_steps(family: str, kind: str, n_images: int) -> int:
    """Steps grow slowly with dataset size; the run saves several checkpoints so both
    under- and over-training can be checked on the results screen (never "fixed by count")."""
    # concept (body type etc.): same as a character for now - one feature shared by all images (unverified)
    base = _BASE_STEPS[(family, "character" if kind == "concept" else kind)]
    factor = min(1.6, max(0.8, (max(n_images, 1) / 20) ** 0.3))
    return int(round(base * factor / 100) * 100)


def defaults(family: str, kind: str, variant: str, n_images: int, vram_mb: int | None, method: str | None = None) -> dict:
    if kind == "enhance":
        return _enhance_defaults(family, method, n_images, vram_mb)
    steps = auto_steps(family, kind, n_images)
    save_every = max(100, int(round(steps / 8 / 50) * 50))
    d = {"max_train_steps": steps, "save_every_n_steps": save_every, "seed": 42, "resolution": 1024,
         "caption_dropout_rate": 0.0}
    if family == "sdxl":
        d.update(learning_rate=4e-4, network_dim=32 if kind == "style" else 16,
                 network_alpha=32 if kind == "style" else 16, train_batch_size=2,
                 optimizer_type="AdamW8bit", lr_scheduler="cosine_with_restarts")
        if vram_mb and vram_mb < 11000:
            d["train_batch_size"] = 1
    elif family == "anima":
        d.update(learning_rate=1e-4, network_dim=32 if kind == "style" else 16,
                 network_alpha=32 if kind == "style" else 16, train_batch_size=2,
                 optimizer_type="AdamW8bit", lr_scheduler="constant", blocks_to_swap=0)
        if vram_mb and vram_mb < 14000:
            d.update(train_batch_size=1, blocks_to_swap=10)
    elif family == "krea2":
        # 12.9B DiT: fp8 (scaled) base + block swap is required on a 16GB card
        swap = 0
        if vram_mb:
            swap = 26 if vram_mb < 13000 else (16 if vram_mb < 20000 else (6 if vram_mb < 30000 else 0))
        d.update(learning_rate=1e-4, network_dim=32, network_alpha=32, train_batch_size=1,
                 optimizer_type="adamw8bit", lr_scheduler="constant", blocks_to_swap=swap)
    elif family == "qwen_image":
        # 20B DiT (60 blocks), fp8 scaled.  musubi-tuner docs/qwen_image.md: 24GB with swap 16, 12GB with swap 45;
        # lr 5e-5, dim 16.  Swap for 16GB interpolated, unverified.
        swap = 0
        if vram_mb:
            swap = 45 if vram_mb < 14000 else (36 if vram_mb < 20000 else (16 if vram_mb < 30000 else 0))
        d.update(learning_rate=5e-5, network_dim=16, network_alpha=16, train_batch_size=1,
                 optimizer_type="adamw8bit", lr_scheduler="constant", blocks_to_swap=swap)
    elif family == "qwen_image21":
        # Experimental (PR #1145 docs): lr 1e-4, dim/alpha 16, swap < 32.  Swap values are guesses, unverified.
        swap = 0
        if vram_mb:
            swap = 24 if vram_mb < 14000 else (16 if vram_mb < 20000 else (8 if vram_mb < 30000 else 0))
        d.update(learning_rate=1e-4, network_dim=16, network_alpha=16, train_batch_size=1,
                 optimizer_type="adamw8bit", lr_scheduler="constant", blocks_to_swap=swap)
    elif family == "zimage":
        # 6B DiT (30 layers + refiners), fp8 scaled.  musubi-tuner docs/zimage.md example: lr 1e-4, dim 32,
        # flow shift 2.0.  Block swap below 14GB is a guess, unverified.
        d.update(learning_rate=1e-4, network_dim=32, network_alpha=32, train_batch_size=1,
                 optimizer_type="adamw8bit", lr_scheduler="constant", blocks_to_swap=0)
        if vram_mb and vram_mb < 14000:
            d["blocks_to_swap"] = 14
    elif family == "moss_image":
        # 5B DiT (42 blocks), sd-scripts docs/moss_image_train_network.md: lr 1e-4 for alpha 1, dim 16, AdamW8bit.
        # VRAM guide (measured there on an RTX PRO 6000): 24GB bf16 DiT, 16GB ConvRot INT8 for DiT and text encoder
        # (9.1GB), 12GB INT8 + block swap 20-36, 8GB not supported.  MOSS-LoRA-Trainer v0.1.2 (推奨モデル.txt)
        # measured ~10.8 / 12GB with INT8 files and swap 16 on an RTX 4070, so 12GB cards get 16; cards below
        # 11.5GB keep the docs' safer 28 (a guess).  None of these has run in this tool.
        int8_dit, int8_te, swap = 1, 1, 0
        if vram_mb:
            int8_dit = 0 if vram_mb >= 19000 else 1
            int8_te = 0 if vram_mb >= 23000 else 1
            swap = 28 if vram_mb < 11500 else 16 if vram_mb < 13500 else 0
        d.update(learning_rate=1e-4, network_dim=16, network_alpha=1.0, train_batch_size=1, optimizer_type="AdamW8bit",
                 lr_scheduler="constant", blocks_to_swap=swap, convrot_int8=int8_dit, convrot_int8_text_encoder=int8_te)
    if family in NO_CAPTION_DROPOUT:
        del d["caption_dropout_rate"]   # not shown, not varied in まとめて予約
    return d


# musubi-tuner families: the dataset config written for them (training._ds_musubi) has no caption dropout
NO_CAPTION_DROPOUT = ("krea2", "qwen_image", "qwen_image21", "zimage")


def _enhance_defaults(family: str, method: str | None, n_images: int, vram_mb: int | None) -> dict:
    """Texture / enhancement LoRA (enhance.py), SDXL only.  Starting points, unverified:
    * words (LECO): sd-scripts docs/train_leco.md "Recommended Starting Settings" for SDXL
      (dim 8-16, lr 1e-4, 500-2000 steps, guidance 1-3).
    * pair: the "copy" LoRA must reproduce the processed images closely, so it gets the same length as a
      character run (which reproduced characters at lr 4e-4 / 1200-1600 steps on Illustrious, VERIFICATION.md);
      the difference LoRA is trained the same way on top of the merged model."""
    if method == "words":
        # Each LECO step first generates part of an image (up to max_denoising_steps with CFG) and then learns, so a
        # step is far heavier than in image training: measured ~18 s/step at 1024px / 40 denoising steps on a
        # 16GB GPU.  768px and 20 denoising steps (training.py): ~9.4 s/step measured, 500 steps ~80 min.
        return {"max_train_steps": 500, "save_every_n_steps": 100, "seed": 42, "resolution": 768, "learning_rate": 1e-4,
                "network_dim": 8, "network_alpha": 4, "optimizer_type": "AdamW8bit", "lr_scheduler": "constant",
                "leco_guidance": 1.5}
    steps = auto_steps(family, "character", n_images)
    d = {"max_train_steps": steps, "save_every_n_steps": max(100, int(round(steps / 6 / 50) * 50)), "seed": 42,
         "resolution": 1024, "caption_dropout_rate": 0.0, "learning_rate": 4e-4, "network_dim": 16, "network_alpha": 16,
         "train_batch_size": 2, "optimizer_type": "AdamW8bit", "lr_scheduler": "constant", "copy_steps": steps}
    if vram_mb and vram_mb < 11000:
        d["train_batch_size"] = 1
    return d


def merged(defaults_: dict, overrides: dict) -> tuple[dict, list[str]]:
    """Apply user overrides; returns (values, list of changed keys).  Unknown keys are ignored.
    "steps_scale" / "save_count" (set when settings were copied from elsewhere) keep the step count
    relative to the automatic value, so it still follows the number of images."""
    ov = overrides or {}
    out = dict(defaults_)
    changed = []
    for k, v in ov.items():
        if k in out and v is not None and v != "" and v != out[k]:
            out[k] = v
            changed.append(k)
    if ov.get("steps_scale") and "max_train_steps" not in ov:
        s = max(100, int(round(defaults_["max_train_steps"] * float(ov["steps_scale"]) / 50) * 50))
        if s != out["max_train_steps"]:
            out["max_train_steps"] = s
            changed.append("max_train_steps")
    if ov.get("save_count") and "save_every_n_steps" not in ov:
        e = max(10, min(out["max_train_steps"], int(round(out["max_train_steps"] / float(ov["save_count"]) / 10) * 10)))
        if e != out["save_every_n_steps"]:
            out["save_every_n_steps"] = e
            changed.append("save_every_n_steps")
    return out, changed


# Settings whose sensible values depend on the model family (engine docs give different values per family).
# They are copied only between projects of the same family.
FAMILY_KEYS = {"learning_rate", "network_dim", "network_alpha", "train_batch_size", "optimizer_type",
               "lr_scheduler", "blocks_to_swap", "convrot_int8", "convrot_int8_text_encoder"}
_STEP_KEYS = {"max_train_steps", "save_every_n_steps"}


def snapshot(family: str, defaults_: dict, overrides: dict) -> dict:
    """The user's changes in a form that can be applied to another project (or kept as a preset).
    Step count and save interval are stored both relative (scale / number of saves) and as fixed values."""
    values, changed = merged(defaults_, overrides)
    snap = {"family": family, "values": {k: values[k] for k in changed if k not in _STEP_KEYS}}
    if "max_train_steps" in changed:
        snap["steps_scale"] = round(values["max_train_steps"] / defaults_["max_train_steps"], 3)
    if "save_every_n_steps" in changed:
        snap["save_count"] = round(values["max_train_steps"] / values["save_every_n_steps"], 3)
    if _STEP_KEYS & set(changed):
        snap["fixed_steps"] = {k: values[k] for k in _STEP_KEYS}   # both, as they were in the source
    return snap


def apply_snapshot(snap: dict, family: str, fixed_steps: bool = False) -> tuple[dict, list[str]]:
    """Overrides for a project of `family` built from a snapshot; returns (overrides, skipped keys).
    Family-dependent keys are skipped when the families differ (the target family's defaults stay)."""
    same = snap.get("family") == family
    ov, skipped = {}, []
    for k, v in (snap.get("values") or {}).items():
        if k not in FIELDS:
            continue
        if k in FAMILY_KEYS and not same:
            skipped.append(k)
        else:
            ov[k] = v
    if fixed_steps and snap.get("fixed_steps"):
        ov.update({k: v for k, v in snap["fixed_steps"].items() if k in _STEP_KEYS})
    else:
        for k in ("steps_scale", "save_count"):
            if snap.get(k):
                ov[k] = snap[k]
    return ov, skipped


# families whose engine can keep part of the model in system memory while training (blocks_to_swap): the largest value
BLOCK_SWAP_MAX = {"anima": 26, "krea2": 26, "qwen_image": 58, "qwen_image21": 31, "zimage": 28, "moss_image": 40}


def validate(family: str, values: dict) -> list[str]:
    """Reject inconsistent settings instead of sending them to the engine."""
    errs = []
    if values["max_train_steps"] < 50:
        errs.append("学習ステップ数は 50 以上にしてください。")
    if values["save_every_n_steps"] <= 0 or values["save_every_n_steps"] > values["max_train_steps"]:
        errs.append("保存間隔は 1 以上、学習ステップ数以下にしてください。")
    if not (1e-7 <= float(values["learning_rate"]) <= 1e-2):
        errs.append("学習率が範囲外です (1e-7 〜 1e-2)。")
    if int(values["network_dim"]) < 1 or int(values["network_dim"]) > 256:
        errs.append("rank は 1〜256 にしてください。")
    if "resolution" in values and (values["resolution"] % 64 or not (512 <= values["resolution"] <= 2048)):
        errs.append("学習解像度は 512〜2048 の 64 の倍数にしてください。")
    if values.get("train_batch_size", 1) < 1:
        errs.append("バッチサイズは 1 以上にしてください。")
    bmax = BLOCK_SWAP_MAX.get(family)
    if bmax is not None and not (0 <= int(values.get("blocks_to_swap", 0)) <= bmax):
        errs.append(f"ブロックスワップ数は 0〜{bmax} にしてください。")
    if family == "krea2" and int(values.get("train_batch_size", 1)) > 1 and int(values.get("blocks_to_swap", 0)) > 0:
        pass  # allowed, just slow
    for k in ("convrot_int8", "convrot_int8_text_encoder"):
        if k in values and int(values[k]) not in (0, 1):
            errs.append(f"「{FIELDS[k][0]}」は 0 か 1 にしてください。")
    if "copy_steps" in values and int(values["copy_steps"]) < 50:
        errs.append("下準備の学習ステップ数は 50 以上にしてください。")
    if "leco_guidance" in values and not (0.1 <= float(values["leco_guidance"]) <= 10):
        errs.append("効果の強さ（学習時）は 0.1〜10 にしてください。")
    return errs


def estimate_disk_gb(family: str, values: dict) -> float:
    """Rough upper bound of disk use for a run (checkpoints + last 2 resume states + caches).  musubi-tuner runs and
    texture LoRA runs from image pairs save no resume states (training.saves_state)."""
    per_rank_mb = {"sdxl": 7.0, "anima": 4.5, "krea2": 20.0, "qwen_image": 36.0, "qwen_image21": 18.0, "zimage": 9.0,
                   "moss_image": 9.0}[family]
    lora_mb = per_rank_mb * int(values["network_dim"])
    n_ckpt = math.ceil(values["max_train_steps"] / max(1, int(values["save_every_n_steps"]))) + 1   # 0: validate() reports it
    state_mb = 0 if family in ("krea2", "qwen_image", "qwen_image21", "zimage") or "copy_steps" in values else lora_mb * 3 * 2
    cache_mb = {"sdxl": 600, "anima": 800, "krea2": 3000, "qwen_image": 3000, "qwen_image21": 3000, "zimage": 1500,
                "moss_image": 2000}[family]
    if "copy_steps" in values:   # texture LoRA from image pairs: a merged base model (fp16) lives in the run until it ends
        cache_mb += 7200 + 600
    return round((lora_mb * n_ckpt + state_mb + cache_mb) / 1024 + 1.0, 1)
