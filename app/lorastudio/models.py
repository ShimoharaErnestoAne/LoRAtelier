"""Base-model registry, family detection and model hashing.

A model family decides the training engine.  LoRAs are never treated as portable across
families (SD1.5 / SDXL / Anima / Krea2 are mutually incompatible).
"""
from __future__ import annotations

import hashlib
import json
import re
import struct
import threading
import uuid
from pathlib import Path

from . import config

# verified: whether this variant has actually been trained + loaded in ComfyUI in this repo's
# verification log (docs/VERIFICATION.md).  Unverified variants are shown with a warning.
# experimental: added on trial with an engine that does not officially support it yet; the UI says it may not work.
VARIANTS = {
    "illustrious": {"family": "sdxl", "label": "Illustrious XL 系 (SDXL)", "caption": "tags", "prediction": "eps", "verified": True},
    "noobai_eps": {"family": "sdxl", "label": "NoobAI-XL eps 系 (SDXL)", "caption": "tags", "prediction": "eps", "verified": False},
    "noobai_vpred": {"family": "sdxl", "label": "NoobAI-XL v-pred 系 (SDXL)", "caption": "tags", "prediction": "v", "verified": False},
    "pony": {"family": "sdxl", "label": "Pony Diffusion V6 系 (SDXL)", "caption": "tags", "prediction": "eps", "verified": True},
    "sdxl_base": {"family": "sdxl", "label": "SDXL 1.0 本家 / 実写系", "caption": "natural", "prediction": "eps", "verified": True},
    "anima": {"family": "anima", "label": "Anima (CircleStone)", "caption": "tags", "prediction": "flow", "verified": True,
              "natural_ok": True},   # also reads natural language: the project's 説明文の作り方 picks tags or sentences
    # Gazingstars123/Anima-2.9B: Anima with 12 inserted blocks (40).  Same text encoder / VAE; ComfyUI counts the blocks;
    # sd-scripts fixes them at 28, so training goes through bridges/anima_blocks.py.  Not trained through this app yet.
    "anima29": {"family": "anima", "label": "Anima 2.9B（40 層）", "caption": "tags", "prediction": "flow", "verified": False,
                "natural_ok": True},
    # Qwen-Image / 2.1 / Z-Image / MOSS-Image: "verified" stays False until training is recorded in
    # docs/VERIFICATION.md (generation of Qwen 2512 fp8, 2.1 GGUF and Z-Image Turbo is).  Qwen and Z-Image are no
    # longer marked experimental.
    "qwen_image": {"family": "qwen_image", "label": "Qwen-Image（2512 など）", "caption": "natural",
                   "prediction": "flow", "verified": False},
    "qwen_image21": {"family": "qwen_image21", "label": "Qwen-Image-2.1", "caption": "natural",
                     "prediction": "flow", "verified": False},
    # Turbo / fine-tuned quantized files serve generation (2026-10-11; works, 2026-10-19).  Training (musubi-tuner
    # docs/krea2.md) needs the bf16 RAW DiT, 26GB and HF-gated; offered since 2026-10-28 after training
    # Krea 2 was reported to work.  The LoRA keeps the checkpoint's module names (blocks.*, txtfusion.*), which ComfyUI maps
    # as lora_unet_* keys: copied, not converted.  Training through this app is unverified.
    "krea2": {"family": "krea2", "label": "Krea 2", "caption": "natural", "prediction": "flow", "verified": False},
    # Z-Image.  Turbo and Base share the DiT structure, so the file cannot tell them
    # apart; the variant (from the file name, else chosen by the user) decides sampling and whether the
    # training adapter is needed.  Turbo is trained with ostris's training adapter merged in (musubi-tuner
    # docs/zimage.md), Base directly.
    "zimage_turbo": {"family": "zimage", "label": "Z-Image Turbo 系", "caption": "natural",
                     "prediction": "flow", "verified": False},
    "zimage_base": {"family": "zimage", "label": "Z-Image Base 系", "caption": "natural",
                    "prediction": "flow", "verified": False},
    # MOSS-Image, trained with sd-scripts v0.12.1 (docs/moss_image_train_network.md).
    # SFT and Base share the DiT structure: the variant comes from the file name, else the user picks it.
    "moss_image_sft": {"family": "moss_image", "label": "MOSS-Image SFT（試験導入・動作未確認）", "caption": "natural",
                       "prediction": "flow", "verified": False, "experimental": True},
    "moss_image_base": {"family": "moss_image", "label": "MOSS-Image Base（試験導入・動作未確認）", "caption": "natural",
                        "prediction": "flow", "verified": False, "experimental": True},
}
UNAVAILABLE_NOTE: dict[str, str] = {}
KREA2_NOTE = ("Krea 2 の学習には RAW 版の本体（bf16、約 26GB）が必要です。Turbo 版や量子化した版は、画像生成と素材の生成にだけ使えます。"
              "学習した LoRA は Turbo 版で使う想定です。")
QWEN_IMAGE_NOTE = ("Qwen-Image-2512 は 20B の大きなモデルです。学習には bf16 版の本体とテキストエンコーダー（合わせて約 57GB）が必要で、"
                   "学習中はメインメモリを多く使います（64GB を推奨）。")
ANIMA29_NOTE = ("Anima 2.9B は、Anima の層を 28 から 40 に増やしたモデルです（試作版）。テキストエンコーダーと VAE は Anima と同じものを使います。"
                "Anima（2B）で作った LoRA は層の並びが違うため、効き方が変わることがあります。")
ZIMAGE_NOTE = "Turbo 系で学習するときは、Turbo 用の学習アダプター（約 0.3GB）を使います。"
MOSS_NOTE = ("MOSS-Image は試験導入中です。このツールでは動作を確認していないため、生成や学習が動かない可能性があります。"
             "学習には MOSS-Image に対応した新しい版の学習エンジン（sd-scripts v0.12.1）を別に入れて使います。")
QWEN21_NOTE = ("学習には、Qwen-Image-2.1 に対応した学習エンジン（musubi-tuner の開発版）を別に入れて使います。"
               "Qwen-Image-2.1 は Qwen Research License のモデルで、商用には使えません。")


VARIANTS["qwen_image"]["note"] = QWEN_IMAGE_NOTE
VARIANTS["qwen_image21"]["note"] = QWEN21_NOTE
VARIANTS["krea2"]["note"] = KREA2_NOTE
VARIANTS["zimage_turbo"]["note"] = ZIMAGE_NOTE
VARIANTS["zimage_base"]["note"] = ZIMAGE_NOTE
VARIANTS["anima29"]["note"] = ANIMA29_NOTE
VARIANTS["moss_image_sft"]["note"] = MOSS_NOTE
VARIANTS["moss_image_base"]["note"] = MOSS_NOTE


def experimental_note(variant: str | None) -> str | None:
    v = VARIANTS.get(variant or "", {})
    return v.get("note") if v.get("experimental") else None


def available(variant: str) -> bool:
    return VARIANTS.get(variant, {}).get("available", True)


def trainable(variant: str) -> bool:
    return available(variant) and VARIANTS.get(variant, {}).get("train", True)

FAMILY_ENGINE = {"sdxl": "sd-scripts", "anima": "sd-scripts", "krea2": "musubi-tuner", "qwen_image": "musubi-tuner",
                 "qwen_image21": "musubi-tuner-qwen21", "zimage": "musubi-tuner", "moss_image": "sd-scripts-moss"}

# files each family needs, beyond the main checkpoint: (key, label, required)
FAMILY_FILES = {
    "sdxl": [("ckpt", "SDXL チェックポイント (.safetensors)", True)],
    "anima": [("ckpt", "Anima DiT (.safetensors)", True), ("te", "Qwen3-0.6B テキストエンコーダー", True),
              ("vae", "Qwen-Image VAE", True)],
    "qwen_image": [("ckpt", "Qwen-Image DiT（学習には bf16 版が必要）", False), ("gen", "生成用の軽量版（GGUF、任意）", False),
                   ("te", "Qwen2.5-VL-7B テキストエンコーダー", True), ("vae", "Qwen-Image VAE", True),
                   ("train_te", "学習用テキストエンコーダー（bf16 の qwen_2.5_vl_7b、学習に必要）", False)],
    "qwen_image21": [("ckpt", "Qwen-Image-2.1 DiT（bf16 版か INT8 版、学習に必要）", False), ("gen", "生成用の軽量版（GGUF、任意）", False),
                     ("te", "Qwen3-VL-8B テキストエンコーダー（ComfyUI 用）", True), ("vae", "Qwen-Image-2.1 VAE（ComfyUI 用）", True),
                     ("train_te", "学習用テキストエンコーダー（qwen3vl_8b の bf16 版か INT8 版、学習に必要）", False),
                     ("train_vae", "学習用 VAE（Diffusers 形式、学習に必要）", False)],
    "krea2": [("turbo", "Krea 2 Turbo など生成用の本体 (.safetensors)", False), ("te", "Qwen3-VL-4B テキストエンコーダー (.safetensors)", True),
              ("vae", "Qwen-Image VAE", True), ("ckpt", "Krea 2 RAW DiT（bf16、学習に必要）", False)],
    "zimage": [("ckpt", "Z-Image DiT（学習には bf16 版が必要）", True), ("te", "Qwen3-4B テキストエンコーダー（qwen_3_4b）", True),
               ("vae", "VAE（ae.safetensors）", True),
               ("adapter", "Turbo 用の学習アダプター（Turbo 系で学習するときに必要）", False)],
    "moss_image": [("ckpt", "MOSS-Image DiT（bf16 版または INT8 ConvRot 版）", True),
                   ("te", "Qwen3.5-9B テキストエンコーダー（qwen3.5_9b_int8_convrot など）", True),
                   ("vae", "FLUX.2 VAE（flux2-vae_small）", True),
                   ("fast", "8 ステップ高速生成 LoRA（SFT 版の生成用、任意）", False),
                   ("fast4", "4 ステップ高速生成 LoRA（SFT 版の生成用、任意）", False)],
}
# families whose registered model needs at least one of these files
FAMILY_ONE_OF = {"qwen_image": ("ckpt", "gen"), "qwen_image21": ("ckpt", "gen"), "krea2": ("turbo", "ckpt")}
# files only generation reads (the comparison falls back to the training files without them): a missing one does
# not stop a training
GEN_ONLY = {"qwen_image": ("gen",), "qwen_image21": ("gen",), "krea2": ("turbo",), "moss_image": ("fast", "fast4")}

# Models that can be added after setup (the setup itself downloads none).  "default": offered on the first
# screen as the standard set.  Files go to the ComfyUI-style folders under models/ (4th item of a file), so a
# model the tool fetched sits where the user puts their own files.  Size in GB; license as declared on the
# model page.  "needs": tokenizer files / optional engines fetched together with the model.
CATALOG = [
    {"id": "illustrious_v2", "variant": "illustrious", "name": "Illustrious-XL v2.0", "license": "creativeml-openrail-m",
     "gated": False, "default": True,
     "files": {"ckpt": ("OnomaAIResearch/Illustrious-XL-v2.0", "Illustrious-XL-v2.0.safetensors", 6.94, "checkpoints")}},
    {"id": "pony_v6", "variant": "pony", "name": "Pony Diffusion V6 XL", "license": "作者の配布ページで確認（独自条件）",
     "gated": False, "default": True,
     "files": {"ckpt": ("AstraliteHeart/pony-diffusion-v6", "v6.safetensors", 6.94, "checkpoints")},
     "aliases": {"ckpt": ["ponyDiffusionV6XL.safetensors"]}},
    {"id": "sdxl_base", "variant": "sdxl_base", "name": "SDXL 1.0 base", "license": "openrail++", "gated": False, "default": True,
     "files": {"ckpt": ("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors", 6.94, "checkpoints")}},
    {"id": "anima_base", "variant": "anima", "name": "Anima base v1.0", "license": "CircleStone Labs Non-Commercial",
     "gated": False, "files": {
         "ckpt": ("circlestone-labs/Anima", "split_files/diffusion_models/anima-base-v1.0.safetensors", 4.18, "diffusion_models"),
         "te": ("circlestone-labs/Anima", "split_files/text_encoders/qwen_3_06b_base.safetensors", 1.19, "text_encoders"),
         "vae": ("circlestone-labs/Anima", "split_files/vae/qwen_image_vae.safetensors", 0.25, "vae")}},
    {"id": "anima_29b", "variant": "anima29", "name": "Anima 2.9B preview v1", "license": "CircleStone Labs Non-Commercial",
     "gated": False, "files": {
         "ckpt": ("Gazingstars123/Anima-2.9B", "Anima-2.9B-preview-v1.safetensors", 5.84, "diffusion_models"),
         "te": ("circlestone-labs/Anima", "split_files/text_encoders/qwen_3_06b_base.safetensors", 1.19, "text_encoders"),
         "vae": ("circlestone-labs/Anima", "split_files/vae/qwen_image_vae.safetensors", 0.25, "vae")}},
    # Qwen-Image-2512: the light GGUF serves generation (picked by VRAM); training needs the bf16 DiT and text encoder
    # (musubi-tuner docs/qwen_image.md: fp8 files cannot be used), added to the same model.  The id of the training
    # entry is the one the full set had before 2026-10-19, so a model added then still shows as added.
    {"id": "qwen_image_2512_gen", "variant": "qwen_image", "name": "Qwen-Image-2512 軽量版（生成用）", "license": "Apache-2.0",
     "gated": False, "merge": True,
     "note": "画像の生成と素材の生成に使う軽量版です（GGUF）。お使いの GPU の VRAM に合わせて、収まる中でいちばん品質の高い段階を選びます。"
             "LoRA を学習するには、下の「Qwen-Image-2512 学習用ファイル」も必要です。",
     "files": {
         "gen": None,   # picked from "choices"
         "te": ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", 9.38, "text_encoders"),
         "vae": ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors", 0.25, "vae")},
     # GGUF by unsloth (Apache-2.0)
     "choices": {"gen": [
         {"id": "Q3_K_M", "label": "Q3_K_M（9.9GB・VRAM 12GB 以下）", "max_vram": 13,
          "file": ("unsloth/Qwen-Image-2512-GGUF", "qwen-image-2512-Q3_K_M.gguf", 9.93, "diffusion_models")},
         {"id": "Q4_K_M", "label": "Q4_K_M（13.2GB・VRAM 16GB）", "max_vram": 19,
          "file": ("unsloth/Qwen-Image-2512-GGUF", "qwen-image-2512-Q4_K_M.gguf", 13.24, "diffusion_models")},
         {"id": "Q6_K", "label": "Q6_K（16.8GB・VRAM 20GB〜）", "max_vram": 23,
          "file": ("unsloth/Qwen-Image-2512-GGUF", "qwen-image-2512-Q6_K.gguf", 16.82, "diffusion_models")},
         {"id": "Q8_0", "label": "Q8_0（21.8GB・VRAM 24GB〜、元の品質に最も近い）", "max_vram": 999,
          "file": ("unsloth/Qwen-Image-2512-GGUF", "qwen-image-2512-Q8_0.gguf", 21.76, "diffusion_models")}]},
     "needs": {"engines": ["ComfyUI-GGUF"]}},
    {"id": "qwen_image_2512", "variant": "qwen_image", "name": "Qwen-Image-2512 学習用ファイル", "license": "Apache-2.0",
     "gated": False, "merge": True, "min_ram": 64,
     "note": "Qwen-Image-2512 で LoRA を学習するためのファイルです（本体 bf16 約 41GB・テキストエンコーダー bf16 約 17GB）。"
             "生成用の軽量版を入れてあれば、同じモデルに追加します。軽量版がなくても、このファイルで生成できます（遅くなります）。"
             "学習中は本体の一部をメインメモリに置くため、メモリ 64GB を推奨します。",
     "files": {
         "ckpt": ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/diffusion_models/qwen_image_2512_bf16.safetensors", 40.86, "diffusion_models"),
         "train_te": ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/text_encoders/qwen_2.5_vl_7b.safetensors", 16.58, "text_encoders"),
         "vae": ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors", 0.25, "vae")},
     "needs": {"tokenizers": [["Qwen/Qwen-Image", ["tokenizer/*"]]]}},
    {"id": "qwen_image21_gen", "variant": "qwen_image21", "name": "Qwen-Image-2.1 Turbo 軽量版（生成用）",
     "license": "Qwen Research License（非商用のみ）", "gated": False, "merge": True,
     "note": "画像の生成と素材の生成に使う軽量版です（GGUF、8 ステップ）。お使いの GPU の VRAM に合わせて、本体とテキストエンコーダーの版を選びます。"
             "LoRA を学習するには、下の「Qwen-Image-2.1 学習用ファイル」も必要です。" + QWEN21_NOTE,
     "files": {
         "gen": None, "te": None,   # picked from "choices"
         "vae": ("Comfy-Org/Qwen-Image-2.1", "vae/qwen_image_2.1_vae_bf16.safetensors", 0.68, "vae")},
     # GGUF of the official Turbo (unsloth).  Recommended by VRAM: the largest that fits with headroom.
     "choices": {
      "te": [
         {"id": "w4a8", "label": "テキストエンコーダー W4A8 版（6.3GB・VRAM 10GB 未満向け）", "max_vram": 10,
          "file": ("Comfy-Org/Qwen-Image-2.1", "text_encoders/qwen3vl_8b_w4a8.safetensors", 6.31, "text_encoders")},
         {"id": "int8", "label": "テキストエンコーダー INT8 版（9.4GB）", "max_vram": 999,
          "file": ("Comfy-Org/Qwen-Image-2.1", "text_encoders/qwen3vl_8b_int8_convrot.safetensors", 9.35, "text_encoders")}],
      "gen": [
         {"id": "Q4_K_M", "label": "Q4_K_M（4.2GB・VRAM 8GB〜）", "max_vram": 10,
          "file": ("unsloth/Qwen-Image-2.1-Turbo-GGUF", "qwen-image-2.1-turbo-Q4_K_M.gguf", 4.20, "diffusion_models")},
         {"id": "Q5_K_M", "label": "Q5_K_M（5.4GB・VRAM 10GB〜）", "max_vram": 14,
          "file": ("unsloth/Qwen-Image-2.1-Turbo-GGUF", "qwen-image-2.1-turbo-Q5_K_M.gguf", 5.39, "diffusion_models")},
         {"id": "Q6_K", "label": "Q6_K（6.3GB・VRAM 14GB〜）", "max_vram": 20,
          "file": ("unsloth/Qwen-Image-2.1-Turbo-GGUF", "qwen-image-2.1-turbo-Q6_K.gguf", 6.27, "diffusion_models")},
         {"id": "Q8_0", "label": "Q8_0（7.6GB・VRAM 20GB〜、元の品質に最も近い）", "max_vram": 999,
          "file": ("unsloth/Qwen-Image-2.1-Turbo-GGUF", "qwen-image-2.1-turbo-Q8_0.gguf", 7.64, "diffusion_models")}]},
     "needs": {"engines": ["ComfyUI-GGUF"]}},
    {"id": "qwen_image21_train", "variant": "qwen_image21", "name": "Qwen-Image-2.1 学習用ファイル",
     "license": "Qwen Research License（非商用のみ）", "gated": False, "merge": True, "min_ram": 32,
     "note": "Qwen-Image-2.1 で LoRA を学習するためのファイルです（本体・学習用テキストエンコーダー・学習用 VAE）。"
             "お使いの GPU の VRAM に合わせて、INT8 版（小さい）か bf16 版（元の精度）を選びます。"
             "生成用の軽量版を入れてあれば、同じモデルに追加します。" + QWEN21_NOTE,
     "files": {
         "ckpt": None, "train_te": None,   # picked from "choices"
         "train_vae": ("Qwen/Qwen-Image-2.1-Turbo", "vae/diffusion_pytorch_model.safetensors", 0.68, "vae",
                       "qwen_image_2.1_vae_diffusers_bf16.safetensors"),
         "vae": ("Comfy-Org/Qwen-Image-2.1", "vae/qwen_image_2.1_vae_bf16.safetensors", 0.68, "vae")},
     # ComfyUI's ConvRot INT8 files are read as bf16 by bridges/qwen21_int8.py (the engine takes only bf16), then
     # quantized to fp8 for training like the bf16 file: half the download for a small loss.  The INT8 text
     # encoder is the same file the generation entry picks above 10GB, so it is not fetched twice.
     "choices": {
      "ckpt": [
         {"id": "int8", "label": "本体 INT8 版（7.3GB・VRAM 20GB 以下向け）", "max_vram": 20.5,
          "file": ("Comfy-Org/Qwen-Image-2.1", "diffusion_models/qwen_image_2.1_int8_convrot.safetensors", 7.26, "diffusion_models")},
         {"id": "bf16", "label": "本体 bf16 版（14.2GB・元の精度）", "max_vram": 999,
          "file": ("Comfy-Org/Qwen-Image-2.1", "diffusion_models/qwen_image_2.1_bf16.safetensors", 14.23, "diffusion_models")}],
      "train_te": [
         {"id": "int8", "label": "学習用テキストエンコーダー INT8 版（9.4GB・VRAM 20GB 以下向け）", "max_vram": 20.5,
          "file": ("Comfy-Org/Qwen-Image-2.1", "text_encoders/qwen3vl_8b_int8_convrot.safetensors", 9.35, "text_encoders")},
         {"id": "bf16", "label": "学習用テキストエンコーダー bf16 版（17.5GB・元の精度）", "max_vram": 999,
          "file": ("Comfy-Org/Qwen-Image-2.1", "text_encoders/qwen3vl_8b_bf16.safetensors", 17.53, "text_encoders")}]},
     "needs": {"engines": ["musubi-tuner-qwen21"], "tokenizers": [["Qwen/Qwen-Image-2.1", ["processor/*"]]]}},
    {"id": "zimage_turbo", "variant": "zimage_turbo", "name": "Z-Image Turbo", "license": "Apache-2.0", "gated": False,
     "note": "8 ステップで生成できる版です。学習には下の学習アダプターも必要です（学習には bf16 版が必要なため、本体は bf16 版を取得します）。",
     "files": {
         "ckpt": ("Comfy-Org/z_image_turbo", "split_files/diffusion_models/z_image_turbo_bf16.safetensors", 12.31, "diffusion_models"),
         "te": ("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors", 8.05, "text_encoders"),
         "vae": ("Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors", 0.34, "vae")},
     "needs": {"tokenizers": [["Tongyi-MAI/Z-Image", ["tokenizer/*"]]]}},
    {"id": "zimage_adapter", "variant": "zimage_turbo", "name": "Z-Image Turbo 学習アダプター", "license": "Apache-2.0",
     "gated": False, "merge": True,
     "note": "Z-Image Turbo 系で LoRA を学習するためのアダプター（ostris 氏作）です。学習のときだけ使い、生成では使いません。"
             "Turbo 系のモデルを登録してあれば、そのモデルに追加します。",
     "files": {"adapter": ("ostris/zimage_turbo_training_adapter", "zimage_turbo_training_adapter_v2.safetensors", 0.34, "loras")},
     "needs": {"tokenizers": [["Tongyi-MAI/Z-Image", ["tokenizer/*"]]]}},
    {"id": "zimage_base", "variant": "zimage_base", "name": "Z-Image Base", "license": "Apache-2.0", "gated": False,
     "note": "学習向けの元のモデルです（学習エンジンの資料でも、学習には Base 系を勧めています）。生成には 25 ステップほどかかります。",
     "files": {
         "ckpt": ("Comfy-Org/z_image", "split_files/diffusion_models/z_image_bf16.safetensors", 12.31, "diffusion_models"),
         "te": ("Comfy-Org/z_image", "split_files/text_encoders/qwen_3_4b.safetensors", 8.05, "text_encoders"),
         "vae": ("Comfy-Org/z_image", "split_files/vae/ae.safetensors", 0.34, "vae")},
     "needs": {"tokenizers": [["Tongyi-MAI/Z-Image", ["tokenizer/*"]]]}},
    *[{"id": f"moss_image_{v}", "variant": f"moss_image_{v}", "name": f"MOSS-Image {v.upper() if v == 'sft' else 'Base'}",
       "license": "Apache-2.0", "gated": False, "experimental": True,
       "note": ("仕上げ済みの版で、画像の生成に向いています。" if v == "sft" else "仕上げ前の元のモデルです。") +
               "本体・テキストエンコーダー・VAE を合わせて約 15GB（VRAM 16GB の場合）を取得します。お使いの GPU に合わせて、"
               "VRAM 20GB 未満では本体を、24GB 未満ではテキストエンコーダーを INT8 版にします。VRAM 8GB では学習できません。" + MOSS_NOTE,
       "files": {"ckpt": None, "te": None,
                 # Apache-2.0 small-decoder VAE (the one MOSS-Image itself ships); flux2-vae.safetensors comes from FLUX.2-dev
                 "vae": ("Comfy-Org/MOSS-Image", "vae/flux2-vae_small.safetensors", 0.25, "vae")},
       "choices": {
           "ckpt": [{"id": "int8", "label": "INT8 ConvRot 版（5.4GB・VRAM 20GB 未満向け）", "max_vram": 19,
                     "file": ("Comfy-Org/MOSS-Image", f"diffusion_models/moss_image_{v}_int8_convrot.safetensors", 5.41, "diffusion_models")},
                    {"id": "bf16", "label": "bf16 版（10.2GB・VRAM 20GB 以上、元の品質）", "max_vram": 999,
                     "file": ("Comfy-Org/MOSS-Image", f"diffusion_models/moss_image_{v}_bf16.safetensors", 10.16, "diffusion_models")}],
           "te": [{"id": "int8", "label": "テキストエンコーダー INT8 版（9.7GB・VRAM 24GB 未満向け）", "max_vram": 23,
                   "file": ("Comfy-Org/MOSS-Image", "text_encoders/qwen3.5_9b_int8_convrot.safetensors", 9.75, "text_encoders")},
                  {"id": "bf16", "label": "テキストエンコーダー bf16 版（19.3GB・VRAM 24GB 以上）", "max_vram": 999,
                   "file": ("Comfy-Org/MOSS-Image", "text_encoders/qwen3.5_9b_bf16.safetensors", 19.31, "text_encoders")}]},
       "needs": {"engines": ["sd-scripts-moss"], "tokenizers": [["OpenMOSS-Team/MOSS-Image-Base", ["tokenizer/*"]]]}}
      for v in ("sft", "base")],
    # lodestones' DMD distillation of MOSS-Image SFT (work in progress): a LoRA that samples in 8 steps without CFG.
    # Model card: lora.safetensors at strength 1, or the newer lora_latest.safetensors (update 2600, 2026-10-17) at 0.7,
    # in ComfyUI's LoRA loader; the demo Space samples plain Euler on the model's own scheduler (shift 5).
    # Fetched since 2026-10-31 (the Reddit post "Moss-Image SFT + new DMD2 lora (for SFT), 8 Steps"):
    # lora_dmd2.safetensors of the Base repo ("DMD2 stage 2 (GAN) on top of DS-DMD 12000 ... stock Euler 4-8 steps
    # cfg 1", commit 2026-10-20; same SHA256 as the Civitai file that post used), at strength 1.  The older files are
    # still used when they are the ones there.  Generation only: comparisons stay on the normal 25-step sampling.
    {"id": "moss_image_sft_dmd", "variant": "moss_image_sft", "name": "MOSS-Image SFT 8 ステップ高速生成 LoRA",
     "license": "Apache-2.0", "gated": False, "merge": True, "experimental": True,
     "needs_model": "先に MOSS-Image SFT を追加してください。この LoRA は MOSS-Image SFT に追加して使います。",
     "note": "MOSS-Image SFT の画像生成と素材生成を、25 ステップから 8 ステップ（CFG なし）に減らす LoRA です（lodestones 氏作の DMD2 版。Base 用として公開されたもので、SFT でも使えるという報告があります）。"
             "生成のときだけ使い、学習と比較画像には使いません。Base 版には使えません。"
             "以前の版を取得済みのときは、この版に切り替わります。"
             "MOSS-Image SFT を登録してあれば、そのモデルに追加します。本体が INT8 版のときに効果が出るかは確認していません。",
     "files": {"fast": ("lodestones/MOSS-Image-Base-8-steps-dmd", "lora_dmd2.safetensors", 1.19, "loras",
                        "moss_image_8steps_dmd2.safetensors")}},
    # akhaliq's 4-step distillation of MOSS-Image SFT, warm-started from lodestones' 8-step DMD.
    # Model card v2.0 (2026-10-18): one LoRA for the stock flow-matching Euler sampler, ComfyUI LoraLoaderModelOnly at
    # strength 1.0, KSampler euler / simple / 4 steps / CFG 1 (its comfyui/moss-image-4step-turbo.json, on the bf16 SFT
    # DiT).  moss_image_sft_4step_v2.0 is the repo's frozen v2.0 file (lora.safetensors moves on).  Unverified here,
    # also on the INT8 DiT.
    {"id": "moss_image_sft_4step", "variant": "moss_image_sft", "name": "MOSS-Image SFT 4 ステップ高速生成 LoRA",
     "license": "Apache-2.0", "gated": False, "merge": True, "experimental": True,
     "needs_model": "先に MOSS-Image SFT を追加してください。この LoRA は MOSS-Image SFT に追加して使います。",
     "note": "MOSS-Image SFT の画像生成と素材生成を 4 ステップ（CFG なし）にする LoRA です（akhaliq 氏作、v2.0）。"
             "生成画面の「高速生成」で、8 ステップの LoRA と選べます。生成のときだけ使い、学習と比較画像には使いません。Base 版には使えません。"
             "作者は bf16 版の本体で試しています。本体が INT8 版のときに効果が出るかは確認していません。",
     "files": {"fast4": ("akhaliq/MOSS-Image-SFT-4-steps", "moss_image_sft_4step_v2.0.safetensors", 1.19, "loras")}},
    {"id": "krea2_raw", "variant": "krea2", "name": "Krea 2 RAW", "license": "Krea 2 の利用条件（モデルのページで確認）", "gated": True,
     "merge": True,
     "note": "Krea 2 の LoRA を学習するためのファイルです（RAW 版の本体と bf16 のテキストエンコーダー）。Hugging Face で利用条件に同意し、"
             "アクセストークンを入力してください。Krea 2 を登録してあれば、そのモデルに追加します。",
     "files": {"ckpt": ("krea/Krea-2-Raw", "raw.safetensors", 26.28, "diffusion_models"),
               "te": ("Comfy-Org/Qwen3-VL", "text_encoders/qwen3vl_4b_bf16.safetensors", 8.88, "text_encoders"),
               "vae": ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors", 0.25, "vae")},
     "needs": {"tokenizers": [["Qwen/Qwen3-VL-4B-Instruct", ["*.json", "*.txt", "merges*", "vocab*", "*.model", "*.jinja"]]]}},
]


def catalog_entry(cid: str) -> dict | None:
    return next((c for c in CATALOG if c["id"] == cid), None)


def strip_prefix(keys) -> list[str]:
    """DiT files are saved with or without ComfyUI's "model.diffusion_model." prefix."""
    pre = "model.diffusion_model."
    return [k[len(pre):] if k.startswith(pre) else k for k in keys]


def read_safetensors_header(path: Path) -> dict:
    """Read only the JSON header (cheap even for 26GB files)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 200_000_000:
            raise ValueError("header too large")
        hdr = json.loads(f.read(n))
    if not isinstance(hdr, dict):
        raise ValueError("safetensors のヘッダーではありません")
    return hdr


def detect(path: Path) -> dict:
    """Guess family/variant from tensor names and metadata.  Returns confidence and evidence;
    the user confirms the variant (e.g. Illustrious vs Pony cannot be told from keys)."""
    path = Path(path)
    if path.suffix.lower() != ".safetensors":
        return {"family": None, "evidence": "safetensors 以外の形式は未対応です", "confidence": "低"}
    return detect_header(read_safetensors_header(path), path.name)


def detect_header(hdr: dict, name: str = "") -> dict:
    """detect() on an already-read safetensors header (also used before downloading, from the first
    bytes of a remote file).  name: file name, used only as a hint."""
    hdr = dict(hdr)
    meta = hdr.pop("__metadata__", {}) or {}
    keys = list(hdr.keys())
    ks = set(keys)
    res = {"family": None, "variant": None, "confidence": "低", "evidence": "", "n_tensors": len(keys),
           "dtypes": sorted({v.get("dtype") for v in hdr.values() if isinstance(v, dict)})}
    if any(k.startswith("conditioner.embedders.1.model.") for k in keys) and any(k.startswith("model.diffusion_model.") for k in keys):
        res.update(family="sdxl", confidence="高", evidence="SDXL の UNet/CLIP-G のキー構造")
        if "v_pred" in ks or "edm_vpred.sigma_max" in ks:
            res.update(variant="noobai_vpred", evidence=res["evidence"] + "、v-prediction 指定キーあり")
        else:
            title = (json.dumps(meta, ensure_ascii=False) + " " + name).lower()   # file name is only a hint
            res["variant"] = "pony" if "pony" in title else "illustrious" if "illustrious" in title else                 "sdxl_base" if "sd_xl_base" in title else None
        return res
    if any("llm_adapter" in k for k in keys):
        res.update(family="anima", variant="anima", confidence="高", evidence="LLM Adapter を持つ DiT (Anima)")
        blocks = {int(m.group(1)) for k in strip_prefix(keys) for m in [re.match(r"(?:net\.)?blocks\.(\d+)\.", k)] if m}
        if blocks and max(blocks) + 1 > 28:
            res.update(variant="anima29", evidence=f"LLM Adapter を持つ DiT (Anima、{max(blocks) + 1} 層)")
        return res
    bare = set(strip_prefix(keys))
    if {"txt_in.text_norm.weight", "modulation.1.weight", "img_in.weight"} <= bare:
        res.update(family="qwen_image21", variant="qwen_image21", confidence="中", evidence="Qwen-Image-2.1 の DiT 構造",
                   note=QWEN21_NOTE)
        return res
    if "txt_norm.weight" in bare and "img_in.weight" in bare and any(k.startswith("transformer_blocks.") for k in bare):
        # the fp8 edit files carry no edit-only tensor; their name ("qwen_image_edit_...") is the only sign
        if "__index_timestep_zero__" in bare or "time_text_embed.addition_t_embedding.weight" in bare or "edit" in name.lower():
            res.update(evidence="Qwen-Image の編集用 / レイヤー分割用モデルです（このツールでは未対応）")
            return res
        res.update(family="qwen_image", variant="qwen_image", confidence="高", evidence="Qwen-Image の DiT 構造（2512 などの画像生成用）",
                   note=QWEN_IMAGE_NOTE)
        if any(str(v.get("dtype", "")).startswith("F8") for v in hdr.values() if isinstance(v, dict)):
            res["note"] = "fp8 版です。生成には使えますが、学習には bf16 版が必要です。" + QWEN_IMAGE_NOTE
            res["fp8"] = True
        return res
    if {"cap_embedder.1.weight", "context_refiner.0.attention.qkv.weight", "noise_refiner.0.attention.qkv.weight"} <= bare:
        # Lumina 2 has the same tensor names; Z-Image is the one with width 3840 (ComfyUI model_detection.py)
        pre = "" if "cap_embedder.1.weight" in hdr else "model.diffusion_model."
        if (hdr[pre + "cap_embedder.1.weight"].get("shape") or [0])[0] != 3840:
            res.update(evidence="Lumina 2 系のモデルです（このツールでは未対応）")
            return res
        if "dec_net.cond_embed.weight" in bare:
            res.update(evidence="Z-Image のピクセル空間版です（このツールでは未対応）")
            return res
        n = name.lower().replace("-", "_")
        variant = "zimage_base" if "base" in n or "de_turbo" in n or "deturbo" in n else "zimage_turbo" if "turbo" in n else None
        res.update(family="zimage", variant=variant, confidence="高", evidence="Z-Image の DiT 構造", note=ZIMAGE_NOTE)
        if variant is None:
            res["evidence"] += "（Turbo 系か Base 系かはファイルから判別できません）"
        if any(str(v.get("dtype", "")).startswith("F8") for v in hdr.values() if isinstance(v, dict)):
            res["note"] = "fp8 版です。生成には使えますが、学習には bf16 版が必要です。" + ZIMAGE_NOTE
            res["fp8"] = True
        return res
    moss = [k[len(p):] for k in keys for p in ("model.diffusion_model.", "diffusion_model.", "") if k.startswith(p)]
    if {"shared_block_modulation.weight", "layers.0.proj.ffn_up_gate.weight"} <= set(moss):
        n = name.lower()
        variant = "moss_image_base" if "base" in n else "moss_image_sft" if "sft" in n else None
        res.update(family="moss_image", variant=variant, confidence="高", evidence="MOSS-Image の DiT 構造", note=MOSS_NOTE)
        if variant is None:
            res["evidence"] += "（SFT 版か Base 版かはファイルから判別できません）"
        # ConvRot INT8 (ComfyUI *_int8_convrot): int8 weights with weight_scale / comfy_quant tensors
        res["int8"] = any(k.endswith((".weight_scale", ".comfy_quant")) for k in keys) or \
            any(str(v.get("dtype", "")) == "I8" for v in hdr.values() if isinstance(v, dict))
        return res
    if any(k.startswith(("txtfusion.", "tmlp.", "txtmlp.")) or ".txtfusion." in k for k in keys):
        res.update(family="krea2", variant="krea2", confidence="中", evidence="Krea 2 系のテキスト融合層")
        res["note"] = KREA2_NOTE
        # fp8 / int8 files are Turbo or fine-tunes made for generation; RAW (training) is bf16
        res["quantized"] = any(str(d).startswith(("F8", "I8")) for d in res["dtypes"]) or "turbo" in name.lower()
        return res
    if any(k.startswith("model.diffusion_model.") for k in keys):
        res.update(evidence="SDXL 以外の Stable Diffusion 系 (SD1.5 等) の可能性。未対応です")
    elif any(k.startswith(("lora_unet_", "lora_te")) or ".lora_down." in k or ".lora_A." in k for k in keys):
        res.update(evidence="LoRA ファイルです。ベースモデル（チェックポイント）を指定してください")
    else:
        res.update(evidence="SDXL / Anima / Qwen-Image / Z-Image のモデル構造が見つかりません")
    return res


def detect_lora(path: Path) -> dict:
    """Which model family a LoRA file was made for.  Metadata first (sd-scripts / modelspec), then
    key names.  family=None means unknown: such a LoRA is not rejected, only a sure mismatch is."""
    return detect_lora_header(read_safetensors_header(Path(path)), Path(path).name)


def is_lora_keys(keys) -> bool:
    """LoRA-type files: LoRA (kohya / diffusers / PEFT, and the older diffusers ".lora.down" layout ComfyUI also reads,
    e.g. lightx2v's pytorch_lora_weights.safetensors), LyCORIS LoKr / LoHa, and ComfyUI weight-diff patches."""
    return any(".lora_down." in k or ".lora_up." in k or ".lora_A." in k or ".lora_B." in k or ".lora.down." in k
               or ".lora.up." in k or ".lokr_w1" in k or ".hada_w1" in k or k.endswith((".diff", ".diff_b", ".lora_A", ".lora_B"))
               for k in keys)   # ".lora_A" at the end: MOSS-Image edit LoRAs as published (layers.N.proj.q.lora_A)


# MOSS-Image edit LoRAs (format "moss-edit-lora" in the file's adapter_config) and the reference layout each was trained
# with, by ComfyUI-Moss-Image-Edit-Native's edit_format names: chinoll's Moss-Image-Edit (and Comfy-Org's copy) carry
# its trainer's model_config; akhaliq's MOSS-Image-Edit-HQ (ai-toolkit) does not, and from armV5 on was trained with
# the text encoder seeing the images at 512 px
EDIT_CHINOLL = "Moss-Image-Edit (chinoll)"
EDIT_HQ = "ai-toolkit (MOSS-Image-Edit-HQ)"
EDIT_HQ512 = "ai-toolkit, 512 px vision (MOSS-Image-Edit-HQ v5)"
EDIT_LABEL = {EDIT_CHINOLL: "Moss-Image-Edit（chinoll）", EDIT_HQ: "MOSS-Image-Edit-HQ", EDIT_HQ512: "MOSS-Image-Edit-HQ v5 以降"}


def moss_edit_format(meta: dict, name: str = "") -> str | None:
    try:
        cfg = json.loads((meta or {}).get("adapter_config") or "null")
    except ValueError:
        return None
    if not isinstance(cfg, dict) or cfg.get("format") != "moss-edit-lora":
        return None
    if "model_config" in (cfg.get("metadata") or {}):
        return EDIT_CHINOLL
    v = re.search(r"armv(\d+)", name.lower())
    return EDIT_HQ512 if v and int(v.group(1)) >= 5 else EDIT_HQ


# sd-scripts lora_generic keys of the MOSS-Image DiT (layers.N.proj.q / k / v / o / ffn_up_gate / ffn_down)
_MOSS_LORA_KEY = re.compile(r"^lora_unet_layers_\d+_proj_(q|k|v|o|ffn_up_gate|ffn_down)\.")


def detect_lora_header(hdr: dict, name: str = "") -> dict:
    hdr = dict(hdr)
    meta = hdr.pop("__metadata__", {}) or {}
    keys = list(hdr.keys())
    arch = (meta.get("modelspec.architecture") or "").lower()
    base = (meta.get("ss_base_model_version") or "").lower()
    if "anima" in arch or "anima" in base or any("llm_adapter" in k for k in keys):
        return {"family": "anima", "evidence": "Anima 用"}
    fmt = moss_edit_format(meta, name)   # the file says so itself (also for edit LoRAs on attention layers only)
    if fmt:
        return {"family": "moss_image", "evidence": f"MOSS-Image 用の編集 LoRA（{EDIT_LABEL[fmt]}）", "edit_format": fmt}
    if arch.startswith("moss-image") or any(_MOSS_LORA_KEY.match(k) or ".proj.ffn_up_gate." in k for k in keys):
        return {"family": "moss_image", "evidence": "MOSS-Image 用"}
    if "xl" in arch or base.startswith("sdxl") or any(k.startswith(("lora_te2_", "text_encoder_2.", "lora_unet_input_blocks_",
                                                                       "lora_unet_output_blocks_", "lora_unet_middle_block_")) for k in keys):
        return {"family": "sdxl", "evidence": "SDXL 用"}
    if "v1" in arch or "v2" in arch or base.startswith(("sd_v1", "sd_v2")) or any(k.startswith(("lora_te_", "lora_unet_down_blocks_")) for k in keys):
        return {"family": "sd15", "evidence": "SD1.5 / SD2 用（SDXL 系では使えません）"}
    if "krea2" in arch or "krea2" in base or any(".txtfusion." in k for k in keys):
        return {"family": "krea2", "evidence": "Krea 2 用"}
    if "z-image" in arch or "zimage" in arch or any(
            k.startswith(("lora_unet_layers_", "lora_unet_context_refiner_", "lora_unet_noise_refiner_"))
            or ".context_refiner." in k or ".noise_refiner." in k or (".layers." in k and ".feed_forward.w" in k) for k in keys):
        return {"family": "zimage", "evidence": "Z-Image 用"}
    if "qwen-image-2.1" in arch or "qwen_image_21" in arch or "qwen_image_21" in base:
        return {"family": "qwen_image21", "evidence": "Qwen-Image-2.1 用"}
    if "qwen-image" in arch or "qwen_image" in arch or base.startswith("qwen_image"):
        return {"family": "qwen_image", "evidence": "Qwen-Image 用"}
    if "flux" in arch or base.startswith("flux") or any("double_blocks" in k for k in keys):
        return {"family": "flux", "evidence": "FLUX 用（このツールでは使えません）"}
    return {"family": None, "evidence": "対応モデルを判別できません"}


FAMILY_LABEL = {"sdxl": "SDXL 系", "anima": "Anima", "krea2": "Krea 2", "sd15": "SD1.5 / SD2", "flux": "FLUX",
                "qwen_image": "Qwen-Image", "qwen_image21": "Qwen-Image-2.1", "zimage": "Z-Image", "moss_image": "MOSS-Image"}


def read_gguf_header(path) -> tuple[dict, list[str]]:
    """String metadata and tensor names of a GGUF file (header only; no tensor data is read).
    path: a file path, or an open binary file (e.g. the first bytes of a remote file in BytesIO)."""
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    import contextlib
    with (open(path, "rb") if isinstance(path, (str, Path)) else contextlib.nullcontext(path)) as f:
        if f.read(4) != b"GGUF":
            raise ValueError("GGUF ではありません")
        version = struct.unpack("<I", f.read(4))[0]
        if version < 2:
            raise ValueError(f"古い GGUF 形式です (v{version})")
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
        if n_tensors > 100_000 or n_kv > 100_000:
            raise ValueError("GGUF header too large")

        def rstr():
            n = struct.unpack("<Q", f.read(8))[0]
            if n > 1 << 24:
                raise ValueError("GGUF header too large")
            return f.read(n).decode("utf-8", "replace")

        def skip(vtype):
            if vtype == 8:
                rstr()
            elif vtype == 9:
                itype, n = struct.unpack("<IQ", f.read(12))
                if itype in sizes:
                    f.seek(sizes[itype] * n, 1)
                else:
                    for _ in range(n):
                        skip(itype)
            else:
                f.seek(sizes[vtype], 1)

        meta = {}
        for _ in range(n_kv):
            key = rstr()
            vtype = struct.unpack("<I", f.read(4))[0]
            if vtype == 8:
                meta[key] = rstr()
            else:
                skip(vtype)
        names = []
        for _ in range(n_tensors):
            names.append(rstr())
            n_dims = struct.unpack("<I", f.read(4))[0]
            f.seek(8 * n_dims + 4 + 8, 1)   # dims, type, offset
    return meta, names


def detect_gguf(path: Path) -> dict:
    """Model family of a GGUF DiT from its tensor names (the architecture field is not reliable:
    Qwen-Image-2.1 files are written as "qwen_image" or without it)."""
    meta, names = read_gguf_header(path)
    bare = set(strip_prefix(names))
    arch = meta.get("general.architecture") or "記載なし"
    if {"txt_in.text_norm.weight", "img_in.weight"} <= bare:
        return {"family": "qwen_image21", "variant": "qwen_image21", "evidence": f"GGUF・Qwen-Image-2.1（{arch}）",
                "note": QWEN21_NOTE}
    if "txt_norm.weight" in bare and "img_in.weight" in bare:
        return {"family": "qwen_image", "variant": "qwen_image", "evidence": f"GGUF・Qwen-Image（{arch}）",
                "note": "GGUF 版は生成だけに使えます（このツールでの Qwen-Image の GGUF 生成は未対応です）。"}
    return {"family": None, "evidence": f"GGUF（{arch}）。このツールでは使えない種類です"}


def comfy_model_dirs(p: Path) -> list[Path]:
    """Model folders of a ComfyUI installation given its folder, a parent of it, or its models folder:
    the models folder, then each base_path of extra_model_paths.yaml that names a folder this tool reads.
    Empty when p is not a ComfyUI folder."""
    install = next((c for c in (p, p / "ComfyUI", p.parent) if (c / "main.py").is_file() and (c / "models").is_dir()), None)
    if not install:
        return []
    out = [install / "models"]
    yml = install / "extra_model_paths.yaml"
    if yml.is_file():
        import re
        kinds = {"checkpoints", "diffusion_models", "unet", "loras", "text_encoders", "clip", "vae"}
        base, used = None, False
        for line in yml.read_text(encoding="utf-8", errors="replace").splitlines() + ["end:"]:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if not line.startswith((" ", "\t")):   # a new section
                if base and used and base.is_dir() and base not in out:
                    out.append(base)
                base, used = None, False
                continue
            m = re.match(r"\s+(\w+):\s*['\"]?(.*?)['\"]?\s*$", line)
            if not m:
                continue
            if m.group(1) == "base_path":
                b = Path(m.group(2))
                base = b if b.is_absolute() else install / b
            elif m.group(1) in kinds:
                used = True
    return out


# Companion files of DiT-only model families, looked up by name in the text_encoders / vae folders.  The first
# name that exists is used.  Qwen-Image training needs the bf16 text encoder; the fp8 one serves generation.
COMPANIONS = {
    "anima": {"te": ["qwen_3_06b_base.safetensors"], "vae": ["qwen_image_vae.safetensors"]},
    "qwen_image": {"te": ["qwen_2.5_vl_7b.safetensors", "qwen_2.5_vl_7b_fp8_scaled.safetensors"],
                   "vae": ["qwen_image_vae.safetensors"], "train_te": ["qwen_2.5_vl_7b.safetensors"]},
    # fp8 first: generation only (no training), and the bf16 one (8.9GB) leaves the 13GB DiT less room (2026-10-27)
    "krea2": {"te": ["qwen3vl_4b_fp8_scaled.safetensors", "qwen3vl_4b_bf16.safetensors"], "vae": ["qwen_image_vae.safetensors"]},
    "qwen_image21": {"te": ["qwen3vl_8b_int8_convrot.safetensors", "qwen3vl_8b_w4a8.safetensors", "qwen3vl_8b_bf16.safetensors"],
                     "vae": ["qwen_image_2.1_vae_bf16.safetensors"],
                     "train_te": ["qwen3vl_8b_bf16.safetensors", "qwen3vl_8b_int8_convrot.safetensors"], "train_vae": ["qwen_image_2.1_vae_diffusers_bf16.safetensors"]},
    "zimage": {"te": ["qwen_3_4b.safetensors", "qwen_3_4b_fp8_mixed.safetensors"], "vae": ["ae.safetensors"],
               "adapter": ["zimage_turbo_training_adapter_v2.safetensors", "zimage_turbo_training_adapter_v1.safetensors"]},
    # w4a8 serves generation only (sd-scripts cannot load it; training.py says so)
    "moss_image": {"te": ["qwen3.5_9b_int8_convrot.safetensors", "qwen3.5_9b_bf16.safetensors", "qwen3.5_9b_w4a8_gptq.safetensors"],
                   "vae": ["flux2-vae_small.safetensors", "flux2-vae.safetensors"],
                   "fast": ["moss_image_8steps_dmd2.safetensors", "moss_image_sft_8steps_dmd_latest.safetensors",
                            "moss_image_sft_8steps_dmd.safetensors"],
                   "fast4": ["moss_image_sft_4step_v2.0.safetensors"]},
}
OPTIONAL_COMPANIONS = ("train_te", "train_vae", "adapter", "fast", "fast4")


def companion_fit(family: str, kind: str, hdr: dict) -> bool | None:
    """Whether a text encoder / VAE file (its safetensors header) fits a model family in ComfyUI.  None: not known
    for that family / kind.  From ComfyUI's comfy/sd.py and the Comfy-Org / unsloth / NVFP4 files (headers checked
    2026-10-26): the Qwen-Image-2.1 VAE is the Wan 2.2 layout with a temporal kernel of 1; the older Wan 2.1 one
    (qwen_image_vae) makes 5-D latents, which the 2.1 model rejects ("too many values to unpack (expected 4)")."""
    hdr = {k: v for k, v in hdr.items() if k != "__metadata__"}
    if kind == "vae":
        if "decoder.middle.0.residual.0.gamma" not in hdr:
            # a Diffusers / Flux / SD layout VAE (decoder.conv_in): the latent channels it decodes (its input
            # channels) say which models it is for: 4 SD / SDXL, 16 Flux.1 ae (Z-Image), 32 Flux.2 (MOSS-Image)
            conv = (hdr.get("decoder.conv_in.weight") or {}).get("shape") or []
            if family in ("qwen_image21", "qwen_image", "anima", "krea2"):
                return False if conv else None
            want = {"sdxl": 4, "zimage": 16, "moss_image": 32}.get(family)
            return None if want is None or len(conv) != 4 else conv[1] == want
        if family in ("sdxl", "zimage", "moss_image"):   # the Wan / Qwen-Image layout is none of these
            return False
        wan22 = "decoder.upsamples.0.upsamples.0.residual.2.weight" in hdr
        head = (hdr.get("decoder.head.2.weight") or {}).get("shape") or []
        q21 = wan22 and len(head) == 5 and head[2] == 1
        if family == "qwen_image21":
            return q21
        if family in ("qwen_image", "anima", "krea2"):
            return not wan22
        return None
    if kind == "te":
        qwen35 = any("linear_attn" in k for k in hdr)   # Qwen3.5 (MOSS-Image)
        emb = next(((v.get("shape") or []) for k, v in hdr.items() if k.endswith("embed_tokens.weight")), [])
        if family == "moss_image":
            return True if qwen35 else (False if len(emb) == 2 else None)
        if qwen35:
            return False if family in ("qwen_image21", "qwen_image", "anima", "krea2", "zimage") else None
        if len(emb) != 2:
            return None
        # the width of the Qwen model: Qwen3-VL 8B / Qwen2.5-VL 7B / Qwen3 0.6B (Anima); 2560 is Qwen3 4B (Z-Image)
        # and Qwen3-VL 4B (Krea 2), told apart by the vision tower the VL files carry
        if family in ("qwen_image21", "qwen_image", "anima"):
            return emb[1] == {"qwen_image21": 4096, "qwen_image": 3584, "anima": 1024}[family]
        if family in ("zimage", "krea2"):
            if emb[1] != 2560:
                return False
            vision = any(k.startswith(("visual.", "model.visual.")) for k in hdr)
            return (not vision) if family == "zimage" else (True if vision else None)
    return None
# DiT-only families: the model file, a text encoder and a VAE (see COMPANIONS)
DIT_FAMILIES = ("anima", "qwen_image", "qwen_image21", "krea2", "zimage", "moss_image")
# families whose GGUF file is registered as the light generation file ("gen"), paired with the bf16 one if found
GGUF_FAMILIES = ("qwen_image", "qwen_image21")
# Folders under models/ that hold model files in the layout this version reads (plus the i2t tool's own folder and
# unfinished downloads).  Model files in any other folder there are listed with where to move them.
STANDARD_DIRS = {"checkpoints", "loras", "diffusion_models", "unet", "text_encoders", "clip", "vae", "i2t", "_incoming"}


# Library (scan / repair / auto-register) and Registry both read, change and write settings["models"]:
# they share one lock so that neither overwrites the other's change.
MODELS_LOCK = threading.RLock()


class Library:
    """Model files the user puts in folders.  The ComfyUI-style folders under models/ (checkpoints,
    loras, diffusion_models, text_encoders, vae) are always read; other folders are read only when
    the user adds them in the settings (e.g. an existing ComfyUI models folder, or Forge's
    Stable-diffusion / Lora folders).

    Checkpoints found here are added to the registry automatically, so the same file is used for
    generation and for training without copying.  DiT-only models (Anima, Qwen-Image) are added when
    their text encoder and VAE are found in the text_encoders / vae folders next to them.  Only file
    headers are read (cached by path/size/mtime), so a scan does not make the UI wait."""

    SUBDIRS = {"checkpoints": "checkpoint", "loras": "lora", "Stable-diffusion": "checkpoint", "Lora": "lora",
               "diffusion_models": "checkpoint", "unet": "checkpoint"}
    COMPANION_DIRS = {"te": ("text_encoders", "clip"), "vae": ("vae", "VAE"), "train_te": ("text_encoders", "clip"),
                      "train_vae": ("vae", "VAE"), "adapter": ("loras", "Lora"), "fast": ("loras", "Lora"),
                      "fast4": ("loras", "Lora")}

    def __init__(self):
        self._lock = MODELS_LOCK
        self._cache: dict[str, dict] = {}

    @staticmethod
    def default_dirs() -> list[Path]:
        # unet: where ComfyUI users keep GGUF models (read when it exists; scan() creates only the first three)
        return [config.MODELS_DIR / d for d in ("checkpoints", "loras", "diffusion_models", "unet")]

    @staticmethod
    def companion_default_dirs() -> list[Path]:
        return [config.MODELS_DIR / "text_encoders", config.MODELS_DIR / "vae"]

    def roots(self) -> list[str]:
        return list(config.settings().get("model_roots", []))

    def add_root(self, path: str) -> list[str]:
        """Add a folder to read.  A ComfyUI installation folder (or its models folder) adds its models
        folder and the extra model folders its extra_model_paths.yaml names, so the files ComfyUI uses
        are used here too, in place.  Returns the folders added."""
        p = Path(path.strip().strip('"'))
        if not path.strip() or not p.is_absolute():
            raise ValueError("フォルダーのパスを、ドライブ名から入力してください（例: D:\\ComfyUI\\models）")
        if not p.is_dir():
            raise ValueError(f"フォルダーが見つかりません: {path}")
        dirs = comfy_model_dirs(p) or [p]
        with self._lock:
            roots = self.roots()
            new = [str(d) for d in dirs if str(d) not in roots]
            if new:
                config.settings().set("model_roots", roots + new)
            return new

    def remove_root(self, path: str) -> None:
        with self._lock:
            config.settings().set("model_roots", [r for r in self.roots() if r != path])

    def _folders(self) -> list[Path]:
        """Folders to read.  An added folder that has checkpoints/loras (ComfyUI) or Stable-diffusion/Lora
        (Forge, A1111) subfolders is read through those; otherwise the folder itself is read.  Whether a
        file is a checkpoint or a LoRA is decided from its contents, not from the folder."""
        out = self.default_dirs()
        for r in self.roots():
            root = Path(r)
            out += [root / s for s in self.SUBDIRS if (root / s).is_dir()] or [root]
        return out

    def _companion_folders(self, kind: str) -> list[Path]:
        """text_encoders / vae folders: those under models/, then the same names under added folders,
        then the older per-family folders of this tool (models/anima, models/shared)."""
        names = self.COMPANION_DIRS[kind]
        out = [config.MODELS_DIR / names[0]]
        for r in self.roots():
            out += [Path(r) / n for n in names]
        out += [config.MODELS_DIR / "anima", config.MODELS_DIR / "shared", config.MODELS_DIR / "krea2"]
        return [d for d in out if d.is_dir()]

    def find_companion(self, family: str, kind: str) -> str | None:
        """First listed name found in the text_encoders / vae folders, also in their subfolders
        (ComfyUI users often keep e.g. text_encoders/Qwen-Image-2.1-GGUF/); a file of such a name that does not
        fit the family is passed over.  Then, for te / vae, any file there whose contents fit (other distributors'
        names, e.g. Qwen-Image-2.1-text_encoder-FP8)."""
        for name in COMPANIONS.get(family, {}).get(kind, []):
            for d in self._companion_folders(kind):
                hit = d / name
                if not hit.is_file():
                    hit = next((x for x in d.rglob(name) if x.is_file()), None)
                if hit and (kind not in ("te", "vae") or self.fits(family, kind, str(hit)) is not False):
                    return str(hit)
        if kind in ("te", "vae"):
            for p in self.companion_files()[kind]:
                if self.fits(family, kind, p):
                    return p
        return None

    def fits(self, family: str, kind: str, path: str) -> bool | None:
        """companion_fit for a file (header cached by size and time).  None: unknown, or not readable here
        (GGUF text encoders)."""
        f = Path(path)
        try:
            st = f.stat()
            key = f"fit|{family}|{kind}|{f}|{st.st_size}|{int(st.st_mtime)}"
            if key not in self._cache:
                self._cache[key] = companion_fit(family, kind, read_safetensors_header(f)) \
                    if f.suffix.lower() == ".safetensors" else None
            return self._cache[key]
        except (OSError, ValueError, struct.error):   # struct.error: a file shorter than a header (cut off while copying)
            return None

    def _repair_companions(self) -> None:
        """A registered model whose text encoder / VAE does not fit it (e.g. the older Qwen-Image VAE chosen for
        Qwen-Image-2.1) gets a fitting one from the folders, when there is one."""
        with self._lock:
            models = config.settings().get("models", [])
            changed = False
            for m in models:
                for kind in ("te", "vae"):
                    p = m.get("files", {}).get(kind)
                    if p and Path(p).is_file() and self.fits(m.get("family"), kind, p) is False:
                        hit = self.find_companion(m.get("family"), kind)
                        if hit:
                            m["files"][kind] = hit
                            changed = True
            if changed:
                config.settings().set("models", models)

    def companion_files(self) -> dict[str, list[str]]:
        """Every text encoder / VAE file in the text_encoders / vae folders (what the user put there, whatever its
        name), for choosing one by hand when a model's own cannot be found by name."""
        out = {}
        for kind in ("te", "vae"):
            seen, files = set(), []
            for d in self._companion_folders(kind):
                for f in sorted([*d.rglob("*.safetensors"), *d.rglob("*.gguf")]):
                    if f.is_file() and str(f).lower() not in seen:
                        seen.add(str(f).lower())
                        files.append(str(f))
            out[kind] = files
        return out

    def register_file(self, path: str, te: str = "", vae: str = "") -> dict:
        """Registers one model file of the folders this tool reads, as chosen in a model list.  A DiT model takes the
        text encoder / VAE given, else the one found by name; when neither is there, ValueError with what is
        needed ("needs": ["te", "vae"]).  A file already registered returns that model."""
        f = Path(path)
        try:
            rf = f.resolve()
            inside = f.is_file() and any(rf.is_relative_to(d.resolve()) for d in self._folders() if d.is_dir())
        except OSError:
            inside = False
        if not inside:
            raise ValueError(f"モデルのフォルダーにないファイルです: {path}")
        info = self._info(f)
        fam = info.get("family")
        if info.get("is_lora") or fam not in ("sdxl", *DIT_FAMILIES) or (info.get("gguf") and fam not in GGUF_FAMILIES):
            raise ValueError(f"このツールで生成に使えないファイルです（{info.get('evidence') or '種類を判別できません'}）")
        for m in config.settings().get("models", []):
            if any(p and str(Path(p)).lower() == str(f).lower() for k, p in m.get("files", {}).items() if k in ("ckpt", "gen", "turbo")):
                return m
        if fam == "sdxl":
            files = {"ckpt": str(f)}
        else:
            main = "gen" if info.get("gguf") else "turbo" if fam == "krea2" and info.get("quantized") else "ckpt"
            files = {main: str(f)}
            picked = {"te": te, "vae": vae}
            needs = []
            for kind in ("te", "vae"):
                p = picked[kind] or self.find_companion(fam, kind)
                if p and not Path(p).is_file():
                    raise ValueError(f"ファイルが見つかりません: {p}")
                if p and self.fits(fam, kind, p) is False:
                    raise ValueError(f"{Path(p).name} は {FAMILY_LABEL.get(fam, fam)} 用の"
                                     f"{'テキストエンコーダー' if kind == 'te' else ' VAE '}ではありません。別のファイルを選んでください")
                if not p:
                    needs.append(kind)
                files[kind] = p
            if needs:
                e = ValueError("テキストエンコーダーと VAE を選んでください" if len(needs) == 2 else
                               "テキストエンコーダーを選んでください" if needs == ["te"] else "VAE を選んでください")
                e.needs = needs
                # the files there, those that fit first; a file known not to fit is shown but cannot be chosen
                e.choices = {k: sorted(({"path": c, "fits": self.fits(fam, k, c)} for c in self.companion_files()[k]),
                                       key=lambda x: {True: 0, None: 1, False: 2}[x["fits"]]) for k in needs}
                raise e
            for kind in OPTIONAL_COMPANIONS:
                if COMPANIONS.get(fam, {}).get(kind):
                    hit = self.find_companion(fam, kind)
                    if hit:
                        files[kind] = hit
        v = info.get("variant")
        m = {"id": uuid.uuid4().hex[:10], "name": f.stem, "variant": v, "family": fam, "files": files, "auto": True,
             "variant_confirmed": fam != "sdxl" and (fam not in ("zimage", "moss_image") or v is not None)}
        with self._lock:
            models = config.settings().get("models", [])
            models.append(m)
            config.settings().set("models", models)
        return m

    def misplaced(self) -> list[dict]:
        """Model files in folders under models/ that this version does not read for them - mainly the per-model
        folders the version of 2026-10-08 downloaded into (models/anima, models/illustrious, ...).  Each comes with
        the folder it belongs in, so the user can be told "move X from here to here"."""
        out = []
        try:
            dirs = sorted(d for d in config.MODELS_DIR.iterdir() if d.is_dir() and d.name.lower() not in STANDARD_DIRS)
        except OSError:
            return out
        for d in dirs:
            for f in sorted([*d.rglob("*.safetensors"), *d.rglob("*.gguf")]):
                dest = self._dest_for(f)
                if dest and not (config.MODELS_DIR / dest[0] / f.name).exists():   # already there: nothing to move
                    sub, what = dest
                    out.append({"name": f.name, "what": what, "from": str(f.parent), "to": str(config.MODELS_DIR / sub),
                                "sub": sub, "old_dir": d.name})
        return out

    def _dest_for(self, f: Path) -> tuple[str, str] | None:
        """(folder under models/, what the file is) for a file found in an old folder; None when unknown."""
        name = f.name.lower()
        for kind, sub, label in (("te", "text_encoders", "テキストエンコーダー"), ("vae", "vae", "VAE")):
            fams = [fam for fam, c in COMPANIONS.items()
                    if any(name == n.lower() for k, ns in c.items() if k in (kind, "train_" + kind) for n in ns)]
            if fams:
                return sub, "・".join(dict.fromkeys(FAMILY_LABEL.get(x, x) for x in fams)) + " の " + label
        info = self._info(f)
        if info["is_lora"]:
            return "loras", "LoRA"
        fam = info.get("family")
        if fam == "sdxl":
            return "checkpoints", "SDXL 系のモデル"
        if fam in DIT_FAMILIES:
            return "diffusion_models", f"{FAMILY_LABEL.get(fam, fam)} の本体"
        return None

    def _repair_paths(self) -> None:
        """A registered model whose file was moved (e.g. from models/anima to models/diffusion_models as the
        settings screen asks) is pointed to the file's new place, found by its name in the folders this tool
        reads.  The model keeps its id, so projects that use it keep working."""
        with self._lock:
            models = config.settings().get("models", [])
            changed = False
            for m in models:
                for k, p in list(m.get("files", {}).items()):
                    if not p or Path(p).exists():
                        continue
                    base = k.removeprefix("train_")
                    folders = self._companion_folders(k) if base in ("te", "vae") else self._folders()
                    name = Path(p).name
                    hit = None
                    for d in folders:
                        if d.is_dir():
                            hit = next((x for x in d.rglob(name) if x.is_file()), None)
                            if hit:
                                break
                    if hit and base not in ("te", "vae") and self._info(hit).get("family") != m.get("family"):
                        hit = None   # same name, different model
                    if hit:
                        m["files"][k] = str(hit)
                        changed = True
            if changed:
                config.settings().set("models", models)

    def _info(self, f: Path) -> dict:
        st = f.stat()
        key = f"{f}|{st.st_size}|{int(st.st_mtime)}"
        if key not in self._cache:
            try:
                if f.suffix.lower() == ".gguf":
                    d = {**detect_gguf(f), "is_lora": False, "gguf": True}
                else:
                    keys = read_safetensors_header(f).keys()
                    if is_lora_keys(keys):
                        d = {**detect_lora(f), "is_lora": True}
                    else:
                        d = {**detect(f), "is_lora": False}
            except Exception as e:   # any malformed header: this file is listed as unreadable, the scan goes on
                d = {"family": None, "evidence": f"読めません: {e}", "is_lora": None}
            self._cache[key] = d
        return {**self._cache[key], "size": st.st_size}

    def scan(self) -> dict:
        for d in self.default_dirs()[:3] + self.companion_default_dirs():
            d.mkdir(parents=True, exist_ok=True)
        ckpts, loras, seen = [], [], set()
        for folder in self._folders():
            if not folder.is_dir():
                continue
            files = sorted([*folder.rglob("*.safetensors"), *folder.rglob("*.gguf")])
            for f in files:
                rp = str(f.resolve()).lower()
                if rp in seen:
                    continue
                seen.add(rp)
                info = self._info(f)
                entry = {"path": str(f), "name": f.stem, "family": info.get("family"), "evidence": info.get("evidence"),
                         "size": info["size"], "folder": str(folder), "note": info.get("note"), "gguf": bool(info.get("gguf")),
                         "fp8": bool(info.get("fp8")), "quantized": bool(info.get("quantized"))}
                if info["is_lora"]:
                    if info.get("edit_format"):   # MOSS-Image edit LoRA: the image page edits the source image with it
                        entry["edit_format"] = info["edit_format"]
                    loras.append(entry)
                elif info["is_lora"] is False:
                    entry["variant_guess"] = info.get("variant")
                    ckpts.append(entry)
        # the same file reachable from two folders (copies) is listed once: name + size
        def dedupe(items):
            out, keys = [], set()
            for it in items:
                k = (it["name"].lower(), it["size"])
                if k not in keys:
                    keys.add(k)
                    out.append(it)
            return out
        ckpts, loras = dedupe(ckpts), dedupe(loras)
        self._repair_paths()   # before registering, so a moved file is not added again as a new model
        self._repair_companions()
        self._register(ckpts)
        return {"checkpoints": ckpts, "loras": loras, "misplaced": self.misplaced()}

    def _register(self, ckpts: list[dict]) -> None:
        """Checkpoints become registry models (an SDXL variant is confirmed later by the user when it
        cannot be read from the file).  A file that is already registered - by path, or the same file
        name and size (e.g. a copy) - is not added again.  A DiT model whose text encoder or VAE is not
        found is listed with the reason and not registered."""
        with self._lock:
            models = config.settings().get("models", [])
            known = set()
            for m in models:
                for p in (m.get("files", {}).get(k) for k in ("ckpt", "gen", "turbo")):
                    if p:
                        known.add(str(Path(p)).lower())
                        try:
                            known.add((Path(p).name.lower(), Path(p).stat().st_size))
                        except OSError:
                            pass
            added = False
            for m in models:   # optional files put in place after the model was registered (e.g. the Z-Image adapter)
                for kind in OPTIONAL_COMPANIONS:
                    if COMPANIONS.get(m.get("family"), {}).get(kind) and not m.get("files", {}).get(kind):
                        hit = self.find_companion(m["family"], kind)
                        if hit:
                            m["files"][kind] = hit
                            added = True
            ggufs = [c for c in ckpts if c["gguf"] and c["family"]]
            for c in ckpts:
                fam = c["family"]
                if fam not in ("sdxl", *DIT_FAMILIES) or (c["gguf"] and fam not in GGUF_FAMILIES):
                    continue
                if c["path"].lower() in known or (Path(c["path"]).name.lower(), c["size"]) in known:
                    continue
                if fam == "sdxl":
                    files = {"ckpt": c["path"]}
                else:
                    main = "gen" if c["gguf"] else "turbo" if fam == "krea2" and c["quantized"] else "ckpt"
                    files = {main: c["path"]}
                    for kind in ("te", "vae"):
                        hit = self.find_companion(fam, kind)
                        if not hit:
                            names = " / ".join(COMPANIONS.get(fam, {}).get(kind, [])) or "対応するファイル"
                            c["evidence"] = (c.get("evidence") or "") + f"。{'テキストエンコーダー' if kind == 'te' else 'VAE'}（{names}）が " \
                                            f"{'text_encoders' if kind == 'te' else 'vae'} フォルダーに見つからないため、まだ選べません"
                            files = None
                            break
                        files[kind] = hit
                    if files is None:
                        continue
                    for kind in OPTIONAL_COMPANIONS:
                        if COMPANIONS.get(fam, {}).get(kind):
                            hit = self.find_companion(fam, kind)
                            if hit:
                                files[kind] = hit
                    if fam in GGUF_FAMILIES and not c["gguf"]:
                        g = next((x for x in ggufs if x["family"] == fam), None)
                        if g:
                            files["gen"] = g["path"]
                            known.add(g["path"].lower())
                    if c["gguf"] and any(x["family"] == fam and not x["gguf"] for x in ckpts):
                        continue   # paired with the bf16 file instead
                v = c.get("variant_guess")
                models.append({"id": uuid.uuid4().hex[:10], "name": c["name"], "variant": v, "family": fam,
                               "files": files, "auto": True, "variant_confirmed": fam != "sdxl" and (fam not in ("zimage", "moss_image") or v is not None)})
                known.add(c["path"].lower())
                added = True
            if added:
                config.settings().set("models", models)


class Registry:
    """Registered base models, stored in global settings."""

    def __init__(self):
        self._lock = MODELS_LOCK

    def list(self) -> list[dict]:
        out = []
        for m in config.settings().get("models", []):
            m = dict(m)
            # the 8-step generation LoRA is optional: without it MOSS-Image samples the normal way
            m["missing"] = [k for k, p in m.get("files", {}).items() if p and k not in ("fast", "fast4") and not Path(p).exists()]
            m["variant_info"] = VARIANTS.get(m.get("variant"), {})
            m["needs_variant"] = bool(m.get("auto")) and not m.get("variant_confirmed") and not m.get("variant")
            out.append(m)
        return out

    def get(self, model_id: str) -> dict | None:
        for m in self.list():
            if m["id"] == model_id:
                return m
        return None

    def add(self, name: str, variant: str, files: dict) -> dict:
        if variant not in VARIANTS:
            raise ValueError("unknown variant")
        if not available(variant):
            raise ValueError(UNAVAILABLE_NOTE.get(variant, "このモデル系列は現在対応していません。"))
        fam = VARIANTS[variant]["family"]
        for key, label, required in FAMILY_FILES[fam]:
            p = files.get(key)
            if required and not p:
                raise ValueError(f"{label} を指定してください")
            if p and not Path(p).is_file():
                raise ValueError(f"ファイルが見つかりません: {p}")
        one_of = FAMILY_ONE_OF.get(fam)
        if one_of and not any(files.get(k) for k in one_of):
            labels = {k: lab for k, lab, _ in FAMILY_FILES[fam]}
            raise ValueError("次のどちらかを指定してください: " + " / ".join(labels[k] for k in one_of))
        with self._lock:
            models = config.settings().get("models", [])
            m = {"id": uuid.uuid4().hex[:10], "name": name, "variant": variant, "family": fam,
                 "files": {k: str(Path(v)) for k, v in files.items() if v}}
            models.append(m)
            config.settings().set("models", models)
            return m

    def update(self, model_id: str, **kw) -> None:
        with self._lock:
            models = config.settings().get("models", [])
            for m in models:
                if m["id"] == model_id:
                    m.update(kw)
            config.settings().set("models", models)

    def remove(self, model_id: str) -> None:
        with self._lock:
            config.settings().set("models", [m for m in config.settings().get("models", []) if m["id"] != model_id])


class Hasher:
    """Background sha256 of large model files.  Results are cached by (path, size, mtime) so
    the UI never waits for hashing and the hash is computed once.  One file at a time, read
    at a capped rate, so hashing a 26GB model does not starve the disk (HDDs especially)."""

    MAX_BYTES_PER_SEC = 120 << 20

    def __init__(self):
        import queue
        self._lock = threading.Lock()
        self._running: set[str] = set()
        self._q: "queue.Queue" = queue.Queue()
        threading.Thread(target=self._loop, daemon=True, name="hasher").start()

    def _loop(self):
        while True:
            p, key = self._q.get()
            try:
                self._run(p, key)
            except Exception:   # e.g. the file was moved or deleted while queued: it is hashed again when asked
                pass

    @staticmethod
    def _key(p: Path) -> str:
        st = p.stat()
        return f"{p}|{st.st_size}|{int(st.st_mtime)}"

    def get(self, path: str) -> str | None:
        p = Path(path)
        if not p.exists():
            return None
        key = self._key(p)
        h = config.settings().get("hash_cache", {}).get(key)
        if h:
            return h
        with self._lock:
            if key in self._running:
                return None
            self._running.add(key)
        self._q.put((p, key))
        return None

    def _run(self, p: Path, key: str):
        try:
            import time
            h = hashlib.sha256()
            t0, n = time.monotonic(), 0
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(8 << 20), b""):
                    h.update(chunk)
                    n += len(chunk)
                    ahead = n / self.MAX_BYTES_PER_SEC - (time.monotonic() - t0)
                    if ahead > 0:
                        time.sleep(ahead)
            with config.settings()._lock:   # loraconf.sha256 writes the same cache
                cache = dict(config.settings().get("hash_cache", {}))
                cache[key] = h.hexdigest()
                config.settings().set("hash_cache", cache)
        finally:
            with self._lock:
                self._running.discard(key)


registry = Registry()
hasher = Hasher()
library = Library()
