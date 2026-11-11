#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
i2t - 画像からポジティブ / ネガティブプロンプトをローカルで生成するツール

  GUI : python i2t.py                （または run.bat をダブルクリック）
  CLI : python i2t.py 画像.png [画像2.jpg ...] [オプション]

推論はすべて PC 内で完結します（外部 API は使いません）。
  1. WD14 タガー (SmilingWolf/wd-*-tagger-v3, ONNX)
       Danbooru タグを推定し、タグ形式のポジティブプロンプトと
       ルールベースのネガティブプロンプト・分析メモを作る。
  2. Ollama の VLM (qwen3-vl など、任意)
       タガーのタグを根拠に、日本語の分析・自然文プロンプト・
       画像固有のネガティブ・再現のコツを補強する。
初回のみタガーモデル (0.4〜1.3GB) を Hugging Face からダウンロードします。
"""
from __future__ import annotations

import argparse
import base64
import csv
import http.client
import io
import json
import os
import queue
import re
import shutil
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageOps

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:  # ヘッドレス環境など
    tk = None  # type: ignore[assignment]

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD  # ドラッグ＆ドロップ (任意)

    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False

APP_NAME = "i2t"  # 画面に出す名前
APP_ID = "i2t"  # ファイル名・CLI・HTTP の User-Agent 用 (User-Agent は ASCII しか送れない)
__version__ = "1.6.0"
FROZEN = bool(getattr(sys, "frozen", False))  # PyInstaller で exe 化されているか
# exe のときは exe の隣、スクリプトのときはスクリプトの隣に config.json と models/ を置く
APP_DIR = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parent
# 同梱リソース (アイコンなど)。exe では一時展開先 (_MEIPASS) に入る
RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", APP_DIR))
CONFIG_PATH = APP_DIR / "config.json"
MODELS_DIR = APP_DIR / "models"
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def resource_path(relative: str) -> Path:
    return RESOURCE_DIR / relative

# 名前: (Hugging Face リポジトリ, 説明)
TAGGER_MODELS: dict[str, tuple[str, str]] = {
    "wd-eva02-large-tagger-v3": ("SmilingWolf/wd-eva02-large-tagger-v3", "最高精度 / 1.26GB"),
    "wd-vit-large-tagger-v3": ("SmilingWolf/wd-vit-large-tagger-v3", "高精度 / 1.26GB"),
    "wd-swinv2-tagger-v3": ("SmilingWolf/wd-swinv2-tagger-v3", "バランス / 467MB"),
    "wd-convnext-tagger-v3": ("SmilingWolf/wd-convnext-tagger-v3", "軽量 / 395MB"),
    "wd-vit-tagger-v3": ("SmilingWolf/wd-vit-tagger-v3", "最軽量 / 379MB"),
    # PixAI Tagger v0.9 (Apache-2.0)。wd-eva02-large-tagger-v3 を土台に、2025-01 までの Danbooru で学習し直したもの
    # (約 13,500 語。新しい作品のキャラに強い)。DeepGHS の ONNX 版を使う。レーティングは出さないので、
    # deepghs/anime_rating (MIT、約 17MB) で補う (RATING_MODEL_URL)
    "pixai-tagger-v0.9": ("deepghs/pixai-tagger-v0.9-onnx", "新しいキャラに強い (2025-01 まで) / 1.27GB"),
    # Camie Tagger v2 (GPL-3.0。モデルはこの PC にダウンロードして使い、i2t と一緒には配らない)。ViT-B/16 を土台に
    # 約 70,500 語 (一般・キャラ・作品・絵師・メタ・年・レーティング) を出す。配布元の camie-tagger-v2.onnx と
    # camie-tagger-v2-metadata.json を使い、語彙は WD と同じ selected_tags.csv の形に直して置く (camie_tags_csv)
    "camie-tagger-v2": ("Camais03/camie-tagger-v2", "語彙が多い (約 7 万語。GPL-3.0) / 789MB"),
}
DEFAULT_TAGGER = "wd-eva02-large-tagger-v3"

ENGINES = {
    "hybrid": "タガー + VLM (Ollama) — 分析・自然文・ネガティブ強化",
    "tagger": "タガーのみ (WD14) — 高速・オフライン",
}
# 出力先モデルのプリセット
#   kind: "tags"  = タグ列 (タガーだけで完結)
#         "anima" = タグ列 + 自然文の補足 (Anima 公式の並び順・品質タグ)
#         "prose" = 自然文のみ (VLM が必要。無い場合は振り分けたタグから英文を組み立てる)
#   negative: True = ネガティブを出す / "optional" = 出すが既定設定では無視される (CFG 1) / False = 非対応なので出さない
TARGETS: dict[str, dict[str, Any]] = {
    "sdxl": {"label": "SDXL / Pony / Illustrious (タグ形式)", "kind": "tags", "negative": True},
    "novelai": {"label": "NovelAI Diffusion V4 / V4.5 (タグ形式)", "kind": "tags", "negative": True},
    "anima": {"label": "Anima (タグ + 自然文)", "kind": "anima", "negative": True},
    "flux": {"label": "Flux.1 / SD3 (自然文) ※VLM 推奨", "kind": "prose", "negative": "optional"},
    "flux2": {"label": "FLUX.2 [pro / max / dev] (自然文・ネガティブ非対応) ※VLM 推奨", "kind": "prose", "negative": False},
    "flux2klein": {"label": "FLUX.2 [klein] (詳しい自然文・ネガティブ非対応) ※VLM 推奨", "kind": "prose", "negative": False},
    "krea2": {"label": "Krea 2 (自然文・ネガティブ非対応) ※VLM 推奨", "kind": "prose", "negative": False},
    "qwen": {"label": "Qwen-Image 2.1 (長文の自然文) ※VLM 推奨", "kind": "prose", "negative": "optional"},
    # Z-Image (Tongyi-MAI): Turbo は蒸留モデルで CFG 0 のためネガティブは効かない、ベースは CFG 3〜5 でネガティブが効く (モデルカード)
    "zimageturbo": {"label": "Z-Image Turbo (詳しい自然文・ネガティブ非対応) ※VLM 推奨", "kind": "prose", "negative": False},
    "zimage": {"label": "Z-Image (詳しい自然文 + ネガティブ) ※VLM 推奨", "kind": "prose", "negative": True},
    # MOSS-Image (OpenMOSS): 学習のキャプションの 8 割が自然文 (技術報告 arXiv 2610.23374) なので自然文だけ。
    # ネガティブはモデルは対応しているが公式の既定は空 (モデルカード) で、i2t は出さない
    "mossimage": {"label": "MOSS-Image (詳しい自然文) ※VLM 推奨", "kind": "prose", "negative": False,
                  "neg_label": "ネガティブプロンプト (MOSS-Image は空が公式の既定。必要なら自分で入れられます)"},
    "chatgpt": {"label": "ChatGPT / GPT Image (依頼文) ※VLM 推奨", "kind": "prose", "negative": False},
    "grok": {"label": "Grok Imagine (短い自然文) ※VLM 推奨", "kind": "prose", "negative": False},
    "nanobanana": {"label": "Nano Banana / Gemini 画像 (情景描写) ※VLM 推奨", "kind": "prose", "negative": False},
    # ネガティブ欄は無く、除外は本文末尾の --no に入れる
    "midjourney": {"label": "Midjourney V8 / Niji 7 (短い描写 + パラメータ) ※VLM 推奨", "kind": "prose", "negative": False},
}
DEFAULT_TARGET = "sdxl"
LEGACY_STYLE_TO_TARGET = {"tags": "sdxl", "natural": "flux"}

# 撮影プリセットの既定。選ぶと本文のレンズ・照明の記述をこの文に置き換える
DEFAULT_CAMERA_LOOKS = [
    {"name": "ポートレート 85mm", "prompt": "professional photography, RAW photographic, natural skin texture, natural lighting, "
     "accurate anatomy, 85mm lens, shallow depth of field, natural color grading"},
    {"name": "35mm フィルム", "prompt": "analog film photography, shot on 35mm film, Kodak Portra 400, natural film grain, "
     "soft natural light, 50mm lens, true-to-life colors"},
    {"name": "スナップ / 街撮り", "prompt": "candid street photography, 35mm lens, available light, natural skin texture, "
     "documentary style, natural color grading"},
    {"name": "スタジオ", "prompt": "professional studio photography, softbox lighting, seamless backdrop, 85mm lens, f/8, "
     "sharp focus, accurate skin tones"},
    {"name": "ファッション誌", "prompt": "high-end fashion editorial photography, medium format camera, 100mm lens, crisp detail, "
     "controlled studio lighting, magazine color grading"},
    {"name": "シネマ", "prompt": "cinematic film still, anamorphic lens, shallow depth of field, subtle film grain, "
     "motivated lighting, cinematic color grading"},
    {"name": "スマホ", "prompt": "casual smartphone photo, natural lighting, deep depth of field, slight sensor noise, unedited colors"},
    {"name": "フラッシュ直当て", "prompt": "direct flash photography, compact point-and-shoot camera, harsh on-camera flash, "
     "dark background, candid snapshot"},
    {"name": "風景 / 広角", "prompt": "landscape photography, 24mm wide-angle lens, f/11, deep depth of field, golden hour light, "
     "natural color grading"},
    {"name": "夜景", "prompt": "night photography, high ISO, available street light, bokeh, natural noise, natural color grading"},
]

DEFAULT_CONFIG: dict[str, Any] = {
    "engine": "tagger",  # 既定はタガーのみ。VLM (Ollama) は使いたい人が切り替える拡張
    "tagger_model": DEFAULT_TAGGER,
    "general_threshold": 0.35,
    "character_threshold": 0.85,
    "style_threshold": 0.15,  # 画風・技法タグだけはこの低い信頼度でも「画風 / 質感」に拾う
    "era_tags": True,  # 年代スタイル分類 (DeepGHS anime_style_ages) を画風タグに足す
    "target": DEFAULT_TARGET,
    "quality_tags": True,
    "use_break": True,  # 複数人物のとき SDXL 系で BREAK 区切りにする
    "mosaic": False,  # オンなら常にプロンプトに censored, mosaic censoring を足す (生成画像にモザイクがかかる)
    "mosaic_strength": "strong",  # normal | strong | max (MOSAIC_LEVELS)
    "split_extras": False,  # カメラ・照明・画風のタグを本文から外し、追加プロンプト側だけに出す
    "split_background": False,  # 背景のタグを本文から外し、本文を被写体だけにする (背景は追加プロンプト側)
    # 仕上げの書き直し (文章の出力先。タガー + VLM のとき): 集めた内容を VLM にもう一度渡して、出力先の書き方で
    # 1 つの文章に書き直させる (画像は見せない)。内容が落ちたら使わない。VLM の呼び出しが 1 回増える
    "polish_prose": True,
    "preview_mosaic": False,  # プレビュー画像をモザイク表示する (表示だけ。出力には影響しない)
    # 結果をチップ (, 区切りの塊) で表示し、選んで Del で消す・日本語訳で見る。Ollama があるときだけ有効
    "result_chips": True,
    # 辞書に無い語の日本語訳に使う Ollama のモデル。分析の VLM とは別 (Index-Translate-2B。
    # 翻訳専用 2B で 40 個 約 5 秒。qwen2.5vl:7b は約 15 秒、qwen3-vl:8b (思考) は訳せず、qwen2.5:7b は誤訳が多かった)
    "translate_model": "hf.co/IndexTeam/Index-Translate-2B-GGUF:Q4_K_M",
    "mode": "single",  # single = 1 枚 / multi = 最大 8 枚 (メイン + ポーズ / 服装 / 背景 / 小物 / 体格 / 作風 / 表情)
    # GPU / VRAM の使用を抑える: タガーと年代判定は CPU (スレッド数は半分)、VLM は使い終わったらすぐ VRAM から降ろす
    "low_vram": True,
    # 人物はメインの 1 人だけ: 一番大きく写っている人物を検出して切り出し、ほかの人はプロンプトに入れない
    "main_person_only": False,
    # 人物ごとに分ける: 2 人以上なら人物を検出して 1 人ずつ切り出してタグ付けし、人数タグは検出した人数で付け直す
    "split_persons": True,
    "gender": "none",  # 性別をそろえる: none | female (全員女性) | male (全員男性)
    # 画風を変える (MOSS-Image だけ): none | photo (実写) | anime | manga | watercolor | thick (厚塗り) | oil | pencil。
    # 元の画像とは違う画風の画像として書く (先頭でその画風だと言い切り、元の画風の語を外し、描き方・質感の文を足す)
    "restyle": "none",
    # 年齢の見た目を書く: 人物ごとに画像から読んだ年齢層を、文とタグに書く (書かないと、生成する側のモデルの癖で年齢が決まる)
    "age_look": True,
    # 同じ顔を避ける: 2 人以上なら「同じ人物・そっくりな人物が並ぶ」タグをネガティブに。双子や描き分けの少ない画風もあるので既定オフ
    "avoid_same_face": False,
    "avoid_duplicates": False,  # 分身を防ぐ (同じ人物の複製・コラージュ・複数アングル・残像・体の重複をネガティブに)
    "person_details": False,  # 人物の特徴を詳しく (VLM で顔立ち・目・眉・鼻・口・耳・髪型・肌・年齢の見た目・体格・目立つ特徴を読む。顔は拡大して見せる)
    "turnaround": False,  # 三面図にする: ポーズ・カメラ・背景・小物を外し、前・後ろ・横の三面図のタグにする
    # 作風の特徴を入れない: 画風・技法・絵か写真か (品質タグ・ネガティブの定番も)・年代・作品名 / 作者名を外す (画風 LoRA 用。2026-10-10)
    "no_style": False,
    "no_identity": False,  # 人物の特徴を入れない: 髪・目・顔立ち・肌の色・キャラ名を外す (LoRA でキャラを指定するとき。2026-10-10)
    "anatomy_guard": True,  # 人物の破綻を防ぐ: 人物がいればネガティブに bad anatomy などを入れ、全身なら顔だけの構図を避ける
    "ollama_setup_asked": False,  # Ollama 未導入時の導入サポート案内を出したか (1 回だけ聞く)
    "window_geometry": "",
    "ollama_host": "http://localhost:11434",
    "ollama_model": "qwen3-vl:8b-instruct",
    "ollama_models": [],  # 手動で追加した VLM 名 (検出一覧に無くても候補に出す)
    "ollama_timeout": 300,  # 秒。VRAM 不足などで VLM が載らない場合はここで諦めてタガー結果を返す
    "describer_model": "",  # 先に画像の説明を書かせるモデル (JOYCAPTION_MODEL など)。空なら使わない
    "last_dir": "",
    "vlm_seconds": {},  # VLM ごとに前回の分析にかかった秒数 (進み具合の見込みに使う)
    # 撮影プリセット (「構図と強さ」欄のプルダウン)。{"name": 画面の表示, "prompt": 入れる英語}。ここを書き換えて追加・編集できる
    "camera_looks": [dict(x) for x in DEFAULT_CAMERA_LOOKS],
}


MODES = {"single": "1 枚", "multi": "最大 8 枚 (メイン + ポーズ / 服装 / 背景 / 小物 / 体格 / 作風 / 表情)"}
GENDER_MODES = {"none": "そろえない", "female": "全員女性", "male": "全員男性"}
RESTYLE_MODES = {"none": "変えない", "photo": "実写", "anime": "アニメ風", "manga": "漫画風 (白黒)", "watercolor": "水彩",
                 "thick": "厚塗り", "oil": "油絵", "pencil": "鉛筆画"}
# 複数枚モードの参照枠。person は必須、ほかは任意で、空の枠の要素は 1 枚目 (person) から取る
REF_SLOTS = {
    "person": "① メイン", "pose": "② ポーズ", "outfit": "③ 服装・小物", "background": "④ 背景", "props": "⑤ 小物",
    "body": "⑥ 体格", "style": "⑦ 作風・質感", "expression": "⑧ 表情",
}
REF_WHAT_JA = {
    "person": "顔・髪・目・表情・キャラ (⑥ が無ければ体格も、⑦ が無ければ画風も)",
    "pose": "姿勢・手足・体の向き・構図 (視線は ⑧ 表情、⑧ が無ければ ①)",
    "outfit": "服・靴・帽子・装飾品・髪飾り (⑤ が無ければ持ち物・小物も)",
    "background": "場所・背景・照明",
    "props": "バッグ・傘・武器・本・飲み物・動物など (手に持つ物・置いてある物)",
    "body": "胸・体型・身長・手足・腰回り・肌の色・年齢の見た目・筋肉",
    # 絵 → 実写 / 実写 → 絵に使う。人種・肌の色は取らない (⑥ / ① のまま)
    # 絵なら色の塗り方、写真なら撮影環境 (カメラ・レンズ・光の当て方・粒子・色の調子) も ⑦ から取る
    "style": "画風・技法 (絵か写真か・線)・色の塗り方・写真の撮影環境 (カメラ・レンズの写り・光の当て方・粒子・色の調子)・"
             "作品名 / 作者名 (VLM の推定)",
    "expression": "表情・視線 (笑顔・赤面・涙・口の開き・目を閉じる・カメラ目線など)",
}

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff")
PREVIEW_SIZE = 256  # 1280 × 720 の画面でも左の列が収まる大きさ
VLM_MAX_SIDE = 896  # VLM に渡す画像の長辺 (大きいほど正確だが遅い)
# 複数枚モードの参照画像の長辺 (役割に足りる大きさに縮めて VLM の読み込みを速くする)
VLM_REF_MAX_SIDE = {"pose": 512, "outfit": 768, "background": 640, "props": 640, "body": 640, "style": 768, "expression": 640}
VLM_NUM_CTX = 12288  # 1 枚 / 2 枚で揃える (変えると Ollama がモデルを読み直して遅くなる)
VLM_NUM_PREDICT = 1600  # JSON 本体の上限
VLM_NUM_PREDICT_REFS = 2800  # 複数枚モードは項目 (ポーズ / 服装) が増えて長くなる (4 枚で 1600 だと途中で切れた)
# 思考モデルの上限。Ollama 0.34.4 + qwen3-vl:8b は think: false を無視して毎回 1000 トークン以上考えるため、
# 1200 では考えるだけで打ち切られて本文が空になっていた
VLM_NUM_PREDICT_THINKING = 4096
VLM_TIMEOUT = 300  # 秒 (config.json の ollama_timeout で上書き可)

# 説明を先に書かせるモデル (describer_model)。JoyCaption Beta One (Llama 3.1 8B + SigLIP の LLaVA 型、画像の説明を
# 書くことに特化し、内容で断らない) の GGUF。この Hugging Face のリポジトリは画像用の部品 (mmproj) も含み、Ollama の
# pull で一緒に入る (Q4_K_M 4.9GB + 0.9GB)。指示に従って JSON を埋めるのは得意ではないので、いつもの VLM の代わりには
# せず、書いた説明をいつもの VLM に参考として渡す (Engine._describe)
JOYCAPTION_MODEL = "hf.co/cinnabrad/llama-joycaption-beta-one-hf-llava-mmproj-gguf:Q4_K_M"
DESCRIBER_PROMPT = "Write a long detailed description for this image."   # JoyCaption Beta One の配布元の例と同じ
DESCRIBER_SYSTEM = "You are a helpful image captioner."
DESCRIBER_VLM = (
    "A captioning model wrote this description of image 1 before you. Use it to notice details you might miss "
    "(clothing parts, accessories, small objects, background, lighting, art style), but check each detail against "
    "the image and the tagger tags and drop what you cannot see; do not copy its wording:\n{text}"
)

ProgressFn = Callable[[str], None]


def log(msg: str) -> None:
    """stderr が無い環境 (pythonw) でも落ちないログ出力。"""
    if sys.stderr:
        try:
            print(msg, file=sys.stderr, flush=True)
        except (OSError, ValueError):
            pass


# --------------------------------------------------------------------------
# 設定
# --------------------------------------------------------------------------


def upright(im: Image.Image) -> Image.Image:
    """EXIF の向き (Orientation) どおりに起こした画像 (向きの情報が無ければそのままの写し)。スマホの縦写真が
    横倒しのままタガー・人物検出・VLM に渡らないように。ブラウザーや画像ビューアーも同じ向きで表示する。"""
    try:
        return ImageOps.exif_transpose(im)
    except Exception:  # noqa: BLE001 - 壊れた EXIF は向きを変えずに使う
        return im.copy()


def load_config() -> dict[str, Any]:
    data: dict[str, Any] = {}
    if CONFIG_PATH.exists():
        try:
            loaded = json.loads(CONFIG_PATH.read_text("utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, json.JSONDecodeError):
            pass
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(data)
    if "target" not in data and "style" in data:
        # 旧設定 (style: tags/natural) からの移行
        cfg["target"] = LEGACY_STYLE_TO_TARGET.get(str(data["style"]), DEFAULT_TARGET)
    cfg.pop("style", None)
    if cfg.get("engine") not in ENGINES:
        cfg["engine"] = DEFAULT_CONFIG["engine"]
    if cfg.get("tagger_model") not in TAGGER_MODELS:
        cfg["tagger_model"] = DEFAULT_TAGGER
    if cfg.get("target") not in TARGETS:
        cfg["target"] = DEFAULT_TARGET
    if cfg.get("mode") == "pose":
        cfg["mode"] = "multi"  # 2 枚モード (v1.0 の開発中) からの移行
    if cfg.get("mode") not in MODES:
        cfg["mode"] = DEFAULT_CONFIG["mode"]
    for gone in ("nsfw_boost", "nsfw_boost_level", "decensor", *(k for k in list(cfg) if k.startswith("comfy_"))):
        cfg.pop(gone, None)  # スケベアップ / 検閲を消すは無くなった (古い config.json のキーは捨てる)
    if cfg.get("gender") not in GENDER_MODES:
        cfg["gender"] = DEFAULT_CONFIG["gender"]
    if data.get("photoreal") is True and "restyle" not in data:
        cfg["restyle"] = "photo"  # 「実写化」のチェックだったころの設定
    cfg.pop("photoreal", None)
    cfg.pop("age", None)  # 年齢の指定は無くなった
    if cfg.get("restyle") not in RESTYLE_MODES:
        cfg["restyle"] = DEFAULT_CONFIG["restyle"]
    # 手で編集した config.json の型違い (null や文字列の true など) は既定値に戻す
    for key, default in DEFAULT_CONFIG.items():
        value = cfg.get(key)
        if isinstance(default, bool):
            ok = isinstance(value, bool)
        elif isinstance(default, (int, float)):
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        else:
            ok = isinstance(value, type(default))
        if not ok:
            cfg[key] = default
    cfg["ollama_models"] = [m for m in cfg["ollama_models"] if isinstance(m, str) and m.strip()]
    cfg["camera_looks"] = [  # 手で編集したときの名前・文の欠けたものは使わない
        {"name": x["name"].strip(), "prompt": x["prompt"].strip()} for x in cfg["camera_looks"]
        if isinstance(x, dict) and isinstance(x.get("name"), str) and isinstance(x.get("prompt"), str)
        and x["name"].strip() and x["prompt"].strip()
    ]
    for key in ("general_threshold", "character_threshold", "style_threshold"):
        cfg[key] = min(0.99, max(0.01, float(cfg[key])))
    return cfg


def save_config(cfg: dict[str, Any]) -> None:
    keep = {k: cfg.get(k, v) for k, v in DEFAULT_CONFIG.items()}
    tmp = CONFIG_PATH.with_name(CONFIG_PATH.name + ".tmp")
    tmp.write_text(json.dumps(keep, ensure_ascii=False, indent=2) + "\n", "utf-8")
    os.replace(tmp, CONFIG_PATH)


# --------------------------------------------------------------------------
# タグの整形
# --------------------------------------------------------------------------

# アンダースコアをスペースに置き換えない顔文字タグ
KAOMOJI = {
    "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<", "3_3",
    "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||",
}


def sd_tag(name: str) -> str:
    """Danbooru 形式のタグ名を Stable Diffusion のプロンプト用に整形する。"""
    tag = name if name in KAOMOJI else name.replace("_", " ")
    return tag.replace("(", r"\(").replace(")", r"\)")


def dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        # 'a \\(b\\)' と 'a (b)'、'a_b' と 'a b' を同じタグとみなす
        key = item.strip().lower().replace("\\(", "(").replace("\\)", ")").replace("_", " ")
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def clean_llm_tags(raw: Any, limit: int = 12) -> list[str]:
    """VLM が返したタグ配列を掃除する（重み構文の除去、小文字化、長さ制限、括弧のエスケープ）。
    タガーのタグ (sd_tag) と同じく () を \\( \\) にする。そのままだと SD 系で強調構文として読まれる。"""
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        tag = item.strip().strip(",.;").lower().replace("_", " ")
        if tag.startswith("(") and ":" in tag and tag.endswith(")"):
            tag = tag[1:].rsplit(":", 1)[0].strip()
        if 0 < len(tag) <= 40 and "\n" not in tag:
            out.append(sd_escape(tag))
        if len(out) >= limit:
            break
    return out


def llm_text(value: Any) -> str:
    """VLM の文字列欄を取り出す。スキーマを無視して配列や数値を返したときも崩れないようにする。"""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(v.strip() for v in value if isinstance(v, str)).strip()
    return ""


# --------------------------------------------------------------------------
# WD14 タガー (ONNX Runtime)
# --------------------------------------------------------------------------


@dataclass
class TagResult:
    rating: str
    rating_conf: float
    general: list[tuple[str, float]]  # (Danbooru タグ名, 信頼度) 信頼度の降順
    characters: list[tuple[str, float]]
    general_probs: dict[str, float]  # しきい値に関係なく全一般タグの信頼度


class DownloadError(Exception):
    pass


class OperationCancelled(Exception):
    """利用者が「■ 中止」を押した (解析・ダウンロード・pull・Ollama 導入)。"""


class PauseGate:
    """「⏸ 一時停止」(段階の切れ目で止める)。pause() の後、処理が次の切れ目 (wait) に来たら
    resume() まで待つ。今の段階 (タガー 1 枚分・VLM の 1 回の呼び出しなど) は途中では止めない。"""

    def __init__(self) -> None:
        self._go = threading.Event()
        self._go.set()

    def pause(self) -> None:
        self._go.clear()

    def resume(self) -> None:
        self._go.set()

    @property
    def paused(self) -> bool:
        return not self._go.is_set()

    def wait(self, cancelled: Callable[[], bool], on_pause: Callable[[], None] | None = None) -> None:
        """一時停止中なら再開まで待つ。待っている間に中止されたら OperationCancelled。"""
        if self._go.is_set():
            return
        if on_pause is not None:
            on_pause()
        while not self._go.wait(0.2):
            if cancelled():
                raise OperationCancelled()


DownloadProgress = Callable[[str, float], None]  # (表示文, 進捗 0.0〜1.0。不明なら -1)


def tagger_files(name: str) -> tuple[Path, Path]:
    local_dir = MODELS_DIR / name
    return local_dir / "model.onnx", local_dir / "selected_tags.csv"


def model_file_ok(path: Path) -> bool:
    """ダウンロード済みのモデルか (途中で止まった小さなファイルは除く)。"""
    try:
        return path.is_file() and path.stat().st_size > 1_000_000
    except OSError:
        return False


def tagger_ready(name: str) -> bool:
    model_path, csv_path = tagger_files(name)
    return csv_path.is_file() and model_file_ok(model_path)


def tagger_size_label(name: str) -> str:
    return TAGGER_MODELS[name][1].split("/")[-1].strip()


def download_file(
    url: str, dest: Path, progress: DownloadProgress | None = None, label: str = "",
    cancel: threading.Event | None = None,
) -> None:
    """URL を dest に保存する。途中経過は .part に書き、完了後に置き換える。
    cancel がセットされたら 1MB ごとの区切りで中止し、.part を消す。"""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise DownloadError(f"保存先フォルダを作成できません ({dest.parent}): {e}") from e
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": f"{APP_ID}/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            last = 0.0
            while True:
                if cancel is not None and cancel.is_set():
                    raise OperationCancelled()
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                now = time.time()
                if progress and (now - last >= 0.2 or done == total):
                    last = now
                    if total:
                        progress(f"{label} {done / total:.0%} ({done / 1e9:.2f} / {total / 1e9:.2f} GB)", done / total)
                    else:
                        progress(f"{label} {done / 1e6:.0f} MB", -1)
        if total and done != total:
            raise DownloadError(f"ダウンロードが途中で終了しました ({done}/{total} バイト)")
        tmp.replace(dest)
    except (DownloadError, OperationCancelled):
        tmp.unlink(missing_ok=True)
        raise
    except (OSError, TimeoutError, http.client.HTTPException, ValueError) as e:  # URLError / HTTPError も OSError の派生
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"ダウンロードに失敗しました: {e}") from e


def make_onnx_session(path: Path, cpu_only: bool = False) -> Any:
    """ONNX Runtime のセッションを作る。GPU (CUDA / DirectML) があれば使い、cpu_only なら CPU だけで動かす。
    CPU のときはスレッド数を半分にして、ほかの処理 (ComfyUI など) の CPU を奪いすぎないようにする。"""
    import onnxruntime as ort  # 起動を軽くするため遅延インポート

    wanted = ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider")
    available = ort.get_available_providers()
    providers = ["CPUExecutionProvider"] if cpu_only else ([p for p in wanted if p in available] or ["CPUExecutionProvider"])
    options = ort.SessionOptions()
    if "DmlExecutionProvider" in providers:
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if cpu_only:
        options.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
    try:
        return ort.InferenceSession(str(path), options, providers=providers)
    except Exception:
        # GPU が使えない状態 (スリープや画面オフのあとにドライバーが CUDA を起動できないなど) では、CPU で動かす
        if providers == ["CPUExecutionProvider"]:
            raise
        return make_onnx_session(path, cpu_only=True)


def tagger_is_pixai(name: str) -> bool:
    """PixAI Tagger (入力は RGB・NCHW・-1〜1 で正方形に引き伸ばす。レーティングは別のモデルで補う)。"""
    return name.startswith("pixai-")


# レーティングを出さないタガー (PixAI) のレーティング。safe / r15 / r18 の 3 つを WD の呼び方に直して使う
RATING_MODEL_URL = f"{HF_ENDPOINT}/deepghs/anime_rating/resolve/main/mobilenetv3_v1_pruned_ls0.1/model.onnx"
RATING_MODEL_PATH = MODELS_DIR / "rating" / "anime_rating_mobilenetv3.onnx"
RATING_LABELS = ["general", "sensitive", "explicit"]   # safe / r15 / r18 の順
CAMIE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # ImageNet
CAMIE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Camie Tagger v2 の配布元が勧める一般タグのしきい値 (モデルカードの Macro 最適値 0.492)。WD 向けの既定値 0.35 のまま
# Camie を使うと、Camie には低すぎてタグが増えすぎる。設定を既定値から変えていればその値を使う
CAMIE_GENERAL_THRESHOLD = 0.49


def tagger_is_camie(name: str) -> bool:
    """Camie Tagger (入力は縦横比を保って 512 に縮め、ImageNet の平均色で余白を埋める。出力はロジット)。"""
    return name.startswith("camie-")


# Camie の分類 → WD の selected_tags.csv の category (0 一般 / 1 絵師 / 3 作品 / 4 キャラ / 5 メタ / 9 レーティング)。
# 年 (year_2005 など) は WD に無いので 7 にする (タグとしては使わない)
CAMIE_CATEGORY = {"general": 0, "artist": 1, "copyright": 3, "character": 4, "meta": 5, "year": 7, "rating": 9}


def camie_tags_csv(metadata_path: Path, csv_path: Path) -> None:
    """Camie の metadata.json (idx_to_tag・tag_to_category) を WD と同じ selected_tags.csv の形にする。
    レーティングは rating_general などの名前を WD の呼び方 (general / sensitive / questionable / explicit) に直す。"""
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    mapping = meta["dataset_info"]["tag_mapping"]
    idx_to_tag, tag_to_cat = mapping["idx_to_tag"], mapping["tag_to_category"]
    tmp = csv_path.with_name(csv_path.name + ".part")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tag_id", "name", "category", "count"])
        for i in range(len(idx_to_tag)):
            tag = idx_to_tag[str(i)]
            cat = CAMIE_CATEGORY.get(tag_to_cat.get(tag, "general"), 0)
            w.writerow([i, tag[len("rating_"):] if cat == 9 and tag.startswith("rating_") else tag, cat, 0])
    os.replace(tmp, csv_path)


def download_tagger(name: str, progress: DownloadProgress | None = None, cancel: threading.Event | None = None) -> None:
    """タガーモデル一式 (selected_tags.csv, model.onnx) を Hugging Face から models/ に取得する。"""
    if name not in TAGGER_MODELS:
        raise ValueError(f"未知のタガーモデルです: {name}")
    repo = TAGGER_MODELS[name][0]
    model_path, csv_path = tagger_files(name)
    if tagger_is_camie(name):   # 配布元のファイル名で取って、model.onnx / selected_tags.csv として置く
        meta_path = model_path.with_name("metadata.json")
        if not model_file_ok(meta_path):
            download_file(f"{HF_ENDPOINT}/{repo}/resolve/main/camie-tagger-v2-metadata.json", meta_path, progress,
                          label=f"{name} / metadata", cancel=cancel)
        if not csv_path.is_file():
            camie_tags_csv(meta_path, csv_path)
        if not model_file_ok(model_path):
            download_file(f"{HF_ENDPOINT}/{repo}/resolve/main/camie-tagger-v2.onnx", model_path, progress,
                          label=f"{name} / model", cancel=cancel)
        return
    for fname, dest in (("selected_tags.csv", csv_path), ("model.onnx", model_path)):
        if dest.is_file() and (fname != "model.onnx" or model_file_ok(dest)):
            continue
        download_file(f"{HF_ENDPOINT}/{repo}/resolve/main/{fname}", dest, progress, label=f"{name} / {fname}", cancel=cancel)
    if tagger_is_pixai(name) and not model_file_ok(RATING_MODEL_PATH):
        download_file(RATING_MODEL_URL, RATING_MODEL_PATH, progress, label="anime_rating", cancel=cancel)


class WD14Tagger:
    # __init__ を通らずに作ったタガー (テストの差し替え用) でも predict が動くよう、後から足した属性の既定値を置く
    camie = False
    pixai = False
    rating_session = None
    outputs = None  # None = モデルの出力すべて (WD タガーは 1 つ)
    logits = False

    def __init__(self, name: str, progress: ProgressFn | None = None, cpu_only: bool = False):
        if name not in TAGGER_MODELS:
            raise ValueError(f"未知のタガーモデルです: {name}")
        model_path, csv_path = tagger_files(name)

        if not tagger_ready(name):
            # 初回のみダウンロード (CLI 用。GUI は事前に確認ダイアログを出す)。以降はローカルのファイルだけを使う。
            if progress:
                progress(f"タガーモデルをダウンロード中 ({name}, {tagger_size_label(name)})... 初回のみ")
            download_tagger(name, (lambda msg, _frac: progress(msg)) if progress else None)

        if progress:
            progress(f"タガーモデルを読み込み中 ({name})...")
        self.session = make_onnx_session(model_path, cpu_only)
        self.provider = self.session.get_providers()[0]
        self.cpu_only = cpu_only
        first = self.session.get_inputs()[0]
        self.input_name = first.name
        self.pixai = tagger_is_pixai(name)
        self.camie = tagger_is_camie(name)
        self.logits = False   # 出力がロジット (sigmoid で確率にする) か
        if self.camie:   # NCHW (1, 3, 512, 512)。出力は initial / refined_predictions (ロジット) / selected_candidates
            self.size = first.shape[2] if isinstance(first.shape[2], int) else 512
            outs = [o.name for o in self.session.get_outputs()]
            self.outputs = ["refined_predictions"] if "refined_predictions" in outs else None
            self.logits = True
            self.rating_session = None
        elif self.pixai:   # NCHW (1, 3, 448, 448)。出力は prediction (確率) / embedding / logits
            self.size = first.shape[2] if isinstance(first.shape[2], int) else 448
            outs = [o.name for o in self.session.get_outputs()]
            self.outputs = ["prediction"] if "prediction" in outs else None
            if not model_file_ok(RATING_MODEL_PATH):
                if progress:
                    progress("レーティングの判定モデルをダウンロード中 (17MB)... 初回のみ")
                download_file(RATING_MODEL_URL, RATING_MODEL_PATH, None, label="anime_rating")
            self.rating_session = make_onnx_session(RATING_MODEL_PATH, cpu_only)
        else:
            self.size = first.shape[1] if isinstance(first.shape[1], int) else 448
            self.outputs = None
            self.rating_session = None

        self.names: list[str] = []
        self.categories: list[int] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.names.append(row["name"])
                self.categories.append(int(row["category"]))
        self.name = name

    def preprocess(self, image: Image.Image) -> np.ndarray:
        rgba = image.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        canvas.alpha_composite(rgba)
        rgb = canvas.convert("RGB")
        if self.camie:   # 配布元の onnx_inference.py: 縦横比を保って LANCZOS で縮め、(124, 116, 104) で 512 四方に、ImageNet の正規化
            w, h = rgb.size
            nw, nh = (self.size, max(1, int(self.size * h / w))) if w > h else (max(1, int(self.size * w / h)), self.size)
            square = Image.new("RGB", (self.size, self.size), (124, 116, 104))
            square.paste(rgb.resize((nw, nh), Image.LANCZOS), ((self.size - nw) // 2, (self.size - nh) // 2))
            arr = (np.asarray(square, dtype=np.float32) / 255.0 - CAMIE_MEAN) / CAMIE_STD
            return np.ascontiguousarray(arr.transpose(2, 0, 1)[None].astype(np.float32))
        if self.pixai:   # 配布元の preprocess.json: 448x448 に引き伸ばし (bilinear)、0〜1、平均 0.5・標準偏差 0.5、CHW
            arr = (np.asarray(rgb.resize((self.size, self.size), Image.BILINEAR), dtype=np.float32) / 255.0 - 0.5) / 0.5
            return np.ascontiguousarray(arr.transpose(2, 0, 1)[None])
        w, h = rgb.size
        side = max(w, h)
        square = Image.new("RGB", (side, side), (255, 255, 255))
        square.paste(rgb, ((side - w) // 2, (side - h) // 2))
        square = square.resize((self.size, self.size), Image.BICUBIC)
        arr = np.asarray(square, dtype=np.float32)[:, :, ::-1]  # RGB -> BGR、0..255 のまま
        return np.ascontiguousarray(arr[None])

    def predict(self, image: Image.Image, general_th: float, character_th: float, flip: bool = False) -> TagResult:
        """flip=True なら左右反転した画像でもタグ付けし、確率を平均する (しきい値の近くのタグの揺れを減らす。時間は 2 倍)。
        タガーの語彙に左右 (left / right) を含む一般タグは無いので、反転してもタグの意味は変わらない。"""
        if self.camie and abs(general_th - DEFAULT_CONFIG["general_threshold"]) < 1e-6:
            general_th = CAMIE_GENERAL_THRESHOLD
        probs = self._probs(image)
        if flip:
            probs = (probs + self._probs(image.transpose(Image.Transpose.FLIP_LEFT_RIGHT))) / 2
        ratings: dict[str, float] = self.rate(image) if self.rating_session is not None else {}
        general: list[tuple[str, float]] = []
        characters: list[tuple[str, float]] = []
        general_probs: dict[str, float] = {}
        for name, category, p in zip(self.names, self.categories, probs.tolist()):
            if category == 9:
                ratings[name] = p
            elif category == 0:
                general_probs[name] = p
                if p >= (general_th if self.camie else min(general_th, LOW_THRESHOLD_TAGS.get(name, general_th))):
                    general.append((name, p))
            elif category == 4 and p >= character_th:
                characters.append((name, p))
        general.sort(key=lambda t: -t[1])
        characters.sort(key=lambda t: -t[1])
        rating = max(ratings, key=ratings.get) if ratings else "unknown"
        return TagResult(rating, ratings.get(rating, 0.0), general, characters, general_probs)

    def _probs(self, image: Image.Image) -> np.ndarray:
        out = np.asarray(self.session.run(self.outputs, {self.input_name: self.preprocess(image)})[0][0], dtype=np.float32)
        return 1.0 / (1.0 + np.exp(-out)) if self.logits else out

    def rate(self, image: Image.Image) -> dict[str, float]:
        """レーティングを出さないタガーのレーティング (deepghs/anime_rating。imgutils と同じ 384x384・-1〜1・CHW)。"""
        rgba = image.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        canvas.alpha_composite(rgba)
        arr = (np.asarray(canvas.convert("RGB").resize((384, 384), Image.BILINEAR), dtype=np.float32) / 255.0 - 0.5) / 0.5
        inp = self.rating_session.get_inputs()[0].name
        out = np.asarray(self.rating_session.run(None, {inp: np.ascontiguousarray(arr.transpose(2, 0, 1)[None])})[0][0],
                         dtype=np.float64)
        if not (out.min() >= 0 and abs(out.sum() - 1.0) < 1e-3):   # 確率でなければ softmax
            out = np.exp(out - out.max())
            out /= out.sum()
        return {label: float(p) for label, p in zip(RATING_LABELS, out)}


# タグごとのしきい値 (全体のしきい値より低い方を使う。WD タガーで測った値なので Camie には使わない)。Danbooru の全年齢 532 枚で実測し (2026-09-27)、
# 当たりが大きく増え、偏りの無い 322 枚で誤検出が少ししか増えなかった値だけ (DEVLOG 38)。
# 例: angry 17% → 42%、surprised 13% → 39%、motion lines 21% → 45%、speed lines 5% → 32%、pov 57% → 78%、from behind 57% → 83%。
# expressionless / frown / scared / from side / dutch angle は誤検出が増えるか効果が小さいので入れない。
# VLM (qwen2.5vl / qwen3.5) に表情・効果線を答えさせても、タガーのしきい値を下げた以上には当たらなかったので使わない
LOW_THRESHOLD_TAGS = {
    **{t: 0.20 for t in ("angry", "sad", "surprised", "crying", "pout", "smirk", "emphasis_lines", "motion_blur",
                         "from_behind", "pov", "wide_shot", "from_below")},
    "motion_lines": 0.15, "embarrassed": 0.15, "speed_lines": 0.10,
}

# --------------------------------------------------------------------------
# NSFW 判定とプロンプトへのモザイク指定
# --------------------------------------------------------------------------

# レーティングが general でもこれらのタグがあれば NSFW とみなす
NSFW_HINT_TAGS = {
    "nude", "completely_nude", "topless", "bottomless", "nipples", "pussy", "penis", "anus", "sex",
    "vaginal", "anal", "fellatio", "paizuri", "cum", "uncensored", "censored", "bar_censor", "mosaic_censoring",
}


def nsfw_suspected(tags: TagResult) -> bool:
    if tags.rating in ("questionable", "explicit"):
        return True
    return bool({name for name, _ in tags.general} & NSFW_HINT_TAGS)


# プロンプト側のモザイク指定: モザイク以外の検閲タグと uncensored は外し、censored + mosaic censoring を入れる
# タガーが出す検閲のタグ。本文には入れず「検出タグ」に表示だけする。
# Danbooru の検閲の無い 110 枚で 20 枚 (35 件、確信度 0.35〜0.94) を誤認した (ポーズで見えにくい所・モザイクに似た模様)。
# モザイクを入れたいときは設定「プロンプトにモザイク指定を追加」を使う。uncensored は対象外
CENSOR_DETECT_TAGS = {
    "censored", "mosaic_censoring", "bar_censor", "blur_censor", "heart_censor", "light_censor", "steam_censor",
    "convenient_censoring", "novelty_censor", "hair_censor", "tail_censor", "character_censor", "soap_censor",
    "glitch_censor", "identity_censor", "censored_identity", "glowing_censor", "pointless_censoring", "censored_nipples",
    "sparkle_censor", "star_censor", "scribble_censor", "shadow_censor", "tape_censor", "censored_text",
}
CENSOR_TYPE_TAGS = {
    "uncensored", "bar censor", "blur censor", "heart censor", "light censor", "steam censor",
    "convenient censoring", "novelty censor", "hair censor", "tail censor", "character censor",
    "soap censor", "glitch censor", "identity censor", "glowing censor", "pointless censoring",
}
# モザイクの強さ: タグ形式は重み付きタグ、自然文は文で強調する
# タグは (タグ名, 重み) の組。重みの書き方は出力先ごとに変える (SD 系 "(tag:1.3)" / NovelAI "1.3::tag::")
MOSAIC_LEVELS: dict[str, dict[str, Any]] = {
    "normal": {
        "label": "標準",
        "tags": [("censored", None), ("mosaic censoring", None)],
        "negative": [("uncensored", None)],
        "sentence": "Explicit areas are covered by a pixelated mosaic censor.",
    },
    "strong": {
        "label": "強め",
        "tags": [("censored", None), ("mosaic censoring", 1.3), ("coarse mosaic", None), ("large pixel blocks", None)],
        "negative": [("uncensored", 1.3)],
        "sentence": "Explicit areas are covered by a heavy, coarse pixelated mosaic censor with large blocks.",
    },
    "max": {
        "label": "最強",
        "tags": [
            ("censored", None), ("mosaic censoring", 1.5), ("coarse mosaic", 1.3),
            ("large pixel blocks", None), ("heavily censored", None),
        ],
        "negative": [("uncensored", 1.5)],
        "sentence": (
            "Explicit areas are completely hidden behind an extremely coarse pixelated mosaic censor "
            "with very large blocks."
        ),
    },
}
DEFAULT_MOSAIC_LEVEL = "strong"
_MOSAIC_SENTENCE_KEY = "pixelated mosaic censor"  # 二重付加の判定用
_MOSAIC_OWN_TAGS = {"censored", "mosaic censoring", "coarse mosaic", "large pixel blocks", "heavily censored", "uncensored"}
_NAI_WEIGHT_RE = re.compile(r"^(-?\d+(?:\.\d+)?)::(.+)::$")


def _tag_key(tag: str) -> str:
    """重み付きタグ ((nude:1.2) / 1.2::nude::) も含めて、比べるための Danbooru 形のタグ名にする。"""
    return norm_tag(_strip_weight(tag)).replace(" ", "_")


def force_gender(tags: TagResult, gender: str) -> TagResult:
    """人数タグを全員女性 / 全員男性にそろえる (girls / boys / others の 3 系列を付け替え)。
    Danbooru の male_focus は「男性キャラだけ」なので、全員男性なら付け、全員女性なら外す。female_focus は廃止タグなので出さない。
    性器・胸などの身体のタグは内容なので変えない。"""
    if gender not in ("female", "male"):
        return tags
    total, _ = person_count_hint(tags)
    if total <= 0:
        return tags
    group = "girl" if gender == "female" else "boy"
    drop = set(PERSON_COUNT_TAGS) | {"male_focus", "female_focus", "other_focus"}
    first_index = next((i for i, (n, _) in enumerate(tags.general) if n in PERSON_COUNT_TAGS), 0)
    rest = [(n, p) for n, p in tags.general if n not in drop]
    counts = [(n, 1.0) for n in count_tags([group] * total)]
    if gender == "male":
        counts.append(("male_focus", 1.0))
    general = rest[:first_index] + counts + rest[first_index:]
    return TagResult(tags.rating, tags.rating_conf, general, list(tags.characters), dict(tags.general_probs))


def force_person_gender(persons: list[dict[str, Any]], gender: str) -> list[dict[str, Any]]:
    """人物ごとの性別 (区画の先頭 1girl / 1boy) をそろえる。"""
    if gender not in ("female", "male"):
        return persons
    subject = "girl" if gender == "female" else "boy"
    out = []
    for person in persons:
        tags = [t for t in person["tags"] if _tag_key(t) not in ("male_focus", "female_focus")]
        out.append(person | {"subject": subject, "tags": tags, "prompt": ", ".join(dedup([SUBJECT_TOKEN[subject]] + tags))})
    return out


# --------------------------------------------------------------------------
# 年齢の見た目 (画像から読んだとおりに書く)
# --------------------------------------------------------------------------
# 出力に年齢が無いと、生成する側のモデルの癖で年齢が決まる (文章で書くモデルは 20 代の大人に、アニメ系のタグのモデルは
# 若く寄る)。人物ごとに VLM が年齢層を選び、タガーの年齢のタグ (child / mature female / old man など) と合わせて、
# 文とタグに書く。年齢を選ばせる欄は VLM_SCHEMA の age (1 人のとき) と persons の age (2 人以上)
AGE_BANDS = ("baby", "toddler", "child", "preteen", "teen", "young adult", "thirties", "forties", "fifties", "sixties", "elderly")
AGE_ENUM = [*AGE_BANDS, "unknown"]
AGE_BAND_JA = {
    "baby": "赤ちゃん", "toddler": "幼児 (1〜3 歳)", "child": "子ども (4〜9 歳)", "preteen": "10〜12 歳", "teen": "13〜17 歳",
    "young adult": "18〜29 歳", "thirties": "30 代", "forties": "40 代", "fifties": "50 代", "sixties": "60 代",
    "elderly": "70 歳以上", "middle-aged": "中年 (タガーの mature のタグから)",
}
AGE_MINOR = ("baby", "toddler", "child", "preteen", "teen")
AGE_OLD = ("sixties", "elderly")
# タガーの年齢のタグ → 年齢層。mature は幅が広いので「中年」とし、VLM が大人の年齢層を選んでいればそちらを使う
AGE_TAG_BANDS = {
    "baby": "baby", "toddler": "toddler", "child": "child", "female_child": "child", "male_child": "child",
    "loli": "child", "shota": "child", "old": "elderly", "old_woman": "elderly", "old_man": "elderly",
    "mature_female": "middle-aged", "mature_male": "middle-aged", "milf": "middle-aged",
}
AGE_VLM_GROUPS = ("baby (under 1), toddler (1-3), child (4-9), preteen (10-12), teen (13-17), young adult (18-29), thirties, "
                  "forties, fifties, sixties, elderly (70 or older)")
AGE_VLM_JUDGE = ("Judge from the face, the skin, the body proportions and the height against the surroundings; for a drawn "
                 "character, from the age the drawing portrays. Do not fall back on young adult: choose the group that "
                 "really fits, for children and for older people as well.")
# 年齢の見た目を書くとき、VLM の文 (自然文・被写体・人物ごとの説明) にも年齢に合った言い方をさせる
AGE_VLM = (
    "Apparent age: in natural_prompt, subject and every description, whenever you refer to a person with a common noun, "
    "use wording that fits the age group you choose for them in the age fields (for example 'a little girl', "
    "'a preteen boy', 'a teenage girl', 'a woman in her forties', 'an elderly man') instead of a generic "
    "'young woman' or 'girl'. This is about the common noun only: where you would write a character's name, keep "
    "writing the name."
)
_AGE_PREFIX = {"baby": "baby", "toddler": "toddler", "child": "little", "preteen": "preteen", "teen": "teenage",
               "elderly": "elderly", "middle-aged": "middle-aged"}
_AGE_DECADE = {"young adult": "twenties", "thirties": "thirties", "forties": "forties", "fifties": "fifties",
               "sixties": "sixties"}


def clean_age(value: Any) -> str:
    """VLM が返した年齢層。決めた語以外 (unknown も) は空にする。"""
    text = str(value or "").strip().lower().replace("_", " ")
    return text if text in AGE_BANDS else ""


def age_from_tags(tag_list: list[str]) -> str:
    """タガーの年齢のタグから年齢層を決める (無ければ空)。幼いほうから見る。"""
    keys = {_tag_key(t) for t in tag_list}
    for band in ("baby", "toddler", "child", "elderly", "middle-aged"):
        if any(AGE_TAG_BANDS.get(k) == band for k in keys):
            return band
    return ""


def merge_age(vlm_band: str, tag_band: str) -> str:
    """VLM が選んだ年齢層とタガーのタグを合わせる。子どもかどうか・高齢かどうかが食い違えばタガーのタグ
    (その語で学習しているぶん確か)、食い違わなければ細かく読める VLM のほう。"""
    if not vlm_band or not tag_band:
        return vlm_band or tag_band
    if (vlm_band in AGE_MINOR) != (tag_band in AGE_MINOR):
        return tag_band
    if tag_band == "elderly" and vlm_band not in AGE_OLD:
        return tag_band
    return vlm_band


def age_noun_en(person: dict[str, Any], article: bool = True) -> str:
    """人物を年齢に合った言い方で呼ぶ名詞句 (a little girl of elementary-school age / a teenage boy /
    a woman in her forties / an elderly man)。年齢が分からなければ a woman / a man / a person。"""
    subject, band = person.get("subject", ""), person.get("age", "")
    minor = band in AGE_MINOR
    noun = ({"girl": "girl", "boy": "boy"} if minor else {"girl": "woman", "boy": "man"}).get(
        subject, "child" if minor else "person")
    if subject not in ("girl", "boy") and band in ("baby", "toddler"):
        phrase = band
    elif subject not in ("girl", "boy") and band == "teen":
        phrase = "teenager"
    else:
        phrase = " ".join(x for x in (_AGE_PREFIX.get(band, ""), noun) if x)
    if band == "child":
        phrase += " of elementary-school age"
    elif band in _AGE_DECADE:
        phrase += f" in {({'girl': 'her', 'boy': 'his'}).get(subject, 'their')} {_AGE_DECADE[band]}"
    return _with_article(phrase, capital=False) if article else phrase


def group_noun_en(persons: list[dict[str, Any]]) -> str:
    """全員をまとめて呼ぶ語 (women / men / girls / boys)。性別か、子どもと大人が混ざれば people。"""
    genders = {p.get("subject") for p in persons}
    minors = {p.get("age") in AGE_MINOR for p in persons}
    if len(genders) != 1 or len(minors) != 1:
        return "people"
    minor = True in minors
    return ({"girl": "girls", "boy": "boys"} if minor else {"girl": "women", "boy": "men"}).get(
        next(iter(genders)), "children" if minor else "people")


def without_age_tags(person: dict[str, Any]) -> list[str]:
    """文に書く人物の特徴。年齢を名詞のほうに書いたときは、年齢のタグ (child / mature female など) を重ねて書かない。"""
    return [t for t in person["tags"] if not (person.get("age") and _tag_key(t) in AGE_TAG_BANDS)]


def age_tags_for(subject: str, band: str) -> list[str]:
    """タグの出力先に足す年齢のタグ (Danbooru の語。10 代と 20 代に当たるタグは無いので足さない)。"""
    known = subject in ("girl", "boy")
    if band in ("baby", "toddler"):
        return [band]
    if band in ("child", "preteen"):
        return ["child"] + ([{"girl": "female child", "boy": "male child"}[subject]] if known else [])
    if band in ("thirties", "forties", "fifties", "middle-aged"):
        return [{"girl": "mature female", "boy": "mature male"}[subject]] if known else []
    if band in AGE_OLD:
        return [{"girl": "old woman", "boy": "old man"}.get(subject, "old")]
    return []


def with_age_tags(person: dict[str, Any]) -> dict[str, Any]:
    """人物の区画の先頭 (1girl などのすぐ後ろ) に年齢のタグを足す。すでにあるタグは足さない。"""
    have = {_tag_key(t) for t in list(person.get("tags") or []) + list(person.get("common") or [])}
    add = [t for t in age_tags_for(person.get("subject", ""), person.get("age", "")) if _tag_key(t) not in have]
    if not add:
        return person
    parts = [x for x in str(person.get("prompt") or "").split(", ") if x]
    head = parts[:1] if parts and _tag_key(parts[0]) in PERSON_COUNT_TAGS else []
    return person | {"tags": add + list(person.get("tags") or []),
                     "prompt": ", ".join(dedup(head + add + parts[len(head):]))}


def age_negatives(people: list[tuple[str, str]]) -> list[str]:
    """年齢がずれないようにネガティブへ入れるタグ。people は人物ごとの (性別, 年齢層)。全員が大人なら幼くなるのを、
    全員が子どもなら大人になるのを抑える。読めなかった人がいるとき・子どもと大人が混ざるときは入れない
    (片方を打ち消してしまう)。"""
    if not people or not all(band for _, band in people):
        return []
    subjects = {s for s, _ in people}
    if all(band not in AGE_MINOR for _, band in people):
        return ["child", "aged down"] + (["loli"] if "girl" in subjects else []) + (["shota"] if "boy" in subjects else [])
    if all(band in AGE_MINOR for _, band in people):
        return ["aged up"] + (["mature female"] if "girl" in subjects else []) + (["mature male"] if "boy" in subjects else [])
    return []


def age_negatives_prose(people: list[tuple[str, str]]) -> list[str]:
    """age_negatives の、文章で書く出力先 (MOSS-Image のネガティブ欄) 向けの語。Danbooru のタグではなく、ふつうの英語で書く。
    全員が大人なら幼く・若くなるのを、全員が子どもなら大人になるのを抑える。読めなかった人がいるとき・子どもと大人が
    混ざるときは入れない。"""
    bands = [band for _, band in people]
    if not bands or not all(bands):
        return []
    if all(b not in AGE_MINOR for b in bands):
        older = all(b not in ("young adult", "thirties") for b in bands)  # 40 代から上だけ: 20 代に寄るのも抑える
        return ["child", "childlike face", "teenager"] + (["young adult", "youthful face"] if older else [])
    if all(b in AGE_MINOR for b in bands):
        return ["adult", "mature face", "middle-aged"] + (["teenager"] if all(b != "teen" for b in bands) else [])
    return []


def single_subject(tags: TagResult) -> str:
    """1 人の画像の性別 (人数タグから)。1girl だけなら girl、1boy だけなら boy、ほかは other。"""
    names = {n for n, _ in tags.general}
    girl, boy = "1girl" in names, "1boy" in names
    return "girl" if girl and not boy else "boy" if boy and not girl else "other"


def assign_ages(
    persons: list[dict[str, Any]], vlm_age: str, numbered: list[dict[str, Any]], tags: TagResult, nsfw: bool,
    single: bool,
) -> tuple[list[dict[str, Any]], str, str]:
    """人物ごとの年齢層を決める。2 人以上なら persons の各人に "age" を入れ、1 人 (single) ならその年齢層を返す。
    VLM の読み (2 人以上は persons の age か、番号付きの枠の番号で結び付けた age) とタガーのタグを merge_age で合わせる。
    性的な内容の画像 (nsfw) では、18 歳未満に見える人物の年齢は書かない。戻り値: (人物, 1 人のときの年齢層, 分析メモの行)。"""
    out = [dict(p) for p in persons]
    if len(out) >= 2:
        by_number = {e.get("number"): e.get("age", "") for e in numbered}
        for i, person in enumerate(out, 1):
            vlm_band = person.get("age") or by_number.get(i, "")
            person["age"] = merge_age(vlm_band, age_from_tags(list(person.get("tags") or []) + list(person.get("common") or [])))
        bands = [p["age"] for p in out]
    elif single:
        bands = [merge_age(vlm_age, age_from_tags([n for n, _ in tags.general]))]
    else:
        return out, "", ""
    held = nsfw and any(b in AGE_MINOR for b in bands)
    if held:
        bands = ["" if b in AGE_MINOR else b for b in bands]
        for person, band in zip(out, bands, strict=False):
            person["age"] = band
    if len(out) >= 2:
        read = "、".join(f"人物 {i} = {AGE_BAND_JA.get(b, '読めませんでした')}" for i, b in enumerate(bands, 1))
    else:
        read = AGE_BAND_JA.get(bands[0], "")
    note = ""
    if any(bands):
        note = f"\n年齢の見た目 (画像から読んで本文に書きました。VLM とタガーの読み取りなので、外れることがあります): {read}"
    if held:
        note += "\n年齢の見た目: 性的な内容と判定した画像では、18 歳未満に見える人物の年齢は書けません"
    return out, ("" if len(out) >= 2 else bands[0]), note


_AGE_SAID = {
    "baby": r"\b(?:baby|babies|infant|newborn)\b",
    "toddler": r"\btoddlers?\b",
    "child": r"\b(?:little|small)\s+(?:girl|boy|child|kid)\b|\byoung\s+child\b|\belementary[- ]school\b",
    "preteen": r"\bpre-?teen|\btween\b",
    "teen": r"\bteen(?:s|age|aged|ager)?\b|\badolescent\b",
    "young adult": r"\btwenties\b|\b20s\b|\byoung\s+(?:adult|woman|man|lady)\b",
    "thirties": r"\bthirties\b|\b30s\b", "forties": r"\bforties\b|\b40s\b", "fifties": r"\bfifties\b|\b50s\b",
    "sixties": r"\bsixties\b|\b60s\b",
    "elderly": r"\b(?:elderly|senior|seventies|eighties|nineties|70s|80s)\b|\bold\s+(?:woman|man|lady|person)\b",
    "middle-aged": r"\bmiddle[- ]aged\b|\bmature\b",
}
_AGE_ADJ = (r"(?:very\s+)?(?:young|younger|little|small|adult|grown-up|grown|mature|middle-aged|elderly|old|older|aged|"
            r"teenage|teen|adolescent|youthful|preteen|pre-teen),?\s+")
# 冠詞と名詞の間は形容詞だけ (a slender young woman)。前置詞や冠詞をまたがない (the back of a woman の the からは取らない)
_AGE_NP_RE = re.compile(
    r"\b(?P<art>an?|the|one)\s+"
    r"(?P<mid>(?:(?!(?:of|with|and|or|in|on|at|by|to|for|from|the|an?|is|as|her|his|their)\b)[a-z][a-z-]*,?\s+){0,3}?)"
    r"(?P<noun>woman|girl|man|boy|lady|gentleman|female|male|person|child|kid|teenager|toddler|baby)\b"
    r"(?P<suffix>\s+of\s+elementary-school\s+age|\s+in\s+(?:her|his|their)\s+(?:early\s+|mid[- ]|late\s+)?"
    r"(?:teens|twenties|thirties|forties|fifties|sixties|seventies|eighties|[1-9]0s))?",
    re.IGNORECASE)
_AGE_FEMALE = {"woman", "girl", "lady", "female"}
_AGE_MALE = {"man", "boy", "gentleman", "male"}


def age_mentioned(text: str, band: str) -> bool:
    """文が、その年齢層だと分かる言い方をもうしているか (引用符の中は画像の文字なので見ない)。"""
    plain = re.sub(r'"[^"]*"', " ", text)
    return bool(band in _AGE_SAID and re.search(_AGE_SAID[band], plain, re.IGNORECASE))


def apply_age_text(text: str, persons: list[dict[str, Any]], main_age: str, subject: str = "other") -> str:
    """文章の本文に年齢を書く。1 人なら、人物を呼んでいる最初の名詞句 (a young woman など) を年齢に合った言い方に
    言い換え、後ろの the woman / the girl も合わせる。名詞句が無ければ一文を足す。2 人以上は、各人の年齢が文から
    読み取れなければ、左からの並びで一文を足す。年齢がもう書いてあれば何もしない。"""
    if len(persons) >= 2:
        known = [p for p in persons if p.get("age")]
        if not known or all(age_mentioned(text, p["age"]) for p in known):
            return text
        nouns = [age_noun_en(p) for p in persons]
        return text.rstrip() + f" From left to right, the people are {', '.join(nouns[:-1])} and {nouns[-1]}."
    if not main_age or age_mentioned(text, main_age):
        return text
    done = [""]

    def named(m: "re.Match[str]") -> str:
        if done[0]:
            return m.group(0)
        noun = m.group("noun").lower()
        who = "girl" if noun in _AGE_FEMALE else "boy" if noun in _AGE_MALE else subject
        done[0] = who
        mid = re.sub(_AGE_ADJ, "", m.group("mid"), flags=re.IGNORECASE)
        phrase = mid + age_noun_en({"subject": who, "age": main_age}, article=False)
        art = m.group("art")
        if art.lower() in ("a", "an"):
            new = "an" if phrase[:1].lower() in "aeiou" else "a"
            art = new.capitalize() if art[:1].isupper() else new
        return f"{art} {phrase}"

    out = _unquoted_sub(_AGE_NP_RE, named, text)
    if not done[0]:
        return out.rstrip() + f" The person shown is {age_noun_en({'subject': subject, 'age': main_age})}."
    # 後ろで同じ人を呼ぶ the young woman / the girl も合わせる (1 人の画像なので、ほかの人を指すことは無い)
    minor = main_age in AGE_MINOR
    again = re.compile(r"\b(the|this|that)\s+(?:" + _AGE_ADJ + r")*(woman|lady|girl|man|gentleman|boy)\b", re.IGNORECASE)

    def same(m: "re.Match[str]") -> str:
        female = m.group(2).lower() in _AGE_FEMALE
        return f"{m.group(1)} " + (("girl" if female else "boy") if minor else ("woman" if female else "man"))

    return _tidy_prose(_unquoted_sub(again, same, out))


GENDER_VLM = {
    "female": "Every person in the image is a woman. Describe all of them as women or girls and use she/her; "
              "fill persons with subject 'girl'.",
    "male": "Every person in the image is a man. Describe all of them as men or boys and use he/him; "
            "fill persons with subject 'boy'.",
}


def _unquoted_sub(pattern: "re.Pattern[str]", repl: Any, text: str, count: int = 0) -> str:
    """二重引用符の外だけを置き換える (引用符の中は画像の文字の書き写しなので変えない)。count は全体での回数 (0 = すべて)。"""
    parts = re.split(r'("[^"]*")', text)
    left = count
    for i in range(0, len(parts), 2):
        if count and left <= 0:
            break
        parts[i], n = pattern.subn(repl, parts[i], count=left if count else 0)
        left -= n
    return "".join(parts)


def _tidy_prose(text: str) -> str:
    """語を抜いた後の文を整える (空白・句読点・文頭の大文字)。"""
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r",(?:\s*,)+", ",", text)
    text = re.sub(r"([.!?])\s*,\s*", r"\1 ", text)
    # 語を抜いた後の冠詞 (an anime girl → an girl → a girl、a stylized elf → a elf → an elf)。u / eu / one で始まる語は
    # 子音の音が多いので触らない (a uniform)
    text = re.sub(r"\b([Aa])n (?=[b-df-gj-np-tv-z])", r"\1 ", text)
    text = re.sub(r"\b([Aa]) (?=(?!one\b|once\b|eu)[aeioAEIO]\w)", r"\1n ", text)
    text = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text.strip())
    return text


def format_weight(tag: str, weight: float | None, target: str) -> str:
    """重み付きタグを出力先の構文で書く。"""
    if weight is None:
        return tag
    if target == "novelai":
        return f"{weight}::{tag}::"
    return f"({tag}:{weight})"


def _strip_weight(tag: str) -> str:
    """'(mosaic censoring:1.3)' や '1.3::mosaic censoring::' → 'mosaic censoring'"""
    t = tag.strip()
    m = _NAI_WEIGHT_RE.match(t)
    if m:
        return m.group(2).strip().lower()
    if t.startswith("(") and t.endswith(")") and ":" in t:
        t = t[1:-1].rsplit(":", 1)[0]
    return t.strip().strip("{}[]").lower()


def _add_mosaic_tags(tag_text: str, level: str, target: str) -> str:
    tags = [t.strip() for t in tag_text.split(",")]
    tags = [t for t in tags if t and _strip_weight(t) not in CENSOR_TYPE_TAGS | _MOSAIC_OWN_TAGS]
    added = [format_weight(tag, w, target) for tag, w in MOSAIC_LEVELS[level]["tags"]]
    return ", ".join(dedup(tags + added))


def add_censor_prompt(
    positive: str, negative: str, kind: str, level: str = DEFAULT_MOSAIC_LEVEL, target: str = DEFAULT_TARGET
) -> tuple[str, str]:
    """生成画像にモザイクがかかるようプロンプトを書き換える (NSFW 判定のときだけ呼ぶ)。"""
    if level not in MOSAIC_LEVELS:
        level = DEFAULT_MOSAIC_LEVEL
    sentence = MOSAIC_LEVELS[level]["sentence"]
    if kind == "prose":
        positive = positive.rstrip()
        if _MOSAIC_SENTENCE_KEY not in positive.lower():
            positive = (positive + " " if positive else "") + sentence
    elif kind == "anima":
        # タグ部分に重み付きタグ、自然文部分にも文を足す (Qwen3 エンコーダは文の方が効く)
        tag_part, sep, sentences = positive.partition(". ")
        tag_part = _add_mosaic_tags(tag_part, level, target)
        if _MOSAIC_SENTENCE_KEY not in sentences.lower():
            sentences = (sentences.rstrip() + " " if sentences.strip() else "") + sentence
        positive = tag_part + ". " + sentences
    else:  # tags (BREAK 区切りがあれば先頭の共有チャンクに入れる)
        main, *chunks = positive.split("\nBREAK ")
        positive = "\nBREAK ".join([_add_mosaic_tags(main, level, target)] + chunks)
    neg_tags = [t.strip() for t in negative.split(",") if t.strip() and _strip_weight(t) not in _MOSAIC_OWN_TAGS]
    added_neg = [format_weight(tag, w, target) for tag, w in MOSAIC_LEVELS[level]["negative"]]
    negative = ", ".join(dedup(neg_tags + added_neg))
    return positive, negative


# --------------------------------------------------------------------------
# 追加プロンプト: カメラ / ライティング / 画風 / 動画用カメラワーク
# --------------------------------------------------------------------------

EXTRA_KINDS = {
    "subject": "被写体 (人物)",
    "pose": "ポーズ",
    "outfit": "服装",
    "objects": "物体・小物",
    "background": "背景 / シーン",
    "camera": "カメラ / 構図",
    "lighting": "ライティング / 色",
    "style": "画風 / 質感",
    "face": "顔の詳細",
    "motion": "動画用カメラワーク (i2v)",
}
EXTRA_SHORT_JA = {
    "subject": "被写体", "pose": "ポーズ", "outfit": "服装", "objects": "物体", "background": "背景",
    "camera": "カメラ", "lighting": "照明", "style": "画風", "face": "顔の詳細", "motion": "動画カメラワーク",
}
# 追加プロンプトタブのコピーボタンの並び (2 段)
EXTRA_BUTTON_ROWS = (("subject", "pose", "outfit", "objects", "background"), ("camera", "lighting", "style", "face", "motion"))


def empty_face_extra() -> dict[str, Any]:
    """「顔の詳細」(人物の特徴を詳しく が VLM で読んだ顔・髪・体格の文)。オフ / 読めなかったときは空。"""
    return {"tags": [], "text": "", "prompt": ""}

BACKGROUND_TAGS = {
    "outdoors", "indoors", "sky", "cloud", "clouds", "tree", "trees", "building", "city", "cityscape", "street",
    "road", "path", "forest", "mountain", "mountains", "ocean", "beach", "sea", "water", "river", "lake", "pond",
    "waterfall", "grass", "field", "meadow", "flower_field", "cherry_blossoms", "autumn_leaves", "snow", "rain",
    "scenery", "landscape", "room", "bedroom", "classroom", "kitchen", "bathroom", "living_room", "hallway",
    "office", "library", "cafe", "restaurant", "shop", "station", "window", "door", "wall", "floor", "ceiling",
    "curtains", "bed", "table", "desk", "chair", "couch", "sofa", "shelf", "bookshelf", "stairs", "fence",
    "railing", "bridge", "castle", "shrine", "torii", "temple", "church", "garden", "park", "pool", "onsen",
    "ruins", "desert", "cave", "space", "planet", "moon", "sun", "star_(sky)", "horizon", "hill", "rock",
    "cliff", "sand", "dirt", "puddle", "reflection", "skyline", "skyscraper", "house", "rooftop", "balcony",
    "alley", "tunnel", "pier", "dock", "festival", "tatami", "shoji", "futon", "wooden_floor", "tiles",
    "brick_wall", "graffiti", "signboard", "vending_machine", "power_lines", "utility_pole", "traffic_light",
    "crosswalk", "sidewalk", "bench", "lamppost", "streetlight", "bush", "hedge", "plant", "potted_plant",
    "palm_tree", "bamboo", "bamboo_forest", "wheat", "rice_paddy", "ice", "iceberg", "volcano", "lava",
    "clock_tower", "ferris_wheel", "amusement_park", "stadium", "gym", "hospital", "school",
    "hot_spring", "bath", "bathtub", "shower_(place)", "elevator", "corridor", "gate", "archway", "pillar",
    "column", "carpet", "rug", "wallpaper_(object)", "fireplace", "stage", "concert", "crowd",
    "school_desk", "school_chair", "office_chair", "swivel_chair", "armchair", "stool", "throne", "kotatsu",
    "bed_sheet", "locker", "chalkboard", "sliding_doors", "poolside",
}
OBJECT_TAGS = {
    "sword", "katana", "weapon", "gun", "rifle", "pistol", "handgun", "knife", "dagger", "staff", "wand",
    "bow_(weapon)", "arrow_(projectile)", "shield", "spear", "polearm", "axe", "scythe", "umbrella", "parasol",
    "cup", "teacup", "mug", "glass", "wine_glass", "bottle", "can", "phone", "cellphone", "smartphone", "book",
    "notebook", "pen", "pencil", "bag", "handbag", "backpack", "shoulder_bag", "suitcase", "box", "gift",
    "flower", "bouquet", "rose", "food", "cake", "bread", "onigiri", "fruit", "apple", "strawberry", "candy",
    "lollipop", "ice_cream", "drink", "coffee", "tea", "camera", "guitar", "piano", "violin", "instrument",
    "microphone", "ball", "basketball", "soccer_ball", "baseball_bat", "racket", "car", "motor_vehicle",
    "motorcycle", "bicycle", "vehicle", "train", "airplane", "aircraft", "boat", "ship", "robot", "mecha",
    "cat", "dog", "bird", "fish", "butterfly", "animal", "horse", "rabbit", "fox", "wolf", "dragon", "snake",
    "stuffed_toy", "stuffed_animal", "teddy_bear", "doll", "balloon", "lantern", "paper_lantern", "candle",
    "lamp", "clock", "mirror", "television", "computer", "laptop", "keyboard_(computer)", "controller",
    "game_console", "handheld_game_console", "cigarette", "smoking_pipe", "hand_fan", "paper_fan", "folding_fan",
    "sign", "banner", "flag", "pillow", "blanket", "towel", "broom", "bucket", "basket", "watering_can",
    "chopsticks", "fork", "spoon", "plate", "bowl", "tray", "bento", "lunchbox", "thermos", "letter",
    "envelope", "paper", "scroll", "map", "compass", "key", "coin", "money", "card", "playing_card",
    "chess_piece", "dice", "toy", "figure", "trophy", "medal", "crystal", "gem", "orb", "skull", "bone",
    "fireworks", "sparkler", "kite", "frisbee", "skateboard", "surfboard", "snowboard", "ski", "helmet_(object)",
    "swim_ring", "pom_pom_(cheerleading)",
}


def classify_subject_tag(name: str, has_humans: bool) -> str | None:
    """本文のタグを被写体 (person) / 物体 (object) / 背景 (background) に振り分ける。
    カメラ・照明・画風は追加プロンプト側なので None を返す。"""
    if classify_extra_tag(name):
        return None
    n = name.lower()
    if n in PERSON_COUNT_TAGS or n in ("solo", "no_humans"):
        return "person" if has_humans else None
    if n in BACKGROUND_TAGS or n.endswith(("_background", "_sky", "_interior", "scape", "_field", "_room", "_floor", "_wall")):
        return "background"
    if n in OBJECT_TAGS or n.endswith(("_(object)", "_(animal)", "_(vehicle)", "_(food)", "_(weapon)", "_(instrument)")):
        return "object"
    return "person" if has_humans else "object"

CAMERA_TAGS = {
    "from_above", "from_below", "from_side", "from_behind", "from_outside", "close-up", "upper_body",
    "lower_body", "full_body", "cowboy_shot", "portrait", "profile", "dutch_angle", "depth_of_field",
    "blurry_background", "blurry_foreground", "bokeh", "pov", "fisheye", "wide_shot", "very_wide_shot",
    "dynamic_angle", "foreshortening", "panorama", "perspective", "vanishing_point", "atmospheric_perspective",
    "feet_out_of_frame", "head_out_of_frame", "out_of_frame", "straight-on", "three_quarter_view",
    "bird's-eye_view", "worm's-eye_view", "selfie", "viewfinder", "letterboxed", "pillarboxed", "rotated",
    "sideways", "upside-down", "multiple_views", "cropped_torso", "cropped_legs", "cropped_arms",
    "facing_viewer", "facing_away", "motion_blur", "motion_lines", "speed_lines", "tilt-shift", "symmetry",
    "negative_space", "zoom_layer", "close_up", "medium_shot", "long_shot", "high_angle", "low_angle",
}
LIGHTING_TAGS = {
    "backlighting", "sidelighting", "rim_light", "sunlight", "dappled_sunlight", "sunbeam", "light_rays",
    "god_rays", "lens_flare", "glowing", "bloom", "shadow", "spotlight", "moonlight", "candlelight",
    "neon_lights", "light_particles", "sparkle", "night", "evening", "sunset", "sunrise", "dusk", "dawn",
    "twilight", "day", "morning", "noon", "overcast", "cloudy", "fog", "mist", "haze", "dark", "bright",
    "high_contrast", "low_contrast", "chiaroscuro", "cinematic_lighting", "dramatic_lighting",
    "soft_lighting", "dim_lighting", "volumetric_lighting", "warm_colors", "cool_colors", "monochrome",
    "greyscale", "sepia", "limited_palette", "muted_color", "pastel_colors", "vibrant_colors", "light",
    "shade", "lightning", "silhouette", "backlit", "golden_hour", "blue_hour", "color_grading",
}
STYLE_TAGS = {
    "realistic", "photorealistic", "3d", "anime_coloring", "anime_screencap", "sketch", "lineart",
    "traditional_media", "flat_color", "cel_shading", "pixel_art", "chibi", "retro_artstyle", "film_grain",
    "chromatic_aberration", "vignetting", "halftone", "screentone", "comic", "4koma", "official_art",
    "game_cg", "concept_art", "faux_traditional_media", "impressionism", "ukiyo-e", "art_nouveau",
    "art_deco", "minimalism", "surreal", "abstract", "no_lineart", "unfinished", "dithering",
    "ligne_claire", "nihonga", "sumi-e", "painterly", "thick_outlines", "oekaki", "vector_art",
    "photo_background", "scan", "scanlines",
}


def classify_extra_tag(name: str) -> str | None:
    """Danbooru タグ名をカメラ / ライティング / 画風のどれかに分類する。該当しなければ None (本文に残す)。"""
    n = name.lower()
    if n in CAMERA_TAGS:
        return "camera"
    if n in LIGHTING_TAGS:
        return "lighting"
    if n in STYLE_TAGS:
        return "style"
    if n in BACKGROUND_TAGS or n in OBJECT_TAGS:
        return None  # traffic_light / desk_lamp などを語尾で照明と誤判定しない
    if n.startswith("looking_") or n.endswith(("_focus", "_shot", "_view")):
        return "camera"
    if n.endswith(("lighting", "_light", "_lights", "_theme")) or n.startswith("light_rays"):
        return "lighting"
    if n.endswith(("_(medium)", "_(style)", "_artstyle")):
        return "style"
    return None


POSE_TAGS = {
    "standing", "sitting", "kneeling", "lying", "squatting", "crouching", "on_back", "on_stomach", "on_side",
    "all_fours", "seiza", "wariza", "indian_style", "butterfly_sitting", "straddling", "walking", "running",
    "jumping", "dancing", "fighting_stance", "bent_over", "arched_back", "head_tilt", "contrapposto",
    "stretching", "reaching", "pointing", "salute", "v", "double_v", "thumbs_up", "waving", "clenched_hand",
    "clenched_hands", "own_hands_together", "hands_together", "own_hands_clasped", "hugging_own_legs",
    "knees_up", "knees_to_chest", "knees_together_feet_apart", "legs_apart", "legs_together", "crossed_legs",
    "crossed_arms", "crossed_ankles", "spread_legs", "spread_arms", "outstretched_arm", "outstretched_arms",
    "outstretched_hand", "outstretched_leg", "arm_support", "fetal_position", "floating", "flying", "midair",
    "falling", "reclining", "twisted_torso", "leg_up", "legs_up", "leg_lift", "one_knee", "on_one_knee",
    "hand_up", "hands_up", "arm_up", "arms_up", "hand_to_own_mouth", "covering_own_mouth", "covering_face",
    "peace_sign", "heart_hands", "finger_to_mouth", "index_finger_raised", "shushing", "pose", "dynamic_pose",
    "action_pose", "posing", "leaning", "sleeping", "stepping", "kicking", "punching", "hugging", "carrying",
    "turning_head", "turning_around", "looking_back", "holding", "adjusting_hair", "adjusting_clothes",
    "hair_tucking", "tiptoes", "standing_on_one_leg", "hand_on_hip", "hands_on_hips", "paw_pose", "claw_pose",
    "fox_shadow_puppet", "w", "ok_sign", "middle_finger", "open_hand", "beckoning", "stroking_own_chin",
    # タガーの語彙で「人物」に落ちて ① から残っていたもの (2026-09-30、複数枚モードでポーズが反映されない報告)
    "hug", "hug_from_behind", "head_rest", "v_arms", "interlocked_fingers", "between_legs", "grabbing", "hugging_object",
    "pillow_hug", "arm_hug", "on_bed", "on_chair", "on_couch", "on_ground", "on_desk", "on_floor", "girl_on_top",
    "boy_on_top", "top-down_bottom-up", "yokozuwari", "m_legs", "eating", "drinking", "reading", "mouth_hold",
    "lap_pillow", "princess_carry", "lifting_person", "back-to-back", "face-to-face", "heads_together", "side-by-side",
    "eye_contact", "facing_another", "headpat", "feeding", "aiming", "riding", "presenting", "elbow_rest", "arm_rest",
    "locked_arms", "steepled_fingers", "pinky_out", "trigger_discipline", "toe_scrunch", "pigeon-toed", "peeking_out",
    "open_hands", "playing_instrument", "taking_picture", "cooking", "bathing", "swimming", "wading", "soaking_feet",
    "waking_up", "yawning", "sheet_grab", "tying_hair", "dressing", "undressing", "singing", "talking", "shrugging",
    "broom_riding", "piggyback", "sleeve_grab", "hand_on_another's_head", "hand_on_another's_shoulder", "unsheathing", "firing",
    "w_arms", "upright_straddle", "finger_gun", "throwing", "hiding", "hair_twirling", "writing", "fighting",
    "cheek-to-cheek", "showering", "exercise", "on_grass", "arm_grab", "leg_grab", "thigh_grab", "torso_grab",
}
_POSE_LIMB_RE = re.compile(
    r"^(?:own_)?(?:arms?|hands?|legs?|knees?|feet|foot|fingers?|elbows?|head)_"
    r"(?:up|down|out|forward|together|apart|crossed|raised|extended|lifted|support|"
    r"(?:behind|on|in|at|over|under|between|around|to|on_own|in_own|near|above|across)_.+)$"
)
_POSE_PREFIX = ("sitting_", "standing_", "kneeling_", "lying_", "squatting_", "leaning_", "holding_", "reaching_",
                "pointing_", "hand_on_", "hands_on_", "hand_in_", "hands_in_", "arm_around_", "arms_around_",
                "grabbing_", "hugging_", "covering_", "carrying_", "lifting_", "sitting_on_", "holding_hands")


def classify_pose_tag(name: str) -> bool:
    """Danbooru タグ名がポーズ (姿勢・手足の位置・持ち方・視線の向き) に当たるか。服や装飾品は含めない。"""
    n = name.lower()
    if n in POSE_TAGS or n.endswith(("_pose", "_stance", "_position", "_grab")):
        return True
    if n.startswith(_POSE_PREFIX):
        return True
    return bool(_POSE_LIMB_RE.match(n))


def pose_source_tag(name: str) -> bool:
    """複数枚モードで ② ポーズ画像から取るタグ: ポーズとカメラ / 構図 (視線・向き・画角を含む)。"""
    return classify_pose_tag(name) or classify_extra_tag(name) == "camera"


OUTFIT_TAGS = {
    "nude", "completely_nude", "topless", "bottomless", "naked_shirt", "barefoot", "bare_shoulders", "bare_arms",
    "bare_legs", "sleeveless", "strapless", "off_shoulder", "off-shoulder", "see-through", "frills", "lace",
    "buttons", "zipper", "crop_top", "tank_top", "tube_top", "halterneck", "midriff", "serafuku", "jewelry",
    "hood_up", "hood_down", "maid", "nurse", "miko", "armor", "gothic_lolita", "lolita_fashion", "casual",
    "formal", "cosplay", "costume",
    # タガーの語彙で「人物」に落ちて ① から残っていたもの (2026-09-30)。付け耳・付け尻尾・ネイル・髪に挿した花も
    # 身に着ける物として服装側 (おすすめどおり。翼・光輪・タトゥー・化粧はキャラの特徴として ① のまま)
    "mary_janes", "hair_flower", "hair_stick", "hair_beads", "bun_cover", "ear_covers", "head_wreath", "nail_polish",
    "toenail_polish", "fake_animal_ears", "fake_tail", "fake_horns", "dog_tags", "name_tag", "id_card", "lanyard",
    "forehead_protector", "double-breasted", "v-neck", "center_frills", "center_opening", "see-through_cleavage",
    "unbuttoned", "partially_unbuttoned", "unzipped", "partially_unzipped", "open_fly", "lowleg", "crotchless",
    "drawstring", "pocket", "breast_pocket", "polka_dot", "camouflage", "latex", "leather", "silk", "skin_tight",
    "pasties", "heart_pasties", "maebari", "harness", "chest_harness", "holster", "thigh_holster", "magical_girl",
    "cheerleader", "nun", "waitress", "office_lady", "police", "policewoman", "race_queen", "bride", "enmaided",
    "nontraditional_miko", "unconventional_maid", "naked_towel", "towel_around_neck", "monocle", "pince-nez",
    "knee_pads", "elbow_pads", "shoulder_pads", "arm_guards", "headset", "earphones", "earpiece", "earclip",
    "goth_fashion", "partially_undressed", "reverse_bunnysuit", "feather_boa", "bandaids_on_nipples", "breast_curtains",
    "rei_no_himo", "multiple_belts", "flip-flops",
}
_OUTFIT_WORDS = {
    "dress", "shirt", "skirt", "jacket", "coat", "hat", "cap", "beret", "gloves", "glove", "boots", "shoes",
    "sneakers", "sandals", "heels", "footwear", "legwear", "thighhighs", "kneehighs", "pantyhose", "socks",
    "leggings", "uniform", "swimsuit", "bikini", "kimono", "yukata", "hakama", "ribbon", "bow", "bowtie",
    "hairband", "headband", "hairclip", "ornament", "earrings", "earring", "necklace", "choker", "glasses",
    "eyewear", "sunglasses", "goggles", "scarf", "apron", "cape", "cloak", "hood", "hoodie", "belt", "sleeves",
    "collar", "neckerchief", "necktie", "tie", "vest", "sweater", "cardigan", "blazer", "pants", "shorts",
    "jeans", "leotard", "bodysuit", "armor", "headwear", "headdress", "headpiece", "bracelet", "wristband",
    "mask", "veil", "gown", "camisole", "bra", "panties", "underwear", "lingerie", "crown", "tiara",
    "headphones", "earmuffs", "bag", "backpack", "pendant", "brooch", "hairpin", "kanzashi", "blouse", "suit",
    "tuxedo", "overalls", "robe", "bathrobe", "pajamas", "armband", "garter", "garters", "corset", "obi", "sash",
    "tabi", "zouri", "geta", "loafers", "slippers", "mittens", "capelet", "shawl", "stole", "poncho", "parka",
    "tabard", "bloomers", "buruma", "sarashi", "fundoshi", "wristwatch", "watch", "anklet", "piercing",
    "scrunchie", "bobbles", "hairpins", "helmet", "visor", "epaulettes", "ascot", "jabot", "cravat",
    "clothes", "clothing", "outfit", "attire",
    # 1 語で書かれる服・柄・飾り (タガーのタグ一覧で ① 側に漏れていたもの)
    "miniskirt", "thighhigh", "t-shirt", "turtleneck", "fishnets", "gauntlets", "suspenders", "denim", "plaid",
    "highleg", "trim", "cuffs", "strap", "print", "buckle", "o-ring", "tassel", "pom", "pelvic", "costume",
    "bunny", "bell", "ring", "eyepatch", "headgear", "sweatband", "legband", "leg_garter", "cuff",
    # 2026-09-30 追加 (tank_top / white_tank_top、各種 cutout、over-kneehighs などが ① に残り ③ から消えていた)
    "top", "tops", "swimwear", "sportswear", "sleepwear", "cutout", "slit", "halter", "seam", "vent", "button",
    "zipper", "frills", "knit", "fabric", "sheer", "sock", "shoe", "sleeve", "stocking", "stockings", "warmers",
    "serafuku", "gakuran", "sundress", "microskirt", "petticoat", "bodystocking", "habit", "haori", "thighlet",
    "vambraces", "bracer", "greaves", "pauldrons", "breastplate", "faulds", "muneate", "sarong", "bandeau", "beanie",
    "fedora", "bonnet", "circlet", "armlet", "bangle", "bandana", "blindfold", "hachimaki", "waistcoat", "lapels",
    "aiguillette", "obijime", "tasuki", "kariginu", "dougi", "loincloth", "babydoll", "nightgown", "jumpsuit",
    "plugsuit", "undershirt", "bustier", "thong", "randoseru", "uwabaki", "kneehigh", "boot", "hakama", "tabard",
    "furisode", "trunks", "kote", "sode", "coattails", "obiage", "tunic", "spacesuit", "pumps", "nightcap",
    "okobo", "cutoffs", "hagoromo", "bunnysuit",
}
# 服装で見え方が決まる体のタグ。複数枚モードでは ③ 服装の画像から取る。
# 胸の大きさ・筋肉など体そのもののタグは ① のまま
EXPOSURE_TAGS = {
    "navel", "cleavage", "armpits", "stomach", "collarbone", "midriff_peek", "zettai_ryouiki", "covered_navel",
    "covered_nipples", "sideboob", "underboob", "cleavage_cutout", "pantyshot", "skindentation", "thighs",
    "groin", "cameltoe", "breasts_out", "nipples", "areolae", "areola_slip", "pussy", "penis", "testicles",
    "pubic_hair", "female_pubic_hair", "male_pubic_hair", "ass_visible_through_thighs", "thigh_gap", "bare_back",
    "back", "ass", "lower_body_exposed", "downblouse", "nipple_slip", "wardrobe_malfunction", "panties_under_pantyhose",
}
# 体格のタグ。複数枚モードで ⑥ があれば ⑥ から取る (胸・体型・身長・手足・腰回りに加え、
# 肌の色・年齢の見た目・男性的な体のタグも)。服で見え方が決まるもの (navel / cleavage / thighs など) は EXPOSURE_TAGS で ③
BODY_TAGS = {
    # 胸
    "breasts", "flat_chest", "small_breasts", "medium_breasts", "large_breasts", "huge_breasts", "gigantic_breasts",
    "alternate_breast_size", "sagging_breasts", "perky_breasts",
    # 体型・身長・手足・腰回り
    "petite", "tall", "tall_female", "tall_male", "short", "skinny", "slim", "slender", "plump", "chubby", "fat", "curvy",
    "voluptuous", "muscular", "muscular_female", "toned", "athletic", "long_legs", "wide_hips", "narrow_waist",
    "thick_thighs", "huge_ass", "flat_ass", "big_belly", "broad_shoulders", "narrow_shoulders", "thick_arms",
    "mini_person", "giant", "giantess", "loli_body",
    # 肌の色
    "dark_skin", "very_dark_skin", "dark-skinned_female", "dark-skinned_male", "pale_skin", "tan", "tanned", "deep_skin",
    "tanlines",
    # 年齢の見た目
    "child", "female_child", "male_child", "toddler", "loli", "shota", "oppai_loli", "mature_female", "mature_male",
    "old", "old_woman", "old_man", "milf", "aged_up", "aged_down",
    # 男性的な体
    "muscular_male", "toned_male", "pectorals", "large_pectorals", "bara", "abs", "biceps", "chest_hair", "arm_hair",
    "leg_hair", "manly",
}
# VLM が体格を書くときの語 (slim figure / athletic build など、BODY_TAGS に無い言い方)
_BODY_WORDS = {
    "build", "figure", "physique", "body", "slim", "slender", "petite", "tall", "curvy", "muscular", "plump", "chubby",
    "athletic", "toned", "skinny", "breasts", "chest", "hips", "waist", "skin", "height", "stature", "proportions",
}


def body_tag(name: str) -> bool:
    """⑥ 体格の画像から取るタグか。"""
    return name.lower() in BODY_TAGS


# 最後の語がこれなら髪型・体の部位 (crown_braid など)。服の語を含んでいても服装にしない
_NOT_OUTFIT_LAST = {
    "braid", "braids", "hair", "bun", "buns", "ponytail", "twintails", "bangs", "eyes", "ears", "tail", "horns",
    "background", "theme", "sky", "floor", "wall", "pupils",
}
_NAIL_COLORS = {
    "black", "red", "pink", "blue", "purple", "green", "yellow", "orange", "white", "aqua", "grey", "gold", "silver",
    "brown", "multicolored",
}


def classify_outfit_tag(name: str) -> bool:
    """服・靴・装飾品 (身に着けている物) のタグか。髪型・髪色・体は含めない (hair_ribbon などの髪飾りは含む)。"""
    n = name.lower()
    if n in OUTFIT_TAGS:
        return True
    if n in OBJECT_TAGS:
        return False  # bow_(weapon) などは物体
    if n.endswith("_(clothing)"):
        return True  # shrug_(clothing) など
    # over-kneehighs / cross-laced_footwear のようにハイフンでつながった語も分けて見る
    words = re.split(r"[_-]", n)
    if words[-1] in _NOT_OUTFIT_LAST:
        return False
    if words[-1] == "nails" and len(words) >= 2 and words[0] in _NAIL_COLORS:
        return True  # black_nails など色の付いた爪 = マニキュア (fingernails / long_fingernails は体のまま)
    # sleeves_past_wrists / skirt_lift / open_clothes のように服の語が途中にあるものも服装
    return any(w in _OUTFIT_WORDS for w in words)


# holding_〇〇 のうち、物ではなく手の位置を表すもの (② ポーズのまま)
_HOLDING_BODY = {
    "hands", "hair", "breasts", "legs", "leg", "head", "breath", "with_feet", "behind_back", "heart",
}


def held_object_tag(name: str) -> bool:
    """holding_umbrella のように物を持っているタグか (holding_hands / holding_another's_wrist などの手の位置は除く)。"""
    n = name.lower()
    if not n.startswith("holding_"):
        return False
    rest = n[len("holding_"):]
    return rest not in _HOLDING_BODY and not rest.startswith(("another's_", "own_", "hands"))


ERA_TAG_RE = re.compile(r"^\d{4}s_\(style\)$")  # 1990s_(style) など。年代は ⑦ 作風・質感では取らない (① のまま)


def reference_category(name: str, style_ref: bool = False) -> str:
    """複数枚モードで、タグをどの枠 (pose / outfit / props / held / background / body / style / expression / person) の画像から取るかを決める。
    held = 物を持っているタグ (小物の枠 → 服装 → ポーズ → 人物の順に受け持つ)。
    style_ref = ⑦ 作風・質感がある: レンズの写り・光の当て方・色調のタグ (撮影プリセットで置き換えるもの、look_replaces) も ⑦ から取る。
    時間帯・天気・光源・画角・アングルは今までどおりの枠。"""
    n = name.lower()
    if n in MONOCHROME_TAGS:
        return "person"  # 白黒の画風は ① から
    if classify_extra_tag(name) == "style" and not ERA_TAG_RE.match(n):
        return "style"  # 画風・技法 (⑦ があれば ⑦、無ければ ①。年代は ① のまま)
    if style_ref and look_replaces(name):
        return "style"
    if held_object_tag(n):
        return "held"
    if gaze_tag(n):
        return "gaze"  # 視線 (⑧ → ② → ①)
    if expression_tag(n):
        return "expression"  # 表情 (⑧ があれば ⑧、無ければ ①)
    if n in EXPOSURE_TAGS:
        return "outfit"  # 服装で見え方が決まる体のタグ
    if body_tag(n):
        return "body"  # 胸・体型・肌の色・年齢の見た目・筋肉 (⑥ があれば ⑥、無ければ ①)
    if pose_source_tag(name):
        return "pose"
    if classify_outfit_tag(name):
        return "outfit"  # 身に着ける物 (服・靴・帽子・装飾品・髪飾り)
    if name.lower() in OBJECT_TAGS or classify_subject_tag(name, True) == "object":
        return "props"  # 手に持つ物・置いてある物 (バッグ・傘・武器・本・飲み物・動物など)
    if classify_extra_tag(name) == "lighting" or classify_subject_tag(name, True) == "background":
        return "background"
    return "person"  # 顔・髪・目・体型・表情・キャラ名・人数


# 複数人を前提にしたタグ (メインの 1 人だけにするときは外す)
MULTI_PERSON_TAGS = {"group", "crowd", "couple", "hetero", "yuri", "yaoi", "multiple_views", "size_difference"}


def force_single_person(tags: TagResult) -> TagResult:
    """人数のタグを 1 人 (1girl / 1boy / 1other と solo) にそろえ、複数人前提のタグを外す。
    どの性別かは、信頼度がいちばん高い人数タグから決める。人数タグが無い・no_humans のときはそのまま。"""
    names = [n for n, _ in tags.general]
    counts = [n for n in names if n in PERSON_COUNT_TAGS]
    if "no_humans" in names or not counts:
        return tags
    group = next(g for g in ("girl", "boy", "other") if g in counts[0])
    general = [(f"1{group}", 1.0), ("solo", 1.0)] + [
        (n, p) for n, p in tags.general if n not in PERSON_COUNT_TAGS and n != "solo" and n not in MULTI_PERSON_TAGS
    ]
    return TagResult(tags.rating, tags.rating_conf, general, list(tags.characters), dict(tags.general_probs))


def slot_owner(category: str, slots: Any) -> str:
    """その種類のタグを受け持つ枠。指定された枠が無ければ 1 枚目 (person)。
    小物は ⑤ があれば ⑤、無ければ ③ 服装が受け持つ (⑤ があるときだけ服装と小物を分ける)。"""
    if category == "held":  # 物を持っているタグ: 小物の枠 → 服装 → ポーズ
        return next((slot for slot in ("props", "outfit", "pose") if slot in slots), "person")
    if category == "gaze":  # 視線: ⑧ 表情 → ② ポーズ → ① (⑧ は 2026-10-10 追加。⑧ が無ければ以前どおり)
        return next((slot for slot in ("expression", "pose") if slot in slots), "person")
    if category in slots:
        return category
    if category == "props" and "outfit" in slots:
        return "outfit"
    return "person"


# 本文に並べる順: 服装 → 小物 → ポーズ → 体格 → 人物 → 背景。
# 人物を先にすると ① のタグだけで CLIP の最初の 75 トークンが埋まり、③ の服・② のポーズが後ろの区画に押し出されて効かなかった。
# 公式の並び (品質 → 人数 → キャラ → 一般タグ) は一般タグの中の順を決めていないので、人数・キャラは compose_positive で先頭のまま
# 画風 (⑦) は以前は人物の中に並んでいたので、人物のすぐ後ろに置く
# 表情 (⑧) は人物のすぐ後ろ (⑧ が無いときは以前どおり人物のタグの中に並べる)
_MERGE_ORDER = ("outfit", "props", "pose", "body", "person", "expression", "style", "background")


def merge_reference_tags(refs: dict[str, TagResult]) -> tuple[TagResult, dict[str, str]]:
    """複数枚モード: 各タグを持ち主の枠の画像から取って 1 つにする。枠が空ならその要素は 1 枚目 (person) から取る。

    refs には必ず "person" を含める。戻り値の辞書は タグ名 → 取った枠。
    """
    person = refs["person"]

    def owner(name: str) -> str:
        return slot_owner(reference_category(name, "style" in refs), refs)

    general: list[tuple[str, float]] = []
    source: dict[str, str] = {}
    for slot in _MERGE_ORDER:
        tags = refs.get(slot)
        if tags is None:
            continue
        for name, prob in tags.general:
            if name not in source and owner(name) == slot:
                general.append((name, prob))
                source[name] = slot
    # レーティング (露出の度合い) は服装で決まるので ③ があれば ③ から。無ければ ①
    rated = refs.get("outfit") or person
    merged = TagResult(
        rating=rated.rating,
        rating_conf=rated.rating_conf,
        general=general,
        characters=list(person.characters),  # キャラ名は常に 1 枚目
        # 画風の判定 (detect_medium) と低い信頼度の画風タグは確率を見るので、⑦ があれば ⑦ の画風の確率を使う
        general_probs={**{n: p for n, p in person.general_probs.items() if owner(n) == "person"},
                       **{n: p for n, p in (refs["style"].general_probs.items() if "style" in refs else ()) if owner(n) == "style"}},
    )
    return merged, source


def order_reference_tags(tags: TagResult, tag_source: dict[str, str], vlm_tags: dict[str, list[str]]) -> list[str]:
    """複数枚モードの本文の一般タグ: 服装 → 小物 → ポーズ → 体格 → 人物 → 背景。VLM が読んだ服装 (③) / ポーズ (②) のタグは
    それぞれタガーの服装 / ポーズのタグの後ろに入れる (同じタグ・表記ゆれは足さない)。"""
    groups: dict[str, list[str]] = {slot: [] for slot in _MERGE_ORDER}
    head: list[str] = []
    for name, _ in tags.general:
        if name in ("solo", "no_humans"):
            head.append(sd_tag(name))  # 人数タグ (1girl など、compose_positive が先頭に置く) のすぐ後ろのまま
            continue
        # 並びはタグの種類で決める (どの画像から来たかではない)。② が空で ① から取ったポーズも、ポーズの位置に並ぶ
        category = reference_category(name, "style" in tag_source.values())
        slot = {"held": "props", "gaze": "expression" if "expression" in tag_source.values() else "pose"}.get(category, category)
        if slot in ("style", "expression") and slot not in tag_source.values():
            slot = "person"  # ⑦ / ⑧ が無いときの画風・表情は、以前どおり人物のタグの中 (信頼度の順) に並べる
        groups.get(slot, groups["person"]).append(sd_tag(name))
    have = {loose_tag_key(t) for g in groups.values() for t in g}
    for slot in ("outfit", "pose", "body", "person", "expression"):
        for t in vlm_tags.get(slot, []):
            if loose_tag_key(t) not in have:
                have.add(loose_tag_key(t))
                groups[slot].append(t)
    order = _MERGE_ORDER
    if "style" in tag_source.values():
        # ⑦ 作風・質感を入れたときは、画風のタグを人数・キャラ名のすぐ後ろに置く (NoobAI などの推奨の並びでも作者・画風は
        # 一般タグの前)。後ろのままだと、長いプロンプトで CLIP の後ろの区画に押し出されて、① の絵柄に負ける
        order = ("style",) + tuple(s for s in _MERGE_ORDER if s != "style")
    return head + [t for slot in order for t in groups[slot]]


# VLM が返しがちな、特徴を表さない区分だけの語 (実機の qwen2.5vl がネガティブに face / hair / body / other outfits を入れた)
_GENERIC_WORDS = {
    "face", "faces", "hair", "body", "bodies", "outfit", "outfits", "pose", "poses", "object", "objects", "setting",
    "settings", "clothes", "clothing", "appearance", "background", "backgrounds", "medium", "other", "wrong", "different",
    "incorrect", "style", "lighting", "light", "colors", "colours", "composition", "angle",
}
# 表情 (① の人物から取る。VLM が ② のポーズのタグに入れてくる)
_EXPRESSION_TAGS = {
    "smile", "smiling", "slight_smile", "grin", "blush", "open_mouth", "closed_mouth", "parted_lips", "expressionless",
    "frown", "pout", "serious", "happy", "sad", "angry", "crying", "tears", "surprised", "serene_expression",
}
# ⑧ 表情の枠が受け持つタグ (表情 + 視線)。表情は Danbooru の Tag Group:Face Tags の
# Emotions / Sexual / Smile / Smug / Surprised・Scared・Sad / Emotes と、目・口の表情 (closed eyes / wink / tongue out など)。
# 視線は looking_〇〇 (looking back は振り返る体の向きなので ② ポーズのまま)
EXPRESSION_SLOT_TAGS = _EXPRESSION_TAGS | {
    "angry", "anger_vein", "annoyed", "clenched_teeth", "glaring", "scowl", "blush_stickers", "embarrassed",
    "full-face_blush", "nose_blush", "bored", "closed_eyes", "confident", "confused", "crazy", "despair", "determined",
    "disappointed", "disdain", "disgust", "distress", "drunk", "ecstasy", "envy", "excited", "exhausted", "flustered",
    "frustrated", "furrowed_brow", "grimace", "guilt", "kubrick_stare", "lonely", "nervous", "nosebleed", "one_eye_closed",
    ":o", "pain", "puffy_cheeks", "raised_eyebrow", "raised_eyebrows", "raised_inner_eyebrows", "rolling_eyes",
    "depressed", "gloom_(expression)", "scared", "panicking", "worried", "shaded_face", "shy", "sigh", "skeptical",
    "sleepy", "squinting", "sulking", "thinking", "pensive", "unamused", "v-shaped_eyebrows", "wince", "struggling",
    "afterglow", "ahegao", "aroused", "naughty_face", "seductive_smile", "torogao", ":d", "crazy_smile", "evil_smile",
    "forced_smile", "evil_grin", "light_smile", "sad_smile", "stifled_laugh", "laughing", "doyagao", "smirk", "smug",
    "horrified", "screaming", "sobbing", "traumatized", "turn_pale", "wavy_mouth", ";)", ";d", "xd", "d:", ":3", ";3",
    "x3", ":p", ";p", ":q", ";q", ">:)", ">:(", ":t", ":/", ":|", "x_mouth", ":<", ":>", "o_o", "^_^", ">_<", "@_@",
    "=_=", "t_t", "half-closed_eyes", "wink", "streaming_tears", "crying_with_eyes_open", "tearing_up", "tongue",
    "tongue_out", "fang_out", "staring", "eye_contact", "light_blush", "heavy_breathing", "drooling", "saliva",
}


# VLM が表情を書くときの語 (gentle smile / soft gaze など、EXPRESSION_SLOT_TAGS に無い言い方)。eyes は髪・目の色と紛れるので入れない
_EXPRESSION_WORDS = {
    "smile", "smiling", "smiles", "blush", "blushing", "tears", "crying", "mouth", "gaze", "looking", "expression",
    "frown", "frowning", "pout", "pouting", "grin", "grinning", "laugh", "laughing", "teeth", "tongue", "eyebrows",
    "brow", "wink", "winking", "smirk", "smirking",
}


def gaze_tag(name: str) -> bool:
    """視線のタグか。⑧ があれば ⑧、無ければ以前どおり ② ポーズ → ① から取る (⑧ を使わないときの結果を変えない)。"""
    n = name.lower()
    return (n.startswith("looking_") and n != "looking_back") or n == "eye_contact"


def expression_tag(name: str) -> bool:
    """⑧ 表情の画像から取るタグか (表情と視線)。"""
    n = name.lower()
    return n in EXPRESSION_SLOT_TAGS or (n.startswith("looking_") and n != "looking_back")


# キャラの外見 (① の担当)。_FEATURE_RE (髪・目) と FACE_TAGS に無いもの。VLM が ② / ③ / ⑥ のタグに入れてくる (網羅テストで検出)
_IDENTITY_TAGS = {
    "heterochromia", "pointy_ears", "animal_ears", "cat_ears", "fox_ears", "dog_ears", "rabbit_ears", "fang", "fangs",
    "mole", "mole_under_eye", "mole_under_mouth", "scar", "tattoo", "makeup", "lipstick", "eyeliner", "eyeshadow", "halo",
    "wings", "horns", "tail", "elf", "blush", "facial_mark",
}
# 場面・雰囲気を表す語。VLM が服装・ポーズのタグに入れてくるので本文の末尾 (VLM の追加タグ) へ回す
_POSE_WORDS = {"posture", "gaze", "sitting", "standing", "kneeling", "lying", "tilted", "crossed", "leaning", "resting"}
_SCENE_WORDS = {"setting", "atmosphere", "light", "lighting", "background", "scene", "outdoor", "indoor", "outdoors", "indoors"}


# 画風ごとの、その画風そのものを表す語。VLM がネガティブに入れると元の画風を打ち消す
_MEDIUM_WORDS = {
    "anime": {"anime", "manga", "cel", "cartoon", "cartoonish", "illustration", "illustrated", "2d", "lineart", "drawing",
              "drawn"},  # flat / shaded は flat chest / shaded face にもなるので語では見ない (flat color は下で)
    "painting": {"painting", "painted", "painterly", "brushstrokes", "watercolor", "illustration", "drawing", "drawn"},
    "photo": {"photo", "photograph", "photographic", "photorealistic", "realistic", "realism", "camera"},
    "3d": {"3d", "render", "rendered", "cgi"},
    "pixel_art": {"pixel", "pixelated", "pixels", "8-bit", "16-bit"},
}
MEDIUM_JA = {"anime": "アニメ・イラスト調", "painting": "絵画調", "photo": "写真・実写調", "3d": "3D", "pixel_art": "ドット絵"}


def contradicts_medium(tag: str, medium: str) -> bool:
    """ネガティブの語が、元の画像の画風そのもの (アニメ絵なら cel shaded / manga style) を打ち消すか。
    ただし lowres / sketch のような品質の語は画風ではないので対象にしない (sketch は完成度の語なのでアニメでも残す)。"""
    words = set(re.findall(r"[a-z0-9-]+", unescape_parens(tag).lower()))
    if medium == "anime" and words <= {"sketch", "unfinished", "lineart"}:
        return False  # 下書き・線画だけの絵を避ける語 (BASE の sketch / lineart / unfinished と同じ扱い)
    if medium == "anime" and re.search(r"\bflat colou?rs?\b", unescape_parens(tag).lower()):
        return True
    return bool(words & _MEDIUM_WORDS.get(medium, set()))


def is_generic_phrase(tag: str) -> bool:
    """face / other outfits / wrong medium のように区分を言うだけか、ほかと比べるだけで、具体的な特徴を表さない語か。"""
    words = re.findall(r"[a-z]+", unescape_parens(tag).lower())
    if words and words[0] in ("other", "different", "wrong", "incorrect"):
        return True  # other beach outfits などは SD には「beach outfit」と読まれ、③ の服まで打ち消す (実機で確認)
    return bool(words) and all(w in _GENERIC_WORDS for w in words)


def sort_ref_vlm_tags(
    vlm_tags: dict[str, list[str]], tagger_tags: dict[str, list[str]] | None = None,
) -> tuple[dict[str, list[str]], list[str], list[str]]:
    """複数枚モードで VLM が返した服装 (③) / ポーズ (②) のタグを、中身で振り分け直す (実機の qwen2.5vl は
    服装のタグに照明・姿勢・持ち物を、ポーズのタグに表情・髪・服を入れ、言い換えを繰り返した)。
    vlm_tags のキーは VLM に頼んだ枠だけ。tagger_tags はその枠の画像のタガーのタグ (組み合わせの重複の判定に使う)。戻り値: (枠ごとのタグ, 本文の末尾へ回すタグ, 外したタグ)。
    - その呼び出しの役割 (② ならポーズ) と判定できるものだけ残す。別の役割のもの (② の画像の服) は、その枠 (③) を
      入れていても移さずに外す (別の画像の中身になるため)
    - 髪・目・顔・表情は ① の人物から取るので外す。場所・照明はその画像の担当ではないので外す。判定できない語は末尾へ
    - 「white dress with headband」のような組み合わせで、語がすべてほかのタグにあるものは重複なので外す"""
    sorted_tags: dict[str, list[str]] = {slot: [] for slot in vlm_tags}
    to_extra: list[str] = []
    dropped: list[str] = []
    for slot, items in vlm_tags.items():
        for tag in items:
            n = norm_tag(tag).replace(" ", "_")
            words = set(re.split(r"[_-]", n))
            singular = "_".join(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in n.split("_"))
            if gaze_tag(n) or (words & {"gaze", "looking"} and n != "looking_back"):
                # 視線: ⑧ を頼んだときは ⑧、無ければ以前どおり ② のポーズ
                dest: str | None = "expression" if "expression" in vlm_tags else "pose"
            elif expression_tag(n) or expression_tag(singular) or (
                    words & _EXPRESSION_WORDS and not n.startswith(("covering_", "hand_", "hands_", "finger_"))):
                dest = "expression"  # 表情は ⑧ (⑧ を頼んだ呼び出しのときだけ残る。② / ③ / ⑥ からは外す)
            elif pose_source_tag(n) or words & _POSE_WORDS:
                dest = "pose"
            elif classify_outfit_tag(n) or classify_outfit_tag(singular):  # VLM は複数形でも書く (miniskirts)
                dest = "outfit"
            elif body_tag(n) or body_tag(singular) or words & _BODY_WORDS:  # 肌の色 (dark skin) は髪・目より先に見る
                dest = "body"
            elif _FEATURE_RE.search(n) or n in FACE_TAGS or n in _EXPRESSION_TAGS or n in _IDENTITY_TAGS:
                dest = None
            elif words & _SCENE_WORDS or classify_extra_tag(n) == "lighting" or classify_subject_tag(n, True) == "background":
                # ② / ③ / ⑥ の画像の場所・照明 (desert, rim light) は背景の担当 (④ / ①) ではないので入れない (網羅テストで検出)
                dest = None
            elif slot == "body":
                dest = None  # ⑥ は体格だけ。体格と判定できない語 (目・顔の特徴など) は入れない
            else:
                # 判定できない語 (loose fit, relaxed, summer) は枠の並びに入れず、
                # これまでどおり本文の末尾 (VLM の追加タグ) へ。服装・ポーズの並びは判定できたタグだけにする (実機で relaxation などが入った)
                to_extra.append(tag)
                continue
            if dest != slot:
                # 別の役割のタグは、その役割の枠を入れていても移さない (③ の画像の one knee を ② のポーズにしない。網羅テストで検出)
                dropped.append(tag)
            else:
                sorted_tags[dest].append(tag)

    def word_set(tag: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", tag.lower()))

    for slot, items in sorted_tags.items():
        items = dedup(items)
        # 組み合わせ (white dress with white headband) は、語がすべてほかの単独のタグにあれば重複
        simple = [t for t in items if not word_set(t) & {"with", "and"}]
        covered = {w for t in simple + list((tagger_tags or {}).get(slot, [])) for w in word_set(t)}
        compound = [t for t in items if t not in simple]
        dropped += [t for t in compound if word_set(t) - {"with", "and", "a", "the"} <= covered]
        items = simple + [t for t in compound if t not in dropped]
        # white / puffy / headband のような、ほかのタグ (white headband) の一部だけの語も重複
        # (組み合わせの一部になっている単独のタグは消さない)
        kept = [t for t in items if not any(word_set(t) < word_set(o) for o in simple)]
        dropped += [t for t in items if t not in kept]
        sorted_tags[slot] = kept
    # 末尾へ回す語も、服装・ポーズのタグの一部だけ (white ⊂ white bikini) なら重複
    placed = [word_set(t) for items in sorted_tags.values() for t in items]
    fragments = [t for t in to_extra if any(word_set(t) < w for w in placed)]
    dropped += fragments
    to_extra = [t for t in dedup(to_extra) if t not in fragments]
    return sorted_tags, to_extra, dedup(dropped)


def clean_style_tags(raw: Any) -> list[str]:
    """⑦ 作風・質感から VLM が返した画風のタグ。画風ではないもの (髪・目・肌・顔・表情・服・ポーズ・体格・場所・物) は外す
    (⑦ は画風だけを取る枠。人種・肌の色と年代は取らない)。"""
    out: list[str] = []
    for tag in clean_llm_tags(raw, limit=10):
        n = norm_tag(tag).replace(" ", "_")
        words = set(re.split(r"[_-]", n))
        if (pose_source_tag(n) or classify_outfit_tag(n) or body_tag(n) or _FEATURE_RE.search(n) or n in FACE_TAGS
                or n in _EXPRESSION_TAGS or n in _IDENTITY_TAGS or n in PERSON_COUNT_TAGS or words & _SCENE_WORDS
                or classify_subject_tag(n, True) in ("background", "object") or ERA_TAG_RE.match(n)
                or re.search(r"(?<![a-z0-9])((1[89]|20)?\d0s)(?![a-z0-9])", n.replace("_", " "))):
            continue
        out.append(tag)
    return out


_STYLE_GENERIC_WORDS = set().union(*_MEDIUM_WORDS.values()) | {
    "art", "artwork", "style", "styled", "realistic", "photography", "photographic", "picture", "image", "comic", "comics",
    "japanese", "western", "american", "modern", "retro", "vintage", "digital", "traditional", "cartoon", "a", "an", "the",
    # 画像サイトの名前 (実機の qwen2.5vl が名前の欄に danbooru と答えた)
    "danbooru", "gelbooru", "safebooru", "booru", "pixiv", "artstation", "deviantart", "twitter", "civitai",
    # 撮影・絵のジャンル (実機で portrait photography と答えた)
    "portrait", "studio", "fashion", "editorial", "street", "landscape", "glamour", "snapshot", "selfie", "cosplay",
    "film", "cinematic", "candid", "headshot", "beauty", "commercial", "stock", "illustrative", "concept",
}


def clean_style_names(raw: Any, exclude: list[str] | None = None) -> list[str]:
    """⑦ 作風・質感から VLM が推定した作品名・作者名 (最大 3)。「by 〜」「〜 style」は外して名前だけにする。
    VLM の推定なので当たるとは限らない (確認と修正は使う人に任せる。利用者の方針)。
    名前ではないもの (実機の qwen2.5vl が返した anime / serafuku のような画風の一般語・⑦ の服) は外す。
    exclude は ⑦ の画像のタガーのタグ (同じ語は画像に写っているものなので名前ではない)。"""
    excluded = {loose_tag_key(t) for t in exclude or []}
    names: list[str] = []
    for item in (raw if isinstance(raw, list) else [])[:10]:
        if len(names) >= 3:
            break
        name = re.sub(r"^(by|in the style of)\s+", "", llm_text(item).strip().strip("\"'"), flags=re.IGNORECASE)
        name = re.sub(r"\s+(art\s*)?style$", "", name, flags=re.IGNORECASE).strip(" .,")
        key = norm_tag(name).replace(" ", "_")
        words = set(re.findall(r"[a-z0-9-]+", name.lower()))
        if not name or len(name) > 60 or "," in name or name.lower() in {n.lower() for n in names}:
            continue
        if words <= _STYLE_GENERIC_WORDS or loose_tag_key(name) in excluded:
            continue
        if (classify_outfit_tag(key) or pose_source_tag(key) or body_tag(key) or _FEATURE_RE.search(key)
                or classify_extra_tag(key) or classify_subject_tag(key, True) in ("background", "object")):
            continue
        names.append(name)
    return names


def foreign_reference_tags(merged: TagResult, ref_tags: dict[str, TagResult], displaced: list[str]) -> list[str]:
    """複数枚モードで、どの画像からも取らなかったタグ: ① から押し出した服・ポーズと、参照画像にあるが
    その枠の担当ではない要素 (② の服、③ の髪・背景など)。合成後の本文にあるものは除く。"""
    have = {n for n, _ in merged.general}
    out = list(displaced)
    for slot, ref in ref_tags.items():
        out += [sd_tag(n) for n, _ in ref.general
                if n not in have and n not in PERSON_COUNT_TAGS and n not in ("solo", "no_humans")
                and slot_owner(reference_category(n, "style" in ref_tags), ref_tags) != slot]
    return dedup(out)


_TRAILING_GENERIC = {"setting", "scene", "background", "view", "area", "environment", "style", "look"}


def mentions_foreign(tag: str, foreign: list[str]) -> bool:
    """タグが foreign のどれかを語として含むか (red shirt ⊃ shirt、hand on chin = hand on own chin)。"""
    def strip_own(text: str) -> str:
        return re.sub(r"\bown\b", " ", text)

    if loose_tag_key(tag) in {loose_tag_key(t) for t in foreign}:
        return True  # 複数形・語順の違い (miniskirts = miniskirt)
    words = re.findall(r"[a-z0-9]+", unescape_parens(tag).lower())
    while len(words) > 1 and words[-1] in _TRAILING_GENERIC:
        words.pop()  # beach setting → beach
    head = words[-1] if words else ""
    multi = [strip_own(t) for t in foreign if len(re.findall(r"[a-z0-9]+", norm_tag(t))) > 1]
    single = {norm_tag(t) for t in foreign if len(re.findall(r"[a-z0-9]+", norm_tag(t))) == 1}
    # 1 語の要素は主な語のときだけ (red shirt ⊃ shirt は外す、sun flare ⊃ sun は光の種類なので外さない)
    return head in single or bool(mentions_any(strip_own(tag), multi))


def vlm_person_tags(person_raw: TagResult, merged: TagResult, ref_tags: dict[str, Any]) -> TagResult:
    """複数枚モードで VLM に渡す ① のタグ。② / ③ などが受け持つ服・ポーズは渡さない (「正解」として渡すと VLM が
    ① の服やポーズで書く)。レーティングは合成後 (③ があれば ③) のもの。"""
    general = [(n, p) for n, p in person_raw.general
               if slot_owner(reference_category(n, "style" in ref_tags), ref_tags) == "person"]
    return TagResult(merged.rating, merged.rating_conf, general, list(person_raw.characters), {})


AMBIENT_MOTION = [
    ("cherry_blossoms", "cherry blossom petals drift through the air"),
    ("petals", "petals drift slowly"),
    ("falling_leaves", "leaves fall gently"),
    ("autumn_leaves", "leaves flutter in the breeze"),
    ("snow", "snow falls softly"),
    ("snowing", "snow falls softly"),
    ("rain", "rain falls steadily"),
    ("wind", "a light wind moves through the scene"),
    ("floating_hair", "hair sways in the breeze"),
    ("wind_lift", "fabric lifts in the wind"),
    ("water", "water ripples softly"),
    ("ocean", "waves roll in slowly"),
    ("waves", "waves roll in slowly"),
    ("river", "the water flows gently"),
    ("fire", "flames flicker"),
    ("smoke", "smoke drifts upward"),
    ("steam", "steam rises slowly"),
    ("clouds", "clouds drift slowly"),
    ("cloudy_sky", "clouds drift slowly"),
    ("grass", "grass sways lightly"),
    ("curtains", "curtains sway"),
    ("light_particles", "light particles float"),
    ("sparkle", "sparkles glint"),
    ("fog", "fog drifts slowly"),
    ("mist", "mist drifts slowly"),
    ("bubbles", "bubbles rise"),
    ("fireworks", "fireworks bloom in the sky"),
    ("candle", "candlelight flickers"),
    ("lens_flare", "the light flare shifts subtly"),
    ("night_sky", "stars twinkle faintly"),
    ("starry_sky", "stars twinkle faintly"),
]


def build_motion_fallback(tags: TagResult) -> str:
    """VLM 無しでも出せる、画像に合った i2v 用のカメラワーク文 (ルールベース)。"""
    have = {n for n, _ in tags.general}
    if have & {"close-up", "portrait", "face_focus", "eye_focus"}:
        camera = "slow push-in toward the face"
    elif have & {"scenery", "wide_shot", "very_wide_shot", "landscape", "no_humans", "panorama"}:
        camera = "slow lateral pan across the scene with gentle parallax"
    elif "from_above" in have:
        camera = "slow crane down toward the subject"
    elif "from_below" in have:
        camera = "slow tilt up along the subject"
    elif have & {"full_body", "cowboy_shot"}:
        camera = "slow dolly-in with slight parallax between subject and background"
    else:
        camera = "slow push-in with slight parallax"
    motions = [text for tag, text in AMBIENT_MOTION if tag in have]
    if "no_humans" not in have:
        if "looking_at_viewer" in have:
            motions.append("the subject holds eye contact and blinks naturally")
        if have & {"smile", "grin", "open_mouth"}:
            motions.append("a subtle change of expression")
        if have & {"long_hair", "very_long_hair"} and "floating_hair" not in have:
            motions.append("hair moves slightly")
        motions.append("subtle breathing and minimal body movement")
    detail = "; ".join(dedup(motions)) or "subtle ambient motion only"
    return f"Camera: {camera}. Motion: {detail}. Keep the framing, lighting and style of the source image; no new objects."


def build_extras(
    tags: TagResult,
    vlm_out: dict[str, Any] | None,
    extra_tags: list[str],
    kind: str,
    style_threshold: float = 0.15,
    era: dict[str, Any] | None = None,
    references: bool = False,
) -> dict[str, dict[str, Any]]:
    """カメラ / ライティング / 画風 / 動画用カメラワークの追加プロンプトを組み立てる。

    画風タグは通常の閾値より低い style_threshold でも拾い、VLM の style_tags と年代スタイル (era) も足す。
    """
    have = {n for n, _ in tags.general}
    has_humans = "no_humans" not in have
    buckets: dict[str, list[str]] = {
        "subject": [], "pose": [], "outfit": [], "objects": [], "background": [], "camera": [], "lighting": [], "style": [],
    }
    subject_bucket = {"person": "subject", "object": "objects", "background": "background"}
    for name, _ in tags.characters:
        buckets["subject"].append(sd_tag(name))  # キャラクター名は被写体
    for name, _ in tags.general:
        group = classify_extra_tag(name)
        if group:
            buckets[group].append(sd_tag(name))
            continue
        if has_humans and classify_pose_tag(name):
            where: str | None = "pose"
        elif has_humans and (classify_outfit_tag(name) or (references and name in EXPOSURE_TAGS)):
            # 複数枚モードでは服で決まる肌の見え方 (navel / cleavage など) も ③ の服装の欄に
            where = "outfit"
        else:
            where = classify_subject_tag(name, has_humans)
        if where:
            buckets[subject_bucket.get(where, where)].append(sd_tag(name))
        if name.startswith("holding_") and name[8:] in OBJECT_TAGS:
            buckets["objects"].append(sd_tag(name[8:]))  # 持っている物も物体側に
    for tag in extra_tags:  # VLM が補ったタグも同じ基準で振り分ける
        group = classify_extra_tag(tag.replace(" ", "_"))
        if group:
            buckets[group].append(tag)
    if vlm_out:  # 複数枚モードの VLM はポーズと服装のタグも返す
        buckets["pose"] += clean_llm_tags(vlm_out.get("pose_tags"), limit=12)
        buckets["outfit"] += clean_llm_tags(vlm_out.get("outfit_tags"), limit=16)
    # 画風・技法タグは信頼度が低く出やすいので、通常の閾値に届かないものも style_threshold 以上なら拾う
    picked = {norm_tag(t) for t in buckets["style"]}
    low_style = [
        (name, p) for name, p in tags.general_probs.items()
        if p >= style_threshold and classify_extra_tag(name) == "style" and norm_tag(sd_tag(name)) not in picked
    ]
    low_style.sort(key=lambda t: -t[1])
    buckets["style"] += [sd_tag(name) for name, _ in low_style]
    # VLM が挙げた画風タグ、年代スタイル分類のタグ
    if vlm_out:
        buckets["style"] += clean_llm_tags(vlm_out.get("style_tags"), limit=10)
    if era and era.get("tags"):
        buckets["style"] += [sd_escape(t) for t in era["tags"]]
    vlm_objects = clean_llm_tags((vlm_out or {}).get("objects"), limit=8) if vlm_out else []
    buckets["objects"] += vlm_objects
    extras: dict[str, dict[str, Any]] = {}
    for key in ("subject", "pose", "outfit", "objects", "background", "camera", "lighting", "style"):
        tag_list = dedup(buckets[key])
        if key == "objects":  # VLM の objects は配列なので文字列化しない
            text = _plain(vlm_objects)
        else:
            text = llm_text((vlm_out or {}).get(key))
        prompt = ", ".join(tag_list) if (kind != "prose" and tag_list) else (text or ", ".join(tag_list))
        extras[key] = {"tags": tag_list, "text": text, "prompt": prompt}
    motion_text = llm_text((vlm_out or {}).get("motion"))
    source = "vlm" if motion_text else "rule"
    if not motion_text:
        motion_text = build_motion_fallback(tags)
    extras["face"] = empty_face_extra()  # 人物の特徴を詳しく がオンなら Engine が埋める
    extras["motion"] = {"tags": [], "text": motion_text, "prompt": motion_text, "source": source}
    return extras


ASPECT_PRESETS = (("16:9", 16 / 9), ("3:2", 1.5), ("4:3", 4 / 3), ("1:1", 1.0), ("3:4", 0.75), ("2:3", 2 / 3), ("9:16", 9 / 16))


def aspect_label(size: tuple[int, int]) -> str:
    """画像サイズから近い縦横比を言葉で返す (例: '3:2, landscape')。"""
    w, h = size
    ratio = (w / h) if h else 1.0
    name = min(ASPECT_PRESETS, key=lambda p: abs(np.log(ratio) - np.log(p[1])))[0]
    orientation = "square" if abs(ratio - 1) < 0.05 else ("landscape" if ratio > 1 else "portrait")
    return f"{name}, {orientation}"


MEDIUM_PHRASE = {
    "anime": "anime-style illustration",
    "photo": "photorealistic photograph",
    "3d": "3D rendered image",
    "pixel_art": "pixel art image",
    "painting": "painterly illustration",
    "unspecified": "image",  # 作風の特徴を入れない (絵か写真かを書かない)
}


def _plain(tag_list: list[str]) -> str:
    return ", ".join(t.replace("\\(", "(").replace("\\)", ")") for t in tag_list)


def _with_article(phrase: str, capital: bool = True) -> str:
    """'a photorealistic photograph' / 'an anime-style illustration' のように冠詞を付ける。"""
    article = "an" if phrase[:1].lower() in "aeiou" else "a"
    if capital:
        article = article.capitalize()
    return f"{article} {phrase}"


# 否定文 ("no other people" など) を書いてよい出力先。GPT Image のガイドは制約として除外を明示するよう勧めるが、
# FLUX.2 は "does not support negative prompts. Focus on describing what you want"、Nano Banana も肯定形で書くよう求める
PROSE_ALLOWS_NEGATION = {"chatgpt"}


# --- 人どうしの動作 ---
# 1 人ずつ切り出すと相手をグレーで塗るので、抱きつく・肩に手などの動作は人物ごとのタグから落ちる。画像全体のタグから取り直し、
# タグの出力先は共有部分、自然文は関係の文 (BFL / Krea 2 / Qwen のガイドの「関係は文で直接書く」。2026-10-17 の生成の
# 実験で、肩に手が Z-Image 4/8 → 8/8、Anima 2/8 → 8/8)、NovelAI はキャラ欄に source# / target# / mutual#
# (docs.novelai.net/en/image/multiplecharacters。公式も「いつも効くわけではない」)。
# 向きは VLM → 骨格 → 分からなければ向きを書かない。mutual# は「お互い」なので一方的な動作には使わない。
# タグはタガー (wd-eva02-large-tagger-v3) の語彙にあるもの。directed の {b} はされる側、mutual はお互いにする動作
INTERACTION_TAGS: dict[str, tuple[str, str]] = {
    "hug": ("directed", "is hugging {b}"),
    "hug_from_behind": ("directed", "is hugging {b} from behind"),
    "incoming_hug": ("directed", "is about to hug {b}"),
    "arm_hug": ("directed", "is hugging {b}'s arm"),
    "headpat": ("directed", "is patting {b}'s head"),
    "cheek_pinching": ("directed", "is pinching {b}'s cheek"),
    "princess_carry": ("directed", "is carrying {b} in a princess carry"),
    "piggyback": ("directed", "is giving {b} a piggyback ride"),
    "shoulder_carry": ("directed", "is carrying {b} on their shoulders"),
    "child_carry": ("directed", "is carrying {b} in their arms"),
    "carrying_person": ("directed", "is carrying {b}"),
    "carrying_over_shoulder": ("directed", "is carrying {b} over their shoulder"),
    "carrying_under_arm": ("directed", "is carrying {b} under their arm"),
    "lifting_person": ("directed", "is lifting {b}"),
    "lap_pillow": ("directed", "is letting {b} rest their head on their lap"),
    "sitting_on_lap": ("directed", "is sitting on {b}'s lap"),
    "sitting_on_person": ("directed", "is sitting on {b}"),
    "sitting_on_shoulder": ("directed", "is sitting on {b}'s shoulder"),
    "lying_on_person": ("directed", "is lying on {b}"),
    "lying_on_lap": ("directed", "is lying on {b}'s lap"),
    "sleeping_on_person": ("directed", "is sleeping on {b}"),
    "leaning_on_person": ("directed", "is leaning on {b}"),
    "head_on_chest": ("directed", "is resting their head on {b}'s chest"),
    "head_on_head": ("directed", "is resting their head on {b}'s head"),
    "behind_another": ("directed", "is standing behind {b}"),
    "feeding": ("directed", "is feeding {b}"),
    "kissing_cheek": ("directed", "is kissing {b} on the cheek"),
    "kissing_forehead": ("directed", "is kissing {b} on the forehead"),
    "kissing_hand": ("directed", "is kissing {b}'s hand"),
    "kissing_neck": ("directed", "is kissing {b}'s neck"),
    "surprise_kiss": ("directed", "is kissing {b} by surprise"),
    "incoming_kiss": ("directed", "is about to kiss {b}"),
    "arm_around_shoulder": ("directed", "has an arm around {b}'s shoulders"),
    "arm_around_waist": ("directed", "has an arm around {b}'s waist"),
    "arm_around_neck": ("directed", "has an arm around {b}'s neck"),
    "arm_around_back": ("directed", "has an arm around {b}'s back"),
    "arms_around_neck": ("directed", "has both arms around {b}'s neck"),
    "arms_around_waist": ("directed", "has both arms around {b}'s waist"),
    "tickling": ("directed", "is tickling {b}"),
    "tickling_armpits": ("directed", "is tickling {b}'s armpits"),
    "slapping": ("directed", "is slapping {b}"),
    "pushing_away": ("directed", "is pushing {b} away"),
    "chasing": ("directed", "is chasing {b}"),
    "whispering": ("directed", "is whispering to {b}"),
    "comforting": ("directed", "is comforting {b}"),
    "hairdressing": ("directed", "is doing {b}'s hair"),
    "looking_at_another": ("directed", "is looking at {b}"),
    "facing_another": ("directed", "is facing {b}"),
    "holding_hands": ("mutual", "are holding hands"),
    "kiss": ("mutual", "are kissing"),
    "french_kiss": ("mutual", "are French kissing"),
    "imminent_kiss": ("mutual", "are about to kiss"),
    "after_kiss": ("mutual", "have just kissed"),
    "mutual_hug": ("mutual", "are hugging each other"),
    "group_hug": ("mutual", "are in a group hug"),
    "cheek-to-cheek": ("mutual", "are cheek to cheek"),
    "face-to-face": ("mutual", "are face to face"),
    "back-to-back": ("mutual", "are standing back to back"),
    "heads_together": ("mutual", "have their heads together"),
    "eye_contact": ("mutual", "are making eye contact"),
    "handshake": ("mutual", "are shaking hands"),
    "fist_bump": ("mutual", "are bumping fists"),
    "fighting": ("mutual", "are fighting"),
}
# hand_on_another's_shoulder / grabbing_another's_hair / pointing_at_another / dressing_another など、相手を another と書くタグ
_ANOTHER_RE = re.compile(r"^(?P<verb>[a-z]+)_(?:(?P<prep>on|in|to|at)_)?another(?:'s_(?P<part>[a-z_]+))?$")
_ANOTHER_NOUN = {"hand": "a hand", "hands": "both hands", "finger": "a finger"}


def interaction_phrase(name: str) -> tuple[str, str] | None:
    """人どうしの動作のタグなら (directed / mutual, 英語の句) を返す。句の {b} は相手。動作でなければ None。"""
    if name in INTERACTION_TAGS:
        return INTERACTION_TAGS[name]
    m = _ANOTHER_RE.match(name)
    if not m:
        return None
    verb, prep, part = m["verb"], m["prep"], m["part"]
    other = "{b}'s " + part.replace("_", " ") if part else "{b}"
    if verb in _ANOTHER_NOUN and prep and part:
        lead = "is holding a finger" if (verb, prep) == ("finger", "to") else "has " + _ANOTHER_NOUN[verb]
        return "directed", f"{lead} {prep} {other}"
    if verb == "head" and prep and part:
        return "directed", f"is resting their head {prep} {other}"
    if verb.endswith("ing"):
        return "directed", f"is {verb}" + (f" {prep}" if prep else "") + f" {other}"
    return None


_HEAD_POINTS, _TORSO_POINTS, _HAND_POINTS = [0, 1, 2, 3, 4], [5, 6, 11, 12], [9, 10]
# 体の部位の名前 → 骨格 (COCO 17 点) の点
BODY_PART_POINTS: dict[str, list[int]] = {
    "head": _HEAD_POINTS, "hair": _HEAD_POINTS, "face": _HEAD_POINTS, "cheek": _HEAD_POINTS, "cheeks": _HEAD_POINTS,
    "chin": [0], "mouth": [0], "ear": [3, 4], "eyes": [1, 2], "neck": [0, 5, 6], "shoulder": [5, 6], "shoulders": [5, 6],
    "arm": [7, 8], "hand": _HAND_POINTS, "chest": _TORSO_POINTS, "breast": _TORSO_POINTS, "breasts": _TORSO_POINTS,
    "back": _TORSO_POINTS, "stomach": _TORSO_POINTS, "clothes": _TORSO_POINTS, "waist": [11, 12], "hip": [11, 12],
    "hips": [11, 12], "ass": [11, 12], "thigh": [11, 12, 13, 14], "thighs": [11, 12, 13, 14], "leg": [11, 12, 13, 14, 15, 16],
    "foot": [15, 16],
}
# 骨格で向きを決める動作: (する側の点, される側の点)。手で触る動作と、頭・口を当てる動作だけ (抱きつくなどは骨格では分からない)
INTERACTION_REACH: dict[str, tuple[list[int], list[int]]] = {
    "headpat": (_HAND_POINTS, _HEAD_POINTS), "cheek_pinching": (_HAND_POINTS, _HEAD_POINTS),
    "arm_around_shoulder": (_HAND_POINTS, [5, 6]), "arm_around_waist": (_HAND_POINTS, [11, 12]),
    "arms_around_waist": (_HAND_POINTS, [11, 12]), "arm_around_neck": (_HAND_POINTS, [0, 5, 6]),
    "arms_around_neck": (_HAND_POINTS, [0, 5, 6]), "kissing_cheek": ([0], _HEAD_POINTS),
    "kissing_forehead": ([0], _HEAD_POINTS), "kissing_hand": ([0], _HAND_POINTS), "kissing_neck": ([0], [0, 5, 6]),
    "head_on_chest": (_HEAD_POINTS, _TORSO_POINTS),
}
SKELETON_TOUCH = 0.8  # する側の点とされる側の部位の距離が、される側の肩幅のこの倍数未満なら「触れている」
SKELETON_CLEAR = 0.6  # 2 番目の候補 (逆向き・別の人) も触れているときは、1 番目がその距離のこの割合以下でないと決めない


def interaction_reach(name: str) -> tuple[list[int], list[int]] | None:
    """骨格で向きを決められる動作なら (する側の点, される側の点)。"""
    if name in INTERACTION_REACH:
        return INTERACTION_REACH[name]
    m = _ANOTHER_RE.match(name)
    if not m or not m["part"] or m["part"] not in BODY_PART_POINTS:
        return None
    if m["verb"] == "head":
        return _HEAD_POINTS, BODY_PART_POINTS[m["part"]]
    if m["verb"] in _ANOTHER_NOUN or m["verb"].endswith("ing"):
        return _HAND_POINTS, BODY_PART_POINTS[m["part"]]
    return None


def skeleton_direction(
    reach: tuple[list[int], list[int]], boxes: list[Box], skeletons: dict[Box, list[list[float]]],
) -> tuple[int, int] | None:
    """骨格で (する側, される側) の番号を決める。される側の肩幅で割った距離が一番近い組を選び、
    触れていない・ほかの組とはっきり差が無いときは None (向きを書かない)。当たり率は未計測 (DEVLOG 78)。"""
    scores = []
    for i, bi in enumerate(boxes):
        for j, bj in enumerate(boxes):
            pi, pj = skeletons.get(bi), skeletons.get(bj)
            if i == j or not pi or not pj:
                continue
            src = [q for q in (_pt(pi, k) for k in reach[0]) if q]
            dst = [q for q in (_pt(pj, k) for k in reach[1]) if q]
            if not src or not dst:
                continue
            left, right = _pt(pj, 5), _pt(pj, 6)
            scale = max(1.0, _dist(left, right) if left and right else (bj[2] - bj[0]) * 0.4)
            scores.append((min(_dist(a, b) for a in src for b in dst) / scale, i, j))
    if not scores:
        return None
    scores.sort()
    best = scores[0]
    if best[0] >= SKELETON_TOUCH:
        return None
    if len(scores) > 1 and scores[1][0] < SKELETON_TOUCH and best[0] > scores[1][0] * SKELETON_CLEAR:
        return None
    return best[1], best[2]


def resolve_interactions(
    names: list[str], boxes: list[Box], skeletons: dict[Box, list[list[float]]] | None,
) -> list[dict[str, Any]]:
    """人どうしの動作ごとに、誰から誰へか (pair = [する側, される側]、お互いなら 2 人) を決める。
    by: skeleton (骨格) / count (2 人だけなのでお互いの動作の 2 人が決まる) / "" (分からない)。VLM はあとで上書きする。"""
    out = []
    for name in names:
        found = interaction_phrase(name)
        if not found:
            continue
        item: dict[str, Any] = {"name": name, "tag": sd_tag(name), "mutual": found[0] == "mutual", "pair": None, "by": ""}
        if item["mutual"] and len(boxes) == 2:
            item["pair"], item["by"] = [0, 1], "count"
        elif not item["mutual"] and skeletons:
            reach = interaction_reach(name)
            direction = skeleton_direction(reach, boxes, skeletons) if reach else None
            if direction:
                item["pair"], item["by"] = list(direction), "skeleton"
        out.append(item)
    return out


def apply_vlm_interactions(
    persons: list[dict[str, Any]], entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """VLM が番号付きの枠で答えた動作の向きで上書きする。番号が範囲外・同じ人どうし・知らない動作は使わない。"""
    items = (persons[0].get("interactions") if persons else None) or []
    if not items or not entries:
        return persons, []
    items = [dict(it) for it in items]
    bad: list[str] = []
    for entry in entries:
        key = loose_tag_key(entry["tag"])
        item = next((it for it in items if loose_tag_key(it["tag"]) == key), None)
        src, dst = entry["source"], entry["target"]
        if item is None or not (1 <= src <= len(persons) and 1 <= dst <= len(persons)) or src == dst:
            bad.append(f"{entry['tag']} ({src} → {dst})")
            continue
        item["pair"], item["by"] = [src - 1, dst - 1], "vlm"
        item["mutual"] = bool(entry["mutual"]) or interaction_phrase(item["name"])[0] == "mutual"
    out = [dict(p) for p in persons]
    out[0]["interactions"] = items
    warnings = ["VLM が返した人どうしの動作の向きのうち、使えないもの (タガーに無い動作・存在しない番号・同じ人) は使いませんでした: "
                + ", ".join(bad)] if bad else []
    return out, warnings


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def _fill_other(phrase: str, other: str) -> str:
    """句の {b} に相手を入れる。相手を位置で呼ぶ長い名詞句なら、所有格は of で書く
    ('the woman on the left's shoulder' ではなく 'the shoulder of the woman on the left')。"""
    if other.startswith("the ") and " " in other[4:]:
        phrase = re.sub(r"\{b\}'s ([a-z][a-z' -]*)$", r"the \g<1> of {b}", phrase)
    return phrase.format(b=other)


def _mutual_phrase(phrase: str) -> str:
    """一方向の句をお互いの句に (is hugging {b} → are hugging each other)。"""
    phrase = re.sub(r"^is ", "are ", re.sub(r"^has ", "have ", phrase))
    return phrase.replace("{b}'s", "each other's").replace("{b}", "each other")


def interaction_lines(persons: list[dict[str, Any]]) -> list[tuple[str, list[int] | None]]:
    """人どうしの動作の英文と、関わる人の番号 (分からなければ None)。向きが分からない動作は向きを書かない
    (2 人: One of them ... the other、3 人以上: One person ... another person)。"""
    out: list[tuple[str, list[int] | None]] = []
    for item in (persons[0].get("interactions") if persons else None) or []:
        found = interaction_phrase(item["name"])
        if not found:
            continue
        kind, phrase = found
        pair = item.get("pair")
        if pair and not all(0 <= k < len(persons) for k in pair):
            pair = None
        everyone = list(range(len(persons))) if len(persons) == 2 else None
        if item.get("mutual"):
            verb = phrase if kind == "mutual" else _mutual_phrase(phrase)
            if pair:
                text = f"{person_lead_en(persons[pair[0]])} and {_lower_first(person_lead_en(persons[pair[1]]))} {verb}"
            else:
                text = ("The two of them " if len(persons) == 2 else "Some of them ") + verb
        elif pair:
            text = f"{person_lead_en(persons[pair[0]])} " + _fill_other(phrase, _lower_first(person_lead_en(persons[pair[1]])))
        else:
            text = ("One of them " + phrase.format(b="the other")) if len(persons) == 2 \
                else "One person " + phrase.format(b="another person")
        out.append((text + ".", pair or everyone))
    return out


def novelai_action_tags(persons: list[dict[str, Any]], index: int) -> list[str]:
    """NovelAI のキャラ欄に入れる動作のタグ (source# / target# / mutual#)。向き (お互いなら 2 人) が分かった動作だけ。"""
    tags = []
    for item in (persons[0].get("interactions") if persons else None) or []:
        pair = item.get("pair")
        if not pair or index not in pair:
            continue
        tag = unescape_parens(item["tag"])
        tags.append(f"mutual#{tag}" if item.get("mutual") else f"source#{tag}" if pair[0] == index else f"target#{tag}")
    return tags


NAI_PRONOUN = {"girl": "She is", "boy": "He is", "other": "They are"}


def novelai_position_sentence(person: dict[str, Any]) -> str:
    """NovelAI のキャラ欄の最後に足す位置の文 (公式: 位置は並び順と食い違わないようにし、自然な文でも補う)。"""
    position = str(person.get("position") or "")
    if not position:
        return ""
    where = position_phrase_en(position) if position.startswith(("left", "right", "center")) else position
    return f"{NAI_PRONOUN.get(person.get('subject', 'other'), 'They are')} {where}."


def position_phrase_en(position: str) -> str:
    """位置の前置詞句。'on the left' / 'on the right' に対し、中央は 'in the center' ('on the center' は英語として誤り)。"""
    return f"in the {position}" if position == "center" or position.startswith("center,") else f"on the {position}"


def describe_persons_en(persons: list[dict[str, Any]], target: str = "") -> str:
    """人物を左から順に 1 人 1 文で書く英文。最初に人数を肯定文で書く (Qwen-Image の left → centre → right の順、
    Anima の「名前だけでなく外見を文で」に合わせる)。否定文は GPT Image 向けのときだけ最後に付ける。"""
    genders = {p["subject"] for p in persons}
    noun = group_noun_en(persons)
    count = NUMBER_EN.get(len(persons), str(len(persons)))
    sentences = [f"{count.capitalize()} {noun} appear in the scene, described from left to right."]
    common = persons[0].get("common") or []
    if common:
        everyone = "Both" if len(persons) == 2 else f"All {count}"
        sentences.append(f"{everyone} share these features: {_plain_prose(common)}.")
    for person in persons:
        who = age_noun_en(person)
        position = str(person["position"])
        lead = position_phrase_en(position).capitalize() if position in ("left", "right", "center") or position.startswith(("left,", "right,", "center,")) \
            else position[:1].upper() + position[1:]
        features = _plain_prose(without_age_tags(person))
        sentences.append(f"{lead}, {who}" + (f" with {features}." if features else "."))
    sentences += [text for text, _ in interaction_lines(persons)]
    if target in PROSE_ALLOWS_NEGATION:
        sentences.append(f"No other people besides these {count}.")
    return " ".join(sentences)


def person_lead_en(person: dict[str, Any]) -> str:
    """人物を位置で呼ぶ英語 ('The woman on the left' / 'The man second from the left')。"""
    who = age_noun_en(person, article=False)
    position = str(person.get("position") or "")
    if position in ("left", "right", "center") or position.startswith(("left,", "right,", "center,")):
        return f"The {who} {position_phrase_en(position)}"
    return f"The {who} {position}".rstrip()


def flux_structured_prompt(
    persons: list[dict[str, Any]], extras: dict[str, dict[str, Any]], medium: str, distinct_faces: bool = False,
) -> str:
    """FLUX.2 の公式ガイドが複数の被写体に勧める JSON 構造化プロンプト
    (scene / subjects[description, position, action] / style / lighting / background / composition)。"""
    subjects = []
    relations = interaction_lines(persons)
    for index, person in enumerate(persons):
        who = age_noun_en(person)
        everything = dedup(list(person.get("common") or []) + person["tags"])
        pose = [t for t in everything if person_tag_priority(t) == 3]
        look = [t for t in everything if person_tag_priority(t) != 3]
        description = f"{who} with {_plain(look)}" if look else who
        if distinct_faces or mixed_gender(persons):
            # 同じ顔を避ける: 人ごとに顔の違いを書く (FLUX.2 はネガティブ非対応なので肯定文で)
            face = GENDER_FACE_EN.get(person["subject"], "a face of their own")
            description += f", {face}" + (", clearly different from the others" if distinct_faces else "")
        subjects.append({
            "description": description,
            "position": person["position"],
            # 人どうしの動作は関わる人の action に (FLUX.2 の JSON に関係の欄は無い。DEVLOG 78)
            "action": " ".join(x for x in [_plain(pose)] + [t for t, who_ in relations if who_ and index in who_] if x),
        })
    unplaced = " ".join(t for t, who_ in relations if not who_)
    data = {
        "scene": " ".join(x for x in (extras["background"]["text"] or _plain(extras["background"]["tags"]), unplaced) if x),
        "subjects": subjects,
        "style": extras["style"]["text"] or ", ".join(x for x in (MEDIUM_PHRASE.get(medium, ""), _plain(extras["style"]["tags"])) if x),
        "lighting": extras["lighting"]["text"] or _plain(extras["lighting"]["tags"]),
        "composition": extras["camera"]["text"] or _plain(extras["camera"]["tags"]),
    }
    return json.dumps({k: v for k, v in data.items() if v}, ensure_ascii=False, indent=2)


COUNT_WORDS_EN = {1: "a", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
# --no に入れない語: SD 向けの品質語 (lowres, worst quality など) は Midjourney では意味が無い。
# 内容の審査・規約はプラットフォーム側が扱うので、こちらでは語を選り分けない (利用者の方針)
MJ_NO_SKIP = {
    "lowres", "bad anatomy", "bad hands", "error", "missing fingers", "extra digit", "fewer digits", "cropped",
    "worst quality", "low quality", "normal quality", "jpeg artifacts", "username", "blurry", "deformed",
    "disfigured", "extra limbs", "artistic error", "scan artifacts", "bad quality", "very displeasing",
    "displeasing", "chromatic aberration", "dithering", "halftone", "screentone", "too many watermarks",
    "negative space", "blank page", "film grain", "score 1", "score 2", "score 3", "multiple views",
}
MJ_NO_MAX = 10


# --------------------------------------------------------------------------
# Danbooru の語 → ふつうの英語 (文章の出力先)
# --------------------------------------------------------------------------
# 文章を読む生成モデル (MOSS-Image / Krea 2 / Qwen-Image / Z-Image / FLUX など) は、Danbooru だけで通じる語
# (cowboy shot、ahoge、zettai ryouiki など) を知らないか、弱くしか効かない。タグから文を組み立てるときは、この表で言い換える。
# 表に無いタグはそのまま書く (long hair、school uniform のように、ふつうの英語として通じるものが大半)。空文字は書かない語
DANBOORU_PLAIN = {
    # 画角・アングル
    "cowboy shot": "framed from the thighs up", "upper body": "upper body shot", "lower body": "lower body shot",
    "full body": "full body shot", "portrait": "close-up portrait", "wide shot": "wide shot showing the surroundings",
    "from above": "seen from above", "from below": "seen from below", "from side": "seen from the side",
    "from behind": "seen from behind", "dutch angle": "tilted camera angle", "pov": "first-person point of view",
    "foreshortening": "strong perspective foreshortening", "fisheye": "fisheye lens distortion",
    # 視線・表情
    "looking at viewer": "looking at the camera", "looking away": "looking away from the camera",
    "looking back": "looking back over the shoulder", "looking afar": "gazing into the distance",
    ":d": "wide open-mouthed smile", ":o": "mouth open in surprise", ":3": "cat-like smile", ":p": "tongue sticking out",
    ";d": "winking with an open-mouthed smile", "^ ^": "eyes closed in a smile", "> <": "eyes squeezed shut",
    "blush": "blushing cheeks", "light blush": "slightly blushing cheeks", "one eye closed": "winking",
    "closed eyes": "eyes closed", "tsurime": "upturned eyes", "tareme": "downturned eyes", "jitome": "half-lidded flat stare",
    "fang": "a small fang", "skin fang": "a small fang", "tongue out": "tongue sticking out",
    "expressionless": "blank expression", "smug": "smug expression", "pout": "pouting", "nervous": "nervous expression",
    # 髪
    "ahoge": "a single upright strand of hair", "sidelocks": "locks of hair framing the face",
    "hair between eyes": "bangs falling between the eyes", "blunt bangs": "straight-cut bangs",
    "hair over one eye": "hair covering one eye", "twintails": "twin tails", "low twintails": "low twin tails",
    "hime cut": "straight hime-style haircut", "bob cut": "bob haircut", "hair bun": "hair in a bun",
    "double bun": "two hair buns", "drill hair": "ringlet curls", "twin drills": "twin ringlet curls",
    "multicolored hair": "multi-coloured hair", "gradient hair": "gradient-coloured hair",
    "streaked hair": "hair with coloured streaks", "colored inner hair": "hair with a different colour on the inside",
    "absurdly long hair": "extremely long hair", "hair intakes": "upswept tufts of hair at the temples",
    "hair flaps": "flap-like tufts of hair", "antenna hair": "two upright strands of hair",
    # 体・特徴
    "heterochromia": "eyes of two different colours", "tan": "tanned skin", "tanlines": "tan lines",
    "dark-skinned female": "dark skin", "dark-skinned male": "dark skin", "mole under eye": "a mole under the eye",
    "mole under mouth": "a mole below the mouth", "thick thighs": "thick thighs", "kemonomimi mode": "animal ears",
    "mature female": "mature woman", "mature male": "mature man",
    # 服
    "zettai ryouiki": "a band of bare thigh between the skirt and the thigh-high socks", "thighhighs": "thigh-high socks",
    "kneehighs": "knee-high socks", "serafuku": "sailor-style school uniform", "gakuran": "Japanese boys' school uniform",
    "detached sleeves": "separate sleeves", "off shoulder": "off-the-shoulder top", "midriff": "bare midriff",
    "navel": "exposed navel", "sleeves past wrists": "sleeves covering the wrists",
    "sleeves past fingers": "sleeves covering the hands", "frills": "frilled trim", "hairclip": "hair clip",
    "hairband": "headband", "legwear": "leg wear", "zipper pull tab": "zipper pull", "shimenawa": "sacred straw rope",
    "miko": "shrine maiden outfit", "yukata": "yukata (light summer kimono)", "hakama": "hakama (pleated Japanese trousers)",
    "geta": "geta (wooden sandals)", "tabi": "tabi (split-toe socks)", "randoseru": "Japanese school backpack",
    # ポーズ
    "v": "making a peace sign", "double v": "making peace signs with both hands", "arm up": "one arm raised",
    "hand up": "one hand raised", "own hands together": "hands clasped together", "hand on own hip": "one hand on the hip",
    "hands on own hips": "hands on the hips", "wariza": "sitting with the legs folded to the sides",
    "seiza": "kneeling in seiza", "indian style": "sitting cross-legged", "yokozuwari": "sitting with both legs to one side",
    "on back": "lying on the back", "on stomach": "lying face down", "on side": "lying on the side",
    "all fours": "on hands and knees", "head tilt": "head tilted", "paw pose": "hands held up like paws",
    "knees up": "knees drawn up", "leg up": "one leg raised", "standing on one leg": "standing on one leg",
    # 背景・光
    "simple background": "plain background", "depth of field": "shallow depth of field", "backlighting": "backlit",
    "day": "daytime", "scenery": "wide view of the scenery", "no humans": "no people",
    # 文には書かない語 (人数は別に数の表現で書く。品質・投稿の種類の語は内容ではない)
    "solo": "", "solo focus": "", "highres": "", "absurdres": "", "lowres": "", "commentary request": "",
    "translation request": "", "virtual youtuber": "", "official alternate costume": "",
}


def plain_english(tag: str) -> str:
    """タグを文章の出力先に書くときの言い方 (重みと括弧のエスケープは外す)。表に無ければそのまま。"""
    text = unescape_parens(str(tag)).strip()
    if "::" in text or re.fullmatch(r"\(.+:-?\d+(?:\.\d+)?\)", text):  # 重み付き ((tag:1.2) / 1.2::tag::) は中身だけ
        text = _strip_weight(text)
    key = text.lower().replace("_", " ")
    if key in DANBOORU_PLAIN:
        return DANBOORU_PLAIN[key]
    # Danbooru の区別用の括弧: watercolor (medium) → watercolor、1990s (style) → 1990s style
    return re.sub(r"\s*\(style\)$", " style", re.sub(r"\s*\(medium\)$", "", text))


def _plain_prose(tag_list: list[str]) -> str:
    """タグの並びを文章の出力先に書く (_plain の、Danbooru の語をふつうの英語にする版)。"""
    return ", ".join(dedup([x for x in (plain_english(t) for t in tag_list) if x]))


def count_phrase_en(tag: str, age: str = "") -> str:
    """人数タグを英語の数の表現にする (2girls → two girls、1boy → a boy)。Midjourney は数を具体的に書くよう勧めている。
    age (1 人のときの年齢層) があれば、年齢に合った言い方にする (1girl → a woman in her forties)。"""
    name = norm_tag(tag).replace(" ", "_")
    if name not in PERSON_COUNT_TAGS or name.startswith("multiple"):
        return ""
    n = PERSON_COUNT_TAGS[name]
    group = next(g for g in ("girl", "boy", "other") if g in name)
    if n == 1 and age:
        return age_noun_en({"subject": group, "age": age})
    noun = {"girl": "girl", "boy": "boy", "other": "person"}[group] if n == 1 else {"girl": "girls", "boy": "boys", "other": "people"}[group]
    word = COUNT_WORDS_EN.get(n, "several") if not name.startswith("6+") else "six or more"
    return f"{word} {noun}"


def compose_midjourney(
    extras: dict[str, dict[str, Any]], medium: str, tags: TagResult, persons: list[dict[str, Any]] | None = None,
    age: str = "",
) -> str:
    """Midjourney 向けの短い描写文 (公式: 短く簡潔に、要素は被写体・画材・環境・光・色・雰囲気・構図、数は具体的に)。
    VLM が無いとき・人物ごとに分けたときに使う。タグは信頼度の高い順に各項目の先頭だけを使う (全体は別タブにある)。"""
    def top(key: str, n: int, skip_counts: bool = True) -> list[str]:
        items = extras.get(key, {}).get("tags", [])
        if skip_counts:
            items = [t for t in items if norm_tag(t).replace(" ", "_") not in PERSON_COUNT_TAGS and norm_tag(t) != "solo"]
        return [unescape_parens(t) for t in items[:n]]

    medium_phrase = MEDIUM_PHRASE.get(medium, "illustration")
    if persons and len(persons) >= 2:
        genders = {p["subject"] for p in persons}
        noun = group_noun_en(persons)
        who = []
        for person in persons:
            one = age_noun_en(person)
            feats = _plain_prose(without_age_tags(person)[:3])
            who.append(f"{person['position']}, {one}" + (f" with {feats}" if feats else ""))
        common = _plain_prose((persons[0].get("common") or [])[:4])
        subject = f"{COUNT_WORDS_EN.get(len(persons), str(len(persons)))} {noun}" + (f" sharing {common}" if common else "")
        subject += " (" + "; ".join(who) + ")"
        subject += "".join(", " + _lower_first(text.rstrip(".")) for text, _ in interaction_lines(persons))
    else:
        counts = [count_phrase_en(t, age) for t in extras["subject"]["tags"]]
        count = next((c for c in counts if c), "")
        looks = _plain_prose([t for t in top("subject", 5) if not (age and _tag_key(t) in AGE_TAG_BANDS)])
        subject = ", ".join(x for x in (count, looks) if x) or "the scene"
    parts = [f"{_with_article(medium_phrase)} of {subject}"]
    if not (persons and len(persons) >= 2):
        outfit, pose = _plain_prose(top("outfit", 4)), _plain_prose(top("pose", 2))
        if outfit:
            parts.append(f"wearing {outfit}")
        if pose:
            parts.append(pose)
    if extras["background"]["text"]:
        parts.append(extras["background"]["text"].rstrip("."))
    else:
        places = top("background", 4)
        where = next((p for p in places if p in ("outdoors", "indoors")), "")
        things = [p for p in places if p != where][:3]
        if where or things:
            parts.append(" ".join(x for x in (where, f"with {_plain_prose(things)} in the background" if things else "") if x))
    first = ", ".join(parts) + "."
    # 画風の語は冒頭の画材 (a photorealistic photograph など) と重なるものを除く
    style = [t for t in top("style", 3) if not set(t.split()) <= set(medium_phrase.replace("-", " ").split())]
    second = [x for x in (_plain_prose(top("lighting", 2)), _plain_prose(top("camera", 2)), _plain_prose(style[:2])) if x]
    return first + (" " + ", ".join(second) + "." if second else "")


def midjourney_params(
    medium: str, size: tuple[int, int], negative: list[str], has_text: bool, body: str = "",
) -> tuple[str, list[str], list[str]]:
    """Midjourney のパラメータ (本文の末尾に付ける。公式 Parameter List: 必ず最後、句読点を付けない)。
    --ar は元画像のピクセルを約分した整数比 (公式: 小数は不可、最大 14:1)。アニメ・イラストは --niji 7、写真は --raw。
    本文に出てくる語は --no に入れない (本文と矛盾する)。negative は大事な順 (画像固有の除外を先) に渡す。
    戻り値: (パラメータ, 矛盾のため外した語, 上限 MJ_NO_MAX 語を超えて入らなかった語)。"""
    import math

    w, h = size
    g = math.gcd(w, h) or 1
    aw, ah = w // g, h // g
    if max(aw, ah) > 1000:  # 約分できない半端なサイズは近い比に丸める (Midjourney は大きな比も約分するが短く読める形に)
        aw, ah = round(w / h * 100), 100
        g2 = math.gcd(aw, ah) or 1
        aw, ah = aw // g2, ah // g2
    if max(aw, ah) / max(1, min(aw, ah)) > 14:
        aw, ah = (14, 1) if aw > ah else (1, 14)
    params = [f"--ar {aw}:{ah}"]
    if medium in ("anime", "pixel_art"):
        params.append("--niji 7")
    elif medium == "photo":
        params.append("--raw")
    body_words = set(re.findall(r"[a-z0-9]+", body.lower()))
    exclude: list[str] = []
    conflicts: list[str] = []
    for item in negative:
        text = unescape_parens(item).strip().lower()
        if not text or text in MJ_NO_SKIP or ":" in text or "(" in text:
            continue
        if any(w in MJ_NO_SKIP for w in text.split()):
            continue
        if has_text and text in ("text", "speech bubble"):
            continue
        if set(re.findall(r"[a-z0-9]+", text)) & body_words:
            conflicts.append(text)
            continue
        exclude.append(text)
    exclude = dedup(exclude)
    exclude, overflow = exclude[:MJ_NO_MAX], exclude[MJ_NO_MAX:]
    if exclude:
        params.append("--no " + ", ".join(exclude))
    return " ".join(params), dedup(conflicts), overflow


def compose_prose_fallback(
    extras: dict[str, dict[str, Any]], medium: str, size: tuple[int, int], target: str, tags: TagResult,
    persons: list[dict[str, Any]] | None = None, use_text: bool = False, style_first: bool = False, age: str = "",
) -> str:
    """VLM 無しで自然文系の出力先向けに、振り分けたタグから構造化した英文を組み立てる。
    人物ごとに分けてあれば、被写体は「左の人: ...; 右の人: ...」と人ごとに書く。
    style_first (⑦ 作風・質感を入れたとき) なら、画風を文の最後ではなく、絵か写真かのすぐ後ろ (先頭) に書く。"""
    if target == "midjourney":
        return compose_midjourney(extras, medium, tags, persons, age)
    # 自然文なので、SD 用の人数タグ (1girl / 2girls / solo) は英語の数の表現 (a girl / two girls) にする
    subject_words = []
    for tag in extras["subject"]["tags"]:
        name = norm_tag(tag).replace(" ", "_")
        if name == "solo" or name.startswith("multiple_"):
            continue
        if age and name in AGE_TAG_BANDS:
            continue  # 年齢は人物の呼び方のほうに書く (a little girl ... / a woman in her forties)
        subject_words.append(count_phrase_en(tag, age) if name in PERSON_COUNT_TAGS else tag)
    subject = _plain_prose([w for w in subject_words if w])
    many = bool(persons and len(persons) >= 2)
    if many:
        subject = describe_persons_en(persons, target)
    pose = _plain_prose(extras.get("pose", {}).get("tags", []))
    outfit = _plain_prose(extras.get("outfit", {}).get("tags", []))
    objects = _plain_prose(extras["objects"]["tags"])
    background = _plain_prose(extras["background"]["tags"])
    camera = _plain_prose(extras["camera"]["tags"])
    lighting = _plain_prose(extras["lighting"]["tags"])
    style = _plain_prose(extras["style"]["tags"])
    if use_text and not many:
        # 最大 4 枚モード: VLM が役割別に書いた文 (① の服・ポーズが混ざったものは呼び出し側で消してある) があれば使う
        def text_or(key: str, fallback: str) -> str:
            return (extras.get(key, {}).get("text") or "").strip().rstrip(".") or fallback

        subject, outfit, pose = text_or("subject", subject), text_or("outfit", outfit), text_or("pose", pose)
        background, camera = text_or("background", background), text_or("camera", camera)
        lighting, style = text_or("lighting", lighting), text_or("style", style)
    medium_phrase = MEDIUM_PHRASE.get(medium, "illustration")
    if target == "krea2" and krea_framing(tags):
        medium_phrase = f"{krea_framing(tags)} {medium_phrase}"  # 画角を最初の文に (Krea 2 だけ)
    aspect = aspect_label(size)
    has_text = bool({n for n, _ in tags.general} & {"text", "english_text", "japanese_text", "speech_bubble", "signboard"})

    if target in ("zimageturbo", "zimage", "mossimage"):
        # MOSS-Image も同じ形 (公式の長いキャプションの観点: 構図・被写体・動作・場所・色・照明・画風。縦横比は数値で指定)
        return compose_zimage(extras, medium_phrase, subject, outfit, pose, objects, background,
                              use_text or many, many, style_first)

    if many and target != "grok":
        # 人物ごとの文を主役に置く。並びは各社ガイドの順: GPT Image は 背景 → 人物 → 細部 → 制約、
        # FLUX / Krea / Qwen / Nano Banana は 被写体 (と動作) → 場所 → 構図 → 画風。
        # 場所・照明・構図・画風は VLM の文があればそれを使う (人物は VLM に任せない)
        def sentence(key: str, label: str, tag_text: str) -> str:
            text = extras[key]["text"]
            if text:
                return text if text.endswith(".") else text + "."
            return f"{label}: {tag_text}." if tag_text else ""

        setting = sentence("background", "The setting", background)
        style_text = extras["style"]["text"]
        style_sentence = ((style_text if style_text.endswith(".") else style_text + ".") if style_text
                          else f"Rendered as {_with_article(medium_phrase, capital=False)}" + (f", {style}." if style else "."))
        rest = [
            f"Objects and props: {objects}." if objects else "",
            sentence("camera", "Composition and camera", camera),
            sentence("lighting", "Lighting and color", lighting),
            "" if style_first else style_sentence,
            f"Aspect ratio {aspect}.",
        ]
        if target == "chatgpt":
            head = [f"Create an image: {_with_article(medium_phrase, capital=False)}.", setting, subject]
            tail = [] if has_text else ["No text, watermark or logo."]
        else:
            head, tail = [subject, setting], []
        if style_first:
            head = head[:1] + [style_sentence] + head[1:] if target == "chatgpt" else [style_sentence] + head
        if target == "krea2" and krea_framing(tags):  # 画角を文の頭に (Krea 2 だけ)
            head = [f"{_with_article(medium_phrase)}."] + head
        return " ".join(x for x in head + rest + tail if x)

    if target == "nanobanana":
        parts = [f"{_with_article(medium_phrase)} of {subject or 'the scene'}" + (f", set in {background}" if background else "") + "."]
        if style and style_first:
            parts.append(f"Rendered in this style: {style}.")
        if outfit:
            parts.append(f"Outfit: {outfit}.")
        if pose:
            parts.append(f"Pose: {pose}.")
        if objects:
            parts.append(f"Notable objects: {objects}.")
        if lighting:
            parts.append(f"The scene is illuminated by {lighting}.")
        if camera:
            parts.append(f"Composition and camera: {camera}.")
        if style and not style_first:
            parts.append(f"Rendered in this style: {style}.")
        parts.append(f"The image should be in a {aspect} format.")
        return " ".join(parts)

    if target == "grok":
        pieces = [medium_phrase]
        for p in ((style, subject, outfit, pose, objects, background, lighting, camera) if style_first
                  else (subject, outfit, pose, objects, background, lighting, camera, style)):
            if p:
                pieces.append(p)
        return "; ".join(pieces) + f". Aspect ratio {aspect}."

    parts = []
    if target == "chatgpt":
        parts.append(f"Create an image: {_with_article(medium_phrase, capital=False)}.")
    else:
        parts.append(f"{_with_article(medium_phrase)}.")
    if style and style_first:
        parts.append(f"Style: {style}.")
    if subject:
        parts.append(f"Subject: {subject}.")
    if outfit and not (persons and len(persons) >= 2):  # 人物ごとに書いたときは服・ポーズも各人の側に入っている
        parts.append(f"Outfit: {outfit}.")
    if pose and not (persons and len(persons) >= 2):
        parts.append(f"Pose: {pose}.")
    if objects:
        parts.append(f"Objects and props: {objects}.")
    if background:
        parts.append(f"Setting: {background}.")
    if camera:
        parts.append(f"Composition and camera: {camera}.")
    if lighting:
        parts.append(f"Lighting and color: {lighting}.")
    if style and not style_first:
        parts.append(f"Style: {style}.")
    parts.append(f"Aspect ratio {aspect}.")
    if target == "chatgpt" and not has_text:
        parts.append("No text, watermark or logo.")
    return " ".join(parts)


def compose_zimage(
    extras: dict[str, dict[str, Any]], medium_phrase: str, subject: str, outfit: str, pose: str, objects: str,
    background: str, use_text: bool, many: bool, style_first: bool = False,
) -> str:
    """Z-Image 向けの VLM 無しの英文。ベース版モデルカードの例文と同じく、被写体・服・ポーズ・背景を文章で書き、
    最後に媒体・画風・照明・カメラを句で並べる。縦横比は数値で指定するのでメモに書き、本文には入れない。
    use_text なら VLM が役割別に書いた文を使う (subject / outfit / pose は呼び出し側で置き換え済み)。
    style_first (⑦ 作風・質感を入れたとき) なら、媒体と画風を最後ではなく先頭に書く (最後だと ① の絵柄に負ける)。"""
    def text(key: str) -> str:
        value = (extras.get(key, {}).get("text") or "").strip().rstrip(".") if use_text else ""
        return value + "." if value else ""

    sentences = []
    if style_first:
        lead = _plain_prose(dedup([medium_phrase] + ([] if text("style") else extras.get("style", {}).get("tags", []))))
        sentences += [lead[:1].upper() + lead[1:] + ".", text("style")]
    outfit_text, pose_text = text("outfit"), text("pose")
    if many:
        # 人物ごとの文 (describe_persons_en)。服・ポーズは人ごとの文に入っている
        sentences.append(subject if subject.endswith(".") else subject + ".")
    elif text("subject"):
        sentences.append(text("subject"))
        if outfit:
            sentences.append(outfit_text or f"The subject wears {outfit}.")
        if pose:
            sentences.append(pose_text or f"Pose: {pose}.")
    else:
        body = subject[:1].upper() + subject[1:]
        if outfit and not outfit_text:
            body += f", wearing {outfit}"
        if pose and not pose_text:
            body += f", {pose}"
        body = body.lstrip(", ")
        if body:
            sentences.append(body[:1].upper() + body[1:] + ".")
        sentences += [x for x in (outfit_text, pose_text) if x]
    if objects:
        sentences.append(f"The scene also includes {objects}.")
    setting = text("background") or (f"The background is {background}." if background else "")
    if setting:
        sentences.append(setting)
    # 照明・カメラ・画風は VLM の文があれば文で、無ければ末尾の句に入れる (どちらも落とさない)
    phrases = [] if style_first else [medium_phrase]
    for key in ("lighting", "camera") if style_first else ("style", "lighting", "camera"):
        if text(key):
            sentences.append(text(key))
        else:
            phrases += extras.get(key, {}).get("tags", [])
    tail = _plain_prose(dedup(phrases))
    if tail:
        sentences.append(tail[:1].upper() + tail[1:] + ".")
    return " ".join(x for x in sentences if x)


# --------------------------------------------------------------------------
# MOSS-Image のキャラ・画風カタログ (hide1/MOSS-Image-Character-Catalog)
# --------------------------------------------------------------------------
# MOSS-Image SFT に LoRA なしで名前を指定して生成した 34,249 枚 (キャラ 33,935 + 画風 314) の一覧。作者は合否で選んでいないので、
# 載っていても名前だけで描けるとは限らない (別人の絵も含む)。キャラ名を「名前 from 作品名」と書くのはこのカタログの生成プロンプトの
# 書き方で、MOSS-Image の公式資料には書き方が無い。出力先が MOSS-Image のときは、カタログに載っているキャラを常にこの形で書く。
# カタログのデータと画像は MIT の範囲外 (カタログの NOTICE.md) なので同梱せず、初めて使うときに利用者に確かめてから
# 取得し、名前・作品名・画像の id だけの索引にして models\ に置く。版は固定する (取得し直しても同じ内容)。
MOSS_CATALOG_REV = "43eac1dbf7207617a654c880da1eeeaf2e5935a9"  # Space の版 (2026-10-31、プロンプトの文章整理を含む)
MOSS_CATALOG_IMAGES_REV = "f99f7a6d281e1710dea0975f147a596232d8fbbe"  # 画像を置いた Dataset の版
MOSS_CATALOG_URL = f"{HF_ENDPOINT}/spaces/hide1/MOSS-Image-Character-Catalog/resolve/{MOSS_CATALOG_REV}/catalog.json.gz"
MOSS_CATALOG_PAGE = "https://huggingface.co/spaces/hide1/MOSS-Image-Character-Catalog"
MOSS_CATALOG_PATH = MODELS_DIR / "moss_catalog" / "index.json"
MOSS_CATALOG_MB = 7
MOSS_CATALOG_VERSION = 1  # 索引の形式 (変えたら作り直す)
_moss_catalog_cache: dict[str, Any] = {"key": None, "data": None}


def moss_name_key(name: str) -> str:
    """キャラ名の比べ方: 大文字小文字・下線と空白・括弧のエスケープの違いを無視する (dawn_\\(pokemon\\) = dawn (pokemon))。"""
    return re.sub(r"\s+", " ", str(name or "").replace("\\", "").replace("_", " ")).strip().lower()


def moss_catalog_image_url(entry_id: str) -> str:
    return f"{HF_ENDPOINT}/datasets/hide1/MOSS-Image-Character-Catalog/resolve/{MOSS_CATALOG_IMAGES_REV}/images/{entry_id[:2]}/{entry_id}.webp"


def moss_catalog_ready() -> bool:
    return load_moss_catalog() is not None


def load_moss_catalog() -> dict[str, Any] | None:
    """取得済みの索引を読む (無い・壊れている・形式が古いなら None)。ファイルが変わるまで読み直さない。"""
    try:
        st = MOSS_CATALOG_PATH.stat()
    except OSError:
        return None
    key = (str(MOSS_CATALOG_PATH), st.st_mtime_ns, st.st_size)
    if _moss_catalog_cache["key"] != key:
        try:
            data = json.loads(MOSS_CATALOG_PATH.read_text("utf-8"))
            ok = data.get("format") == MOSS_CATALOG_VERSION and isinstance(data.get("characters"), dict)
        except (OSError, ValueError, AttributeError):
            ok, data = False, None
        _moss_catalog_cache.update(key=key, data=data if ok else None)
    return _moss_catalog_cache["data"]


def build_moss_catalog_index(raw: dict[str, Any]) -> dict[str, Any]:
    """catalog.json → 索引 {characters: {名前の比べ方: [名前, 作品名, id]}, styles: [[名前, プロンプト, id]]}。
    同じ名前が 2 つあれば先のものを使う (v0.1.2 では 1 件)。"""
    chars: dict[str, list[str]] = {}
    styles: list[list[str]] = []
    for e in raw.get("entries") or []:
        if not isinstance(e, dict) or e.get("hidden") or e.get("deleted"):
            continue
        name, eid = str(e.get("name") or "").strip(), str(e.get("id") or "")
        if not name or not re.fullmatch(r"[0-9a-f]{32}", eid):
            continue
        if e.get("kind") == "character":
            chars.setdefault(moss_name_key(name), [name, str(e.get("copyright") or "").strip(), eid])
        elif e.get("kind") == "style" and str(e.get("copy_prompt") or "").strip():
            styles.append([name, str(e["copy_prompt"]).strip(), eid])
    if not chars:
        raise DownloadError("MOSS-Image のキャラ一覧の中身を読み取れませんでした (形式が変わった可能性があります)")
    return {"format": MOSS_CATALOG_VERSION, "rev": MOSS_CATALOG_REV, "source": MOSS_CATALOG_PAGE,
            "characters": chars, "styles": styles}


def download_moss_catalog(progress: DownloadProgress | None = None, cancel: threading.Event | None = None) -> None:
    """カタログ (約 7MB) を取得して索引にする。途中で失敗しても前の索引は残る (書き込みは置き換えで行う)。"""
    import gzip

    gz = MOSS_CATALOG_PATH.with_name("catalog.json.gz")
    download_file(MOSS_CATALOG_URL, gz, progress, label="MOSS-Image のキャラ一覧", cancel=cancel)
    try:
        try:
            raw = json.loads(gzip.decompress(gz.read_bytes()).decode("utf-8"))
        except (OSError, ValueError, EOFError) as e:
            raise DownloadError(f"MOSS-Image のキャラ一覧を展開できませんでした: {e}") from e
        index = build_moss_catalog_index(raw)
        tmp = MOSS_CATALOG_PATH.with_name(MOSS_CATALOG_PATH.name + ".tmp")
        tmp.write_text(json.dumps(index, ensure_ascii=False), "utf-8")
        tmp.replace(MOSS_CATALOG_PATH)
    finally:
        gz.unlink(missing_ok=True)


def moss_catalog_character(name: str) -> dict[str, str] | None:
    """キャラ名 (タガーの名前でも、表記ゆれがあってもよい) → カタログの {name, copyright, id, image}。無ければ None。"""
    data = load_moss_catalog()
    hit = (data or {}).get("characters", {}).get(moss_name_key(name))
    if not hit:
        return None
    return {"name": hit[0], "copyright": hit[1], "id": hit[2], "image": moss_catalog_image_url(hit[2])}


def _name_pattern(name: str) -> str:
    """本文中のキャラ名を探す正規表現: 括弧はエスケープの有無、空白は下線も同じとみなす。前後は語の切れ目。"""
    parts = []
    for ch in name:
        if ch in "()":
            parts.append(r"\\?" + re.escape(ch))
        elif ch in " _":
            parts.append(r"[ _]")
        else:
            parts.append(re.escape(ch))
    return r"(?<![\w])" + "".join(parts) + r"(?![\w])"


def moss_names_with_series(positive: str, characters: list[tuple[str, float]]) -> tuple[str, str]:
    """出力先が MOSS-Image のとき、本文のキャラ名をカタログの書き方「名前 from 作品名」にする (最初に出てくる 1 か所)。
    戻り値: (本文, 分析メモに足す行)。カタログが無い・載っていない・本文に名前が無いときは本文を変えず、理由をメモに書く。"""
    if not characters:
        return positive, ""
    if load_moss_catalog() is None:
        return positive, "\nMOSS-Image のキャラ一覧: 取得していないので、キャラ名に作品名を付けていません"
    lines = []
    for raw, _p in characters:
        shown = sd_tag(raw)
        entry = moss_catalog_character(raw)
        if not entry:
            lines.append(f"{shown} は載っていません")
            continue
        series = entry["copyright"]
        head = f"{shown} は載っています (作品名 {series or 'なし'}。生成例: {entry['image']})"
        m = re.search(_name_pattern(moss_name_key(raw)), positive, re.IGNORECASE)
        if not series:
            lines.append(head)
        elif not m:
            lines.append(head + "。本文に名前が無いので作品名は付けていません")
        elif re.match(r"\s+from\s+" + re.escape(series) + r"(?![\w])", positive[m.end():], re.IGNORECASE):
            lines.append(head + "。本文にすでに作品名があります")
        else:
            positive = positive[:m.end()] + f" from {series}" + positive[m.end():]
            lines.append(head + f"。本文は「{m.group(0)} from {series}」と書きました")
    note = ("\nMOSS-Image のキャラ一覧 (LoRA なしの生成例。選別されていないので、載っていても描けるとは限りません): "
            + " / ".join(lines))
    return positive, note


# --------------------------------------------------------------------------
# 人物検出 (DeepGHS anime_person_detection, MIT) — 「人物はメインの 1 人だけ」用
# --------------------------------------------------------------------------

PERSON_MODEL_URL = f"{HF_ENDPOINT}/deepghs/anime_person_detection/resolve/main/person_detect_v1.3_s/model.onnx"
PERSON_MODEL_PATH = MODELS_DIR / "person" / "person_detect_v1.3_s.onnx"
PERSON_MIN_CONF = 0.324  # モデル付属の threshold.json (F1 が最大になる値)
PERSON_INPUT = 640  # 推論時の長辺 (imgutils と同じ)
PERSON_NMS_IOU = 0.7
# 小さい枠のこの割合以上が大きい枠の中に入っていれば同じ人 (全身と上半身に二重に枠が付いたもの) とみなす。
# NMS の IoU は大小の枠が入れ子だと小さくなり、二重の枠が残って同じ内容の人物ができていた
PERSON_CONTAIN_RATIO = 0.85
PERSON_MARGIN = 0.08  # 切り出すときに枠の外へ広げる割合 (髪や手先が切れないように)

Box = tuple[int, int, int, int]  # (左, 上, 右, 下) のピクセル座標


class PersonDetector:
    """画像の中の人物の位置を検出する (YOLOv8 系、約 44MB。初回のみ取得)。
    アニメ・イラスト向けに学習されたモデルだが、実写の人物にもおおむね反応する。小さいので常に CPU で動かす。"""

    model_url, model_path, min_conf = PERSON_MODEL_URL, PERSON_MODEL_PATH, PERSON_MIN_CONF
    label, label_ja = "anime_person_detection", "人物検出モデル"

    def __init__(self, progress: ProgressFn | None = None, cancel: threading.Event | None = None):
        if not model_file_ok(self.model_path):
            if progress:
                progress(f"{self.label_ja}をダウンロード中 (44MB)... 初回のみ")
            download_file(
                self.model_url, self.model_path, (lambda m, _f: progress(m)) if progress else None,
                label=self.label, cancel=cancel,
            )
        self.session = make_onnx_session(self.model_path, cpu_only=True)
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, image: Image.Image) -> list[tuple[float, Box]]:
        """人物の枠を (信頼度, 枠) の一覧で返す (信頼度の高い順、重なりは NMS で除く)。"""
        rgb = image.convert("RGB")
        w, h = rgb.size
        scale = min(1.0, PERSON_INPUT / max(w, h))
        # 入力は 32 の倍数 (YOLO のストライド)
        nw, nh = (max(32, int(np.ceil(v * scale / 32)) * 32) for v in (w, h))
        x = np.asarray(rgb.resize((nw, nh), Image.BILINEAR), dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        pred = self.session.run(None, {self.input_name: np.ascontiguousarray(x)})[0][0]  # (5, N): cx, cy, w, h, score
        if pred.shape[0] != 5 and pred.shape[1] == 5:  # 出力は (4 座標 + 信頼度, 候補数)。逆向きなら直す
            pred = pred.T
        candidates: list[tuple[float, Box]] = []
        for cx, cy, bw, bh, score in pred[:5].T:
            if score < self.min_conf:
                continue
            box = (
                int(max(0.0, (cx - bw / 2) * w / nw)), int(max(0.0, (cy - bh / 2) * h / nh)),
                int(min(float(w), (cx + bw / 2) * w / nw)), int(min(float(h), (cy + bh / 2) * h / nh)),
            )
            if box[2] > box[0] and box[3] > box[1]:
                candidates.append((float(score), box))
        candidates.sort(key=lambda c: -c[0])
        kept: list[tuple[float, Box]] = []
        for score, box in candidates:
            if all(box_iou(box, other) <= PERSON_NMS_IOU for _, other in kept):
                kept.append((score, box))
        # 入れ子の枠 (同じ人の全身と上半身など) は大きい方だけ残す
        kept.sort(key=lambda c: -box_area(c[1]))
        unique: list[tuple[float, Box]] = []
        for score, box in kept:
            if all(box_containment(box, other) < PERSON_CONTAIN_RATIO for _, other in unique):
                unique.append((score, box))
        return sorted(unique, key=lambda c: -c[0])


# 頭部検出 (DeepGHS anime_head_detection, MIT)。人物検出の F1 0.86 に対して頭部は 0.92 (モデルカード)。
# Danbooru の複数人 180 枚 (2026-10-15) で人数を正しく数えられたのは 人物検出 55% / 頭部検出 71%。
# 人物検出の失敗で一番多いのは 2 人以上を 1 つの枠にまとめるもの (31 枚) → 頭 1 つ = 1 人として枠を分ける
HEAD_MODEL_URL = f"{HF_ENDPOINT}/deepghs/anime_head_detection/resolve/main/head_detect_v2.0_s/model.onnx"
HEAD_MODEL_PATH = MODELS_DIR / "head" / "head_detect_v2.0_s.onnx"
HEAD_MIN_CONF = 0.413  # モデル付属の threshold.json (F1 が最大になる値)
HEAD_MAIN_RATIO = 0.2  # 一番大きい頭の面積に対してこれ未満の頭は使わない (奥に小さく写った人。人物の枠と同じく数えない)
HEAD_IN_BOX = 0.5  # 頭の面積のこの割合以上が人物の枠に入っていれば、その枠の人の頭とみなす


class HeadDetector(PersonDetector):
    """頭の位置を検出する (人物検出と同じ YOLOv8 系、約 44MB。初回のみ取得、常に CPU)。"""

    model_url, model_path, min_conf = HEAD_MODEL_URL, HEAD_MODEL_PATH, HEAD_MIN_CONF
    label, label_ja = "anime_head_detection", "頭部検出モデル"


def refine_person_boxes(
    size: tuple[int, int], persons: list[tuple[float, "Box"]], heads: list[tuple[float, "Box"]],
) -> tuple[list[tuple[float, "Box"]], int, int]:
    """頭の位置で人物の枠を直す。頭 1 つ = 1 人。
    - 2 つ以上の頭が入った枠は、隣り合う頭の中心の中間で縦に分ける (人物検出が 2 人を 1 つにまとめたもの)
    - どの枠にも入らない頭には、頭の大きさから枠を作る (幅は頭の 3 倍、下へ頭 7 つ分)
    - 頭の無い枠 (後ろ姿で頭が隠れる・頭が画面の外など) はそのまま残す
    頭が 1 つも無ければ人物検出のまま。戻り値: (枠, 分けた枠の数, 頭から作った枠の数)。"""
    if not heads:
        return persons, 0, 0
    w, h = size
    largest = max(box_area(b) for _, b in heads)
    heads = [(s, b) for s, b in heads if box_area(b) >= largest * HEAD_MAIN_RATIO]
    boxes = [b for _, b in persons]
    out: list[tuple[float, Box]] = []
    used: set[Box] = set()
    split: set[Box] = set()
    added = 0
    for score, head in heads:
        owners = [b for b in boxes if box_containment(head, b) >= HEAD_IN_BOX]
        if not owners:
            hw, hh = head[2] - head[0], head[3] - head[1]
            cx = (head[0] + head[2]) / 2
            out.append((score, (int(max(0, cx - hw * 1.5)), int(max(0, head[1] - hh * 0.2)),
                                int(min(w, cx + hw * 1.5)), int(min(h, head[1] + hh * 7)))))
            added += 1
            continue
        box = min(owners, key=box_area)
        used.add(box)
        inside = sorted((b for _, b in heads if box_containment(b, box) >= HEAD_IN_BOX), key=lambda b: b[0] + b[2])
        if len(inside) < 2:
            out.append((score, box))
            continue
        split.add(box)
        k = inside.index(head)
        centers = [(b[0] + b[2]) / 2 for b in inside]
        left = box[0] if k == 0 else int((centers[k - 1] + centers[k]) / 2)
        right = box[2] if k == len(inside) - 1 else int((centers[k] + centers[k + 1]) / 2)
        out.append((score, (max(left, box[0]), box[1], min(right, box[2]), box[3])))
    out += [(s, b) for s, b in persons if b not in used and not any(box_containment(hb, b) >= HEAD_IN_BOX for _, hb in heads)]
    return list(dict.fromkeys(out)), len(split), added


def box_area(box: Box) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def box_iou(a: Box, b: Box) -> float:
    inter = box_area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def box_containment(inner: Box, outer: Box) -> float:
    """inner の面積のうち outer の中に入っている割合。"""
    inter = box_area((max(inner[0], outer[0]), max(inner[1], outer[1]), min(inner[2], outer[2]), min(inner[3], outer[3])))
    return inter / box_area(inner) if box_area(inner) else 0.0


def main_person_box(detections: list[tuple[float, Box]]) -> Box | None:
    """一番大きく写っている人物 (枠の面積が最大) の枠。"""
    return max((box for _, box in detections), key=box_area, default=None)


PERSON_SPLIT_MAX = 6  # 人物ごとに分けるのは左から 6 人まで (NovelAI V4 / V4.5 のキャラ欄も 6 人まで)
# NovelAI の出力先だけ 22 人まで (公式: V5 は 22 人。V5 の位置指定の詳しい仕様は公式に無いので書き方は V4 のまま)
PERSON_SPLIT_MAX_NOVELAI = 22
# 一番大きい人の面積に対してこれ未満なら「奥に小さく写った人」。Danbooru の人数タグは
# "A group of unnamed people in the background or a crowd is not counted" なので、人数と人物ごとの区画に入れない
PERSON_BACKGROUND_RATIO = 0.35
PERSON_LEAK_RATIO = 0.75  # 同じタグが複数人に出たとき、最大の信頼度のこの割合未満の人からは外す (隣の人の写り込み)
PERSON_GAZE_TAGS = {"facing_viewer", "facing_away", "profile"}  # カメラ扱いだが人ごとに違うもの
# 画角 (1 枚に 1 つ当てはまる)。Danbooru 322 枚の実測で、しきい値で切らずに一番高いものを選ぶと 83% が正解
FRAMING_TAGS = ("close-up", "portrait", "upper_body", "cowboy_shot", "full_body", "wide_shot")
_FRAMING_ANY = set(FRAMING_TAGS) | {"very_wide_shot", "lower_body", "head_out_of_frame", "feet_out_of_frame"}
# Krea 2 の文の頭に置く画角 (Krea 2 だけ。全身の画像で顔のアップになった。
# Krea 2 公式の prompting.md の例は画角を文の頭に置く: "close-up anime portrait of a young woman" など)
KREA_FRAMING_EN = {
    "full_body": "full-body", "upper_body": "upper-body", "cowboy_shot": "cowboy-shot (mid-thigh up)", "portrait": "portrait",
    "close-up": "close-up", "wide_shot": "wide-shot", "very_wide_shot": "very wide-shot", "lower_body": "lower-body",
}
_FRAMING_WORDS_RE = re.compile(r"\b(full[- ]body|upper[- ]body|cowboy[- ]shot|close[- ]up|portrait|wide[- ]shot|lower[- ]body|"
                               r"head[- ]to[- ]toe|full[- ]length|bust shot|headshot|medium shot|long shot)\b", re.IGNORECASE)


def krea_framing(tags: "TagResult") -> str:
    """画角のタグ (信頼度の高い順で最初のもの) → Krea 2 の文の頭に置く言い方。無ければ空。"""
    return next((KREA_FRAMING_EN[n] for n, _ in tags.general if n in KREA_FRAMING_EN), "")


def krea_lead_with_framing(text: str, framing: str) -> str:
    """VLM の自然文の最初の文に画角が無ければ、頭に画角の文を足す (Krea 2 だけ)。"""
    first = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
    if not framing or _FRAMING_WORDS_RE.search(first):
        return text
    return f"{framing[0].upper()}{framing[1:]} shot. {text.strip()}"


def scene_camera_tag(name: str) -> bool:
    """画像全体の構図として読むカメラのタグ (画角・アングル・効果線など)。視線・顔の向きは人ごとなので含めない。"""
    return classify_extra_tag(name) == "camera" and not name.startswith("looking_") and name not in PERSON_GAZE_TAGS


def ensure_framing(tags: TagResult) -> TagResult:
    """画角のタグが 1 つも無ければ、確信度が一番高い画角を 1 つ足す (人がいない絵には足さない)。"""
    names = {n for n, _ in tags.general}
    if "no_humans" in names or names & _FRAMING_ANY or not tags.general_probs:
        return tags
    best = max(FRAMING_TAGS, key=lambda t: tags.general_probs.get(t, 0.0))
    prob = float(tags.general_probs.get(best, 0.0))
    if prob <= 0.0:
        return tags
    return TagResult(tags.rating, tags.rating_conf, list(tags.general) + [(best, prob)], list(tags.characters),
                     dict(tags.general_probs))


def camera_from_whole(crop: TagResult, whole: TagResult) -> TagResult:
    """切り出した人物のタグのうち、画角・アングル・効果線は切り出す前の画像全体のものに入れ替える
    (切り出すと構図が変わる。複数人の実測で画像全体の方が当たる)。"""
    general = [(n, p) for n, p in crop.general if not scene_camera_tag(n)]
    general += [(n, p) for n, p in whole.general if scene_camera_tag(n)]
    general.sort(key=lambda t: -t[1])
    probs = {n: p for n, p in crop.general_probs.items() if not scene_camera_tag(n)}
    probs.update({n: p for n, p in whole.general_probs.items() if scene_camera_tag(n)})
    return TagResult(crop.rating, crop.rating_conf, general, list(crop.characters), probs)
# 切り出しとグレーの塗りつぶしで出てしまうタグ (元の絵には無いので人物のタグにしない)
CROP_ARTIFACT_TAGS = {
    "border", "grey_border", "gray_border", "white_border", "black_border", "outside_border", "letterboxed",
    "pillarboxed", "grey_background", "gray_background", "simple_background", "cropped", "cropped_legs",
    "cropped_torso", "cropped_arms", "out_of_frame", "head_out_of_frame", "feet_out_of_frame",
}


SKELETON_MIN_POINTS = 4  # 見えている骨格の点がこれ未満の人は、骨格での塗り分けに使わない
SKELETON_CLOSER = 0.7  # 重なった部分で、ほかの人の骨格までの距離が自分の骨格までのこの割合未満なら、ほかの人の部分として塗る
SKELETON_STEP = 4  # 距離の計算は 4 ピクセルおき (速さのため。塗る範囲はあとで元の大きさに戻す)


def skeleton_distance(points: list[list[float]], region: Box, step: int = SKELETON_STEP) -> np.ndarray | None:
    """region の中の各点 (step おき) から、骨格の骨 (見えている点を結んだ線分) までの最短距離。
    見えている点が SKELETON_MIN_POINTS 未満なら None。"""
    visible = [i for i, p in enumerate(points) if p[2] >= POSE_MIN_CONF]
    if len(visible) < SKELETON_MIN_POINTS:
        return None
    xs = np.arange(region[0], region[2], step, dtype=np.float32)
    ys = np.arange(region[1], region[3], step, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    segs = [(a, b) for a, b in POSE_BONES if a in visible and b in visible] or [(i, i) for i in visible]
    best = np.full(gx.shape, np.inf, dtype=np.float32)
    for a, b in segs:
        ax, ay = points[a][0], points[a][1]
        dx, dy = points[b][0] - ax, points[b][1] - ay
        t = np.clip(((gx - ax) * dx + (gy - ay) * dy) / max(dx * dx + dy * dy, 1e-6), 0.0, 1.0)
        best = np.minimum(best, np.hypot(gx - (ax + t * dx), gy - (ay + t * dy)))
    return best


def crop_person(
    image: Image.Image, box: Box, others: list[Box], margin: float = PERSON_MARGIN,
    skeletons: dict[Box, list[list[float]]] | None = None,
) -> Image.Image:
    """人物の枠を少し広げて切り出し、写り込んだほかの人物の部分をグレーで塗る (特徴が別の人に移るのを抑える)。
    自分の枠と重なる部分は、骨格 (skeletons: 枠 → 17 点) があれば、ほかの人の骨格の方がはっきり近いところだけ塗る
    (抱き合う・肩を組むなど枠が重なる場面で、重なった所に写る他人の髪・服を外すため)。骨格が無ければ塗らない。"""
    from PIL import ImageDraw

    w, h = image.size
    mx, my = (box[2] - box[0]) * margin, (box[3] - box[1]) * margin
    region = (int(max(0, box[0] - mx)), int(max(0, box[1] - my)), int(min(w, box[2] + mx)), int(min(h, box[3] + my)))
    crop = image.convert("RGB").crop(region)
    own = crop.crop((box[0] - region[0], box[1] - region[1], box[2] - region[0], box[3] - region[1]))
    draw = ImageDraw.Draw(crop)
    for other in others:
        x0, y0 = max(other[0], region[0]) - region[0], max(other[1], region[1]) - region[1]
        x1, y1 = min(other[2], region[2]) - region[0], min(other[3], region[3]) - region[1]
        if x1 > x0 and y1 > y0:
            draw.rectangle((x0, y0, x1, y1), fill=(128, 128, 128))
    crop.paste(own, (box[0] - region[0], box[1] - region[1]))
    own_points = (skeletons or {}).get(box)
    overlapping = [o for o in others if box_area((max(box[0], o[0]), max(box[1], o[1]), min(box[2], o[2]), min(box[3], o[3]))) > 0]
    if own_points is None or not overlapping:
        return crop
    own_dist = skeleton_distance(own_points, region)
    if own_dist is None:
        return crop
    gray = np.zeros(own_dist.shape, dtype=bool)
    xs = np.arange(region[0], region[2], SKELETON_STEP)[None, :]
    ys = np.arange(region[1], region[3], SKELETON_STEP)[:, None]
    for other in overlapping:
        other_points = skeletons.get(other) if skeletons else None
        other_dist = skeleton_distance(other_points, region) if other_points else None
        if other_dist is None:
            continue
        inside = (xs >= other[0]) & (xs < other[2]) & (ys >= other[1]) & (ys < other[3])
        gray |= inside & (other_dist < own_dist * SKELETON_CLOSER)
    if not gray.any():
        return crop
    mask = Image.fromarray((gray * 255).astype(np.uint8)).resize(
        (gray.shape[1] * SKELETON_STEP, gray.shape[0] * SKELETON_STEP), Image.NEAREST).crop((0, 0, crop.width, crop.height))
    crop.paste((128, 128, 128), mask=mask)
    return crop


# 1 人に 1 つのことが多い特徴 (髪の色・目の色)。2 色が出た人の弱い方が、ほかの人の強い色と同じなら写り込みとして外す。
# その人に多色の髪・オッドアイのタグがあれば外さない
_COLOR_WORDS = {
    "black", "blonde", "brown", "red", "blue", "green", "pink", "purple", "white", "grey", "orange", "aqua", "silver",
    "light_brown", "light_blue", "light_purple", "light_green", "dark_blue", "dark_green", "yellow", "dark_red",
}
COLOR_GROUPS = {
    "hair": {"multicolored_hair", "two-tone_hair", "gradient_hair", "streaked_hair", "colored_inner_hair", "split-color_hair"},
    "eyes": {"heterochromia", "multicolored_eyes"},
}


def color_group(name: str) -> str | None:
    """髪の色 / 目の色のタグなら "hair" / "eyes"。それ以外は None。"""
    for part in COLOR_GROUPS:
        if name.endswith("_" + part) and name[: -len(part) - 1] in _COLOR_WORDS:
            return part
    return None


def minor_color_leaks(crops: list[TagResult]) -> list[set[str]]:
    """人物ごとに、写り込みとみなして外す色のタグ。条件 (すべて満たすとき):
    その人に同じ種類 (髪 / 目) の色が 2 つ以上あり、それが一番強い色ではなく、ほかの誰かの一番強い色で、
    その人の方が確度が高く、その人に多色のタグ (multicolored_hair / heterochromia など) が無い。枠の重なりは問わない。"""
    tops: list[dict[str, tuple[str, float]]] = []
    for t in crops:
        top: dict[str, tuple[str, float]] = {}
        for name, prob in t.general:
            group = color_group(name)
            if group and prob > top.get(group, ("", 0.0))[1]:
                top[group] = (name, prob)
        tops.append(top)
    out: list[set[str]] = []
    for i, t in enumerate(crops):
        names = {n for n, _ in t.general}
        drop: set[str] = set()
        for name, prob in t.general:
            group = color_group(name)
            if group is None or names & COLOR_GROUPS[group] or tops[i][group][0] == name:
                continue
            if any(j != i and top.get(group, ("", 0.0))[0] == name and top[group][1] > prob for j, top in enumerate(tops)):
                drop.add(name)
        out.append(drop)
    return out


def is_person_tag(name: str) -> bool:
    """人物ごとに分けるタグか (外見・服・ポーズ・表情・視線)。人数・背景・照明・画風・構図は画像全体の共有側。"""
    if name in PERSON_COUNT_TAGS or name in ("solo", "no_humans") or name in MULTI_PERSON_TAGS:
        return False
    if name in CROP_ARTIFACT_TAGS or name in MONOCHROME_TAGS:
        return False  # 白黒は画像全体の画風
    extra = classify_extra_tag(name)
    if extra == "camera":
        return name.startswith("looking_") or name in PERSON_GAZE_TAGS
    if extra is not None:
        return False
    return reference_category(name) != "background"


GENDER_MIN_PROB = 0.2  # 1girl / 1boy のどちらかがこれ以上なら、確率の高い方を性別にする


GENDER_JA = {"girl": "女性", "boy": "男性", "other": "その他"}


def genders_from_whole(crops: list["TagResult"], whole: "TagResult") -> list[str] | None:
    """画像全体の人数タグ (例: 1boy, 2girls) の合計が人物の数と同じなら、その男女の数に合わせて性別を割り当てる
    (1boy - 1girl の確率の差が大きい順に男性)。合わなければ None。
    1 人ずつ切り出すと男女の手がかり (並んだ体格の差など) が減るため。Danbooru の複数人 180 枚 (2026-10-16) で
    人数は合うのに性別だけ外れた画像が 19 枚 → 10 枚。"""
    numbers = {"girl": 0, "boy": 0}
    for name, _ in whole.general:
        m = re.fullmatch(r"(\d)(girl|boy)s?", name)
        if m:
            numbers[m.group(2)] = max(numbers[m.group(2)], int(m.group(1)))
    if sum(numbers.values()) != len(crops) or any(n in PERSON_COUNT_TAGS for n, _ in whole.general if "other" in n):
        return None
    order = sorted(range(len(crops)), key=lambda i: -(crops[i].general_probs.get("1boy", 0.0) - crops[i].general_probs.get("1girl", 0.0)))
    out = ["girl"] * len(crops)
    for i in order[: numbers["boy"]]:
        out[i] = "boy"
    return out


def person_gender(tags: TagResult) -> str:
    """切り出した人物の性別 (girl / boy / other)。1 人を表す 1girl / 1boy の確率を比べて決める
    (切り出しに隣の人が少し写ると 2girls などが一番高くなることがあるため)。どちらも低ければ、信頼度がいちばん高い人数タグから。"""
    girl, boy = tags.general_probs.get("1girl", 0.0), tags.general_probs.get("1boy", 0.0)
    if max(girl, boy) >= GENDER_MIN_PROB:
        return "girl" if girl >= boy else "boy"
    counts = [n for n, _ in tags.general if n in PERSON_COUNT_TAGS]
    return next((g for g in ("girl", "boy", "other") if counts and g in counts[0]), "other")


def count_tag_name(group: str, n: int) -> str | None:
    """人数タグの名前 (girl, 3 → 3girls)。Danbooru に無い数は 6+ / multiple にまとめる。0 以下は None。"""
    if n <= 0:
        return None
    name = f"{n}{group}" if n == 1 else f"{n}{group}s"
    if name in PERSON_COUNT_TAGS:
        return name
    return f"6+{group}s" if f"6+{group}s" in PERSON_COUNT_TAGS else f"multiple_{group}s"


def count_tags(genders: list[str], with_multiple: bool = True) -> list[str]:
    """性別ごとの人数から人数タグを作る (例: girl, girl, boy → 2girls, multiple girls, 1boy)。
    Danbooru では 2girls などに multiple_girls が自動で付く (tag implication) ので、学習データに合わせて併記する。
    NovelAI は公式の例が 2girls だけなので with_multiple=False で数だけにする。"""
    tags: list[str] = []
    for group in ("girl", "boy", "other"):
        n = genders.count(group)
        name = count_tag_name(group, n)
        if name is None:
            continue
        tags.append(name)
        if with_multiple and n >= 2 and f"multiple_{group}s" != name:
            tags.append(f"multiple_{group}s")
    return tags


def generation_size(size: tuple[int, int]) -> tuple[int, int]:
    """元画像の縦横比に近い SDXL の生成サイズ (suggest_size と同じ区分)。"""
    w, h = size
    ratio = w / h if h else 1.0
    if ratio < 0.8:
        return 832, 1216
    if ratio > 1.25:
        return 1216, 832
    return 1024, 1024


CLIP_CHUNK_TOKENS = 75  # CLIP のチャンク長。BREAK の区画がこれを超えると次の区画にはみ出し、ほかの人の特徴と混ざる


def clip_tokens(text: str) -> int:
    """CLIP のトークン数の見積もり (語と記号を数え、サブワード分割の分を 2 割ほど足す)。"""
    pieces = re.findall(r"[a-z]+|[0-9]|[^\sa-z0-9]", text.lower())
    return int(np.ceil(len(pieces) * 1.2))


_FEATURE_RE = re.compile(r"(^|_)(hair|eyes|ponytail|twintails|braid|braids|bangs|ahoge|bun|buns|sidelocks|pupils|skin)($|_)")
# 顔立ち・性別らしさのタグ (Danbooru に実在)。同じ顔になるのを防ぐため、髪・目と同じく区画の先頭側に置く
FACE_TAGS = {
    "tsurime", "tareme", "jitome", "sanpaku", "thick_eyebrows", "facial_hair", "beard", "stubble", "mustache",
    "adam's_apple", "mature_male", "mature_female", "manly", "bishounen", "freckles", "mole_under_eye", "eyelashes",
    "long_eyelashes", "muscular_male", "toned_male",
}


def person_tag_priority(tag: str) -> int:
    """人物の区画の中での並び順。その人を見分ける手がかり (髪・目 → 服 → ポーズ・視線 → その他) を先に置く。
    CLIP は区画の前の方ほど強く効くので、削らずに並べ替えるだけで取り違えが減る。"""
    n = norm_tag(tag).replace(" ", "_")
    if _FEATURE_RE.search(n) or n in FACE_TAGS:
        return 1
    if classify_outfit_tag(n):
        return 2
    if classify_pose_tag(n) or n.startswith("looking_") or n in PERSON_GAZE_TAGS:
        return 3
    return 4


_NUMBER_WORDS = ["two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
                 "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "twenty-one", "twenty-two"]
_ORDINAL_WORDS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh",
                  "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth",
                  "twentieth", "twenty-first", "twenty-second"]
ORDINAL_EN = dict(enumerate(_ORDINAL_WORDS, 1))  # NovelAI V5 の 22 人まで
NUMBER_EN = dict(enumerate(_NUMBER_WORDS, 2))
NAI_SUBJECT = {"girl": "girl", "boy": "boy", "other": "other"}  # NovelAI のキャラ欄の先頭 (公式: 数字は付けない)


def person_position(box: Box, size: tuple[int, int], largest: int, index: int = 0, total: int = 1) -> tuple[str, str]:
    """枠の位置を (英語, 日本語) で返す。3 人までは left / center / right、4 人以上は重ならないよう
    「左から n 番目」。小さく写っていれば奥 (background) を付ける。"""
    if total >= 4:
        where, where_ja = f"{ORDINAL_EN.get(index + 1, str(index + 1))} from the left", f"左から {index + 1} 番目"
    else:
        cx = (box[0] + box[2]) / 2 / max(1, size[0])
        where = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "center"
        where_ja = POSITION_JA[where]
    if box_area(box) < largest * PERSON_BACKGROUND_RATIO:
        return f"{where}, in the background", f"{where_ja} (奥)"
    return where, where_ja


def person_area(box: Box, size: tuple[int, int]) -> dict[str, Any]:
    """枠の位置と大きさ。ComfyUI の領域指定ノードの入力に合わせて 2 通りで持つ。
    - x / y / w / h: 画像に対する % (表示用・Regional Prompter の比率の計算用)
    - pct: 0〜1 の比率 (Conditioning (Set Area with Percentage) の x / y / width / height)
    - px: SDXL の生成サイズでのピクセル (Conditioning (Set Area) の入力。8 の倍数、幅と高さは 64 以上)"""
    w, h = max(1, size[0]), max(1, size[1])
    fx, fy, fw, fh = box[0] / w, box[1] / h, (box[2] - box[0]) / w, (box[3] - box[1]) / h
    gw, gh = generation_size(size)

    def px(v: float, total: int, minimum: int = 0) -> int:
        return max(minimum, min(total, int(round(v * total / 8)) * 8))

    return {
        "x": round(fx * 100), "y": round(fy * 100), "w": round(fw * 100), "h": round(fh * 100),
        "pct": {"x": round(fx, 2), "y": round(fy, 2), "width": round(fw, 2), "height": round(fh, 2)},
        "px": {"x": px(fx, gw), "y": px(fy, gh), "width": px(fw, gw, 64), "height": px(fh, gh, 64), "canvas": f"{gw}x{gh}"},
    }


def nai_position(box: Box, size: tuple[int, int]) -> str:
    """NovelAI V4 / V4.5 の Character Positions (Custom、5×5 のマス) で置く場所。
    公式資料にマスの名前 (A1 など) の記載は無いので、左から何列目・上から何行目で表す。"""
    cx = (box[0] + box[2]) / 2 / max(1, size[0])
    cy = (box[1] + box[3]) / 2 / max(1, size[1])
    return f"左から {min(4, int(cx * 5)) + 1} 列目・上から {min(4, int(cy * 5)) + 1} 行目"


def regional_columns(persons: list[dict[str, Any]]) -> list[int] | None:
    """人物が左右に並んでいれば、Regional Prompter の列分割の比率 (合計 100 前後) を返す。
    境目は隣り合う人物の中心の中間。人物どうしが大きく重なって並んでいないなら None。"""
    areas = [p.get("area") for p in persons]
    if len(persons) < 2 or not all(areas):
        return None
    centers = [a["x"] + a["w"] / 2 for a in areas]
    if any(b - a < 5 for a, b in zip(centers, centers[1:], strict=False)):
        return None  # 中心が近すぎる (前後に重なっている) と列では分けられない
    edges = [0.0] + [(a + b) / 2 for a, b in zip(centers, centers[1:], strict=False)] + [100.0]
    return [max(1, round(b - a)) for a, b in zip(edges, edges[1:], strict=False)]


def crop_box(image: Image.Image, box: Box, margin: float = PERSON_MARGIN) -> Image.Image:
    """枠を少し広げて切り出す。"""
    w, h = image.size
    mx, my = (box[2] - box[0]) * margin, (box[3] - box[1]) * margin
    return image.crop((
        int(max(0, box[0] - mx)), int(max(0, box[1] - my)), int(min(w, box[2] + mx)), int(min(h, box[3] + my)),
    ))


# --------------------------------------------------------------------------
# 年代スタイル分類 (DeepGHS anime_style_ages, openrail)
# --------------------------------------------------------------------------
# 骨格 (ポーズ編集) — RTMPose-s (OpenMMLab mmpose 公式の ONNX、Apache-2.0)
# 必要スペックを上げない (CPU、LLM 不要)。モデルは「画像から骨格を取る」を押したときだけ確認の上で取得。
# 実測 (Ryzen / CPU 1 スレッド): 1 人 16ms、モデル 21MB。全身が見える絵はアニメでも正確、隠れた下半身は崩れる (手で直す)
# --------------------------------------------------------------------------
POSE_MODEL_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip"
)
POSE_MODEL_PATH = MODELS_DIR / "pose" / "rtmpose-s_body7_256x192.onnx"
POSE_MODEL_MB = 21
POSE_INPUT_W, POSE_INPUT_H = 192, 256  # pipeline.json の TopDownAffine image_size
POSE_MEAN = np.array([123.675, 116.28, 103.53], np.float32)  # pipeline.json の Normalize
POSE_STD = np.array([58.395, 57.12, 57.375], np.float32)
POSE_MIN_CONF = 0.3  # これ未満の点は「見えていない」扱い (タグにも OpenPose 画像にも使わない)
# COCO 17 点 (RTMPose body7 の出力の並び)
KEYPOINTS = [
    ("nose", "鼻"), ("left_eye", "左目"), ("right_eye", "右目"), ("left_ear", "左耳"), ("right_ear", "右耳"),
    ("left_shoulder", "左肩"), ("right_shoulder", "右肩"), ("left_elbow", "左ひじ"), ("right_elbow", "右ひじ"),
    ("left_wrist", "左手首"), ("right_wrist", "右手首"), ("left_hip", "左腰"), ("right_hip", "右腰"),
    ("left_knee", "左ひざ"), ("right_knee", "右ひざ"), ("left_ankle", "左足首"), ("right_ankle", "右足首"),
]
POSE_BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
              (0, 1), (0, 2), (1, 3), (2, 4)]
# 立ち姿の雛形 (幅 600 × 高さ 900 のキャンバス、人物の左右 = 画像の右左)
POSE_TEMPLATE_SIZE = (600, 900)
POSE_TEMPLATE = [
    (300, 130), (315, 115), (285, 115), (335, 125), (265, 125), (370, 210), (230, 210), (390, 330), (210, 330),
    (410, 470), (190, 470), (330, 450), (270, 450), (350, 630), (250, 630), (355, 810), (245, 810),
]


def pose_model_ready() -> bool:
    return model_file_ok(POSE_MODEL_PATH)


def face_models_ready() -> bool:
    """顔の切り出し (人物の特徴を詳しく) に使う骨格モデルと人物検出モデルがそろっているか。"""
    return pose_model_ready() and model_file_ok(PERSON_MODEL_PATH)


FACE_MODELS_JA = [f"骨格モデル RTMPose-s (約 {POSE_MODEL_MB}MB、OpenMMLab、Apache-2.0)", "人物検出モデル (約 44MB、DeepGHS、MIT)"]


def missing_face_models() -> list[str]:
    """顔の切り出しに足りないモデル (確認ダイアログに出す名前)。"""
    return [name for name, ok in zip(FACE_MODELS_JA, (pose_model_ready(), model_file_ok(PERSON_MODEL_PATH)), strict=True) if not ok]


def ensure_face_models(progress: ProgressFn | None = None, cancel: threading.Event | None = None) -> None:
    """顔の切り出しに使うモデルを取得する (無いものだけ)。呼ぶ側が利用者の同意を得てから呼ぶ。"""
    if not pose_model_ready():
        PoseEstimator(progress, cancel)
    if not model_file_ok(PERSON_MODEL_PATH):
        PersonDetector(progress, cancel)


class PoseEstimator:
    """人物の骨格 (17 点) を推定する。人物の枠は PersonDetector で探し、いちばん大きい人を使う。常に CPU。"""

    def __init__(self, progress: ProgressFn | None = None, cancel: threading.Event | None = None):
        if not pose_model_ready():
            import zipfile

            if progress:
                progress(f"骨格モデルをダウンロード中 ({POSE_MODEL_MB}MB)... 初回のみ")
            archive = POSE_MODEL_PATH.with_suffix(".zip")
            download_file(POSE_MODEL_URL, archive, (lambda m, _f: progress(m)) if progress else None,
                          label="RTMPose-s", cancel=cancel)
            try:
                with zipfile.ZipFile(archive) as zf:
                    member = next(n for n in zf.namelist() if n.endswith("end2end.onnx"))
                    tmp = POSE_MODEL_PATH.with_name(POSE_MODEL_PATH.name + ".part")
                    tmp.write_bytes(zf.read(member))
                    tmp.replace(POSE_MODEL_PATH)
            except (OSError, StopIteration, zipfile.BadZipFile) as e:
                raise DownloadError(f"骨格モデルの展開に失敗しました: {e}") from e
            finally:
                archive.unlink(missing_ok=True)
        self.session = make_onnx_session(POSE_MODEL_PATH, cpu_only=True)

    def estimate(self, image: Image.Image, box: Box | None = None) -> list[list[float]]:
        """骨格の 17 点を [x, y, 信頼度] (元画像の座標) で返す。box は人物の枠 (無ければ画像全体)。"""
        rgb = image.convert("RGB")
        x0, y0, x1, y1 = box or (0, 0, rgb.width, rgb.height)
        # TopDownGetBboxCenterScale (padding 1.25) → 192:256 に合わせて切り出す
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        bw, bh = (x1 - x0) * 1.25, (y1 - y0) * 1.25
        if bw / bh > POSE_INPUT_W / POSE_INPUT_H:
            bh = bw * POSE_INPUT_H / POSE_INPUT_W
        else:
            bw = bh * POSE_INPUT_W / POSE_INPUT_H
        left, top, sx, sy = cx - bw / 2, cy - bh / 2, bw / POSE_INPUT_W, bh / POSE_INPUT_H
        patch = rgb.transform((POSE_INPUT_W, POSE_INPUT_H), Image.AFFINE, (sx, 0, left, 0, sy, top), resample=Image.BILINEAR)
        x = ((np.asarray(patch, np.float32) - POSE_MEAN) / POSE_STD).transpose(2, 0, 1)[None]
        simcc_x, simcc_y = self.session.run(None, {self.session.get_inputs()[0].name: np.ascontiguousarray(x)})
        xs, ys = simcc_x[0].argmax(1) / 2.0, simcc_y[0].argmax(1) / 2.0  # SimCC の分割比 2
        conf = np.minimum(simcc_x[0].max(1), simcc_y[0].max(1))
        return [[float(left + px * sx), float(top + py * sy), float(c)] for px, py, c in zip(xs, ys, conf)]


def _pt(points: list[list[float]], i: int) -> tuple[float, float] | None:
    p = points[i]
    return (p[0], p[1]) if len(p) < 3 or p[2] >= POSE_MIN_CONF else None


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _angle(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    """b を頂点とする角度 (度)。"""
    v1, v2 = np.subtract(a, b), np.subtract(c, b)
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


FACE_MIN_SIDE = 32  # 顔の切り出しがこれより小さければ (遠景の人など) 切り出さない (拡大しても細部は増えない)
FACE_VLM_SIDE = 512  # VLM に渡す顔の長辺 (これより小さい切り出しは拡大する)


def face_box(points: list[list[float]], size: tuple[int, int]) -> Box | None:
    """骨格の頭の点 (鼻・目・耳) から顔のまわり (髪を含む正方形) の枠を返す。頭の点が 2 つ未満なら None。
    顔の幅は 耳の間 ≒ 顔の幅、目の間 × 2.5、肩幅 × 0.5 のうち大きい方 (横顔で目の間が縮んでも顔を切らないように)。"""
    head = [pt for pt in (_pt(points, i) for i in range(5)) if pt]
    if len(head) < 2:
        return None
    widths = []
    for a, b, scale in ((3, 4, 1.0), (1, 2, 2.5), (5, 6, 0.5)):
        pa, pb = _pt(points, a), _pt(points, b)
        if pa and pb:
            widths.append(_dist(pa, pb) * scale)
    if not widths:
        return None
    side = max(widths) * 2.0  # 顔の幅の 2 倍 (髪・あご・耳まで入る)
    cx = sum(p[0] for p in head) / len(head)
    cy = sum(p[1] for p in head) / len(head)
    w, h = size
    box = (int(max(0, cx - side / 2)), int(max(0, cy - side / 2)), int(min(w, cx + side / 2)), int(min(h, cy + side / 2)))
    if min(box[2] - box[0], box[3] - box[1]) < FACE_MIN_SIDE:
        return None
    return box


def pose_tags_from_points(points: list[list[float]]) -> list[str]:
    """骨格の 17 点から Danbooru のポーズのタグを規則で作る (画像の座標、y は下向き)。
    タグ名はタガーの語彙 (wd-eva02-large-tagger-v3 の selected_tags.csv) にあるものだけ。
    2D の骨格なので向き (正面 / 後ろ) や座っている物は分からない (向きはカメラの指定で決める)。見えていない点は使わない。"""
    p = {i: _pt(points, i) for i in range(17)}
    ls, rs, lh, rh = p[5], p[6], p[11], p[12]
    if not (ls and rs):
        return []
    mid_s = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    mid_h = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2) if lh and rh else None
    torso = _dist(mid_s, mid_h) if mid_h else _dist(ls, rs) * 1.6
    torso = max(torso, 1.0)
    heads = [p[i] for i in (0, 1, 2, 3, 4) if p[i]]
    head = (float(np.mean([h[0] for h in heads])), float(np.mean([h[1] for h in heads]))) if heads else None
    head_top = (min(h[1] for h in heads) - 0.25 * torso) if heads else mid_s[1] - 0.6 * torso
    tags: list[str] = []

    def both(flags: list[bool], two: str, one: str) -> None:
        if all(flags):
            tags.append(two)
        elif any(flags):
            tags.append(one)

    # 腕
    up, on_head, on_hip, out, face, sides = [], [], [], [], [], []
    for sh, el, wr, hip in ((ls, p[7], p[9], lh), (rs, p[8], p[10], rh)):
        if not wr:
            for flags in (up, on_head, on_hip, out, face, sides):
                flags.append(False)
            continue
        is_up = wr[1] < head_top + 0.1 * torso
        near_head = bool(head) and _dist(wr, head) < 0.45 * torso and wr[1] < sh[1]
        up.append(is_up and not near_head)
        on_head.append(near_head and wr[1] < head[1] if head else False)
        face.append(near_head and not (wr[1] < head[1] if head else False))
        straight = bool(el) and _angle(sh, el, wr) > 150
        # 腰に手はひじが曲がる (腕を下ろしただけ・ポケットに手は手首が腰の近くでもひじが伸びている)
        on_hip.append(bool(hip) and bool(el) and _dist(wr, hip) < 0.3 * torso and _angle(sh, el, wr) < 135)
        sides.append(straight and bool(hip) and wr[1] > hip[1] - 0.15 * torso and abs(wr[0] - sh[0]) < 0.45 * torso)
        out.append(straight and abs(wr[1] - sh[1]) < 0.35 * torso and abs(wr[0] - sh[0]) > 0.8 * torso)
    both(up, "arms_up", "arm_up")
    both(on_head, "hands_on_own_head", "hand_on_own_head")
    both(on_hip, "hands_on_own_hips", "hand_on_own_hip")
    both(out, "outstretched_arms", "outstretched_arm")
    if any(face):
        tags.append("hand_on_own_face")
    if all(sides):
        tags.append("arms_at_sides")
    lw, rw = p[9], p[10]
    if lw and rw and mid_h:
        # 左手首が右肩の側、右手首が左肩の側にあり、胸の高さ → 腕組み
        l_side, r_side = np.sign(ls[0] - mid_s[0]), np.sign(rs[0] - mid_s[0])
        crossed = np.sign(lw[0] - mid_s[0]) == r_side and np.sign(rw[0] - mid_s[0]) == l_side
        chest = all(mid_s[1] - 0.1 * torso < w[1] < mid_h[1] for w in (lw, rw))
        if crossed and chest and all(abs(w[0] - mid_s[0]) < 0.8 * torso for w in (lw, rw)):
            tags.append("crossed_arms")
    # 脚
    lk, rk, la, ra = p[13], p[14], p[15], p[16]
    if lh and rh and lk and rk and la and ra:
        hip_w = max(_dist(lh, rh), 0.15 * torso)
        ang_l, ang_r = _angle(lh, lk, la), _angle(rh, rk, ra)
        hip_y = mid_h[1]
        if min(ang_l, ang_r) < 110 and max(lk[1], rk[1]) < hip_y + 0.35 * torso and min(la[1], ra[1]) > hip_y:
            tags.append("squatting")
        elif all(abs(k[1] - a[1]) < 0.25 * torso for k, a in ((lk, la), (rk, ra))) and max(lk[1], rk[1]) > hip_y + 0.4 * torso:
            tags.append("kneeling")
        elif any(abs(k[1] - a[1]) < 0.2 * torso for k, a in ((lk, la), (rk, ra))) and min(ang_l, ang_r) < 120:
            tags.append("on_one_knee")
        elif min(ang_l, ang_r) > 150 and min(la[1], ra[1]) - hip_y > 0.8 * torso:
            tags.append("standing")
        if any(k[1] < hip_y - 0.05 * torso for k in (lk, rk)) and max(la[1], ra[1]) > hip_y + 0.8 * torso:
            tags.append("leg_up")
        frontal = _dist(lh, rh) > 0.2 * torso  # 横向きだと腰が重なって左右の比較ができない
        ankle_gap = abs(la[0] - ra[0])
        if frontal and "squatting" not in tags:
            if (la[0] - ra[0]) * (lh[0] - rh[0]) < 0 and abs(la[1] - ra[1]) < 0.4 * torso:
                tags.append("crossed_legs")
            elif ankle_gap > 2.2 * hip_w:
                tags.append("legs_apart")
            elif ankle_gap < 0.6 * hip_w:
                tags.append("legs_together")
    # 頭の傾き・上半身の傾き
    le, re_ = p[1], p[2]
    if le and re_ and _dist(le, re_) > 0.05 * torso:
        tilt = abs(np.degrees(np.arctan2(le[1] - re_[1], abs(le[0] - re_[0]) + 1e-6)))
        if tilt > 12:
            tags.append("head_tilt")
    if mid_h and _dist(ls, rs) > 0.3 * torso:
        lean = abs(np.degrees(np.arctan2(mid_s[0] - mid_h[0], mid_h[1] - mid_s[1])))
        if lean > 18:
            tags.append("leaning_to_the_side")
    return dedup(tags)


# OpenPose の 18 点・骨・色 (ControlNet 公式 annotator/openpose/util.py の draw_bodypose と同じ。stickwidth 4、骨は 0.6 で重ねる)
_OPENPOSE_FROM_COCO = [0, None, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]  # None = 首 (両肩の中点)
_OPENPOSE_LIMBS = [(2, 3), (2, 6), (3, 4), (4, 5), (6, 7), (7, 8), (2, 9), (9, 10), (10, 11), (2, 12), (12, 13), (13, 14),
                   (2, 1), (1, 15), (15, 17), (1, 16), (16, 18)]  # 1 始まり (公式の limbSeq の先頭 17 本)
_OPENPOSE_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
    (255, 0, 170), (255, 0, 85),
]


def render_openpose(points: list[list[float]], size: tuple[int, int]) -> Image.Image:
    """骨格を ControlNet (OpenPose) 用の画像にする (黒地、公式の色と太さ)。見えていない点と、それにつながる骨は描かない。"""
    from PIL import ImageDraw

    w, h = size
    coco = [_pt(points, i) for i in range(17)]
    neck = ((coco[5][0] + coco[6][0]) / 2, (coco[5][1] + coco[6][1]) / 2) if coco[5] and coco[6] else None
    op = [neck if src is None else coco[src] for src in _OPENPOSE_FROM_COCO]
    canvas = np.zeros((h, w, 3), np.uint8)
    base = Image.fromarray(canvas)
    draw = ImageDraw.Draw(base)
    for i, pt in enumerate(op):
        if pt:
            draw.ellipse([pt[0] - 4, pt[1] - 4, pt[0] + 4, pt[1] + 4], fill=_OPENPOSE_COLORS[i])
    arr = np.asarray(base, np.float32)
    for i, (a, b) in enumerate(_OPENPOSE_LIMBS):
        pa, pb = op[a - 1], op[b - 1]
        if not (pa and pb):
            continue
        layer = Image.fromarray(arr.astype(np.uint8))
        length = _dist(pa, pb)
        angle = np.arctan2(pb[1] - pa[1], pb[0] - pa[0])
        mx, my = (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2
        t = np.linspace(0, 2 * np.pi, 36)
        ex, ey = length / 2 * np.cos(t), 4 * np.sin(t)
        poly = [(mx + x * np.cos(angle) - y * np.sin(angle), my + x * np.sin(angle) + y * np.cos(angle)) for x, y in zip(ex, ey)]
        ImageDraw.Draw(layer).polygon(poly, fill=_OPENPOSE_COLORS[i])
        arr = arr * 0.4 + np.asarray(layer, np.float32) * 0.6
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))



# --------------------------------------------------------------------------

ERA_MODEL_URL = f"{HF_ENDPOINT}/deepghs/anime_style_ages/resolve/main/mobilenetv3_v0_dist/model.onnx"
ERA_MODEL_PATH = MODELS_DIR / "style" / "anime_style_ages_mobilenetv3.onnx"
ERA_LABELS = ["1970s-", "1980s", "1990s", "2000s", "2010s", "2015s", "2020s"]
ERA_INPUT = 384
ERA_MIN_CONF = 0.35  # これ未満はタグにせず分析メモだけに出す
ERA_TAGS = {  # 年代 → 画風タグ (Danbooru の年代スタイル + NoobAI / NovelAI が学習している year タグ)
    "1970s-": ["1970s (style)", "retro artstyle"],
    "1980s": ["1980s (style)", "retro artstyle"],
    "1990s": ["1990s (style)", "retro artstyle"],
    "2000s": ["2000s (style)", "year 2005"],
    "2010s": ["2010s (style)", "year 2013"],
    "2015s": ["2010s (style)", "year 2017"],
    "2020s": ["year 2023"],
}
ERA_JA = {
    "1970s-": "1970 年代以前", "1980s": "1980 年代", "1990s": "1990 年代", "2000s": "2000 年代",
    "2010s": "2010 年代前半", "2015s": "2010 年代後半", "2020s": "2020 年代",
}


class EraClassifier:
    """アニメ・イラストの年代スタイルを推定する (mobilenetv3、約 17MB。初回のみ取得)。"""

    def __init__(self, progress: ProgressFn | None = None, cancel: threading.Event | None = None, cpu_only: bool = False):
        if not model_file_ok(ERA_MODEL_PATH):
            if progress:
                progress("年代スタイル分類モデルをダウンロード中 (17MB)... 初回のみ")
            download_file(
                ERA_MODEL_URL, ERA_MODEL_PATH, (lambda m, _f: progress(m)) if progress else None,
                label="anime_style_ages", cancel=cancel,
            )
        self.session = make_onnx_session(ERA_MODEL_PATH, cpu_only)
        self.cpu_only = cpu_only
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image: Image.Image) -> dict[str, Any]:
        rgb = image.convert("RGB").resize((ERA_INPUT, ERA_INPUT), Image.BILINEAR)
        arr = (np.asarray(rgb, dtype=np.float32) / 255.0 - 0.5) / 0.5  # imgutils と同じ正規化
        tensor = np.ascontiguousarray(arr.transpose(2, 0, 1)[None])
        out = np.squeeze(self.session.run(None, {self.input_name: tensor})[0], axis=0).astype(np.float64)
        if out.min() >= 0 and abs(out.sum() - 1.0) < 1e-3:
            probs = out  # このモデルは確率を出力する (softmax を重ねると最大でも約 0.31 に潰れ、しきい値に届かない)
        else:
            probs = np.exp(out - out.max())
            probs /= probs.sum()
        order = probs.argsort()[::-1]
        top = ERA_LABELS[int(order[0])]
        return {
            "label": top,
            "label_ja": ERA_JA[top],
            "conf": round(float(probs[order[0]]), 3),
            "tags": list(ERA_TAGS[top]) if probs[order[0]] >= ERA_MIN_CONF else [],
            "ranking": [(ERA_LABELS[int(i)], round(float(probs[i]), 3)) for i in order[:3]],
        }


# --------------------------------------------------------------------------
# ルールベース: 画風判定・ネガティブ・分析メモ
# --------------------------------------------------------------------------

MEDIUM_JA = {
    "anime": "アニメ・イラスト調",
    "photo": "写真・実写調",
    "3d": "3DCG",
    "pixel_art": "ドット絵",
    "painting": "絵画・アナログ画材風",
}
RATING_JA = {"general": "全年齢", "sensitive": "やや際どい", "questionable": "きわどい", "explicit": "成人向け"}

QUALITY_TAGS = {
    "anime": ["masterpiece", "best quality", "very aesthetic", "absurdres"],
    "photo": ["best quality", "ultra detailed", "photorealistic", "8k uhd", "sharp focus"],
    "3d": ["best quality", "highly detailed", "sharp focus"],
    "pixel_art": ["best quality", "pixel art"],
    "painting": ["masterpiece", "best quality", "highly detailed"],
}

BASE_NEGATIVES = {
    "anime": [
        "lowres", "bad anatomy", "bad hands", "text", "error", "extra digits",
        "fewer digits", "cropped", "worst quality", "low quality", "jpeg artifacts", "signature",
        "watermark", "username", "blurry", "realistic", "photorealistic", "3d",
    ],
    "photo": [
        "cartoon", "anime", "illustration", "painting", "drawing", "3d render", "cgi", "deformed",
        "disfigured", "bad anatomy", "extra limbs", "bad hands", "blurry", "lowres", "jpeg artifacts",
        "watermark", "text", "signature",
    ],
    "3d": [
        "2d", "flat color", "sketch", "lineart", "lowres", "bad anatomy", "bad hands", "deformed",
        "blurry", "jpeg artifacts", "text", "watermark", "signature",
    ],
    "pixel_art": [
        "blurry", "smooth shading", "photorealistic", "3d render", "lowres", "jpeg artifacts",
        "text", "watermark",
    ],
    "painting": [
        "photorealistic", "3d render", "anime screencap", "flat color", "lowres", "bad anatomy",
        "bad hands", "blurry", "jpeg artifacts", "text", "watermark", "signature",
    ],
}

MODEL_FAMILY_JA = {
    "anime": "Illustrious / NoobAI / Pony 系の SDXL モデル",
    "photo": "実写系 SDXL (Juggernaut, RealVis など) または Flux",
    "3d": "3D 系の SDXL チェックポイント",
    "pixel_art": "ピクセルアート LoRA を併用した SDXL / SD1.5",
    "painting": "絵画系の SDXL モデルまたは Flux",
}

# --- Anima (circlestone-labs/Anima) 公式の推奨 ---
ANIMA_QUALITY = ["masterpiece", "best quality", "score_7"]
ANIMA_NEGATIVE = [
    "worst quality", "low quality", "score_1", "score_2", "score_3", "artist name",
    "blurry", "jpeg artifacts", "chromatic aberration",
]
ANIMA_RATING = {"general": "safe", "sensitive": "sensitive", "questionable": "nsfw", "explicit": "explicit"}

# --- NovelAI Diffusion V4.5 Full 公式の推奨 (Add Quality Tags / Undesired Content: Heavy) ---
NAI_QUALITY = ["location", "very aesthetic", "masterpiece", "no text"]
NAI_NEGATIVE = [
    "lowres", "artistic error", "film grain", "scan artifacts", "worst quality", "bad quality",
    "jpeg artifacts", "very displeasing", "chromatic aberration", "dithering", "halftone", "screentone",
    "multiple views", "logo", "too many watermarks", "negative space", "blank page",
]


def unescape_parens(text: str) -> str:
    """NovelAI は () を特別扱いしないので、SD 向けのエスケープを外す。"""
    return text.replace("\\(", "(").replace("\\)", ")")
# Anima の並び順で先頭に置く人数タグ
COUNT_TAGS = (
    "1girl", "2girls", "3girls", "4girls", "5girls", "6+girls", "multiple_girls",
    "1boy", "2boys", "3boys", "4boys", "5boys", "6+boys", "multiple_boys",
    "1other", "2others", "3others", "multiple_others",
)

# VLM に渡す出力先ごとの natural_prompt の書き方
# Z-Image (Turbo / ベース共通)。公式のプロンプト拡張 (Space の pe.py) の決まり: 細部まで客観的・具象的に描写し、
# 画像内の文字は英語の二重引用符で一字一句そのまま、比喩・感情的な修辞・8K や傑作などのメタタグ・描画の指示は書かない。
# 長さは公式に数字が無い (公式の担当者: 長く詳しいプロンプトが最良、diffusers の既定は 512 トークンで超えると切れる) ので
# 150〜300 語、形はベース版モデルカードの例文に合わせて「内容は文章 → 末尾に媒体・照明・色調・被写界深度の句」
ZIMAGE_VLM_GUIDANCE = (
    "natural_prompt: a detailed, objective English description of 150-300 words for Z-Image. First describe the "
    "content in complete sentences: the subjects (count, appearance, hair, face, expression), their garments and "
    "materials, pose and action, objects, then the background and setting, keeping every visual detail of the image. "
    "Then end with one short comma-separated list of phrases for the medium and style, lighting, colour palette, "
    "camera framing and depth of field (for example 'Photograph, natural light, soft shadows, neutral navy and cream "
    "palette, medium depth of field.'). Transcribe every legible text in the image verbatim inside English double "
    "quotes and give its position, size and material. Be objective and concrete: no metaphors, no figurative or "
    "emotional rhetoric, no meta tags or quality words such as 8K or masterpiece, and no drawing instructions."
)
# MOSS-Image。学習用の長いキャプションの指示 (USERGUIDE.md: 構図・主な被写体・動作・場所・色・照明・画風を詳しく、
# 文字は訳さず一字一句) と公式の Prompt Enhancer (skills/moss-image-prompt-enhancer/SKILL.md: ひとまとまりの文章、位置関係は言葉で、
# 品質タグの重ね付け・決まった語数・処理の説明は書かない、英語が既定)。長さは公式が語数を決めないよう言うので数字を入れない
MOSS_VLM_GUIDANCE = (
    "natural_prompt: a detailed English caption for MOSS-Image written as cohesive prose (one or more paragraphs). "
    "Describe the composition, the key subjects (count, appearance, hair, face, expression), their garments, actions "
    "and pose, objects, the setting, colours, lighting, camera framing and the artistic style or medium in detail, "
    "keeping every visual detail of the image. State spatial relations in words (left, right, foreground, background). "
    "Transcribe every legible text in the image verbatim inside double quotes, in its original language, without "
    "translating it. No tag lists, no weights, no quality words such as masterpiece or 8k, and no notes about the process."
)
TARGET_VLM_GUIDANCE = {
    "sdxl": (
        "natural_prompt: one short English sentence summarising the scene (it is only shown as reference)."
    ),
    "novelai": (
        "natural_prompt: one short English sentence summarising the scene (it is only shown as reference)."
    ),
    "anima": (
        "natural_prompt: two or three fluent English sentences that will be appended AFTER a Danbooru tag list "
        "for the Anima model. Describe the scene, mood, composition and art style; do not repeat the tags "
        "one by one, do not use tag syntax, no quality words."
    ),
    # BFL の FLUX.2 ガイド: 30〜80 語が標準、前に置いたものほど重視 (被写体 → 動作 → 画風 → 状況 → 細部)、
    # キーワードの羅列でなく描写する文、ネガティブ非対応なので欲しいものを書く
    "flux2": (
        "natural_prompt: one descriptive English paragraph of 30-80 words for FLUX.2 (up to about 120 words only for "
        "complex scenes). Word order matters: main subject first, then the key action, the critical style, the essential "
        "context, and secondary details last. Descriptive prose with concrete visual language (materials, camera and lens, "
        "light), not keyword lists. No quality words such as masterpiece or 8k. Describe only what should be in the image; "
        "never write what should be absent."
    ),
    # klein は prompt upsampling が無いので詳しく書く (docs.bfl.ai/flux_2)
    "flux2klein": (
        "natural_prompt: one detailed, descriptive English paragraph of 50-80 words for FLUX.2 [klein], which does not "
        "expand short prompts, so spell out every important visual detail. Word order matters: main subject first, then "
        "the key action, the critical style, the essential context, and secondary details last. Descriptive prose, not "
        "keyword lists. No quality words. Describe only what should be in the image; never write what should be absent."
    ),
    "flux": (
        "natural_prompt: one fluent English paragraph of 60-120 words for Flux / SD3. Subject first, then "
        "setting, lighting, colour, composition, and finally medium and style. No lists, no tag syntax, "
        "no quality words such as masterpiece or 8k."
    ),
    "krea2": (
        "natural_prompt: a vivid, flowing English description of 60-150 words for Krea 2, written as you "
        "would brief a skilled photographer or illustrator. Use specific visual language: materials, "
        "textures, camera and lens terms (close-up, wide angle, shallow depth of field), lighting and mood. "
        "Name the medium descriptively (e.g. 'photograph on 35mm film', '1990s cel animation anime style', "
        "'flat 2d vector illustration'). Begin the first sentence with the shot type / framing seen in the image "
        "(e.g. 'A full-body photograph of ...', 'An upper-body portrait of ...'). "
        "No tag lists, no weights, no quality boosters, no negative wording."
    ),
    "qwen": (
        "natural_prompt: a detailed English description of 200-350 words (10-16 sentences) for Qwen-Image 2.1. "
        "Open with one sentence naming the medium, style and main subject. Then walk the frame: background "
        "first, then the subject details from top to bottom (build, posture, expression, garments, materials). "
        "Describe every legible text in the image in double quotes with its exact characters, placement, "
        "colour and weight; use 'indistinct' for unreadable text. Describe the lighting in its own sentence, "
        "then close with the overall composition and framing. Use colour modifiers (deep navy, soft dusty "
        "rose), name materials (brushed metal, matte plastic). Avoid quality boosters (masterpiece, 8k, "
        "highly detailed), instructional language, generic summaries and numerical ages."
    ),
    "zimageturbo": ZIMAGE_VLM_GUIDANCE,
    "zimage": ZIMAGE_VLM_GUIDANCE,
    "mossimage": MOSS_VLM_GUIDANCE,
    "chatgpt": (
        "natural_prompt: an image-generation request for ChatGPT (GPT Image) of 80-150 words in fluent English. "
        "Begin with the medium and shot ('A photorealistic close-up photo of ...', 'An anime-style illustration of ...'). "
        "Describe the subject, setting, composition, lighting, colour palette and mood in complete sentences. "
        "If the image contains text, give the exact text in double quotes; otherwise end with 'No text, watermark or logo.' "
        "Finish with the aspect ratio in words (for example 'Aspect ratio 3:2, landscape'). "
        "No tag lists, no weights, no negative-prompt syntax."
    ),
    "grok": (
        "natural_prompt: a concise, vivid English prompt for Grok Imagine of 40-90 words: medium and style first, "
        "then subject, setting, lighting and mood as cinematic description with concrete nouns and adjectives. "
        "No tag lists, no weights, no negative-prompt syntax. End with the aspect ratio in words."
    ),
    "nanobanana": (
        "natural_prompt: a narrative English prompt for Nano Banana (Gemini image generation) of 80-150 words that follows this shape: "
        "'A [medium] [shot type] of [subject], [action or expression], set in [environment]. The scene is illuminated by "
        "[lighting], creating a [mood] atmosphere. Captured with [camera and lens] (for photos) or rendered in [style] "
        "(for illustrations), emphasising [key textures and details]. The image should be in a [aspect ratio] format.' "
        "Describe the scene instead of listing keywords. Phrase exclusions positively (say 'an empty street' rather than "
        "'no cars'). Give any visible text exactly in double quotes."
    ),
    # Midjourney 公式 Prompt Basics: "Short and simple prompts typically generate the best images", 数は "three cats" のように、
    # 否定は本文に書かず --no で。パラメータはツールが付けるので VLM には書かせない
    "midjourney": (
        "natural_prompt: one or two short English sentences (15-45 words) for Midjourney that describe the subject, "
        "medium, environment, lighting, colour, mood and composition. Use specific numbers for how many people or "
        "objects there are (for example 'three women'). Do not make long lists or give instructions. Do not write "
        "negations (no 'no cars', no 'without'). No tag syntax, no weights, no '::', no parameters such as --ar. "
        "Give any visible text exactly in double quotes."
    ),
}


PHOTO_THRESHOLD = 0.5
# ⑦ 作風・質感があるときの写真の境目。実機で ⑦ の写真の realistic が 0.5 に届かず「アニメ・イラスト調」と判定され、
# Krea 2 の本文が An anime-style illustration になった (⑦ だけ下げる)
STYLE_REF_PHOTO_THRESHOLD = 0.25


def photo_score(tags: TagResult) -> float:
    p = tags.general_probs.get
    return max(p("realistic", 0), p("photorealistic", 0), p("photo_(medium)", 0))


def detect_medium(tags: TagResult, photo_threshold: float = PHOTO_THRESHOLD) -> str:
    p = tags.general_probs.get
    if photo_score(tags) >= photo_threshold:
        return "photo"
    if p("3d", 0) >= 0.5:
        return "3d"
    if p("pixel_art", 0) >= 0.5:
        return "pixel_art"
    painting = ("traditional_media", "watercolor_(medium)", "oil_painting_(medium)", "painting_(medium)")
    if max(p(t, 0) for t in painting) >= 0.5:
        return "painting"
    return "anime"


# 人物の崩れを表すタグ。Danbooru に実在して件数が多いもの (bad_anatomy 1.8 万、bad_hands 3.4 千、extra_digits 538、
# fewer_digits 6.3 千、bad_feet 3.9 千、bad_proportions 2.4 千。2026-09-26 に Danbooru API で確認)。
# 前の 4 つは Illustrious-XL v0.1 の公式例のネガティブにも入っている。
# 使わないもの: missing_fingers / extra_limbs / fused_fingers / multiple_navels (Danbooru に無く学習されていない)、
# missing_limb (実体は amputee)、anatomical_nonsense (意図的な誇張の意味)、long_torso (186 件で弱い)
ANATOMY_NEGATIVES = {
    "tags": ["bad anatomy", "bad hands", "extra digits", "fewer digits", "bad feet", "bad proportions"],
    # NovelAI V4.5 の公式プリセット Undesired Content "Human Focus" にある解剖のタグ
    "novelai": ["bad anatomy", "bad hands", "artistic error"],
    # Anima の公式ネガティブに解剖のタグは無い。Danbooru のタグで学習しているので実在するタグだけ足す
    "anima": ["bad anatomy", "bad hands", "extra digits", "fewer digits"],
}
# Danbooru の構図タグ: full_body は "Whole body"、portrait は "Face through shoulders"。全身の絵が顔だけにならないように
FRAMING_NEGATIVES = {"full_body": ["portrait", "close-up"], "wide_shot": ["portrait", "close-up"],
                     "cowboy_shot": ["portrait", "close-up"], "feet_out_of_frame": ["portrait", "close-up"]}


def anatomy_negatives(tags: TagResult, target: str, kind: str) -> list[str]:
    """人物がいるときに足す、崩れと構図のネガティブ。自然文系は公式が否定を勧めない (FLUX.2) ので何も足さない。
    Z-Image (ベース) はネガティブが効くので足す。"""
    have = {n for n, _ in tags.general}
    if (kind == "prose" and target != "zimage") or "no_humans" in have or not (have & set(PERSON_COUNT_TAGS)):
        return []
    out = list(ANATOMY_NEGATIVES.get(target, ANATOMY_NEGATIVES["tags"]))
    for framing, negatives in FRAMING_NEGATIVES.items():
        if framing in have:
            out += negatives
    return dedup(out)


ANATOMY_NOTE = (
    "\n人物の破綻 (関節・欠損・指・胴長・ヘソが増えるなど): ネガティブに Danbooru にあるタグ (bad anatomy, bad hands, extra digits, "
    "fewer digits, bad feet, bad proportions) を入れています。fewer digits は 4 本指のキャラも抑えます。"
    "ヘソが増える・胴長を表す学習済みのタグは無いので、プロンプトだけでは防げません。生成後に直す道具が確実です: "
    "ADetailer (A1111 / Forge、顔・手・人物を検出して描き直す)、ComfyUI Impact Pack の FaceDetailer / Detailer (SEGS)、"
    "手は MeshGraphormer Hand Refiner (comfyui_controlnet_aux)。ポーズが崩れるときは ControlNet (OpenPose) も有効です。"
)


# 画面の中に勝手に出てくる小窓・カットイン・ワイプ・コマ割りを防ぐネガティブ。Danbooru に実在して件数の多いもの
# (2026-09-26 に確認: multiple_views 27 万, cut-in 775, inset 2.7 千, chibi_inset 3.3 万 (隅に添えたデフォルメキャラ、ワイプ風),
# photo_inset 5.3 千, zoom_layer 4 万, split_screen 3 千)。comic / 2koma / 4koma は Illustrious-XL v0.1 の公式例のネガティブ。
# reaction_inset (反応の小窓) は定義はワイプそのものだが 148 件しかなく学習が弱いので入れない。元画像にあるものは入れない
CUT_IN_NEGATIVES = [
    "multiple views", "cut-in", "inset", "chibi inset", "photo inset", "zoom layer", "split screen", "comic", "2koma", "4koma",
]


# 白黒に関係するタグ (Danbooru の定義): monochrome = 単色の濃淡、greyscale = 完全に色の無い白黒、sepia、
# spot_color = ほぼ白黒で一部だけ色、partially_colored = 一部だけ着色して残りは線画、greyscale_with_colored_background、
# limited_palette = 使う色がごく少ない、halftone = 点の大小で灰色の濃淡を表す (漫画のトーン)。
# black_and_white / grayscale / screentone は Danbooru に無い (0 件) ので扱わない。red_theme などの色の傾向は残す
MONOCHROME_TAGS = {
    "monochrome", "greyscale", "sepia", "spot_color", "partially_colored", "greyscale_with_colored_background",
    "limited_palette", "halftone",
}
# 文 (VLM の自然文など) の中の白黒を表す言い方。品質チェックで残っていないか見る
MONOCHROME_WORDS = ["monochrome", "greyscale", "grayscale", "black and white", "black-and-white", "sepia", "halftone"]
MONOCHROME_VLM = (
    "Describe the image as a full-colour image: give natural, plausible colours to hair, eyes, skin, clothing and "
    "background. Do not mention monochrome, greyscale, black and white, sepia, halftone or screentone anywhere."
)


# 同じ人物・そっくりな人物が並ぶことを表すタグ (Danbooru で確認 2026-09-26): clone 9.5 千 (完全に同じキャラが同じ場面に複数)、
# dual_persona 6.4 万 / multiple_persona 1.8 万 (同じキャラの別バージョンが 2 人 / 3 人以上)、twins 4.7 万 (そっくりに描かれやすい)、
# look-alike 3.9 千 (そっくりな 2 キャラ)。same_face は定義はまさにこれだが 43 件で学習が弱いので使わない。
# multiple_views (同じキャラを 1 枚に複数回) はカットイン対策で常に入れている。元画像にあるもの (双子の絵など) は入れない
SAME_FACE_NEGATIVES = ["clone", "dual persona", "multiple persona", "twins", "look-alike"]
# 分身を防ぐ (既定オフ)。タグはすべてタガーの語彙 (Danbooru) で確かめたもの:
# 分身・複製 clone / dual persona / multiple persona、コラージュ・複数アングル collage / reference sheet / variations / turnaround
# (multiple views / split screen は CUT_IN_NEGATIVES で常に入る)、残像 afterimage、体の重複 extra faces / multiple heads / extra arms。
# 二重露光 (double exposure) は Danbooru に無いので入れられない。元画像にあるものは「写っているものは外す」で除かれる
DUPLICATE_NEGATIVES = [
    "clone", "dual persona", "multiple persona", "collage", "reference sheet", "variations", "turnaround", "afterimage",
    "extra faces", "multiple heads", "extra arms",
]
MJ_DUPLICATE_NO = ["clone", "collage", "afterimage"]  # Midjourney の --no は短い一般語だけ (語数の上限があるため)


def duplicate_guard_text(target: str, count: int) -> str:
    """分身を防ぐ: ネガティブの無い自然文の出力先に足す文。各社とも公式にこの題の説明は無いので、一般的な書き方
    (肯定文で具体的に。GPT Image だけは除外を本文に書ける) に合わせる。効き目は保証できない。"""
    who = "The character appears" if count <= 1 else f"Each of the {NUMBER_EN.get(count, count)} people appears"
    text = f"{who} only once, in a single continuous scene seen from one viewpoint."
    if target in PROSE_ALLOWS_NEGATION:
        text += " Do not duplicate anyone, and do not make a collage, split panels or multiple views."
    return text
# Midjourney の --no は語数に上限を設けているので、Danbooru 固有の言い回し (dual persona など) を除いた短い形
MJ_SAME_FACE_NO = ["clone", "twins", "look-alike"]
GENDER_FACE_EN = {"girl": "a feminine face", "boy": "a masculine face"}


def face_features(person: dict[str, Any], limit: int = 4) -> list[str]:
    """その人の顔・髪・目の特徴 (見分けるための特徴)。"""
    return [t for t in dedup(list(person.get("common") or []) + person["tags"]) if person_tag_priority(t) == 1][:limit]


def distinct_faces_text(persons: list[dict[str, Any]], target: str, full: bool, count_hint: int = 0) -> str:
    """ネガティブが使えない自然文の出力先向けに、一人ひとり違う顔であることを書く文。
    full = 「同じ顔を避ける」がオン (全部書く)。False なら男女が混ざるときの性別の顔だけ。
    公式に「双子・同じ顔」を扱った資料は無いので、各社の一般的な書き方 (肯定文で具体的に。GPT Image は除外を本文に) に合わせる。"""
    n = len(persons) or count_hint
    if n < 2 or (not full and not mixed_gender(persons)):
        return ""
    short = target == "midjourney"  # Midjourney は短い描写が公式の推奨
    sentences: list[str] = []
    if persons:
        clauses = []
        for person in persons:
            who = {"girl": "the woman", "boy": "the man"}.get(person.get("subject", ""), "the person")
            position = str(person.get("position") or "")
            where = f" {position_phrase_en(position)}" if position in ("left", "right", "center") else ""
            face = GENDER_FACE_EN.get(person.get("subject", ""), "a face of their own")
            feats = _plain(face_features(person)) if full and not short else ""
            if full or person.get("subject") in GENDER_FACE_EN:
                clauses.append(f"{who}{where} has {face}" + (f" with {feats}" if feats else ""))
        if full:
            sentences.append("Each person has a distinct face" if short
                             else "Each person has their own distinct face, clearly different from the others")
        if clauses and (not short or not full):
            sentences.append("; ".join(clauses))
    elif full:
        sentences.append(f"Each of the {NUMBER_EN.get(n, n)} people has their own distinct face, clearly different from the others")
    if full and target in PROSE_ALLOWS_NEGATION:
        sentences.append("They are not twins or clones; each person has a different face")  # GPT Image は除外を本文に書ける
    return " ".join(x[:1].upper() + x[1:] + "." for x in sentences if x)


def remove_monochrome(tags: TagResult) -> tuple[TagResult, list[str]]:
    """白黒のタグを外す。戻り値: (外した後のタグ, 外したタグ)。"""
    removed = [n for n, _ in tags.general if n in MONOCHROME_TAGS]
    if not removed:
        return tags, []
    general = [(n, p) for n, p in tags.general if n not in MONOCHROME_TAGS]
    probs = {n: p for n, p in tags.general_probs.items() if n not in MONOCHROME_TAGS}
    return TagResult(tags.rating, tags.rating_conf, general, list(tags.characters), probs), removed


def build_negative(tags: TagResult, medium: str, target: str = DEFAULT_TARGET) -> list[str]:
    have = {name for name, _ in tags.general}
    if target == "anima":
        neg = list(ANIMA_NEGATIVE)
    elif target == "novelai":
        neg = list(NAI_NEGATIVE)
    else:
        neg = list(BASE_NEGATIVES[medium])

    def add(*items: str) -> None:
        for item in items:
            if item not in neg:
                neg.append(item)

    if "solo" in have:
        add("multiple views", "2girls", "3girls", "2boys", "crowd")
    add(*CUT_IN_NEGATIVES)  # 元画像にあるものは最後の「写っているものは外す」で除かれる
    for group in ("girl", "boy", "other"):  # 2 人以上なら前後の人数をネガティブに入れて、人数の増減を抑える
        numeric = [PERSON_COUNT_TAGS[t] for t in have if t in PERSON_COUNT_TAGS and group in t and not t.startswith("multiple")]
        if numeric and max(numeric) >= 2:
            n = max(numeric)
            # 1 人少ない側が 1girl などになるときは入れない (人物ごとの区画の先頭が 1girl なので矛盾する)
            for other in (count_tag_name(group, n - 1) if n >= 3 else None, count_tag_name(group, n + 1) if n < 6 else None):
                if other and other not in have:
                    add(other.replace("_", " "))
    if "no_humans" in have:
        add("1girl", "1boy", "humans")
    if "1girl" in have and not ({"1boy", "2boys", "multiple_boys", "male_focus"} & have):
        add("1boy")
    if "1boy" in have and not ({"1girl", "2girls", "multiple_girls"} & have):
        add("1girl")
    if {"monochrome", "greyscale"} & have:
        add("colored", "vibrant colors")
    else:
        add("monochrome", "greyscale")
    text_tags = {"text", "english_text", "japanese_text", "speech_bubble", "signboard"}
    mark_tags = {"signature", "watermark", "artist_name", "logo"}
    if not (text_tags & have):
        add("text", "speech bubble")
    if not (mark_tags & have):
        add("signature", "watermark", "artist name", "logo")
    if "outdoors" in have and "indoors" not in have:
        add("indoors")
    if "indoors" in have and "outdoors" not in have:
        add("outdoors")
    if "night" in have and "day" not in have:
        add("daylight", "day")
    if "day" in have and "night" not in have:
        add("night", "dark")
    if {"simple_background", "white_background", "grey_background"} & have:
        add("complex background", "cluttered background")
    if medium in ("anime", "painting") and not ({"sketch", "lineart", "unfinished"} & have):
        add("sketch", "lineart", "unfinished")
    # 画像に写っているもの (ポジティブに入るタグ) はネガティブから外す。BASE_NEGATIVES の text / watermark なども含む
    drop = {n.replace("_", " ") for n in have}
    if text_tags & have:
        drop |= {"text", "speech bubble"}
    if mark_tags & have:
        drop |= {"signature", "watermark", "artist name", "logo"}
    return [n for n in neg if norm_tag(n) not in drop]


def build_analysis(tags: TagResult, medium: str, size: tuple[int, int], general_th: float) -> str:
    lines = [f"画風: {MEDIUM_JA[medium]}"]
    rating_ja = RATING_JA.get(tags.rating, tags.rating)
    lines.append(f"レーティング: {rating_ja} ({tags.rating}, {tags.rating_conf:.0%})")
    if tags.characters:
        chars = ", ".join(f"{sd_tag(n)} ({p:.0%})" for n, p in tags.characters)
        lines.append(f"キャラクター: {chars}")
    top = ", ".join(sd_tag(n) for n, _ in tags.general[:10])
    lines.append(f"主な要素: {top}")
    lines.append(
        f"検出タグ数: {len(tags.general)} (しきい値 {general_th:.2f}) / 画像サイズ {size[0]}×{size[1]}"
    )
    return "\n".join(lines)


def suggest_size(size: tuple[int, int]) -> str:
    w, h = size
    ratio = w / h if h else 1.0
    if ratio < 0.8:
        return "縦長 832×1216 (SDXL) / 512×768 (SD1.5)"
    if ratio > 1.25:
        return "横長 1216×832 (SDXL) / 768×512 (SD1.5)"
    return "正方形 1024×1024 (SDXL) / 512×512 (SD1.5)"


def build_notes(medium: str, size: tuple[int, int], target: str = DEFAULT_TARGET) -> str:
    size_hint = f"生成サイズは元画像の比率に合わせて {suggest_size(size)} が目安です。"
    if target == "anima":
        return (
            "Anima: CFG 4〜5、30〜50 ステップ、512²〜1536² の範囲で生成。タグと自然文は自由に混在できます。"
            "絵師タグを足す場合は @ を付けます (例: @artist_name)。" + size_hint
        )
    if target == "flux2":
        return (
            "FLUX.2 [dev]: steps 50 (28 が妥協点)、guidance 4 (公式モデルカード)。[pro] / [max] は BFL の API。"
            "ネガティブは非対応なので、避けたいものは本文で肯定的に言い換えます (BFL の公式ガイド)。" + size_hint
        )
    if target == "flux2klein":
        return (
            "FLUX.2 [klein] 4B: steps 4、guidance 1.0、1024×1024 前後 (公式モデルカード。4B は Apache 2.0、9B は非商用)。"
            "短いプロンプトを補う機能が無いので、本文は詳しく書いています。ネガティブは非対応です。" + size_hint
        )
    if target == "flux":
        return (
            "Flux dev: guidance 3.5 前後、CFG 1 のためネガティブプロンプトは通常無視されます "
            "(CFG > 1 にした場合のみ有効)。" + size_hint
        )
    if target == "krea2":
        return (
            "Krea 2: 自然文のみで指示します。ネガティブプロンプトと重み構文は非対応 (Turbo は CFG 1.0) のため出力できません。"
            "避けたい要素は本文で肯定形に言い換えます。最大 2K まで生成できます。"
        )
    if target in ("zimageturbo", "zimage"):
        # 出典: huggingface.co/Tongyi-MAI/Z-Image-Turbo と /Z-Image のモデルカード、diffusers の ZImagePipeline、
        # ComfyUI の comfy/text_encoders/z_image.py (max_length の上限なし、disable_weights=True)
        common = (
            f"生成サイズは幅と高さの数値で元画像の比率 ({aspect_label(size)}) に合わせます (本文には書いていません)。"
            "diffusers は既定で 512 トークンを超えた分を切り捨てます (max_sequence_length で変更可)。ComfyUI は切り捨てません。"
            "重みの構文 (tag:1.3) は ComfyUI でも効きません。"
        )
        if target == "zimageturbo":
            return (
                "Z-Image Turbo: 8 ステップ (diffusers では num_inference_steps=9)、guidance 0、公式の例は 1024×1024 (公式モデルカード)。"
                "蒸留モデルで CFG を使わないため、ネガティブは効かないので出力しません。" + common
            )
        return (
            "Z-Image: 28〜50 ステップ、CFG 3.0〜5.0、512²〜2048² の範囲で縦横比は自由 (公式モデルカード)。"
            "cfg_normalization は一般の画風で False、写実で True が公式の推奨です。ネガティブは CFG で効きます。" + common
        )
    if target == "mossimage":
        # 出典: github.com/OpenMOSS/MOSS-Image の USERGUIDE.md (Inference Parameters)、huggingface.co/OpenMOSS-Team/MOSS-Image-Base
        return (
            "MOSS-Image: 50 ステップ以上、CFG 3.0 (3.0〜6.0。短いプロンプトでは上げる)、shift 5.0、Euler (公式 USERGUIDE)。"
            f"生成サイズは約 1M / 1.5M / 2M ピクセルで、幅と高さは 16 の倍数にし、元画像の比率 ({aspect_label(size)}) に合わせます (本文には書いていません)。"
            "ネガティブ欄は空です (公式の既定も空)。公式が試したネガティブの例は nsfw, worst quality, bad anatomy, watermark, jpeg artifact です。"
            "重みの構文 (tag:1.3) は公式資料に書き方がありません。ComfyUI では ComfyUI Manager で MOSS-Image のプラグインを入れます。"
        )
    if target == "qwen":
        return (
            "Qwen-Image 2.1: ネイティブ 2K (約 4MP)、25 ステップ、Euler / simple、CFG 1 が公式設定。"
            "CFG 1 ではネガティブは無視されます。画像内の文字は英語の二重引用符で正確に書くと再現されます。"
        )
    if target == "novelai":
        return (
            "NovelAI Diffusion V4.5: 品質タグは公式の Add Quality Tags 相当 (location, very aesthetic, masterpiece, no text)、"
            "Undesired Content は Heavy プリセット + 画像由来の除外。強調は {tag} または 1.3::tag::、括弧はエスケープ不要。"
            "複数人物は「人物ごと」タブの各行を Character 1, 2 … に貼り、位置を指定してください (本文には共有タグだけ入れています)。"
            + size_hint
        )
    if target == "chatgpt":
        return (
            "ChatGPT (GPT Image): 自然文の依頼文だけで指示します。ネガティブは非対応のため出力できません。避けたい要素は本文に "
            "'No text, watermark or logo.' のように書きます。縦横比は本文で指定 (1:1 / 3:2 / 2:3 など)。"
            "画像内の文字は二重引用符で正確に。"
        )
    if target == "grok":
        return (
            "Grok Imagine: 短めで具体的な自然文が向きます。ネガティブ・重み構文は非対応のため出力できません。"
            "画像を動画化するときは「追加プロンプト」の動画用カメラワークを続けて入力すると意図が伝わります。"
        )
    if target == "nanobanana":
        return (
            "Nano Banana (Gemini 2.5 Flash Image / Nano Banana Pro): キーワードの羅列より情景の描写が効きます。"
            "ネガティブは非対応のため出力できません。避けたい要素は肯定形で ('no cars' より 'an empty street')。縦横比は本文または生成設定で指定。"
            "画像内の文字は二重引用符で正確に。"
        )
    if target == "midjourney":
        return (
            "Midjourney (docs.midjourney.com、2026-09 時点で既定 V8.2): 公式は「短く簡潔なプロンプトが最良、長いリストや"
            "細かい指示は混乱させる」としているので、本文は短い描写文にしています (全部のタグは「追加プロンプト」「検出タグ」タブ)。"
            "パラメータは末尾に付けています: --ar は元画像の比率、除外は --no (本文の否定文は効かない)。"
            "アニメ・イラストには --niji 7 (公式: 東洋・アニメの美学とイラスト向け)、写真には --raw (公式: 写真らしい画像になりやすい) を"
            "付けています (公式の説明から選んだもので、明示的な推奨ではありません。外すと既定の V8.2)。"
            "V8.2 / Niji 7 は :: の重み付けに対応していないので使っていません。元画像を Imagine bar にアップロードすると"
            "画像プロンプトとして使えます (Discord ではプロンプトの先頭に画像の URL)。"
        )
    return f"{MODEL_FAMILY_JA[medium]}が向いています。" + size_hint


# --------------------------------------------------------------------------
# Ollama VLM
# --------------------------------------------------------------------------

VLM_SYSTEM = """You help reverse-engineer image-generation prompts for Stable Diffusion / Flux.
You receive one image plus tags detected by a specialised tagger with confidence scores.
Treat the tagger tags as ground truth for WHAT is in the image. Your job is to add what a tagger cannot see well: composition, camera, lighting, colour palette, mood, art style, and the overall scene description.

Rules
- Describe only what is visible. Do not invent objects, people or text that are not there.
- Never identify a real person by name; describe appearance and expression instead. Well-known fictional characters may be named.
- Write every field in English only.
- "natural_prompt": one fluent English paragraph of 60-120 words, like a caption for Flux / SD3. Subject first, then setting, lighting, colour, composition, and finally medium and style. No lists, no tag syntax.
- "extra_tags": up to 12 lowercase Danbooru-style tags for composition, camera, lighting, colour, atmosphere and art style that are missing from the tagger list. No weights, no duplicates of given tags.
- "negative_additions": up to 12 short lowercase phrases that must NOT appear if someone regenerates this image: the wrong medium, wrong colours, unwanted extra objects or people, wrong time of day, wrong mood. Do not repeat the base negatives you are given.
- "persons": if the image shows two or more people, one entry per person (at most 6), ordered left to right. subject is girl / boy / other. position is a short phrase such as "left", "center", "right", "foreground left", "background". tags are the lowercase Danbooru-style tags that belong to THAT person only: hair, eyes, expression, body, clothing, accessories, pose, and the character name if the tagger found one. Take them from the tagger list where they fit and add what is missing. Never put scene, background, lighting or composition tags into a person. description is one English sentence about that person. When there is at most one person, return an empty array.
- When there are several people, natural_prompt must describe each person separately, identifying them by position, and must never merge one person's hair, clothing or expression with another's.
- "camera": one English sentence about the camera and composition only: shot type (close-up / medium / full body / wide), angle (eye level, low angle, high angle, dutch), lens feel (wide angle, 50mm, 85mm portrait, telephoto compression, fisheye), depth of field and focus, framing and where the subject sits in the frame.
- "lighting": one English sentence about light and colour only: direction and quality of the key light, fill and rim light, time of day, colour palette or grading, contrast and mood.
- "style": one or two English sentences about the medium and rendering only, concrete enough to reproduce the look: medium (anime / illustration / photo / 3d), line work (thin, thick, sketchy, clean, no lineart), shading (cel, soft, painterly, watercolor, airbrush), colour palette (pastel, vivid, muted, monochrome, limited palette), texture (grain, paper, halftone, glossy), era (1990s cel anime, 2000s, 2010s, modern), and the closest style family (for example 'modern pixiv illustration', 'Kyoto Animation-like', 'Ghibli-like', 'webtoon', 'game CG', 'retro cel anime', 'Western cartoon'). Do not name real individual artists; studios and genres are fine. No subject description.
- "style_tags": up to 10 lowercase Danbooru-style tags describing the art style only (e.g. anime coloring, cel shading, soft shading, thick outlines, sketch, watercolor (medium), pastel colors, muted color, 1990s (style), film grain, flat color, painterly). Empty array for photographs.
- "motion": one or two English sentences for an image-to-video model (LTX-2, Wan, Kling): first the camera movement (static, slow push-in, pan, tilt, orbit, crane, handheld) that suits this composition, then plausible subject and environment motion (hair, cloth, breathing, blink, petals, water, smoke, light). Keep it subtle and physically plausible for THIS image; do not add new objects or change the framing.
- "subject": one English sentence describing only the main subject: the person or people (appearance, clothing, pose, expression) or, if there are no people, the main object or creature. No background, no camera, no style words.
- "background": one English sentence describing only the environment behind and around the subject: location, architecture, nature, weather, furniture fixed to the scene. No people, no camera, no style words.
- "objects": up to 8 short lowercase noun phrases for notable movable objects, props or animals in the frame (things held, on tables, vehicles, pets). Exclude clothing worn by people and exclude the background itself. Empty array if none.
- "pose_tags": when there is one person, up to 12 lowercase Danbooru-style tags for that person's exact pose: body posture, what they sit or lean on, arm, hand and leg positions, head tilt (e.g. sitting on bench, hand on lap, crossed legs, head tilt, arm support). Only what is clearly visible. When there are two or more people, return an empty array and put the poses into persons.
- "outfit_tags": when there is one person, up to 16 lowercase Danbooru-style tags for the clothing, footwear, headwear and accessories worn, with colours (e.g. white dress, puffy short sleeves, black thighhighs). When there are two or more people, return an empty array and put the outfits into persons.
- In each persons entry, "number" is the number of the box drawn around that person when numbered boxes are shown (otherwise the order from left, starting at 1), "pose_tags" (up to 12) is that person's exact pose and "outfit_tags" (up to 16) is what that person wears.
- "interactions": an empty array unless the instructions below list actions between people.
- "age": when the image shows exactly one person or human-like character, the apparent age group that person is shown as: baby (under 1), toddler (1-3), child (4-9), preteen (10-12), teen (13-17), young adult (18-29), thirties, forties, fifties, sixties, elderly (70 or older). "unknown" when there is no person, when there are several people, or when it cannot be judged. Judge from the face, the skin, the body proportions and the height against the surroundings; for a drawn character, from the age the drawing portrays. Do not fall back on young adult: choose the group that really fits, for children and for older people as well.
- In each persons entry, "age" is that person's apparent age group, chosen in the same way ("unknown" when it cannot be judged).
"""

# VLM には日本語の欄 (analysis / notes) を書かせない。1 回の呼び出しで日本語と英語を書き分けさせると、先に書いた日本語に
# 引きずられて英語の欄 (自然文・タグ) まで日本語になることがあり、本文に日本語が混ざった (2026-10-04 利用者の指摘。
# VLM は英語の欄だけ。分析メモ・メモは i2t が日本語で組み立てる部分だけにする)
VLM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "natural_prompt": {"type": "string"},
        "extra_tags": {"type": "array", "items": {"type": "string"}},
        "negative_additions": {"type": "array", "items": {"type": "string"}},
        "persons": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "number": {"type": "integer"},
                    "subject": {"type": "string", "enum": ["girl", "boy", "other"]},
                    "position": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "pose_tags": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
                    "outfit_tags": {"type": "array", "items": {"type": "string"}, "maxItems": 16},
                    "description": {"type": "string"},
                    "age": {"type": "string", "enum": AGE_ENUM},
                },
                "required": ["number", "subject", "position", "tags", "pose_tags", "outfit_tags", "description", "age"],
            },
        },
        "age": {"type": "string", "enum": AGE_ENUM},
        "camera": {"type": "string"},
        "lighting": {"type": "string"},
        "style": {"type": "string"},
        "motion": {"type": "string"},
        "subject": {"type": "string"},
        "background": {"type": "string"},
        "objects": {"type": "array", "items": {"type": "string"}},
        "style_tags": {"type": "array", "items": {"type": "string"}},
        "pose_tags": {"type": "array", "items": {"type": "string"}},
        "outfit_tags": {"type": "array", "items": {"type": "string"}},
        "interactions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tag": {"type": "string"},
                    "source": {"type": "integer"},
                    "target": {"type": "integer"},
                    "mutual": {"type": "boolean"},
                },
                "required": ["tag", "source", "target", "mutual"],
            },
        },
    },
    "required": [
        "natural_prompt", "extra_tags", "negative_additions", "persons", "age",
        "camera", "lighting", "style", "motion", "subject", "background", "objects", "style_tags",
        "pose_tags", "outfit_tags", "interactions",
    ],
}

# 複数枚モードで ② ポーズ / ③ 服装を VLM に 1 枚ずつ見せる呼び出し。
# 3 枚まとめて見せると ① の頬杖を「② のポーズ」として hand on chin と書くなど ① が混ざり、語が一致しないので外せなかった。
# 1 枚だけ見せれば ① は混ざらない (実機: qwen2.5vl / qwen3.5 とも 3 回中 3 回、① の混入なし)
VLM_PART_NUM_PREDICT = 700  # タグ 12〜16 個と 1〜2 文なので短くてよい (出力が短いほど速い)
VLM_DETAIL_NUM_PREDICT = 2400  # 人物の特徴を詳しく: 欄ごとに複数の特徴 + タグ 32 + 2〜5 文 (以前の 900 では足りない)


def vlm_part_system(slot: str, with_objects: bool) -> str:
    """② / ③ を 1 枚だけ見せるときの指示。with_objects = ③ が小物も受け持つ (⑤ が無い)。"""
    if slot == "style":
        # ⑦ 作風・質感: 絵 → 実写 / 実写 → 絵に使う。作品名・作者名はタガーに無いので VLM だけが推定する。
        # 絵なら色の塗り方、写真ならその写真の撮影環境まで読む
        lines = [
            "You read the ART STYLE and LOOK of one reference image for an image-generation prompt, so that another image "
            "can be made to look the same way.",
            "Ignore who is shown, their face, hair, eyes, skin, ethnicity, clothing, pose, objects and the background: "
            "only how the image is made matters (medium, rendering, line work, colouring, shading, proportions of the style).",
            "If it is a drawing, painting or CG, read how it is coloured: the shading method (cel shading with one or two "
            "shadow tones, soft gradient shading, thick painterly brushwork, watercolour washes and bleeding, airbrush, "
            "markers, flat colour), the colour of the shadows, how highlights and rim light are drawn, the colour and "
            "weight of the line art or its absence, and finishing effects (glow, blur overlays, paper or canvas texture).",
            "If it is a photograph, read the shooting setup that produced its look: the kind of camera (smartphone, "
            "film camera, DSLR or mirrorless, instant film, disposable camera), the lens look (wide angle or telephoto, "
            "depth of field and bokeh), the lighting setup and its quality and direction (studio softbox, ring light, "
            "on-camera flash, window light, hard direct sun, overcast daylight), film stock look or digital, grain or "
            "noise, colour grading, exposure, contrast and sharpness. Do not describe the place, the time of day or what "
            "is shown.",
            '- "style_tags": up to 10 lowercase Danbooru-style tags for the medium and rendering only (e.g. anime coloring, '
            "cel shading, soft shading, thick outlines, flat color, watercolor (medium), oil painting (medium), 3d, "
            "photorealistic, film grain, flash photography, depth of field, soft lighting, high contrast, american comics, "
            "chibi). Do not give the era or decade of the style.",
            '- "style_names": up to 3 names of the franchise, series or artist whose visual style this image most clearly '
            "follows, in lowercase English as commonly written (for example: akira toriyama, dragon ball, naruto, "
            "the powerpuff girls, dc comics, studio ghibli). Only names you are confident about; an empty array when unsure "
            "or when the image is an ordinary photograph. Never give the names of the characters shown, and never "
            "generic words such as anime, manga, photo, realistic or clothing names.",
            '- "style": two or three English sentences describing only the medium and the look: for a drawing how it is '
            "coloured, for a photograph how it was shot. Never mention the person, what they wear, their pose or the "
            "setting.",
        ]
    elif slot == "expression":
        # ⑧ 表情: 表情と視線だけ
        lines = [
            "You read the FACIAL EXPRESSION and GAZE of the person in one reference image for an image-generation prompt.",
            "Ignore the person's identity, hair, eye colour, face shape, skin, clothing, body, pose and the background: "
            "only the expression and where the eyes look matter.",
            '- "expression_tags": up to 10 lowercase Danbooru-style tags for the expression and gaze only (e.g. smile, '
            "open mouth, blush, closed eyes, one eye closed, tears, pout, angry, surprised, smug, looking at viewer, "
            "looking away, looking up).",
        ]
    elif slot == "body":
        lines = [
            "You read the BODY TYPE of the person in one reference image for an image-generation prompt.",
            "Ignore the person's face, hair, eyes, expression, clothing, pose and the background: only the body matters.",
            '- "body_tags": up to 10 lowercase Danbooru-style tags for the body: breast size, build, height, limb and hip '
            "proportions, skin tone, apparent age of the body, muscles (e.g. large breasts, slim, petite, wide hips, "
            "thick thighs, long legs, dark skin, muscular, abs).",
        ]
    elif slot == "pose":
        lines = [
            "You read the POSE of the person in one reference image for an image-generation prompt.",
            "Ignore the person's face, hair, clothing, colours, body type and the background: only the pose matters.",
            '- "pose_tags": up to 12 lowercase Danbooru-style tags for the exact pose: body posture, arm, hand and leg '
            "positions, head tilt, gaze, and the framing (e.g. standing, arms up, v, hand on own head, head tilt, full body).",
            '- "pose": one or two English sentences describing only the pose.',
            '- "camera": one English sentence about framing, camera angle and composition.',
        ]
    else:
        lines = [
            "You read the OUTFIT worn by the person in one reference image for an image-generation prompt.",
            "Ignore the person's face, hair, eyes, body, pose and the background: only what is worn matters.",
            '- "outfit_tags": up to 16 lowercase Danbooru-style tags for the clothing, footwear, headwear, accessories and '
            "hair ornaments, with colours and details (e.g. white bikini, frilled bikini, side-tie bikini, straw hat, hat ribbon).",
            '- "outfit": one or two English sentences describing only the outfit: garments, colours, materials, accessories.',
        ]
        if with_objects:
            lines.append('- "objects": up to 8 short lowercase noun phrases for objects held or carried (bags, umbrellas, '
                         "weapons, books, drinks). Empty array if none.")
    lines += [
        "Tags from a specialised tagger are given with confidence scores. Treat them as ground truth for what is visible.",
        "Only what is clearly visible. Never repeat a tag. Everything is English.",
    ]
    return "\n".join(lines) + "\n"


def vlm_part_schema(slot: str, with_objects: bool) -> dict[str, Any]:
    arr = {"type": "array", "items": {"type": "string"}}
    if slot == "style":
        props = {"style_tags": arr | {"maxItems": 10}, "style_names": arr | {"maxItems": 3},
                 "style": {"type": "string", "maxLength": 900}}
    elif slot == "body":
        props = {"body_tags": arr | {"maxItems": 10}}
    elif slot == "expression":
        props = {"expression_tags": arr | {"maxItems": 10}}
    elif slot == "pose":
        props = {"pose_tags": arr | {"maxItems": 12}, "pose": {"type": "string"}, "camera": {"type": "string"}}
    else:
        props = {"outfit_tags": arr | {"maxItems": 16}, "outfit": {"type": "string"}}
        if with_objects:
            props["objects"] = arr | {"maxItems": 8}
    return {"type": "object", "properties": props, "required": list(props)}


# 複数枚モードで ② から取った画角タグの重み。実機 (animagine-xl-3.1、3 シード) で full body が重み無し 0/3 → 1.3 で 2/3。
# ネガティブに近い画角を足す・先頭に移すだけでは 0/3 だった。重みの書式は A1111 wiki の (tag:1.3) と NovelAI docs の 1.3::tag::。
# Anima (Qwen3 テキストエンコーダ) で重みが効くかは公式資料で確かめられていないので入れない。自然文系・Midjourney には書式が無い
REF_FRAMING_WEIGHT = 1.3
REF_FRAMING_WEIGHT_TARGETS = ("sdxl", "novelai")
# ② の向き・アングルのタグにも同じ重みを付ける。② が後ろ姿のとき、重み無しは後ろ姿 0/3、1.3 で 2/3
REF_ANGLE_TAGS = {
    "from_behind", "facing_away", "from_side", "profile", "from_above", "from_below", "three_quarter_view", "dutch_angle",
    "straight-on", "high_angle", "low_angle",
}
# ② が後ろ向き (from_behind / facing_away) のとき外す、正面からしか見えない露出タグ (③ や ① が正面の絵だと入る)。
# 実機: navel / stomach / cleavage / collarbone を外すと後ろ姿 0/3 → 3/3。外したものは警告欄に出す
BACK_VIEW_TAGS = {"from_behind", "facing_away"}
FRONT_ONLY_EXPOSURE = {
    "navel", "covered_navel", "stomach", "midriff_peek", "cleavage", "cleavage_cutout", "collarbone", "underboob",
    "downblouse", "covered_nipples", "nipples", "areolae", "areola_slip", "nipple_slip", "breasts_out", "groin", "cameltoe",
    "pussy", "penis", "testicles", "pubic_hair", "female_pubic_hair", "male_pubic_hair",
}


# --------------------------------------------------------------------------
# 構図と強さの指定 (ボーン / カメラ / 強さ)。LLM・GPU 不要の規則だけで動く
# controls = {"pose": {"tags": [...]}, "camera": {"direction", "height", "distance", "tilt"}, "strength": {枠: "strong"|"weak"}}
# --------------------------------------------------------------------------
CAMERA_OPTIONS: dict[str, list[tuple[str, str, list[str]]]] = {
    # (値, 画面の表示, 入れるタグ)。タグ名はタガーの語彙 (selected_tags.csv) で確かめたもの
    "direction": [("", "指定しない", []), ("front", "正面", ["facing_viewer"]),
                  ("three_quarter", "斜め", []),  # 斜め 45 度のタグは Danbooru 系の語彙に無い (自然文系にだけ書く)
                  ("side", "横", ["from_side"]), ("back", "後ろ", ["from_behind", "facing_away"])],
    "height": [("", "指定しない", []), ("below", "下から", ["from_below"]), ("eye", "目線", []), ("above", "上から", ["from_above"])],
    "distance": [("", "指定しない", []), ("face", "顔", ["portrait"]), ("upper", "上半身", ["upper_body"]),
                 ("cowboy", "膝上", ["cowboy_shot"]), ("full", "全身", ["full_body"]), ("wide", "引き", ["wide_shot"])],
    "tilt": [("", "指定しない", []), ("none", "傾けない", []), ("dutch", "傾ける", ["dutch_angle"])],
}
CAMERA_LABELS_JA = {"direction": "向き", "height": "高さ", "distance": "距離", "tilt": "傾き"}
# 指定したら外す、同じ種類の元のタグ
CAMERA_DIMENSION_TAGS = {
    "direction": {"facing_viewer", "from_side", "profile", "from_behind", "facing_away", "straight-on", "three_quarter_view"},
    "height": {"from_above", "from_below", "straight-on", "high_angle", "low_angle", "bird's-eye_view", "worm's-eye_view"},
    "distance": set(_FRAMING_ANY) | {"close-up", "portrait", "upper_body", "cowboy_shot", "full_body", "wide_shot",
                                     "very_wide_shot", "lower_body", "feet_out_of_frame", "head_out_of_frame"},
    "tilt": {"dutch_angle"},
}
CAMERA_PROSE_EXTRA = {"three_quarter": "three-quarter view", "eye": "eye-level shot"}  # 自然文系だけに書く言い方
# 撮影プリセット (controls["look"] = {"name", "prompt"})。選ぶと、画像から読んだレンズ・照明の質・色調のタグを外して
# プリセットの文に置き換える (全出力先)。
# 画角・アングル・視線 (構図) と、時間帯・天気・光源 (night / sunset / sunlight / neon lights など場面の内容) は外さない
LOOK_REPLACED_TAGS = {
    # レンズ・写り
    "depth_of_field", "blurry_background", "blurry_foreground", "bokeh", "fisheye", "tilt-shift", "motion_blur",
    "chromatic_aberration", "film_grain", "vignetting", "lens_flare", "bloom",
    # 照明の当て方・質と色調
    "backlighting", "backlit", "sidelighting", "rim_light", "spotlight", "light_rays", "god_rays", "high_contrast",
    "low_contrast", "chiaroscuro", "color_grading", "warm_colors", "cool_colors", "muted_color", "pastel_colors",
    "vibrant_colors",
}
# VLM の自然文に入れてもらう指示: 撮影の文 (look_prose で文にしたもの) を、画像から読んだレンズ・照明の代わりに自然文へ
# 織り込ませる。書かれなかったときは後から文で足す (look_in_text)
LOOK_VLM = (
    "Describe the photography in natural_prompt with this fixed camera and lighting description instead of what the image "
    "suggests, woven naturally into your sentences (keep every named camera, lens, film stock and setting as written): "
    "\"{look}\" Do not describe any other lens, focal length, depth of field, film stock, grain, lighting quality or colour "
    "grading. Still describe the shot type, angle, time of day, weather and visible light sources."
)


LOOK_NONE = "指定しない"
LOOK_PARTS_MENU = "組み合わせ..."  # 撮影の欄で選ぶと、項目ごとに選ぶ画面を開く


def find_look(cfg: dict[str, Any], name: str) -> dict[str, str] | None:
    """撮影プリセットを名前で探す (config.json の camera_looks)。"""
    return next((dict(x) for x in cfg.get("camera_looks") or [] if x.get("name") == name), None)


def look_replaces(tag: str) -> bool:
    """撮影プリセットを選んだときに外すタグか (表記ゆれ: 空白 / 下線・重み付きも同じとみなす)。"""
    key = _strip_weight(norm_tag(tag)).replace(" ", "_")
    return key in LOOK_REPLACED_TAGS or key.endswith("_lighting")


def look_phrases(prompt: str) -> list[str]:
    """撮影プリセットの文 → タグ形式の出力先に入れる語 (カンマ区切り)。"""
    return dedup([x.strip() for x in (prompt or "").split(",") if x.strip()])

# 撮影の組み合わせ。
# 例: "Cinematic photo, shot on Fujifilm X-T4, 50mm f/1.8 lens, clean smooth skin with natural texture, balanced ambient
# lighting, subtle film grain, shallow depth of field"。項目ごとに選んだ語をこの順につなぎ、撮影プリセットの文と同じに使う
# (選んだ項目だけ。プリセットとは同時に使わない)。値は英語の語句そのまま、表示は日本語。
LOOK_PART_LABELS_JA = {"style": "写り", "camera": "カメラ", "lens": "レンズ", "texture": "質感", "light": "光",
                       "film": "フィルム", "grain": "粒子", "dof": "被写界深度"}
LOOK_PARTS: dict[str, list[tuple[str, str]]] = {
    "style": [
        ("Cinematic photo", "シネマ風の写真"), ("cinematic film still", "映画のワンシーン"),
        ("professional photography", "プロの写真"), ("RAW photo", "RAW 写真"),
        ("portrait photography", "ポートレート"), ("analog film photography", "フィルム写真"),
        ("candid snapshot", "スナップ"), ("street photography", "街撮り"),
        ("documentary photography", "ドキュメンタリー"), ("editorial fashion photography", "ファッション誌"),
        ("studio photography", "スタジオ撮影"), ("lifestyle photography", "ライフスタイル"),
        ("travel photography", "旅行写真"), ("fine art photography", "アート写真"),
        ("vintage photograph", "古い写真"), ("polaroid photo", "ポラロイド写真"),
        ("award-winning photograph", "受賞作のような写真"), ("casual smartphone photo", "スマホの写真"),
    ],
    "camera": [
        # ミラーレス・一眼レフ
        ("shot on Fujifilm X-T4", "富士 X-T4"), ("shot on Fujifilm X-T5", "富士 X-T5"),
        ("shot on Fujifilm X-H2S", "富士 X-H2S"), ("shot on Fujifilm X-Pro3", "富士 X-Pro3"),
        ("shot on Fujifilm X100V", "富士 X100V"), ("shot on Fujifilm X100VI", "富士 X100VI"),
        ("shot on Sony A7 IV", "ソニー α7 IV"), ("shot on Sony A7R V", "ソニー α7R V"),
        ("shot on Sony A7S III", "ソニー α7S III"), ("shot on Sony A1", "ソニー α1"),
        ("shot on Sony A9 III", "ソニー α9 III"), ("shot on Sony ZV-E10", "ソニー ZV-E10"),
        ("shot on Canon EOS R5", "キヤノン EOS R5"), ("shot on Canon EOS R6 Mark II", "キヤノン EOS R6 Mark II"),
        ("shot on Canon EOS R3", "キヤノン EOS R3"), ("shot on Canon EOS 5D Mark IV", "キヤノン EOS 5D Mark IV"),
        ("shot on Canon EOS-1D X Mark III", "キヤノン EOS-1D X Mark III"),
        ("shot on Nikon Z8", "ニコン Z8"), ("shot on Nikon Z9", "ニコン Z9"), ("shot on Nikon Z6III", "ニコン Z6III"),
        ("shot on Nikon Zf", "ニコン Zf"), ("shot on Nikon D850", "ニコン D850"),
        ("shot on Panasonic Lumix S5 II", "パナソニック S5 II"), ("shot on Panasonic Lumix GH6", "パナソニック GH6"),
        ("shot on OM System OM-1", "OM SYSTEM OM-1"), ("shot on Pentax K-1 Mark II", "ペンタックス K-1 Mark II"),
        ("shot on Ricoh GR III", "リコー GR III"), ("shot on Ricoh GR IIIx", "リコー GR IIIx"),
        ("shot on Leica M11", "ライカ M11"), ("shot on Leica Q3", "ライカ Q3"), ("shot on Leica SL2", "ライカ SL2"),
        ("shot on Leica M10 Monochrom", "ライカ M10 モノクローム"),
        # 中判
        ("shot on Fujifilm GFX100S", "富士 GFX100S（中判）"), ("shot on Fujifilm GFX100 II", "富士 GFX100 II（中判）"),
        ("shot on Hasselblad X2D 100C", "ハッセルブラッド X2D（中判）"), ("shot on Hasselblad 907X", "ハッセルブラッド 907X（中判）"),
        ("shot on Phase One IQ4", "フェーズワン IQ4（中判）"), ("shot on Pentax 645Z", "ペンタックス 645Z（中判）"),
        # フィルムカメラ
        ("shot on Leica M6", "ライカ M6（フィルム）"), ("shot on Canon AE-1", "キヤノン AE-1（フィルム）"),
        ("shot on Nikon F3", "ニコン F3（フィルム）"), ("shot on Pentax K1000", "ペンタックス K1000（フィルム）"),
        ("shot on Olympus OM-1 film camera", "オリンパス OM-1（フィルム）"), ("shot on Contax T2", "コンタックス T2（フィルム）"),
        ("shot on Yashica T4", "ヤシカ T4（フィルム）"), ("shot on Mamiya RZ67", "マミヤ RZ67（中判フィルム）"),
        ("shot on Pentax 67", "ペンタックス 67（中判フィルム）"), ("shot on Hasselblad 500C/M", "ハッセルブラッド 500C/M（中判フィルム）"),
        ("shot on Rolleiflex 2.8F", "ローライフレックス 2.8F（二眼レフ）"), ("shot on Polaroid SX-70", "ポラロイド SX-70"),
        ("shot on Fujifilm Instax Mini", "チェキ（instax mini）"), ("shot on a disposable film camera", "使い捨てカメラ"),
        ("shot on a Holga toy camera", "ホルガ（トイカメラ）"),
        # シネマカメラ
        ("shot on ARRI Alexa 35", "ARRI Alexa 35（映画）"), ("shot on ARRI Alexa Mini LF", "ARRI Alexa Mini LF（映画）"),
        ("shot on RED V-Raptor", "RED V-Raptor（映画）"), ("shot on Sony Venice 2", "ソニー VENICE 2（映画）"),
        ("shot on Panavision Millennium DXL2", "パナビジョン DXL2（映画）"),
        ("shot on Blackmagic URSA Mini Pro 12K", "Blackmagic URSA 12K（映画）"),
        ("shot on Canon EOS C300 Mark III", "キヤノン C300 Mark III（映像）"), ("shot on Sony FX3", "ソニー FX3（映像）"),
        # スマホ・そのほか
        ("shot on iPhone 16 Pro", "iPhone 16 Pro"), ("shot on iPhone 15 Pro", "iPhone 15 Pro"),
        ("shot on Google Pixel 9 Pro", "Google Pixel 9 Pro"), ("shot on Samsung Galaxy S24 Ultra", "Galaxy S24 Ultra"),
        ("shot on a GoPro HERO12", "GoPro HERO12"), ("shot on a DJI Mavic 3 drone", "ドローン（DJI Mavic 3）"),
        ("shot on a 2000s compact digital camera", "2000 年代のコンデジ"), ("shot on a webcam", "ウェブカメラ"),
        ("CCTV security camera footage", "防犯カメラ"),
    ],
    "lens": [
        ("8mm fisheye lens", "8mm 魚眼"), ("14mm ultra-wide lens", "14mm 超広角"), ("24mm f/1.4 lens", "24mm f/1.4 広角"),
        ("28mm f/2 lens", "28mm f/2"), ("35mm f/1.4 lens", "35mm f/1.4"), ("40mm pancake lens", "40mm パンケーキ"),
        ("50mm f/1.8 lens", "50mm f/1.8"), ("50mm f/1.2 lens", "50mm f/1.2"), ("Leica Summilux 50mm f/1.4 lens", "ライカ ズミルックス 50mm"),
        ("Zeiss Otus 55mm f/1.4 lens", "ツァイス Otus 55mm"), ("58mm f/1.4 lens", "58mm f/1.4"),
        ("85mm f/1.8 lens", "85mm f/1.8"), ("85mm f/1.4 lens", "85mm f/1.4"), ("85mm f/1.2 lens", "85mm f/1.2"),
        ("100mm macro lens", "100mm マクロ"), ("105mm f/1.4 lens", "105mm f/1.4"), ("135mm f/1.8 lens", "135mm f/1.8"),
        ("200mm f/2 lens", "200mm f/2"), ("70-200mm f/2.8 telephoto lens", "70-200mm 望遠ズーム"),
        ("400mm super-telephoto lens", "400mm 超望遠"), ("24mm tilt-shift lens", "24mm ティルトシフト"),
        ("anamorphic lens", "アナモルフィック"), ("Cooke S4 cinema prime lens", "クック S4（シネマ単焦点）"),
        ("vintage Helios 44-2 lens, swirly bokeh", "ヘリオス 44-2（ぐるぐるボケ）"),
        ("Petzval 85mm lens, swirly bokeh", "ペッツバール 85mm"), ("Canon 50mm f/0.95 dream lens", "キヤノン 50mm f/0.95"),
        ("soft focus lens", "ソフトフォーカス"),
    ],
    "texture": [
        ("clean smooth skin with natural texture", "なめらかで自然な肌"), ("natural skin texture with visible pores", "毛穴まで見える自然な肌"),
        ("realistic skin texture, fine details", "リアルな肌と細部"), ("soft matte skin", "マットな肌"),
        ("dewy glowing skin", "つやのある肌"), ("freckles and natural imperfections", "そばかす・自然なむら"),
        ("airbrushed flawless skin", "修整した肌"), ("porcelain skin", "陶器のような肌"),
        ("ultra-detailed textures, sharp focus", "細部まで鮮明"), ("crisp fabric texture", "布の質感がはっきり"),
        ("soft dreamy glow", "ふんわり柔らかい"), ("high clarity, micro-contrast", "くっきり（高精細）"),
    ],
    "light": [
        ("balanced ambient lighting", "バランスのよい環境光"), ("soft natural window light", "窓からの柔らかい光"),
        ("golden hour sunlight", "夕方の金色の光"), ("blue hour light", "ブルーアワー"), ("overcast diffused light", "曇りの拡散光"),
        ("harsh midday sun", "真昼の強い日差し"), ("soft studio softbox lighting", "スタジオのソフトボックス"),
        ("Rembrandt lighting", "レンブラントライト"), ("butterfly lighting", "バタフライライト"), ("split lighting", "スプリットライト"),
        ("rim lighting", "リムライト（輪郭光）"), ("backlighting", "逆光"), ("high-key lighting", "ハイキー（明るく白い）"),
        ("low-key lighting", "ローキー（暗く陰影）"), ("dramatic chiaroscuro lighting", "強い明暗"),
        ("cinematic motivated lighting", "映画的な照明"), ("volumetric light rays", "光の筋"),
        ("neon lights", "ネオン"), ("teal and orange color grading", "ティール＆オレンジ"),
        ("warm candlelight", "ろうそくの光"), ("warm tungsten practical lights", "電球色の照明"),
        ("direct on-camera flash", "フラッシュ直当て"), ("moonlight", "月明かり"),
    ],
    "film": [
        ("Kodak Portra 400 film", "コダック Portra 400"), ("Kodak Portra 160 film", "コダック Portra 160"),
        ("Kodak Portra 800 film", "コダック Portra 800"), ("Kodak Ektar 100 film", "コダック Ektar 100"),
        ("Kodak Gold 200 film", "コダック Gold 200"), ("Kodak UltraMax 400 film", "コダック UltraMax 400"),
        ("Kodak Ektachrome E100 slide film", "コダック Ektachrome（リバーサル）"), ("Kodachrome 64 film", "コダクローム 64"),
        ("Kodak Tri-X 400 black and white film", "コダック Tri-X 400（白黒）"), ("Kodak T-Max 100 black and white film", "コダック T-Max 100（白黒）"),
        ("Kodak Vision3 500T motion picture film", "コダック Vision3 500T（映画用）"),
        ("Fujifilm Superia 400 film", "富士 SUPERIA 400"), ("Fujicolor C200 film", "富士 C200"),
        ("Fujifilm Pro 400H film", "富士 PRO 400H"), ("Fujifilm Velvia 50 slide film", "富士 Velvia 50（リバーサル）"),
        ("Fujifilm Provia 100F slide film", "富士 PROVIA 100F（リバーサル）"),
        ("Ilford HP5 Plus 400 black and white film", "イルフォード HP5（白黒）"), ("Ilford Delta 3200 black and white film", "イルフォード Delta 3200（白黒）"),
        ("CineStill 800T film, halation", "CineStill 800T（ハレーション）"), ("CineStill 50D film", "CineStill 50D"),
        ("Lomography Color 100 film", "ロモグラフィー Color 100"), ("Polaroid instant film", "ポラロイドフィルム"),
        ("Instax instant film", "チェキのフィルム"), ("expired film, color shifts", "期限切れフィルム（色ずれ）"),
    ],
    "grain": [
        ("subtle film grain", "うっすら粒子"), ("fine film grain", "細かい粒子"), ("heavy film grain", "粗い粒子"),
        ("clean digital image, no grain", "粒子なし（デジタル）"), ("slight sensor noise", "わずかなノイズ"),
        ("high ISO noise", "高感度ノイズ"),
    ],
    "dof": [
        ("shallow depth of field", "浅い"), ("very shallow depth of field, creamy bokeh", "とても浅い（とろけるボケ）"),
        ("background softly blurred", "背景がやわらかくぼける"), ("moderate depth of field", "ほどほど"),
        ("deep depth of field, everything in focus", "深い（全体にピント）"), ("macro, razor-thin focus", "マクロ（ピントがごく薄い）"),
        ("tilt-shift miniature effect", "ミニチュア風（ティルトシフト）"),
    ],
}
LOOK_PARTS_NAME = "組み合わせ"


def compose_look_parts(parts: dict[str, str] | None) -> str:
    """撮影の組み合わせ {項目: 英語の語句} → 撮影プリセットの文。候補にない値は使わない。"""
    parts = parts if isinstance(parts, dict) else {}
    out = []
    for key, options in LOOK_PARTS.items():
        value = str(parts.get(key) or "").strip()
        if value and any(value == v for v, _ in options):
            out.append(value)
    return ", ".join(out)

# 撮影の語 → 文法の通った英文 (自然文の出力先すべてと
# Anima の文の部分、撮影プリセットにも使う)。プリセットは config.json で自由に書けるので、語句を 1 つずつ写り・カメラ・レンズ・
# フィルム・光・質感・粒子・被写界深度・色に振り分け、決まった型の 2 文にする。振り分けられない語は 2 文目に with で足す。
_LOOK_FILM_WORDS = ("portra", "ektar", "kodak", "kodachrome", "ektachrome", "tri-x", "t-max", "vision3", "superia", "fujicolor",
                    "velvia", "provia", "pro 400h", "ilford", "hp5", "delta 3200", "cinestill", "lomography", "instant film",
                    "expired film", "35mm film", "120 film", "motion picture film")


def _look_kind(p: str) -> str:
    """撮影の語句 1 つの種類。"""
    s = p.lower()
    if "grain" in s or "noise" in s:
        return "grain"
    if "depth of field" in s or "bokeh" in s or "blurred" in s or "tilt-shift miniature" in s or "focus" in s and "lens" not in s:
        return "dof"
    if "film still" in s or "footage" in s or s.endswith(("photographic", " style")):
        return "style"
    if "flash" in s and "photo" not in s:
        return "light"
    if any(re.search(r"\b" + re.escape(w) + r"\b", s) for w in _LOOK_FILM_WORDS) or s.endswith(" film") and "camera" not in s:
        return "film"
    if s.startswith("shot on") or "camera" in s:
        return "camera"
    if "lens" in s or re.search(r"\b\d+(\.\d+)?mm\b", s) or re.fullmatch(r"f/\d+(\.\d+)?", s) or "anamorphic" in s or "fisheye" in s:
        return "lens"
    if "grading" in s or "colors" in s or "colours" in s or "color" in s or "tones" in s:
        return "color"
    if "photo" in s or "snapshot" in s or "photograph" in s or s.endswith(" style"):
        return "style"
    if "light" in s or "lit" in s.split() or "flash" in s or "sun" in s or "golden hour" in s or "blue hour" in s \
            or "neon" in s or "chiaroscuro" in s or s in ("backlit", "moonlight"):
        return "light"
    if "skin" in s or "texture" in s or "detail" in s or "clarity" in s or "glow" in s or "anatomy" in s or "imperfection" in s:
        return "texture"
    return "other"


def _a(np: str) -> str:
    """名詞句に a / an を付ける (数字は読み方で: 8mm → an 8mm)。"""
    word = np.split()[0] if np.split() else ""
    first = word.lower()
    if first in ("a", "an", "the") or not first:
        return np
    vowel = first[0] in "aeiou" or re.match(r"(8|11|18|80|8\d)(\b|mm|\.)", first) or first in ("f/8", "hdr")
    if word.isupper() and len(word) > 1 and word[0] in "FHLMNRSX" and word != "RAW":   # 頭字語: an RGB, an SLR (RAW は語)
        vowel = True
    return ("an " if vowel else "a ") + np


def _and(items: list[str]) -> str:
    items = [x for x in items if x]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1] if items else ""


def look_prose(prompt: str) -> str:
    """撮影の文 (カンマ区切りの語) → 2 文ほどの英文。例: "Cinematic photo, shot on Fujifilm X-T4, 50mm f/1.8 lens, clean smooth
    skin with natural texture, balanced ambient lighting, subtle film grain, shallow depth of field" →
    "A cinematic photo shot on a Fujifilm X-T4 with a 50mm f/1.8 lens. Clean smooth skin with natural texture, lit by balanced
    ambient lighting, with subtle film grain and a shallow depth of field." """
    groups: dict[str, list[str]] = {}
    for p in look_phrases(prompt):
        groups.setdefault(_look_kind(p), []).append(p.strip().rstrip("."))
    style = groups.get("style", [])
    head = ""
    if style:
        s = re.sub(r"\bphotography$", "photograph", style[0])
        s = s if s[:3].isupper() and s.split()[0].isupper() else s[:1].lower() + s[1:]   # RAW photo は大文字のまま
        head = s if s.lower().endswith("footage") else _a(s)
    first: list[str] = []
    cams = groups.get("camera", [])
    for c in cams[:1]:
        rest = c[8:] if c.lower().startswith("shot on ") else c
        first.append("shot on " + (rest if rest[:1].isdigit() else _a(rest)))
    lenses = groups.get("lens", [])
    lens_words = [x for x in lenses if not re.fullmatch(r"f/\d+(\.\d+)?", x.lower())]
    apertures = [x for x in lenses if re.fullmatch(r"f/\d+(\.\d+)?", x.lower())]
    if lens_words:
        first.append("with " + _and([_a(x if "lens" in x.lower() or "mm" not in x.lower() else x + " lens") for x in lens_words]))
    if apertures:
        first.append("at " + apertures[0])
    films = [re.sub(r"^shot on ", "", x, flags=re.I) for x in groups.get("film", [])]
    if films:
        first.append(("on " if cams else "shot on ") + _and(films))
    if not head and first:
        head = "a photo"
    sentences = []
    if head:
        sentences.append(" ".join([head] + first))
    extra_styles = style[1:]
    second: list[str] = []
    second += groups.get("texture", [])
    lights = [x for x in groups.get("light", []) if x.lower() != "backlit"]
    if lights:
        second.append("lit by " + _and(lights))
    if len(lights) != len(groups.get("light", [])):
        second.append("backlit")
    withs = []
    withs += groups.get("grain", [])
    withs += [x if x.lower().startswith(("a ", "an ", "the ", "background", "macro", "everything")) or "depth of field" not in x.lower()
              else _a(x) for x in groups.get("dof", [])]
    withs += groups.get("color", [])
    withs += groups.get("other", [])
    withs = [_a(x) if x.lower().endswith(("effect", "background", "backdrop")) else x for x in withs]
    if withs:
        second.append("with " + _and(withs))
    for x in extra_styles:   # 2 つ目からの写り: "documentary style" → in a documentary style
        second.append("in " + _a(x if x.lower().endswith("style") else x + " style"))
    if second:
        sentences.append(", ".join(second))
    out = " ".join(s[:1].upper() + s[1:] + "." for s in sentences if s)
    return re.sub(r"\s+", " ", out).strip()


def look_in_text(prompt: str, text: str) -> bool:
    """撮影の語が本文 (VLM の自然文) にすでに書かれているか。カメラ・レンズ・フィルムの名前は全部、なければ語の半分以上。"""
    low = (text or "").lower()
    phrases = look_phrases(prompt)
    names = [re.sub(r"^shot on (an? )?|\s+(lens|film)$", "", p.lower()).strip() for p in phrases
             if _look_kind(p) in ("camera", "lens", "film")]
    if names:
        return all(n in low for n in names)
    return bool(phrases) and sum(p.lower() in low for p in phrases) * 2 >= len(phrases)


# 写真の撮影情報 (EXIF)。⑦ 作風・質感の画像から、⑦ が空なら ① (1 枚モードの画像) から読む。
# 記録された値なので推定ではない。撮影プリセットの文と同じ形 (カンマ区切りの語) にして、タグ形式は語のまま、自然文は look_prose で文にする。
# 位置情報 (GPS)・撮影日時・機器の製造番号・持ち主の名前は読まない (プロンプトにもメモにも出さない)
_EXIF_IFD0 = {"make": 0x010F, "model": 0x0110}
_EXIF_SUB = {"exposure": 0x829A, "fnumber": 0x829D, "iso": 0x8827, "flash": 0x9209, "focal": 0x920A, "focal35": 0xA405,
             "lens": 0xA434}


def _exif_text(v: Any) -> str:
    if isinstance(v, bytes):
        v = v.decode("utf-8", "ignore")
    return re.sub(r"\s+", " ", str(v or "").replace("\x00", " ")).strip() if isinstance(v, (str, bytes)) else ""


def _exif_num(v: Any) -> float | None:
    if isinstance(v, (tuple, list)):
        v = v[0] if v else None
    try:
        x = float(v)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return x if x > 0 and x == x and x != float("inf") else None


def camera_exif(path: Path | None) -> dict[str, Any] | None:
    """写真の撮影情報 (カメラ・レンズ・焦点距離・絞り・シャッター速度・ISO・フラッシュ)。無ければ None。
    SNS から保存した写真・スクリーンショット・生成画像には残っていないことが多い。"""
    if not path:
        return None
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            sub = exif.get_ifd(0x8769)
    except Exception:  # noqa: BLE001 - 壊れた EXIF は読まない
        return None
    info: dict[str, Any] = {k: _exif_text(exif.get(t)) for k, t in _EXIF_IFD0.items()}
    info["lens"] = _exif_text(sub.get(_EXIF_SUB["lens"]))
    for k in ("exposure", "fnumber", "iso", "focal", "focal35"):
        info[k] = _exif_num(sub.get(_EXIF_SUB[k]))
    flash = sub.get(_EXIF_SUB["flash"])
    info["flash"] = bool(int(flash) & 1) if isinstance(flash, int) else None
    if not (info["model"] or info["focal"] or info["fnumber"]):
        return None
    return info


def _exif_camera(info: dict[str, Any]) -> str:
    make, model = info.get("make") or "", info.get("model") or ""
    word = make.split()[0] if make else ""
    if model and word and word.lower() not in model.lower():
        return f"{word} {model}"
    return model or make


def _exif_small_sensor(info: dict[str, Any]) -> bool:
    """スマートフォン・コンパクトカメラ (35mm 換算が実際の焦点距離の 3 倍以上 = 小さいセンサー)。レンズ名は書かない
    (iPhone の「back triple camera 6.86mm f/1.78」のような名前はレンズの種類を表さない)。"""
    f, f35 = info.get("focal"), info.get("focal35")
    return bool(f and f35 and f35 / f >= 3)


def _fmt_g(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".")


def exif_look_prompt(info: dict[str, Any]) -> str:
    """撮影情報 → 撮影プリセットと同じ形の語 (例: shot on Canon EOS R5, 85mm f/1.2 lens, 1/200s shutter speed, ISO 100,
    flash photography)。焦点距離は 35mm 換算があればそちら (写る範囲が分かる値)。"""
    parts = []
    cam = _exif_camera(info)
    if cam:
        parts.append(f"shot on {cam}")
    focal = info.get("focal35") or info.get("focal")
    lens = " ".join(x for x in (f"{round(focal)}mm" if focal else "", f"f/{_fmt_g(info['fnumber'])}" if info.get("fnumber") else "") if x)
    if lens:
        parts.append(lens + " lens")
    t = info.get("exposure")
    if t:
        parts.append((f"1/{round(1 / t)}s" if t < 1 else f"{_fmt_g(t)}s") + " shutter speed")
    if info.get("iso"):
        parts.append(f"ISO {round(info['iso'])}")
    if info.get("flash"):
        parts.append("flash photography")
    return ", ".join(parts)


def exif_prose(info: dict[str, Any]) -> str:
    """撮影情報 → 自然文の出力先に足す 1 文 (例: A photograph shot on Canon EOS R5 with an 85mm f/1.2 lens at 1/200s and
    ISO 100.)。カメラ名には冠詞を付けない (撮影プリセットの「shot on Fujifilm X-T4」と同じ書き方。SONY などの読み方で迷わない)。"""
    head = "A flash photograph" if info.get("flash") else "A photograph"
    words = [head]
    cam = _exif_camera(info)
    if cam:
        words.append(f"shot on {cam}")
    focal = info.get("focal35") or info.get("focal")
    lens = " ".join(x for x in (f"{round(focal)}mm" if focal else "", f"f/{_fmt_g(info['fnumber'])}" if info.get("fnumber") else "") if x)
    if lens:
        words.append(f"with {_a(lens + ' lens')}")
    tail = []
    t = info.get("exposure")
    if t:
        tail.append(f"1/{round(1 / t)}s" if t < 1 else f"{_fmt_g(t)}s")
    if info.get("iso"):
        tail.append(f"ISO {round(info['iso'])}")
    if tail:
        words.append("at " + " and ".join(tail))
    return " ".join(words) + "."


def exif_summary_ja(info: dict[str, Any]) -> str:
    """分析メモに出す撮影情報 (記録された値をそのまま)。"""
    out = []
    cam = _exif_camera(info)
    if cam:
        out.append(f"カメラ {cam}")
    if info.get("lens") and not _exif_small_sensor(info):
        out.append(f"レンズ {info['lens']}")
    if info.get("focal"):
        f35 = info.get("focal35")
        out.append(f"焦点距離 {_fmt_g(info['focal'])}mm" + (f" (35mm 換算 {round(f35)}mm)" if f35 and round(f35) != round(info["focal"]) else ""))
    if info.get("fnumber"):
        out.append(f"絞り f/{_fmt_g(info['fnumber'])}")
    t = info.get("exposure")
    if t:
        out.append("シャッター速度 " + (f"1/{round(1 / t)} 秒" if t < 1 else f"{_fmt_g(t)} 秒"))
    if info.get("iso"):
        out.append(f"ISO {round(info['iso'])}")
    if info.get("flash") is not None:
        out.append("フラッシュ発光" if info["flash"] else "フラッシュなし")
    return " / ".join(out)


# 三面図にする。画像から読んだポーズ・カメラ・背景 (照明を含む)・小物を外し、三面図のタグにする。
# タグは Danbooru wiki で確認: reference_sheet (22,894 件。character_sheet / design_sheet / settei の別名)、turnaround
# (4,931 件。「A character, often full body, shown from multiple views about their vertical axis」、multiple_views を含意)、
# multiple_views (277,053 件。「Images with multiple views should not be tagged solo」→ solo は外す。1girl は wiki の例のまま残す)、
# 正面 straight-on (63,181 件)、横 from_side、後ろ from_behind。白い無地の背景は simple_background / white_background
# 並びは前 → 後ろ → 横
TURNAROUND_TAGS = ["reference sheet", "turnaround", "multiple views", "full body", "standing", "straight-on", "from behind",
                   "from side", "simple background", "white background"]
# 自然文の出力先に足す文。各社とも公式に三面図の書き方の説明は無いので、一般的な書き方 (肯定文で見え方を書く)
TURNAROUND_SENTENCE = ("Character turnaround reference sheet: the same character shown three times side by side in the same "
                       "standing pose, full body, as a front view, a back view and a side view, on a plain white background.")
TURNAROUND_VLM = (
    "The result will be turned into a character turnaround reference sheet (front, back and side views, standing, plain "
    "white background) afterwards. In natural_prompt, subject, extra_tags and pose_tags, do not describe the pose or "
    "gesture, the camera angle or framing, the background, location or lighting, or objects that are held or placed."
)
# 人物の特徴を入れない (LoRA などでキャラを指定するとき。髪と目・顔立ち・キャラ名 / 作品名・肌の色を外す。
# 服装・体格・ポーズ・表情は残す)
_IDENTITY_FEATURE_RE = re.compile(
    r"(^|_)(hair|haired|eyes|ponytail|twintails|braid|braids|bangs|ahoge|bun|buns|sidelocks|pupils|eyebrows|hairstyle)($|_)")
IDENTITY_FACE_TAGS = {
    "tsurime", "tareme", "jitome", "sanpaku", "thick_eyebrows", "facial_hair", "beard", "stubble", "mustache", "freckles",
    "mole", "mole_under_eye", "mole_under_mouth", "eyelashes", "long_eyelashes", "heterochromia", "facial_mark",
    "bishounen", "slit_pupils", "symbol-shaped_pupils",
}
SKIN_COLOR_TAGS = {
    "dark_skin", "very_dark_skin", "dark-skinned_female", "dark-skinned_male", "pale_skin", "tan", "tanned", "deep_skin",
    "colored_skin", "blue_skin", "green_skin", "grey_skin", "red_skin", "purple_skin", "white_skin", "black_skin",
    "pink_skin", "yellow_skin", "orange_skin", "brown_skin",
}
NO_IDENTITY_VLM = (
    "The character's identity is given separately (for example by a LoRA). In natural_prompt, subject, extra_tags, "
    "pose_tags and outfit_tags, never mention the character's name or series, the hair (colour, length, style), eye "
    "colour, facial features or skin colour. Describe the outfit, body type, pose, expression and scene only."
)


# 作風の特徴を入れない (画風 LoRA を使うとき。画風・技法のタグ、絵か写真か (と、それに合わせた品質タグ・
# ネガティブの定番セット)、年代、作品名・作者名を外す)
MEDIUM_WORD_SET = set().union(*_MEDIUM_WORDS.values()) | {"2d", "cgi", "screencap", "artstyle", "pixel"}
NO_STYLE_VLM = (
    "The art style is given separately (for example by a style LoRA). In natural_prompt, subject, extra_tags and the "
    "other fields, never mention the art style, medium or rendering (anime, illustration, photo, realistic, 3d, "
    "painting, cel shading, watercolor and so on), an era or decade of style, or the names of artists or series. "
    "Also do not describe skin or surface texture (wet, shiny, glossy or detailed skin, dew drops), lips or makeup."
)


def medium_word(tag: str) -> bool:
    """絵か写真か (画材・描き方) を表す語を含むか (realistic / anime / 3d render / illustration など)。"""
    return bool(set(re.findall(r"[a-z0-9-]+", unescape_parens(tag).lower().replace("_", " "))) & MEDIUM_WORD_SET)


# 質感の描き込み (肌のつや・濡れ・唇・化粧)。塗りの密度に効くので「作風の特徴を入れない」で外す。
# 汗 (sweat)・赤面 (blush)・そばかすは画面の内容・表情なので外さない
TEXTURE_TAGS = {
    "wet_skin", "shiny_skin", "oily_skin", "glistening_skin", "sweaty_skin", "dew_drops", "dewdrop", "water_drop_on_skin",
    "lips", "glossy_lips", "shiny_lips", "lipstick", "makeup", "eyeshadow", "eyeliner", "mascara", "lip_gloss",
    "pink_lips", "red_lips", "nude_lips", "skin_texture", "detailed_skin", "realistic_skin", "skin_pores", "pores",
    "subsurface_scattering", "glossy_skin", "porcelain_skin", "smooth_skin", "dewy_skin",
}
_TEXTURE_WORDS = {"glossy", "shiny", "oily", "wet", "dewy", "glistening", "detailed", "realistic", "textured", "smooth",
                  "porcelain", "luminous", "radiant", "pores", "subsurface"}


def texture_tag(tag: str) -> bool:
    """質感の描き込みのタグか (VLM の言い回し detailed skin texture / glossy lips なども)。"""
    n = _tag_name(tag)
    if n in TEXTURE_TAGS or n.endswith(("_lipstick", "_eyeshadow", "_makeup")):
        return True
    words = set(n.split("_"))
    return bool(words & {"skin", "lips", "lip"}) and bool(words & _TEXTURE_WORDS)


def style_removes(tag: str) -> bool:
    """「作風の特徴を入れない」で外すタグか: 画風・技法・年代と、絵か写真かを表す語、質感の描き込み。"""
    n = _tag_name(tag)
    return classify_extra_tag(n) == "style" or bool(ERA_TAG_RE.match(n)) or medium_word(n) or texture_tag(n)


# --------------------------------------------------------------------------
# 仕上げの書き直し (文章の出力先。VLM を使うとき)
# --------------------------------------------------------------------------
# 文章の出力先の本文は、条件によって作り方が違う: 1 枚だけなら VLM が画像を見て書いた文、複数枚モード・2 人以上なら
# 枠ごとの内容を決まった順に並べた文 (VLM の文が無い枠はタグの並びのまま)。並べた文は「Subject: ... Outfit: ...」の
# ような見出しとタグの列で、文章を読む生成モデルには弱い。集めた内容を VLM にもう一度渡して (画像は見せない)、
# 出力先の書き方で 1 つの文章に書き直させる。VLM が画像を見て書いた文 (1 枚だけ) は書き直さない (呼び出しが増えて遅くなる)。
# 書き直しが内容を落としたり、外したはずの要素を書いたりしたら使わず、元の文のままにする
POLISH_SYSTEM = (
    "You are given a draft prompt for an image generation model. The draft was assembled from separate notes about one "
    "image (subject, outfit, pose, objects, setting, lighting, camera, medium and style), so parts of it are labels and "
    "comma-separated tag lists. Rewrite it as the final prompt in fluent English.\n"
    "Rules:\n"
    "- Keep every fact of the draft: the people and how many there are, their age, hair, eyes, face, body, clothing with "
    "its colours and details, pose, objects, setting, lighting, camera framing, medium and style. Do not drop anything.\n"
    "- Keep every name of a character, a series or an artist exactly as the draft writes it. Names are not tag jargon: "
    "do not translate them, do not replace them with a description, and do not drop them.\n"
    "- Do not invent anything that is not in the draft.\n"
    "- Replace labels such as 'Subject:' or 'Outfit:' and tag lists with natural sentences.\n"
    "- Replace Danbooru tag jargon with plain descriptive English (1girl -> a woman or a girl as the draft says, cowboy "
    "shot -> framed from the thighs up, ahoge -> a single upright strand of hair, and so on).\n"
    "- Keep text inside double quotes exactly as it is.\n"
    "- Do not add a medium or an art style that the draft does not state, and no quality words such as masterpiece.\n"
    "- Output only the prompt: no explanations, no headings, no markdown."
)
POLISH_STYLE_FIRST = "Begin with the medium and the art style or look, then describe the content."
def _mentions_loose(text: str, phrase: str) -> bool:
    """phrase の語が全部 text に出てくるか。並びは問わない (long hair は long pink hair に出ている) し、複数形の s も同じとみなす。
    文の中の語の並びはタグと違うので、「書かれているか」はこちらで確かめる (厳しく見ると、書いてあるのに足させてしまう)。"""
    words = set(re.findall(r"[a-z0-9]+", unescape_parens(text).lower()))
    words |= {w[:-1] for w in words if w.endswith("s")}
    need = re.findall(r"[a-z0-9]+", unescape_parens(phrase).lower())
    return bool(need) and all(w in words or w.rstrip("s") in words for w in need)


def polish_guidance(target: str) -> str:
    """出力先ごとの書き方 (メインの VLM への指示から、JSON の欄名を外したもの)。"""
    text = TARGET_VLM_GUIDANCE.get(target, "")
    return re.sub(r"^natural_prompt:\s*", "", text).strip()


def polish_problem(draft: str, text: str, keep: list[str], forbidden: list[str], names: list[str] | None = None) -> str:
    """書き直した文を使えない理由 (使えるなら空文字)。keep は本文に入れるタグ、forbidden は外したはずのタグ、
    names はキャラ名・作品名・作者名 (元の文にあった名前が 1 つでも落ちていたら使わない。名前は言い換えが利かない)。"""
    words = len(re.findall(r"[A-Za-z]+", text))
    draft_words = len(re.findall(r"[A-Za-z]+", draft))
    if words < max(8, draft_words * 0.5):
        return "短すぎます (内容が落ちています)"
    if words > max(120, draft_words * 3):
        return "長すぎます (元に無い内容が足されています)"
    if re.search(r"(?m)^\s*(?:#|\*\*|- )|\b(?:Here is|Here's|Note:|Explanation:)", text):
        return "説明や見出しが入っています"
    lost_quotes = [q for q in re.findall(r'"[^"]+"', draft) if q not in text]
    if lost_quotes:
        return "画像の文字 (" + ", ".join(lost_quotes)[:80] + ") が変わっています"
    lost_names = [t for t in mentions_any(draft, list(names or []))
                  if not _mentions_loose(text, t) and not _mentions_loose(text, plain_english(t))]
    if lost_names:
        return "キャラ名・作品名 (" + ", ".join(dedup(lost_names))[:120] + ") が落ちています"
    leaked = mentions_any(text, [t for t in forbidden if not mentions_any(draft, [t])])
    if leaked:
        return "外した要素 (" + ", ".join(dedup(leaked))[:120] + ") を書いています"
    had = mentions_any(draft, keep)
    lost = [t for t in had if not _mentions_loose(text, t) and not _mentions_loose(text, plain_english(t))]
    if len(lost) > max(3, len(had) * 0.4):
        return "内容が落ちています (" + ", ".join(dedup(lost))[:120] + ")"
    return ""


# --------------------------------------------------------------------------
# 画風を変える (MOSS-Image)
# --------------------------------------------------------------------------
# 元の画像とは違う画風で作る: 実写 / アニメ風 / 漫画風 / 水彩 / 厚塗り / 油絵 / 鉛筆画。
# ⑦ 作風・質感に見本を入れても元の絵柄が勝つことがある。原因は 2 つ: 人物の説明が元の画像から書かれること (元の画風の
# 言い方のまま) と、画風が文の後ろに回ること。種類を選ぶと、先頭でその画風だと言い切り、元の画風の語を外すか言い換え、
# その画風にしかない描き方・質感の文を最後に足す。⑦ が無くても使える。⑦ の画像が選んだ種類と同じ系統 (写真 / 絵画 / 絵) なら、
# ⑦ から読んだ文 (写真なら撮り方、絵なら塗り方) を先頭の直後に置く。MOSS-Image での効き目は実機で確かめていない
PHOTOREAL_VLM = (
    "The result must be a real photograph of real people taken with a camera, whatever the reference image is (anime, "
    "illustration, painting or 3D). In natural_prompt, subject and every other text field, describe the person as a real "
    "human being as they would look in a photograph: natural human facial proportions and eye size, real skin, real hair "
    "(an unnatural hair colour is dyed hair), real fabrics and materials. Translate drawn features into their real-life "
    "counterpart instead of copying them (for example huge sparkling eyes become clear blue eyes). Never say that the "
    "source is a drawing, and never use the words anime, manga, cartoon, illustration, drawing, painting, cel shading, "
    "line art, 2D, 3D, render, CGI, digital art, stylized, realistic or photorealistic. When the image shows a named "
    "character, keep the character's name and describe a real person who is that character in real life. Begin "
    "natural_prompt with 'A photograph of'."
)
RESTYLE_VLM = (
    "The result will be made as {what}, whatever the reference image is (a photograph, an anime illustration, a painting "
    "or 3D). In natural_prompt, subject and every other text field, describe the content only - the people, their "
    "clothing, pose, objects, the setting and the light - and never the medium or rendering of the reference image: do "
    "not use the words photograph, photo, realistic, photorealistic, anime, manga, cartoon, illustration, painting, "
    "watercolor, oil, sketch, cel shading, line art, 2D, 3D, render or CGI. Begin natural_prompt with '{start} of'."
)
PHOTOREAL_PEOPLE = (
    "Real human skin with visible pores, fine vellus hair and slight natural unevenness, natural human facial proportions "
    "and eye size, individually visible strands of hair, real fabric with weave and wrinkles, natural light with true "
    "optical depth of field."
)
PHOTOREAL_SCENE = "Real materials and surfaces with their natural wear, natural light with true optical depth of field."
# 種類ごとの書き方。medium = 絵か写真かの判定に入れる値 (⑦ がこれと同じなら ⑦ の文を使う)、noun = 画材の語の言い換え先、
# lead = 先頭の文、tail = 最後に足す描き方・質感の文、negative = MOSS-Image のネガティブ欄 (公式の既定は空。CFG 1 の高速 LoRA
# では効かない)、names = キャラ名を本文に入れるか (どの種類も入れる。実写でも、誰の実写かは名前で決まる)
RESTYLES: dict[str, dict[str, Any]] = {
    "photo": {
        "medium": "photo", "noun": "photograph", "names": True,
        "lead": "A real photograph of a real person, taken with a camera.",
        "lead_many": "A real photograph of real people, taken with a camera.",
        "lead_scene": "A real photograph, taken with a camera.",
        "tail": PHOTOREAL_PEOPLE, "tail_scene": PHOTOREAL_SCENE,
        "negative": ["anime", "illustration", "drawing", "painting", "cartoon", "manga", "3d render", "cgi", "digital art",
                     "doll", "plastic skin"],
    },
    "anime": {
        "medium": "anime", "noun": "anime illustration", "names": True,
        "lead": "An anime illustration with clean line art and cel shading.",
        "tail": "Clean, even outlines, flat cel-shaded colours with one or two shadow tones and crisp highlights, large "
                "expressive eyes and simplified facial features.",
        "negative": ["photograph", "photorealistic", "3d render", "cgi", "painterly", "sketch"],
    },
    "manga": {
        "medium": "anime", "noun": "black-and-white manga illustration", "names": True,
        "lead": "A black-and-white manga illustration drawn in ink.",
        "tail": "Black ink line art on white paper, screentone and hatching for the shading, no colour.",
        "negative": ["colour", "photograph", "3d render", "cgi", "painting", "grey wash"],
    },
    "watercolor": {
        "medium": "painting", "noun": "watercolour painting", "names": True,
        "lead": "A watercolour painting on textured paper.",
        "tail": "Transparent washes of colour that bleed into each other, visible paper grain, soft edges and light pencil "
                "lines under the paint.",
        "negative": ["photograph", "3d render", "cgi", "cel shading", "flat colour", "digital art"],
    },
    "thick": {
        "medium": "painting", "noun": "digital painting", "names": True,
        "lead": "A digital painting with thick, opaque painterly brushwork.",
        "tail": "Visible brush strokes, blended opaque colours, soft edges without line art, painted light and shadow.",
        "negative": ["photograph", "3d render", "cgi", "line art", "cel shading", "flat colour"],
    },
    "oil": {
        "medium": "painting", "noun": "oil painting", "names": True,
        "lead": "An oil painting on canvas.",
        "tail": "Thick layered oil paint with visible brush strokes and canvas texture, rich blended colours and soft "
                "painted edges.",
        "negative": ["photograph", "3d render", "cgi", "line art", "cel shading", "anime"],
    },
    "pencil": {
        "medium": "anime", "noun": "pencil drawing", "names": True,
        "lead": "A pencil drawing on paper.",
        "tail": "Graphite pencil lines with hatching and soft smudged shading on white paper, no colour.",
        "negative": ["colour", "photograph", "3d render", "cgi", "painting", "ink"],
    },
}
# タグから外す語: 絵か写真かと描き方を表す語 (camera は持ち物のタグにもあるので見ない)。実写にするときは写真の語は残す
_RESTYLE_PHOTO_WORDS = {"photo", "photograph", "photographic"}
_RESTYLE_DROP_WORDS = (set().union(*_MEDIUM_WORDS.values()) - {"camera"}
                       | {"2d", "cgi", "screencap", "artstyle", "pixel", "photorealistic", "realistic", "realism"})
_PR_DRAWN = r"(?:anime|manga|cartoon|comic|toon|cel[- ]shaded)"
# 絵の画材を表す名詞の句 → その種類の語 (anime-style illustration / digital painting / 3D rendering / an anime image など)
_RESTYLE_NOUN_RE = re.compile(
    r"\b(?:(?:" + _PR_DRAWN + r"(?:[- ]?style[d]?)?|digital|2d|3d|cgi?|rendered|illustrated|hand[- ]drawn|painterly|"
    r"styli[sz]ed|(?:photo|hyper|ultra|semi)[- ]?realistic|realistic)[ -]+)*"
    r"(?:illustration|drawing|artwork|digital painting|digital art|rendering|render|screencap|screenshot|cartoon|sketch)s?\b"
    r"|\b(?:" + _PR_DRAWN + r"|digital|oil|watercolou?r|2d|3d|cgi?|rendered|illustrated|styli[sz]ed|hand[- ]drawn)[ -]"
    r"(?:painting|art|image|picture|portrait|scene|graphic)s?\b"
    r"|\bpainting(?= of\b)", re.IGNORECASE)
# 写真を表す名詞の句 (実写以外にするとき) → その種類の語
_RESTYLE_PHOTO_NOUN_RE = re.compile(
    r"\b(?:(?:real|real-life|candid|professional|raw|(?:photo|hyper|ultra|semi)[- ]?realistic|realistic)[ -]+)*"
    r"(?:photograph|photo|snapshot)s?\b", re.IGNORECASE)
# 画風の言い方の句 (in an anime style、in a clean cel-shaded anime style など) → 外す。
# 「is rendered in a ... style with soft lighting」は「has soft lighting」にし、何も残らない文 (The image.) は消す
_PR_STYLE_WORDS = (r"(?:" + _PR_DRAWN + r"|painterly|illustrated|watercolou?r|sketch|pixel[- ]art|photo-?realistic|realistic|"
                   r"photographic|3d|cgi?)(?:[- ](?:" + _PR_DRAWN + r"|art))*[- ]?style")
_RESTYLE_IS_STYLE_RE = re.compile(
    r"\b(?:is|are) (?:rendered|drawn|painted|done|illustrated|made) in (?:an?|the) (?:[\w-]+ ){0,3}?" + _PR_STYLE_WORDS
    + r"\b( with\b)?", re.IGNORECASE)
_RESTYLE_STYLE_RE = re.compile(  # コンマで挟まれた句 (, rendered in an anime style,) は後ろのコンマも外す
    r"(,\s*)?\b(?:rendered |drawn |painted |done )?in (?:an?|the) (?:[\w-]+ ){0,3}?" + _PR_STYLE_WORDS + r"\b(?(1),?)",
    re.IGNORECASE)
_RESTYLE_EMPTY_RE = re.compile(
    r"(?:(?<=[.!?] )|^)(?:The|This) (?:image|picture|scene|artwork|photograph|style)(?: (?:is|are|has))?\s*\.\s*", re.IGNORECASE)
# 残った絵・CG の語 → 外す
_RESTYLE_WORD_RE = re.compile(
    r"\b(?:anime|manga|cartoon|toon|chibi)(?:[- ]?(?:style[d]?|like|esque|inspired))?\b|\bcartoonish\b|\bcel[- ]shad(?:ed|ing)\b"
    r"|\bline[- ]?art\b|\bstyli[sz]ed\b|\bhand[- ]drawn\b|\billustrated\b|\bpainterly\b|\b[23]d\b|\bcgi?\b"
    r"|\b(?:digitally )?rendered\b", re.IGNORECASE)
# realistic / photorealistic: 実写にするときは real に (「本物らしく描いた」という語で、CG の説明にも使われる)、ほかの種類では外す
_RESTYLE_REAL_RE = re.compile(r"\b(?:(?:photo|hyper|ultra|semi)[- ]?)?realistic\b|\bphotoreal\b|\bphotographic\b", re.IGNORECASE)


def restyle_vlm(kind: str) -> str:
    """VLM への指示: 元の画像の画風を書かせず、選んだ種類の画像として書かせる。"""
    if kind == "photo":
        return PHOTOREAL_VLM
    noun = RESTYLES[kind]["noun"]
    return RESTYLE_VLM.format(what=_with_article(noun, capital=False), start=_with_article(noun))


def restyle_removes(tag: str, kind: str, keep_style: bool = False) -> bool:
    """「画風を変える」で外すタグか: 絵か写真かと描き方を表す語、年代、画風・技法のタグ。keep_style なら、
    ⑦ から読んだ画風のタグ (絵か写真かを表す語を含まないもの) は残す。"""
    n = _tag_name(tag)
    drop = _RESTYLE_DROP_WORDS - (_RESTYLE_PHOTO_WORDS if kind == "photo" else set())
    if set(re.findall(r"[a-z0-9-]+", n.replace("_", " "))) & drop or ERA_TAG_RE.match(n):
        return True
    return classify_extra_tag(n) == "style" and not keep_style


def restyle_text(text: str, kind: str, people: int, shot: str = "") -> tuple[str, list[str]]:
    """本文を、選んだ種類の画像として書き直す。戻り値: (本文, 外した・言い換えた語の一覧)。
    先頭にその種類だと言い切る文、その直後に ⑦ から読んだ文 (shot。本文の中にあれば前へ移す)、最後に描き方・質感の文を置く。
    元の画風の語 (anime-style illustration、photograph など) は、その種類の語にするか外す。people は人数 (0 = 人物なし)。"""
    spec = RESTYLES[kind]
    shot = shot.strip()
    if shot:
        sentence = shot.rstrip(".") + "."
        text = text.replace(sentence, "", 1) if sentence in text else text.replace(shot, "", 1)
        shot = sentence
    lead = spec.get("lead_scene", spec["lead"]) if people < 1 else spec.get("lead_many", spec["lead"]) if people >= 2 else spec["lead"]
    tail = spec.get("tail_scene", spec["tail"]) if people < 1 else spec["tail"]
    noun = spec["noun"]
    changed: list[str] = []

    mark = "@@LOOK@@"  # 言い換え先の語 (anime illustration など) が、後の「絵の語を外す」で削られないよう、最後に入れる

    def to_noun(m: "re.Match[str]") -> str:
        if m.group(0).lower() != noun:
            changed.append(f"{m.group(0).lower()} → {noun}")
        return mark

    def drop(m: "re.Match[str]") -> str:
        changed.append(m.group(0).strip(" ,").lower() + " (外した)")
        return ""

    def is_style(m: "re.Match[str]") -> str:
        changed.append(m.group(0).lower() + (" → has" if m.group(1) else " (外した)"))
        return "has" if m.group(1) else ""

    def real(m: "re.Match[str]") -> str:
        if kind != "photo":
            return drop(m)
        changed.append(m.group(0).lower() + " → real")
        return "Real" if m.group(0)[:1].isupper() else "real"

    body = " ".join(x for x in (shot, text.strip()) if x)
    body = _unquoted_sub(_RESTYLE_IS_STYLE_RE, is_style, body)
    body = _unquoted_sub(_RESTYLE_STYLE_RE, drop, body)
    body = _unquoted_sub(_RESTYLE_NOUN_RE, to_noun, body)
    if kind != "photo":
        body = _unquoted_sub(_RESTYLE_PHOTO_NOUN_RE, to_noun, body)
    body = _unquoted_sub(_RESTYLE_WORD_RE, drop, body)
    body = _unquoted_sub(_RESTYLE_REAL_RE, real, body)
    body = _RESTYLE_EMPTY_RE.sub("", _tidy_prose(body.replace(mark, noun))).strip()
    return " ".join(x for x in (lead, body, tail) if x), dedup(changed)


def _tag_name(tag: str) -> str:
    return _strip_weight(norm_tag(tag)).replace(" ", "_")


def identity_tag(tag: str) -> bool:
    """「人物の特徴を入れない」で外すタグか: 髪・目・顔立ち・肌の色 (髪飾り・表情・ポーズは除く)。キャラ名は別に外す。"""
    n = _tag_name(tag)
    if expression_tag(n) or classify_outfit_tag(n) or classify_pose_tag(n) or n.startswith("looking_"):
        return False  # closed eyes / hair ornament / hand in own hair などは表情・服装・ポーズ
    return n in SKIN_COLOR_TAGS or n in IDENTITY_FACE_TAGS or bool(_IDENTITY_FEATURE_RE.search(n))


def turnaround_removes(tag: str) -> bool:
    """「三面図にする」で外すタグか: ポーズ・カメラ (画角・アングル)・背景と照明・小物、と solo。表情・視線・服装は残す。"""
    n = _tag_name(tag)
    if n == "solo":
        return True
    if expression_tag(n) or n in EXPOSURE_TAGS:
        return False
    if held_object_tag(n) or n in OBJECT_TAGS or classify_subject_tag(n, True) == "object":
        return True
    if classify_outfit_tag(n):
        return False
    if pose_source_tag(n) or classify_extra_tag(n) == "lighting" or classify_subject_tag(n, True) == "background":
        return True
    # VLM の言い回し (park bench など) は、語のどれかが背景・物のタグなら外す (服装は上で残している)
    words = [w for w in re.split(r"[_-]", n) if w]
    return len(words) > 1 and any(classify_subject_tag(w, True) in ("background", "object") for w in words)


STRENGTH_GROUPS = [("person", "人物"), ("pose", "ポーズ"), ("outfit", "服装"), ("body", "体格"), ("props", "小物"),
                   ("background", "背景"), ("style", "画風")]
STRENGTH_WEIGHT = {"strong": 1.2, "weak": 0.8}


def strength_group(name: str, style_ref: bool = False) -> str:
    """強さの調整の枠: reference_category を 7 つにまとめる (持ち物は小物、服で決まる見え方は服装)。
    画風は、⑦ 作風・質感があるか「画風」の強さを選んだとき (呼び出し側が style_ref を True にする) は画風の枠。
    どちらでもなければ以前どおり人物の枠。"""
    category = reference_category(name, style_ref)
    if category == "style" and not style_ref:
        return "person"
    # 表情 (⑧) は強さの欄が無いので人物の枠、視線は以前どおりポーズの枠
    return {"held": "props", "expression": "person", "gaze": "pose"}.get(category, category)


def camera_tags(camera: dict[str, str]) -> dict[str, list[str]]:
    """カメラの指定 → 種類ごとの入れるタグ (指定しない種類は含めない)。"""
    out: dict[str, list[str]] = {}
    for dim, options in CAMERA_OPTIONS.items():
        value = (camera or {}).get(dim) or ""
        if value:
            out[dim] = next((tags for v, _, tags in options if v == value), [])
    return out


def apply_controls(
    tags: TagResult, tag_source: dict[str, str], controls: dict[str, Any] | None,
) -> tuple[TagResult, dict[str, str], list[str], list[str], dict[str, bool]]:
    """ボーン (ポーズのタグ) とカメラの指定を、タグの結果に反映する。
    戻り値: (タグ, 出どころ, 外したタグ, 注意, {"pose": ボーンで置き換えた, "camera": カメラを指定した})。
    置き換えたタグの出どころは "control" (② から取った画角と同じく重みの対象)。"""
    controls = controls or {}
    names = [n for n, _ in tags.general]
    probs = dict(tags.general)
    source = dict(tag_source)
    removed: list[str] = []
    notes: list[str] = []
    added: list[str] = []
    pose = controls.get("pose") or {}
    flags = {"pose": False, "camera": False}
    if pose.get("tags") is not None:
        # ボーンのポーズは姿勢・手足だけを置き換える (視線・画角・向きはカメラや ② のまま、持ち物は小物の担当)
        flags["pose"] = True
        drop = [n for n in names if reference_category(n) == "pose" and classify_extra_tag(n) != "camera"
                and n not in _FRAMING_ANY and not n.startswith("looking_")]
        removed += [n for n in drop if n not in pose["tags"]]
        names = [n for n in names if n not in drop]
        added += [t for t in pose["tags"] if t not in names]
    chosen = camera_tags(controls.get("camera") or {})
    if chosen:
        flags["camera"] = True
        for dim, new in chosen.items():
            drop = [n for n in names if n in CAMERA_DIMENSION_TAGS[dim]]
            removed += [n for n in drop if n not in new]
            names = [n for n in names if n not in drop]
            added += [t for t in new if t not in names and t not in added]
        if (controls.get("camera") or {}).get("direction") == "back":
            if "looking_at_viewer" in names:
                # 後ろ向きで視線がこちら = 振り返り (Danbooru の looking_back)。正面を向かせないように置き換える
                names = [("looking_back" if n == "looking_at_viewer" else n) for n in names]
                source["looking_back"] = "control"
                notes.append("カメラを「後ろ」にしたので、looking at viewer を looking back (振り返り) にしました")
            front = [n for n in names if n in FRONT_ONLY_EXPOSURE]
            if front:
                names = [n for n in names if n not in front]
                removed += front
                notes.append("後ろからの構図なので、正面からしか見えない要素を外しました (残すと正面向きで描かれやすい): "
                             + ", ".join(sd_tag(n) for n in front))
        if (controls.get("camera") or {}).get("direction") == "three_quarter":
            notes.append("斜めの向きは Danbooru 系のタグ一覧に無いので、タグ形式の出力先には書けません (自然文系には書きます)")
    for t in added:
        source[t] = "control"
    # 指定したタグは先頭に置く (1 枚モードは信頼度順なので、末尾だと CLIP の後ろの区画に押し出されて効かない)
    head = [(n, probs.get(n, 1.0)) for n in names if n in ("solo", "no_humans") or n in PERSON_COUNT_TAGS]
    general = head + [(t, 1.0) for t in added if t not in probs] + [(n, probs.get(n, 1.0)) for n in names if (n, probs.get(n, 1.0)) not in head]
    merged = TagResult(tags.rating, tags.rating_conf, general, list(tags.characters), dict(tags.general_probs))
    return merged, source, dedup(removed), notes, flags


def apply_strength(shared: list[str], strength: dict[str, str] | None, target: str, kind: str,
                   style_ref: bool = False) -> tuple[list[str], list[str]]:
    """強さの調整。重みの書式がある出力先 (sdxl / novelai) は重み 1.2 / 0.8、ほかのタグ形式は「強い」の枠を前に並べる。
    自然文系・Midjourney には重みも並べ替えも付けられない。戻り値: (本文の一般タグ, 注意)。"""
    chosen = {g: v for g, v in (strength or {}).items() if v in STRENGTH_WEIGHT}
    if not chosen:
        return shared, []
    if kind == "prose":
        return shared, ["強さの調整: この出力先 (自然文) には重みを付けられません"]
    style_ref = style_ref or "style" in chosen  # 「画風」の強さを選んだら、⑦ が無くても画風のタグは画風の枠

    def key(t: str) -> str:
        return _strip_weight(norm_tag(t)).replace(" ", "_")

    fixed = {"solo", "no_humans"} | set(PERSON_COUNT_TAGS)
    if target in REF_FRAMING_WEIGHT_TARGETS:
        out = []
        for t in shared:
            group = strength_group(key(t), style_ref)
            if key(t) in fixed or group not in chosen or t.startswith("(") or "::" in t:
                out.append(t)  # 人数タグ・すでに重みのあるタグ (② の画角など) はそのまま
            else:
                out.append(format_weight(t, STRENGTH_WEIGHT[chosen[group]], target))
        return out, []
    strong = [t for t in shared if key(t) not in fixed and chosen.get(strength_group(key(t), style_ref)) == "strong"]
    head = [t for t in shared if key(t) in fixed]
    rest = [t for t in shared if t not in strong and t not in head]
    return head + strong + rest, ["強さの調整: この出力先には重みの書式が無いので、「強い」の枠を前に並べるだけにしました"]



# 人物の特徴を詳しく。写真の人物にも、オリジナルキャラなどイラストのキャラにも使う。
# 描くたびに変わらない外見 (顔立ち・目・髪型・体格・角や耳などの特徴) だけを読み、服・ポーズ・表情・背景は読まない (ほかの欄が受け持つ)
PERSON_DETAIL_FIELDS = [
    ("medium", "画風"), ("face_shape", "輪郭・顔の比率"), ("eyes", "目"), ("eyebrows", "眉"), ("nose", "鼻"), ("mouth", "口"),
    ("ears", "耳"), ("skin", "肌"), ("hair", "髪"), ("age_look", "年齢の見た目"), ("build", "体格"), ("distinctive", "目立つ特徴"),
]
# 画風を問わない言い方にする (2026-10-01 利用者の指摘「極端。絵画や劇画などに対応していない」: 最初はアニメの言い方
# (tsurime / ahoge / blush marks / highlights) を例にしていて、油絵や劇画にもアニメの言い方を当てはめていた)。
# 各欄は「見る観点」の一覧で、ここに挙げた語に限らず、見えるものは全部書かせる (2026-10-01 利用者の指摘「顔の細かい
# ディテールが取れていない」: 以前は欄ごとに一言だったので、一重 / 二重のような違いが落ちていた)
PERSON_DETAIL_HINTS = {
    "medium": "what kind of image this is, in a few words (e.g. photograph, oil painting, watercolour painting, gekiga / realistic "
              "manga, anime illustration, western comic, 3D render, pixel art, pencil sketch). Decide from the image itself",
    "face_shape": "outline of the face, forehead, temples, cheekbones, cheeks, jaw, chin, and the proportions and spacing of the "
                  "features (e.g. long midface, short chin, wide-set features)",
    "eyes": "eyelids (single, double, hooded and so on), eye shape, size, slant of the outer corners, inner corners, spacing, "
            "depth, iris and pupil colour and look, eyelashes, under-eye area",
    "eyebrows": "shape, arch, angle, length, thickness, density, colour and the gap to the eyes",
    "nose": "bridge height and width, tip, nostrils and wings, length, profile, and how the image draws it",
    "mouth": "lip thickness (upper and lower), width, shape of the cupid's bow, corners, philtrum, teeth or fangs when they are a "
             "fixed trait",
    "ears": "size, shape, position and angle when visible (also pointed or animal ears)",
    "skin": "tone, undertone, texture, shine, wrinkles, lines, pores, blemishes and marks on the face",
    "hair": "colour (including roots, streaks, gradient), length, texture, thickness, style, bangs, parting, hairline, sideburns, "
            "volume, ends, or baldness",
    "age_look": "apparent age range of the face and body (do not guess a real identity)",
    "build": "height impression, body type, shoulders, neck, chest, waist, hips, arms, legs, proportions",
    "distinctive": "fixed distinctive features: moles (with position), freckles, scars, tattoos, facial hair, dimples, asymmetry, "
                   "animal ears, horns, tail, wings, halo, heterochromia. Not clothing or accessories",
}
PERSON_DETAIL_TAG_MAX = 32  # VLM_ARRAY_LIMITS と同じ理由で上限を付ける (無いと同じ語を繰り返して JSON が切れる)


def vlm_detail_system(with_build: bool, medium_hint: str = "", with_face: bool = False) -> str:
    """人物 1 人の特徴を詳しく読む指示。with_build=False (⑥ 体格を別の画像から取る) なら体格は読まない。
    medium_hint はタガーから判定した画風 (参考。劇画などは判定できないので、最終的な判断は VLM に任せる)。
    with_face は顔を拡大した 2 枚目の画像を渡すとき。"""
    fields = [k for k, _ in PERSON_DETAIL_FIELDS if with_build or k != "build"]
    lines = [
        "You describe the fixed appearance of ONE person or character in the image so that the same person or character can be "
        "drawn again in another picture. The image can be any kind: a photograph, a painting (oil, watercolour), gekiga or "
        "realistic manga, an anime or game illustration, a western comic, a 3D render, pixel art or a sketch.",
    ]
    if with_face:
        lines.append(
            "Image 1 shows the whole person (use it for hair length, build and overall look). Image 2 is a close-up of the same "
            "person's face cut from the original: read the face, eyes, eyebrows, nose, mouth, ears and skin from image 2."
        )
    lines += [
        "Be thorough. The goal is that someone who never saw the image can draw a face that is recognisably this one and not a "
        "generic face. For each field, write every visible detail that sets this person apart from other people drawn in the "
        "same way: shape, size, angle, position, colour, left/right differences. Several details per field, separated by commas. "
        "Skip statements that fit almost anyone (such as 'two eyes', 'normal nose', 'average face'). Do not use vague words "
        "such as medium, average, normal, typical, standard or natural on their own: name the actual shape, size or colour, "
        "or leave the detail out.",
        "First decide what kind of image it is"
        + (f" (an image tagger guessed: {MEDIUM_PHRASE.get(medium_hint, medium_hint)}; it is often wrong for paintings and gekiga)"
           if medium_hint else "")
        + ". Then describe the features in plain, neutral words that fit that kind of image and how it renders the face. "
        "Use anime-specific terms (such as tsurime, ahoge or blush stickers) only for anime-style images, and do not describe a "
        "painting or a drawing as if it were a photograph.",
        "Describe only what stays the same between pictures: face, eyes, eyebrows, nose, mouth, skin, hair"
        + (", body build" if with_build else "") + " and fixed distinctive features.",
        "Do not describe clothing, accessories, pose, expression, lighting or background. Never identify a real person by name.",
        "Each field is English phrases separated by commas (the points listed for each field are what to look at, not a limit). "
        "Write an empty string if it cannot be seen.",
    ]
    lines += [f'- "{k}": {PERSON_DETAIL_HINTS[k]}.' for k in fields]
    lines += [
        f'- "tags": up to {PERSON_DETAIL_TAG_MAX} lowercase short tags for these features only, matching the kind of image '
        "(Danbooru-style tags where one exists, e.g. square jaw, thick eyebrows, wrinkles, beard, long hair, hair over one eye, "
        "mole under eye, freckles, dark skin, slim; otherwise short plain phrases such as double eyelids or high nose bridge; "
        "for anime-style images also e.g. tsurime, ahoge).",
        '- "summary": two to five English sentences that describe the face in detail, then the hair'
        + (", the build" if with_build else "")
        + " and distinctive features, for an image-generation prompt. Keep the specific details from the fields.",
        "Only what is clearly visible. Never repeat a tag. Everything is English.",
    ]
    return "\n".join(lines) + "\n"


def vlm_detail_schema(with_build: bool) -> dict[str, Any]:  # noqa: D103
    props: dict[str, Any] = {k: {"type": "string", "maxLength": 400} for k, _ in PERSON_DETAIL_FIELDS if with_build or k != "build"}
    props["medium"]["maxLength"] = 120
    props["tags"] = {"type": "array", "items": {"type": "string", "maxLength": 60}, "maxItems": PERSON_DETAIL_TAG_MAX}
    props["summary"] = {"type": "string", "maxLength": 1500}
    return {"type": "object", "properties": props, "required": list(props)}


def person_detail_tags(raw: Any, with_build: bool) -> list[str]:
    """VLM が返した特徴のタグから、人物 (顔・髪・目) と体格のものだけを残す (服・ポーズ・背景が混ざっても入れない)。"""
    out = []
    for tag in clean_llm_tags(raw, limit=PERSON_DETAIL_TAG_MAX):
        n = norm_tag(tag).replace(" ", "_")
        category = reference_category(n)
        if category == "body" and not with_build:
            continue
        if set(re.split(r"[_-]", n)) & _DETAIL_MEDIUM_WORDS:
            continue  # 画風の言葉 (gekiga / realistic manga など) は特徴ではない (画風は画風の欄の担当)
        words = set(re.split(r"[_-]", n))
        if not words & _DETAIL_FACE_WORDS and any(classify_subject_tag(w, True) == "background" for w in (n, n.split("_")[-1])):
            continue  # school classroom のように語彙に無い言い方の背景も外す (実機の qwen3-vl で混ざった。nose bridge は残す)
        if category in ("person", "body") and n not in _EXPRESSION_TAGS:
            out.append(tag)
    return out


# 情報の少ない言い方 (medium size / typical human ear shape / depth natural など) は外す。
# あいまいな語 (_VAGUE_WORDS) があり、残りが部位と属性の名前だけ (_VAGUE_FILLER) の言い方。medium brown hair のように
# 具体的な語が 1 つでもあれば残す。外したものは分析メモに出す
_VAGUE_WORDS = {
    "medium", "average", "normal", "normally", "typical", "typically", "standard", "regular", "moderate", "moderately",
    "natural", "naturally", "ordinary", "common", "usual", "unremarkable", "generic", "balanced", "proportionate", "proportional",
}
_VAGUE_FILLER = {
    "size", "sized", "shape", "shaped", "length", "width", "height", "thickness", "density", "volume", "angle", "position",
    "positioned", "placement", "spacing", "spaced", "set", "depth", "colour", "color", "texture", "tone", "undertone", "appearance",
    "impression", "type", "build", "proportions", "features", "feature", "face", "facial", "look", "human", "ear", "ears", "eye",
    "eyes", "nose", "lip", "lips", "mouth", "bridge", "nostrils", "wings", "tip", "chin", "jaw", "forehead", "cheeks",
    "cheekbones", "eyebrow", "eyebrows", "brows", "eyelashes", "eyelids", "iris", "pupil", "pupils", "hair", "skin", "body",
    "profile", "shine", "upper", "and", "lower", "a", "an", "the", "of", "with", "in", "for", "midface", "anime", "hairline",
}
_EMPTY_PHRASES = {"none", "n/a", "na", "not visible", "nothing", "nothing distinctive", "no distinctive features", "unknown"}


def is_vague_phrase(phrase: str) -> bool:
    """顔の特徴の言い方が、あいまいな語と部位・属性の名前だけでできているか (例: medium size, typical human ear shape)。"""
    text = re.sub(r"[()]", " ", phrase.lower()).strip(" .")
    if text in _EMPTY_PHRASES:
        return True
    words = [w for w in re.split(r"[\s_\-/]+", text) if w]
    return bool(words) and bool(set(words) & _VAGUE_WORDS) and set(words) <= _VAGUE_WORDS | _VAGUE_FILLER


def drop_vague_detail(detail: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """VLM が読んだ特徴の各欄とタグから、情報の少ない言い方を外す。戻り値: (外した後, 外した言い方)。"""
    out = dict(detail)
    removed: list[str] = []
    for key, _ in PERSON_DETAIL_FIELDS:
        if key == "medium" or not isinstance(out.get(key), str):
            continue
        parts = [p.strip() for p in re.split(r",(?![^()]*\))", out[key]) if p.strip()]
        vague = [p for p in parts if is_vague_phrase(p)]
        removed += vague
        out[key] = ", ".join(p for p in parts if p not in vague)
    if isinstance(out.get("tags"), list):
        vague = [t for t in out["tags"] if isinstance(t, str) and is_vague_phrase(t)]
        removed += vague
        out["tags"] = [t for t in out["tags"] if t not in vague]
    return out, dedup(removed)


_DETAIL_FACE_WORDS = {
    "face", "eye", "eyes", "eyelid", "eyelids", "eyelashes", "eyebrow", "eyebrows", "brow", "brows", "nose", "lip", "lips", "mouth",
    "cheek", "cheeks", "cheekbones", "chin", "jaw", "jawline", "forehead", "ear", "ears", "skin", "teeth", "hair", "hairline",
}
_DETAIL_MEDIUM_WORDS = {
    "manga", "gekiga", "geki", "ga", "painting", "painted", "illustration", "anime", "photo", "photograph", "photographic",
    "render", "sketch", "comic", "comics", "watercolor", "watercolour", "oil", "style", "realistic", "digital", "artwork",
}


def person_detail_lines(detail: dict[str, Any]) -> list[str]:
    """分析メモに出す、項目ごとの一覧。"""
    return [f"  {label}: {llm_text(detail.get(key))}" for key, label in PERSON_DETAIL_FIELDS if llm_text(detail.get(key))]


# メインの呼び出しに見せない枠 (② / ③ は上の 1 枚ずつの呼び出しが受け持つ)
VLM_SEPARATE_SLOTS = ("pose", "outfit", "body", "style", "expression")


REF_VLM_ROLE = {
    "pose": "the POSE reference: take ONLY the body pose, gesture, hand and leg positions, head direction, gaze, "
            "framing, camera angle and composition from it. Ignore the look of any person in it (face, hair, clothing, "
            "colours, body type, art style) and its background.",
    "outfit": "the OUTFIT reference: take ONLY the clothing, footwear, headwear, accessories and hair ornaments worn "
              "(design, colours, materials, details) from it. Ignore the face, hair, body, pose, art style, background and "
              "objects that are held.",
    "background": "the BACKGROUND reference: take ONLY the location, environment, time of day, weather and lighting "
                  "from it. Ignore any people in it.",
    "props": "the PROPS reference: take ONLY the small objects from it (things held or carried such as bags, umbrellas, "
             "weapons, books, drinks, phones, and animals or items placed near the character). Ignore people, clothing, "
             "pose and background.",
}


def separate_rule(separate: tuple[str, ...], has_props: bool) -> str:
    """メインの呼び出しへの指示: 別の呼び出しが 1 枚ずつ読む枠 (② / ③ / ⑥) は書かせず、image 1 のその要素も書かせない。"""
    names = {"pose": "pose and composition", "outfit": "outfit", "body": "body type", "style": "art style",
             "expression": "facial expression and gaze"}
    fields = [k for x in separate for k in (
        ("pose_tags", "pose", "camera") if x == "pose" else ("outfit_tags", "outfit") + (() if has_props else ("objects",))
        if x == "outfit" else ("style", "style_tags") if x == "style" else ())]
    what = {"pose": "pose", "outfit": "clothing", "body": "body type (breasts, build, skin tone, apparent age of the body)",
            "style": "art style or medium (anime, photo, painting and so on)",
            "expression": "facial expression or gaze (smile, blush, tears, open mouth, closed eyes, looking at viewer)"}
    return (
        "- The " + ", the ".join(names[x] for x in separate) + " are read from separate references in another step. "
        + (f"Leave {', '.join(chr(34) + k + chr(34) for k in fields)} out, and never" if fields else "Never")
        + " describe the " + " or the ".join(what[x] for x in separate) + " of image 1 anywhere (not in subject or extra_tags)."
    )


def vlm_reference_system(slots: list[str], separate: tuple[str, ...] = ()) -> str:
    """複数枚モードの VLM への指示。slots は画像の並び (先頭は必ず person)。空の枠の要素は image 1 から取らせる。
    separate は、画像を見せずに別の呼び出し (vlm_part_system) が受け持つ枠 (② ポーズ / ③ 服装)。"""
    idx = {slot: i + 1 for i, slot in enumerate(slots)}

    def src(slot: str) -> str:
        return f"image {idx.get(slot, 1)}"

    lines = [
        f"You help write image-generation prompts for Stable Diffusion / Flux from {len(slots)} reference images.",
        "Image 1 is the CHARACTER reference: take the person's identity and look from it (character name, face, hair, "
        "eyes, body type, skin" + ("" if "expression" in separate else ", expression")
        + ("" if "style" in separate else ", art style and medium") + ").",
    ]
    for slot in slots[1:]:
        role = REF_VLM_ROLE[slot]
        if slot == "outfit" and "props" not in idx:  # ⑤ が無いときは ③ が小物も受け持つ
            role = role.replace("and objects that are held.", "") + " Also take the held or carried objects (bags, umbrellas, weapons, books) from it."
        lines.append(f"Image {idx[slot]} is {role}")
    missing = [slot for slot in ("pose", "outfit", "background", "props")
               if slot not in idx and slot not in separate and not (slot == "props" and ("outfit" in idx or "outfit" in separate))]
    if missing:
        names = {"pose": "the pose and composition", "outfit": "the outfit", "background": "the background and lighting",
                 "props": "the held or placed objects"}
        lines.append("Take " + ", ".join(names[m] for m in missing) + " from image 1, because no separate reference was given.")
    lines += [
        "The goal is a prompt for ONE new image that shows the character of image 1 combined with these references.",
        "Tags from a specialised tagger are given for each image with confidence scores. Treat them as ground truth for what is visible.",
        "",
        "Rules",
        "- Describe only one person: the character of image 1. Never mix the face, hair or body of people in the other images into the character.",
        "- Never identify a real person by name. Well-known fictional characters may be named.",
        *([separate_rule(separate, "props" in idx)] if separate else []),
        "- Write every field in English only.",
        '- "age": the apparent age group of the character of image 1: ' + AGE_VLM_GROUPS + '. "unknown" when it cannot be '
        "judged. " + AGE_VLM_JUDGE,
        '- "extra_tags": up to 12 lowercase Danbooru-style tags for composition, camera, lighting and atmosphere that are missing from the tag lists.',
        f'- "pose_tags": up to 12 lowercase Danbooru-style tags for the pose in {src("pose")} only (e.g. standing, sitting, arms up, hand on hip, crossed legs).',
        f'- "pose": one or two English sentences describing only the pose in {src("pose")}: body orientation, torso, arms, hands, legs, head tilt, gaze.',
        f'- "outfit_tags": up to 16 lowercase Danbooru-style tags for the clothing, footwear, headwear and accessories worn in {src("outfit")} only.',
        f'- "outfit": one or two English sentences describing only the outfit in {src("outfit")}: garments, colours, materials, accessories.',
        (
            '- "negative_additions": up to 12 short lowercase phrases to avoid: extra people, and concrete features of the people in '
            "the other reference images that are not present in image 1 (only features you can actually see there). Each entry "
            if any(slot in idx for slot in ("pose", "outfit")) else
            # ② / ③ を別の呼び出しで読むときは、メインは人物の写った参照画像を見ていない (見ていない特徴を挙げて wearing hat が入った)
            '- "negative_additions": up to 12 short lowercase phrases to avoid, such as extra people or a wrong medium. '
            "Do not list hair, eye or clothing features. Each entry "
        ) + (
            "must be a concrete visual feature, never a bare category word such as face, hair, body, outfit or pose, and never a "
            "feature of image 1. Never put the outfit, "
            "pose, objects or setting that you take from the reference images into negative_additions. "
            "Do not repeat the base negatives you are given."
        ),
        f'- "camera": one English sentence about camera and composition taken from {src("pose")}.',
        f'- "lighting": one English sentence about light and colour taken from {src("background")}.',
        '- "style": one or two English sentences about the medium and rendering of image 1. Do not name real individual artists.',
        '- "style_tags": up to 10 lowercase Danbooru-style tags describing the art style of image 1 only. Empty array for photographs.',
        '- "subject": one English sentence describing the character of image 1 (face, hair, eyes, '
        + ("" if "body" in separate else "body, ") + ("" if "expression" in separate else "expression")
        + ") WITHOUT outfit and pose" + (" and WITHOUT facial expression or gaze" if "expression" in separate else "") + ".",
        f'- "background": one English sentence describing the environment in {src("background")}. No people.',
        f'- "objects": up to 8 short lowercase noun phrases for the objects held or placed in {src("props") if "props" in idx else src("outfit")} only (bags, umbrellas, weapons, books, drinks, animals).',
        '- "motion": one or two English sentences for an image-to-video model: a camera movement that suits the composition, '
        "then subtle plausible motion of the character.",
    ]
    # 別の呼び出しが受け持つ欄の説明は載せない (スキーマからも外している。vlm_ref_schema)
    skip = set()
    if "pose" in separate:
        skip |= {"pose_tags", "pose", "camera"}
    if "outfit" in separate:
        skip |= {"outfit_tags", "outfit"} | (set() if "props" in idx else {"objects"})
    if "style" in separate:
        skip |= {"style", "style_tags"}
    lines = [x for x in lines if not any(x.startswith(f'- "{k}":') for k in skip)]
    return "\n".join(lines) + "\n"


VLM_REF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        **{k: v for k, v in VLM_SCHEMA["properties"].items() if k not in ("persons", "pose_tags", "outfit_tags", "interactions")},
        "pose": {"type": "string"},
        "pose_tags": {"type": "array", "items": {"type": "string"}},
        "outfit": {"type": "string"},
        "outfit_tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": [k for k in VLM_SCHEMA["required"] if k not in ("persons", "pose_tags", "outfit_tags", "interactions")]
    + ["pose", "pose_tags", "outfit", "outfit_tags"],
}

VLM_TEXT_MAX_CHARS: dict[str, int] = {}  # 欄ごとの長さの上限 (無い欄は 600 文字)


def vlm_ref_schema(separate: tuple[str, ...] = (), has_props: bool = False) -> dict[str, Any]:
    """複数枚モードのメインの呼び出しのスキーマ。出力が長いほど遅いので、使わない欄は書かせない (軽量化 2026-09-30):
    natural_prompt は複数枚モードでは使わない (本文は枠ごとのタグと文から組み立てる。実機で出力 1,020 トークン中の大きな部分)、
    ② / ③ を 1 枚ずつ読むときはその欄 (pose_tags / pose / camera、outfit_tags / outfit / ⑤ が無ければ objects) も外す。"""
    drop = {"natural_prompt"}
    if "pose" in separate:
        drop |= {"pose_tags", "pose", "camera"}
    if "outfit" in separate:
        drop |= {"outfit_tags", "outfit"} | (set() if has_props else {"objects"})
    if "style" in separate:
        drop |= {"style", "style_tags"}
    props = {k: v for k, v in VLM_REF_SCHEMA["properties"].items() if k not in drop}
    # 文字列欄の長さの上限。無いと思考モデル (qwen3.5) が subject などに外見の描写を延々と書き、出力の上限で JSON が切れた
    # (実機 5 回中 3 回、各 58 秒)。1〜4 文の欄なので十分な長さ
    props = {k: (v | {"maxLength": VLM_TEXT_MAX_CHARS.get(k, 600)} if v.get("type") == "string" else v) for k, v in props.items()}
    return {"type": "object", "properties": props, "required": [k for k in VLM_REF_SCHEMA["required"] if k not in drop]}


# 配列の最大個数。無いと小型 VLM が同じ語を延々と繰り返し、出力の上限で JSON が途中で切れる (4 枚で実際に起きた)
VLM_ARRAY_LIMITS = {
    "extra_tags": 12, "negative_additions": 12, "objects": 8, "style_tags": 10, "persons": 6,
    "pose_tags": 12, "outfit_tags": 16, "interactions": 6,
}
for _schema in (VLM_SCHEMA, VLM_REF_SCHEMA):
    for _key, _limit in VLM_ARRAY_LIMITS.items():
        if _key in _schema["properties"]:
            _schema["properties"][_key]["maxItems"] = _limit
VLM_SCHEMA["properties"]["persons"]["items"]["properties"]["tags"]["maxItems"] = 30

# --- 複数人物の扱い ---
SUBJECT_TOKEN = {"girl": "1girl", "boy": "1boy", "other": "1other"}
PERSON_COUNT_TAGS = {
    "1girl": 1, "2girls": 2, "3girls": 3, "4girls": 4, "5girls": 5, "6+girls": 6, "multiple_girls": 3,
    "1boy": 1, "2boys": 2, "3boys": 3, "4boys": 4, "5boys": 5, "6+boys": 6, "multiple_boys": 3,
    "1other": 1, "2others": 2, "3others": 3, "multiple_others": 3,
}
POSITION_JA = {
    "left": "左", "right": "右", "center": "中央", "centre": "中央", "middle": "中央",
    "foreground": "手前", "front": "手前", "background": "奥", "back": "奥",
    "top": "上", "bottom": "下", "upper": "上", "lower": "下", "far": "遠く", "near": "近く",
}


def person_count_hint(tags: TagResult) -> tuple[int, list[str]]:
    """タガーの人数タグから人数の目安を出す (性別ごとに最大値を取り、二重計上を避ける)。"""
    have = {name for name, _ in tags.general}
    found = [t for t in PERSON_COUNT_TAGS if t in have]
    total = 0
    for group in ("girl", "boy", "other"):
        numeric = [PERSON_COUNT_TAGS[t] for t in found if group in t and not t.startswith("multiple")]
        if numeric:
            total += max(numeric)  # 2girls などの明示的な数を優先
        elif any(group in t for t in found):
            total += PERSON_COUNT_TAGS[f"multiple_{group}s"]
    return total, found


def position_ja(position: str) -> str:
    words = [w for w in re.split(r"[\s\-_/,]+", position.strip().lower()) if w]
    return " ".join(POSITION_JA.get(w, w) for w in words) or position


def sd_escape(tag: str) -> str:
    return tag.replace("\\(", "(").replace("\\)", ")").replace("(", "\\(").replace(")", "\\)")


def norm_tag(tag: str) -> str:
    return tag.replace("\\(", "(").replace("\\)", ")").replace("_", " ").strip().lower()


def parse_persons(raw: Any) -> list[dict[str, Any]]:
    """VLM の persons 配列を整形する。人数タグ・solo は人物側から除き、性別トークンを先頭に付ける。"""
    if not isinstance(raw, list):
        return []
    persons: list[dict[str, Any]] = []
    for item in raw[:6]:
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject") or "other").strip().lower()
        if subject not in SUBJECT_TOKEN:
            subject = "other"
        tags = [
            sd_escape(t) for t in clean_llm_tags(item.get("tags"), limit=30)
            if t.replace(" ", "_") not in PERSON_COUNT_TAGS and t != "solo"
        ]
        # その人のポーズ・服装のタグも足す (表記ゆれで同じものは足さない)
        have = {loose_tag_key(t) for t in tags}
        for t in clean_llm_tags(item.get("pose_tags"), limit=12) + clean_llm_tags(item.get("outfit_tags"), limit=16):
            if loose_tag_key(t) not in have:
                have.add(loose_tag_key(t))
                tags.append(t)
        position = str(item.get("position") or "").strip()
        description = str(item.get("description") or "").strip()
        persons.append({
            "subject": subject,
            "position": position,
            "position_ja": position_ja(position),
            "tags": tags,
            "description": description,
            "age": clean_age(item.get("age")),
            "prompt": ", ".join(dedup([SUBJECT_TOKEN[subject]] + tags)),
        })
    return persons


class OllamaError(Exception):
    pass


def normalize_host(host: str) -> str:
    """Ollama の URL を 'http://host:port' の形にそろえる。スキーム省略 (localhost:11434) は http:// を補い、
    http / https 以外 (file:// など) は受け付けない。"""
    host = (host or "").strip() or DEFAULT_CONFIG["ollama_host"]
    if not re.match(r"^[a-z][a-z0-9+.-]*://", host, re.IGNORECASE):
        host = "http://" + host
    parts = urllib.parse.urlsplit(host)
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        raise OllamaError(f"Ollama の URL は http:// か https:// で指定してください: {host}")
    return host.rstrip("/")


def _ollama_error_text(body: str) -> str:
    """Ollama のエラー応答 ({"error": "..."}) から本文を取り出す。"""
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return body[:300]
    return str(data.get("error", body))[:300] if isinstance(data, dict) else body[:300]


class OllamaVLM:
    def __init__(self, host: str, model: str, timeout: int = VLM_TIMEOUT):
        self.host = normalize_host(host)
        self.model = model
        self.timeout = timeout
        self.keep_alive: int | str | None = None  # 0 なら応答後すぐ VRAM から降ろす (None は Ollama の既定 5 分)
        self._capabilities: set[str] | None = None
        self.cancel_event: threading.Event | None = None  # セットされていたら /api/chat を中止する
        self._conn: http.client.HTTPConnection | None = None  # 実行中の /api/chat の接続 (abort 用)

    def abort(self) -> None:
        """別スレッドから実行中の /api/chat の接続を切る。Ollama も接続断で生成をやめ、GPU を解放する。"""
        conn = self._conn
        if conn is None:
            return
        try:
            if conn.sock is not None:
                conn.sock.shutdown(socket.SHUT_RDWR)  # 受信待ちのスレッドを起こす (close だけでは起きない)
        except OSError:
            pass
        conn.close()

    def _cancelled(self) -> bool:
        return bool(self.cancel_event and self.cancel_event.is_set())

    def _http(self, path: str, payload: dict[str, Any] | None = None, timeout: int | None = None,
              abortable: bool = False) -> dict[str, Any]:
        """Ollama に JSON を送り (payload が無ければ GET)、JSON の辞書を受け取る。失敗はすべて OllamaError。
        abortable=True なら abort() で別スレッドから中止できる (urllib は応答待ちの間に切れないので http.client で
        接続を保持する)。"""
        if abortable and self._cancelled():
            raise OperationCancelled()
        parts = urllib.parse.urlsplit(self.host)
        conn_cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
        conn = conn_cls(parts.hostname or "localhost", parts.port, timeout=timeout or self.timeout)
        if abortable:
            self._conn = conn
        try:
            conn.connect()
            if abortable and self._cancelled():  # 接続前に abort() が来ていた場合 (sock が無く切れなかった)
                raise OperationCancelled()
            body = json.dumps(payload).encode("utf-8") if payload is not None else None
            conn.request(
                "POST" if payload is not None else "GET", parts.path.rstrip("/") + path, body=body,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            text = resp.read().decode("utf-8", "replace")
        except (OSError, http.client.HTTPException) as e:
            if abortable and self._cancelled():
                raise OperationCancelled() from e
            if isinstance(e, TimeoutError):
                raise OllamaError("Ollama の応答がタイムアウトしました。") from e
            raise OllamaError(
                f"Ollama に接続できません ({self.host})。Ollama が起動しているか確認してください: {e}"
            ) from e
        finally:
            if abortable:
                self._conn = None
            conn.close()
        if abortable and self._cancelled():
            raise OperationCancelled()
        if resp.status >= 400:
            raise OllamaError(f"Ollama がエラーを返しました ({resp.status}): {_ollama_error_text(text)}")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise OllamaError(f"Ollama の応答を読めませんでした: {text[:200]}") from e
        if not isinstance(data, dict):
            raise OllamaError("Ollama の応答が想定した形式ではありません。")
        return data

    def _chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """/api/chat を送る。「■ 中止」で接続を切れる。"""
        return self._http("/api/chat", payload, abortable=True)

    def unload(self) -> None:
        """モデルを VRAM から降ろす (keep_alive 0。読み込まれていなければ何もしない)。失敗しても処理は続ける。"""
        try:
            self._http("/api/generate", {"model": self.model, "keep_alive": 0}, timeout=30)
        except OllamaError:
            pass

    def list_models(self) -> list[str]:
        models = self._http("/api/tags", timeout=10).get("models")
        if not isinstance(models, list):
            return []
        return [str(m["name"]) for m in models if isinstance(m, dict) and m.get("name")]

    def pull(self, name: str, progress: ProgressFn | None = None, cancel: threading.Event | None = None) -> None:
        """モデルを Ollama に取得する (ollama pull 相当)。進捗は progress(文字列) で通知する。
        cancel がセットされたら接続を切って中止する (Ollama は取得を止め、次回は続きから取得する)。"""
        url = self.host + "/api/pull"
        payload = json.dumps({"model": name, "stream": True}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:  # チャンク間 120 秒無応答で諦める
                for raw_line in resp:
                    if cancel is not None and cancel.is_set():
                        raise OperationCancelled()
                    line = raw_line.decode("utf-8", "replace").strip()
                    if not line:
                        continue
                    try:
                        info = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if info.get("error"):
                        raise OllamaError(f"取得に失敗しました: {info['error']}")
                    status = str(info.get("status") or "")
                    total = info.get("total") or 0
                    done = info.get("completed") or 0
                    if progress:
                        if total:
                            progress(f"{name} を取得中: {done / total:.0%} ({done / 1e9:.2f} / {total / 1e9:.2f} GB) {status}")
                        else:
                            progress(f"{name} を取得中: {status}")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            try:
                body = json.loads(body).get("error", body)
            except json.JSONDecodeError:
                pass
            raise OllamaError(f"取得に失敗しました ({e.code}): {body}") from e
        except urllib.error.URLError as e:
            raise OllamaError(f"Ollama に接続できません ({self.host}): {e.reason}") from e
        except TimeoutError as e:
            raise OllamaError("取得中に応答が途絶えました。ネットワークを確認してください。") from e
        except (OSError, http.client.HTTPException) as e:
            raise OllamaError(f"取得中に接続が切れました: {e}") from e

    def capabilities(self) -> set[str]:
        if self._capabilities is None:
            caps = self._http("/api/show", {"model": self.model}, timeout=30).get("capabilities")
            self._capabilities = {c for c in caps if isinstance(c, str)} if isinstance(caps, list) else set()
        return self._capabilities

    @staticmethod
    def encode_image(image: Image.Image, max_side: int = VLM_MAX_SIDE) -> str:
        rgb = image.convert("RGB")
        w, h = rgb.size
        scale = max_side / max(w, h)
        if scale < 1:
            rgb = rgb.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        rgb.save(buf, "JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def analyze(
        self,
        image: Image.Image,
        tags: TagResult,
        medium: str,
        base_negative: list[str],
        guidance: str = "",
        refs: list[tuple[str, Image.Image, TagResult]] | None = None,
        separate: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """画像を分析する。refs (枠名, 画像, タグ) を渡すと複数枚モード (image 1 = 人物、以降は refs の順) で分析する。
        separate の枠 (② / ③) は analyze_part が 1 枚ずつ読むので、ここでは画像を見せない。"""
        caps = self.capabilities()
        if "vision" not in caps:
            raise OllamaError(f"モデル {self.model} は画像入力に対応していません。qwen3-vl などを指定してください。")

        tag_lines = [f"{sd_tag(n)} ({p:.2f})" for n, p in tags.general[:60]]
        char_lines = [f"{sd_tag(n)} ({p:.2f})" for n, p in tags.characters]
        pair = bool(refs)
        refs = [r for r in (refs or []) if r[0] not in separate]
        if pair:
            tag_block = f"Image 1 (character) tagger tags (confidence): {', '.join(tag_lines)}\n"
            for i, (slot, _img, ref_tags) in enumerate(refs, 2):
                ref_lines = [f"{sd_tag(n)} ({p:.2f})" for n, p in ref_tags.general[:50]]
                tag_block += f"Image {i} ({slot}) tagger tags (confidence): {', '.join(ref_lines) or 'none'}\n"
        else:
            tag_block = f"Tagger tags (confidence): {', '.join(tag_lines)}\n"
        user_text = (
            f"Detected medium: {medium}\n"
            f"Rating: {tags.rating} ({tags.rating_conf:.2f})\n"
            f"Characters: {', '.join(char_lines) or 'none'}\n"
            + tag_block
            + f"Base negatives already used: {', '.join(base_negative)}\n\n"
            + (f"Target model instructions.\n{guidance}\n\n" if guidance else "")
            + (
                "Look at all images (" + ", ".join(
                    ["image 1 = character"] + [f"image {i} = {slot}" for i, (slot, _, _) in enumerate(refs, 2)]
                ) + ") and return the JSON." if pair else "Look at the image and return the JSON."
            )
            + " Write every field in English."
        )
        images = [self.encode_image(image)]
        images += [self.encode_image(img, VLM_REF_MAX_SIDE.get(slot, VLM_MAX_SIDE)) for slot, img, _ in refs]
        thinking = "thinking" in caps
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "format": vlm_ref_schema(separate, any(slot == "props" for slot, _, _ in refs)) if pair else VLM_SCHEMA,
            "options": {
                "temperature": 0.2,
                "num_predict": VLM_NUM_PREDICT_THINKING if thinking else (VLM_NUM_PREDICT_REFS if pair else VLM_NUM_PREDICT),
                "num_ctx": VLM_NUM_CTX,
            },
            "messages": [
                {"role": "system", "content": vlm_reference_system(["person"] + [slot for slot, _, _ in refs], separate)
                 if pair else VLM_SYSTEM},
                {"role": "user", "content": user_text, "images": images},
            ],
        }
        if thinking:
            payload["think"] = False  # 思考モードを切って高速化 (無視するモデルもある)
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive  # 0 = 応答後すぐ VRAM から降ろす (省 VRAM)
        data = self._chat(payload)
        message = data.get("message") or {}
        content = str(message.get("content") or "")
        # think を無視したモデルが本文に <think>...</think> を混ぜることがあるので外す
        content = re.sub(r"(?s)^\s*<think>.*?</think>", "", content).strip()
        if not content and data.get("done_reason") == "length":
            raise OllamaError(
                f"VLM ({self.model}) が考える過程 (thinking) だけで出力の上限に達し、本文を返しませんでした。"
                "思考しないモデル (qwen3-vl:8b-instruct / qwen2.5vl:7b など) を使うと速く確実です。"
            )
        try:
            obj = json.loads(content)
        except json.JSONDecodeError as e:
            reason = "出力の上限で途中まで" if data.get("done_reason") == "length" else ""
            raise OllamaError(f"VLM の応答を JSON として読めませんでした{f' ({reason})' if reason else ''}: {content[:300]}") from e
        if not isinstance(obj, dict):
            raise OllamaError("VLM の応答が想定した形式ではありません。")
        return obj


    def describe(self, image: Image.Image) -> str:
        """画像の説明を書かせる (describer_model。JoyCaption など、説明を書くことに特化したモデル)。"""
        if "vision" not in self.capabilities():
            raise OllamaError(f"モデル {self.model} は画像入力に対応していません。")
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0.3, "top_p": 0.9, "num_predict": 600},
            "messages": [
                {"role": "system", "content": DESCRIBER_SYSTEM},
                {"role": "user", "content": DESCRIBER_PROMPT, "images": [self.encode_image(image)]},
            ],
        }
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        data = self._chat(payload)
        text = str((data.get("message") or {}).get("content") or "")
        return re.sub(r"(?s)^\s*<think>.*?</think>", "", text).strip()

    def rewrite(self, draft: str, guidance: str, style_first: bool = False) -> str:
        """仕上げの書き直し: 組み立てた本文を、出力先の書き方で 1 つの文章に書き直させる (画像は渡さない)。"""
        thinking = "thinking" in self.capabilities()
        system = POLISH_SYSTEM + ("\n- " + POLISH_STYLE_FIRST if style_first else "")
        if guidance:
            system += "\nHow the final prompt should be written: " + guidance
        user = "Draft:\n" + draft
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": VLM_NUM_PREDICT_THINKING if thinking else VLM_NUM_PREDICT,
                        "num_ctx": VLM_NUM_CTX},  # num_ctx はメインと揃える (変えるとモデルを読み直す)
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if thinking:
            payload["think"] = False
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        data = self._chat(payload)
        text = re.sub(r"(?s)^\s*<think>.*?</think>", "", str((data.get("message") or {}).get("content") or "")).strip()
        text = re.sub(r"^```[a-z]*\s*|\s*```$", "", text).strip()
        return re.sub(r"^(?:Final prompt|Prompt)\s*:\s*", "", text, flags=re.IGNORECASE).strip().strip("'").strip()

    def analyze_detail(self, image: Image.Image, tags: list[str], with_build: bool,
                       keep_alive: int | str | None = None, medium_hint: str = "",
                       face: Image.Image | None = None) -> dict[str, Any]:
        """人物 1 人 (切り出した画像) の特徴を詳しく読む (人物の特徴を詳しく)。
        face は元画像から切り出した顔 (2 枚目として渡す。全身の縮小画像では顔が 100px ほどになり細部が読めないため)。"""
        thinking = "thinking" in self.capabilities()
        images = [self.encode_image(image, VLM_MAX_SIDE)] + ([self.encode_image(face, VLM_MAX_SIDE)] if face is not None else [])
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "format": vlm_detail_schema(with_build),
            "options": {"temperature": 0.2, "num_predict": VLM_NUM_PREDICT_THINKING if thinking else VLM_DETAIL_NUM_PREDICT,
                        "num_ctx": VLM_NUM_CTX},
            "messages": [
                {"role": "system", "content": vlm_detail_system(with_build, medium_hint, with_face=face is not None)},
                {"role": "user", "content": f"Tagger tags: {', '.join(tags[:40]) or 'none'}\n\n"
                                            + ("Image 1: the whole person. Image 2: close-up of the face. " if face is not None else "")
                                            + "Look at the image and return the JSON.",
                 "images": images},
            ],
        }
        if thinking:
            payload["think"] = False
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        data = self._chat(payload)
        content = re.sub(r"(?s)^\s*<think>.*?</think>", "", str((data.get("message") or {}).get("content") or "")).strip()
        try:
            obj = json.loads(content)
        except json.JSONDecodeError as e:
            if data.get("done_reason") == "length":
                # 思考するモデル (qwen3-vl:8b など) は think: false でも内部で考えて上限を使い切ることがある (2026-10-01 実機)
                raise OllamaError(
                    f"VLM ({self.model}) の出力が上限に達して途中で切れました (人物の特徴)。"
                    + ("思考しないモデル (qwen3-vl:8b-instruct / qwen2.5vl:7b など) を使うと確実です。" if thinking else "")
                ) from e
            raise OllamaError(f"VLM の応答を JSON として読めませんでした (人物の特徴): {content[:200]}") from e
        if not isinstance(obj, dict):
            raise OllamaError("VLM の応答が想定した形式ではありません (人物の特徴)。")
        return obj

    def analyze_part(self, image: Image.Image, slot: str, tags: TagResult, with_objects: bool,
                     keep_alive: int | str | None = None) -> dict[str, Any]:
        """複数枚モードの ② ポーズ / ③ 服装を 1 枚だけ見せて読む (ほかの画像が混ざらない)。
        keep_alive は次の呼び出しまでモデルを VRAM に残すため (読み直しを省く)。"""
        tag_lines = [f"{sd_tag(n)} ({p:.2f})" for n, p in tags.general[:50]]
        thinking = "thinking" in self.capabilities()
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "format": vlm_part_schema(slot, with_objects),
            "options": {"temperature": 0.2, "num_predict": VLM_NUM_PREDICT_THINKING if thinking else VLM_PART_NUM_PREDICT,
                        "num_ctx": VLM_NUM_CTX},  # num_ctx はメインと揃える (変えるとモデルを読み直す)
            "messages": [
                {"role": "system", "content": vlm_part_system(slot, with_objects)},
                {"role": "user", "content": f"Tagger tags (confidence): {', '.join(tag_lines) or 'none'}\n\n"
                                            "Look at the image and return the JSON.",
                 "images": [self.encode_image(image, VLM_REF_MAX_SIDE.get(slot, VLM_MAX_SIDE))]},
            ],
        }
        if thinking:
            payload["think"] = False
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        data = self._chat(payload)
        content = re.sub(r"(?s)^\s*<think>.*?</think>", "", str((data.get("message") or {}).get("content") or "")).strip()
        try:
            obj = json.loads(content)
        except json.JSONDecodeError as e:
            raise OllamaError(f"VLM の応答を JSON として読めませんでした ({REF_SLOTS[slot]}): {content[:200]}") from e
        if not isinstance(obj, dict):
            raise OllamaError(f"VLM の応答が想定した形式ではありません ({REF_SLOTS[slot]})。")
        return obj


# --- Hugging Face の URL から VLM を追加 ---
# Ollama は hf.co/{ユーザー}/{リポジトリ}[:量子化] で Hugging Face の GGUF を取得できる (HF 公式 docs/hub/ollama)。
# 量子化を付けなければ Q4_K_M、無ければ Ollama がリポジトリの中から選ぶ。画像用ファイル (mmproj-*.gguf) も一緒に取り込まれる
# (Ollama 0.34 で hf.co/HauhauCS/Qwen3.5-9B-...:Q6_K が vision 付きで入ることを確認)
HF_API = "https://huggingface.co/api/models/"
_HF_URL_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?(?:huggingface\.co|hf\.co)/"
    r"(?P<repo>[A-Za-z0-9][\w.-]*/[\w.-]+?)"
    r"(?::(?P<tag>[\w.-]+))?"
    r"(?:/(?:tree|blob|resolve)/[^/]+(?:/(?P<file>[^?#]*))?)?/?(?:[?#].*)?$"
)


class HfModelError(Exception):
    """Hugging Face のリポジトリから取得できない理由 (画面にそのまま出す)。"""


def parse_hf_model(text: str) -> tuple[str, str | None] | None:
    """VLM 欄の文字が Hugging Face の URL (huggingface.co/ か hf.co/ で始まる) なら (リポジトリ, 量子化またはファイル名) を返す。
    ユーザー/リポジトリ だけの形は Ollama のモデル名 (namespace/model) と見分けられないので扱わない。"""
    m = _HF_URL_RE.match(text.strip())
    if not m:
        return None
    repo = m.group("repo")
    if repo.endswith(".git"):
        repo = repo[:-4]
    tag = m.group("tag")
    file = (m.group("file") or "").rstrip("/")
    if file:
        name = file.rsplit("/", 1)[-1]
        if name.lower().endswith(".gguf"):
            tag = name  # HF 公式: ファイル名をそのまま量子化のタグに使える
        elif name:
            tag = None  # README などを指していたらリポジトリ全体
    return repo, tag


def hf_repo_info(repo: str, timeout: int = 20) -> dict[str, Any]:
    """Hugging Face の公開 API でリポジトリの情報 (ファイル一覧とサイズ、gated) を取る。"""
    req = urllib.request.Request(HF_API + urllib.parse.quote(repo, safe="/") + "?blobs=true",
                                 headers={"User-Agent": f"{APP_ID}/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 404):
            raise HfModelError(
                f"Hugging Face のリポジトリ「{repo}」が見つからないか非公開のため、中身を確かめられません。取得できません。"
            ) from e
        raise HfModelError(f"Hugging Face に問い合わせできませんでした ({e.code})。取得できません。") from e
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
        raise HfModelError(f"Hugging Face に問い合わせできませんでした ({e})。取得できません。") from e
    if not isinstance(data, dict):
        raise HfModelError("Hugging Face の応答が想定した形式ではありません。取得できません。")
    return data


def check_hf_model(text: str, info: dict[str, Any] | None = None) -> dict[str, Any]:
    """URL を確かめて、Ollama に渡す名前と確認用の情報を返す。取得できない理由があれば HfModelError。
    戻り値: {"name": hf.co/..., "repo", "file": 取る GGUF (分かれば), "size": バイト (分かれば), "gated": bool}"""
    parsed = parse_hf_model(text)
    if parsed is None:
        raise HfModelError("Hugging Face の URL として読めません (例: https://huggingface.co/ユーザー/リポジトリ)。")
    repo, tag = parsed
    if info is None:
        info = hf_repo_info(repo)
    files = [(str(f.get("rfilename") or ""), f.get("size")) for f in info.get("siblings") or [] if isinstance(f, dict)]
    ggufs = [(n, sz) for n, sz in files if n.lower().endswith(".gguf") and "mmproj" not in n.lower().rsplit("/", 1)[-1]]
    if not ggufs:
        raise HfModelError(
            f"「{repo}」には GGUF ファイルがありません。Ollama が Hugging Face から取得できるのは GGUF だけなので、取得できません"
            " (同じモデルの GGUF 版のリポジトリを探してください)。"
        )
    if not any("mmproj" in n.lower().rsplit("/", 1)[-1] for n, _ in files):
        raise HfModelError(
            f"「{repo}」には画像用ファイル (mmproj) がありません。画像を読めないので i2t では使えず、取得できません。"
        )
    chosen: tuple[str, Any] | None = None
    if tag:
        low = tag.lower()
        if low.endswith(".gguf"):
            chosen = next(((n, sz) for n, sz in ggufs if n.rsplit("/", 1)[-1].lower() == low), None)
            if chosen is None and "mmproj" in low:
                raise HfModelError("指定されたのは画像用ファイル (mmproj) です。本体のモデルの URL を入れてください。")
        else:
            chosen = next(((n, sz) for n, sz in ggufs if low in n.rsplit("/", 1)[-1].lower()), None)
        if chosen is None:
            quants = ", ".join(n.rsplit("/", 1)[-1] for n, _ in ggufs)
            raise HfModelError(f"指定された「{tag}」が「{repo}」にありません。あるファイル: {quants}")
    else:
        # Ollama の既定: Q4_K_M があればそれ (無ければ Ollama がリポジトリから選ぶので、ここでは分からない)
        chosen = next(((n, sz) for n, sz in ggufs if "q4_k_m" in n.lower()), None)
    return {
        "name": f"hf.co/{repo}" + (f":{tag}" if tag else ""),
        "repo": repo,
        "file": chosen[0] if chosen else "",
        "size": chosen[1] if chosen and isinstance(chosen[1], int) else None,
        "gated": bool(info.get("gated")),
    }


# --------------------------------------------------------------------------
# パイプライン
# --------------------------------------------------------------------------


@dataclass
class PromptResult:
    image: str
    engine: str
    tagger_model: str
    vlm_model: str
    detected_medium: str
    rating: str
    analysis: str
    positive_prompt: str
    negative_prompt: str
    notes: str
    target: str = DEFAULT_TARGET
    tags: list[dict[str, Any]] = field(default_factory=list)
    persons: list[dict[str, Any]] = field(default_factory=list)
    nsfw: bool = False  # NSFW 判定 (questionable / explicit または露出タグ)
    mosaic_prompt: bool = False  # NSFW 判定でプロンプトにモザイク指定を入れたか
    extras: dict[str, dict[str, Any]] = field(default_factory=dict)  # camera / lighting / style / motion
    era: dict[str, Any] | None = None  # 年代スタイル分類の結果 (label / label_ja / conf / tags / ranking)
    references: dict[str, str] = field(default_factory=dict)  # 複数枚モードの参照画像 (pose / outfit / background → パス)
    controls: dict[str, Any] = field(default_factory=dict)  # ボーン・カメラ・強さの指定 (Engine.run の controls)
    main_person: dict[str, Any] | None = None  # メインの 1 人だけ: {"count": 検出人数, "box": [左, 上, 右, 下] | None}
    formats: dict[str, str] = field(default_factory=dict)  # 人物ごとタブに出す、各ツールに貼る形 (見出し → 本文)
    background_people: list[list[int]] = field(default_factory=list)  # 奥に小さく写っていて人数に数えなかった人の枠
    quality_issues: list[str] = field(default_factory=list)  # 品質の自己チェックで見つかった問題
    elapsed_sec: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def extras_text(self) -> str:
        """追加プロンプト (カメラ / ライティング / 画風 / 動画用カメラワーク) の表示用テキスト。"""
        lines: list[str] = []
        for key, label in EXTRA_KINDS.items():
            e = self.extras.get(key) or {}
            lines.append(f"[{label}]")
            if e.get("tags"):
                lines.append("tags: " + ", ".join(e["tags"]))
            if e.get("text"):
                lines.append(e["text"] + ("  (ルールベース)" if e.get("source") == "rule" else ""))
            if not e.get("tags") and not e.get("text"):
                lines.append("(「人物の特徴を詳しく」をオンにすると、VLM が読んだ顔の細部が出ます)" if key == "face" else "(該当なし)")
            lines.append("")
        return "\n".join(lines).rstrip()

    def tag_lines(self) -> str:
        def mark(t: dict[str, Any]) -> str:
            if t["kind"] == "character":
                return "  [character]"
            source = t.get("source")
            return f"  [{REF_SLOTS[source]}]" if source in REF_SLOTS and source != "person" else ""

        return "\n".join(f"{t['tag']}  {t['conf']:.2f}" + mark(t) for t in self.tags)

    def person_lines(self) -> str:
        """人物ごとのプロンプト (Regional Prompting / Attention Couple / NovelAI の Character 欄などに貼る用)。"""
        lines: list[str] = []
        for i, p in enumerate(self.persons, 1):
            where = p.get("position_ja") or p.get("position") or "?"
            lines.append(f"[{i}] {where}: {p['prompt']}")
            if p.get("description"):
                lines.append(f"    {p['description']}")
            if p.get("area"):
                a = p["area"]
                lines.append(f"    範囲 (画像に対する %): 左 {a['x']} / 上 {a['y']} / 幅 {a['w']} / 高さ {a['h']}")
            if self.target == "novelai" and p.get("nai_position"):
                lines.append(f"    NovelAI の Character Positions (Custom、5×5 のマス): {p['nai_position']}")
            if self.target == "novelai" and p.get("undesired"):
                lines.append("    特徴が漏れたときにこのキャラの Undesired Content に入れる候補: " + ", ".join(p["undesired"]))
            if p.get("dropped"):
                lines.append("    写り込みの可能性で外したタグ (隣の人と重なっていて、その人の方が確度が高い): " + ", ".join(p["dropped"]))
            if p.get("dropped_colors"):
                lines.append("    写り込みの可能性で外した色 (この人の 2 番目の色が、ほかの人の一番強い色と同じ): "
                             + ", ".join(p["dropped_colors"]))
        if self.background_people:
            lines += ["", f"[奥に小さく写った人: {len(self.background_people)} 人]",
                      "Danbooru の人数タグは背景の人や群衆を数えないので、人数タグと人物ごとの区画には入れていません。"
                      "描かせたいときは共有部分に crowd などを足してください。"]
        actions = (self.persons[0].get("interactions") if self.persons else None) or []
        if actions:
            by_ja = {"vlm": "VLM で判定", "skeleton": "骨格で判定", "count": "2 人だけなので"}
            lines += ["", "[人どうしの動作 (共有部分に入れ、自然文では関係の文に)]"]
            for it in actions:
                pair = it.get("pair")
                if not pair:
                    who = "向きが分からないので、向きを書いていません"
                elif it.get("mutual"):
                    who = f"人物 {pair[0] + 1} と人物 {pair[1] + 1} (お互い、{by_ja.get(it.get('by'), '')})"
                else:
                    who = f"人物 {pair[0] + 1} → 人物 {pair[1] + 1} ({by_ja.get(it.get('by'), '')})"
                lines.append(f"  {unescape_parens(it['tag'])}: {who}")
        for title, text in self.formats.items():
            lines += ["", f"[{title}]", text]
        return "\n".join(lines)

    def to_text(self) -> str:
        parts = [
            "\n".join(
                [f"# {REF_SLOTS['person']}: {self.image}"]
                + [f"# {REF_SLOTS[k]}: {v}" for k, v in self.references.items()]
            ) if self.references else f"# {self.image}",
            "",
            "## 分析",
            self.analysis,
            "",
            "## Positive",
            self.positive_prompt,
            "",
            "## Negative",
            self.negative_prompt or "(この出力先はネガティブプロンプト非対応のため出力していません)",
            "",
        ]
        if self.persons:
            parts += ["## 人物ごと (Regional Prompting 用)", self.person_lines(), ""]
        if self.quality_issues:
            parts += ["## 品質チェック"] + [f"- {q}" for q in self.quality_issues] + [""]
        if self.mosaic_prompt:
            parts += ["※ NSFW 判定のため、ポジティブにモザイク指定 (censored, mosaic censoring 系) を、ネガティブに uncensored を追加しました", ""]
        if self.extras:
            parts += ["## 追加プロンプト", self.extras_text(), ""]
        parts += [
            "## メモ",
            self.notes,
            "",
            "## 検出タグ (信頼度)",
            self.tag_lines(),
            "",
        ]
        if self.warnings:
            parts += ["## 警告"] + [f"- {w}" for w in self.warnings] + [""]
        parts.append(
            f"(target: {self.target} / engine: {self.engine} / tagger: {self.tagger_model}"
            + (f" / vlm: {self.vlm_model}" if self.vlm_model else "")
            + f" / medium: {self.detected_medium} / {self.elapsed_sec:.1f}s)"
        )
        return "\n".join(parts) + "\n"


class Engine:
    """読み込んだモデルを保持し、複数の画像で使い回す。"""

    def __init__(self, cfg: dict[str, Any], progress: ProgressFn | None = None):
        self.cfg = cfg
        self.progress = progress or (lambda _msg: None)
        # 進み具合 (0.0〜1.0)。VLM の間は (見込み秒数, 始まり, 終わり) も渡す。見込みが無ければ None
        self.fraction: Callable[[float, tuple[float, float, float] | None], None] = lambda _f, _est=None: None
        self.last_vlm: tuple[str, float] | None = None  # (モデル, 実際にかかった秒数)
        self._tagger: WD14Tagger | None = None
        self._vlm: OllamaVLM | None = None
        self._writer: OllamaVLM | None = None   # 説明を書いている最中の describer (中止で接続を切る)
        self.describe_seconds = 0.0             # describer にかかった秒数 (VLM の見込み時間から除く)
        self._era: EraClassifier | None = None
        self._person: PersonDetector | None = None
        self._head: HeadDetector | None = None
        self._pose: PoseEstimator | None = None
        self._cancel = threading.Event()
        self.pause_gate = PauseGate()

    def reset_cancel(self) -> None:
        """解析を始める前に呼ぶ (中止フラグを下ろす)。"""
        self._cancel.clear()
        self.pause_gate.resume()

    def cancel(self) -> None:
        """実行中の run() を中止する。VLM の応答待ちなら接続を切ってすぐ止める。
        タガーの推論中は切れないので、その処理が終わった時点で止まる。"""
        self._cancel.set()
        if self._vlm is not None:
            self._vlm.abort()
        writer = self._writer
        if writer is not None:
            writer.abort()

    def _check(self) -> None:
        if self._cancel.is_set():
            raise OperationCancelled()
        if self.pause_gate.paused:  # 段階の切れ目で一時停止 (再開まで待つ)
            self.pause_gate.wait(self._cancel.is_set, self._on_paused)
            self.progress("再開しました")

    def _on_paused(self) -> None:
        self.progress("一時停止中 (「▶ 再開」で続きから。「■ 中止」で止められます)")
        if self._low_vram() and self._vlm is not None:
            self._vlm.unload()  # 止まっている間は VRAM を空ける (ComfyUI と共有のため)

    def _low_vram(self) -> bool:
        return bool(self.cfg.get("low_vram", True))

    def era_classifier(self) -> EraClassifier:
        if self._era is None or self._era.cpu_only != self._low_vram():
            self._era = EraClassifier(self.progress, self._cancel, cpu_only=self._low_vram())
        return self._era

    def person_detector(self) -> PersonDetector:
        if self._person is None:
            self._person = PersonDetector(self.progress, self._cancel)
        return self._person

    def head_detector(self) -> HeadDetector:
        if self._head is None:
            self._head = HeadDetector(self.progress, self._cancel)
        return self._head

    def detect_persons(
        self, image: Image.Image, warnings: list[str],
    ) -> tuple[list[tuple[float, Box]], list[tuple[float, Box]]]:
        """人物を検出し、頭の位置で枠を直す (refine_person_boxes)。戻り値: (人物の枠, 頭の枠)。
        頭部検出に失敗したら人物検出のまま続ける (頭の枠は空)。"""
        detections = self.person_detector().detect(image)
        try:
            heads = self.head_detector().detect(image)
        except OperationCancelled:
            raise
        except Exception as e:  # noqa: BLE001
            warnings.append(f"頭部検出をスキップしました (人物検出の枠のまま): {describe_error(e)}")
            return detections, []
        refined, split, added = refine_person_boxes(image.size, detections, heads)
        if split or added:
            parts = ([f"2 人以上が入った枠 {split} つを頭の位置で分けました"] if split else []) + (
                [f"人物検出で見つからなかった {added} 人を頭の位置から足しました"] if added else [])
            warnings.append("人物の検出: " + "、".join(parts) + "。")
        return refined, heads

    def face_image(self, image: Image.Image, box: Box | None) -> tuple[Image.Image | None, str]:
        """人物の枠 (無ければ一番大きい人を探す) から骨格を推定し、元画像から顔を切り出す (人物の特徴を詳しく)。
        戻り値: (顔の画像 | None, 切り出せなかった理由)。モデルはここでは取得しない (GUI / CLI が確認してから取得する)。"""
        if not face_models_ready():
            return None, "骨格モデル・人物検出モデルが無いため"
        if self._pose is None:
            self._pose = PoseEstimator()
        if box is None:
            box = main_person_box(self.person_detector().detect(image))
        found = face_box(self._pose.estimate(image, box), image.size)
        if found is None:
            return None, "顔の位置が分からなかったため"
        if box_area(found) >= box_area(box or (0, 0, image.width, image.height)) * 0.5:
            return None, ""  # 顔のアップの絵: 人物の画像がすでに顔なので、もう 1 枚は渡さない
        face = image.convert("RGB").crop(found)
        if max(face.size) < FACE_VLM_SIDE:
            # 全身の絵では顔が 200px ほど。Qwen-VL は画像を一定の大きさのパッチで読むので、小さいままだと細部に割く量が少ない
            scale = FACE_VLM_SIDE / max(face.size)
            face = face.resize((round(face.width * scale), round(face.height * scale)), Image.LANCZOS)
        return face, ""

    def tagger(self) -> WD14Tagger:
        name = self.cfg.get("tagger_model") or DEFAULT_TAGGER
        cpu_only = self._low_vram()
        if self._tagger is None or self._tagger.name != name or self._tagger.cpu_only != cpu_only:
            self._tagger = None  # 先に手放して、古いセッションの VRAM を解放してから作り直す
            self._tagger = WD14Tagger(name, self.progress, cpu_only=cpu_only)
        return self._tagger

    def vlm(self) -> OllamaVLM:
        host = normalize_host(str(self.cfg.get("ollama_host") or ""))
        model = self.cfg.get("ollama_model") or DEFAULT_CONFIG["ollama_model"]
        try:
            timeout = max(30, int(self.cfg.get("ollama_timeout") or VLM_TIMEOUT))
        except (TypeError, ValueError):
            timeout = VLM_TIMEOUT
        if self._vlm is None or self._vlm.host != host or self._vlm.model != model:
            self._vlm = OllamaVLM(host, model, timeout)
        self._vlm.timeout = timeout
        self._vlm.keep_alive = 0 if self._low_vram() else None
        self._vlm.cancel_event = self._cancel
        return self._vlm

    def _describer_label(self) -> str:
        name = str(self.cfg.get("describer_model") or "")
        return "JoyCaption" if name == JOYCAPTION_MODEL else name

    def _describe(self, image: Image.Image, vlm: OllamaVLM, warnings: list[str]) -> str:
        """describer_model が設定されていれば、いつもの VLM の前に画像の説明を書かせる。失敗しても解析は続ける
        (理由は warnings に入れる)。いつもの VLM を続けて読むので、省 VRAM 設定ならすぐ VRAM から降ろす。"""
        name = str(self.cfg.get("describer_model") or "").strip()
        if not name or name == vlm.model:
            return ""
        if "vision" not in vlm.capabilities():   # いつもの VLM が画像を読めないなら、説明を書かせても使えない
            return ""
        self._check()
        self.progress(f"{self._describer_label()} で画像の説明を書いています... (「■ 中止」で止められます)")
        writer = OllamaVLM(vlm.host, name, vlm.timeout)
        writer.keep_alive = 0 if self._low_vram() else None
        writer.cancel_event = self._cancel
        started = time.time()
        self._writer = writer
        try:
            self._check()   # 中止が _writer を置く直前に来たとき
            return writer.describe(image)
        except OperationCancelled:
            raise
        except Exception as e:  # noqa: BLE001 - 説明が書けなくても、いつもの VLM で続ける
            if self._cancel.is_set():
                raise OperationCancelled() from e
            warnings.append(f"{self._describer_label()} の説明をスキップしました: {describe_error(e)}")
            return ""
        finally:
            self._writer = None
            self.describe_seconds = time.time() - started

    @staticmethod
    def _open(path: Path) -> Image.Image:
        with Image.open(path) as opened:
            opened.load()  # アニメーション GIF は先頭フレーム
            return upright(opened)

    def run(self, path: Path, refs: dict[str, Path] | None = None, controls: dict[str, Any] | None = None) -> PromptResult:
        """1 枚を解析する。refs ({"pose" / "outfit" / "background": パス}) を渡すと複数枚モード (path = ① メイン)。
        refs に無い枠の要素は 1 枚目から取る。controls はボーン・カメラ・強さの指定 (apply_controls / apply_strength)。"""
        cfg = self.cfg
        warnings: list[str] = []
        refs = {k: Path(v) for k, v in (refs or {}).items() if k in REF_SLOTS and k != "person" and v}
        general_th = float(cfg.get("general_threshold", 0.35))
        character_th = float(cfg.get("character_threshold", 0.85))

        tagger = self.tagger()
        started = time.time()  # モデル読み込みを除いた 1 枚あたりの処理時間
        self.fraction(0.02, None)
        inputs = self._tag_inputs(tagger, path, refs, general_th, character_th, warnings)
        timing: list[tuple[str, float]] = [("タガーと検出", time.time() - started)]  # どこに時間がかかったか (分析メモの最後に出す)
        self.fraction(0.35, None)
        controls = controls or {}
        control_flags = {"pose": False, "camera": False}
        if controls.get("pose") or controls.get("camera"):
            import dataclasses

            new_tags, new_source, removed, notes, control_flags = apply_controls(inputs.tags, inputs.tag_source, controls)
            displaced = dict(inputs.displaced)
            if removed:  # 置き換えた元のタグは、VLM が書いても戻さない (① から押し出したタグと同じ扱い)
                displaced["control"] = dedup(displaced.get("control", []) + [sd_tag(n) for n in removed])
            inputs = dataclasses.replace(inputs, tags=new_tags, tag_source=new_source, displaced=displaced)
            warnings += notes
        overridden = control_flags["pose"] or control_flags["camera"]
        look_spec = controls.get("look") if isinstance(controls.get("look"), dict) else {}
        if compose_look_parts(controls.get("look_parts")):  # 撮影の組み合わせ (項目ごとに選んだ語。プリセットより優先)
            look_spec = {"name": LOOK_PARTS_NAME, "prompt": compose_look_parts(controls.get("look_parts"))}
        look = str(look_spec.get("prompt") or "").strip()  # 撮影プリセットの文 (選んでいなければ空)
        turnaround = bool(cfg.get("turnaround", False))  # 三面図にする
        no_identity = bool(cfg.get("no_identity", False))  # 人物の特徴を入れない (LoRA 用)
        no_style = bool(cfg.get("no_style", False))  # 作風の特徴を入れない (画風 LoRA 用)
        # 年齢の見た目を書く。人物の特徴を入れないときは書かない (年齢もその人の特徴)
        age_on = bool(cfg.get("age_look", True)) and not no_identity
        run_target = cfg.get("target") if cfg.get("target") in TARGETS else DEFAULT_TARGET
        # 画風を変える: MOSS-Image だけ
        restyle = cfg.get("restyle") if cfg.get("restyle") in RESTYLES and run_target == "mossimage" else "none"
        if restyle != "none" and no_style:
            no_style = False
            warnings.append("「画風を変える」を選んでいるので、「作風の特徴を入れない」は使いませんでした (画風を書く必要があります)")
        nsfw_before = nsfw_suspected(inputs.tags)  # 表示用 (ステータス欄 / JSON)。条件には使わない
        mono_removed: list[str] = []
        if cfg.get("remove_monochrome", False):
            # 元画像が白黒でもカラーで出す: 白黒のタグを外し (以降の追加プロンプト・画風判定・ネガティブにも効く)、ネガティブへ回す
            import dataclasses

            new_tags, mono_removed = remove_monochrome(inputs.tags)
            if mono_removed:
                inputs = dataclasses.replace(
                    inputs, tags=new_tags, person_raw=remove_monochrome(inputs.person_raw)[0],
                    persons=[p | {"tags": [t for t in p["tags"] if _tag_key(t) not in MONOCHROME_TAGS],
                                  "prompt": ", ".join(t for t in p["prompt"].split(", ") if _tag_key(t) not in MONOCHROME_TAGS)}
                             for p in inputs.persons],
                )
        gender = cfg.get("gender") if cfg.get("gender") in GENDER_MODES else "none"
        if gender != "none":
            # 長髪の男性・短髪の女性の取り違えを防ぐため、人数タグと人物ごとの性別を指定の性別にそろえる
            import dataclasses

            inputs = dataclasses.replace(
                inputs, tags=force_gender(inputs.tags, gender), person_raw=force_gender(inputs.person_raw, gender),
                persons=force_person_gender(inputs.persons, gender),
            )
        detected = inputs.tags  # 「検出タグ」に出す (検閲のタグも含む)
        if any(n in CENSOR_DETECT_TAGS for n, _ in inputs.tags.general):
            # タガーの検閲のタグは誤認が多いので本文には入れない (検出タグには確信度付きで残る)
            import dataclasses

            def _drop(t: TagResult) -> TagResult:
                return TagResult(t.rating, t.rating_conf, [(n, q) for n, q in t.general if n not in CENSOR_DETECT_TAGS],
                                 list(t.characters), dict(t.general_probs))

            inputs = dataclasses.replace(
                inputs, tags=_drop(inputs.tags), person_raw=_drop(inputs.person_raw),
                persons=[pp | {"tags": [t for t in pp["tags"] if _tag_key(t) not in CENSOR_DETECT_TAGS],
                               "prompt": ", ".join(t for t in pp["prompt"].split(", ") if _tag_key(t) not in CENSOR_DETECT_TAGS)}
                         for pp in inputs.persons],
            )
        tags, ref_tags, tag_source, size = inputs.tags, inputs.ref_tags, inputs.tag_source, inputs.size
        medium = detect_medium(tags, STYLE_REF_PHOTO_THRESHOLD if "style" in ref_tags else PHOTO_THRESHOLD)
        # ⑦ 作風・質感が、選んだ種類と同じ系統 (写真 / 絵画 / 絵) か。同じなら ⑦ から読んだ画風の文とタグを使う
        style_matches = restyle != "none" and "style" in ref_tags and medium == RESTYLES[restyle]["medium"]
        if restyle != "none":
            medium = RESTYLES[restyle]["medium"]  # 元の画像に関係なく、選んだ種類として書く (文の画材、VLM への手がかり)
        nsfw = nsfw_before or nsfw_suspected(tags)
        target = cfg.get("target") if cfg.get("target") in TARGETS else DEFAULT_TARGET
        kind = TARGETS[target]["kind"]

        negative = build_negative(tags, medium, target)
        if mono_removed:
            negative = dedup(negative + [t.replace("_", " ") for t in mono_removed] + ["monochrome", "greyscale"])
        same_face = bool(cfg.get("avoid_same_face", False)) and person_count_hint(tags)[0] >= 2
        if same_face and kind != "prose":
            negative = dedup(negative + SAME_FACE_NEGATIVES)  # 元画像にあるものは後の「写っているものは外す」で除かれる
        elif same_face and TARGETS[target]["negative"] in ("optional", True):
            negative = dedup(negative + SAME_FACE_NEGATIVES)  # Qwen-Image / Flux / Z-Image のネガティブ欄
        avoid_dup = bool(cfg.get("avoid_duplicates", False))
        if avoid_dup and (kind != "prose" or TARGETS[target]["negative"] in ("optional", True)):
            negative = dedup(negative + DUPLICATE_NEGATIVES)  # 元画像にあるもの (双子・設定画など) は後で外れる
        if cfg.get("anatomy_guard", True):
            guard = anatomy_negatives(tags, target, kind)
            if guard:
                negative = dedup(negative + guard)
                notes_anatomy = ANATOMY_NOTE
            else:
                notes_anatomy = ""
        else:
            notes_anatomy = ""
        analysis = build_analysis(tags, medium, size, general_th)
        notes = build_notes(medium, size, target) + notes_anatomy
        pair_header = ""
        if ref_tags:
            pair_header, notes_add = reference_header(path, refs, tags, tag_source)
            if inputs.displaced.get("style"):
                # ① の画風を残すと「アニメ塗り + 写真」のように混ざるので外す。外したことはメモに出す
                pair_header += "⑦ 作風・質感があるので、① の画風のタグを外しました: " + ", ".join(inputs.displaced["style"]) + "\n"
            if "style" in ref_tags:
                pair_header += (f"絵か写真かの判定は ⑦ 作風・質感から: {MEDIUM_JA.get(medium, medium)} (⑦ の写真らしさ "
                                f"{photo_score(tags):.2f}。⑦ があるときは {STYLE_REF_PHOTO_THRESHOLD} 以上で写真)\n")
            analysis = pair_header + analysis
            notes += notes_add

        if inputs.main_person is not None:
            analysis += "\n" + main_person_note(inputs.main_person)
        if len(inputs.persons) >= 2 and kind == "tags" and target != "novelai":
            # A1111 の wiki: BREAK は 75 トークンのチャンクを区切るだけ。人ごとの分離は領域指定の仕組みが担う
            # (Regional Prompter / Attention Couple)。ComfyUI 標準の CLIP Text Encode は BREAK を解釈しない
            notes += (
                "\n複数人物: BREAK はプロンプトの区切り (75 トークンのチャンク) で、それだけでは人ごとの特徴は分かれません。"
                "人ごとに分けるには「人物ごと」タブの Regional Prompter 用 (A1111 / Forge) か、ComfyUI 領域指定の値を使ってください。"
                "ComfyUI 標準の CLIP Text Encode は BREAK に対応していないので、BREAK 入りの本文は CLIP Text Encode++ (smZ Nodes) "
                "に入れるか、BREAK で分けて Conditioning (Concat) でつないでください。"
            )
        elif len(inputs.persons) >= 2 and target == "novelai":
            notes += (
                "\n複数人物 (NovelAI 公式の書き方): 本文をベースプロンプトに、「人物ごと」の各行を Character 1, 2 … に貼ります "
                "(先頭は girl / boy / other、数字は付けない)。配置は並び順 (左から) が手がかりになります。"
                "正確に置くときは Character Positions を Custom にし、各行の位置 (5×5 のマス) に置いてください。V4 / V4.5 は 6 人、V5 は 22 人まで"
                "です (i2t は 22 人まで分けます。V5 の位置指定の詳しい仕様は公式に無いので、位置のマスは V4 の 5×5 で出しています)。"
                "人どうしの動作は、向きが分かったものを各行に source# / target# / mutual# で入れています (公式: いつも効くわけではない)。"
            )
        background_people = inputs.persons[0].get("background_people", []) if inputs.persons else []
        if background_people:
            analysis += (
                f"\n奥に小さく写った {len(background_people)} 人は、Danbooru の数え方 (背景の人は人数に数えない) に合わせて "
                "人数と人物ごとの区画に入れていません"
            )
        if inputs.persons:
            analysis += (
                f"\n人物ごと: {len(inputs.persons)} 人を検出し、1 人ずつ切り出してタグ付けしました (左から "
                + " / ".join(p["position_ja"] for p in inputs.persons) + "。プレビューの番号付きの枠)"
            )
        # 年代は ① の画像で判定する (⑦ 作風・質感では取らない)
        era_info = self._era_step(inputs.image, medium, warnings)
        self.fraction(0.40, None)
        if era_info:
            line = f"年代スタイル: {era_info['label_ja']} ({era_info['label']}, {era_info['conf']:.0%})"
            if not era_info["tags"]:
                line += "  ※確度が低いためタグには入れていません"
            analysis += "\n" + line

        hint_count, hint_tags = person_count_hint(tags)
        use_vlm = cfg.get("engine", DEFAULT_CONFIG["engine"]) == "hybrid"
        vlm = VlmOutcome()
        if use_vlm:
            model = str(cfg.get("ollama_model") or "")
            seconds = (cfg.get("vlm_seconds") or {}).get(model) if isinstance(cfg.get("vlm_seconds"), dict) else None
            est = float(seconds) if isinstance(seconds, (int, float)) and seconds > 0 else 0.0
            self.fraction(0.40, (est, 0.40, 0.95) if est else None)
            vlm_started = time.time()
            self.describe_seconds = 0.0
            vlm = self._run_vlm(
                numbered_boxes_image(inputs.vlm_image, [p["box"] for p in inputs.persons])
                if len(inputs.persons) >= 2 and not ref_tags else inputs.vlm_image,
                vlm_person_tags(inputs.person_raw, tags, ref_tags) if ref_tags else tags, medium, negative, target, size,
                [(slot, inputs.ref_images[slot], ref_tags[slot]) for slot in REF_SLOTS if slot in ref_tags],
                hint_count, hint_tags,
                extra_guidance="\n".join(x for x in (
                    MAIN_PERSON_VLM if inputs.main_person is not None
                    else split_person_vlm(inputs.persons, target) if inputs.persons else "",
                    GENDER_VLM.get(gender, ""),
                    MONOCHROME_VLM if mono_removed else "",
                    TURNAROUND_VLM if turnaround else "",
                    NO_IDENTITY_VLM if no_identity else "",
                    NO_STYLE_VLM if no_style else "",
                    restyle_vlm(restyle) if restyle != "none" else "",
                    AGE_VLM if age_on else "",
                ) if x),
                look=look_prose(look) if look else "",
                describe_image=inputs.vlm_image,
                persons_guided=inputs.main_person is not None or bool(inputs.persons),
                # 枠ごとの内容を並べる条件 (複数枚モード・ボーン / カメラの指定・2 人以上) では、この後に仕上げの書き直しがある
                polish_next=(kind == "prose" and target != "midjourney" and bool(cfg.get("polish_prose", True))
                             and (bool(ref_tags) or overridden or len(inputs.persons) >= 2)),
            )
            warnings += vlm.warnings
            timing.append(("VLM" + (f" ({1 + len([s for s in ref_tags if s in VLM_SEPARATE_SLOTS])} 回)"
                                    if any(s in VLM_SEPARATE_SLOTS for s in ref_tags) else ""), time.time() - vlm_started))
            if vlm.ok:
                self.last_vlm = (model, round(time.time() - vlm_started - self.describe_seconds, 1))
            self.fraction(0.95, None)
            if vlm.analysis:
                analysis = pair_header + vlm.analysis
            if vlm.notes:
                notes = vlm.notes if target == "sdxl" else f"{notes}\n{vlm.notes}"
            own_style = [n for n in vlm.negative if not is_generic_phrase(n) and contradicts_medium(n, medium)]
            if own_style:
                # アニメ絵のネガティブに cel shaded / manga style など、元の画像の画風そのものを打ち消す語 (実機の qwen3.5 で確認)
                warnings.append("VLM がネガティブに元の画像の画風 (" + MEDIUM_JA.get(medium, medium) + ") を打ち消す語を挙げたので外しました: "
                                + ", ".join(own_style))
            generic = [n for n in vlm.negative if is_generic_phrase(n) or n in own_style]
            if generic:
                # face / hair / other outfits などはキャラの顔・髪まで打ち消してしまう (実機の qwen2.5vl で確認)
                if [n for n in generic if n not in own_style]:
                    warnings.append("VLM がネガティブに挙げた、具体的な特徴を表さない語を外しました: "
                                    + ", ".join(n for n in generic if n not in own_style))
            negative += [n for n in vlm.negative if n not in generic]
        elif kind == "prose":
            warnings.append(
                f"{TARGETS[target]['label']}: VLM 無しのため、タグから組み立てた英文を出力しました "
                "(「タガー + VLM」にすると画像を見た自然な文になります)。"
            )

        self._check()
        if inputs.persons:
            # 人物ごとの内容はタガーで 1 人ずつ付けたものを使う。VLM の persons は並び順が左からとは限らず
            # (実機の qwen2.5vl で「左から 2 番目」の人に「中央の女性」の説明が来た)、順番では結び付けない。
            # ポーズ・服装だけは、画像に描いた番号付きの枠の番号で結び付けて足す
            persons = [dict(p) for p in inputs.persons]
            if vlm.ok:
                persons, numbered_warn = add_numbered_person_tags(persons, vlm.numbered)
                warnings += numbered_warn
                persons, inter_warn = apply_vlm_interactions(persons, vlm.interactions)
                warnings += inter_warn
        else:
            persons = vlm.persons if not ref_tags and inputs.main_person is None else []
            persons = force_person_gender(persons, gender)
        detail_tags_main: list[str] = []
        detail_summary = ""
        face_extra = empty_face_extra()
        if cfg.get("person_details", False):
            details_started = time.time()
            persons, detail_tags_main, detail_summary, detail_notes, detail_warn, face_extra = self._person_details(
                inputs, persons, use_vlm, with_build="body" not in ref_tags, medium=medium)
            timing.append(("人物の特徴を詳しく", time.time() - details_started))
            warnings += detail_warn
            if detail_notes:
                analysis += "\n" + detail_notes
        if hint_count >= 2 and not vlm.ok and not ref_tags and not inputs.persons:
            warnings.append(
                f"複数人物 ({hint_count} 人) を検出しました。人物ごとの属性の振り分けには「タガー + VLM」が必要です。"
            )
        if control_flags["pose"]:  # ボーンで決めたポーズがあれば、VLM が画像から読んだポーズは使わない
            vlm.pose_tags, vlm.ref_pose_tags = [], []
        extra_tags = list(vlm.extra_tags)
        if control_flags["camera"]:  # カメラを指定した種類は、VLM の追加タグからも外す
            chosen = camera_tags(controls.get("camera") or {})
            clash = set().union(*(CAMERA_DIMENSION_TAGS[d] for d in chosen))
            extra_tags = [t for t in extra_tags if norm_tag(t).replace(" ", "_") not in clash]
        if not ref_tags and len(inputs.persons) < 2 and (hint_count <= 1 or inputs.main_person is not None):
            # 1 枚モード・1 人: VLM が読んだポーズ・服装のタグも本文に入れる (タガーは細かいポーズを拾えないため)
            extra_tags = dedup(extra_tags + vlm.pose_tags + vlm.outfit_tags)
        if mono_removed:  # VLM が補ったタグの白黒の語も外す
            extra_tags = [t for t in extra_tags if _tag_key(t) not in MONOCHROME_TAGS and not mentions_any(t, MONOCHROME_WORDS)]
        displaced_all = [t for tags_ in inputs.displaced.values() for t in tags_]
        if displaced_all:
            # VLM は ② / ③ の役割を守らず ① の服やポーズを挙げることがある (qwen2.5vl で確認)。① から押し出したタグは戻さない
            displaced_keys = {loose_tag_key(t) for t in displaced_all}
            dropped = [t for t in extra_tags if loose_tag_key(t) in displaced_keys]  # 表記ゆれ (複数形・語順など) も同じタグとみなす
            extra_tags = [t for t in extra_tags if t not in dropped]
            if dropped:
                warnings.append("VLM が ① の服・ポーズのタグを挙げたので外しました: " + ", ".join(dropped))
        # 複数枚モード: VLM が ③ / ② から読んだ服装・ポーズのタグ (① から押し出したもの・白黒の語は除く)
        ref_vlm: dict[str, list[str]] = {}
        if "outfit" in ref_tags:
            ref_vlm["outfit"] = list(vlm.ref_outfit_tags)
        if "pose" in ref_tags:
            ref_vlm["pose"] = list(vlm.ref_pose_tags)
        if "body" in ref_tags:
            ref_vlm["body"] = list(vlm.ref_body_tags)
        if "expression" in ref_tags:
            ref_vlm["expression"] = list(vlm.ref_expression_tags)
        ref_vlm, scene_tags, off_role = sort_ref_vlm_tags(
            ref_vlm, {slot: [sd_tag(n) for n, _ in ref_tags[slot].general] for slot in ref_vlm})
        extra_tags = dedup(extra_tags + scene_tags)
        if off_role:
            warnings.append("VLM が服装・ポーズのタグに入れた、枠の役割に合わないタグ (① の髪・顔・表情や重複の言い換え) を外しました: "
                            + ", ".join(off_role))
        if displaced_all or mono_removed:
            displaced_keys = {loose_tag_key(t) for t in displaced_all}
            for slot, items in ref_vlm.items():
                dropped = [t for t in items if loose_tag_key(t) in displaced_keys]
                if dropped:
                    warnings.append(f"VLM が {REF_SLOTS[slot]} のタグに ① の服・ポーズを挙げたので外しました: " + ", ".join(dropped))
                ref_vlm[slot] = [
                    t for t in items if t not in dropped
                    and not (mono_removed and (_tag_key(t) in MONOCHROME_TAGS or mentions_any(t, MONOCHROME_WORDS)))
                ]
        if ref_tags:
            # VLM は別の画像の要素を混ぜる (実機の qwen2.5vl: ③ の服装のタグに ① の red shirt / white skirt と ② の black jacket、
            # 追加タグに ③ の beach)。どの画像からも取らなかったタグ (① から押し出したもの、参照画像の持ち主でない要素) を
            # 語として含む VLM のタグは外す。追加タグ (構図・照明・雰囲気) に服装・ポーズが入っていたら、それも外す (服装・ポーズの欄が受け持つ)
            foreign = foreign_reference_tags(tags, ref_tags, displaced_all)
            merged_keys = {loose_tag_key(sd_tag(n)) for n, _ in tags.general}

            def is_foreign(t: str) -> bool:  # 本文にあるタグ (arms up) と表記ゆれで同じなら、ほかの画像の arm up とみなさない
                return loose_tag_key(t) not in merged_keys and mentions_foreign(t, foreign)

            removed: list[str] = []
            for slot in list(ref_vlm):
                bad = [t for t in ref_vlm[slot] if is_foreign(t)]
                removed += bad
                ref_vlm[slot] = [t for t in ref_vlm[slot] if t not in bad]
            removed += [t for t in extra_tags if is_foreign(t)]
            role = [t for t in extra_tags if t not in removed and t in vlm.extra_tags and (
                pose_source_tag(norm_tag(t).replace(" ", "_")) and classify_extra_tag(norm_tag(t).replace(" ", "_")) != "camera"
                or classify_outfit_tag(norm_tag(t).replace(" ", "_")) or body_tag(norm_tag(t).replace(" ", "_")))]
            extra_tags = [t for t in extra_tags if t not in removed and t not in role]
            if removed:
                warnings.append("VLM がほかの画像の要素 (その枠から取らない服・ポーズ・背景など) を挙げたので外しました: "
                                + ", ".join(dedup(removed)))
            if role:
                warnings.append("VLM の追加タグ (構図・照明・雰囲気の欄) にあった服・ポーズ・体格のタグは、それぞれの枠が受け持つので外しました: "
                                + ", ".join(dedup(role)))
        char_tags = [sd_tag(n) for n, _ in tags.characters]
        general_tags = [sd_tag(n) for n, _ in tags.general]
        if len(persons) >= 2:
            # 人物に割り当てたタグを共有タグから外す (要素の混在を防ぐ)
            assigned = {norm_tag(t) for p in persons for t in p["tags"]}
            shared_general = [t for t in general_tags if norm_tag(t) not in assigned]
            shared_chars = [t for t in char_tags if norm_tag(t) not in assigned]
        else:
            shared_general, shared_chars = general_tags, char_tags
        if ref_tags:
            # 服装 → 小物 → ポーズ → 人物 → 背景の順 (merge_reference_tags と同じ)。VLM が読んだ服装・ポーズのタグは
            # 末尾ではなく、それぞれタガーの服装・ポーズのタグの後ろに置く (後ろの区画に押し出されて効かないため)
            shared_general = order_reference_tags(tags, tag_source, ref_vlm | {"person": detail_tags_main})
        elif detail_tags_main:
            # 1 枚モード: 詳しく読んだ特徴は人数タグのすぐ後ろに (末尾だと効かない)
            have = {loose_tag_key(t) for t in shared_general}
            add = [t for t in detail_tags_main if loose_tag_key(t) not in have]
            head = [t for t in shared_general if norm_tag(t).replace(" ", "_") in PERSON_COUNT_TAGS or norm_tag(t) in ("solo",)]
            shared_general = head + add + [t for t in shared_general if t not in head]

        # 追加プロンプト (被写体 / ポーズ / 服装 / 物体 / 背景 / カメラ / ライティング / 画風 / 動画用カメラワーク)
        try:
            style_th = min(0.95, max(0.01, float(cfg.get("style_threshold", 0.15))))
        except (TypeError, ValueError):
            style_th = 0.15
        extras = build_extras(tags, vlm.out, extra_tags, kind, style_th, era_info, references=bool(ref_tags))
        extras["face"] = face_extra
        if "style" in ref_tags and extras["style"]["text"]:
            # 画風の文が判定 (⑦ から) と食い違う (写真なのに anime / illustration / 2D など) なら使わず、⑦ の画風タグで書く
            other_media = set().union(*(w for m, w in _MEDIUM_WORDS.items() if m != medium)) - _MEDIUM_WORDS.get(medium, set())
            clash = sorted(set(re.findall(r"[a-z0-9-]+", extras["style"]["text"].lower())) & other_media)
            if clash:
                extras["style"]["text"] = ""
                extras["style"]["prompt"] = ", ".join(extras["style"]["tags"])
                warnings.append(f"VLM の画風の文が ⑦ の判定 ({MEDIUM_JA.get(medium, medium)}) と食い違う語 ({', '.join(clash)}) を"
                                "含んでいたので、画風は ⑦ のタグで書きました")
        if "style" in ref_tags and extras["style"]["text"]:
            # ⑦ の VLM の画風の文に ⑦ の服・人物が入っていたら使わない (実機の qwen2.5vl:
            # 「a person wearing a serafuku, a traditional Japanese school uniform」が本文に入った)
            style_foreign = [sd_tag(n) for n, _ in ref_tags["style"].general
                             if reference_category(n, True) != "style" and n not in PERSON_COUNT_TAGS and n not in ("solo", "no_humans")]
            # 写真の光の当て方 (window light / sun lighting など) は撮影環境の語なので、場所の混入として数えない (2026-11-02)
            leaked = mentions_any(re.sub(r"\b[\w-]+\s+(light|lighting|lit|sunlight)\b", " ", extras["style"]["text"],
                                         flags=re.IGNORECASE), style_foreign)
            if re.search(r"\bwear(s|ing)?\b", extras["style"]["text"], re.IGNORECASE):
                leaked.append("wearing")
            if leaked:
                extras["style"]["text"] = ""
                extras["style"]["prompt"] = ", ".join(extras["style"]["tags"])
                warnings.append("VLM の画風の文に ⑦ の服・人物 (" + ", ".join(dedup(leaked))
                                + ") が入っていたので、画風はタグで書きました")
        if "style" in ref_tags:
            # ⑦ 作風・質感から VLM が推定した作品名・作者名 (タガーには無い)。作者と作品は分けず、どの出力先でも名前そのままで書く。
            # タグ形式はキャラ名のすぐ後ろ (公式の並び: NovelAI docs「characters, series,
            # everything else」、Anima / NoobAI のモデルカード「character, series, artist, general tags」)。
            # 自然文は「in the style of 名前」(BFL の FLUX.1 の例 / Gemini の画像生成ガイドの書き方)
            names = vlm.style_names
            if names and kind == "prose":
                phrase = "in the style of " + " and ".join(names)
                extras["style"]["tags"] = [phrase] + extras["style"]["tags"]
                if extras["style"]["text"]:
                    extras["style"]["text"] = phrase[0].upper() + phrase[1:] + ". " + extras["style"]["text"]
                extras["style"]["prompt"] = extras["style"]["text"] or ", ".join(extras["style"]["tags"])
            elif names:
                name_tags = [sd_escape(n.lower()) for n in names]
                extras["style"]["tags"] = dedup(name_tags + extras["style"]["tags"])
                extras["style"]["prompt"] = ", ".join(extras["style"]["tags"])
                shared_chars = dedup(shared_chars + name_tags)
            if names:
                analysis += "\n⑦ 作風・質感: VLM が推定した作品名・作者名: " + ", ".join(names) + " (VLM の推定です)"
            elif not use_vlm:
                analysis += "\n⑦ 作風・質感: 作品名・作者名は VLM でだけ推定できます (「タガーのみ」ではできません)"
            elif vlm.ok:
                analysis += "\n⑦ 作風・質感: VLM は作品名・作者名を挙げませんでした"
        if control_flags["pose"] and extras["pose"]["text"]:  # VLM のポーズの文は画像のポーズなので使わずタグで書く
            extras["pose"]["text"] = ""
            extras["pose"]["prompt"] = ", ".join(extras["pose"]["tags"])
        if control_flags["camera"]:
            if extras["camera"]["text"]:
                extras["camera"]["text"] = ""
            camera = controls.get("camera") or {}
            if kind == "prose":  # タグの無い向き・高さ (斜め・目線) は自然文系にだけ言葉で書く
                extras["camera"]["tags"] += [CAMERA_PROSE_EXTRA[v] for v in camera.values() if v in CAMERA_PROSE_EXTRA]
            extras["camera"]["prompt"] = ", ".join(extras["camera"]["tags"])
        natural_prompt = vlm.natural_prompt
        if mono_removed:
            # 白黒の情報を消すときは、VLM の文 (自然文・追加プロンプトの各欄) に白黒の言い方が残っていたら使わずタグで書く
            if natural_prompt and mentions_any(natural_prompt, MONOCHROME_WORDS):
                natural_prompt = ""
                warnings.append("白黒の情報を消す: VLM の自然文が白黒で書いていたので、タグから組み立てた文にしました")
            for key in extras:
                if key != "motion" and extras[key]["text"] and mentions_any(extras[key]["text"], MONOCHROME_WORDS):
                    extras[key]["text"] = ""
                    extras[key]["prompt"] = ", ".join(extras[key]["tags"])
        if inputs.displaced:
            # VLM の役割別の文 (被写体・服装・ポーズ) に ① の服・ポーズが混ざっていたら、その文は使わずタグで書く
            displaced_all = [t for tags_ in inputs.displaced.values() for t in tags_]
            displaced_keys = {loose_tag_key(t) for t in displaced_all}
            for key in extras:  # VLM の outfit_tags / pose_tags も各欄に直接入るので、そこからも ① のタグを外す
                if key == "motion":
                    continue
                kept_tags = [t for t in extras[key]["tags"] if loose_tag_key(t) not in displaced_keys]
                if len(kept_tags) != len(extras[key]["tags"]):
                    extras[key]["tags"] = kept_tags
                    extras[key]["prompt"] = ", ".join(kept_tags) if (kind != "prose" or not extras[key]["text"]) else extras[key]["prompt"]
            for key in ("subject", "outfit", "pose"):
                text = extras[key]["text"]
                leaked = mentions_any(text, displaced_all) if text else []
                if leaked:
                    extras[key]["text"] = ""
                    extras[key]["prompt"] = ", ".join(extras[key]["tags"])
                    warnings.append(f"VLM の「{EXTRA_KINDS[key]}」の文に ① の {', '.join(leaked)} が入っていたので、タグで書きました")
        if cfg.get("split_extras", False):
            # カメラ・照明・画風のタグは本文から外して追加プロンプト側だけに出す
            shared_general = [t for t in shared_general if classify_extra_tag(norm_tag(t).replace(" ", "_")) is None]
            extra_tags = [t for t in extra_tags if classify_extra_tag(norm_tag(t).replace(" ", "_")) is None]
        if cfg.get("split_background", False):
            # 背景のタグは本文から外し、本文を被写体だけにする (背景は追加プロンプトの「背景 / シーン」)
            has_humans = "no_humans" not in {n for n, _ in tags.general}
            shared_general = [
                t for t in shared_general
                if classify_subject_tag(norm_tag(t).replace(" ", "_"), has_humans) != "background"
            ]

        if target in REF_FRAMING_WEIGHT_TARGETS and (("pose" in ref_tags) or control_flags["camera"]):
            # ② の画角 (full body など) と向き (from behind など) は服装・人物のタグに負けて出ないことがあるので重みを上げる
            shared_general = [
                format_weight(t, REF_FRAMING_WEIGHT, target)
                if norm_tag(t).replace(" ", "_") in _FRAMING_ANY | REF_ANGLE_TAGS | {"facing_viewer"}
                and inputs.tag_source.get(norm_tag(t).replace(" ", "_")) in ("pose", "control") else t
                for t in shared_general
            ]
        # 年齢の見た目: 人物ごとに VLM が選んだ年齢層とタガーの年齢のタグから決めて、文とタグに書く。
        # 人物の特徴を入れないときは書かない (年齢もその人の特徴)。人物ごとに分けていない 2 人以上の画像は、誰の年齢か
        # 結び付けられないので書かない
        main_age = ""
        age_moss_negative: list[str] = []
        if age_on:
            age_single = len(persons) < 2 and (hint_count <= 1 or inputs.main_person is not None)
            persons, main_age, age_note = assign_ages(
                persons, vlm.age if vlm.ok else "", vlm.numbered if vlm.ok else [], tags, nsfw, age_single)
            analysis += age_note
            people_ages = ([(p.get("subject", ""), p.get("age", "")) for p in persons] if len(persons) >= 2
                           else [(single_subject(tags), main_age)] if age_single else [])
            if kind != "prose":  # タグの出力先と Anima (タグの部分): 年齢のタグを足し、ずれる方向をネガティブで抑える
                if len(persons) >= 2:
                    persons = [with_age_tags(p) for p in persons]
                else:
                    have_age = {_tag_key(t) for t in shared_general}
                    add_age = [t for t in age_tags_for(single_subject(tags), main_age) if _tag_key(t) not in have_age]
                    at = max((i for i, t in enumerate(shared_general)
                              if _tag_key(t) in PERSON_COUNT_TAGS or _tag_key(t) == "solo"), default=-1) + 1
                    shared_general = shared_general[:at] + add_age + shared_general[at:]
                negative = dedup(negative + age_negatives(people_ages))
            elif target == "mossimage":
                # MOSS-Image はネガティブ欄が使える (公式の既定は空)。年齢は本文の言い方に加えて、ずれる方向をここでも抑える
                age_moss_negative = age_negatives_prose(people_ages)
                if age_moss_negative:
                    analysis += ("\n年齢の見た目: MOSS-Image のネガティブ欄に、年齢がずれる方向を抑える語を入れました ("
                                 + ", ".join(age_moss_negative) + "。CFG 1 の高速 LoRA では効きません)")
        compose_extras = extras
        compose_tags = tags
        option_removed: list[str] = []  # 画風を変えるで外したタグ (分析メモは本文を整えるところで書く)
        removed_all: list[str] = []  # 三面図・人物の特徴・作風の特徴などで外したタグ (仕上げの書き直しが戻さないかも確かめる)
        # キャラ名を本文に入れない種類か (今はどの種類も入れる。RESTYLES の names)
        no_names = restyle != "none" and not RESTYLES[restyle]["names"]
        if turnaround or no_identity or no_style or restyle != "none":
            # 撮影プリセットと同じく、本文・人物ごと・VLM の追加タグから外す。追加プロンプトのタブは画像のまま残す
            char_keys = {_tag_name(sd_tag(n)) for n, _ in tags.characters}
            style_name_keys = {_tag_name(sd_escape(n.lower())) for n in vlm.style_names}  # ⑦ の作品名・作者名
            char_keys |= {_tag_name(t) for t in shared_chars} - style_name_keys

            def drops_base(t: str) -> bool:  # キャラ名は追加プロンプトの被写体の欄にも入っている
                return ((turnaround and turnaround_removes(t)) or (no_identity and (identity_tag(t) or _tag_name(t) in char_keys))
                        or (no_style and (style_removes(t) or _tag_name(t) in style_name_keys
                                          or t.lower().startswith("in the style of"))))

            def drops(t: str) -> bool:
                # 画風を変える: 元の画風のタグ、⑦ の作品名・作者名、(実写にするとき) キャラ名。VLM の自然文は捨てずに、
                # 後で文の中の語を言い換える (restyle_text)
                return (drops_base(t)
                        or (restyle != "none" and (restyle_removes(t, restyle, style_matches) or _tag_name(t) in style_name_keys
                                                   or t.lower().startswith("in the style of")
                                                   or (no_names and _tag_name(t) in char_keys))))

            removed_all = [t for t in shared_general + extra_tags + [x for p in persons for x in p.get("tags", [])] if drops(t)]
            removed_all = [t for t in shared_chars if drops(t)] + removed_all
            shared_chars = [t for t in shared_chars if not drops(t)]  # ⑦ の作品名・作者名はキャラ名ではないので人物の特徴では外さない
            # 自然文の組み立て (VLM 無し) はタグからも直接読むので、外したタグとキャラ名を除いたものを渡す
            compose_tags = TagResult(tags.rating, tags.rating_conf, [(n, q) for n, q in tags.general if not drops(n)],
                                     [] if no_identity or no_names else list(tags.characters), dict(tags.general_probs))
            shared_general = [t for t in shared_general if not drops(t)]
            extra_tags = [t for t in extra_tags if not drops(t)]
            persons = [p | {"tags": [x for x in p.get("tags", []) if not drops(x)],
                            "prompt": ", ".join(x for x in str(p.get("prompt") or "").split(", ") if x and not drops(x))}
                       for p in persons]
            compose_extras = {k: dict(v) for k, v in extras.items()}
            blank = ((["pose", "camera", "background", "objects", "lighting"] if turnaround else [])
                     + (["subject", "face"] if no_identity else [])
                     # 画風を変える: 元の画像の画風の文は使わない (⑦ が同じ系統なら、⑦ から読んだ文は残す)
                     + (["style"] if no_style or (restyle != "none" and not style_matches) else []))
            for key in compose_extras:
                if key == "motion":
                    continue
                compose_extras[key]["tags"] = [t for t in extras[key].get("tags", []) if not drops(t)]
                if key in blank:  # VLM の文はその要素を書くので使わず、残ったタグで書く
                    compose_extras[key]["text"] = ""
                compose_extras[key]["prompt"] = compose_extras[key]["text"] or ", ".join(compose_extras[key]["tags"])
            leaked = (mentions_any(natural_prompt, [_strip_weight(t) for t in removed_all if drops_base(t)])
                      if natural_prompt else [])
            if natural_prompt and no_style and medium_word(natural_prompt):
                leaked.append("絵か写真か (" + ", ".join(sorted(set(re.findall(r"[a-z0-9-]+", natural_prompt.lower())) & MEDIUM_WORD_SET)) + ")")
            if leaked:
                natural_prompt = ""
                warnings.append("VLM の自然文が外した要素 (" + ", ".join(dedup(leaked))[:200] + ") を書いていたので、タグから組み立てた文にしました")
            what = "・".join(x for x in ("三面図にする (ポーズ・カメラ・背景・小物を外しました)" if turnaround else "",
                                        "人物の特徴を入れない (髪・目・顔立ち・肌の色・キャラ名を外しました)" if no_identity else "",
                                        "作風の特徴を入れない (画風・技法・絵か写真か・年代・作品名 / 作者名を外し、品質タグ・ネガティブも"
                                        "絵か写真かに合わせた語を外しました)" if no_style else "") if x)
            removed_base = [t for t in removed_all if drops_base(t)]
            if what:
                analysis += "\n" + what + (f": {', '.join(dedup(removed_base))}" if removed_base else "") + "。追加プロンプトのタブは画像のままです"
            option_removed = dedup([t for t in removed_all if not drops_base(t)])
        if turnaround and kind != "prose":
            have = {_tag_name(t) for t in shared_general}
            shared_general = [t for t in TURNAROUND_TAGS if _tag_name(t) not in have] + shared_general
        shared_general, strength_notes = apply_strength(shared_general, controls.get("strength"), target, kind,
                                                         style_ref="style" in ref_tags)
        warnings += strength_notes
        look_tags: list[str] = []
        if look:
            # 撮影プリセット: 画像から読んだレンズ・照明の質・色調を外し、プリセットの文に置き換える (追加プロンプトの欄は画像のまま)
            look_removed = [t for t in shared_general + extra_tags if look_replaces(t)]
            shared_general = [t for t in shared_general if not look_replaces(t)]
            extra_tags = [t for t in extra_tags if not look_replaces(t)]
            compose_extras = {k: dict(v) for k, v in compose_extras.items()}
            for key in ("camera", "lighting"):
                # VLM 無し・人物ごとの文の組み立て用。VLM のカメラ・照明の文はレンズ・照明の質も書くので使わずタグで書く
                compose_extras[key]["tags"] = [t for t in compose_extras[key]["tags"] if not look_replaces(t)]
                compose_extras[key]["text"] = ""
                look_removed += [t for t in extras[key]["tags"] if look_replaces(t)]
            if kind not in ("prose", "anima"):  # タグ形式はタグで。自然文と Anima (文の部分) は後で文にして足す
                have = {norm_tag(t) for t in shared_general}
                look_tags = [t for t in look_phrases(look) if norm_tag(t) not in have]
                shared_general = shared_general + look_tags
            analysis += (f"\n撮影プリセット「{look_spec.get('name', '')}」: 本文のレンズ・照明をプリセットの文に置き換えました"
                         + (f" (外した画像由来のタグ: {', '.join(dedup(look_removed))})" if look_removed else "")
                         + "。追加プロンプトのカメラ・照明の欄は画像のままです")
        # 写真の撮影情報 (EXIF): ⑦ 作風・質感の画像から、⑦ が空なら ① から。撮影プリセットを選んでいれば
        # プリセットを優先する。作風の特徴を入れないときも入れない (撮影の語は作風の一部)
        exif_from = "⑦" if refs.get("style") else "①"
        exif = camera_exif(refs.get("style") or path)
        exif_look = exif_look_prompt(exif) if exif else ""
        exif_used = bool(exif_look) and not look and not no_style
        if exif:
            analysis += (f"\n撮影情報 (EXIF、{exif_from} の画像に記録された値): {exif_summary_ja(exif)}"
                         + ("。撮影プリセットを選んでいるので、本文にはプリセットのほうを入れました" if look and exif_look
                            else "。「作風の特徴を入れない」がオンなので、本文には入れていません" if no_style and exif_look else ""))
        if exif_used and kind not in ("prose", "anima"):
            have = {norm_tag(t) for t in shared_general}
            shared_general = shared_general + [t for t in look_phrases(exif_look) if norm_tag(t) not in have]
        if overridden and kind == "prose" and not ref_tags and vlm.natural_prompt and len(inputs.persons) < 2:
            warnings.append("ボーン・カメラを指定したので、VLM の自然文ではなく、指定を反映したタグから組み立てた文にしました")
        self.fraction(0.95 if use_vlm else 0.9, None)
        positive, compose_warnings = compose_positive(
            kind, target, tags=compose_tags, medium=medium, size=size, extras=compose_extras,
            shared_chars=shared_chars, shared_general=shared_general, extra_tags=extra_tags,
            natural_prompt=natural_prompt, persons=persons,
            quality_on=bool(cfg.get("quality_tags", True)), use_break=bool(cfg.get("use_break", True)),
            vlm_attempted=use_vlm, references=bool(ref_tags) or overridden, no_style=no_style,
            style_first="style" in ref_tags and not no_style, age=main_age,
        )
        warnings += compose_warnings
        # 仕上げの書き直しは、枠ごとの内容を並べた文のときだけ (compose_positive が VLM の自然文を使わない条件と同じ)。
        # VLM が画像を見て書いた文 (1 枚だけ) はそのまま使う: VLM をもう一度呼ぶと、その分だけ遅くなる
        assembled = not natural_prompt or bool(ref_tags) or overridden or (len(persons) >= 2 and bool(persons[0].get("box")))
        polish_started = time.time()
        if kind == "prose" and target != "midjourney" and vlm.ok and assembled and cfg.get("polish_prose", True):
            positive, polish_note, polish_warn = self._polish(
                positive, target, shared_general + extra_tags, forbidden=displaced_all + removed_all,
                style_first=("style" in ref_tags and not no_style) or restyle != "none",
                names=shared_chars)  # キャラ名と ⑦ の作品名・作者名: 書き直しで落ちていたら使わない
            analysis += polish_note
            warnings += polish_warn
            timing.append(("仕上げの書き直し", time.time() - polish_started))
        if age_on and kind in ("prose", "anima") and (main_age or len(persons) >= 2):
            # 年齢を文に書く (VLM の文が a young woman などと書いていたら、読んだ年齢に合わせて言い換える)。
            # Anima はタグの後ろの文の部分だけ
            age_head, age_sep, age_tail = positive.partition(". ") if kind == "anima" else ("", "", positive)
            if age_tail:
                positive = age_head + age_sep + apply_age_text(age_tail, persons, main_age, single_subject(tags))
        if target == "mossimage" and not no_identity and not no_names:
            # カタログに載っているキャラは「名前 from 作品名」(カタログの書き方)。実写にするときはキャラ名を書かない
            positive, catalog_note = moss_names_with_series(positive, list(tags.characters))
            analysis += catalog_note
        if turnaround and kind == "prose":  # 三面図の一文 (撮影プリセットなどより前)
            positive = positive.rstrip() + " " + TURNAROUND_SENTENCE
        if look and kind in ("prose", "anima"):
            # 自然文と Anima (文の部分) は文法の通った文で足す。VLM がすでに自然文に書いていれば足さない。
            # Midjourney のパラメータ・顔の文などはこの後ろに付く
            if not look_in_text(look, positive):
                sep = " " if kind == "prose" or ". " in positive else ". "
                positive = positive.rstrip() + sep + look_prose(look)
            look_tags = look_phrases(look)
        if exif_used and kind in ("prose", "anima") and not look_in_text(exif_look, positive):
            sep = " " if kind == "prose" or ". " in positive else ". "
            positive = positive.rstrip() + sep + exif_prose(exif)
        if kind == "prose" and (same_face or mixed_gender(persons)):
            faces = distinct_faces_text(persons, target, full=same_face, count_hint=person_count_hint(tags)[0])
            if faces:
                positive = positive.rstrip() + " " + faces
        # 外見の文を本文に入れる出力先: 自然文と Anima (タグ + 自然文。公式が「名前だけでなく外見を文で」)
        looks_in_body = kind in ("prose", "anima")
        if no_identity:
            detail_summary = ""  # 外見の文は髪・目・顔を書くので入れない
            persons = [p | {"detail_description": ""} for p in persons]
        if detail_summary and looks_in_body and len(persons) < 2:
            positive = positive.rstrip().rstrip(".") + ". Appearance: " + detail_summary.rstrip(".") + "."
        elif looks_in_body and len(persons) >= 2:
            # 人物ごとに読んだ外見の文も、位置で呼んで本文へ (タグだけだと顔の細部が落ちる)
            looks = [f"{person_lead_en(p)}: {p['detail_description'].rstrip('.')}." for p in persons if p.get("detail_description")]
            if looks:
                positive = positive.rstrip().rstrip(".") + ". " + " ".join(looks)
        if face_extra["text"]:
            analysis += "\n顔の詳細: " + (
                "本文に外見の文を入れました" if looks_in_body
                else f"本文には特徴のタグ ({'人物ごとの区画' if len(persons) >= 2 else '人数タグの後ろ'}) を入れ、"
                     "文は本文に入れていません (タグの出力先のため)。文は追加プロンプトの「顔の詳細」にあります"
            )
        have_multi = {n for n, _ in tags.general} & {"multiple_views", "collage", "reference_sheet", "variations", "turnaround",
                                                      "split_screen", "dual_persona", "multiple_persona", "clone", "afterimage"}
        if turnaround:
            have_multi = have_multi | {"turnaround"}  # 三面図は複数アングルの絵なので、分身を防ぐの文・--no は入れない
        if avoid_dup and kind == "prose" and target != "midjourney" and not have_multi:
            # ネガティブの無い自然文の出力先は本文に一文 (元画像が設定画・分身の絵なら書かない)
            positive = positive.rstrip() + " " + duplicate_guard_text(target, max(len(persons), person_count_hint(tags)[0]))
        elif avoid_dup and kind == "prose" and have_multi:
            warnings.append("分身を防ぐ: 元画像が複数アングル・分身の絵なので、自然文には書きませんでした")
        # ポジティブに入れたタグはネガティブから外す (VLM のネガティブ追加分がタグと矛盾することがある)。
        # 表記ゆれ (語順・複数形) も同じとみなし、VLM が足した分を外したときは画面に出す
        in_positive = char_tags + general_tags + shared_general + extra_tags + look_tags + [t for key in extras if key != "motion" for t in extras[key]["tags"]]
        if turnaround:  # 三面図の語 (分身を防ぐの reference sheet / turnaround など) はネガティブに残さない (自然文の出力先も)
            in_positive += TURNAROUND_TAGS
        present = {norm_tag(t) for t in in_positive}
        present_loose = {loose_tag_key(t) for t in in_positive}
        vlm_negative = {norm_tag(n) for n in vlm.negative}
        clash = [n for n in negative if norm_tag(n) in vlm_negative
                 and (norm_tag(n) in present or loose_tag_key(n) in present_loose
                      or mentions_any(n, [_strip_weight(t) for t in in_positive]))]  # wearing hat ⊃ hat (③ の帽子を打ち消す)
        if clash:
            warnings.append("VLM がネガティブにポジティブと同じ要素を挙げたので外しました: " + ", ".join(dedup(clash)))
        negative = [n for n in negative if norm_tag(n) not in present and n not in clash]
        if no_style:  # 絵か写真かに合わせたネガティブの定番 (アニメ絵の realistic / 3d、写真の anime / illustration など) を外す
            negative = [n for n in negative if not medium_word(n)]

        if restyle != "none":
            # 画風を変える: 先頭でその画風だと書き、⑦ から読んだ文をその直後に、描き方・質感の文を最後に置く
            shot = compose_extras["style"]["text"] if style_matches else ""
            positive, restyle_words = restyle_text(positive, restyle, max(len(persons), hint_count), shot)
            # 実写にするとき: キャラ名のタグは外してあるが、VLM が自然文に名前を書いていれば残る (文は言い換えない)
            names_left = mentions_any(positive, [unescape_parens(sd_tag(n)) for n, _ in tags.characters]) if no_names else []
            if names_left:
                warnings.append("画風を変える: VLM の文にキャラ名 (" + ", ".join(names_left) + ") が残っています "
                                "(名前は元の絵柄を呼び出すことがあります。気になるときは本文から消してください)")
            analysis += (f"\n画風を変える (MOSS-Image): {RESTYLE_MODES[restyle]}。先頭でその画風だと書き、描き方・質感の文を足しました"
                         + ("。⑦ 作風・質感から読んだ文は先頭の直後に置きました" if shot else "")
                         + (f"。外した・言い換えた語: {', '.join(restyle_words)}" if restyle_words else "")
                         + (f"。外したタグ: {', '.join(option_removed)}" if option_removed else "")
                         + ("。キャラ名は絵柄を呼び出すので本文に入れていません"
                            if no_names and tags.characters and not names_left else "")
                         + (f"。キャラ名のタグは外しましたが、VLM の文に名前 ({', '.join(names_left)}) が残っています"
                            if names_left else "")
                         + "。ネガティブには、ほかの画風を抑える語を入れました (CFG 1 の高速 LoRA では効きません)")
            if "style" in ref_tags and not style_matches:
                warnings.append(f"画風を変える: ⑦ 作風・質感の画像が「{RESTYLE_MODES[restyle]}」と同じ系統と判定されなかったので、"
                                "⑦ の画風は使いませんでした")
        positive, negative_text, persons, mosaic_prompt_applied = finalize_prompts(
            positive, negative, persons, kind, target, True,
            # モザイク指定はオンなら画像に関係なく常に入れる
            mosaic_level=cfg.get("mosaic_strength") if cfg.get("mosaic", False) else None,
        )
        if restyle != "none":
            negative_text = ", ".join(RESTYLES[restyle]["negative"])  # MOSS-Image のネガティブ欄 (ふだんは空)
        if age_moss_negative:  # 画風を変えるの語があれば、その後ろに足す
            negative_text = ", ".join(dedup([t.strip() for t in negative_text.split(",") if t.strip()] + age_moss_negative))
        if target == "novelai":
            for person, undesired in zip(persons, novelai_undesired(persons), strict=True):
                person["undesired"] = undesired
        elif (kind in ("tags", "anima") or TARGETS[target]["negative"] in ("optional", True)) and mixed_gender(persons) \
                and "androgynous" not in {_tag_key(t) for t in general_tags}:
            negative_text = ", ".join(dedup([t.strip() for t in negative_text.split(",") if t.strip()] + ["androgynous"]))
        formats = paste_formats(positive, persons, kind, target, extras, medium, distinct_faces=same_face)
        if target == "midjourney":
            has_text = bool({n for n, _ in tags.general} & {"text", "english_text", "japanese_text", "speech_bubble", "signboard"})
            # 画像固有の除外 (VLM が挙げたもの) を先に、定番の除外 (画風違いなど) を後に
            ordered = list(vlm.negative) + [n for n in negative if n not in vlm.negative]
            if same_face:  # 同じ顔を避ける: --no に (V7 / V8 / Niji 7 で効くかは公式資料で未確認)
                ordered = MJ_SAME_FACE_NO + ordered
            if avoid_dup and not have_multi:  # 分身を防ぐ: --no に (効き目は公式資料で未確認)
                ordered = dedup(MJ_DUPLICATE_NO + ordered)
            if mixed_gender(persons):
                ordered = ["androgynous"] + ordered
            params, conflicts, overflow = midjourney_params("unspecified" if no_style else medium, size, ordered, has_text, positive)
            positive = positive.rstrip() + " " + params
            if conflicts:
                warnings.append("Midjourney: 本文と矛盾するため --no に入れなかった語: " + ", ".join(conflicts))
            if overflow:
                warnings.append(f"Midjourney: --no は {MJ_NO_MAX} 語までにしました。入れなかった語: " + ", ".join(overflow))
        quality_issues = quality_check(
            positive, negative_text, persons, kind, target, inputs.displaced, color_only=bool(mono_removed),
        )
        if mono_removed:
            analysis += "\n白黒の情報を消す: " + ", ".join(t.replace("_", " ") for t in mono_removed) + " を外してネガティブへ (カラーで出す)"
        if quality_issues:
            analysis += "\n品質チェック: " + " / ".join(quality_issues)
            warnings += [f"品質チェック: {q}" for q in quality_issues]
        elif len(persons) >= 2:
            analysis += "\n品質チェック: 人数タグ・各人の特徴・区画の長さ・ネガティブとの矛盾に問題はありません"

        total = time.time() - started
        if total >= 1.0:  # どこが遅いかを見分けられるように (モデルの読み込みは含まない)
            analysis += ("\nかかった時間: " + "、".join(f"{label} {secs:.1f} 秒" for label, secs in timing if secs >= 0.05)
                         + f" (合計 {total:.1f} 秒)")
        tag_records = [
            {"tag": sd_tag(n), "raw": n, "conf": round(p, 4), "kind": "character"} for n, p in tags.characters
        ] + [
            {"tag": sd_tag(n), "raw": n, "conf": round(p, 4), "kind": "general"}
            | ({"source": tag_source[n]} if tag_source.get(n, "person") != "person" else {})
            for n, p in detected.general
        ]

        return PromptResult(
            image=str(path),
            engine=cfg.get("engine", DEFAULT_CONFIG["engine"]),
            tagger_model=tagger.name,
            vlm_model=vlm.model,
            target=target,
            detected_medium=medium,
            rating=tags.rating,
            analysis=analysis,
            positive_prompt=positive,
            negative_prompt=negative_text,
            notes=notes,
            tags=tag_records,
            persons=persons,
            nsfw=nsfw,
            mosaic_prompt=mosaic_prompt_applied,
            extras=extras,
            era=era_info,
            references={slot: str(p) for slot, p in refs.items()},
            controls=dict(controls),
            main_person=inputs.main_person,
            formats=formats,
            background_people=background_people,
            quality_issues=quality_issues,
            elapsed_sec=round(time.time() - started, 2),
            warnings=warnings,
        )

    def _tag_inputs(
        self, tagger: WD14Tagger, path: Path, refs: dict[str, Path], general_th: float, character_th: float,
        warnings: list[str],
    ) -> TagInputs:
        """① メインと参照画像をタガーにかけ、複数枚モードならタグを持ち主の枠ごとに統合する。
        「メインの 1 人だけ」なら ① (と ② ポーズ / ③ 服装) の画像から一番大きい人物を切り出してタグ付けし、
        背景・照明だけは切り出す前の画像全体から取る。"""
        # 同じ画像を複数の枠に使ったとき (② と ③ に同じ画像など) は 1 回だけ開き、タガーも 1 回だけかける
        opened: dict[Path, Image.Image] = {}

        def open_once(p: Path) -> Image.Image:
            key = Path(p).resolve()
            if key not in opened:
                opened[key] = self._open(p)
            return opened[key]

        image = open_once(path)
        ref_images = {slot: open_once(p) for slot, p in refs.items()}
        # 構図・画角はポーズ画像、無ければ背景画像から取るので、縦横比もそれに合わせる
        size = next((ref_images[s].size for s in ("pose", "background") if s in ref_images), image.size)
        main_only = bool(self.cfg.get("main_person_only", False))
        main_info: dict[str, Any] | None = None
        person_image = image
        pose_whole_image: Image.Image | None = None
        if main_only:
            person_image, main_info = self._main_person(image, REF_SLOTS["person"] if refs else "画像", warnings)
            cropped: dict[int, Image.Image] = {}
            for slot in ("pose", "outfit", "body"):
                if slot in ref_images:
                    before = ref_images[slot]
                    if id(before) not in cropped:
                        cropped[id(before)], _ = self._main_person(before, REF_SLOTS[slot], warnings)
                    ref_images[slot] = cropped[id(before)]
                    if slot == "pose" and ref_images[slot] is not before:
                        pose_whole_image = before  # 構図 (画角・アングル) は切り出す前の ② 全体から読む
        self._check()
        self.progress(f"タガーで推論中 ({tagger.name}, {tagger.provider})...")
        tagged: dict[int, TagResult] = {}

        def predict_once(img: Image.Image) -> TagResult:
            if id(img) not in tagged:
                tagged[id(img)] = tagger.predict(img, general_th, character_th)
            return tagged[id(img)]

        tags = predict_once(person_image)
        self.fraction(0.15, None)
        whole: TagResult | None = None
        if person_image is not image:  # 切り出したときは、背景・照明を取るために画像全体もタガーにかける
            self._check()
            self.progress("タガーで推論中 (背景用に画像全体)...")
            whole = predict_once(image)
        ref_tags: dict[str, TagResult] = {}
        for slot, ref_image in ref_images.items():
            self._check()
            self.progress(f"タガーで推論中 ({REF_SLOTS[slot]} の画像)...")
            ref_tags[slot] = predict_once(ref_image)
            self.fraction(0.15 + 0.15 * len(ref_tags) / max(1, len(ref_images)), None)
        if whole is not None:
            tags = camera_from_whole(tags, whole)  # 画角・アングルは切り出す前の ① 全体から
        if pose_whole_image is not None:
            self._check()
            self.progress("タガーで推論中 (② ポーズの画像全体の構図)...")
            ref_tags["pose"] = camera_from_whole(ref_tags["pose"], predict_once(pose_whole_image))
        # 画角は一番確信度の高いものを 1 つ必ず入れる (構図を取る画像: ② があれば ②、無ければ ①)
        if "pose" in ref_tags:
            ref_tags["pose"] = ensure_framing(ref_tags["pose"])
        else:
            tags = ensure_framing(tags)
        if main_only:
            tags = force_single_person(tags)
        elif self.cfg.get("split_persons", True) and not refs:
            split = self._split_persons(tagger, image, tags, general_th, character_th, warnings)
            if split is not None:
                merged, persons = split
                return TagInputs(
                    image=image, vlm_image=image, ref_images={}, size=size, tags=merged, person_raw=merged,
                    ref_tags={}, tag_source={}, main_person=None, persons=persons,
                )
        merged, tag_source = tags, {}
        if ref_tags or whole is not None:
            if ref_tags:
                warnings += reference_warnings(tags, ref_tags)
            parts = {"person": tags, **ref_tags}
            background_from_whole = whole is not None and "background" not in parts
            if background_from_whole:
                parts["background"] = whole
            merged, tag_source = merge_reference_tags(parts)
            if background_from_whole:  # 画像全体は ① と同じ画像なので、出どころは ① のまま
                tag_source = {n: ("person" if src == "background" else src) for n, src in tag_source.items()}
            if main_only:
                merged = force_single_person(merged)
            if "pose" in ref_tags and {n for n, s_ in tag_source.items() if s_ == "pose"} & BACK_VIEW_TAGS:
                # ② が後ろ姿なのに正面でしか見えない露出タグ (③ / ① の絵が正面) があると、正面向きで描かれる
                front = [n for n, _ in merged.general if n in FRONT_ONLY_EXPOSURE]
                if front:
                    merged = TagResult(merged.rating, merged.rating_conf, [(n, p) for n, p in merged.general if n not in front],
                                       list(merged.characters), dict(merged.general_probs))
                    tag_source = {n: s_ for n, s_ in tag_source.items() if n not in front}
                    warnings.append("② が後ろ姿なので、正面からしか見えない要素を外しました (残すと正面向きで描かれやすい): "
                                    + ", ".join(sd_tag(n) for n in front))
        chest = {n for n, _ in merged.general} & {"flat_chest", "small_breasts"}
        showing = {n for n, _ in merged.general} & {"cleavage", "underboob", "sideboob", "cleavage_cutout", "breasts_out"}
        if "body" in ref_tags and chest and showing:
            # 削らずに知らせるだけ (どちらを優先するかは使う人が決める)
            warnings.append(f"⑥ の体格 ({', '.join(sd_tag(n) for n in sorted(chest))}) と ③ の服の見え方 "
                            f"({', '.join(sd_tag(n) for n in sorted(showing))}) が合わないことがあります。必要ならどちらかを消してください。")
        displaced: dict[str, list[str]] = {}
        for slot, ref in ref_tags.items():
            in_ref = {n for n, _ in ref.general}
            ref_words = [set(n.split("_")) for n in in_ref]
            # 参照画像のタグの語だけでできているタグ (③ が black_dress のときの ① の dress) は ① 固有とは言えないので除く
            gone = [
                sd_tag(n) for n, _ in tags.general
                if slot_owner(reference_category(n, "style" in ref_tags), ref_tags) == slot and n not in in_ref
                and not any(set(n.split("_")) <= w for w in ref_words)
            ]
            if gone:
                displaced[slot] = gone
        return TagInputs(
            image=image, vlm_image=person_image, ref_images=ref_images, size=size, tags=merged, person_raw=tags,
            ref_tags=ref_tags, tag_source=tag_source, main_person=main_info, displaced=displaced,
        )

    def _split_persons(
        self, tagger: WD14Tagger, image: Image.Image, whole: TagResult, general_th: float, character_th: float,
        warnings: list[str],
    ) -> tuple[TagResult, list[dict[str, Any]]] | None:
        """2 人以上写っていれば 1 人ずつ切り出してタグ付けし、(共有 + 全員のタグ, 人物の一覧) を返す。
        人数タグは検出した人数と各人の性別で付け直す。1 人以下や検出に失敗したときは None (いつもどおり画像全体)。"""
        self._check()
        self.progress("人物を検出中...")
        try:
            detections, _heads = self.detect_persons(image, warnings)
        except OperationCancelled:
            raise
        except Exception as e:  # noqa: BLE001
            warnings.append(f"人物ごとの分割をスキップしました: {describe_error(e)}")
            return None
        if len(detections) < 2:
            return None
        all_boxes = [box for _, box in detections]
        largest = max(box_area(b) for b in all_boxes)
        # 奥に小さく写った人は Danbooru の数え方に合わせて人数に入れない (人物ごとタブに表示だけする)。
        # 頭の大きさで決める案は、奥の人まで数えて人数が多すぎる画像が倍になったので使わない (DEVLOG 77)
        main_boxes = [b for b in all_boxes if box_area(b) >= largest * PERSON_BACKGROUND_RATIO]
        background_boxes = [b for b in all_boxes if box_area(b) < largest * PERSON_BACKGROUND_RATIO]
        if len(main_boxes) < 2:
            return None  # 主な人物が 1 人なら、いつもどおり画像全体で
        limit = PERSON_SPLIT_MAX_NOVELAI if self.cfg.get("target") == "novelai" else PERSON_SPLIT_MAX
        if len(main_boxes) > limit:
            warnings.append(f"人物を {len(main_boxes)} 人検出しました。人物ごとに分けるのは大きい順に {limit} 人までです。")
        boxes = sorted(main_boxes, key=box_area, reverse=True)[:limit]
        boxes.sort(key=lambda b: (b[0] + b[2]) / 2)  # 左から順
        skeletons = self._person_skeletons(image, boxes, warnings)
        crops: list[tuple[Box, TagResult]] = []
        for i, box in enumerate(boxes, 1):
            self._check()
            self.progress(f"人物 {i} / {len(boxes)} をタグ付け中...")
            self.fraction(0.15 + 0.2 * (i - 1) / len(boxes), None)
            crop = crop_person(image, box, [b for b in all_boxes if b != box], skeletons=skeletons)
            crops.append((box, tagger.predict(crop, general_th, character_th, flip=True)))
        color_leaks = minor_color_leaks([t for _, t in crops])
        whole_genders = genders_from_whole([t for _, t in crops], whole)
        own_genders = [person_gender(t) for _, t in crops]
        if whole_genders and whole_genders != own_genders:
            changed = [f"人物 {i + 1} を {GENDER_JA[g]}" for i, (g, o) in enumerate(zip(whole_genders, own_genders, strict=True)) if g != o]
            warnings.append(f"性別を画像全体の人数タグの男女の数に合わせました ({', '.join(changed)})。")
        # 同じタグが複数人に出ても、基本は全員に残す (同じ服・同じ髪色は普通にある)。
        # 外すのは「そのタグの信頼度がもっと高い人が、この人の枠に実際に重なっている」(写り込みの可能性が高い) ときだけで、
        # 外したタグは人物ごとタブに理由付きで出す (黙って捨てない)
        best: dict[str, tuple[float, int]] = {}
        for i, (_, t) in enumerate(crops):
            for name, prob in t.general:
                if prob > best.get(name, (0.0, -1))[0]:
                    best[name] = (prob, i)
        persons: list[dict[str, Any]] = []
        person_general: list[tuple[str, float]] = []
        characters: list[tuple[str, float]] = []
        for i, (box, t) in enumerate(crops):
            own: list[tuple[str, float]] = []
            dropped: list[str] = []
            dropped_colors: list[str] = []
            for name, prob in t.general:
                if not is_person_tag(name) or interaction_phrase(name):  # 人どうしの動作は共有へ (下の interactions)
                    continue
                if name in color_leaks[i]:
                    dropped_colors.append(sd_tag(name))
                    continue
                top_prob, owner = best[name]
                leaked = (
                    owner != i and prob < top_prob * PERSON_LEAK_RATIO
                    and box_area((max(box[0], boxes[owner][0]), max(box[1], boxes[owner][1]),
                                  min(box[2], boxes[owner][2]), min(box[3], boxes[owner][3]))) > 0
                )
                (dropped if leaked else own).append((name, prob) if not leaked else sd_tag(name))
            subject = whole_genders[i] if whole_genders else own_genders[i]
            position, where_ja = person_position(box, image.size, largest, i, len(crops))
            persons.append({
                "subject": subject, "position": position, "position_ja": where_ja,
                "characters": [sd_tag(n) for n, _ in t.characters], "tags": [sd_tag(n) for n, _ in own],
                "dropped": dropped, "dropped_colors": dropped_colors, "description": "", "box": list(box),
                "area": person_area(box, image.size),
                "nai_position": nai_position(box, image.size),
            })
            person_general += own
            characters += t.characters
        # 全員に共通する特徴 (全員が school uniform など) は共有部分へ移す。消すのではなく置き場所を変えるだけ
        common = set.intersection(*({norm_tag(t) for t in p["tags"]} for p in persons)) if persons else set()
        for person in persons:
            person["tags"] = sorted(
                (t for t in person["tags"] if norm_tag(t) not in common), key=person_tag_priority,
            )  # sorted は安定なので、同じ優先度の中は信頼度の順のまま
            names = person["characters"] + person["tags"]
            person["prompt"] = ", ".join(dedup([SUBJECT_TOKEN[person["subject"]]] + names))
            person["tags"] = names
        common_tags = sorted(
            {sd_tag(n) for n, _ in person_general if norm_tag(n) in common}, key=person_tag_priority,
        )
        for person in persons:
            person["common"] = common_tags  # タグ形式では共有部分に入る。自然文では「全員に共通」として書く
        common_general = [(n, p) for n, p in person_general if norm_tag(n) in common]
        # 人どうしの動作は画像全体のタグから取り直す (切り出しでは相手が塗られて落ちる。切り出しにだけ出たものも足す)
        interactions = {n: p for n, p in whole.general if interaction_phrase(n)}
        for _, t in crops:
            for name, prob in t.general:
                if interaction_phrase(name) and prob > interactions.get(name, 0.0):
                    interactions[name] = prob
        interaction_tags = sorted(interactions.items(), key=lambda x: -x[1])
        # 共有: 画像全体のタグから背景・照明・画風・構図、全員に共通の特徴、検出した人数から付け直した人数タグ
        shared = [
            (n, p) for n, p in whole.general
            if not is_person_tag(n) and n not in PERSON_COUNT_TAGS and n not in ("solo", "multiple_views")
        ]
        # 画風・質感 (realistic / photo_background など) は画像全体の性質なので、切り出しで強く出たものも共有側に足す
        shared += [(n, p) for _, t in crops for n, p in t.general if classify_extra_tag(n) == "style"]
        counts = [(n, 1.0) for n in count_tags([p["subject"] for p in persons])]
        seen: set[str] = set()
        general = []
        # 人物ごとのタグも入れておく (追加プロンプトと分析用。本文の共有部分からは人物側に割り当てた分が外れる)
        for name, prob in counts + common_general + shared + interaction_tags + person_general:
            if name not in seen:
                seen.add(name)
                general.append((name, prob))
        chars = list({n: (n, p) for n, p in whole.characters + characters}.values())
        # 画風の判定 (detect_medium) は信頼度を見るので、画風のタグは画像全体と切り出しのうち高い方を使う
        probs = dict(whole.general_probs)
        for _, t in crops:
            for name, prob in t.general_probs.items():
                if classify_extra_tag(name) == "style" and prob > probs.get(name, 0.0):
                    probs[name] = prob
        merged = TagResult(whole.rating, whole.rating_conf, general, chars, probs)
        if persons:
            persons[0]["background_people"] = [list(b) for b in sorted(background_boxes, key=lambda b: b[0])]
            names = [n for n, _ in interaction_tags]
            persons[0]["interactions"] = resolve_interactions(names, boxes, self._interaction_skeletons(image, boxes, skeletons, names, warnings))
        # 画像全体のタグの人数・性別と、1 人ずつの判定が食い違えば知らせる (どちらが正しいかは画像を見て決めてもらう)
        whole_counts = [n for n, _ in whole.general if n in PERSON_COUNT_TAGS and not n.startswith("multiple")]
        split_counts = [n for n in count_tags([p["subject"] for p in persons], with_multiple=False)]
        if whole_counts and sorted(whole_counts) != sorted(split_counts):
            warnings.append(
                f"人数・性別の判定が食い違っています: 画像全体のタグでは {', '.join(sd_tag(n) for n in whole_counts)}、"
                f"1 人ずつの判定では {', '.join(sd_tag(n) for n in split_counts)} です (本文は 1 人ずつの判定を使っています)。"
            )
        return merged, persons

    def _person_skeletons(self, image: Image.Image, boxes: list[Box], warnings: list[str]) -> dict[Box, list[list[float]]] | None:
        """枠が重なっている人の骨格 (切り出しの塗り分けに使う)。骨格モデルが無い・重なりが無いときは None (取得はしない)。"""
        overlapped = [b for b in boxes if any(
            o != b and box_area((max(b[0], o[0]), max(b[1], o[1]), min(b[2], o[2]), min(b[3], o[3]))) > 0 for o in boxes)]
        if not overlapped or not pose_model_ready():
            return None
        try:
            if self._pose is None:
                self._pose = PoseEstimator()
            return {b: self._pose.estimate(image, b) for b in overlapped}
        except OperationCancelled:
            raise
        except Exception as e:  # noqa: BLE001
            warnings.append(f"骨格での塗り分けをスキップしました: {describe_error(e)}")
            return None

    def _interaction_skeletons(
        self, image: Image.Image, boxes: list[Box], skeletons: dict[Box, list[list[float]]] | None, names: list[str],
        warnings: list[str],
    ) -> dict[Box, list[list[float]]] | None:
        """骨格で向きを決められる動作があれば、全員の骨格 (塗り分けで取った分は使い回す)。骨格モデルが無ければ None。"""
        if not any(interaction_reach(n) for n in names) or not pose_model_ready():
            return None
        try:
            if self._pose is None:
                self._pose = PoseEstimator()
            have = dict(skeletons or {})
            for box in boxes:
                if box not in have:
                    self._check()
                    have[box] = self._pose.estimate(image, box)
            return have
        except OperationCancelled:
            raise
        except Exception as e:  # noqa: BLE001
            warnings.append(f"骨格で人どうしの動作の向きを決めるのをスキップしました: {describe_error(e)}")
            return None

    def _main_person(self, image: Image.Image, label: str, warnings: list[str]) -> tuple[Image.Image, dict[str, Any]]:
        """一番大きく写っている人物を検出する。2 人以上いればその人物を切り出した画像を返し、1 人以下ならそのまま返す。
        検出に失敗しても (モデルを取得できないなど) 画像全体で続ける。"""
        self._check()
        self.progress(f"人物を検出中 ({label})...")
        try:
            detections, _heads = self.detect_persons(image, warnings)
        except OperationCancelled:
            raise
        except Exception as e:  # noqa: BLE001
            warnings.append(f"人物検出をスキップしました ({label}): {describe_error(e)}")
            return image, {"count": 0, "box": None, "error": True}
        box = main_person_box(detections)
        info: dict[str, Any] = {"count": len(detections), "box": list(box) if box else None}
        if len(detections) >= 2 and box is not None:
            return crop_box(image, box), info
        return image, info

    def _era_step(self, image: Image.Image, medium: str, warnings: list[str]) -> dict[str, Any] | None:
        """年代スタイル分類 (アニメ・イラスト系のみ)。失敗しても続行する。"""
        if not self.cfg.get("era_tags", True) or medium == "photo":
            return None
        try:
            self._check()
            self.progress("年代スタイルを判定中...")
            return self.era_classifier().predict(image)
        except OperationCancelled:
            raise
        except Exception as e:  # noqa: BLE001
            warnings.append(f"年代スタイル分類をスキップしました: {describe_error(e)}")
            return None

    def _person_details(
        self, inputs: "TagInputs", persons: list[dict[str, Any]], use_vlm: bool, with_build: bool, medium: str = "",
    ) -> tuple[list[dict[str, Any]], list[str], str, str, list[str], dict[str, Any]]:
        """人物の特徴を詳しく読む (VLM)。2 人以上に分けたときは 1 人ずつ切り出して、それぞれの区画に足す。
        顔は元画像から拡大して切り出し、人物全体の画像と一緒に渡す (細部を読むため。2026-10-01 利用者の指摘)。
        戻り値: (人物, 1 人のときのタグ, 1 人のときの外見の文, 分析メモ, 注意, 追加プロンプトの「顔の詳細」)。"""
        if not use_vlm:
            return persons, [], "", "", ["人物の特徴を詳しく: VLM (「タガー + VLM」) が必要なため、できませんでした"], empty_face_extra()
        vlm = self.vlm()
        keep = "2m" if vlm.keep_alive == 0 else None
        if persons and persons[0].get("box"):
            targets = [(i, crop_box(inputs.image, tuple(p["box"])), p["tags"], tuple(p["box"])) for i, p in enumerate(persons)]
        else:
            main_box = (getattr(inputs, "main_person", None) or {}).get("box")
            targets = [(-1, inputs.vlm_image, [sd_tag(n) for n, _ in inputs.tags.general], tuple(main_box) if main_box else None)]
        notes, warnings = [], []
        main_tags: list[str] = []
        main_summary = ""
        face_lines: list[str] = []
        no_face: list[str] = []
        out = [dict(p) for p in persons]
        for index, image, tags, box in targets:
            self._check()
            who = f"人物 {index + 1}" if index >= 0 else "人物"
            try:
                face, why = self.face_image(inputs.image, box)
            except OperationCancelled:
                raise
            except Exception as e:  # noqa: BLE001 - 顔を切り出せなくても人物全体の画像で読む
                face, why = None, f"骨格の推定に失敗したため ({describe_error(e)})"
            if why:
                no_face.append(f"{who}: {why}")
            self.progress(f"VLM で{who}の特徴を詳しく読んでいます ({vlm.model})...")
            try:
                detail = vlm.analyze_detail(image, tags, with_build, keep_alive=keep, medium_hint=medium, face=face)
            except OperationCancelled:
                raise
            except Exception as e:  # noqa: BLE001 - 読めなくてもほかの結果は返す
                if self._cancel.is_set():
                    raise OperationCancelled() from e
                warnings.append(f"人物の特徴を詳しく ({who}): 読めませんでした ({describe_error(e)})")
                continue
            detail, vague = drop_vague_detail(detail)
            found = person_detail_tags(detail.get("tags"), with_build)
            summary = llm_text(detail.get("summary"))
            lines = person_detail_lines(detail)
            if vague:
                lines.append("  (情報の少ない言い方を外しました: " + ", ".join(vague) + ")")
            if lines:
                notes.append(f"{who}の特徴 (VLM{'、顔は拡大して読んだ' if face is not None else ''}):\n" + "\n".join(lines))
            if index < 0:
                main_tags, main_summary = found, summary
                if summary:
                    face_lines.append(summary)
                continue
            person = out[index]
            if summary:
                person["detail_description"] = summary
                face_lines.append(f"{person_lead_en(person)}: {summary}")
            have = {loose_tag_key(t) for t in person["tags"]} | {loose_tag_key(t) for t in person.get("common") or []}
            added = [t for t in found if loose_tag_key(t) not in have]
            if added:
                person["tags"] = person["tags"] + added
                person["prompt"] = ", ".join(dedup(person["prompt"].split(", ") + added))
                person["detail_added"] = added
        if no_face:
            warnings.append("人物の特徴を詳しく: 顔を拡大して切り出せなかったので、人物全体の画像だけで読みました ("
                            + " / ".join(no_face) + ")")
        if not with_build:
            notes.append("(体格は ⑥ 体格の画像から取るので、① からは読んでいません)")
        text = "\n".join(face_lines)
        face_extra = {"tags": list(main_tags), "text": text, "prompt": text or ", ".join(main_tags)}
        return out, main_tags, main_summary, "\n".join(notes), warnings, face_extra

    def _polish(self, draft: str, target: str, shared: list[str], forbidden: list[str],
                style_first: bool, names: list[str] | None = None) -> tuple[str, str, list[str]]:
        """仕上げの書き直し (POLISH_SYSTEM): 枠ごとの内容を並べた文を、出力先の書き方で 1 つの文章にする。
        戻り値: (本文, 分析メモに足す行, 注意)。失敗・内容が落ちたときは元の文のまま (中止だけは投げる)。"""
        if not draft.strip():
            return draft, "", []
        try:
            vlm = self.vlm()
            self._check()
            self.progress(f"VLM で文を仕上げています ({vlm.model})...")
            text = vlm.rewrite(draft, polish_guidance(target), style_first)
        except OperationCancelled:
            raise
        except Exception as e:  # noqa: BLE001 - 仕上げは無くても結果は出せる
            if self._cancel.is_set():
                raise OperationCancelled() from e
            return draft, "", [f"仕上げの書き直し: できませんでした ({describe_error(e)})。組み立てた文のままです"]
        problem = polish_problem(draft, text, [_strip_weight(t) for t in shared], [_strip_weight(t) for t in forbidden],
                                 [unescape_parens(_strip_weight(t)) for t in names or []])
        if problem:
            return draft, "", [f"仕上げの書き直し: 書き直した文は使いませんでした ({problem})。元の文のままです"]
        return text, "\n仕上げの書き直し: 枠ごとに集めた内容を、1 つの文章に書き直しました (設定の「文を仕上げる」でオフにできます)", []

    def _run_vlm(
        self, image: Image.Image, tags: TagResult, medium: str, negative: list[str], target: str,
        size: tuple[int, int], vlm_refs: list[tuple[str, Image.Image, TagResult]], hint_count: int, hint_tags: list[str],
        extra_guidance: str = "", look: str = "", describe_image: Image.Image | None = None,
        persons_guided: bool = False, polish_next: bool = False,
    ) -> VlmOutcome:
        """VLM で分析する。失敗してもタガーの結果で続けられるよう、理由を warnings に入れて空の結果を返す
        (中止だけは OperationCancelled を投げる)。"""
        result = VlmOutcome()
        try:
            vlm = self.vlm()
            self._check()
            self.progress(f"VLM で分析中 ({vlm.model})... 数十秒かかることがあります (「■ 中止」で止められます)")
            guidance = TARGET_VLM_GUIDANCE.get(target, "")
            guidance += f"\nSource image aspect ratio: {aspect_label(size)} ({size[0]}x{size[1]})."
            if look:  # 撮影プリセット・組み合わせ: 文にした撮影の語を自然文に織り込ませる (人数の指示とは独立に足す)
                guidance += "\n" + LOOK_VLM.format(look=look)
            if extra_guidance:
                guidance += "\n" + extra_guidance
            if not vlm_refs and not persons_guided and hint_count >= 2:   # 人物ごとの指示が無いときだけ
                guidance += (
                    f"\nThe tagger suggests {hint_count} people ({', '.join(hint_tags)}). Fill persons with one "
                    "entry per person and keep each person's attributes separate."
                )
            described = self._describe(image if describe_image is None else describe_image, vlm, result.warnings)
            if described:
                guidance += "\n" + DESCRIBER_VLM.format(text=described)
            separate = tuple(slot for slot, _, _ in vlm_refs if slot in VLM_SEPARATE_SLOTS)
            parts: dict[str, dict[str, Any]] = {}
            for slot, ref_image, ref_tags in vlm_refs:
                if slot not in separate:
                    continue
                self._check()
                self.progress(f"VLM で {REF_SLOTS[slot]} の画像を分析中 ({vlm.model})...")
                try:
                    # 続けてメインの呼び出しがあるので、省 VRAM 設定でもそれまではモデルを降ろさない (読み直しの時間を省く)
                    parts[slot] = vlm.analyze_part(ref_image, slot, ref_tags, with_objects=slot == "outfit"
                                                   and all(s_ != "props" for s_, _, _ in vlm_refs),
                                                   keep_alive="2m" if vlm.keep_alive == 0 else None)
                except OperationCancelled:
                    raise
                except Exception as e:  # noqa: BLE001 - 1 枚分が失敗してもタガーの結果で続ける
                    if self._cancel.is_set():
                        raise OperationCancelled() from e
                    result.warnings.append(f"VLM の {REF_SLOTS[slot]} の分析をスキップしました: {describe_error(e)}")
            self._check()
            self.progress(f"VLM で分析中 ({vlm.model})...")
            keep_before = vlm.keep_alive
            if polish_next and keep_before == 0:
                vlm.keep_alive = "2m"  # 続けて仕上げの書き直しがあるので、省 VRAM 設定でもそれまではモデルを降ろさない
            try:
                out = vlm.analyze(image, tags, medium, negative, guidance, vlm_refs, separate)
            finally:
                vlm.keep_alive = keep_before
            if "style" in separate:
                # ⑦ があるときの画風は ⑦ の呼び出しだけから取る。思考モデルはスキーマを守らずに ① の画風を書いてくる
                # ことがあり、⑦ の呼び出しが失敗するとそれが残った (実機: Krea 2 の Style: が ① の 2D anime aesthetic)
                out.pop("style", None)
                out.pop("style_tags", None)
            for slot, part in parts.items():  # 1 枚ずつ読んだ結果で上書きする (メインには見せていない)
                out.update({k: v for k, v in part.items()
                            if k in ("pose_tags", "pose", "camera", "outfit_tags", "outfit", "objects", "body_tags",
                                     "style_tags", "style", "style_names", "expression_tags")})
        except OperationCancelled:
            raise
        except Exception as e:  # noqa: BLE001 - VLM が落ちてもタガーの結果は返す
            if self._cancel.is_set():
                raise OperationCancelled() from e
            result.warnings.append(f"VLM をスキップしました: {describe_error(e)}")
            return result
        slots = {slot for slot, _, _ in vlm_refs}
        result.out = out
        result.model = vlm.model
        result.analysis = llm_text(out.get("analysis"))
        if described:   # 説明を書かせたモデルの文も分析の欄に残す (いつもの VLM が何を参考にしたかわかるように)
            result.analysis = (result.analysis + "\n\n" if result.analysis else "") + f"[{self._describer_label()} の説明]\n{described}"
        result.notes = llm_text(out.get("notes"))
        result.natural_prompt = llm_text(out.get("natural_prompt"))
        result.age = clean_age(out.get("age"))
        result.extra_tags = clean_llm_tags(out.get("extra_tags"))
        # 参照画像から VLM が拾ったポーズ・服装のタグも本文に入れる。末尾ではなく服装・ポーズのタグの並びに置くので別に持つ
        if "pose" in slots:
            result.ref_pose_tags = clean_llm_tags(out.get("pose_tags"), limit=12)
        if "outfit" in slots:
            result.ref_outfit_tags = clean_llm_tags(out.get("outfit_tags"), limit=16)
        if "body" in slots:
            result.ref_body_tags = clean_llm_tags(out.get("body_tags"), limit=10)
        if "expression" in slots:
            result.ref_expression_tags = clean_llm_tags(out.get("expression_tags"), limit=10)
        if "style" in slots:
            out["style_tags"] = clean_style_tags(out.get("style_tags"))  # 画風以外 (髪・服・ポーズ・場所) は外す
            style_seen = next((t for slot_, _, t in vlm_refs if slot_ == "style"), None)
            result.style_names = clean_style_names(
                out.get("style_names"), [sd_tag(n) for n, _ in style_seen.general] if style_seen else None)
        result.extra_tags = dedup(result.extra_tags)
        result.negative = clean_llm_tags(out.get("negative_additions"))
        result.persons = parse_persons(out.get("persons")) if not slots else []
        if not slots:
            # 1 枚モード: 1 人のときのポーズ・服装のタグ (2 人以上なら persons の中)
            result.pose_tags = clean_llm_tags(out.get("pose_tags"), limit=12)
            result.outfit_tags = clean_llm_tags(out.get("outfit_tags"), limit=16)
            raw_persons = out.get("persons") if isinstance(out.get("persons"), list) else []
            for entry in raw_persons[:VLM_ARRAY_LIMITS["persons"]]:
                if not isinstance(entry, dict):
                    continue
                num = entry.get("number")
                if isinstance(num, int) and not isinstance(num, bool):
                    result.numbered.append({
                        "number": num,
                        "pose_tags": clean_llm_tags(entry.get("pose_tags"), limit=12),
                        "outfit_tags": clean_llm_tags(entry.get("outfit_tags"), limit=16),
                        "age": clean_age(entry.get("age")),
                    })
            raw_inter = out.get("interactions") if isinstance(out.get("interactions"), list) else []
            for entry in raw_inter[:VLM_ARRAY_LIMITS["interactions"]]:
                if not isinstance(entry, dict) or not isinstance(entry.get("tag"), str):
                    continue
                src, dst = entry.get("source"), entry.get("target")
                if all(isinstance(v, int) and not isinstance(v, bool) for v in (src, dst)):
                    result.interactions.append({"tag": entry["tag"].strip().lower()[:60], "source": src, "target": dst,
                                                "mutual": entry.get("mutual") is True})
        if "thinking" in vlm.capabilities() and not any(
                out.get(k) for k in ("natural_prompt", "subject", "pose_tags", "outfit_tags", "body_tags", "extra_tags")):
            # 思考オフ (think: false) で正常に返れば出さない。空の結果のときだけ理由の候補として出す
            result.warnings.append(
                f"{vlm.model} は思考モデルのため遅く、結果が空になることがあります。"
                "qwen3-vl:8b-instruct や qwen2.5vl:7b など思考しないモデルがおすすめです。"
            )
        return result


@dataclass
class VlmOutcome:
    """Engine._run_vlm の結果。out が None なら VLM は使われていない (未使用 / 失敗)。"""

    out: dict[str, Any] | None = None
    model: str = ""
    analysis: str = ""
    notes: str = ""
    natural_prompt: str = ""
    age: str = ""  # 1 人のときの年齢層 (AGE_BANDS。読めなければ空)
    extra_tags: list[str] = field(default_factory=list)
    negative: list[str] = field(default_factory=list)
    persons: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    pose_tags: list[str] = field(default_factory=list)  # 1 枚モード・1 人のときのポーズ
    outfit_tags: list[str] = field(default_factory=list)  # 1 枚モード・1 人のときの服装
    ref_pose_tags: list[str] = field(default_factory=list)  # 複数枚モード: ② から読んだポーズ
    ref_outfit_tags: list[str] = field(default_factory=list)  # 複数枚モード: ③ から読んだ服装
    ref_body_tags: list[str] = field(default_factory=list)  # 複数枚モード: ⑥ から読んだ体格
    ref_expression_tags: list[str] = field(default_factory=list)  # 複数枚モード: ⑧ から読んだ表情・視線
    style_names: list[str] = field(default_factory=list)  # 複数枚モード: ⑦ から VLM が推定した作品名・作者名
    numbered: list[dict[str, Any]] = field(default_factory=list)  # 番号付きの枠で読ませた人物ごとのポーズ・服装
    interactions: list[dict[str, Any]] = field(default_factory=list)  # 人どうしの動作の向き (番号付きの枠の番号)

    @property
    def ok(self) -> bool:
        return self.out is not None


@dataclass
class TagInputs:
    """Engine._tag_inputs の結果。"""

    image: Image.Image  # ① メインの元画像 (年代判定に使う)
    vlm_image: Image.Image  # VLM に渡す ① (メインの 1 人だけなら切り出した画像)
    ref_images: dict[str, Image.Image]  # 参照画像 (ポーズ / 服装は切り出し後)
    size: tuple[int, int]  # 縦横比に使うサイズ
    tags: TagResult  # 統合したタグ
    person_raw: TagResult  # ① のタグ (VLM に渡す)
    ref_tags: dict[str, TagResult]
    tag_source: dict[str, str]  # タグ名 → 取った枠
    main_person: dict[str, Any] | None  # メインの 1 人だけのときの検出結果
    # ① にあったが、ほかの枠に任せたので外したタグ (服装 / ポーズ / 背景 → タグ名)。出力に戻ってこないかの点検に使う
    displaced: dict[str, list[str]] = field(default_factory=dict)
    persons: list[dict[str, Any]] = field(default_factory=list)  # 人物ごとに分けたときの人物 (左から順)


def numbered_boxes_image(image: Image.Image, boxes: list[Any]) -> Image.Image:
    """人物ごとに番号付きの枠を描いた画像 (VLM に誰がどの番号かを見せる。元の画像は変えない)。"""
    from PIL import ImageDraw, ImageFont

    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    side = max(out.size)
    width = max(3, side // 200)
    size = max(24, side // 18)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # 古い Pillow
        font = ImageFont.load_default()
    colors = [(255, 40, 40), (40, 160, 255), (40, 200, 60), (255, 170, 0), (200, 60, 255), (0, 210, 210)]
    for i, box in enumerate(boxes, 1):
        x0, y0, x1, y1 = (int(v) for v in box)
        color = colors[(i - 1) % len(colors)]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=width)
        label = str(i)
        tx, ty = x0 + width, y0 + width
        l, t, r, b = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle((l - 4, t - 4, r + 4, b + 4), fill=color)
        draw.text((tx, ty), label, fill=(255, 255, 255), font=font)
    return out


def add_numbered_person_tags(
    persons: list[dict[str, Any]], numbered: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """番号付きの枠で VLM に読ませた人物ごとのポーズ・服装のタグを、その番号の人物の区画に足す。
    番号が範囲外・重複しているものは使わない (並び順では結び付けない)。戻り値: (人物, 注意)。"""
    warnings: list[str] = []
    seen: set[int] = set()
    bad: list[str] = []
    out = [dict(p) for p in persons]
    for entry in numbered:
        num = entry["number"]
        if not 1 <= num <= len(out) or num in seen:
            bad.append(str(num))
            continue
        seen.add(num)
        person = out[num - 1]
        have = {loose_tag_key(t) for t in person["tags"]} | {loose_tag_key(t) for t in person.get("common") or []}
        added = [t for t in dedup(entry["pose_tags"] + entry["outfit_tags"]) if loose_tag_key(t) not in have]
        if added:
            person["tags"] = person["tags"] + added
            person["prompt"] = ", ".join(dedup(person["prompt"].split(", ") + added))
            person["vlm_added"] = added
    if bad:
        warnings.append("VLM が存在しない / 重複した人物の番号 (" + ", ".join(bad) + ") を返したので、その分のポーズ・服装は使いませんでした")
    missing = [str(i) for i in range(1, len(out) + 1) if i not in seen]
    if numbered and missing:
        warnings.append("VLM が人物 " + ", ".join(missing) + " のポーズ・服装を返さなかったので、タガーのタグだけで書きました")
    return out, warnings


def split_person_vlm(persons: list[dict[str, Any]], target: str = "") -> str:
    """人物ごとに分けたとき、VLM に人物の並びとタグを伝える指示。"""
    lines = [
        f"The image contains exactly {len(persons)} people. They were detected and tagged one by one, from left to right. "
        "Each of them is outlined in the image by a coloured box with a large number (1 = leftmost). The boxes and numbers "
        "are only markers added for you: never mention them in any text field.",
    ]
    for i, person in enumerate(persons, 1):
        lines.append(f"  person {i} (box {i}, {person['position']}): {_plain(person['tags'][:25])}")
    lines.append(
        "Fill persons with exactly one entry per person in this same order, set number to the box number, and give "
        "pose_tags and outfit_tags for the person inside THAT box only (look at the box, not at the list order). "
        "Keep each person's attributes separate. "
        f"natural_prompt must start with the exact number of people (\"{NUMBER_EN.get(len(persons), len(persons))} "
        "people/women/men\"), then describe each person in order from left to right with their position first "
        "(for example \"On the far left, ...\"), and never move one person's hair, clothing or pose to another."
        + (" End with a short constraint that there are no other people." if target in PROSE_ALLOWS_NEGATION
           else " Use only affirmative descriptions; do not write negations such as 'no other people'.")
    )
    actions = [it["tag"] for it in (persons[0].get("interactions") if persons else None) or []]
    if actions:
        # 向き (する側 / される側) はタガーでは分からないので VLM に聞く (VLM → 骨格 → 向きなし)
        lines.append(
            "The tagger found these actions between people: " + ", ".join(unescape_parens(a) for a in actions) + ". "
            "For each one you can see, add one interactions entry: tag exactly as listed, source = the box number of the "
            "person who does it, target = the box number of the person it is done to, mutual = true only if both do it "
            "to each other. Leave out an action you cannot see. Do not add actions that are not listed."
        )
    return "\n".join(lines)


MAIN_PERSON_VLM = (
    "Describe only the single main person (the largest, most prominent one). If other people are visible, "
    "leave them out of every field; the prompt must show only this one person."
)


def main_person_note(info: dict[str, Any]) -> str:
    """分析メモに出す「メインの 1 人だけ」の結果。"""
    if info.get("error"):
        return "メインの人物だけ: 人物を検出できなかったため画像全体を使いました"
    count = int(info.get("count") or 0)
    if count >= 2:
        return f"メインの人物だけ: 人物 {count} 人を検出し、一番大きく写っている 1 人だけを使いました (プレビューの枠)"
    if count == 1:
        return "メインの人物だけ: 人物は 1 人でした (人数のタグを 1 人にそろえました)"
    return "メインの人物だけ: 人物を検出できませんでした (人数のタグだけ 1 人にそろえました)"


def loose_tag_key(tag: str) -> tuple[str, ...]:
    """表記ゆれ (大文字小文字・_ と空白・複数形・語順・重み) を無視して比べるためのキー。"""
    words = re.findall(r"[a-z0-9]+", _strip_weight(tag).lower())
    return tuple(sorted(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words))


def mentions_any(text: str, tags: list[str]) -> list[str]:
    """文やタグ列の中に、tags のどれかが語として出てくるか (出てきたタグを返す)。"""
    low = " " + re.sub(r"[^a-z0-9]+", " ", unescape_parens(text).lower()) + " "
    found = []
    for tag in tags:
        words = re.sub(r"[^a-z0-9]+", " ", norm_tag(tag)).strip()
        if words and f" {words} " in low:
            found.append(tag)
    return found


def reference_warnings(person: TagResult, ref_tags: dict[str, TagResult]) -> list[str]:
    """複数枚モードで、参照画像の中身が役割に合っていないときの注意。"""
    warnings: list[str] = []
    person_count, _ = person_count_hint(person)
    if person_count >= 2:
        warnings.append(f"人物画像に {person_count} 人写っています。1 人を想定しているため、外見が混ざることがあります。")
    if "pose" in ref_tags:
        pose_have = {n for n, _ in ref_tags["pose"].general}
        pose_count, _ = person_count_hint(ref_tags["pose"])
        if pose_count >= 2:
            warnings.append(f"ポーズ画像に {pose_count} 人写っています。ポーズのタグが混ざることがあります。")
        if "no_humans" in pose_have:
            warnings.append("ポーズ画像に人物が検出されませんでした (no_humans)。")
        if not any(classify_pose_tag(n) for n in pose_have):
            warnings.append("ポーズ画像からポーズのタグを検出できませんでした。「一般」のしきい値を下げるか、VLM を使ってください。")
    if "outfit" in ref_tags and not any(classify_outfit_tag(n) for n, _ in ref_tags["outfit"].general):
        warnings.append("服装画像から服・装飾品のタグを検出できませんでした。「一般」のしきい値を下げるか、VLM を使ってください。")
    if "body" in ref_tags:
        body_have = {n for n, _ in ref_tags["body"].general}
        if not any(body_tag(n) for n in body_have):
            warnings.append("体格画像から体格のタグを検出できませんでした。「一般」のしきい値を下げるか、VLM を使ってください。")
        person_have = {n for n, _ in person.general}

        def sex(have: set[str]) -> str:
            girl, boy = bool(have & {"1girl", "2girls", "multiple_girls"}), bool(have & {"1boy", "2boys", "multiple_boys"})
            return "girl" if girl and not boy else "boy" if boy and not girl else ""

        if sex(body_have) and sex(person_have) and sex(body_have) != sex(person_have):
            warnings.append(
                f"⑥ 体格の画像は{'男性' if sex(body_have) == 'boy' else '女性'}、① メインは{'男性' if sex(person_have) == 'boy' else '女性'}です。"
                "性別は ① のまま、体格のタグ (筋肉・胸など) は ⑥ から取るので、体つきが性別と合わないことがあります。"
            )
    if "props" in ref_tags and not any(reference_category(n) == "props" for n, _ in ref_tags["props"].general):  # noqa: SIM102
        warnings.append("小物画像から小物のタグを検出できませんでした。「一般」のしきい値を下げるか、VLM を使ってください。")
    if "style" in ref_tags and not any(reference_category(n, True) == "style" for n, _ in ref_tags["style"].general):
        warnings.append("作風画像から画風・技法のタグを検出できませんでした (絵か写真かの判定は ⑦ から行います)。"
                        "「設定」の画風のしきい値を下げるか、VLM を使ってください。")
    return warnings


def reference_header(path: Path, refs: dict[str, Path], tags: TagResult, tag_source: dict[str, str]) -> tuple[str, str]:
    """複数枚モードの分析メモの見出し (どの画像から何を取ったか) と、メモに足す注意。"""
    used = " / ".join(
        f"{REF_SLOTS[slot]}: " + (
            refs[slot].name if slot in refs else path.name if slot == "person"
            else "(③ から)" if slot_owner(slot, refs) == "outfit" else "(1 枚目)"
        )
        for slot in REF_SLOTS
    )
    lines = []
    for slot in refs:
        picked = ", ".join(sd_tag(n) for n, _ in tags.general if tag_source.get(n) == slot)
        lines.append(f"{REF_SLOTS[slot]}: {picked or '(なし)'}")
    header = f"複数枚モード ({used})\n" + "\n".join(lines) + "\n"
    notes = ""
    if "pose" in refs:
        notes += (
            "\nポーズを正確に再現したいときは、② ポーズ画像を ControlNet (OpenPose / DWPose) の入力にし、"
            "このプロンプトと併用してください。"
        )
    if "outfit" in refs and tags.characters:
        notes += "\nキャラクター名のタグは標準の衣装を呼び出します。③ の服装と混ざるときは、キャラ名タグの重みを下げるか外してください。"
    return header, notes


def compose_positive(
    kind: str, target: str, *, tags: TagResult, medium: str, size: tuple[int, int], extras: dict[str, dict[str, Any]],
    shared_chars: list[str], shared_general: list[str], extra_tags: list[str], natural_prompt: str,
    persons: list[dict[str, Any]], quality_on: bool, use_break: bool, vlm_attempted: bool, references: bool = False,
    no_style: bool = False, style_first: bool = False, age: str = "",
) -> tuple[str, list[str]]:
    """出力先の書式でポジティブプロンプトを組み立てる。戻り値: (ポジティブ, 注意)。
    references (最大 4 枚モード) では、VLM のまとめ文は ① の服・ポーズを持ち込むので使わず、枠ごとの内容から書く。"""
    warnings: list[str] = []
    multi = len(persons) >= 2
    detected = len(persons) >= 2 and bool(persons[0].get("box"))
    if no_style:  # 作風の特徴を入れない: 文に絵か写真かを書かず、品質タグも絵か写真かを表す語を外す
        quality_medium, medium = medium, "unspecified"
    else:
        quality_medium = medium
    if kind == "prose" and detected:
        # VLM の自然文は人物をまとめて書き、各人の特徴を落とす (実機の qwen2.5vl で「Each woman is wearing ...」になった)。
        # 人物は 1 人ずつ確かめたタグから書き、場所・照明・構図・画風には VLM の文があれば使う
        return compose_prose_fallback(extras, medium, size, target, tags, persons, style_first=style_first, age=age), warnings
    if kind == "prose" and references:
        # VLM の自然文は ② / ③ の役割を守らず ① の服やポーズで書くことがある (服装・ポーズが ① に負ける原因)。
        # 服装は ③、ポーズは ② から取ったタグ (と VLM の役割別の文) で書く
        return compose_prose_fallback(extras, medium, size, target, tags, persons, use_text=True,
                                      style_first=style_first, age=age), warnings
    if kind == "prose" and natural_prompt:
        if target == "krea2":  # 画角を文の頭に (Krea 2 だけ)
            return krea_lead_with_framing(natural_prompt, krea_framing(tags)), warnings
        return natural_prompt, warnings
    if kind == "anima":
        # Anima 公式の並び: [品質/レーティング] [1girl 等] [キャラ] [一般タグ] + 自然文
        head = list(ANIMA_QUALITY) if quality_on else []
        head.append(ANIMA_RATING.get(tags.rating, "safe"))
        counts = [sd_tag(n) for n, _ in tags.general if n in COUNT_TAGS]
        rest = [t for t in shared_general if t not in counts]
        positive = ", ".join(dedup(head + counts + shared_chars + rest + extra_tags))
        if detected:  # 人物は 1 人ずつ確かめた内容で書き (Anima: 名前だけでなく外見を文で)、場面は VLM の文があれば足す
            scene = " ".join(x for x in (extras["background"]["text"], extras["lighting"]["text"]) if x)
            sentences = describe_persons_en(persons, target) + (" " + scene if scene else "")
        elif references:  # 最大 4 枚モード: 服装 (③) とポーズ (②) をはっきり文で書く
            parts = []
            for key, lead in (("subject", ""), ("outfit", "Wearing "), ("pose", "Pose: ")):
                items = extras[key]["tags"]
                if key == "subject":
                    # 被写体のタグは前のタグ列に全部あるので、文では見分ける特徴 (髪・目・顔) だけ書く
                    # (全部書くと ① の比重が倍になる)
                    items = [t for t in items if person_tag_priority(t) == 1]
                text = extras[key]["text"] or (lead + _plain(items) if items else "")
                if text:
                    parts.append(text.rstrip(".") + ".")
            sentences = " ".join(parts)
        else:
            sentences = natural_prompt or " ".join(p["description"] for p in persons if p["description"])
        return (positive + ". " + sentences if sentences else positive), warnings
    if kind == "prose":
        # VLM が無い / 失敗したときは、振り分けたタグから構造化した英文を組み立てる
        if vlm_attempted:
            warnings.append("VLM から自然文が得られなかったため、タグから組み立てた英文を出力しました。")
        return compose_prose_fallback(extras, medium, size, target, tags, persons, style_first=style_first, age=age), warnings
    if not quality_on:
        quality: list[str] = []
    elif target == "novelai":
        quality = list(NAI_QUALITY)
    else:
        quality = [q for q in QUALITY_TAGS[quality_medium] if not (no_style and medium_word(q))]
    # 並びは NoobAI の推奨 (品質 → 人数 → キャラ → 一般タグ)
    counts = [t for t in shared_general if norm_tag(t).replace(" ", "_") in PERSON_COUNT_TAGS]
    if target == "novelai":  # NovelAI の公式例は人数だけ (2girls)。multiple girls は付けない
        counts = [t for t in counts if not norm_tag(t).startswith("multiple ")]
    rest = [t for t in shared_general if norm_tag(t).replace(" ", "_") not in PERSON_COUNT_TAGS]
    main = dedup(quality + counts + shared_chars + rest + extra_tags)
    if multi and target == "novelai":
        # NovelAI V4 は人物ごとのプロンプトを Character 欄に入れる。本文は共有タグだけにする。
        # 向きが分かった人どうしの動作はキャラ欄 (source# / target# / mutual#) に移す。分からないものは本文に残す
        moved = {norm_tag(it["tag"]) for it in (persons[0].get("interactions") or []) if it.get("pair")}
        main = [t for t in main if norm_tag(t) not in moved]
        warnings.append("複数人物: 「人物ごと」タブの各行を NovelAI の Character 1, 2 … に貼ってください (本文は共有タグのみ)。")
        return ", ".join(main), warnings
    if multi and use_break:
        # A1111 / ComfyUI の BREAK で人物ごとにチャンクを分ける
        return ", ".join(main) + "".join("\nBREAK " + p["prompt"] for p in persons), warnings
    if multi:
        return ", ".join(dedup(main + [t for p in persons for t in p["tags"]])), warnings
    return ", ".join(main), warnings


def quality_check(
    positive: str, negative_text: str, persons: list[dict[str, Any]], kind: str, target: str,
    displaced: dict[str, list[str]] | None = None, color_only: bool = False,
) -> list[str]:
    """出力したプロンプトを機械的に点検する (最適化の途中で品質が落ちていないか)。問題の説明の一覧を返す。
    - ポジティブとネガティブの両方にあるタグ
    - 人数タグと人物の数のずれ
    - 人物ごとに検出した特徴がプロンプトに入っていない (取りこぼし)
    - 人物の区画が CLIP の 75 トークンを超えて、次の人の区画にはみ出している"""
    issues: list[str] = []
    if color_only:
        left = mentions_any(positive.split(" --")[0], MONOCHROME_WORDS + [t.replace("_", " ") for t in MONOCHROME_TAGS])
        if left:
            issues.append("白黒の情報を消す設定なのに、プロンプトに " + ", ".join(left) + " が残っています (VLM の文の可能性)")
    names = {"outfit": "③ 服装", "pose": "② ポーズ", "background": "④ 背景", "body": "⑥ 体格", "style": "⑦ 作風・質感",
             "control": "ボーン・カメラ"}
    for slot, gone in (displaced or {}).items():
        leaked = mentions_any(positive, gone)
        if leaked:
            issues.append(f"{names.get(slot, slot)} を指定したのに、① の {', '.join(leaked)} がプロンプトに残っています")
    if target == "midjourney":
        body, sep, params = positive.partition(" --")
        if "::" in body or re.search(r"\([^()]*:\d", body):
            issues.append("Midjourney の本文に重み付け (:: や (tag:1.2)) が入っています (V8.2 / Niji 7 は非対応)")
        if not sep or not re.match(r"ar \d+:\d+", params):
            issues.append("Midjourney のパラメータ (--ar 整数:整数) が末尾にありません")
        if re.search(r"\b(no|without)\b", body, re.IGNORECASE):
            issues.append("Midjourney の本文に否定文があります (公式: 除外は --no で)")
        words = len(re.findall(r"[A-Za-z]+", body))
        if words > 80:
            issues.append(f"Midjourney の本文が約 {words} 語と長めです (公式: 短く簡潔なプロンプトが最良)")
        no_part = positive.split(" --no ", 1)[1] if " --no " in positive else ""
        clash = (set(re.findall(r"[a-z0-9]+", no_part.lower())) & set(re.findall(r"[a-z0-9]+", body.lower()))) - {"a", "the", "and", "of"}
        if clash:
            issues.append("Midjourney の本文と --no の両方に出てくる語があります: " + ", ".join(sorted(clash)))
    if kind == "prose":
        return issues
    tag_part = positive.partition(". ")[0] if kind == "anima" else positive
    chunks = tag_part.split("\nBREAK ")
    pos_tags = {norm_tag(t) for chunk in chunks for t in chunk.split(",") if t.strip()}
    neg_tags = {norm_tag(t) for t in negative_text.split(",") if t.strip()}
    clash = sorted(pos_tags & neg_tags)
    if clash:
        issues.append("ポジティブとネガティブの両方に入っています: " + ", ".join(clash))
    if len(persons) >= 2:
        seen_prompts: dict[str, int] = {}
        for i, person in enumerate(persons, 1):
            key = ", ".join(sorted(norm_tag(t) for t in person["tags"]))
            if key in seen_prompts:
                issues.append(
                    f"人物 {seen_prompts[key]} と人物 {i} のプロンプトが同じです (同じ人を二重に数えた、重なっていて見分けられない、"
                    "VLM が同じ人を 2 回書いた、のいずれか。人数が合っているか画像で確かめてください)"
                )
            else:
                seen_prompts[key] = i
    if len(persons) >= 2 and persons[0].get("box"):
        expected = {norm_tag(sd_tag(t)) for t in count_tags([p["subject"] for p in persons], with_multiple=target != "novelai")}
        # 人数タグは共有部分 (最初の区画) で見る。人物の区画の先頭の 1girl などは各人の性別なので数えない
        shared_tags = {norm_tag(t) for t in chunks[0].split(",") if t.strip()}
        present = {t for t in shared_tags if t.replace(" ", "_") in PERSON_COUNT_TAGS}
        if present != expected:
            issues.append(
                f"人数タグが人物の数 ({len(persons)} 人) と合いません: 出力 {', '.join(sorted(present)) or 'なし'} / "
                f"期待 {', '.join(sorted(expected))}"
            )
        if target != "novelai":  # NovelAI は人物ごとを Character 欄に貼るので本文には無い
            text = unescape_parens(positive).lower()  # Anima は人物ごとの特徴を文の部分に書く
            for i, person in enumerate(persons, 1):
                missing = [t for t in person["tags"] if norm_tag(t) not in pos_tags and norm_tag(t) not in text]
                if missing:
                    issues.append(f"人物 {i} の特徴がプロンプトに入っていません: {', '.join(missing)}")
        for i, chunk in enumerate(chunks[1:], 1):
            tokens = clip_tokens(chunk)
            if tokens > CLIP_CHUNK_TOKENS:
                issues.append(
                    f"人物 {i} の区画が約 {tokens} トークンで {CLIP_CHUNK_TOKENS} を超え、次の区画にはみ出します "
                    "(特徴が混ざる原因になります。人物ごとタブで不要なタグを消してください)"
                )
    return issues


def paste_formats(
    positive: str, persons: list[dict[str, Any]], kind: str, target: str, extras: dict[str, dict[str, Any]], medium: str,
    distinct_faces: bool = False,
) -> dict[str, str]:
    """人物ごとタブに出す、ツールにそのまま貼れる形 (見出し → 本文)。書式はそれぞれの公式資料に合わせている。"""
    out: dict[str, str] = {}
    if len(persons) < 2 or not persons[0].get("area"):
        return out
    if kind == "tags" and target != "novelai":
        ratios = regional_columns(persons)
        if ratios:
            # Regional Prompter の README: 共通部分は ADDCOMM、領域は BREAK で区切る。左右の分割は Columns (旧版 Horizontal)
            shared = positive.split("\nBREAK ")[0]
            body = " BREAK\n".join(p["prompt"] for p in persons)
            out["Regional Prompter (A1111 / Forge)"] = (
                f"Main Splitting: Columns (旧版は Horizontal) / Divide Ratio: {','.join(map(str, ratios))} / "
                "モード: Attention (小さい人物が混ざるときは Latent)\n"
                f"{shared} ADDCOMM\n{body}"
            )
            region_neg = region_negatives(persons)
            if any(region_neg):
                # README: "negative prompts can also be separated by BREAK"。領域の順に並べる (共通のネガティブは先頭)
                out["Regional Prompter のネガティブ (領域ごと。男女の顔の取り違えを防ぐ)"] = (
                    "[ネガティブの本文] ADDCOMM\n" + " BREAK\n".join(", ".join(n) or " " for n in region_neg)
                )
        lines = []
        for i, p in enumerate(persons, 1):
            pct, px = p["area"]["pct"], p["area"]["px"]
            lines.append(
                f"人物 {i}: Set Area with Percentage → x {pct['x']}, y {pct['y']}, width {pct['width']}, height {pct['height']}"
                f" / Set Area ({px['canvas']}) → x {px['x']}, y {px['y']}, width {px['width']}, height {px['height']}"
            )
            lines.append(f"  プロンプト: {p['prompt']}")
        # 領域の先頭はその人の 1girl など。全体の人数 (Forge Couple の勧め) は ComfyUI の Set Area では人が増えた (DEVLOG 82)
        out["ComfyUI 領域指定 (各人のプロンプトを CLIP Text Encode → 領域指定 → Conditioning (Combine))"] = "\n".join(lines)
    if target == "novelai":
        # NovelAI 公式: キャラ欄の代わりに「ベース | キャラ1 | キャラ2」と書ける (キャラ欄とは併用できない)
        out["NovelAI 1 行形式 (| 区切り。キャラ欄とは併用できない)"] = " | ".join([positive] + [p["prompt"] for p in persons])
    if target == "flux2":  # BFL の FLUX.2 [pro] / [max] ガイドの JSON。klein は公式に記載が無いので出さない
        out["FLUX.2 構造化プロンプト (JSON。公式ガイドが複数の被写体に推奨)"] = flux_structured_prompt(persons, extras, medium, distinct_faces)
    if target in ("sdxl", "anima", "flux", "flux2", "flux2klein", "krea2", "qwen", "zimageturbo", "zimage", "mossimage"):
        # 同じ顔になるのを防ぐ一番確実な手: 生成後に顔を 1 人ずつ、その人のプロンプトで描き直す
        faces = face_prompts(persons)
        out["ComfyUI FaceDetailer / Detailer の wildcard (顔を左から 1 人ずつ描き直す。Impact Pack の [ASC] + [SEP])"] = (
            "[ASC]\n" + "\n".join(f"{f} [SEP]" for f in faces)
        )
        out["ADetailer のプロンプト (設定の Sort bounding boxes by を Position (left to right) に)"] = " [SEP] ".join(faces)
    return out


def mixed_gender(persons: list[dict[str, Any]]) -> bool:
    return len({p.get("subject") for p in persons} & {"girl", "boy"}) == 2


def region_negatives(persons: list[dict[str, Any]]) -> list[list[str]]:
    """人ごとのネガティブ (Regional Prompter の領域ごと / NovelAI のキャラ別 Undesired Content)。
    男女が混ざるときは、別の性別と性別があいまいな顔 (Danbooru の androgynous) を打ち消す。
    元の人に androgynous などのタグがあれば入れない (その人の特徴なので)。"""
    out = []
    mixed = mixed_gender(persons)
    for person in persons:
        own = {_tag_key(t) for t in person["tags"]}
        negs: list[str] = []
        if mixed and person.get("subject") in ("girl", "boy"):
            negs += ["1boy" if person["subject"] == "girl" else "1girl"]
            if "androgynous" not in own:
                negs.append("androgynous")
        out.append(negs)
    return out


def novelai_undesired(persons: list[dict[str, Any]]) -> list[list[str]]:
    """NovelAI のキャラ別 Undesired Content の候補。公式は「特徴が別のキャラに漏れたら、そのキャラの
    Undesired Content で打ち消す」と勧めているので、ほかの人の髪・目・顔の特徴のうち自分に無いものを挙げる。
    男女が混ざるときは別の性別 (NovelAI のキャラ欄の書き方で girl / boy) と androgynous も入れる。"""
    features = [{t for t in p["tags"] if person_tag_priority(t) == 1} for p in persons]
    result = []
    for i, own in enumerate(features):
        others = [t for j, f in enumerate(features) if j != i for t in sorted(f)]
        uc = [t for t in others if norm_tag(t) not in {norm_tag(x) for x in own}]
        uc += [NAI_SUBJECT.get(t[1:], t) if t in ("1girl", "1boy") else t for t in region_negatives(persons)[i]]
        result.append(dedup(uc))
    return result


def face_prompts(persons: list[dict[str, Any]]) -> list[str]:
    """顔を 1 人ずつ描き直すとき (ADetailer / FaceDetailer) の人ごとのプロンプト: 性別 + 髪・目・顔立ちの特徴。
    全員に共通の特徴 (common) も入れる (顔だけを描き直すので共有部分が無い)。"""
    out = []
    for person in persons:
        features = [t for t in list(person.get("common") or []) + person["tags"] if person_tag_priority(t) == 1]
        out.append(", ".join(dedup([SUBJECT_TOKEN.get(person.get("subject", "other"), "1other")] + features)))
    return out


def finalize_prompts(
    positive: str, negative: list[str], persons: list[dict[str, Any]], kind: str, target: str, nsfw: bool,
    mosaic_level: str | None,
) -> tuple[str, str, list[dict[str, Any]], bool]:
    """モザイク指定・NovelAI の括弧・ネガティブ非対応の出力先を反映する。
    mosaic_level は設定でモザイクが有効なときの強さ (無効なら None)。戻り値: (ポジティブ, ネガティブ, 人物, モザイク適用)。"""
    negative_text = ", ".join(dedup(negative))
    mosaic_applied = False
    if nsfw and mosaic_level is not None:
        # NSFW 判定ならプロンプトにモザイク指定を足す (生成画像にモザイクがかかる)
        level = mosaic_level if mosaic_level in MOSAIC_LEVELS else DEFAULT_MOSAIC_LEVEL
        positive, negative_text = add_censor_prompt(positive, negative_text, kind, level, target)
        mosaic_applied = True
    if target == "novelai":
        # NovelAI は () を特別扱いしないのでエスケープを外す (Character 欄に貼る人物ごとのプロンプトも)
        positive = unescape_parens(positive)
        negative_text = unescape_parens(negative_text)
        # キャラ欄: 性別 → 特徴 → 動作 (source# / target# / mutual#) → 位置の文 (公式の例もタグの後ろに文)
        persons = [
            p | {
                "prompt": ", ".join(dedup([NAI_SUBJECT.get(p.get("subject", "other"), "other")] + [unescape_parens(t) for t in p["tags"]]
                                          + novelai_action_tags(persons, i)))
                + (". " + novelai_position_sentence(p) if len(persons) >= 2 and novelai_position_sentence(p) else ""),
                "tags": [unescape_parens(t) for t in p["tags"]],
            }
            for i, p in enumerate(persons)
        ]
    if TARGETS[target]["negative"] is False:
        # ネガティブ非対応の出力先 (Krea 2 / ChatGPT / Grok / Nano Banana) には出さない。
        # 避けたい要素は本文側 (VLM の指示や 'No text, watermark or logo.') で表現している
        negative_text = ""
    return positive, negative_text, persons, mosaic_applied


def describe_error(e: BaseException) -> str:
    """例外を日本語の説明に変換する。"""
    if isinstance(e, (OllamaError, DownloadError)):
        return str(e)
    if isinstance(e, OperationCancelled):
        return "中止しました"
    name = type(e).__name__
    module = type(e).__module__ or ""
    if module.startswith("onnxruntime"):
        return f"ONNX Runtime でエラーが起きました ({name}: {e})"
    if isinstance(e, MemoryError):
        return "メモリが不足しました。小さいタガーモデルに切り替えてください。"
    if isinstance(e, OSError):  # PIL.UnidentifiedImageError も含む
        return f"ファイルを読み込めません: {e}"
    return f"{name}: {e}"


# --------------------------------------------------------------------------
# 画像に埋め込まれた生成情報 (元のプロンプト) の読み取り
#   A1111 / Forge: PNG の tEXt "parameters" (JPEG / WebP は EXIF の UserComment)。書式は modules/processing.py の
#     create_infotext: "<prompt>\nNegative prompt: <negative>\nSteps: ..., Sampler: ..."
#   ComfyUI: PNG の "prompt" (API 形式のノード JSON) と "workflow" (nodes.py の SaveImage)
#   NovelAI: PNG の "Comment" (JSON。prompt / uc、V4 は v4_prompt.caption.base_caption / char_captions)
#   Fooocus: "parameters" が JSON、InvokeAI: "invokeai_metadata"
#   メタデータが消された画像 (SNS に上げた画像・スクリーンショット・貼り付けたビットマップ) からは読めない
# --------------------------------------------------------------------------

def _decode_user_comment(raw: Any) -> str:
    """EXIF の UserComment (先頭 8 バイトが文字コード) を文字列にする。"""
    if isinstance(raw, str):
        return raw.strip("\x00 ")
    if not isinstance(raw, bytes):
        return ""
    head, body = raw[:8], raw[8:]
    if head.startswith(b"UNICODE"):
        # A1111 は piexif の UserComment.dump(encoding="unicode") = UTF-16 BE。LE で書く道具もあるので先頭で見分ける
        encoding = "utf-16-be" if body[:1] == b"\x00" else "utf-16-le"
        return body.decode(encoding, "replace").strip("\x00 ")
    if head.startswith(b"ASCII") or head == b"\x00" * 8:
        return body.decode("utf-8", "replace").strip("\x00 ")
    return raw.decode("utf-8", "replace").strip("\x00 ")


def _parse_a1111(text: str) -> tuple[str, str, str]:
    """A1111 形式の infotext → (ポジティブ, ネガティブ, 設定の行)。"""
    lines = text.strip().split("\n")
    settings = lines.pop().strip() if lines and re.match(r"\s*Steps: ", lines[-1]) else ""
    body = "\n".join(lines)
    if body.startswith("Negative prompt: "):
        return "", body[len("Negative prompt: "):].strip(), settings
    positive, _, negative = body.partition("\nNegative prompt: ")
    return positive.strip(), negative.strip(), settings


def _comfy_texts(graph: dict[str, Any], ref: Any, side: str, depth: int = 0) -> list[str]:
    """ComfyUI の API 形式のグラフで、サンプラーの入力 (positive / negative) からテキストのノードまでたどる。
    ControlNet の適用や Conditioning の結合を挟んでも、同じ側 (side) の入力を追う。"""
    if depth > 20 or not (isinstance(ref, list) and ref):
        return []
    node = graph.get(str(ref[0]))
    if not isinstance(node, dict):
        return []
    inputs = node.get("inputs") or {}
    texts: list[str] = []
    for key in ("text", "text_g", "text_l", "prompt", "string", "value", "t5xxl", "clip_l"):
        value = inputs.get(key)
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())
        elif isinstance(value, list):
            texts += _comfy_texts(graph, value, side, depth + 1)
    for key in (side, "conditioning", "conditioning_1", "conditioning_2", "conditioning_to", "conditioning_from"):
        if isinstance(inputs.get(key), list):
            texts += _comfy_texts(graph, inputs[key], side, depth + 1)
    return list(dict.fromkeys(texts))  # text_g と text_l が同じ文のときなどの重複を外す


def _parse_comfy(prompt_json: str) -> tuple[str, str, str]:
    graph = json.loads(prompt_json)
    if not isinstance(graph, dict):
        raise ValueError("ComfyUI のグラフではありません")
    positives: list[str] = []
    negatives: list[str] = []
    settings: list[str] = []
    for node_id in sorted(graph, key=lambda k: (len(str(k)), str(k))):
        node = graph[node_id]
        if not isinstance(node, dict) or not re.search(r"Sampler|Guider", str(node.get("class_type", ""))):
            continue
        inputs = node.get("inputs") or {}
        pos = _comfy_texts(graph, inputs.get("positive") or inputs.get("conditioning"), "positive")
        neg = _comfy_texts(graph, inputs.get("negative"), "negative")
        positives += [t for t in pos if t not in positives]
        negatives += [t for t in neg if t not in negatives]
        if not settings and (pos or "seed" in inputs or "noise_seed" in inputs):
            for key in ("seed", "noise_seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"):
                if key in inputs and not isinstance(inputs[key], list):
                    settings.append(f"{key}: {inputs[key]}")
    for node in graph.values():  # 使ったモデル・LoRA
        inputs = (node.get("inputs") or {}) if isinstance(node, dict) else {}
        for key in ("ckpt_name", "unet_name", "lora_name"):
            if isinstance(inputs.get(key), str):
                settings.append(f"{key}: {inputs[key]}")
    return "\n\n".join(positives), "\n\n".join(negatives), ", ".join(dict.fromkeys(settings))


def _parse_novelai(comment: str) -> tuple[str, str, str]:
    data = json.loads(comment)
    if not isinstance(data, dict):
        raise ValueError("NovelAI の Comment ではありません")

    def v4(key: str) -> str:
        caption = ((data.get(key) or {}).get("caption") or {}) if isinstance(data.get(key), dict) else {}
        base = str(caption.get("base_caption") or "").strip()
        chars = [str(c.get("char_caption") or "").strip() for c in caption.get("char_captions") or [] if isinstance(c, dict)]
        lines = [base] + [f"Character {i}: {c}" for i, c in enumerate(chars, 1) if c]
        return "\n".join(x for x in lines if x)

    positive = v4("v4_prompt") or str(data.get("prompt") or "").strip()
    negative = v4("v4_negative_prompt") or str(data.get("uc") or "").strip()
    keys = ("steps", "scale", "cfg_rescale", "seed", "sampler", "noise_schedule", "width", "height")
    settings = ", ".join(f"{k}: {data[k]}" for k in keys if k in data)
    return positive, negative, settings


def generation_info_from_image(im: Image.Image) -> dict[str, str] | None:
    """開いた画像の生成情報。{"source", "positive", "negative", "settings", "raw"}。無ければ None。
    読めない書式は raw (元のデータ) だけを返す (positive が空)。"""
    info = {k: v for k, v in (im.info or {}).items() if isinstance(v, str)}
    if not any(k in info for k in ("parameters", "prompt", "Comment", "invokeai_metadata")):
        try:
            exif = im.getexif()
            comment = _decode_user_comment(exif.get_ifd(0x8769).get(0x9286))  # Exif IFD の UserComment
        except Exception:  # noqa: BLE001 - 壊れた EXIF は読まない
            comment = ""
        if comment:
            info["parameters"] = comment
    candidates: list[tuple[str, str, Callable[[str], tuple[str, str, str]]]] = []
    params = info.get("parameters", "").strip()
    if params.startswith("{"):
        def fooocus(text: str) -> tuple[str, str, str]:
            data = json.loads(text)
            rest = ", ".join(f"{k}: {v}" for k, v in data.items() if k not in ("prompt", "negative_prompt", "full_prompt",
                                                                                 "full_negative_prompt") and not isinstance(v, (dict, list)))
            return str(data.get("prompt") or "").strip(), str(data.get("negative_prompt") or "").strip(), rest
        candidates.append(("Fooocus", params, fooocus))
    elif params:
        candidates.append(("A1111 / Forge", params, _parse_a1111))
    if info.get("Comment", "").strip().startswith("{"):
        candidates.append(("NovelAI", info["Comment"], _parse_novelai))
    if info.get("prompt", "").strip().startswith("{"):
        candidates.append(("ComfyUI", info["prompt"], _parse_comfy))
    if info.get("invokeai_metadata", "").strip().startswith("{"):
        def invoke(text: str) -> tuple[str, str, str]:
            data = json.loads(text)
            model = data.get("model") if isinstance(data.get("model"), dict) else {}
            keys = ("seed", "steps", "cfg_scale", "scheduler", "width", "height")
            rest = [f"{k}: {data[k]}" for k in keys if k in data] + ([f"model: {model['name']}"] if model.get("name") else [])
            return str(data.get("positive_prompt") or "").strip(), str(data.get("negative_prompt") or "").strip(), ", ".join(rest)
        candidates.append(("InvokeAI", info["invokeai_metadata"], invoke))
    if not candidates:
        return None
    for source, raw, parse in candidates:
        try:
            positive, negative, settings = parse(raw)
        except (ValueError, TypeError, AttributeError):
            continue
        if positive or negative:  # ComfyUI の元のデータは API 形式 (prompt)。workflow は画面の配置情報で長いので出さない
            return {"source": source, "positive": positive, "negative": negative, "settings": settings, "raw": raw}
    source, raw, _ = candidates[0]
    return {"source": source + " (書式を読み取れませんでした)", "positive": "", "negative": "", "settings": "", "raw": raw}


def read_generation_info(path: Path) -> dict[str, str] | None:
    try:
        with Image.open(path) as im:
            return generation_info_from_image(im)
    except Exception:  # noqa: BLE001 - 読めない画像は「生成情報なし」
        return None


# --------------------------------------------------------------------------
# クリップボードの画像 (Ctrl+V)
# --------------------------------------------------------------------------

PASTE_DIR = Path(os.environ.get("TEMP") or os.environ.get("TMP") or str(APP_DIR)) / "i2t_paste"  # 自動では消さない


def clipboard_image() -> Image.Image | list[Path] | None:
    """クリップボードの画像。ビットマップなら Image、エクスプローラーでコピーしたファイルならパスの一覧、無ければ None。"""
    try:
        from PIL import ImageGrab

        grabbed = ImageGrab.grabclipboard()
    except Exception:  # noqa: BLE001 - 非対応の環境・読めない形式は「画像なし」
        return None
    if isinstance(grabbed, Image.Image):
        return grabbed
    if isinstance(grabbed, list):
        paths = [Path(p) for p in grabbed if isinstance(p, str)]
        return paths or None
    return None


def save_pasted_image(image: Image.Image, folder: Path | None = None) -> Path:
    """貼り付けたビットマップを解析・履歴のために PNG にする (OSError はそのまま上げる)。"""
    folder = folder or PASTE_DIR
    folder.mkdir(parents=True, exist_ok=True)
    now = time.time()
    stem = time.strftime("paste-%Y%m%d-%H%M%S", time.localtime(now)) + f"-{int(now * 1000) % 1000:03d}"
    path = folder / f"{stem}.png"
    image.save(path)
    return path


# --------------------------------------------------------------------------
# 結果のチップ表示 (, 区切りの塊ごとに表示し、選んで Del で消す・Ctrl+Z で戻す。
# 右上のボタンで日本語訳に切り替える。訳は Ollama で作り translations.json に保存して使い回す。
# Ollama があるときだけチップ表示にし、無ければ今までどおりの文字の欄。設定でも切り替えられる)
# --------------------------------------------------------------------------

TRANSLATIONS_PATH = APP_DIR / "translations.json"
TRANSLATE_BATCH = 30  # 1 回の問い合わせで訳す塊の数 (多いと出力が長くなり、途中で切れやすい)
_CHIP_WEIGHT_RE = re.compile(r"^\((?P<core>.+):(?P<w>-?\d+(?:\.\d+)?)\)$")  # A1111 の (tag:1.3)
_CHIP_NAI_WEIGHT_RE = re.compile(r"^(?P<w>-?\d+(?:\.\d+)?)::(?P<core>.+?)::$")  # NovelAI の 1.3::tag::


def split_prompt_chips(text: str) -> list[list[str]]:
    """プロンプト → 行ごとの塊の一覧。, で区切るが、括弧 ( ) [ ] { } の中の , では区切らない (重みのまとまりを壊さない)。
    空の行も空の一覧として残す (Regional Prompter などの改行を保つ)。"""
    lines: list[list[str]] = []
    for line in str(text or "").split("\n"):
        chunks: list[str] = []
        depth, start = 0, 0
        for i, ch in enumerate(line):
            if ch in "([{" and (i == 0 or line[i - 1] != "\\"):
                depth += 1
            elif ch in ")]}" and (i == 0 or line[i - 1] != "\\"):
                depth = max(0, depth - 1)
            elif ch == "," and depth == 0:
                chunks.append(line[start:i])
                start = i + 1
        chunks.append(line[start:])
        lines.append([c.strip() for c in chunks if c.strip()])
    while lines and not lines[-1]:
        lines.pop()
    return lines


def join_prompt_chips(lines: list[list[str]]) -> str:
    return "\n".join(", ".join(line) for line in lines)


def chip_core(chunk: str) -> tuple[str, str]:
    """塊 → (訳に使う語, 重み)。(tag:1.3) と 1.3::tag:: の重みを外し、括弧のエスケープと _ を戻す。重みが無ければ ""。"""
    text, weight = chunk.strip(), ""
    m = _CHIP_WEIGHT_RE.match(text) or _CHIP_NAI_WEIGHT_RE.match(text)
    if m:
        text, weight = m.group("core").strip(), m.group("w")
    text = text.replace("\\(", "(").replace("\\)", ")").replace("_", " ")
    return re.sub(r"\s+", " ", text).strip(), weight


def lookup_ja(table: dict[str, str], core: str) -> str:
    """辞書を引く。VLM の書き方の揺れ (off-shoulder / off shoulder、bow tie / bowtie、frilly / frilled、複数形) も同じ語とみなす。"""
    key = core.lower().strip()
    if key in table:
        return table[key]
    spaced = re.sub(r"\s+", " ", key.replace("-", " "))
    singular = spaced[:-1] if spaced.endswith("s") and not spaced.endswith("ss") else spaced  # 最後の語の複数形だけ
    for cand in (spaced, spaced.replace(" ", ""), key.replace(" ", "-"), re.sub(r"\bfrilly\b", "frilled", spaced), singular):
        if cand in table:
            return table[cand]
    return ""


def chip_label_ja(chunk: str, table: dict[str, str]) -> str:
    """日本語表示のときのチップの文字。訳が無ければ英語のまま、重みは後ろに ×1.3 で添える。"""
    core, weight = chip_core(chunk)
    ja = lookup_ja(table, core)
    if not ja:
        return chunk
    return f"{ja} ×{weight}" if weight else ja


def load_translations(path: Path | None = None) -> dict[str, str]:
    try:
        data = json.loads((path or TRANSLATIONS_PATH).read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in data.items() if isinstance(v, str) and v.strip()} if isinstance(data, dict) else {}


# 同梱の辞書: タガーの一般タグ 8,106 個を Index-Translate-2B で訳したもの (tools/make_tag_dict.py)
BUNDLED_DICT_PATH = resource_path("assets/tags_ja.json")
# 利用者が置く既存の辞書 (Tag Autocomplete の翻訳 CSV の形: 1 列目がタグ、2 列目が訳)。同梱はしない (ライセンスは置く人の判断)
USER_DICT_DIR = APP_DIR / "dict"


def load_tag_dictionary(bundled: Path | None = None, folder: Path | None = None) -> dict[str, str]:
    """チップの日本語訳に使う辞書: 同梱の辞書の上に、dict フォルダの CSV を重ねる (CSV が優先)。鍵は chip_core の小文字。"""
    import csv

    table: dict[str, str] = {}
    try:
        data = json.loads((bundled or BUNDLED_DICT_PATH).read_text("utf-8"))
        if isinstance(data, dict):
            table.update({str(k): str(v) for k, v in data.items() if isinstance(v, str) and v.strip()})
    except (OSError, ValueError):
        pass
    folder = folder or USER_DICT_DIR
    for path in sorted(folder.glob("*.csv")) if folder.is_dir() else []:
        try:
            with open(path, encoding="utf-8-sig", newline="") as f:
                for row in csv.reader(f):
                    if len(row) >= 2 and row[0].strip() and row[1].strip():
                        key = chip_core(sd_tag(row[0].strip()))[0].lower()
                        if key:
                            table[key] = row[1].strip()
        except (OSError, UnicodeDecodeError, csv.Error):
            continue
    return table


def save_translations(table: dict[str, str], path: Path | None = None) -> bool:
    try:
        (path or TRANSLATIONS_PATH).write_text(json.dumps(table, ensure_ascii=False, indent=1, sort_keys=True) + "\n", "utf-8")
        return True
    except OSError:
        return False


# 翻訳専用モデル Index-Translate (IndexTeam、Apache-2.0)。モデルカードの書き方: システムプロンプト無し、
# 「请将以下文本翻译为{target-language}，直接输出翻译结果，不要进行任何解释。」+ 空行 + 原文、enable_thinking: false、temperature 0。
# JSON の指示には従わず英語を写すだけだった (2026-10-10 実機)。1 行に 1 つ並べると行ごとに訳を返す (40 個 5 秒、数も一致)
INDEX_TRANSLATE_PROMPT = "请将以下文本翻译为日语，直接输出翻译结果，不要进行任何解释。\n\n"


# 翻訳専用モデル PLaMo 3 Translate (Preferred Networks、PLaMo Community License: 個人・非商用は登録なしで無料)。
# 渡し方は配布元の chat_template.jinja が作る PFML3 そのもの (モデルカードも本番では PFML3 を直接組むよう勧めている)。
# Ollama は GGUF のテンプレートを使わないので /api/generate の raw で渡す。先頭の <|plamo:bos|> はトークナイザーが付ける
# (GGUF の tokenizer.ggml.add_bos_token = true)。止めるのは <|plamo:tag|> / <|plamo:bos|>、greedy (temperature 0)
PLAMO_STOP = ["<|plamo:tag|>", "<|plamo:bos|>"]
# 「訳のモデル」の候補 (取得済みのモデルに加えて出す)。PLaMo は 2B (約 1.7 GB) と 8B (約 5.3 GB)
TRANSLATE_CHOICES = [
    "hf.co/IndexTeam/Index-Translate-2B-GGUF:Q4_K_M",
    "hf.co/pfnet/plamo-3-translate-2b-GGUF:Q4_K_M",
    "hf.co/pfnet/plamo-3-translate-8b-GGUF:Q4_K_M",
]


def is_plamo_translate(name: str) -> bool:
    return "plamo" in name.lower() and "translate" in name.lower()


def is_translation_model(name: str) -> bool:
    """訳すことしかできないモデル (指示に従って JSON で答える用途には使えない)。"""
    return "index-translate" in name.lower() or is_plamo_translate(name)


# 訳す前に挟む「状況の説明」(PFML3 の context 要素。配布元の説明では、原文を解釈するための参考の文)。
# 一般向けの翻訳モデルなので、何も渡さないと Danbooru タグを直訳したり、訳さずに写したりする。
# タグ・短い語句と、英語の文で内容を伝えるモデル (Krea 2・MOSS-Image・Qwen-Image など) の文とで分ける
PLAMO_CONTEXT = {
    ("ja", "tags"): "画像生成 AI のプロンプトに並べる英語のタグや短い語句を、1 行に 1 つずつ日本語にします。"
                    "多くはアニメ絵の Danbooru タグで、服装・髪・体の部位・ポーズ・表情・画角・背景・画質を表す語です。"
                    "タグとしての意味で、短い日本語にします。",
    ("ja", "prose"): "画像生成 AI に、作りたい画像の内容を説明する英語の文です。"
                     "人物・服装・場面・光・撮り方が伝わる、自然な日本語にします。",
    ("en", "tags"): "画像生成 AI のプロンプトに書く語です。Danbooru タグのような、短い英語の語句にします。",
    ("en", "prose"): "画像生成 AI に、作りたい画像の内容を英語の文で伝えるためのものです。自然な英語の文にします。",
}


def looks_like_sentence(text: str) -> bool:
    """タグ・短い語句ではなく文 (英語は 6 語以上か文の句読点、日本語は句点か 25 文字以上)。"""
    t = text.strip()
    return len(t.split()) >= 6 or bool(re.search(r"[.!?。！？]", t)) or (" " not in t and len(t) >= 25)


def plamo_prompt(text: str, lang: str = "ja", context: str | None = None) -> str:
    """PFML3 の翻訳依頼 (chat_template.jinja の出力と同じ。原文に予約の <|plamo:...|> があれば崩して渡す)。
    context は dataset の直後に入れる状況の説明。"""
    text = text.replace("<|plamo:", "<| plamo:")
    ctx = "<|plamo:tag|>context<|plamo:msg|>" + context.replace("<|plamo:", "<| plamo:") if context else ""
    return ("<|plamo:tag|>dataset<|plamo:msg|>translation" + ctx + "<|plamo:tag|>input<|plamo:msg|>" + text
            + "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>" + lang + "<|plamo:msg|>")


def _translate_plain(vlm: "OllamaVLM", text: str, keep_alive: int | str | None, context: str | None = None) -> str:
    if is_plamo_translate(vlm.model):
        req: dict[str, Any] = {
            "model": vlm.model, "stream": False, "raw": True, "prompt": plamo_prompt(text, "ja", context),
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 64 + 40 * (text.count("\n") + 1) + len(text),
                        "stop": PLAMO_STOP},
        }
        if keep_alive is not None:
            req["keep_alive"] = keep_alive
        data = vlm._http("/api/generate", req, abortable=True)
        return str(data.get("response") or "").strip()
    payload: dict[str, Any] = {
        "model": vlm.model, "stream": False, "think": False,
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 64 + 40 * (text.count("\n") + 1)},
        "messages": [{"role": "user", "content": INDEX_TRANSLATE_PROMPT + text}],
    }
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    data = vlm._chat(payload)
    return re.sub(r"(?s)^\s*<think>.*?</think>", "", str((data.get("message") or {}).get("content") or "")).strip()


def translate_chips_ja(vlm: "OllamaVLM", words: list[str], keep_alive: int | str | None = None,
                       checkpoint: Callable[[], None] | None = None) -> dict[str, str]:
    """英語の塊 (タグ・短い句) を Ollama で日本語に訳す → {小文字の英語: 日本語}。訳せなかった語は入れない。
    結び付けは並び順 (スキーマで個数を固定して、抜けや増えを起こさない)。"""
    out: dict[str, str] = {}
    words = dedup([w for w in words if w])
    if is_translation_model(vlm.model):
        plamo = is_plamo_translate(vlm.model)
        tag_ctx = PLAMO_CONTEXT[("ja", "tags")] if plamo else None
        prose = [w for w in words if plamo and looks_like_sentence(w)]   # 文は 1 つずつ (並べると混ざる)
        tags = [w for w in words if w not in prose]
        got: dict[str, str] = {}
        alone: set[str] = set()
        for start in range(0, len(tags), TRANSLATE_BATCH):
            if checkpoint is not None:
                checkpoint()  # 一時停止・中止はまとめて訳す単位の切れ目で
            batch = tags[start:start + TRANSLATE_BATCH]
            lines = [x.strip() for x in _translate_plain(vlm, "\n".join(batch), keep_alive, tag_ctx).split("\n") if x.strip()]
            if len(lines) != len(batch):  # 行の数が合わなければ並び順で結び付けられないので、1 つずつ訳し直す
                lines = [_translate_plain(vlm, w, keep_alive, tag_ctx).split("\n")[0].strip() for w in batch]
                alone.update(batch)
            got.update(zip(batch, lines, strict=True))
        if plamo:  # 並べて渡すと訳さずに写すことがあるので、写しただけの語は 1 つだけで渡し直す
            for w in [w for w in tags if w not in alone and (got.get(w) or "").lower() == w.lower()]:
                if checkpoint is not None:
                    checkpoint()
                got[w] = _translate_plain(vlm, w, keep_alive, tag_ctx).split("\n")[0].strip()
        for w in prose:
            if checkpoint is not None:
                checkpoint()
            got[w] = "".join(x.strip() for x in _translate_plain(vlm, w, keep_alive, PLAMO_CONTEXT[("ja", "prose")]).split("\n") if x.strip())
        for word, ja in got.items():
            if ja and ja.lower() != word.lower():  # 訳さずに写しただけ (1girl など) は入れない (英語のまま表示)
                out[word.lower()] = ja
        return out
    thinking = "thinking" in vlm.capabilities()
    for start in range(0, len(words), TRANSLATE_BATCH):
        if checkpoint is not None:
            checkpoint()
        batch = words[start:start + TRANSLATE_BATCH]
        payload: dict[str, Any] = {
            "model": vlm.model,
            "stream": False,
            "format": {"type": "object", "required": ["ja"], "properties": {"ja": {
                "type": "array", "minItems": len(batch), "maxItems": len(batch),
                "items": {"type": "string", "maxLength": 200}}}},
            # 思考するモデル (qwen3-vl:8b) は think: false でも内部で考え、短い上限だと考えるだけで使い切る (2026-10-10 実機)
            "options": {"temperature": 0.1, "num_ctx": VLM_NUM_CTX,
                        "num_predict": (VLM_NUM_PREDICT_THINKING if thinking else 200) + 60 * len(batch)},
            "messages": [
                {"role": "system", "content":
                    "You translate image-generation prompt fragments into natural, concise Japanese so a Japanese user "
                    "can understand each one. Most items are Danbooru tags for anime illustrations: translate them with the "
                    "meaning they have as Danbooru tags (clothing, body parts, poses, camera framing, image-quality terms), "
                    "not the literal dictionary meaning. Other items are English phrases or sentences: translate them "
                    "naturally. Translate every item, keep the order, and return exactly one Japanese string per item. "
                    "Write character names and series titles in their usual Japanese form. Do not add explanations."},
                {"role": "user", "content": "Items:\n" + "\n".join(f"{i + 1}. {w}" for i, w in enumerate(batch))},
            ],
        }
        if thinking:
            payload["think"] = False
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        data = vlm._chat(payload)
        content = re.sub(r"(?s)^\s*<think>.*?</think>", "", str((data.get("message") or {}).get("content") or "")).strip()
        try:
            obj = json.loads(content)
        except json.JSONDecodeError as e:
            if data.get("done_reason") == "length":
                raise OllamaError(
                    f"{vlm.model} の出力が上限に達して途中で切れました (日本語訳)。"
                    + ("思考しないモデル (qwen3-vl:8b-instruct / qwen2.5vl:7b など) を使うと確実です。" if thinking else "")
                ) from e
            raise OllamaError(f"日本語訳の応答を JSON として読めませんでした: {content[:200]}") from e
        items = obj.get("ja") if isinstance(obj, dict) else None
        if not isinstance(items, list) or len(items) != len(batch):
            raise OllamaError("日本語訳の応答が想定した形式ではありません (数が合いません)。")
        for word, ja in zip(batch, items):
            text = llm_text(ja)
            if text:
                out[word.lower()] = text
    return out


# --------------------------------------------------------------------------
# 履歴 (解析結果を 50 件まで。結果 + 128px のサムネイル、exe の隣の history\)
# --------------------------------------------------------------------------

HISTORY_DIR = APP_DIR / "history"
HISTORY_LIMIT = 50
HISTORY_THUMB = 128


def history_add(result: PromptResult, thumb: Image.Image | None, extra: dict[str, Any] | None = None,
                folder: Path | None = None) -> Path | None:
    """結果を履歴に足し、古いものから HISTORY_LIMIT 件を超えた分を消す。書き込めなければ None (動作は続ける)。"""
    folder = folder or HISTORY_DIR
    now = time.time()
    stem = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + f"-{int(now * 1000) % 1000:03d}"
    data = asdict(result)
    data["_history"] = {"saved_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)), **(extra or {})}
    try:
        folder.mkdir(parents=True, exist_ok=True)
        n = 1
        while (folder / f"{stem}.json").exists():  # 同じミリ秒に 2 件
            stem, n = f"{stem.split('_')[0]}_{n}", n + 1
        path = folder / f"{stem}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", "utf-8")
        if thumb is not None:
            small = thumb.convert("RGB")
            small.thumbnail((HISTORY_THUMB, HISTORY_THUMB), Image.LANCZOS)
            small.save(folder / f"{stem}.jpg", quality=85)
    except OSError:
        return None
    for old in sorted(folder.glob("*.json"))[:-HISTORY_LIMIT]:
        history_delete(old.stem, folder)
    return path


def history_entries(folder: Path | None = None) -> list[dict[str, Any]]:
    """履歴の一覧 (新しい順)。読めないファイルは飛ばす。"""
    folder = folder or HISTORY_DIR
    entries: list[dict[str, Any]] = []
    for path in sorted(folder.glob("*.json"), reverse=True) if folder.is_dir() else []:
        try:
            data = json.loads(path.read_text("utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        meta = data.get("_history") if isinstance(data.get("_history"), dict) else {}
        thumb = path.with_suffix(".jpg")
        entries.append({
            "id": path.stem, "path": path, "thumb": thumb if thumb.is_file() else None,
            "saved_at": str(meta.get("saved_at") or path.stem), "image": str(data.get("image") or ""),
            "target": str(data.get("target") or ""), "positive": str(data.get("positive_prompt") or ""),
            "references": data.get("references") if isinstance(data.get("references"), dict) else {},
        })
    return entries


def history_result(path: Path) -> tuple[PromptResult, dict[str, Any]]:
    """履歴のファイル → (結果, 付帯情報)。知らない欄は無視する (版が違っても読めるように)。"""
    from dataclasses import fields

    data = json.loads(Path(path).read_text("utf-8"))
    meta = data.pop("_history", {}) if isinstance(data, dict) else {}
    names = {f.name for f in fields(PromptResult)}
    return PromptResult(**{k: v for k, v in data.items() if k in names}), meta if isinstance(meta, dict) else {}


def history_update_prompts(path: Path, positive: str, negative: str) -> bool:
    """チップで消した結果を履歴に反映する。書き込めなければ False (動作は続ける)。"""
    try:
        data = json.loads(Path(path).read_text("utf-8"))
        if not isinstance(data, dict):
            return False
        data["positive_prompt"], data["negative_prompt"] = positive, negative
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", "utf-8")
        return True
    except (OSError, ValueError):
        return False


def history_delete(stem: str, folder: Path | None = None) -> None:
    folder = folder or HISTORY_DIR
    for suffix in (".json", ".jpg"):
        try:
            (folder / f"{stem}{suffix}").unlink(missing_ok=True)
        except OSError:
            pass


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def attach_parent_console() -> None:
    """GUI サブシステムの exe を端末から CLI として使ったとき、呼び出し元のコンソールに出力する。"""
    if not (FROZEN and sys.platform == "win32") or sys.stdout is not None:
        return
    try:
        import ctypes

        if ctypes.windll.kernel32.AttachConsole(-1):  # ATTACH_PARENT_PROCESS
            sys.stdout = open("CONOUT$", "w", encoding="utf-8", errors="replace", buffering=1)  # noqa: SIM115
            sys.stderr = open("CONOUT$", "w", encoding="utf-8", errors="replace", buffering=1)  # noqa: SIM115
    except Exception:  # noqa: BLE001 - コンソールが無ければ黙って諦める (--out で結果は保存できる)
        pass


def cli_controls(args: argparse.Namespace, engine: "Engine", log: Callable[[str], None]) -> dict[str, Any]:
    """CLI の構図と強さ (--bones-from / --camera-* / --strength) を Engine.run の controls にする。
    --bones-from は骨格モデル (RTMPose-s 約 21MB) が無ければ取得する (オプションを付けたことを同意とみなす)。"""
    controls: dict[str, Any] = {}
    if args.bones_from:
        src = Path(args.bones_from)
        if not src.is_file():
            raise ValueError(f"骨格を取る画像が見つかりません: {args.bones_from}")
        with Image.open(src) as im:
            image = upright(im).convert("RGB")
        log(f"[{APP_NAME}] 骨格を推定中: {src.name}")
        estimator = PoseEstimator(progress=lambda m: log(f"[{APP_NAME}] {m}"))
        box = main_person_box(engine.person_detector().detect(image))
        points = estimator.estimate(image, box)
        tags = pose_tags_from_points(points)
        hidden = [KEYPOINTS[i][1] for i, pt in enumerate(points) if pt[2] < POSE_MIN_CONF]
        log(f"[{APP_NAME}] ボーンのポーズ: {', '.join(sd_tag(t) for t in tags) or '(タグにできる姿勢なし)'}"
            + (f" (見えていない関節: {'、'.join(hidden)})" if hidden else ""))
        controls["pose"] = {"tags": tags}
        if args.openpose_out:
            render_openpose(points, image.size).save(args.openpose_out)
            log(f"[{APP_NAME}] OpenPose 画像を保存しました: {args.openpose_out}")
    elif args.openpose_out:
        raise ValueError("--openpose-out は --bones-from と一緒に指定してください")
    camera = {dim: getattr(args, f"camera_{dim}") for dim in CAMERA_OPTIONS if getattr(args, f"camera_{dim}", None)}
    if camera:
        controls["camera"] = camera
    if getattr(args, "look", None):
        look = find_look(engine.cfg, args.look)
        if look is None:
            raise ValueError(f"撮影プリセット「{args.look}」がありません (config.json の camera_looks: "
                             + " / ".join(x["name"] for x in engine.cfg.get("camera_looks") or []) + ")")
        controls["look"] = look
    if args.strength:
        groups = {g for g, _ in STRENGTH_GROUPS}
        strength: dict[str, str] = {}
        for item in args.strength.split(","):
            group, _, value = item.strip().partition("=")
            if group not in groups or value not in ("strong", "weak", "normal"):
                raise ValueError(f"--strength の指定が正しくありません: {item} (例: outfit=strong,background=weak。"
                                 f"枠は {' / '.join(g for g, _ in STRENGTH_GROUPS)}、強さは strong / normal / weak)")
            if value != "normal":
                strength[group] = value
        if strength:
            controls["strength"] = strength
    return controls


def run_cli(args: argparse.Namespace) -> int:
    attach_parent_console()
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    cfg = load_config()
    if args.engine:
        cfg["engine"] = args.engine
    if args.tagger:
        cfg["tagger_model"] = args.tagger
    if args.vlm:
        cfg["ollama_model"] = args.vlm
    if args.describer:
        cfg["describer_model"] = args.describer
    if args.host:
        cfg["ollama_host"] = args.host
    # --style は ⑦ 作風・質感の画像。旧オプション (--style tags / natural = 出力先) と同じ名前なので、その値で
    # 同じ名前のファイルが無ければ旧オプションとして扱う (古い使い方を壊さない)
    legacy_style = args.style in LEGACY_STYLE_TO_TARGET and not Path(args.style).is_file()
    if args.target:
        cfg["target"] = args.target
    elif legacy_style:
        cfg["target"] = LEGACY_STYLE_TO_TARGET[args.style]
    if args.no_quality:
        cfg["quality_tags"] = False
    if args.no_break:
        cfg["use_break"] = False
    if args.mosaic:
        cfg["mosaic"] = True
    if args.mosaic_strength:
        cfg["mosaic_strength"] = args.mosaic_strength
    if args.split_extras:
        cfg["split_extras"] = True
    if args.split_background:
        cfg["split_background"] = True
    if args.no_polish:
        cfg["polish_prose"] = False
    if args.no_age_look:
        cfg["age_look"] = False
    if args.general_threshold is not None:
        cfg["general_threshold"] = args.general_threshold
    if args.character_threshold is not None:
        cfg["character_threshold"] = args.character_threshold
    if args.style_threshold is not None:
        cfg["style_threshold"] = args.style_threshold
    if args.no_era:
        cfg["era_tags"] = False
    if args.main_person:
        cfg["main_person_only"] = True
    if args.no_split_persons:
        cfg["split_persons"] = False
    if args.gender:
        cfg["gender"] = args.gender
    if args.restyle:
        cfg["restyle"] = args.restyle
    if args.no_anatomy_guard:
        cfg["anatomy_guard"] = False
    if args.color:
        cfg["remove_monochrome"] = True
    if args.avoid_same_face:
        cfg["avoid_same_face"] = True
    if args.avoid_duplicates:
        cfg["avoid_duplicates"] = True
    if args.person_details:
        cfg["person_details"] = True
    if args.turnaround:
        cfg["turnaround"] = True
    if args.no_identity:
        cfg["no_identity"] = True
    if args.no_style:
        cfg["no_style"] = True
    refs: dict[str, Path] = {}
    for slot, raw_ref in (("pose", args.pose), ("outfit", args.outfit), ("background", args.background), ("props", args.props),
                          ("body", args.body), ("style", None if legacy_style else args.style),
                          ("expression", args.expression)):
        if raw_ref:
            refs[slot] = Path(raw_ref)
            if not refs[slot].is_file():
                log(f"[error] {REF_SLOTS[slot]} の画像が見つかりません: {raw_ref}")
                return 1

    engine = Engine(cfg, progress=lambda m: log(f"[{APP_NAME}] {m}"))
    try:
        controls = cli_controls(args, engine, log)
    except ValueError as e:
        log(f"[error] {e}")
        return 1
    except Exception as e:  # noqa: BLE001 - 骨格モデルの取得・推定の失敗
        log(f"[error] 骨格を取れませんでした: {describe_error(e)}")
        return 1
    if cfg.get("person_details") and cfg.get("engine") == "hybrid" and not face_models_ready():
        # 顔の切り出しに使うモデル (骨格 21MB・人物検出 44MB)。--person-details を付けたことを同意とみなす (--bones-from と同じ)
        try:
            ensure_face_models(lambda m: log(f"[{APP_NAME}] {m}"))
        except Exception as e:  # noqa: BLE001 - 取れなくても人物全体の画像で読む
            log(f"[warn] 顔の切り出しに使うモデルを取得できませんでした (人物全体の画像だけで読みます): {describe_error(e)}")
    results: list[PromptResult] = []
    failures = 0
    saved: set[Path] = set()

    for raw in args.images:
        path = Path(raw)
        if not path.is_file():
            log(f"[skip] ファイルが見つかりません: {raw}")
            failures += 1
            continue
        ref_note = ", ".join(f"{REF_SLOTS[k]}: {v.name}" for k, v in refs.items())
        log(f"[{APP_NAME}] 解析中: {path.name}" + (f" ({ref_note})" if ref_note else ""))
        try:
            result = engine.run(path, refs, controls)
        except Exception as e:  # noqa: BLE001 - 1 枚失敗しても残りを続行する
            log(f"[error] {path.name}: {describe_error(e)}")
            failures += 1
            continue
        for w in result.warnings:
            log(f"[warn] {path.name}: {w}")
        if result.mosaic_prompt:
            log(f"[mosaic] {path.name}: NSFW 判定のためプロンプトにモザイク指定を追加")
        results.append(result)
        if not args.json:
            print(result.to_text())
        if args.save:
            suffix = "".join(f"_{k}-{v.stem}" for k, v in refs.items()) + "_prompt.txt"
            out = path.with_name(path.stem + suffix)
            if out in saved:  # a.png と a.jpg のように名前が同じ画像は拡張子も付けて上書きを防ぐ
                out = path.with_name(f"{path.stem}_{path.suffix.lstrip('.')}{suffix}")
            saved.add(out)
            out.write_text(result.to_text(), "utf-8")
            log(f"[saved] {out}")

    if args.json:
        print(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2))

    if args.out:
        if args.json:
            content = json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2) + "\n"
        else:
            content = "\n".join(r.to_text() for r in results)
        Path(args.out).write_text(content, "utf-8")
        log(f"[saved] {args.out}")

    return 1 if failures else 0


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Ollama 導入サポート (Windows)
# --------------------------------------------------------------------------

OLLAMA_INSTALLER_URL = "https://ollama.com/download/OllamaSetup.exe"
OLLAMA_INSTALLER_GB = 1.6  # 2026-09 時点で約 1.57GB (一時ファイル)
OLLAMA_APP_GB = 4.0  # 公式: バイナリの導入に 4GB 以上
# (モデル名, 容量 GB, 推奨 VRAM GB, 説明)
VLM_CATALOG: list[tuple[str, float, int, str]] = [
    ("qwen3-vl:2b-instruct", 1.9, 4, "軽量。CPU でも動く。精度は控えめ"),
    ("qwen3-vl:4b-instruct", 3.3, 6, "バランス型"),
    ("qwen3-vl:8b-instruct", 6.1, 8, "既定。12GB あれば余裕 (無印の qwen3-vl:8b は思考版で遅い)"),
    ("qwen3-vl:32b-instruct", 21.0, 24, "高精度。RTX 3090 / 4090 級"),
    # Qwen3.5 (画像も読める。MOSS-Image のテキストエンコーダーと同じ系統)。
    # 大きさは ollama.com/library/qwen3.5 の表示。思考は他と同じく think: false で切る (効くかは未確認)
    ("qwen3.5:4b", 3.4, 6, "Qwen3.5 の軽量版 (画像も読める)"),
    ("qwen3.5:9b", 6.6, 10, "Qwen3.5。MOSS-Image のテキストエンコーダーと同じ系統"),
    ("qwen3.5:27b", 17.0, 24, "Qwen3.5 の高精度版。RTX 3090 / 4090 級"),
    # Gemma 4 (Google。全サイズが画像を読める)。大きさは ollama.com/library/gemma4 の表示。26b は MoE (動くのは約 4B 分)
    ("gemma4:e4b", 6.6, 8, "Gemma 4 の軽量版"),
    ("gemma4:12b", 7.7, 10, "Gemma 4"),
    ("gemma4:26b", 16.0, 20, "Gemma 4 の MoE 版。大きさの割に速い"),
    ("gemma4:31b", 19.0, 24, "Gemma 4 の高精度版。RTX 3090 / 4090 級"),
    # Gemma 4 の検閲なし版 (huihui_ai、abliteration で答えを断る働きを外したもの。画像を見る力は元と同じ)。
    # 大きさは ollama.com/huihui_ai/gemma-4-abliterated/tags の表示 (e4b は公式より大きい)。31b は作り直した v2
    ("huihui_ai/gemma-4-abliterated:e4b", 9.6, 12, "Gemma 4 の検閲なし版 (軽量)"),
    ("huihui_ai/gemma-4-abliterated:12b", 7.6, 10, "Gemma 4 の検閲なし版"),
    ("huihui_ai/gemma-4-abliterated:26b", 18.0, 22, "Gemma 4 の検閲なし版 (MoE)"),
    ("huihui_ai/gemma-4-abliterated:31b-v2", 20.0, 24, "Gemma 4 の検閲なし版 (高精度)。RTX 3090 / 4090 級"),
    # Qwen3.6 / Qwen3.8 (画像も読める。27b 以上しかない)。大きさは ollama.com/library/qwen3.6・qwen3.8 の表示
    ("qwen3.6:27b", 17.0, 24, "Qwen3.6。RTX 3090 / 4090 級"),
    ("qwen3.6:35b-a3b", 23.0, 24, "Qwen3.6 の MoE 版 (動くのは約 3B 分)。大きさの割に速い"),
    ("qwen3.8:27b", 18.0, 24, "Qwen3.8。RTX 3090 / 4090 級"),
    # Qwen の検閲なし版 (huihui_ai、abliteration)。大きさは ollama.com/huihui_ai/... の表示。
    # 名前の付いた派生版 (9b-Claude など、別の目的で追加学習したもの) は入れない
    ("huihui_ai/qwen3-vl-abliterated:8b-instruct", 6.1, 8, "Qwen3-VL の検閲なし版 (2025 年のまま更新なし)"),
    ("huihui_ai/qwen3.5-abliterated:4B", 3.3, 6, "Qwen3.5 の検閲なし版 (軽量)"),
    ("huihui_ai/qwen3.5-abliterated:9b", 6.6, 10, "Qwen3.5 の検閲なし版"),
    ("huihui_ai/qwen3.5-abliterated:27b", 17.0, 24, "Qwen3.5 の検閲なし版 (高精度)。RTX 3090 / 4090 級"),
    ("huihui_ai/qwen3.6-abliterated:27b", 17.0, 24, "Qwen3.6 の検閲なし版。RTX 3090 / 4090 級"),
    ("huihui_ai/qwen3.6-abliterated:35b-a3b", 24.0, 24, "Qwen3.6 の検閲なし版 (MoE)"),
    ("huihui_ai/qwen3.8-abliterated:27b-q3_K", 14.0, 16, "Qwen3.8 の検閲なし版を強く縮めたもの (16GB 向け。精度は少し落ちる)"),
    ("huihui_ai/qwen3.8-abliterated:27b-q4_K_M", 18.0, 24, "Qwen3.8 の検閲なし版。RTX 3090 / 4090 級"),
]
VLM_SIZES = {name: size for name, size, _, _ in VLM_CATALOG}


def _no_window_flags() -> int:
    if sys.platform != "win32":
        return 0
    import subprocess

    return subprocess.CREATE_NO_WINDOW


def _hidden_run(args: list[str], timeout: int = 10) -> str:
    import subprocess

    try:
        out = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, creationflags=_no_window_flags(),
            encoding="utf-8", errors="replace",
        )
        return (out.stdout or "") + (out.stderr or "")
    except (OSError, subprocess.SubprocessError):
        return ""


def find_ollama_exe() -> Path | None:
    """PATH → レジストリのアンインストール情報 → 既定の導入先の順に ollama.exe を探す。"""
    import shutil

    found = shutil.which("ollama")
    if found:
        return Path(found)
    if sys.platform == "win32":
        try:
            import winreg

            for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
                    try:
                        root = winreg.OpenKey(
                            hive, r"Software\Microsoft\Windows\CurrentVersion\Uninstall", 0, winreg.KEY_READ | view
                        )
                    except OSError:
                        continue
                    with root:
                        for i in range(winreg.QueryInfoKey(root)[0]):
                            try:
                                name = winreg.EnumKey(root, i)
                                with winreg.OpenKey(root, name) as sub:
                                    display = str(winreg.QueryValueEx(sub, "DisplayName")[0])
                                    if not display.lower().startswith("ollama"):
                                        continue
                                    location = str(winreg.QueryValueEx(sub, "InstallLocation")[0])
                            except OSError:
                                continue
                            exe = Path(location) / "ollama.exe"
                            if exe.is_file():
                                return exe
        except ImportError:
            pass
        default = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        if default.is_file():
            return default
    return None


def default_ollama_install_dir() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Programs" / "Ollama"


def ollama_models_dir() -> Path:
    """OLLAMA_MODELS (プロセス環境 → ユーザー環境変数) か既定の ~/.ollama/models。"""
    env = os.environ.get("OLLAMA_MODELS")
    if not env and sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                env = str(winreg.QueryValueEx(key, "OLLAMA_MODELS")[0])
        except OSError:
            env = ""
    return Path(env) if env else Path.home() / ".ollama" / "models"


def ollama_version(exe: Path) -> str:
    text = _hidden_run([str(exe), "--version"], timeout=10)
    m = re.search(r"(\d+\.\d+\.\d+)", text)
    return m.group(1) if m else ""


def free_gb(path: Path) -> float:
    import shutil

    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    try:
        return shutil.disk_usage(str(p)).free / 1e9
    except OSError:
        return 0.0


def system_specs() -> dict[str, Any]:
    """GPU 名 / VRAM / メモリを調べる (NVIDIA は nvidia-smi、それ以外は名前だけ)。"""
    specs: dict[str, Any] = {"gpu": "", "vram_gb": None, "ram_gb": 0.0}
    text = _hidden_run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], timeout=8)
    line = next((ln.strip() for ln in text.splitlines() if "," in ln), "")
    if line:
        name, _, mem = line.rpartition(",")
        specs["gpu"] = name.strip()
        try:
            specs["vram_gb"] = round(float(mem.strip()) / 1024, 1)
        except ValueError:
            pass
    elif sys.platform == "win32":
        out = _hidden_run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join '; '"],
            timeout=15,
        )
        specs["gpu"] = out.strip()
    if sys.platform == "win32":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                specs["ram_gb"] = round(status.ullTotalPhys / 1e9, 1)
        except Exception:  # noqa: BLE001
            pass
    return specs


def model_fit(vram_gb: float | None, ram_gb: float, need_vram: int, size_gb: float) -> str:
    """PC スペックに対するモデルの適合度: 推奨 / 可 / 遅い (CPU) / 不可"""
    if vram_gb is not None and vram_gb >= need_vram:
        return "可"
    if ram_gb and ram_gb >= size_gb + 6:
        return "遅い (CPU)"
    return "不可"


def recommend_vlm(vram_gb: float | None, ram_gb: float) -> str:
    """VRAM / メモリに合わせた推奨モデル。速度と精度のバランスで 8b までを推奨し、32b は 24GB 以上でも「可」扱い。"""
    if vram_gb is None or vram_gb < 6:
        if vram_gb is not None and vram_gb >= 4:
            return "qwen3-vl:2b-instruct"
        return "qwen3-vl:4b-instruct" if ram_gb >= 16 else "qwen3-vl:2b-instruct"
    if vram_gb < 8:
        return "qwen3-vl:4b-instruct"
    return "qwen3-vl:8b-instruct"


def start_ollama(exe: Path) -> None:
    """トレイアプリ (ollama app.exe) があればそれを、無ければ ollama serve をバックグラウンドで起動する。"""
    import subprocess

    flags = _no_window_flags()
    if sys.platform == "win32":
        flags |= subprocess.DETACHED_PROCESS
    app = exe.parent / "ollama app.exe"
    if app.is_file():
        subprocess.Popen([str(app)], creationflags=flags, close_fds=True)
    else:
        subprocess.Popen(
            [str(exe), "serve"], creationflags=flags, close_fds=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


def wait_ollama(host: str, timeout: int = 60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            OllamaVLM(host, "").list_models()
            return True
        except OllamaError:
            time.sleep(1.0)
    return False


def set_ollama_models_dir(path: Path) -> None:
    """OLLAMA_MODELS をユーザー環境変数と現在のプロセスに設定する (Ollama の再起動後に有効)。"""
    import subprocess

    path.mkdir(parents=True, exist_ok=True)
    os.environ["OLLAMA_MODELS"] = str(path)
    if sys.platform == "win32":
        subprocess.run(["setx", "OLLAMA_MODELS", str(path)], capture_output=True, creationflags=_no_window_flags(), timeout=20)


OLLAMA_SIGNER = "Ollama Inc."  # 公式インストーラ / ollama.exe の署名者 (2026-09 に確認)


def authenticode_signer(path: Path) -> str:
    """Windows の署名検証 (WinVerifyTrust) で exe の署名と証明書チェーンを確かめ、署名者名を返す。
    署名が無い・改ざんされている・信頼できない場合は OllamaError。"""
    import ctypes
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                    ("Data4", ctypes.c_ubyte * 8)]

    class WintrustFileInfo(ctypes.Structure):
        _fields_ = [("cbStruct", wintypes.DWORD), ("pcwszFilePath", wintypes.LPCWSTR), ("hFile", wintypes.HANDLE),
                    ("pgKnownSubject", ctypes.POINTER(GUID))]

    class WintrustData(ctypes.Structure):
        _fields_ = [("cbStruct", wintypes.DWORD), ("pPolicyCallbackData", ctypes.c_void_p),
                    ("pSIPClientData", ctypes.c_void_p), ("dwUIChoice", wintypes.DWORD),
                    ("fdwRevocationChecks", wintypes.DWORD), ("dwUnionChoice", wintypes.DWORD),
                    ("pFile", ctypes.POINTER(WintrustFileInfo)), ("dwStateAction", wintypes.DWORD),
                    ("hWVTStateData", wintypes.HANDLE), ("pwszURLReference", wintypes.LPWSTR),
                    ("dwProvFlags", wintypes.DWORD), ("dwUIContext", wintypes.DWORD),
                    ("pSignatureSettings", ctypes.c_void_p)]

    class ProviderSigner(ctypes.Structure):  # CRYPT_PROVIDER_SGNR の先頭部分
        _fields_ = [("cbStruct", wintypes.DWORD), ("sftVerifyAsOf", wintypes.FILETIME),
                    ("csCertChain", wintypes.DWORD), ("pasCertChain", ctypes.c_void_p)]

    class ProviderCert(ctypes.Structure):  # CRYPT_PROVIDER_CERT の先頭部分
        _fields_ = [("cbStruct", wintypes.DWORD), ("pCert", ctypes.c_void_p)]

    # WINTRUST_ACTION_GENERIC_VERIFY_V2
    action = GUID(0x00AAC56B, 0xCD44, 0x11D0, (ctypes.c_ubyte * 8)(0x8C, 0xC2, 0x00, 0xC0, 0x4F, 0xC2, 0x95, 0xEE))
    file_info = WintrustFileInfo(ctypes.sizeof(WintrustFileInfo), str(path), None, None)
    data = WintrustData()
    data.cbStruct = ctypes.sizeof(WintrustData)
    data.dwUIChoice = 2  # WTD_UI_NONE
    data.dwUnionChoice = 1  # WTD_CHOICE_FILE
    data.pFile = ctypes.pointer(file_info)
    data.dwStateAction = 1  # WTD_STATEACTION_VERIFY
    wintrust = ctypes.WinDLL("wintrust")
    wintrust.WinVerifyTrust.argtypes = [wintypes.HWND, ctypes.POINTER(GUID), ctypes.c_void_p]
    wintrust.WinVerifyTrust.restype = wintypes.LONG
    wintrust.WTHelperProvDataFromStateData.argtypes = [wintypes.HANDLE]
    wintrust.WTHelperProvDataFromStateData.restype = ctypes.c_void_p
    wintrust.WTHelperGetProvSignerFromChain.argtypes = [ctypes.c_void_p, wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    wintrust.WTHelperGetProvSignerFromChain.restype = ctypes.c_void_p
    status = wintrust.WinVerifyTrust(None, ctypes.byref(action), ctypes.byref(data))
    try:
        if status != 0:
            raise OllamaError(f"{path.name} の署名を確認できませんでした (0x{status & 0xFFFFFFFF:08X})。実行を中止しました。")
        provider = wintrust.WTHelperProvDataFromStateData(data.hWVTStateData)
        signer_ptr = wintrust.WTHelperGetProvSignerFromChain(provider, 0, False, 0) if provider else None
        if not signer_ptr:
            raise OllamaError(f"{path.name} の署名者を読み取れませんでした。実行を中止しました。")
        signer = ProviderSigner.from_address(signer_ptr)
        if not signer.csCertChain or not signer.pasCertChain:
            raise OllamaError(f"{path.name} の証明書を読み取れませんでした。実行を中止しました。")
        cert = ProviderCert.from_address(signer.pasCertChain).pCert
        crypt32 = ctypes.WinDLL("crypt32")
        crypt32.CertGetNameStringW.argtypes = [ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                               wintypes.LPWSTR, wintypes.DWORD]
        crypt32.CertGetNameStringW.restype = wintypes.DWORD
        name = ctypes.create_unicode_buffer(256)
        crypt32.CertGetNameStringW(cert, 4, 0, None, name, 256)  # CERT_NAME_SIMPLE_DISPLAY_TYPE
        return name.value
    finally:
        data.dwStateAction = 2  # WTD_STATEACTION_CLOSE
        wintrust.WinVerifyTrust(None, ctypes.byref(action), ctypes.byref(data))


def install_ollama(installer: Path, install_dir: Path, progress: ProgressFn | None = None) -> Path:
    """公式インストーラをサイレント実行する (Inno Setup: /SILENT /DIR=)。管理者権限は不要。
    実行前に署名を確かめ、Ollama Inc. の有効な署名でなければ実行しない (途中で差し替えられたファイル対策)。"""
    import subprocess

    signer = authenticode_signer(installer)
    if signer != OLLAMA_SIGNER:
        raise OllamaError(f"インストーラの署名者が想定と違います ({signer or '不明'})。実行を中止しました。")

    if progress:
        progress("Ollama をインストール中... (数分かかります)")
    result = subprocess.run(
        [str(installer), "/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART", f"/DIR={install_dir}"],
        timeout=1800,
    )
    if result.returncode != 0:
        raise OllamaError(f"インストーラが終了コード {result.returncode} で終了しました。")
    exe = install_dir / "ollama.exe"
    if not exe.is_file():
        found = find_ollama_exe()
        if not found:
            raise OllamaError("インストール後に ollama.exe が見つかりません。")
        exe = found
    return exe


def gather_ollama_info(host: str, model: str) -> dict[str, Any]:
    """導入サポート画面に出す情報をまとめて集める (バックグラウンドで呼ぶ)。"""
    info: dict[str, Any] = {"reachable": False, "models": [], "vision": [], "exe": None, "version": "", "specs": {}}
    try:
        client = OllamaVLM(host, model)
        info["models"] = client.list_models()
        info["reachable"] = True
        for name in info["models"]:
            try:
                if "vision" in OllamaVLM(host, name).capabilities():
                    info["vision"].append(name)
            except OllamaError:
                continue
    except OllamaError:
        pass
    exe = find_ollama_exe()
    info["exe"] = exe
    if exe:
        info["version"] = ollama_version(exe)
    info["specs"] = system_specs()
    info["models_dir"] = ollama_models_dir()
    return info


class OllamaSetupDialog:
    """Ollama の状態・PC スペック・保存先・必要容量を見せ、導入 / 起動 / 接続と使うモデルを選ばせる。"""

    def __init__(self, parent: "tk.Tk", cfg: dict[str, Any], info: dict[str, Any]):
        self.result: dict[str, Any] | None = None
        self.info = info
        specs = info.get("specs") or {}
        vram = specs.get("vram_gb")
        ram = float(specs.get("ram_gb") or 0)
        reachable = bool(info.get("reachable"))
        exe = info.get("exe")
        installed = reachable or exe is not None
        pulled = set(info.get("models") or [])

        win = self.win = tk.Toplevel(parent)
        win.title(f"{APP_NAME} - Ollama 導入サポート")
        win.transient(parent)
        win.resizable(False, False)
        body = ttk.Frame(win, padding=14)
        body.pack(fill="both", expand=True)

        if reachable:
            state = f"接続済み: {cfg.get('ollama_host')}  (取得済みモデル {len(pulled)} 件)"
        elif exe is not None:
            state = f"導入済み・未起動: {exe}" + (f"  (v{info.get('version')})" if info.get("version") else "")
        else:
            state = "未導入 (ollama.exe が見つかりません)"
        ttk.Label(body, text=f"状態: {state}", wraplength=620, justify="left").pack(anchor="w")
        gpu = specs.get("gpu") or "不明"
        vram_text = f"{vram:.1f} GB" if vram is not None else "不明 (NVIDIA 以外または未検出)"
        ttk.Label(
            body, text=f"PC: GPU {gpu} / VRAM {vram_text} / メモリ {ram:.0f} GB", wraplength=620, justify="left",
        ).pack(anchor="w", pady=(2, 8))

        grid = ttk.Frame(body)
        grid.pack(fill="x")
        grid.columnconfigure(1, weight=1)
        row = 0
        self.install_var = tk.StringVar(value=str(default_ollama_install_dir()))
        if not installed:
            ttk.Label(grid, text="インストール先").grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
            ttk.Entry(grid, textvariable=self.install_var, width=52).grid(row=row, column=1, sticky="we", pady=2)
            ttk.Button(grid, text="参照...", width=7, command=lambda: self._browse(self.install_var)).grid(row=row, column=2, padx=(4, 0))
            row += 1
        ttk.Label(grid, text="モデル保存先").grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
        self.models_var = tk.StringVar(value=str(info.get("models_dir") or ollama_models_dir()))
        ttk.Entry(grid, textvariable=self.models_var, width=52).grid(row=row, column=1, sticky="we", pady=2)
        ttk.Button(grid, text="参照...", width=7, command=lambda: self._browse(self.models_var)).grid(row=row, column=2, padx=(4, 0))
        row += 1
        self.space_var = tk.StringVar()
        ttk.Label(grid, textvariable=self.space_var, foreground="#666", wraplength=620, justify="left").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(2, 6)
        )

        ttk.Label(body, text="使うモデル (PC のスペックから判定):").pack(anchor="w")
        recommended = recommend_vlm(vram, ram)
        pulled_vision = [m for m in (info.get("vision") or []) if m in VLM_SIZES]
        default_model = cfg.get("ollama_model") if cfg.get("ollama_model") in pulled else (pulled_vision[-1] if pulled_vision else recommended)
        self.model_var = tk.StringVar(value=default_model)
        self.custom_var = tk.StringVar(value="" if default_model in VLM_SIZES else str(default_model))
        for name, size, need, desc in VLM_CATALOG:
            fit = model_fit(vram, ram, need, size)
            mark = "推奨" if name == recommended and fit == "可" else fit
            state_txt = "取得済み" if name in pulled else f"{size:.1f} GB"
            ttk.Radiobutton(
                body, variable=self.model_var, value=name,
                text=f"{name:<14} {state_txt:>8}   VRAM {need}GB 以上   [{mark}]   {desc}",
            ).pack(anchor="w")
        other = ttk.Frame(body)
        other.pack(anchor="w", pady=(2, 0))
        ttk.Radiobutton(other, variable=self.model_var, value="__custom__", text="その他 (名前を入力)").pack(side="left")
        ttk.Entry(other, textvariable=self.custom_var, width=28).pack(side="left", padx=(4, 0))
        if default_model not in VLM_SIZES:
            self.model_var.set("__custom__")

        self.pull_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(body, text="続けて選択したモデルを取得する (未取得のとき)", variable=self.pull_var).pack(anchor="w", pady=(6, 0))
        self.need_var = tk.StringVar()
        ttk.Label(body, textvariable=self.need_var, wraplength=620, justify="left").pack(anchor="w", pady=(4, 8))

        buttons = ttk.Frame(body)
        buttons.pack(fill="x")
        if not installed:
            primary, action = "ダウンロードしてインストール", "install"
        elif not reachable:
            primary, action = "Ollama を起動して使う", "start"
        else:
            primary, action = "この設定で使う", "connect"
        self.action = action
        ttk.Button(buttons, text=primary, command=self._ok).pack(side="right")
        ttk.Button(buttons, text="閉じる", command=self._cancel).pack(side="right", padx=(0, 8))
        ttk.Label(
            buttons, foreground="#666", wraplength=380, justify="left",
            text="管理者権限は不要です。インストーラは公式サイト (ollama.com) から取得します。",
        ).pack(side="left")

        for var in (self.install_var, self.models_var, self.model_var, self.custom_var):
            var.trace_add("write", lambda *_: self._refresh())
        self._refresh()
        win.protocol("WM_DELETE_WINDOW", self._cancel)
        win.bind("<Escape>", lambda _e: self._cancel())
        win.update_idletasks()
        px, py = parent.winfo_rootx(), parent.winfo_rooty()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        win.geometry(f"+{px + max(0, (pw - win.winfo_width()) // 2)}+{py + max(0, (ph - win.winfo_height()) // 4)}")
        win.grab_set()
        win.focus_set()

    def _browse(self, var: "tk.StringVar") -> None:
        chosen = filedialog.askdirectory(parent=self.win, initialdir=var.get() or str(Path.home()))
        if chosen:
            var.set(chosen)

    def selected_model(self) -> str:
        if self.model_var.get() == "__custom__":
            return self.custom_var.get().strip()
        return self.model_var.get()

    def _refresh(self) -> None:
        model = self.selected_model()
        size = VLM_SIZES.get(model, 0.0)
        pulled = model in set(self.info.get("models") or [])
        parts = []
        if self.action == "install":
            parts.append(f"インストール先の空き: {free_gb(Path(self.install_var.get() or '.')):.0f} GB")
        parts.append(f"モデル保存先の空き: {free_gb(Path(self.models_var.get() or '.')):.0f} GB")
        self.space_var.set("  /  ".join(parts))
        need = []
        if self.action == "install":
            need.append(f"Ollama 本体 約 {OLLAMA_APP_GB:.0f} GB + インストーラ 約 {OLLAMA_INSTALLER_GB:.1f} GB (一時)")
        if model and not pulled:
            need.append(f"モデル {model} 約 {size:.1f} GB" if size else f"モデル {model} (容量は取得時に表示)")
        elif model:
            need.append(f"モデル {model} は取得済み")
        self.need_var.set("必要容量: " + " / ".join(need) if need else "必要容量: 追加のダウンロードはありません")

    def _ok(self) -> None:
        model = self.selected_model()
        if not model:
            messagebox.showinfo(APP_NAME, "使うモデルを選ぶか名前を入力してください。", parent=self.win)
            return
        self.result = {
            "action": self.action,
            "install_dir": Path(self.install_var.get().strip() or str(default_ollama_install_dir())),
            "models_dir": Path(self.models_var.get().strip() or str(ollama_models_dir())),
            "model": model,
            "pull": bool(self.pull_var.get()) and model not in set(self.info.get("models") or []),
        }
        self.win.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.win.destroy()

    def show(self) -> dict[str, Any] | None:
        self.win.wait_window()
        return self.result


class TaggerDownloadDialog:
    """タガーモデルが無いときに、どのモデルを取得して使うかを選ばせるダイアログ。"""

    def __init__(self, parent: "tk.Tk", default_name: str, reason: str):
        self.result: str | None = None
        win = self.win = tk.Toplevel(parent)
        win.title(f"{APP_NAME} - タガーモデルの取得")
        win.transient(parent)
        win.resizable(False, False)
        body = ttk.Frame(win, padding=14)
        body.pack(fill="both", expand=True)
        ttk.Label(
            body,
            text=(
                f"{reason}\n"
                "タグ推定に使う WD14 タガーモデルを選んでダウンロードしてください (初回のみ)。\n"
                f"保存先: {MODELS_DIR}"
            ),
            justify="left",
            wraplength=520,
        ).pack(anchor="w", pady=(0, 10))
        self.choice = tk.StringVar(value=default_name if default_name in TAGGER_MODELS else DEFAULT_TAGGER)
        for name, (_repo, desc) in TAGGER_MODELS.items():
            state = "取得済み" if tagger_ready(name) else "未取得"
            ttk.Radiobutton(body, text=f"{name}   {desc}   [{state}]", value=name, variable=self.choice).pack(anchor="w", pady=1)
        ttk.Label(
            body, foreground="#666", wraplength=520, justify="left",
            text="迷ったら eva02-large (最高精度) を選んでください。GPU が無い PC では vit / convnext が軽快です。",
        ).pack(anchor="w", pady=(8, 10))
        buttons = ttk.Frame(body)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="ダウンロードして使う", command=self._ok).pack(side="right")
        ttk.Button(buttons, text="あとで", command=self._cancel).pack(side="right", padx=(0, 8))
        win.protocol("WM_DELETE_WINDOW", self._cancel)
        win.bind("<Return>", lambda _e: self._ok())
        win.bind("<Escape>", lambda _e: self._cancel())
        win.update_idletasks()
        px, py = parent.winfo_rootx(), parent.winfo_rooty()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        win.geometry(f"+{px + max(0, (pw - win.winfo_width()) // 2)}+{py + max(0, (ph - win.winfo_height()) // 3)}")
        win.grab_set()
        win.focus_set()

    def _ok(self) -> None:
        self.result = self.choice.get()
        self.win.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.win.destroy()

    def show(self) -> str | None:
        self.win.wait_window()
        return self.result


class PoseEditor:
    """ボーン (骨格 17 点) を編集する画面。画像から自動で取るか、立ち姿の雛形から始めて、点をドラッグで直す。
    右クリックで点の表示 / 非表示 (見えていない関節はタグにも OpenPose 画像にも使わない)。LLM・GPU 不要。"""

    VIEW_H = 560

    def __init__(self, app: "App", source: Path | None):
        self.app = app
        self.win = tk.Toplevel(app.root)
        self.win.title("ボーンでポーズを編集")
        self.win.transient(app.root)
        self.image: Image.Image | None = None
        if source is not None:
            try:
                with Image.open(source) as im:
                    self.image = upright(im).convert("RGB")
            except OSError:
                self.image = None
        self.size = self.image.size if self.image else POSE_TEMPLATE_SIZE
        self.scale = self.VIEW_H / self.size[1]
        vw = max(240, int(self.size[0] * self.scale))
        self.canvas = tk.Canvas(self.win, width=vw, height=self.VIEW_H, bg="#202020", highlightthickness=0)
        self.canvas.grid(row=0, column=0, rowspan=8, padx=6, pady=6)
        self._photo = None
        if self.image is not None:
            from PIL import ImageEnhance, ImageTk

            shown = ImageEnhance.Brightness(self.image.resize((vw, self.VIEW_H), Image.BILINEAR)).enhance(0.55)
            self._photo = ImageTk.PhotoImage(shown)
        if app.pose_control and tuple(app.pose_control["size"]) == tuple(self.size):
            self.points = [list(p) for p in app.pose_control["points"]]
        else:
            self.points = self._template()
        self.drag: int | None = None
        side = ttk.Frame(self.win, padding=(0, 6, 6, 6))
        side.grid(row=0, column=1, sticky="n")
        self.auto_btn = ttk.Button(side, text="画像から骨格を取る", command=self.auto_detect,
                                   state="normal" if self.image is not None else "disabled")
        self.auto_btn.pack(fill="x")
        ttk.Button(side, text="立ち姿に戻す", command=self.reset).pack(fill="x", pady=(4, 0))
        ttk.Button(side, text="OpenPose 画像を保存...", command=self.save_openpose).pack(fill="x", pady=(12, 0))
        ttk.Label(side, text="ControlNet (OpenPose) 用。\n使わなければ保存不要", foreground="#666").pack(anchor="w")
        ttk.Label(side, text="プロンプトに入るポーズ:").pack(anchor="w", pady=(12, 0))
        self.tags_var = tk.StringVar()
        ttk.Label(side, textvariable=self.tags_var, wraplength=220, foreground="#0a5").pack(anchor="w")
        ttk.Label(side, text=("点をドラッグで移動、右クリックで表示 / 非表示。\n"
                              "向き (正面 / 後ろ) は骨格では決まらないので、\n「カメラ」の向きで指定してください。"),
                  foreground="#666", justify="left").pack(anchor="w", pady=(12, 0))
        self.status_var = tk.StringVar()
        ttk.Label(side, textvariable=self.status_var, wraplength=220, foreground="#a60").pack(anchor="w", pady=(8, 0))
        buttons = ttk.Frame(side)
        buttons.pack(fill="x", pady=(16, 0))
        ttk.Button(buttons, text="適用", command=self.apply).pack(side="left")
        ttk.Button(buttons, text="キャンセル", command=self.win.destroy).pack(side="left", padx=(6, 0))
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._move)
        self.canvas.bind("<ButtonRelease-1>", lambda _e: setattr(self, "drag", None))
        self.canvas.bind("<Button-3>", self._toggle)
        self.redraw()

    def _template(self) -> list[list[float]]:
        tw, th = POSE_TEMPLATE_SIZE
        k = min(self.size[0] / tw, self.size[1] / th)
        ox, oy = (self.size[0] - tw * k) / 2, (self.size[1] - th * k) / 2
        return [[ox + x * k, oy + y * k, 1.0] for x, y in POSE_TEMPLATE]

    def redraw(self) -> None:
        c = self.canvas
        c.delete("all")
        if self._photo is not None:
            c.create_image(0, 0, image=self._photo, anchor="nw")
        k = self.scale
        for a, b in POSE_BONES:
            pa, pb = self.points[a], self.points[b]
            if pa[2] >= POSE_MIN_CONF and pb[2] >= POSE_MIN_CONF:
                c.create_line(pa[0] * k, pa[1] * k, pb[0] * k, pb[1] * k, fill="#ff5050", width=3)
        for i, (x, y, conf) in enumerate(self.points):
            visible = conf >= POSE_MIN_CONF
            r = 6
            c.create_oval(x * k - r, y * k - r, x * k + r, y * k + r, outline="#fff" if visible else "#888",
                          fill="#3c3" if visible else "", width=2, tags=(f"p{i}",))
        tags = pose_tags_from_points(self.points)
        self.tags_var.set(", ".join(sd_tag(t) for t in tags) or "(タグにできる姿勢なし)")

    def _nearest(self, x: float, y: float) -> int | None:
        k = self.scale
        best = min(range(17), key=lambda i: (self.points[i][0] * k - x) ** 2 + (self.points[i][1] * k - y) ** 2)
        d = ((self.points[best][0] * k - x) ** 2 + (self.points[best][1] * k - y) ** 2) ** 0.5
        return best if d <= 14 else None

    def _press(self, event: Any) -> None:
        self.drag = self._nearest(event.x, event.y)

    def _move(self, event: Any) -> None:
        if self.drag is None:
            return
        k = self.scale
        self.points[self.drag][0] = min(max(event.x / k, 0), self.size[0])
        self.points[self.drag][1] = min(max(event.y / k, 0), self.size[1])
        self.points[self.drag][2] = max(self.points[self.drag][2], 1.0)  # 動かした点は見えている扱い
        self.redraw()

    def _toggle(self, event: Any) -> None:
        i = self._nearest(event.x, event.y)
        if i is not None:
            self.points[i][2] = 0.0 if self.points[i][2] >= POSE_MIN_CONF else 1.0
            self.redraw()

    def reset(self) -> None:
        self.points = self._template()
        self.redraw()

    def auto_detect(self) -> None:
        """画像から骨格を取る。モデル (RTMPose-s 21MB、人物検出 44MB) が無ければ確認してから取得する (初回のみ)。"""
        missing = []
        if not pose_model_ready():
            missing.append(f"骨格モデル RTMPose-s (約 {POSE_MODEL_MB}MB、OpenMMLab、Apache-2.0)")
        if not model_file_ok(PERSON_MODEL_PATH):
            missing.append("人物検出モデル (約 44MB、DeepGHS、MIT)")
        if missing and not messagebox.askyesno(
            APP_NAME, "画像から骨格を取るには、次のモデルを取得します (初回のみ。CPU で動き、GPU は使いません):\n\n"
            + "\n".join("・" + m for m in missing) + f"\n\n保存先: {MODELS_DIR}\n取得しますか？", parent=self.win):
            return
        self.auto_btn.config(state="disabled")
        self.status_var.set("骨格を推定中..." if not missing else "モデルを取得中... (初回のみ)")
        result: dict[str, Any] = {}

        def work() -> None:
            try:
                estimator = PoseEstimator()
                detector = self.app.engine.person_detector()
                box = main_person_box(detector.detect(self.image))
                result["points"] = estimator.estimate(self.image, box)
            except Exception as e:  # noqa: BLE001 - 画面に出す
                result["error"] = describe_error(e)

        thread = threading.Thread(target=work, daemon=True)
        thread.start()

        def wait() -> None:  # Tk はこのスレッド (UI) からだけ触る
            if thread.is_alive():
                self.win.after(100, wait)
                return
            self.auto_btn.config(state="normal")
            if "error" in result:
                self.status_var.set("骨格を取れませんでした: " + result["error"])
                return
            self.points = [list(p) for p in result["points"]]
            hidden = [KEYPOINTS[i][1] for i, p in enumerate(self.points) if p[2] < POSE_MIN_CONF]
            self.status_var.set("見えていない関節 (灰色、タグに使わない): " + "、".join(hidden) if hidden else "")
            self.redraw()

        self.win.after(100, wait)

    def save_openpose(self) -> None:
        path = filedialog.asksaveasfilename(parent=self.win, title="OpenPose 画像を保存", defaultextension=".png",
                                            filetypes=[("PNG", "*.png")], initialfile="openpose.png")
        if path:
            render_openpose(self.points, self.size).save(path)
            self.status_var.set(f"保存しました: {Path(path).name}")

    def apply(self) -> None:
        self.app.set_pose_control(self.points, self.size)
        self.win.destroy()


# 手順書 (「？ ヘルプ」/ F1。やりたいこと → 手順の形。機能を変えたら一緒に直す)
# 見出しの行は「■」で始める (太字で表示)
HELP_SECTIONS: list[tuple[str, str]] = [
    ("はじめに", """■ i2t でできること
画像 1 枚から、画像生成 AI (Stable Diffusion / NovelAI / Flux など) に貼るプロンプトを、この PC の中だけで作ります。
画像がインターネットに送られることはありません。

■ 最初の準備
1. 初めて起動すると、タガー (画像からタグを読むモデル) を取得するか聞かれます。「ダウンロードして使う」を押します (models フォルダに保存)
2. 文章のプロンプトや詳しい分析も使いたいときは、Ollama (VLM) を入れます (→「VLM (Ollama) を使う」)
   入れなくても、エンジンを「タガーのみ」にすれば使えます

■ 画面の見方
・左: プレビュー、「構図と強さ」、「出力」(エンジン・出力先など)
・右: ポジティブ / ネガティブプロンプト、下のタブ (追加プロンプト・人物ごと・検出タグ・分析メモ・メモ)
・下端: 「▶ 分析する」「■ 中止」、進み具合、状態の表示
・右上: 「？ ヘルプ」(この画面)、「⚙ 設定...」
"""),
    ("画像 1 枚からプロンプトを作る", """■ 手順
1. 上部の「モード」で「1 枚」を選ぶ
2. 画像を入れる (ウィンドウにドロップ / 「画像を選択...」/ Ctrl+V。→「画像を入れる方法」)
3. 左の「出力」欄で「出力先」を選ぶ (使う画像生成 AI。→「出力先を選ぶ」)
4. 「▶ 分析する」を押す (タガーのみなら数秒、VLM ありなら数十秒)
5. 右の「ポジティブプロンプト」の「コピー」(またはネガティブと一緒に「両方コピー」) で、画像生成 AI に貼る

■ ポイント
・結果はタグごとの塊で出ます。要らない塊を選んで Del で消せます (→「結果を選り分ける」)
・設定でオフにすると、結果のテキストをそのまま書き換えられます。保存・コピーは書き換えた内容で行われます
・途中でやめるときは「■ 中止」(または Esc キー)
・「⏸ 一時停止」で止めて、「▶ 再開」で続きから進められます。止まるのは今の段階 (タガー 1 枚分・VLM の 1 回の呼び出しなど) が
  終わってからです。止まっている間は、「GPU / VRAM の使用を抑える」がオンなら VLM を GPU から降ろします。日本語訳も同じように止められます
・2 人以上写っていると、1 人ずつ切り出して「人物ごと」タブに分けて書きます。人数を頭の位置で数え直したときや、
  画像全体の人数・性別と食い違ったときは、警告欄に出ます。写り込みとして外したタグ・色は「人物ごと」タブに理由付きで出ます
・抱きつく・肩に手を置くといった人どうしの動作は、画像全体から取り直して共有部分に入れ、自然文では「右の男性が左の女性の肩に
  手を置いている」のような関係の文にします。誰が誰にしているかは VLM (「タガー + VLM」のとき) か骨格で決め、分からなければ
  向きを書きません。決めた結果は「人物ごと」タブの [人どうしの動作] で確かめられます
"""),
    ("画像を入れる方法", """■ ドラッグ＆ドロップ
画像ファイルをウィンドウのどこかに落とします。最大 8 枚モードでは、プレビューの落とした枠に入ります。

■ ファイルを選ぶ
1 枚モードは上部の「画像を選択...」、最大 8 枚モードはプレビューの枠をクリックします。

■ 貼り付け (Ctrl+V)
ブラウザの「画像をコピー」やスクリーンショットを、そのまま Ctrl+V で読み込めます。
・最大 8 枚モードでは、マウスを置いている枠に入ります (プレビューの外なら空いている最初の枠)
・ファイルの無い画像は、一時フォルダ (%TEMP%\\i2t_paste) に PNG で保存してから使います (自動では消しません)
・文字の入力欄で文字をコピーしているときは、普通の文字の貼り付けになります

■ 外す
プレビューを右クリックで外せます。上部の「✕ 画像をクリア」で全部外せます。
"""),
    ("複数の画像を組み合わせる (最大 8 枚)", """■ 手順
1. 上部の「モード」で「最大 8 枚」を選ぶ
2. プレビューが 8 つの枠に分かれます (上段 ① メイン ② ポーズ ③ 服装 ④ 背景、下段 ⑤ 小物 ⑥ 体格 ⑦ 作風・質感 ⑧ 表情)
3. ① メインは必ず入れます。②〜⑧ は使いたいものだけ入れます (空の枠の要素は ① から取ります)
4. 「▶ 分析する」を押す

■ 枠ごとに取るもの
① メイン: キャラ・顔・髪・目 (⑥ が無ければ体格、⑦ が無ければ画風、⑧ が無ければ表情も)
② ポーズ: 姿勢・手足・体の向き・構図 (画角・アングル)・縦横比 (⑧ が無ければ視線も)
③ 服装: 服・靴・装飾品・髪飾り (⑤ が無ければ持ち物も)
④ 背景: 場所・天候・時間帯・照明
⑤ 小物: 手に持つ物・置いてある物
⑥ 体格: 胸・体型・身長・肌の色・筋肉
⑦ 作風・質感: 画風・技法 (絵か写真か)、絵なら色の塗り方、写真なら撮影環境 (カメラ・レンズの写り・光の当て方・粒子・色の調子) と
   作品名・作者名 (→「絵を実写にする / 写真を絵にする」)。⑦ を入れると、写り・光の当て方・色調は ① ④ ではなく ⑦ から取ります
⑧ 表情: 表情 (笑顔・赤面・涙・口の開き・目を閉じるなど) と視線 (カメラ目線・よそ見など)。
   視線は ⑧ → ② → ① の順に、入っている枠から取ります (振り返り looking back は体の向きなので ② のまま)

■ ポイント
・同じ画像をほかの枠にも使えます (枠を右クリック →「この画像を ○ にも使う」)
・どのタグをどの画像から取ったかは「分析メモ」と「検出タグ」に出ます
・ポーズを正確に再現したいときは、② の画像を ControlNet (OpenPose) にも使ってください
"""),
    ("絵を実写にする / 写真を絵にする", """■ 絵を実写にする
1. 「モード」で「最大 8 枚」を選ぶ
2. ① メインに、実写にしたい絵を入れる
3. ⑦ 作風・質感に、写真を入れる (人物の写真など)
4. 出力先を選び、「▶ 分析する」を押す
→ ① の画風のタグは外して ⑦ の画風 (写真) にします。品質タグ・ネガティブも写真用に切り替わります

■ 写真を絵にする
① に写真、⑦ に目指す画風の絵を入れます。あとは同じです。

■ 知っておくこと
・絵か写真かの判定は ⑦ から行います (⑦ の写真らしさが 0.25 以上で写真)。判定と使った値は「分析メモ」に出ます
・「タガー + VLM」なら、VLM が ⑦ の作品名・作者名 (例: dragon ball) を推定して本文に入れます。推定なので違うことがあります。違っていたら消してください。「タガーのみ」では名前は出せません
・⑦ からは人種・肌の色 (⑥ / ①) と年代 (設定の「年代スタイル」で ① から) は取りません
・⑦ からは、絵なら色の塗り方 (アニメ塗り・厚塗り・水彩のにじみ・影の色・ハイライトなど)、写真なら撮影環境
  (カメラの種類・レンズの写り・光の当て方・粒子・色の調子) も取ります。細かく読めるのは「タガー + VLM」のときだけです。
  「タガーのみ」では film grain / soft lighting のようなタグの範囲になります
・⑦ を入れると、ぼけ・逆光・色調のような写りと光の当て方のタグは ⑦ から取ります (① ④ の分は外します)。
  時間帯・天気・光源 (夕焼け・ネオンなど)・画角は今までどおり ① ② ④ から取ります
・写真に撮影情報 (EXIF: カメラ・レンズ・焦点距離・絞り・シャッター速度・ISO・フラッシュ) が残っていれば、それも本文に入れます。
  ⑦ が空なら ① の写真から読みます。読んだ値は「分析メモ」に出ます。位置情報・撮影日時・製造番号は使いません。
  SNS から保存した写真やスクリーンショットには撮影情報が残っていないことが多いです。
  撮影プリセット (構図と強さの「撮影」) を選んでいるときはプリセットを使い、撮影情報は分析メモに出すだけにします
・ネガティブ非対応の出力先 (Krea 2 など) では「アニメ絵を避ける」指定を書けないので、写真の指定は本文だけになります
・MOSS-Image で、⑦ に写真を入れても 3DCG のようになるときは、「出力」欄の「画風変更」で「実写」を選んでください
  (→「画風を変える (MOSS-Image)」)
"""),
    ("ポーズ・カメラ・強さを変える", """左の「構図と強さ」欄で、画像から読んだポーズや構図を上書きできます (LLM・GPU 不要)。何も指定しなければ画像のままです。

■ ポーズを変える
1. 「ボーン編集...」を押す (「画像から骨格を取る」で今の画像のポーズから始められます)
2. 骨格の点をドラッグして動かす (右クリックで点の表示 / 非表示)
3. 「適用」を押すと、ポーズのタグがボーンの形に置き換わります。「解除」で元に戻ります
・「OpenPose 画像を保存...」で ControlNet 用の画像も作れます

■ カメラを変える
「カメラ」の向き・高さ・距離・傾きを選びます。

■ 撮影の雰囲気を変える (撮影プリセット)
「撮影」でポートレート 85mm・35mm フィルム・スタジオ・シネマ・スマホ・夜景などを選びます。
画像から読んだレンズ・ぼけ・照明の当て方・色調のタグを外し、選んだ英語の文に置き換えます (全出力先)。
画角・アングル・時間帯・天気・光源 (夕日やネオンなど) はそのまま残ります。外したタグは「分析メモ」に出ます。
自然文の出力先と Anima では、選んだ語を文法の通った英文にして入れます (VLM を使うときは VLM の文に織り込みます)。
「組み合わせ...」を選ぶと、写り・カメラ・レンズ・質感・光・フィルム・粒子・被写界深度を項目ごとに選べます
(例: シネマ風の写真 + 富士 X-T4 + 50mm f/1.8 + なめらかで自然な肌 + バランスのよい環境光 + うっすら粒子 + 浅い)。
カメラは富士・ソニー・キヤノン・ニコン・ライカ・中判・フィルムカメラ・映画用・スマホなど約 70 種類です。
候補は config.json の "camera_looks" を書き換えると追加・編集できます (i2t を開き直すと反映)。

■ 強さを変える
「強さ」で、人物・ポーズ・服装・体格・小物・背景ごとに「弱い / 普通 / 強い」を選びます。
SDXL / NovelAI は重み (1.2 / 0.8)、ほかのタグ形式は並び順で表します。自然文の出力先には付けられません。
"""),
    ("三面図を作る", """キャラクターを前・後ろ・横から描いた三面図 (設定画) のプロンプトにします。
1. 左の「出力」欄の「三面図にする」にチェックを入れる
2. 「▶ 分析する」を押す

■ どう変わるか
・画像から読んだポーズ・カメラ (画角・アングル)・背景 (照明を含む)・小物のタグを外します
・reference sheet, turnaround, multiple views, full body, standing, straight-on (前), from behind (後ろ), from side (横),
  simple background, white background を入れます (Danbooru wiki で確かめたタグ)。solo は外します
・自然文の出力先には「前・後ろ・横を並べた三面図、白い無地の背景」の一文を足します
・髪・目・服装・体格・表情はそのまま残ります
・外したタグは「分析メモ」に出ます。「追加プロンプト」タブは画像のままです
・「分身を防ぐ」がオンでも、三面図の語 (reference sheet など) はネガティブに入れません
"""),
    ("LoRA でキャラを指定するとき (人物の特徴を入れない)", """LoRA などでキャラクターを指定するとき、画像の服装・体格・ポーズ・表情だけを使うための機能です。
1. 左の「出力」欄の「LoRA 用に入れない:」の「人物の特徴」にチェックを入れる
2. 「▶ 分析する」を押す
3. 結果をコピーし、LoRA のトリガーワード (や <lora:名前:1>) は自分で足してください

■ 外すもの
・髪 (色・長さ・髪型) と目の色
・顔立ち (つり目・たれ目・そばかす・ほくろ・ひげなど)
・キャラクター名・作品名
・肌の色

■ 残すもの
服装 (髪飾りを含む)・体格・ポーズ・表情・視線・背景・画風

■ 知っておくこと
・「タガー + VLM」では、VLM にも髪・目・顔・肌・キャラ名を書かないよう伝えます。それでも VLM の文が外したものを書いていたら、
  その文は使わずタグから組み立てた文にします (ステータス欄の「注意」に出ます)
・外したタグは「分析メモ」に出ます。「追加プロンプト」タブは画像のままです
・「三面図にする」と一緒に使えます
"""),
    ("画風 LoRA を使うとき (作風の特徴を入れない)", """画風 LoRA などで絵柄を指定するとき、画像の絵柄を持ち込まないための機能です。
1. 左の「出力」欄の「LoRA 用に入れない:」の「作風の特徴」にチェックを入れる
2. 「▶ 分析する」を押す

■ 外すもの
・画風・技法のタグ (anime coloring, cel shading, watercolor, thick outlines など)
・絵か写真か (realistic, photorealistic, anime, 3d など) と、それに合わせた品質タグの語 (写真の photorealistic など) と
  ネガティブの定番 (アニメ絵の realistic / 3d、写真の anime / illustration など)
・年代 (1990s (style) など)
・⑦ 作風・質感から VLM が推定した作品名・作者名
・質感の描き込み (wet skin・shiny skin・dew drops などの肌のつや・濡れ、lips・makeup などの唇・化粧)。
  汗・赤面・そばかす・肌の色は残します
・品質タグ (masterpiece など) と照明・レンズの語は外しません (品質タグは「⚙ 設定...」の「品質タグを先頭に付ける」で外せます)
・自然文の出力先では「アニメ調のイラスト」などの書き出しを「An image」にします。Midjourney の --niji / --raw も付けません

■ 知っておくこと
・「タガー + VLM」では、VLM にも画風・絵か写真か・作者名を書かないよう伝えます。それでも VLM の文が書いていたら、
  その文は使わずタグから組み立てた文にします
・外したタグは「分析メモ」に出ます。「追加プロンプト」タブ (画風の欄など) は画像のままです
・「人物の特徴」「三面図にする」と一緒に使えます
"""),
    ("出力先を選ぶ", """左の「出力」欄の「出力先」で、使う画像生成 AI を選びます。書き方 (タグ / 文章、ネガティブの有無) が変わります。

■ タグ形式
・SDXL / Pony / Illustrious: タグを並べる形。ネガティブあり
・NovelAI V4 / V4.5 / V5: NovelAI の書き方。2 人以上は「人物ごと」タブの各行を Character 欄に貼る (各行の最後に位置の文、向きが分かった動作は source# / target# / mutual#。22 人まで分けます。V4 / V4.5 のキャラ欄は 6 人までです)
・Anima: タグ + 自然文

■ 自然文 (VLM 推奨)
Flux.1 / SD3、FLUX.2、Krea 2、Qwen-Image、Z-Image Turbo、Z-Image、MOSS-Image、ChatGPT、Grok、Nano Banana、Midjourney
・Z-Image Turbo はネガティブが効かないので出しません。Z-Image (ベース) はネガティブを出します
・MOSS-Image のネガティブ欄は空です (公式の既定も空)。必要なら自分で入れてください。「画風を変える」と「年齢の見た目を書く」は、
  それぞれの語をこの欄に入れます (CFG 1 の高速 LoRA では、ネガティブ欄は効きません)
・MOSS-Image では、キャラ一覧 (hide1 の MOSS-Image Character & Style Catalog。LoRA なしの生成例で、選別されていません) に載っているキャラの名前を「名前 from 作品名」で書きます (例: dawn (pokemon) from pokemon。一覧の書き方で、公式資料には書き方がありません)。一覧は初めて MOSS-Image で分析するときに確かめてから取得します (約 7MB)。載っているかどうかと生成例の画像のアドレスは「分析」タブに出ます
・VLM が無いときは、タグから組み立てた英文になります
・ネガティブ非対応の出力先では、ネガティブ欄は空です

■ メモ タブ
選んだ出力先のおすすめ設定 (ステップ数・CFG など) が出ます。
"""),
    ("結果をコピー・保存する", """■ コピー
・各欄の「コピー」: その欄だけ
・「両方コピー」: ポジティブとネガティブをまとめて
・「追加プロンプト」タブ: 被写体・ポーズ・服装・背景・カメラ・照明・画風・動画用カメラワークなどを別々にコピー

■ 保存
「TXT に保存...」「JSON に保存...」で、テキスト欄で書き換えた内容を保存します。
"""),
    ("結果を選り分ける (要るタグ・要らないタグ)", """ポジティブ / ネガティブはタグごとの塊 (, で区切った 1 つずつ) で出ます。
自然文の出力先も , で区切った塊になります。

■ 要らないタグを消す
1. 塊をクリックして選ぶ (Ctrl+クリックで選び足す、Shift+クリックで範囲を選ぶ)
2. Del (または BackSpace) を押すと、選んだ塊が消えます
3. 間違えたら Ctrl+Z で 1 つずつ戻せます
→ 消した結果は、コピー・保存・履歴にもそのまま入ります

■ 日本語で意味を確かめる
1. ポジティブ欄の右上の「日本語」を押す
2. 塊の表示が日本語になります (重みは ×1.3 のように後ろに付きます)。「English」で英語に戻ります
・タガーの出すタグ (一般タグ 8,106 個) と、i2t が足す品質タグ・ネガティブの定番・撮影プリセットの語は、同梱の辞書ですぐに
  日本語になります (Ollama は要りません)。キャラ名は英語のまま表示します
・辞書に無い語 (VLM が足した言い回し・自然文など) は、Ollama の「訳のモデル」(「⚙ 設定...」の VLM 欄。
  既定 hf.co/IndexTeam/Index-Translate-2B-GGUF:Q4_K_M。PLaMo 3 Translate の 2B / 8B も選べ、「取得」で Ollama に取得できます) が訳し、i2t.exe の隣の translations.json に登録します (次からはすぐ出ます)。
  Ollama が無いときは、辞書に無い語だけ英語のままです
・日本語で見ていても、コピー・保存は英語のままです
・辞書も訳のモデルも機械翻訳なので、違っていることがあります。直したいときは translations.json に「"英語": "日本語"」で書くと、
  そちらが優先されます

■ 手持ちの辞書を使う
Tag Autocomplete 用の翻訳辞書 (1 列目がタグ、2 列目が訳の CSV) を i2t.exe の隣の dict フォルダに置くと、同梱の辞書より優先して使います
(i2t を開き直すと反映)

■ 知っておくこと
・塊の表示では書き足しはできません。書き足したいときは「⚙ 設定...」の「結果をタグごとの塊で表示する」をオフにすると、
  今までどおり文字の欄で書き換えられます (オンに戻すと、書き換えた内容をまた塊に分けます)
"""),
    ("履歴・画像の生成情報", """■ 履歴
解析が終わるたびに、結果と小さなサムネイルを i2t.exe の隣の history フォルダに残します (新しい順に 50 件。古いものから消えます)。
1. 「履歴...」を押す
2. 一覧の行をダブルクリック (または「開く」)
→ 結果を画面に戻します。元画像が残っていれば、プレビューにも読み込みます
・タグの塊を消した・戻したときは、履歴にもその結果が入ります
・「削除」「すべて削除...」で消せます

■ 画像の生成情報
AI で作った画像に、作ったときのプロンプトが入っていると、ステータスに「生成情報あり」と出ます。
「生成情報...」を押すと、元のポジティブ / ネガティブ / 設定を見られます (i2t の推定と比べられます)。
・読める形式: A1111 / Forge、ComfyUI、NovelAI、Fooocus、InvokeAI
・SNS に上げた画像・スクリーンショット・貼り付けた画像など、情報が消えた画像からは読めません
"""),
    ("先に画像の説明を書かせる (JoyCaption など)", """「⚙ 設定...」の「説明のモデル」にモデルを入れると、いつもの VLM の前に、そのモデルが画像の説明を書きます。
説明は VLM に参考として渡され、「分析メモ」にも残ります。欄が空なら使いません (初期設定は空)。

・候補に出る JoyCaption Beta One は、画像の説明を書くことに特化したモデルです (約 5.8GB)。Ollama にまだ無ければ、
  欄の右の「取得」で取得できます
・VLM の呼び出しが 1 回増えるので、そのぶん時間がかかります
・説明を書けなかったときは、ステータス欄の「注意」に出して、いつもの分析を続けます
・コマンドラインでは --describer モデル名
"""),
    ("年齢の見た目を書く", """出力に年齢が書かれていないと、生成する側のモデルの癖で年齢が決まります (文章で書くモデルは 20 代の大人に、
アニメ系のタグのモデルは若く寄りがちです)。元の画像と同じ年齢で出るように、人物ごとに画像から読んだ年齢層を本文に書きます。

■ 読み方
・「タガー + VLM」のとき、VLM が人物ごとに年齢層を選びます (赤ちゃん / 幼児 / 子ども / 10〜12 歳 / 13〜17 歳 / 18〜29 歳 /
  30 代 / 40 代 / 50 代 / 60 代 / 70 歳以上)。2 人以上のときも 1 人ずつ読みます
・タガーが年齢のタグ (child / mature female / old man など) を出していれば、それも使います。子どもかどうか・高齢かどうかが
  VLM とタガーで食い違うときは、タガーのタグに合わせます
・「タガーのみ」のときは、タガーの年齢のタグだけから決めます
・読んだ結果は「分析メモ」の「年齢の見た目」の行に出ます。VLM とタガーの読み取りなので、外れることがあります

■ 書き方
・文章の出力先: 人物を年齢に合った言い方で書きます (a little girl of elementary-school age / a teenage boy /
  a woman in her forties / an elderly man など)。VLM の文が別の言い方 (a young woman など) をしていたら、言い換えます
・タグの出力先: 年齢のタグを足します (child・female child / mature female / old man など。10 代と 20 代に当たる
  Danbooru のタグは無いので足しません)。全員が大人のときはネガティブに child / aged down などを、全員が子どもの
  ときは aged up / mature female などを入れます
・Anima はタグと文の両方に書きます
・MOSS-Image: 本文の言い方に加えて、ネガティブ欄にも、ずれる方向を抑える語を入れます (全員が大人のときは child /
  childlike face / teenager、40 代から上だけなら young adult / youthful face も。全員が子どものときは adult /
  mature face / middle-aged など)。CFG 1 の高速 LoRA (8 ステップ・4 ステップ) では、ネガティブ欄は効きません

■ 年齢が書かれないとき
・「人物の特徴を入れない」がオンのとき (年齢もその人の特徴として外れます)
・2 人以上の画像で、人物ごとに分けていないとき (誰の年齢か結び付けられないので、書けません)
・性的な内容と判定した画像の、18 歳未満に見える人物 (この機能では年齢を書けません。タガーと VLM が出した結果は、
  そのまま出ます)
・「⚙ 設定...」の「年齢の見た目を書く」をオフにしたとき
"""),
    ("文章の出力先の文 (MOSS-Image / Krea 2 など)", """文章で書く出力先 (MOSS-Image / Krea 2 / Qwen-Image / Z-Image / FLUX など) の本文は、条件によって作り方が違います。

■ 文の作り方
・1 枚だけ + 「タガー + VLM」: VLM が画像を見て書いた文を使います
・複数枚モード (② 〜 ⑧ のどれかを使う)・2 人以上・ボーンやカメラを指定したとき: 枠ごとの内容を決まった順に並べます
・「タガーのみ」: タグを文の形に並べます

■ 文を仕上げる (「タガー + VLM」で、枠ごとの内容を並べた文のとき。初期設定はオン)
・枠ごとの内容を並べた文は、見出しとタグの列になりがちです。集めた内容を VLM にもう一度渡して、出力先の書き方で
  1 つの文章に書き直します (画像は見せません)
・1 枚だけのときは VLM が画像を見て書いた文をそのまま使い、書き直しはしません (VLM をもう一度呼ぶと、そのぶん遅くなるため)
・書き直した文が内容を落としていたり、キャラ名・作品名を落としていたり、外したはずの要素を書いていたり、画像の文字を変えていたりしたら使わず、元の文のままにします
  (ステータス欄の「注意」に出ます)
・VLM の呼び出しが 1 回増えます。「⚙ 設定...」の「文を仕上げる」でオフにできます
・どこに時間がかかったかは、「分析メモ」の最後の「かかった時間」の行に出ます

■ Danbooru の語
タグから文を組み立てるときは、Danbooru だけで通じる語をふつうの英語にします
(cowboy shot → framed from the thighs up、ahoge → a single upright strand of hair、serafuku → sailor-style school uniform など)。
タグの出力先 (SDXL 系・NovelAI・Anima のタグ列) は、タグのままです。

■ ⑦ 作風・質感を入れたとき
・文章の出力先: 絵か写真かと画風を、文の最後ではなく先頭に書きます
・タグの出力先: 画風のタグを、人数・キャラ名のすぐ後ろに置きます (前は服装・ポーズ・人物の後ろでした)
・「構図と強さ」の「強さ」に「画風」があります。強い / 弱いを選ぶと、重みの書式がある出力先では画風のタグに重みを付けます
  (⑦ が無くても使えます。文章の出力先には重みを付けられません)
"""),
    ("画風を変える (MOSS-Image)", """元の画像とは違う画風で作りたいとき、⑦ 作風・質感に見本を入れても、元の絵柄が勝つことがあります
(絵を実写にしたら 3DCG のようになる、など)。人物の説明が元の画像から書かれることと、画風が文の後ろに回ることが原因です。
「画風を変える」で種類を選ぶと、その画風の画像として書き直します。
1. 出力先を MOSS-Image にする
2. 左の「出力」欄の「画風変更」で種類を選ぶ (実写 / アニメ風 / 漫画風 (白黒) / 水彩 / 厚塗り / 油絵 / 鉛筆画)
3. 「▶ 分析する」を押す (⑦ 作風・質感は入れなくても使えます)

■ やること
・本文の先頭で、その画風だと書きます (実写なら「実在の人物をカメラで撮った写真」、水彩なら「紙に描いた水彩画」など)
・元の画風の語を外すか、その種類の語に言い換えます (実写にするなら anime-style illustration → photograph、
  水彩にするなら photograph → watercolour painting など)。実写にするときは realistic / photorealistic を real にします
  (「本物らしく描いた」という語で、CG の説明にも使われるため)
・その画風にしかない描き方・質感の文を最後に足します (実写: 肌のきめ・髪の一本一本、アニメ風: 均一な線と 2 段階の影、
  水彩: にじむ絵の具と紙の目、厚塗り: 筆の跡、鉛筆画: ハッチング、など)
・⑦ の画像が選んだ種類と同じ系統 (写真 / 絵画 / 絵) なら、⑦ から読んだ文 (写真なら撮り方、絵なら塗り方) を先頭の直後に置きます。
  系統が違うときは ⑦ の画風は使いません (ステータス欄の「注意」に出ます)
・キャラクター名は、どの種類でも本文に残ります (実写でも同じです。誰の実写かは名前で決まるため)。
  実写にしたのに元の絵柄に寄るときは、本文の名前を消して試してください
・ネガティブ欄に、ほかの画風を抑える語を入れます。CFG 1 の高速 LoRA では効きません
・「タガー + VLM」では、VLM にも元の画像の画風を書かないよう伝えます (実写にするときは、絵特有の大きな目などを実際の人の
  特徴に言い換えさせます)

■ 知っておくこと
・MOSS-Image 以外の出力先では何もしません
・「作風の特徴を入れない」と一緒には使えません (「画風を変える」を優先します)
・外した語・言い換えた語は「分析メモ」に出ます
"""),
    ("設定 (⚙ 設定...)", """よく変える設定は本画面の「出力」欄、それ以外は右上の「⚙ 設定...」にあります。変えた内容はすぐ反映され、config.json に保存されます。

■ 本画面の「出力」欄
・エンジン: 「タガー + VLM」(詳しい・遅い) / 「タガーのみ」(速い・オフライン)
・出力先、モザイク指定 (強さ)、分身を防ぐ、性別をそろえる、画風変更 (MOSS-Image)、白黒をカラーで出す、
  人物の特徴を詳しく (VLM)、三面図にする、LoRA 用に入れない (人物の特徴 / 作風の特徴)

■ 「⚙ 設定...」
・タガー: モデル、しきい値 (一般 / キャラクター / 画風。下げるとタグが増える)
・VLM (Ollama): モデル、「追加 (pull)」、訳のモデル (結果の日本語訳用)、Ollama URL、「再接続」、「導入...」
・プロンプトの書き方、人物、動作・表示 (プレビューのモザイク表示、結果をタグごとの塊で表示する など)
"""),
    ("VLM (Ollama) を使う", """VLM は画像を見て文章を書く AI です。自然文の出力先や、詳しい分析・人物の振り分けに使います。この PC の中で動きます。
最初は「タガーのみ」で動きます。VLM は拡張機能 (推奨) で、無くても全部の出力先を使えます。

■ 入れる
1. 「出力」欄のエンジンを「タガー + VLM」にすると、Ollama が無ければ導入サポートを開くか聞きます (1 回だけ)。
   または「⚙ 設定...」の「導入...」を押す
2. 保存先・必要な容量・おすすめのモデルを確かめて進める (既に入っていれば、起動とモデルの取得だけ)

■ 使う
1. 「出力」欄のエンジンを「タガー + VLM」にする
2. 「⚙ 設定...」の VLM のモデルを選ぶ (無ければ名前を入れて「追加 (pull)」)

■ 注意
・GPU のメモリをほかのアプリ (ComfyUI など) が使っていると、失敗したり遅くなったりします。失敗してもタガーの結果は出ます
"""),
    ("困ったとき", """■ 分析が失敗する・遅い
・VLM が失敗する: ほかのアプリが GPU を使っていないか確かめる。「タガーのみ」でも使えます
・Ollama に接続できない: 「⚙ 設定...」の「再接続」、または「導入...」から起動

■ タグが少ない / 多い
「⚙ 設定...」のしきい値 (一般) を下げる / 上げる。画風のタグが拾えないときは「画風」のしきい値を下げる

■ 実写化したのに絵になる
「分析メモ」の「絵か写真かの判定は ⑦ 作風・質感から」の行を見てください。写真らしさの値が 0.25 未満なら、別の写真を ⑦ に使ってください
MOSS-Image で、写真と判定されているのに 3DCG のようになるときは、「出力」欄の「画風変更」で「実写」を選んでください (→「画風を変える (MOSS-Image)」)

■ 結果の中身を変えたい
タグの塊で出ているときは、要らない塊を選んで Del で消します (→「結果を選り分ける」)。書き足したいときは
「⚙ 設定...」の「結果をタグごとの塊で表示する」をオフにして、テキスト欄で直接書き換えてください。

■ 警告欄
外したタグや、うまくいかなかった理由は、ステータス欄の「注意」と「分析メモ」に出ます。
"""),
]


def fit_min_size(win: Any, min_width: int) -> None:
    """窓の最小サイズを、中身 (ボタンを含む) が隠れない高さにする。表示倍率で高さが変わるので実際に測る
    (2026-10-04 利用者の指摘: 履歴の窓でボタンが下に隠れていた)。"""
    win.update_idletasks()
    win.minsize(min_width, win.winfo_reqheight())
    width, height = win.winfo_width(), win.winfo_height()
    if height < win.winfo_reqheight():  # 初期サイズが中身より低ければ広げる
        win.geometry(f"{max(width, min_width)}x{win.winfo_reqheight()}")


@dataclass
class _Pointer:
    """ドロップのイベントの代わり (貼り付けたときのマウスの位置。複数枚モードで入れる枠を決める)。"""
    x_root: int
    y_root: int


class PromptChips:
    """ポジティブ / ネガティブの欄。チップ表示 (, 区切りの塊ごと) と、今までどおりの文字の欄を切り替える。
    チップ表示: クリックで選ぶ (Ctrl+クリックで追加、Shift+クリックで範囲)、Del / BackSpace で消す、Ctrl+Z で戻す。
    書き足しはできない。コピー・保存は常に英語 (チップを , でつないだ文字)。"""

    CHIP_BG, CHIP_SEL, CHIP_FG = "#e8eef6", "#3b78c4", "#1d2733"

    def __init__(self, box: "tk.Text", on_change: Callable[[], None]):
        self.box = box
        self.on_change = on_change
        self.chip_mode = False
        self.ja = False
        self.table: dict[str, str] = {}
        self.lines: list[list[str]] = []
        self.selected: set[tuple[int, int]] = set()
        self._anchor: tuple[int, int] | None = None
        self._undo: list[list[list[str]]] = []
        self._labels: list["tk.Label"] = []
        for seq in ("<Delete>", "<BackSpace>"):
            box.bind(seq, self._on_delete_key, add="+")
        for seq in ("<Control-z>", "<Control-Z>"):
            box.bind(seq, self._on_undo_key, add="+")
        box.bind("<Configure>", lambda _e: self._rewrap(), add="+")

    # ---- 中身 ----
    def get_text(self) -> str:
        if self.chip_mode:
            return join_prompt_chips(self.lines)
        return self.box.get("1.0", "end-1c").strip()

    def set_text(self, text: str) -> None:
        self.lines = split_prompt_chips(text)
        self.selected.clear()
        self._anchor = None
        self._undo.clear()
        if self.chip_mode:
            self._render()
        else:
            for label in self._labels:
                label.destroy()
            self._labels = []
            self.box.config(state="normal", cursor="xterm", insertwidth=2)
            self.box.delete("1.0", "end")
            self.box.insert("1.0", text)

    def set_mode(self, chip_mode: bool) -> None:
        """チップ表示 ↔ 文字の欄。今の中身を引き継ぐ (文字の欄で書き換えた内容もチップに分け直す)。"""
        if chip_mode == self.chip_mode:
            return
        text = self.get_text()
        self.chip_mode = chip_mode
        self.set_text(text)

    def set_language(self, ja: bool, table: dict[str, str]) -> None:
        self.ja, self.table = ja, table
        if self.chip_mode:
            self._render()

    def chunks(self) -> list[str]:
        return [c for line in self.lines for c in line]

    # ---- 描画 ----
    def _wrap_width(self) -> int:
        return max(120, self.box.winfo_width() - 30)

    def _render(self) -> None:
        box = self.box
        box.config(state="normal", cursor="arrow", insertwidth=0)  # 文字カーソルは出さない (書き足しはできない)
        box.delete("1.0", "end")
        for label in self._labels:
            label.destroy()
        self._labels = []
        wrap = self._wrap_width()
        for li, line in enumerate(self.lines):
            for ci, chunk in enumerate(line):
                sel = (li, ci) in self.selected
                text = chip_label_ja(chunk, self.table) if self.ja else chunk
                label = tk.Label(box, text=text, font=("Yu Gothic UI", 10), padx=5, pady=1, bd=1, relief="solid",
                                 bg=self.CHIP_SEL if sel else self.CHIP_BG, fg="white" if sel else self.CHIP_FG,
                                 wraplength=wrap, justify="left", cursor="hand2")
                label.bind("<Button-1>", lambda e, k=(li, ci): self._on_click(k, e))
                box.window_create("end", window=label, padx=2, pady=2)
                self._labels.append(label)
            if li < len(self.lines) - 1:
                box.insert("end", "\n")
        box.config(state="disabled")

    def _rewrap(self) -> None:
        if self.chip_mode and self._labels:
            wrap = self._wrap_width()
            for label in self._labels:
                label.config(wraplength=wrap)

    # ---- 操作 ----
    def _on_click(self, key: tuple[int, int], event: Any) -> str:
        self.box.focus_set()
        ctrl, shift = bool(event.state & 0x0004), bool(event.state & 0x0001)
        if shift and self._anchor is not None:
            order = [(li, ci) for li, line in enumerate(self.lines) for ci in range(len(line))]
            a, b = sorted((order.index(self._anchor), order.index(key)))
            self.selected = set(order[a:b + 1])
        elif ctrl:
            self.selected ^= {key}
            self._anchor = key
        else:
            self.selected = {key}
            self._anchor = key
        self._render()
        return "break"

    def delete_selected(self) -> bool:
        if not self.chip_mode or not self.selected:
            return False
        self._undo.append([list(line) for line in self.lines])
        kept: list[list[str]] = []
        for li, line in enumerate(self.lines):
            rest = [c for ci, c in enumerate(line) if (li, ci) not in self.selected]
            if rest or not line:  # 消して空になった行は詰める (元から空の行は残す)
                kept.append(rest)
        self.lines = kept
        self.selected.clear()
        self._anchor = None
        self._render()
        self.on_change()
        return True

    def undo(self) -> bool:
        if not self.chip_mode or not self._undo:
            return False
        self.lines = self._undo.pop()
        self.selected.clear()
        self._anchor = None
        self._render()
        self.on_change()
        return True

    def _on_delete_key(self, _event: Any) -> str | None:
        if not self.chip_mode:
            return None  # 文字の欄なら普通の削除
        self.delete_selected()
        return "break"

    def _on_undo_key(self, _event: Any) -> str | None:
        if not self.chip_mode:
            return None  # 文字の欄なら tk.Text の元に戻す
        self.undo()
        return "break"


class App:
    def __init__(self, root: "tk.Tk", cfg: dict[str, Any], initial_image: str | None = None):
        self.root = root
        self.cfg = cfg
        # 枠ごとの画像。person は 1 枚モードの画像 / 複数枚モードの ① メイン
        self.slot_paths: dict[str, Path | None] = {slot: None for slot in REF_SLOTS}
        self.slot_thumbs: dict[str, Image.Image | None] = {slot: None for slot in REF_SLOTS}  # 元のサムネイル
        self.slot_infos: dict[str, str] = {slot: "" for slot in REF_SLOTS}  # "幅×高さ"
        self.slot_sizes: dict[str, tuple[int, int]] = {}  # 元画像のサイズ (プレビューに枠を描くため)
        self.slot_gen: dict[str, dict[str, str] | None] = {slot: None for slot in REF_SLOTS}  # 画像に入っていた生成情報
        self._history_win: Any = None
        self._gen_win: Any = None
        self.result: PromptResult | None = None
        self.queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        self.engine = Engine(cfg, progress=lambda m: self.queue.put(("status", m)))
        self.engine.fraction = lambda f, est=None: self.queue.put(("frac", (f, est)))
        self._vlm_estimate: tuple[float, float, float, float] | None = None  # (始めた時刻, 見込み秒数, 始まり, 終わり)
        self.busy = False
        self._modal_open = False  # 確認ダイアログを開いている間は Ollama 導入の案内を重ねない
        self._preview = None  # PhotoImage の参照を保持（GC 防止）
        self._known_models: list[str] = []
        self._ollama_ok = False  # Ollama につながったか (辞書に無い語の訳に使う)
        self._history_path: Path | None = None  # 今の結果の履歴ファイル (チップで消した結果を反映する)
        # 訳の表: 同梱の辞書 → dict フォルダの CSV → translations.json (Ollama で訳した語・利用者が直した訳) の順に上書き
        self._user_translations = load_translations()
        self._translations = {**load_tag_dictionary(), **self._user_translations}
        self._ja = False  # チップを日本語で見せているか

        root.title(f"{APP_NAME} v{__version__} - 画像 → プロンプト (ローカル)")
        try:
            root.iconbitmap(default=str(resource_path("assets/i2t.ico")))
        except tk.TclError:
            pass
        self._apply_geometry()
        self._build()
        self._fit_to_content()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        if initial_image:
            self.load_image(Path(initial_image))
        self._probe_ollama_async()
        root.after(100, self._poll)
        root.after(400, self._first_run_check)  # 初回起動: タガーモデルが無ければ取得を確認する

    @property
    def image_path(self) -> Path | None:
        return self.slot_paths["person"]

    # ---- ウィンドウ ----
    def _apply_geometry(self) -> None:
        """画面サイズに合わせた初期サイズにし、前回のサイズ・位置があれば復元する。"""
        root = self.root
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        width, height = min(1240, int(sw * 0.9)), min(940, int(sh * 0.88))
        x, y = max(0, (sw - width) // 2), max(0, (sh - height) // 2 - 20)
        m = re.fullmatch(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)", str(self.cfg.get("window_geometry") or ""))
        if m:
            w2, h2, x2, y2 = (int(v) for v in m.groups())
            if 600 <= w2 <= sw and 400 <= h2 <= sh and -50 <= x2 < sw - 200 and -50 <= y2 < sh - 200:
                width, height, x, y = w2, h2, x2, y2
        root.geometry(f"{width}x{height}+{x}+{y}")
        root.minsize(900, 600)

    def _fit_to_content(self) -> None:
        """設定の項目が増えて収まらないことがあるので、中身が全部入る大きさまでウィンドウを広げる (画面に収まる範囲で)。
        それでも入らない小さな画面では、左の設定欄をスクロールで見る。"""
        root = self.root
        root.update_idletasks()
        need_w = self._left_inner.winfo_reqwidth() + 700  # 右の結果欄にも十分な幅を残す
        need_h = (self._left_inner.winfo_reqheight() + self._top_bar.winfo_reqheight()
                  + self._status_bar.winfo_reqheight() + 30)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        max_w, max_h = int(sw * 0.96), int(sh * 0.90)  # タスクバーとタイトルバーの分を残す
        m = re.fullmatch(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)", root.geometry())
        if not m:
            return
        w, h, x, y = (int(v) for v in m.groups())
        new_w, new_h = min(max(w, need_w), max_w), min(max(h, need_h), max_h)
        if (new_w, new_h) == (w, h):
            return
        # 広げた分がはみ出すなら画面の内側に寄せる
        x = max(0, min(x, sw - new_w))
        y = max(0, min(y, sh - new_h - 40))
        root.geometry(f"{new_w}x{new_h}+{x}+{y}")

    def _save_cfg(self) -> None:
        """設定を保存する。書き込めなくても (読み取り専用のフォルダなど) 動作は続ける。"""
        try:
            save_config(self.cfg)
        except OSError:
            pass

    def _on_close(self) -> None:
        if self.busy:
            if not messagebox.askyesno(APP_NAME, "処理の途中です。中止して終了しますか？", parent=self.root):
                return
            if self._job_cancel is not None:
                self._job_cancel()  # 解析・ダウンロード・pull を止める (インストール中はインストーラが続行する)
        try:
            self.cfg["window_geometry"] = self.root.geometry()
        except tk.TclError:
            pass
        self._save_cfg()
        self.root.destroy()

    # ---- 画面構築 ----
    def _build(self) -> None:
        root = self.root
        outer = ttk.Frame(root, padding=8)
        outer.pack(fill="both", expand=True)

        top = ttk.Frame(outer)
        top.pack(fill="x", pady=(0, 4))
        self._top_bar = top
        ttk.Label(top, text="モード").pack(side="left")
        self.mode_var = tk.StringVar(value=self.cfg.get("mode") if self.cfg.get("mode") in MODES else "single")
        for key, label in MODES.items():
            ttk.Radiobutton(top, text=label, value=key, variable=self.mode_var, command=self._on_mode_change).pack(
                side="left", padx=(4, 0)
            )
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)
        self.choose_btn = ttk.Button(top, text="画像を選択...", command=lambda: self.choose_image("person"))
        self.choose_btn.pack(side="left")
        # 複数枚モードの枠ごとのボタン (① メイン...〜⑦ 作風・質感...) は置かない: 7 つ並ぶと上部がはみ出て見えなくなるため。
        # 枠の画像はプレビューの枠のクリック / ドロップ / Ctrl+V で選ぶ
        self.clear_btn = ttk.Button(top, text="✕ 画像をクリア", command=self.clear_images)
        ttk.Button(top, text="⚙ 設定...", command=self.open_settings).pack(side="right")
        ttk.Button(top, text="？ ヘルプ", command=self.open_help).pack(side="right", padx=(0, 4))
        root.bind("<F1>", lambda _e: self.open_help())
        self._help_win: Any = None
        self.path_var = tk.StringVar(value="画像が選択されていません")
        self.path_label = ttk.Label(top, textvariable=self.path_var, anchor="w")
        self.path_label.pack(side="left", fill="x", expand=True, padx=8)

        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        # 左の列 (プレビュー・設定・実行) は、画面に入りきらないときだけスクロールできるようにする
        left_outer = ttk.Frame(body)
        left_outer.pack(side="left", fill="y", padx=(0, 10))
        self._left_canvas = tk.Canvas(left_outer, highlightthickness=0, borderwidth=0)
        self._left_scroll = ttk.Scrollbar(left_outer, orient="vertical", command=self._left_canvas.yview)
        self._left_canvas.configure(yscrollcommand=self._left_scroll.set)
        self._left_canvas.pack(side="left", fill="y")
        left = ttk.Frame(self._left_canvas)
        self._left_inner = left
        self._left_canvas.create_window((0, 0), window=left, anchor="nw")

        def update_scroll(_event: Any = None) -> None:
            self._left_canvas.configure(scrollregion=(0, 0, left.winfo_reqwidth(), left.winfo_reqheight()),
                                        width=left.winfo_reqwidth())
            if left.winfo_reqheight() > self._left_canvas.winfo_height() > 1:
                if not self._left_scroll.winfo_manager():
                    self._left_scroll.pack(side="left", fill="y")
            elif self._left_scroll.winfo_manager():
                self._left_scroll.pack_forget()
                self._left_canvas.yview_moveto(0)

        left.bind("<Configure>", update_scroll)
        self._left_canvas.bind("<Configure>", update_scroll)

        def on_wheel(event: Any) -> None:
            if self._left_scroll.winfo_manager():
                self._left_canvas.yview_scroll(int(-event.delta / 120), "units")

        # マウスが左の列の上にあるときだけホイールでスクロール
        left_outer.bind("<Enter>", lambda _e: self.root.bind_all("<MouseWheel>", on_wheel))
        left_outer.bind("<Leave>", lambda _e: self.root.unbind_all("<MouseWheel>"))
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)

        # ---- 左: プレビュー ----
        self.canvas = tk.Canvas(
            left, width=PREVIEW_SIZE, height=PREVIEW_SIZE, bg="#202020",
            highlightthickness=1, highlightbackground="#888",
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Button-3>", self._on_canvas_right_click)
        self.preview_hint_var = tk.StringVar()
        ttk.Label(left, textvariable=self.preview_hint_var, foreground="#666", wraplength=PREVIEW_SIZE + 40,
                  justify="left").pack(anchor="w")
        self.preview_mosaic_var = tk.BooleanVar(value=bool(self.cfg.get("preview_mosaic", False)))

        # ---- 左: 構図と強さ (ボーン・カメラ・強さ。LLM・GPU 不要、今回の分析だけに効く) ----
        self._build_controls(left)

        # ---- 左: よく変える設定 (エンジン・出力先・モザイク指定・性別・白黒だけを本画面に。
        # ほかは上部の「⚙ 設定...」の画面へ。1280 × 720 の画面でも収まるように) ----
        s = ttk.LabelFrame(left, text="出力", padding=(6, 0))
        s.pack(fill="x", pady=(1, 0))
        s.columnconfigure(1, weight=1)
        pad = {"padx": (0, 6), "pady": 0}  # 三面図・人物の特徴を入れないの行を足しても 1152 × 633 に収まるよう、行の間隔は 0
        ttk.Label(s, text="エンジン").grid(row=0, column=0, sticky="w", **pad)
        self.engine_var = tk.StringVar(value=ENGINES[self.cfg["engine"]])
        ttk.Combobox(s, textvariable=self.engine_var, values=list(ENGINES.values()), state="readonly", width=34).grid(
            row=0, column=1, sticky="we", pady=0)
        self.engine_var.trace_add("write", lambda *_a: self._on_engine_change())
        ttk.Label(s, text="出力先").grid(row=1, column=0, sticky="w", **pad)
        self.target_labels = {v["label"]: k for k, v in TARGETS.items()}
        self.target_var = tk.StringVar(value=TARGETS[self.cfg["target"]]["label"])
        ttk.Combobox(s, textvariable=self.target_var, values=list(self.target_labels), state="readonly", width=34,
                     height=len(self.target_labels)).grid(  # 全部の出力先をスクロールなしで出す (既定の 10 行では MOSS-Image 以降が隠れる)
            row=1, column=1, sticky="we", pady=0)
        mrow = ttk.Frame(s)
        mrow.grid(row=2, column=0, columnspan=2, sticky="w", pady=0)
        self.mosaic_var = tk.BooleanVar(value=bool(self.cfg.get("mosaic", False)))
        ttk.Checkbutton(mrow, text="モザイク指定を追加  強さ", variable=self.mosaic_var).pack(side="left")
        self.mosaic_level_labels = {v["label"]: k for k, v in MOSAIC_LEVELS.items()}
        current_level = self.cfg.get("mosaic_strength") if self.cfg.get("mosaic_strength") in MOSAIC_LEVELS else DEFAULT_MOSAIC_LEVEL
        self.mosaic_level_var = tk.StringVar(value=MOSAIC_LEVELS[current_level]["label"])
        ttk.Combobox(mrow, textvariable=self.mosaic_level_var, values=list(self.mosaic_level_labels), state="readonly",
                     width=6).pack(side="left", padx=(4, 12))
        self.dup_var = tk.BooleanVar(value=bool(self.cfg.get("avoid_duplicates", False)))
        ttk.Checkbutton(mrow, text="分身を防ぐ", variable=self.dup_var).pack(side="left")
        grow = ttk.Frame(s)
        grow.grid(row=3, column=0, columnspan=2, sticky="w", pady=0)
        ttk.Label(grow, text="性別をそろえる").pack(side="left")
        self.gender_labels = {v: k for k, v in GENDER_MODES.items()}
        current_gender = self.cfg.get("gender") if self.cfg.get("gender") in GENDER_MODES else "none"
        self.gender_var = tk.StringVar(value=GENDER_MODES[current_gender])
        ttk.Combobox(grow, textvariable=self.gender_var, values=list(self.gender_labels), state="readonly", width=10).pack(
            side="left", padx=(6, 12))
        # 画風を変える (MOSS-Image だけ)。行を増やすと 1152 × 633 に収まらないので、性別の行に置く
        ttk.Label(grow, text="画風変更").pack(side="left")
        self.restyle_labels = {v: k for k, v in RESTYLE_MODES.items()}
        current_restyle = self.cfg.get("restyle") if self.cfg.get("restyle") in RESTYLE_MODES else "none"
        self.restyle_var = tk.StringVar(value=RESTYLE_MODES[current_restyle])
        ttk.Combobox(grow, textvariable=self.restyle_var, values=list(self.restyle_labels), state="readonly", width=8).pack(
            side="left", padx=(4, 8))
        self.color_var = tk.BooleanVar(value=bool(self.cfg.get("remove_monochrome", False)))
        ttk.Checkbutton(grow, text="白黒をカラーで出す", variable=self.color_var).pack(side="left")
        self.details_var = tk.BooleanVar(value=bool(self.cfg.get("person_details", False)))
        drow = ttk.Frame(s)
        drow.grid(row=4, column=0, columnspan=2, sticky="w", pady=0)
        ttk.Checkbutton(drow, text="人物の特徴を詳しく (顔立ち・髪型・体格。VLM)", variable=self.details_var,
                        command=self._on_details_toggle).pack(side="left")
        self.turnaround_var = tk.BooleanVar(value=bool(self.cfg.get("turnaround", False)))
        ttk.Checkbutton(drow, text="三面図にする", variable=self.turnaround_var).pack(side="left", padx=(12, 0))
        # LoRA でキャラ・画風を指定するとき。行を増やすと 1152 × 633 に収まらないので 1 行にまとめる
        trow = ttk.Frame(s)
        trow.grid(row=5, column=0, columnspan=2, sticky="w", pady=0)
        ttk.Label(trow, text="LoRA 用に入れない:").pack(side="left")
        self.no_identity_var = tk.BooleanVar(value=bool(self.cfg.get("no_identity", False)))
        ttk.Checkbutton(trow, text="人物の特徴", variable=self.no_identity_var).pack(side="left", padx=(6, 0))
        self.no_style_var = tk.BooleanVar(value=bool(self.cfg.get("no_style", False)))
        ttk.Checkbutton(trow, text="作風の特徴", variable=self.no_style_var).pack(side="left", padx=(12, 0))

        # ---- 頻繁に変えない設定 (「⚙ 設定...」の画面。起動時に作って隠しておき、開くときに見せる) ----
        self._build_settings_window()

        # ---- 下端: 実行と状態バー (分析ボタン・進捗・状態・Ollama。左の列の高さを取らないよう 1 行に) ----
        bar = ttk.Frame(outer)
        bar.pack(side="bottom", fill="x", pady=(4, 0), before=body)
        self._status_bar = bar
        self.run_btn = ttk.Button(bar, text="▶ 分析する", command=self.start_analysis, state="disabled", width=16)
        self.run_btn.pack(side="left")
        self.stop_btn = ttk.Button(bar, text="■ 中止", width=8, command=self.cancel_job, state="disabled")
        self.stop_btn.pack(side="left", padx=(4, 0))
        # 一時停止 (段階の切れ目で止める。分析と日本語訳)
        self.pause_btn = ttk.Button(bar, text="⏸ 一時停止", width=11, command=self.toggle_pause, state="disabled")
        self.pause_btn.pack(side="left", padx=(4, 10))
        self._job_gate: PauseGate | None = None
        self._job_cancel: Callable[[], None] | None = None  # 実行中の処理 (解析・ダウンロード・pull・導入) の中止
        root.bind("<Escape>", lambda _e: self.cancel_job())
        # 分析は段階ごとのパーセント (VLM の間は前回の時間からの見込み)。ダウンロードは実際の割合。それ以外は動いていることだけ
        # 待機中は割合の表示で空にしておく (不確定の表示は止めても左端に緑の塊が残るため)
        self.progress = ttk.Progressbar(bar, mode="determinate", maximum=1.0, value=0.0, length=150)
        self.progress.pack(side="left")
        self.pct_var = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.pct_var, width=6).pack(side="left", padx=(6, 0))
        self.ollama_var = tk.StringVar(value="Ollama: 確認中...")
        ttk.Label(bar, textvariable=self.ollama_var, foreground="#666").pack(side="right", padx=(8, 0))
        self.status_var = tk.StringVar(value="画像を選択してください")
        ttk.Label(bar, textvariable=self.status_var, anchor="w").pack(side="left", fill="x", expand=True, padx=(4, 0))

        # ---- 右: 結果 ----
        # ポジティブとネガティブは同じ高さ。右上に日本語 / 英語の切り替え (チップ表示のときだけ)
        self.pos_box = self._text_section(right, "ポジティブプロンプト", height=5, copy=True, grow=True)
        side = next(w for w in self.pos_box.master.winfo_children() if isinstance(w, ttk.Frame))  # コピーボタンの列
        self.ja_btn = ttk.Button(side, text="日本語", width=8, command=self.toggle_language)
        self.neg_box = self._text_section(right, "ネガティブプロンプト", height=5, copy=True, grow=True)
        self.pos_chips = PromptChips(self.pos_box, self._on_chips_change)
        self.neg_chips = PromptChips(self.neg_box, self._on_chips_change)
        self._apply_chip_mode()
        self.tabs = ttk.Notebook(right)
        self.tabs.pack(fill="both", expand=True, pady=(4, 2))
        self.extras_box = self._extras_tab()
        self.persons_box = self._tab_text("人物ごと", copy=True)
        self.tags_box = self._tab_text("検出タグ (信頼度)", copy=True)
        self.analysis_box = self._tab_text("分析メモ")
        self.notes_box = self._tab_text("メモ (推奨設定・コツ)")

        buttons = ttk.Frame(right)
        buttons.pack(fill="x", pady=(4, 0))
        ttk.Button(buttons, text="TXT に保存...", command=lambda: self.save("txt")).pack(side="left")
        ttk.Button(buttons, text="JSON に保存...", command=lambda: self.save("json")).pack(side="left", padx=6)
        ttk.Button(buttons, text="両方コピー", command=self.copy_both).pack(side="left", padx=6)
        ttk.Button(buttons, text="履歴...", command=self.open_history).pack(side="left", padx=6)
        self.gen_btn = ttk.Button(buttons, text="生成情報...", command=self.open_generation_info, state="disabled")
        self.gen_btn.pack(side="left", padx=6)
        # Ctrl+V: クリップボードの画像を読み込む (文字の入力欄では、文字があれば普通の貼り付け)
        root.bind_all("<Control-v>", self._on_paste_key, add="+")
        root.bind_all("<Control-V>", self._on_paste_key, add="+")

        if DND_AVAILABLE:
            self._register_dnd(root)
            self.status_var.set("画像を選択するか、ウィンドウにドロップしてください")
        self._on_mode_change(initial=True)

    def _text_section(self, parent: "ttk.Frame", title: str, height: int, copy: bool = False, grow: bool = False) -> "tk.Text":
        frame = ttk.LabelFrame(parent, text=title)
        frame.pack(fill="both", expand=grow, pady=3)
        box = tk.Text(frame, height=height, wrap="word", font=("Yu Gothic UI", 10), undo=True)
        if copy:  # ボタンを先に右端へ (文字の欄を先に置くと、窓が狭いときにボタンが消える)
            side = ttk.Frame(frame)
            side.pack(side="right", anchor="n", padx=4, pady=4)
            ttk.Button(side, text="コピー", width=8, command=lambda b=box: self.copy(b)).pack(side="bottom")
        box.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        return box

    def _tab_text(self, title: str, copy: bool = False) -> "tk.Text":
        frame = ttk.Frame(self.tabs, padding=4)
        self.tabs.add(frame, text=title)
        box = tk.Text(frame, height=6, wrap="word", font=("Yu Gothic UI", 10), undo=True)
        if copy:  # ボタンを先に右端へ (窓が狭いときに消えないように)
            ttk.Button(frame, text="コピー", width=8, command=lambda b=box: self.copy(b)).pack(side="right", anchor="n", padx=(4, 0))
        box.pack(side="left", fill="both", expand=True)
        return box

    def _extras_tab(self) -> "tk.Text":
        """追加プロンプト (カメラ / 照明 / 画風 / 動画カメラワーク) のタブ。項目ごとのコピーボタン付き。"""
        frame = ttk.Frame(self.tabs, padding=4)
        self.tabs.add(frame, text="追加プロンプト")
        for row_keys in EXTRA_BUTTON_ROWS:
            bar = ttk.Frame(frame)
            bar.pack(fill="x", pady=(0, 3))
            for key in row_keys:
                ttk.Button(
                    bar, text=f"{EXTRA_SHORT_JA[key]}をコピー", command=lambda k=key: self.copy_extra(k),
                ).pack(side="left", padx=(0, 4))
        box = tk.Text(frame, height=6, wrap="word", font=("Yu Gothic UI", 10), undo=True)
        box.pack(fill="both", expand=True)
        return box

    def _on_details_toggle(self) -> None:
        """人物の特徴を詳しく をオンにしたとき、顔の切り出しに使うモデルが無ければ確認してから取得する (初回のみ)。
        取得しなくても使える (人物全体の画像だけで読む)。"""
        missing = missing_face_models()
        if not self.details_var.get() or not missing:
            return
        if not messagebox.askyesno(
            APP_NAME, "顔を拡大して細部を読むには、次のモデルを取得します (初回のみ。CPU で動き、GPU は使いません):\n\n"
            + "\n".join("・" + m for m in missing) + f"\n\n保存先: {MODELS_DIR}\n"
            "取得しない場合は、人物全体の画像だけで読みます (顔の細部は読みにくくなります)。取得しますか？", parent=self.root):
            return
        if not self._start_busy("顔の切り出しに使うモデルを取得中... (初回のみ)"):
            return

        def work() -> None:
            try:
                ensure_face_models(lambda m: self.queue.put(("status", m)))
                self.queue.put(("dl_done", ("顔の切り出しモデル", None)))
            except Exception as e:  # noqa: BLE001 - 画面に出す
                self.queue.put(("error", e))

        threading.Thread(target=work, daemon=True).start()

    def copy_extra(self, key: str) -> None:
        extra = (self.result.extras.get(key) if self.result else None) or {}
        text = str(extra.get("prompt") or "").strip()
        if not text:
            self.status_var.set(f"{EXTRA_KINDS.get(key, key)} のプロンプトはまだありません")
            return
        self._to_clipboard(text)
        self.status_var.set(f"{EXTRA_KINDS.get(key, key)} をコピーしました")

    # ---- Ollama ----
    def _vlm_choices(self, detected: list[str]) -> list[str]:
        """検出したモデル + 手動で追加した名前 + 現在の設定値をまとめた候補一覧。"""
        extra = [m for m in (self.cfg.get("ollama_models") or []) if isinstance(m, str) and m.strip()]
        return dedup(list(detected) + extra + [str(self.cfg.get("ollama_model") or "")])

    def _probe_ollama_async(self) -> None:
        self.ollama_var.set("Ollama: 確認中...")
        threading.Thread(target=self._probe_ollama, daemon=True).start()

    def _probe_ollama(self) -> None:
        """Ollama に接続し、画像入力に対応したモデルだけをドロップダウンに載せる。"""
        try:
            host = self.cfg["ollama_host"]
            models = OllamaVLM(host, self.cfg["ollama_model"]).list_models()
            self.queue.put(("ollama", (models, None)))  # まず全モデルを即表示
            vision: list[str] = []
            for name in models:  # /api/show は Ollama が生成中だと遅いので後追いで絞り込む
                try:
                    if "vision" in OllamaVLM(host, name).capabilities():
                        vision.append(name)
                except OllamaError:
                    continue
            self.queue.put(("ollama", (models, vision)))
        except Exception as e:  # noqa: BLE001
            self.queue.put(("ollama_error", describe_error(e)))

    def _on_engine_change(self) -> None:
        """エンジンを切り替えたら保存し、「タガー + VLM」(拡張) にしたときは Ollama を確かめる。
        未接続なら _poll の ollama_error が導入サポートを案内する (既定はタガーのみなので、起動時には聞かない)。"""
        engine = {label: key for key, label in ENGINES.items()}.get(self.engine_var.get(), DEFAULT_CONFIG["engine"])
        if engine == self.cfg.get("engine"):
            return
        self.cfg["engine"] = engine
        self._save_cfg()
        if engine == "hybrid":
            self._probe_ollama_async()

    def reconnect_ollama(self) -> None:
        self._collect_settings()
        self._probe_ollama_async()

    def pull_vlm(self) -> None:
        """VLM 欄に入力されたモデルを Ollama に取得する (ollama pull)。"""
        if self.busy:
            return
        name = self.vlm_var.get().strip()
        if not name:
            messagebox.showinfo(
                APP_NAME, "追加するモデル名か Hugging Face の URL を VLM 欄に入力してください\n"
                "(例: qwen3-vl:8b-instruct / https://huggingface.co/ユーザー/リポジトリ)", parent=self.root,
            )
            return
        if parse_hf_model(name) is not None:
            self._check_hf_model(name)
            return
        self._collect_settings()
        if not messagebox.askyesno(
            APP_NAME,
            f"Ollama ({self.cfg['ollama_host']}) にモデル「{name}」を取得しますか？\n数 GB のダウンロードになることがあります。",
            parent=self.root,
        ):
            return
        self._start_pull(name)

    def pull_translate(self) -> None:
        """訳のモデル欄のモデルを Ollama に取得する (取得後も VLM 欄は変えない)。"""
        if self.busy:
            return
        name = self.translate_var.get().strip()
        if not name:
            return
        self._collect_settings()
        if not messagebox.askyesno(
            APP_NAME,
            f"Ollama ({self.cfg['ollama_host']}) に訳のモデル「{name}」を取得しますか？\n数 GB のダウンロードになることがあります。",
            parent=self.root,
        ):
            return
        self._start_pull(name, translate=True)

    def pull_describer(self) -> None:
        """説明のモデル欄のモデルを Ollama に取得する (取得後も VLM 欄は変えない)。"""
        if self.busy:
            return
        name = self.describer_var.get().strip()
        if not name:
            return
        self._collect_settings()
        if not messagebox.askyesno(
            APP_NAME,
            f"Ollama ({self.cfg['ollama_host']}) に説明のモデル「{name}」を取得しますか？\n数 GB のダウンロードになることがあります。",
            parent=self.root,
        ):
            return
        self._start_pull(name, describer=True)

    def _check_hf_model(self, text: str) -> None:
        """Hugging Face の URL: 公開 API でファイル一覧を確かめてから (別スレッド)、取得してよいか聞く。"""
        self._collect_settings()
        if not self._start_busy("Hugging Face のリポジトリを確認中..."):
            return

        def worker() -> None:
            try:
                self.queue.put(("hf_checked", check_hf_model(text)))
            except HfModelError as e:
                self.queue.put(("pull_error", str(e)))
            except Exception as e:  # noqa: BLE001
                self.queue.put(("pull_error", describe_error(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_hf_checked(self, info: dict[str, Any]) -> None:
        self._finish()
        lines = [f"Hugging Face の「{info['repo']}」を Ollama ({self.cfg['ollama_host']}) に取得しますか？", ""]
        if info["file"]:
            size = f" ({info['size'] / 1e9:.1f} GB)" if info["size"] else ""
            lines.append(f"取得するファイル: {info['file'].rsplit('/', 1)[-1]}{size} と画像用ファイル (mmproj)")
        else:
            lines.append("取得するファイル: Ollama がリポジトリの中から選びます (Q4_K_M が無いため)")
        lines.append(f"Ollama での名前: {info['name']}")
        if info["gated"]:
            lines += ["", "このリポジトリは利用条件への同意が必要です (gated)。Hugging Face で同意し、Ollama の SSH 鍵を"
                          " Hugging Face に登録していないと取得に失敗します。"]
        if not messagebox.askyesno(APP_NAME, "\n".join(lines), parent=self.root):
            self.status_var.set("取得をやめました")
            return
        self.vlm_var.set(info["name"])
        self._start_pull(info["name"])

    def _start_pull(self, name: str, translate: bool = False, describer: bool = False) -> None:
        """確認済みのモデル名を Ollama に取得する (導入サポートからも呼ばれる)。translate: 訳のモデルとして、
        describer: 説明のモデルとして取得する (どちらも VLM 欄は変えない)。"""
        cancel = threading.Event()
        if not self._start_busy(f"{name} を取得中...", cancel.set):
            return
        host = self.cfg["ollama_host"]

        def worker() -> None:
            try:
                OllamaVLM(host, name).pull(name, progress=lambda m: self.queue.put(("status", m)), cancel=cancel)
                self.queue.put(("pulled_describer" if describer else "pulled_translate" if translate else "pulled", name))
            except OperationCancelled:
                self.queue.put(("cancelled", f"{name} の取得を中止しました (もう一度「追加 (pull)」を押すと続きから取得します)"))
            except Exception as e:  # noqa: BLE001
                self.queue.put(("pull_error", describe_error(e)))

        threading.Thread(target=worker, daemon=True).start()

    # ---- Ollama 導入サポート ----
    def open_ollama_setup(self) -> None:
        """状態と PC スペックを調べてから導入サポート画面を開く。既に入っていれば再インストールはしない。"""
        if self.busy:
            return
        self._collect_settings()
        self._start_busy("Ollama の状態と PC のスペックを確認中...")
        host, model = self.cfg["ollama_host"], self.cfg["ollama_model"]

        def worker() -> None:
            try:
                self.queue.put(("ollama_setup_ready", gather_ollama_info(host, model)))
            except Exception as e:  # noqa: BLE001
                self.queue.put(("ollama_setup_error", describe_error(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_ollama_setup_ready(self, info: dict[str, Any]) -> None:
        self._finish()
        self._modal_open = True
        try:
            result = OllamaSetupDialog(self.root, self.cfg, info).show()
        finally:
            self._modal_open = False
        if not result:
            self.status_var.set("Ollama の導入サポートを閉じました")
            return
        self.cfg["ollama_model"] = result["model"]
        self.vlm_var.set(result["model"])
        self._save_cfg()
        cancel = threading.Event()
        if not self._start_busy("Ollama の準備を始めます...", cancel.set):
            return
        host = self.cfg["ollama_host"]
        current_models_dir = Path(str(info.get("models_dir") or ollama_models_dir()))

        def report(msg: str, frac: float = -1.0) -> None:
            self.queue.put(("dl_progress", (msg, frac)))

        def worker() -> None:
            tmp_dir: Path | None = None
            try:
                exe = info.get("exe")
                if result["action"] == "install":
                    import tempfile

                    # 固定名の %TEMP%\\OllamaSetup.exe だと実行までの間に差し替えられうるので、専用の一時フォルダに置く
                    tmp_dir = Path(tempfile.mkdtemp(prefix="i2t_ollama_"))
                    installer = tmp_dir / "OllamaSetup.exe"
                    report(f"Ollama のインストーラをダウンロード中 (約 {OLLAMA_INSTALLER_GB:.1f} GB)...", -1.0)
                    download_file(OLLAMA_INSTALLER_URL, installer, report, label="OllamaSetup.exe", cancel=cancel)
                    if cancel.is_set():
                        raise OperationCancelled()
                    # ここから先 (インストール・起動) は途中で止めると壊れるので中止を受け付けない
                    self.queue.put(("stop_state", False))
                    if result["models_dir"] != current_models_dir:
                        set_ollama_models_dir(result["models_dir"])
                    exe = install_ollama(installer, result["install_dir"], lambda m: report(m, -1.0))
                    report("Ollama を起動しています...", -1.0)
                    start_ollama(exe)
                elif result["action"] == "start":
                    self.queue.put(("stop_state", False))
                    if result["models_dir"] != current_models_dir:
                        set_ollama_models_dir(result["models_dir"])
                    report("Ollama を起動しています...", -1.0)
                    start_ollama(Path(exe))
                elif result["models_dir"] != current_models_dir:
                    set_ollama_models_dir(result["models_dir"])
                    report("モデル保存先を変更しました。Ollama を再起動すると反映されます", -1.0)
                if result["action"] in ("install", "start") and not wait_ollama(host, timeout=90):
                    raise OllamaError("Ollama を起動しましたが応答がありません。しばらく待ってから「再接続」を押してください。")
                self.queue.put(("ollama_setup_done", result))
            except OperationCancelled:
                self.queue.put(("cancelled", "Ollama の導入を中止しました (インストーラのダウンロード途中のファイルは削除しました)"))
            except Exception as e:  # noqa: BLE001
                self.queue.put(("ollama_setup_error", describe_error(e)))
            finally:
                if tmp_dir is not None:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

        threading.Thread(target=worker, daemon=True).start()

    # ---- タガーモデルの取得 ----
    @staticmethod
    def _tagger_label(name: str) -> str:
        desc = TAGGER_MODELS[name][1]
        return f"{name}  ({desc}{' / 取得済み' if tagger_ready(name) else ''})"

    def _refresh_tagger_labels(self) -> None:
        """取得状況が変わったあとにタガー一覧の表示を作り直す。"""
        self.tagger_labels = {self._tagger_label(k): k for k in TAGGER_MODELS}
        self.tagger_box["values"] = list(self.tagger_labels)
        self.tagger_var.set(self._tagger_label(self.cfg["tagger_model"]))

    def _first_run_check(self) -> None:
        if self.busy or tagger_ready(self.cfg["tagger_model"]):
            return
        if any(tagger_ready(n) for n in TAGGER_MODELS):
            reason = f"設定されているタガーモデル「{self.cfg['tagger_model']}」がまだ取得されていません。"
        else:
            reason = "はじめまして。タガーモデルがまだこの PC にありません。"
        self._ask_download(reason)

    def _ask_download(self, reason: str, on_done: Callable[[], None] | None = None) -> None:
        """取得するモデルを選ばせ、OK なら取得して設定に紐づける。取得済みを選んだ場合はそのまま紐づける。"""
        self._modal_open = True
        try:
            chosen = TaggerDownloadDialog(self.root, self.cfg["tagger_model"], reason).show()
        finally:
            self._modal_open = False
        if not chosen:
            self.status_var.set("タガーモデルが未取得のため解析できません。「▶ 分析する」を押すと再度確認します")
            return
        self.cfg["tagger_model"] = chosen
        self._refresh_tagger_labels()
        self._save_cfg()
        if tagger_ready(chosen):
            self.status_var.set(f"タガーモデル {chosen} を使用します")
            if on_done:
                on_done()
            return
        self._start_tagger_download(chosen, on_done)

    def _start_tagger_download(self, name: str, on_done: Callable[[], None] | None) -> None:
        cancel = threading.Event()
        if not self._start_busy(f"{name} ({tagger_size_label(name)}) をダウンロード中...", cancel.set, determinate=True):
            return

        def report(msg: str, frac: float) -> None:
            self.queue.put(("dl_progress", (msg, frac)))

        def worker() -> None:
            try:
                download_tagger(name, report, cancel)
                self.queue.put(("dl_done", (name, on_done)))
            except OperationCancelled:
                self.queue.put(("cancelled", "タガーモデルのダウンロードを中止しました (「▶ 分析する」を押すと再度確認します)"))
            except Exception as e:  # noqa: BLE001
                self.queue.put(("dl_error", describe_error(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _start_moss_catalog_download(self, on_done: Callable[[], None] | None) -> None:
        cancel = threading.Event()
        if not self._start_busy(f"MOSS-Image のキャラ一覧 (約 {MOSS_CATALOG_MB}MB) を取得中...", cancel.set, determinate=True):
            return

        def worker() -> None:
            try:
                download_moss_catalog(lambda msg, frac: self.queue.put(("dl_progress", (msg, frac))), cancel)
                self.queue.put(("dl_done", ("MOSS-Image のキャラ一覧", on_done)))
            except OperationCancelled:
                self.queue.put(("cancelled", "キャラ一覧の取得を中止しました (キャラ名はそのまま書きます。次に起動したときにもう一度確かめます)"))
            except Exception as e:  # noqa: BLE001
                self._moss_catalog_asked = False  # 失敗したら次の「▶ 分析する」でもう一度確かめる
                self.queue.put(("dl_error", describe_error(e)))

        threading.Thread(target=worker, daemon=True).start()

    # ---- ドラッグ＆ドロップ ----
    def _register_dnd(self, widget: Any) -> None:
        """ウィンドウ内のどこにドロップしても受け取れるよう、全ウィジェットを登録する。"""
        try:
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<DropEnter>>", self._on_drag_enter)
            widget.dnd_bind("<<DropLeave>>", self._on_drag_leave)
            widget.dnd_bind("<<Drop>>", self._on_drop)
        except (tk.TclError, AttributeError):
            pass
        for child in widget.winfo_children():
            self._register_dnd(child)

    def _on_drag_enter(self, event: Any) -> Any:
        self.canvas.config(highlightbackground="#3b82f6", highlightthickness=3)
        return event.action

    def _on_drag_leave(self, event: Any) -> Any:
        self.canvas.config(highlightbackground="#888", highlightthickness=1)
        return event.action

    def _on_drop(self, event: Any) -> Any:
        self._on_drag_leave(event)
        if self.busy:
            self.status_var.set("処理の途中は画像を変更できません (「■ 中止」で止められます)")
            return event.action
        try:
            raw_paths = self.root.tk.splitlist(event.data)
        except tk.TclError:
            # Tcl のリストとして読めない (括弧の対応が崩れた名前など) ときは 1 つのパスとみなし、外側の {} だけ外す
            data = str(event.data)
            raw_paths = [data[1:-1] if data.startswith("{") and data.endswith("}") else data]
        # splitlist が {} を外しているので、ここでは削らない (末尾が } の名前を壊さない)
        self._open_files([Path(str(raw)) for raw in raw_paths], event)
        return event.action

    def _open_files(self, paths: list[Path], event: Any) -> None:
        """ドロップ / 貼り付けたファイル・フォルダを読み込む。event は x_root / y_root (複数枚モードで入れる枠) を持つ。"""
        files: list[Path] = []
        for p in paths:
            if p.is_dir():  # フォルダなら中の画像を先頭から
                try:
                    files.extend(sorted(c for c in p.iterdir() if c.suffix.lower() in IMAGE_EXTS))
                except OSError as e:
                    self.status_var.set(f"フォルダを読めません: {p} ({e})")
                    return
            elif p.is_file():
                files.append(p)
        if not files:
            self.status_var.set("画像ファイルが見つかりませんでした")
            return
        # 拡張子が未知でも画像かもしれないので試す。失敗時はダイアログを出さない
        images = [f for f in files if f.suffix.lower() in IMAGE_EXTS] or files[:1]
        if self.multi_mode():
            self._drop_multi(images, event)
            return
        if self.load_image(images[0], quiet=True) and len(images) > 1:
            self.status_var.set(f"{len(images)} 件のうち先頭の画像を読み込みました (GUI は 1 枚ずつ。複数枚は CLI で)")

    # ---- 貼り付け (Ctrl+V) ----
    def _on_paste_key(self, event: Any) -> Any:
        widget = event.widget
        try:
            if widget.winfo_toplevel() is not self.root:  # 設定・履歴などのほかのウィンドウでは普通の貼り付けだけ
                return None
        except (AttributeError, tk.TclError):
            return None
        if isinstance(widget, (tk.Text, tk.Entry, ttk.Entry)):
            # 文字の入力欄: クリップボードに文字があれば、もう普通に貼り付けられている (画像は読み込まない)
            try:
                self.root.clipboard_get()
                return None
            except tk.TclError:
                pass
        self.paste_image(quiet_if_empty=isinstance(widget, (tk.Text, tk.Entry, ttk.Entry)))
        return "break"

    def paste_image(self, quiet_if_empty: bool = False) -> None:
        """クリップボードの画像を読み込む。複数枚モードではマウスの下の枠、外なら空いている最初の枠。"""
        grabbed = clipboard_image()
        if grabbed is None:
            if not quiet_if_empty:
                self.status_var.set("クリップボードに画像がありません")
            return
        if self.busy:
            self.status_var.set("処理の途中は画像を変更できません (「■ 中止」で止められます)")
            return
        px, py = self.root.winfo_pointerxy()
        pointer = _Pointer(px, py)
        if isinstance(grabbed, list):  # エクスプローラーでコピーしたファイル: ドロップと同じ扱い
            self._open_files(grabbed, pointer)
            return
        try:
            path = save_pasted_image(grabbed)
        except (OSError, ValueError) as e:
            self.status_var.set(f"貼り付けた画像を保存できません ({PASTE_DIR}): {e}")
            return
        if self.multi_mode():
            self._drop_multi([path], pointer)
        else:
            self.load_image(path, quiet=True)
        if any(p == path for p in self.slot_paths.values()):
            self.status_var.set(f"貼り付けました ({path.name} として一時保存)。" + self.status_var.get())

    SLOT_COLUMNS = 4  # 複数枚モードのプレビューは 4 列 × 2 段 (上: ① ② ③ ④、下: ⑤ ⑥ ⑦ ⑧)

    def _slot_cell(self, index: int) -> tuple[int, int, int, int]:
        """枠の番号 → プレビュー上の四角 (x0, y0, 幅, 高さ)。"""
        cw, ch = PREVIEW_SIZE // self.SLOT_COLUMNS, PREVIEW_SIZE // 2
        return (index % self.SLOT_COLUMNS) * cw, (index // self.SLOT_COLUMNS) * ch, cw, ch

    def _slot_at(self, x: int, y: int) -> str | None:
        """複数枚モードのプレビュー上の座標 → 枠名 (上段 ① メイン / ② ポーズ / ③ 服装 / ④ 背景、下段 ⑤ 小物 / ⑥ 体格 / ⑦ 作風・質感 / ⑧ 表情)。"""
        if not (0 <= x < PREVIEW_SIZE and 0 <= y < PREVIEW_SIZE):
            return None
        cw, ch = PREVIEW_SIZE // self.SLOT_COLUMNS, PREVIEW_SIZE // 2
        index = min(y // ch, 1) * self.SLOT_COLUMNS + min(x // cw, self.SLOT_COLUMNS - 1)
        slots = list(REF_SLOTS)
        return slots[index] if index < len(slots) else None

    def _drop_multi(self, images: list[Path], event: Any) -> None:
        """複数枚モードのドロップ。複数ファイルは ① → ⑥ の順に入れる。1 枚ならプレビューの落とした枠、
        プレビューの外なら空いている最初の枠へ。"""
        if len(images) >= 2:
            for slot in REF_SLOTS:  # 入らなかった枠に前の画像が残って知らずに混ざらないよう、先に空にする
                self._clear_slot(slot)
            for slot, image in zip(REF_SLOTS, images, strict=False):
                self.load_image(image, quiet=True, slot=slot)
            if len(images) > len(REF_SLOTS):
                self.status_var.set(f"{len(images)} 件のうち先頭の {len(REF_SLOTS)} 枚を読み込みました")
            return
        slot = None
        try:
            slot = self._slot_at(int(event.x_root) - self.canvas.winfo_rootx(), int(event.y_root) - self.canvas.winfo_rooty())
        except (AttributeError, ValueError, TypeError, tk.TclError):
            pass
        if slot is None:
            slot = next((k for k in REF_SLOTS if self.slot_paths[k] is None), None)
        if slot is None:
            self.status_var.set("差し替えるときは、プレビューの差し替えたい枠 (上段 ①〜④、下段 ⑤〜⑧) にドロップしてください")
            return
        self.load_image(images[0], quiet=True, slot=slot)

    def _on_canvas_click(self, event: Any) -> None:
        """複数枚モード: 枠をクリックするとその枠の画像を選ぶ。"""
        if self.multi_mode() and not self.busy:
            slot = self._slot_at(event.x, event.y)
            if slot:
                self.choose_image(slot)

    def _on_canvas_right_click(self, event: Any) -> None:
        """プレビューを右クリックすると画像を外す。複数枚モードではその枠だけ (空の枠は 1 枚目から取る)。"""
        if self.busy:
            return
        slot = self._slot_at(event.x, event.y) if self.multi_mode() else "person"
        if not slot or self.slot_paths[slot] is None:
            return
        if not self.multi_mode():
            self.remove_slot(slot)
            return
        # 複数枚モード: 右クリックでメニュー (外す / この画像をほかの枠にも使う)
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label=f"{REF_SLOTS[slot]} を外す", command=lambda: self.remove_slot(slot))
        menu.add_separator()
        for other, label in REF_SLOTS.items():
            if other == slot:
                continue
            same = self.slot_paths[other] == self.slot_paths[slot]
            note = " (同じ画像)" if same else " (差し替え)" if self.slot_paths[other] is not None else ""
            menu.add_command(label=f"この画像を {label} にも使う{note}", state="disabled" if same else "normal",
                             command=lambda o=other: self.copy_slot(slot, o))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def remove_slot(self, slot: str) -> None:
        if self.busy or self.slot_paths[slot] is None:
            return
        self._clear_slot(slot)
        self._render_preview()
        self._update_path_label()
        self._update_run_state()
        self.status_var.set(f"{REF_SLOTS[slot]} を外しました" + ("" if slot == "person" else " (1 枚目から取ります)"))

    def copy_slot(self, src: str, dst: str) -> None:
        """同じ画像をほかの枠にも使う (ポーズと服装を同じ画像から、など)。"""
        if self.busy or self.slot_paths[src] is None:
            return
        self.slot_paths[dst], self.slot_thumbs[dst] = self.slot_paths[src], self.slot_thumbs[src]
        self.slot_infos[dst] = self.slot_infos[src]
        self.slot_gen[dst] = self.slot_gen[src]
        if src in self.slot_sizes:
            self.slot_sizes[dst] = self.slot_sizes[src]
        self._render_preview()
        self._update_path_label()
        self._update_run_state()
        self.status_var.set(f"{REF_SLOTS[src]} の画像を {REF_SLOTS[dst]} にも使います")

    def _clear_slot(self, slot: str) -> None:
        self.slot_sizes.pop(slot, None)
        self.slot_paths[slot] = None
        self.slot_thumbs[slot] = None
        self.slot_infos[slot] = ""
        self.slot_gen[slot] = None

    # ---- モード ----
    def multi_mode(self) -> bool:
        return self.mode_var.get() == "multi"

    def clear_images(self) -> None:
        """選んだ画像をすべて外す (解析中は受け付けない)。結果の欄はそのまま残す。"""
        if self.busy:
            return
        for slot in REF_SLOTS:
            self._clear_slot(slot)
        self._render_preview()
        self._update_path_label()
        self._update_run_state()
        self.status_var.set("画像をクリアしました")

    def _on_mode_change(self, initial: bool = False) -> None:
        multi = self.multi_mode()
        self.preview_hint_var.set(
            "クリックで選択、右クリックで外す / ほかの枠にも使う"
            if multi else "右クリックで画像を外す"
        )
        self.clear_btn.pack_forget()
        if multi:
            self.choose_btn.pack_forget()
        else:
            self.choose_btn.pack(side="left", before=self.path_label)
        self.clear_btn.pack(side="left", padx=(4, 0), before=self.path_label)
        if not initial:
            self.cfg["mode"] = self.mode_var.get()
            self._save_cfg()
        self._update_path_label()
        self._render_preview()
        self._update_run_state()
        if not self.busy and not initial:
            if multi:
                self.status_var.set(
                    "① メインは必須、②〜⑧ は任意です (空の枠は 1 枚目から取ります)。プレビューの各枠にドロップ / "
                    "クリックで選択、右クリックで外す・ほかの枠にも使う"
                )
            elif self.image_path:
                self.status_var.set("「▶ 分析する」を押してください")

    def _ready(self) -> bool:
        return self.image_path is not None

    def _update_run_state(self) -> None:
        if not self.busy:
            self.run_btn.config(state="normal" if self._ready() else "disabled")

    def _update_path_label(self) -> None:
        self.gen_btn.config(state="normal" if self.generation_infos() else "disabled")
        if self.multi_mode():
            parts = []
            for slot, label in REF_SLOTS.items():
                path = self.slot_paths[slot]
                if path:
                    parts.append(f"{label.split()[0]} {path.name} ({self.slot_infos[slot]})")
                else:
                    parts.append(f"{label.split()[0]} " + ("未選択" if slot == "person" else "(1 枚目)"))
            self.path_var.set("   ".join(parts))
        elif self.image_path:
            self.path_var.set(f"{self.image_path}  ({self.slot_infos['person']})")
        else:
            self.path_var.set("画像が選択されていません")

    def selected_refs(self) -> dict[str, Path]:
        """複数枚モードで選ばれている参照画像 (① メインを除く)。1 枚モードなら空。"""
        if not self.multi_mode():
            return {}
        return {slot: p for slot, p in self.slot_paths.items() if slot != "person" and p is not None}

    # ---- 画像 ----
    def choose_image(self, slot: str = "person") -> None:
        if self.busy:
            return
        initial = self.cfg.get("last_dir") or str(Path.home())
        patterns = " ".join(f"*{ext}" for ext in IMAGE_EXTS)
        title = f"{REF_SLOTS[slot]} の画像を選択" if self.multi_mode() else "画像を選択"
        chosen = filedialog.askopenfilename(
            title=title,
            initialdir=initial if Path(initial).is_dir() else str(Path.home()),
            filetypes=[("画像ファイル", patterns), ("すべてのファイル", "*.*")],
        )
        if chosen:
            self.load_image(Path(chosen), slot=slot)

    def load_image(self, path: Path, quiet: bool = False, slot: str = "person") -> bool:
        """画像をプレビューに読み込む。quiet=True ならダイアログを出さずステータス表示だけにする。
        slot: person = 1 枚モードの画像 / 複数枚モードの ① メイン、pose / outfit / background / props / body / style / expression = ②〜⑧ の枠。"""
        if self.busy:
            return False
        try:
            with Image.open(path) as im:
                im.load()
                gen_info = generation_info_from_image(im)
                shown = upright(im)
                width, height = shown.size
                thumb = shown.convert("RGB")
                thumb.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE), Image.LANCZOS)
        except Exception as e:  # noqa: BLE001
            message = f"画像を開けません: {path.name} ({describe_error(e)})"
            if quiet:
                self.status_var.set(message)
            else:
                messagebox.showerror(APP_NAME, message, parent=self.root)
            return False

        if path.parent != PASTE_DIR:  # 貼り付けた画像の一時フォルダは「画像を選択」の開始場所にしない
            self.cfg["last_dir"] = str(path.parent)
        slot = slot if slot in REF_SLOTS else "person"
        self.slot_paths[slot], self.slot_thumbs[slot], self.slot_infos[slot] = path, thumb, f"{width}×{height}"
        self.slot_sizes[slot] = (width, height)
        self.slot_gen[slot] = gen_info
        self._render_preview()
        self._update_path_label()
        self._update_run_state()
        if self._gen_win is not None:
            self._close_generation_info()  # 前の画像の生成情報を出したままにしない
        if self._ready():
            given = {k for k, v in self.slot_paths.items() if v is not None}
            empty = [k for k in REF_SLOTS if k != "person" and k not in given]
            from_first = [REF_SLOTS[k] for k in empty if slot_owner(k, given) == "person"]
            from_outfit = [REF_SLOTS[k] for k in empty if slot_owner(k, given) == "outfit"]
            parts = ([f"{'、'.join(from_first)} は 1 枚目から"] if from_first else []) + \
                    ([f"{'、'.join(from_outfit)} は ③ から"] if from_outfit else [])
            note = f" ({'、'.join(parts)}取ります)" if self.multi_mode() and parts else ""
            self.status_var.set("「▶ 分析する」を押してください" + note)
        elif self.multi_mode():
            self.status_var.set("① メインの画像も選んでください (プレビュー左上にドロップ / クリック)")
        if gen_info:
            self.status_var.set(self.status_var.get() + f"  / 生成情報あり ({gen_info['source']}。「生成情報...」で表示)")
        return True

    def _render_preview(self) -> None:
        """保持しているサムネイルをプレビューに描く。モザイク表示が有効なら粗くピクセル化する (表示だけ)。"""
        from PIL import ImageTk

        def mosaic(image: Image.Image, cell: int) -> Image.Image:
            w, h = image.size
            return image.resize((max(1, w // cell), max(1, h // cell)), Image.BOX).resize((w, h), Image.NEAREST)

        cell = max(20, PREVIEW_SIZE // 8)  # 320px のプレビューで 1 マス 40px (以前の倍の粗さ)
        if self.multi_mode():
            # 複数枚モード: 4 列 × 2 段の枠 (上段 ① メイン〜④ 背景、下段 ⑤ 小物 / ⑥ 体格 / ⑦ 作風・質感 / ⑧ 表情)。枠が狭いので見出しは短く
            self.canvas.delete("all")
            sheet = Image.new("RGB", (PREVIEW_SIZE, PREVIEW_SIZE), (32, 32, 32))
            for i, slot in enumerate(REF_SLOTS):
                source = self.slot_thumbs[slot]
                if source is None:
                    continue
                x0, y0, cw, ch = self._slot_cell(i)
                thumb = source.copy()
                thumb.thumbnail((cw - 6, ch - 22), Image.LANCZOS)
                if self.preview_mosaic_var.get():
                    thumb = mosaic(thumb, max(5, cell // 4))
                sheet.paste(thumb, (x0 + (cw - thumb.width) // 2, y0 + 18 + (ch - 18 - thumb.height) // 2))
            self._preview = ImageTk.PhotoImage(sheet)
            self.canvas.create_image(PREVIEW_SIZE // 2, PREVIEW_SIZE // 2, image=self._preview)
            _, _, cw, ch = self._slot_cell(0)
            for c in range(1, self.SLOT_COLUMNS):
                self.canvas.create_line(c * cw, 0, c * cw, PREVIEW_SIZE, fill="#666", dash=(3, 3))
            self.canvas.create_line(0, ch, PREVIEW_SIZE, ch, fill="#666", dash=(3, 3))
            for i, (slot, label) in enumerate(REF_SLOTS.items()):
                x0, y0, cw, ch = self._slot_cell(i)
                self.canvas.create_text(x0 + cw // 2, y0 + 9, text=label.split("・")[0], fill="#ddd",
                                        font=("Yu Gothic UI", 9, "bold"))
                same = next((other for other in list(REF_SLOTS)[:i] if self.slot_paths[slot] is not None
                             and self.slot_paths[other] == self.slot_paths[slot]), None)
                if same:  # 同じ画像を使っている枠が分かるように (画像の上でも読めるよう暗い帯を敷く)
                    self.canvas.create_rectangle(x0 + 2, y0 + ch - 20, x0 + cw - 2, y0 + ch - 1, fill="#202020",
                                                 outline="#ffd966")
                    self.canvas.create_text(x0 + cw // 2, y0 + ch - 10, text=f"= {REF_SLOTS[same].split()[0]} と同じ",
                                            fill="#ffd966", font=("Yu Gothic UI", 9, "bold"))
                if self.slot_thumbs[slot] is None:
                    empty = {"person": "必須", "props": "空なら ③\nか 1 枚目"}.get(slot, "空なら\n1 枚目")
                    hint = empty + ("\nドロップ /\nクリック" if DND_AVAILABLE else "\nクリック")
                    self.canvas.create_text(x0 + cw // 2, y0 + ch // 2 + 8, text=hint, fill="#777", justify="center",
                                            font=("Yu Gothic UI", 9))
            return
        source = self.slot_thumbs["person"]
        self.canvas.delete("all")
        if source is None:
            placeholder = "ここに画像をドロップ\nまたは「画像を選択...」" if DND_AVAILABLE else "プレビュー"
            self.canvas.create_text(
                PREVIEW_SIZE // 2, PREVIEW_SIZE // 2, text=placeholder, fill="#777", justify="center",
                font=("Yu Gothic UI", 11),
            )
            return
        image = mosaic(source, cell) if self.preview_mosaic_var.get() else source
        self._preview = ImageTk.PhotoImage(image)
        self.canvas.create_image(PREVIEW_SIZE // 2, PREVIEW_SIZE // 2, image=self._preview)
        self._draw_person_boxes(source.size)

    def _draw_person_boxes(self, thumb_size: tuple[int, int]) -> None:
        """検出した人物の枠をプレビューに描く (今表示している画像の結果のときだけ)。
        メインの 1 人だけなら選んだ人を赤枠、人物ごとに分けたなら左から番号付きの枠。"""
        result = self.result
        size = self.slot_sizes.get("person")
        if not (result and size and Path(result.image) == self.image_path):
            return
        scale = thumb_size[0] / size[0]
        ox, oy = (PREVIEW_SIZE - thumb_size[0]) / 2, (PREVIEW_SIZE - thumb_size[1]) / 2
        boxes: list[tuple[list[int], str, str]] = []
        if result.main_person and result.main_person.get("box"):
            boxes.append((result.main_person["box"], "", "#ff4d4d"))
        colors = ("#ff4d4d", "#3b82f6", "#22c55e", "#f59e0b", "#a855f7", "#14b8a6")
        for i, person in enumerate(result.persons):
            if person.get("box"):
                boxes.append((person["box"], str(i + 1), colors[i % len(colors)]))
        for box in result.background_people:  # 奥の人 (人数に数えない) は灰色の点線
            x0, y0, x1, y1 = (v * scale for v in box)
            self.canvas.create_rectangle(ox + x0, oy + y0, ox + x1, oy + y1, outline="#9ca3af", width=1, dash=(3, 3))
        for box, label, color in boxes:
            x0, y0, x1, y1 = (v * scale for v in box)
            self.canvas.create_rectangle(ox + x0, oy + y0, ox + x1, oy + y1, outline=color, width=2)
            if label:
                self.canvas.create_text(ox + x0 + 3, oy + y0 + 2, text=label, anchor="nw", fill=color,
                                        font=("Yu Gothic UI", 10, "bold"))

    def _toggle_chips(self) -> None:
        self.cfg["result_chips"] = bool(self.chips_var.get())
        self._save_cfg()
        self._apply_chip_mode()

    def _toggle_preview_mosaic(self) -> None:
        self.cfg["preview_mosaic"] = bool(self.preview_mosaic_var.get())
        self._save_cfg()
        self._render_preview()

    # ---- 解析 ----
    def _collect_settings(self, normalize: bool = True) -> None:
        """画面の設定を cfg に入れて保存する。normalize=False (設定画面で入力中) なら、しきい値の表示は書き換えない。"""
        engines = {label: key for key, label in ENGINES.items()}
        self.cfg["engine"] = engines.get(self.engine_var.get(), DEFAULT_CONFIG["engine"])
        self.cfg["tagger_model"] = self.tagger_labels.get(self.tagger_var.get(), DEFAULT_TAGGER)
        self.cfg["target"] = self.target_labels.get(self.target_var.get(), DEFAULT_TARGET)
        self.cfg["quality_tags"] = bool(self.quality_var.get())
        self.cfg["use_break"] = bool(self.break_var.get())
        self.cfg["mosaic"] = bool(self.mosaic_var.get())
        self.cfg["mosaic_strength"] = self.mosaic_level_labels.get(self.mosaic_level_var.get(), DEFAULT_MOSAIC_LEVEL)
        self.cfg["split_extras"] = bool(self.split_var.get())
        self.cfg["split_background"] = bool(self.split_bg_var.get())
        self.cfg["polish_prose"] = bool(self.polish_var.get())
        self.cfg["age_look"] = bool(self.age_var.get())
        self.cfg["preview_mosaic"] = bool(self.preview_mosaic_var.get())
        self.cfg["result_chips"] = bool(self.chips_var.get())
        self.cfg["translate_model"] = self.translate_var.get().strip() or DEFAULT_CONFIG["translate_model"]
        self.cfg["describer_model"] = self.describer_var.get().strip()
        self.cfg["ollama_model"] = self.vlm_var.get().strip() or DEFAULT_CONFIG["ollama_model"]
        host = self.host_var.get().strip() or DEFAULT_CONFIG["ollama_host"]
        if not re.match(r"^https?://", host, re.IGNORECASE):
            host = "http://" + host
        self.cfg["ollama_host"] = host.rstrip("/")
        if normalize and self.host_var.get() != self.cfg["ollama_host"]:  # 入力中は書き換えない (http:// を勝手に足さない)
            self.host_var.set(self.cfg["ollama_host"])
        self.cfg["era_tags"] = bool(self.era_var.get())
        self.cfg["low_vram"] = bool(self.low_vram_var.get())
        self.cfg["main_person_only"] = bool(self.main_person_var.get())
        self.cfg["split_persons"] = bool(self.split_persons_var.get())
        self.cfg["gender"] = self.gender_labels.get(self.gender_var.get(), "none")
        self.cfg["restyle"] = self.restyle_labels.get(self.restyle_var.get(), "none")
        self.cfg["anatomy_guard"] = bool(self.anatomy_var.get())
        self.cfg["remove_monochrome"] = bool(self.color_var.get())
        self.cfg["avoid_same_face"] = bool(self.same_face_var.get())
        self.cfg["avoid_duplicates"] = bool(self.dup_var.get())
        self.cfg["person_details"] = bool(self.details_var.get())
        self.cfg["turnaround"] = bool(self.turnaround_var.get())
        self.cfg["no_identity"] = bool(self.no_identity_var.get())
        self.cfg["no_style"] = bool(self.no_style_var.get())
        self.cfg["mode"] = self.mode_var.get() if self.mode_var.get() in MODES else "single"
        for key, var in (
            ("general_threshold", self.general_th_var),
            ("character_threshold", self.character_th_var),
            ("style_threshold", self.style_th_var),
        ):
            try:
                value = float(var.get())
                if not normalize and not 0.05 <= value <= 0.99:
                    continue  # 入力の途中 ("0." = 0 など) は前の値のまま
                self.cfg[key] = min(0.99, max(0.01, value))
            except ValueError:
                if not normalize:
                    continue  # 入力の途中 ("0." など) は前の値のまま
                self.cfg[key] = DEFAULT_CONFIG[key]
            if normalize and var.get() != f"{self.cfg[key]:.2f}":
                var.set(f"{self.cfg[key]:.2f}")
        self._save_cfg()

    def start_analysis(self) -> None:
        if not self._ready() or self.busy:
            return
        self._collect_settings()
        if not tagger_ready(self.cfg["tagger_model"]):
            # モデル未取得なら確認してから取得し、完了後に解析を続ける
            self._ask_download(
                f"タガーモデル「{self.cfg['tagger_model']}」がまだ取得されていません。",
                on_done=self.start_analysis,
            )
            return
        if self.cfg.get("target") == "mossimage" and not moss_catalog_ready() and not getattr(self, "_moss_catalog_asked", False):
            # 初めて MOSS-Image に出力するときに確かめてから取得する。断ったらこの起動中は聞かない
            self._moss_catalog_asked = True
            if messagebox.askyesno(
                APP_NAME, "MOSS-Image では、キャラ名を一覧の書き方「名前 from 作品名」(例: dawn (pokemon) from pokemon) で書きます。\n"
                f"そのために MOSS-Image のキャラ一覧 (約 {MOSS_CATALOG_MB}MB、hide1 の MOSS-Image Character & Style Catalog) を取得します (初回のみ)。\n\n"
                f"取得元: {MOSS_CATALOG_PAGE}\n保存先: {MOSS_CATALOG_PATH.parent}\n\n"
                "取得しない場合は、キャラ名をそのまま書きます。取得しますか？", parent=self.root):
                self._start_moss_catalog_download(self.start_analysis)
                return
        # 解析中に設定を変えても (「再接続」などで cfg が書き換わっても) 走っている解析には影響させない
        self.engine.cfg = dict(self.cfg)
        self.engine.reset_cancel()
        if not self._start_busy("準備中...", self.engine.cancel):
            return
        self._begin_pausable(self.engine.pause_gate)
        for chips in (self.pos_chips, self.neg_chips):
            chips.set_text("")
        for box in (self.extras_box, self.persons_box, self.tags_box, self.analysis_box, self.notes_box):
            box.delete("1.0", "end")
        self.result = None
        self._history_path = None
        threading.Thread(target=self._worker, args=(self.image_path, self.selected_refs(), self.controls()), daemon=True).start()

    # ---- 設定画面 (頻繁に変えない設定。変えたらすぐ反映・保存) ----
    def _build_settings_window(self) -> None:
        win = tk.Toplevel(self.root)
        win.withdraw()
        win.title(f"{APP_NAME} - 設定")
        win.transient(self.root)
        win.protocol("WM_DELETE_WINDOW", self.close_settings)
        self.settings_win = win
        body = ttk.Frame(win, padding=10)
        body.pack(fill="both", expand=True)
        pad = {"padx": (0, 6), "pady": 2}

        t = ttk.LabelFrame(body, text="タガー", padding=(6, 4))
        t.pack(fill="x")
        t.columnconfigure(1, weight=1)
        ttk.Label(t, text="モデル").grid(row=0, column=0, sticky="w", **pad)
        self.tagger_labels = {self._tagger_label(k): k for k in TAGGER_MODELS}
        self.tagger_var = tk.StringVar(value=self._tagger_label(self.cfg["tagger_model"]))
        self.tagger_box = ttk.Combobox(t, textvariable=self.tagger_var, values=list(self.tagger_labels), state="readonly", width=48)
        self.tagger_box.grid(row=0, column=1, sticky="we", pady=2)
        ttk.Label(t, text="しきい値").grid(row=1, column=0, sticky="w", **pad)
        th = ttk.Frame(t)
        th.grid(row=1, column=1, sticky="w", pady=2)
        ttk.Label(th, text="一般").pack(side="left")
        self.general_th_var = tk.StringVar(value=f"{self.cfg['general_threshold']:.2f}")
        ttk.Spinbox(th, from_=0.05, to=0.95, increment=0.05, textvariable=self.general_th_var, width=5, format="%.2f").pack(
            side="left", padx=(4, 10))
        ttk.Label(th, text="キャラクター").pack(side="left")
        self.character_th_var = tk.StringVar(value=f"{self.cfg['character_threshold']:.2f}")
        ttk.Spinbox(th, from_=0.05, to=0.99, increment=0.05, textvariable=self.character_th_var, width=5, format="%.2f").pack(
            side="left", padx=(4, 10))
        ttk.Label(th, text="画風").pack(side="left")
        self.style_th_var = tk.StringVar(value=f"{float(self.cfg.get('style_threshold', 0.15)):.2f}")
        ttk.Spinbox(th, from_=0.05, to=0.95, increment=0.05, textvariable=self.style_th_var, width=5, format="%.2f").pack(
            side="left", padx=4)

        # 枠の間は 5 (説明のモデルの行を足したとき、1280 × 720 の画面に収まるよう 8 から詰めた)
        v = ttk.LabelFrame(body, text="VLM (Ollama)", padding=(6, 4))
        v.pack(fill="x", pady=(5, 0))
        v.columnconfigure(1, weight=1)
        ttk.Label(v, text="モデル").grid(row=0, column=0, sticky="w", **pad)
        self.vlm_var = tk.StringVar(value=self.cfg["ollama_model"])
        self.vlm_box = ttk.Combobox(v, textvariable=self.vlm_var, values=self._vlm_choices([]), width=36)
        self.vlm_box.grid(row=0, column=1, sticky="we", pady=2)
        self.pull_btn = ttk.Button(v, text="追加 (pull)", width=11, command=self.pull_vlm)
        self.pull_btn.grid(row=0, column=2, padx=(4, 0), pady=2)
        ttk.Label(v, text="訳のモデル").grid(row=1, column=0, sticky="w", **pad)
        self.translate_var = tk.StringVar(value=str(self.cfg.get("translate_model") or DEFAULT_CONFIG["translate_model"]))
        self.translate_box = ttk.Combobox(v, textvariable=self.translate_var,
                                          values=dedup([self.translate_var.get()] + TRANSLATE_CHOICES), width=36)
        self.translate_box.grid(row=1, column=1, sticky="we", pady=2)
        tb = ttk.Frame(v)
        tb.grid(row=1, column=2, padx=(4, 0), pady=2, sticky="w")
        self.tr_pull_btn = ttk.Button(tb, text="取得", width=6, command=self.pull_translate)
        self.tr_pull_btn.pack(side="left")
        ttk.Label(tb, text="結果の日本語訳用", foreground="#666").pack(side="left", padx=(4, 0))
        # 先に画像の説明を書かせるモデル (describer_model)。空なら使わない
        ttk.Label(v, text="説明のモデル").grid(row=2, column=0, sticky="w", **pad)
        self.describer_var = tk.StringVar(value=str(self.cfg.get("describer_model") or ""))
        self.describer_box = ttk.Combobox(v, textvariable=self.describer_var,
                                          values=["", *dedup([self.describer_var.get(), JOYCAPTION_MODEL])], width=36)
        self.describer_box.grid(row=2, column=1, sticky="we", pady=2)
        db = ttk.Frame(v)
        db.grid(row=2, column=2, padx=(4, 0), pady=2, sticky="w")
        self.desc_pull_btn = ttk.Button(db, text="取得", width=6, command=self.pull_describer)
        self.desc_pull_btn.pack(side="left")
        ttk.Label(db, text="先に説明を書かせる (空 = 使わない)", foreground="#666").pack(side="left", padx=(4, 0))
        ttk.Label(v, text="Ollama URL").grid(row=3, column=0, sticky="w", **pad)
        self.host_var = tk.StringVar(value=self.cfg["ollama_host"])
        ttk.Entry(v, textvariable=self.host_var, width=36).grid(row=3, column=1, sticky="we", pady=2)
        hb = ttk.Frame(v)
        hb.grid(row=3, column=2, padx=(4, 0), pady=2, sticky="w")
        ttk.Button(hb, text="再接続", width=6, command=self.reconnect_ollama).pack(side="left")
        self.setup_btn = ttk.Button(hb, text="導入...", width=6, command=self.open_ollama_setup)
        self.setup_btn.pack(side="left", padx=(3, 0))

        w = ttk.LabelFrame(body, text="プロンプトの書き方", padding=(6, 4))
        w.pack(fill="x", pady=(5, 0))

        def check(parent: Any, var: "tk.BooleanVar", text: str) -> None:
            ttk.Checkbutton(parent, text=text, variable=var).pack(anchor="w")

        self.quality_var = tk.BooleanVar(value=bool(self.cfg["quality_tags"]))
        check(w, self.quality_var, "品質タグ (masterpiece, best quality ...) を先頭に付ける")
        self.break_var = tk.BooleanVar(value=bool(self.cfg["use_break"]))
        check(w, self.break_var, "複数人物は BREAK で人物ごとに区切る (SDXL 系。A1111 / Forge / CLIP Text Encode++ 用、ComfyUI 標準ノードは非対応)")
        self.split_var = tk.BooleanVar(value=bool(self.cfg.get("split_extras", False)))
        check(w, self.split_var, "カメラ・照明・画風のタグを本文から分離する (追加プロンプト側に出す)")
        self.split_bg_var = tk.BooleanVar(value=bool(self.cfg.get("split_background", False)))
        check(w, self.split_bg_var, "背景のタグを本文から分離する (本文は被写体だけ、背景は追加プロンプト側)")
        self.polish_var = tk.BooleanVar(value=bool(self.cfg.get("polish_prose", True)))
        check(w, self.polish_var, "文を仕上げる (文章の出力先・VLM: 集めた内容を 1 つの文章に書き直す。VLM の呼び出しが 1 回増える)")
        self.age_var = tk.BooleanVar(value=bool(self.cfg.get("age_look", True)))
        check(w, self.age_var, "年齢の見た目を書く (人物ごとに画像から読んだ年齢層を、文とタグに書く。元の画像と年齢がずれるのを防ぐ)")
        self.era_var = tk.BooleanVar(value=bool(self.cfg.get("era_tags", True)))
        check(w, self.era_var, "年代スタイル (1990s / 2010s など) を判定して画風タグに足す (イラスト系のみ、17MB のモデルを初回取得)")
        self.anatomy_var = tk.BooleanVar(value=bool(self.cfg.get("anatomy_guard", True)))
        check(w, self.anatomy_var, "人物の破綻を防ぐ (ネガティブに bad anatomy, bad hands, extra digits などを入れ、全身の絵は顔だけの構図を避ける)")
        self.same_face_var = tk.BooleanVar(value=bool(self.cfg.get("avoid_same_face", False)))
        check(w, self.same_face_var, "同じ顔を避ける (2 人以上ならネガティブに clone, twins など。ネガティブの無い出力先は本文で書き分け)")

        h = ttk.LabelFrame(body, text="人物", padding=(6, 4))
        h.pack(fill="x", pady=(5, 0))
        self.main_person_var = tk.BooleanVar(value=bool(self.cfg.get("main_person_only", False)))
        check(h, self.main_person_var, "人物はメインの 1 人だけ (一番大きく写っている人を検出し、ほかの人はプロンプトに入れない。初回 44MB 取得)")
        self.split_persons_var = tk.BooleanVar(value=bool(self.cfg.get("split_persons", True)))
        check(h, self.split_persons_var, "2 人以上なら人物ごとに分ける (1 人ずつ切り出してタグ付けし、人数と特徴の取り違えを防ぐ)")

        o = ttk.LabelFrame(body, text="動作・表示", padding=(6, 4))
        o.pack(fill="x", pady=(5, 0))
        self.low_vram_var = tk.BooleanVar(value=bool(self.cfg.get("low_vram", True)))
        check(o, self.low_vram_var, "GPU / VRAM の使用を抑える (タガーは CPU で、VLM は使い終わったらすぐ VRAM を空ける。VLM は毎回読み込み分遅くなる)")
        ttk.Checkbutton(o, text="プレビューをモザイク表示 (NSFW 対策。出力には影響しません)", variable=self.preview_mosaic_var,
                        command=self._toggle_preview_mosaic).pack(anchor="w")
        self.chips_var = tk.BooleanVar(value=bool(self.cfg.get("result_chips", True)))
        ttk.Checkbutton(o, text="結果をタグごとの塊で表示する (選んで Del で消す・Ctrl+Z で戻す・日本語訳)",
                        variable=self.chips_var, command=self._toggle_chips).pack(anchor="w")

        foot = ttk.Frame(body)
        foot.pack(fill="x", pady=(6, 0))
        ttk.Label(foot, text="変えた内容はすぐ反映され、config.json に保存されます", foreground="#666").pack(side="left")
        ttk.Button(foot, text="閉じる", command=self.close_settings).pack(side="right")
        # 変えたらすぐ反映・保存 (入力中の数値を書き換えないよう、少し待ってから保存する)
        self._settings_save_job: str | None = None
        for var in (self.tagger_var, self.general_th_var, self.character_th_var, self.style_th_var, self.vlm_var, self.translate_var,
                    self.describer_var, self.host_var,
                    self.quality_var, self.break_var, self.split_var, self.split_bg_var, self.polish_var, self.age_var, self.era_var,
                    self.anatomy_var,
                    self.same_face_var, self.main_person_var, self.split_persons_var, self.low_vram_var):
            var.trace_add("write", lambda *_a: self._schedule_settings_save())

    def _schedule_settings_save(self) -> None:
        if self._settings_save_job is not None:
            self.root.after_cancel(self._settings_save_job)
        self._settings_save_job = self.root.after(600, self._save_settings_now)

    def _save_settings_now(self) -> None:
        self._settings_save_job = None
        self._collect_settings(normalize=False)

    def open_settings(self) -> None:
        win = self.settings_win
        win.deiconify()
        win.update_idletasks()
        px, py = self.root.winfo_rootx(), self.root.winfo_rooty()
        win.geometry(f"+{px + 40}+{py + 40}")
        win.lift()
        win.focus_set()

    def close_settings(self) -> None:
        if self._settings_save_job is not None:
            self.root.after_cancel(self._settings_save_job)
            self._settings_save_job = None
        self._collect_settings()  # しきい値の書き方も整えて保存する
        self.settings_win.withdraw()

    # ---- 構図と強さ ----
    def _build_controls(self, parent: Any) -> None:
        box = ttk.LabelFrame(parent, text="構図と強さ (LLM・GPU 不要。指定しなければ画像のまま)", padding=(6, 0))
        box.pack(fill="x", pady=(1, 0))
        self.pose_control: dict[str, Any] | None = None
        row = ttk.Frame(box)
        row.pack(fill="x")
        ttk.Label(row, text="ポーズ").pack(side="left", padx=(0, 4))
        ttk.Button(row, text="ボーン編集...", command=self.open_pose_editor).pack(side="left", padx=(0, 4))
        ttk.Button(row, text="解除", command=self.clear_pose_control, width=5).pack(side="left")
        # 撮影プリセット (候補は config.json の camera_looks)。行を増やすと 1152 × 633 に収まらないので、ポーズの行の右端に置く
        self.look_var = tk.StringVar(value=LOOK_NONE)
        self.look_parts: dict[str, str] = {}  # 撮影の組み合わせ (「組み合わせ...」で選ぶ)
        look_box = ttk.Combobox(row, textvariable=self.look_var, state="readonly", width=16,
                                values=[LOOK_NONE] + [x["name"] for x in self.cfg["camera_looks"]] + [LOOK_PARTS_MENU])
        look_box.pack(side="right")
        look_box.bind("<<ComboboxSelected>>", lambda _e: self.open_look_parts() if self.look_var.get() == LOOK_PARTS_MENU else None)
        ttk.Label(row, text="撮影").pack(side="right", padx=(6, 4))
        self.pose_control_var = tk.StringVar(value="指定なし")
        ttk.Label(row, textvariable=self.pose_control_var, foreground="#555").pack(side="left", padx=(6, 0))
        grid = ttk.Frame(box)
        grid.pack(fill="x", pady=(1, 0))
        ttk.Label(grid, text="カメラ").grid(row=0, column=0, rowspan=2, sticky="nw", padx=(0, 4))
        self.camera_vars: dict[str, tk.StringVar] = {}
        for i, (dim, options) in enumerate(CAMERA_OPTIONS.items()):
            r, c = divmod(i, 2)
            ttk.Label(grid, text=CAMERA_LABELS_JA[dim]).grid(row=r, column=1 + c * 2, sticky="e", padx=(6, 2), pady=1)
            var = tk.StringVar(value=options[0][1])
            ttk.Combobox(grid, textvariable=var, values=[label for _, label, _ in options], state="readonly", width=9).grid(
                row=r, column=2 + c * 2, sticky="w", pady=1)
            self.camera_vars[dim] = var
        grid2 = ttk.Frame(box)
        grid2.pack(fill="x", pady=(1, 1))
        ttk.Label(grid2, text="強さ").grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.strength_vars: dict[str, tk.StringVar] = {}
        for i, (group, label) in enumerate(STRENGTH_GROUPS):  # 4 つずつ 2 行 (行を増やすと 1152 × 633 に収まらない)
            ttk.Label(grid2, text=label).grid(row=i // 4, column=1 + (i % 4) * 2, sticky="e", padx=(4, 2), pady=1)
            var = tk.StringVar(value="普通")
            ttk.Combobox(grid2, textvariable=var, values=["弱い", "普通", "強い"], state="readonly", width=4).grid(
                row=i // 4, column=2 + (i % 4) * 2, sticky="w")
            self.strength_vars[group] = var

    def controls(self) -> dict[str, Any]:
        """画面の指定を Engine.run の controls にする (指定の無い項目は含めない)。"""
        out: dict[str, Any] = {}
        if self.pose_control is not None:
            out["pose"] = {"tags": list(self.pose_control["tags"])}
        camera = {}
        for dim, var in self.camera_vars.items():
            value = next((v for v, label, _ in CAMERA_OPTIONS[dim] if label == var.get()), "")
            if value:
                camera[dim] = value
        if camera:
            out["camera"] = camera
        look = find_look(self.cfg, self.look_var.get())
        if look:
            out["look"] = dict(look)
        elif self.look_var.get() == LOOK_PARTS_MENU and compose_look_parts(self.look_parts):
            out["look_parts"] = dict(self.look_parts)
        strength = {g: {"強い": "strong", "弱い": "weak"}[v.get()] for g, v in self.strength_vars.items() if v.get() in ("強い", "弱い")}
        if strength:
            out["strength"] = strength
        return out

    def open_look_parts(self) -> None:
        """撮影の組み合わせ: 写り・カメラ・レンズ・質感・光・フィルム・粒子・被写界深度を項目ごとに選ぶ。"""
        win = tk.Toplevel(self.root)
        win.title("撮影の組み合わせ")
        win.transient(self.root)
        win.resizable(False, False)
        frame = ttk.Frame(win, padding=10)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="項目ごとに選びます。選んだ項目の英語の語を、この順につないでプロンプトに入れます。",
                  foreground="#555").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        variables: dict[str, tk.StringVar] = {}
        for i, (key, options) in enumerate(LOOK_PARTS.items(), start=1):
            ttk.Label(frame, text=LOOK_PART_LABELS_JA[key]).grid(row=i, column=0, sticky="e", padx=(0, 6), pady=2)
            labels = [LOOK_NONE] + [f"{label}  ({value})" for value, label in options]
            current = next((f"{label}  ({value})" for value, label in options if value == self.look_parts.get(key)), LOOK_NONE)
            var = tk.StringVar(value=current)
            ttk.Combobox(frame, textvariable=var, values=labels, state="readonly", width=56,
                         height=min(len(labels), 30)).grid(row=i, column=1, sticky="w", pady=2)
            variables[key] = var
        preview = tk.StringVar()

        def chosen() -> dict[str, str]:
            out = {}
            for key, var in variables.items():
                value = next((v for v, label in LOOK_PARTS[key] if var.get() == f"{label}  ({v})"), "")
                if value:
                    out[key] = value
            return out

        def update(*_a: Any) -> None:
            preview.set(compose_look_parts(chosen()) or "(何も選んでいません)")

        for var in variables.values():
            var.trace_add("write", update)
        update()
        n = len(LOOK_PARTS) + 1
        ttk.Label(frame, textvariable=preview, wraplength=560, foreground="#333").grid(row=n, column=0, columnspan=2, sticky="w", pady=(8, 4))

        def ok() -> None:
            self.look_parts = chosen()
            self.look_var.set(LOOK_PARTS_MENU if self.look_parts else LOOK_NONE)
            win.destroy()

        def cancel() -> None:
            self.look_var.set(LOOK_PARTS_MENU if compose_look_parts(self.look_parts) else LOOK_NONE)
            win.destroy()

        buttons = ttk.Frame(frame)
        buttons.grid(row=n + 1, column=0, columnspan=2, sticky="e")
        ttk.Button(buttons, text="OK", command=ok).pack(side="right")
        ttk.Button(buttons, text="キャンセル", command=cancel).pack(side="right", padx=(0, 6))
        win.protocol("WM_DELETE_WINDOW", cancel)
        win.grab_set()

    def clear_pose_control(self) -> None:
        self.pose_control = None
        self.pose_control_var.set("指定なし")

    def set_pose_control(self, points: list[list[float]], size: tuple[int, int]) -> None:
        tags = pose_tags_from_points(points)
        self.pose_control = {"points": [list(p) for p in points], "size": size, "tags": tags}
        text = ", ".join(sd_tag(t) for t in tags) or "(タグにできる姿勢なし)"
        self.pose_control_var.set(text if len(text) <= 26 else text[:25] + "…")  # 欄の幅に収める (全文は編集画面に出る)

    def open_pose_editor(self) -> None:
        if self.busy:
            return
        # 元にする画像: 複数枚モードで ② があれば ②、無ければ ① (1 枚モードは読み込んだ画像)
        src = (self.slot_paths.get("pose") if self.multi_mode() else None) or self.slot_paths.get("person")
        PoseEditor(self, src)

    def _start_busy(self, status: str = "", cancel: Callable[[], None] | None = None, determinate: bool = False) -> bool:
        """重い処理を始める前に呼ぶ (ボタンを無効にし、進捗表示を始める)。ほかの処理の最中なら False を返し、
        二重に動かさない。cancel を渡すと「■ 中止」で呼ばれる。"""
        if self.busy:
            return False
        self.busy = True
        self.run_btn.config(state="disabled")
        self.pull_btn.config(state="disabled")
        self.tr_pull_btn.config(state="disabled")
        self.desc_pull_btn.config(state="disabled")
        if determinate:
            self.progress.config(mode="determinate", maximum=1.0, value=0.0)
        else:
            self.progress.config(mode="indeterminate", maximum=100, value=0.0)
            self.progress.start(12)
        if status:
            self.status_var.set(status)
        if cancel is not None:
            self._begin_job(cancel)
        return True

    def _begin_pausable(self, gate: PauseGate) -> None:
        """一時停止できる処理 (分析・日本語訳) を始めたときに呼ぶ。"""
        gate.resume()
        self._job_gate = gate
        self.pause_btn.config(state="normal", text="⏸ 一時停止")

    def toggle_pause(self) -> None:
        """「⏸ 一時停止」/「▶ 再開」。止まるのは今の段階 (タガー 1 枚分・VLM の 1 回の呼び出しなど) が終わってから。"""
        gate = self._job_gate
        if gate is None:
            return
        if gate.paused:
            gate.resume()
            self.pause_btn.config(text="⏸ 一時停止")
            self.status_var.set("再開しました")
        else:
            gate.pause()
            self.pause_btn.config(text="▶ 再開")
            self.status_var.set("一時停止します (今の段階が終わったところで止まります)")

    def _begin_job(self, cancel: Callable[[], None]) -> None:
        """中止できる処理を始めたときに呼ぶ。「■ 中止」を押せるようにする。"""
        self._job_cancel = cancel
        self.stop_btn.config(state="normal")

    def cancel_job(self) -> None:
        """「■ 中止」/ Esc。解析は VLM の応答待ちならすぐ、タガーの推論中ならその処理が終わった時点で止まる。
        ダウンロードは 1MB ごと、pull は進捗の区切りごとに止まる。"""
        if self._job_cancel is None:
            return
        self.stop_btn.config(state="disabled")
        self.status_var.set("中止しています...")
        self._job_cancel()

    def _worker(self, path: Path, refs: dict[str, Path] | None = None, controls: dict[str, Any] | None = None) -> None:
        try:
            self.queue.put(("ok", self.engine.run(path, refs, controls)))
        except OperationCancelled:
            self.queue.put(("cancelled", "中止しました"))
        except Exception as e:  # noqa: BLE001 - UI スレッドで表示する
            self.queue.put(("err", e))

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "status":
                    self.status_var.set(payload)
                elif kind == "frac":
                    frac, est = payload
                    if self.busy:
                        self.progress.stop()
                        self.progress.config(mode="determinate", maximum=1.0, value=frac)
                        self.pct_var.set(f"{frac:.0%}")
                        self._vlm_estimate = (time.time(), *est) if est else None
                elif kind == "ollama":
                    models, vision = payload  # vision は絞り込み前なら None
                    self._known_models = models
                    self.translate_box["values"] = dedup(models + [self.translate_var.get().strip()] + TRANSLATE_CHOICES)  # 文字専用のモデルも選べる
                    # 説明のモデルは画像を見るので、画像対応のモデルから (絞り込み前は全部)
                    self.describer_box["values"] = ["", *dedup([self.describer_var.get().strip(), JOYCAPTION_MODEL]
                                                               + list(vision if vision else models))]
                    self.vlm_box["values"] = self._vlm_choices(vision if vision else models)
                    current = self.vlm_var.get().strip()
                    hit = current in models or any(m.startswith(current + ":") for m in models)
                    if vision is None:
                        text = f"Ollama: 接続OK (全 {len(models)} モデル、画像対応を確認中...)"
                    else:
                        text = f"Ollama: 接続OK (画像対応 {len(vision)} / 全 {len(models)})"
                        if not vision:
                            text += " / 画像対応モデルがありません。VLM 欄に名前を入れて「追加 (pull)」"
                    if not hit:
                        text += f" / 「{current}」は未取得です。「追加 (pull)」で取得できます"
                    self.ollama_var.set(text)
                    if not self._ollama_ok:
                        self._ollama_ok = True
                        self._apply_chip_mode()
                elif kind == "translated":
                    self._finish()
                    self._user_translations.update(payload)
                    self._translations.update(payload)
                    saved = save_translations(self._user_translations)
                    self._set_language(True)
                    self.status_var.set(f"日本語で表示しています (新しく訳した語 {len(payload)} 個"
                                        + ("" if saved else "。translations.json に保存できませんでした") + ")")
                elif kind == "translate_error":
                    self._finish()
                    self._set_language(True)  # 辞書にある語は日本語で見せる
                    self.status_var.set("辞書に無い語は英語のままです (訳のモデルで訳せませんでした: " + str(payload) + ")")
                elif kind == "ollama_error":
                    if self._ollama_ok:
                        self._ollama_ok = False
                        self._apply_chip_mode()
                    self.ollama_var.set(f"Ollama: 未接続。「タガーのみ」で利用できます。導入や起動は「導入...」から ({payload})")
                    if (not self.cfg.get("ollama_setup_asked") and self.cfg.get("engine") == "hybrid"
                            and not self.busy and not self._modal_open):  # ほかのダイアログに重ねない
                        # 初回だけ、導入サポートを開くか聞く (既に入っていれば起動だけを案内する)
                        self.cfg["ollama_setup_asked"] = True
                        self._save_cfg()
                        installed = find_ollama_exe() is not None
                        question = (
                            "Ollama は導入済みですが起動していません。起動して使いますか？"
                            if installed else
                            "Ollama が見つかりません。導入サポート (保存先・必要容量・スペックの確認、ダウンロードとインストール) を開きますか？\n"
                            "「いいえ」でも「タガーのみ」で使えます。"
                        )
                        if messagebox.askyesno(APP_NAME, question, parent=self.root):
                            self.root.after(50, self.open_ollama_setup)
                elif kind == "ollama_setup_ready":
                    self._on_ollama_setup_ready(payload)
                elif kind == "ollama_setup_done":
                    self._finish()
                    setup: dict[str, Any] = payload
                    self.status_var.set(f"Ollama の準備ができました (モデル: {setup['model']})")
                    self._probe_ollama_async()
                    if setup.get("pull"):
                        self._start_pull(setup["model"])  # 間を空けると、その間に別の処理を始められてしまう
                elif kind == "ollama_setup_error":
                    self._finish()
                    self.status_var.set("Ollama 導入サポート: " + str(payload))
                    messagebox.showerror(APP_NAME, str(payload), parent=self.root)
                elif kind == "pulled":
                    self._finish()
                    extra = [m for m in (self.cfg.get("ollama_models") or []) if isinstance(m, str)]
                    if payload not in extra:
                        extra.append(payload)
                    self.cfg["ollama_models"] = extra
                    self.cfg["ollama_model"] = payload
                    self.vlm_var.set(payload)
                    self._save_cfg()
                    self.status_var.set(f"取得完了: {payload}")
                    self._probe_ollama_async()
                elif kind == "pulled_translate":
                    self._finish()
                    self.cfg["translate_model"] = payload
                    self.translate_var.set(payload)
                    self._save_cfg()
                    self.status_var.set(f"取得完了 (訳のモデル): {payload}")
                    self._probe_ollama_async()
                elif kind == "pulled_describer":
                    self._finish()
                    self.cfg["describer_model"] = payload
                    self.describer_var.set(payload)
                    self._save_cfg()
                    self.status_var.set(f"取得完了 (説明のモデル): {payload}")
                    self._probe_ollama_async()
                elif kind == "hf_checked":
                    self._on_hf_checked(payload)
                elif kind == "pull_error":
                    self._finish()
                    self.status_var.set("取得エラー: " + str(payload))
                    messagebox.showerror(APP_NAME, str(payload), parent=self.root)
                elif kind == "dl_progress":
                    msg, frac = payload
                    self.status_var.set(msg)
                    if frac >= 0:
                        self.progress.stop()  # 不確定表示の自動送りを止めてから割合を出す
                        self.progress.config(mode="determinate", maximum=1.0, value=frac)
                elif kind == "dl_done":
                    name, on_done = payload
                    self._finish()
                    self._refresh_tagger_labels()
                    self.status_var.set(f"ダウンロード完了: {name} を使用します")
                    if on_done:
                        self.root.after(50, on_done)
                elif kind == "dl_error":
                    self._finish()
                    self.status_var.set("ダウンロードエラー: " + str(payload))
                    messagebox.showerror(
                        APP_NAME,
                        f"{payload}\n\n「▶ 分析する」を押すと再度ダウンロードを確認します。",
                        parent=self.root,
                    )
                elif kind == "ok":
                    self._finish(done=True)
                    if self.engine.last_vlm:  # VLM ごとにかかった時間を覚えて、次の見込みに使う
                        model, secs = self.engine.last_vlm
                        seconds = dict(self.cfg.get("vlm_seconds") or {})
                        seconds[model] = secs
                        self.cfg["vlm_seconds"] = seconds
                        self.engine.last_vlm = None
                        self._save_cfg()
                    self._show_result(payload)
                    level = self.engine.cfg.get("mosaic_strength")
                    self._history_path = history_add(payload, self.slot_thumbs["person"],
                                                     {"mosaic_strength": level, "mode": self.mode_var.get()})
                    if self._history_win is not None:
                        self._fill_history()
                    if self._ja and self._chips_enabled():  # 日本語で見ている間は、新しい結果も訳して見せる
                        self._translate_and_show()
                elif kind == "cancelled":
                    self._finish()
                    self.status_var.set(str(payload or "中止しました"))
                elif kind == "stop_state":
                    # 途中で止めると壊れる工程 (インストール中など) では中止を受け付けない
                    if self._job_cancel is not None:
                        self.stop_btn.config(state="normal" if payload else "disabled")
                        if not payload:
                            self._job_cancel = None  # Esc でも止めようとしない
                else:
                    self._finish()
                    message = describe_error(payload)
                    self.status_var.set("エラー: " + message)
                    messagebox.showerror(APP_NAME, message, parent=self.root)
        except queue.Empty:
            pass
        if self.busy and self._vlm_estimate is not None and not (self._job_gate and self._job_gate.paused):
            # VLM の間: 前回かかった時間から見込みで進める (見込みを超えても終わりの手前で止める)
            started, est, lo, hi = self._vlm_estimate
            frac = lo + (hi - lo) * min(1.0, (time.time() - started) / est)
            self.progress.config(value=frac)
            self.pct_var.set(f"{frac:.0%}")
        self.root.after(100, self._poll)

    def _finish(self, done: bool = False) -> None:
        """処理の終わり。done=True (解析が完了) ならバーを 100% まで埋めて残す。それ以外 (中止・エラー・ほかの処理) は空に戻す。"""
        self.busy = False
        self._job_cancel = None
        self.stop_btn.config(state="disabled")
        if self._job_gate is not None:
            self._job_gate.resume()
            self._job_gate = None
        self.pause_btn.config(state="disabled", text="⏸ 一時停止")
        self.progress.stop()
        self.progress.config(mode="determinate", maximum=1.0, value=1.0 if done else 0.0)
        self.pct_var.set("100%" if done else "")
        self._vlm_estimate = None
        self.run_btn.config(state="normal" if self._ready() else "disabled")
        self.pull_btn.config(state="normal")
        self.tr_pull_btn.config(state="normal")
        self.desc_pull_btn.config(state="normal")

    def _show_result(self, result: PromptResult, mosaic_level: str | None = None) -> None:
        """結果を画面に出す。mosaic_level は解析に使ったモザイクの強さ (履歴から開いたとき。無ければ今回の解析の設定)。"""
        self.result = result
        self._render_preview()  # メインの人物の枠を描く
        persons_text = result.person_lines() or (
            "(複数枚モードでは人物ごとの振り分けは行いません)" if result.references else "(複数人物は検出されませんでした)"
        )
        analysis_text = result.analysis
        if result.mosaic_prompt:
            # 解析に使った設定 (解析後に画面で変えていても、その時の強さを出す)
            level = mosaic_level if mosaic_level is not None else self.engine.cfg.get("mosaic_strength")
            level = level if level in MOSAIC_LEVELS else DEFAULT_MOSAIC_LEVEL
            analysis_text += (
                f"\n\n[NSFW 判定] モザイク指定 (強さ: {MOSAIC_LEVELS[level]['label']}) をポジティブに追加し、"
                "ネガティブに uncensored を追加しました"
            )
        self.pos_chips.set_text(result.positive_prompt)
        self.neg_chips.set_text(result.negative_prompt)
        for box, text in (
            (self.extras_box, result.extras_text()),
            (self.persons_box, persons_text),
            (self.tags_box, result.tag_lines()),
            (self.analysis_box, analysis_text),
            (self.notes_box, result.notes),
        ):
            box.delete("1.0", "end")
            box.insert("1.0", text)
        self.tabs.select((self.persons_box if result.persons else self.extras_box).master)
        if TARGETS.get(result.target, {}).get("neg_label"):
            self.neg_box.master.config(text=TARGETS[result.target]["neg_label"])
        elif TARGETS.get(result.target, {}).get("negative") is False:
            self.neg_box.master.config(text="ネガティブプロンプト (この出力先は非対応のため空)")
        elif TARGETS.get(result.target, {}).get("negative") == "optional":
            self.neg_box.master.config(text="ネガティブプロンプト (CFG 1 では無視されます)")
        else:
            self.neg_box.master.config(text="ネガティブプロンプト")

        status = f"完了 ({result.elapsed_sec:.1f} 秒)  medium={result.detected_medium}  rating={result.rating}"
        if result.persons:
            status += f"  人物={len(result.persons)}"
        if result.vlm_model:
            status += f"  vlm={result.vlm_model}"
        if result.nsfw:
            status += "  NSFW" + (" (モザイク指定を追加)" if result.mosaic_prompt else "")
        if result.warnings:
            status += "\n注意: " + " / ".join(result.warnings)
        self.status_var.set(status)

    def _result_from_boxes(self) -> PromptResult:
        """テキストボックス上でユーザーが編集した内容を反映した結果を返す。"""
        assert self.result is not None
        import dataclasses

        return dataclasses.replace(
            self.result,
            analysis=self.analysis_box.get("1.0", "end-1c").strip(),
            positive_prompt=self.pos_chips.get_text(),
            negative_prompt=self.neg_chips.get_text(),
            notes=self.notes_box.get("1.0", "end-1c").strip(),
        )

    # ---- チップ表示と日本語訳 ----
    def _chips_enabled(self) -> bool:
        """チップ表示にするか: 設定でオンのとき (辞書で訳せるので Ollama は要らない)。"""
        return bool(self.cfg.get("result_chips", True))

    def _apply_chip_mode(self) -> None:
        on = self._chips_enabled()
        for chips in (self.pos_chips, self.neg_chips):
            chips.set_mode(on)
        if on:
            self.ja_btn.pack(side="top", pady=(0, 4))
        else:
            self.ja_btn.pack_forget()
            if self._ja:
                self._set_language(False)

    def _on_chips_change(self) -> None:
        """チップを消した・戻したとき: 履歴にも反映する。"""
        if self._history_path is not None:
            history_update_prompts(self._history_path, self.pos_chips.get_text(), self.neg_chips.get_text())

    def _set_language(self, ja: bool) -> None:
        self._ja = ja
        self.ja_btn.config(text="English" if ja else "日本語")
        for chips in (self.pos_chips, self.neg_chips):
            chips.set_language(ja, self._translations)

    def toggle_language(self) -> None:
        """右上の「日本語 / English」。日本語にするとき、まだ訳していない塊だけを Ollama で訳す (訳は保存して使い回す)。"""
        if self._ja:
            self._set_language(False)
            return
        self._translate_and_show()

    def _translate_and_show(self) -> None:
        # キャラ名は訳さない (英語のまま表示。Ollama にも送らない)
        names = {chip_core(str(t.get("tag") or ""))[0].lower() for t in (self.result.tags if self.result else [])
                 if t.get("kind") == "character"}
        missing = dedup([core for chips in (self.pos_chips, self.neg_chips) for core, _w in map(chip_core, chips.chunks())
                         if core and not lookup_ja(self._translations, core) and core.lower() not in names])
        # 辞書にある語はすぐ日本語にする (辞書に無い語を Ollama が訳し終えるまで待たせない。2026-10-10 利用者の指摘)
        self._set_language(True)
        if not missing:
            return
        if not self._ollama_ok:  # Ollama が無ければ辞書にある語だけ日本語にする
            self.status_var.set(f"辞書に無い {len(missing)} 語は英語のままです (訳すには Ollama と訳のモデルが必要です)")
            return
        model = str(self.cfg.get("translate_model") or DEFAULT_CONFIG["translate_model"])
        vlm = OllamaVLM(self.cfg["ollama_host"], model, int(self.cfg.get("ollama_timeout") or VLM_TIMEOUT))
        vlm.cancel_event = threading.Event()

        def cancel() -> None:
            vlm.cancel_event.set()
            vlm.abort()

        if not self._start_busy(f"日本語に訳しています ({len(missing)} 個、{vlm.model})...", cancel):
            self.status_var.set("ほかの処理の最中は訳せません")
            return
        keep = 0 if self.cfg.get("low_vram", True) else None
        gate = PauseGate()
        self._begin_pausable(gate)

        def checkpoint() -> None:
            if vlm.cancel_event.is_set():
                raise OperationCancelled()
            gate.wait(vlm.cancel_event.is_set, lambda: self.queue.put(("status", "一時停止中 (「▶ 再開」で続きから)")))

        def work() -> None:
            try:
                self.queue.put(("translated", translate_chips_ja(vlm, missing, keep_alive=keep, checkpoint=checkpoint)))
            except OperationCancelled:
                self.queue.put(("cancelled", "日本語訳を中止しました"))
            except Exception as e:  # noqa: BLE001 - 画面に出す
                self.queue.put(("translate_error", describe_error(e)))

        threading.Thread(target=work, daemon=True).start()

    # ---- コピー / 保存 ----
    def _to_clipboard(self, text: str) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()  # ウィンドウを閉じてもクリップボードに残るようにする

    def copy(self, box: "tk.Text") -> None:
        chips = self.pos_chips if box is self.pos_box else self.neg_chips if box is self.neg_box else None
        text = chips.get_text() if chips else box.get("1.0", "end-1c").strip()  # チップは日本語表示でも英語でコピー
        if not text:
            return
        self._to_clipboard(text)
        self.status_var.set("クリップボードにコピーしました")

    def copy_both(self) -> None:
        pos = self.pos_chips.get_text()
        neg = self.neg_chips.get_text()
        if not pos and not neg:
            return
        if neg:
            self._to_clipboard(f"Positive:\n{pos}\n\nNegative:\n{neg}")
            self.status_var.set("ポジティブ / ネガティブをまとめてコピーしました")
        else:
            self._to_clipboard(pos)
            self.status_var.set("ポジティブをコピーしました (この出力先にネガティブはありません)")

    def save(self, kind: str) -> None:
        if not self.result:
            messagebox.showinfo(APP_NAME, "まだ解析結果がありません。", parent=self.root)
            return
        result = self._result_from_boxes()
        src = Path(result.image)
        ext = ".json" if kind == "json" else ".txt"
        filetypes = [("JSON", "*.json")] if kind == "json" else [("テキスト", "*.txt")]
        out = filedialog.asksaveasfilename(
            title="保存先を選択",
            defaultextension=ext,
            initialdir=str(src.parent),
            initialfile=f"{src.stem}_prompt{ext}",
            filetypes=filetypes + [("すべてのファイル", "*.*")],
            parent=self.root,
        )
        if not out:
            return
        content = json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n" if kind == "json" else result.to_text()
        try:
            Path(out).write_text(content, "utf-8")
        except OSError as e:
            messagebox.showerror(APP_NAME, f"保存できませんでした: {e}", parent=self.root)
            return
        self.status_var.set(f"保存しました: {out}")

    # ---- ヘルプ (手順書。右上のボタン / F1 で別ウィンドウ、左に目次・右に本文) ----
    def open_help(self) -> None:
        if self._help_win is not None:
            self._help_win.deiconify()
            self._help_win.lift()
            return
        win = tk.Toplevel(self.root)
        win.title(f"{APP_NAME} - ヘルプ (手順書)")
        win.transient(self.root)
        win.protocol("WM_DELETE_WINDOW", self.close_help)
        self._help_win = win
        # 閉じるボタンを先に下端へ (本文を先に置くと、窓が低いときにボタンが隠れる)
        ttk.Button(win, text="閉じる", command=self.close_help).pack(side="bottom", anchor="e", padx=8, pady=(0, 8))
        body = ttk.Frame(win, padding=6)
        body.pack(fill="both", expand=True)
        toc = tk.Listbox(body, width=28, height=8, exportselection=False, font=("Yu Gothic UI", 10), activestyle="none")
        for title, _ in HELP_SECTIONS:
            toc.insert("end", title)
        toc.pack(side="left", fill="y")
        text = tk.Text(body, wrap="word", font=("Yu Gothic UI", 10), padx=10, pady=8, relief="flat", height=10)
        scroll = ttk.Scrollbar(body, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        text.pack(side="left", fill="both", expand=True, padx=(6, 0))
        scroll.pack(side="left", fill="y")
        text.tag_configure("title", font=("Yu Gothic UI", 13, "bold"), spacing3=6)
        text.tag_configure("heading", font=("Yu Gothic UI", 10, "bold"), spacing1=8)
        self._help_toc, self._help_text = toc, text

        def show(_event: Any = None) -> None:
            picked = toc.curselection()
            index = picked[0] if picked else 0
            title, content = HELP_SECTIONS[index]
            text.config(state="normal")
            text.delete("1.0", "end")
            text.insert("end", title + "\n", "title")
            for line in content.strip("\n").split("\n"):
                text.insert("end", line + "\n", "heading" if line.startswith("■") else ())
            text.config(state="disabled")
            text.yview_moveto(0)

        toc.bind("<<ListboxSelect>>", show)
        toc.selection_set(0)
        show()
        win.bind("<Escape>", lambda _e: self.close_help())
        win.geometry("860x520")
        fit_min_size(win, 520)

    def close_help(self) -> None:
        if self._help_win is not None:
            try:
                self._help_win.destroy()
            except tk.TclError:
                pass
        self._help_win = None

    # ---- 生成情報 (画像に入っていた元のプロンプト。ステータスに出し、ボタンで別ウィンドウ) ----
    def generation_infos(self) -> dict[str, dict[str, str]]:
        """今の画像の生成情報 (枠 → 情報)。1 枚モードは画像 1 枚、複数枚モードは各枠。同じ画像を使う枠は 1 つにまとめる。"""
        slots = list(REF_SLOTS) if self.multi_mode() else ["person"]
        infos: dict[str, dict[str, str]] = {}
        seen: set[Path] = set()
        for slot in slots:
            path, info = self.slot_paths[slot], self.slot_gen[slot]
            if path is not None and info and path not in seen:
                infos[slot] = info
                seen.add(path)
        return infos

    def open_generation_info(self) -> None:
        infos = self.generation_infos()
        if not infos:
            self.status_var.set("この画像に生成情報はありません")
            return
        self._close_generation_info()
        win = tk.Toplevel(self.root)
        win.title(f"{APP_NAME} - 画像に入っていた生成情報")
        win.transient(self.root)
        win.protocol("WM_DELETE_WINDOW", self._close_generation_info)
        self._gen_win = win
        ttk.Button(win, text="閉じる", command=self._close_generation_info).pack(side="bottom", anchor="e", padx=8, pady=(0, 8))
        ttk.Label(win, text="画像を作ったときのプロンプトです (画像に保存されていたもの。i2t の推定ではありません)",
                  foreground="#666").pack(anchor="w", padx=8, pady=(8, 0))
        notebook = ttk.Notebook(win)
        notebook.pack(fill="both", expand=True, padx=6, pady=6)
        for slot, info in infos.items():
            path = self.slot_paths[slot]
            frame = ttk.Frame(notebook, padding=6)
            notebook.add(frame, text=REF_SLOTS[slot] if self.multi_mode() else path.name)
            ttk.Label(frame, text=f"{path.name}  —  {info['source']}").pack(anchor="w")
            for label, key, height in (("ポジティブ", "positive", 3), ("ネガティブ", "negative", 2),
                                       ("設定", "settings", 1), ("元のデータ", "raw", 2)):
                box = self._text_section(frame, label + ("" if info[key] else " (なし)"), height=height, copy=True,
                                         grow=key in ("positive", "raw"))
                box.insert("1.0", info[key])
        win.geometry("720x480")
        fit_min_size(win, 480)

    def _close_generation_info(self) -> None:
        if self._gen_win is not None:
            try:
                self._gen_win.destroy()
            except tk.TclError:
                pass
        self._gen_win = None

    # ---- 履歴 (50 件、結果 + 128px のサムネイル、「履歴...」で別ウィンドウ) ----
    def open_history(self) -> None:
        if self._history_win is not None:
            self._history_win.deiconify()
            self._history_win.lift()
            self._fill_history()
            return
        win = tk.Toplevel(self.root)
        win.title(f"{APP_NAME} - 履歴")
        win.transient(self.root)
        win.protocol("WM_DELETE_WINDOW", self.close_history)
        self._history_win = win
        ttk.Style(win).configure("History.Treeview", rowheight=HISTORY_THUMB // 2 + 8)
        # ボタンの列を先に下端へ置く (一覧を先に置くと、窓が低いときにボタンが下にはみ出して隠れた。2026-10-04 利用者の指摘)
        bar = ttk.Frame(win, padding=(6, 0, 6, 6))
        bar.pack(side="bottom", fill="x")
        ttk.Label(win, text=f"新しい順に {HISTORY_LIMIT} 件まで (古いものから消えます)。保存先: {HISTORY_DIR}",
                  foreground="#666").pack(side="bottom", anchor="w", padx=8)
        ttk.Button(bar, text="開く", command=self.open_history_entry).pack(side="left")
        ttk.Button(bar, text="削除", command=self.delete_history_entries).pack(side="left", padx=6)
        ttk.Button(bar, text="すべて削除...", command=self.clear_history).pack(side="left")
        ttk.Button(bar, text="閉じる", command=self.close_history).pack(side="right")
        frame = ttk.Frame(win, padding=6)
        frame.pack(fill="both", expand=True)
        columns = {"saved_at": ("日時", 140), "image": ("画像", 180), "target": ("出力先", 80), "positive": ("ポジティブ", 300)}
        # 行数の要求は小さく (既定の 10 行 × 72px が窓の高さを超えてボタンを押し出していた)。窓の残りの高さに広がる
        tree = ttk.Treeview(frame, columns=list(columns), show="tree headings", style="History.Treeview", height=3)
        tree.heading("#0", text="")
        tree.column("#0", width=HISTORY_THUMB // 2 + 24, stretch=False)
        for key, (title, width) in columns.items():
            tree.heading(key, text=title, anchor="w")
            tree.column(key, width=width, stretch=key == "positive", anchor="w")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="left", fill="y")
        tree.bind("<Double-1>", lambda _e: self.open_history_entry())
        tree.bind("<Return>", lambda _e: self.open_history_entry())
        self._history_tree = tree
        self._history_images: list[Any] = []  # PhotoImage の参照を保持 (GC 防止)
        self._fill_history()
        win.geometry("820x460")
        fit_min_size(win, 520)

    def close_history(self) -> None:
        if self._history_win is not None:
            try:
                self._history_win.destroy()
            except tk.TclError:
                pass
        self._history_win = None

    def _fill_history(self) -> None:
        from PIL import ImageTk

        tree = self._history_tree
        tree.delete(*tree.get_children())
        self._history_images = []
        for entry in history_entries():
            image: Any = ""
            if entry["thumb"] is not None:
                try:
                    with Image.open(entry["thumb"]) as im:
                        small = im.convert("RGB")
                    small.thumbnail((HISTORY_THUMB // 2, HISTORY_THUMB // 2), Image.LANCZOS)
                    if self.preview_mosaic_var.get():  # プレビューのモザイク表示に合わせる
                        w, h = small.size
                        small = small.resize((max(1, w // 8), max(1, h // 8)), Image.BOX).resize((w, h), Image.NEAREST)
                    image = ImageTk.PhotoImage(small)
                    self._history_images.append(image)
                except (OSError, ValueError):
                    image = ""
            name = Path(entry["image"]).name
            if entry["references"]:
                name += f" ほか {len(entry['references'])} 枚"
            positive = re.sub(r"\s+", " ", entry["positive"])[:200]
            tree.insert("", "end", iid=entry["id"], image=image,
                        values=(entry["saved_at"], name, entry["target"], positive))

    def open_history_entry(self) -> None:
        """選んだ履歴の結果を画面に戻す。元画像が残っていればプレビューにも読み込む。"""
        selection = self._history_tree.selection()
        if not selection:
            return
        if self.busy:
            self.status_var.set("処理の途中は履歴を開けません (「■ 中止」で止められます)")
            return
        path = HISTORY_DIR / f"{selection[0]}.json"
        try:
            result, meta = history_result(path)
        except (OSError, ValueError, TypeError) as e:
            messagebox.showerror(APP_NAME, f"履歴を読み込めません: {e}", parent=self._history_win)
            return
        refs = {slot: Path(p) for slot, p in (result.references or {}).items() if slot in REF_SLOTS and slot != "person"}
        mode = meta.get("mode") if meta.get("mode") in MODES else ("multi" if refs else "single")
        if self.mode_var.get() != mode:
            self.mode_var.set(mode)
            self._on_mode_change()
        for slot in REF_SLOTS:
            self._clear_slot(slot)
        missing: list[str] = []
        for slot, image in [("person", Path(result.image))] + list(refs.items()):
            if not (image.is_file() and self.load_image(image, quiet=True, slot=slot)):
                missing.append(image.name)
        self._render_preview()
        self._update_path_label()
        self._update_run_state()
        self._show_result(result, mosaic_level=meta.get("mosaic_strength"))
        self._history_path = path
        status = f"履歴を開きました ({meta.get('saved_at') or selection[0]}、出力先: {result.target})"
        if missing:
            status += f"  / 元画像が見つかりません (結果だけ表示): {', '.join(missing)}"
        self.status_var.set(status)
        if self._ja and self._chips_enabled():
            self._translate_and_show()

    def delete_history_entries(self) -> None:
        for stem in self._history_tree.selection():
            history_delete(stem)
        self._fill_history()

    def clear_history(self) -> None:
        entries = history_entries()
        if not entries or not messagebox.askyesno(
                APP_NAME, f"履歴 {len(entries)} 件をすべて削除しますか？ (元に戻せません)", parent=self._history_win):
            return
        for entry in entries:
            history_delete(entry["id"])
        self._fill_history()


def run_gui(initial_image: str | None) -> int:
    if tk is None:
        log("tkinter が利用できません。CLI モード (python i2t.py 画像.png) を使ってください。")
        return 2
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)  # 高 DPI でにじまないようにする
        except Exception:  # noqa: BLE001
            pass
    root = TkinterDnD.Tk() if DND_AVAILABLE else tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    App(root, load_config(), initial_image)
    root.mainloop()
    return 0


# --------------------------------------------------------------------------
# エントリポイント
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=APP_ID,
        description="画像からポジティブ / ネガティブプロンプトをローカルで生成します。引数なしで GUI が起動します。",
    )
    p.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    p.add_argument("images", nargs="*", help="画像ファイル (複数可)。省略すると GUI を起動")
    p.add_argument("--gui", action="store_true", help="GUI を起動する (画像を指定するとプレビューに読み込む)")
    p.add_argument("--engine", choices=list(ENGINES), help="hybrid: タガー + VLM / tagger: タガーのみ")
    p.add_argument("--tagger", choices=list(TAGGER_MODELS), help=f"タガーモデル (既定: {DEFAULT_TAGGER})")
    p.add_argument("--vlm", metavar="MODEL", help="Ollama の画像対応モデル (既定: qwen3-vl:8b-instruct)")
    p.add_argument("--describer", metavar="MODEL",
                   help="先に画像の説明を書かせるモデル (JoyCaption Beta One: " + JOYCAPTION_MODEL + ")。書いた説明を VLM に参考として渡す")
    p.add_argument("--host", metavar="URL", help="Ollama のホスト (既定: http://localhost:11434)")
    p.add_argument(
        "--target", choices=list(TARGETS),
        help="出力先モデル: sdxl (タグ) / anima (タグ+自然文) / flux, flux2, flux2klein, krea2, qwen, zimageturbo, zimage, mossimage, chatgpt, grok, nanobanana, midjourney (自然文、VLM 推奨)。既定 sdxl",
    )
    p.add_argument("--no-quality", action="store_true", help="品質タグ (masterpiece 等) を付けない")
    p.add_argument("--no-break", action="store_true", help="複数人物でも BREAK で区切らず 1 つのタグ列にする (SDXL 系)")
    p.add_argument("--mosaic", action="store_true", help="プロンプトにモザイク指定 (censored, mosaic censoring) を追加する")
    p.add_argument("--split-extras", action="store_true", help="カメラ・照明・画風のタグを本文から外し、追加プロンプト側だけに出す")
    p.add_argument("--split-background", action="store_true", help="背景のタグを本文から外し、本文を被写体だけにする (背景は追加プロンプト側)")
    p.add_argument("--no-age-look", action="store_true",
                   help="年齢の見た目を書かない (人物ごとに画像から読んだ年齢層を、文とタグに書く処理を省く)")
    p.add_argument("--no-polish", action="store_true",
                   help="文を仕上げない (文章の出力先・VLM: 集めた内容を VLM に 1 つの文章へ書き直させる処理を省く)")
    p.add_argument("--mosaic-prompt", dest="mosaic", action="store_true", help=argparse.SUPPRESS)  # 旧名
    p.add_argument("--mosaic-strength", choices=list(MOSAIC_LEVELS), help="モザイク指定の強さ (既定 strong)")
    p.add_argument("--general-threshold", type=float, metavar="F", help="一般タグのしきい値 (既定 0.35)")
    p.add_argument("--character-threshold", type=float, metavar="F", help="キャラクタータグのしきい値 (既定 0.85)")
    p.add_argument("--style-threshold", type=float, metavar="F", help="画風タグを「画風 / 質感」に拾うしきい値 (既定 0.15)")
    p.add_argument("--no-era", action="store_true", help="年代スタイル分類を行わない")
    p.add_argument(
        "--main-person", action="store_true",
        help="人物はメインの 1 人だけ (一番大きく写っている人物を検出して切り出し、ほかの人はプロンプトに入れない)",
    )
    p.add_argument(
        "--no-split-persons", action="store_true",
        help="2 人以上でも人物ごとに分けない (既定では人物を検出して 1 人ずつタグ付けし、人数タグを付け直す)",
    )
    p.add_argument("--gender", choices=["female", "male"], help="人物の性別をそろえる (female: 全員女性 / male: 全員男性)")
    p.add_argument("--restyle", choices=list(RESTYLES),
                   help="画風を変える (MOSS-Image だけ): photo (実写) / anime (アニメ風) / manga (漫画風・白黒) / watercolor (水彩) / "
                        "thick (厚塗り) / oil (油絵) / pencil (鉛筆画)。元の画像とは違う画風の画像として書く")
    p.add_argument("--no-anatomy-guard", action="store_true", help="人物の破綻を防ぐネガティブ (bad anatomy など) を足さない")
    p.add_argument("--color", action="store_true", help="白黒の情報を消す: 元画像が白黒でも白黒のタグを外してカラーで出す")
    p.add_argument("--person-details", action="store_true",
                   help="人物の特徴を詳しく: VLM で顔立ち・目・眉・鼻・口・髪型・肌・年齢の見た目・体格・目立つ特徴を読んで本文に入れる "
                        "(写真の人物にもイラストのキャラにも。--engine hybrid が必要)")
    p.add_argument("--avoid-duplicates", action="store_true",
                   help="分身を防ぐ: 同じ人物の複製・コラージュ・複数アングル・残像・顔や腕の重複をネガティブに入れる "
                        "(自然文の出力先は本文に一文)")
    p.add_argument("--turnaround", action="store_true",
                   help="三面図にする: 画像のポーズ・カメラ・背景・小物を外し、前・後ろ・横の三面図 (reference sheet, turnaround) にする")
    p.add_argument("--no-identity", action="store_true",
                   help="人物の特徴を入れない (LoRA 用): 髪・目・顔立ち・肌の色・キャラ名を外し、服装・体格・ポーズ・表情は残す")
    p.add_argument("--no-style", action="store_true",
                   help="作風の特徴を入れない (画風 LoRA 用): 画風・技法・絵か写真か (品質タグ・ネガティブの定番も)・年代・作品名 / 作者名を外す")
    p.add_argument("--avoid-same-face", action="store_true",
                   help="同じ顔を避ける: 2 人以上ならネガティブに clone, dual persona, multiple persona, twins, look-alike (ネガティブの無い出力先は本文で書き分け)")
    p.add_argument("--pose", metavar="IMAGE", help="複数枚モード: ② ポーズ (姿勢・手足・体の向き・構図) の参照画像")
    p.add_argument("--outfit", metavar="IMAGE", help="複数枚モード: ③ 服装 (服・靴・帽子・装飾品) の参照画像")
    p.add_argument("--background", metavar="IMAGE", help="複数枚モード: ④ 背景 (場所・背景・照明) の参照画像")
    p.add_argument("--props", metavar="IMAGE", help="複数枚モード: ⑤ 小物 (手に持つ物・置いてある物) の参照画像")
    p.add_argument("--body", metavar="IMAGE", help="複数枚モード: ⑥ 体格 (胸・体型・身長・肌の色・筋肉など) の参照画像。"
                   "同じ画像をほかの枠と同時に指定してもよい")
    p.add_argument("--style", metavar="IMAGE", help="複数枚モード: ⑦ 作風・質感 (画風・技法・色の塗り方・写真の撮影環境・作品名 / 作者名 (VLM)) の参照画像。"
                   "絵か写真かの判定もこの画像から行い、① の画風のタグは外す (絵 → 実写 / 実写 → 絵)")
    p.add_argument("--expression", metavar="IMAGE", help="複数枚モード: ⑧ 表情 (表情・視線) の参照画像")
    # 構図と強さ (LLM・GPU 不要)。GUI の「構図と強さ」欄と同じ
    p.add_argument("--bones-from", metavar="IMAGE", help="この画像から骨格を取り、そのポーズ (arms up など) を画像のポーズの代わりに使う "
                   "(骨格モデル RTMPose-s 約 21MB を初回のみ取得、CPU)")
    p.add_argument("--openpose-out", metavar="PNG", help="--bones-from の骨格を ControlNet (OpenPose) 用の画像に保存する")
    for dim in CAMERA_OPTIONS:
        values = [v for v, _, _ in CAMERA_OPTIONS[dim] if v]
        p.add_argument(f"--camera-{dim}", choices=values, help=f"カメラの{CAMERA_LABELS_JA[dim]}: " + " / ".join(
            f"{v} ({label})" for v, label, _ in CAMERA_OPTIONS[dim] if v))
    p.add_argument("--look", metavar="NAME", help="撮影プリセットの名前 (config.json の camera_looks。例: \"ポートレート 85mm\")。"
                   "画像から読んだレンズ・照明の質をプリセットの文に置き換える")
    p.add_argument("--strength", metavar="SPEC", help="枠ごとの強さ。例: outfit=strong,background=weak "
                   "(枠: person / pose / outfit / body / props / background / style、強さ: strong / normal / weak)")
    p.add_argument("--json", action="store_true", help="結果を JSON で出力する")
    p.add_argument("--save", action="store_true", help="各画像の隣に <名前>_prompt.txt を保存する")
    p.add_argument("--out", metavar="FILE", help="全結果をこのファイルにまとめて書き出す")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.gui or not args.images:
        return run_gui(args.images[0] if args.images else None)
    return run_cli(args)


if __name__ == "__main__":
    sys.exit(main())
