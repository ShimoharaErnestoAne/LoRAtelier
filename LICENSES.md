# 使用している既存実装・モデルと利用条件

このツールは学習・推論・解析を独自実装せず、以下の既存実装を呼び出します。コード・モデルの重み・データの利用条件はそれぞれ別です。モデルの重みは各自が利用条件に同意して入手します（本ツールの配布物には含めません）。

## 学習エンジン・生成環境（コード）

| 名称 | 用途 | ライセンス | 組み込み方 |
|---|---|---|---|
| i2t（作者: ShimoharaErnestoAne。LoRAtelier と同じ） | 「画像からプロンプト」（`engines/i2t/` に版を固定して同梱、解析用環境で実行） | PolyForm Noncommercial 1.0.0 と追加の条件（LoRAtelier と同じ） | 同梱の訳の辞書 `assets/tags_ja.json` は WD Tagger v3 のタグ一覧（Apache-2.0, SmilingWolf）を Index-Translate-2B（Apache-2.0, IndexTeam）で訳し、手直ししたもの。実行時に取得するモデル（WD14 タガー、DeepGHS の人物検出・年代判定、RTMPose-s）はそれぞれの利用条件に従う |
| [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) | SDXL / Anima の LoRA 学習 | Apache-2.0 | 別プロセスで実行（`engines.lock.json` の commit に固定） |
| [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner) | Qwen-Image・Z-Image の LoRA 学習（Krea 2 用の部分は積み残し） | Apache-2.0 | 別プロセスで実行（commit 固定） |
| [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner) dev ブランチ | Qwen-Image-2.1 の LoRA 学習 | Apache-2.0 | PR #1145 を取り込んだ commit（9702b25）に固定。Qwen-Image-2.1 学習用ファイルを追加したときだけ取得します |
| sd-scripts v0.12.1（[kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) 04ca541） | MOSS-Image の LoRA 学習（試験導入・未検証） | Apache-2.0 | 固定版とは別の場所（`engines/sd-scripts-moss`）に commit 固定で取得し、学習用環境を共用します。MOSS-Image を追加したときだけ取得し、高速化用の triton-windows（MIT）も学習用環境に入れます |
| [suigin10/MOSS-LoRA-Trainer](https://github.com/suigin10/MOSS-LoRA-Trainer) v0.1.2 | sd-scripts で途中から再開したときのステップ数の補正（`app/lorastudio/bridges/sd_resume.py`） | MIT（Copyright (c) 2026 suigin10） | `moss_trainer/resume_bridge.py` の補正部分を手直しして使っています。学習率の再計算と state の整理の部分は使っていません |
| [leejet/ComfyUI-GGUF](https://github.com/leejet/ComfyUI-GGUF)（[city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) の派生） | Qwen-Image-2512 / 2.1 の GGUF 版での生成 | Apache-2.0 | ComfyUI のカスタムノードとして commit 固定（edd981b）で取得。本家は Qwen-Image-2.1 の GGUF を判別できないため、対応を足した派生版を使っています。ComfyUI-GGUF#487 の不具合への小さな修正を当てています（`engines/patches/`） |
| [pamparamm/ComfyUI-ppm](https://github.com/pamparamm/ComfyUI-ppm) | 画像生成の NegPip（マイナスの強さ・MOSS-Image の強調）、左右で描き分ける（Attention Couple）、白飛びを抑える（RescaleCFGPost）、スケジューラー ays / gits | AGPL-3.0-or-later | commit 543dbe3 のソースコードを、手を加えずにライセンス文（LICENSE）ごと配布用 zip の `engines/bundled/ComfyUI-ppm` に同梱しています（元のソースは上のリンク先）。ComfyUI を初めて起動するときに ComfyUI のカスタムノードの場所へコピーします（同梱分がなければ git で取得）。本ツールのコードとはリンクしていません（ComfyUI の別プロセスの中で動き、HTTP API 経由でノード名を指定して使うだけです） |
| [PedroMarinhoDev/ComfyUI-Moss-Image-Edit-Native](https://github.com/PedroMarinhoDev/ComfyUI-Moss-Image-Edit-Native) | 画像生成で、MOSS-Image の編集用 LoRA を使って元画像をプロンプトの指示どおりに編集する（MOSSImageEditTextEncode） | GPL-3.0 | commit d5cd7b2 のソースコードを、手を加えずにライセンス文（LICENSE）ごと配布用 zip の `engines/bundled/ComfyUI-Moss-Image-Edit-Native` に同梱しています（README 用の画像・見本の画像は省いています。元のソースは上のリンク先）。ComfyUI を初めて起動するときに ComfyUI のカスタムノードの場所へコピーします（同梱分がなければ git で取得）。本ツールのコードとはリンクしていません（ComfyUI の別プロセスの中で動き、HTTP API 経由でノード名を指定して使うだけです）。編集用 LoRA（chinoll/Moss-Image-Edit、akhaliq/MOSS-Image-Edit-HQ など）は同梱せず、ダウンロードもしません |
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | 比較画像の生成・読み込み確認 | GPL-3.0 | 別プロセスとして HTTP API 経由で使用。本ツールのコードとはリンクしていません |

## 素材解析（コード／モデル）

| 名称 | 用途 | ライセンス |
|---|---|---|
| [deepghs/imgutils](https://github.com/deepghs/imgutils) | アニメ系の人物・頭部検出、画風分類の呼び出し | MIT |
| SmilingWolf/wd-eva02-large-tagger-v3 | タグ推定（説明文・偏り判定） | Apache-2.0 |
| deepghs/anime_person_detection, anime_face_detection | 人物・顔検出 | MIT |
| deepghs/ccip_onnx | キャラクター同一性の推定 | OpenRAIL |
| deepghs/anime_real_cls | 写真／イラストの推定 | OpenRAIL |
| torchvision Faster R-CNN ResNet50-FPN v2 (COCO) | 写真の人物検出 | BSD-3-Clause（torchvision） |
| torchvision Mask R-CNN ResNet50-FPN v2 (COCO) | マスク学習用の、写真の人物の範囲 | BSD-3-Clause（torchvision） |
| [skytnt/anime-seg](https://huggingface.co/skytnt/anime-seg)（ISNet、imgutils 経由） | マスク学習用の、イラストの人物の範囲 | Apache-2.0 |
| google/siglip2-so400m-patch16-384 | 画像特徴（類似・偏り・評価補助） | Apache-2.0 |
| Qwen/Qwen3-VL-4B-Instruct | 自然文の説明文生成 | Apache-2.0 |

[IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) は PolyForm Noncommercial 等の記載があるため採用していません（画質は機械的な指標で判定）。

## UI・サーバー

FastAPI (MIT), Uvicorn (BSD-3), Pillow (MIT-CMU), NumPy (BSD-3)。依存一覧と版は `envs/*.lock.txt`。

タグの入力補完で使う読みの表 `app/lorastudio/static/tag_readings.json` は、i2t の訳の辞書の日本語を [Janome](https://github.com/mocobeta/janome) 0.5.0（Apache-2.0）と同梱の IPADIC 2.7.0（IPADIC のライセンス。Copyright 2000-2007 Nara Institute of Science and Technology）で読みに直して作ったものです（作り方は `app/scripts/make_tag_readings.py`）。Janome はこの表を作るときだけ使い、アプリには含めていません。

## 参考にした既存ツール

- [LoRA Ease](https://huggingface.co/spaces/multimodalart/lora-ease)（MIT）: キャラ／スタイルの分岐と、用途に応じた設定の考え方を参考にしました。コードはクラウド前提・古い依存のため取り込んでいません。
- [OneTrainer](https://github.com/Nerogar/OneTrainer), [AI Toolkit](https://github.com/ostris/ai-toolkit): 比較検討の対象。採用していません。

## ベースモデル（各自が入手）

| モデル | ライセンス（モデルページの表記） |
|---|---|
| Illustrious-XL v2.0 | creativeml-openrail-m |
| SDXL 1.0 base | CreativeML Open RAIL++-M |
| Pony Diffusion V6 XL | 独自ライセンス（配布元で確認） |
| Anima (circlestone-labs) | CircleStone Labs Non-Commercial License（非商用） |
| Anima 2.9B（Gazingstars123/Anima-2.9B、Anima を元にしたモデル） | CircleStone Labs Non-Commercial License（非商用） |
| Qwen-Image-2512（Comfy-Org/Qwen-Image_ComfyUI） | Apache-2.0 |
| Qwen-Image-2512 GGUF（unsloth/Qwen-Image-2512-GGUF） | Apache-2.0 |
| Qwen-Image-2.1 / 2.1 Turbo（Comfy-Org/Qwen-Image-2.1、Qwen/Qwen-Image-2.1-Turbo） | Qwen Research License（非商用のみ） |
| Qwen-Image-2.1 Turbo GGUF（unsloth/Qwen-Image-2.1-Turbo-GGUF） | 元モデルと同じ Qwen Research License（非商用のみ） |
| Z-Image Turbo / Base（Comfy-Org/z_image_turbo、Comfy-Org/z_image、Tongyi-MAI/Z-Image のトークナイザー） | Apache-2.0 |
| Z-Image Turbo 学習アダプター（ostris/zimage_turbo_training_adapter） | Apache-2.0 |
| MOSS-Image SFT / Base、Qwen3.5-9B テキストエンコーダー、FLUX.2 small decoder VAE（Comfy-Org/MOSS-Image、OpenMOSS-Team/MOSS-Image-Base のトークナイザー） | Apache-2.0（VAE は Apache-2.0 の small decoder 版を使います。FLUX.2-dev の flux2-vae.safetensors は非商用ライセンスのため取得しません） |
| Krea 2 RAW / Turbo（積み残し・未対応） | Krea 2 Community License（Hugging Face でログインと同意が必要） |

作った LoRA の利用・配布は、ベースモデルのライセンスと学習素材の権利の両方に従ってください。

## 取り込んだコードのライセンス全文

`app/lorastudio/bridges/sd_resume.py` は、MOSS-LoRA-Trainer の次のライセンスのもとでコードを手直しして使っています。

```
MIT License

Copyright (c) 2026 suigin10

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
