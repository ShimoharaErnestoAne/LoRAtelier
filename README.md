# LoRAtelier

[English](#english) | [日本語](#日本語)

<a id="english"></a>

## English

**This software comes with no warranty. Use it at your own risk. The author is not responsible for anything that results from its use.**

The license is the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) with the additional terms below (full text in `LICENSE`). The additional terms take precedence over PolyForm.

**Additional terms**

1. Above all, do not use this software to infringe the copyright or other rights of others.
2. You may use this software only for personal hobby purposes. Any other purpose, including research, education, use by organizations or work, is not permitted.
3. Do not publish or distribute anything you make with this software (LoRAs, images, captions and so on). However, you may publish things made only from material you hold the rights to (such as your own drawings or your own photos).
4. If you distribute this software, pass these terms along with it.
5. If you do not follow these terms, your permission to use this software ends.

The existing code and models it uses follow their own terms (`LICENSES.md`). The bundled ComfyUI custom nodes (`engines/bundled`) keep their original licenses (AGPL-3.0 / GPL-3.0).

LoRAtelier is a local tool for making character LoRAs and style LoRAs: add images → fix only what needs fixing → train → compare and choose → export. Training uses existing engines (sd-scripts / musubi-tuner) as they are, and comparison images are generated with ComfyUI.

The user interface (screens and help) is in Japanese.

### Starting

| What to do | File |
|---|---|
| First time only: prepare the environment (downloads several GB; on a PC without Python, a private Python is set up inside this folder) | `setup.bat` |
| Start (a browser opens) | `start.bat` |
| Start so that other PCs on the same network can use it too | `start_intranet.bat` |

- When started with `start.bat`, the screen is at `http://127.0.0.1:7865/`. Only this PC can connect (no public link is created).
- When started with `start_intranet.bat`, other PCs and tablets on the same network (home or office LAN) can open it in their browsers too. The address to open (for example `http://192.168.1.10:7865/`) is shown in the black window that starts.
  - There is no password. Anyone on the same network can operate it, so use it only on networks you trust. It is not exposed to the internet.
  - If other PCs cannot open it, allow python.exe through the Windows firewall (private networks). If a permission dialog appears the first time you start it, you can allow it there.
  - Training and image generation run on the GPU of the PC that started it. Exported LoRAs and model folders are on that PC, and "Open folder" opens on that PC's screen.
  - It cannot run at the same time as `start.bat`. To switch, close the running black window first.
- Requirements: Windows, an NVIDIA GPU (12GB VRAM or more recommended), 32GB RAM or more recommended (while training SDXL, the training process alone uses about 15GB).
- Images are never sent outside. Apart from the first download of AI models (for material analysis, about 15GB), it works without a network connection.

### How to use

The "ヘルプ" (Help) button at the top right explains the operations, where to put models and what to do when something goes wrong (the text is in `app/lorastudio/static/help.html`).

1. First time only, prepare a base model. Setup does not download models.
   - "標準モデルをダウンロード" (Download standard models) on the home screen gets three SDXL models (Illustrious-XL v2.0, Pony Diffusion V6 XL, SDXL 1.0 base).
   - If you already have model files, just put them in `models\checkpoints` and they appear in the list (Anima and Qwen-Image main files go in `models\diffusion_models`, text encoders in `models\text_encoders`, VAEs in `models\vae`).
   - Anima, Qwen-Image-2512, Qwen-Image-2.1, MOSS-Image and others can be added as a set from **Settings** → "おすすめのモデル" (Recommended models). You can also paste a Hugging Face or Civitai link, or drop a downloaded file (PixAI and others); the contents are checked and added as a model or LoRA.
2. On the home screen, choose **Character LoRA / Style LoRA**.
3. **Materials**: drag and drop images (pasting with Ctrl+V also works). They appear in the list right away and the checks start automatically.
   - "要確認を優先" (Needs review first) puts images that may have problems at the top. The reason, the confidence and whether it is a fact or an estimate are shown.
   - Use / exclude, cropping (by dragging) and caption edits are saved automatically. Ctrl+Z undoes them.
   - Excluding is only a suggestion. Original images are never deleted.
4. **Start training** (one click). Settings are chosen automatically. To change them, open "学習" (Training) → "詳細設定" (Advanced settings).
5. **Results**: choose from comparison images lined up by no LoRA / saved checkpoints / strengths, then **export**.

### When something goes wrong

| What you see | Where to look |
|---|---|
| Training "failed" | The red explanation and suggested fixes on the training screen. Details are in the technical log (`data/projects/<id>/runs/<run>/logs/`) |
| Out of GPU memory | In the advanced settings, set the batch size to 1, lower the resolution or increase block swap |
| AI analysis does not progress | The status at the top right (it waits while training). `data/analysis_worker.log` |
| No comparison images | `data/comfyui.log` |
| Errors on the screen | `data/server_errors.log` |
| Original images not found | Shown on the materials screen. If you point to a folder, only files with matching hashes are restored |

### Where data is kept

- `data/projects/<id>/` … projects (`sources/` has copies of the original images, `project.db` the edits, `runs/` the fixed data and results of each training, `exports/` the export folder of older versions)
- `models/loras/<LoRA name>/` … exported LoRAs (the LoRA file, 使い方.txt (usage notes) and lora_info.json)
- `data/settings.json` … registered models
- "Saving the project" is automatic, "saving for resume" is the in-training save (including optimizer state), and "export" is the LoRA you use in image generation software.

### Supported models

The supported families are SDXL (Illustrious / NoobAI / Pony / SDXL 1.0) and Anima. Qwen-Image-2512, Qwen-Image-2.1 and Z-Image are also supported. Image generation has been confirmed for all of them; for training, only a short Z-Image Turbo run has been tried (training of the Qwen models is unverified). When added from "Recommended models" in the settings, a lighter version for generation (GGUF quantization level, text encoder version) is chosen to fit the GPU's VRAM, and the bf16 version the training engine needs is fetched separately for training. Before training starts, it also checks that main memory meets the recommendation (64GB for Qwen-Image-2512). Training for 2.1 uses the development version of the musubi-tuner training engine. MOSS-Image is experimental, and both generation and training are unverified (training uses a separately installed sd-scripts v0.12.1 that supports MOSS-Image, and INT8 loading and block swap are chosen automatically to fit the VRAM). Krea 2 supports image generation and material generation (generation has been confirmed to work). Training uses the RAW main model (selectable since 2026-10-28; training from this tool is unverified). If you add your ComfyUI folder in the settings, ComfyUI's models and LoRAs can be used as they are, without copying. LoRAs cannot be shared between different model families.

The feature that makes a character LoRA from AI-generated images lets you choose the checkpoint used for generation, file by file. Images are generated on a plain white background, and background instructions in the prompt (`city street`, `outdoors` and so on) are removed before generating (the removed words are shown on the screen).

"画像からプロンプト" (Image to prompt) at the top right gives you every feature of i2t (a tool that makes prompts for Stable Diffusion and others from images).

"画像生成" (Image generation) at the top right generates images from just a model, LoRAs, a prompt and an image size. If you like the result, you can copy those settings into the generation settings of a project that makes a character LoRA from AI-generated images (the image size and seed are not copied).

Sources and licenses: `LICENSES.md`

---

<a id="日本語"></a>

## 日本語

**このソフトは無保証です。自己責任で使ってください。使用によって生じたことについて、作者は責任を負いません。**

ライセンスは [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) に、次の追加の条件を加えたものです（全文は `LICENSE`）。追加の条件は PolyForm より優先します。

**追加の条件**

1. 大前提として、他人の著作権やその他の権利を侵害することに、このソフトを使わないでください。
2. このソフトは、個人の趣味の目的でだけ使えます。研究・教育・団体での使用や仕事など、それ以外の目的では使えません。
3. このソフトで作ったもの（LoRA・画像・説明文など）を公開・配布しないでください。ただし、自分が権利を持つ素材（自分で描いた絵、自分で撮った写真など）だけを使って作ったものは、公開できます。
4. このソフトを配布するときは、この条件も一緒に渡してください。
5. これらの条件を守らなかった場合、このソフトを使う許可はなくなります。

使っている既存のコードとモデルは、それぞれの利用条件に従います（`LICENSES.md`）。同梱している ComfyUI のカスタムノード（`engines/bundled`）は、それぞれ元のライセンス（AGPL-3.0 / GPL-3.0）のままです。

キャラクター LoRA とスタイル LoRA を、画像を入れて → 必要な所だけ直して → 学習 → 比較して選ぶ → 書き出す、の流れで作るローカルツールです。学習は既存のエンジン（sd-scripts / musubi-tuner）、比較画像の生成は ComfyUI をそのまま使います。

画面とヘルプは日本語です。

### 起動

| 操作 | ファイル |
|---|---|
| 初回だけ: 環境の準備（数 GB のダウンロード。Python が入っていない PC では、このフォルダーの中に専用の Python を用意します） | `setup.bat` |
| 起動（ブラウザーが開きます） | `start.bat` |
| 同じネットワークの他の PC からも使えるように起動 | `start_intranet.bat` |

- `start.bat` で起動したときの画面は `http://127.0.0.1:7865/` です。この PC からだけ接続できます（外部公開のリンクは作りません）。
- `start_intranet.bat` で起動すると、同じネットワーク（家庭や職場の LAN）の他の PC やタブレットのブラウザーからも開けます。開くアドレス（例: `http://192.168.1.10:7865/`）は、起動した黒いウィンドウに表示されます。
  - パスワードはありません。同じネットワークにいる人は誰でも操作できるので、信頼できるネットワークだけで使ってください。インターネットには公開されません。
  - 他の PC から開けないときは、Windows のファイアウォールで python.exe の通信（プライベート ネットワーク）を許可してください。初めて起動したときに許可を求める画面が出た場合は、そこで許可できます。
  - 学習や画像生成は、起動した PC の GPU で動きます。書き出した LoRA やモデルの置き場所も起動した PC のフォルダーで、「フォルダーを開く」も起動した PC の画面に開きます。
  - `start.bat` と同時には起動できません。切り替えるときは、起動中の黒いウィンドウを閉じてから起動してください。
- 動作環境: Windows、NVIDIA GPU（VRAM 12GB 以上推奨）、RAM 32GB 以上推奨（SDXL の学習中は学習プロセスだけで約 15GB 使います）。
- 画像は外部に送信しません。初回の AI モデル取得（素材解析用、約 15GB）以外は、通信なしで動きます。

### 使い方

画面右上の「ヘルプ」に、操作の説明・モデルの置き場所・うまくいかないときの対処をまとめています（本文は `app/lorastudio/static/help.html`）。

1. 初回だけ、ベースモデルを用意します。セットアップではモデルを取得しません。
   - ホームの「標準モデルをダウンロード」で、SDXL 系 3 種（Illustrious-XL v2.0・Pony Diffusion V6 XL・SDXL 1.0 base）を取得できます。
   - 手元にモデルファイルがあれば、`models\checkpoints` に置くだけで一覧に出ます（Anima・Qwen-Image の本体は `models\diffusion_models`、テキストエンコーダーは `models\text_encoders`、VAE は `models\vae`）。
   - Anima・Qwen-Image-2512・Qwen-Image-2.1・MOSS-Image などは、**設定** →「おすすめのモデル」から一式で追加できます。Hugging Face・Civitai のリンクを貼るか、ダウンロードしたファイル（PixAI など）をドロップしても、中身を確認してモデルや LoRA を追加できます。
2. ホームで **キャラクター LoRA / スタイル LoRA** を選ぶ。
3. **素材**: 画像をドラッグ＆ドロップ（Ctrl+V での貼り付けも可）。すぐに一覧に出て、確認が自動で始まります。
   - 「要確認を優先」で、問題のありそうな画像が先頭に並びます。理由・確信度・事実か推定かを表示します。
   - 採用／除外、切り抜き（ドラッグ）、説明文の編集は自動保存。Ctrl+Z で元に戻せます。
   - 除外は提案です。元の画像は削除しません。
4. **学習を開始**（1 回のクリック）。設定は自動で決まります。変えたい場合は「学習」→「詳細設定」。
5. **結果**: LoRA なし / 途中の保存版 / 強さ を並べた比較画像から選び、**書き出す**。

### うまくいかないとき

| 表示 | 確認する場所 |
|---|---|
| 学習が「失敗」 | 学習画面の赤字の説明と対処案。詳細は「技術ログ」（`data/projects/<id>/runs/<run>/logs/`） |
| GPU メモリ不足 | 詳細設定でバッチサイズ 1、解像度を下げる、ブロックスワップを増やす |
| AI 解析が進まない | 右上の表示（学習中は待機します）。`data/analysis_worker.log` |
| 比較画像が出ない | `data/comfyui.log` |
| 画面のエラー | `data/server_errors.log` |
| 元画像が見つからない | 素材画面で表示。フォルダーを指定すると、ハッシュが一致するファイルだけを戻します |

### データの場所

- `data/projects/<id>/` … プロジェクト（`sources/` に元画像のコピー、`project.db` に編集内容、`runs/` に学習ごとの固定データと結果、`exports/` は以前の版の書き出し先）
- `models/loras/<LoRA 名>/` … 書き出した LoRA（LoRA 本体・使い方.txt・lora_info.json）
- `data/settings.json` … 登録したモデル
- 「プロジェクトの保存」は自動、「再開用の保存」は学習中の途中保存（オプティマイザー状態を含む）、「書き出し」は生成ソフトで使う LoRA です。

### 対応モデル

対応する系列は SDXL（Illustrious / NoobAI / Pony / SDXL 1.0）と Anima です。Qwen-Image-2512・Qwen-Image-2.1・Z-Image にも対応します。どれも画像生成は確認済みで、学習は Z-Image Turbo の短い学習だけを試しています（Qwen 系の学習は未検証）。設定画面の「おすすめのモデル」から追加すると、GPU の VRAM に合わせて生成用の軽量版（GGUF の量子化の段階、テキストエンコーダーの版）を選び、学習用には学習エンジンが求める bf16 版を別に取得します。学習を始める前に、メインメモリが推奨（Qwen-Image-2512 は 64GB）に足りるかも確認します。2.1 の学習には学習エンジン musubi-tuner の開発版を使います。MOSS-Image は試験導入中で、生成も学習も未検証です（学習には MOSS-Image に対応した sd-scripts v0.12.1 を別に入れて使い、VRAM に合わせて INT8 での読み込みやブロックスワップを自動で選びます）。Krea 2 は画像生成と素材の生成に対応しています（生成は動作を確認済み）。学習は RAW 版の本体で行います（2026-10-28 から選べます。このツールからの学習は未検証です）。ComfyUI のフォルダーを設定で追加すると、ComfyUI のモデルと LoRA をコピーせずにそのまま使えます。系列が違うモデルの間で LoRA は共用できません。

AI 生成画像からキャラ LoRA を作る機能では、生成に使う checkpoint をファイルごとに選べます。背景は白一色で生成し、プロンプトに書かれた背景の指定（`city street`、`outdoors` など）は外して生成します（外した語は画面に表示します）。

画面右上の「画像からプロンプト」では、i2t（画像から Stable Diffusion などのプロンプトを作るツール）の機能をすべて使えます。

画面右上の「画像生成」では、モデル・LoRA・プロンプト・画像サイズだけを指定して画像を生成できます。生成した結果が良ければ、その設定を AI 生成画像からキャラ LoRA を作るプロジェクトの生成設定にコピーできます（画像サイズと seed はコピーしません）。

出典とライセンス: `LICENSES.md`
