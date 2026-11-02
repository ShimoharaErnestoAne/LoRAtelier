# LoRAtelier

[English](#english) | [日本語](#日本語)

<a id="english"></a>

## English

**This software comes with no warranty. Use it at your own risk. The author is not responsible for anything that results from its use.**

The license is the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) with the additional terms below (full text in `LICENSE`). The additional terms take precedence over PolyForm.

**Additional terms**

1. Above all, do not use this software to infringe the copyright or other rights of others.
2. You may use this software only as an individual, for personal hobby purposes. You may not use it for research, education, work or similar purposes, or within companies, schools or other organizations.
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

**このソフトは無保証です。使うときは自己責任でお願いします。使ったことで何か問題が起きても、作者は責任を負いません。**

ライセンスは、PolyForm Noncommercial License 1.0.0 に、次の「追加の条件」を足したものです（全文は `LICENSE` にあります）。PolyForm と追加の条件の内容が食い違うときは、追加の条件のほうに従ってください。

**追加の条件**

1. 他人の著作権などの権利を侵害する使い方は、決してしないでください。これがすべての前提です。
2. 使ってよいのは、個人が趣味で使う場合だけです。研究や教育、仕事のために使うことはできません。会社や学校などの団体で使うこともできません。
3. このソフトで作った LoRA や画像、説明文などは、公開も配布もしないでください。ただし、自分で描いた絵や自分で撮った写真のように、自分が権利を持っている素材だけから作ったものは、公開してもかまいません。
4. このソフトを人に渡すときは、この条件も一緒に渡してください。
5. この条件を守れない場合は、このソフトを使うことはできません。

このソフトの中で使っている他のプログラムやモデルには、それぞれに利用条件があります（`LICENSES.md` にまとめています）。一緒に入れている ComfyUI 用の拡張（`engines/bundled`）は、元のライセンス（AGPL-3.0 / GPL-3.0）のまま使っています。

LoRAtelier は、キャラクター LoRA とスタイル LoRA を自分の PC の中だけで作るためのツールです。画像を入れたら、気になるところだけを直して学習し、できあがりを比較画像で見比べて、良いものを書き出します。学習には既存の学習エンジン（sd-scripts と musubi-tuner）を、比較画像の生成には ComfyUI を、手を加えずにそのまま使っています。

画面とヘルプは日本語です。

### 起動のしかた

| やりたいこと | 使うファイル |
|---|---|
| 最初に一度だけ、使うための準備をする（数 GB をダウンロードします。Python が入っていない PC では、このフォルダーの中に専用の Python を用意します） | `setup.bat` |
| 起動する（ブラウザーが開きます） | `start.bat` |
| 同じネットワークにある他の PC からも使えるように起動する | `start_intranet.bat` |

- `start.bat` で起動すると、ブラウザーに `http://127.0.0.1:7865/` が開きます。この画面を開けるのは、起動した PC だけです。インターネットに公開するリンクは作りません。
- `start_intranet.bat` で起動すると、家や職場の LAN につながっている他の PC やタブレットのブラウザーからも開けるようになります。開くアドレス（たとえば `http://192.168.1.10:7865/`）は、起動したときの黒いウィンドウに出ます。
  - パスワードはかかっていません。同じネットワークにいる人なら誰でも操作できてしまうので、信頼できるネットワークの中だけで使ってください。インターネットからは開けません。
  - 他の PC から開けないときは、Windows のファイアウォールの設定で、python.exe の通信（プライベート ネットワーク）を許可してください。最初に起動したときに許可を求める画面が出たら、そこで許可することもできます。
  - 学習や画像の生成に使われるのは、起動した PC の GPU です。書き出した LoRA やモデルも、起動した PC のフォルダーに置かれます。「フォルダーを開く」を押したときも、起動した PC の画面でフォルダーが開きます。
  - `start.bat` と同時に起動することはできません。切り替えたいときは、先に起動中の黒いウィンドウを閉じてください。
- 動かすには、Windows と NVIDIA の GPU が必要です。GPU のメモリ（VRAM）は 12GB 以上、PC のメモリは 32GB 以上をおすすめします。SDXL の学習中は、学習の処理だけで 15GB ほどメモリを使います。
- 画像を外に送ることはありません。最初に素材の解析用の AI モデル（約 15GB）をダウンロードしたあとは、ネットにつながっていなくても動きます。

### 使い方

操作のしかたや、モデルの置き場所、うまくいかないときの対処は、画面の右上にある「ヘルプ」にまとめてあります（中身は `app/lorastudio/static/help.html` です）。

1. 最初に一度だけ、もとになるモデル（ベースモデル）を用意します。準備（`setup.bat`）の段階ではモデルをダウンロードしません。
   - ホーム画面の「標準モデルをダウンロード」を押すと、SDXL 系の 3 つのモデル（Illustrious-XL v2.0、Pony Diffusion V6 XL、SDXL 1.0 base）をダウンロードできます。
   - モデルのファイルをすでに持っているなら、`models\checkpoints` に入れるだけで一覧に出てきます。Anima や Qwen-Image の場合は、本体を `models\diffusion_models` に、テキストエンコーダーを `models\text_encoders` に、VAE を `models\vae` に入れてください。
   - Anima、Qwen-Image-2512、Qwen-Image-2.1、MOSS-Image などは、**設定** の「おすすめのモデル」から、必要なファイルをまとめて追加できます。Hugging Face や Civitai のリンクを貼ったり、ダウンロードしたファイル（PixAI のものなど）を画面に落としたりしても、中身を確かめたうえでモデルや LoRA として追加します。
2. ホーム画面で、**キャラクター LoRA** か **スタイル LoRA** を選びます。
3. **素材** の画面に、画像をドラッグ＆ドロップします（Ctrl+V で貼り付けることもできます）。画像はすぐに一覧に並び、自動で確認が始まります。
   - 「要確認を優先」を選ぶと、問題がありそうな画像が先頭に並びます。それぞれの画像には、気になる理由と、その判定がどのくらい確かか、はっきりした事実なのか推測なのかが表示されます。
   - 画像を使うかどうかの切り替えや、切り抜き（ドラッグで範囲を決めます）、説明文の書き直しは、すべて自動で保存されます。Ctrl+Z で元に戻すこともできます。
   - 「除外」はあくまで提案です。元の画像が消えることはありません。
4. **学習を開始** を押します。押すのは 1 回だけです。設定は自動で決まりますが、自分で変えたいときは「学習」の「詳細設定」で変えられます。
5. **結果** の画面で、LoRA を使わない場合、途中で保存した版、強さを変えた場合を並べた比較画像を見て、良いものを選んで **書き出し** ます。

### うまくいかないとき

| 困っていること | 見るところ・試すこと |
|---|---|
| 学習が「失敗」になった | 学習の画面に、赤い字で原因と対処のしかたが出ます。詳しい記録は「技術ログ」（`data/projects/<id>/runs/<run>/logs/`）にあります。 |
| GPU のメモリが足りない | 詳細設定で、バッチサイズを 1 にするか、解像度を下げるか、ブロックスワップを増やしてみてください。 |
| AI の解析が進まない | 画面右上の表示を見てください。学習中は、学習が終わるまで解析が待たされます。記録は `data/analysis_worker.log` にあります。 |
| 比較画像ができない | `data/comfyui.log` を見てください。 |
| 画面にエラーが出る | `data/server_errors.log` を見てください。 |
| 元の画像が見つからない | 素材の画面に、見つからない画像が表示されます。今その画像があるフォルダーを指定すると、中身が同じファイルだけを探して戻します。 |

### データの置き場所

- `data/projects/<id>/` に、プロジェクトごとのデータがあります。`sources/` には元の画像のコピー、`project.db` には編集した内容、`runs/` には学習ごとのデータと結果が入っています。`exports/` は、以前の版で書き出しに使っていたフォルダーです。
- `models/loras/<LoRA の名前>/` に、書き出した LoRA があります。LoRA のファイルのほか、使い方.txt と lora_info.json も入っています。
- `data/settings.json` には、登録したモデルが記録されています。
- 「プロジェクトの保存」は自動で行われます。「再開用の保存」は、学習の途中から続けるための保存で、オプティマイザーの状態も含みます。「書き出し」は、画像生成ソフトで使うための LoRA を出力することです。

### 使えるモデル

- **SDXL 系と Anima**：SDXL 系（Illustrious、NoobAI、Pony、SDXL 1.0）と Anima に対応しています。
- **Qwen-Image と Z-Image**：Qwen-Image-2512、Qwen-Image-2.1、Z-Image にも対応しています。どれも画像の生成は確かめてあります。学習のほうは、Z-Image Turbo で短く学習してみただけで、Qwen 系の学習はまだ確かめていません。
  - 設定の「おすすめのモデル」から追加すると、生成用には GPU のメモリに合った軽い版（GGUF の量子化の段階や、テキストエンコーダーの版）を選びます。学習用には、学習エンジンが必要とする bf16 の版を別にダウンロードします。
  - 学習を始める前に、PC のメモリが足りているかも確かめます（Qwen-Image-2512 は 64GB をおすすめします）。
  - Qwen-Image-2.1 の学習には、学習エンジン musubi-tuner の開発中の版を使います。
- **MOSS-Image**：試しに入れている段階で、生成も学習もまだ確かめていません。学習には、MOSS-Image に対応した sd-scripts v0.12.1 を別に入れて使います。INT8 で読み込むかどうかや、ブロックスワップの量は、GPU のメモリに合わせて自動で決めます。
- **Krea 2**：画像の生成と、素材の生成に使えます。生成は動くことを確かめてあります。学習には RAW 版の本体を使います。2026-10-28 から選べるようにしましたが、このツールからの学習はまだ確かめていません。
- **ComfyUI のモデル**：設定で ComfyUI のフォルダーを追加すると、ComfyUI に入っているモデルや LoRA を、コピーせずにそのまま使えます。
- **系統の違うモデル**：系統の違うモデルの間では、LoRA を使い回せません。

AI で生成した画像からキャラ LoRA を作るときは、生成に使う checkpoint を、ファイルごとに選べます。背景は白一色で生成します。プロンプトに背景の指定（`city street` や `outdoors` など）が書かれていても、その語は外して生成します。外した語は画面に表示します。

画面右上の「画像からプロンプト」では、i2t（画像から Stable Diffusion などのプロンプトを作るツール）の機能をすべて使えます。

画面右上の「画像生成」では、モデルと LoRA、プロンプト、画像の大きさを決めるだけで画像を生成できます。気に入った結果が出たら、その設定を、AI 生成画像からキャラ LoRA を作るプロジェクトの生成設定に写せます（画像の大きさと seed は写しません）。

使っているプログラムやモデルの出典とライセンスは、`LICENSES.md` にまとめています。
