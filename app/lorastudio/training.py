"""Training runs: frozen run snapshots, engine adapters and the serial GPU job runner.

* A run snapshot freezes images (with crops applied), captions, model files, all settings,
  the preset version and the engine commit.  Edits made during training go to the project
  and only affect the *next* run.
* Engines are called with their official TOML config files (no shell string building).
* Progress comes from the engine's own step counter (tqdm "steps: n/N"), never synthesised.
* Stop = "safe stop at the next checkpoint save"; force stop kills the process tree.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import traceback
from pathlib import Path

import toml
from PIL import Image, ImageOps

from . import config, masks, multichar, presets, traincache
from .events import bus
from .gpu import gpu, query_gpus, shared_gpu_memory_gb, system_ram_gb
from .models import FAMILY_ENGINE, VARIANTS, hasher, library, registry

STATUS_LABEL = {"queued": "待機", "preparing": "準備", "training": "学習", "evaluating": "評価", "done": "完了",
                "cancelled": "中止", "failed": "失敗"}
ACTIVE = {"queued", "preparing", "training", "evaluating"}
QUEUE_FILE = config.DATA_DIR / "queue.json"   # waiting runs, restored at startup (server.resume_pending_on_startup)


def engine_cwd(engine: str) -> Path:
    return {"sd-scripts": config.SD_SCRIPTS_DIR, "musubi-tuner": config.MUSUBI_DIR}.get(engine) or config.ENGINES_DIR / engine


def engine_env(engine: str) -> dict:
    """Process environment for an engine.  The experimental musubi-tuner checkout (Qwen-Image-2.1, PR #1145)
    shares the train env with the pinned musubi-tuner, which is installed there as an editable package; its
    own src/ goes first on sys.path so `musubi_tuner` resolves to the PR's code."""
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", HF_HUB_DISABLE_TELEMETRY="1",
               PYTHONUNBUFFERED="1", TQDM_MININTERVAL="1")
    env.pop("PYTHONHOME", None)
    # The editable installs in the train env record absolute paths, so they break when the install folder is
    # moved or copied; putting the checkout on PYTHONPATH keeps `library` / `musubi_tuner` importable even
    # for scripts in sub-folders (networks/*.py), whose own folder is all Python adds by itself.
    if engine in ("musubi-tuner", "musubi-tuner-qwen21"):
        env["PYTHONPATH"] = str(engine_cwd(engine) / "src")
    elif engine in ("sd-scripts", "sd-scripts-moss"):
        # the newer sd-scripts (MOSS-Image) shares the train env with the pinned one, installed there as an
        # editable package: its own folder goes first so `library` / `networks` resolve to its code
        env["PYTHONPATH"] = str(engine_cwd(engine))
    return env


def engine_commit(engine: str) -> str:
    d = engine_cwd(engine)
    head = d / ".git" / "HEAD"
    try:
        ref = head.read_text().strip()
        if ref.startswith("ref:"):
            p = d / ".git" / ref.split(" ", 1)[1]
            if p.exists():
                return p.read_text().strip()[:12]
            packed = (d / ".git" / "packed-refs").read_text()
            for line in packed.splitlines():
                if line.endswith(ref.split(" ", 1)[1]):
                    return line.split()[0][:12]
        return ref[:12]
    except OSError:
        return "unknown"


# ======================================================================== preflight
def _family_file_errors(model: dict) -> list[str]:
    """Files the training engine needs beyond what generation needs."""
    f, fam = model["files"], model["family"]
    errs = []
    if fam == "qwen_image":
        # musubi-tuner docs/qwen_image.md: the bf16 DiT and the bf16 text encoder; fp8 / GGUF files cannot be used
        from .models import detect
        ck = f.get("ckpt") or ""
        try:
            bad = not ck or ck.lower().endswith(".gguf") or (Path(ck).exists() and bool(detect(Path(ck)).get("fp8")))
        except (OSError, ValueError):
            bad = False
        te = f.get("train_te") or f.get("te") or ""
        if bad or "fp8" in Path(te).name.lower():
            errs.append("Qwen-Image の学習には bf16 版の本体とテキストエンコーダー（qwen_2.5_vl_7b.safetensors）が必要です"
                        "（fp8 版・GGUF 版は生成にだけ使えます）。設定画面の「おすすめのモデル」から「Qwen-Image-2512 学習用ファイル」を追加してください。")
    if fam == "qwen_image21":
        if not f.get("ckpt") or not qwen21_train_te(f) or not f.get("train_vae"):
            errs.append("Qwen-Image-2.1 の学習には学習用ファイル（本体・学習用テキストエンコーダー・学習用 VAE）が必要です。"
                        "設定画面の「おすすめのモデル」から「Qwen-Image-2.1 学習用ファイル」を追加してください。")
        from .addons import engine_ready
        if not engine_ready("musubi-tuner-qwen21"):
            errs.append("Qwen-Image-2.1 の学習エンジン（musubi-tuner の開発版）が入っていません。「Qwen-Image-2.1 学習用ファイル」を追加すると導入されます。")
    if fam == "zimage":
        from .models import detect
        try:
            fp8 = bool(f.get("ckpt")) and Path(f["ckpt"]).exists() and detect(Path(f["ckpt"])).get("fp8")
        except (OSError, ValueError):
            fp8 = False
        if fp8:
            errs.append("Z-Image の学習には bf16 版の本体が必要です（このファイルは fp8 版で、生成にだけ使えます）。")
        if "fp8" in Path(f.get("te", "")).name.lower():
            errs.append("Z-Image の学習には bf16 のテキストエンコーダー（qwen_3_4b.safetensors）が必要です。")
        if model.get("variant") == "zimage_turbo" and not f.get("adapter"):
            errs.append("Z-Image Turbo 系の学習には、Turbo 用の学習アダプターが必要です。設定画面の「おすすめのモデル」から"
                        "「Z-Image Turbo 学習アダプター」を追加してください（約 0.3GB）。")
    if fam == "krea2":
        # musubi-tuner docs/krea2.md: trained on the bf16 RAW DiT; Turbo and quantized files are for generation
        from .models import detect
        ck = f.get("ckpt") or ""
        try:
            bad = not ck or "turbo" in Path(ck).name.lower() or (Path(ck).exists() and bool(detect(Path(ck)).get("quantized")))
        except (OSError, ValueError):
            bad = False
        if bad:
            errs.append("Krea 2 の学習には RAW 版の本体（bf16）が必要です（Turbo 版や量子化した版は生成にだけ使えます）。"
                        "設定画面の「おすすめのモデル」から「Krea 2 RAW」を追加するか、モデルの「Krea 2 RAW DiT」に RAW 版のファイルを指定してください。")
        if "fp8" in Path(f.get("te", "")).name.lower():
            errs.append("Krea 2 の学習には bf16 のテキストエンコーダー（qwen3vl_4b_bf16.safetensors）が必要です。")
    if fam == "moss_image":
        from .addons import engine_ready
        if not engine_ready("sd-scripts-moss"):
            errs.append("MOSS-Image の学習エンジン（sd-scripts v0.12.1）が入っていません。設定画面の「おすすめのモデル」から"
                        "MOSS-Image を追加すると導入されます。")
        if "w4a8" in Path(f.get("te", "")).name.lower():
            errs.append("MOSS-Image の学習には、W4A8 版のテキストエンコーダーは使えません（生成には使えます）。"
                        "qwen3.5_9b_int8_convrot.safetensors か qwen3.5_9b_bf16.safetensors を text_encoders フォルダーに置いてください。")
    return errs


def qwen21_train_te(files: dict) -> str | None:
    """The text encoder Qwen-Image-2.1 training reads: bf16, or ComfyUI's ConvRot INT8 (made bf16 again by
    bridges/qwen21_int8.py).  The W4A8 one is for generation only."""
    if files.get("train_te"):
        return files["train_te"]
    te = files.get("te") or ""
    name = Path(te).name.lower()
    return te if te and "w4a8" not in name and ("bf16" in name or is_int8_file(te)) else None


def is_int8_file(path: str | None) -> bool:
    """A ConvRot INT8 file (ComfyUI *_int8_convrot): sd-scripts loads it as INT8 by itself, and quantizing it again
    is not wanted."""
    if not path:
        return False
    if "int8" in Path(path).name.lower():
        return True
    from .models import detect
    try:
        return bool(Path(path).exists() and detect(Path(path)).get("int8"))
    except (OSError, ValueError):
        return False


def missing_file_messages(model: dict, keys: list[str] | None = None) -> list[str]:
    """One message per missing file of a registered model (or of `keys` of them), saying where to move it when it
    is found in an old folder of this tool (e.g. models/anima of the 2026-10-08 version)."""
    from .models import FAMILY_FILES
    labels = {k: lab for k, lab, _ in FAMILY_FILES.get(model["family"], [])}
    old = {x["name"].lower(): x for x in library.misplaced()}
    out = []
    for k in model["missing"] if keys is None else keys:
        name = Path(model["files"][k]).name
        what = labels.get(k, "モデルファイル")
        x = old.get(name.lower())
        if x:
            out.append(f"「{model['name']}」の {what}（{name}）を、{x['from']} から {x['to']} に移してください。")
        else:
            out.append(f"「{model['name']}」の {what}（{name}）が見つかりません。登録していた場所: {model['files'][k]}")
    return out


_plan_gpu_cache = {"t": 0.0, "v": []}


def _plan_gpus(ttl: float = 10.0) -> list[dict]:
    """nvidia-smi (0.1-0.5 s) at most every ttl seconds: the plan is asked for on every settings change."""
    if time.time() - _plan_gpu_cache["t"] > ttl:
        _plan_gpu_cache.update(t=time.time(), v=query_gpus())
    return _plan_gpu_cache["v"]


_write_checks: dict[str, tuple[float, str | None]] = {}   # folder -> (checked at, error or None)
WRITE_CHECK_TTL = 60


def _write_error(folder: Path) -> str | None:
    """Can the run folder be written?  Remembered for a minute: the plan is asked for on every settings change."""
    key = os.path.normcase(str(folder))
    hit = _write_checks.get(key)
    if hit and time.time() - hit[0] < WRITE_CHECK_TTL:
        return hit[1]
    err = None
    try:
        probe = Path(folder) / ".write_test"
        probe.write_text("x")
        probe.unlink()
    except OSError as e:
        err = str(e)
    _write_checks[key] = (time.time(), err)
    return err


_detect_cache: dict[str, dict] = {}


def model_train_errors(model: dict) -> list[str]:
    """What stops training with this registered model, whatever the project: its kind, missing files, files the
    training engine needs (bf16, not fp8 / GGUF ...).  preflight blocks on these; the model lists show them."""
    from .models import GEN_ONLY, UNAVAILABLE_NOTE, VARIANTS, detect, trainable
    errors = []
    if not model.get("variant") or model["variant"] not in VARIANTS:
        return ["モデルの系統が未確認です。設定画面で選んでください。"]
    fam = model["family"]
    need = [k for k in model["missing"] if k not in GEN_ONLY.get(fam, ())]
    if need:
        errors += missing_file_messages(model, need)
    if not trainable(model["variant"]):
        errors.append(UNAVAILABLE_NOTE.get(model["variant"], "このモデル系列は現在対応していません。"))
    errors += _family_file_errors(model)
    if not need and model["files"].get("ckpt"):
        ck = Path(model["files"]["ckpt"])
        try:   # header only: cheap even for multi-GB files; kept by size and time for the model lists
            st = ck.stat()
            key = f"{ck}|{st.st_size}|{int(st.st_mtime)}"
            if key not in _detect_cache:
                _detect_cache[key] = detect(ck)
            det = _detect_cache[key]
            if det.get("family") != fam:
                errors.append(f"登録されたモデルの中身が「{fam}」系ではありません（{det.get('evidence') or '判定不能'}）。設定画面でファイルを確認してください。")
            elif det.get("fp8"):
                errors.append("学習には bf16 版のモデル本体が必要です（登録されているのは fp8 版で、生成にだけ使えます）。")
        except (OSError, ValueError) as e:
            errors.append(f"モデルファイルを読めません: {e}")
    return errors


def preflight(project, used_images: list[dict], values: dict) -> dict:
    """Checks before training.  'errors' block the start; 'warnings' are shown but don't."""
    meta = project.get_meta()
    errors, warnings, info = [], [], {}
    model = registry.get(meta.get("model_id") or "")
    if not model:
        errors.append("ベースモデルが未設定です。設定画面でモデルを登録してください。")
        return {"errors": errors, "warnings": warnings, "info": info}
    if not model.get("variant"):
        kinds = {"zimage": "Turbo 系 / Base 系", "moss_image": "SFT 版 / Base 版"}.get(model.get("family"), "Illustrious / Pony など")
        errors.append(f"「{model['name']}」の系統（{kinds}）が未確認です。設定画面で選んでください。")
        return {"errors": errors, "warnings": warnings, "info": info}
    fam = model["family"]
    if model["missing"]:   # moved since the last scan of the model folders?
        library._repair_paths()
        model = registry.get(model["id"])
    from .models import GEN_ONLY
    gen_only = GEN_ONLY.get(fam, ())
    need = [k for k in model["missing"] if k not in gen_only]
    errors += model_train_errors(model)
    if len(need) < len(model["missing"]):
        warnings += ["学習には使わない生成用のファイルが見つかりません（学習はできます。学習後の比較画像は、学習用の本体で生成します）。" + m
                     for m in missing_file_messages(model, [k for k in model["missing"] if k in gen_only])]
    from .models import experimental_note
    if experimental_note(model["variant"]):
        warnings.append(experimental_note(model["variant"]))
    if not VARIANTS[model["variant"]]["verified"]:
        warnings.append(f"「{VARIANTS[model['variant']]['label']}」はこのツールでの実学習検証が未完了です。")
    enh = None
    if meta.get("kind") == "enhance":
        from . import enhance
        enh = enhance.clean(meta.get("enhance"))
        if fam not in enhance.FAMILIES:
            errors.append("質感・補強 LoRA は、今は SDXL 系のモデル（Illustrious・Pony・NoobAI・SDXL 1.0 など）だけで作れます。")
        warnings.append("質感・補強 LoRA は試験的な機能です。初期設定は資料にもとづく目安で、このツールでの実学習検証が済んでいません。")
    if enh and enh["method"] == "words":
        if not (enh["positive"] or enh["negative"]):
            errors.append("強めたいもの・弱めたいもののどちらかを入れてください。")
    elif len(used_images) < 3:
        errors.append(f"採用画像が {len(used_images)} 枚です。3 枚以上必要です。")
    if enh and enh["method"] == "words":
        used_images = []   # made from prompts only; material images are not used
    # _src: images of a multi-character project live in the member projects (multichar.gather)
    missing = [im for im in used_images if not (Path(im["_src"]) if im.get("_src") else project.source_path(im)).exists()]
    if missing:
        errors.append(f"元画像が見つからない素材が {len(missing)} 枚あります。素材画面で再指定してください。")
    waiting = [im for im in used_images if im.get("caption_status") in ("pending", "running") and not im.get("caption_edited")]
    empty = [im for im in used_images if not (im.get("caption") or "").strip()]
    cap_failed = [im for im in used_images if im.get("caption_status") == "failed" and not im.get("caption_edited")]
    if cap_failed:
        warnings.append(f"説明文の自動生成に失敗した画像が {len(cap_failed)} 枚あります。その画像の説明文は古いまま（呼び出し語を変える前の"
                        "ものなど）の可能性があります。素材画面でその画像を開き、「説明文を作り直す」か、説明文を確認してください。")
    # Captions are not required: an image without one is trained with the trigger alone (see fallback_caption).
    if waiting:
        warnings.append(f"説明文の自動生成がまだ終わっていない画像が {len(waiting)} 枚あります。このまま開始すると、その画像は"
                        "説明文が呼び出し語だけの状態で学習します。説明文がないと、写っている服・ポーズ・背景なども呼び出し語の一部として"
                        "覚えやすくなり、LoRA を使ったときに素材と同じ服や構図に寄りやすくなります。素材画面で説明文の生成が終わるのを"
                        "待ってから開始することをおすすめします。学習や画像生成など GPU を使う処理が動いているあいだは、説明文の生成は止まって順番を待ちます。"
                        "待ち時間を短くしたい場合は、素材画面の「説明文の作り方」で、すぐに作れる方法に変えられます。")
    elif empty:
        warnings.append(f"説明文が空の画像が {len(empty)} 枚あります。このまま開始すると、その画像は説明文が呼び出し語だけの状態で学習します。"
                        "説明文がないと、写っている服・ポーズ・背景なども呼び出し語の一部として覚えやすくなり、LoRA を使ったときに"
                        "素材と同じ服や構図に寄りやすくなります。")
    if not config.TRAIN_PYTHON.exists():
        errors.append("学習環境が未導入です。初回セットアップを実行してください。")
    gpus = _plan_gpus()
    info["gpu"] = gpus[0] if gpus else None
    if not gpus:
        errors.append("NVIDIA GPU が見つかりません。素材の編集はできますが、学習は GPU が必要です（CPU 学習へは切り替えません）。")
    else:
        need = {"sdxl": 10000, "anima": 10000, "krea2": 12000, "qwen_image": 12000, "qwen_image21": 12000, "zimage": 12000,
                "moss_image": 10000}[fam]
        if fam == "moss_image" and gpus[0]["vram_total_mb"] < 10000:
            # sd-scripts docs: training fits 5.2GB, but caching the text encoder outputs needs about 8.3GB
            errors.append(f"VRAM {gpus[0]['vram_total_mb'] // 1024}GB では MOSS-Image の学習はできません（学習エンジンの資料でも"
                          "VRAM 8GB は未対応です。説明文の前処理だけで約 8.3GB 使います）。")
        elif gpus[0]["vram_total_mb"] < need:
            warnings.append(f"VRAM {gpus[0]['vram_total_mb'] // 1024}GB は推奨 {need // 1024}GB 未満です。メモリ不足で失敗する可能性があります。")
    # block swap keeps part of the DiT in system memory (musubi-tuner docs/qwen_image.md recommends 64GB)
    ram_need = {"qwen_image": 64, "qwen_image21": 32, "zimage": 32}.get(fam)
    ram = system_ram_gb() if ram_need else None
    info["ram_gb"] = ram
    if ram and ram < ram_need - 2 and int(values.get("blocks_to_swap", 0)) > 0:
        warnings.append(f"メインメモリ {ram:.0f}GB は、このモデルの学習の推奨 {ram_need}GB 未満です。本体の一部をメインメモリに置いて学習するため、"
                        "足りないと非常に遅くなるか、途中で失敗することがあります。")
    short = last_vram_short(project, model["id"])
    if short and not _vram_lighter(values, short):
        warnings.append(f"前回の学習（{short['name']}）では VRAM が足りず、大幅に遅くなりました。同じ設定では今回も遅くなる見込みです。"
                        "次のどれかを試してください: " + vram_advice(fam, values) + "。")
    need_gb = presets.estimate_disk_gb(fam, values)
    free_gb = shutil.disk_usage(project.runs_dir).free / 1e9
    info["disk_need_gb"], info["disk_free_gb"] = need_gb, round(free_gb, 1)
    if free_gb < need_gb + 2:
        errors.append(f"空き容量が不足しています（必要 約{need_gb}GB / 空き {free_gb:.1f}GB）。")
    werr = _write_error(project.runs_dir)
    if werr:
        errors.append(f"書き込みできません: {project.runs_dir} ({werr})")
    errors += presets.validate(fam, values)
    return {"errors": errors, "warnings": warnings, "info": info}


# ======================================================================== snapshot
PREP_VERSION = 1   # needs_prep / _export_image output; part of the latent cache key (traincache._keys)
RAW_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}   # files sd-scripts and musubi-tuner read (others are skipped)


def needs_prep(src: Path) -> bool:
    """True when the engines would not see the file as the app shows it: they apply no EXIF rotation (phone
    photos would be learnt sideways), drop the alpha channel (transparent parts become the colour under them,
    usually black, while the analysis composites them on white), and skip TIFF / GIF."""
    if src.suffix.lower() not in RAW_EXT:
        return True
    try:
        with Image.open(src) as im:
            if im.mode in ("RGBA", "LA", "PA", "P") or "transparency" in im.info:
                return im.mode != "P" or "transparency" in im.info
            return im.getexif().get(0x0112, 1) != 1
    except Exception:
        return False   # unreadable: left to materialize, which reports it


def _export_image(src: Path, dest: Path, crop, prep: bool = False, fit: int | None = None) -> None:
    """prep: upright and on white (needs_prep).  fit: scaled down to about fit x fit pixels (pair runs)."""
    if not crop and not prep and not fit:
        try:
            os.link(src, dest)  # same volume: instant, no extra space
            return
        except OSError:
            shutil.copyfile(src, dest)
            return
    with Image.open(src) as im:
        im.load()
        im = ImageOps.exif_transpose(im)
        if crop:
            im = im.crop(tuple(int(v) for v in crop))
        im = masks.on_white(im)
        if fit and im.width * im.height > fit * fit:
            k = (fit * fit / (im.width * im.height)) ** 0.5
            im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
        im.save(dest, "PNG", compress_level=1)   # lossless either way; faster to write


def _enhance_snap(meta):
    if meta.get("kind") != "enhance":
        return None
    from .enhance import clean
    return clean(meta.get("enhance"))


def create_run(project, used_images: list[dict], values: dict, changed: list[str], name: str | None = None,
               members: list[dict] | None = None) -> int:
    meta = project.get_meta()
    model = registry.get(meta["model_id"])
    fam = model["family"]
    engine = FAMILY_ENGINE[fam]
    now = time.time()
    with project.tx() as db:
        cur = db.execute("INSERT INTO runs(created, status, name, config, run_dir) VALUES(?,?,?,?,?)",
                         (now, "queued", name or "", "{}", ""))
        run_id = cur.lastrowid
    try:
        return _write_run(project, run_id, meta, model, fam, engine, now, used_images, values, changed, name, members)
    except Exception as e:
        # never enqueued: without this the row would stay 待機 with an empty snapshot until the next restart
        with project.tx() as db:
            db.execute("UPDATE runs SET status='failed', ended=?, error=? WHERE id=?",
                       (time.time(), json.dumps({"code": "prepare", "message": f"学習の準備に失敗しました: {e}",
                                                 "fixes": ["空き容量と書き込み権限を確認する"]}, ensure_ascii=False), run_id))
        raise


def _write_run(project, run_id, meta, model, fam, engine, now, used_images, values, changed, name, members=None) -> int:
    run_dir = project.runs_dir / f"{run_id:04d}_{time.strftime('%Y%m%d-%H%M%S')}"
    train_files = {k: p for k, p in model["files"].items() if k not in ("fast", "fast4")}   # the fast LoRAs serve generation only
    ds_dir = run_dir / "dataset"
    ds_dir.mkdir(parents=True)
    (run_dir / "output").mkdir()
    # The snapshot (which images, captions, crops) is fixed *now*, in memory and in run.json.
    # Image files are written later by materialize() in the job's preparing phase: sources/ are
    # immutable copies, so deferring the file export cannot change what gets trained, and the
    # "start" click returns immediately.
    items = []
    fallback = f"{meta['trigger']} style" if meta.get("kind") == "style" else meta["trigger"]   # same head as captions.tag_caption
    if meta.get("kind") == "enhance" and (meta.get("enhance") or {}).get("method") == "words":
        used_images = []
    mask_mode = masks.active(meta, fam)
    mask_set = masks.setting(meta)
    enh = _enhance_snap(meta)
    # pair runs: the partner effects (soften, grain) work on pixels, so both sides are made at about the training
    # size, as in the preview; at full size of a large photo they almost vanish once the engine scales it down
    fit = int(values.get("resolution") or 1024) if enh and enh.get("method") == "pair" else None
    mask_dir = run_dir / "masks"
    for im in used_images:
        src = Path(im["_src"]) if im.get("_src") else project.source_path(im)
        mf = masks.usable(im, mask_mode) if mask_mode else None
        prep = needs_prep(src)
        ext = ".png" if im.get("crop") or mf or prep or fit else src.suffix.lower()
        stem = f"{im['id']:05d}"
        mask = None
        if mf:
            # frozen with the run (「マスクを作り直す」 rewrites the project's file); its content is part of the cache key
            mask_dir.mkdir(exist_ok=True)
            frozen = mask_dir / f"{stem}.png"
            shutil.copyfile(mf, frozen)
            digest = hashlib.sha1(frozen.read_bytes()).hexdigest()[:12]
            mask = {"file": str(frozen), "key": f"{masks.MASK_VERSION}:{mask_mode}:{mask_set['floor']}:{digest}"}
        items.append({"id": im["id"], "sha": im["sha"], "crop": im.get("crop"), "caption": (im.get("caption") or "").strip() or fallback,
                      "file": stem + ext, "src": str(src),
                      **({"prep": PREP_VERSION} if prep else {}), **({"fit": fit} if fit else {}),
                      **({"mask": mask} if mask else {})})
    out_name = re.sub(r"[^A-Za-z0-9_-]", "_", meta["trigger"]) + f"_r{run_id}"
    snap = {
        "project": project.id, "kind": meta["kind"], "outfit_mode": meta.get("outfit_mode"), "trigger": meta["trigger"],
        "concept_words": meta.get("concept_words") or [],
        "concept_target": meta.get("concept_target") or "body",
        "members": multichar.snap_members(members, items) if members else None,
        "enhance": enh,
        "mask": {"mode": mask_mode, "floor": mask_set["floor"], "n_masked": sum(1 for it in items if it.get("mask"))}
                if mask_mode else None,
        "model": {"id": model["id"], "name": model["name"], "variant": model["variant"], "family": fam,
                  "files": train_files, "sha256": {k: hasher.get(p) for k, p in train_files.items()}},
        "engine": engine, "engine_commit": engine_commit(engine), "preset_version": presets.PRESET_VERSION,
        "values": values, "changed": changed, "items": items, "output_name": out_name, "created": now,
    }
    config.atomic_write_json(run_dir / "run.json", snap)
    with project.tx() as db:
        db.execute("UPDATE runs SET config=?, run_dir=?, name=? WHERE id=?",
                   (json.dumps(snap, ensure_ascii=False), str(run_dir), name or f"#{run_id}", run_id))
    return run_id


def materialize(run_dir: Path, snap: dict) -> None:
    """Write the frozen dataset files (idempotent; a partly written dataset is completed)."""
    ds_dir = run_dir / "dataset"
    ds_dir.mkdir(exist_ok=True)
    for it in snap["items"]:
        img = ds_dir / it["file"]
        if not img.exists():
            tmp = ds_dir / ("_tmp_" + it["file"])
            if it.get("mask"):   # masked training: the mask goes into the alpha channel (sd-scripts alpha_mask)
                with Image.open(it["src"]) as im:
                    im.load()
                    masks.apply(ImageOps.exif_transpose(im), Path(it["mask"]["file"]), it.get("crop"),
                                snap["mask"]["floor"]).save(tmp, "PNG", compress_level=1)
            else:
                _export_image(Path(it["src"]), tmp, it.get("crop"), bool(it.get("prep")), it.get("fit"))
            os.replace(tmp, img)
        txt = img.with_suffix(".txt")
        if not txt.exists():
            txt.write_text(it["caption"], encoding="utf-8")
    enh = snap.get("enhance")
    if enh and enh["method"] == "pair":
        from .enhance import effect, make_partner
        op = effect(enh["effect"])["op"]
        pd = run_dir / "dataset_partner"
        pd.mkdir(exist_ok=True)
        for it in snap["items"]:
            dest = pd / (Path(it["file"]).stem + ".png")
            if not dest.exists():
                make_partner(ds_dir / it["file"], dest, op)
            txt = dest.with_suffix(".txt")
            if not txt.exists():
                txt.write_text(it["caption"], encoding="utf-8")


def pair_dirs(run_dir: Path, snap: dict) -> tuple[Path, Path]:
    """(copy side, target side) of a pair run: the LoRA learns the change from the copy side to the target side."""
    from .enhance import effect
    orig, partner = run_dir / "dataset", run_dir / "dataset_partner"
    return (partner, orig) if effect(snap["enhance"]["effect"])["direction"] == "remove" else (orig, partner)


# ======================================================================== engine adapters
def _ds_sdscripts(run_dir: Path, v: dict, image_dir: Path | None = None, name: str = "dataset.toml",
                  alpha_mask: bool = False) -> Path:
    sub = {"image_dir": str(image_dir or run_dir / "dataset"), "num_repeats": 1,
           "caption_dropout_rate": float(v.get("caption_dropout_rate", 0))}
    if alpha_mask:   # the loss is multiplied by the images' alpha channel (masked training, masks.py)
        sub["alpha_mask"] = True
    ds = {"general": {"caption_extension": ".txt", "shuffle_caption": False, "keep_tokens": 1},
          "datasets": [{"resolution": v["resolution"], "batch_size": v["train_batch_size"], "enable_bucket": True,
                        "min_bucket_reso": 512, "max_bucket_reso": 2048, "bucket_reso_steps": 64, "bucket_no_upscale": False,
                        "subsets": [sub]}]}
    p = run_dir / name
    p.write_text(toml.dumps(ds), encoding="utf-8")
    return p


def _ds_musubi(run_dir: Path, v: dict) -> Path:
    r = v["resolution"]
    ds = {"general": {"resolution": [r, r], "caption_extension": ".txt", "batch_size": v["train_batch_size"],
                      "enable_bucket": True, "bucket_no_upscale": False},
          "datasets": [{"image_directory": str(run_dir / "dataset"), "cache_directory": str(run_dir / "cache"),
                        "num_repeats": 1}]}
    p = run_dir / "dataset.toml"
    p.write_text(toml.dumps(ds), encoding="utf-8")
    return p


STATE_ENGINES = ("sd-scripts", "sd-scripts-moss")   # the engines whose saved states this tool can resume from


def saves_state(snap: dict) -> bool:
    """Whether a run writes resume states.  musubi-tuner runs cannot be resumed (resume_block_reason), so their
    states would only cost disk and time at every save; texture LoRA runs have no resume either."""
    return not snap.get("enhance") and snap.get("engine") in STATE_ENGINES


def _common_save(v, out_dir, out_name, state: bool = True):
    cfg = {"output_dir": str(out_dir), "output_name": out_name, "save_model_as": "safetensors",
           "max_train_steps": int(v["max_train_steps"]), "save_every_n_steps": int(v["save_every_n_steps"]),
           "save_state": state,
           "seed": int(v["seed"]), "learning_rate": float(v["learning_rate"]),
           "network_dim": int(v["network_dim"]), "network_alpha": float(v["network_alpha"]),
           "optimizer_type": v["optimizer_type"], "lr_scheduler": v["lr_scheduler"],
           "gradient_checkpointing": True, "mixed_precision": "bf16",
           # every run reads only the cached latents / text encoder outputs (a few small files per step, measured
           # 1.8 ms per SDXL image), so loader processes buy no speed; each spawned one costs ~700MB RAM and ~6 s
           "max_data_loader_n_workers": 0}
    if state:
        cfg["save_last_n_steps_state"] = int(v["save_every_n_steps"]) * 2
    return cfg


SD_RESUME = Path(__file__).resolve().parent / "bridges" / "sd_resume.py"
QWEN21_INT8 = Path(__file__).resolve().parent / "bridges" / "qwen21_int8.py"   # reads ComfyUI INT8 files as bf16
ANIMA_BLOCKS = Path(__file__).resolve().parent / "bridges" / "anima_blocks.py"   # Anima DiT at the file's block count


def _sd_script(script: str, resume: str | None) -> list[str]:
    """A resumed sd-scripts run goes through bridges/sd_resume.py, which fixes the engine's step counting.  Anima runs
    go through bridges/anima_blocks.py, which builds the DiT with the checkpoint's block count (Anima 2.9B: 40; the
    engine fixes it at 28) and applies the resume fix too."""
    if script.startswith("anima_"):
        return [str(ANIMA_BLOCKS)] + (["--resume-fix"] if resume else []) + [script]
    return [str(SD_RESUME), script] if resume else [script]


def hf_tokenizer_cached(repo: str, revision: str = "main") -> bool:
    """The tokenizer/ folder of a Hugging Face repo is already in the local Hub cache (the layout huggingface_hub
    writes: models--<org>--<name>/refs/<revision> -> snapshots/<commit>/).  Then the prefetch process (Python start-up
    and a request to the Hub) is skipped; when unsure, False, and the prefetch runs as before."""
    hub = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if not hub:
        home = os.environ.get("HF_HOME") or os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.join(Path.home(), ".cache"),
                                                         "huggingface")
        hub = os.path.join(home, "hub")
    d = Path(hub) / ("models--" + repo.replace("/", "--"))
    try:
        commit = (d / "refs" / revision).read_text(encoding="utf-8").strip()
    except OSError:
        return False
    tok = d / "snapshots" / commit / "tokenizer"
    return bool(commit) and (tok / "tokenizer_config.json").is_file() and (tok / "tokenizer.json").is_file()


def build_commands(snap: dict, run_dir: Path, resume: str | None) -> list[tuple[str, list[str], Path]]:
    """Returns [(phase, argv, cwd)].  argv is a list (no shell)."""
    fam = snap["model"]["family"]
    v = snap["values"]
    files = snap["model"]["files"]
    out_dir = run_dir / "output"
    py = str(config.TRAIN_PYTHON)
    launch = [py, "-m", "accelerate.commands.launch", "--num_processes", "1", "--num_machines", "1",
              "--mixed_precision", "bf16", "--dynamo_backend", "no", "--num_cpu_threads_per_process", "2"]
    cmds = []
    if snap.get("enhance"):
        return _enhance_commands(snap, run_dir, launch)
    if fam in ("sdxl", "anima"):
        ds = _ds_sdscripts(run_dir, v, alpha_mask=bool(snap.get("mask")))
        cfg = _common_save(v, out_dir, snap["output_name"])
        cfg.update(dataset_config=str(ds), pretrained_model_name_or_path=files["ckpt"], network_train_unet_only=True,
                   cache_latents=True, cache_latents_to_disk=True, cache_text_encoder_outputs=True,
                   cache_text_encoder_outputs_to_disk=True, sdpa=True, save_precision="bf16")
        if fam == "sdxl":
            cfg.update(network_module="networks.lora", no_half_vae=True, max_token_length=225, save_precision="fp16")
            cfg.update(_sdxl_te_cache(v))
            if VARIANTS[snap["model"]["variant"]]["prediction"] == "v":
                cfg.update(v_parameterization=True, zero_terminal_snr=True)
            else:
                cfg.update(min_snr_gamma=5)
            if v["lr_scheduler"] == "cosine_with_restarts":
                cfg["lr_scheduler_num_cycles"] = 1
            script = "sdxl_train_network.py"
        else:
            cfg.update(network_module="networks.lora_anima", qwen3=files["te"], vae=files["vae"],
                       timestep_sampling="sigmoid", discrete_flow_shift=1.0, qwen_image_vae_2d=True)
            if int(v.get("blocks_to_swap", 0)) > 0:
                cfg["blocks_to_swap"] = int(v["blocks_to_swap"])
            script = "anima_train_network.py"
        if resume:
            # --resume alone restores optimizer/scheduler but restarts the step count at 0 and trains the full
            # max_train_steps again.  skip_until_initial_step continues from the step stored in train_state.json;
            # its step counting after the first epoch is fixed by bridges/sd_resume.py.
            cfg["resume"] = resume
            cfg["skip_until_initial_step"] = True
        cfg_path = run_dir / ("train_resume.toml" if resume else "train.toml")
        cfg_path.write_text(toml.dumps(cfg), encoding="utf-8")
        cmds.append(("training", launch + _sd_script(script, resume) + ["--config_file", str(cfg_path)], config.SD_SCRIPTS_DIR))
    elif fam == "krea2":
        ds = _ds_musubi(run_dir, v)
        if not resume:
            cmds.append(("preparing", [py, "krea2_cache_latents.py", "--dataset_config", str(ds), "--vae", files["vae"]],
                         config.MUSUBI_DIR))
            cmds.append(("preparing", [py, "krea2_cache_text_encoder_outputs.py", "--dataset_config", str(ds),
                                       "--text_encoder", files["te"], "--batch_size", "1"], config.MUSUBI_DIR))
        cfg = _common_save(v, out_dir, snap["output_name"], state=False)   # no resume (resume_block_reason)
        cfg.update(dit=files["ckpt"], vae=files["vae"], dataset_config=str(ds), sdpa=True,
                   timestep_sampling="krea2_shift", weighting_scheme="none", network_module="networks.lora_krea2",
                   fp8_base=True, fp8_scaled=True)
        if int(v.get("blocks_to_swap", 0)) > 0:
            cfg["blocks_to_swap"] = int(v["blocks_to_swap"])
        if resume:
            cfg["resume"] = resume
        cfg_path = run_dir / ("train_resume.toml" if resume else "train.toml")
        cfg_path.write_text(toml.dumps(cfg), encoding="utf-8")
        cmds.append(("training", launch + ["krea2_train_network.py", "--config_file", str(cfg_path)], config.MUSUBI_DIR))
    elif fam == "qwen_image":
        # musubi-tuner docs/qwen_image.md.  2512 has the original architecture (same tensors and size), so it is
        # trained as model_version original; the docs do not name 2512 (unverified until docs/VERIFICATION.md says so).
        ds = _ds_musubi(run_dir, v)
        te = files.get("train_te") or files["te"]   # the bf16 encoder (a light model's own "te" may be fp8)
        if not resume:
            cmds.append(("preparing", [py, "qwen_image_cache_latents.py", "--dataset_config", str(ds), "--vae", files["vae"],
                                       "--model_version", "original"], config.MUSUBI_DIR))
            cmds.append(("preparing", [py, "qwen_image_cache_text_encoder_outputs.py", "--dataset_config", str(ds),
                                       "--text_encoder", te, "--batch_size", "1", "--model_version", "original",
                                       "--fp8_vl"], config.MUSUBI_DIR))
        cfg = _common_save(v, out_dir, snap["output_name"], state=False)   # no resume (resume_block_reason)
        cfg.update(dit=files["ckpt"], vae=files["vae"], text_encoder=te, model_version="original",
                   dataset_config=str(ds), sdpa=True, timestep_sampling="shift", discrete_flow_shift=2.2,
                   weighting_scheme="none", network_module="networks.lora_qwen_image", fp8_base=True, fp8_scaled=True)
        if int(v.get("blocks_to_swap", 0)) > 0:
            cfg["blocks_to_swap"] = int(v["blocks_to_swap"])
        if resume:
            cfg["resume"] = resume
        cfg_path = run_dir / ("train_resume.toml" if resume else "train.toml")
        cfg_path.write_text(toml.dumps(cfg), encoding="utf-8")
        cmds.append(("training", launch + ["qwen_image_train_network.py", "--config_file", str(cfg_path)], config.MUSUBI_DIR))
    elif fam == "qwen_image21":
        # musubi-tuner dev branch (PR #1145 merged there 2026-10-12; docs/qwen_image_21.md).  The VAE must have Diffusers
        # keys (train_vae); the text encoder is bf16 or ComfyUI INT8, which bridges/qwen21_int8.py reads as bf16 (the DiT too).
        cwd = engine_cwd("musubi-tuner-qwen21")
        ds = _ds_musubi(run_dir, v)
        te = qwen21_train_te(files) or files["te"]
        if not resume:
            cmds.append(("preparing", [py, "qwen_image_21_cache_latents.py", "--dataset_config", str(ds),
                                       "--vae", files["train_vae"], "--vae_tiling"], cwd))
            cmds.append(("preparing", [py, str(QWEN21_INT8), "qwen_image_21_cache_text_encoder_outputs.py", "--dataset_config", str(ds),
                                       "--text_encoder", te, "--batch_size", "1", "--fp8_vl"], cwd))
        cfg = _common_save(v, out_dir, snap["output_name"], state=False)   # no resume (resume_block_reason)
        cfg.update(dit=files["ckpt"], dataset_config=str(ds), sdpa=True, timestep_sampling="shift", discrete_flow_shift=3.0,
                   weighting_scheme="none", network_module="networks.lora_qwen_image_21", fp8_base=True, fp8_scaled=True)
        if int(v.get("blocks_to_swap", 0)) > 0:
            cfg["blocks_to_swap"] = int(v["blocks_to_swap"])
        if resume:
            cfg["resume"] = resume
        cfg_path = run_dir / ("train_resume.toml" if resume else "train.toml")
        cfg_path.write_text(toml.dumps(cfg), encoding="utf-8")
        cmds.append(("training", launch + [str(QWEN21_INT8), "qwen_image_21_train_network.py", "--config_file", str(cfg_path)], cwd))
    elif fam == "moss_image":
        # sd-scripts v0.12.1 docs/moss_image_train_network.md (a separate checkout, sd-scripts-moss).  The timestep
        # options reproduce the pre-training distribution (doc section 7.2).  INT8 / block swap follow the VRAM tier
        # (presets.py); a ConvRot INT8 file is loaded as INT8 by the engine itself.  The tokenizer (~20MB) is fetched
        # from Hugging Face first (cached after the first run).
        cwd = engine_cwd("sd-scripts-moss")
        ds = _ds_sdscripts(run_dir, v)
        if not resume and not hf_tokenizer_cached("OpenMOSS-Team/MOSS-Image-Base"):
            code = ("from huggingface_hub import snapshot_download as d\n"
                    "d('OpenMOSS-Team/MOSS-Image-Base', allow_patterns=['tokenizer/*'])\n")
            cmds.append(("preparing", [py, "-c", code], cwd))
        cfg = _common_save(v, out_dir, snap["output_name"])
        cfg.update(dataset_config=str(ds), pretrained_model_name_or_path=files["ckpt"], text_encoder=files["te"],
                   vae=files["vae"], network_module="networks.lora_generic", network_train_unet_only=True,
                   cache_latents=True, cache_latents_to_disk=True, cache_text_encoder_outputs=True,
                   cache_text_encoder_outputs_to_disk=True, timestep_sampling="sigma", weighting_scheme="logit_normal",
                   logit_mean=-1.09, logit_std=0.95, discrete_flow_shift=1.0, save_precision="bf16")
        if int(v.get("convrot_int8", 0)) and not is_int8_file(files["ckpt"]):
            cfg["convrot_int8"] = True
        if int(v.get("convrot_int8_text_encoder", 0)) and not is_int8_file(files["te"]):
            cfg["convrot_int8_text_encoder"] = True
        if int(v.get("blocks_to_swap", 0)) > 0:
            cfg["blocks_to_swap"] = int(v["blocks_to_swap"])
        if resume:   # same engine semantics as sd-scripts (resume_block_reason)
            cfg["resume"] = resume
            cfg["skip_until_initial_step"] = True
        cfg_path = run_dir / ("train_resume.toml" if resume else "train.toml")
        cfg_path.write_text(toml.dumps(cfg), encoding="utf-8")
        cmds.append(("training", launch + _sd_script("moss_image_train_network.py", resume) + ["--config_file", str(cfg_path)], cwd))
    elif fam == "zimage":
        # musubi-tuner docs/zimage.md.  Turbo (distilled) is trained with ostris's training adapter merged into the
        # DiT (--base_weights); the LoRA is then used on Turbo without the adapter.  The tokenizer is loaded from
        # Tongyi-MAI/Z-Image on Hugging Face; fetched first (cached after the first run).
        ds = _ds_musubi(run_dir, v)
        if not resume and not hf_tokenizer_cached("Tongyi-MAI/Z-Image"):
            code = ("from huggingface_hub import snapshot_download as d\n"
                    "d('Tongyi-MAI/Z-Image', allow_patterns=['tokenizer/*'])\n")
            cmds.append(("preparing", [py, "-c", code], config.MUSUBI_DIR))
        if not resume:
            cmds.append(("preparing", [py, "zimage_cache_latents.py", "--dataset_config", str(ds), "--vae", files["vae"]],
                         config.MUSUBI_DIR))
            cmds.append(("preparing", [py, "zimage_cache_text_encoder_outputs.py", "--dataset_config", str(ds),
                                       "--text_encoder", files["te"], "--batch_size", "1"], config.MUSUBI_DIR))
        cfg = _common_save(v, out_dir, snap["output_name"], state=False)   # no resume (resume_block_reason)
        cfg.update(dit=files["ckpt"], vae=files["vae"], text_encoder=files["te"], dataset_config=str(ds), sdpa=True,
                   timestep_sampling="shift", discrete_flow_shift=2.0, weighting_scheme="none",
                   network_module="networks.lora_zimage", fp8_base=True, fp8_scaled=True)
        if snap["model"]["variant"] == "zimage_turbo" and files.get("adapter"):
            cfg.update(base_weights=[files["adapter"]], base_weights_multiplier=[1.0])
        if int(v.get("blocks_to_swap", 0)) > 0:
            cfg["blocks_to_swap"] = int(v["blocks_to_swap"])
        if resume:
            cfg["resume"] = resume
        cfg_path = run_dir / ("train_resume.toml" if resume else "train.toml")
        cfg_path.write_text(toml.dumps(cfg), encoding="utf-8")
        cmds.append(("training", launch + ["zimage_train_network.py", "--config_file", str(cfg_path)], config.MUSUBI_DIR))
    return cmds


def _sdxl_network_cfg(snap: dict, v: dict, ckpt: str, ds: Path, out_dir: Path, out_name: str) -> dict:
    cfg = _common_save(v, out_dir, out_name)
    cfg.update(dataset_config=str(ds), pretrained_model_name_or_path=ckpt, network_train_unet_only=True,
               cache_latents=True, cache_latents_to_disk=True, cache_text_encoder_outputs=True,
               cache_text_encoder_outputs_to_disk=True, sdpa=True, network_module="networks.lora", no_half_vae=True,
               max_token_length=225, save_precision="fp16")
    if VARIANTS[snap["model"]["variant"]]["prediction"] == "v":
        cfg.update(v_parameterization=True, zero_terminal_snr=True)
    else:
        cfg.update(min_snr_gamma=5)
    cfg.update(_sdxl_te_cache(v))
    return cfg


def _sdxl_te_cache(v: dict) -> dict:
    """sd-scripts' SDXL training refuses a caption dropout together with cached text encoder outputs (Anima and
    MOSS support it): with a dropout the text encoders run every step instead (slower, more VRAM)."""
    if float(v.get("caption_dropout_rate") or 0) > 0:
        return {"cache_text_encoder_outputs": False, "cache_text_encoder_outputs_to_disk": False}
    return {}


def _enhance_commands(snap: dict, run_dir: Path, launch: list[str]) -> list[tuple]:
    """Texture / enhancement LoRA (enhance.py).  Returns (phase, argv, cwd, total steps of that phase).
    pair:  copy  - LoRA that reproduces the copy-side images (kept in run_dir/copy, not offered as a result)
           merge - that LoRA merged into the base model (run_dir/copy/merged.safetensors, deleted when the run ends)
           training - the result LoRA on the target-side images, trained on the merged model
    words: training - LECO slider from prompts only.
    No optimizer states are saved (no resume): the pair run's later phase needs the merged model, which is
    deleted at the end, and the LECO script has no resume support in this tool."""
    v = snap["values"]
    files = snap["model"]["files"]
    out_dir = run_dir / "output"
    enh = snap["enhance"]
    py = str(config.TRAIN_PYTHON)
    sd = config.SD_SCRIPTS_DIR
    cmds = []
    if enh["method"] == "words":
        from .enhance import leco_prompts
        pf = run_dir / "leco_prompts.toml"
        pf.write_text(toml.dumps(leco_prompts(enh, snap["model"]["variant"], float(v.get("leco_guidance", 1.5)),
                                                        int(v.get("resolution", 768)))), encoding="utf-8")
        cfg = {"pretrained_model_name_or_path": files["ckpt"], "prompts_file": str(pf), "output_dir": str(out_dir),
               "output_name": snap["output_name"], "save_model_as": "safetensors", "save_precision": "fp16",
               "network_module": "networks.lora", "network_dim": int(v["network_dim"]), "network_alpha": float(v["network_alpha"]),
               "learning_rate": float(v["learning_rate"]), "optimizer_type": v["optimizer_type"], "lr_scheduler": v["lr_scheduler"],
               "max_train_steps": int(v["max_train_steps"]), "save_every_n_steps": int(v["save_every_n_steps"]),
               "max_denoising_steps": 20, "seed": int(v["seed"]), "mixed_precision": "bf16", "sdpa": True,
               "gradient_checkpointing": True}
        if VARIANTS[snap["model"]["variant"]]["prediction"] == "v":
            cfg.update(v_parameterization=True, zero_terminal_snr=True)
        cfg_path = run_dir / "train.toml"
        cfg_path.write_text(toml.dumps(cfg), encoding="utf-8")
        # --prompts_file is a required argument, which argparse checks before the config file is read
        cmds.append(("training", launch + ["sdxl_train_leco.py", "--config_file", str(cfg_path), "--prompts_file", str(pf)],
                     sd, int(v["max_train_steps"])))
        return cmds
    copy_dir, target_dir = pair_dirs(run_dir, snap)
    work = run_dir / "copy"
    work.mkdir(exist_ok=True)
    copy_steps = int(v.get("copy_steps") or v["max_train_steps"])
    vc = dict(v, max_train_steps=copy_steps, save_every_n_steps=copy_steps)
    cfg = _sdxl_network_cfg(snap, vc, files["ckpt"], _ds_sdscripts(run_dir, vc, copy_dir, "dataset_copy.toml"), work, "copy")
    cfg["save_state"] = False
    cfg.pop("save_last_n_steps_state", None)
    p1 = run_dir / "train_copy.toml"
    p1.write_text(toml.dumps(cfg), encoding="utf-8")
    cmds.append(("copy", launch + ["sdxl_train_network.py", "--config_file", str(p1)], sd, copy_steps))
    merged = work / "merged.safetensors"
    cmds.append(("merge", [py, "networks/sdxl_merge_lora.py", "--sd_model", files["ckpt"], "--save_to", str(merged),
                           "--models", str(work / "copy.safetensors"), "--ratios", "1.0", "--precision", "float",
                           "--save_precision", "fp16"], sd, None))
    cfg = _sdxl_network_cfg(snap, v, str(merged), _ds_sdscripts(run_dir, v, target_dir), out_dir, snap["output_name"])
    cfg["save_state"] = False
    cfg.pop("save_last_n_steps_state", None)
    p2 = run_dir / "train.toml"
    p2.write_text(toml.dumps(cfg), encoding="utf-8")
    cmds.append(("training", launch + ["sdxl_train_network.py", "--config_file", str(p2)], sd, int(v["max_train_steps"])))
    return cmds


# ======================================================================== error translation
ERROR_RULES = [
    (re.compile(r"out of memory|OutOfMemoryError|CUDA error: out of memory", re.I), "oom",
     "GPU メモリ不足で停止しました。",
     ["詳細設定でバッチサイズを 1 にする", "学習解像度を下げる（例: 1024 → 896）", "ブロックスワップ数を増やす（対応モデルのみ）",
      "他の GPU を使うアプリを閉じる"]),
    (re.compile(r"No space left|not enough space on the disk|Errno 28", re.I), "disk",
     "ディスクの空き容量が不足して停止しました。", ["不要な保存版やキャッシュを削除する", "空きのあるドライブにプロジェクトを置く"]),
    (re.compile(r"FileNotFoundError|No such file or directory", re.I), "missing",
     "必要なファイルが見つかりませんでした。", ["設定画面でモデルファイルの場所を確認する"]),
    (re.compile(r"safetensors_rust\.SafetensorError|HeaderTooLarge|invalid load key|Error while deserializing", re.I), "badmodel",
     "モデルファイルを読み込めませんでした（破損または非対応形式）。", ["モデルを再ダウンロードする", "モデルの種類（系列）の指定を確認する"]),
    (re.compile(r"KeyError|size mismatch|Missing key", re.I), "incompatible",
     "モデルの構造が学習スクリプトと一致しません。", ["モデルの系列（SDXL / Anima）の指定を確認する"]),
]


def translate_error(log_tail: str, rc: int | None) -> dict:
    for rx, code, msg, fixes in ERROR_RULES:
        if rx.search(log_tail):
            return {"code": code, "message": msg, "fixes": fixes, "rc": rc}
    return {"code": "unknown", "message": f"学習エンジンがエラーで終了しました（終了コード {rc}）。", "fixes": ["技術ログを確認する"], "rc": rc}


# ======================================================================== runner
STEP_RE = re.compile(r"steps:\s*\d+%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([\d:]+)<([\d:?]+),\s*([\d.?]+)\s*(s/it|it/s)?(?:,\s*avr_loss=([\d.eE+-]+|nan))?")
CACHE_RE = re.compile(r"(caching latents|cache_latents|Caching latents|caching text encoder|Encoding|cache text encoder)", re.I)
TQDM_RE = re.compile(r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)")

# Running out of VRAM on Windows shows up as a run that suddenly becomes many times slower (measured: 1.39 s/step
# -> 410 s/step), not as an error.  The shared memory counter sees it directly; the step time is the fallback when
# the counter cannot be read.  One slow window is usually a checkpoint save, so it has to last.
SLOW_WINDOW = 10     # steps per speed measurement
SLOW_FACTOR = 3.0    # times slower than the fastest window of the run
SLOW_WINDOWS = 2     # windows in a row
VRAM_KEYS = ("train_batch_size", "resolution", "blocks_to_swap", "convrot_int8", "convrot_int8_text_encoder")


def vram_advice(family: str, values: dict) -> str:
    """What to change after VRAM ran out, the ones that leave the result as it is first."""
    tips = []
    bmax = presets.BLOCK_SWAP_MAX.get(family)
    swap = int(values.get("blocks_to_swap", 0) or 0)
    if bmax and swap < bmax:
        tips.append(f"ブロックスワップ数を増やす（今は {swap}、最大 {bmax}。学習の結果は変わらず、1 ステップが少し遅くなります）")
    if family == "moss_image" and not (int(values.get("convrot_int8", 1)) and int(values.get("convrot_int8_text_encoder", 1))):
        tips.append("本体・テキストエンコーダーを INT8 で読み込む（1 にする）")
    if int(values.get("train_batch_size", 1) or 1) > 1:
        tips.append("バッチサイズを 1 にする（学習の進み方が変わるので、結果の比較画像で確かめてください）")
    if int(values.get("resolution", 1024) or 1024) > 1024:
        tips.append("学習解像度を 1024 にする")
    tips.append("ほかに GPU を使っているソフトを終了する")
    return "、".join(tips)


def _vram_lighter(now: dict, then: dict) -> bool:
    """The settings now need less VRAM than the ones of a run that ran out (any one change counts)."""
    def n(d, k, default):
        try:
            return int(d.get(k, default))
        except (TypeError, ValueError):
            return default
    return (n(now, "train_batch_size", 1) < n(then, "train_batch_size", 1) or n(now, "resolution", 1024) < n(then, "resolution", 1024)
            or n(now, "blocks_to_swap", 0) > n(then, "blocks_to_swap", 0)
            or any(n(now, k, 0) > n(then, k, 0) for k in ("convrot_int8", "convrot_int8_text_encoder")))


def last_vram_short(project, model_id: str) -> dict | None:
    """The newest run of this model that reached training, when VRAM ran out in it (else None: a later run that went
    well means the problem was dealt with)."""
    for r in project.runs():
        p = r.get("progress") or {}
        if (r.get("config") or {}).get("model", {}).get("id") != model_id or not p.get("step"):
            continue
        return {"name": r.get("name") or f"#{r['id']}", **p["vram_short"]} if p.get("vram_short") else None
    return None


def update_vram_warning(job) -> None:
    """The warning on the training screen from what was seen (shared memory, step time), and a mark kept with the run
    so the next start can say the same settings will be slow again."""
    pr = job.progress
    spill, slow = pr.get("spill_gb"), pr.get("slow_x")
    if not (spill or slow):
        pr.pop("warning", None)
        return
    fam, values = getattr(job, "vram_ctx", ("", {}))
    seen = (f"VRAM が足りず、メインメモリ（約 {spill:.1f}GB）で代用しているため大幅に遅くなっています。" if spill else
            f"学習の速さが、この回でいちばん速かったときの約 {slow:.0f} 分の 1 に落ちています。VRAM が足りず、メインメモリで代用している可能性があります。")
    pr["warning"] = seen + "停止して、次のどれかを試してください: " + vram_advice(fam, values) + "。"
    pr["vram_short"] = {k: values[k] for k in VRAM_KEYS if k in values}


def resume_block_reason(snap: dict, state_path) -> str | None:
    """Why a run cannot continue from this saved state, or None.  Checked against the pinned engines:
    - musubi-tuner (Krea 2 / Qwen-Image / Z-Image) restores the weights and optimizer but restarts at step 0
      and trains max_train_steps again ("TODO skip until initial step"), writing over the earlier step files.
    - sd-scripts with skip_until_initial_step skips whole epochs and then sets global_step to the remainder
      within the epoch (train_network.py), so once the saved step reaches one epoch the step count restarts.
      A resumed run goes through bridges/sd_resume.py, which keeps the saved step (from MOSS-LoRA-Trainer);
      checked on the pinned sources against a model of the engine's loop (test_sd_resume.py), not by a real resume."""
    if snap.get("enhance"):
        return "質感・補強 LoRA は途中からの再開に対応していません。"
    if snap.get("engine") not in ("sd-scripts", "sd-scripts-moss"):
        return ("このモデルの学習エンジン（musubi-tuner）は、保存地点からの再開でステップ数が 0 から数え直され、"
                "設定したステップ数を最初から学習し直してしまうため、再開できません。保存済みの LoRA は比較・書き出しに使えます。")
    try:
        st = json.loads((Path(state_path) / "train_state.json").read_text(encoding="utf-8"))
        saved = int(st["current_step"])
    except (OSError, ValueError, KeyError, TypeError):
        return "再開用の保存地点の情報（train_state.json）を読めません。"
    if saved >= int(snap["values"]["max_train_steps"]):
        return "この保存地点は学習の最後です。続きはありません。"
    return None


def lora_file_complete(path: Path) -> bool:
    """A .safetensors file cut off mid-write (crash, console closed) still has its header; its size tells."""
    import struct
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n > 200_000_000 or 8 + n > size:
                return False
            hdr = json.loads(f.read(n))
        end = max((t["data_offsets"][1] for k, t in hdr.items() if k != "__metadata__"), default=0)
        return size >= 8 + n + end
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, struct.error):
        return False


def state_is_complete(path: Path) -> bool:
    """A resumable state dir must contain train_state.json, the model and a complete optimizer file.
    torch.save writes zip archives; a file cut off mid-write has no central directory."""
    import zipfile
    path = Path(path)
    if not (path / "train_state.json").exists():
        return False
    bins = list(path.glob("*.bin"))
    if not bins or not any(path.glob("*.safetensors")):
        return False
    return all(zipfile.is_zipfile(b) for b in bins)


def kill_tree(pid: int):
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,
                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


_job_seq = iter(range(1, 1 << 62))


class Job:
    def __init__(self, project, run_id, resume=None):
        self.project, self.run_id, self.resume = project, run_id, resume
        self.uid = next(_job_seq)   # names a waiting job for reordering (tasks have no run id)
        self.proc: subprocess.Popen | None = None
        self.stop_requested_at: float | None = None
        self.stop_after_step: int | None = None
        self.force = False
        self.output_name: str | None = None      # of the run snapshot, read once (polled every 2 s)
        self.vram_baseline: float | None = None  # shared GPU memory before the engine started; None = not measured
        self.eval_eta: int | None = None         # seconds the comparison after training took last time, if known
        self.vram_ctx: tuple[str, dict] = ("", {})   # (family, values) of the run, for the VRAM advice
        self.progress = {"phase": "queued", "step": 0, "total": None, "loss": None, "speed": None, "eta_s": None,
                         "message": "待機中"}


def _log_error(what: str, tb: str) -> None:
    try:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.DATA_DIR / "server_errors.log", "a", encoding="utf-8") as f:
            f.write(time.strftime("%F %T ") + what + "\n" + tb + "\n")
    except OSError:
        pass


class Runner:
    def __init__(self, projects):
        self.projects = projects
        self.jobs: list[Job] = []
        self.current: Job | None = None
        self._cond = threading.Condition()
        self.after_train_hooks = []   # fn(project, run_id) -> called after a successful run
        # paused: the job running now finishes, the waiting ones do not start (kept across restarts)
        self.paused = bool(config.settings().get("queue_paused", False))
        threading.Thread(target=self._loop, daemon=True, name="job-runner").start()

    # ---------------------------------------------------------------- queue
    def enqueue(self, project, run_id, resume=None):
        with self._cond:
            self.jobs.append(Job(project, run_id, resume))
            self._save_queue()
            self._cond.notify_all()
        self._set(project, run_id, status="queued", progress={"phase": "queued", "message": "待機中"})
        bus.publish("queue")

    def _save_queue(self):
        """The waiting training runs, in order, so a reservation survives a restart of the app (call under _cond).
        Comparison tasks are not kept: they are marked interrupted at startup and can be added again."""
        try:
            config.atomic_write_json(QUEUE_FILE, [{"project": j.project.id, "run": j.run_id, "resume": j.resume}
                                                  for j in self.jobs if not getattr(j, "task", None)])
        except OSError as e:
            _log_error("予約の保存", repr(e))

    def waiting(self) -> list[Job]:
        with self._cond:
            return list(self.jobs)

    def move(self, uid: int, to: str) -> bool:
        """Reorder a waiting job: to = "up" / "down" / "top"."""
        with self._cond:
            i = next((n for n, j in enumerate(self.jobs) if j.uid == uid), None)
            if i is None:
                return False
            j = self.jobs.pop(i)
            n = 0 if to == "top" else max(0, i - 1) if to == "up" else min(len(self.jobs), i + 1)
            self.jobs.insert(n, j)
            self._save_queue()
        bus.publish("queue")
        return True

    def set_paused(self, on: bool):
        with self._cond:
            self.paused = bool(on)
            self._cond.notify_all()
        try:
            config.settings().set("queue_paused", bool(on))
        except OSError as e:
            _log_error("順番待ちの一時停止の保存", repr(e))
        bus.publish("queue")

    def enqueue_task(self, project, label, fn, on_cancel=None):
        """Other GPU work (comparison generation) shares the same serial queue as training.  on_cancel marks
        the task's own record cancelled when it is taken out of the queue before it starts (without it, the
        task cannot be cancelled from the queue)."""
        job = Job(project, None)
        job.task, job.label, job.on_cancel = fn, label, on_cancel
        with self._cond:
            self.jobs.append(job)
            self._cond.notify_all()
        bus.publish("queue")
        return job

    def cancel_task(self, uid: int) -> bool:
        """Take a waiting task (not a training run) out of the queue."""
        with self._cond:
            j = next((x for x in self.jobs if x.uid == uid and getattr(x, "task", None)), None)
            if j is None or not getattr(j, "on_cancel", None):
                return False
            self.jobs.remove(j)
        try:
            j.on_cancel()
        except Exception:
            _log_error(f"GPU タスク「{j.label}」の取り消し", traceback.format_exc())
        bus.publish("queue")
        return True

    def find(self, project_id, run_id) -> Job | None:
        for j in ([self.current] if self.current else []) + self.jobs:
            if j and j.project.id == project_id and j.run_id == run_id and not getattr(j, "task", None):
                return j
        return None

    def stop(self, project, run_id, force=False) -> dict:
        with self._cond:
            for j in self.jobs:
                if j.project.id == project.id and j.run_id == run_id and not getattr(j, "task", None):
                    self.jobs.remove(j)
                    self._save_queue()
                    self._set(project, run_id, status="cancelled", ended=time.time(),
                              progress={"phase": "cancelled", "message": "開始前に中止しました"})
                    return {"accepted": True, "message": "開始前の学習を中止しました。"}
        j = self.current
        if not j or j.project.id != project.id or j.run_id != run_id:
            return {"accepted": False, "message": "実行中の学習ではありません。"}
        if j.progress.get("phase") == "evaluating":
            # the LoRA is finished; comparison images stop after the image being generated
            j.eval_cancel = True
            j.progress["message"] = "比較画像の生成を中止しています（今の 1 枚が終わったら止まります）。学習した LoRA は残っています。"
            self._publish(j)
            return {"accepted": True, "message": j.progress["message"]}
        if not force and j.progress.get("phase") == "training":
            every = max(1, int(j.project.run(run_id)["config"]["values"]["save_every_n_steps"]))
            total = int(j.progress.get("total") or j.project.run(run_id)["config"]["values"]["max_train_steps"])
            stop_at = ((j.progress.get("step") or 0) // every + 1) * every
            if stop_at >= total:
                # the next save is the end of the run: let it finish (a stop there would mark a finished run 中止)
                return {"accepted": False, "message": f"次の保存地点が学習の最後（ステップ {total}）です。このまま完了まで進みます。"
                                                      "すぐに止めたい場合は強制停止してください。"}
        j.stop_requested_at = time.time()
        if force or j.progress.get("phase") != "training":
            j.force = True
            if j.proc and j.proc.poll() is None:
                kill_tree(j.proc.pid)
            return {"accepted": True, "message": "強制停止しました。最後に保存された地点までのデータが残ります。"}
        j.stop_after_step = stop_at
        j.progress["message"] = f"停止要求を受け付けました。次の保存地点（ステップ {j.stop_after_step}）で停止します。"
        self._publish(j)
        return {"accepted": True, "message": j.progress["message"], "stop_at": j.stop_after_step}

    # ---------------------------------------------------------------- internals
    def _set(self, project, run_id, **fields):
        cols, vals = [], []
        for k, v in fields.items():
            cols.append(f"{k}=?")
            vals.append(json.dumps(v, ensure_ascii=False) if k in ("progress", "error") and v is not None else v)
        vals.append(run_id)
        with project.tx() as db:
            db.execute(f"UPDATE runs SET {', '.join(cols)} WHERE id=?", vals)
        bus.publish("run", project=project.id, run=run_id, **{k: v for k, v in fields.items() if k in ("status", "progress")})

    def _publish(self, job: Job):
        bus.publish("run", project=job.project.id, run=job.run_id, progress=job.progress)

    def _loop(self):
        while True:
            with self._cond:
                self._cond.wait_for(lambda: bool(self.jobs) and not self.paused)
                job = self.jobs.pop(0)
                self.current = job
                self._save_queue()
            bus.publish("queue")
            if getattr(job, "task", None):
                try:
                    gpu.acquire(job.label)
                    job.task()
                except Exception:
                    # the task marks its own row failed when it can; the queue must keep running either way
                    _log_error(f"GPU タスク「{job.label}」", traceback.format_exc())
                finally:
                    gpu.release(job.label)
                    self.current = None
                continue
            owner = f"学習 #{job.run_id}"
            try:
                # the frozen dataset files are written while the GPU is being freed (ComfyUI and the analysis worker
                # stopping take seconds); the snapshot is fixed, so nothing written later depends on that order
                prep = self._materialize_async(job)
                gpu.acquire(owner, on_wait=lambda job=job: self._waiting_for_analysis(job))
                self._run(job, prep)
            except Exception as e:
                tb = traceback.format_exc()
                try:
                    (Path(job.project.run(job.run_id)["run_dir"]) / "runner_error.log").write_text(tb, encoding="utf-8")
                except Exception:
                    _log_error(f"学習 #{job.run_id}", tb)
                try:
                    self._set(job.project, job.run_id, status="failed", ended=time.time(),
                              error={"code": "internal", "message": f"内部エラー: {e}", "fixes": ["技術ログを確認する"]})
                except Exception:
                    _log_error(f"学習 #{job.run_id} の失敗の記録", traceback.format_exc())
            finally:
                gpu.release(owner)
                self.current = None

    def _materialize_async(self, job: Job) -> dict:
        run = job.project.run(job.run_id)
        box = {}

        def work():
            try:
                materialize(Path(run["run_dir"]), run["config"])
            except BaseException as e:
                box["error"] = e
        box["thread"] = threading.Thread(target=work, daemon=True, name="materialize")
        box["thread"].start()
        return box

    def _waiting_for_analysis(self, job: Job):
        """The GPU is ours, but an analysis request (material analysis, captions, i2t) has to end first."""
        job.progress["message"] = "GPU を使う解析（素材の解析・説明文の作成・i2t など）が終わるのを待っています"
        self._set(job.project, job.run_id, progress=dict(job.progress))

    def _measure_vram_baseline(self, job: Job):
        def work():
            job.vram_baseline = shared_gpu_memory_gb()   # None when the counter cannot be read: no warnings then
        threading.Thread(target=work, daemon=True, name="vram-baseline").start()

    @staticmethod
    def _eval_estimate(project, snap: dict) -> int | None:
        """Seconds the comparison right after training took for the latest run on the same model and kind (recorded
        by evaluate as results.timing).  None without such a run: the remaining time then covers training only."""
        try:
            rows = project.q("SELECT r.config AS config, e.results AS results FROM evals e JOIN runs r ON r.id = e.run_id "
                             "WHERE e.status='done' AND e.id IN (SELECT MIN(id) FROM evals GROUP BY run_id) "
                             "ORDER BY e.id DESC LIMIT 5")
            for r in rows:
                cfg = json.loads(r["config"] or "{}")
                if (cfg.get("model") or {}).get("id") != snap["model"]["id"] or cfg.get("kind") != snap.get("kind"):
                    continue
                sec = ((json.loads(r["results"] or "{}")).get("timing") or {}).get("seconds")
                if sec:
                    return int(sec)
        except Exception:
            pass
        return None

    def _register_checkpoints(self, job: Job, out_dir: Path, final: bool = False):
        """final=True after the engine process has exited: nothing can still be being written."""
        if not job.output_name:
            job.output_name = job.project.run(job.run_id)["config"]["output_name"]
        name = job.output_name
        found = []
        for p in out_dir.iterdir():
            m = re.fullmatch(re.escape(name) + r"-step(\d+)(-state)?(\.safetensors)?", p.name)
            if m and (m.group(2) or m.group(3)):
                found.append((int(m.group(1)), str(p), "state" if m.group(2) else "lora"))
            elif p.name == name + ".safetensors":
                found.append((None, str(p), "lora"))
            elif p.name == name + "-state":
                found.append((None, str(p), "state"))
        new_steps = []
        with job.project.tx() as db:
            known = {r["path"] for r in db.execute("SELECT path FROM checkpoints WHERE run_id=?", (job.run_id,))}
            for step, path, kind in found:
                if path in known:
                    continue
                try:
                    if not final and kind == "lora" and time.time() - os.path.getmtime(path) < 3:
                        continue  # possibly still being written; pick up next poll
                except OSError:
                    continue      # removed in the meantime (engine rotation)
                if kind == "lora" and not lora_file_complete(Path(path)):
                    continue      # cut off mid-write (crash) or still being written
                if kind == "state" and not state_is_complete(Path(path)):
                    continue   # still being written (or broken): never offer it for resume
                db.execute("INSERT OR IGNORE INTO checkpoints(run_id, step, path, kind, created) VALUES(?,?,?,?,?)",
                           (job.run_id, step, path, kind, time.time()))
                new_steps.append((step, kind))
            # states removed by the engine's rotation (save_last_n_steps_state)
            for r in db.execute("SELECT id, path FROM checkpoints WHERE run_id=? AND kind='state'", (job.run_id,)).fetchall():
                if not Path(r["path"]).exists():
                    db.execute("DELETE FROM checkpoints WHERE id=?", (r["id"],))
        if new_steps:
            bus.publish("checkpoints", project=job.project.id, run=job.run_id)
        return new_steps

    def _run(self, job: Job, prep: dict | None = None):
        run = job.project.run(job.run_id)
        snap = run["config"]
        run_dir = Path(run["run_dir"])
        out_dir = run_dir / "output"
        logs = run_dir / "logs"
        logs.mkdir(exist_ok=True)
        self._set(job.project, job.run_id, status="preparing", error=None,
                  progress={"phase": "preparing", "message": "準備中（モデル読み込み・キャッシュ作成）"})
        job.progress.update(phase="preparing", message="準備中（モデル読み込み・キャッシュ作成）")
        self._measure_vram_baseline(job)   # once per run, in the background (PowerShell takes 1-2 s)
        job.eval_eta = self._eval_estimate(job.project, snap)
        if prep is None:
            materialize(run_dir, snap)
        else:
            prep["thread"].join()
            if "error" in prep:
                raise prep["error"]
        try:   # encoder caches of earlier runs with the same images, captions and model
            reused = traincache.prime(job.project.path, run_dir, snap) if traincache.keep() else 0
        except Exception:
            reused = 0   # only a speed-up: the engine encodes what is missing
        if reused:
            job.progress["message"] = f"準備中（前の回の前処理を {reused} 件使い、残りを作成）"
            self._set(job.project, job.run_id, progress=dict(job.progress))
        cmds = [traincache.with_skip_existing(c) for c in build_commands(snap, run_dir, job.resume)]
        if reused:
            try:
                cmds = traincache.without_complete(cmds, traincache.complete(run_dir, snap))
            except Exception:
                pass   # the engine skips the existing files itself (after loading its encoder)
        total = int(snap["values"]["max_train_steps"])
        try:
            return self._run_cmds(job, snap, run_dir, out_dir, logs, cmds, total)
        finally:
            if not traincache.keep():   # the user keeps no training caches: this run's go when it ends
                try:
                    traincache.clear_run(run_dir)
                except Exception:
                    pass
            elif (job.progress.get("step") or 0) > 0:   # caching is finished once training steps run
                try:
                    traincache.harvest(job.project.path, run_dir, snap)
                except Exception:
                    pass
            merged = run_dir / "copy" / "merged.safetensors"   # texture LoRA from image pairs: ~7GB, only needed while training
            if merged.exists():
                try:
                    merged.unlink()
                except OSError:
                    pass

    def _run_cmds(self, job: Job, snap, run_dir: Path, out_dir: Path, logs: Path, cmds, total: int):
        for i, cmd in enumerate(cmds):
            phase, argv, cwd = cmd[:3]
            if job.stop_requested_at:
                break
            log_path = logs / f"{i:02d}_{phase}.log"
            rc, tail = self._exec(job, phase, argv, cwd, log_path, cmd[3] if len(cmd) > 3 and cmd[3] else total, out_dir)
            if job.stop_requested_at:
                break
            if rc != 0:
                err = translate_error(tail, rc)
                err["log"] = str(log_path)
                self._register_checkpoints(job, out_dir, final=True)
                self._set(job.project, job.run_id, status="failed", ended=time.time(), error=err,
                          progress={**job.progress, "phase": "failed", "message": err["message"]})
                return
        self._register_checkpoints(job, out_dir, final=True)
        if job.stop_requested_at:
            msg = "強制停止しました。" if job.force else "保存地点で停止しました。"
            self._set(job.project, job.run_id, status="cancelled", ended=time.time(),
                      progress={**job.progress, "phase": "cancelled", "message": msg})
            return
        final = out_dir / (snap["output_name"] + ".safetensors")
        if not final.exists():
            self._set(job.project, job.run_id, status="failed", ended=time.time(),
                      error={"code": "nooutput", "message": "学習は終了しましたが LoRA ファイルが見つかりません。", "fixes": ["技術ログを確認する"]},
                      progress={**job.progress, "phase": "failed"})
            return
        job.progress.update(phase="evaluating", step=total, total=total, eta_s=None, message="学習完了。比較画像を準備中")
        self._set(job.project, job.run_id, status="evaluating", progress=dict(job.progress))
        for fn in self.after_train_hooks:
            fn(job.project, job.run_id)

    def _exec(self, job: Job, phase, argv, cwd, log_path: Path, total: int, out_dir: Path):
        engine = next((e for e in ("sd-scripts", "musubi-tuner", "musubi-tuner-qwen21", "sd-scripts-moss") if Path(cwd) == engine_cwd(e)), "")
        env = engine_env(engine)
        flags = (getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
                 | getattr(subprocess, "CREATE_NO_WINDOW", 0))
        watch_stop = threading.Event()

        def watch_vram():
            # On Windows, running out of VRAM spills into system RAM instead of raising OOM and the run
            # silently becomes many times slower; warn with a concrete fix.
            while not watch_stop.wait(20):
                baseline = job.vram_baseline
                if job.progress.get("phase") != "training" or baseline is None:
                    continue   # no baseline (counter unreadable): never warn on a guess
                g = shared_gpu_memory_gb()
                if g is not None and g - baseline > 1.5:
                    job.progress["spill_gb"] = round(g - baseline, 1)
                else:
                    job.progress.pop("spill_gb", None)
                update_vram_warning(job)

        snap = job.project.run(job.run_id)["config"]
        job.vram_ctx = (snap["model"]["family"], snap["values"])
        threading.Thread(target=watch_vram, daemon=True, name="vram-watch").start()
        # runs without optimizer states (texture LoRA, musubi-tuner) stop safely at the next saved LoRA file instead
        stop_kind = "state" if saves_state(snap) else "lora"
        if phase == "merge":
            job.progress["message"] = "準備中（下準備 2/2: 覚え込ませた LoRA をモデルに合成しています）"
            self._publish(job)
        with open(log_path, "ab") as logf:
            logf.write(("$ " + " ".join(argv) + "\n").encode("utf-8"))
            job.proc = subprocess.Popen(argv, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                        creationflags=flags)
            if job.force:
                kill_tree(job.proc.pid)   # force stop arrived while the previous command was ending
            try:
                return self._read_engine(job, phase, total, out_dir, logf, stop_kind)
            finally:
                watch_stop.set()
                if job.proc.poll() is None:   # an exception while reading: the engine must not keep the GPU
                    kill_tree(job.proc.pid)
                self._publish(job)

    def _read_engine(self, job: Job, phase, total: int, out_dir: Path, logf, stop_kind: str):
        tail = []
        buf = b""
        last_pub = 0.0
        t_first = None
        step_first = None
        win = None          # (time, step) where the current speed window began
        best = None         # fastest window of this command (s/step)
        slow_n = 0          # slow windows in a row
        last_poll = 0.0
        stream = job.proc.stdout
        while True:
            chunk = stream.read1(65536) if hasattr(stream, "read1") else stream.read(4096)
            if not chunk:
                break
            logf.write(chunk)
            buf += chunk
            parts = re.split(rb"[\r\n]", buf)
            buf = parts.pop()
            for raw in parts:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                tail.append(line)
                if len(tail) > 300:
                    del tail[:100]
                m = STEP_RE.search(line)
                if m and phase == "copy":
                    # texture LoRA from image pairs: the preparatory LoRA is not the result; report it as preparation
                    job.progress["message"] = f"準備中（下準備 1/2: 加工前の画像を覚え込ませています {m.group(1)}/{m.group(2)}）"
                elif m:
                    step, tot = int(m.group(1)), int(m.group(2))
                    if total and tot != total:
                        # after a resume sd-scripts' bar spans (max_train_steps - skipped remainder);
                        # report against the configured total so the UI shows the real step number
                        step, tot = step + (total - tot), total
                    if job.progress.get("phase") != "training":
                        job.progress["phase"] = "training"
                        self._set(job.project, job.run_id, status="training")
                    now = time.time()
                    if step > 0 and t_first is None:
                        t_first, step_first = now, step
                    loss = m.group(7)
                    job.progress.update(step=step, total=tot, loss=float(loss) if loss and loss != "nan" else None)
                    if win is None or step < win[1]:
                        win = (now, step)
                    elif step - win[1] >= SLOW_WINDOW:
                        w = (now - win[0]) / (step - win[1])
                        win = (now, step)
                        best = w if best is None else min(best, w)
                        slow_n = slow_n + 1 if w > best * SLOW_FACTOR else 0
                        if slow_n >= SLOW_WINDOWS:
                            job.progress["slow_x"] = round(w / best, 1)
                            job.progress["recent_speed"] = w
                        else:
                            job.progress.pop("slow_x", None)
                            job.progress.pop("recent_speed", None)
                        update_vram_warning(job)
                    if t_first is not None and step - step_first >= 5:
                        # after a slowdown the average since the start would promise far too early an end
                        spd = job.progress.get("recent_speed") or (now - t_first) / (step - step_first)
                        job.progress["speed"] = round(spd, 3)
                        # plus the comparison images made right after training, as long as they took last time
                        job.progress["eta_s"] = int(spd * (tot - step)) + (job.eval_eta or 0)
                        job.progress["eval_s"] = job.eval_eta
                    else:
                        job.progress["eta_s"] = None   # 計測中
                    if not job.stop_after_step:
                        job.progress["message"] = "学習中"
                    hist = job.progress.setdefault("loss_hist", [])
                    if job.progress["loss"] is not None and (not hist or hist[-1][0] != step):
                        hist.append([step, round(job.progress["loss"], 5)])
                        if len(hist) > 400:
                            job.progress["loss_hist"] = hist[::2]
                elif phase == "preparing" or job.progress.get("phase") == "preparing":
                    if CACHE_RE.search(line) or TQDM_RE.search(line):
                        mm = TQDM_RE.search(line)
                        label = "キャッシュ作成" if CACHE_RE.search(line) else "モデル読み込み・キャッシュ作成"
                        job.progress["message"] = f"準備中（{label}）" + (f" {mm.group(2)}/{mm.group(3)}" if mm else "")
            now = time.time()
            if now - last_poll > 2:
                last_poll = now
                new = self._register_checkpoints(job, out_dir) if out_dir.exists() else []
                if (job.stop_after_step and (job.progress.get("step") or 0) > job.stop_after_step
                        and any(c["step"] == job.stop_after_step and c["kind"] == stop_kind
                                and (stop_kind == "lora" or state_is_complete(Path(c["path"])))
                                for c in job.project.checkpoints(job.run_id))):
                    # the engine reported a step after the save point, so the save has completed
                    job.progress["message"] = f"ステップ {job.stop_after_step} を保存しました。停止します。"
                    kill_tree(job.proc.pid)
            if now - last_pub > 1.0:
                last_pub = now
                self._publish(job)
                job.project.q("UPDATE runs SET progress=? WHERE id=?",
                              (json.dumps(job.progress, ensure_ascii=False), job.run_id))
        rc = job.proc.wait()
        return rc, "\n".join(tail[-200:])
