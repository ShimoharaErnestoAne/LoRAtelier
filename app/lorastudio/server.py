"""HTTP API + static UI.  Bound to 127.0.0.1 only; no share links."""
from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import shutil
import sqlite3
import threading
import time
import uuid
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from . import captions, config, generate, judge, masks, presets, promptkeep, runcompare
from .analysis import ANALYZER_VERSION, Analyzer, worker
from .events import bus
from .evaluate import EXTRA_MODELS_KEY, MAX_EXTRA_MODELS, _extra_spec, comfy, default_spec, prompt_sets, run_eval, run_prompts
from .export import export as do_export
from .gpu import gpu, query_gpus
from . import addons, i2tlink, imagegen, linking, loraconf, moss_catalog, multichar, pipeline, projectio, promptedit, relocate, remote, traincache
from .models import FAMILY_FILES, FAMILY_LABEL, VARIANTS, catalog_entry, detect, hasher, library, registry
from .store import ProjectManager, ProjectMoving, default_trigger
from .training import (ACTIVE, QUEUE_FILE, STATUS_LABEL, Job, Runner, create_run, model_train_errors, preflight, resume_block_reason,
                       state_is_complete)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
HEAVY_AI_KEYS = {"embed", "ccip", "subject_embed"}

app = FastAPI(title="LoRAtelier", docs_url=None, redoc_url=None)
projects = ProjectManager()
analyzer = Analyzer(projects)
runner = Runner(projects)
_gpu_cache = {"t": 0, "v": []}


def gpus_cached():
    """nvidia-smi takes 0.1-0.5 s: the first call waits for it, later calls return the last answer and
    refresh it in the background (so /api/status, polled every few seconds, never waits)."""
    if _gpu_cache["t"] == 0:
        _gpu_cache.update(t=time.time(), v=query_gpus())
    elif time.time() - _gpu_cache["t"] > 10 and not _gpu_cache.get("busy"):
        _gpu_cache.update(t=time.time(), busy=True)

        def refresh():
            try:
                _gpu_cache["v"] = query_gpus()
            finally:
                _gpu_cache["busy"] = False
        threading.Thread(target=refresh, daemon=True).start()
    return _gpu_cache["v"]


def P(pid: str):
    try:
        return projects.get(pid)
    except ProjectMoving:
        raise HTTPException(409, "このプロジェクトはほかの置き場所へ移動中です。移動が終わるまでお待ちください。")
    except KeyError:
        raise HTTPException(404, "プロジェクトが見つかりません")


# ====================================================================== evaluation glue
class EvalCancelled(Exception):
    pass


def _eval_progress(project, run_id):
    def cb(done, total, eta):
        job = runner.current
        # the training run's own comparison, or one added later (a task labelled "比較 #<run>")
        live = job if job and job.project.id == project.id and (
            job.run_id == run_id or getattr(job, "label", None) == f"比較 #{run_id}") else None
        if live and getattr(live, "eval_cancel", False):
            raise EvalCancelled()   # called between images: the image being generated is finished first
        progress = {"phase": "evaluating", "step": done, "total": total, "eta_s": eta, "message": f"比較画像を生成中 {done}/{total}"}
        if live:
            live.progress.update(progress)   # /runs shows the live progress of the current job
        bus.publish("run", project=project.id, run=run_id, progress=progress)
    return cb


def _after_train(project, run_id):
    run = project.run(run_id)
    eid = None
    try:
        spec = default_spec(project, run)
        with project.tx() as db:
            eid = db.execute("INSERT INTO evals(run_id, created, status, spec) VALUES(?,?,?,?)",
                             (run_id, time.time(), "queued", json.dumps(spec, ensure_ascii=False))).lastrowid
        run_eval(project, run_id, eid, _eval_progress(project, run_id))
        msg = "完了"
    except EvalCancelled:
        project.q("UPDATE evals SET status='cancelled', error=? WHERE id=?", ("中止しました", eid))
        msg = "学習は完了しました。比較画像の生成は中止しました（「比較を追加」でやり直せます）。"
    except Exception as e:
        if eid is not None:
            project.q("UPDATE evals SET status='failed', error=? WHERE id=?", (str(e), eid))
        msg = f"学習は完了しました。比較画像の生成に失敗: {e}"
    runner._set(project, run_id, status="done", ended=time.time(),
                progress={"phase": "done", "message": msg, "step": run["config"]["values"]["max_train_steps"],
                          "total": run["config"]["values"]["max_train_steps"]})


runner.after_train_hooks.append(_after_train)


# ====================================================================== status / events
@app.get("/api/status")
def status():
    cur = runner.current
    return {
        "gpus": gpus_cached(), "worker": worker.status, "worker_available": worker.available(),
        "comfy_available": comfy.available(), "train_env": config.TRAIN_PYTHON.exists(),
        "analysis_queue": analyzer.queue_size(), "gpu_owner": gpu.owner,
        "job": {"project": cur.project.id, "run": cur.run_id, "progress": cur.progress,
                "kind": (cv := _current_view(cur))["kind"], "message": cv["message"]} if cur else None,
        "queue_paused": runner.paused,
        "blur_images": config.settings().get("blur_images", False), "keep_train_cache": traincache.keep(),
        "queued": [{"project": j.project.id, "run": j.run_id} for j in runner.jobs],
        "first_run_done": bool(config.settings().get("models")),   # no file checks: asked every 4 s
        "downloads": sum(1 for d in _downloads_view() if d["state"] == "running"),
        "boot": BOOT, "can_restart": LAUNCH is not None,
    }


LAUNCH: dict | None = None   # how this server was started (set by __main__; None in tests): 設定 → 再起動 needs it


def _restart_losses() -> list[str]:
    """What a restart stops now, for the question before it (nothing listed: it restarts at once)."""
    out = []
    cur = runner.current
    if cur:
        v = _job_view(cur)
        what = (f"学習 #{v['run']}" if v["kind"] == "train" else v["label"] or "処理")
        where = f"「{v['project_name']}」の" if v["project_name"] and v["kind"] != "imagegen" else ""
        out.append(f"{where}{what}が途中で止まります" + ("（再起動後は「中断」になり、保存した地点があれば学習の履歴から続きを再開できます）"
                                                    if v["kind"] == "train" else ""))
    waiting = [j for j in runner.jobs if getattr(j, "task", None) is not None]
    if waiting:
        out.append(f"順番待ちの画像の生成・比較画像 {len(waiting)} 件は、再起動後に自動では始まりません（予約した学習は順番待ちのまま残ります）")
    dl = sum(1 for d in _downloads_view() if d["state"] == "running")
    if dl:
        out.append(f"ダウンロード中の {dl} 件が止まります")
    if analyzer.queue_size():
        out.append("素材の解析が止まります（再起動後に続きから解析します）")
    return out


@app.get("/api/restart")
def restart_check():
    return {"losses": _restart_losses(), "can_restart": LAUNCH is not None}


@app.post("/api/restart")
def restart_app():
    """設定 → 再起動: a new server is started in a new window, then this one ends (its window closes, and ComfyUI, the
    workers and training end with it).  The open page waits for the new one and reloads itself."""
    from .procjob import relaunch
    if LAUNCH is None:
        raise HTTPException(409, "この起動方法では再起動できません。start.bat から起動してください。")
    try:
        relaunch(["--port", str(LAUNCH["port"])] + (["--no-browser"] if LAUNCH.get("no_browser") else []))
    except RuntimeError as e:
        raise HTTPException(500, str(e))

    def end():
        time.sleep(0.8)   # the answer reaches the page first
        os._exit(0)       # exit code 0: start.bat does not pause, so its window closes
    threading.Thread(target=end, daemon=True, name="restart").start()
    return {"ok": True, "boot": BOOT}


def _downloads_view() -> list[dict]:
    """Every download of this session for the job page, newest first: the models of 設定 → モデルの追加 (addons) and
    the files from links / dropped files (remote).  cancel: how it is stopped (None: it cannot be)."""
    out = []
    for cid, st in list(addons.state().items()):
        e = catalog_entry(cid) or {}
        out.append({"key": f"model:{cid}", "what": "モデル", "name": e.get("name") or cid, "cancel": f"/api/downloads/{cid}/cancel",
                    **{k: st.get(k) for k in ("state", "done_bytes", "total_bytes", "message", "error", "started")}})
    for st in remote.state():
        up = st.get("src") == "upload"
        out.append({"key": f"link:{st['id']}", "what": "ファイルの受け取り" if up else "リンクから取得", "name": st.get("name") or "",
                    "cancel": None if up else f"/api/import/download/{st['id']}/cancel",
                    **{k: st.get(k) for k in ("state", "done_bytes", "total_bytes", "message", "error", "started")}})
    return sorted(out, key=lambda d: -(d.get("started") or 0))


@app.get("/api/events")
def events(request: Request):
    q = bus.subscribe()

    def gen():
        try:
            yield "retry: 2000\n\n"
            while not getattr(q, "dropped", False):   # dropped as too slow: end, so the browser connects again
                try:
                    msg = q.get(timeout=15)
                    yield f"data: {msg}\n\n"
                except queue.Empty:
                    yield ": ping\n\n"
        finally:
            bus.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# ====================================================================== models
@app.get("/api/models")
def list_models():
    library._repair_paths()   # files moved out of an old folder are found again (no-op when nothing is missing)
    out = registry.list()
    for m in out:
        m["hash"] = {k: hasher.get(p) for k, p in m["files"].items()} if not m["missing"] else {}
    return {"models": out, "variants": VARIANTS, "family_files": FAMILY_FILES, "misplaced": library.misplaced()}


@app.post("/api/models/detect")
def models_detect(body: dict):
    p = Path(body.get("path", "").strip().strip('"'))
    if not p.is_file():
        raise HTTPException(400, f"ファイルが見つかりません: {p}")
    try:
        return detect(p)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        raise HTTPException(400, f"モデルを読めません: {e}")


@app.post("/api/models")
def models_add(body: dict):
    try:
        files = {k: v.strip().strip('"') for k, v in (body.get("files") or {}).items() if v and v.strip()}
        m = registry.add(body.get("name") or "model", body["variant"], files)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))
    for p in m["files"].values():
        hasher.get(p)   # start background hashing now; never blocks
    return m


@app.patch("/api/models/{mid}")
def models_update(mid: str, body: dict):
    m = registry.get(mid)
    if not m:
        raise HTTPException(404)
    files = dict(m["files"])
    for k, v in (body.get("files") or {}).items():
        if v:
            if not Path(v).is_file():
                raise HTTPException(400, f"ファイルが見つかりません: {v}")
            files[k] = v
        else:
            files.pop(k, None)
    extra = {"name": body["name"]} if body.get("name") else {}
    if body.get("variant"):
        v = VARIANTS.get(body["variant"])
        if not v or v["family"] != m["family"] or v.get("available") is False:
            raise HTTPException(400, "このモデルには選べない系統です")
        extra.update(variant=body["variant"], variant_confirmed=True)
    registry.update(mid, files=files, **extra)
    return registry.get(mid)


@app.delete("/api/models/{mid}")
def models_delete(mid: str):
    registry.remove(mid)
    return {"ok": True}


@app.get("/api/library")
def get_library():
    """Model files placed in models/checkpoints, models/loras and the folders the user added."""
    scan = library.scan()
    by_path = {str(Path(f)).lower(): m for m in registry.list() for k, f in m["files"].items() if k in ("ckpt", "gen", "turbo")}
    for c in scan["checkpoints"]:
        m = by_path.get(c["path"].lower())
        c["model_id"] = m["id"] if m else None
        c["family_label"] = FAMILY_LABEL.get(c["family"], "不明")
    for lo in scan["loras"]:
        lo["family_label"] = FAMILY_LABEL.get(lo["family"], "不明")
        lo["conf"] = loraconf.get(lo["path"])
    comp = library.companion_files()
    # whether each text encoder / VAE file fits each model family (read from the file; True / False / None unknown)
    fits = {kind: {p: {fam: library.fits(fam, kind, p) for fam in generate.GEN_FAMILIES} for p in comp[kind]} for kind in ("te", "vae")}
    models = [m for m in registry.list() if not m["missing"]]
    for m in models:   # what would stop training with it (the same checks as before training starts)
        m["train_errors"] = model_train_errors(m)
    return {"default_dirs": [str(d) for d in library.default_dirs() if d.is_dir()], "roots": library.roots(), **scan,
            "models": models, "companions": comp, "companion_fits": fits}


@app.post("/api/models/from_file")
def models_from_file(body: dict):
    """A model file chosen in a model list that is not registered yet (e.g. its text encoder / VAE has another name
    than the usual one): registered with the text encoder / VAE chosen there.  {"needs": [...]} when they are to be chosen."""
    try:
        m = library.register_file(str(body.get("path") or ""), str(body.get("te") or ""), str(body.get("vae") or ""))
    except ValueError as e:
        if getattr(e, "needs", None):
            return {"needs": e.needs, "choices": e.choices, "message": str(e)}
        raise HTTPException(400, str(e))
    return {"id": m["id"]}


def _library_lora(path: str) -> str:
    """Only LoRA files the library lists (models/loras and the folders the user added) can be configured."""
    f = Path(path or "")
    ok = False
    try:   # checked on this one file; scanning every LoRA folder per save is slow with many LoRAs (HDD)
        rf = f.resolve()
        ok = (f.is_file() and f.suffix.lower() in (".safetensors", ".gguf")
              and any(rf.is_relative_to(d.resolve()) for d in library._folders() if d.is_dir())
              and bool(library._info(f).get("is_lora")))
    except OSError:
        pass
    if not ok:
        raise HTTPException(404, "その LoRA は一覧にありません。フォルダーを読み直してください。")
    return path


@app.put("/api/loras/settings")
def lora_settings_save(body: dict):
    p = _library_lora(body.get("path"))
    try:
        return loraconf.save(p, body.get("triggers"), body.get("auto", True), body.get("strength"))
    except (TypeError, ValueError) as e:
        raise HTTPException(400, f"保存できませんでした: {e}")


@app.post("/api/loras/civitai")
def lora_civitai_lookup(body: dict):
    """Look the LoRA up on Civitai by its sha256 (only the hash is sent) and record the trigger words."""
    p = _library_lora(body.get("path"))
    try:
        return loraconf.civitai_lookup(p, body.get("civitai_token") or os.environ.get("CIVITAI_TOKEN"))
    except remote.NeedToken:
        raise HTTPException(401, "Civitai に問い合わせるには API キーが必要でした。")
    except OSError as e:
        raise HTTPException(502, f"Civitai に接続できませんでした（{e}）。ネットへの接続を確認してください。")


@app.post("/api/loras/settings/reset")
def lora_settings_reset(body: dict):
    return loraconf.reset(_library_lora(body.get("path")))


@app.post("/api/library/roots")
def library_add_root(body: dict):
    try:
        added = library.add_root((body.get("path") or "").strip().strip('"'))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {**get_library(), "added": added}


@app.post("/api/library/roots/remove")
def library_remove_root(body: dict):
    library.remove_root(body.get("path") or "")
    return get_library()


@app.get("/api/catalog")
def get_catalog():
    gpus = gpus_cached()
    from .gpu import system_ram_gb
    return {"items": addons.catalog_view(), "imports": remote.state(), "models_dir": str(config.MODELS_DIR),
            "vram_gb": round(gpus[0]["vram_total_mb"] / 1024) if gpus else None, "ram_gb": system_ram_gb()}


@app.post("/api/downloads/{did}")
def start_download(did: str, body: dict | None = None):
    token = (body or {}).get("token") or os.environ.get("HF_TOKEN")
    try:
        return addons.start(did, token, (body or {}).get("choices"))
    except KeyError:
        raise HTTPException(404)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/downloads/{did}/cancel")
def cancel_download(did: str):
    return {"ok": addons.cancel(did)}


@app.post("/api/import/download/{key}/cancel")
def cancel_import_download(key: str):
    return {"ok": remote.cancel(key)}


@app.get("/api/import/state")
def import_state():
    """Downloads from links (the LoRA box on the generation pages follows its own download here)."""
    return {"imports": remote.state()}


@app.post("/api/import/resolve")
def import_resolve(body: dict):
    """Files behind a Hugging Face / Civitai link, each checked from its header before downloading."""
    try:
        return remote.resolve(body.get("link") or "", body.get("hf_token") or os.environ.get("HF_TOKEN"),
                              body.get("civitai_token") or os.environ.get("CIVITAI_TOKEN"))
    except remote.NeedToken as e:
        site = str(e)
        return JSONResponse({"need_token": site, "detail": "このリンクの取得にはログインが必要です。" +
                             ("Hugging Face のアクセストークンを入れてください（モデルのページで利用条件への同意も必要です）。" if site == "hf"
                              else "Civitai の API キーを入れてください（Civitai の設定画面で作れます）。")}, status_code=401)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except OSError as e:
        raise HTTPException(502, f"サイトに接続できませんでした: {e}")


@app.post("/api/import/download")
def import_download(body: dict):
    try:
        return remote.start_download(body["url"], body["name"], body.get("site") or "", body.get("token"),
                                     body.get("variant_hint"), body.get("size"))
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/import/upload")
async def import_upload(request: Request, name: str):
    """Raw request body = the file (no multipart, so a multi-GB file is written once, straight to models/)."""
    try:
        st, tmp = remote.begin_upload(name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    ok, err = True, None
    from starlette.concurrency import run_in_threadpool
    try:
        with open(tmp, "wb") as f:
            buf, n = [], 0
            async for chunk in request.stream():
                buf.append(chunk)
                n += len(chunk)
                st["done_bytes"] += len(chunk)
                if n >= 8 << 20:   # written off the event loop: a slow disk must not stop the whole UI
                    await run_in_threadpool(f.write, b"".join(buf))
                    buf, n = [], 0
            if buf:
                await run_in_threadpool(f.write, b"".join(buf))
    except Exception as e:   # the browser closed the page, the disk is full, ...
        ok, err = False, str(e)
    st = await run_in_threadpool(remote.finish_upload, st, tmp, ok, err)
    if st["state"] == "failed":
        raise HTTPException(400, st["error"])
    return st


@app.post("/api/models_dir/open")
def open_models_dir(body: dict | None = None):
    """Open a models subfolder in Explorer so the user can drop files in."""
    sub = (body or {}).get("sub") or "checkpoints"
    d = config.MODELS_DIR / sub
    # the standard folders, or an old folder of this tool that still holds models (shown as "move from here")
    if sub not in ("checkpoints", "loras", "diffusion_models", "text_encoders", "vae") and \
            not any(x["old_dir"] == sub for x in library.misplaced()):
        raise HTTPException(400)
    d.mkdir(parents=True, exist_ok=True)
    before = _explorer_windows()
    os.startfile(str(d))
    threading.Thread(target=_raise_explorer, args=(d.name, before), daemon=True).start()
    return {"ok": True}


def _explorer_windows() -> dict:
    """Visible Explorer windows: hwnd -> title ("<folder> - エクスプローラー", or "<folder> とその他 N 個のタブ - ..." on
    Windows 11 when the folder opens as a tab)."""
    import ctypes
    from ctypes import wintypes
    u = ctypes.windll.user32
    out = {}

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(h, _):
        cls, txt = ctypes.create_unicode_buffer(64), ctypes.create_unicode_buffer(512)
        u.GetClassNameW(h, cls, 64)
        if cls.value == "CabinetWClass" and u.IsWindowVisible(h):
            u.GetWindowTextW(h, txt, 512)
            out[h] = txt.value
        return True

    u.EnumWindows(cb, 0)
    return out


def _raise_explorer(name: str, before: dict):
    """Explorer started from this background server opens behind the browser (Windows does not let a
    background process take the foreground).  Find the window that newly shows the folder (a new window, or an
    existing one whose active tab changed to it) and bring it to the front; a simulated Alt press is the
    commonly used way to be allowed to call SetForegroundWindow."""
    import ctypes
    u = ctypes.windll.user32
    for _ in range(30):   # the window appears within about a second
        time.sleep(0.1)
        hit = [h for h, t in _explorer_windows().items()
               if (t == name or t.startswith(name + " ")) and before.get(h) != t]
        if hit:
            u.keybd_event(0x12, 0, 0, 0)        # Alt down
            u.keybd_event(0x12, 0, 2, 0)        # Alt up
            u.ShowWindow(hit[0], 9)             # SW_RESTORE
            u.SetForegroundWindow(hit[0])
            return


# ====================================================================== projects
def _project_info(p):
    meta = p.get_meta()
    meta["id"] = p.id
    meta["history"] = p.history_state()
    meta["model"] = registry.get(meta.get("model_id") or "")
    v = VARIANTS.get((meta["model"] or {}).get("variant") or "")
    meta["caption_style"] = captions.project_style(meta) if v else None
    meta["caption_natural_ok"] = bool(v and v.get("natural_ok"))
    meta["caption_mode_default"] = captions.default_caption_mode(meta)
    meta["prompt_sets"] = prompt_sets(meta.get("kind"), captions.concept_target(meta))
    if multichar.is_multi(meta):
        _imgs, members = multichar.gather(projects, p, _used_images)
        meta["member_info"] = [{k: m.get(k) for k in ("pid", "name", "trigger", "n_used", "missing", "kind", "multi")} for m in members]
    return meta


@app.get("/api/projects")
def list_projects():
    return projects.list()


@app.post("/api/projects")
def create_project(body: dict):
    kind = body.get("kind")
    target = body.get("concept_target") or "body"
    label = {"character": "キャラ", "enhance": "質感"}.get(kind, "スタイル")
    enh = body.get("enhance") or {}
    if kind == "enhance" and enh.get("method") == "words" and enh.get("effect") == "custom":
        label = "スライダー"
    if kind == "concept":
        label = captions.CONCEPT_TARGETS.get(target, captions.CONCEPT_TARGETS["body"])["short"]
    name = (body.get("name") or "").strip() or time.strftime(label + "_%m%d_%H%M")
    try:
        source = "generate" if body.get("source") == "generate" and kind in ("character", "style") else "images"
        members = None
        if kind == "character" and body.get("source") == "multi":
            source, label = "multi", "複数キャラ"
            members = _clean_members(body.get("members"))
            if not (body.get("name") or "").strip():
                name = time.strftime(label + "_%m%d_%H%M")
        model_id = body.get("model_id")
        if members and not model_id:   # 「あとで選ぶ」: the first member's model
            model_id = projects.get(members[0]).meta("model_id")
        p = projects.create(name, kind, model_id, body.get("outfit_mode", "include"), source,
                            body.get("concept_words"), body.get("enhance"), target)
        if members is not None:
            p.set_meta(members=members, trigger="multi_" + p.meta("trigger").split("_", 1)[1])
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _project_info(p)


def _clean_members(ids, own: str | None = None) -> list[str]:
    """Member list of a multi-character project: existing character projects, each once, never the project itself."""
    out = []
    for pid in ids or []:
        if not isinstance(pid, str) or pid == own or pid in out:
            continue
        try:
            m = projects.get(pid).get_meta()
        except (KeyError, OSError):
            raise ValueError("まとめるプロジェクトが見つかりません")
        if m.get("kind") != "character" or multichar.is_multi(m):
            raise ValueError(f"「{m.get('name')}」はキャラクター LoRA のプロジェクトではありません")
        out.append(pid)
    if len(out) > multichar.MAX_MEMBERS:
        raise ValueError(f"一度にまとめられるのは {multichar.MAX_MEMBERS} キャラまでです")
    return out


def _busy_reason(pid: str) -> str | None:
    """Work that still uses the project's files (training, comparison images, material generation)."""
    jobs = ([runner.current] if runner.current else []) + list(runner.jobs)
    for j in jobs:
        if j.project.id == pid:
            what = getattr(j, "label", None) or f"学習 #{j.run_id}"
            return f"{what} が実行中または順番待ちです。止めてから操作してください。"
    return None


@app.post("/api/projects/import")
def import_project(file: UploadFile = File(...)):
    try:
        p = projectio.import_zip(projects, file.file)
    except zipfile.BadZipFile:
        raise HTTPException(400, "取り込めませんでした: zip ファイルとして読めません（壊れているか、途中までしかありません）。")
    except ValueError as e:
        raise HTTPException(400, f"取り込めませんでした: {e}")
    except OSError as e:
        raise HTTPException(400, f"取り込めませんでした（ファイルの書き込みに失敗）: {e}")
    _resume_analysis(p)
    return {**_project_info(p), "loras": _restore_exports(p)}


def _resume_analysis(p):
    """Queue again what was left unfinished (app closed meanwhile, or a project brought from a zip)."""
    for im in p.images():
        if im["fast_status"] == "pending":
            analyzer.submit_fast(p, im["id"])
        elif im["ai_status"] in ("pending", "running"):
            analyzer.submit_ai(p, im["id"])
        elif im["ai_status"] == "done" and not (im.get("ai_key") or "").startswith(ANALYZER_VERSION + ":"):
            analyzer.submit_ai(p, im["id"], priority=8)   # analyzer changed: refresh in the background
        elif im["ai_status"] == "done":
            if im["caption_status"] in ("pending", "running", "failed") and im.get("ai"):
                analyzer.ensure_caption(p, im["id"], force=True)   # a failed caption gets one more try
            analyzer.ensure_mask(p, im["id"])   # masks waiting (or missing: not in a zip) are made


def _restore_exports(p) -> dict:
    """LoRAs exported from the project before it was zipped: the exported folders are outside the project and not
    in the zip, but a zip with the training results has the saved LoRAs they came from.  Each one whose folder is
    not on this PC (another PC, or deleted) is written again into the LoRA folder, so it shows up in the LoRA list
    and in image generation.  Returns {"added": [names], "failed": [messages]}."""
    out = {"added": [], "failed": []}
    rows = p.q("SELECT * FROM exports ORDER BY id DESC")
    seen = set()
    for r in rows:
        ck = r["ckpt_path"]
        if ck in seen:
            continue
        seen.add(ck)
        same = [x for x in rows if x["ckpt_path"] == ck]
        if any(Path(x["dest"]).exists() for x in same):
            continue   # still here (imported on the PC it was exported on)
        if not Path(ck).exists() or not any(c["path"] == ck for c in p.checkpoints(r["run_id"])):
            continue   # a zip without the training results
        info = json.loads(r["info"] or "{}")
        try:
            res = do_export(p, r["run_id"], ck, info.get("tested_strength"), None, info.get("evaluation") or None)
        except Exception as e:
            out["failed"].append(f"{Path(r['dest']).name}: {e}")
            continue
        with p.tx() as db:   # the old rows point at folders that do not exist here
            db.executemany("DELETE FROM exports WHERE id=?", [(x["id"],) for x in same])
        out["added"].append(Path(res["dir"]).name)
    return out


@app.get("/api/projects/{pid}/export_size")
def project_export_size(pid: str):
    p = P(pid)
    return {"base": projectio.export_size(p, False), "with_runs": projectio.export_size(p, True), "busy": _busy_reason(pid)}


@app.get("/api/projects/{pid}/export.zip")
def project_export(pid: str, runs: int = 0):
    p = P(pid)
    if runs and (why := _busy_reason(pid)):
        raise HTTPException(400, why)
    from urllib.parse import quote
    fname = f"{p.meta('name') or pid}_{time.strftime('%Y%m%d')}{'_学習結果込み' if runs else ''}.zip"
    fname = "".join("_" if c in '\\/:*?"<>|' else c for c in fname)
    return StreamingResponse(projectio.export_stream(p, bool(runs)), media_type="application/zip",
                             headers={"Content-Disposition": f"attachment; filename=\"project.zip\"; filename*=UTF-8''{quote(fname)}"})


@app.delete("/api/projects/{pid}")
def delete_project(pid: str):
    P(pid)
    if why := _busy_reason(pid):
        raise HTTPException(400, why)
    analyzer.forget_project(pid)
    try:
        projectio.delete_project(projects, pid)
    except KeyError:
        raise HTTPException(404, "プロジェクトが見つかりません")
    except OSError as e:
        raise HTTPException(400, f"削除できませんでした: {e}")
    return {"deleted": pid}


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    return _project_info(P(pid))


@app.patch("/api/projects/{pid}")
def update_project(pid: str, body: dict):
    p = P(pid)
    meta = p.get_meta()
    allowed = {"name", "trigger", "outfit_mode", "model_id", "reference_ids", "train_overrides", "concept_words", "enhance",
               "caption_mode", "mask"} | ({"members"} if multichar.is_multi(meta) else set())
    upd = {k: v for k, v in body.items() if k in allowed}
    if "members" in upd:
        try:
            upd["members"] = _clean_members(upd["members"], pid)
        except ValueError as e:
            raise HTTPException(400, str(e))
    if "mask" in upd:
        try:
            m = upd["mask"] or {}
            upd["mask"] = {"on": bool(m.get("on")), "floor": float(m.get("floor", masks.DEFAULT_FLOOR))}
        except (TypeError, ValueError, AttributeError):
            raise HTTPException(400, "マスク学習の設定が正しくありません")
        if not 0 <= upd["mask"]["floor"] <= 1:
            raise HTTPException(400, "マスクの外の効き方は 0 から 1 の間で指定してください")
    if "caption_mode" in upd and upd["caption_mode"] not in captions.CAPTION_MODES:
        raise HTTPException(400, "説明文の作り方の指定が正しくありません")
    if "enhance" in upd:
        if meta.get("kind") != "enhance":
            raise HTTPException(400, "質感・補強 LoRA のプロジェクトではありません")
        from .enhance import clean
        new = clean({**(meta.get("enhance") or {}), **(upd["enhance"] or {})})
        if new["method"] != (meta.get("enhance") or {}).get("method"):
            upd["train_overrides"] = {}   # the two methods have different settings
        upd["enhance"] = new
    if "concept_words" in upd:
        upd["concept_words"] = captions.clean_concept_words(upd["concept_words"])
    if "trigger" in upd:
        t = (upd["trigger"] or "").strip()
        if not t:
            raise HTTPException(400, "呼び出し語を入力してください")
        if "," in t:
            raise HTTPException(400, "呼び出し語にカンマは使えません")
        upd["trigger"] = t
    reset = []
    if "model_id" in upd and "train_overrides" not in upd:
        # another family: its own engine values (learning rate, rank ...) start from that family's defaults
        old_m, new_m = registry.get(meta.get("model_id") or ""), registry.get(upd["model_id"] or "")
        ov = meta.get("train_overrides") or {}
        if old_m and new_m and old_m["family"] != new_m["family"] and presets.FAMILY_KEYS & set(ov):
            reset = [presets.FIELDS[k][0] for k in ov if k in presets.FAMILY_KEYS and k in presets.FIELDS]
            upd["train_overrides"] = {k: v for k, v in ov.items() if k not in presets.FAMILY_KEYS}
    if upd.get("train_overrides") == {}:
        upd["settings_import"] = None   # "すべて初期値に戻す" also clears the note about copied settings
    p.set_meta(**upd)
    if "trigger" in upd and upd["trigger"] != meta.get("trigger"):
        # auto captions are regenerated by policy; edited captions get a plain word replacement (undoable)
        old, new = meta["trigger"], upd["trigger"]
        changes = [(im["id"], "caption", captions.replace_trigger(im["caption"], old, new))
                   for im in p.images() if im["caption_edited"] and old in im["caption"]]
        if changes:
            p.edit(f"呼び出し語を {new} に変更", changes)
    _after_meta(p, meta, upd.keys())
    return {**_project_info(p), "reset_settings": reset}


def _after_meta(p, before: dict, keys):
    """Captions, identity and masks that follow a change of these project settings (before: the old meta)."""
    if {"trigger", "outfit_mode", "model_id", "concept_words", "caption_mode"} & keys:
        analyzer.carry_trigger(p, before)
        for im in p.images():
            analyzer.ensure_caption(p, im["id"])
    if "reference_ids" in keys:
        threading.Thread(target=analyzer.compute_identity, args=(p,), daemon=True).start()
    if {"mask", "model_id", "members"} & keys:
        threading.Thread(target=_queue_masks, args=(p,), daemon=True).start()


def _queue_masks(p):
    """Masks for every image the project's runs can use (the members' images for a multi-character project)."""
    meta = p.get_meta()
    mode = masks.active(meta, (registry.get(meta.get("model_id") or "") or {}).get("family"))
    if not mode:
        return
    owners = []
    if multichar.is_multi(meta):
        for mid in multichar.member_ids(meta):
            try:
                owners.append(projects.get(mid))
            except (KeyError, OSError):
                pass
    else:
        owners = [p]
    for o in owners:
        for im in o.images():
            analyzer.ensure_mask(o, im["id"], mode)


# ====================================================================== images
def _ctx(p, imgs, meta):
    ident = {int(k): v for k, v in ((meta.get("identity") or {}).get("map") or {}).items()}
    meta = dict(meta)
    meta["reference_ids"] = meta.get("reference_ids") or meta.get("auto_reference") or []
    return judge.build_context(imgs, meta, ident)


def _light(im: dict, res: dict) -> dict:
    ai = im.get("ai") or {}
    general, chars = captions.tags_from_ai(ai) if ai else ([], [])
    f = im.get("fast") or {}
    return {"id": im["id"], "name": im["orig_name"], "w": im["width"], "h": im["height"], "crop": im["crop"],
            "decision": im["decision"], "caption": im["caption"], "caption_edited": bool(im["caption_edited"]),
            "caption_status": im["caption_status"], "caption_auto": im.get("caption_auto"),
            "fast_status": im["fast_status"], "ai_status": im["ai_status"], "ai_error": im.get("ai_error"),
            "verdict": res["verdict"], "verdict_label": judge.VERDICT_LABEL[res["verdict"]], "use": res["use"],
            "checks": res["checks"], "domain": ai.get("domain"), "tags": general[:60], "chars": chars,
            "subject": ai.get("subject"), "persons": ai.get("persons"), "file_size": f.get("file_size"),
            "jpeg_quality": f.get("jpeg_quality"), "missing": bool(f.get("missing")),
            "mask": im.get("mask"), "mask_off": bool(im.get("mask_off"))}


@app.get("/api/projects/{pid}/images")
def list_images(pid: str):
    p = P(pid)
    meta = p.get_meta()
    imgs = p.images()
    ctx = _ctx(p, imgs, meta)
    results = {im["id"]: judge.check_image(im, ctx) for im in imgs}
    report = judge.dataset_report(imgs, results, meta)
    # MOSS-Image character LoRA: is the character in hide1's catalog (drawn by name without a LoRA)?  Information only
    report["moss_catalog"] = moss_catalog.character_hint(imgs, results, meta, (registry.get(meta.get("model_id") or "") or {}).get("family"))
    return {"images": [_light(im, results[im["id"]]) for im in imgs], "report": report,
            "history": p.history_state(), "reference_ids": ctx["reference_ids"]}


def _ingest(p, src_iter, after_add=None, failed=None):
    """src_iter yields (filename, origin_path|None, reader) ; reader(fh) writes bytes.
    after_add(image_id, filename) runs before analysis is queued (e.g. to attach generation info).
    A file that cannot be read (locked, no permission, disk error) is skipped and listed in `failed`."""
    if multichar.is_multi(p.get_meta()):
        raise HTTPException(400, "複数キャラ LoRA には画像を直接入れられません。キャラごとのプロジェクトに入れてください。")
    ids = []
    incoming = p.path / "incoming"
    incoming.mkdir(exist_ok=True)
    for name, origin, write in src_iter:
        if Path(name).suffix.lower() not in IMAGE_EXT:
            continue
        tmp = incoming / (uuid.uuid4().hex + ".part")
        try:
            h = hashlib.sha256()
            with open(tmp, "wb") as f:
                write(f, h)
            sha = h.hexdigest()
            ext = Path(name).suffix.lower()
            dest = p.sources / f"{sha[:20]}{ext}"
            if not dest.exists():
                os.replace(tmp, dest)
            iid, new = p.add_image_file(dest, name, sha, origin)
        except Exception as e:
            if failed is None:
                raise
            failed.append(f"{name}: {e}")
            continue
        finally:
            tmp.unlink(missing_ok=True)
        ids.append(iid)
        if new:
            p.set_fields(iid, caption_status="pending")
        if after_add:
            after_add(iid, name)
        analyzer.submit_fast(p, iid)
    return ids


@app.post("/api/projects/{pid}/upload")
def upload(pid: str, files: list[UploadFile] = File(...)):
    p = P(pid)

    def it():
        for uf in files:
            def write(f, h, uf=uf):
                while chunk := uf.file.read(1 << 20):
                    h.update(chunk)
                    f.write(chunk)
            yield uf.filename or "image", None, write

    failed = []
    ids = _ingest(p, it(), failed=failed)
    return {"ids": ids, "failed": failed}


@app.post("/api/projects/{pid}/import_folder")
def import_folder(pid: str, body: dict):
    p = P(pid)
    folder = Path((body.get("path") or "").strip().strip('"'))
    if not folder.is_dir():
        raise HTTPException(400, f"フォルダーが見つかりません: {folder}")
    recursive = bool(body.get("recursive"))
    files = sorted((folder.rglob("*") if recursive else folder.iterdir()))
    files = [f for f in files if f.is_file() and f.suffix.lower() in IMAGE_EXT]

    def it():
        for fp in files:
            def write(f, h, fp=fp):
                with open(fp, "rb") as src:
                    while chunk := src.read(1 << 20):
                        h.update(chunk)
                        f.write(chunk)
            yield fp.name, str(fp), write

    def run():
        failed = []
        try:
            ids = _ingest(p, it(), failed=failed)
        except Exception as e:   # e.g. the project folder itself became unwritable
            ids, failed = [], failed + [str(e)]
        bus.publish("import", project=p.id, added=len(ids), failed=failed)

    threading.Thread(target=run, daemon=True).start()
    return {"queued": len(files)}


@app.get("/api/projects/{pid}/thumb/{iid}")
def thumb(pid: str, iid: int):
    p = P(pid)
    f = p.thumbs / f"{iid}.webp"
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f, media_type="image/webp", headers={"Cache-Control": "max-age=31536000, immutable"})


@app.get("/api/projects/{pid}/preview/{iid}")
def preview(pid: str, iid: int):
    """Detail-view image: <=1600px cached rendition (full-size originals are only read once)."""
    p = P(pid)
    im = p.image(iid)
    if not im:
        raise HTTPException(404)
    cache = p.thumbs / f"{iid}_1600.webp"
    src = p.source_path(im)
    if not cache.exists():
        if not src.exists():
            raise HTTPException(404, "元画像が見つかりません")
        with Image.open(src) as x:
            x.load()
            x = ImageOps.exif_transpose(x).convert("RGB")
            x.thumbnail((1600, 1600))
            tmp = cache.with_name(f"{cache.name}.{uuid.uuid4().hex[:8]}.tmp")   # never serve a half-written file
            x.save(tmp, "WEBP", quality=88)
            os.replace(tmp, cache)
    return FileResponse(cache, media_type="image/webp", headers={"Cache-Control": "max-age=3600"})


@app.get("/api/projects/{pid}/images/{iid}")
def image_detail(pid: str, iid: int):
    p = P(pid)
    meta = p.get_meta()
    imgs = p.images()
    ctx = _ctx(p, imgs, meta)
    im = next((i for i in imgs if i["id"] == iid), None)
    if not im:
        raise HTTPException(404)
    analyzer.prioritize(p, [iid])
    d = _light(im, judge.check_image(im, ctx))
    ai = im.get("ai") or {}
    d["tags_scored"] = {k: v for k, v in sorted((ai.get("tags") or {}).get("general", {}).items(), key=lambda x: -x[1])[:80]}
    d["fast"] = im.get("fast")
    d["origin_path"] = im.get("origin_path")
    d["suggested_crop"] = suggest_crop(im)
    mode = analyzer.mask_mode(p)   # also a member's image of a multi-character project trained with masks
    if mode:
        d["mask"] = {"mode": mode, "off": bool(im.get("mask_off")), **masks.state(im, mode)}
    return d


@app.get("/api/projects/{pid}/images/{iid}/mask_preview")
def mask_preview(pid: str, iid: int):
    """The image with the part outside the mask darkened (what the training counts mainly stays bright)."""
    import io
    p = P(pid)
    im = p.image(iid)
    mode = analyzer.mask_mode(p)
    f = masks.image_mask_file(im, mode) if im and mode else None
    if not f or not f.exists():
        raise HTTPException(404, "マスクがまだありません")
    with Image.open(p.source_path(im)) as img:
        img.load()
        img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((1024, 1024))
    with Image.open(f) as m:
        m = m.convert("L").resize(img.size, Image.BILINEAR)
    floor = masks.setting(p.get_meta())["floor"]
    lo = int(round(60 + 120 * floor))   # darkened, but the picture stays readable
    dark = img.point(lambda v: v * lo // 255)
    out = Image.composite(img, dark, m)
    buf = io.BytesIO()
    out.save(buf, "WEBP", quality=85)
    return StreamingResponse(io.BytesIO(buf.getvalue()), media_type="image/webp", headers={"Cache-Control": "no-store"})


@app.post("/api/projects/{pid}/images/{iid}/mask_redo")
def mask_redo(pid: str, iid: int):
    p = P(pid)
    mode = analyzer.mask_mode(p)
    if not mode or not p.image(iid):
        raise HTTPException(400, "このプロジェクトではマスク学習を使っていません")
    analyzer.ensure_mask(p, iid, mode, force=True)
    return {"ok": True}


def suggest_crop(im):
    ai = im.get("ai") or {}
    subj = ai.get("subject")
    if not subj or im.get("crop") or subj.get("area_ratio", 1) > 0.45:
        return None
    w, h = im["width"], im["height"]
    x0, y0, x1, y1 = subj["bbox"]
    bw, bh = x1 - x0, y1 - y0
    m = 0.18
    x0, x1 = max(0, x0 - bw * m), min(w, x1 + bw * m)
    y0, y1 = max(0, y0 - bh * m), min(h, y1 + bh * m * 0.6)
    # keep at least 768px short side when the source allows it
    need = 768
    if min(x1 - x0, y1 - y0) < need:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half_w, half_h = max((x1 - x0) / 2, min(need, w) / 2), max((y1 - y0) / 2, min(need, h) / 2)
        x0, x1 = max(0, cx - half_w), min(w, cx + half_w)
        y0, y1 = max(0, cy - half_h), min(h, cy + half_h)
    return [int(x0), int(y0), int(x1), int(y1)]


@app.post("/api/projects/{pid}/edit")
def edit(pid: str, body: dict):
    p = P(pid)
    changes = []
    crop_changed = []
    for c in body.get("changes", []):
        field, value = c["field"], c.get("value")
        if field == "crop" and value is not None:
            im = p.image(c["id"])
            if im is None:
                raise HTTPException(404, "画像が見つかりません")
            x0, y0, x1, y1 = [int(round(v)) for v in value]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(im["width"], x1), min(im["height"], y1)
            if x1 - x0 < 64 or y1 - y0 < 64:
                raise HTTPException(400, "切り抜き範囲が小さすぎます")
            value = [x0, y0, x1, y1]
        changes.append((c["id"], field, value))
        if field == "caption":
            changes.append((c["id"], "caption_edited", 1))
        if field == "crop":
            crop_changed.append(c["id"])
    try:
        hid = p.edit(body.get("label") or "編集", changes)
    except ValueError as e:
        raise HTTPException(400, str(e))
    for iid in crop_changed:
        analyzer.submit_ai(p, iid, priority=2)
    return {"history_id": hid, "history": p.history_state()}


@app.post("/api/projects/{pid}/bulk_caption")
def bulk_caption(pid: str, body: dict):
    p = P(pid)
    op = body.get("op")
    if op not in ("add", "remove", "replace", "replace_tag", "reset_auto"):
        raise HTTPException(400, "unknown op")
    ids = body.get("ids") or []
    changes = []
    tags = [t.strip() for t in (body.get("tags") or "").split(",") if t.strip()]
    for iid in ids:
        im = p.image(iid)
        if not im:
            continue
        cap = im["caption"]
        if op == "add":
            new = captions.add_tags(cap, tags, body.get("position", "end"))
        elif op == "remove":
            new = captions.remove_tags(cap, tags)
        elif op == "replace_tag":
            new = captions.replace_tag(cap, body.get("find") or "", body.get("replace") or "")
        elif op == "replace":
            find = body.get("find") or ""
            new = cap.replace(find, body.get("replace") or "") if find else cap
        elif op == "reset_auto":
            if im.get("caption_auto") is None:
                continue
            changes += [(iid, "caption", im["caption_auto"]), (iid, "caption_edited", 0)]
            continue
        else:
            raise HTTPException(400, "unknown op")
        if new != cap:
            changes += [(iid, "caption", new), (iid, "caption_edited", 1)]
    label = {"add": "タグを追加", "remove": "タグを削除", "replace": "文字列を置換", "replace_tag": "タグを置き換え",
             "reset_auto": "自動説明文に戻す"}[op]
    hid = p.edit(f"{label}（{len(ids)}枚）", changes)
    return {"changed": len({c[0] for c in changes}), "history_id": hid, "history": p.history_state()}


@app.post("/api/projects/{pid}/undo")
def undo(pid: str):
    p = P(pid)
    r = p.undo()
    for iid in (r or {}).get("analyze_ids", []):
        analyzer.submit_ai(p, iid)  # crop may have changed back; cached results are reused
    return {"undone": r, "history": p.history_state()}


@app.post("/api/projects/{pid}/redo")
def redo(pid: str):
    p = P(pid)
    r = p.redo()
    for iid in (r or {}).get("analyze_ids", []):
        analyzer.submit_ai(p, iid)
    return {"redone": r, "history": p.history_state()}


@app.post("/api/projects/{pid}/reanalyze")
def reanalyze(pid: str, body: dict):
    p = P(pid)
    ids = body.get("ids") or [im["id"] for im in p.images()]
    for iid in ids:
        im = p.image(iid)
        if im and im["fast_status"] != "done":
            analyzer.submit_fast(p, iid)
        else:
            analyzer.submit_ai(p, iid, priority=3, force=bool(body.get("force")))
            if body.get("captions"):
                analyzer.ensure_caption(p, iid, force=True)
    return {"queued": len(ids)}


@app.post("/api/projects/{pid}/auto_crop")
def auto_crop(pid: str, body: dict):
    p = P(pid)
    changes = []
    for iid in body.get("ids") or []:
        im = p.image(iid)
        sc = suggest_crop(im) if im else None
        if sc:
            changes.append((iid, "crop", sc))
    p.edit(f"自動切り抜き（{len(changes)}枚）", changes)
    for iid, _, _ in changes:
        analyzer.submit_ai(p, iid, priority=2)
    return {"changed": len(changes), "history": p.history_state()}


@app.post("/api/projects/{pid}/relink")
def relink(pid: str, body: dict):
    """Restore missing source files from a folder the user points to (that folder only)."""
    p = P(pid)
    folder = Path((body.get("path") or "").strip().strip('"'))
    if not folder.is_dir():
        raise HTTPException(400, "フォルダーが見つかりません")
    missing: dict[str, list] = {}   # the same file may be in the project twice (other names)
    for im in p.images():
        if not p.source_path(im).exists():
            missing.setdefault(im["sha"], []).append(im)
    # files of another size cannot match: they are not read (sizes are known for images checked before)
    sizes = {(im.get("fast") or {}).get("file_size") for ims in missing.values() for im in ims}
    fixed = 0
    for fp in folder.iterdir():
        if not missing or not fp.is_file() or fp.suffix.lower() not in IMAGE_EXT:
            continue
        if None not in sizes and fp.stat().st_size not in sizes:
            continue
        h = hashlib.sha256()
        with open(fp, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        for im in missing.pop(h.hexdigest(), []):
            import shutil
            shutil.copyfile(fp, p.source_path(im))
            analyzer.submit_fast(p, im["id"])
            fixed += 1
    return {"fixed": fixed, "still_missing": sum(len(v) for v in missing.values())}


# ====================================================================== material generation
def _gen_state(p):
    recipe = p.meta("gen_recipe") or {}
    batches = []
    for b in p.q("SELECT * FROM gen_batches ORDER BY id DESC"):
        spec = json.loads(b["spec"])
        rows = p.q("SELECT * FROM gen_images WHERE batch_id=? ORDER BY idx", (b["id"],))
        imgs = [{"id": r["id"], "idx": r["idx"], "file": r["file"], "seed": r["seed"], "picked": bool(r["picked"]),
                 "ccip_diff": r["ccip_diff"], "differs": r["ccip_diff"] is not None and r["ccip_diff"] > generate.SAME_CHAR,
                 "mismatch": promptkeep.reason(json.loads(r["prompt_check"])) if r["prompt_check"] else "",
                 "image_id": r["image_id"], "label": generate.shot_label(json.loads(r["shot"])), "prompt": r["prompt"]}
                for r in rows if not r["deleted"]]
        d = generate.batch_dir(p, b["id"])
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) if d.exists() else 0
        batches.append({"id": b["id"], "created": b["created"], "status": b["status"], "message": b["message"],
                        "total": len(spec["shots"]), "remaining": len(spec["shots"]) - len(rows),
                        "gen_model_name": spec.get("gen_model_name"), "images": imgs, "size_mb": round(size / 2**20, 1)})
    model = registry.get(recipe.get("gen_model_id") or "")
    chk = generate.check_recipe(recipe, p.meta("kind"), p)
    return {"recipe": recipe, "batches": batches, "default_sampling": generate.default_sampling(model, _fast(recipe)),
            "angles": generate.ANGLE_LABEL, "framings": generate.FRAMING_LABEL, "exprs": generate.EXPR_LABEL,
            "poses": generate.POSE_LABEL,
            "groups": generate.STYLE_GROUP_LABEL, "check": chk["errors"], "notes": chk["notes"]}


@app.get("/api/projects/{pid}/gen")
def gen_state(pid: str):
    return _gen_state(P(pid))


@app.put("/api/projects/{pid}/gen/recipe")
def gen_recipe(pid: str, body: dict):
    p = P(pid)
    keep = ("gen_model_id", "loras", "positive", "negative", "count", "sampling", "seed", "exclude_angles", "exclude_framings",
            "exclude_exprs", "exclude_poses", "exclude_groups", "init_images", "denoise")
    recipe = {k: body.get(k) for k in keep if k in body}
    if "init_images" in recipe:
        recipe["init_images"] = [Path(str(n)).name for n in recipe["init_images"] or []][:generate.MAX_SOURCES]
        _prune_gen_sources(p, recipe["init_images"])
    before = (p.meta("gen_recipe") or {}).get("gen_model_id")
    p.set_meta(gen_recipe=recipe)
    # the LoRA is trained on the generation model unless the user chose another one: the training
    # model follows the generation model while it is unset or still equal to the previous one
    cur = p.meta("model_id")
    if recipe.get("gen_model_id") and registry.get(recipe["gen_model_id"]) and (not cur or cur == before) \
            and recipe["gen_model_id"] != cur:
        old = p.get_meta()
        p.set_meta(model_id=recipe["gen_model_id"])
        _after_meta(p, old, {"model_id"})   # caption style (tags / sentences) and masks follow the model
    chk = generate.check_recipe(recipe, p.meta("kind"), p)
    return {"check": chk["errors"], "notes": chk["notes"], "project": _project_info(p),
            "default_sampling": generate.default_sampling(registry.get(recipe.get("gen_model_id") or ""), _fast(recipe))}


def _fast(form: dict) -> str | None:
    """The form's choice of the 8-step generation LoRA (MOSS-Image SFT): "on" / "off" / unset."""
    return (form.get("sampling") or {}).get("fast")


def _queue_batch(p, bid):
    def cancelled():
        generate.control.flags.pop((p.id, bid), None)   # a pause / stop asked while queued
        generate._set_status(p, bid, "cancelled", "開始前に取り消しました")
    runner.enqueue_task(p, f"素材生成 #{bid}", lambda: generate.run_batch(p, bid), on_cancel=cancelled)


def _prune_gen_sources(p, keep: list[str]):
    """Source images the recipe no longer lists (each batch's spec keeps the names it used: those stay too)."""
    d = generate.gen_source_dir(p)
    if not d.is_dir():
        return
    used = set(keep)
    for b in p.q("SELECT spec FROM gen_batches"):
        used.update(json.loads(b["spec"]).get("init_images") or [])
    for f in d.glob("*.png"):   # a file just added may not be in the saved recipe yet
        if f.name not in used and time.time() - f.stat().st_mtime > 600:
            f.unlink(missing_ok=True)


@app.post("/api/projects/{pid}/gen/source")
async def gen_source(pid: str, request: Request):
    """A source image for image-to-image material generation (the raw file as the body)."""
    from starlette.concurrency import run_in_threadpool
    p = P(pid)
    data = await request.body()
    if not data:
        raise HTTPException(400, "画像が空です")
    if len(data) > 100 * 2**20:
        raise HTTPException(400, "画像が大きすぎます（100MB まで）")
    try:
        return await run_in_threadpool(generate.save_source_image, data, generate.gen_source_dir(p))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/projects/{pid}/gen/source/{fname}")
def gen_source_image(pid: str, fname: str):
    f = generate.source_file(generate.gen_source_dir(P(pid)), fname)
    if not f:
        raise HTTPException(404)
    return FileResponse(f, media_type="image/png", headers={"Cache-Control": "max-age=86400"})


@app.post("/api/projects/{pid}/gen/start")
def gen_start(pid: str, body: dict | None = None):
    p = P(pid)
    recipe = p.meta("gen_recipe") or {}
    errs = generate.check_recipe(recipe, p.meta("kind"), p)["errors"]
    if errs:
        raise HTTPException(400, " / ".join(errs))
    count = int((body or {}).get("count") or recipe.get("count") or 48)
    if not 1 <= count <= 200:
        raise HTTPException(400, "枚数は 1〜200 枚で指定してください")
    try:
        bid = generate.new_batch(p, recipe, count)
    except ValueError as e:
        raise HTTPException(400, str(e))
    _queue_batch(p, bid)
    bus.publish("gen", project=p.id, batch=bid, status="queued", message="順番待ち")
    return {"batch_id": bid}


@app.post("/api/projects/{pid}/gen/consistency")
def gen_consistency(pid: str):
    """Check how alike the generated candidates are (queued on the GPU queue like generation)."""
    p = P(pid)

    def task():
        try:
            msg = generate.consistency_message(generate.check_consistency(p)) or "顔立ちを比べる候補が足りません（3 枚以上必要です）"
        except Exception as e:
            msg = f"顔立ちのそろい具合を確認できませんでした（{e}）"
        try:   # candidates made before 2026-11-02 have not been checked against the prompt yet
            kept = generate.prompt_check_message(generate.check_prompt(p))
        except Exception as e:
            kept = f"プロンプトの指定どおりかを確認できませんでした（{e}）"
        msg = "。".join(x for x in (msg, kept) if x)
        bus.publish("gen", project=p.id, batch=0, status="consistency", message=msg)
    runner.enqueue_task(p, "生成候補のそろい具合の確認", task, on_cancel=lambda: None)
    return {"ok": True}


@app.post("/api/projects/{pid}/gen/pick")
def gen_pick(pid: str, body: dict):
    """Which generated images are checked for import (kept across restarts)."""
    p = P(pid)
    with p.tx() as db:
        for i in body.get("ids") or []:
            db.execute("UPDATE gen_images SET picked=?, pick_manual=1 WHERE id=?", (1 if body.get("picked") else 0, int(i)))
    return {"ok": True}


@app.post("/api/projects/{pid}/gen/import")
def gen_import(pid: str, body: dict):
    """Add the picked generated images to the material.  They are copied like any other material;
    the shot they were generated with is attached so the caption can name it exactly."""
    return {"added": len(_import_generated(P(pid), body.get("ids") or []))}


def _import_generated(p, ids) -> list:
    rows = []
    for i in ids:
        r = p.q("SELECT * FROM gen_images WHERE id=? AND image_id IS NULL", (int(i),))
        if r:
            rows.append(r[0])
    by_name = {}

    def it():
        for r in rows:
            src = generate.batch_dir(p, r["batch_id"]) / r["file"]
            if not src.exists():
                continue
            name = f"生成{r['batch_id']:03d}_{r['file']}"
            by_name[name] = r

            def write(f, h, src=src):
                with open(src, "rb") as fh:
                    while chunk := fh.read(1 << 20):
                        h.update(chunk)
                        f.write(chunk)
            yield name, str(src), write

    def attach(iid, name):
        r = by_name[name]
        p.set_fields(iid, gen={"batch": r["batch_id"], "idx": r["idx"], "seed": r["seed"], "prompt": r["prompt"],
                               "shot": json.loads(r["shot"]), "shot_set": generate.STYLE_SET_VERSION if "subject" in json.loads(r["shot"]) else generate.SHOT_SET_VERSION})
        p.q("UPDATE gen_images SET image_id=? WHERE id=?", (iid, r["id"]))

    return _ingest(p, it(), attach)


@app.post("/api/projects/{pid}/gen/{bid}/files/delete")
def gen_delete(pid: str, bid: int, body: dict):
    """Free disk space.  scope "unpicked": the images not picked and not added to the material.
    scope "all": every generated file of the batch and the batch itself.  Material already added is
    a separate copy in the project and is never touched."""
    p = P(pid)
    rows = p.q("SELECT status FROM gen_batches WHERE id=?", (bid,))
    if not rows:
        raise HTTPException(404)
    if rows[0]["status"] in ("queued", "running"):
        raise HTTPException(400, "生成中は削除できません。一時停止か中止をしてから削除してください。")
    d = generate.batch_dir(p, bid)
    scope = body.get("scope")
    if scope == "unpicked":
        targets = p.q("SELECT id, file FROM gen_images WHERE batch_id=? AND picked=0 AND image_id IS NULL AND deleted=0", (bid,))
    elif scope == "all":
        targets = p.q("SELECT id, file FROM gen_images WHERE batch_id=? AND deleted=0", (bid,))
    else:
        raise HTTPException(400, "unknown scope")
    freed = 0
    for r in targets:
        for f in (d / r["file"], d / "thumbs" / (Path(r["file"]).stem + ".webp")):
            if f.exists():
                freed += f.stat().st_size
                f.unlink()
    with p.tx() as db:
        if scope == "all":
            db.execute("DELETE FROM gen_images WHERE batch_id=?", (bid,))
            db.execute("DELETE FROM gen_batches WHERE id=?", (bid,))
        else:
            db.executemany("UPDATE gen_images SET deleted=1, picked=0 WHERE id=?", [(r["id"],) for r in targets])
    if scope == "all" and d.exists():
        import shutil
        shutil.rmtree(d, ignore_errors=True)
    bus.publish("gen", project=p.id, batch=bid, status="deleted")
    return {"deleted": len(targets), "freed_mb": round(freed / 2**20, 1)}


@app.post("/api/projects/{pid}/gen/{bid}/{action}")
def gen_control(pid: str, bid: int, action: str):
    """pause: stop after the current image, resumable.  stop: interrupt now.  resume: continue."""
    p = P(pid)
    rows = p.q("SELECT status FROM gen_batches WHERE id=?", (bid,))
    if not rows:
        raise HTTPException(404)
    st = rows[0]["status"]
    if action in ("pause", "stop"):
        if action == "stop" and st == "paused":
            generate._set_status(p, bid, "cancelled", "中止しました（生成済みの画像は残っています）")
            return {"message": "中止しました"}
        if st not in ("queued", "running"):
            raise HTTPException(400, "実行中の生成ではありません")
        if st == "queued":
            # not started: a flag is enough (run_batch checks it first).  request("stop") would send ComfyUI's
            # global /interrupt and fail whatever else is generating now (a comparison, the image page)
            generate.control.flags[(p.id, bid)] = action
        else:
            generate.control.request(p.id, bid, action)
        msg = "一時停止を受け付けました。今の 1 枚が終わったら止まります。" if action == "pause" else "中止を受け付けました。"
        p.q("UPDATE gen_batches SET message=? WHERE id=?", (msg, bid))
        bus.publish("gen", project=p.id, batch=bid, status=st, message=msg)
        return {"message": msg}
    if action == "resume":
        if st not in ("paused", "failed", "cancelled"):
            raise HTTPException(400, "再開できる状態ではありません")
        generate.control.flags.pop((p.id, bid), None)
        generate._set_status(p, bid, "queued", "順番待ち")
        _queue_batch(p, bid)
        return {"message": "再開します"}
    raise HTTPException(404)


@app.get("/api/projects/{pid}/gen/{bid}/img/{fname}")
def gen_image(pid: str, bid: int, fname: str, thumb: int = 0):
    p = P(pid)
    f = generate.batch_dir(p, bid) / Path(fname).name
    if not f.exists():
        raise HTTPException(404)
    if not thumb:
        return FileResponse(f, media_type="image/png", headers={"Cache-Control": "max-age=86400"})
    return _thumb_response(f, f.parent / "thumbs" / (f.stem + ".webp"), 384)


def _thumb_response(f: Path, t: Path, size: int):
    """A small WebP of a generated PNG for lists and grids (the PNG itself is kept for zoom and download)."""
    if not t.exists() or t.stat().st_mtime < f.stat().st_mtime:
        t.parent.mkdir(exist_ok=True)
        with Image.open(f) as x:
            x = x.convert("RGB")
            x.thumbnail((size, size))
            tmp = t.with_name(f"{t.name}.{uuid.uuid4().hex[:8]}.tmp")
            x.save(tmp, "WEBP", quality=85)
            os.replace(tmp, t)
    return FileResponse(t, media_type="image/webp", headers={"Cache-Control": "max-age=86400"})


# ====================================================================== files shared with ComfyUI (hard links)
@app.get("/api/comfy_link")
def comfy_link_state():
    cd = linking.comfy_dir()
    return {"comfy_dir": str(cd) if cd else None, "scan": linking.scan_state(),
            "export_dir": str(cd / "models" / "loras" / linking.EXPORT_SUBDIR) if cd else None}


@app.put("/api/comfy_link")
def comfy_link_set(body: dict):
    try:
        linking.set_comfy_dir(body.get("comfy_dir"))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return comfy_link_state()


@app.post("/api/comfy_link/scan")
def comfy_link_scan():
    if not linking.comfy_dir():
        raise HTTPException(400, "先に ComfyUI のフォルダーを指定してください")
    linking.start_scan()
    return comfy_link_state()


@app.post("/api/comfy_link/apply")
def comfy_link_apply():
    """Replace the copies found by the last scan with hard links (after the user saw the list)."""
    sc = linking.scan_state()
    if sc.get("state") != "done":
        raise HTTPException(400, "先に「重複を調べる」を実行してください")
    if runner.current or runner.jobs:
        raise HTTPException(409, "学習や生成の実行中・順番待ちの間はまとめられません。終わってからもう一度押してください。")
    res = linking.link_duplicates(sc["items"])
    linking.start_scan()   # show the new state
    return res


# ====================================================================== image -> prompt (i2t, 画像からプロンプト)
def _i2t(fn):
    try:
        return fn()
    except (RuntimeError, ValueError) as e:
        raise HTTPException(400, str(e))


@app.get("/api/i2t/state")
def i2t_state():
    c = i2tlink.client
    if not c.available():
        return {"available": False, "message": "i2t を動かす環境がありません（setup.bat を実行してください）"}
    return _i2t(lambda: {"available": True, "consts": c.consts(), "config": c.call("load_config"), "job": c.job,
                         "last_done": c.last_done, "models_dir": str(i2tlink.models_dir()),
                         "default_models_dir": str(config.MODELS_DIR / "i2t")})


@app.put("/api/i2t/config")
def i2t_config(body: dict):
    """The whole i2t settings dict (i2t keeps only the keys it knows and checks their values)."""
    return _i2t(lambda: i2tlink.client.call("save_config", body))


@app.put("/api/i2t/models_dir")
def i2t_models_dir(body: dict):
    p = (body.get("path") or "").strip().strip('"')
    if p and not Path(p).is_absolute():
        raise HTTPException(400, "フォルダーのパスを、ドライブ名から入力してください")
    if i2tlink.client.job:
        raise HTTPException(409, "処理の最中は変えられません")
    config.settings().set("i2t_models_dir", p or None)
    i2tlink.client.restart()
    return {"models_dir": str(i2tlink.models_dir())}


@app.post("/api/i2t/upload")
async def i2t_upload(request: Request, name: str = "image.png"):
    from starlette.concurrency import run_in_threadpool
    data = await request.body()
    if not data:
        raise HTTPException(400, "画像が空です")
    if len(data) > 200 * 2**20:
        raise HTTPException(400, "画像が大きすぎます（200MB まで）")
    p = await run_in_threadpool(i2tlink.save_upload, name, data)
    try:
        with Image.open(p) as im:
            size = list(im.size)
    except Exception:
        p.unlink(missing_ok=True)
        raise HTTPException(400, "画像として開けません（PNG / JPEG / WebP / GIF / BMP / TIFF に対応）")
    info = await run_in_threadpool(lambda: _i2t(lambda: i2tlink.client.call("read_generation_info", str(p))))
    return {"path": str(p), "name": Path(name).name, "size": size, "geninfo": info}


@app.get("/api/i2t/file")
def i2t_file(path: str, thumb: int = 0, mosaic: int = 0):
    """mosaic: blocks across the longer side (the preview's NSFW mosaic; display only)."""
    p = i2tlink.allowed_file(path)
    if not p:
        raise HTTPException(404)
    if not thumb and not mosaic:
        return FileResponse(p)
    t = config.storage_dir("i2t") / "thumbs" / (hashlib.sha1(f"{p}|{p.stat().st_mtime_ns}|{mosaic}".encode()).hexdigest() + ".webp")
    if not t.exists():
        t.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(p) as x:
            x = ImageOps.exif_transpose(x)
            x.seek(0)
            x = x.convert("RGB")
            x.thumbnail((512, 512))
            if mosaic:
                k = max(x.size) / max(2, mosaic)
                small = x.resize((max(1, round(x.width / k)), max(1, round(x.height / k))), Image.BOX)
                x = small.resize(x.size, Image.NEAREST)
            tmp = t.with_name(f"{t.stem}.{threading.get_ident()}.tmp")
            x.save(tmp, "WEBP", quality=88)
            os.replace(tmp, t)
    return FileResponse(t, media_type="image/webp", headers={"Cache-Control": "max-age=31536000, immutable"})


@app.post("/api/i2t/use_file")
def i2t_use_file(body: dict):
    """An image LoRAtelier already has (e.g. a 画像生成 result) used on the i2t screen without uploading."""
    p = i2tlink.allowed_file(body.get("path") or "")
    if not p:
        raise HTTPException(404, "画像が見つかりません")
    with Image.open(p) as im:
        size = list(im.size)
    return {"path": str(p), "name": p.name, "size": size, "geninfo": _i2t(lambda: i2tlink.client.call("read_generation_info", str(p)))}


@app.post("/api/i2t/call")
def i2t_call(body: dict):
    """Plain i2t functions the screens use (the worker accepts only its whitelist)."""
    return {"value": _i2t(lambda: i2tlink.client.call(body["fn"], *(body.get("args") or []), timeout=float(body.get("timeout") or 120)))}


@app.get("/api/moss_catalog/styles")
def moss_catalog_styles():
    """The catalog's styles for tag completion on MOSS-Image models (empty until the list was fetched)."""
    return {"ready": moss_catalog.load() is not None, "styles": moss_catalog.styles()}


@app.post("/api/i2t/job")
def i2t_job(body: dict):
    op = body.pop("op", "")
    if op == "analyze":
        def found(path):   # the place the image is in now (it may have been moved on the settings page)
            p = i2tlink.allowed_file(path)
            if not p:
                raise HTTPException(400, f"画像が見つかりません: {Path(path).name}")
            return str(p)
        body["path"] = found(body.get("path") or "")
        if body.get("refs"):
            body["refs"] = {k: found(v) for k, v in body["refs"].items()}
    return _i2t(lambda: i2tlink.client.start_job(op, **body))


@app.get("/api/i2t/job")
def i2t_job_state():
    return {"job": i2tlink.client.job, "last_done": i2tlink.client.last_done}


@app.post("/api/i2t/{what}")
def i2t_control(what: str):
    if what not in ("cancel", "pause", "resume"):
        raise HTTPException(404)
    _i2t(lambda: i2tlink.client.control(what))
    return {"ok": True}


@app.post("/api/i2t_openpose")
def i2t_openpose(body: dict):
    """The pose editor's 「OpenPose 画像を保存...」: the PNG is returned for the browser to save."""
    out = i2tlink.DATA_DIR / "openpose.png"
    _i2t(lambda: i2tlink.client.call("save_openpose", body["points"], body["size"], str(out)))
    return FileResponse(out, media_type="image/png", filename="openpose.png")


# ====================================================================== plain image generation (画像生成)
@app.get("/api/imagegen")
def imagegen_state():
    form = imagegen.get_form()
    model = registry.get(form.get("model_id") or "")
    chk = imagegen.check(form)
    hist = imagegen.history()
    for e in hist:
        if e["status"] == "running":
            e["progress"] = imagegen.progress(e["id"])
    ds, edit = imagegen.request_sampling(model, form)
    return {"form": form, "history": hist, "default_sampling": ds, "edit": edit,
            "native": imagegen.native_size(model), "check": chk["errors"], "notes": chk["notes"],
            "presets": _presets_view(), "attention": imagegen.attention_installed(),
            "workflow": _workflow_view(form.get("workflow"))}


def _workflow_view(wid) -> dict | None:
    """The workflow's meta and its sampler's CFG (the negative-prompt note goes by it, not by the form's model)."""
    m = imagegen.comfyflow.meta(wid)
    if m:
        graph = imagegen.comfyflow.load(wid)
        m = {**m, "cfg": imagegen.comfyflow.cfg_of(graph) if graph else None,
             "negative": imagegen.comfyflow.has_negative(graph) if graph else True}
    return m


def _presets_view() -> list[dict]:
    """Saved presets with what the list page needs now: whether the model file is still there, and the
    thumbnail request's current state (順番待ち / 生成中 from the generation history)."""
    status = {e["id"]: e["status"] for e in imagegen.history()}
    out = []
    for p in imagegen.presets():
        model = registry.get(p.get("model_id") or "")
        t = dict(p.get("thumb") or {})
        if t.get("status") == "queued":
            t["status"] = "running" if status.get(t.get("entry")) == "running" else "queued"
        out.append({**p, "thumb": t, "model_missing": not model or bool(model.get("missing"))})
    return out


@app.get("/api/imagegen/presets")
def imagegen_presets():
    return {"presets": _presets_view()}


@app.post("/api/imagegen/presets")
def imagegen_preset_save(body: dict):
    try:
        preset = imagegen.save_preset(body, None if body.get("no_thumb") else runner)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"preset": preset, "presets": _presets_view()}


@app.post("/api/imagegen/presets/{pid}/thumb")
def imagegen_preset_thumb(pid: str, body: dict):
    """「サムネイルを作り直す」: one more image of the preset (a new random seed unless body has one)."""
    try:
        state = imagegen.make_preset_thumb(pid, body, runner)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return {"thumb": state, "presets": _presets_view()}


@app.get("/api/imagegen/preset_thumb/{fname}")
def imagegen_preset_thumb_file(fname: str):
    f = imagegen.PRESET_THUMBS / Path(fname).name
    if not f.is_file():
        raise HTTPException(404, "サムネイルがありません")
    return FileResponse(f, media_type="image/webp", headers={"Cache-Control": "max-age=86400"})


@app.delete("/api/imagegen/presets/{pid}")
def imagegen_preset_delete(pid: str):
    gone = imagegen.delete_preset(pid)
    if not gone:
        raise HTTPException(404, "その設定は見つかりません")
    return {"deleted": gone, "presets": _presets_view()}


@app.put("/api/imagegen/form")
def imagegen_form(body: dict):
    keep = ("model_id", "loras", "positive", "negative", "width", "height", "count", "seed", "sampling", "init_image", "denoise", "regions",
            "vae", "te", "attention", "workflow", "workflow_offer")
    form = {k: body.get(k) for k in keep if k in body}
    imagegen.save_form(form)
    model = registry.get(form.get("model_id") or "")
    chk = imagegen.check(form)
    ds, edit = imagegen.request_sampling(model, form)
    return {"check": chk["errors"], "notes": chk["notes"], "default_sampling": ds, "edit": edit,
            "native": imagegen.native_size(model)}


@app.put("/api/ui")
def ui_conf(body: dict):
    if "blur_images" in body:
        config.settings().set("blur_images", bool(body["blur_images"]))
    freed = None
    if "keep_train_cache" in body:
        keep = bool(body["keep_train_cache"])
        config.settings().set("keep_train_cache", keep)
        if not keep:   # the caches kept so far go now; the run training now keeps its own until it ends
            job = runner.current
            freed = 0
            for d in projects.folders():
                skip = tuple(r.name for r in (d / "runs").glob(f"{job.run_id:04d}_*")) if job and job.project.id == d.name else ()
                freed += traincache.clear_project(d, skip)
    return {"blur_images": config.settings().get("blur_images", False), "keep_train_cache": traincache.keep(), "freed": freed}


def _free_bytes(p: Path) -> int | None:
    for d in (p, *p.parents):   # the folder itself may not exist yet
        try:
            return shutil.disk_usage(d).free
        except OSError:
            continue
    return None   # e.g. its drive is not connected


@app.get("/api/storage")
def storage_conf():
    out = {}
    for kind in config.STORAGE:
        dirs = config.storage_dirs(kind)
        out[kind] = {"base": config.storage_base(kind), "dir": str(dirs[0]), "default": str(config.STORAGE[kind][0]),
                     "free": _free_bytes(dirs[0]),
                     "others": [{"dir": str(d), "exists": d.is_dir()} for d in dirs[1:]]}
    return out


@app.put("/api/storage")
def set_storage(body: dict):
    try:
        config.set_storage_base(body.get("kind"), body.get("base"))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return storage_conf()


@app.get("/api/storage/items")
def storage_items(kind: str):
    """What can be moved, by place: projects one by one, generated / added images by place."""
    if kind not in config.STORAGE:
        raise HTTPException(400, "unknown kind")
    places = [{"dir": str(d), "exists": d.is_dir()} for d in config.storage_dirs(kind)]
    if kind == "projects":
        out = []
        for x in projects.list():
            path = projects.path_of(x["id"])
            if path is not None:
                out.append({"id": x["id"], "name": x["name"], "kind": x.get("kind"), "updated": x.get("updated"),
                            "place": str(path.parent), "moving": bool(x.get("moving"))})
        return {"places": places, "projects": out}
    return {"places": [{**pl, **(relocate.place_summary(kind, Path(pl["dir"])) if pl["exists"] else {"files": 0, "bytes": 0})}
                       for pl in places]}


def _move_busy(pid: str) -> str | None:
    if why := _busy_reason(pid):
        return why
    with analyzer._qlock:
        if any(it[1] == pid for it in analyzer._queued) or (analyzer.current and analyzer.current[1] == pid):
            return "画像の解析が残っています。解析が終わってから移してください。"
    if any(t.get("project") == pid and t.get("status") not in pipeline.ENDED for t in tasks.list()):
        return "LoRA 作成タスクが終わっていません。終わってから移してください。"
    return None


@app.post("/api/storage/move")
def storage_move(body: dict):
    try:
        return relocate.start(body.get("kind"), body.get("items") or [], body.get("dest") or "", projects, _move_busy)
    except relocate.MoveError as e:
        raise HTTPException(400, str(e))


@app.get("/api/storage/move")
def storage_move_state():
    return relocate.status()


@app.post("/api/storage/move/cancel")
def storage_move_cancel():
    return {"accepted": relocate.cancel(), **relocate.status()}


@app.get("/api/llm")
def llm_state():
    try:
        return {**promptedit.models(), "error": None}
    except promptedit.LlmError as e:
        return {**promptedit.get_conf(), "models": [], "error": str(e)}


@app.put("/api/llm")
def llm_conf(body: dict):
    promptedit.set_conf(body.get("host"), body.get("model"))
    return llm_state()


def negative_used(model: dict | None, sampling: dict | None) -> bool:
    """At CFG 1 (Krea 2, Z-Image Turbo, Qwen-Image-2.1 ...) the sampler never uses the negative prompt."""
    cfg = (sampling or {}).get("cfg")
    ds = generate.default_sampling(model, (sampling or {}).get("fast"))
    try:
        cfg = float(cfg) if cfg not in (None, "") else float(ds["cfg"])
    except (TypeError, ValueError):
        cfg = float(ds["cfg"])
    return cfg > 1.0


@app.post("/api/imagegen/rewrite")
def imagegen_rewrite(body: dict):
    """Rewrite the prompt by the user's instruction with the local LLM (Ollama).  The LoRAs' trigger words
    already in the prompt are kept."""
    positive = body.get("positive") or ""
    keep = []
    for lo in body.get("loras") or []:
        if lo.get("path") and Path(lo["path"]).is_file():
            keep += [w for w in loraconf.get(lo["path"])["triggers"] if loraconf._has(positive, w) and w not in keep]
    model = registry.get(body.get("model_id") or "")
    style = generate.prompt_style(model)
    try:
        return promptedit.rewrite(positive, body.get("negative") or "", body.get("instruction") or "", style, keep,
                                  body.get("llm_model") or None, negative_used=negative_used(model, body.get("sampling")),
                                  history=body.get("history"))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except promptedit.LlmError as e:
        raise HTTPException(502, str(e))


@app.post("/api/imagegen/init")
async def imagegen_init(request: Request):
    """A source image for image-to-image (the raw file as the body)."""
    from starlette.concurrency import run_in_threadpool
    data = await request.body()
    if not data:
        raise HTTPException(400, "画像が空です")
    if len(data) > 100 * 2**20:
        raise HTTPException(400, "画像が大きすぎます（100MB まで）")
    try:
        return await run_in_threadpool(imagegen.save_init, data)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/imagegen/init_from")
def imagegen_init_from(body: dict):
    """The source image from an image already on this PC's screens (the i2t page's main image)."""
    p = i2tlink.allowed_file(str(body.get("path") or ""))
    if not p:
        raise HTTPException(404, "画像が見つかりません")
    try:
        return imagegen.save_init(p.read_bytes())
    except (OSError, ValueError) as e:
        raise HTTPException(400, str(e))


@app.get("/api/imagegen/init/{fname}")
def imagegen_init_image(fname: str):
    f = imagegen.init_file(fname)
    if not f:
        raise HTTPException(404)
    return FileResponse(f, media_type="image/png", headers={"Cache-Control": "max-age=86400"})


def _stem(name: str) -> str:
    """A file name as the other tools write it ("SDXL\\foo.safetensors", "foo") -> "foo" for matching."""
    base = re.split(r"[\\/]", str(name or ""))[-1]
    return re.sub(r"\.(safetensors|ckpt|pt|pth|bin|gguf|sft)$", "", base, flags=re.I).strip().lower()


@app.post("/api/imagegen/pnginfo")
async def imagegen_pnginfo(request: Request):
    """「画像から設定を読み込む」: the prompt, model, LoRAs and parameters written in a generated image (the raw
    file as the body), matched with this PC's files.  The form is not changed here; the page puts them in."""
    from starlette.concurrency import run_in_threadpool
    from . import pnginfo
    data = await request.body()
    if not data:
        raise HTTPException(400, "画像が空です")
    if len(data) > 100 * 2**20:
        raise HTTPException(400, "画像が大きすぎます（100MB まで）")
    try:
        info = await run_in_threadpool(pnginfo.read, data)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not info:
        raise HTTPException(400, "この画像には生成の設定が入っていません。SNS に投稿した画像やスクリーンショットでは、設定が消えていることがあります。")
    # this app's own image whose history entry is still there: its settings exactly as they were typed
    own = next((e for e in imagegen.history() if e["id"] == info.get("own_entry")), None)
    if own:
        s = own["settings"]
        form = {"model_id": s["model_id"], "loras": s["loras"], "positive": s["positive"], "negative": s["negative"],
                "width": s["width"], "height": s["height"], "seed": info["seed"] if info["seed"] is not None else s["seed"],
                "sampling": s.get("sampling_overrides") or {}, "init_image": s.get("init_image"), "denoise": s.get("denoise"),
                "regions": {"left": s["regions"][0], "right": s["regions"][1]} if s.get("regions") else None,
                "vae": s.get("vae") or "", "te": s.get("te") or "", "attention": s.get("attention") or "",
                "workflow": s.get("workflow") or "", "workflow_offer": s.get("workflow") or ""}
        missing = [] if registry.get(s["model_id"]) else ["生成モデルのファイルが見つかりません。モデルを選び直してください。"]
        return {"source": "LoRAtelier（この画面で生成した画像）", "form": form, "model_file": None, "notes": missing}
    notes = list(info["notes"])
    # a ComfyUI image: its workflow is kept, and generation runs it as it is unless the page turns that off
    # (opening the page again starts without it, with its 「workflow のまま生成する」 button: workflow_offer)
    workflow = None
    if info.get("graph"):
        workflow = imagegen.comfyflow.save(info.pop("graph"), info["source"], imagegen.comfyflow.summary(info))
        if imagegen.comfyflow._known is None and comfy.running():   # node classes of the ComfyUI here, asked once
            try:
                imagegen.comfyflow.remember_nodes(await run_in_threadpool(comfy._get, "/object_info", 30))
            except (OSError, ValueError):
                pass
        miss = imagegen.comfyflow.missing_nodes(imagegen.comfyflow.load(workflow["id"]))
        workflow["missing_nodes"] = miss
    scan = await run_in_threadpool(library.scan)
    form = {"positive": info["positive"], "negative": info["negative"], "seed": info["seed"],
            "sampling": {k: v for k, v in info["sampling"].items() if k in imagegen.SAMPLING_KEYS}, "regions": None}
    if info["width"] and info["height"]:
        form["width"], form["height"] = info["width"], info["height"]
    model_file = None
    if info["model_name"]:
        want = _stem(info["model_name"])
        reg = next((m for m in registry.list() if not m["missing"] and m["family"] in generate.GEN_FAMILIES
                    and any(_stem(m["files"].get(k) or "") == want for k in ("gen", "turbo", "ckpt"))), None)
        if reg:
            form["model_id"] = reg["id"]
        elif c := next((c for c in scan["checkpoints"] if _stem(c["path"]) == want and c["family"] in generate.GEN_FAMILIES), None):
            model_file = c["path"]   # not set up yet: the page chooses it in the model list, which sets it up
        else:
            notes.append(f"モデル「{info['model_name']}」はこのパソコンのモデルのフォルダーに見つからなかったので、モデルは今のままです。")
    # the VAE / CLIP named in the image: from the vae / text_encoders folders; not found: the model's own
    comp = await run_in_threadpool(library.companion_files)
    for kind, key, what in (("vae", "vae_name", "VAE"), ("te", "te_name", "CLIP")):
        form[kind] = ""
        if info[key]:
            hit = next((p for p in comp[kind] if _stem(p) == _stem(info[key])), None)
            if hit:
                form[kind] = hit
            else:
                notes.append(f"{what}「{info[key]}」は {'vae' if kind == 'vae' else 'text_encoders'} のフォルダーに見つからなかったので、"
                             f"{what} はモデルのものを使います。")
    form["attention"] = ""
    if info["attention"]:
        if imagegen.attention_installed().get(info["attention"], True):
            form["attention"] = info["attention"]
        else:
            notes.append(f"元の画像は {imagegen.ATTENTION_LABEL[info['attention']]} を使っていますが、ComfyUI の環境に入っていないため、"
                         "アテンションは標準にしました。")
    loras = []
    for lo in info["loras"]:
        # also under another name of the very same file (MOSS-Image-Base-8-steps-dmd2 = moss_image_8steps_dmd2 ...)
        names = imagegen.comfyflow.same_names(lo["name"])
        hit = next((x for x in scan["loras"] if _stem(x["path"]) == _stem(lo["name"])), None) or             next((x for x in scan["loras"] if imagegen.comfyflow._stem(x["path"]) in names), None)
        if hit:
            loras.append({"path": hit["path"], "strength": lo["strength"]})
        else:
            notes.append(f"LoRA「{lo['name']}」（強さ {lo['strength']:g}）は LoRA のフォルダーに見つからなかったので、入れていません。")
    form["loras"] = loras
    form["workflow"] = form["workflow_offer"] = workflow["id"] if workflow and not workflow["missing_nodes"] else ""
    return {"source": info["source"], "form": form, "model_file": model_file, "notes": notes, "workflow": workflow}


@app.post("/api/imagegen/start")
def imagegen_start(body: dict):
    imagegen.save_form({k: v for k, v in body.items()})
    try:
        return imagegen.submit(body, runner)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/imagegen/{eid}/cancel")
def imagegen_cancel(eid: str):
    if not imagegen.cancel(eid):
        raise HTTPException(400, "実行中・順番待ちの生成ではありません")
    return {"ok": True}


@app.delete("/api/imagegen/{eid}")
def imagegen_delete(eid: str):
    if not imagegen.delete(eid):
        raise HTTPException(400, "実行中・順番待ちの生成は消せません。先に中止してください。")
    return {"ok": True}


def _clean_jpeg(f: Path) -> bytes:
    """The image as a JPEG carrying its pixels only: the prompt, the ComfyUI workflow and every other text / EXIF /
    ICC block of the PNG are left out (a new image is made from the pixel values, nothing of the PNG's info is
    passed on).  Transparent parts become white."""
    import io
    with Image.open(f) as im:
        im.load()
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.getchannel("A"))
            im = bg
        px = Image.frombytes("RGB", im.size, im.convert("RGB").tobytes())
    out = io.BytesIO()
    px.save(out, "JPEG", quality=95, subsampling=0)
    return out.getvalue()


@app.get("/api/imagegen/img/{fname}")
def imagegen_image(fname: str, dl: int = 0, thumb: int = 0, jpg: int = 0):
    f = imagegen.img_file(Path(fname).name)
    if not f.exists():
        raise HTTPException(404)
    if jpg:   # 「JPG（情報なし）」: no prompt / workflow inside
        from fastapi.responses import Response
        return Response(_clean_jpeg(f), media_type="image/jpeg",
                        headers={"Content-Disposition": f'attachment; filename="{f.stem}.jpg"'})
    if dl:
        return FileResponse(f, media_type="image/png", filename=f.name)
    if thumb:
        return _thumb_response(f, f.parent / "thumbs" / (f.stem + ".webp"), 640)
    return FileResponse(f, media_type="image/png", headers={"Cache-Control": "max-age=86400"})


@app.get("/api/imagegen/{eid}/download.zip")
def imagegen_zip(eid: str, jpg: int = 0):
    """All images of one request in a zip (PNG is already compressed: stored as is; jpg=1: JPEGs without the
    prompt / workflow, as _clean_jpeg)."""
    e = next((x for x in imagegen.history() if x["id"] == eid), None)
    files = [imagegen.img_file(im["file"]) for im in (e or {}).get("images", [])]
    files = [f for f in files if f.exists()]
    if not files:
        raise HTTPException(404, "ダウンロードできる画像がありません")
    import tempfile
    from starlette.background import BackgroundTask
    fd, tmp = tempfile.mkstemp(suffix=".zip")
    os.close(fd)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
        for f in files:
            if jpg:
                zf.writestr(f.stem + ".jpg", _clean_jpeg(f))
            else:
                zf.write(f, f.name)
    name = time.strftime("images_%Y%m%d-%H%M%S", time.localtime(e["created"])) + ("_jpg" if jpg else "") + ".zip"
    return FileResponse(tmp, media_type="application/zip", filename=name, background=BackgroundTask(os.unlink, tmp))


@app.post("/api/imagegen/{eid}/copy")
def imagegen_copy(eid: str, body: dict):
    """Copy a result's settings into a project's material-generation recipe (a new project if none is given).
    The project's own choices (number of images, excluded angles) are kept."""
    try:
        return imagegen.copy_to_project(projects, eid, body.get("project_id"), body.get("name"))
    except KeyError:
        raise HTTPException(404, "生成結果またはプロジェクトが見つかりません")
    except ValueError as e:
        raise HTTPException(400, str(e))


# ====================================================================== training
def _used_images(p):
    meta = p.get_meta()
    if multichar.is_multi(meta):
        return multichar.gather(projects, p, _used_images)[0]
    imgs = p.images()
    ctx = _ctx(p, imgs, meta)
    return [im for im in imgs if judge.check_image(im, ctx)["use"]]


def _members(p):
    """Member list for the run snapshot of a multi-character project (None for any other project)."""
    return multichar.gather(projects, p, _used_images)[1] if multichar.is_multi(p.get_meta()) else None


def _plan(p):
    meta = p.get_meta()
    model = registry.get(meta.get("model_id") or "")
    used = _used_images(p)
    if not model:
        return {"model": None, "used": len(used), "preflight": {"errors": ["ベースモデルが未設定です。"], "warnings": [], "info": {}}}
    if not model.get("variant"):
        return {"model": model, "used": len(used), "preflight": preflight(p, used, {})}
    d = _defaults(meta, model, len(used))
    ov = meta.get("train_overrides") or {}
    values, changed = presets.merged(d, ov)
    pf = preflight(p, used, values)
    if multichar.is_multi(meta):
        _imgs, members = multichar.gather(projects, p, _used_images)
        e, w = multichar.check(members, lambda mid: (VARIANTS.get((registry.get(mid or "") or {}).get("variant") or "") or {}).get("caption"))
        pf["errors"] = e + [x for x in pf["errors"] if not x.startswith("採用画像が ")]
        pf["warnings"] = w + pf["warnings"]
        if any(registry.get(m.get("model_id") or "") and (registry.get(m["model_id"]) or {}).get("family") != model["family"]
               for m in members if not m.get("missing")):
            pf["warnings"].insert(0, "まとめるキャラの中に、別の系列のモデルを選んだプロジェクトがあります。説明文はそのまま使われます。")
    mp = _mask_plan(meta, model, used)
    if mp.get("summary"):
        sm, what = mp["summary"], masks.MODE_LABEL[mp["mode"]]
        if sm["waiting"]:
            pf["warnings"].append(f"{sm['waiting']} 枚のマスクを作っています。いま学習を始めると、その画像はマスクなしで学習します。")
        if sm["none"]:
            pf["warnings"].append(f"{sm['none']} 枚は人物が見つからなかったため、マスクなしで学習します。")
        if sm["failed"]:
            pf["warnings"].append(f"{sm['failed']} 枚はマスクを作れなかったため、マスクなしで学習します（素材の画面で作り直せます）。")
        if sm["small"]:
            pf["warnings"].append(f"{sm['small']} 枚は{what}として見つかった範囲がとても小さい画像です。素材の画面でマスクを確かめてください。")
    return {"model": model, "used": len(used), "defaults": d, "values": values, "changed": changed, "mask": mp,
            "fields": presets.FIELDS, "preflight": pf, "preset_version": presets.PRESET_VERSION,
            "relative": {k: ov[k] for k in ("steps_scale", "save_count") if ov.get(k)},
            "imported": meta.get("settings_import")}


def _mask_plan(meta, model, used):
    mode = masks.mode_for(meta)
    out = {"mode": mode, "mode_label": masks.MODE_LABEL.get(mode), "unsupported": masks.unsupported_reason(meta, model["family"]),
           **masks.setting(meta)}
    if out["on"] and not out["unsupported"]:
        out["summary"] = masks.summary(used, mode)
    return out


def _defaults(meta, model, n_used):
    g = gpus_cached()
    return presets.defaults(model["family"], meta["kind"], model["variant"], n_used, g[0]["vram_total_mb"] if g else None,
                            (meta.get("enhance") or {}).get("method"))


# ---- texture / enhancement LoRA (enhance.py)
@app.get("/api/enhance/effects")
def enhance_effects():
    from . import enhance
    return {"methods": enhance.METHODS, "families": list(enhance.FAMILIES),
            "effects": {k: {x: e[x] for x in ("label", "desc", "positive", "negative", "subject", "direction")}
                        for k, e in enhance.EFFECTS.items()}}


@app.get("/api/projects/{pid}/enhance/preview/{iid}")
def enhance_preview(pid: str, iid: int, side: str = "partner"):
    """One material image and its processed partner (pair method), downscaled, so the user can see what the
    LoRA will learn the difference of."""
    from . import enhance
    p = P(pid)
    im = p.image(iid)
    if not im or p.meta("kind") != "enhance":
        raise HTTPException(404)
    src = p.source_path(im)
    if not src.exists():
        raise HTTPException(404, "元画像が見つかりません")
    with Image.open(src) as img:
        img.load()
        img = ImageOps.exif_transpose(img).convert("RGB")
    if im.get("crop"):
        img = img.crop(tuple(int(v) for v in im["crop"]))
    # process at a size close to training (long side ~1024) so the strength of the effect looks the same
    img.thumbnail((1024, 1024))
    if side == "partner":
        img = enhance.process(img, enhance.effect(enhance.clean(p.meta("enhance"))["effect"])["op"])
    import io
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return StreamingResponse(io.BytesIO(buf.getvalue()), media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/projects/{pid}/train/plan")
def train_plan(pid: str):
    return _plan(P(pid))


# ---- carrying training settings to other projects: saved presets and "copy from another project"
def _snapshot(p):
    """The project's changed settings as a snapshot, or None (no model, or nothing changed)."""
    meta = p.get_meta()
    model = registry.get(meta.get("model_id") or "")
    ov = meta.get("train_overrides") or {}
    if not model or not model.get("variant") or not ov:
        return None
    snap = presets.snapshot(model["family"], _defaults(meta, model, len(_used_images(p))), ov)
    return snap if snap["values"] or snap.get("fixed_steps") else None


def _snap_summary(snap):
    parts = [f"{presets.FIELDS[k][0]} {v}" for k, v in snap["values"].items() if k in presets.FIELDS]
    if snap.get("steps_scale"):
        parts.append(f"学習ステップ数 自動の {round(snap['steps_scale'], 2):g} 倍")
    if snap.get("save_count"):
        parts.append(f"途中保存 約 {round(snap['save_count']):g} 回")
    return "・".join(parts)


def _train_presets():
    return config.settings().get("train_presets") or []


@app.get("/api/projects/{pid}/train/sources")
def train_sources(pid: str):
    """Where settings can be read from: saved presets and other projects with changed settings."""
    p = P(pid)
    model = registry.get(p.meta("model_id") or "")
    fam = model["family"] if model else None
    out = {"family": fam, "presets": [], "projects": []}
    for pr in _train_presets():
        out["presets"].append({"id": pr["id"], "name": pr["name"], "created": pr["created"],
                               "family_label": FAMILY_LABEL.get(pr["snap"]["family"], pr["snap"]["family"]),
                               "same_family": pr["snap"]["family"] == fam, "summary": _snap_summary(pr["snap"])})
    for info in projects.list():
        if info["id"] == pid or info.get("error"):
            continue
        try:
            snap = _snapshot(projects.get(info["id"]))
        except (KeyError, sqlite3.Error):
            continue
        if snap:
            out["projects"].append({"id": info["id"], "name": info["name"], "kind": info["kind"],
                                    "concept_target": info.get("concept_target"),
                                    "enhance": info.get("enhance"), "family_label": FAMILY_LABEL.get(snap["family"], snap["family"]),
                                    "same_family": snap["family"] == fam, "summary": _snap_summary(snap)})
    return out


@app.post("/api/projects/{pid}/train/presets")
def train_preset_save(pid: str, body: dict):
    """Save this project's changed settings as a named preset (same name: replaced)."""
    p = P(pid)
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "プリセットの名前を入力してください")
    snap = _snapshot(p)
    if not snap:
        raise HTTPException(400, "初期値から変えた設定がないため、保存するものがありません。")
    with config.settings()._lock:
        lst = [x for x in _train_presets() if x["name"] != name]
        lst.append({"id": uuid.uuid4().hex[:12], "name": name, "created": time.time(), "kind": p.meta("kind"), "snap": snap})
        config.settings().set("train_presets", lst)
    return {"name": name, "summary": _snap_summary(snap)}


@app.delete("/api/train/presets/{preset_id}")
def train_preset_delete(preset_id: str):
    with config.settings()._lock:
        lst = _train_presets()
        if not any(x["id"] == preset_id for x in lst):
            raise HTTPException(404, "プリセットが見つかりません")
        config.settings().set("train_presets", [x for x in lst if x["id"] != preset_id])
    return {"deleted": preset_id}


@app.post("/api/projects/{pid}/train/apply")
def train_apply(pid: str, body: dict):
    """Replace this project's changed settings with a preset's or another project's.
    Returns the previous settings so the UI can undo."""
    p = P(pid)
    meta = p.get_meta()
    model = registry.get(meta.get("model_id") or "")
    if not model:
        raise HTTPException(400, "先にこのプロジェクトの「LoRA を作るモデル」を選んでください。")
    if body.get("preset_id"):
        pr = next((x for x in _train_presets() if x["id"] == body["preset_id"]), None)
        if not pr:
            raise HTTPException(404, "プリセットが見つかりません")
        snap, src = pr["snap"], f"プリセット「{pr['name']}」"
    else:
        try:
            other = projects.get(body.get("project_id") or "")
        except KeyError:
            raise HTTPException(404, "プロジェクトが見つかりません")
        snap = _snapshot(other)
        if not snap:
            raise HTTPException(400, "そのプロジェクトには初期値から変えた設定がありません。")
        src = f"プロジェクト「{other.meta('name')}」"
    ov, skipped = presets.apply_snapshot(snap, model["family"], bool(body.get("fixed_steps")))
    prev = {"train_overrides": meta.get("train_overrides") or {}, "settings_import": meta.get("settings_import")}
    p.set_meta(train_overrides=ov, settings_import={
        "from": src, "skipped": skipped, "from_family": FAMILY_LABEL.get(snap["family"], snap["family"])})
    return {"project": _project_info(p), "skipped": skipped, "previous": prev}


@app.post("/api/projects/{pid}/train/restore")
def train_restore(pid: str, body: dict):
    """Undo of train/apply: put back the settings returned as "previous"."""
    p = P(pid)
    p.set_meta(train_overrides=body.get("train_overrides") or {}, settings_import=body.get("settings_import"))
    return _project_info(p)


@app.post("/api/projects/{pid}/train/start")
def train_start(pid: str, body: dict | None = None):
    p = P(pid)
    plan = _plan(p)
    if plan["preflight"]["errors"]:
        raise HTTPException(400, " / ".join(plan["preflight"]["errors"]))
    used = _used_images(p)
    rid = create_run(p, used, plan["values"], plan["changed"], (body or {}).get("name"), _members(p))
    ahead = 1 if runner.current else 0
    runner.enqueue(p, rid)
    return {"run_id": rid, "ahead": ahead + len(runner.waiting()) - 1}


# Settings that can be varied in one go ("まとめて予約"): each value becomes its own run in the queue.
BATCH_FIELDS = ("learning_rate", "max_train_steps", "network_dim", "seed", "resolution", "caption_dropout_rate")
BATCH_MAX = 8


@app.post("/api/projects/{pid}/train/batch")
def train_batch(pid: str, body: dict):
    """Reserve several runs that differ in one setting.  All of them use the current material and the other
    settings; each is frozen now like a normal start.  rank changes keep the alpha / rank ratio."""
    p = P(pid)
    field = body.get("field")
    if field not in BATCH_FIELDS:
        raise HTTPException(400, "まとめて変えられない設定です")
    plan = _plan(p)
    if plan["preflight"]["errors"]:
        raise HTTPException(400, " / ".join(plan["preflight"]["errors"]))
    if field not in (plan["values"] or {}):
        raise HTTPException(400, "このモデルでは使わない設定です")
    typ = presets.FIELDS[field][1]
    try:
        vals = list(dict.fromkeys(int(float(x)) if typ == "int" else float(x) for x in body.get("values") or []))
    except (TypeError, ValueError):
        raise HTTPException(400, "値は数値で入れてください（カンマ区切り）")
    if not vals:
        raise HTTPException(400, "試す値を入れてください")
    if len(vals) > BATCH_MAX:
        raise HTTPException(400, f"一度に予約できるのは {BATCH_MAX} 件までです")
    label = presets.FIELDS[field][0]
    variants = []
    for x in vals:
        v = dict(plan["values"])
        v[field] = x
        if field == "network_dim" and v.get("network_alpha") and plan["values"].get("network_dim"):
            v["network_alpha"] = round(float(plan["values"]["network_alpha"]) * x / float(plan["values"]["network_dim"]), 4)
        if field == "max_train_steps" and v["save_every_n_steps"] > x:
            v["save_every_n_steps"] = x
        errs = presets.validate(plan["model"]["family"], v)
        if errs:
            raise HTTPException(400, f"{label} {x:g}: " + " / ".join(errs))
        variants.append((x, v, [k for k in v if v[k] != plan["defaults"].get(k)]))
    need = sum(presets.estimate_disk_gb(plan["model"]["family"], v) for _x, v, _c in variants)
    free = shutil.disk_usage(p.runs_dir).free / 1e9
    if free < need + 2:
        raise HTTPException(400, f"空き容量が足りません（{len(variants)} 件で 約{need:.1f}GB 必要 / 空き {free:.1f}GB）。件数を減らしてください。")
    used, members = _used_images(p), _members(p)
    ids = []
    for x, v, changed in variants:
        rid = create_run(p, used, v, changed, f"{label} {x:g}", members)
        runner.enqueue(p, rid)
        ids.append(rid)
    return {"run_ids": ids}


# ---- the job queue (training runs and other GPU work, one at a time)
def _job_view(j) -> dict:
    task = getattr(j, "task", None) is not None
    pid = j.project.id
    name, run_name = None, None
    if hasattr(j.project, "meta"):
        try:
            name = j.project.meta("name")
            if not task:
                r = j.project.run(j.run_id)
                run_name = r["name"] if r else None
        except sqlite3.Error:
            pass
    label = getattr(j, "label", None) or ""
    kind = ("train" if not task else "gen" if label.startswith("素材生成") else "eval" if label.startswith("比較")
            else "imagegen" if label == imagegen.LABEL else "other")
    return {"uid": j.uid, "kind": kind, "project": pid, "project_name": name, "run": j.run_id,
            "run_name": run_name, "label": label or None, "detail": getattr(j, "detail", None),
            "resume": bool(j.resume), "cancellable": not task or bool(getattr(j, "on_cancel", None))}


def _num_after(label: str) -> int | None:
    try:
        return int(label.rsplit("#", 1)[1])
    except (IndexError, ValueError):
        return None


def _current_view(j) -> dict:
    """The running job with what it is doing (message, fraction 0-1 or None) and how it can be stopped."""
    v = _job_view(j)
    msg, frac, controls = (j.progress or {}).get("message"), None, []
    if v["kind"] == "train":
        pr = j.progress or {}
        if pr.get("total") and pr.get("step") is not None:
            frac = min(1.0, pr["step"] / pr["total"])
        if pr.get("phase") == "evaluating":
            controls = [{"action": "stop", "label": "比較画像を中止", "title": "学習した LoRA は残ります"}]
        elif pr.get("phase") == "training":
            controls = [{"action": "stop", "label": "次の保存地点で止める", "title": "保存した地点までの結果が残り、学習の履歴から続きを再開できます"},
                        {"action": "force", "label": "強制停止", "title": "すぐに止めます。最後に保存した地点より後の学習は失われます", "danger": True}]
        else:
            controls = [{"action": "force", "label": "中止", "danger": True}]
    elif v["kind"] == "gen":
        bid = _num_after(v["label"])
        try:
            row = j.project.q("SELECT message FROM gen_batches WHERE id=?", (bid,))
        except (sqlite3.Error, AttributeError):
            row = []
        msg = row[0]["message"] if row else msg
        m = re.search(r"(\d+)\s*/\s*(\d+)\s*枚", msg or "")
        if m and int(m.group(2)):
            frac = int(m.group(1)) / int(m.group(2))
        controls = [{"action": "pause", "label": "一時停止", "title": "今の 1 枚が終わったら止まります。素材の画面の「再開」で続きから生成できます"},
                    {"action": "stop", "label": "中止", "title": "すぐに止めます。生成済みの画像は残ります", "danger": True}]
    elif v["kind"] == "eval":
        pr = j.progress or {}
        if pr.get("total"):
            frac, msg = min(1.0, (pr.get("step") or 0) / pr["total"]), pr.get("message")
        else:
            msg = "準備中"
        controls = [{"action": "stop", "label": "中止", "title": "今の 1 枚が終わったら止まります。できた画像は残ります", "danger": True}]
    elif v["kind"] == "imagegen":
        pr = imagegen.progress(getattr(j, "entry", "")) or {}
        if pr.get("count"):
            done = max(0, (pr.get("image") or 1) - 1) + ((pr.get("step") or 0) / pr["steps"] if pr.get("steps") else 0)
            frac = min(1.0, done / pr["count"])
            msg = f"{pr.get('image') or 1} / {pr['count']} 枚目を生成中" if pr.get("phase") == "sampling" else "モデルを読み込んでいます"
        controls = [{"action": "stop", "label": "中止", "danger": True}]
    if msg in (None, "", "待機中") and v["kind"] != "train":
        msg = "実行中"
    return {**v, "progress": j.progress, "message": msg, "fraction": frac, "controls": controls}


@app.get("/api/queue")
def queue_state():
    cur = runner.current
    return {"current": _current_view(cur) if cur else None,
            "waiting": [_job_view(j) for j in runner.waiting()], "paused": runner.paused,
            "i2t": i2tlink.client.view(),   # 画像からプロンプト: its own worker, shown on the job page too
            "downloads": _downloads_view()}


@app.post("/api/queue/pause")
def queue_pause(body: dict):
    runner.set_paused(bool(body.get("on")))
    return queue_state()


@app.post("/api/queue/current/{action}")
def queue_current(action: str, body: dict):
    """Stop or pause the job running now.  body.uid must name it (a click meant for a job that has just
    ended must not stop the next one)."""
    j = runner.current
    if not j or j.uid != body.get("uid"):
        raise HTTPException(409, "その処理はもう終わっています。表示を更新しました。")
    v = _current_view(j)
    if action not in {c["action"] for c in v["controls"]}:
        raise HTTPException(400, "この処理では、その操作はできません")
    if v["kind"] == "train":
        return runner.stop(j.project, j.run_id, force=action == "force")
    if v["kind"] == "gen":
        return gen_control(j.project.id, _num_after(v["label"]), action)
    if v["kind"] == "eval":
        j.eval_cancel = True
        return {"accepted": True, "message": "中止を受け付けました。今の 1 枚が終わったら止まります。"}
    if v["kind"] == "imagegen":
        imagegen.cancel(getattr(j, "entry", ""))
        return {"accepted": True, "message": "中止を受け付けました。"}
    raise HTTPException(400, "この処理は止められません")


@app.post("/api/queue/{uid}/move")
def queue_move(uid: int, body: dict):
    if body.get("to") not in ("up", "down", "top"):
        raise HTTPException(400, "to must be up, down or top")
    if not runner.move(uid, body["to"]):
        raise HTTPException(404, "順番待ちの中に見つかりません（もう始まったか、取り消されています）")
    return queue_state()


@app.post("/api/queue/{uid}/cancel")
def queue_cancel(uid: int):
    j = next((x for x in runner.waiting() if x.uid == uid), None)
    if not j:
        raise HTTPException(400, "順番待ちの中に見つかりません（もう始まったか、取り消されています）")
    if getattr(j, "task", None):
        if not runner.cancel_task(uid):
            raise HTTPException(400, "この処理は順番待ちから取り消せません")
        return {"accepted": True, "message": f"「{j.label}」を取り消しました。"}
    return runner.stop(j.project, j.run_id)


def _eval_cancelled(p, rid, eid, msg="開始前に取り消しました"):
    p.q("UPDATE evals SET status='cancelled', error=? WHERE id=?", (msg, eid))
    bus.publish("eval", project=p.id, run=rid, eval=eid, status="cancelled")


# ---- LoRA creation tasks (pipeline.py): material generation -> analysis and captions -> training
def _pipeline_start_training(p, name):
    plan = _plan(p)
    if plan["preflight"]["errors"]:
        raise ValueError(" / ".join(plan["preflight"]["errors"]))
    rid = create_run(p, _used_images(p), plan["values"], plan["changed"], name, _members(p))
    runner.enqueue(p, rid)
    return rid


def _pipeline_stop_batch(p, bid):
    st = p.q("SELECT status FROM gen_batches WHERE id=?", (bid,))
    if st and st[0]["status"] == "queued":
        generate.control.flags[(p.id, bid)] = "stop"
    elif st and st[0]["status"] == "running":
        generate.control.request(p.id, bid, "stop")


def _pipeline_import(p, bid):
    ids = [r["id"] for r in p.q("SELECT id FROM gen_images WHERE batch_id=? AND picked=1 AND deleted=0 AND image_id IS NULL", (bid,))]
    return len(_import_generated(p, ids))


pipeline.hooks.queue_batch = _queue_batch
pipeline.hooks.stop_batch = _pipeline_stop_batch
pipeline.hooks.import_batch = _pipeline_import
pipeline.hooks.start_training = _pipeline_start_training
pipeline.hooks.stop_run = lambda p, rid: runner.stop(p, rid)
pipeline.hooks.analysis_busy = lambda: analyzer.queue_size() > 0 or analyzer.current is not None   # also the item in work
tasks = pipeline.Pipeline(projects)
PIPE_MAX_GEN = 200


def _task_view(t):
    try:
        name = projects.get(t["project"]).meta("name")
    except (KeyError, sqlite3.Error):
        name = None
    return {**t, "project_name": name}


@app.get("/api/tasks")
def tasks_state():
    return {"tasks": [_task_view(t) for t in tasks.list()], "paused": tasks.paused}


@app.post("/api/tasks/pause")
def tasks_pause(body: dict):
    tasks.set_paused(bool(body.get("on")))
    return tasks_state()


@app.post("/api/projects/{pid}/tasks")
def task_add(pid: str, body: dict):
    """Register a LoRA creation task: generate gen_count images (0: use the material as it is), wait for the
    analysis and captions, then train with the project's settings at that time."""
    p = P(pid)
    try:
        n = int(body.get("gen_count") or 0)
    except (TypeError, ValueError):
        raise HTTPException(400, "枚数は数値で入れてください")
    if not 0 <= n <= PIPE_MAX_GEN:
        raise HTTPException(400, f"生成する枚数は 0〜{PIPE_MAX_GEN} 枚で指定してください")
    if not registry.get(p.meta("model_id") or ""):
        raise HTTPException(400, "先にこのプロジェクトの「LoRA を作るモデル」を選んでください。")
    if n:
        errs = generate.check_recipe(p.meta("gen_recipe") or {}, p.meta("kind"), p)["errors"]
        if errs:
            raise HTTPException(400, "素材の生成の設定が足りません: " + " / ".join(errs))
    elif not p.images() and (p.meta("enhance") or {}).get("method") != "words":
        raise HTTPException(400, "素材がありません。素材を入れるか、生成する枚数を指定してください。")
    t = tasks.add(p, n, body.get("name"))
    ahead = len([x for x in tasks.list() if x["status"] in pipeline.ACTIVE + ("waiting",)]) - 1
    return {"task": _task_view(t), "ahead": ahead}


@app.post("/api/tasks/{tid}/move")
def task_move(tid: str, body: dict):
    if body.get("to") not in ("up", "down", "top"):
        raise HTTPException(400, "to must be up, down or top")
    if not tasks.move(tid, body["to"]):
        raise HTTPException(404, "待っているタスクの中に見つかりません（もう始まったか、取り消されています）")
    return tasks_state()


@app.post("/api/tasks/{tid}/cancel")
def task_cancel(tid: str):
    try:
        return {"message": tasks.cancel(tid)}
    except KeyError:
        raise HTTPException(404, "取り消せるタスクが見つかりません（もう終わっている可能性があります）")


@app.post("/api/tasks/clear")
def task_clear():
    return {"removed": tasks.remove_ended()}


@app.get("/api/projects/{pid}/runs")
def runs(pid: str):
    p = P(pid)
    out = []
    rows = p.runs()
    older = {r["id"]: next((o["config"] for o in rows if o["id"] < r["id"] and o["config"].get("items") is not None), None)
             for r in rows}
    for r in rows:
        cfg = r["config"]
        live = runner.find(pid, r["id"])
        ck = p.checkpoints(r["id"])
        for c in ck:
            if c["kind"] == "state":
                c["complete"] = state_is_complete(Path(c["path"]))
        evals = [dict(e) for e in p.q("SELECT id, created, status, error FROM evals WHERE run_id=? ORDER BY id", (r["id"],))]
        out.append({"id": r["id"], "name": r["name"], "created": r["created"], "ended": r["ended"], "status": r["status"],
                    "status_label": STATUS_LABEL.get(r["status"], r["status"]),
                    "progress": live.progress if live else r["progress"], "error": json.loads(r["error"]) if r["error"] else None,
                    "values": cfg.get("values"), "changed": cfg.get("changed"), "n_images": len(cfg.get("items", [])),
                    "model": (cfg.get("model") or {}).get("name"), "model_id": (cfg.get("model") or {}).get("id"),
                    "model_family": (cfg.get("model") or {}).get("family"), "engine": cfg.get("engine"),
                    "engine_commit": cfg.get("engine_commit"), "preset_version": cfg.get("preset_version"),
                    "checkpoints": ck, "evals": evals, "changes": runcompare.change_summary(older[r["id"]], cfg),
                    **_resume_info(r, ck)})
    return out


def _resume_state(run: dict, checkpoints: list[dict]) -> tuple[dict | None, str | None]:
    """The newest complete intermediate state and, if the run cannot continue from it, why."""
    states = [c for c in checkpoints if c["kind"] == "state" and c["step"] is not None and state_is_complete(Path(c["path"]))]
    if not states:
        return None, "再開用の保存地点（オプティマイザー状態を含む）がありません。LoRA の途中保存からの完全な再開はできません。"
    st = max(states, key=lambda c: c["step"])
    return st, resume_block_reason(run["config"], st["path"])


def _resume_info(run: dict, checkpoints: list[dict]) -> dict:
    if run["status"] not in ("cancelled", "failed"):
        return {"can_resume": False}
    st, why = _resume_state(run, checkpoints)
    return {"can_resume": bool(st) and not why, **({"resume_note": why} if st and why else {})}


@app.post("/api/projects/{pid}/runs/{rid}/stop")
def run_stop(pid: str, rid: int, body: dict | None = None):
    return runner.stop(P(pid), rid, force=bool((body or {}).get("force")))


@app.post("/api/projects/{pid}/runs/{rid}/resume")
def run_resume(pid: str, rid: int):
    p = P(pid)
    r = _run_or_404(p, rid)
    if r["status"] in ACTIVE:
        raise HTTPException(400, "実行中です")
    st, why = _resume_state(r, p.checkpoints(rid))
    if not st or why:
        raise HTTPException(400, why)
    runner.enqueue(p, rid, resume=st["path"])
    return {"resumed_from_step": st["step"]}


@app.delete("/api/projects/{pid}/runs/{rid}")
def run_delete(pid: str, rid: int):
    p = P(pid)
    r = p.run(rid)
    if r is None:
        raise HTTPException(404)
    if r["status"] in ACTIVE or runner.find(pid, rid):
        raise HTTPException(400, "実行中の学習は削除できません。止めてから削除してください。")
    if p.q("SELECT 1 FROM evals WHERE run_id=? AND status IN ('queued','running')", (rid,)):
        raise HTTPException(400, "比較画像の生成中です。終わってから削除してください。")
    projectio.delete_run(p, rid)
    return {"deleted": rid}


@app.get("/api/app_logs")
def app_logs():
    return projectio.app_logs()


@app.get("/api/app_logs/zip")
def app_logs_zip():
    from fastapi.responses import Response
    return Response(projectio.app_logs_zip(), media_type="application/zip",
                    headers={"Content-Disposition": f"attachment; filename=\"LoRAtelier_logs_{time.strftime('%Y%m%d-%H%M%S')}.zip\""})


@app.get("/api/app_logs/{name}")
def app_log_tail(name: str, tail: int = 300):
    try:
        return projectio.app_log_tail(name, max(1, min(tail, 2000)))
    except ValueError:
        raise HTTPException(404, "そのログはありません")


@app.post("/api/app_logs/clear")
def app_logs_clear():
    return projectio.clear_app_logs()


@app.get("/api/projects/{pid}/runs/{rid}/log")
def run_log(pid: str, rid: int, tail: int = 200):
    p = P(pid)
    r = _run_or_404(p, rid)
    logs = sorted((Path(r["run_dir"]) / "logs").glob("*.log"))
    if not logs:
        return {"text": ""}
    with open(logs[-1], "rb") as fh:   # only the end: a long run's log is large
        fh.seek(max(0, fh.seek(0, 2) - 200_000))
        data = fh.read().decode("utf-8", "replace")
    lines = [l for l in data.replace("\r", "\n").split("\n") if l.strip()]
    return {"file": str(logs[-1]), "text": "\n".join(lines[-tail:])}


# ====================================================================== evaluation / export
def _run_or_404(p, rid: int) -> dict:
    r = p.run(rid)
    if r is None:
        raise HTTPException(404, "その学習の回は見つかりません")
    return r


@app.get("/api/projects/{pid}/runs/{rid}/evals")
def evals(pid: str, rid: int):
    p = P(pid)
    out = []
    for e in p.q("SELECT * FROM evals WHERE run_id=? ORDER BY id DESC", (rid,)):
        d = dict(e)
        d["spec"] = json.loads(d["spec"])
        d["results"] = json.loads(d["results"]) if d["results"] else None
        if d["results"] and d["status"] == "done" and "recommend" not in d["results"]:   # evaluations made before it existed
            from .evaluate import recommend
            kind = p.run(rid)["config"]["kind"]
            d["results"]["recommend"] = recommend(kind, d["results"].get("images") or [], d["results"].get("scores") or {})
        out.append(d)
    return out


@app.get("/api/eval_models")
def eval_models_get():
    """Other models every comparison after training is also made with (kept for all projects)."""
    return {"ids": config.settings().get(EXTRA_MODELS_KEY, []), "max": MAX_EXTRA_MODELS}


@app.put("/api/eval_models")
def eval_models_put(body: dict):
    ids = list(dict.fromkeys(str(x) for x in body.get("ids") or []))
    known = {m["id"] for m in registry.list()}
    if any(i not in known for i in ids):
        raise HTTPException(400, "登録されていないモデルが含まれています。モデルの一覧を読み込み直してください。")
    # up to MAX_EXTRA_MODELS per family: a comparison uses only the models of the LoRA's family
    fam = {m["id"]: m.get("family") for m in registry.list()}
    for f in {fam[i] for i in ids}:
        if sum(1 for i in ids if fam[i] == f) > MAX_EXTRA_MODELS:
            raise HTTPException(400, f"同じ系列のモデルは {MAX_EXTRA_MODELS} つまで選べます。")
    config.settings().set(EXTRA_MODELS_KEY, ids)
    return {"ids": ids}


@app.post("/api/projects/{pid}/runs/{rid}/eval")
def eval_more(pid: str, rid: int, body: dict):
    p = P(pid)
    run = _run_or_404(p, rid)
    if run["status"] not in ("done", "cancelled", "failed"):
        raise HTTPException(400, "学習の終了後に比較できます")
    spec = default_spec(p, run)
    for k in ("checkpoints", "strengths", "seeds", "include_base"):
        if k in body:
            spec[k] = body[k]
    try:
        # an entry given twice would submit the same graph twice; ComfyUI answers the second from its cache
        # with the file already moved away, and the whole comparison fails
        spec["strengths"] = list(dict.fromkeys(round(float(x), 4) for x in spec["strengths"]))
        spec["seeds"] = list(dict.fromkeys(int(x) for x in spec["seeds"]))
        spec["checkpoints"] = list(dict.fromkeys(spec["checkpoints"]))
        spec["include_base"] = bool(spec["include_base"])
    except (TypeError, ValueError):
        raise HTTPException(400, "強さは数値、seed は整数で指定してください。")
    if any(not (0 <= s < 2**63) for s in spec["seeds"]):
        raise HTTPException(400, "seed は 0 以上の整数で指定してください。")
    if body.get("prompt_sets"):
        spec["prompts"] = run_prompts(run, [str(x) for x in body["prompt_sets"]])
    if body.get("extra_prompts"):
        spec["prompts"] = list(dict.fromkeys(spec["prompts"] + [x.strip() for x in body["extra_prompts"] if x.strip()]))
    if not spec["prompts"]:
        raise HTTPException(400, "プロンプトの種類を 1 つ以上選ぶか、プロンプトを入れてください。")
    if "extra_models" in body:   # the same saved versions and strengths on each chosen model
        spec.pop("extra_models", None)
        spec.pop("extra_checkpoints", None)
        spec.update(_extra_spec(run, [str(x) for x in body["extra_models"] or []], spec["checkpoints"]))
    per = len(spec["checkpoints"]) * len(spec["strengths"]) + (1 if spec["include_base"] else 0)
    per += len(spec.get("extra_models") or []) * (len(spec.get("extra_checkpoints") or []) * len(spec["strengths"])
                                                  + (1 if spec["include_base"] else 0))
    n = per * len(spec["seeds"]) * len(spec["prompts"])
    if n > 160:
        raise HTTPException(400, f"生成枚数が多すぎます（{n} 枚）。保存版・強度・seed の組み合わせを減らしてください（上限 160）。")
    with p.tx() as db:
        eid = db.execute("INSERT INTO evals(run_id, created, status, spec) VALUES(?,?,?,?)",
                         (rid, time.time(), "queued", json.dumps(spec, ensure_ascii=False))).lastrowid

    def task():
        try:
            run_eval(p, rid, eid, _eval_progress(p, rid))
        except EvalCancelled:
            _eval_cancelled(p, rid, eid, "中止しました（できた画像は残っています）")
        except Exception as e:
            p.q("UPDATE evals SET status='failed', error=? WHERE id=?", (str(e), eid))
            bus.publish("eval", project=p.id, run=rid, eval=eid, status="failed")
    runner.enqueue_task(p, f"比較 #{rid}", task, on_cancel=lambda: _eval_cancelled(p, rid, eid))
    return {"eval_id": eid, "n_images": n}


# ---- comparing runs with each other
def _compare_runs(p, ids) -> list[dict]:
    try:
        ids = [int(x) for x in ids]
    except (TypeError, ValueError):
        raise HTTPException(400, "比べる回の指定が正しくありません")
    if not 2 <= len(ids) <= 6:
        raise HTTPException(400, "比べる回を 2〜6 個選んでください")
    out = []
    for rid in ids:
        r = p.run(rid)
        if not r or not r["config"].get("items") and not r["config"].get("values"):
            raise HTTPException(404, f"学習 #{rid} が見つかりません")
        evals = []
        for e in p.q("SELECT * FROM evals WHERE run_id=? ORDER BY id", (rid,)):
            d = dict(e)
            d["spec"] = json.loads(d["spec"])
            d["results"] = json.loads(d["results"]) if d["results"] else None
            evals.append(d)
        out.append({"id": rid, "name": r["name"] or f"#{rid}", "status": r["status"], "created": r["created"],
                    "snap": r["config"], "evals": evals, "run": r})
    return out


@app.get("/api/projects/{pid}/compare")
def compare_runs(pid: str, runs: str, mode: str = "final", strength: float = 1.0):
    """Two or more runs side by side: settings, material and caption changes against the first run, and the
    comparison images that were made under the same conditions."""
    p = P(pid)
    rs = _compare_runs(p, runs.split(","))
    base = rs[0]["snap"]
    pending = [r["id"] for r in rs if any(e["status"] in ("queued", "running") for e in r["evals"])]
    return {"runs": [{"id": r["id"], "name": r["name"], "status": r["status"], "status_label": STATUS_LABEL.get(r["status"], r["status"]),
                      "created": r["created"]} for r in rs],
            "settings": runcompare.settings_rows([r["snap"] for r in rs]),
            "material": [{"run": r["id"], **runcompare.material_diff(base, r["snap"])} for r in rs[1:]],
            "images": runcompare.compare_images(rs, mode if mode in ("final", "recommend") else "final", strength),
            "generating": pending}


@app.post("/api/projects/{pid}/compare/generate")
def compare_generate(pid: str, body: dict):
    """Comparison images for every chosen run under the same conditions (prompts of the same kinds with each run's
    own trigger, the same seeds, the final version at one strength or each run's recommended one)."""
    p = P(pid)
    rs = _compare_runs(p, body.get("runs") or [])
    for r in rs:
        if r["status"] not in ("done", "cancelled", "failed") or not any(c["kind"] == "lora" for c in p.checkpoints(r["id"])):
            raise HTTPException(400, f"{r['name']} はまだ比べられる LoRA がありません（学習の終了後に比べられます）")
    mode = body.get("mode") if body.get("mode") in ("final", "recommend") else "final"
    try:
        strength = round(float(body.get("strength", 1.0)), 4)
        seeds = list(dict.fromkeys(int(x) for x in (body.get("seeds") or [1234, 5678])))
    except (TypeError, ValueError):
        raise HTTPException(400, "強さは数値、seed は整数で指定してください。")
    if not seeds or any(not (0 <= x < 2**63) for x in seeds):
        raise HTTPException(400, "seed は 0 以上の整数で指定してください。")
    sets = [str(x) for x in body.get("prompt_sets") or ["standard"]]
    extra = [x.strip() for x in body.get("extra_prompts") or [] if x.strip()]

    def prompts_for(r):
        own = [x.replace("{T}", r["snap"].get("trigger") or "") for x in extra]
        return list(dict.fromkeys(run_prompts(r["run"], sets) + own))

    def size_for(r):
        d = default_spec(p, r["run"])
        return d["width"], d["height"]
    specs = runcompare.align_specs(rs, mode, strength, seeds, prompts_for, size_for)
    n = sum((len(sp["checkpoints"]) * len(sp["strengths"]) + (1 if sp["include_base"] else 0)) * len(sp["seeds"]) * len(sp["prompts"])
            for _rid, sp in specs)
    if n > 240:
        raise HTTPException(400, f"生成枚数が多すぎます（{n} 枚）。回・seed・プロンプトの種類を減らしてください（上限 240）。")
    queued = []
    for rid, sp in specs:
        with p.tx() as db:
            eid = db.execute("INSERT INTO evals(run_id, created, status, spec) VALUES(?,?,?,?)",
                             (rid, time.time(), "queued", json.dumps(sp, ensure_ascii=False))).lastrowid

        def task(rid=rid, eid=eid):
            try:
                run_eval(p, rid, eid, _eval_progress(p, rid))
            except EvalCancelled:
                _eval_cancelled(p, rid, eid, "中止しました（できた画像は残っています）")
            except Exception as e:
                p.q("UPDATE evals SET status='failed', error=? WHERE id=?", (str(e), eid))
                bus.publish("eval", project=p.id, run=rid, eval=eid, status="failed")
        runner.enqueue_task(p, f"比較 #{rid}", task, on_cancel=lambda rid=rid, eid=eid: _eval_cancelled(p, rid, eid))
        queued.append({"run": rid, "eval_id": eid})
    return {"queued": queued, "n_images": n}


@app.get("/api/projects/{pid}/runs/{rid}/evals/{eid}/{fname}")
def eval_image(pid: str, rid: int, eid: int, fname: str, thumb: int = 0):
    p = P(pid)
    r = p.run(rid)
    if not r:
        raise HTTPException(404)
    f = (Path(r["run_dir"]) / "eval" / f"{eid:03d}" / Path(fname).name)
    if not f.exists():
        raise HTTPException(404)
    if thumb:
        return _thumb_response(f, f.parent / "thumbs" / (f.stem + ".webp"), 384)
    return FileResponse(f, media_type="image/png", headers={"Cache-Control": "max-age=86400"})


@app.post("/api/projects/{pid}/runs/{rid}/export")
def export_lora(pid: str, rid: int, body: dict):
    p = P(pid)
    ck = body.get("ckpt_path")
    if not any(c["path"] == ck for c in p.checkpoints(rid)):
        raise HTTPException(400, "保存版を選んでください")
    ev = p.q("SELECT spec, results FROM evals WHERE run_id=? AND status='done' ORDER BY id DESC LIMIT 1", (rid,))
    summary = None
    if ev:
        res = json.loads(ev[0]["results"] or "{}")
        summary = {"conditions": res.get("conditions"), "prompts": json.loads(ev[0]["spec"])["prompts"],
                   "other_models": [{"name": o["name"], "ratio": o.get("ratio"), "reasons": o.get("reasons")}
                                    for o in res.get("other_models") or []] or None,
                   "seeds": json.loads(ev[0]["spec"])["seeds"], "strengths": json.loads(ev[0]["spec"])["strengths"],
                   "lora_key_warnings": res.get("n_lora_key_warnings")}
    try:
        out = do_export(p, rid, ck, body.get("strength"), (body.get("dest") or "").strip().strip('"') or None, summary)
    except (OSError, RuntimeError, ValueError) as e:
        raise HTTPException(400, f"書き出せませんでした: {e}")
    # offered next to the result: "学習データ（x.xGB）を破棄しますか？"
    out["leftover_bytes"] = projectio.training_leftovers(p, _runs_in_use(p))["bytes"]
    return out


def _runs_in_use(p) -> set[int]:
    """Runs whose files are still read: queued or training, or with comparison images being made."""
    busy = {r["id"] for r in p.runs() if r["status"] in ACTIVE or runner.find(p.id, r["id"])}
    return busy | {r["run_id"] for r in p.q("SELECT run_id FROM evals WHERE status IN ('queued','running')")}


def _all_leftovers(discard: bool) -> dict:
    """「すべてのプロジェクトの学習データを破棄する」 (settings): training_leftovers of every project."""
    total, n, skipped = 0, 0, False
    for d in projects.folders():
        try:
            p = projects.get(d.name)
        except (KeyError, OSError, sqlite3.Error):   # also ProjectMoving: left for later
            skipped = True
            continue
        busy = _runs_in_use(p)
        skipped = skipped or bool(busy)
        got = projectio.discard_training_leftovers(p, busy) if discard else projectio.training_leftovers(p, busy)["bytes"]
        total += got
        n += bool(got)
    return {"bytes": total, "projects": n, "skipped": skipped}


@app.get("/api/training_leftovers")
def all_leftovers():
    return _all_leftovers(False)


@app.post("/api/training_leftovers/discard")
def discard_all_leftovers():
    r = _all_leftovers(True)
    return {"freed": r["bytes"], "skipped": r["skipped"]}


@app.post("/api/projects/{pid}/training_leftovers/discard")
def discard_training_leftovers(pid: str):
    """The saved LoRAs that were not exported, the resume states and the training caches of every run.  The
    material, captions and exported LoRAs stay (the project can be trained again)."""
    p = P(pid)
    busy = _runs_in_use(p)
    freed = projectio.discard_training_leftovers(p, busy)
    return {"freed": freed, "skipped_runs": sorted(busy)}


@app.get("/api/projects/{pid}/exports")
def exports(pid: str):
    p = P(pid)
    return [dict(r) | {"info": json.loads(r["info"] or "{}"), "exists": Path(r["dest"]).exists()}
            for r in p.q("SELECT * FROM exports ORDER BY id DESC")]


# ====================================================================== static
@app.get("/", response_class=HTMLResponse)
def index():
    # version the scripts by file time so the browser never keeps an old app.js after an update
    html = (config.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    for f in ("app.js", "style.css", "i2t.js"):
        v = int((config.STATIC_DIR / f).stat().st_mtime)
        html = html.replace(f'"/static/{f}"', f'"/static/{f}?v={v}"')
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")


@app.exception_handler(Exception)
def on_error(request, exc):
    import traceback
    (config.DATA_DIR / "server_errors.log").open("a", encoding="utf-8").write(
        time.strftime("%F %T ") + request.url.path + "\n" + traceback.format_exc() + "\n")
    return JSONResponse({"detail": f"内部エラー: {type(exc).__name__}: {exc}（詳細は設定画面の「ログ」欄で見られます）"}, status_code=500)


BOOT = time.time()   # this start of the app: rows created later belong to it


def resume_pending_on_startup():
    """After a restart, runs that were in progress are marked as interrupted (not silently resumed)."""
    imagegen.recover()
    threading.Thread(target=projectio.empty_trash, daemon=True).start()
    try:
        saved = json.loads(QUEUE_FILE.read_text(encoding="utf-8")) if QUEUE_FILE.exists() else []
    except (OSError, ValueError):
        saved = []
    order = {(q.get("project"), q.get("run")): (i, q.get("resume")) for i, q in enumerate(saved) if isinstance(q, dict)}
    reserved = []   # runs reserved but not started yet: they keep their place in the queue
    for info in projects.list():
        try:
            p = projects.get(info["id"])
        except KeyError:
            continue
        for r in p.runs():
            if r["created"] >= BOOT:   # started after this start (the page answers while this runs)
                continue
            if r["status"] == "queued" and r["run_dir"] and (r["progress"] or {}).get("phase", "queued") == "queued":
                pos, resume = order.get((p.id, r["id"]), (None, None))
                if pos is not None or not p.checkpoints(r["id"]):   # not in the file: only a run that never trained
                    reserved.append((pos if pos is not None else len(order), r["created"], p, r["id"], resume))
                    continue
            if r["status"] in ACTIVE:
                out_dir = Path(r["run_dir"] or ".") / "output"
                if r["run_dir"] and out_dir.is_dir():
                    try:   # saves written after the runner's last 2 s poll; cut-off files are skipped
                        runner._register_checkpoints(Job(p, r["id"]), out_dir, final=True)
                    except Exception:
                        pass
                runner._set(p, r["id"], status="failed", ended=time.time(),
                            error={"code": "interrupted", "message": "アプリの終了により中断されました。",
                                   "fixes": ["再開用の保存地点があれば「再開」できます"]})
        for b in p.q("SELECT id FROM gen_batches WHERE status IN ('queued','running') AND created < ?", (BOOT,)):
            generate._set_status(p, b["id"], "paused", "アプリの終了で止まりました。「再開」で続きを生成できます。")
        for e in p.q("SELECT id FROM evals WHERE status IN ('queued','running') AND created < ?", (BOOT,)):
            p.q("UPDATE evals SET status='failed', error='アプリの終了により中断されました' WHERE id=?", (e["id"],))
        _resume_analysis(p)
    for _pos, _created, p, rid, resume in sorted(reserved, key=lambda x: (x[0], x[1])):
        if resume and not Path(resume).exists():   # never train a resumed run again from the start over its saves
            runner._set(p, rid, status="failed", ended=time.time(),
                        error={"code": "interrupted", "message": "再開に使う保存地点が見つからないため、予約を取り消しました。",
                               "fixes": ["学習の履歴で、ほかの保存地点から再開できるか確認する"]})
            continue
        runner.enqueue(p, rid, resume=resume)
    tasks.recover()
