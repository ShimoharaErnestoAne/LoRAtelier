"""Connection to the resident i2t worker (worker/i2t_worker.py): the 「画像からプロンプト」 feature.

The worker runs the bundled copy of i2t in the analysis environment.  Short calls (settings,
history, chips, generation info, ...) wait for their reply; long jobs (analysis, model downloads, Ollama
pull / setup, translation, pose estimation) return an id at once and report through the event bus
("i2t" events: progress / fraction / stop_state / done), one at a time like in the i2t window.

GPU: with i2t's default "GPU / VRAM の使用を抑える" the tagger runs on the CPU and only Ollama uses the GPU; the VLM
stays loaded for a short while between images and is unloaded as soon as another job takes the GPU (unload_kept_vlm).
When that setting is off, an analysis waits while training or generation holds the GPU, and the worker
(with its GPU models) is ended when such a job starts.
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import threading
import time
from concurrent.futures import Future
from pathlib import Path

from . import config
from .events import bus
from .gpu import gpu

WORKER_SCRIPT = config.APP_DIR / "worker" / "i2t_worker.py"
I2T_DIR = config.ENGINES_DIR / "i2t"
DATA_DIR = config.DATA_DIR / "i2t"
IMAGES_DIR = DATA_DIR / "images"   # the default place of added images (see save_upload)
LONG_JOBS = ("analyze", "tagger_download", "face_models", "pull", "translate", "pose_estimate", "ollama_setup",
             "translate_en", "readings", "moss_catalog")
# what the job page calls each long job (画像からプロンプト runs them in its own worker, outside the GPU queue)
JOB_LABELS = {"analyze": "画像の分析", "tagger_download": "タガーモデルの取得", "face_models": "顔用モデルの取得",
              "pull": "説明 AI（VLM）の取得", "translate": "タグの日本語訳", "pose_estimate": "骨格の推定",
              "ollama_setup": "Ollama の準備", "translate_en": "日本語の語を英語に訳す", "readings": "タグの読みを調べる",
              "moss_catalog": "MOSS-Image のキャラ一覧の取得"}
PAUSABLE = ("analyze", "translate")


def models_dir() -> Path:
    p = config.settings().get("i2t_models_dir")
    return Path(p) if p else config.MODELS_DIR / "i2t"


class I2tClient:
    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._pending: dict[int, Future] = {}
        self._rid = itertools.count(1)
        self.status = "stopped"          # stopped / starting / ready / failed
        self.job: dict | None = None     # the running long job: {rid, op, message, fraction, cancellable}
        self.last_done: dict | None = None
        self._consts = None
        self._restarts: list[float] = []   # unexpected ends (stop() is not counted)
        self.kept_vlm: dict | None = None  # {host, model}: the VLM an analysis left loaded in Ollama (unload_kept_vlm)

    def available(self) -> bool:
        return config.ANALYSIS_PYTHON.exists() and (I2T_DIR / "i2t.py").exists()

    def _start(self):
        if not self.available():
            raise RuntimeError("i2t を動かす環境がありません（setup.bat を実行してください）")
        if self.proc is not None:   # the worker ended by itself (stop() clears proc): limit the restarts
            now = time.time()
            self._restarts = [t for t in self._restarts if now - t < 600] + [now]
            if len(self._restarts) > 3:
                self.status = "failed"
                raise RuntimeError("i2t のワーカーが繰り返し停止したため、自動再起動を止めました。"
                                   "data\\i2t_worker.log を確認し、しばらくしてからもう一度試してください。")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        log = open(config.DATA_DIR / "i2t_worker.log", "ab")
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
        try:
            self.proc = subprocess.Popen([str(config.ANALYSIS_PYTHON), "-u", str(WORKER_SCRIPT), str(I2T_DIR), str(DATA_DIR),
                                          str(models_dir())], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                         env=env, creationflags=flags, cwd=str(DATA_DIR))
        finally:
            log.close()   # the worker has its own handle
        self.status = "starting"
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True, name="i2t-reader").start()

    def _reader(self, proc):
        for raw in proc.stdout:
            try:
                msg = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            ev = msg.get("event")
            if ev == "ready":
                self.status = "ready"
                continue
            if ev:
                if self.job and msg.get("rid") == self.job["rid"]:
                    if ev == "progress":
                        self.job["message"] = msg["message"]
                        if msg.get("fraction", -1) >= 0:
                            self.job["fraction"] = msg["fraction"]
                    elif ev == "fraction":
                        self.job["fraction"], self.job["est"] = msg["value"], msg.get("est")
                    elif ev == "stop_state":
                        self.job["cancellable"] = msg["cancellable"]
                bus.publish("i2t", **msg)
                continue
            rid = msg.get("rid")
            if self.job and rid == self.job["rid"]:
                done = {**msg, "op": self.job["op"], "finished": time.time()}
                self.last_done = done
                self.job = None
                if done["op"] == "analyze":
                    self._release_gpu()
                    if msg.get("vlm_kept"):
                        self.kept_vlm = msg["vlm_kept"]
                        if gpu.owner or gpu.analysis_active:   # the GPU was taken while it ran: free it now
                            threading.Thread(target=unload_kept_vlm, daemon=True, name="i2t-unload").start()
                bus.publish("i2t", event="done", **{k: v for k, v in done.items() if k != "trace"})
            fut = self._pending.pop(rid, None)
            if fut is not None:
                fut.set_result(msg)
        if self.proc is not None and self.proc is not proc:
            return   # a newer worker runs now: its job and GPU hold are not this one's (old calls time out)
        self.status = "stopped" if self.status != "failed" else "failed"
        self._release_gpu()   # the worker died during an analysis: training must not wait for it forever
        if self.job:
            done = {"rid": self.job["rid"], "op": self.job["op"], "ok": False, "error": "i2t のワーカーが終了しました"}
            self.last_done, self.job = done, None
            bus.publish("i2t", event="done", **done)
        for rid, fut in list(self._pending.items()):
            self._pending.pop(rid, None)
            fut.set_result({"ok": False, "error": "i2t のワーカーが終了しました"})

    _gpu_held = False

    def _release_gpu(self):
        if self._gpu_held:
            self._gpu_held = False
            gpu.end_analysis()

    def _send(self, obj: dict):
        with self._lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            self.proc.stdin.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
            self.proc.stdin.flush()

    def call(self, fn: str, *args, timeout: float = 120, **kwargs):
        """A plain i2t function (whitelisted in the worker).  Raises RuntimeError with i2t's message."""
        rid = next(self._rid)
        fut: Future = Future()
        self._pending[rid] = fut
        try:
            self._send({"rid": rid, "op": "call", "fn": fn, "args": list(args), "kwargs": kwargs})
        except OSError as e:
            self._pending.pop(rid, None)
            raise RuntimeError(f"i2t のワーカーへ送信できません: {e}")
        try:
            res = fut.result(timeout=timeout)
        except TimeoutError:
            self._pending.pop(rid, None)
            raise RuntimeError("i2t の応答がありません（時間切れ）")
        if not res.get("ok"):
            raise RuntimeError(res.get("error") or "i2t でエラーが起きました")
        return res.get("value")

    def start_job(self, op: str, **kw) -> dict:
        """Start a long job; returns at once.  Only one runs at a time (like the i2t window)."""
        if op not in LONG_JOBS:
            raise ValueError(op)
        if self.job:
            raise RuntimeError("ほかの処理の最中です。終わるか中止してからもう一度押してください。")
        rid = next(self._rid)
        self.job = {"rid": rid, "op": op, "message": "準備中...", "fraction": None, "est": None, "cancellable": True,
                    "started": time.time()}
        if op == "analyze" and not (kw.get("cfg") or {}).get("low_vram", True):
            self.job["message"] = "GPU が空くのを待っています（学習・生成の後に始まります）" if gpu.owner else "準備中..."
            threading.Thread(target=self._send_after_gpu, args=(rid, op, kw), daemon=True).start()
        else:
            self._send_job(rid, op, kw)
        return dict(self.job)

    def _send_after_gpu(self, rid, op, kw):
        gpu.begin_analysis()
        if not self.job or self.job["rid"] != rid:   # cancelled while waiting
            gpu.end_analysis()
            return
        self._gpu_held = True
        self._send_job(rid, op, kw)

    def _send_job(self, rid, op, kw):
        try:
            self._send({"rid": rid, "op": op, **kw})
        except (OSError, RuntimeError) as e:
            self._release_gpu()
            done = {"rid": rid, "op": op, "ok": False, "error": str(e)}
            self.last_done, self.job = done, None
            bus.publish("i2t", event="done", **done)

    def view(self) -> dict | None:
        """The running long job as the job page shows it (None when nothing runs)."""
        j = self.job
        if not j:
            return None
        return {"op": j["op"], "label": JOB_LABELS.get(j["op"], j["op"]), "message": j.get("message") or "",
                "fraction": j.get("fraction"), "cancellable": j.get("cancellable", True) is not False,
                "pausable": j["op"] in PAUSABLE, "paused": bool(j.get("paused")), "started": j.get("started")}

    def control(self, what: str):
        """cancel / pause / resume the running job."""
        if self.job and what in ("pause", "resume"):
            # kept here so the i2t page and the job page show the same state, whichever one was pressed
            self.job["paused"] = what == "pause"
            bus.publish("i2t", event="paused", rid=self.job["rid"], paused=self.job["paused"])
        if what == "cancel" and self.job and self.job["op"] == "analyze" and self.job["message"].startswith("GPU が空く"):
            done = {"rid": self.job["rid"], "op": "analyze", "ok": False, "cancelled": True, "error": "中止しました"}
            self.last_done, self.job = done, None
            bus.publish("i2t", event="done", **done)
            return
        if self.proc is not None and self.proc.poll() is None:
            self._send({"op": what})

    def consts(self) -> dict:
        if self._consts is None:
            self._consts = self.call("consts")
        else:   # which tagger models are present changes after a download
            self._consts["tagger_models"] = {k: {**v, "ready": self.call("tagger_ready", k)} for k, v in self._consts["tagger_models"].items()}
        return self._consts

    def restart(self):
        """After the models folder changed: the worker reads it at start."""
        self.stop()
        self._consts = None

    def stop(self):
        with self._lock:
            if self.proc is not None and self.proc.poll() is None:
                from .training import kill_tree
                kill_tree(self.proc.pid)
            self.proc = None
            self.status = "stopped"


client = I2tClient()


def _release_for_exclusive():
    """Training / generation takes the GPU: end the worker only if it may hold GPU models (low_vram off)."""
    try:
        cfg = json.loads((DATA_DIR / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not cfg.get("low_vram", True) and not client.job:
        client.stop()


gpu.on_exclusive(_release_for_exclusive)


def unload_kept_vlm():
    """GPU hook: training, generation or material analysis takes the GPU - unload the VLM an analysis left loaded
    in Ollama (the worker keeps it for a while between images; see i2t_worker.VLM_KEEP)."""
    kept, client.kept_vlm = client.kept_vlm, None
    if not kept:
        return
    import urllib.request
    req = urllib.request.Request(kept["host"] + "/api/generate", headers={"Content-Type": "application/json"},
                                 data=json.dumps({"model": kept["model"], "keep_alive": 0}).encode())
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read()
    except (OSError, ValueError):
        pass   # Ollama is gone: nothing is loaded; it unloads by itself after VLM_KEEP otherwise


gpu.on_exclusive(unload_kept_vlm)
gpu.on_analysis(unload_kept_vlm)


def save_upload(name: str, data: bytes) -> Path:
    """An image added to the i2t screen (file, drop or paste) is kept under data/i2t/images, named by its content."""
    import hashlib
    images_dir = config.storage_dir("i2t") / "images"   # a folder set on the settings page, else IMAGES_DIR
    images_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c for c in Path(name).name if c not in '\\/:*?"<>|') or "image.png"
    dest = images_dir / f"{hashlib.sha256(data).hexdigest()[:12]}_{safe}"
    if not dest.exists():
        tmp = dest.with_name(dest.name + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, dest)
    return dest


def allowed_file(path: str) -> Path | None:
    """Files the screen may show: images added here, i2t's history thumbnails, and LoRAtelier's own
    generated images (so they can be analysed too)."""
    p = Path(path).resolve()
    for kind in ("i2t", "imagegen"):   # an image moved to another place on the settings page is found there
        dirs = [d.resolve() for d in config.storage_dirs(kind)]
        for root in dirs:
            try:
                rel = p.relative_to(root)
            except ValueError:
                continue
            return next((d / rel for d in [root, *dirs] if (d / rel).is_file()), None)
    for root in (DATA_DIR, config.DATA_DIR / "imagegen"):
        try:
            p.relative_to(root.resolve())
            return p if p.is_file() else None
        except ValueError:
            continue
    return None
