"""Staged analysis pipeline.

Stage 1 (fast, CPU, in-process thread pool): decode, facts, hashes, thumbnail -> UI immediately.
Stage 2 (AI, resident worker process on the GPU): tags, people, identity features, embeddings.
Stage 3 (captions): tag captions are built in-process; natural-language captions go to the VLM.

Results are cached by (file sha, crop, analyzer version) so identical material is never
re-analysed, and the cache key changes whenever the analyzer or preprocessing changes.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import queue
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from . import captions, config, fastcheck
from .events import bus
from .gpu import gpu

ANALYZER_VERSION = "ai-4"   # keep in sync with worker/analysis_worker.py
WORKER_SCRIPT = config.APP_DIR / "worker" / "analysis_worker.py"


def ai_key(sha: str, crop) -> str:
    c = "full" if not crop else "-".join(str(int(v)) for v in crop)
    return f"{ANALYZER_VERSION}:{sha[:24]}:{c}"


def _cache_path(key: str) -> Path:
    return config.ANALYSIS_CACHE_DIR / (hashlib.sha1(key.encode()).hexdigest() + ".json")


def cache_get(key: str):
    p = _cache_path(key)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    return None


def cache_put(key: str, value) -> None:
    config.atomic_write_json(_cache_path(key), value)


def _other_image(old_key: str | None, new_key: str) -> bool:
    """The analysis is of another picture (file or crop), not only of a newer analyzer: the caption is made again."""
    return bool(old_key) and old_key.split(":")[-2:] != new_key.split(":")[-2:]


def _no_caption(img) -> dict:
    """A failed analysis makes no caption; an image waiting for one would show "being made" forever."""
    return {"caption_status": "skipped"} if img and img.get("caption_status") in ("pending", "running") else {}


LOAD_LIMIT_S = 2 * 3600   # a worker call may wait this long for a model download / load
GPU_RETRY_S = 300         # a worker analysing on the CPU because CUDA failed tries the GPU again after this

class WorkerClient:
    """Owns the analysis worker process.  Restarts it on crash, at most 3 times per 10 min."""

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._pending: dict[int, Future] = {}
        self._rid = itertools.count(1)
        self._restarts: list[float] = []
        self.status = "stopped"      # stopped/starting/ready/ready_cpu/loading:<model>/failed
        self.last_error = None
        self.gpu_error = None        # CUDA failed to start in the running worker: it analyses on the CPU
        self.cpu_since = None

    def available(self) -> bool:
        return config.ANALYSIS_PYTHON.exists() and WORKER_SCRIPT.exists()

    def _start(self):
        now = time.time()
        self._restarts = [t for t in self._restarts if now - t < 600]
        if len(self._restarts) >= 3:
            self.status = "failed"
            raise RuntimeError("解析ワーカーが繰り返し停止したため、自動再起動を止めました。技術ログを確認してください。")
        self._restarts.append(now)
        log = open(config.DATA_DIR / "analysis_worker.log", "ab")
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
        try:
            self.proc = subprocess.Popen([str(config.ANALYSIS_PYTHON), "-u", str(WORKER_SCRIPT)], stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=log, env=env, creationflags=flags,
                                         cwd=str(config.DATA_DIR))
        finally:
            log.close()   # the worker has its own handle
        self.status = "starting"
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True, name="worker-reader").start()

    def _reader(self, proc):
        for raw in proc.stdout:
            try:
                msg = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if "event" in msg:
                ev = msg["event"]
                if ev == "ready":
                    self.gpu_error = msg.get("gpu_error")
                    self.cpu_since = time.time() if self.gpu_error else None
                self.status = (("ready_cpu" if self.gpu_error else "ready") if ev in ("ready", "loaded")
                               else f"loading:{msg.get('model')}")
                bus.publish("worker", status=self.status)
                continue
            fut = self._pending.pop(msg.get("rid"), None)
            if fut is not None:
                if self.status.startswith("loading"):
                    self.status = "ready_cpu" if self.gpu_error else "ready"
                    bus.publish("worker", status=self.status)
                fut.set_result(msg)
        # process ended
        if self.proc is proc:
            self.status = "stopped"
        for rid, fut in list(self._pending.items()):
            if getattr(fut, "proc", proc) is proc:   # requests sent to a newer process are not failed
                self._pending.pop(rid, None)
                fut.set_result({"ok": False, "error": "解析ワーカーが終了しました"})

    def call(self, op: str, timeout: float = 900, **kw) -> dict:
        with self._lock:
            if (self.cpu_since and time.time() - self.cpu_since > GPU_RETRY_S and not self._pending
                    and self.proc is not None and self.proc.poll() is None):
                self._quit_locked()   # a GPU broken by a sleep or a driver reset often works again: try it
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            rid = next(self._rid)
            fut: Future = Future()
            fut.proc = self.proc
            self._pending[rid] = fut
            try:
                self.proc.stdin.write((json.dumps({"op": op, "rid": rid, **kw}, ensure_ascii=False) + "\n").encode("utf-8"))
                self.proc.stdin.flush()
            except OSError as e:
                self._pending.pop(rid, None)
                return {"ok": False, "error": f"解析ワーカーへ送信できません: {e}"}
        # a model loading for the first time is downloaded first (the caption VLM is several GB): the time limit
        # counts from when loading ends, up to LOAD_LIMIT_S
        deadline, load_end = time.time() + timeout, time.time() + LOAD_LIMIT_S
        while True:
            try:
                return fut.result(timeout=max(0.1, min(5.0, deadline - time.time())))
            except TimeoutError:
                now = time.time()
                if self.status.startswith("loading") and now < load_end:
                    deadline = max(deadline, now + timeout)
                elif now >= deadline:
                    self._pending.pop(rid, None)
                    self.kill()
                    return {"ok": False, "error": "解析がタイムアウトしました"}

    def release_vram(self):
        """Before training: end the worker process so that *all* VRAM (torch + every cached
        ONNX session) is returned.  It restarts lazily on the next request; this planned
        restart does not count toward the crash-restart limit."""
        with self._lock:
            if self.proc is None or self.proc.poll() is not None:
                return
            self._quit_locked()
        bus.publish("worker", status=self.status)

    def _quit_locked(self):
        proc = self.proc
        try:
            proc.stdin.write(b'{"op":"quit","rid":0}\n')
            proc.stdin.flush()
            proc.wait(timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            from .training import kill_tree
            kill_tree(proc.pid)
        self.proc = None
        self.status = "stopped"
        self.gpu_error = self.cpu_since = None
        if self._restarts:
            self._restarts.pop()  # the next start is planned, not a crash recovery

    def unload(self):
        """Before a job that only needs some VRAM (ComfyUI generation): unload the models but keep the process.
        Starting it again (torch, CUDA, ONNX runtime) takes much longer than reloading the models."""
        if self.proc is None or self.proc.poll() is not None:
            return
        if not self.call("release", timeout=60).get("ok"):
            self.release_vram()

    def kill(self):
        if self.proc is not None and self.proc.poll() is None:
            from .training import kill_tree
            kill_tree(self.proc.pid)  # venv launcher + child interpreter


worker = WorkerClient()


WORKER_TASKS = ("生成候補のそろい具合の確認",)   # GPU task labels (server.py) that use this worker themselves


def _release_worker_vram():
    from .evaluate import comfy_user
    if gpu.owner in WORKER_TASKS:
        return
    if comfy_user(gpu.owner):
        worker.unload()
    else:   # training: every MB of VRAM, so the process ends
        worker.release_vram()


gpu.on_exclusive(_release_worker_vram)


CAPTION_BATCH = 8   # images per VLM call; the worker halves the batch when the GPU runs out of memory


class Analyzer:
    """Queues per project.  AI requests are processed one at a time on the GPU, highest priority
    first (e.g. the image the user is looking at)."""
    _multi_cache: tuple[float, list] = (0.0, [])   # (when, settings of the multi-character projects)

    def __init__(self, projects):
        self.projects = projects
        self.fast_pool = ThreadPoolExecutor(max_workers=max(2, min(4, (os.cpu_count() or 4) // 3)),
                                            thread_name_prefix="fastcheck")
        self.q: queue.PriorityQueue = queue.PriorityQueue()
        self._seq = itertools.count()
        self._queued: set[tuple] = set()
        self._forced: set[tuple] = set()   # (project, image) to analyse again without the cache
        self._qlock = threading.Lock()
        self.current: tuple | None = None
        threading.Thread(target=self._dispatch_loop, daemon=True, name="ai-dispatch").start()

    # ---------------------------------------------------------------- stage 1
    def submit_fast(self, project, image_id: int):
        self.fast_pool.submit(self._fast, project, image_id)

    def _fast(self, project, image_id):
        img = project.image(image_id)
        if img is None:
            return
        try:
            facts = fastcheck.analyze(project.source_path(img), project.thumbs / f"{image_id}.webp")
        except Exception as e:  # never leave an image looking "fine" because the check crashed
            facts = {"ok": False, "error": f"確認処理で例外: {type(e).__name__}: {e}"}
        fields = {"fast": facts, "fast_status": "done" if facts.get("ok") else "failed"}
        if facts.get("ok"):
            fields.update(width=facts["width"], height=facts["height"])
        else:
            fields.update(ai_status="skipped", caption_status="skipped")
        project.set_fields(image_id, **fields)
        bus.image_changed(project.id, image_id)
        if facts.get("ok"):
            self.submit_ai(project, image_id)

    # ---------------------------------------------------------------- stage 2/3
    def submit_ai(self, project, image_id: int, priority: int = 5, force: bool = False):
        img = project.image(image_id)
        if img is None or img["fast_status"] != "done":
            return
        key = ai_key(img["sha"], img.get("crop"))
        if not force and img.get("ai_key") == key and img["ai_status"] == "done":
            self.ensure_caption(project, image_id)
            self.ensure_mask(project, image_id)
            return
        cached = None if force else cache_get(key)
        if cached is not None:
            project.set_fields(image_id, ai=cached, ai_key=key, ai_status="done", ai_error=None)
            bus.image_changed(project.id, image_id)
            self.ensure_caption(project, image_id, force=_other_image(img.get("ai_key"), key))
            self.ensure_mask(project, image_id)
            self.after_ai(project)
            return
        project.set_fields(image_id, ai_status="pending")
        if force:
            with self._qlock:
                self._forced.add((project.id, image_id))
        self._enqueue(priority, ("analyze", project.id, image_id))

    def _enqueue(self, priority, item):
        with self._qlock:
            if item in self._queued:
                if priority < 5:  # re-prioritise: push again with better priority; stale copy is skipped
                    self.q.put((priority, next(self._seq), item))
                return
            self._queued.add(item)
            self.q.put((priority, next(self._seq), item))

    def forget_project(self, pid: str, wait: float = 60):
        """Drop a project's queued work and wait until the image being processed (if it is theirs) is done."""
        with self._qlock:
            self._queued = {it for it in self._queued if it[1] != pid}
        end = time.time() + wait
        while self.current and self.current[1] == pid and time.time() < end:
            time.sleep(0.1)

    def prioritize(self, project, image_ids):
        for i in image_ids:
            img = project.image(i)
            if img and img["ai_status"] == "pending":
                self._enqueue(1, ("analyze", project.id, i))

    @staticmethod
    def _caption_plan(meta: dict, img: dict) -> tuple[str, str, str | None]:
        """How this image's auto caption is made now: ("now", key, text) when it can be made at once (tags,
        or the shot of generated material), ("vlm", key, None) when the VLM has to write it."""
        style = captions.project_style(meta)
        kind, outfit, trig = meta["kind"], meta.get("outfit_mode", "include"), meta["trigger"]
        concept = meta.get("concept_words") or []
        key = captions.policy_key(kind, outfit, style, trig, concept, captions.concept_target(meta))
        mono = kind != "style" and captions.is_monochrome(img)
        mode = meta.get("caption_mode") or captions.default_caption_mode(meta)
        text = None
        if style == "tags":
            text = captions.tag_caption(img["ai"], kind, outfit, trig, concept)
        elif mode == "tags":
            key += ":t"
            text = captions.tag_sentence(img["ai"], kind, outfit, trig, concept)
        elif mode == "gen" and (img.get("gen") or {}).get("shot"):
            from .generate import gen_caption
            text = gen_caption(img["gen"]["shot"], kind, outfit, trig, captions.clothing_tags(img["ai"]))
            if text:
                key += ":g"
        if text is None:
            return "vlm", key + (":mono" if mono else ""), None
        if mono:
            text = captions.mark_monochrome(text, kind, style)
        return "now", key + (":mono" if mono else ""), text

    def ensure_caption(self, project, image_id: int, force: bool = False):
        meta = project.get_meta()
        img = project.image(image_id)
        if img is None or not img.get("ai"):
            return
        how, img_key, text = self._caption_plan(meta, img)
        if not force and img.get("caption_key") == img_key and img["caption_status"] == "done":
            return
        if how == "now":
            project.set_auto_caption(image_id, text, img_key)
            bus.image_changed(project.id, image_id)
            return
        style = captions.project_style(meta)
        mono = meta["kind"] != "style" and captions.is_monochrome(img)
        key = captions.policy_key(meta["kind"], meta.get("outfit_mode", "include"), style, meta["trigger"],
                                  meta.get("concept_words") or [], captions.concept_target(meta))
        ckey = f"cap:{key}:{img['sha'][:24]}:{img.get('crop')}"
        cached = cache_get(ckey)
        if cached is not None:
            text = captions.mark_monochrome(cached["text"], meta["kind"], style) if mono else cached["text"]
            project.set_auto_caption(image_id, text, img_key)
            bus.image_changed(project.id, image_id)
            return
        project.set_fields(image_id, caption_status="pending")
        self._enqueue(7, ("caption", project.id, image_id))

    def carry_trigger(self, project, old_meta: dict):
        """After only the trigger changed: the VLM captions written for the old trigger are kept with the word
        replaced, instead of writing every caption again (the instructions differ only in that name)."""
        meta = project.get_meta()
        old, new = old_meta.get("trigger") or "", meta["trigger"]
        same = ("kind", "outfit_mode", "model_id", "concept_words", "concept_target", "caption_mode")
        if not old or old == new or any(old_meta.get(k) != meta.get(k) for k in same):
            return
        style = captions.project_style(meta)
        if style == "tags":
            return
        args = (meta["kind"], meta.get("outfit_mode", "include"), style)
        k_old = captions.policy_key(*args, old, meta.get("concept_words") or [], captions.concept_target(meta))
        k_new = captions.policy_key(*args, new, meta.get("concept_words") or [], captions.concept_target(meta))
        for img in project.images():
            tail = f"{img['sha'][:24]}:{img.get('crop')}"
            if cache_get(f"cap:{k_new}:{tail}") is None:
                prev = cache_get(f"cap:{k_old}:{tail}")
                if prev is not None:
                    cache_put(f"cap:{k_new}:{tail}", {"text": captions.replace_trigger(prev["text"], old, new)})

    # ---------------------------------------------------------------- masks (masked training, masks.py)
    def ensure_mask(self, project, image_id: int, mode: str | None = None, force: bool = False):
        """Queue the image's mask when the project trains with masks (or for `mode`, e.g. a multi-character
        project asking for its members' images).  Needs the AI analysis (photo or illustration)."""
        from . import masks
        if mode is None:
            mode = self.mask_mode(project)
            if not mode:
                return
        img = project.image(image_id)
        if img is None or img["removed"] or not img.get("ai"):
            return
        item = (f"mask:{mode}", project.id, image_id)
        # pending / running left over from before a restart are queued again
        if not force and (masks.state(img, mode)["status"] in ("done", "none", "failed") or item in self._queued
                          or self.current == item):
            return
        self._set_mask(project, img, mode, {"status": "pending"})
        if force:
            with self._qlock:
                self._forced.add((project.id, image_id, mode))
        self._enqueue(8, item)

    def mask_mode(self, project) -> str | None:
        """The mask mode the project's images need: its own, or for a character project, that of a multi-character
        project it is a member of (images added to a member later are masked too)."""
        from . import masks, multichar
        from .models import registry
        meta = project.get_meta()
        mode = masks.active(meta, (registry.get(meta.get("model_id") or "") or {}).get("family"))
        if mode or meta.get("kind") != "character" or multichar.is_multi(meta):
            return mode
        for pm in self._multi_metas():
            if project.id in multichar.member_ids(pm):
                mode = masks.active(pm, (registry.get(pm.get("model_id") or "") or {}).get("family"))
                if mode:
                    return mode
        return None

    def _multi_metas(self) -> list[dict]:
        """Settings of the multi-character projects (looked up again after a few seconds: asked once per image)."""
        from . import multichar
        now = time.time()
        if now - self._multi_cache[0] < 5:
            return self._multi_cache[1]
        out = []
        for d in self.projects.folders():
            try:
                m = self.projects.get(d.name).get_meta()
            except Exception:   # a project being deleted or a broken one: not a member's owner
                continue
            if multichar.is_multi(m):
                out.append(m)
        self._multi_cache = (now, out)
        return out

    @staticmethod
    def _set_mask(project, img, mode, st):
        from .masks import MASK_VERSION
        cur = dict(img.get("mask") or {})
        cur[mode] = {**st, "version": MASK_VERSION}
        project.set_fields(img["id"], mask=cur)
        bus.image_changed(project.id, img["id"])

    def _do_mask(self, project, image_id, mode):
        from . import masks
        img = project.image(image_id)
        if img is None or img["removed"] or not img.get("ai"):
            return
        crop = masks.crop_of(img, mode)
        out = masks.mask_file(img["sha"], mode, crop)
        ckey = (f"mask:{masks.MASK_VERSION}:{mode}:{img['sha'][:24]}:{(img['ai'].get('domain') or {}).get('label')}"
                + masks.crop_tag(crop))
        with self._qlock:
            forced = (project.id, image_id, mode) in self._forced
            self._forced.discard((project.id, image_id, mode))
        cached = None if forced else cache_get(ckey)
        if cached is not None and (cached.get("status") != "done" or out.exists()):
            # the same file (and crop) was masked before
            self._set_mask(project, img, mode, {**cached, "crop": crop})
            return
        self._set_mask(project, img, mode, {"status": "running", "crop": crop})
        out.parent.mkdir(parents=True, exist_ok=True)
        res = worker.call("mask", path=str(project.source_path(img)), mode=mode, crop=crop,
                          domain=((img["ai"].get("domain") or {}).get("label")), out=str(out), timeout=600)
        img = project.image(image_id) or img
        if res.get("ok"):
            cache_put(ckey, res["result"])
            self._set_mask(project, img, mode, {**res["result"], "crop": crop})
        else:
            self._set_mask(project, img, mode, {"status": "failed", "error": res.get("error"), "crop": crop})
        if masks.crop_of(img, mode) != crop and not img["removed"]:
            # cropped again meanwhile (ensure_mask skipped it: this item was running): a mask for the new crop
            self._set_mask(project, img, mode, {"status": "pending", "crop": masks.crop_of(img, mode)})
            self._enqueue(8, (f"mask:{mode}", project.id, image_id))

    def queue_size(self) -> int:
        # the priority queue may hold stale duplicates after re-prioritisation; count real pending items
        with self._qlock:
            return len(self._queued)

    def _more_captions(self, pid: str) -> list[int]:
        """Further queued captions of the same project, taken out of the queue to be written in one batch."""
        with self.q.mutex:
            waiting = sorted(self.q.queue)
        out = []
        with self._qlock:
            for _p, _s, it in waiting:
                if len(out) >= CAPTION_BATCH - 1:
                    break
                if it[0] == "caption" and it[1] == pid and it in self._queued:
                    self._queued.discard(it)   # its heap entry is skipped as stale later
                    out.append(it[2])
        return out

    def _dispatch_loop(self):
        while True:
            prio, _seq, item = self.q.get()
            with self._qlock:
                if item not in self._queued:
                    continue  # stale duplicate entry
                self._queued.discard(item)
            op, pid, image_id = item
            if op == "caption":
                image_id = [image_id] + self._more_captions(pid)
            try:
                self._dispatch_one(op, pid, image_id, item)
            except Exception as e:   # one bad item must not stop the queue for the rest of the session
                # RuntimeError carries a message for the user (e.g. the worker stopped restarting)
                self._mark_failed(op, pid, image_id, str(e) if isinstance(e, RuntimeError) else
                                  f"解析中にエラーが起きました: {type(e).__name__}: {e}")

    def _mark_failed(self, op, pid, image_id, message):
        try:
            project = self.projects.get(pid)
            if op == "analyze":
                project.set_fields(image_id, ai_status="failed", ai_error=message, **_no_caption(project.image(image_id)))
                bus.image_changed(pid, image_id)
            elif op.startswith("mask:"):
                img = project.image(image_id)
                if img:
                    self._set_mask(project, img, op.split(":", 1)[1], {"status": "failed", "error": message})
            else:
                for i in image_id:
                    project.set_fields(i, caption_status="failed")
                    bus.image_changed(pid, i)
        except Exception:
            pass   # the project is gone, or the database cannot be written either

    def _dispatch_one(self, op, pid, image_id, item):
        try:
            project = self.projects.get(pid)
        except KeyError as e:
            from .store import ProjectMoving
            if isinstance(e, ProjectMoving):   # being moved to another place: do it once the move is over
                threading.Timer(10, self._enqueue, (5, item)).start()
            return
        if not worker.available():
            if op.startswith("mask:"):
                self._mark_failed(op, pid, image_id, "解析環境が未導入です（初回セットアップを実行してください）")
                return
            msg = "解析環境が未導入です（初回セットアップを実行してください）"
            if op == "analyze":
                project.set_fields(image_id, ai_status="failed", ai_error=msg, **_no_caption(project.image(image_id)))
                bus.image_changed(pid, image_id)
            else:   # captions: the analysis itself is fine
                self._mark_failed(op, pid, image_id, msg)
            return
        if not gpu.begin_analysis(timeout=0):
            bus.publish("analysis", state="paused", reason=gpu.owner)
            gpu.begin_analysis()
            bus.publish("analysis", state="running")
        self.current = item
        try:
            if op == "analyze":
                self._do_analyze(project, image_id)
            elif op.startswith("mask:"):
                self._do_mask(project, image_id, op.split(":", 1)[1])
            else:
                self._do_captions(project, image_id)
        finally:
            self.current = None
            gpu.end_analysis()
            bus.publish("analysis", state="running", queue=self.queue_size())

    def _do_analyze(self, project, image_id):
        img = project.image(image_id)
        if img is None or img["removed"]:
            return
        key = ai_key(img["sha"], img.get("crop"))
        recaption = _other_image(img.get("ai_key"), key)
        with self._qlock:
            forced = (project.id, image_id) in self._forced
            self._forced.discard((project.id, image_id))
        # the same file (e.g. added twice) may have been analysed while this one was queued
        cached = None if forced else cache_get(key)
        if cached is not None:
            project.set_fields(image_id, ai=cached, ai_key=key, ai_status="done", ai_error=None)
            bus.image_changed(project.id, image_id)
            self.ensure_caption(project, image_id, force=recaption)
            self.ensure_mask(project, image_id)
            self.after_ai(project)
            return
        project.set_fields(image_id, ai_status="running")
        bus.image_changed(project.id, image_id)
        res = worker.call("analyze", path=str(project.source_path(img)), crop=img.get("crop"))
        cur = project.image(image_id)
        if cur is None or ai_key(cur["sha"], cur.get("crop")) != key:
            return  # crop changed meanwhile; a new request has been queued
        if res.get("ok"):
            cache_put(key, res["result"])
            project.set_fields(image_id, ai=res["result"], ai_key=key, ai_status="done", ai_error=None)
            bus.image_changed(project.id, image_id)
            self.ensure_caption(project, image_id, force=recaption)
            self.ensure_mask(project, image_id)
            self.after_ai(project)
        else:
            project.set_fields(image_id, ai_status="failed", ai_error=res.get("error"), **_no_caption(cur))
            bus.image_changed(project.id, image_id)

    def _do_captions(self, project, image_ids: list[int]):
        meta = project.get_meta()
        imgs = []
        for i in image_ids:
            img = project.image(i)
            if img is None or img["removed"] or not img.get("ai"):
                continue
            if self._caption_plan(meta, img)[0] == "now":   # the caption mode changed while it was queued
                self.ensure_caption(project, i)
                continue
            imgs.append(img)
        if not imgs:
            return
        style = captions.project_style(meta)
        concept = meta.get("concept_words") or []
        key = captions.policy_key(meta["kind"], meta.get("outfit_mode", "include"), style, meta["trigger"], concept,
                                  captions.concept_target(meta))
        for img in imgs:
            project.set_fields(img["id"], caption_status="running")
            bus.image_changed(project.id, img["id"])
        instr = captions.nl_instruction(meta["kind"], meta.get("outfit_mode", "include"), meta["trigger"], concept,
                                        captions.concept_target(meta))
        res = worker.call("captions", paths=[str(project.source_path(img)) for img in imgs],
                          crops=[img.get("crop") for img in imgs], instruction=instr, timeout=300 + 120 * len(imgs))
        texts = (res.get("texts") or []) if res.get("ok") else []
        for n, img in enumerate(imgs):
            cur = project.image(img["id"])
            # changed meanwhile (crop, removal, an undo that put a cached caption back): the result is only cached
            stale = (cur is None or cur["removed"] or cur["sha"] != img["sha"] or cur.get("crop") != img.get("crop")
                     or cur["caption_status"] != "running")
            if n < len(texts) and texts[n] is not None:   # None: the worker could not read that image
                text = captions.finalize_nl(texts[n], meta["kind"], meta["trigger"])
                cache_put(f"cap:{key}:{img['sha'][:24]}:{img.get('crop')}", {"text": text})
                if stale:
                    continue
                mono = meta["kind"] != "style" and captions.is_monochrome(img)
                if mono:
                    text = captions.mark_monochrome(text, meta["kind"], style)
                project.set_auto_caption(img["id"], text, key + (":mono" if mono else ""))
            elif not stale:
                project.set_fields(img["id"], caption_status="failed")
            bus.image_changed(project.id, img["id"])

    # ---------------------------------------------------------------- dataset-level identity
    def after_ai(self, project):
        """Debounced recomputation of character identity once analyses settle."""
        if project.meta("kind") != "character":
            return
        t = getattr(project, "_ident_timer", None)
        if t is not None:
            t.cancel()
        project._ident_timer = threading.Timer(2.0, self.compute_identity, args=(project,))
        project._ident_timer.daemon = True
        project._ident_timer.start()

    def compute_identity(self, project):
        imgs = [i for i in project.images() if i.get("ai") and i["ai_status"] == "done"]
        meta = project.get_meta()
        refs = [r for r in (meta.get("reference_ids") or []) if any(i["id"] == r for i in imgs)]
        anime = [i for i in imgs if i["ai"].get("ccip")]
        ident = {}
        sig = hashlib.sha1(json.dumps([[i["id"], i["ai_key"]] for i in imgs] + [refs]).encode()).hexdigest()
        if (meta.get("identity") or {}).get("sig") == sig:
            return
        if len(anime) >= 2:
            feats = [i["ai"]["ccip"] for i in anime]
            if not gpu.begin_analysis(timeout=0):
                return  # GPU busy; recomputed after the next analysis
            try:
                res = worker.call("ccip_diff", feats=feats, refs=feats, timeout=300)
            except RuntimeError:
                return   # the worker stopped restarting: the image list shows that already
            finally:
                gpu.end_analysis()
            if not res.get("ok"):
                # not judged is not "the same character": shown on each image, and tried again next time (no sig)
                project.set_meta(identity={"sig": None, "map": {str(i["id"]): {"method": "failed"} for i in anime}})
                bus.publish("identity", project=project.id)
                return
            if res.get("ok"):
                mat = res["matrix"]
                ids = [i["id"] for i in anime]
                auto = False
                if not refs:
                    # automatic reference: the image most similar to the others (medoid)
                    best = min(range(len(ids)), key=lambda a: sorted(mat[a])[len(ids) // 2])
                    refs = [ids[best]]
                    auto = True
                ref_idx = [ids.index(r) for r in refs if r in ids]
                if not ref_idx:
                    # the chosen references have no CCIP (e.g. a photo): compare with the automatic one instead
                    ref_idx = [min(range(len(ids)), key=lambda a: sorted(mat[a])[len(ids) // 2])]
                for a, iid in enumerate(ids):
                    ident[iid] = {"method": "ccip", "diff": round(min(mat[a][r] for r in ref_idx), 4)}
                project.set_meta(auto_reference=refs if auto else [])
        photos = [i for i in imgs if not i["ai"].get("ccip") and i["ai"].get("subject_embed")]
        if len(photos) >= 4:
            import numpy as np
            E = np.array([i["ai"]["subject_embed"] for i in photos])
            c = E.mean(0)
            c /= np.linalg.norm(c)
            s = E @ c
            mu, sd = s.mean(), s.std() + 1e-6
            for i, v in zip(photos, s):
                ident[i["id"]] = {"method": "embed", "sim": round(float(v), 4), "outlier": bool(v < mu - 2.5 * sd and v < 0.75)}
        project.set_meta(identity={"sig": sig, "map": {str(k): v for k, v in ident.items()}})
        bus.publish("identity", project=project.id)
