"""LoRA creation tasks: material generation, analysis and captions, then training, registered ahead and run
one task after another.

A task only drives the existing steps: the generation batch and the training run go through the serial GPU
queue like the buttons, and the analysis is the normal background analysis.  The task waits for each step and
starts the next one, so several projects can be registered and left to run unattended.  Training uses the
project's settings at the time the task reaches it (the material does not exist before).

Tasks are kept in data/pipeline.json, so registered tasks survive a restart; a task that was generating
continues its batch, one whose training was interrupted is marked failed (the run can be resumed from the
training page).
"""
from __future__ import annotations

import json
import threading
import time
import traceback
import uuid

from . import config
from .events import bus

FILE = config.DATA_DIR / "pipeline.json"
ACTIVE = ("generating", "analyzing", "training")
ENDED = ("done", "failed", "cancelled")
KEEP_ENDED = 30            # finished tasks shown in the list
ANALYSIS_STALL_S = 300     # analysis queue empty and nothing changed this long: the remaining images are not coming
POLL_S = 2.0
RUN_ACTIVE = ("queued", "preparing", "training", "evaluating")


class Hooks:
    """Set by server.py (the steps live there)."""
    queue_batch = None      # (project, batch_id) -> None
    stop_batch = None       # (project, batch_id) -> None
    import_batch = None     # (project, batch_id) -> number of images added
    start_training = None   # (project, name) -> run id; ValueError with the reason when it cannot start
    stop_run = None         # (project, run_id) -> None
    analysis_busy = None    # () -> bool: the analysis queue still has items


hooks = Hooks()


class Pipeline:
    def __init__(self, projects):
        self.projects = projects
        self._lock = threading.RLock()
        self.tasks: list[dict] = []
        self._pending_seen: dict[str, tuple[int, float]] = {}   # task id -> (images left, since)
        self._thread = None
        # paused: the stage running now (generation, analysis, training) goes on, but no task moves to its
        # next stage and no waiting task starts (kept across restarts)
        self.paused = bool(config.settings().get("pipeline_paused", False))
        # read now, not only in recover() (which runs after the startup scan): a task added or cleared in the
        # meantime would otherwise save a list without the saved tasks
        self.load()

    def set_paused(self, on: bool):
        self.paused = bool(on)
        try:
            config.settings().set("pipeline_paused", self.paused)
        except OSError:
            pass
        bus.publish("pipeline", task=None)
        self.wake()

    # ---------------------------------------------------------------- storage
    def load(self):
        try:
            import json
            data = json.loads(FILE.read_text(encoding="utf-8")) if FILE.exists() else []
        except (OSError, ValueError):
            data = []
        with self._lock:
            self.tasks = [t for t in data if isinstance(t, dict) and t.get("id")]

    def _save(self):
        try:
            config.atomic_write_json(FILE, self.tasks)
        except OSError:
            pass

    def _changed(self, t: dict | None = None):
        self._save()
        bus.publish("pipeline", task=t["id"] if t else None)

    def _ended(self, t: dict) -> bool:
        with self._lock:
            return t["status"] in ENDED

    def _set(self, t: dict, **fields):
        with self._lock:
            if t["status"] in ENDED:   # cancelled while a step was running: the step's result is dropped
                return
            t.update(fields)
            if fields.get("status") in ENDED:
                t["ended"] = time.time()
                ended = [x for x in self.tasks if x["status"] in ENDED]
                for old in ended[:-KEEP_ENDED] if len(ended) > KEEP_ENDED else []:
                    self.tasks.remove(old)
            self._changed(t)

    # ---------------------------------------------------------------- API
    def list(self) -> list[dict]:
        with self._lock:
            return [dict(t) for t in self.tasks]

    def add(self, project, gen_count: int, name: str | None = None) -> dict:
        t = {"id": uuid.uuid4().hex[:10], "project": project.id, "name": (name or "").strip() or None,
             "gen_count": int(gen_count), "status": "waiting", "message": "順番待ち", "batch": None, "run": None,
             "created": time.time(), "started": None, "ended": None}
        with self._lock:
            self.tasks.append(t)
            self._changed(t)
        self.wake()
        return dict(t)

    def move(self, tid: str, to: str) -> bool:
        """Reorder among the waiting tasks: to = "up" / "down" / "top"."""
        with self._lock:
            waiting = [t for t in self.tasks if t["status"] == "waiting"]
            i = next((n for n, t in enumerate(waiting) if t["id"] == tid), None)
            if i is None:
                return False
            t = waiting.pop(i)
            n = 0 if to == "top" else max(0, i - 1) if to == "up" else min(len(waiting), i + 1)
            waiting.insert(n, t)
            rest = [x for x in self.tasks if x["status"] != "waiting"]
            self.tasks = rest + waiting
            self._changed(t)
        return True

    def cancel(self, tid: str) -> str:
        with self._lock:
            t = next((x for x in self.tasks if x["id"] == tid), None)
            if not t or t["status"] in ENDED:
                raise KeyError(tid)
            st = t["status"]
            self._set(t, status="cancelled", message="取り消しました")
        p = self._project(t)
        if p is not None and st == "generating" and t.get("batch"):
            hooks.stop_batch(p, t["batch"])
            return "取り消しました。素材の生成も中止します（生成済みの画像は残ります）。"
        if p is not None and st == "training" and t.get("run"):
            r = hooks.stop_run(p, t["run"]) or {}
            return "取り消しました。" + (r.get("message") or "学習は次の保存地点で止まります。")
        return "取り消しました。"

    def remove_ended(self) -> int:
        with self._lock:
            n = len([t for t in self.tasks if t["status"] in ENDED])
            self.tasks = [t for t in self.tasks if t["status"] not in ENDED]
            self._changed()
        return n

    # ---------------------------------------------------------------- processing
    def start(self):
        if self._thread is None:
            self._wake = threading.Event()
            self._thread = threading.Thread(target=self._loop, daemon=True, name="pipeline")
            self._thread.start()

    def wake(self):
        if self._thread is not None:
            self._wake.set()

    def _loop(self):
        while True:
            try:
                self.step()
            except Exception:
                _log(traceback.format_exc())
            self._wake.wait(POLL_S)
            self._wake.clear()

    def _project(self, t):
        try:
            return self.projects.get(t["project"])
        except KeyError:
            return None

    def step(self):
        """Advance the current task by at most one step (called every few seconds)."""
        with self._lock:
            t = next((x for x in self.tasks if x["status"] in ACTIVE), None) \
                or next((x for x in self.tasks if x["status"] == "waiting"), None)
        if t is None:
            return
        p = self._project(t)
        if p is None:
            self._set(t, status="failed", message="プロジェクトが見つかりません（削除された可能性があります）")
            return
        try:
            getattr(self, "_" + t["status"])(t, p)
        except Exception as e:
            _log(traceback.format_exc())
            self._set(t, status="failed", message=f"内部エラー: {e}")

    def _held(self, t, msg: str) -> bool:
        if self.paused and t["message"] != msg:
            self._set(t, message=msg)
        return self.paused

    def _waiting(self, t, p):
        if self.paused:
            return
        self._set(t, started=time.time())
        if t["gen_count"] > 0:
            from . import generate
            from .models import registry
            recipe = p.meta("gen_recipe") or {}
            errs = generate.check_recipe(recipe, p.meta("kind"), p)["errors"]
            if errs or not registry.get(recipe.get("gen_model_id") or ""):
                self._set(t, status="failed", message="素材の生成を始められません: " + (" / ".join(errs) or "生成モデルが見つかりません"))
                return
            bid = generate.new_batch(p, recipe, t["gen_count"])
            hooks.queue_batch(p, bid)
            if self._ended(t):   # cancelled while the batch was being made
                hooks.stop_batch(p, bid)
                return
            self._set(t, status="generating", batch=bid, message=f"素材を生成します（{t['gen_count']} 枚、順番待ち）")
            return
        self._set(t, status="analyzing", message="解析と説明文の作成を待っています")

    def _generating(self, t, p):
        rows = p.q("SELECT status, message FROM gen_batches WHERE id=?", (t["batch"],))
        if not rows:
            self._set(t, status="failed", message="素材の生成の記録が見つかりません（削除された可能性があります）")
            return
        st, msg = rows[0]["status"], rows[0]["message"] or ""
        if st in ("queued", "running"):
            if msg and f"素材の生成: {msg}" != t["message"]:
                self._set(t, message=f"素材の生成: {msg}")
            return
        if st != "done":
            self._set(t, status="failed", message=f"素材の生成が終わりませんでした（{msg}）。素材画面の「AI で生成」から再開できます。")
            return
        if self._held(t, "素材の生成が終わりました。一時停止中のため、素材に加える前で止めています"):
            return
        n = hooks.import_batch(p, t["batch"])
        if n == 0:
            self._set(t, status="failed", message="素材に加えられる生成画像がありませんでした（すべて選択から外れています）。")
            return
        self._set(t, status="analyzing", imported=n, message=f"生成した {n} 枚を素材に加えました。解析と説明文の作成を待っています")

    def _analyzing(self, t, p):
        # a count, not p.images(): this runs every POLL_S seconds and the rows carry the large analysis results
        left = p.q("SELECT COUNT(*) AS n FROM images WHERE removed=0 AND (fast_status='pending' OR "
                   "ai_status IN ('pending','running') OR caption_status IN ('pending','running'))")[0]["n"]
        now = time.time()
        if left:
            seen = self._pending_seen.get(t["id"])
            busy = hooks.analysis_busy()
            if seen is None or seen[0] != left or busy:
                # the stall timer starts when the analysis has nothing left to do
                self._pending_seen[t["id"]] = (left, now if seen is None or seen[0] != left or busy else seen[1])
                msg = f"解析と説明文の作成を待っています（残り {left} 枚）"
                if msg != t["message"]:
                    self._set(t, message=msg)
                return
            if now - seen[1] < ANALYSIS_STALL_S:
                return
            # nothing is analysing these any more (e.g. the analysis AI could not start): train with what there is
        if self._held(t, "解析が終わりました。一時停止中のため、学習を始める前で止めています"):
            return
        self._pending_seen.pop(t["id"], None)
        try:
            rid = hooks.start_training(p, t.get("name"))
        except ValueError as e:
            self._set(t, status="failed", message=f"学習を始められません: {e}")
            return
        if self._ended(t):   # cancelled while the run was being reserved
            hooks.stop_run(p, rid)
            return
        self._set(t, status="training", run=rid, message=f"学習 #{rid} を開始します" +
                  (f"（解析が終わらなかった画像 {left} 枚はそのまま使います）" if left else ""))

    def _training(self, t, p):
        # the status columns only, not the whole frozen snapshot (config), every POLL_S seconds
        rows = p.q("SELECT status, progress, error FROM runs WHERE id=?", (t["run"],))
        if not rows:
            self._set(t, status="failed", message="学習の記録が見つかりません（削除された可能性があります）")
            return
        # error is stored as JSON text (Project.run leaves it as text too, so .get() on it failed before)
        run = {"status": rows[0]["status"], "progress": _json(rows[0]["progress"]), "error": _json(rows[0]["error"])}
        st = run["status"]
        if st in RUN_ACTIVE:
            msg = f"学習 #{t['run']}: " + ((run.get("progress") or {}).get("message") or "待機中")
            if msg != t["message"]:
                self._set(t, message=msg)
            return
        if st == "done":
            self._set(t, status="done", message=f"完了しました（学習 #{t['run']}）")
        elif st == "cancelled":
            self._set(t, status="cancelled", message=f"学習 #{t['run']} が中止されました")
        else:
            err = (run.get("error") or {}).get("message") or "失敗しました"
            self._set(t, status="failed", message=f"学習 #{t['run']}: {err}")

    # ---------------------------------------------------------------- startup
    def recover(self):
        """After a restart (server.resume_pending_on_startup has already paused batches and restored the
        reserved runs): a task that was generating continues its batch."""
        self.load()
        for t in self.list():
            if t["status"] != "generating":
                continue
            p = self._project(t)
            if p is None:
                continue
            rows = p.q("SELECT status FROM gen_batches WHERE id=?", (t["batch"],))
            if rows and rows[0]["status"] == "paused":
                from . import generate
                generate._set_status(p, t["batch"], "queued", "順番待ち")
                hooks.queue_batch(p, t["batch"])
        self.start()


def _json(v):
    try:
        return json.loads(v) if v else None
    except (TypeError, ValueError):
        return None


def _log(tb: str):
    try:
        with open(config.DATA_DIR / "server_errors.log", "a", encoding="utf-8") as f:
            f.write(time.strftime("%F %T ") + "LoRA 作成タスク\n" + tb + "\n")
    except OSError:
        pass
