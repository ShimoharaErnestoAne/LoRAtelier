"""Per-project persistent store (SQLite, WAL) with undo/redo history.

Original images are copied into <project>/sources and never modified.  Everything the
user edits (adoption, crop, caption) is stored as edit information in project.db.
SQLite transactions give crash-safe autosave: a write either commits or is rolled back.
"""
from __future__ import annotations

import json
import re
import secrets
import shutil
import sqlite3
import threading
import time
import unicodedata
from pathlib import Path

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS images (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  sha TEXT NOT NULL,
  src_rel TEXT NOT NULL,
  orig_name TEXT NOT NULL,
  origin_path TEXT,
  added_at REAL NOT NULL,
  width INTEGER, height INTEGER,
  fast TEXT,              -- JSON: mechanical facts (decode, size, hashes, blur, exposure...)
  fast_status TEXT NOT NULL DEFAULT 'pending',   -- pending/done/failed
  ai TEXT,                -- JSON: AI analysis result
  ai_key TEXT,            -- cache key (analyzer version + crop) the ai result belongs to
  ai_status TEXT NOT NULL DEFAULT 'pending',     -- pending/running/done/failed
  ai_error TEXT,
  decision TEXT,          -- NULL = follow automatic verdict, 'use', 'exclude'
  crop TEXT,              -- JSON [x0,y0,x1,y1] in source pixels, user-set; NULL = none
  caption TEXT NOT NULL DEFAULT '',
  caption_auto TEXT,      -- last auto-generated caption (kept for "reset to auto")
  caption_key TEXT,       -- policy key the auto caption was generated with
  caption_edited INTEGER NOT NULL DEFAULT 0,
  caption_status TEXT NOT NULL DEFAULT 'pending', -- pending/running/done/failed
  removed INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS images_sha ON images(sha);
CREATE TABLE IF NOT EXISTS history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL, label TEXT NOT NULL, ops TEXT NOT NULL, undone INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created REAL NOT NULL, status TEXT NOT NULL, name TEXT NOT NULL,
  config TEXT NOT NULL, run_dir TEXT NOT NULL,
  progress TEXT, error TEXT, ended REAL, parent_run INTEGER
);
CREATE TABLE IF NOT EXISTS checkpoints (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL, step INTEGER, path TEXT NOT NULL, kind TEXT NOT NULL, created REAL NOT NULL,
  UNIQUE(run_id, path)
);
CREATE TABLE IF NOT EXISTS evals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL, created REAL NOT NULL, status TEXT NOT NULL,
  spec TEXT NOT NULL, results TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS gen_batches (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created REAL NOT NULL, status TEXT NOT NULL, message TEXT, spec TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS gen_images (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  batch_id INTEGER NOT NULL, idx INTEGER NOT NULL, file TEXT NOT NULL, seed INTEGER, prompt TEXT,
  shot TEXT, picked INTEGER NOT NULL DEFAULT 1, image_id INTEGER, created REAL NOT NULL,
  deleted INTEGER NOT NULL DEFAULT 0,   -- file deleted by the user; the row stays so resume skips it
  UNIQUE(batch_id, idx)
);
CREATE TABLE IF NOT EXISTS exports (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL, created REAL NOT NULL, ckpt_path TEXT NOT NULL, dest TEXT NOT NULL, info TEXT
);
"""

UNDOABLE_FIELDS = {"decision", "crop", "caption", "caption_edited", "removed", "mask_off"}
JSON_FIELDS = {"fast", "ai", "crop", "gen", "mask"}
PARSE_CACHED = {"fast", "ai"}   # the large ones (embeddings); see Project._row


def _slug(text: str) -> str:
    """ASCII-only folder id (the display name stays in meta) so ids are safe in URLs,
    engine output names and ComfyUI file prefixes."""
    text = unicodedata.normalize("NFKC", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("._-")
    return (text[:30] or "p") + "_" + secrets.token_hex(2)


def default_trigger(kind: str, target: str = "body") -> str:
    """Trigger words that are unlikely to collide with real vocabulary (e.g. 'chr_k7q2')."""
    alphabet = "abcdefghjkmnpqrstuvwxyz23456789"
    tail = "".join(secrets.choice(alphabet) for _ in range(4))
    if kind == "concept":
        from .captions import CONCEPT_TARGETS
        return (CONCEPT_TARGETS.get(target) or CONCEPT_TARGETS["body"])["trigger"] + tail
    return {"character": "chr_", "enhance": "enh_"}.get(kind, "sty_") + tail


class Project:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.id = self.path.name
        self.sources = self.path / "sources"
        self.thumbs = self.path / "thumbs"
        self.runs_dir = self.path / "runs"
        self.exports_dir = self.path / "exports"
        for d in (self.sources, self.thumbs, self.runs_dir, self.exports_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._parsed: dict[tuple[int, str], tuple[str, dict]] = {}
        self.db = sqlite3.connect(self.path / "project.db", check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        # WAL + NORMAL: a crash of the app never corrupts or half-applies an edit (each edit is one
        # transaction).  Only an OS crash / power loss can drop the last few commits.  FULL would
        # fsync on every keystroke-save, which costs 100ms+ per edit on an HDD.
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript("BEGIN;" + SCHEMA + "COMMIT;")
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(images)")}
        if "gen" not in cols:   # added with material generation: the shot an image was generated with
            self.db.execute("ALTER TABLE images ADD COLUMN gen TEXT")
        if "mask" not in cols:   # masked training: the masks made for the image (masks.py) and the user's "no mask"
            self.db.execute("ALTER TABLE images ADD COLUMN mask TEXT")
            self.db.execute("ALTER TABLE images ADD COLUMN mask_off INTEGER NOT NULL DEFAULT 0")
        gcols = {r[1] for r in self.db.execute("PRAGMA table_info(gen_images)")}
        if "deleted" not in gcols:
            self.db.execute("ALTER TABLE gen_images ADD COLUMN deleted INTEGER NOT NULL DEFAULT 0")
        # consistency check of generated candidates: CCIP feature, difference to the set's centre, and
        # whether the user set the pick by hand (then the automatic check never changes it)
        # prompt_check: what the tagger saw against the user's prompt (generate.check_prompt, 2026-11-02)
        for col, typ in (("ccip", "TEXT"), ("ccip_diff", "REAL"), ("pick_manual", "INTEGER NOT NULL DEFAULT 0"),
                         ("prompt_check", "TEXT")):
            if col not in gcols:
                self.db.execute(f"ALTER TABLE gen_images ADD COLUMN {col} {typ}")

    # ---------------------------------------------------------------- basics
    def close(self):
        with self._lock:
            self.db.close()

    def tx(self):
        """Context manager: BEGIN IMMEDIATE ... COMMIT/ROLLBACK under the project lock."""
        proj = self

        class _Tx:
            def __enter__(self_inner):
                proj._lock.acquire()
                try:
                    proj.db.execute("BEGIN IMMEDIATE")
                except BaseException:   # e.g. a locked or vanished database: __exit__ will not run, free the lock here
                    proj._lock.release()
                    raise
                return proj.db

            def __exit__(self_inner, exc_type, exc, tb):
                try:
                    if exc_type is not None:
                        proj.db.execute("ROLLBACK")
                    else:
                        try:
                            proj.db.execute("COMMIT")
                        except sqlite3.Error:
                            # e.g. disk full: do not leave the shared connection inside the transaction
                            try:
                                proj.db.execute("ROLLBACK")
                            except sqlite3.Error:
                                pass
                            raise
                finally:
                    proj._lock.release()
                return False

        return _Tx()

    def q(self, sql, args=()):
        with self._lock:
            return self.db.execute(sql, args).fetchall()

    # ---------------------------------------------------------------- meta
    def get_meta(self) -> dict:
        rows = self.q("SELECT key, value FROM meta")
        return {r["key"]: json.loads(r["value"]) for r in rows}

    def meta(self, key, default=None):
        rows = self.q("SELECT value FROM meta WHERE key=?", (key,))
        return json.loads(rows[0]["value"]) if rows else default

    def set_meta(self, **kv):
        with self.tx() as db:
            for k, v in kv.items():
                db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", (k, json.dumps(v, ensure_ascii=False)))

    # ---------------------------------------------------------------- images
    def _row(self, r) -> dict:
        d = dict(r)
        for k in JSON_FIELDS:
            if d.get(k):
                if k in PARSE_CACHED:
                    key = (d["id"], k)
                    hit = self._parsed.get(key)
                    if hit is None or hit[0] != d[k]:
                        hit = self._parsed[key] = (d[k], json.loads(d[k]))
                    d[k] = dict(hit[1])   # a fresh top level: a caller adding a key does not change the cache
                else:
                    d[k] = json.loads(d[k])
        return d

    def images(self, include_removed=False) -> list[dict]:
        sql = "SELECT * FROM images" + ("" if include_removed else " WHERE removed=0") + " ORDER BY id"
        return [self._row(r) for r in self.q(sql)]

    def image(self, image_id: int) -> dict | None:
        rows = self.q("SELECT * FROM images WHERE id=?", (image_id,))
        return self._row(rows[0]) if rows else None

    def source_path(self, img: dict) -> Path:
        return self.path / img["src_rel"]

    def add_image_file(self, data_path: Path, orig_name: str, sha: str, origin_path: str | None = None) -> tuple[int, bool]:
        """Register an image (already hashed).  Returns (id, is_new).  Exact duplicates of an
        existing, non-removed image are still registered (they are shown as duplicates and the
        user decides), but share the stored source file."""
        ext = Path(orig_name).suffix.lower() or ".bin"
        rel = f"sources/{sha[:20]}{ext}"
        dest = self.path / rel
        if not dest.exists():
            tmp = dest.with_name(dest.name + ".part")
            shutil.copyfile(data_path, tmp)
            tmp.replace(dest)
        with self.tx() as db:
            # re-adding an image that the user removed earlier restores it instead of duplicating
            row = db.execute("SELECT id FROM images WHERE sha=? AND removed=1", (sha,)).fetchone()
            if row:
                db.execute("UPDATE images SET removed=0 WHERE id=?", (row["id"],))
                return row["id"], False
            cur = db.execute(
                "INSERT INTO images(sha, src_rel, orig_name, origin_path, added_at) VALUES(?,?,?,?,?)",
                (sha, rel, orig_name, origin_path, time.time()),
            )
            return cur.lastrowid, True

    def set_fields(self, image_id: int, **fields):
        """Non-undoable system updates (analysis results etc.)."""
        if not fields:
            return
        cols, vals = [], []
        for k, v in fields.items():
            cols.append(f"{k}=?")
            vals.append(json.dumps(v, ensure_ascii=False) if k in JSON_FIELDS and v is not None else v)
        vals.append(image_id)
        with self.tx() as db:
            db.execute(f"UPDATE images SET {', '.join(cols)} WHERE id=?", vals)

    def set_auto_caption(self, image_id: int, caption: str, key: str) -> bool:
        """Store an auto caption.  Never overwrites a caption the user has edited:
        the check and the write happen in one transaction."""
        with self.tx() as db:
            row = db.execute("SELECT caption_edited FROM images WHERE id=?", (image_id,)).fetchone()
            if row is None:
                return False
            if row["caption_edited"]:
                db.execute(
                    "UPDATE images SET caption_auto=?, caption_key=?, caption_status='done' WHERE id=?",
                    (caption, key, image_id),
                )
                return False
            db.execute(
                "UPDATE images SET caption=?, caption_auto=?, caption_key=?, caption_status='done' WHERE id=?",
                (caption, caption, key, image_id),
            )
            return True

    # ---------------------------------------------------------------- user edits (undoable)
    def edit(self, label: str, changes: list[tuple[int, str, object]]) -> int | None:
        """Apply user edits [(image_id, field, value)] atomically and record them for undo."""
        ops = []
        with self.tx() as db:
            for image_id, field, value in changes:
                if field not in UNDOABLE_FIELDS:
                    raise ValueError(f"field not editable: {field}")
                row = db.execute(f"SELECT {field} FROM images WHERE id=?", (image_id,)).fetchone()
                if row is None:
                    continue
                before = row[field]
                after = json.dumps(value) if field in JSON_FIELDS and value is not None else value
                if before == after:
                    continue
                db.execute(f"UPDATE images SET {field}=? WHERE id=?", (after, image_id))
                ops.append({"id": image_id, "field": field, "before": before, "after": after})
            if not ops:
                return None
            db.execute("DELETE FROM history WHERE undone=1")
            cur = db.execute(
                "INSERT INTO history(ts, label, ops) VALUES(?,?,?)", (time.time(), label, json.dumps(ops, ensure_ascii=False))
            )
            return cur.lastrowid

    def _apply_history(self, undo: bool) -> dict | None:
        with self.tx() as db:
            if undo:
                row = db.execute("SELECT * FROM history WHERE undone=0 ORDER BY id DESC LIMIT 1").fetchone()
            else:
                row = db.execute("SELECT * FROM history WHERE undone=1 ORDER BY id ASC LIMIT 1").fetchone()
            if row is None:
                return None
            ops = json.loads(row["ops"])
            for op in (reversed(ops) if undo else ops):
                db.execute(f"UPDATE images SET {op['field']}=? WHERE id=?", (op["before"] if undo else op["after"], op["id"]))
            # a caption that is no longer the user's again shows the current auto caption: while it was
            # edited, new auto captions (e.g. after the trigger changed) were kept in caption_auto only
            for iid in sorted({o["id"] for o in ops if o["field"] in ("caption", "caption_edited")}):
                db.execute("UPDATE images SET caption=caption_auto WHERE id=? AND caption_edited=0 AND caption_auto IS NOT NULL",
                           (iid,))
            db.execute("UPDATE history SET undone=? WHERE id=?", (1 if undo else 0, row["id"]))
            return {"label": row["label"], "ids": sorted({o["id"] for o in ops}),
                    # only these can need analysis again (crop back, or an image back in use for masks)
                    "analyze_ids": sorted({o["id"] for o in ops if o["field"] in ("crop", "removed")})}

    def undo(self):
        return self._apply_history(undo=True)

    def redo(self):
        return self._apply_history(undo=False)

    def history_state(self) -> dict:
        u = self.q("SELECT label FROM history WHERE undone=0 ORDER BY id DESC LIMIT 1")
        r = self.q("SELECT label FROM history WHERE undone=1 ORDER BY id ASC LIMIT 1")
        return {"undo": u[0]["label"] if u else None, "redo": r[0]["label"] if r else None}

    # ---------------------------------------------------------------- runs
    def runs(self) -> list[dict]:
        out = []
        for r in self.q("SELECT * FROM runs ORDER BY id DESC"):
            d = dict(r)
            d["config"] = json.loads(d["config"])
            d["progress"] = json.loads(d["progress"]) if d["progress"] else None
            out.append(d)
        return out

    def run(self, run_id: int) -> dict | None:
        rows = self.q("SELECT * FROM runs WHERE id=?", (run_id,))
        if not rows:
            return None
        d = dict(rows[0])
        d["config"] = json.loads(d["config"])
        d["progress"] = json.loads(d["progress"]) if d["progress"] else None
        return d

    def checkpoints(self, run_id: int) -> list[dict]:
        return [dict(r) for r in self.q("SELECT * FROM checkpoints WHERE run_id=? ORDER BY step, id", (run_id,))]


class ProjectMoving(KeyError):
    """The project is being moved to another place (relocate.py): not usable until the move ends."""


class ProjectManager:
    def __init__(self, root: Path | None = None):
        """root: one fixed folder.  None: the folders of the settings page (config.storage_dirs), new projects
        going to the current one."""
        self._fixed = Path(root) if root is not None else None
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass   # e.g. the HDD set as the place is not connected: the projects elsewhere stay usable
        self._open: dict[str, Project] = {}
        self._moving: dict[str, str] = {}   # project id -> its name, while relocate.py moves it
        self._lock = threading.Lock()

    @property
    def root(self) -> Path:
        """Where new projects are made."""
        return self._fixed or config.storage_dir("projects")

    def roots(self) -> list[Path]:
        return [self._fixed] if self._fixed else config.storage_dirs("projects")

    def folders(self) -> list[Path]:
        """Every project folder, in all the places (a folder name found twice: the first place wins)."""
        out: dict[str, Path] = {}
        for r in self.roots():
            try:
                # ".moving_<id>" (a copy being made by relocate.py), ".trash" and ".importing" are not projects
                found = [d for d in r.iterdir() if not d.name.startswith(".") and (d / "project.db").exists()]
            except OSError:
                continue   # not there (yet), or its drive is not connected
            for d in found:
                out.setdefault(d.name, d)
        return list(out.values())

    def path_of(self, pid: str) -> Path | None:
        if not pid or pid.startswith("."):
            return None
        for r in self.roots():
            path = r / pid
            try:
                if (path / "project.db").exists() and path.resolve().parent == r.resolve():
                    return path
            except OSError:
                continue
        return None

    def list(self) -> list[dict]:
        def mtime(p: Path) -> float:
            try:
                return p.stat().st_mtime
            except OSError:   # deleted since folders() looked
                return 0.0

        out = []
        for d in sorted(self.folders(), key=mtime, reverse=True):
            try:
                p = self.get(d.name)
                m = p.get_meta()
                n = p.q("SELECT COUNT(*) c FROM images WHERE removed=0")[0]["c"]
                # edits are written to project.db (and its WAL), which does not always touch the folder
                updated = max(f.stat().st_mtime for f in (d, d / "project.db", d / "project.db-wal") if f.exists())
                out.append({"id": d.name, "name": m.get("name", d.name), "kind": m.get("kind"), "concept_target": m.get("concept_target"),
                            "enhance": m.get("enhance"),
                            "model_id": m.get("model_id"), "source": m.get("source", "images"), "images": n,
                            "updated": updated, "created": m.get("created") or d.stat().st_ctime})
            except ProjectMoving:
                out.append({"id": d.name, "name": self._moving.get(d.name) or d.name, "moving": True,
                            "error": "ほかの置き場所へ移動中です。終わるまで開けません。"})
            except KeyError:
                continue   # deleted (or moved away) since folders() looked
            except (sqlite3.Error, OSError, ValueError) as e:
                # one folder that cannot be read (a broken database or settings value, a drive that went away) is
                # listed with its error; the other projects stay usable
                out.append({"id": d.name, "name": d.name, "error": str(e)})
        return out

    def create(self, name: str, kind: str, model_id: str | None, outfit_mode: str = "include", source: str = "images",
               concept_words=None, enhance=None, concept_target: str = "body") -> Project:
        if kind not in ("character", "style", "concept", "enhance"):
            raise ValueError("kind must be character, style, concept or enhance")
        from .captions import CONCEPT_TARGETS
        if kind == "concept" and concept_target not in CONCEPT_TARGETS:
            raise ValueError(f"unknown concept target: {concept_target}")
        base = time.strftime("%Y%m%d-%H%M%S") + "_" + _slug(name)
        root = self.root
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ValueError(f"プロジェクトの置き場所 {root} に書き込めません（{e}）。設定画面の「データの置き場所」を確認してください。") from e
        path, i = root / base, 1
        while path.exists() or self.path_of(path.name):
            i += 1
            path = root / f"{base}_{i}"
        p = Project(path)
        p.set_meta(name=name, kind=kind, model_id=model_id, trigger=default_trigger(kind, concept_target),
                   outfit_mode=outfit_mode, train_overrides={}, created=time.time(), reference_ids=[], source=source)
        if kind == "concept":
            from .captions import clean_concept_words
            p.set_meta(concept_words=clean_concept_words(concept_words), concept_target=concept_target)
        if kind == "enhance":
            from .enhance import clean
            p.set_meta(enhance=clean(enhance))
        with self._lock:
            self._open[p.id] = p
        return p

    def get(self, pid: str) -> Project:
        with self._lock:
            if pid in self._moving:
                raise ProjectMoving(pid)
            if pid not in self._open:
                path = self.path_of(pid)
                if path is None:
                    raise KeyError(pid)
                self._open[pid] = Project(path)
            return self._open[pid]
