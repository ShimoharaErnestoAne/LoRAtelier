"""Paths and global (cross-project) settings."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

ROOT = Path(os.environ.get("LORASTUDIO_ROOT", Path(__file__).resolve().parents[2]))
APP_DIR = ROOT / "app"
STATIC_DIR = Path(__file__).resolve().parent / "static"
ENGINES_DIR = ROOT / "engines"
ENVS_DIR = ROOT / "envs"
MODELS_DIR = ROOT / "models"
DATA_DIR = Path(os.environ.get("LORASTUDIO_DATA", ROOT / "data"))
PROJECTS_DIR = DATA_DIR / "projects"
SETTINGS_FILE = DATA_DIR / "settings.json"
ANALYSIS_CACHE_DIR = DATA_DIR / "analysis_cache"

SD_SCRIPTS_DIR = ENGINES_DIR / "sd-scripts"
MUSUBI_DIR = ENGINES_DIR / "musubi-tuner"
COMFY_DIR = ENGINES_DIR / "ComfyUI"

TRAIN_PYTHON = ENVS_DIR / "train" / "Scripts" / "python.exe"
ANALYSIS_PYTHON = ENVS_DIR / "analysis" / "Scripts" / "python.exe"
COMFY_PYTHON = ENVS_DIR / "comfy" / "Scripts" / "python.exe"

HOST = "127.0.0.1"
# address the UI server listens on; start_intranet.bat sets 0.0.0.0 so other PCs on the LAN can open it.
# Everything else (ComfyUI, probes, the browser on this PC) keeps using HOST.
LISTEN = os.environ.get("LORASTUDIO_LISTEN", HOST)
PORT = int(os.environ.get("LORASTUDIO_PORT", "7865"))
COMFY_PORT = int(os.environ.get("LORASTUDIO_COMFY_PORT", "8199"))


def atomic_write_text(path: Path, text: str) -> None:
    """Write via temp file + fsync + os.replace so a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # one temp name per write: two requests saving the same file at once (e.g. form autosave + start)
    # must not share a temp file, which Windows refuses with PermissionError
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.{time.monotonic_ns()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        # Windows briefly refuses the replace while another thread or an antivirus scanner has the target open
        for attempt in range(10):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def atomic_write_json(path: Path, data) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


class Settings:
    """Global settings: registered base models, hash cache, comfy settings, UI prefs."""

    _lock = threading.RLock()

    def __init__(self, path: Path = SETTINGS_FILE):
        self.path = path
        self.data = {"models": [], "hash_cache": {}, "first_run_done": False}
        if path.exists():
            try:
                loaded = json.loads(self._read(path))
                if not isinstance(loaded, dict):
                    raise ValueError("settings.json is not an object")
                self.data.update(loaded)
            except ValueError:   # JSONDecodeError and UnicodeDecodeError are ValueErrors: the file itself is broken
                # keep a copy of the broken file instead of silently discarding it (an earlier copy is kept too).
                # If it cannot be moved aside, do not go on with defaults: the first save would replace it for good.
                broken = path.with_name(f"{path.stem}.broken-{time.strftime('%Y%m%d-%H%M%S')}.json")
                try:
                    os.replace(path, broken)
                except OSError:
                    try:
                        import shutil
                        shutil.copy2(path, broken)
                    except OSError as e:
                        raise RuntimeError(f"設定ファイル {path} が壊れていて、退避もできませんでした（{e}）。"
                                           "ほかのソフトがこのファイルを開いていないか確認してから、起動し直してください。") from e

    @staticmethod
    def _read(path: Path, tries: int = 5) -> str:
        """A read that fails for a moment (antivirus or backup software holding the file) is retried.  A file
        that still cannot be read stops the start: going on with defaults would lose the registered models,
        storage folders and LoRA settings, and the first save would overwrite the real file with them."""
        for i in range(tries):
            try:
                return path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                raise
            except OSError as e:
                if i == tries - 1:
                    raise RuntimeError(f"設定ファイル {path} を読めませんでした（{e}）。ほかのソフト（ウイルス対策・"
                                       "バックアップなど）がこのファイルを開いていないか確認してから、起動し直してください。") from e
                time.sleep(0.5 * (i + 1))
        return ""

    def save(self) -> None:
        with self._lock:
            atomic_write_json(self.path, self.data)

    def get(self, key, default=None):
        with self._lock:
            return self.data.get(key, default)

    def set(self, key, value) -> None:
        with self._lock:
            self.data[key] = value
            self.save()


_settings: Settings | None = None


def settings() -> Settings:
    global _settings
    if _settings is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _settings = Settings()
    return _settings


# Large data that can live on another drive (e.g. an HDD, to keep the SSD free): kind -> (default folder,
# subfolder made in the base folder set on the settings page).  Small files (settings, history) stay in DATA_DIR.
STORAGE = {
    "projects": (PROJECTS_DIR, "projects"),
    "imagegen": (DATA_DIR / "imagegen" / "images", "imagegen"),
    "i2t": (DATA_DIR / "i2t", "i2t"),   # images added on the i2t page and their thumbnails (history and config stay)
}


def _same(a: Path, b: Path) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def storage_base(kind: str) -> str | None:
    return (settings().get("storage_dirs") or {}).get(kind) or None


def storage_dir(kind: str) -> Path:
    """Folder new data of this kind is written to."""
    default, sub = STORAGE[kind]
    base = storage_base(kind)
    return Path(base) / sub if base else default


def storage_dirs(kind: str) -> list[Path]:
    """Every folder data of this kind is looked for in: the current one first, then the default and the folders
    used before (what was made there stays where it is and stays usable)."""
    out = []
    for p in [storage_dir(kind), STORAGE[kind][0], *map(Path, (settings().get("storage_dirs_old") or {}).get(kind, []))]:
        if not any(_same(p, q) for q in out):
            out.append(p)
    return out


def set_storage_base(kind: str, base: str | None) -> Path:
    """Set the base folder (None or "": back to the default).  The folder is made and a test file written,
    so a path that cannot be used is refused here, not at the next save."""
    if kind not in STORAGE:
        raise ValueError(f"unknown storage kind: {kind}")
    base = (base or "").strip().strip('"') or None
    if base:
        p = Path(base)
        if not p.is_absolute() or not p.drive:
            raise ValueError("ドライブ名から始まるフォルダーのパスを入力してください（例: D:\\LoRAtelier）。")
        dest = p / STORAGE[kind][1]
        try:
            dest.mkdir(parents=True, exist_ok=True)
            probe = dest / f".write_test.{os.getpid()}.tmp"
            probe.write_bytes(b"ok")
            probe.unlink()
        except OSError as e:
            raise ValueError(f"{dest} に書き込めません（{e}）。") from e
        base = str(p)
    with Settings._lock:
        before = storage_dir(kind)
        dirs = dict(settings().get("storage_dirs") or {})
        old = {k: list(v) for k, v in (settings().get("storage_dirs_old") or {}).items()}
        if base:
            dirs[kind] = base
        else:
            dirs.pop(kind, None)
        if not _same(before, STORAGE[kind][0]) and not any(_same(before, Path(q)) for q in old.get(kind, [])):
            old.setdefault(kind, []).append(str(before))
        s = settings()
        s.data["storage_dirs"] = dirs
        s.data["storage_dirs_old"] = old
        s.save()
    return storage_dir(kind)
