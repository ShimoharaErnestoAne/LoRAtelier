"""Every test runs on made-up files in temporary folders, never on what is really in this installation.

The two folders that hold the user's own files - data (projects, material, captions, training results, generated
images, settings) and models (LoRAs among them) - are pointed at a temporary folder here, before any test module
imports the app.  A test's own fixture may point them somewhere else (its tmp_path); when it ends, they come back to
this temporary folder, not to the real ones.  A test that still ends up on a real folder fails (the check below).
"""
from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

_REAL_ROOT = Path(__file__).resolve().parents[2]
_TMP = Path(tempfile.mkdtemp(prefix="loratelier_tests_"))
# set, not setdefault: a value inherited from the shell could be the real folder
os.environ["LORASTUDIO_DATA"] = str(_TMP / "data")
os.environ["LORASTUDIO_MODELS"] = str(_TMP / "models")
atexit.register(shutil.rmtree, _TMP, True)


def _inside(path, folder: Path) -> bool:
    try:
        Path(path).resolve().relative_to(folder.resolve())
        return True
    except (ValueError, OSError):
        return False


@pytest.fixture(autouse=True)
def _made_up_files_only():
    """Fails the test whose app modules were looking at the real data or models folder."""
    yield
    cfg = sys.modules.get("lorastudio.config")
    if cfg is None:
        return
    for name in ("DATA_DIR", "MODELS_DIR"):
        for real in (_REAL_ROOT / "data", _REAL_ROOT / "models"):
            assert not _inside(getattr(cfg, name), real), (
                f"config.{name} was the real folder {real}: tests must use made-up files in a temporary folder only")
