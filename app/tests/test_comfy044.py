"""ComfyUI v0.44.0 update (2026-11-09): the pins agree with each other, and the ComfyUI-GGUF fix patched in before the
rename to LoRAtelier ("LoRA Studio patch") counts as applied (otherwise it is reported as missing, and patching it
again fails because the original lines are gone)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lorastudio import addons, config  # noqa: E402

LOCK = json.loads((config.ROOT / "engines.lock.json").read_text(encoding="utf-8"))["engines"]


def _gguf_patch():
    return addons._patch(LOCK["ComfyUI-GGUF"])


def test_gguf_patch_from_before_the_rename_counts_as_applied(tmp_path):
    p = _gguf_patch()
    ops = tmp_path / "ops.py"
    ops.write_text(p.NEW.replace("LoRAtelier patch", "LoRA Studio patch"), encoding="utf-8")
    before = ops.read_bytes()
    assert p.applied(tmp_path)
    p.apply(tmp_path)                       # nothing to do, no "修正箇所が見つかりません"
    assert ops.read_bytes() == before


def test_gguf_patch_applies_once_to_the_original(tmp_path):
    p = _gguf_patch()
    ops = tmp_path / "ops.py"
    ops.write_bytes(("x = 1\n" + p.OLD).replace("\n", "\r\n").encode())
    assert not p.applied(tmp_path)
    p.apply(tmp_path)
    assert p.applied(tmp_path) and b"\r\n" in ops.read_bytes() and b"\r\r" not in ops.read_bytes()
    once = ops.read_bytes()
    p.apply(tmp_path)
    assert ops.read_bytes() == once


def test_bundled_copies_are_the_pinned_commits():
    """ensure_node copies a bundled pack only when its mark is the pinned commit; a stale copy would be fetched with
    git instead (slow, and fails offline)."""
    for name in (addons.PPM, addons.MOSS_EDIT):
        mark = (addons.bundled_dir(name) / addons.BUNDLE_MARK).read_text(encoding="utf-8").strip()
        assert mark == LOCK[name]["commit"], name
        assert (addons.bundled_dir(name) / "LICENSE").is_file()


def test_comfy_lock_matches_the_pinned_comfyui():
    """envs/comfy.lock.txt carries the packages ComfyUI v0.44.0's requirements.txt pins exactly."""
    assert LOCK["ComfyUI"]["commit"].startswith("2c75dc1b")
    lock = dict(ln.split("==", 1) for ln in (config.ROOT / "envs" / "comfy.lock.txt").read_text(encoding="utf-8").split()
                if "==" in ln)
    for pkg, ver in {"comfyui-frontend-package": "1.57.6", "comfyui-workflow-templates": "0.11.101",
                     "comfyui-embedded-docs": "0.5.16", "comfy-aimdo": "0.6.0", "comfy-kitchen": "0.2.40"}.items():
        assert lock[pkg] == ver, pkg
