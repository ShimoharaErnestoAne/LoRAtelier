"""Files shared with ComfyUI by hard links.  A mistake here replaces a model with a different file or
leaves the old data on disk, so the rules are tested on small files in a temporary folder."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "data"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config, evaluate, linking
    config._settings = None
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "tool models")
    monkeypatch.setattr(evaluate, "COMFY_MODELS", tmp_path / "engine comfy models")
    inst = tmp_path / "Comfy UI"
    (inst / "models" / "checkpoints").mkdir(parents=True)
    (inst / "main.py").write_text("")
    linking.set_comfy_dir(str(inst))
    return {"ln": linking, "tmp": tmp_path, "inst": inst, "tool": tmp_path / "tool models", "staged": tmp_path / "engine comfy models"}


def _w(p: Path, data: bytes) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def test_identical_copy_becomes_one_file_and_staged_links_follow(env):
    ln = env["ln"]
    comfy = _w(env["inst"] / "models" / "checkpoints" / "pony_civitai.safetensors", b"A" * 1000)
    tool = _w(env["tool"] / "sdxl" / "pony.safetensors", b"A" * 1000)
    staged = env["staged"] / "checkpoints" / "lsg_1" / "pony.safetensors"
    staged.parent.mkdir(parents=True)
    os.link(tool, staged)                                   # the tool's own ComfyUI link to the old file
    _w(env["tool"] / "sdxl" / "other.safetensors", b"B" * 1000)   # same size, other contents
    items = ln.find_duplicates()
    assert [(Path(i["tool"]).name, i["status"]) for i in items] == [("pony.safetensors", "same")]
    r = ln.link_duplicates(items)
    assert r == {"linked": 1, "freed_bytes": 1000, "errors": []}
    assert os.path.samefile(tool, comfy) and os.path.samefile(staged, comfy)
    assert comfy.stat().st_nlink == 3 and tool.read_bytes() == b"A" * 1000
    assert [i["status"] for i in ln.find_duplicates()] == ["linked"]   # nothing left to do


def test_changed_file_is_not_replaced(env):
    ln = env["ln"]
    _w(env["inst"] / "models" / "checkpoints" / "x.safetensors", b"A" * 100)
    tool = _w(env["tool"] / "x.safetensors", b"A" * 100)
    items = ln.find_duplicates()
    tool.write_bytes(b"C" * 100)                            # changed between the list and the button
    r = ln.link_duplicates(items)
    assert r["linked"] == 0 and r["errors"] and tool.read_bytes() == b"C" * 100


def test_export_goes_only_to_its_own_folder_without_overwriting(env):
    ln = env["ln"]
    lora = _w(env["tmp"] / "exports" / "chara_v1" / "chara_v1.safetensors", b"L" * 10)
    mine = _w(env["inst"] / "models" / "loras" / "LoRAStudio" / "chara_v1.safetensors", b"other project")
    r = ln.link_export(lora)
    p = Path(r["path"])
    assert r["kind"] == "hardlink" and p.parent == env["inst"] / "models" / "loras" / "LoRAStudio"
    assert p.name == "chara_v1_2.safetensors" and os.path.samefile(p, lora)
    assert mine.read_bytes() == b"other project"


def test_not_a_comfyui_folder_is_refused(env):
    with pytest.raises(ValueError):
        env["ln"].set_comfy_dir(str(env["tmp"]))
