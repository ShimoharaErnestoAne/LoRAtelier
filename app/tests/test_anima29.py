"""Anima 2.9B (Gazingstars123/Anima-2.9B: Anima with 40 blocks instead of 28).  sd-scripts builds every Anima DiT with
28 blocks, so training goes through bridges/anima_blocks.py, which builds it with the checkpoint's block count.
Checked with header-only fake files and, when the training environment is installed, against the pinned engine's own
loader with empty weights.  No real model, GPU run or generation."""
from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lorastudio import config, evaluate, generate, presets, training  # noqa: E402
from lorastudio.bridges.anima_blocks import count_blocks  # noqa: E402
from lorastudio.models import VARIANTS, catalog_entry, detect_header  # noqa: E402

ENGINE = ROOT.parent / "engines" / "sd-scripts"
BRIDGE = ROOT / "lorastudio" / "bridges" / "anima_blocks.py"


def _header(n_blocks: int, prefix: str = "net.") -> dict:
    t = {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}
    h = {f"{prefix}blocks.{i}.mlp.layer1.weight": t for i in range(n_blocks)}
    h.update({f"{prefix}llm_adapter.blocks.{i}.cross_attn.q_proj.weight": t for i in range(60)},
             **{f"{prefix}x_embedder.proj.1.weight": t})
    return h


def _write(path: Path, header: dict) -> Path:
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0\0")
    return path


def test_count_blocks(tmp_path):
    assert count_blocks(_write(tmp_path / "a.safetensors", _header(40))) == 40
    assert count_blocks(_write(tmp_path / "b.safetensors", _header(28, "model.diffusion_model."))) == 28
    assert count_blocks(_write(tmp_path / "c.safetensors", _header(28, ""))) == 28   # the adapter's 60 blocks do not count
    (tmp_path / "bad.safetensors").write_bytes(b"not a safetensors file")
    assert count_blocks(tmp_path / "bad.safetensors") is None
    assert count_blocks(tmp_path / "missing.safetensors") is None


def test_detected_as_anima29():
    r = detect_header(_header(40))
    assert (r["family"], r["variant"]) == ("anima", "anima29") and "40 層" in r["evidence"]
    assert detect_header(_header(28))["variant"] == "anima"
    assert detect_header(_header(28, "model.diffusion_model."))["variant"] == "anima"


def test_catalog_and_settings():
    c = catalog_entry("anima_29b")
    assert c["variant"] == "anima29" and c["files"]["ckpt"][0] == "Gazingstars123/Anima-2.9B"
    assert c["files"]["te"] == catalog_entry("anima_base")["files"]["te"]   # same companions as Anima
    assert VARIANTS["anima29"]["family"] == "anima" and not VARIANTS["anima29"]["verified"]
    m = {"family": "anima", "variant": "anima29", "files": {}}
    assert evaluate.sampling_for(m, {})["scheduler"] == "sgm_uniform"
    assert generate.default_sampling(m)["scheduler"] == "sgm_uniform"
    assert evaluate.sampling_for({"family": "anima", "variant": "anima", "files": {}}, {})["scheduler"] == "simple"
    assert evaluate.QUALITY["anima29"] == evaluate.QUALITY["anima"]
    small = presets.defaults("anima", "character", "anima29", 20, 12000)
    assert small["train_batch_size"] == 1 and small["blocks_to_swap"] == 16
    assert presets.defaults("anima", "character", "anima", 20, 16000)["train_batch_size"] == 2
    assert presets.defaults("anima", "character", "anima29", 20, 16000)["train_batch_size"] == 1


def test_anima_runs_go_through_the_bridge():
    assert training._sd_script("anima_train_network.py", None) == [str(BRIDGE), "anima_train_network.py"]
    assert training._sd_script("anima_train_network.py", "state") == [str(BRIDGE), "--resume-fix", "anima_train_network.py"]
    assert training._sd_script("sdxl_train_network.py", None) == ["sdxl_train_network.py"]
    assert training._sd_script("sdxl_train_network.py", "s")[0] == str(training.SD_RESUME)


CHECK = r"""
import sys
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
import anima_blocks
anima_blocks.install()
anima_blocks.install()   # twice: still one wrapper
from library import anima_models, anima_utils
anima_utils.load_safetensors_with_lora_and_fp8 = lambda **kw: {}
anima_models.Anima.load_state_dict = lambda self, sd, strict=False, assign=False: ([], [])
out = []
for path in sys.argv[3:]:
    model = anima_utils.load_anima_model("cpu", path, "torch", False, "cpu", None, False)
    out.append(len(model.blocks))
print(out)
"""


def test_engine_builds_the_file_block_count(tmp_path):
    """The pinned engine's load_anima_model, with the bridge installed, builds a 40-block DiT for a 40-block file and
    a 28-block one for Anima 2B (empty weights; the weight loading itself is skipped)."""
    if not (config.TRAIN_PYTHON.exists() and (ENGINE / "library" / "anima_utils.py").exists()):
        pytest.skip("the training environment is not installed")
    f40 = _write(tmp_path / "anima29.safetensors", _header(40))
    f28 = _write(tmp_path / "anima.safetensors", _header(28, "model.diffusion_model."))
    r = subprocess.run([str(config.TRAIN_PYTHON), "-c", CHECK, str(ENGINE), str(BRIDGE.parent), str(f40), str(f28)],
                       cwd=ENGINE, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().splitlines()[-1] == "[40, 28]"
    assert "Anima DiT with 40 blocks" in r.stdout
