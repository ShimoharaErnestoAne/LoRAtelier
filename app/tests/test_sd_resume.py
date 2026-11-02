"""A resumed sd-scripts run must continue at the saved step even after the first epoch (bridges/sd_resume.py,
from MOSS-LoRA-Trainer).  The patch is applied to the pinned engine source when it is installed, and the patched
resume lines are run against a model of the engine's epoch / batch-skip loop.  No GPU run or real resume."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lorastudio.bridges.sd_resume import patch_source  # noqa: E402

ENGINE = ROOT.parent / "engines" / "sd-scripts" / "train_network.py"


def _patched() -> str:
    if not ENGINE.exists():
        pytest.skip("sd-scripts is not installed")
    return patch_source(ENGINE.read_text(encoding="utf-8-sig"))


def _resume_block(src: str) -> str:
    lines = src.splitlines()
    i = next(n for n, line in enumerate(lines) if line.strip() == "resume_global_step = initial_step")
    return "\n".join(line[8:] for line in lines[i:i + 6])


def _run(block: str, saved: int, n_batches: int, accum: int, max_steps: int):
    """The engine's loop after the resume lines (train_network.py): skip_first_batches(initial_step - 1),
    then one more batch is skipped; an optimizer step every `accum` batches and at the end of the epoch."""
    per_epoch = -(-n_batches // accum)
    ns = {"initial_step": saved, "num_update_steps_per_epoch": per_epoch, "current_step": SimpleNamespace(value=0),
          "args": SimpleNamespace(gradient_accumulation_steps=accum, skip_until_initial_step=True)}
    exec(block, {}, ns)
    global_step, initial_step, first, saves = ns["global_step"], ns["initial_step"], None, []
    for epoch in range(ns["epoch_to_start"], -(-max_steps // per_epoch)):
        batches = list(range(n_batches))
        if initial_step > 0:
            batches = batches[initial_step - 1:]
            initial_step = 1
        for i, batch in enumerate(batches):
            if initial_step > 0:
                initial_step -= 1
                continue
            first = first or (epoch, batch)
            if (batch + 1) % accum == 0 or batch == n_batches - 1:
                global_step += 1
                saves.append(global_step)
                if global_step >= max_steps:
                    return first, saves
    return first, saves


def test_patch_applies_to_the_pinned_engine():
    out = _patched()
    assert "global_step = resume_global_step" in out and "for skip_epoch in" not in out
    assert "current_step.value + 1" not in out


@pytest.mark.parametrize("saved, n_batches, accum", [(600, 20, 1), (7, 20, 1), (45, 20, 2), (31, 21, 2)])
def test_resume_continues_at_the_saved_step(saved, n_batches, accum):
    first, saves = _run(_resume_block(_patched()), saved, n_batches, accum, 1600)
    per_epoch = -(-n_batches // accum)
    assert first == (saved // per_epoch, (saved % per_epoch) * accum)   # the batch right after the saved step
    assert saves[0] == saved + 1 and saves[-1] == 1600 and len(saves) == 1600 - saved   # no step trained twice


def test_a_changed_engine_stops_instead_of_resuming_wrong():
    with pytest.raises(RuntimeError, match="再開の補正ができません"):
        patch_source("class NetworkTrainer:\n    def train(self, args):\n        pass\n")
