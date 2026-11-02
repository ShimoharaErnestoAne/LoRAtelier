"""Carrying training settings to another project (presets / copy from a project): values that depend on
the model family must not leak into another family, and the step count must follow the target's images."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lorastudio import presets  # noqa: E402


def _changed(family, kind, n, overrides):
    d = presets.defaults(family, kind, "x", n, 16000)
    return d, presets.snapshot(family, d, overrides)


def test_same_family_copies_everything_and_steps_follow_image_count():
    d20, snap = _changed("sdxl", "character", 20, {"learning_rate": 2e-4, "network_dim": 32, "seed": 7,
                                                   "max_train_steps": d_steps(20) * 3 // 2})
    assert snap["values"] == {"learning_rate": 2e-4, "network_dim": 32, "seed": 7}
    ov, skipped = presets.apply_snapshot(snap, "sdxl")
    assert skipped == []
    d60 = presets.defaults("sdxl", "character", "x", 60, 16000)
    values, changed = presets.merged(d60, ov)
    assert values["learning_rate"] == 2e-4 and values["network_dim"] == 32
    # 1.5x of the target's own automatic value, not the source's fixed number
    assert values["max_train_steps"] == int(round(d60["max_train_steps"] * 1.5 / 50) * 50)
    assert values["max_train_steps"] != d20["max_train_steps"] * 3 // 2


def d_steps(n):
    return presets.defaults("sdxl", "character", "x", n, 16000)["max_train_steps"]


def test_other_family_keeps_family_defaults():
    _, snap = _changed("sdxl", "character", 20, {"learning_rate": 4e-3, "optimizer_type": "Prodigy",
                                                 "network_dim": 64, "seed": 9, "caption_dropout_rate": 0.1})
    ov, skipped = presets.apply_snapshot(snap, "anima")
    assert set(skipped) == {"learning_rate", "optimizer_type", "network_dim"}
    da = presets.defaults("anima", "character", "x", 20, 16000)
    values, changed = presets.merged(da, ov)
    assert values["learning_rate"] == da["learning_rate"] and values["optimizer_type"] == da["optimizer_type"]
    assert values["seed"] == 9 and values["caption_dropout_rate"] == 0.1
    assert set(changed) == {"seed", "caption_dropout_rate"}


def test_save_count_is_kept_and_fixed_steps_option():
    d, snap = _changed("sdxl", "style", 20, {"max_train_steps": 3000, "save_every_n_steps": 300})
    ov, _ = presets.apply_snapshot(snap, "sdxl")
    values, _ = presets.merged(presets.defaults("sdxl", "style", "x", 80, 16000), ov)
    assert round(values["max_train_steps"] / values["save_every_n_steps"]) == 10
    assert not presets.validate("sdxl", values)
    ov, _ = presets.apply_snapshot(snap, "sdxl", fixed_steps=True)
    values, _ = presets.merged(presets.defaults("sdxl", "style", "x", 80, 16000), ov)
    assert (values["max_train_steps"], values["save_every_n_steps"]) == (3000, 300)


def test_typed_value_wins_over_relative_scale():
    d = presets.defaults("sdxl", "character", "x", 20, 16000)
    values, changed = presets.merged(d, {"steps_scale": 2.0, "max_train_steps": 900})
    assert values["max_train_steps"] == 900 and changed == ["max_train_steps"]
