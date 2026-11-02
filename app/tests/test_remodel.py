"""「別のモデル向けに作る」 (results page): the project's model is switched to another family.  Risk: the old
family's engine values (learning rate, rank ...) are sent to the new family's engine."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "データ 空白"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config, models
    config._settings = None
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    ck = tmp_path / "base.safetensors"
    ck.write_bytes(b"0")
    sdxl = models.registry.add("SDXL", "sdxl_base", {"ckpt": str(ck)})
    sdxl2 = models.registry.add("SDXL 2", "sdxl_base", {"ckpt": str(ck)})
    anima = models.registry.add("Anima", "anima", {"ckpt": str(ck), "te": str(ck), "vae": str(ck)})
    from lorastudio import server
    return {"server": server, "sdxl": sdxl, "sdxl2": sdxl2, "anima": anima}


def test_family_values_go_back_to_defaults_only_when_the_family_changes(env):
    s = env["server"]
    p = s.projects.create("キャラ", "character", env["sdxl"]["id"])
    ov = {"learning_rate": 5e-5, "network_dim": 64, "max_train_steps": 1200}
    p.set_meta(train_overrides=ov)
    r = s.update_project(p.id, {"model_id": env["sdxl2"]["id"]})   # same family: kept
    assert r["reset_settings"] == [] and p.meta("train_overrides") == ov
    r = s.update_project(p.id, {"model_id": env["anima"]["id"]})
    assert p.meta("train_overrides") == {"max_train_steps": 1200}
    assert len(r["reset_settings"]) == 2 and r["model_id"] == env["anima"]["id"]
