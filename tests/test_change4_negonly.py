"""C4 negative-crossover-only effective-loss contract."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from cardioresp4d.training.source_first_config import validate_source_first_config
from cardioresp4d.training.trainer import effective_loss_weights


ROOT = Path(__file__).resolve().parents[1]


def test_change4_negonly_config_only_enables_negative_crossover() -> None:
    baseline = yaml.safe_load((ROOT / "configs" / "source_first.yaml").read_text(encoding="utf-8"))
    config = yaml.safe_load((ROOT / "configs" / "source_first_change4_negonly.yaml").read_text(encoding="utf-8"))
    validate_source_first_config(config, ROOT)
    weights = effective_loss_weights(config["training"]["loss_weights"])
    assert weights["cardiac_leakage_in_resp"] == 1e-4
    assert weights["respiratory_leakage_in_card"] == 1e-4
    assert weights["cardiac_target_concentration"] == 0.
    assert weights["cardiac_pca_waveform"] == 0.
    expected = copy.deepcopy(baseline)
    expected["training"]["loss_weights"].update({
        "cardiac_leakage_in_resp": 1e-4,
        "respiratory_leakage_in_card": 1e-4,
        "cardiac_target_concentration": 0.0,
        "cardiac_pca_waveform": 0.0,
    })
    assert config == expected
