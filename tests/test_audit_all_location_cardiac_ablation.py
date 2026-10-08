"""Checkpoint-stage compatibility for the read-only cardiac ablation audit."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_all_location_cardiac_ablation as ablation  # noqa: E402


@pytest.mark.parametrize("stage", ("stage3a", "stage3b"))
def test_cardiac_enabled_checkpoint_stages_are_accepted(stage: str) -> None:
    assert ablation.validate_cardiac_ablation_stage(stage) == stage


@pytest.mark.parametrize("stage", ("stage2c", "stage1"))
def test_noncardiac_checkpoint_stages_are_rejected(stage: str) -> None:
    with pytest.raises(ValueError, match="cardiac-enabled checkpoint"):
        ablation.validate_cardiac_ablation_stage(stage)


def test_ablation_keeps_paired_psf_rng_for_resp_only_and_joint() -> None:
    calls: list[tuple[str, torch.Tensor]] = []

    class Model:
        def predict(self, _observation, pixels, stage):
            sample = torch.rand(pixels.shape[0])
            calls.append((stage, sample))
            return {"predicted_intensity": sample + (1. if stage == "stage3b" else 0.)}

    observation = SimpleNamespace(image=torch.zeros(1, 1, 2), view="SAX", slice_id="s1", dynamic_frame_id=7, timestamp_s=1.5)
    ablation._row(Model(), observation, "stage3b", 0, torch.tensor([[0., 0.], [1., 0.]]), diagnostic_seed=3)
    assert [stage for stage, _ in calls] == ["stage2c", "stage3b"]
    torch.testing.assert_close(calls[0][1], calls[1][1])
