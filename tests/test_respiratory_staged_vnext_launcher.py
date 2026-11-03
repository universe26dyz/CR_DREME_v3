"""Launcher-only contracts for the formally fixed respiratory-vNext protocol."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_respiratory_staged_vnext import (  # noqa: E402
    _deep_merge,
    checkpoint_name_for_segment,
    resume_next_segment_index,
)
from cardioresp4d.training.respiratory_staged_vnext import respiratory_staged_vnext_segments  # noqa: E402


def test_launcher_deep_merge_preserves_base_and_applies_vnext_overlay() -> None:
    merged = _deep_merge({"training": {"optimizer": {"stage1": {"canonical_lr": 1.}}, "seed": 0}, "model": {"psf": {"n_samples": 8}}}, {"training": {"optimizer": {"resp_a1": {"canonical_lr": 2e-4}}, "loss_weights": {"respiratory_pca_waveform": 1e-3}}})
    assert merged["training"]["seed"] == 0
    assert merged["training"]["optimizer"]["stage1"]["canonical_lr"] == 1.
    assert merged["training"]["optimizer"]["resp_a1"]["canonical_lr"] == 2e-4
    assert merged["training"]["loss_weights"]["respiratory_pca_waveform"] == 1e-3


def test_launcher_checkpoint_names_preserve_required_segment_endpoints() -> None:
    names = [checkpoint_name_for_segment(segment) for segment in respiratory_staged_vnext_segments()]
    assert names == ["checkpoint_A1_end.pt", "checkpoint_A_end.pt", "checkpoint_B_end.pt", "checkpoint_C1_end.pt", "checkpoint_C2_end.pt", "checkpoint_C_end.pt", "checkpoint_D_end.pt", "checkpoint_E_end.pt"]


def test_resume_only_accepts_completed_segment_boundaries_and_matching_provenance() -> None:
    segments = respiratory_staged_vnext_segments()
    provenance = {"effective_config": "abc", "manifest": "m", "respiratory_pca": "p"}
    checkpoint = {"checkpoint_schema": 1, "provenance": provenance, "runner_state": {"completed": [{"segment": segments[0].name, "segment_step": segments[0].steps}], "global_step": segments[0].steps, "current_segment": segments[0].name, "segment_step": segments[0].steps}}
    assert resume_next_segment_index(checkpoint, segments, provenance) == 1
    checkpoint["runner_state"]["segment_step"] = segments[0].steps - 1
    with pytest.raises(ValueError, match="segment boundary"):
        resume_next_segment_index(checkpoint, segments, provenance)
    checkpoint["runner_state"]["segment_step"] = segments[0].steps
    with pytest.raises(ValueError, match="provenance"):
        resume_next_segment_index(checkpoint, segments, {**provenance, "manifest": "different"})
