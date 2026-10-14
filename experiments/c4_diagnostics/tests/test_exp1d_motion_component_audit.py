from __future__ import annotations

import pytest
import torch

from experiments.c4_diagnostics.exp1d_motion_component_audit import (
    component_dynamic_metrics,
    project_dvf_to_slice_components,
)


def test_slice_basis_projection_uses_row_column_and_normal_directions() -> None:
    dvf = torch.tensor([[[1.0, 2.0, 3.0]]])
    row = torch.tensor([0.0, 1.0, 0.0])
    column = torch.tensor([0.0, 0.0, 1.0])
    normal = torch.tensor([1.0, 0.0, 0.0])
    projected = project_dvf_to_slice_components(dvf, row, column, normal)
    assert torch.equal(projected, torch.tensor([[[2.0, 3.0, 1.0]]]))


def test_component_metrics_are_exact_for_identical_dynamic_fields() -> None:
    target = torch.tensor([[[1.0], [2.0]], [[-1.0], [-2.0]]])
    values = component_dynamic_metrics(target, target)
    assert values["rms_ratio"] == pytest.approx(1.0)
    assert values["correlation"] == pytest.approx(1.0)
    assert values["endpoint_error_mm"] == pytest.approx(0.0)
    assert values["normalized_epe"] == pytest.approx(0.0)


def test_component_metrics_detect_sign_reversal() -> None:
    target = torch.tensor([[[1.0], [2.0]], [[-1.0], [-2.0]]])
    values = component_dynamic_metrics(-target, target)
    assert values["rms_ratio"] == pytest.approx(1.0)
    assert values["correlation"] == pytest.approx(-1.0)
