from __future__ import annotations

import torch
from types import SimpleNamespace

from cardioresp4d.diagnostics.change5c import stable_diagnostic_seed
from cardioresp4d.diagnostics.stage_visual import dvf_static_dynamic_metrics, finite_array_stats, select_stack_locations, temporal_maps_and_metrics


def test_temporal_statistics_preserve_float_arrays_and_recovery() -> None:
    acquired = torch.tensor([[[0., 2.]], [[2., 4.]]]); predicted = acquired / 2
    maps, metrics = temporal_maps_and_metrics(acquired, predicted, torch.tensor([[True, False]]))
    assert maps["predicted_temporal_std"].dtype == torch.float32
    assert metrics["temporal_std_recovery_ratio"] == .5
    assert finite_array_stats(predicted)["nonzero_fraction"] > 0.


def test_static_dynamic_dvf_decomposition_uses_physical_mm() -> None:
    dvf = torch.tensor([[[[2., 0., 0.]]], [[[4., 0., 0.]]]])
    mean, dynamic, metrics = dvf_static_dynamic_metrics(dvf)
    torch.testing.assert_close(mean, torch.tensor([[[3., 0., 0.]]]))
    torch.testing.assert_close(dynamic[:, 0, 0, 0], torch.tensor([-1., 1.]))
    assert metrics["static_to_dynamic_RMS_ratio"] == 3.


def test_fixed_location_selection_is_physical_and_reuses_persisted_locations() -> None:
    observations = [
        SimpleNamespace(qc_valid=True, view="SAX", slice_id=f"SAX_s{index:03d}", center_mm=torch.tensor([0., 0., float(index)]), normal=torch.tensor([0., 0., 1.]))
        for index in range(5)
    ]
    selected = select_stack_locations(observations)
    assert selected["SAX"] == ["SAX_s000", "SAX_s002", "SAX_s004"]
    assert select_stack_locations(observations, existing={"SAX": ["SAX_s002"]})["SAX"] == ["SAX_s002"]


def test_fixed_location_psf_seed_is_independent_of_checkpoint_label() -> None:
    first = stable_diagnostic_seed(17, "SAX", "SAX_s026", 0, purpose="stage_visual_shared_psf")
    second = stable_diagnostic_seed(17, "SAX", "SAX_s026", 0, purpose="stage_visual_shared_psf")
    assert first == second
