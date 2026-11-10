from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def test_boundary_discovery_orders_and_deduplicates_checkpoint_last(tmp_path: Path) -> None:
    from cardioresp4d.visualization.dynamic_audit import discover_boundary_checkpoints, resolve_dynamic_checkpoint_root

    names = ["checkpoint_L3_joint_end.pt", "checkpoint_L1_joint_end.pt", "checkpoint_L1_warm_end.pt", "checkpoint_L2_warm_end.pt", "checkpoint_L2_joint_end.pt", "checkpoint_L3_warm_end.pt"]
    for name in names: (tmp_path / name).write_bytes(name.encode())
    (tmp_path / "checkpoint_last.pt").write_bytes((tmp_path / "checkpoint_L3_joint_end.pt").read_bytes())
    result = discover_boundary_checkpoints(tmp_path)
    assert [item.label for item in result] == ["L1_warm", "L1_joint", "L2_warm", "L2_joint", "L3_warm", "L3_joint"]


def test_boundary_discovery_resolves_experiment_or_direct_training_root_and_no_root(tmp_path: Path) -> None:
    from cardioresp4d.visualization.dynamic_audit import discover_boundary_checkpoints, resolve_dynamic_checkpoint_root

    training = tmp_path / "04_training"; training.mkdir()
    (training / "checkpoint_L1_warm_end.pt").write_bytes(b"warm")
    assert resolve_dynamic_checkpoint_root(tmp_path) == training
    assert resolve_dynamic_checkpoint_root(training) == training
    assert [item.label for item in discover_boundary_checkpoints(tmp_path)] == ["L1_warm"]
    assert resolve_dynamic_checkpoint_root(tmp_path / "missing") is None


def test_fixed_location_selection_never_mixes_slice_ids_and_spans_time() -> None:
    from cardioresp4d.visualization.dynamic_audit import select_fixed_location_frames
    from cardioresp4d.training.sampler import DynamicObservation

    rows = []
    for view in ("SAX", "2CH", "4CH"):
        for location, z in (("low", 0.), ("middle", 5.), ("high", 10.)):
            for time in (0., 2., 4.):
                rows.append(DynamicObservation(torch.zeros(1, 2, 2), view, location, len(rows), torch.tensor([0., 0., z]), torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.]), torch.ones(2), 1., True, "", time, 0.))
    selected = select_fixed_location_frames(rows, frames_per_location=3)
    assert set(selected) == {("SAX", "middle"), ("2CH", "middle"), ("4CH", "middle")}
    assert all(len({frame.slice_id for frame in frames}) == 1 and [frame.timestamp_s for frame in frames] == [0., 2., 4.] for frames in selected.values())


def test_dynamic_audit_active_levels_centering_grid_and_field_labels() -> None:
    from cardioresp4d.visualization.dynamic_audit import active_respiratory_levels, fixed_physical_grid, posthoc_center_scores, dynamic_field_labels

    assert [active_respiratory_levels(stage) for stage in ("stage2a", "stage2b", "stage2c")] == [1, 2, 3]
    scores = torch.tensor([[1., 2.], [3., 4.]])
    torch.testing.assert_close(posthoc_center_scores(scores), torch.tensor([[-1., -1.], [1., 1.]]))
    grid_a, spacing_a = fixed_physical_grid(torch.zeros(3), torch.ones(3), (4, 4, 4)); grid_b, spacing_b = fixed_physical_grid(torch.zeros(3), torch.ones(3), (4, 4, 4))
    torch.testing.assert_close(grid_a, grid_b); torch.testing.assert_close(spacing_a, spacing_b)
    assert dynamic_field_labels() == {"raw_mbc": "raw MBC basis field", "weighted_dvf": "score-weighted DVF"}


def test_exp1_checkpoint_bounds_use_model_state_and_current_nested_metadata() -> None:
    from cardioresp4d.visualization.dynamic_audit import canonical_internal_bounds_from_checkpoint

    bounds = torch.tensor([[-7.0, -6.0, -5.0], [7.0, 6.0, 5.0]])
    checkpoint = {
        "model": {"canonical.inr.bounding_box": bounds.clone()},
        "static_initialization": {"direct_state_admission": {"internal_inr_bbox": bounds.tolist()}},
    }
    actual = canonical_internal_bounds_from_checkpoint(checkpoint, torch.device("cpu"))
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, bounds)


def test_exp1_checkpoint_bounds_reject_nested_metadata_conflict() -> None:
    from cardioresp4d.visualization.dynamic_audit import canonical_internal_bounds_from_checkpoint

    bounds = torch.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]])
    conflicting = bounds.clone(); conflicting[1, 0] += 0.1
    checkpoint = {
        "model": {"canonical.inr.bounding_box": bounds},
        "static_initialization": {"direct_state_admission": {"internal_inr_bbox": conflicting.tolist()}},
    }
    with pytest.raises(ValueError, match="conflicts"):
        canonical_internal_bounds_from_checkpoint(checkpoint, torch.device("cpu"))


def test_exp1_checkpoint_bounds_support_legacy_top_level_metadata() -> None:
    from cardioresp4d.visualization.dynamic_audit import canonical_internal_bounds_from_checkpoint

    bounds = torch.tensor([[-4.0, -5.0, -6.0], [4.0, 5.0, 6.0]])
    checkpoint = {"model": {"canonical.inr.bounding_box": bounds}, "static_initialization": {"internal_inr_bbox": bounds.tolist()}}
    torch.testing.assert_close(canonical_internal_bounds_from_checkpoint(checkpoint, torch.device("cpu")), bounds)


@pytest.mark.parametrize(
    "bounds",
    [
        torch.zeros(3),
        torch.tensor([[-1.0, float("nan"), -1.0], [1.0, 1.0, 1.0]]),
        torch.tensor([[1.0, -1.0, -1.0], [0.0, 1.0, 1.0]]),
    ],
)
def test_exp1_checkpoint_bounds_reject_malformed_model_state(bounds: torch.Tensor) -> None:
    from cardioresp4d.visualization.dynamic_audit import canonical_internal_bounds_from_checkpoint

    with pytest.raises(ValueError):
        canonical_internal_bounds_from_checkpoint({"model": {"canonical.inr.bounding_box": bounds}}, torch.device("cpu"))


def test_reconstruct_exp1_checkpoint_passes_checkpoint_bounds_to_builder(monkeypatch) -> None:
    import cardioresp4d.visualization.dynamic_audit as dynamic_audit

    bounds = torch.tensor([[-8.0, -7.0, -6.0], [8.0, 7.0, 6.0]])
    captured = {}

    class FakeModel:
        def __init__(self):
            self.canonical = SimpleNamespace(inr=SimpleNamespace(train=lambda: None))

        def to(self, device):
            return self

        def load_state_dict(self, state, strict=True):
            captured["strict"] = strict

        def eval(self):
            return self

    def fake_builder(*args, **kwargs):
        captured.update(kwargs)
        return FakeModel()

    monkeypatch.setattr(dynamic_audit, "build_source_first_model", fake_builder)
    checkpoint = {
        "effective_config": {},
        "model": {"canonical.inr.bounding_box": bounds.clone(), "encoding.box_offsets": torch.zeros(1)},
        "static_initialization": {"direct_state_admission": {"internal_inr_bbox": bounds.tolist()}},
    }
    dynamic_audit.reconstruct_exp1_checkpoint(checkpoint, {"world_min_mm": [-100.0] * 3, "world_max_mm": [100.0] * 3}, [SimpleNamespace(dynamic_frame_id=0)], torch.device("cpu"))
    torch.testing.assert_close(captured["canonical_internal_bounds_world_mm"], bounds)
    assert captured["canonical_world_to_internal_transform"].shape == (3, 3)
    assert captured["strict"] is True


def test_boundary_subset_validates_and_preserves_requested_protocol_order(tmp_path: Path) -> None:
    from cardioresp4d.visualization.dynamic_audit import discover_boundary_checkpoints, select_boundary_subset

    for name in ("L1_warm", "L1_joint"):
        (tmp_path / f"checkpoint_{name}_end.pt").write_bytes(name.encode())
    boundaries = discover_boundary_checkpoints(tmp_path)
    assert [item.label for item in select_boundary_subset(boundaries, "L1_joint,L1_warm")] == ["L1_joint", "L1_warm"]
    import pytest
    with pytest.raises(ValueError, match="unknown boundary"):
        select_boundary_subset(boundaries, "L4_warm")


def test_precomputed_basis_weighting_matches_score_weighted_field() -> None:
    from cardioresp4d.models.bspline_mbc import RespiratoryMBC
    from cardioresp4d.models.cardioresp_motion import ScoreWeightedMBCField
    from cardioresp4d.visualization.dynamic_audit import weighted_dvf_from_precomputed_basis

    mbc = RespiratoryMBC(torch.tensor([-1., -1., -1.]), torch.tensor([1., 1., 1.]), resolutions=(4, 5, 6))
    with torch.no_grad():
        for level in mbc.levels:
            level.controls.copy_(torch.randn_like(level.controls))
    points = torch.randn(2, 11, 3).clamp(-.8, .8); scores = torch.randn(2, 3, 3)
    expected = ScoreWeightedMBCField(mbc, scores)(points)
    basis = mbc(points)
    torch.testing.assert_close(weighted_dvf_from_precomputed_basis(basis, scores), expected)


def test_location_local_pooling_does_not_turn_location_offsets_into_dynamics() -> None:
    from cardioresp4d.visualization.dynamic_audit import location_local_rms_metrics, pool_location_metrics_by_view

    first = torch.tensor([[100., 100., 100.], [100., 100., 100.]])
    second = torch.tensor([[-50., -50., -50.], [-50., -50., -50.]])
    metrics = location_local_rms_metrics((first, second))
    assert metrics["dynamic_rms"] == 0.0
    assert metrics["total_rms"] > 0.0
    by_view = pool_location_metrics_by_view({"SAX": (first,), "2CH": (second,)})
    assert set(by_view) == {"SAX", "2CH"} and by_view["SAX"]["dynamic_rms"] == 0.0


def test_vector_field_plane_renderer_writes_magnitude_and_signed_components(tmp_path: Path) -> None:
    from cardioresp4d.visualization.adapters.exp1_reference import _render_vector_field_planes

    field = torch.zeros(16 * 16 * 16, 3); field[:, 0] = 1.; field[:, 1] = -2.; field[:, 2] = .5
    _render_vector_field_planes(tmp_path, "dvf", field, label="score-weighted DVF")
    assert {path.name for path in tmp_path.iterdir()} == {"dvf_magnitude.png", "dvf_dx.png", "dvf_dy.png", "dvf_dz.png"}
