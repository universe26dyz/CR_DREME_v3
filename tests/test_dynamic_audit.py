from __future__ import annotations

from pathlib import Path

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
