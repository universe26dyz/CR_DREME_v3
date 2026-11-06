from __future__ import annotations

from pathlib import Path

import torch


def test_boundary_discovery_orders_and_deduplicates_checkpoint_last(tmp_path: Path) -> None:
    from cardioresp4d.visualization.dynamic_audit import discover_boundary_checkpoints

    names = ["checkpoint_L3_joint_end.pt", "checkpoint_L1_joint_end.pt", "checkpoint_L1_warm_end.pt", "checkpoint_L2_warm_end.pt", "checkpoint_L2_joint_end.pt", "checkpoint_L3_warm_end.pt"]
    for name in names: (tmp_path / name).write_bytes(name.encode())
    (tmp_path / "checkpoint_last.pt").write_bytes((tmp_path / "checkpoint_L3_joint_end.pt").read_bytes())
    result = discover_boundary_checkpoints(tmp_path)
    assert [item.label for item in result] == ["L1_warm", "L1_joint", "L2_warm", "L2_joint", "L3_warm", "L3_joint"]


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
