from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch


def _geometry(z: float) -> dict:
    return {
        "center_mm": [0.0, 0.0, z], "row_direction": [1.0, 0.0, 0.0],
        "column_direction": [0.0, 1.0, 0.0], "normal": [0.0, 0.0, 1.0],
        "pixel_spacing_mm": [1.0, 1.0], "slice_thickness_mm": 2.0,
        "image_shape": [1, 3, 4],
    }


def test_adapter_selection_and_exp1_scientific_status() -> None:
    from cardioresp4d.visualization import adapter_for

    adapter = adapter_for("exp1_reference")
    assert adapter.name == "exp1_reference"
    assert adapter.scientific_status["experiment_scientific_status"] == "INVALID_PENDING_STATIC_PIPELINE_REPAIR"
    with pytest.raises(ValueError, match="unknown experiment"):
        adapter_for("other")


def test_output_directory_must_not_overlap_authoritative_artifacts(tmp_path: Path) -> None:
    from cardioresp4d.visualization.contract import ensure_safe_output_dir

    experiment = tmp_path / "experiment"; authoritative = experiment / "02_static_nesvor"
    authoritative.mkdir(parents=True)
    with pytest.raises(ValueError, match="authoritative"):
        ensure_safe_output_dir(authoritative, [authoritative / "model.pt"])
    output = experiment / "05_visualization"
    ensure_safe_output_dir(output, [authoritative / "model.pt"])
    assert output.is_dir()


def test_representatives_are_physical_ordered_and_placeholders_are_distinct() -> None:
    from cardioresp4d.visualization.geometry_utils import ordered_representative_locations, stack_tiles

    records = [
        {"view": "SAX", "slice_id": "last", "stack_index": 5, "geometry": _geometry(10)},
        {"view": "SAX", "slice_id": "first", "stack_index": 0, "geometry": _geometry(0)},
        {"view": "SAX", "slice_id": "middle", "stack_index": 2, "geometry": _geometry(4)},
    ]
    assert [row["slice_id"] for row in ordered_representative_locations(records)] == ["first", "middle", "last"]
    tiles = stack_tiles(records, lattice_size=6)
    assert tiles[1]["label"] == "MISSING PLACEHOLDER" and not tiles[1]["real"]
    assert tiles[2]["slice_id"] == "middle" and tiles[2]["real"]


def test_lps_ras_roundtrip_and_geometry_fault_detection() -> None:
    from cardioresp4d.visualization.geometry_utils import compare_plane_corners_lps, detect_geometry_faults, plane_corners_lps

    corners = plane_corners_lps(_geometry(3.0))
    affine_lps = np.array([[1., 0., 0., -1.5], [0., 1., 0., -1.], [0., 0., 1., 3.], [0., 0., 0., 1.]])
    affine_ras = np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine_lps
    assert compare_plane_corners_lps(corners, affine_ras, (4, 3, 1), 0)["max_error_mm"] == pytest.approx(0.0)
    assert "axis_swap" in detect_geometry_faults(corners, corners[[1, 0, 3, 2]])
    flipped = corners.copy(); flipped[:, 0] *= -1
    assert "sign_flip" in detect_geometry_faults(corners, flipped)
    assert "reversed_stack_order" in detect_geometry_faults(np.array([[0., 0., 0.], [0., 0., 2.]]), np.array([[0., 0., 2.], [0., 0., 0.]]))


def test_rendering_does_not_mutate_arrays_and_paired_window_is_shared(tmp_path: Path) -> None:
    from cardioresp4d.visualization.renderers import render_paired_images

    left = np.array([[0., 1.], [2., 3.]], dtype=np.float32); right = left + 2
    before = left.copy()
    policy = render_paired_images(tmp_path / "pair.png", [left, right], ["left", "right"], window=(0., 5.))
    np.testing.assert_array_equal(left, before)
    assert policy == {"kind": "fixed", "vmin": 0.0, "vmax": 5.0}
    assert (tmp_path / "pair.png").is_file()


def test_pose_deltas_and_numeric_slice_filter(tmp_path: Path) -> None:
    from cardioresp4d.visualization.geometry_utils import pose_delta, numeric_corrected_slices

    original = _geometry(0.0); corrected = _geometry(2.0); corrected["normal"] = [0.0, 1.0, 0.0]
    delta = pose_delta(original, corrected)
    assert delta["translation_mm"] == pytest.approx(2.0)
    assert delta["normal_angle_deg"] == pytest.approx(90.0)
    for filename in ("0.nii.gz", "1.nii.gz", "mask_0.nii.gz", "x.nii.gz"):
        (tmp_path / filename).write_bytes(b"x")
    assert [path.name for path in numeric_corrected_slices(tmp_path)] == ["0.nii.gz", "1.nii.gz"]


def test_v0_representative_frames_use_one_location_and_true_relative_seconds(monkeypatch, tmp_path: Path) -> None:
    from cardioresp4d.visualization.adapters.exp1_reference import Exp1ReferenceAdapter
    import cardioresp4d.visualization.adapters.exp1_reference as module

    rows = []
    samples = {}
    for slice_id, z in (("a", 0.), ("b", 10.)):
        for frame_index, timestamp in enumerate((10., 13., 20.)):
            row = {"view": "SAX", "slice_id": slice_id, "qc_valid": "1", "image_orientation_patient": "[1,0,0,0,1,0]", "image_position_patient": f"[0,0,{z}]"}
            rows.append(row); samples[len(rows) - 1] = {"rescaled_image": np.full((3, 3), frame_index + z), "timestamp_s": timestamp, "slice_id": slice_id, "source_file_token": f"tok-{slice_id}-{frame_index}"}

    class FakeDataset:
        _rows = rows
        def __getitem__(self, index): return samples[index]

    captured = []
    monkeypatch.setattr(Exp1ReferenceAdapter, "_dataset", staticmethod(lambda *_: FakeDataset()))
    monkeypatch.setattr(module, "render_montage", lambda *args, **kwargs: captured.append((args, kwargs)) or {"kind": "fixed"})
    Exp1ReferenceAdapter()._v0_raw(Path("manifest"), Path("qc"), tmp_path)
    contacts = [item for item in captured if "temporal_contact_sheet" in str(item[0][0])]
    assert len(contacts) == 1
    labels = contacts[0][0][2]
    panels = contacts[0][0][1]
    assert len({label.split(" frame")[0] for label in labels}) == 1
    assert all("Δt=" in label for label in labels)
    assert any("Δt=3s" in label for label in labels) and any("Δt=10s" in label for label in labels)
    assert all(float(panel[0, 0]) >= 10. for panel in panels)


def test_clip_fractions_aggregate_full_series_and_display_windows_are_independent() -> None:
    from cardioresp4d.visualization.image_utils import aggregate_clip_fractions

    frames = [np.array([0., 1., 2.]), np.array([100., 101., 102.])]
    result = aggregate_clip_fractions(frames, lower=1., upper=101.)
    assert result["low_clip_fraction"] == pytest.approx(2 / 6)
    assert result["high_clip_fraction"] == pytest.approx(2 / 6)


def test_support_export_uses_official_nesvor_transpose_and_round_trips(tmp_path: Path) -> None:
    from cardioresp4d.visualization.image_utils import export_support_mask
    import cardioresp4d.adapters.nesvor_inr  # registers the vendored NeSVoR path
    from nesvor.image.image_utils import load_nii_volume

    internal = np.zeros((2, 3, 5), dtype=bool); internal[0, 1, 4] = True; internal[1, 2, 0] = True
    path = export_support_mask(internal, np.eye(4), tmp_path / "mask.nii.gz")
    assert nib.load(str(path)).shape == (5, 3, 2)
    restored, _, _ = load_nii_volume(path)
    np.testing.assert_array_equal(restored.astype(bool), internal)


def test_support_bbox_uses_all_eight_corners_and_reports_lps() -> None:
    from cardioresp4d.visualization.image_utils import support_metrics

    affine = np.array([[1., 1., 0., 0.], [0., 1., 1., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    values = np.ones((2, 2, 2), dtype=np.uint8)
    result = support_metrics(values, affine)
    assert result["voxel_index_bbox"] == [[0, 0, 0], [1, 1, 1]]
    assert result["world_bbox_ras_mm"] == [[0., 0., 0.], [2., 2., 1.]]
    assert "world_bbox_lps_mm" in result


def test_pose_delta_uses_full_so3_rotation_when_row_is_unchanged() -> None:
    from cardioresp4d.visualization.geometry_utils import pose_delta

    original = _geometry(0.)
    corrected = _geometry(0.)
    corrected["column_direction"] = [0., 0., 1.]
    corrected["normal"] = [0., -1., 0.]
    result = pose_delta(original, corrected)
    assert result["rotation_deg"] == pytest.approx(90.)
    assert result["normal_angle_deg"] == pytest.approx(90.)


def test_v7_representative_indices_cover_all_views() -> None:
    from cardioresp4d.visualization.adapters.exp1_reference import Exp1ReferenceAdapter

    poses = {("SAX", str(i)): object() for i in range(3)} | {("2CH", str(i)): object() for i in range(2)} | {("4CH", str(i)): object() for i in range(4)}
    selected = Exp1ReferenceAdapter._representative_pose_keys(poses)
    assert {view for view, _ in selected} == {"SAX", "2CH", "4CH"}


def test_cli_returns_nonzero_after_mandatory_failure_but_allows_partial_hooks(monkeypatch, tmp_path: Path) -> None:
    import scripts.render_experiment_audit as cli

    class FakeAdapter:
        name = "fake"
        scientific_status = {}
        def source_artifacts(self, *args): return {}
        def render(self, **kwargs):
            return {**{name: {"render_status": "rendered", "manual_admission": "pending"} for name in ("V0_raw_acquisition", "V1_normalization", "V2_temporal_mean", "V3_stack_geometry", "V4_registration", "V5_static_support", "V6_static_inr", "V7_static_reprojection")}, "V8_dynamic_zero_motion": {"render_status": "partial"}, "V9_dynamic_boundaries": {"render_status": "partial"}}
    monkeypatch.setattr(cli, "adapter_for", lambda _: FakeAdapter())
    monkeypatch.setattr(cli, "write_audit_manifest", lambda path, **kwargs: (path.write_text("{}"), path)[1])
    monkeypatch.setattr(cli, "write_html_report", lambda *args: None)
    monkeypatch.setattr(cli, "ensure_safe_output_dir", lambda path, _: path.mkdir(parents=True, exist_ok=True) or path)
    monkeypatch.setattr(cli, "json", json)
    monkeypatch.setattr(cli.sys, "argv", ["render", "--experiment", "fake", "--experiment-root", str(tmp_path), "--manifest", str(tmp_path / "m"), "--qc-table", str(tmp_path / "q"), "--canonical-domain", str(tmp_path / "d"), "--output-dir", str(tmp_path / "out")])
    assert cli.main() == 0


def test_cli_fails_closed_when_mandatory_stage_failed(monkeypatch, tmp_path: Path) -> None:
    import scripts.render_experiment_audit as cli
    class FakeAdapter:
        name = "fake"; scientific_status = {}
        def source_artifacts(self, *args): return {}
        def render(self, **kwargs): return {"V0_raw_acquisition": {"render_status": "failed", "reason": "boom"}, **{f"V{i}_x": {"render_status": "rendered"} for i in range(1, 10)}}
    monkeypatch.setattr(cli, "adapter_for", lambda _: FakeAdapter())
    monkeypatch.setattr(cli, "write_audit_manifest", lambda path, **kwargs: (path.write_text("{}"), path)[1])
    monkeypatch.setattr(cli, "write_html_report", lambda *args: None)
    monkeypatch.setattr(cli.sys, "argv", ["render", "--experiment", "fake", "--experiment-root", str(tmp_path), "--manifest", str(tmp_path / "m"), "--qc-table", str(tmp_path / "q"), "--canonical-domain", str(tmp_path / "d"), "--output-dir", str(tmp_path / "out")])
    assert cli.main() == 2


def test_psf_free_and_psf_sampled_lattice_helpers_are_distinct_and_deterministic() -> None:
    from cardioresp4d.visualization.image_utils import evaluate_inr_on_nifti_lattice, evaluate_inr_psf_on_nifti_lattice
    import cardioresp4d.adapters.nesvor_inr
    class Model(torch.nn.Module):
        def forward(self, points): return points[..., 0]
        def sample_batch(self, points, transformation, sigma, n_samples): return points[:, None, :].expand(-1, n_samples, -1) + torch.randn(points.shape[0], n_samples, 3) * float(sigma)
    target = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4)); model = Model()
    center = evaluate_inr_on_nifti_lattice(model, target, device=torch.device("cpu")); sampled_a = evaluate_inr_psf_on_nifti_lattice(model, target, device=torch.device("cpu"), psf_resolution=1., n_samples=4, seed=7); sampled_b = evaluate_inr_psf_on_nifti_lattice(model, target, device=torch.device("cpu"), psf_resolution=1., n_samples=4, seed=7)
    assert center.shape == sampled_a.shape == target.shape
    np.testing.assert_array_equal(sampled_a, sampled_b)


def test_representative_plane_specs_preserve_fixed_axis_coordinates() -> None:
    from cardioresp4d.visualization.image_utils import representative_plane_specs

    target = nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.float32), np.diag([2., 3., 4., 1.]))
    planes = representative_plane_specs(target)
    assert [plane["axis"] for plane in planes] == ["sagittal", "coronal", "axial"]
    assert planes[0]["shape"] == (1, 5, 6) and planes[0]["voxel_index"] == (2, None, None)
    assert planes[1]["shape"] == (4, 1, 6) and planes[1]["voxel_index"] == (None, 2, None)
    assert planes[2]["shape"] == (4, 5, 1) and planes[2]["voxel_index"] == (None, None, 3)
    np.testing.assert_allclose(planes[2]["affine"][:3, 3], [0., 0., 12.])


def test_support_selection_is_capped_deterministic_and_spans_support() -> None:
    from cardioresp4d.visualization.image_utils import select_support_voxel_indices

    mask = np.zeros((20, 1, 1), dtype=bool); mask[:2] = True; mask[-2:] = True
    first, total = select_support_voxel_indices(mask, max_points=3, chunk_size=2)
    second, total_again = select_support_voxel_indices(mask, max_points=3, chunk_size=2)
    assert total == total_again == 4 and len(first) == 3
    np.testing.assert_array_equal(first, second)
    assert first[:, 0].min() == 0 and first[:, 0].max() == 19


def test_psf_plane_and_support_agreement_use_bounded_explicit_batches() -> None:
    from cardioresp4d.visualization.image_utils import (evaluate_inr_psf_plane, evaluate_inr_psf_points,
                                                        representative_plane_specs, support_point_agreement)

    class Model(torch.nn.Module):
        def forward(self, points): return points[..., 0]
        def sample_batch(self, points, transformation, sigma, n_samples): return points[:, None, :].expand(-1, n_samples, -1)
    model = Model(); points = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    values = evaluate_inr_psf_points(model, points, device=torch.device("cpu"), psf_resolution=1., n_samples=128, seed=4, point_batch_size=2)
    assert values.shape == (4,)
    target = nib.Nifti1Image(np.zeros((3, 4, 5), dtype=np.float32), np.diag([2., 3., 4., 1.])); spec = representative_plane_specs(target)[0]
    plane_a = evaluate_inr_psf_plane(model, target, spec, device=torch.device("cpu"), psf_resolution=1., n_samples=2, seed=4, point_batch_size=3)
    plane_b = evaluate_inr_psf_plane(model, target, spec, device=torch.device("cpu"), psf_resolution=1., n_samples=2, seed=4, point_batch_size=3)
    np.testing.assert_array_equal(plane_a, plane_b)
    np.testing.assert_allclose(plane_a, 2. * spec["fixed_index"])
    coordinates = np.indices((3, 3, 3), dtype=np.float32); formal = (2. * coordinates[0]).astype(np.float32); mask = np.ones_like(formal, dtype=bool)
    agreement = support_point_agreement(model, formal, mask, np.diag([2., 1., 1., 1.]), device=torch.device("cpu"), psf_resolution=1., n_samples=2, seed=4, max_points=5, point_batch_size=2)
    assert agreement["sample_count"] == 5 and agreement["total_support_count"] == 27
    assert agreement["sampling_policy"] == "uniform_index_space_ranks_over_sorted_support"
    assert agreement["mae"] == pytest.approx(0.) and agreement["rmse"] == pytest.approx(0.) and agreement["correlation"] == pytest.approx(1.)


def test_v6_metadata_does_not_claim_full_volume_psf_derivative() -> None:
    import inspect
    from cardioresp4d.visualization.adapters.exp1_reference import Exp1ReferenceAdapter

    source = inspect.getsource(Exp1ReferenceAdapter._v6_inr)
    assert "static_inr_psf_unmasked" not in source
    assert "evaluate_inr_psf_on_nifti_lattice" not in source


def test_support_export_preserves_values_and_unmasked_lattice(tmp_path: Path) -> None:
    from cardioresp4d.visualization.image_utils import export_support_mask, save_unmasked_static_inr

    mask = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.bool)
    mask_path = export_support_mask(mask, np.eye(4), tmp_path / "mask.nii.gz")
    np.testing.assert_array_equal(nib.load(str(mask_path)).get_fdata(), mask.numpy().transpose(2, 1, 0))
    target = nib.Nifti1Image(np.zeros((2, 3, 4), dtype=np.float32), np.diag([1., 2., 3., 1.]))
    result = save_unmasked_static_inr(np.ones((2, 3, 4), dtype=np.float32), target, tmp_path / "inr.nii.gz")
    image = nib.load(str(result))
    assert image.shape == target.shape
    np.testing.assert_allclose(image.affine, target.affine)


def test_manifest_and_html_report_include_hashes_display_policy_and_stage_status(tmp_path: Path) -> None:
    from cardioresp4d.visualization.report import write_audit_manifest, write_html_report

    source = tmp_path / "source.txt"; source.write_text("source")
    manifest = write_audit_manifest(tmp_path / "audit_manifest.json", adapter="exp1_reference", sources={"input": source}, stages={"V0_raw_acquisition": {"render_status": "rendered", "manual_admission": "pending"}}, display_policies={"pair": {"kind": "fixed", "vmin": 0., "vmax": 1.}})
    payload = json.loads(manifest.read_text())
    assert payload["source_artifacts"]["input"]["sha256"] and payload["display_window_policies"]["pair"]["kind"] == "fixed"
    report = write_html_report(tmp_path / "report.html", payload, {"V0_raw_acquisition": {"render_status": "rendered", "manual_admission": "pending"}})
    assert "V0_raw_acquisition" in report.read_text()
