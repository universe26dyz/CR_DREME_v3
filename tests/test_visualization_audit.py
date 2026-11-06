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


def test_support_export_preserves_values_and_unmasked_lattice(tmp_path: Path) -> None:
    from cardioresp4d.visualization.image_utils import export_support_mask, save_unmasked_static_inr

    mask = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.bool)
    mask_path = export_support_mask(mask, np.eye(4), tmp_path / "mask.nii.gz")
    np.testing.assert_array_equal(nib.load(str(mask_path)).get_fdata(), mask.numpy())
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
