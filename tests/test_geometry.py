"""
功能：验证 DICOM PS3.3 病人世界坐标、受试者归一化和几何质控的公开契约。
论文来源：Necessary adaptation: physical-coordinate handling for local image-domain MRI.
输入：手工推导的各向异性平面几何和临时 manifest。
输出：针对坐标变换、边界、归一化矩阵与 QC 工件的 unittest 断言。
主要步骤：构造已知几何、调用公开 API，并比较精确的物理坐标和误差界。
是否属于原论文直接实现 / 必要适配 / 可选实验：Necessary adaptation.
命令行使用示例：python -m unittest tests.test_geometry -v
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cardioresp4d.geometry.coordinate_normalization import WorldNormalizer  # noqa: E402
from cardioresp4d.geometry.geometry_qc import choose_qc_planes, run_geometry_qc  # noqa: E402
from cardioresp4d.geometry.world_geometry import DicomPlane  # noqa: E402


GEOMETRY = {
    "image_position_patient": [10.0, 20.0, 30.0],
    # First triplet advances with DICOM column index; second with row index.
    "image_orientation_patient": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    # DICOM PixelSpacing is (row spacing, column spacing).
    "pixel_spacing": [2.0, 3.0],
    "slice_thickness": 8.0,
    "spacing_between_slices": 10.0,
    "rows": 4,
    "columns": 5,
}


class DicomPlaneTest(unittest.TestCase):
    """Exercise the physical-coordinate convention with hand-derived values."""

    def setUp(self) -> None:
        """Build one axial plane with intentionally unequal row and column spacing."""
        self.plane = DicomPlane.from_geometry(GEOMETRY)

    def test_pixel_to_world_uses_column_spacing_for_column_index(self) -> None:
        """A spacing swap changes this hand-derived patient-world coordinate."""
        np.testing.assert_allclose(
            self.plane.pixel_to_world(column=2.0, row=3.0),
            [16.0, 26.0, 30.0],
            atol=1e-12,
        )

    def test_world_to_pixel_inverts_plane_coordinates(self) -> None:
        """In-plane physical coordinates invert to the public (column, row) order."""
        pixel = np.array([1.25, 2.5])
        recovered = self.plane.world_to_pixel(self.plane.pixel_to_world(*pixel))
        np.testing.assert_allclose(recovered, pixel, atol=1e-12)

    def test_world_to_pixel_inverts_rounded_dicom_direction_cosines(self) -> None:
        """A small stored IOP cross-term must not prevent a sub-micrometre in-plane inverse."""
        geometry = dict(GEOMETRY)
        geometry["image_orientation_patient"] = [1.0, 0.0, 0.0, 1e-7, 1.0, 0.0]
        plane = DicomPlane.from_geometry(geometry)
        pixel = np.array([4.25, 2.5])
        np.testing.assert_allclose(plane.world_to_pixel(plane.pixel_to_world(pixel)), pixel, atol=1e-12)

    def test_pixel_to_world_preserves_raw_nonunit_dicom_iop(self) -> None:
        """Forward coordinates and serialization must use raw stored IOP, not normalized vectors."""
        geometry = dict(GEOMETRY)
        raw_iop = [0.999999, 0.0, 0.0, 0.0, 1.000001, 0.0]
        geometry["image_orientation_patient"] = raw_iop
        plane = DicomPlane.from_geometry(geometry)

        np.testing.assert_allclose(plane.pixel_to_world(2.0, 3.0), [15.999994, 26.000006, 30.0], atol=1e-12)
        np.testing.assert_allclose(plane.to_dict()["image_orientation_patient"], raw_iop, atol=0.0)

    def test_normal_and_orientation_are_orthonormal(self) -> None:
        """The normal is row-direction cross column-direction with unit orthogonal axes."""
        np.testing.assert_allclose(self.plane.normal, [0.0, 0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(self.plane.orientation_matrix.T @ self.plane.orientation_matrix, np.eye(3), atol=1e-12)

    def test_bounds_and_corners_follow_zero_based_pixel_extent(self) -> None:
        """Corners include exactly the valid integer pixel indices, not one-past-the-end."""
        self.assertTrue(self.plane.contains_pixel(4.0, 3.0))
        self.assertFalse(self.plane.contains_pixel(5.0, 3.0))
        expected = np.array([[10.0, 20.0, 30.0], [22.0, 20.0, 30.0], [22.0, 26.0, 30.0], [10.0, 26.0, 30.0]])
        np.testing.assert_allclose(self.plane.corners(), expected, atol=1e-12)


class WorldNormalizerTest(unittest.TestCase):
    """Verify serializable subject-bounding-box normalization on plane corners."""

    def test_normalizer_maps_all_plane_corners_to_symmetric_bounds_and_inverts_batches(self) -> None:
        """The fitted matrices map the aggregate box to [-1, 1] and preserve point batches."""
        first = DicomPlane.from_geometry(GEOMETRY)
        second_geometry = dict(GEOMETRY)
        second_geometry["image_position_patient"] = [10.0, 20.0, 40.0]
        second = DicomPlane.from_geometry(second_geometry)
        normalizer = WorldNormalizer.from_planes([first, second])

        corners = np.concatenate([first.corners(), second.corners()])
        normalised = normalizer.normalize(corners)
        np.testing.assert_allclose(normalised.min(axis=0), [-1.0, -1.0, -1.0], atol=1e-12)
        np.testing.assert_allclose(normalised.max(axis=0), [1.0, 1.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(normalizer.denormalize(normalised), corners, atol=1e-12)
        payload = normalizer.to_dict()
        restored = WorldNormalizer.from_dict(payload)
        np.testing.assert_allclose(restored.normalize(corners), normalised, atol=1e-12)
        np.testing.assert_allclose(restored.forward_matrix @ restored.inverse_matrix, np.eye(4), atol=1e-12)


class GeometryQcTest(unittest.TestCase):
    """Check that QC writes inspectable JSON and PNG artifacts from a manifest."""

    def test_qc_writes_errors_below_one_micrometre(self) -> None:
        """QC reports both requested round trips and renders the selected plane geometry."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("view", "slice_id", "frame_index", *GEOMETRY.keys()))
                writer.writeheader()
                for view, offset in (("SAX", 0.0), ("2CH", 15.0), ("4CH", 30.0)):
                    geometry = dict(GEOMETRY)
                    geometry["image_position_patient"] = [10.0, 20.0 + offset, 30.0]
                    writer.writerow({
                        "view": view,
                        "slice_id": f"{view}_s01",
                        "frame_index": 0,
                        **{key: json.dumps(value) if isinstance(value, list) else value for key, value in geometry.items()},
                    })

            report_path, image_path = run_geometry_qc(manifest, root / "qc")
            self.assertTrue(report_path.is_file())
            self.assertTrue(image_path.is_file())
            with report_path.open(encoding="utf-8") as handle:
                report = json.load(handle)
            self.assertEqual({"SAX", "2CH", "4CH"}, {plane["view"] for plane in report["planes"]})
            self.assertLess(report["round_trip_errors"]["pixel_to_world_to_pixel_max_error_pixels"], 1e-9)
            self.assertLess(report["round_trip_errors"]["world_to_pixel_to_world_max_error_mm"], 1e-6)
            self.assertEqual((4, 4), np.asarray(report["world_normalizer"]["forward_matrix"]).shape)

    def test_qc_rejects_manifest_missing_a_required_view(self) -> None:
        """QC must refuse a partial acquisition instead of silently omitting a mandated view."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "partial_manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("view", "slice_id", "frame_index", *GEOMETRY.keys()))
                writer.writeheader()
                for view in ("SAX", "2CH"):
                    writer.writerow({
                        "view": view,
                        "slice_id": f"{view}_s01",
                        "frame_index": 0,
                        **{key: json.dumps(value) if isinstance(value, list) else value for key, value in GEOMETRY.items()},
                    })
            with self.assertRaisesRegex(ValueError, "SAX, 2CH, 4CH"):
                run_geometry_qc(manifest, root / "qc")

    def test_qc_sampling_orders_physical_planes_by_stack_normal(self) -> None:
        """Representative plane sampling follows physical stack position, not slice-id lexicography."""
        records = []
        for slice_id, z_position in (("SAX_a", 20.0), ("SAX_b", 0.0), ("SAX_c", 10.0)):
            geometry = dict(GEOMETRY)
            geometry["image_position_patient"] = [10.0, 20.0, z_position]
            records.append(({"view": "SAX", "slice_id": slice_id, "frame_index": "0"}, DicomPlane.from_geometry(geometry)))
        selected = choose_qc_planes(records, max_per_view=3)
        self.assertEqual([0.0, 10.0, 20.0], [float(plane.pixel_to_world(0.0, 0.0)[2]) for _, plane in selected])


if __name__ == "__main__":
    unittest.main()
