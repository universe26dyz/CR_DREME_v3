"""
功能：验证 DREME-style 独立局部 cardiac world-space box 及三视图物理投影 QC。
论文来源：DREME-MR local cardiac coordinate box；三视图 DICOM 投影属于 necessary adaptation。
输入：手工几何、受试者归一化范围，以及无 PHI 的 50-frame 合成 DICOM manifest。
输出：盒子边界/角点、平面截交、代表层选择与 QC 工件的 unittest 断言。
主要步骤：先验证同一个 3D 盒子的解析投影，再运行 SAX/2CH/4CH temporal-mean overlay。
是否属于原论文直接实现 / 必要适配 / 可选实验：DREME-style direct concept + necessary adaptation.
命令行使用示例：python -m unittest tests.test_roi -v
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cardioresp4d.geometry.coordinate_normalization import WorldNormalizer  # noqa: E402
from cardioresp4d.geometry.world_geometry import DicomPlane  # noqa: E402
from cardioresp4d.roi.cardiac_box import CardiacBox, load_cardiac_box_config  # noqa: E402
from cardioresp4d.roi.roi_qc import (  # noqa: E402
    box_plane_intersection,
    choose_closest_planes,
    project_box_to_plane,
    run_roi_qc,
)


AXIAL_GEOMETRY = {
    "image_position_patient": [10.0, 20.0, 30.0],
    "image_orientation_patient": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "pixel_spacing": [2.0, 3.0],
    "slice_thickness": 8.0,
    "spacing_between_slices": 10.0,
    "rows": 8,
    "columns": 8,
}


class CardiacBoxTest(unittest.TestCase):
    """Exercise the shared axis-aligned patient-world box contract."""

    def test_corners_bounds_and_normalized_round_trip_are_consistent(self) -> None:
        box = CardiacBox(center_mm=[4.0, 5.0, 6.0], size_mm=[4.0, 6.0, 8.0])
        normalizer = WorldNormalizer.from_corners([[0.0, 0.0, 0.0], [10.0, 12.0, 14.0]])

        self.assertEqual((8, 3), box.corners_mm.shape)
        np.testing.assert_allclose(box.min_mm, [2.0, 2.0, 2.0])
        np.testing.assert_allclose(box.max_mm, [6.0, 8.0, 10.0])
        np.testing.assert_allclose(np.unique(box.corners_mm[:, 0]), [2.0, 6.0])
        np.testing.assert_allclose(np.unique(box.corners_mm[:, 1]), [2.0, 8.0])
        np.testing.assert_allclose(np.unique(box.corners_mm[:, 2]), [2.0, 10.0])
        normalized = box.normalized(normalizer)
        np.testing.assert_allclose(normalizer.denormalize(normalized["corners"]), box.corners_mm)
        np.testing.assert_allclose(normalizer.denormalize(normalized["center"]), box.center_mm)

    def test_rejects_nonpositive_size_and_box_outside_subject_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            CardiacBox([0.0, 0.0, 0.0], [2.0, 0.0, 2.0])
        normalizer = WorldNormalizer.from_corners([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
        with self.assertRaisesRegex(ValueError, "outside subject normalization bounds"):
            CardiacBox([9.0, 5.0, 5.0], [4.0, 2.0, 2.0]).validate_within(normalizer)

    def test_loads_subject_values_only_from_explicit_local_roi_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "subject_local.yaml"
            config.write_text(
                "roi:\n"
                "  cardiac_box_center_mm: [4.0, 5.0, 6.0]\n"
                "  cardiac_box_size_mm: [8.0, 10.0, 12.0]\n",
                encoding="utf-8",
            )

            box = load_cardiac_box_config(config)

        np.testing.assert_allclose(box.center_mm, [4.0, 5.0, 6.0])
        np.testing.assert_allclose(box.size_mm, [8.0, 10.0, 12.0])


class CardiacProjectionTest(unittest.TestCase):
    """Verify projection and true plane intersection using hand-derived coordinates."""

    def test_projects_same_eight_world_corners_in_public_column_row_order(self) -> None:
        plane = DicomPlane.from_geometry(AXIAL_GEOMETRY)
        box = CardiacBox(center_mm=[16.0, 24.0, 30.0], size_mm=[6.0, 4.0, 8.0])

        projection = project_box_to_plane(box, plane)

        np.testing.assert_allclose(projection["world_corners_mm"], box.corners_mm)
        np.testing.assert_allclose(np.unique(projection["pixel_corners"][:, 0]), [1.0, 3.0])
        np.testing.assert_allclose(np.unique(projection["pixel_corners"][:, 1]), [1.0, 3.0])
        np.testing.assert_allclose(np.unique(np.abs(projection["off_plane_distance_mm"])), [4.0])

    def test_intersection_uses_box_edges_not_projected_corner_hull(self) -> None:
        plane = DicomPlane.from_geometry(AXIAL_GEOMETRY)
        box = CardiacBox(center_mm=[16.0, 24.0, 30.0], size_mm=[6.0, 4.0, 8.0])

        intersection = box_plane_intersection(box, plane)

        self.assertEqual((4, 3), intersection["world_mm"].shape)
        self.assertEqual((4, 2), intersection["pixel"].shape)
        np.testing.assert_allclose(np.sort(intersection["pixel"][:, 0]), [1.0, 1.0, 3.0, 3.0])
        np.testing.assert_allclose(np.sort(intersection["pixel"][:, 1]), [1.0, 1.0, 3.0, 3.0])
        np.testing.assert_allclose(intersection["world_mm"][:, 2], 30.0)

    def test_intersection_preserves_single_vertex_tangent_contact(self) -> None:
        """A tangent plane touching only the upper box corner yields one audited contact point."""
        root_two = np.sqrt(2.0)
        root_six = np.sqrt(6.0)
        plane = DicomPlane.from_geometry({
            **AXIAL_GEOMETRY,
            "image_position_patient": [1.0, 1.0, 1.0],
            "image_orientation_patient": [
                1.0 / root_two, -1.0 / root_two, 0.0,
                1.0 / root_six, 1.0 / root_six, -2.0 / root_six,
            ],
        })
        box = CardiacBox(center_mm=[0.0, 0.0, 0.0], size_mm=[2.0, 2.0, 2.0])

        intersection = box_plane_intersection(box, plane)

        self.assertEqual((1, 3), intersection["world_mm"].shape)
        np.testing.assert_allclose(intersection["world_mm"][0], [1.0, 1.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(intersection["pixel"][0], [0.0, 0.0], atol=1e-12)

    def test_closest_plane_selection_uses_absolute_physical_distance(self) -> None:
        box = CardiacBox([0.0, 0.0, 8.0], [2.0, 2.0, 2.0])
        records = []
        for slice_id, z in (("SAX_001", 40.0), ("SAX_999", 10.0)):
            geometry = dict(AXIAL_GEOMETRY)
            geometry["image_position_patient"] = [0.0, 0.0, z]
            records.append(({"view": "SAX", "slice_id": slice_id}, DicomPlane.from_geometry(geometry)))

        chosen = choose_closest_planes(records, box.center_mm, required_views=("SAX",))

        self.assertEqual("SAX_999", chosen["SAX"][0]["slice_id"])
        self.assertAlmostEqual(2.0, chosen["SAX"][2])

    def test_closest_plane_selection_rejects_missing_view(self) -> None:
        record = ({"view": "SAX", "slice_id": "SAX_1"}, DicomPlane.from_geometry(AXIAL_GEOMETRY))
        with self.assertRaisesRegex(ValueError, "missing 2CH, 4CH"):
            choose_closest_planes([record], [16.0, 24.0, 30.0])


class RoiQcArtifactTest(unittest.TestCase):
    """Run the temporal-mean QC boundary with a tiny synthetic three-view acquisition."""

    def test_qc_writes_auditable_json_and_three_pngs_from_50_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            fieldnames = (
                "dicom_path", "view", "slice_id", "frame_index", "timestamp_s",
                "image_position_patient", "image_orientation_patient", "pixel_spacing",
                "slice_thickness", "spacing_between_slices", "rows", "columns",
            )
            geometries = {
                "SAX": ([0.0, 0.0, 3.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
                "2CH": ([0.0, 3.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
                "4CH": ([3.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            }
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for view, (origin, orientation) in geometries.items():
                    slice_id = f"{view}_slice"
                    for frame in range(50):
                        path = root / f"{view}_{frame:02d}.dcm"
                        pixels = np.arange(64, dtype=np.uint16).reshape(8, 8) + frame
                        _write_dicom(path, pixels, origin, orientation)
                        writer.writerow({
                            "dicom_path": path,
                            "view": view,
                            "slice_id": slice_id,
                            "frame_index": frame,
                            "timestamp_s": frame * 0.17,
                            "image_position_patient": json.dumps(origin),
                            "image_orientation_patient": json.dumps(orientation),
                            "pixel_spacing": json.dumps([1.0, 1.0]),
                            "slice_thickness": 8.0,
                            "spacing_between_slices": 10.0,
                            "rows": 8,
                            "columns": 8,
                        })

            report_path, image_paths = run_roi_qc(
                manifest,
                root / "qc",
                CardiacBox([3.0, 3.0, 3.0], [4.0, 4.0, 4.0]),
                expected_frames_per_slice=50,
            )

            self.assertTrue(report_path.is_file())
            self.assertEqual({"SAX", "2CH", "4CH"}, set(image_paths))
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in image_paths.values()))
            with report_path.open(encoding="utf-8") as handle:
                report = json.load(handle)
            self.assertEqual(50, report["expected_frames_per_slice"])
            self.assertEqual([3.0, 3.0, 3.0], report["cardiac_box"]["center_mm"])
            self.assertEqual({"SAX", "2CH", "4CH"}, set(report["views"]))
            for view_report in report["views"].values():
                self.assertEqual(50, view_report["frame_count"])
                self.assertAlmostEqual(0.0, view_report["plane_distance_to_box_center_mm"])
                self.assertEqual(4, view_report["intersection"]["vertex_count"])
                self.assertTrue(view_report["intersection"]["all_vertices_in_image"])

    def test_qc_rejects_frame_expectations_other_than_project_contract_50(self) -> None:
        """The public API cannot redefine the acquisition as 49 or 51 frames per slice."""
        with tempfile.TemporaryDirectory() as directory:
            missing_manifest = Path(directory) / "not_needed_for_contract_validation.csv"
            for expected_frames in (49, 51):
                with self.subTest(expected_frames=expected_frames):
                    with self.assertRaisesRegex(ValueError, "exactly 50"):
                        run_roi_qc(
                            missing_manifest,
                            Path(directory) / "qc",
                            CardiacBox([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]),
                            expected_frames_per_slice=expected_frames,
                        )


def _write_dicom(path: Path, pixels: np.ndarray, origin: list[float], orientation: list[float]) -> None:
    """Write one unscaled synthetic image with the physical tags used by the manifest."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "MR"
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.ImagePositionPatient = origin
    dataset.ImageOrientationPatient = orientation
    dataset.PixelSpacing = [1.0, 1.0]
    dataset.SliceThickness = 8.0
    dataset.SpacingBetweenSlices = 10.0
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = pixels.astype(np.uint16).tobytes()
    dataset.save_as(path, write_like_original=False)


if __name__ == "__main__":
    unittest.main()
