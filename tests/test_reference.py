"""
功能：验证 S2V-DREME-style SAX temporal-average initial reference 的排序、轴顺序和物理仿射。
论文来源：S2V-DREME Stage I 每个固定层时间平均后堆叠；DICOM LPS 到 NIfTI RAS+ 是 necessary adaptation。
输入：非方形、50-frame 合成 SAX DICOM manifest。
输出：针对 initial_reference.nii.gz / JSON / QC PNG 的 unittest 断言。
主要步骤：制造目录顺序与物理顺序相反的层，验证 (column,row,slice) 及 LPS→RAS。
是否属于原论文直接实现 / 必要适配 / 可选实验：Direct S2V-DREME principle + necessary coordinate adaptation.
命令行使用示例：python -m unittest tests.test_reference -v
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cardioresp4d.data.dataset import _normalise_image  # noqa: E402
from cardioresp4d.reference.build_initial_reference import build_initial_reference  # noqa: E402


FIELDS = (
    "dicom_path", "view", "slice_id", "frame_index", "timestamp_s",
    "image_position_patient", "image_orientation_patient", "pixel_spacing",
    "slice_thickness", "spacing_between_slices", "rows", "columns",
)


class InitialReferenceTest(unittest.TestCase):
    def test_whole_hard_invalid_sax_location_is_finite_masked_supervision_hole(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            layers = tuple(
                (f"SAX_{index + 1}", float(index * 10), np.full((2, 3), index + 1, dtype=np.uint16))
                for index in range(5)
            )
            _write_manifest(manifest, root / "dicom", layers)
            _write_qc_table(manifest, root / "acquisition_qc" / "acquisition_qc.csv", invalid_slice_id="SAX_3")

            reference_path, metadata_path, _ = build_initial_reference(manifest, root / "reference")
            reference = nib.load(reference_path)
            mask = nib.load(root / "reference" / "initial_reference_valid_mask.nii.gz")
            values = np.asanyarray(reference.dataobj)
            mask_values = np.asanyarray(mask.dataobj)
            self.assertEqual((3, 2, 5), reference.shape)
            np.testing.assert_allclose(reference.affine[:3, 2], [0.0, 0.0, 10.0])
            np.testing.assert_allclose(mask.affine, reference.affine)
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue(np.all(mask_values[:, :, 2] == 0))
            self.assertTrue(np.all(mask_values[:, :, [0, 1, 3, 4]] == 1))
            metadata = json.loads(metadata_path.read_text())
            self.assertEqual(["SAX_3"], metadata["missing_slice_ids"])
            self.assertEqual(1, metadata["missing_slice_count"])
            self.assertEqual("mask_zero_means_no_stage1a_reference_supervision", metadata["supervision_contract"])
            self.assertIn("linear", metadata["placeholder_method"])

    def test_temporal_means_are_transposed_and_sorted_by_physical_position(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            # DICOM arrays are (rows=2, columns=3). Names deliberately disagree
            # with physical z ordering to protect against directory-based stacking.
            layers = (
                ("SAX_a_z20", 20.0, np.array([[21, 22, 23], [24, 25, 31]], dtype=np.uint16)),
                ("SAX_m_z10", 10.0, np.array([[11, 12, 13], [14, 15, 21]], dtype=np.uint16)),
                ("SAX_z_z00", 0.0, np.array([[1, 2, 3], [4, 5, 11]], dtype=np.uint16)),
            )
            _write_manifest(manifest, root, layers, temporal_flip=True)

            nii_path, metadata_path, qc_path = build_initial_reference(manifest, root / "reference")

            image = nib.load(nii_path)
            volume = np.asanyarray(image.dataobj)
            self.assertEqual((3, 2, 3), volume.shape)
            for index, (_, _, pixels) in enumerate(reversed(layers)):
                known_mean = 0.5 * (_normalise_image(pixels) + _normalise_image(np.fliplr(pixels)))
                np.testing.assert_allclose(volume[:, :, index], known_mean.T, atol=1e-6)
            self.assertTrue(qc_path.is_file() and qc_path.stat().st_size > 0)
            with metadata_path.open(encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.assertEqual("SAX", metadata["source_view"])
            self.assertEqual("(column,row,slice)", metadata["array_order"])
            self.assertEqual(["SAX_z_z00", "SAX_m_z10", "SAX_a_z20"], metadata["sorted_slice_ids"])
            self.assertEqual([50, 50, 50], metadata["frames_per_slice"])
            self.assertEqual([3, 2, 3], metadata["shape"])
            self.assertEqual(8.0, metadata["acquisition_slice_thickness_mm"])
            self.assertEqual("IPP centre-to-centre spacing for stack/world geometry",
                             metadata["stack_slice_spacing_semantics"])
            self.assertIn("thick-slice renderer", metadata["acquisition_slice_thickness_semantics"])
            mask = nib.load(root / "reference" / "initial_reference_valid_mask.nii.gz")
            self.assertEqual(volume.shape, mask.shape)
            self.assertTrue(np.all(np.asanyarray(mask.dataobj) == 1))
            self.assertEqual([], metadata["missing_slice_ids"])

    def test_affines_use_dicom_column_row_steps_and_explicit_lps_to_ras(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            layers = (
                ("SAX_2", 14.0, np.arange(6, dtype=np.uint16).reshape(2, 3) + 20),
                ("SAX_0", -6.0, np.arange(6, dtype=np.uint16).reshape(2, 3)),
                ("SAX_1", 4.0, np.arange(6, dtype=np.uint16).reshape(2, 3) + 10),
            )
            _write_manifest(manifest, root, layers, origin_xy=(7.0, 8.0), pixel_spacing=(2.0, 3.0))

            nii_path, metadata_path, _ = build_initial_reference(manifest, root / "reference")

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            lps = np.asarray(metadata["dicom_lps_affine"])
            expected_lps = np.array([
                [3.0, 0.0, 0.0, 7.0],
                [0.0, 2.0, 0.0, 8.0],
                [0.0, 0.0, 10.0, -6.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
            np.testing.assert_allclose(lps, expected_lps)
            expected_ras = np.diag([-1.0, -1.0, 1.0, 1.0]) @ expected_lps
            np.testing.assert_allclose(metadata["nifti_ras_affine"], expected_ras)
            np.testing.assert_allclose(nib.load(nii_path).affine, expected_ras)
            self.assertEqual(0.0, metadata["geometry_validation"]["max_origin_affine_residual_mm"])

    def test_rejects_nonparallel_geometry_duplicate_locations_and_non_50_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pixels = np.arange(6, dtype=np.uint16).reshape(2, 3)
            cases = {
                "duplicate": (
                    (("SAX_0", 0.0, pixels), ("SAX_1", 0.0, pixels)),
                    None,
                    "duplicate physical slice",
                ),
                "orientation": (
                    (("SAX_0", 0.0, pixels), ("SAX_1", 10.0, pixels)),
                    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    "orientation",
                ),
            }
            for name, (layers, second_orientation, message) in cases.items():
                with self.subTest(name=name):
                    manifest = root / f"{name}.csv"
                    _write_manifest(manifest, root / name, layers, second_orientation=second_orientation)
                    with self.assertRaisesRegex(ValueError, message):
                        build_initial_reference(manifest, root / f"out_{name}")

            manifest = root / "count.csv"
            _write_manifest(manifest, root / "count", (("SAX_0", 0.0, pixels),), frame_count=49)
            with self.assertRaisesRegex(ValueError, "exactly 50"):
                build_initial_reference(manifest, root / "out_count")

    def test_rejects_inconsistent_matrix_spacing_and_irregular_origins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pixels = np.arange(6, dtype=np.uint16).reshape(2, 3)
            layers = (("SAX_0", 0.0, pixels), ("SAX_1", 10.0, pixels), ("SAX_2", 25.0, pixels))
            manifest = root / "irregular.csv"
            _write_manifest(manifest, root / "irregular", layers)
            with self.assertRaisesRegex(ValueError, "affine residual"):
                build_initial_reference(manifest, root / "out_irregular", origin_tolerance_mm=0.5)

            manifest = root / "spacing.csv"
            _write_manifest(manifest, root / "spacing", layers[:2], second_pixel_spacing=(2.0, 4.0))
            with self.assertRaisesRegex(ValueError, "spacing"):
                build_initial_reference(manifest, root / "out_spacing")


def _write_manifest(
    manifest: Path,
    root: Path,
    layers: tuple[tuple[str, float, np.ndarray], ...],
    *,
    origin_xy: tuple[float, float] = (0.0, 0.0),
    pixel_spacing: tuple[float, float] = (2.0, 3.0),
    second_pixel_spacing: tuple[float, float] | None = None,
    second_orientation: list[float] | None = None,
    frame_count: int = 50,
    temporal_flip: bool = False,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for layer_index, (slice_id, z, pixels) in enumerate(layers):
            orientation = second_orientation if layer_index == 1 and second_orientation else [1, 0, 0, 0, 1, 0]
            spacing = second_pixel_spacing if layer_index == 1 and second_pixel_spacing else pixel_spacing
            origin = [origin_xy[0], origin_xy[1], z]
            for frame_index in range(frame_count):
                path = root / f"{slice_id}_{frame_index:02d}.dcm"
                frame_pixels = np.fliplr(pixels) if temporal_flip and frame_index >= frame_count // 2 else pixels
                _write_dicom(path, frame_pixels, origin, orientation, spacing)
                writer.writerow({
                    "dicom_path": path,
                    "view": "SAX",
                    "slice_id": slice_id,
                    "frame_index": frame_index,
                    "timestamp_s": frame_index * 0.17,
                    "image_position_patient": json.dumps(origin),
                    "image_orientation_patient": json.dumps(orientation),
                    "pixel_spacing": json.dumps(spacing),
                    "slice_thickness": 8.0,
                    "spacing_between_slices": 10.0,
                    "rows": pixels.shape[0],
                    "columns": pixels.shape[1],
                })


def _write_dicom(path: Path, pixels: np.ndarray, origin: list[float], orientation: list[float], spacing: tuple[float, float]) -> None:
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
    dataset.PixelSpacing = list(spacing)
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


def _write_qc_table(manifest: Path, path: Path, *, invalid_slice_id: str) -> None:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("view", "slice_id", "frame_index", "qc_valid", "qc_reason"))
        writer.writeheader()
        for row in rows:
            invalid = row["slice_id"] == invalid_slice_id
            writer.writerow({"view": row["view"], "slice_id": row["slice_id"], "frame_index": row["frame_index"], "qc_valid": "0" if invalid else "1", "qc_reason": "manual_exclusion" if invalid else "not_evaluated"})


if __name__ == "__main__":
    unittest.main()
