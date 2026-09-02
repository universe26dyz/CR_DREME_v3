"""
功能：验证 CardioResp 4D MRI 的 DICOM 扫描、清单与数据集装载契约。
论文来源：Necessary adaptation: image-domain DICOM ingestion for the local free-breathing acquisition.
输入：临时生成的、无 PHI 的合成 DICOM 数据。
输出：针对扫描顺序、清单字段及图像归一化的 unittest 断言。
主要步骤：生成 DICOM，调用公开 API，并断言可复现的输出。
是否属于原论文直接实现 / 必要适配 / 可选实验：Necessary adaptation.
命令行使用示例：python -m unittest tests.test_data -v
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
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cardioresp4d.config import AppConfig, config_from_mapping  # noqa: E402
from cardioresp4d.data.build_manifest import build_manifest  # noqa: E402
from cardioresp4d.data.dataset import CardioRespDataset, _rescaled_pixels  # noqa: E402
from cardioresp4d.data.inspect_dataset import scan_dicom_frames, write_inspection  # noqa: E402


def nested_config_mapping(dicom_root: Path, results_dir: Path, expected_frames: int = 50) -> dict:
    """Build the module-scoped Phase-1 YAML shape used by configuration tests."""
    return {
        "project": {"results_dir": str(results_dir)},
        "data": {"dicom_root": str(dicom_root), "views": ["SAX", "2CH", "4CH"],
                 "expected_frames_per_slice": expected_frames},
        "geometry": {"required_views": ["SAX", "2CH", "4CH"]},
        "frequency": {},
        "roi": {"required_views": ["SAX", "2CH", "4CH"]},
        "reference": {"source_view": "SAX"},
    }


def write_dicom(
    path: Path,
    acquisition_time: str,
    pixels: np.ndarray,
    slope: float = 1.0,
    intercept: float = 0.0,
) -> None:
    """Write a minimal MR-like DICOM frame for an isolated test."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.SeriesInstanceUID = generate_uid()
    dataset.StudyInstanceUID = generate_uid()
    dataset.PatientName = "Synthetic^Person"
    dataset.PatientID = "synthetic-id"
    dataset.Modality = "MR"
    dataset.SeriesDescription = path.parent.name
    dataset.ProtocolName = path.parent.name
    dataset.AcquisitionTime = acquisition_time
    dataset.ContentTime = "235959.999"  # Scanner must not use this field.
    dataset.InstanceNumber = int(path.stem)
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.ImagePositionPatient = [1.0, 2.0, 3.0]
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.PixelSpacing = [1.5, 2.5]
    dataset.SliceThickness = 8.0
    dataset.SpacingBetweenSlices = 10.0
    dataset.RescaleSlope = slope
    dataset.RescaleIntercept = intercept
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = pixels.astype(np.uint16).tobytes()
    dataset.save_as(path, write_like_original=False)


class DataPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name) / "dicom"
        self.sax = self.root / "SAX_s01_1001"
        self.two_ch = self.root / "2ch_s01_2001"
        self.scout = self.root / "scout_sax_001"
        for directory in (self.sax, self.two_ch, self.scout):
            directory.mkdir(parents=True)
        pixels = np.array([[0, 10], [20, 100]], dtype=np.uint16)
        for index in range(50):
            # Deliberately reverse filename order so ordering must use AcquisitionTime.
            write_dicom(
                self.sax / f"{50 - index:08d}.dcm",
                f"1200{index:02d}.000",
                pixels,
                slope=2.0,
                intercept=-10.0,
            )
            write_dicom(
                self.two_ch / f"{50 - index:08d}.dcm",
                f"1300{index:02d}.000",
                pixels,
            )
        write_dicom(self.scout / "00000001.dcm", "140000.000", pixels)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_scan_filters_views_and_orders_each_fixed_slice_by_acquisition_time(self) -> None:
        (self.sax / "00000050.dcm").rename(self.sax / "00000050.DCM")
        frames = scan_dicom_frames(self.root, ["SAX", "2CH"])

        self.assertEqual(100, len(frames))
        self.assertEqual({"SAX", "2CH"}, {frame["view"] for frame in frames})
        sax_frames = [frame for frame in frames if frame["view"] == "SAX"]
        two_ch_frames = [frame for frame in frames if frame["view"] == "2CH"]
        self.assertEqual(50, len(sax_frames))
        self.assertEqual(50, len(two_ch_frames))
        self.assertEqual(list(range(50)), [frame["frame_index"] for frame in sax_frames])
        self.assertEqual(sorted(frame["timestamp_s"] for frame in sax_frames), [frame["timestamp_s"] for frame in sax_frames])
        self.assertEqual(43200.0, sax_frames[0]["timestamp_s"])
        self.assertNotIn("ContentTime", sax_frames[0])

    def test_scan_rejects_selected_slice_with_extra_frame(self) -> None:
        write_dicom(
            self.two_ch / "00000051.dcm",
            "130050.000",
            np.array([[0, 10], [20, 100]], dtype=np.uint16),
        )

        with self.assertRaisesRegex(ValueError, "expected 50 frames"):
            scan_dicom_frames(self.root, ["2CH"])

    def test_scan_rejects_selected_slice_with_incomplete_frame_count(self) -> None:
        (self.two_ch / "00000050.dcm").unlink()

        with self.assertRaisesRegex(ValueError, "expected 50 frames"):
            scan_dicom_frames(self.root, ["2CH"])

    def test_manifest_uses_only_non_phi_columns(self) -> None:
        output_dir = Path(self.tempdir.name) / "results"
        config = AppConfig(
            dicom_root=self.root,
            views=("SAX", "2CH"),
            results_dir=output_dir,
            manifest_stem="manifest",
        )

        csv_path, json_path = build_manifest(config)

        self.assertTrue(csv_path.is_file())
        self.assertTrue(json_path.is_file())
        with csv_path.open(newline="", encoding="utf-8") as handle:
            columns = csv.DictReader(handle).fieldnames or []
        forbidden = {
            "patient_name",
            "patient_id",
            "patient_birth_date",
            "accession_number",
            "study_instance_uid",
        }
        self.assertTrue(forbidden.isdisjoint({column.lower() for column in columns}))
        with json_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.assertTrue(forbidden.isdisjoint(manifest["frames"][0]))

    def test_config_and_manifest_reject_results_inside_dicom_root(self) -> None:
        with self.assertRaisesRegex(ValueError, "results_dir"):
            config_from_mapping(
                nested_config_mapping(self.root, self.root / "results"),
                self.root,
            )
        with self.assertRaisesRegex(ValueError, "results_dir"):
            build_manifest(AppConfig(self.root, ("SAX",), self.root / "results"))

    def test_config_requires_exactly_50_frames_per_slice(self) -> None:
        for expected_frames in (49, 51):
            with self.subTest(expected_frames=expected_frames):
                with self.assertRaisesRegex(ValueError, "exactly 50"):
                    config_from_mapping(
                        nested_config_mapping(self.root, Path(self.tempdir.name) / "results", expected_frames),
                        self.root,
                    )

    def test_inspection_rejects_output_inside_dicom_root(self) -> None:
        with self.assertRaisesRegex(ValueError, "output"):
            write_inspection(self.root, ["SAX"], self.root / "inspection.json")

    def test_dataset_applies_rescale_and_percentile_normalization(self) -> None:
        output_dir = Path(self.tempdir.name) / "results"
        csv_path, _ = build_manifest(
            AppConfig(self.root, ("SAX",), output_dir, "manifest")
        )

        sample = CardioRespDataset(csv_path)[0]

        self.assertEqual("SAX", sample["view"])
        self.assertEqual("SAX_s01_1001", sample["slice_id"])
        self.assertEqual("modality_lut", sample["rescale_status"])
        self.assertEqual(43200.0, sample["timestamp_s"])
        self.assertEqual([1.0, 2.0, 3.0], sample["geometry"]["image_position_patient"])
        self.assertEqual(np.float32, sample["image"].dtype)
        self.assertAlmostEqual(0.0, float(sample["image"].min()))
        self.assertAlmostEqual(1.0, float(sample["image"].max()))
        self.assertGreater(float(sample["image"][0, 1]), 0.0)

    def test_dataset_keeps_stored_pixels_when_both_rescale_tags_are_absent(self) -> None:
        csv_path, _ = build_manifest(
            AppConfig(self.root, ("SAX",), Path(self.tempdir.name) / "results", "manifest")
        )
        dicom_path = self.sax / "00000050.dcm"
        dataset = pydicom.dcmread(dicom_path)
        del dataset.RescaleSlope
        del dataset.RescaleIntercept
        dataset.save_as(dicom_path, write_like_original=False)

        stored_pixels = pydicom.dcmread(dicom_path).pixel_array.astype(np.float32)
        np.testing.assert_array_equal(stored_pixels, _rescaled_pixels(pydicom.dcmread(dicom_path)))
        self.assertEqual("identity_without_rescale_tags", CardioRespDataset(csv_path)[0]["rescale_status"])

    def test_dataset_rejects_partial_rescale_metadata(self) -> None:
        csv_path, _ = build_manifest(
            AppConfig(self.root, ("SAX",), Path(self.tempdir.name) / "results", "manifest")
        )
        dicom_path = self.sax / "00000050.dcm"
        dataset = pydicom.dcmread(dicom_path)
        del dataset.RescaleSlope
        dataset.save_as(dicom_path, write_like_original=False)

        with self.assertRaisesRegex(ValueError, "partial rescale"):
            CardioRespDataset(csv_path)[0]

    @unittest.skipUnless(
        os.environ.get("CARDIORESP4D_REAL_DICOM_ROOT"),
        "Set CARDIORESP4D_REAL_DICOM_ROOT to run the real first-frame loader test.",
    )
    def test_real_first_frame_loader(self) -> None:
        real_root = Path(os.environ["CARDIORESP4D_REAL_DICOM_ROOT"])
        with tempfile.TemporaryDirectory() as output_directory:
            csv_path, _ = build_manifest(
                AppConfig(real_root, ("SAX", "2CH", "4CH"), Path(output_directory), "manifest")
            )
            sample = CardioRespDataset(csv_path)[0]

        self.assertEqual("identity_without_rescale_tags", sample["rescale_status"])
        self.assertTrue(np.isfinite(sample["image"]).all())
        self.assertEqual((240, 213), sample["image"].shape)


if __name__ == "__main__":
    unittest.main()
