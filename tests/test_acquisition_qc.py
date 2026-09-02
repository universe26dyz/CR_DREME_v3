"""Test robust within-slice acquisition corruption masking without deleting DICOM."""
from __future__ import annotations
import csv, sys, tempfile, unittest
from pathlib import Path
import numpy as np
import nibabel as nib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cardioresp4d.outlier_qc.acquisition_qc import analyze_frame_block, write_qc_table
from cardioresp4d.data.dataset import CardioRespDataset
from cardioresp4d.reference.build_initial_reference import build_initial_reference
from tests.test_reference import _write_manifest


def normal_block():
    y, x = np.mgrid[:24, :20]
    base = 100 + 30*np.exp(-((x-10)**2+(y-12)**2)/30)
    return np.stack([np.roll(base, i % 3 - 1, axis=1) for i in range(50)]).astype(np.float32)


class AcquisitionQcTest(unittest.TestCase):
    def test_normal_motion_is_retained_but_global_drop_and_local_bright_are_flagged(self):
        images = normal_block()
        normal = analyze_frame_block(images, ["identity_without_rescale_tags"]*50)
        self.assertGreaterEqual(sum(x["qc_valid"] for x in normal), 47)

        corrupted = images.copy(); corrupted[7] *= .1; corrupted[19, 5:10, 5:10] += 1000
        rows = analyze_frame_block(corrupted, ["identity_without_rescale_tags"]*50)
        self.assertFalse(rows[7]["qc_valid"])
        self.assertFalse(rows[19]["qc_valid"])
        self.assertIn("intensity", rows[7]["qc_reason"])
        self.assertTrue(all(k in rows[0] for k in ("qc_ncc","qc_intensity_scale","qc_residual","rescale_status")))

    def test_qc_table_is_independent_and_phi_free(self):
        rows = analyze_frame_block(normal_block(), ["identity_without_rescale_tags"]*50)
        for i, row in enumerate(rows): row.update(source_file_token=f"f{i:03d}", view="SAX", slice_id="SAX_s001", frame_index=i)
        with tempfile.TemporaryDirectory() as d:
            path = write_qc_table(rows, Path(d)/"acquisition_qc.csv")
            text = path.read_text()
            self.assertNotIn("dicom_path", text)
            with path.open(newline="", encoding="utf-8") as handle:
                parsed = list(csv.DictReader(handle))
            self.assertEqual("1", parsed[0]["qc_valid"])

    def test_initial_reference_arithmetic_mean_ignores_invalid_frame(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); manifest = root / "manifest.csv"
            pixels = np.arange(20, dtype=np.uint16).reshape(4, 5)
            _write_manifest(manifest, root / "dicom", (("slice_a", 0.0, pixels), ("slice_b", 2.0, pixels + 10)))
            qc_rows = []
            for slice_id in ("slice_a", "slice_b"):
                for frame in range(50):
                    valid = not (slice_id == "slice_a" and frame == 0)
                    qc_rows.append({"source_file_token": "", "view": "SAX", "slice_id": slice_id,
                                    "frame_index": frame, "qc_valid": valid, "qc_reason": "valid" if valid else "synthetic_corruption",
                                    "qc_ncc": 1, "qc_intensity_scale": 1, "qc_residual": 0, "rescale_status": "identity_without_rescale_tags"})
            write_qc_table(qc_rows, root / "acquisition_qc" / "acquisition_qc.csv")
            dataset = CardioRespDataset(manifest)
            expected = np.mean(np.stack([dataset[i]["image"] for i, row in enumerate(dataset._rows) if row["slice_id"] == "slice_a"]), axis=0).T
            nifti, metadata, _ = build_initial_reference(manifest, root / "reference")
            volume = np.asanyarray(nib.load(nifti).dataobj)
            np.testing.assert_allclose(volume[:, :, 0], expected, atol=1e-6)
            self.assertIn("49", metadata.read_text())
            for row in qc_rows:
                if row["slice_id"] == "slice_a":
                    row["qc_valid"] = False; row["qc_reason"] = "all_invalid_test"
            write_qc_table(qc_rows, root / "acquisition_qc" / "acquisition_qc.csv")
            with self.assertRaisesRegex(ValueError, "no valid frames"):
                build_initial_reference(manifest, root / "reference_all_invalid")


if __name__ == "__main__": unittest.main()
