"""Test the explicit multi-view canonical-domain contract."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import nibabel as nib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardioresp4d.geometry.canonical_domain import build_canonical_domain  # noqa: E402
from cardioresp4d.roi.cardiac_box import CardiacBox  # noqa: E402


class CanonicalDomainTest(unittest.TestCase):
    def test_domain_uses_multiview_support_and_cardiac_box_not_reference_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            fields = ("view", "slice_id", "frame_index", "image_position_patient",
                      "image_orientation_patient", "pixel_spacing", "slice_thickness",
                      "spacing_between_slices", "rows", "columns")
            rows = (
                ("SAX", [0, 0, 0], [1, 0, 0, 0, 1, 0]),
                ("2CH", [0, 4, 0], [1, 0, 0, 0, 0, 1]),
                ("4CH", [4, 0, 0], [0, 1, 0, 0, 0, 1]),
            )
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
                for view, origin, orientation in rows:
                    writer.writerow({"view": view, "slice_id": f"{view}_1", "frame_index": 0,
                                     "image_position_patient": json.dumps(origin),
                                     "image_orientation_patient": json.dumps(orientation),
                                     "pixel_spacing": json.dumps([1, 1]), "slice_thickness": 8,
                                     "spacing_between_slices": 2, "rows": 9, "columns": 9})
            report, outputs = build_canonical_domain(
                manifest, CardiacBox(np.array([4.0, 4.0, 0.0]), np.array([4.0, 4.0, 4.0])),
                root / "geometry", coverage_spacing_mm=2.0,
            )
            payload = json.loads(report.read_text())
            self.assertEqual("multi_view_acquisition_supported_full_fov", payload["derivation_rule"])
            self.assertEqual([0.0, 0.0, -2.0], payload["world_min_mm"])
            self.assertEqual([8.0, 8.0, 8.0], payload["world_max_mm"])
            self.assertEqual([4.0, 4.0, 5.0], payload["scale_mm_per_normalized_unit"])
            self.assertEqual("legacy_not_used_by_mainline", payload["initial_reference_support"]["role"])
            self.assertTrue(all(path.is_file() for path in outputs.values()))
            normalised = np.asarray(payload["cardiac_box"]["normalized"]["corners"])
            self.assertTrue(np.all(normalised >= -1.0) and np.all(normalised <= 1.0))
            self.assertEqual(1, int(np.asanyarray(nib.load(outputs["canonical_domain_mask"]).dataobj).min()))
            self.assertEqual(1, int(np.asanyarray(nib.load(outputs["canonical_domain_mask"]).dataobj).max()))
            self.assertTrue(np.any(np.asanyarray(nib.load(outputs["coverage_plane_center_sax"]).dataobj) == 0))
            self.assertGreaterEqual(int(np.asanyarray(nib.load(outputs["coverage_psf_union"]).dataobj).sum()), int(np.asanyarray(nib.load(outputs["coverage_plane_center_sax"]).dataobj).sum()))
            self.assertLessEqual(int(np.asanyarray(nib.load(outputs["coverage_view_count"]).dataobj).max()), 3)
            self.assertGreater(int(np.asanyarray(nib.load(outputs["coverage_observation_count"]).dataobj).max()), 0)

            reference_mask = root / "subject_specific_mask.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4)), reference_mask)
            masked_report, _ = build_canonical_domain(
                manifest, CardiacBox(np.array([4.0, 4.0, 0.0]), np.array([4.0, 4.0, 4.0])),
                root / "geometry_masked", initial_reference_mask_path=reference_mask, coverage_spacing_mm=2.0,
            )
            masked = json.loads(masked_report.read_text())["initial_reference_support"]
            self.assertTrue(masked["available"])
            self.assertNotIn("path", masked)


if __name__ == "__main__":
    unittest.main()
