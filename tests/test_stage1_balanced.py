"""Focused contracts for balanced cardiac-ROI Stage-1B sampling."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from cardioresp4d.training.sampling import (
    BalancedViewEpochSampler,
    pixel_indices_to_world,
    roi_pixel_pool,
    sample_pixels,
)
from cardioresp4d.training.stage1 import (
    _load_checkpoint,
    train_stage1a,
    train_stage1b_balanced,
)
from cardioresp4d.training.static_qc import run_final_static_qc


def _geometry(*, origin=(0.0, 0.0, 0.0), orientation=(1, 0, 0, 0, 1, 0)):
    return {
        "image_position_patient": list(origin),
        "image_orientation_patient": list(orientation),
        "pixel_spacing": [2.0, 3.0],
        "slice_thickness": 4.0,
        "rows": 4,
        "columns": 4,
    }


class Stage1BalancedSamplingTests(unittest.TestCase):
    def test_pixel_indices_to_world_respects_dicom_row_column_semantics(self):
        axial = _geometry(origin=(10, 20, 30))
        actual = pixel_indices_to_world([2], [3], axial)
        np.testing.assert_allclose(actual, [[19, 24, 30]])
        oblique = _geometry(
            origin=(1, 2, 3),
            orientation=(0, 1, 0, -1, 0, 0),
        )
        np.testing.assert_allclose(
            pixel_indices_to_world([2], [3], oblique), [[-3, 11, 3]]
        )

    def test_roi_pool_preserves_whole_slice_when_heart_is_absent(self):
        geometry = _geometry()
        partial = roi_pixel_pool(geometry, [0, 0, -1], [3, 2, 1], margin_mm=0)
        self.assertTrue(partial["intersects_roi"])
        self.assertEqual(set(partial["roi"]), {0, 1, 4, 5})
        self.assertEqual(len(partial["rows"]), 16)
        full = roi_pixel_pool(geometry, [-1, -1, -1], [100, 100, 1], margin_mm=0)
        self.assertEqual(len(full["roi"]), 16)
        absent = roi_pixel_pool(geometry, [100, 100, -1], [101, 101, 1], margin_mm=0)
        self.assertFalse(absent["intersects_roi"])
        self.assertEqual(len(absent["rows"]), 16)

    def test_roi_global_sampling_is_reproducible_and_has_explicit_draw_count(self):
        pool = roi_pixel_pool(_geometry(), [0, 0, -1], [3, 2, 1], margin_mm=0)
        one = sample_pixels(pool, 100, 0.8, torch.Generator().manual_seed(9))
        two = sample_pixels(pool, 100, 0.8, torch.Generator().manual_seed(9))
        np.testing.assert_array_equal(one[0], two[0])
        np.testing.assert_array_equal(one[1], two[1])
        self.assertEqual(one[2], 80)
        absent = roi_pixel_pool(_geometry(), [100, 100, -1], [101, 101, 1], margin_mm=0)
        rows, columns, n_roi = sample_pixels(absent, 100, 0.8, torch.Generator().manual_seed(9))
        self.assertEqual((len(rows), len(columns), n_roi), (100, 100, 0))

    def test_balanced_epoch_has_one_observation_of_each_view_per_step(self):
        source = {"SAX": list(range(5)), "2CH": list(range(3)), "4CH": list(range(4))}
        sampler = BalancedViewEpochSampler(source, seed=11)
        batches = list(sampler.epoch())
        self.assertEqual(sampler.steps_per_epoch, 5)
        self.assertTrue(all(set(batch) == {"SAX", "2CH", "4CH"} for batch in batches))
        for view, expected in source.items():
            self.assertTrue(set(expected).issubset({batch[view] for batch in batches}))

    def test_missing_sax_location_is_not_synthesized_for_balanced_epoch(self):
        sampler = BalancedViewEpochSampler(
            {"SAX": ["s16", "s18"], "2CH": ["two"], "4CH": ["four"]}, seed=4
        )
        sax = {batch["SAX"] for batch in sampler.epoch()}
        self.assertEqual(sax, {"s16", "s18"})
        self.assertNotIn("s17", sax)

    def test_balanced_training_cpu_writes_checkpoint_metadata_and_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            domain = root / "domain.json"
            domain.write_text(json.dumps({
                "world_min_mm": [0, 0, 0], "world_max_mm": [3, 3, 3],
                "world_to_normalized": [[2 / 3, 0, 0, -1], [0, 2 / 3, 0, -1], [0, 0, 2 / 3, -1], [0, 0, 0, 1]],
                "cardiac_box": {"center_mm": [1.5, 1.5, 1.5], "size_mm": [3, 3, 3]},
            }))
            reference = root / "reference.nii.gz"
            mask = root / "mask.nii.gz"
            volume = np.linspace(0, 1, 64, dtype=np.float32).reshape(4, 4, 4)
            affine = np.diag([-1.0, -1.0, 1.0, 1.0])
            nib.save(nib.Nifti1Image(volume, affine), reference)
            nib.save(nib.Nifti1Image(np.ones_like(volume), affine), mask)
            stage1a = train_stage1a(reference, mask, domain, root / "stage1a", steps=2, batch_size=8, profile="smoke", device="cpu")
            manifest = root / "mean_slice_manifest.csv"
            fields = ["mean_slice_id", "view", "slice_id", "image_file", "geometry_json"]
            with manifest.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                for view, z in (("SAX", 0), ("2CH", 1), ("4CH", 2)):
                    filename = f"{view}.npy"
                    np.save(root / filename, np.full((4, 4), 0.5, dtype=np.float32))
                    geometry = _geometry(origin=(0, 0, z))
                    writer.writerow({"mean_slice_id": view, "view": view, "slice_id": view, "image_file": filename, "geometry_json": json.dumps(geometry)})
            outcome = train_stage1b_balanced(stage1a["checkpoint"], manifest, domain, root / "stage1b", epochs=1, pixels_per_view=8, device="cpu")
            self.assertEqual(outcome["visits"], {"SAX": 1, "2CH": 1, "4CH": 1})
            self.assertTrue((root / "stage1b" / "stage1b_curve.csv").is_file())
            payload = torch.load(outcome["checkpoint"], map_location="cpu", weights_only=True)
            self.assertEqual(payload["model_config"], stage1a["model_config"])
            self.assertEqual(payload["steps_per_epoch"], 1)
            self.assertEqual(payload["n_mean_slices_per_view"], {"SAX": 1, "2CH": 1, "4CH": 1})
            before = torch.load(stage1a["checkpoint"], map_location="cpu", weights_only=True)["model_state"]
            self.assertTrue(any(not torch.equal(before[name], value) for name, value in payload["model_state"].items()))
            self.assertTrue(np.isfinite(payload["curve"][0]["total_mse"]))
            loaded = _load_checkpoint(Path(outcome["checkpoint"]))
            self.assertTrue(any(torch.count_nonzero(value).item() for value in loaded.parameters()))
            source = root / "source_manifest.csv"
            source_fields = ["view", "slice_id", "image_position_patient", "image_orientation_patient", "pixel_spacing", "slice_thickness", "rows", "columns"]
            with source.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=source_fields); writer.writeheader()
                for slice_id, view, z in (("SAX", "SAX", 0), ("s17", "SAX", 1), ("2CH", "2CH", 1), ("4CH", "4CH", 2)):
                    geometry = _geometry(origin=(0, 0, z))
                    writer.writerow({"view": view, "slice_id": slice_id, **{key: json.dumps(geometry[key]) if isinstance(geometry[key], list) else geometry[key] for key in ("image_position_patient", "image_orientation_patient", "pixel_spacing", "slice_thickness", "rows", "columns")}})
            qc = run_final_static_qc(
                _load_checkpoint(Path(stage1a["checkpoint"])), loaded, manifest, source, domain, root / "final_qc",
                overview_spacing_mm=4.0, cardiac_export_spacing_mm=1.5, cardiac_roi_margin_mm=1.5,
                eval_chunk_pixels=8, stage1b_metadata=payload,
            )
            for filename in ("canonical_overview.nii.gz", "stage1a_cardiac_1p5mm.nii.gz", "stage1b_cardiac_1p5mm.nii.gz", "stage1a_vs_stage1b_cardiac_difference.nii.gz", "stage1b_per_slice_metrics.csv", "stage1b_per_view_metrics.json", "s17_stage1a_prediction.png", "s17_stage1b_prediction.png", "s17_stage1b_minus_stage1a.png", "s17_neighboring_sax_qc.png", "s17_through_plane_qc.png", "stage1_coverage_summary.json"):
                self.assertTrue((root / "final_qc" / filename).is_file(), filename)
            self.assertTrue(qc["s17"]["s17_absence_confirmed"])

    def test_balanced_training_defers_calibrated_uncertainty(self):
        with self.assertRaises(NotImplementedError):
            train_stage1b_balanced("unused", "unused", "unused", "unused", uncertainty_mode="calibrated")


if __name__ == "__main__":
    unittest.main()
