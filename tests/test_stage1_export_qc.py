"""Focused physical-export and final static-QC contracts for Stage 1."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cardioresp4d.models.hash_inr import CanonicalINR
from cardioresp4d.models.uncertainty import NamespacedObservationUncertainty
from cardioresp4d.training.export import (
    lps_to_ras,
    make_export_grid,
    query_physical_plane,
    ras_to_lps,
    export_volume,
)
from cardioresp4d.training.static_qc import (
    aggregate_by_view,
    calibration_summary,
    evaluate_mean_slices,
    select_representative_rows,
)


class _LinearINR(torch.nn.Module):
    def forward(self, points, return_features=False):
        image = (0.5 + 0.1 * points.sum(-1, keepdim=True)).clamp(0, 1)
        return (image, points) if return_features else image


class _OffsetINR(_LinearINR):
    def __init__(self, offset: float):
        super().__init__()
        self.offset = offset

    def forward(self, points, return_features=False):
        image = super().forward(points, return_features=False) + self.offset
        return (image, points) if return_features else image


def _domain(root: Path) -> Path:
    path = root / "domain.json"
    path.write_text(json.dumps({
        "world_min_mm": [0, 0, 0], "world_max_mm": [6, 6, 6],
        "world_to_normalized": [[1 / 3, 0, 0, -1], [0, 1 / 3, 0, -1], [0, 0, 1 / 3, -1], [0, 0, 0, 1]],
        "cardiac_box": {"center_mm": [3, 3, 3], "size_mm": [4, 4, 4]},
    }))
    return path


def _geometry(origin=(0, 0, 0), orientation=(1, 0, 0, 0, 1, 0)):
    return {"image_position_patient": list(origin), "image_orientation_patient": list(orientation),
            "pixel_spacing": [1.5, 1.5], "slice_thickness": 4.0, "rows": 4, "columns": 4}


class Stage1ExportQcTests(unittest.TestCase):
    def test_spacing_driven_grid_and_lps_ras_roundtrip(self):
        grid = make_export_grid([1, 2, 3], [4, 5, 6], 1.5)
        self.assertEqual(grid.shape, (3, 3, 3))
        np.testing.assert_allclose(grid.spacing_mm, [1.5, 1.5, 1.5])
        lps = np.array([2.5, 3.5, 4.5])
        voxel = grid.lps_to_voxel(lps)
        ras_from_affine = grid.affine_ras @ np.r_[voxel, 1.0]
        np.testing.assert_allclose(ras_to_lps(ras_from_affine[:3]), lps)
        np.testing.assert_allclose(lps_to_ras(lps), ras_from_affine[:3])

    def test_same_grid_export_and_difference_are_not_fixed_64_cubed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); domain = _domain(root); model_a, model_b = _OffsetINR(0.0), _OffsetINR(0.1)
            first = export_volume(model_a, domain, [0, 0, 0], [6, 6, 6], 1.5, root / "a.nii.gz")
            second = export_volume(model_b, domain, [0, 0, 0], [6, 6, 6], 1.5, root / "b.nii.gz", grid=first.grid)
            self.assertEqual(first.grid.shape, second.grid.shape)
            np.testing.assert_allclose(first.grid.affine_ras, second.grid.affine_ras)
            self.assertNotEqual(first.grid.shape, (64, 64, 64))
            self.assertTrue(np.isfinite(first.data).all() and np.isfinite(second.data).all())
            np.testing.assert_allclose(second.data - first.data, 0.1, atol=1e-6)

    def test_oblique_physical_plane_is_continuous_and_finite(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = query_physical_plane(_LinearINR(), _domain(Path(tmp)), _geometry(orientation=(0, 1, 0, 0, 0, 1)))
            self.assertEqual(image.shape, (4, 4))
            self.assertTrue(np.isfinite(image).all())

    def test_chunked_evaluation_matches_full_and_uses_all_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); domain = _domain(root); manifest = root / "mean_slice_manifest.csv"
            fields = ("mean_slice_id", "view", "slice_id", "image_file", "geometry_json")
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
                for view, z in (("SAX", 0), ("2CH", 2), ("4CH", 4)):
                    filename = f"{view}.npy"; np.save(root / filename, np.full((4, 4), 0.5, np.float32))
                    writer.writerow({"mean_slice_id": view, "view": view, "slice_id": view.lower(), "image_file": filename, "geometry_json": json.dumps(_geometry((0, 0, z)))})
            full = evaluate_mean_slices(_LinearINR(), manifest, domain, root / "full", chunk_pixels=64)
            chunked = evaluate_mean_slices(_LinearINR(), manifest, domain, root / "chunked", chunk_pixels=3)
            self.assertEqual([row["n_pixels"] for row in full], [16, 16, 16])
            np.testing.assert_allclose([row["MSE"] for row in full], [row["MSE"] for row in chunked])
            aggregate = aggregate_by_view(full)
            self.assertEqual(set(aggregate), {"SAX", "2CH", "4CH"})

    def test_physical_sorting_and_uncertainty_calibration_are_separate(self):
        rows = [
            {"slice_id": "z20", "geometry_json": json.dumps(_geometry((0, 0, 20)))},
            {"slice_id": "z00", "geometry_json": json.dumps(_geometry((0, 0, 0)))},
            {"slice_id": "z10", "geometry_json": json.dumps(_geometry((0, 0, 10)))},
        ]
        self.assertEqual([row["slice_id"] for row in select_representative_rows(rows)], ["z00", "z10", "z20"])
        summary = calibration_summary(
            squared_pixel_residual=np.array([1.0, 4.0, 9.0]), pixel_variance=np.array([2.0, 8.0, 18.0]),
            slice_mse=np.array([1.0, 4.0]), observation_variance=np.array([3.0, 12.0]), total_variance=np.array([5.0, 10.0, 20.0]),
        )
        self.assertGreater(summary["pixel_level"]["pearson_r"], 0.99)
        self.assertGreater(summary["observation_level"]["pearson_r"], 0.99)
        uncertainty = NamespacedObservationUncertainty(3, num_mean_slices=2, num_dynamic_frames=1, embedding_dim=2)
        uncertainty.initialize_observation_variance("mean_slice", torch.tensor([0, 1]), torch.tensor([0.2, 0.7]))
        values = torch.nn.functional.softplus(uncertainty.log_variances["mean_slice"].weight.squeeze()) + uncertainty.epsilon
        torch.testing.assert_close(values, torch.tensor([0.2, 0.7]), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
