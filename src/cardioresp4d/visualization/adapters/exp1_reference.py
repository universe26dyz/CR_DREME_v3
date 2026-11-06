"""Read-only adapter for the finished Exp1 static-NeSVoR reference artifacts."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import numpy as np
import torch

from ..contract import ExperimentVisualizationAdapter, STAGES, stage_status
from ..geometry_utils import (LPS_RAS, compare_plane_corners_lps, location_token,
                              numeric_corrected_slices, ordered_representative_locations,
                              plane_corners_lps, pose_delta, stack_tiles)
from ..image_utils import (evaluate_inr_on_nifti_lattice, export_support_mask,
                           save_unmasked_static_inr, support_metrics, volume_summary)
from ..renderers import fixed_window, render_montage, render_paired_images


class Exp1ReferenceAdapter(ExperimentVisualizationAdapter):
    name = "exp1_reference"

    @property
    def scientific_status(self) -> Mapping[str, str]:
        return {"experiment_scientific_status": "INVALID_PENDING_STATIC_PIPELINE_REPAIR", "reason": "formal static NeSVoR output visually invalid"}

    def source_artifacts(self, experiment_root: Path, manifest: Path, qc_table: Path, canonical_domain: Path) -> dict[str, Path]:
        static = experiment_root / "02_static_nesvor"; mean = experiment_root / "01_temporal_mean"
        paths = {"manifest": manifest, "qc_table": qc_table, "canonical_domain": canonical_domain,
                 "mean_slice_manifest": mean / "mean_slice_manifest.json", "static_reference": static / "static_reference.nii.gz",
                 "static_model": static / "model.pt", "corrected_pose_manifest": static / "corrected_pose_manifest.json",
                 "corrected_slices_directory": static / "slices"}
        return {name: path for name, path in paths.items() if path.exists()}

    def render(self, *, experiment_root: Path, manifest: Path, qc_table: Path, canonical_domain: Path, output_dir: Path, device: str) -> dict[str, dict[str, Any]]:
        for stage in STAGES: (output_dir / stage).mkdir(parents=True, exist_ok=True)
        state: dict[str, dict[str, Any]] = {}
        workers = (
            ("V0_raw_acquisition", lambda: self._v0_raw(manifest, qc_table, output_dir / "V0_raw_acquisition")),
            ("V1_normalization", lambda: self._v1_normalization(manifest, qc_table, output_dir / "V1_normalization")),
            ("V2_temporal_mean", lambda: self._v2_temporal_mean(experiment_root, output_dir / "V2_temporal_mean")),
            ("V3_stack_geometry", lambda: self._v3_geometry(experiment_root, canonical_domain, output_dir / "V3_stack_geometry")),
            ("V4_registration", lambda: self._v4_registration(experiment_root, output_dir / "V4_registration")),
            ("V5_static_support", lambda: self._v5_support(experiment_root, output_dir / "V5_static_support", device)),
            ("V6_static_inr", lambda: self._v6_inr(experiment_root, output_dir / "V6_static_inr", device)),
            ("V7_static_reprojection", lambda: self._v7_reprojection(experiment_root, output_dir / "V7_static_reprojection", device)),
        )
        for name, worker in workers:
            try:
                state[name] = worker()
            except Exception as error:  # Individual audit stages must explain rather than silently disappear.
                state[name] = stage_status("failed", f"{type(error).__name__}: {error}")
        state["V8_dynamic_zero_motion"] = self.render_v8_dynamic_zero_motion()
        state["V9_dynamic_boundaries"] = self.render_v9_dynamic_boundaries()
        return state

    @staticmethod
    def _dataset(manifest: Path, qc_table: Path):
        from cardioresp4d.data.dataset import CardioRespDataset
        return CardioRespDataset(manifest, valid_only=False, qc_table_path=qc_table, normalization_mode="per_series")

    @staticmethod
    def _ordered_rows(dataset) -> dict[str, list[tuple[dict[str, str], int]]]:
        groups: dict[tuple[str, str], list[tuple[dict[str, str], int]]] = defaultdict(list)
        for index, row in enumerate(dataset._rows):
            if row["qc_valid"] not in ("0", "false", "False"):
                groups[(row["view"].upper(), row["slice_id"])].append((row, index))
        by_view: dict[str, list[tuple[dict[str, str], int]]] = defaultdict(list)
        for (view, _), rows in groups.items():
            first = rows[0][0]; orientation = np.asarray(json.loads(first["image_orientation_patient"]), dtype=float)
            position = np.asarray(json.loads(first["image_position_patient"]), dtype=float); normal = np.cross(orientation[:3], orientation[3:])
            by_view[view].append((first | {"_projection": float(np.dot(position, normal))}, rows[0][1]))
        return {view: sorted(rows, key=lambda item: item[0]["_projection"]) for view, rows in by_view.items()}

    @staticmethod
    def _frame_indices(rows: list[tuple[dict[str, str], int]]) -> list[int]:
        return sorted(set(np.linspace(0, len(rows) - 1, min(5, len(rows))).round().astype(int).tolist()))

    def _v0_raw(self, manifest: Path, qc_table: Path, output: Path) -> dict[str, Any]:
        dataset = self._dataset(manifest, qc_table); selected = self._ordered_rows(dataset); rendered = 0
        for view, locations in selected.items():
            panels: list[np.ndarray] = []; labels: list[str] = []
            for row, first_index in [locations[index] for index in sorted(set((0, len(locations) // 2, len(locations) - 1)))]:
                frames = [(candidate, idx) for candidate, idx in enumerate(dataset._rows) if candidate["view"].upper() == view and candidate["slice_id"] == row["slice_id"] and candidate["qc_valid"] not in ("0", "false", "False")]
                for local_index in self._frame_indices(frames):
                    sample = dataset[frames[local_index][1]]; image = np.asarray(sample["rescaled_image"], dtype=np.float32); stats = np.percentile(image, (50, 95, 99))
                    relative_time = local_index / max(1, len(frames) - 1)
                    panels.append(image); labels.append(f"{view} {location_token(view, row['slice_id'])} frame {local_index} t={relative_time:.0%}\nmin/P50/P95/P99/max={image.min():.3g}/{stats[0]:.3g}/{stats[1]:.3g}/{stats[2]:.3g}/{image.max():.3g}")
            if panels:
                render_montage(output / f"{view}_representative_frames.png", panels, labels, columns=5); rendered += len(panels)
            view_indices = [index for index, row in enumerate(dataset._rows) if row["view"].upper() == view and row["qc_valid"] not in ("0", "false", "False")]
            if view_indices:
                contact = sorted(set(np.linspace(0, len(view_indices) - 1, min(20, len(view_indices))).round().astype(int).tolist()))
                contact_samples = [dataset[view_indices[index]] for index in contact]
                render_montage(output / f"{view}_temporal_contact_sheet.png", [np.asarray(sample["rescaled_image"]) for sample in contact_samples], [f"{view} {location_token(view, sample['slice_id'])}" for sample in contact_samples], columns=5)
        return stage_status("rendered", frame_panels=rendered, display_window_policy="per_image_robust")

    def _v1_normalization(self, manifest: Path, qc_table: Path, output: Path) -> dict[str, Any]:
        dataset = self._dataset(manifest, qc_table); rows: list[dict[str, Any]] = []; images_before: dict[str, list[np.ndarray]] = defaultdict(list); images_after: dict[str, list[np.ndarray]] = defaultdict(list)
        grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, row in enumerate(dataset._rows):
            if row["qc_valid"] not in ("0", "false", "False"): grouped[(row["view"].upper(), row["slice_id"])].append(index)
        representative = {(view, row["slice_id"]) for view, locations in self._ordered_rows(dataset).items() for row, _ in [locations[index] for index in sorted(set((0, len(locations) // 2, len(locations) - 1))) ]}
        for (view, slice_id), indices in sorted(grouped.items()):
            sample = dataset[indices[0]]; raw, normalized = np.asarray(sample["rescaled_image"]), np.asarray(sample["image"])
            key = dataset._normalization_key(dataset._rows[indices[0]]); params = dataset.normalization_parameters[key]
            rows.append({"view": view, "opaque_slice_id": location_token(view, slice_id), "P1": params["lower"], "P99": params["upper"], "P99_minus_P1": params["upper"] - params["lower"], "low_clip_fraction": float(np.mean(raw <= params["lower"])), "high_clip_fraction": float(np.mean(raw >= params["upper"]))})
            images_before[view].append(raw.ravel()); images_after[view].append(normalized.ravel())
            if (view, slice_id) in representative:
                for frame_index in self._frame_indices([(dataset._rows[index], index) for index in indices]):
                    frame = dataset[indices[frame_index]]; raw_frame, normal_frame = np.asarray(frame["rescaled_image"]), np.asarray(frame["image"])
                    render_paired_images(output / f"{location_token(view, slice_id)}_frame_{frame_index:03d}.png", [raw_frame, normal_frame], ["modality-rescaled", "per-series normalized"], window=(min(float(raw_frame.min()), 0.), max(float(raw_frame.max()), 1.)))
        with (output / "normalization_parameters.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["view"]); writer.writeheader(); writer.writerows(rows)
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(8, 4)); axis.scatter(range(len(rows)), [row["P99_minus_P1"] for row in rows]); axis.set_xlabel("physical location order within grouped views"); axis.set_ylabel("P99-P1"); figure.tight_layout(); figure.savefig(output / "location_scale.png", dpi=140); plt.close(figure)
        return stage_status("rendered", location_count=len(rows), display_window_policy="paired_fixed_per_location")

    def _mean(self, experiment_root: Path) -> tuple[dict[str, Any], Path]:
        directory = experiment_root / "01_temporal_mean"; return json.loads((directory / "mean_slice_manifest.json").read_text()), directory

    def _v2_temporal_mean(self, experiment_root: Path, output: Path) -> dict[str, Any]:
        manifest, directory = self._mean(experiment_root); table: list[dict[str, Any]] = []
        for view in ("SAX", "2CH", "4CH"):
            records = [record for record in manifest["locations"] if record["view"].upper() == view]
            if not records: continue
            image = nib.load(str(directory / records[0]["stack_file"])); data = np.asanyarray(image.dataobj)
            lattice = manifest.get("stack_lattices", {}).get(view, {}); tiles = stack_tiles(records, lattice_size=int(lattice.get("lattice_size", data.shape[2])))
            pages = [tiles[start:start + 12] for start in range(0, len(tiles), 12)]
            for page_index, page in enumerate(pages):
                panels = [data[..., int(tile["stack_index"])].T if tile["real"] else np.zeros(data.shape[:2][::-1]) for tile in page]
                render_montage(output / f"{view}_mean_page_{page_index:02d}.png", panels, [tile["label"] for tile in page], columns=4)
            for tile in tiles:
                if tile["real"]:
                    values = data[..., int(tile["stack_index"])]
                    table.append({"view": view, "opaque_slice_id": location_token(view, str(tile["slice_id"])), "lattice_index": tile["stack_index"], **{key: value for key, value in volume_summary(values).items() if key in {"min", "p1", "p50", "p95", "p99", "max", "zero_fraction"}}})
        with (output / "temporal_mean_intensity.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]) if table else ["view"]); writer.writeheader(); writer.writerows(table)
        return stage_status("rendered", location_count=len(table), rescaled_signal="not_rendered")

    def _v3_geometry(self, experiment_root: Path, canonical_domain: Path, output: Path) -> dict[str, Any]:
        manifest, directory = self._mean(experiment_root); geometry: dict[str, Any] = {}; errors: list[float] = []
        for stack in sorted({record["stack_file"] for record in manifest["locations"]}):
            image = nib.load(str(directory / stack)); header = image.header; records = [record for record in manifest["locations"] if record["stack_file"] == stack]
            rows = []
            for record in records:
                result = compare_plane_corners_lps(plane_corners_lps(record["geometry"]), image.affine, image.shape, int(record["stack_index"])); errors.append(result["max_error_mm"]); rows.append({"opaque_slice_id": location_token(record["view"], record["slice_id"]), "stack_index": record["stack_index"], **result})
            geometry[stack] = {"shape": list(image.shape), "voxel_sizes": list(header.get_zooms()), "qform": image.get_qform().tolist(), "sform": image.get_sform().tolist(), "affine": image.affine.tolist(), "orientation_codes": list(nib.aff2axcodes(image.affine)), "affine_determinant": float(np.linalg.det(image.affine[:3, :3])), "corner_round_trip": rows}
        (output / "stack_geometry.json").write_text(json.dumps(geometry, indent=2) + "\n")
        import matplotlib.pyplot as plt
        figure = plt.figure(figsize=(8, 7)); axis = figure.add_subplot(projection="3d")
        all_corners = []
        for record in manifest["locations"]:
            corners = plane_corners_lps(record["geometry"]); loop = np.vstack((corners[[0, 1, 3, 2]], corners[0]))
            axis.plot(loop[:, 0], loop[:, 1], loop[:, 2], label=record["view"] if record["view"] not in axis.get_legend_handles_labels()[1] else None)
            all_corners.append(corners)
        domain = json.loads(canonical_domain.read_text()); box = domain.get("cardiac_box", {})
        if "min_mm" in box and "max_mm" in box:
            lower, upper = np.asarray(box["min_mm"]), np.asarray(box["max_mm"]); axis.scatter(*lower, c="black", marker="x"); axis.scatter(*upper, c="black", marker="x")
        axis.set(xlabel="LPS x (mm)", ylabel="LPS y (mm)", zlabel="LPS z (mm)"); axis.legend(); figure.tight_layout(); figure.savefig(output / "stack_geometry_lps_3d.png", dpi=140); plt.close(figure)
        points = np.concatenate(all_corners, axis=0); figure, axes = plt.subplots(1, 3, figsize=(12, 4))
        for axis, pair, title in zip(axes, ((0, 1), (0, 2), (1, 2)), ("LPS XY", "LPS XZ", "LPS YZ")):
            axis.scatter(points[:, pair[0]], points[:, pair[1]], s=6); axis.set_title(title); axis.set_aspect("equal")
        figure.tight_layout(); figure.savefig(output / "stack_geometry_lps_projections.png", dpi=140); plt.close(figure)
        return stage_status("rendered", maximum_world_corner_error_mm=max(errors, default=0.), geometry_file="stack_geometry.json")

    def _v4_registration(self, experiment_root: Path, output: Path) -> dict[str, Any]:
        from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import load_corrected_pose_map
        poses = load_corrected_pose_map(experiment_root / "02_static_nesvor" / "corrected_pose_manifest.json"); rows = []
        for (view, slice_id), pose in sorted(poses.items()): rows.append({"view": view, "opaque_slice_id": location_token(view, slice_id), **pose_delta(pose.original_geometry, pose.corrected_geometry)})
        with (output / "registration_deltas.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["view"]); writer.writeheader(); writer.writerows(rows)
        corrected = numeric_corrected_slices(experiment_root / "02_static_nesvor" / "slices")
        if corrected:
            panels = [np.asanyarray(nib.load(str(path)).dataobj).squeeze().T for path in corrected[:12]]; render_montage(output / "corrected_slices.png", panels, [f"corrected {index}" for index in range(len(panels))], columns=4)
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2, figsize=(10, 4)); axes[0].hist([row["translation_mm"] for row in rows]); axes[0].set_title("translation magnitude (mm)"); axes[1].hist([row["rotation_deg"] for row in rows]); axes[1].set_title("rotation angle (degrees)"); figure.tight_layout(); figure.savefig(output / "registration_distributions.png", dpi=140); plt.close(figure)
        figure = plt.figure(figsize=(8, 7)); axis = figure.add_subplot(projection="3d")
        for pose in poses.values():
            for geometry, colour in ((pose.original_geometry, "tab:blue"), (pose.corrected_geometry, "tab:orange")):
                corners = plane_corners_lps(geometry); loop = np.vstack((corners[[0, 1, 3, 2]], corners[0])); axis.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=colour, alpha=.45)
        axis.set(xlabel="LPS x (mm)", ylabel="LPS y (mm)", zlabel="LPS z (mm)"); figure.tight_layout(); figure.savefig(output / "registration_lps_overlay.png", dpi=140); plt.close(figure)
        return stage_status("rendered", location_count=len(rows), corrected_numeric_slice_count=len(corrected))

    @staticmethod
    def _load_static_model(experiment_root: Path, device: str):
        from cardioresp4d.adapters import nesvor_inr
        payload = torch.load(experiment_root / "02_static_nesvor" / "model.pt", map_location="cpu", weights_only=False)
        state, args = payload["model"], payload["args"]; backend = nesvor_inr.detect_checkpoint_encoding_backend(state)
        with nesvor_inr._upstream_encoding_backend(backend):
            model = nesvor_inr.INR(state["bounding_box"], args); model.load_state_dict(state, strict=True)
        return payload, model.to(torch.device(device)), backend

    def _v5_support(self, experiment_root: Path, output: Path, device: str) -> dict[str, Any]:
        payload, _, _ = self._load_static_model(experiment_root, device); mask = payload["mask"]
        from nesvor.image.image_utils import transformation2affine
        affine = transformation2affine(mask.image, mask.transformation, float(mask.resolution_x), float(mask.resolution_y), float(mask.resolution_z))
        path = export_support_mask(mask.mask, affine, output / "static_support_mask.nii.gz"); data = np.asanyarray(nib.load(str(path)).dataobj)
        center = tuple(size // 2 for size in data.shape); panels = [data[center[0], :, :], data[:, center[1], :], data[:, :, center[2]]]
        render_montage(output / "support_orthogonal.png", panels, ["axial", "coronal", "sagittal"], columns=3, window=(0., 1.))
        render_montage(output / "support_projections.png", [data.max(axis=0), data.max(axis=1), data.max(axis=2), data.mean(axis=0), data.mean(axis=1), data.mean(axis=2)], ["max x", "max y", "max z", "mean x", "mean y", "mean z"], columns=3, window=(0., 1.))
        metrics = support_metrics(data, affine); (output / "support_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        return stage_status("rendered", **metrics)

    def _v6_inr(self, experiment_root: Path, output: Path, device: str) -> dict[str, Any]:
        target = nib.load(str(experiment_root / "02_static_nesvor" / "static_reference.nii.gz")); payload, model, backend = self._load_static_model(experiment_root, device)
        values = evaluate_inr_on_nifti_lattice(model, target, device=torch.device(device)); unmasked = save_unmasked_static_inr(values, target, output / "static_inr_unmasked.nii.gz")
        mask = payload["mask"]; from nesvor.image.image_utils import transformation2affine
        mask_affine = transformation2affine(mask.image, mask.transformation, float(mask.resolution_x), float(mask.resolution_y), float(mask.resolution_z))
        support = export_support_mask(mask.mask, mask_affine, output / "static_support_mask_for_comparison.nii.gz")
        from nibabel.processing import resample_from_to
        support_on_static = resample_from_to(nib.load(str(support)), target, order=0); nib.save(support_on_static, str(output / "support_on_static_lattice.nii.gz"))
        masked = np.asanyarray(target.dataobj); support_values = np.asanyarray(support_on_static.dataobj); panel = [values[..., values.shape[2] // 2].T, support_values[..., support_values.shape[2] // 2].T, masked[..., masked.shape[2] // 2].T]
        raw_window = (min(float(values.min()), float(masked.min())), max(float(values.max()), float(masked.max())))
        render_paired_images(output / "inr_support_masked_raw.png", panel, ["unmasked INR", "support mask", "masked static reference"], window=raw_window)
        render_paired_images(output / "inr_support_masked_robust.png", panel, ["unmasked INR", "support mask", "masked static reference"], window=fixed_window(np.concatenate((values.ravel(), masked.ravel())), robust=True))
        summary = {"unmasked": volume_summary(values), "masked": volume_summary(masked), "encoding_backend": backend}; (output / "intensity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return stage_status("rendered", **summary)

    def _v7_reprojection(self, experiment_root: Path, output: Path, device: str) -> dict[str, Any]:
        """Use pinned NeSVoR ``sample_slices``; no central-plane substitute is used."""
        payload, model, _ = self._load_static_model(experiment_root, device)
        from nesvor.image import load_slices
        from nesvor.inr.sample import sample_slices
        files = numeric_corrected_slices(experiment_root / "02_static_nesvor" / "slices")
        chosen = [files[index] for index in sorted(set((0, len(files) // 2, len(files) - 1)))] if files else []
        if not chosen: return stage_status("partial", "no numeric corrected slices available")
        observed = load_slices([str(path) for path in chosen], torch.device(device)); simulated = sample_slices(model, observed, payload["mask"], float(getattr(payload["args"], "output_psf_factor", 1.0)), int(getattr(payload["args"], "n_inference_samples", 128)))
        panels: list[np.ndarray] = []; labels: list[str] = []
        for index, (source, predicted) in enumerate(zip(observed, simulated)):
            raw, estimate = source.image.squeeze().detach().cpu().numpy(), predicted.image.squeeze().detach().cpu().numpy(); panels.extend((raw, estimate, np.abs(raw - estimate))); labels.extend((f"corrected observed {index}", "NeSVoR PSF reprojection", "absolute residual"))
        render_montage(output / "reprojections.png", panels, labels, columns=3)
        return stage_status("rendered", slice_count=len(chosen), simulation_path="pinned_nesvor_sample_slices")
