"""Read-only adapter for the finished Exp1 static-NeSVoR reference artifacts."""
from __future__ import annotations

import csv
import hashlib
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
from ..image_utils import (aggregate_clip_fractions, evaluate_inr_on_nifti_lattice,
                           evaluate_inr_psf_plane, export_support_mask,
                           representative_plane_specs, save_unmasked_static_inr,
                           nesvor_volume_to_device, support_metrics,
                           support_point_agreement, validate_nesvor_device_contract,
                           volume_summary)
from ..report import json_safe
from ..renderers import fixed_window, render_independent_images, render_montage, render_paired_images
from ..dynamic_audit import (CheckpointBoundary, discover_boundary_checkpoints, resolve_dynamic_checkpoint_root,
                             fixed_physical_grid, mbc_points_from_world_grid, weighted_dvf_from_precomputed_basis,
                             posthoc_center_scores, predict_observation_slice,
                             reconstruct_exp1_checkpoint,
                             select_fixed_location_frames, static_dynamic_rms)


def _render_vector_field_planes(directory: Path, prefix: str, field: torch.Tensor, *, label: str, shape: tuple[int, int, int] = (16, 16, 16)) -> None:
    """Render central-z magnitude and signed component planes for [N,3] fields."""
    vector = field.detach().cpu().reshape(*shape, 3)
    plane = vector[:, :, shape[2] // 2, :]
    magnitude = np.linalg.norm(plane.numpy(), axis=-1)
    render_independent_images(directory / f"{prefix}_magnitude.png", [magnitude], [f"{label} — |field|"], [(0., max(float(magnitude.max()), 1e-6))])
    bound = max(float(np.abs(plane.numpy()).max()), 1e-6)
    for index, component in enumerate(("dx", "dy", "dz")):
        render_independent_images(directory / f"{prefix}_{component}.png", [plane.numpy()[..., index]], [f"{label} — {component}"], [(-bound, bound)])


class Exp1ReferenceAdapter(ExperimentVisualizationAdapter):
    name = "exp1_reference"

    @property
    def scientific_status(self) -> Mapping[str, str]:
        return {"experiment_scientific_status": "STATIC_REFERENCE_VERIFIED_DYNAMIC_IDENTIFIABILITY_FAILURE_OBSERVED", "reason": "Stage 0/static NeSVoR was verified by geometry, support, center/PSF INR agreement and reprojection; the active scientific issue is temporal motion identifiability/collapse in Exp1."}

    def source_artifacts(self, experiment_root: Path, manifest: Path, qc_table: Path, canonical_domain: Path) -> dict[str, Path]:
        static = experiment_root / "02_static_nesvor"; mean = experiment_root / "01_temporal_mean"
        paths = {"manifest": manifest, "qc_table": qc_table, "canonical_domain": canonical_domain,
                 "mean_slice_manifest": mean / "mean_slice_manifest.json", "mean_stack_SAX": mean / "mean_stack_SAX.nii.gz", "mean_stack_2CH": mean / "mean_stack_2CH.nii.gz", "mean_stack_4CH": mean / "mean_stack_4CH.nii.gz",
                 "static_reference_manifest": static / "static_reference_manifest.json", "nesvor_static_command": static / "nesvor_static_command.json", "nesvor_effective_args": static / "nesvor_effective_args.json", "static_reference": static / "static_reference.nii.gz",
                 "static_model": static / "model.pt", "corrected_pose_manifest": static / "corrected_pose_manifest.json",
                 "corrected_slices_directory": static / "slices"}
        training = resolve_dynamic_checkpoint_root(experiment_root)
        if training is not None:
            for name in ("checkpoint_L1_warm_end.pt", "checkpoint_L1_joint_end.pt", "checkpoint_L2_warm_end.pt", "checkpoint_L2_joint_end.pt", "checkpoint_L3_warm_end.pt", "checkpoint_L3_joint_end.pt", "checkpoint_last.pt", "boundary_diagnostics.json", "training_report.json"):
                paths[f"training_{name}"] = training / name
        return {name: path for name, path in paths.items() if path.exists()}

    def render(self, *, experiment_root: Path, manifest: Path, qc_table: Path, canonical_domain: Path, output_dir: Path, device: str, stages: tuple[str, ...] | None = None) -> dict[str, dict[str, Any]]:
        requested = set(STAGES if stages is None else stages)
        unknown = requested - set(STAGES)
        if unknown:
            raise ValueError(f"unknown visualization stage(s): {', '.join(sorted(unknown))}")
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
            if name not in requested:
                state[name] = stage_status("skipped", "stage not requested")
                continue
            try:
                state[name] = worker()
            except Exception as error:  # Individual audit stages must explain rather than silently disappear.
                state[name] = stage_status("failed", f"{type(error).__name__}: {error}")
        for name, worker in (("V8_dynamic_zero_motion", self.render_v8_dynamic_zero_motion), ("V9_dynamic_boundaries", self.render_v9_dynamic_boundaries)):
            if name not in requested:
                state[name] = stage_status("skipped", "stage not requested")
                continue
            try:
                state[name] = worker(experiment_root=experiment_root, manifest=manifest, qc_table=qc_table, canonical_domain=canonical_domain, output=output_dir / name, device=device)
            except Exception as error:
                state[name] = stage_status("failed", f"{type(error).__name__}: {error}")
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

    @staticmethod
    def _series_frames(dataset, view: str, slice_id: str) -> list[tuple[dict[str, str], int]]:
        frames = [(row, index) for index, row in enumerate(dataset._rows) if row["view"].upper() == view and row["slice_id"] == slice_id and row["qc_valid"] not in ("0", "false", "False")]
        return sorted(frames, key=lambda item: float(item[0].get("timestamp_s", item[1])))

    @staticmethod
    def _bounded_sample(arrays: list[np.ndarray], *, per_array: int = 8192, total: int = 100000) -> np.ndarray:
        sampled = [array[np.linspace(0, array.size - 1, min(array.size, per_array), dtype=int)] for array in (np.asarray(value).reshape(-1) for value in arrays) if array.size]
        if not sampled:
            return np.empty(0, dtype=np.float32)
        values = np.concatenate(sampled)
        return values[np.linspace(0, values.size - 1, min(values.size, total), dtype=int)]

    def _v0_raw(self, manifest: Path, qc_table: Path, output: Path) -> dict[str, Any]:
        dataset = self._dataset(manifest, qc_table); selected = self._ordered_rows(dataset); rendered = 0
        figures: dict[str, Any] = {}
        for view, locations in selected.items():
            panels: list[np.ndarray] = []; labels: list[str] = []
            for row, first_index in [locations[index] for index in sorted(set((0, len(locations) // 2, len(locations) - 1)))]:
                frames = self._series_frames(dataset, view, row["slice_id"])
                first_time = float(frames[0][0].get("timestamp_s", dataset[frames[0][1]].get("timestamp_s", 0.)))
                for local_index in self._frame_indices(frames):
                    sample = dataset[frames[local_index][1]]; image = np.asarray(sample["rescaled_image"], dtype=np.float32); stats = np.percentile(image, (50, 95, 99))
                    frame_row = frames[local_index][0]; frame_number = frame_row.get("frame_index", local_index); relative_time = float(sample.get("timestamp_s", frame_row.get("timestamp_s", first_time))) - first_time
                    panels.append(image); labels.append(f"{view} {location_token(view, row['slice_id'])} frame {frame_number} Δt={relative_time:.3g}s\nmin/P50/P95/P99/max={image.min():.3g}/{stats[0]:.3g}/{stats[1]:.3g}/{stats[2]:.3g}/{image.max():.3g}")
            if panels:
                name = f"{view}_representative_frames.png"; figures[name] = {"policy": render_montage(output / name, panels, labels, columns=5)}; rendered += len(panels)
            if locations:
                middle_row, _ = locations[len(locations) // 2]; frames = self._series_frames(dataset, view, middle_row["slice_id"]); first_time = float(frames[0][0].get("timestamp_s", dataset[frames[0][1]].get("timestamp_s", 0.))); contact = self._frame_indices(frames); contact_samples = [dataset[frames[index][1]] for index in contact]
                labels = [f"{view} {location_token(view, middle_row['slice_id'])} frame {frames[index][0].get('frame_index', index)} Δt={float(contact_samples[offset].get('timestamp_s', frames[index][0].get('timestamp_s', first_time))) - first_time:.3g}s" for offset, index in enumerate(contact)]
                name = f"{view}_temporal_contact_sheet.png"; policy = render_montage(output / name, [np.asarray(sample["rescaled_image"]) for sample in contact_samples], labels, columns=5); figures[name] = {"policy": policy, "location": location_token(view, middle_row["slice_id"])}
        return stage_status("rendered", frame_panels=rendered, figures=figures)

    def _v1_normalization(self, manifest: Path, qc_table: Path, output: Path) -> dict[str, Any]:
        dataset = self._dataset(manifest, qc_table); rows: list[dict[str, Any]] = []; images_before: dict[str, list[np.ndarray]] = defaultdict(list); images_after: dict[str, list[np.ndarray]] = defaultdict(list); figures: dict[str, Any] = {}
        grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, row in enumerate(dataset._rows):
            if row["qc_valid"] not in ("0", "false", "False"): grouped[(row["view"].upper(), row["slice_id"])].append(index)
        representative = {(view, row["slice_id"]) for view, locations in self._ordered_rows(dataset).items() for row, _ in [locations[index] for index in sorted(set((0, len(locations) // 2, len(locations) - 1))) ]}
        for (view, slice_id), indices in sorted(grouped.items()):
            key = dataset._normalization_key(dataset._rows[indices[0]]); params = dataset.normalization_parameters[key]
            clip = aggregate_clip_fractions((np.asarray(dataset[index]["rescaled_image"]) for index in indices), lower=params["lower"], upper=params["upper"])
            rows.append({"view": view, "opaque_slice_id": location_token(view, slice_id), "P1": params["lower"], "P99": params["upper"], "P99_minus_P1": params["upper"] - params["lower"], **clip})
            images_before[view].extend(self._bounded_sample([np.asarray(dataset[index]["rescaled_image"])]) for index in indices); images_after[view].extend(self._bounded_sample([np.asarray(dataset[index]["image"])]) for index in indices)
            if (view, slice_id) in representative:
                for frame_index in self._frame_indices([(dataset._rows[index], index) for index in indices]):
                    frame = dataset[indices[frame_index]]; raw_frame, normal_frame = np.asarray(frame["rescaled_image"]), np.asarray(frame["image"]); raw_window = fixed_window(raw_frame, robust=True); name = f"{location_token(view, slice_id)}_frame_{frame_index:03d}.png"
                    figures[name] = render_independent_images(output / name, [raw_frame, normal_frame], ["modality-rescaled", "per-series normalized"], [raw_window, (0., 1.)])
        with (output / "normalization_parameters.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["view"]); writer.writeheader(); writer.writerows(rows)
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, max(1, len(images_before)), figsize=(5 * max(1, len(images_before)), 4), squeeze=False)
        for axis, (view, values) in zip(axes.ravel(), sorted(images_before.items())):
            row_by_token = {row["opaque_slice_id"]: row for row in rows}; ordered_tokens = [location_token(view, location[0]["slice_id"]) for location in self._ordered_rows(dataset).get(view, [])]; view_rows = [row_by_token[token] for token in ordered_tokens if token in row_by_token]; axis.plot(range(len(view_rows)), [row["P99_minus_P1"] for row in view_rows], marker="o"); axis.set_title(view); axis.set_xlabel("physical location order"); axis.set_ylabel("P99-P1")
            raw_sample = self._bounded_sample(values); normalized_sample = self._bounded_sample(images_after[view]); hist_name = f"{view}_intensity_histograms.png"; hist_figure, hist_axes = plt.subplots(1, 2, figsize=(9, 4)); hist_axes[0].hist(raw_sample, bins=64); hist_axes[0].set_title(f"{view} modality-rescaled"); hist_axes[1].hist(normalized_sample, bins=64, range=(0., 1.)); hist_axes[1].set_title(f"{view} normalized"); hist_figure.tight_layout(); hist_figure.savefig(output / hist_name, dpi=140); plt.close(hist_figure); figures[hist_name] = {"raw": "per-view deterministic bounded sample", "normalized": "per-view deterministic bounded sample"}
        figure.tight_layout(); figure.savefig(output / "location_scale.png", dpi=140); plt.close(figure); figures["location_scale.png"] = {"policy": "physical_order_by_view"}
        return stage_status("rendered", location_count=len(rows), figures=figures)

    def _mean(self, experiment_root: Path) -> tuple[dict[str, Any], Path]:
        directory = experiment_root / "01_temporal_mean"; return json.loads((directory / "mean_slice_manifest.json").read_text()), directory

    def _v2_temporal_mean(self, experiment_root: Path, output: Path) -> dict[str, Any]:
        manifest, directory = self._mean(experiment_root); table: list[dict[str, Any]] = []; figures: dict[str, Any] = {}
        for view in ("SAX", "2CH", "4CH"):
            records = [record for record in manifest["locations"] if record["view"].upper() == view]
            if not records: continue
            image = nib.load(str(directory / records[0]["stack_file"])); data = np.asanyarray(image.dataobj)
            lattice = manifest.get("stack_lattices", {}).get(view, {}); tiles = stack_tiles(records, lattice_size=int(lattice.get("lattice_size", data.shape[2])))
            pages = [tiles[start:start + 12] for start in range(0, len(tiles), 12)]
            for page_index, page in enumerate(pages):
                panels = [data[..., int(tile["stack_index"])].T if tile["real"] else np.zeros(data.shape[:2][::-1]) for tile in page]
                name = f"{view}_mean_page_{page_index:02d}.png"; figures[name] = {"policy": render_montage(output / name, panels, [tile["label"] for tile in page], columns=4)}
            for tile in tiles:
                if tile["real"]:
                    values = data[..., int(tile["stack_index"])]
                    table.append({"view": view, "opaque_slice_id": location_token(view, str(tile["slice_id"])), "lattice_index": tile["stack_index"], **{key: value for key, value in volume_summary(values).items() if key in {"min", "p1", "p50", "p95", "p99", "max", "zero_fraction"}}})
        with (output / "temporal_mean_intensity.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]) if table else ["view"]); writer.writeheader(); writer.writerows(table)
        return stage_status("rendered", location_count=len(table), rescaled_signal="not_rendered", figures=figures)

    def _v3_geometry(self, experiment_root: Path, canonical_domain: Path, output: Path) -> dict[str, Any]:
        manifest, directory = self._mean(experiment_root); geometry: dict[str, Any] = {}; errors: list[tuple[float, str, str]] = []; tolerance_mm = 1e-3
        for stack in sorted({record["stack_file"] for record in manifest["locations"]}):
            image = nib.load(str(directory / stack)); header = image.header; records = [record for record in manifest["locations"] if record["stack_file"] == stack]
            rows = []
            for record in records:
                result = compare_plane_corners_lps(plane_corners_lps(record["geometry"]), image.affine, image.shape, int(record["stack_index"])); errors.append((result["max_error_mm"], location_token(record["view"], record["slice_id"]), stack)); rows.append({"opaque_slice_id": location_token(record["view"], record["slice_id"]), "stack_index": record["stack_index"], **result})
            geometry[stack] = {"shape": list(image.shape), "voxel_sizes": [float(value) for value in header.get_zooms()], "qform": image.get_qform().tolist(), "sform": image.get_sform().tolist(), "affine": image.affine.tolist(), "orientation_codes": list(nib.aff2axcodes(image.affine)), "affine_determinant": float(np.linalg.det(image.affine[:3, :3])), "corner_round_trip": rows}
        (output / "stack_geometry.json").write_text(json.dumps(json_safe(geometry), indent=2) + "\n")
        import matplotlib.pyplot as plt
        figure = plt.figure(figsize=(8, 7)); axis = figure.add_subplot(projection="3d")
        all_corners = []
        colours = {"SAX": "tab:blue", "2CH": "tab:orange", "4CH": "tab:green"}
        for record in manifest["locations"]:
            corners = plane_corners_lps(record["geometry"]); loop = np.vstack((corners[[0, 1, 3, 2]], corners[0]))
            view = record["view"].upper(); axis.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=colours.get(view, "black"), label=view if view not in axis.get_legend_handles_labels()[1] else None)
            all_corners.append(corners)
        domain = json.loads(canonical_domain.read_text()); box = domain.get("cardiac_box", {})
        if "min_mm" in box and "max_mm" in box:
            lower, upper = np.asarray(box["min_mm"]), np.asarray(box["max_mm"]); corners = np.asarray([[x, y, z] for x in (lower[0], upper[0]) for y in (lower[1], upper[1]) for z in (lower[2], upper[2])]); edges = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)); [axis.plot(corners[list(edge), 0], corners[list(edge), 1], corners[list(edge), 2], color="black", alpha=.5) for edge in edges]
        axis.set(xlabel="LPS x (mm)", ylabel="LPS y (mm)", zlabel="LPS z (mm)"); axis.legend(); figure.tight_layout(); figure.savefig(output / "stack_geometry_lps_3d.png", dpi=140); plt.close(figure)
        points = np.concatenate(all_corners, axis=0); figure, axes = plt.subplots(1, 3, figsize=(12, 4))
        for axis, pair, title in zip(axes, ((0, 1), (0, 2), (1, 2)), ("LPS XY", "LPS XZ", "LPS YZ")):
            axis.scatter(points[:, pair[0]], points[:, pair[1]], s=6); axis.set_title(title); axis.set_aspect("equal")
        figure.tight_layout(); figure.savefig(output / "stack_geometry_lps_projections.png", dpi=140); plt.close(figure)
        worst = max(errors, default=(0., None, None)); return stage_status("rendered", maximum_world_corner_error_mm=worst[0], worst_opaque_location=worst[1], worst_stack=worst[2], numerical_tolerance_mm=tolerance_mm, geometry_warning="world corner error exceeds tolerance" if worst[0] > tolerance_mm else None, geometry_file="stack_geometry.json")

    def _v4_registration(self, experiment_root: Path, output: Path) -> dict[str, Any]:
        from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import load_corrected_pose_map
        poses = load_corrected_pose_map(experiment_root / "02_static_nesvor" / "corrected_pose_manifest.json"); rows = []
        for (view, slice_id), pose in sorted(poses.items()): rows.append({"view": view, "opaque_slice_id": location_token(view, slice_id), **pose_delta(pose.original_geometry, pose.corrected_geometry)})
        with (output / "registration_deltas.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["view"]); writer.writeheader(); writer.writerows(rows)
        corrected = numeric_corrected_slices(experiment_root / "02_static_nesvor" / "slices")
        if corrected:
            pages = [corrected[start:start + 12] for start in range(0, len(corrected), 12)]
            for page_index, page in enumerate(pages):
                labels = []
                panels = []
                for path in page:
                    numeric_index = int(path.name.split(".", 1)[0]); match = next((pose for pose in poses.values() if Path(pose.corrected_slice_file).name == path.name), None)
                    labels.append(f"{match.view if match else 'unknown'} {location_token(match.view, match.slice_id) if match else 'unknown'}\nphysical {match.source_stack_index if match else numeric_index} / NIfTI {numeric_index}"); panels.append(np.asanyarray(nib.load(str(path)).dataobj).squeeze().T)
                render_montage(output / f"corrected_slices_page_{page_index:03d}.png", panels, labels, columns=4)
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2, figsize=(10, 4)); axes[0].hist([row["translation_mm"] for row in rows]); axes[0].set_title("translation magnitude (mm)"); axes[1].hist([row["rotation_deg"] for row in rows]); axes[1].set_title("rotation angle (degrees)"); figure.tight_layout(); figure.savefig(output / "registration_distributions.png", dpi=140); plt.close(figure)
        def overlay(path: Path, selected_poses: Mapping[tuple[str, str], Any]) -> None:
            figure = plt.figure(figsize=(8, 7)); axis = figure.add_subplot(projection="3d"); labels = {"original": "tab:blue", "corrected": "tab:orange"}
            for pose in selected_poses.values():
                for label, geometry in (("original", pose.original_geometry), ("corrected", pose.corrected_geometry)):
                    corners = plane_corners_lps(geometry); loop = np.vstack((corners[[0, 1, 3, 2]], corners[0])); axis.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=labels[label], alpha=.45, label=label if label not in axis.get_legend_handles_labels()[1] else None)
            axis.set(xlabel="LPS x (mm)", ylabel="LPS y (mm)", zlabel="LPS z (mm)"); axis.legend(); figure.tight_layout(); figure.savefig(path, dpi=140); plt.close(figure)
        overlay(output / "registration_lps_overlay.png", poses)
        for view in ("SAX", "2CH", "4CH"):
            overlay(output / f"registration_{view}_overlay.png", {key: value for key, value in poses.items() if key[0].upper() == view})
        largest_translation = max(rows, key=lambda row: row["translation_mm"], default=None); largest_rotation = max(rows, key=lambda row: row["rotation_deg"], default=None); largest_normal = max(rows, key=lambda row: row["normal_angle_deg"], default=None)
        return stage_status("rendered", location_count=len(rows), corrected_numeric_slice_count=len(corrected), largest_translation=largest_translation, largest_rotation=largest_rotation, largest_normal_angle=largest_normal)

    @staticmethod
    def _representative_pose_keys(poses: Mapping[tuple[str, str], Any]) -> list[tuple[str, str]]:
        selected: list[tuple[str, str]] = []
        for view in ("SAX", "2CH", "4CH"):
            keys = sorted((key for key in poses if key[0].upper() == view), key=lambda key: (str(getattr(poses[key], "source_stack_file", "")), int(getattr(poses[key], "source_stack_index", 0)), str(key[1])))
            selected.extend(keys[index] for index in sorted(set((0, len(keys) // 2, len(keys) - 1))) if keys)
        return selected

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
        center = tuple(size // 2 for size in data.shape); planes = {"sagittal": {"voxel_index": [center[0], None, None], "image": data[center[0], :, :].T}, "coronal": {"voxel_index": [None, center[1], None], "image": data[:, center[1], :].T}, "axial": {"voxel_index": [None, None, center[2]], "image": data[:, :, center[2]].T}}
        for value in planes.values():
            indices = [index if index is not None else 0 for index in value["voxel_index"]]; value["world_mm_ras"] = nib.affines.apply_affine(affine, indices).tolist()
        (output / "orthogonal_planes.json").write_text(json.dumps({key: {"voxel_index": value["voxel_index"], "world_mm_ras": value["world_mm_ras"]} for key, value in planes.items()}, indent=2) + "\n")
        render_montage(output / "support_orthogonal.png", [planes[name]["image"] for name in ("sagittal", "coronal", "axial")], [f"{name} ({planes[name]['world_mm_ras']})" for name in ("sagittal", "coronal", "axial")], columns=3, window=(0., 1.))
        render_montage(output / "support_projections.png", [data.max(axis=0), data.max(axis=1), data.max(axis=2), data.mean(axis=0), data.mean(axis=1), data.mean(axis=2)], ["max x", "max y", "max z", "mean x", "mean y", "mean z"], columns=3, window=(0., 1.))
        metrics = support_metrics(data, affine); (output / "support_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        return stage_status("rendered", figures={"support_orthogonal.png": {"window": (0., 1.), "axis_convention": "sagittal=X fixed, coronal=Y fixed, axial=Z fixed"}, "support_projections.png": {"window": (0., 1.)}}, **metrics)

    @staticmethod
    def _dynamic_context(experiment_root: Path, manifest: Path, qc_table: Path, canonical_domain: Path, device: str):
        from scripts.train_source_first import observations_from_manifest
        from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import apply_corrected_static_poses, load_corrected_pose_map
        from cardioresp4d.training.build_model import uses_s2v_style_film
        domain = json.loads(canonical_domain.read_text()); static = experiment_root / "02_static_nesvor"; pose_map = load_corrected_pose_map(static / "corrected_pose_manifest.json")
        checkpoint_paths = discover_boundary_checkpoints(experiment_root); observations_by_config = {}
        for boundary in checkpoint_paths:
            checkpoint = torch.load(boundary.path, map_location="cpu", weights_only=False); config = checkpoint.get("effective_config", {})
            key = (str(config.get("training", {}).get("normalization", {}).get("mode", "per_series")), uses_s2v_style_film(config))
            if key not in observations_by_config:
                observations, _, _ = observations_from_manifest(manifest, qc_table, torch.device(device), normalization_mode=key[0], assign_slice_positions=key[1]); observations_by_config[key] = apply_corrected_static_poses(observations, pose_map)
        return domain, checkpoint_paths, observations_by_config

    def render_v8_dynamic_zero_motion(self, *, experiment_root: Path, manifest: Path, qc_table: Path, canonical_domain: Path, output: Path, device: str) -> dict[str, Any]:
        output.mkdir(parents=True, exist_ok=True); domain, boundaries, observations_by_config = self._dynamic_context(experiment_root, manifest, qc_table, canonical_domain, device)
        if not boundaries:
            return stage_status("partial", "no Exp1 boundary checkpoints found")
        summary: dict[str, Any] = {"checkpoints": [], "metrics": [], "checkpoint_root": str(boundaries[0].path.parent), "policy": "stage1 canonical-only zero-motion prediction", "metric_cohort": "representative fixed physical middle location per view; early/middle/late QC-valid frames"}
        for boundary in boundaries:
            checkpoint = torch.load(boundary.path, map_location="cpu", weights_only=False); config = checkpoint["effective_config"]; key = (str(config.get("training", {}).get("normalization", {}).get("mode", "per_series")), bool(config.get("model", {}).get("film", {}).get("implementation") in {"s2v_style_single_frame", "s2vdreme_reference_8layer"})); observations = observations_by_config[key]; model = reconstruct_exp1_checkpoint(checkpoint, domain, observations, torch.device(device)); selected = select_fixed_location_frames(observations, frames_per_location=3); checkpoint_dir = output / f"checkpoint_{boundary.label}"; checkpoint_dir.mkdir(parents=True, exist_ok=True); rows = []; figures = {}
            for (view, slice_id), frames in selected.items():
                for frame in frames:
                    acquired = frame.image[0].detach().cpu(); prediction = predict_observation_slice(model, frame, "stage1", seed=20260823); residual = (acquired - prediction.cpu()).abs(); intensity_window = fixed_window(np.concatenate((acquired.numpy().ravel(), prediction.cpu().numpy().ravel())), robust=True); residual_window = (0., max(float(residual.max()), 1e-6)); name = f"{view}_{location_token(view, slice_id)}_frame_{frame.dynamic_frame_id:05d}.png"; figures[name] = render_independent_images(checkpoint_dir / name, [acquired.numpy(), prediction.cpu().numpy(), residual.numpy()], ["observed dynamic frame", "canonical-only zero-motion prediction", "absolute residual"], [intensity_window, intensity_window, residual_window]); rows.append({"view": view, "slice_id": slice_id, "dynamic_frame_id": int(frame.dynamic_frame_id), "timestamp_s": float(frame.timestamp_s), "mse": float(residual.square().mean())})
            by_view = {view: float(np.mean([row["mse"] for row in rows if row["view"] == view])) for view in ("SAX", "2CH", "4CH") if any(row["view"] == view for row in rows)}; summary["checkpoints"].append(boundary.label); summary["metrics"].append({"checkpoint": boundary.label, "stage": boundary.stage, "representative_reprojection_mse_by_view": by_view, "representative_reprojection_mse": float(np.mean([row["mse"] for row in rows])), "metric_cohort": summary["metric_cohort"], "frames": rows, "figures": figures})
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n"); return stage_status("rendered", checkpoint_count=len(boundaries), summary_file="summary.json", policy=summary["policy"])

    def render_v9_dynamic_boundaries(self, *, experiment_root: Path, manifest: Path, qc_table: Path, canonical_domain: Path, output: Path, device: str) -> dict[str, Any]:
        output.mkdir(parents=True, exist_ok=True)
        domain, boundaries, observations_by_config = self._dynamic_context(experiment_root, manifest, qc_table, canonical_domain, device)
        if not boundaries:
            return stage_status("partial", "no Exp1 boundary checkpoints found")
        rows: list[dict[str, Any]] = []
        for boundary in boundaries:
            checkpoint = torch.load(boundary.path, map_location="cpu", weights_only=False)
            config = checkpoint["effective_config"]
            key = (str(config.get("training", {}).get("normalization", {}).get("mode", "per_series")), bool(config.get("model", {}).get("film", {}).get("implementation") in {"s2v_style_single_frame", "s2vdreme_reference_8layer"}))
            observations = observations_by_config[key]
            model = reconstruct_exp1_checkpoint(checkpoint, domain, observations, torch.device(device))
            selected = select_fixed_location_frames(observations, frames_per_location=3)
            all_frames_by_location: dict[tuple[str, str], list[Any]] = defaultdict(list)
            for observation in observations:
                if observation.qc_valid:
                    all_frames_by_location[(observation.view.upper(), observation.slice_id)].append(observation)
            for frames_all in all_frames_by_location.values():
                frames_all.sort(key=lambda item: (float(item.timestamp_s), int(item.dynamic_frame_id)))
            checkpoint_dir = output / f"checkpoint_{boundary.label}"
            for name in ("scores", "dvf", "mbc", "reprojection"):
                (checkpoint_dir / name).mkdir(parents=True, exist_ok=True)
            grid, _ = fixed_physical_grid(model.canonical_lower_world_mm, model.canonical_upper_world_mm, (16, 16, 16))
            raw_mbc = model.respiratory_mbc.raw_active(mbc_points_from_world_grid(grid), active_levels=boundary.active_levels).detach()
            per_level_rms = raw_mbc.square().mean(dim=(0, 2, 3)).sqrt()
            per_level_max = raw_mbc.abs().amax(dim=(0, 2, 3))
            for level in range(boundary.active_levels):
                _render_vector_field_planes(checkpoint_dir / "mbc", f"L{level + 1}", raw_mbc[0, level], label="RAW MBC BASIS FIELD — not score-weighted DVF")
            np.savez_compressed(checkpoint_dir / "mbc" / "raw_mbc_basis.npz", world_grid_mm=grid.detach().cpu().numpy(), raw_mbc_basis_mm=raw_mbc.cpu().numpy(), per_level_rms_mm=per_level_rms.cpu().numpy(), per_level_max_mm=per_level_max.cpu().numpy(), label="RAW MBC BASIS FIELD", score_weighted_label="score-weighted DVF")
            location_rows = []
            for key_location, frames in selected.items():
                full_frames = all_frames_by_location[key_location]
                raw_scores = torch.cat([model.encode_motion_scores_raw(frame)["resp_scores"][:, :boundary.active_levels] for frame in full_frames], dim=0)
                centered = posthoc_center_scores(raw_scores)
                dvfs = torch.cat([weighted_dvf_from_precomputed_basis(raw_mbc, model.encode_motion_scores_raw(frame)["resp_scores"][:, :boundary.active_levels]) for frame in full_frames], dim=0)
                token = location_token(*key_location)
                np.savez_compressed(checkpoint_dir / "dvf" / f"{key_location[0]}_{token}.npz", world_grid_mm=grid.detach().cpu().numpy(), dvf_mm=dvfs.detach().cpu().numpy(), dvf_mean_mm=dvfs.mean(0).detach().cpu().numpy(), dvf_dynamic_mm=(dvfs - dvfs.mean(0)).detach().cpu().numpy())
                import matplotlib.pyplot as plt
                _render_vector_field_planes(checkpoint_dir / "dvf", f"{key_location[0]}_{token}_frame_middle", dvfs[len(dvfs) // 2], label="score-weighted DVF")
                figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
                flat = raw_scores.detach().cpu().numpy().reshape(len(full_frames), -1); flat_centered = centered.detach().cpu().numpy().reshape(len(full_frames), -1)
                for index in range(flat.shape[1]):
                    axes[0].plot([frame.timestamp_s for frame in full_frames], flat[:, index], marker="o", label=f"L{index // 3 + 1}.c{index % 3}")
                    axes[1].plot([frame.timestamp_s for frame in full_frames], flat_centered[:, index], marker="o")
                axes[0].set_title(f"raw score — {key_location[0]}/{key_location[1]}"); axes[1].set_title("POST-HOC DIAGNOSTIC: raw score - temporal mean (not used in Exp1 training)"); axes[1].set_xlabel("timestamp (s)"); axes[0].legend(ncol=3, fontsize=7); figure.tight_layout(); figure.savefig(checkpoint_dir / "scores" / f"{key_location[0]}_{token}.png", dpi=140); plt.close(figure)
                figure, axis = plt.subplots(figsize=(8, 4)); times = [frame.timestamp_s for frame in full_frames]; dvf_rms = dvfs.square().mean(dim=(-1, -2)).sqrt().detach().cpu().numpy(); dvf_max = dvfs.square().sum(dim=-1).sqrt().amax(dim=-1).detach().cpu().numpy(); axis.plot(times, dvf_rms, marker="o", label="DVF RMS (mm)"); axis.plot(times, dvf_max, marker="o", label="DVF max magnitude (mm)"); axis.set(xlabel="timestamp (s)", title=f"DVF magnitude vs time — {key_location[0]}/{key_location[1]}"); axis.legend(); axis.grid(alpha=.25); figure.tight_layout(); figure.savefig(checkpoint_dir / "dvf" / f"{key_location[0]}_{token}_vs_time.png", dpi=140); plt.close(figure)
                prediction_mse = []
                for frame in frames:
                    prediction = predict_observation_slice(model, frame, boundary.stage, seed=20260823)
                    observed = frame.image[0].detach().cpu().numpy(); predicted = prediction.detach().cpu().numpy(); residual = np.abs(observed - predicted); shared_window = fixed_window(np.concatenate((observed.ravel(), predicted.ravel())), robust=True); residual_window = (0., max(float(residual.max()), 1e-6))
                    render_independent_images(checkpoint_dir / "reprojection" / f"{key_location[0]}_{token}_{frame.dynamic_frame_id:05d}.png", [observed, predicted, residual], ["observed", f"full motion prediction ({boundary.stage})", "absolute residual"], [shared_window, shared_window, residual_window]); prediction_mse.append(float(np.mean((predicted - observed) ** 2)))
                score_metrics = static_dynamic_rms(raw_scores); dvf_metrics = static_dynamic_rms(dvfs)
                location_rows.append({"view": key_location[0], "slice_id": key_location[1], "mse": float(np.mean(prediction_mse)), **{f"raw_score_{name}": value for name, value in score_metrics.items()}, "dvf_static_rms_mm": dvf_metrics["static_rms"], "dvf_dynamic_rms_mm": dvf_metrics["dynamic_rms"], "dvf_static_dynamic_ratio": dvf_metrics["static_dynamic_ratio"], "mbc_rms_mm": float(per_level_rms.mean()), "mbc_max_mm": float(per_level_max.max())})
            per_view = {view: float(np.mean([item["mse"] for item in location_rows if item["view"] == view])) for view in ("SAX", "2CH", "4CH") if any(item["view"] == view for item in location_rows)}
            row = {"checkpoint": boundary.label, "stage": boundary.stage, "active_levels": boundary.active_levels, "representative_reprojection_mse": float(np.mean([item["mse"] for item in location_rows])), **{f"representative_reprojection_mse_{view}": value for view, value in per_view.items()}, "metric_cohort": "representative fixed physical middle location per view; early/middle/late frames"}
            for metric in ("raw_score_static_rms", "raw_score_dynamic_rms", "raw_score_static_dynamic_ratio", "dvf_static_rms_mm", "dvf_dynamic_rms_mm", "dvf_static_dynamic_ratio", "mbc_rms_mm", "mbc_max_mm"):
                row[metric] = float(np.mean([item[metric] for item in location_rows]))
            rows.append(row)
        fields = sorted({key for row in rows for key in row});
        with (output / "boundary_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        (output / "boundary_summary.json").write_text(json.dumps(rows, indent=2) + "\n"); return stage_status("rendered", checkpoint_count=len(boundaries), checkpoint_root=str(boundaries[0].path.parent), boundary_summary="boundary_summary.json", cohort_policy="one physical middle location per view; full temporal sequence for score/DVF curves; early/middle/late frames for reprojection", posthoc_centering="diagnostic only")

    def _v6_inr(self, experiment_root: Path, output: Path, device: str) -> dict[str, Any]:
        target = nib.load(str(experiment_root / "02_static_nesvor" / "static_reference.nii.gz")); payload, model, backend = self._load_static_model(experiment_root, device)
        center_point_batch_size = 8192; psf_point_batch_size = 4096; max_support_points = 100000; diagnostic_seed = 20261106
        values = evaluate_inr_on_nifti_lattice(model, target, device=torch.device(device), chunk_size=center_point_batch_size)
        save_unmasked_static_inr(values, target, output / "static_inr_center_unmasked.nii.gz")
        args = payload["args"]; output_resolution = float(getattr(args, "output_resolution", min(target.header.get_zooms()[:3]))); psf_factor = float(getattr(args, "output_psf_factor", 1.)); n_samples = int(getattr(args, "n_inference_samples", 128)); psf_resolution = output_resolution * psf_factor
        mask = payload["mask"]; from nesvor.image.image_utils import transformation2affine
        mask_affine = transformation2affine(mask.image, mask.transformation, float(mask.resolution_x), float(mask.resolution_y), float(mask.resolution_z))
        support = export_support_mask(mask.mask, mask_affine, output / "static_support_mask_for_comparison.nii.gz")
        from nibabel.processing import resample_from_to
        support_on_static = resample_from_to(nib.load(str(support)), target, order=0); nib.save(support_on_static, str(output / "support_on_static_lattice.nii.gz"))
        masked = np.asanyarray(target.dataobj).astype(np.float32); support_values = np.asanyarray(support_on_static.dataobj).astype(np.float32); support_mask = support_values > .5
        figures: dict[str, Any] = {}; plane_metadata: list[dict[str, Any]] = []
        for spec in representative_plane_specs(target):
            axis = spec["axis"]; axis_index = int(spec["axis_index"]); fixed_index = int(spec["fixed_index"])
            psf_values = evaluate_inr_psf_plane(model, target, spec, device=torch.device(device), psf_resolution=psf_resolution, n_samples=n_samples, seed=diagnostic_seed, point_batch_size=psf_point_batch_size)
            plane_target = nib.Nifti1Image(np.zeros(spec["shape"], dtype=np.float32), spec["affine"])
            save_unmasked_static_inr(psf_values, plane_target, output / f"psf_plane_{axis}.nii.gz")
            center_plane = np.take(values, fixed_index, axis=axis_index); psf_plane = np.squeeze(psf_values, axis=axis_index); support_plane = np.take(support_values, fixed_index, axis=axis_index); formal_plane = np.take(masked, fixed_index, axis=axis_index)
            shared_window = fixed_window(np.concatenate((psf_plane.reshape(-1), formal_plane.reshape(-1))), robust=True)
            figure_name = f"inr_psf_plane_{axis}.png"; figure_policy = render_independent_images(output / figure_name, [center_plane, psf_plane, support_plane, formal_plane], ["PSF-free INR center", f"PSF plane ({axis})", "support mask", "formal static"], [fixed_window(center_plane, robust=True), shared_window, (0., 1.), shared_window])
            figures[figure_name] = {"policy": "aligned_center_psf_support_formal", "fixed_voxel_index": spec["voxel_index"], "world_coordinate_mm_ras": spec["world_coordinate"], "psf_formal_shared_window": shared_window, "mask_window": (0., 1.), **figure_policy}
            plane_metadata.append({"axis": axis, "fixed_voxel_index": spec["voxel_index"], "world_coordinate_mm_ras": spec["world_coordinate"], "shape": list(spec["shape"]), "nifti": f"psf_plane_{axis}.nii.gz", "display_window": {"center": fixed_window(center_plane, robust=True), "psf_formal_shared": shared_window, "support": (0., 1.)}, "value_summary": volume_summary(psf_values)})
        agreement = support_point_agreement(model, masked, support_mask, target.affine, device=torch.device(device), psf_resolution=psf_resolution, n_samples=n_samples, seed=diagnostic_seed, max_points=max_support_points, point_batch_size=psf_point_batch_size)
        summary = {"center_unmasked_full_volume": volume_summary(values), "formal_static_full_volume": volume_summary(masked), "psf_representative_planes": plane_metadata, "support_point_psf_agreement": agreement, "encoding_backend": backend, "quantity": {"center": "psf_free_inr_center_value", "psf": "same_sampling_semantics_independent_deterministic_diagnostic_realization"}, "output_resolution": output_resolution, "psf_factor": psf_factor, "psf_resolution": psf_resolution, "n_samples": n_samples, "diagnostic_rng_seed": diagnostic_seed, "center_point_batch_size": center_point_batch_size, "psf_point_batch_size": psf_point_batch_size, "max_support_points": max_support_points}
        (output / "intensity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return stage_status("rendered", figures=figures, **summary)

    def _v7_reprojection(self, experiment_root: Path, output: Path, device: str) -> dict[str, Any]:
        """Use pinned NeSVoR ``sample_slices``; no central-plane substitute is used."""
        payload, model, _ = self._load_static_model(experiment_root, device)
        from nesvor.image import load_slices
        from nesvor.inr.sample import sample_slices
        corrected_dir = experiment_root / "02_static_nesvor" / "slices"; files = numeric_corrected_slices(corrected_dir); pose_path = experiment_root / "02_static_nesvor" / "corrected_pose_manifest.json"
        from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import load_corrected_pose_map
        poses = load_corrected_pose_map(pose_path); selected_keys = self._representative_pose_keys(poses); selected_names = {Path(poses[key].corrected_slice_file).name for key in selected_keys}; chosen = [path for path in files if path.name in selected_names]
        if not chosen: return stage_status("partial", "no numeric corrected slices available")
        all_observed = load_slices(str(corrected_dir), torch.device(device)); by_index = {int(path.name.split(".", 1)[0]): item for path, item in zip(files, all_observed)}; observed = [by_index[int(path.name.split(".", 1)[0])] for path in chosen]
        mask_device = nesvor_volume_to_device(payload["mask"], torch.device(device)); device_contract = validate_nesvor_device_contract(model, observed, mask_device, torch.device(device)); simulated = sample_slices(model, observed, mask_device, float(getattr(payload["args"], "output_psf_factor", 1.0)), int(getattr(payload["args"], "n_inference_samples", 128)))
        figures: dict[str, Any] = {}
        for index, (source, predicted) in enumerate(zip(observed, simulated)):
            raw, estimate = source.image.squeeze().detach().cpu().numpy(), predicted.image.squeeze().detach().cpu().numpy(); residual = np.abs(raw - estimate); intensity_window = fixed_window(np.concatenate((raw.ravel(), estimate.ravel())), robust=True); residual_window = (0., max(float(residual.max()), 1e-6)); key = next((candidate for candidate in selected_keys if Path(poses[candidate].corrected_slice_file).name == chosen[index].name), None); label = f"{key[0]}_{location_token(key[0], key[1])}" if key else f"slice_{index:03d}"; name = f"reprojection_{label}.png"; figures[name] = render_independent_images(output / name, [raw, estimate, residual], ["corrected observed", "NeSVoR PSF reprojection", "absolute residual"], [intensity_window, intensity_window, residual_window]); figures[name]["observed_predicted_window"] = intensity_window; figures[name]["residual_window"] = residual_window; figures[name]["policy"] = "shared observed/predicted; separate nonnegative residual"
        return stage_status("rendered", slice_count=len(chosen), simulation_path="nesvor.inr.sample.sample_slices(directory, device)", effective_device=device_contract["requested_device"], device_contract=device_contract, figures=figures)
