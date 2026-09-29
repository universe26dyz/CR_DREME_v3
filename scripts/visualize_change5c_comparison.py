#!/usr/bin/env python3
"""Make fixed-scale, cross-checkpoint Change5C reprojection comparison panels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from cardioresp4d.diagnostics.change5c import jacobian_summary, quiver_subsample_indices, world_to_grid_index


def _input(value: str) -> tuple[str, Path]:
    if "=" not in value: raise argparse.ArgumentTypeError("--input must be LABEL=VISUALIZATION_ROOT")
    label, path = value.split("=", 1); return label, Path(path)


def _location_dir(root: Path, view: str, slice_id: str) -> Path:
    return root / f"visualize_{view}_{slice_id}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=_input, required=True); parser.add_argument("--locations-json", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); inputs = args.input; locations = json.loads(args.locations_json.read_text(encoding="utf-8"))
    locations = locations.get("locations", locations); args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for PNG comparison panels") from exc
    for location in locations:
        view, slice_id = location["view"], location["slice_id"]
        series = {label: sorted(_location_dir(root, view, slice_id).glob("reprojection_*.npz")) for label, root in inputs}
        count = min((len(paths) for paths in series.values()), default=0)
        if not count: continue
        all_images = [np.load(path) for paths in series.values() for path in paths[:count]]
        intensity = np.concatenate([np.r_[item["acquired"].reshape(-1), item["resp_only"].reshape(-1), item["resp_plus_card"].reshape(-1)] for item in all_images])
        residual = max(float(np.abs(item["acquired"] - item["resp_only"]).max()) for item in all_images); effect = max(float(np.abs(item["resp_plus_card"] - item["resp_only"]).max()) for item in all_images)
        frame_paths = []
        for frame in range(count):
            identities = [(int(np.load(series[label][frame])["dynamic_frame_id"]), float(np.load(series[label][frame])["timestamp_s"])) for label, _ in inputs]
            if len(set(identities)) != 1:
                raise ValueError(f"cross-checkpoint frame mismatch at {view}/{slice_id} frame {frame}: {identities}")
            figure, axes = plt.subplots(len(inputs), 6, figsize=(18, 3.2 * len(inputs)), squeeze=False)
            for row, (label, _) in enumerate(inputs):
                item = np.load(series[label][frame]); acquired, resp, joint = item["acquired"], item["resp_only"], item["resp_plus_card"]
                panels = (acquired, resp, joint, np.abs(acquired - resp), np.abs(acquired - joint), np.abs(joint - resp))
                titles = ("Acquired", "Resp-only", "Resp+Card", "|Acq-Resp|", "|Acq-Joint|", "|Joint-Resp|")
                mse_resp, mse_joint = np.mean((acquired - resp) ** 2), np.mean((acquired - joint) ** 2)
                for column, (panel, title) in enumerate(zip(panels, titles)):
                    axes[row, column].imshow(panel, cmap="gray", vmin=float(intensity.min()) if column < 3 else 0., vmax=float(intensity.max()) if column < 3 else effect if column == 5 else residual)
                    axes[row, column].set_title(title); axes[row, column].axis("off")
                effect_values = np.abs(joint - resp)
                axes[row, 0].set_ylabel(f"{label}\nMSE {mse_resp:.4g}->{mse_joint:.4g}\ngain {100*(mse_resp-mse_joint)/max(mse_resp, np.finfo(np.float32).eps):.2f}%\nmean/p95 effect {effect_values.mean():.4g}/{np.quantile(effect_values,.95):.4g}")
            figure.suptitle(f"{view}/{slice_id} frame {frame}: fixed scales across checkpoints")
            figure.tight_layout(); path = args.output_dir / f"{view}_{slice_id}_frame_{frame:03d}_comparison.png"; figure.savefig(path, dpi=150); plt.close(figure); frame_paths.append(path)
            dvf_paths = {label: _location_dir(root, view, slice_id) / f"dvf_{frame:03d}.npz" for label, root in inputs}
            if all(path.exists() for path in dvf_paths.values()):
                dvf_data = {label: np.load(path) for label, path in dvf_paths.items()}
                maximum = max(float(np.linalg.norm(item[name], axis=-1).max()) for item in dvf_data.values() for name in ("cardiac_dvf_mm", "respiratory_dvf_mm", "total_dvf_mm"))
                figure, axes = plt.subplots(len(inputs), 4, figsize=(12, 3.2 * len(inputs)), squeeze=False)
                for row, (label, _) in enumerate(inputs):
                    item = dvf_data[label]; shape = item["cardiac_dvf_mm"].shape[:3]; grid = item["observation_world_mm"]
                    center = world_to_grid_index(torch.as_tensor(item["cardiac_center_world_mm"]), torch.as_tensor(grid[0, 0, 0]), torch.as_tensor(grid[-1, -1, -1]), shape)
                    magnitudes = [np.linalg.norm(item[name], axis=-1)[center[0]] for name in ("cardiac_dvf_mm", "respiratory_dvf_mm", "total_dvf_mm")]
                    jacobian = item["jacobian_observation_to_reference"][center[0]]
                    for column, (panel, title) in enumerate(zip([*magnitudes, jacobian], ("Cardiac DVF mm", "Resp DVF mm", "Total DVF mm", "Jacobian"))):
                        axes[row, column].imshow(panel, cmap="magma" if column < 3 else "coolwarm", vmin=0. if column < 3 else None, vmax=maximum if column < 3 else None); axes[row, column].set_title(title); axes[row, column].axis("off")
                    card = np.linalg.norm(item["cardiac_dvf_mm"], axis=-1); jacobian = item["jacobian_observation_to_reference"]
                    stats = jacobian_summary(torch.as_tensor(jacobian))
                    axes[row, 0].set_ylabel(f"{label}\ncard RMS {np.sqrt(np.mean(card**2)):.3g}\np95 {np.quantile(card,.95):.3g}, max {card.max():.3g}\nJ {stats['min']:.3g}/{stats['p01']:.3g}/{stats['median']:.3g}/{stats['p99']:.3g}/{stats['max']:.3g}; <=0 {stats['fraction_leq_zero']:.3g}")
                    y_indices, z_indices = quiver_subsample_indices(shape[1], min(shape[1], 16)), quiver_subsample_indices(shape[2], min(shape[2], 16))
                    yy, zz = np.meshgrid(y_indices, z_indices, indexing="ij"); vectors = item["cardiac_dvf_mm"][center[0]][np.ix_(y_indices, z_indices)]
                    axes[row, 0].quiver(zz, yy, vectors[..., 2], vectors[..., 1], color="cyan", scale=max(maximum, np.finfo(np.float32).eps), scale_units="xy")
                figure.suptitle(f"{view}/{slice_id} frame {frame}: fixed DVF scale across checkpoints")
                figure.tight_layout(); figure.savefig(args.output_dir / f"{view}_{slice_id}_frame_{frame:03d}_dvf_jacobian.png", dpi=150); plt.close(figure)
        try:
            import imageio.v2 as imageio
            imageio.mimsave(args.output_dir / f"{view}_{slice_id}_comparison.gif", [imageio.imread(path) for path in frame_paths], duration=.35)
        except (ImportError, OSError):
            pass
        volume_paths = {label: _location_dir(root, view, slice_id) / "conditioned_dynamic_4d.npz" for label, root in inputs}
        if all(path.exists() for path in volume_paths.values()):
            volumes = {label: np.load(path) for label, path in volume_paths.items()}
            intensity_max = max(float(max(item["resp_only_implied_4d"].max(), item["conditioned_dynamic_4d"].max())) for item in volumes.values())
            effect_max = max(float(np.abs(item["cardiac_effect_4d"]).max()) for item in volumes.values())
            figure, axes = plt.subplots(len(inputs), 9, figsize=(24, 3.2 * len(inputs)), squeeze=False)
            for row, (label, _) in enumerate(inputs):
                item = volumes[label]; grid = item["world_grid_mm"]; shape = item["conditioned_dynamic_4d"].shape[1:]
                center_world = np.load(volume_paths[label].parent / "dvf_000.npz")["cardiac_center_world_mm"]
                center = world_to_grid_index(torch.as_tensor(center_world), torch.as_tensor(grid[0, 0, 0]), torch.as_tensor(grid[-1, -1, -1]), shape)
                panels, titles = [], []
                for image, name in ((item["resp_only_implied_4d"][0], "Resp-only"), (item["conditioned_dynamic_4d"][0], "Resp+Card"), (np.abs(item["cardiac_effect_4d"][0]), "|Cardiac effect|")):
                    panels.extend((image[center[0]], image[:, center[1], :], image[:, :, center[2]])); titles.extend((f"{name} XY", f"{name} XZ", f"{name} YZ"))
                for column, (panel, title) in enumerate(zip(panels, titles)):
                    effect_column = column >= 6
                    axes[row, column].imshow(panel, cmap="gray", vmin=0. if effect_column else None, vmax=effect_max if effect_column else intensity_max); axes[row, column].set_title(title); axes[row, column].axis("off")
                axes[row, 0].set_ylabel(label)
            figure.suptitle(f"{view}/{slice_id}: observation-conditioned 3D, not synchronized cine")
            figure.tight_layout(); figure.savefig(args.output_dir / f"{view}_{slice_id}_cardiac_effect_3d.png", dpi=150); plt.close(figure)
    (args.output_dir / "README.txt").write_text("Comparison panels use shared intensity, residual, and cardiac-effect scales across checkpoints. GIF creation is optional; PNG contact-sheet frames are always emitted. Observation-conditioned outputs are not a globally synchronized physiological cine.\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
