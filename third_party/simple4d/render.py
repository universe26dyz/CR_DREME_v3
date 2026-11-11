"""Export volumes, displacement fields, the canonical anatomy and motion codes of a finished run.

Volumes are arrays [rows, columns, slices]; displacement fields [rows, columns, slices, 3]
with components (rows, columns, slices) in mm. With ``--nifti`` (requires nibabel) the
files are written as NIfTI with the voxel spacing in the header.

Examples:
    # codes of all frames and volumes of the held-out frames
    python render.py --run runs/scan01 --data scan.npz --heldout --dvf
    # every 10th frame of the scan, plus the canonical anatomy
    python render.py --run runs/scan01 --data scan.npz --frames 0:-1:10 --template --nifti
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from simple4d import SliceSeries, load_config, load_run


def parse_frames(spec, n):
    if ":" in spec:
        return list(range(n))[slice(*[int(v) if v else None for v in spec.split(":")])]
    return [int(v) % n for v in spec.split(",")]


def save(path, arr, spacing, nifti):
    if nifti:
        import nibabel as nib
        affine = np.diag([spacing[1], spacing[2], spacing[0], 1.0])  # rows, columns, slices
        nib.save(nib.Nifti1Image(arr.astype(np.float32), affine), path + ".nii.gz")
    else:
        np.save(path + ".npy", arr.astype(np.float32))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", default="model_final.pth")
    p.add_argument("--out", default=None, help="output directory (default: <run>/export)")
    p.add_argument("--frames", default=None, help="frame indices, e.g. '0,5,9' or a slice '0:-1:10'")
    p.add_argument("--heldout", action="store_true", help="export all held-out frames")
    p.add_argument("--dvf", action="store_true", help="also export the forward displacement fields")
    p.add_argument("--template", action="store_true", help="also export the canonical anatomy")
    p.add_argument("--nifti", action="store_true")
    p.add_argument("--device", type=int, default=None)
    a = p.parse_args()

    cfg = load_config(os.path.join(a.run, "config.yaml"))
    series = SliceSeries.load(a.data, cfg.depth_crop)
    model, cfg = load_run(a.run, series, a.checkpoint, a.device)
    out = a.out or os.path.join(a.run, "export")
    os.makedirs(out, exist_ok=True)

    codes = model.encode_frames(series.frames, series.slice_index)
    np.savez(os.path.join(out, "codes.npz"), codes=codes, slice_index=series.slice_index)
    print(f"codes of {len(codes)} frames -> {out}/codes.npz")

    frames = []
    if a.frames:
        frames += parse_frames(a.frames, len(series))
    if a.heldout:
        frames += list(series.split(cfg.n_holdout)[1])
    for j in sorted(set(int(f) for f in frames)):
        save(os.path.join(out, f"volume_{j:05d}"), model.render_volume(codes[j]), series.spacing, a.nifti)
        if a.dvf:
            save(os.path.join(out, f"dvf_{j:05d}"), model.dvf_volume_mm(codes[j]), series.spacing, a.nifti)
    if frames:
        print(f"{len(set(frames))} volumes{' and displacement fields' if a.dvf else ''} -> {out}")
    if a.template:
        save(os.path.join(out, "template"), model.template_volume(), series.spacing, a.nifti)
        print(f"canonical anatomy -> {out}/template")


if __name__ == "__main__":
    main()
