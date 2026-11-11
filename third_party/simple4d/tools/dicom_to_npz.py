"""Convert a 2D cine slice series (DICOM) to the SIMPLE-4D input format (.npz).

Two acquisition orderings are supported.

sweep        slice positions are swept cyclically, 0, 1, ..., D-1, 0, 1, ... (e.g. the
             interleaved 1.5 T protocol). Frames are ordered by InstanceNumber (or file name)
             within each series, and several series of one session can be concatenated in the
             given order; the position of frame i of a series is i mod D.
interleaved  the temporal order is given by the instance numbers modulo a skip number
             (e.g. the 0.5 T HASTE cine): frames are sorted by (InstanceNumber mod skip,
             InstanceNumber) and the position index is the rank of SliceLocation. The number of
             frames per position may differ.

Images are window/level normalised to [0, 1] after an optional left-right flip and crop.

Examples:
    python tools/dicom_to_npz.py sweep series_03/ series_05/ series_07/ -o scan.npz \
        --positions 64 --flip --crop 70 310 5 295 --window 300 --level 120
    python tools/dicom_to_npz.py interleaved haste_cine/ -o scan.npz --skip 50 \
        --flip --window 600 --level 300
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pydicom


def list_dicoms(folder):
    files = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
             if f.lower().endswith((".dcm", ".ima")) or "." not in f]
    if not files:
        raise FileNotFoundError(f"no DICOM files in {folder}")
    return files


def headers(files):
    return [pydicom.dcmread(f, stop_before_pixels=True) for f in files]


def load_frame(path, flip, crop, window, level):
    img = pydicom.dcmread(path).pixel_array.astype(np.float32)
    if flip:
        img = img[:, ::-1]
    if crop:
        r0, r1, c0, c1 = crop
        img = img[r0:r1, c0:c1]
    if img.max() == img.min():
        return img
    lo = level - window / 2
    return ((np.clip(img, lo, lo + window) - lo) / window).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["sweep", "interleaved"])
    p.add_argument("series", nargs="+", help="DICOM folder(s); several folders only in sweep mode")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--positions", type=int, default=None, help="sweep: number of slice positions D")
    p.add_argument("--sort", choices=["instance", "name"], default="instance", help="sweep: frame order in a series")
    p.add_argument("--skip", type=int, default=50, help="interleaved: skip number of the instance ordering")
    p.add_argument("--flip", action="store_true", help="flip the columns (left-right)")
    p.add_argument("--crop", type=int, nargs=4, default=None, metavar=("R0", "R1", "C0", "C1"))
    p.add_argument("--window", type=float, default=300)
    p.add_argument("--level", type=float, default=120)
    p.add_argument("--slice_spacing", type=float, default=None, help="override the slice spacing (mm)")
    p.add_argument("--depth_crop", type=int, nargs=2, default=(0, 0), metavar=("START", "END"),
                   help="slice positions excluded from the photometric loss")
    a = p.parse_args()

    if a.mode == "sweep":
        if a.positions is None:
            p.error("sweep mode needs --positions")
        paths, index = [], []
        for folder in a.series:
            files = list_dicoms(folder)
            if a.sort == "instance":
                inst = [int(h.InstanceNumber) for h in headers(files)]
                files = [files[k] for k in np.argsort(inst, kind="stable")]
            paths += files
            index += [i % a.positions for i in range(len(files))]
        h0 = headers(paths[:2])
        if a.slice_spacing is not None:
            dz = a.slice_spacing
        elif float(getattr(h0[0], "SpacingBetweenSlices", 0) or 0) != 0:
            dz = float(h0[0].SpacingBetweenSlices)
        else:
            dz = abs(float(h0[1].SliceLocation) - float(h0[0].SliceLocation))
        D = a.positions
    else:
        if len(a.series) != 1:
            p.error("interleaved mode takes one series")
        files = list_dicoms(a.series[0])
        hs = headers(files)
        info = sorted(((int(h.InstanceNumber), round(float(h.SliceLocation), 1), f) for h, f in zip(hs, files)),
                      key=lambda t: (t[0] % a.skip, t[0]))
        locations = sorted({t[1] for t in info})
        rank = {loc: k for k, loc in enumerate(locations)}
        paths = [t[2] for t in info]
        index = [rank[t[1]] for t in info]
        D = len(locations)
        dz = a.slice_spacing if a.slice_spacing is not None else (
            abs(locations[1] - locations[0]) if D > 1 else float(hs[0].SpacingBetweenSlices))
        h0 = hs
    row, col = (float(v) for v in getattr(h0[0], "PixelSpacing", [1.0, 1.0]))

    frames = np.stack([load_frame(f, a.flip, a.crop, a.window, a.level) for f in paths])
    index = np.asarray(index, dtype=np.int16)
    np.savez_compressed(a.output, frames=frames.astype(np.float32), slice_index=index,
                        spacing=np.array([dz, row, col]), num_positions=D, depth_crop=np.array(a.depth_crop))
    counts = np.bincount(index, minlength=D)
    print(f"{a.output}: {len(frames)} frames of {frames.shape[1]}x{frames.shape[2]} pixels, {D} positions "
          f"({counts.min()}-{counts.max()} frames per position), spacing {dz:.2f} / {row:.3f} / {col:.3f} mm")


if __name__ == "__main__":
    main()
