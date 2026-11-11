"""Evaluate a finished run on its held-out frames.

Example:
    python evaluate.py --run runs/scan01 --data scan.npz --dvf_regularity
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from simple4d import SliceSeries, load_config, load_run
from simple4d.evaluation import dvf_regularity, heldout_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="output directory of train.py")
    p.add_argument("--data", required=True, help="the .npz the run was trained on")
    p.add_argument("--checkpoint", default="model_final.pth")
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--dvf_regularity", action="store_true",
                   help="also report Jacobian determinant and inverse consistency (mm) for held-out codes")
    p.add_argument("--n_regularity", type=int, default=8, help="number of held-out codes for --dvf_regularity")
    a = p.parse_args()

    cfg = load_config(os.path.join(a.run, "config.yaml"))
    series = SliceSeries.load(a.data, cfg.depth_crop)
    model, cfg = load_run(a.run, series, a.checkpoint, a.device)
    _, held_idx = series.split(cfg.n_holdout)

    res = heldout_metrics(model, series, held_idx)
    s = res["summary"]
    print(f"{res['n_frames']} held-out frames")
    for v, name in (("v1", "held-out self-consistency (V1)"), ("v2", "prediction with target withheld (V2)")):
        print(f"  {name}: PSNR {s[v + '_psnr']:.2f} dB, SSIM {s[v + '_ssim']:.3f}, MAE {s[v + '_mae']:.4f}")

    if a.dvf_regularity:
        codes = model.encode_frames(series.frames, series.slice_index)
        sel = np.linspace(0, len(held_idx) - 1, min(a.n_regularity, len(held_idx))).round().astype(int)
        reg = [dvf_regularity(model, codes[held_idx[i]]) for i in sel]
        res["dvf_regularity"] = reg
        print(f"  Jacobian determinant min {min(r['jacobian_min'] for r in reg):.3f}, "
              f"folding {np.mean([r['folding_percent'] for r in reg]):.4f} %")
        if "ice_mean_mm" in reg[0]:
            print(f"  inverse-consistency error {np.mean([r['ice_mean_mm'] for r in reg]):.3f} mm (mean), "
                  f"{np.mean([r['ice_p99_mm'] for r in reg]):.3f} mm (99th percentile)")

    with open(os.path.join(a.run, "heldout_metrics.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
