# SIMPLE-4D

Surrogate-free 4D MRI from standard fast multi-slice 2D MRI, using implicit neural representations.

[Paper](https://arxiv.org/abs/2607.17906) | [PDF](https://arxiv.org/pdf/2607.17906) | [Project page](https://ttlmh.github.io/simple-4d/)

This repository contains the reference implementation of

> M. Li, X. Wu, X. Li, O. Pusterla, P. C. Cattin, S. Safai, A. J. Lomax, Y. Zhang.
> *Portable surrogate-free 4D MRI from standard fast multi-slice 2D MRI via implicit neural representations.*
> arXiv:2607.17906, 2026. [doi:10.48550/arXiv.2607.17906](https://doi.org/10.48550/arXiv.2607.17906)

SIMPLE-4D fits a patient-specific model to one interleaved 2D cine scan. Every acquired slice is encoded into a motion code. A bidirectional deformation network, conditioned on that code, maps the slice into a canonical anatomy. The canonical anatomy is a multi-resolution hash-grid INR, and a thick-slice forward model links it to the acquired slices. After fitting, the model can render a 3D volume and displacement field for the respiratory state of every acquired frame. No navigator, external surrogate or respiratory binning is needed.

```
acquired slice ──► motion encoder ──► code z_t
                                        │
frame coordinates x ──► deformation network (x, z_t) ──► canonical coordinates ──► canonical anatomy f_θ
                                        ▲                                              │
                        inverse deformation (cycle loss)        thick-slice integration ▼
                                                                         predicted slice ◄─► L1 loss
```

## Installation

```bash
conda create -n simple4d python=3.10 -y && conda activate simple4d
pip install torch --index-url https://download.pytorch.org/whl/cu128   # choose the build for your CUDA
pip install -r requirements.txt
# recommended: tiny-cuda-nn hash encoding (fast, used for all results in the paper)
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

If tiny-cuda-nn is not installed, a pure-PyTorch hash encoding is used instead. It is several times slower and does not reproduce the paper results exactly.

## Data

**The MRI data of the paper cannot be shared** (identifiable volunteer data; ethics approvals EKNZ 2019-01060 and EKNZ 2023-00674). The code runs on your own scans. A synthetic phantom is included for testing.

### Input format

A scan is a single `.npz` file:

| key             | shape / type        | content |
|-----------------|---------------------|---------|
| `frames`        | float32 `[N, H, W]` | preprocessed slices in [0, 1], **in acquisition order** |
| `slice_index`   | int `[N]`           | slice-position index 0 … D−1 (sorted along the slice normal) |
| `spacing`       | float `[3]`         | (slice spacing, row spacing, column spacing) in mm |
| `num_positions` | int (optional)      | D (default: `max(slice_index) + 1`) |
| `depth_crop`    | int `[2]` (optional)| slice positions excluded from the loss at the start / end of the stack |

The slice positions do not have to be sampled uniformly. In the 0.5 T HASTE data, for example, one central position is acquired about 19 times more often than the others.

### Converting DICOM

```bash
# positions swept cyclically 0..D-1 (1.5 T protocol of the paper; several series of a session are concatenated)
python tools/dicom_to_npz.py sweep series_A/ series_B/ series_C/ -o scan.npz \
    --positions 64 --flip --crop 70 310 5 295 --window 300 --level 120

# temporal order given by (InstanceNumber mod skip, InstanceNumber) (0.5 T HASTE cine of the paper)
python tools/dicom_to_npz.py interleaved haste_cine/ -o scan.npz --skip 50 --flip --window 600 --level 300
```

The crop and window/level values above are the ones used in the paper. Adapt them to your field of view and contrast.

### Synthetic phantom

```bash
python tools/make_phantom.py -o phantom.npz
python train.py --data phantom.npz --out_dir runs/phantom --n_holdout 16 --iters 2000
```

## Usage

```bash
# fit one scan (paper configuration; about 20-25 min for 768 slices of 240x290 on one RTX 4090)
python train.py --data scan.npz --out_dir runs/scan01                     # 1.5 T: last 64 frames held out
python train.py --config configs/paper_0p5T_haste.yaml --data scan.npz --out_dir runs/scan02   # 0.5 T

# held-out evaluation (also run at the end of training) and DVF regularity in mm
python evaluate.py --run runs/scan01 --data scan.npz --dvf_regularity

# export motion codes, volumes / displacement fields of selected frames, canonical anatomy
python render.py --run runs/scan01 --data scan.npz --heldout --dvf --template --nifti
python render.py --run runs/scan01 --data scan.npz --frames 0:-1:10
```

All options are listed by `python train.py --help`. A YAML file passed with `--config` sets defaults, and command-line options override them.

**Evaluation.** The last `--n_holdout` frames of the scan are never seen during fitting. Two held-out metrics are reported:
- **V1, held-out self-consistency:** the held-out slice is encoded, and the slice at the same position is rendered from that estimate.
- **V2, prediction with the target withheld:** the motion code is interpolated from the two temporally adjacent frames, and the target slice is not used at all.

PSNR, SSIM (7×7 window) and MAE are computed on [0, 1] intensities.

**Outputs.**
- Volumes are arrays `[rows, columns, slices]`.
- Displacement fields are `[rows, columns, slices, 3]`, with components along (rows, columns, slices) in mm. The forward field maps frame → canonical.
- Codes are posterior means.

## Reproducing the configurations of the paper

| configuration | file |
|---|---|
| SIMPLE-4D, 1.5 T (defaults) | `configs/paper_1p5T.yaml` |
| SIMPLE-4D, 0.5 T HASTE | `configs/paper_0p5T_haste.yaml` |
| concatenation conditioning | `configs/ablations/concat.yaml` |
| FiLM conditioning | `configs/ablations/film.yaml` |
| SIREN canonical anatomy | `configs/ablations/siren_template.yaml` |
| deterministic encoder | `configs/ablations/deterministic_encoder.yaml` |
| unidirectional DVF | `configs/ablations/unidirectional.yaml` |
| point (thin-slice) supervision | `configs/ablations/point_slice.yaml` |

The paper used the seeds `--seed 6 --loader_seed 12`, plus repeated runs with other seeds. Hash-grid training uses atomic GPU operations, so repeated runs agree closely but not bit for bit.

## Repository layout

```
simple4d/
  networks.py     hash-grid / SIREN canonical anatomy, motion encoders, conditioned deformation networks
  model.py        forward model (bidirectional DVF, thick-slice integration), training step, inference
  losses.py       Jacobian and DVF-magnitude regularisers
  coords.py       coordinate conventions and sampling
  data.py         input format and dataset
  evaluation.py   held-out metrics, DVF regularity
  config.py       options and defaults
train.py  evaluate.py  render.py
tools/dicom_to_npz.py  tools/make_phantom.py
configs/
```

## License

[PolyForm Noncommercial License 1.0.0](LICENSE.md). You may use, modify and share the code for **non-commercial purposes**, including academic research, teaching and personal study. **Commercial use is not permitted.** For a commercial licence, contact the corresponding author (ye.zhang@psi.ch).

Third-party components keep their own licences: PyTorch (BSD-style) and tiny-cuda-nn (BSD-3-Clause), which are installed separately.

## Citation

If you use this code, please cite our [paper](https://arxiv.org/abs/2607.17906) (also available in [`CITATION.cff`](CITATION.cff)):

```bibtex
@misc{li2026simple4d,
  title         = {Portable surrogate-free {4D} {MRI} from standard fast multi-slice {2D} {MRI} via implicit neural representations},
  author        = {Muheng Li and Xinyang Wu and Xia Li and Orso Pusterla and Philippe C. Cattin and Sairos Safai and Antony J. Lomax and Ye Zhang},
  year          = {2026},
  eprint        = {2607.17906},
  archivePrefix = {arXiv},
  primaryClass  = {physics.med-ph},
  doi           = {10.48550/arXiv.2607.17906},
  url           = {https://arxiv.org/abs/2607.17906}
}
```
