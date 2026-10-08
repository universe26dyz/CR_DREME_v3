# C4 paper-aligned protocol audit

`source_first_change4_paperaligned.yaml` is a transferable DREME-style C4 baseline within CR_DREME_v3, not an exact DREME-MR reproduction. It preserves the source-first architecture, NeSVoR PSF/INR, SINR MBC, sequential pullback, cardiac box, true-timestamp Eq.8/Eq.9 implementation, and image-domain MSE data term.

| DREME setting | Current CR_DREME behavior | transfer status | final decision |
|---|---|---|---|
| Progressive schedule | Existing stages are retained; formal C4 splits them into scheduling-only segments | DIRECT_TRANSFER | S1A 500, S1B 1300, three 50+200 respiratory segments, S3A 50, S3B 3650 |
| Batch size 32 | Image-domain observations are view/location balanced and backpropagated one at a time | IMAGE_DOMAIN_ADAPTATION | `observations_per_update=32`, gradient-mean accumulation, one temporal loss/update |
| Spatial INR LR | Same trainable canonical INR interface | DIRECT_TRANSFER | S1A `2e-4`; S1B/Stage2/Stage3b `5e-5` |
| Motion-model LR | FiLM/SINR are the project's motion realization | IMAGE_DOMAIN_ADAPTATION | `5e-4` for active FiLM/MBC groups |
| TV weight | Existing NeSVoR image regularizer supports TV, but its zero-difference backward is unstable in this runtime | IMAGE_DOMAIN_ADAPTATION | `dreme_tv_stable` Eq.5 form, image weight `2e-6`; read-only loss-scale/backward gate required |
| MBC normalization weight | Current Eq.6-style SINR numerical implementation | IMAGE_DOMAIN_ADAPTATION | `1e-2`; no automatic retuning |
| Zero-mean score weight | Current Eq.7-style score implementation | DIRECT_TRANSFER | `1e-4` |
| Eq.8 weight | Existing true-timestamp cardiac leakage in respiratory score, with paired baseline subtraction | DIRECT_TRANSFER | `1e-1` |
| Eq.9 weight | Existing true-timestamp respiratory leakage in cardiac score | DIRECT_TRANSFER | `5e-2` |
| S3A cardiac initialization | The historical generic Stage3a freezes shared/respiratory FiLM and respiratory MBC, but that is not the paper-aligned C4 adaptation | IMAGE_DOMAIN_ADAPTATION | Keep canonical frozen for 50 updates while full FiLM, all three respiratory MBC levels, and cardiac MBC train; uncertainty remains frozen |
| K-space L1 data consistency | Project has reconstructed DICOM image observations and no raw k-space | NOT_TRANSFERABLE | Retain current image-domain MSE; do not claim k-space equivalence |
| Stage-I approximate-volume target | Source-first deliberately has no approximate NUFFT volume target | NOT_TRANSFERABLE | Retain no-motion image-domain Stage1; align only budget/LR structure |
| B-spline MBC implementation | Pinned upstream SINR B-spline FFD adapters | DIRECT_TRANSFER | Unchanged |
| Cardiac local coordinate system | Existing local cardiac box and taper | DIRECT_TRANSFER | Unchanged |
| Three respiratory + one cardiac level | Existing three progressive respiratory levels and one cardiac MBC | DIRECT_TRANSFER | Unchanged |
| Long Stage-III full joint optimization | Existing Stage3b unfreezes canonical, full FiLM, respiratory and cardiac MBC | DIRECT_TRANSFER | 3650-update Stage3b with uncertainty off |

The C4 config sets target concentration and PCA waveform to zero, leaves Stage3c at zero in the launcher, and uses no explicit DVF smoothness for this baseline (`smooth_resp=smooth_card=0`). This is a protocol control, not a change to generic regularizer support or to Change5A/Change5B.

For C4 only, `dreme_tv_stable` implements DREME Eq.5 in this NeSVoR/PSF image-domain runtime as `mean(abs(density - flip(density)) / sqrt(dx2))`. It is forward-equivalent to the pinned upstream TV expression but uses PyTorch's finite zero subgradient for `abs`, rather than backpropagating through `sqrt(d_density**2)`. It does not modify vendored NeSVoR or any historical generic/Change5 regularizer semantics.

The prior paper-aligned S3A mapping incorrectly inherited the generic card-head-only warm-up. The corrected scheduling override keeps the canonical field frozen while all active motion components train, which is the required image-domain adaptation rather than an exact DREME-MR reproduction. The transfer remains non-exact because the project has no raw k-space data or approximate-volume target, uses its existing 2-D geometry-conditioned FiLM/SINR realization, and evaluates Eq.8/Eq.9 on true timestamps.
