# v3 Source-first implementation report

Date: 2026-09-09. Specification: `CR_DREME_v3_FULL_PIPELINE_2026-09-09_SOURCE_FIRST_NESVOR_SINR.md`.

## Completed source audit

`SOURCE_PROVENANCE.md` records the actual repository, immutable commit, source
file, reused symbol, license, and adapter for every requested primitive.
Unmodified pinned checkouts exist in `third_party/NeSVoR`, `third_party/film`,
and external `/home/universe/SVR/code/external/SINR`. The supplied DREME-MR, S2V-DREME, NISF++, SIMPLE-4D,
and PCA/FFT papers were reviewed. Only the direct source calls listed in the
provenance file are described as upstream reuse.

## Implemented changes

- Added NeSVoR world-mm INR, PSF, and dynamic-frame uncertainty adapters.
- Added a direct FiLM primitive adapter and changed the geometry motion encoder
  to call it instead of locally multiplying gamma and beta.
- Changed canonical-domain coverage outputs to a continuous
  `canonical_domain_mask`, sparse plane-center QC maps, practical 3-sigma
  NeSVoR-PSF support maps, per-view count, and per-observation count. Coverage
  zeros are explicitly not canonical-volume holes.
- Added identity/numerical adapter tests, external SINR FFD adapter, unified
  Stage1/Stage2a-b-c/Stage3 trainer, and expanded canonical-domain tests.
- Added the source-first configuration and third-party notices.

## Mainline status

The local CPU runtime now has external SINR MBC/FFD adapters and a unified
Stage 1 / Stage 2a-b-c / Stage 3 trainer. It never constructs an initial V,
mean-slice training set, Stage1A, or Stage1B. The cardiac MBC requires an
explicit local cardiac box within the full-FOV canonical bounds.

## Verification actually run

- Read pinned NeSVoR `INR`, `INR.sample_batch`, `resolution2sigma`,
  `sigma_net`, `log_var_slice`, and image-regularization source.
- Read pinned SINR `Siren`, `BSplineSiren`, `CubicBSplineFFDTransform`,
  `cubic_bspline1d`, and `conv1d` source.
- Read pinned FiLM primitive source and confirmed the official `gamma * x + beta` operation.
- Confirmed commits: NeSVoR `2e96a91...`, SINR `1a524ca...`, FiLM `fe43ddf...`.
- Ran CPU source adapter tests (NeSVoR/PSF/uncertainty/FiLM: 6/6), external
  SINR adapter tests (3/3), coverage test (1/1), and unified progressive
  synthetic tests (3/3). The Stage3 test confirmed non-None gradients for
  INR, FiLM, respiratory SINR, cardiac SINR, and uncertainty.
- Ran the source-first configuration validator red→green cycle: it rejects
  legacy canonical/SINR, cardiac-only, and unbalanced settings, and accepts
  the pinned source contract (2/2 tests passed).
- Ran a static source-identity audit that verified all three pinned commit
  IDs, the two MIT license files, the absent SINR license file, and the direct
  NeSVoR/SINR/FiLM import statements.

The Python runtime is local `knesvr_torch` CPU-only (PyTorch 2.5.1,
`cuda=False`). `omegaconf` was installed for SINR's own import path; `ipdb` and
`termcolor` were installed for FiLM's own import path. No real-data experiment
was launched, by design: it remains a GPU-server task.

## Remaining issues

1. Execute the shortest real-data Stage1 + Stage2a validation on the GPU server
   after copying this checkout and setting `CARDIORESP4D_SINR_ROOT`.
2. Frequency-leakage regularization based on the Phase-1 subject-specific,
   nonuniform-timestamp PCA/FFT bands has not yet been connected to the
   trainer's score-history objective; it must be added before a long Stage3
   scientific run. The current short GPU command deliberately stops at Stage2a.
