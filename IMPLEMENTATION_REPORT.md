# v3 Source-first implementation report

Date: 2026-09-10. Specification: `CR_DREME_v3_FULL_PIPELINE_2026-09-09_SOURCE_FIRST_NESVOR_SINR.md`; second-round contract: `v3_change1_codex.md`.

## Completed source audit

`SOURCE_PROVENANCE.md` records the actual repository, immutable commit, source
file, reused symbol, license, and adapter for every requested primitive.
Unmodified pinned source trees exist in `third_party/NeSVoR`, `third_party/SINR`,
and `third_party/film`; `third_party/SOURCE_LOCK.json` records their identity.
The supplied DREME-MR, S2V-DREME, NISF++, SIMPLE-4D,
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
- Corrected geometry units (canonical-normalized position Fourier features;
  separate direction and acquisition inputs), oblique DICOM-basis PSF sampling,
  SINR logical control-grid versus dense-grid semantics, NeSVoR variance
  aggregation, true timestamp propagation, 80/20 cardiac sampling, and real
  stage freeze/unfreeze with per-stage Adam recreation.

## Mainline status

The local CPU runtime now has vendored-source SINR MBC/FFD adapters and a unified
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
- Ran `python -m unittest tests.test_v3_change1_training
  tests.test_source_backed_adapters tests.test_sinr_adapter
  tests.test_v3_change1_contracts tests.test_unified_progressive_smoke
  tests.test_source_first_config` in local `knesvr_torch`: 23/23 passed.
  This includes direct source identity, oblique PSF variance, logical
  8/12/16/16 control grids, variance equality, irregular timestamp frequency
  gradients, 80/20 sampling, real stage freeze/unfreeze, central pullback and
  unified Stage1→2a→2b→2c→3 synthetic smoke.
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

1. Run the shortest real-data Stage1 + Stage2a validation on the GPU server
   after replacing the template `configs/frequency_bands.json` with Phase-1
   subject-specific bands.
2. The tiny CPU smoke is not a real-data reprojection or long-training result;
   those remain GPU-server work.
