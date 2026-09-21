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

### MBC topology audit — v3_change2

DREME-MR defines one scalar score per level and Cartesian displacement
component: respiratory scores `[B,3,3]`, cardiac scores `[B,1,3]`. S2V-DREME
uses a SINR generator that emits a 3D displacement vector per control point.
The compatible contract is therefore `resp_mbc [B,3,N,3]` and `card_mbc
[B,1,N,3]`, with element-wise Cartesian weighting. The former local 9-channel
`[basis,xyz]` construction was not justified by either tensor definition and
has been removed from the formal adapter; upstream SINR is still called
directly with output dimension 3.

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

### v3_change2 status (not yet a GPU gate)

- Implemented and CPU-tested: Phase-1 aggregate prior parser, 3-channel
  Cartesian SINR MBC topology, Eq.6 MBC normalization, Eq.7 zero-mean score,
  nonuniform Eq.8/Eq.9 loss primitives, full-span temporal subsampling,
  runtime source-lock verification and CLI frequency-path precedence.
- Still required before claiming v3_change2 complete: config-to-model builder
  for every formal hyperparameter, separate respiratory/cardiac smoothness on
  configured 16³ grids, stage-specific learning-rate groups, and the requested
  fixture repair for all-repository legacy tests.
- Full `unittest discover -s tests`: 105 passed, 1 skipped, 5 errors in
  `tests.test_data` (single-view fixtures rejected by formal three-view Phase-1
  validator), and 2 legacy initial-reference/QC expectation failures. These
  are not hidden or excluded.

1. Run the shortest real-data Stage1 + Stage2a validation on the GPU server
   after replacing the template `configs/frequency_bands.json` with Phase-1
   subject-specific bands.
2. The tiny CPU smoke is not a real-data reprojection or long-training result;
   those remain GPU-server work.

## v3_change3 closure repair — 2026-09-10

## v3_change4 schedule and local-prior repair — 2026-09-14

Phase-1 respiratory and cardiac candidates are now independently resolved per
`view/slice_id`, with separate provenance and independently auditable global
fallback. Eq.9 consumes that resolved local respiratory prior. Stage3 is split
into Stage3a cardiac-head/MBC warm-up (MSE), Stage3b full joint refinement
(MSE), and Stage3c uncertainty refinement (Gaussian NLL). The canonical FOV,
cardiac local box, NUDFT `/N`, Eq.8/Eq.9 weights, source lock and uncertainty
architecture are unchanged. The portable SINR device, CPU checkpoint RNG, and
nested frequency-prior JSON report fixes are formal runtime behavior.

The v3_change3 repair moved the formal mainline from masking/fallback behavior
to explicit runtime contracts. `training/stage_contract.py` owns Stage1,
Stage2a/b/c and Stage3 call/gradient/regularizer ownership. Stage1 now calls
only NeSVoR PSF→INR; Stage2 calls only active respiratory source-SINR levels;
Stage3 adds cardiac source-SINR and dynamic-frame uncertainty. Respiratory
level scheduling is a non-trainable buffer, not a learnable amplitude gate.

`dreme_mbc_normalization` now preserves level×Cartesian axes while reducing
only batch/spatial samples. Eq.8 uses complex paired Fourier subtraction;
NUDFT uses relative timestamps, score centring and sample-count normalization.
The Phase-1 `slice_key=view/slice_id` producer schema was checked, enabling
per-location cardiac evidence only when that exact identity exists; otherwise
the report marks global fallback provenance.

`training/build_model.py` is the sole formal model constructor. It applies
canonical INR, PSF, SINR cps/grid/taper, FiLM, uncertainty, image regularizer
and separate respiratory/full-FOV vs cardiac/local-box smoothness settings;
effective config is introspected from objects. Runtime state now validates
hard-QC/identity/timestamps, records seed/RNG, retains optimizer state across
progression, writes resumable checkpoint state and emits per-step metrics.

Actually run locally in `knesvr_torch` (PyTorch 2.5.1 CPU): final full
`python -m unittest discover -s tests` — 126 passed, 1 explicit skip; targeted
stage/factory/frequency/runtime/data tests — 23 passed, 1 explicit skip. A
final real-data no-step CPU preflight passed on the locally available DYL0709
manifest sidecar plus full Phase-1 QC/domain/frequency artifacts: source lock
verified, 7200 observations (7150 valid, 50 hard-invalid) validated, 143
location candidates resolved, and Stage1–Stage3 forwards were finite. No GPU
or long training was run.

## v3_change5A cardiac target-band concentration — 2026-09-18

Change5A preserves the Change4 model, source-backed adapters, canonical INR,
PSF, uncertainty schedule, SINR MBCs, sampling, Stage1–Stage3 schedule, and
DREME Eq.8/Eq.9. It adds only a local Phase-1-cardiac-band target/total NUDFT
power ratio for Stage3a/b/c. Power is aggregated over all cardiac-score
channels before the ratio, and the denominator is a timestamp-derived non-DC
grid. Eq.8/Eq.9 are source-derived negative crossover suppression; Change5A is
a project-specific image-domain adaptation introduced because Change4 could
satisfy those suppression losses while cardiac score energy remained in
low-frequency nuisance bands. `source_first_change5a.yaml` sets the audit
weight to `0.001`; missing legacy config keys remain zero.

## v3_change5B per-location PCA waveform weak supervision — 2026-09-21

Change5B adds no model architecture or trainable projection. A cached provider
reads each reliable Phase-1 location's explicitly 1-based `selected_pc` from
its own `pca_psd.npz`, strictly matches only the sampler's valid timestamps,
and applies a differentiable ridge best-linear projection from `[T,3]` cardiac
scores to that fixed image-derived waveform. The loss is `1-corr²`, making it
invariant to PCA sign/scale and cardiac-score basis permutations/rotations.
Missing/unreliable locations receive no global substitute; malformed timestamp
alignment fails explicitly. Change5B remains project-specific weak supervision,
not ECG ground truth nor a DREME/S2V-DREME method claim.

## Change5C audit, diagnostics, and visualization — 2026-09-21

`SCIENTIFIC_AUDIT_CHANGE4_CHANGE5.md` records a source-level PASS/PASS WITH
DOCUMENTED LIMITATION audit of forward motion/PSF, Stage3a ownership,
Change4, Change5A, Change5B, and historical diagnostics. No material scientific
blocker was found. A CPU numerical regression proves that a Stage3a optimizer
step after Stage2c changes only the cardiac FiLM head/cardiac MBC. A small
non-scientific numerical repair makes PCA nonfinite-input skips finite and
zero-gradient.

The opt-in all-location diagnostic retains every valid location with a skip
reason when unevaluable, preserves legacy probe-only motion fields, adds
aggregate motion naming/population, and writes JSON/CSV. Companion comparison,
gradient-audit, and visualization CLIs are read-only; server commands are in
`CHANGE5C_AUDIT_DIAGNOSTIC_VIS_SERVER_RUN.md`. No GPU or real-data result was
run locally.
