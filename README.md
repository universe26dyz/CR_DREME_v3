# CardioResp4D v3 source-first status

The v3 specification is implemented only where the audited source licence and
runtime prerequisites permit it. `SOURCE_PROVENANCE.md` is the definitive
module-level record.

## Direct upstream source

- NeSVoR `INR`, its HashEmbedder fallback, `resolution2sigma`, Gaussian sample
  semantics, and sigma-network builder are imported from the pinned MIT source
  via `src/cardioresp4d/adapters/nesvor_*.py`.
- FiLM modulation is imported from the pinned MIT `FiLM` primitive via
  `src/cardioresp4d/adapters/film.py`.
- SINR `BSplineSiren` and `CubicBSplineFFDTransform` are imported directly
  from the pinned vendored source under `third_party/SINR`.
- PyTorch Gaussian NLL is used through its official loss class.

## Necessary adapters and paper-derived work

Adapters only translate patient-world millimetres, dynamic-frame IDs, and
motion insertion between PSF samples and the canonical query. The full-FOV
canonical-domain/coverage construction, asynchronous geometry-conditioned
encoder, DREME low-rank organization, sequential pullback, frequency priors,
and progressive orchestration are paper-derived necessary adaptations.

## Legacy / ablation modules

`models/hash_inr.py`, `models/sinr_mbc.py`, `models/bspline_mbc.py`,
`rendering/psf_renderer.py`, `models/uncertainty.py`, and
`training/stage1.py` remain for historical reproduction only. They must not be
selected by a v3 mainline runtime. Stage1A/Stage1B and initial-reference / mean
slice construction are not v3 mainline prerequisites.

## Vendored source lock

`third_party/SOURCE_LOCK.json` fixes repository URLs, commits and SHA256 hashes
for every runtime source file. NeSVoR, SINR and FiLM are ordinary vendored
directories, not gitlinks. `vasl12/SINR` is public research code with no
explicit upstream license file; it is not labelled MIT/Apache and is unmodified
inside this repository. No custom SINR/FFD fallback is selected by mainline.

## Clean local CPU source tests

Install the narrow adapter dependency set into the existing environment when
needed (PyTorch build selection remains platform-specific):

```bash
conda run -n knesvr_torch python -m pip install -r requirements-source-first.txt
conda run -n knesvr_torch python -m pip install -e . --no-deps
```

```bash
conda run --no-capture-output -n knesvr_torch \
  python -m unittest discover -s tests
```

## CPU real-data preflight, then first GPU validation

Run this no-step CPU validation before a GPU run. It validates the exact
manifest/QC/domain/frequency/source-lock contract, reports normalization groups
and location priors, and executes one no-grad forward for each stage.

```bash
python scripts/preflight_source_first.py \
  --source-config configs/source_first.yaml \
  --frequency-bands /absolute/path/to/results/frequency/frequency_bands.json \
  --manifest /absolute/path/to/results/dicom_manifest.csv \
  --qc-table /absolute/path/to/results/acquisition_qc/acquisition_qc.csv \
  --canonical-domain /absolute/path/to/results/domain/canonical_domain.json \
  --output-dir /absolute/path/to/results/training/source_first_preflight \
  --device cpu
```

## v3_change4 first GPU real-data validation

After copying this checkout to the GPU host, resume the existing Stage2c
checkpoint with Stage3a only, inspect the diagnostic, then run Stage3b only.
Do not automatically run Stage3c: it is the uncertainty-only refinement stage.

```bash
python scripts/train_source_first.py \
  --source-config configs/source_first.yaml \
  --frequency-bands /absolute/path/to/results/frequency/frequency_bands.json \
  --manifest /absolute/path/to/results/dicom_manifest.csv \
  --qc-table /absolute/path/to/results/acquisition_qc/acquisition_qc.csv \
  --canonical-domain /absolute/path/to/results/domain/canonical_domain.json \
  --output-dir /absolute/path/to/results/training/source_first_short \
  --device cuda --pixel-samples 256 --seed 0 --resume /absolute/path/to/stage2c/source_first_last.pt \
  --stage1-steps 0 --stage2a-steps 0 --stage2b-steps 0 --stage2c-steps 0 \
  --stage3a-steps 100 --stage3b-steps 0 --stage3c-steps 0
```

Then inspect the resulting checkpoint with `scripts/diagnose_change4_checkpoint.py`
and resume that checkpoint with `--stage3a-steps 0 --stage3b-steps 100
--stage3c-steps 0`. The legacy `--stage3-steps` option is a deprecated error.
This entrypoint uses individual valid dynamic frames; it never creates or
loads `initial_reference.nii.gz`, mean slices, Stage1A, or Stage1B.

## Change5A cardiac target-band concentration ablation

`configs/source_first_change5a.yaml` changes only
`training.loss_weights.cardiac_target_concentration` to `0.001`.  It evaluates
each fixed location against its resolved Phase-1 local cardiac band on a
timestamp-derived non-DC grid, summing all three cardiac-score channel powers
before the target/total ratio.  A valid narrow band between grid centres uses
its nearest centre; this is a project-specific image-domain adaptation.

DREME Eq.8/Eq.9 remain source-derived negative crossover suppression.  The
Change5A concentration term is not attributed to DREME-MR or S2V-DREME: it was
added after Change4 showed that Eq.9 alone permits cardiac score power in other
low-frequency nuisance bands.

For the GPU ablation, start both arms from the same Change4 Stage2c checkpoint,
seed, pixel samples, data/QC/domain and frequency prior. Run Stage3a=100 then
Stage3b=100 with Stage3c=0; compare weight `0` against `0.001` and do not use a
previously trained Stage3a/Stage3b checkpoint as the starting point.

## Change5B per-location PCA waveform weak supervision

`configs/source_first_change5b.yaml` retains Change5A concentration at `0.003`
and adds `cardiac_pca_waveform: 0.001`. For each fixed `view/slice_id`, it
loads that location's reliable Phase-1 `selected_pc` (explicitly 1-based),
strictly matches its own valid timestamps, and weakly supervises the current
three-dimensional cardiac-score subspace with a differentiable ridge fit and
`corr²`. This is sign-, scale-, permutation-, and rotation-insensitive; it is
an image-derived local surrogate, not ECG ground truth and never a global phase
label. See [CHANGE5B_SERVER_RUN.md](CHANGE5B_SERVER_RUN.md) for the GPU-only
Stage3a=1000 protocol.

## Change5C audit-only diagnostics and visualization

`scripts/diagnose_change4_checkpoint.py --all-locations` preserves default
probe-compatible output and additionally writes complete per-location JSON/CSV
records, explicit eligibility/skip reasons, grouped summaries, and separately
named aggregate motion metrics. `scripts/compare_all_location_diagnostics.py`
performs exact-key paired descriptive comparisons. `scripts/audit_stage3a_loss_gradients.py`
does no optimizer step and reports raw/configured-weighted gradients. The
visualization tool exports read-only reprojections and observation-conditioned
implied 3D dynamics, DVFs, and pullback Jacobians; it never claims globally
synchronized cardiac cine. See
[CHANGE5C_AUDIT_DIAGNOSTIC_VIS_SERVER_RUN.md](CHANGE5C_AUDIT_DIAGNOSTIC_VIS_SERVER_RUN.md).
