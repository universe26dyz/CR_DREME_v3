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
