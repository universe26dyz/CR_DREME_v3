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
- SINR SIREN and cubic B-spline FFD are imported directly from the independent
  pinned external checkout via `src/cardioresp4d/adapters/sinr_mbc.py`.
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

## External SINR dependency

`vasl12/SINR` is a public research-code external pinned dependency at
`1a524ca7ae453b55310595fe957245088a108233`. It has no explicit upstream
license file, so it is not labelled MIT/Apache. No SINR source is copied or
modified in CR_DREME; its adapter directly imports the checkout. No custom
SINR/FFD fallback is selected by the v3 mainline.

## First GPU real-data validation

After copying this checkout and creating an independent SINR checkout on the
GPU host, run only the short Stage1 -> Stage2a chain:

```bash
CARDIORESP4D_SINR_ROOT=/absolute/path/to/SINR \
python scripts/train_source_first.py \
  --source-config configs/source_first.yaml \
  --manifest /absolute/path/to/results/dicom_manifest.csv \
  --qc-table /absolute/path/to/results/acquisition_qc/acquisition_qc.csv \
  --canonical-domain /absolute/path/to/results/domain/canonical_domain.json \
  --output-dir /absolute/path/to/results/training/source_first_short \
  --device cuda --stage1-steps 100 --stage2a-steps 100 \
  --stage2b-steps 0 --stage2c-steps 0 --stage3-steps 0
```

This entrypoint uses individual valid dynamic frames; it never creates or
loads `initial_reference.nii.gz`, mean slices, Stage1A, or Stage1B.
