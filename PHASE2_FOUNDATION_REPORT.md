# Phase 1 seal and Phase 2 P0 foundation

## Scope completed

- Phase 1 hard-QC policy was sealed in `ce1ca92`: only
  `slice_local_scale_absolute` and `manual_exclusion` can make a frame
  invalid.  `low_ncc`, `global_intensity_scale`, `scale_corrected_residual`,
  `slice_local_structure`, and `slice_local_scale_robust` remain diagnostics.
- Added an explicit canonical reconstruction domain.  It is the patient-world
  AABB of valid multi-view acquisition support union the DREME-style cardiac
  box; it is deliberately not inferred from `initial_reference.nii.gz`.
  `initial_reference_valid_mask`, when supplied, is recorded only as Stage-1
  supervision support and does not change this domain.
- The one directed local coverage run wrote
  `result_phase1/geometry/canonical_domain.json`, per-view and union coverage
  NIfTIs, and `domain_coverage_qc.png`.  It verified the cardiac box and the
  SAX/2CH/4CH target support are inside the canonical domain.  The JSON stores
  `world_to_normalized`, `normalized_to_world`, and
  `scale_mm_per_normalized_unit`, preserving explicit mm semantics for later
  B-spline, DVF, smoothness, and Jacobian calculations.

## Phase 2 P0 modules added

- `models/hash_inr.py`: continuous hashed-grid canonical INR.
- `models/bspline_mbc.py`: physical-mm cubic respiratory multi-level and
  cardiac local B-spline MBCs, score-weighted DVF construction, and zero
  cardiac-box boundary.
- `models/cardioresp_motion.py`: score-weighted fields and explicitly defined
  observation-to-reference sequential pullback:
  `x_ref = y + d_c(y) + d_r(y + d_c(y))`.
- `rendering/thick_slice_renderer.py`: direct INR query at five
  Gauss-Legendre samples through the DICOM physical slice thickness.
- `models/uncertainty.py`: positive quadrature-aggregated pixel variance plus
  acquired-frame variance, for a later Gaussian NLL (no standalone variance
  penalty).

## Targeted verification

- Phase-1 hard-QC regression plus canonical-domain deterministic test: 6/6.
- Phase-2 deterministic mathematics and zero-motion forward/backward smoke
  tests: 6/6.
- New modules compile cleanly; `git diff --check` is clean.

## Deliberately not implemented

Motion encoder, frequency/loss stack, Gaussian NLL integration, progressive
training, inference, and evaluation remain the next Phase-2/3 work.  No full
Phase-1 rerun or real-data training was performed in this increment.
