# Task 5 report — SAX temporal-average initial reference

## Scope and provenance

- Implemented only Phase 1 Task 5.
- Direct S2V-DREME Stage-I principle: average all 50 dynamic frames at each fixed SAX location, physically order the locations, and stack the means as the initial reference.
- Necessary adaptation: authoritative DICOM patient LPS geometry is retained separately from the NIfTI RAS+ affine; DICOM `(rows, columns)` pixels are explicitly transposed into volume `(column, row, slice)` order.
- No 2CH/4CH fusion, resampling invention, INR, model, or Phase-2 code was added.

## TDD evidence

1. RED: `python -m unittest tests.test_reference -v` failed with `ModuleNotFoundError: cardioresp4d.reference` before implementation.
2. GREEN: `4/4` Task-5 tests pass. They cover:
   - true 50-frame temporal averaging with a known changing non-square image sequence;
   - directory/slice-ID order deliberately disagreeing with physical position;
   - `(column,row,slice)` output and the otherwise-hidden DICOM transpose;
   - hand-derived DICOM column/row/slice affine columns;
   - explicit `diag(-1,-1,1,1)` LPS-to-RAS conversion and reopened NIfTI affine;
   - rejection of non-50 counts, duplicate locations, nonparallel orientations, inconsistent matrix/spacing, and an irregular stack exceeding the configured origin-affine residual tolerance.

## Files and interfaces

- `src/cardioresp4d/reference/build_initial_reference.py`
  - `build_initial_reference(manifest_path, output_dir, expected_frames_per_slice=50, ...)`
  - writes `initial_reference.nii.gz`, `initial_reference.json`, and `reference_qc.png`;
  - validates each temporal location and the cross-slice sampling grid before reading all pixels;
  - uses the existing `CardioRespDataset` intensity boundary: modality transform when present, per-frame 1st/99th percentile normalization, clipping to `[0,1]`, then temporal average;
  - constructs `dicom_lps_affine` with columns `raw IOP[:3]*PixelSpacing[1]`, `raw IOP[3:]*PixelSpacing[0]`, and `unit SAX normal*median physical position step`;
  - constructs the actual NIfTI affine as `diag(-1,-1,1,1) @ dicom_lps_affine` and reopens the saved file to validate shape, finiteness, and affine.
- `src/cardioresp4d/reference/__init__.py`: package boundary.
- `tests/test_reference.py`: four synthetic tests using non-square images and 50 frames per location.
- `pyproject.toml`: declares the required `nibabel` runtime dependency.

## Real-data validation

- Input: ignored `results/dicom_manifest.csv`; only its 2,500 SAX frames were consumed.
- Output shape: `(213, 240, 50)` = `(column, row, slice)`, `float32`.
- Values: all finite; min `0.0`, max `1.0`, mean `0.23163329`.
- Source layers: 50 SAX locations, each with exactly 50 frames. First/last physically sorted IDs are `SAX_s1_801` and `SAX_s50_5701`.
- Reopened NIfTI voxel sizes: approximately `[1.5011737, 1.5011737, 1.9999981]` mm.
- Physical position steps from `ImagePositionPatient`: `1.9999905` to `2.0000105` mm; maximum residual from the regular affine prediction is `0.0001005` mm.
- All 2,500 real SAX frames report `identity_without_rescale_tags`; this is explicitly recorded rather than inventing slope/intercept values.
- Artifacts (Git-ignored):
  - `results/reference/initial_reference.nii.gz` (8.9 MB)
  - `results/reference/initial_reference.json`
  - `results/reference/reference_qc.png`
- The QC PNG was visually inspected. The central SAX temporal mean and both orthogonal stack views are anatomically recognizable and spatially continuous.
- Orthogonal QC panels use physical y-spacing/x-spacing aspect ratios, so their displayed anisotropy reflects the approximately 1.50 x 1.50 x 2.00 mm grid.

## Verification

- Focused Task-5 suite: `4/4` passed.
- Full Phase-1 suite at this point: `41` tests run, `40` passed, `1` optional environment-gated real-loader test skipped; no failures.
- `python -m compileall -q src tests`: passed.
- `python -m cardioresp4d.reference.build_initial_reference --help`: passed.
- Real CLI build, NIfTI reopen, finite/range/shape/affine audit: passed.
- `git diff --check`: passed before commit.

## Limitations and risks

- The actual `ImagePositionPatient` sampling interval is about `2 mm`, while the DICOM fields report `SliceThickness=8 mm` and `SpacingBetweenSlices=10 mm`. The reference affine deliberately follows measured physical origins, not the contradictory/nonrepresentative spacing tag. The metadata retains the measured steps and residuals for audit.
- This is an anisotropic, highly overlapping acquired SAX stack. It is not an isotropic resampling or multi-view fused volume; any later regridding belongs to an explicit downstream stage.
- Per-frame percentile normalization is inherited from the Task-1 loader for consistency. It can alter absolute inter-frame intensity scale; the exact policy and output statistics are recorded in JSON.
