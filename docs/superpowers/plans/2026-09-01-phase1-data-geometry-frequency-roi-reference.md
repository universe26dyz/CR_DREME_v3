# CardioResp 4D MRI Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify the complete pre-network pipeline for the local 50-frame SAX/2CH/4CH DICOM acquisition.

**Architecture:** One authoritative DICOM manifest carries acquisition time and physical plane metadata through all modules. Geometry, frequency discovery, a shared 3D cardiac box, and the SAX temporal-average reference consume that manifest without modifying source data; all generated artifacts live under ignored `results/`.

**Tech Stack:** Python 3.9, NumPy, SciPy, pydicom, PyYAML, Matplotlib, nibabel, standard-library `csv` and `unittest`.

**Spec:** `/home/universe/Documents/ChatGPT/4dsvr_CR_DREME/CardioResp_4D_Method_and_Codex_Prompt.md`

## Global Constraints

- Implement Phase 1 only; do not create `models/`, `rendering/`, `losses/`, or training code.
- Treat `/home/universe/SVR/data/DYL0709/DYL_20260709_dicom` as read-only.
- Current acquisition has 50 temporal positions per fixed slice, not 25.
- Use DICOM patient coordinates and an explicit `(column, row)` pixel API.
- Do not fabricate missing DICOM fields or frequency candidates.
- Keep local absolute paths in ignored `configs/subject_local.yaml`; commit only portable templates.
- Every Python module starts with the required provenance/input/output/CLI docstring.

---

### Task 1: Repository, configuration, DICOM inspection, manifest, and loader

**Files:**
- Create: `.gitignore`, `pyproject.toml`, `configs/default.yaml`, `configs/subject_example.template.yaml`
- Create: `src/cardioresp4d/config.py`, `src/cardioresp4d/data/inspect_dataset.py`, `src/cardioresp4d/data/build_manifest.py`, `src/cardioresp4d/data/dataset.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces `scan_dicom_frames(root, views) -> list[dict]`, `build_manifest(config) -> (Path, Path)`, and `CardioRespDataset` returning image, timestamp, view, slice ID, and geometry.

- [ ] Write synthetic-DICOM tests proving view filtering, 50-frame temporal ordering, AcquisitionTime conversion, no PHI columns, rescale handling, and percentile normalization.
- [ ] Run `python -m unittest tests.test_data -v`; verify failure because modules do not exist.
- [ ] Implement the minimal scanner, JSON inspection, CSV/JSON manifest, config loader, and dataset.
- [ ] Re-run the test and all module `--help` commands; require success.
- [ ] Run inspection and manifest generation on the real read-only dataset and validate row/series counts.

### Task 2: DICOM patient-world geometry, normalization, and QC

**Files:**
- Create: `src/cardioresp4d/geometry/world_geometry.py`, `coordinate_normalization.py`, `geometry_qc.py`
- Test: `tests/test_geometry.py`

**Interfaces:**
- Produces `DicomPlane`, `pixel_to_world(column,row)`, `world_to_pixel(world)`, `WorldNormalizer`, and a JSON/PNG QC report.

- [ ] Write hand-derived anisotropic-spacing tests that fail if row/column spacing is swapped, plus inverse, orthonormality, bounds, and plane-corner tests.
- [ ] Run the geometry tests and verify the expected missing-module failure.
- [ ] Implement the DICOM PS3.3 mapping and subject bounding-box normalization with serializable forward/inverse matrices.
- [ ] Re-run tests and generate real SAX/2CH/4CH 3D plane QC.
- [ ] Require sub-micrometre numerical round-trip error and inspect the rendered plane image.

### Task 3: Fixed-slice image-domain PCA, PSD, and frequency candidates

**Files:**
- Create: `src/cardioresp4d/frequency/pca_motion.py`, `frequency_bands.py`
- Test: `tests/test_pca_frequency.py`

**Interfaces:**
- Produces PCA temporal PCs/explained variance, one-sided PSD, per-slice candidates, and transparent cross-slice consensus JSON.

- [ ] Write synthetic image-series tests with independently specified 0.24 Hz and 1.20 Hz signals, irregular-time rejection, and insufficient-resolution null candidates.
- [ ] Verify RED, implement NumPy SVD and SciPy periodogram using actual AcquisitionTime sampling.
- [ ] Verify GREEN and generate NPZ/CSV/PNG outputs for representative real slices.
- [ ] Run all eligible real slices; record duration, median dt, frequency resolution, Nyquist, variance, candidates, and confidence limitations without inventing peaks.

### Task 4: Shared 3D cardiac box and multi-view projection QC

**Files:**
- Create: `src/cardioresp4d/roi/cardiac_box.py`, `roi_qc.py`
- Test: `tests/test_roi.py`

**Interfaces:**
- Produces one axis-aligned patient-world `CardiacBox` and projects its same eight corners through each representative DICOM plane.

- [ ] Write tests for world/normalized bounds and hand-derived projection coordinates.
- [ ] Verify RED, implement config/volume-centred box construction and projection.
- [ ] Verify GREEN and generate SAX/2CH/4CH overlay PNGs plus JSON.
- [ ] Visually verify the same 3D box covers the cardiac region in all three views; adjust only the explicit local config if needed.

### Task 5: SAX temporal-average initial reference

**Files:**
- Create: `src/cardioresp4d/reference/build_initial_reference.py`
- Test: `tests/test_reference.py`

**Interfaces:**
- Produces `initial_reference.nii.gz` in `(column,row,slice)` array order and metadata JSON with source view, affine, spacing, slice ordering, frame counts, and normalization.

- [ ] Write a synthetic two-slice test with known means and affine columns.
- [ ] Verify RED, implement per-location temporal averaging and stack ordering along the median SAX normal.
- [ ] Verify GREEN, build the real reference, reopen it with nibabel, and validate shape/finite values/affine consistency.

### Task 6: Phase-1 runner, full validation, report, and verified Git snapshot

**Files:**
- Create: `scripts/run_pipeline.py`, `tests/test_phase1_pipeline.py`, `PHASE1_REPORT.md`
- Modify: `configs/default.yaml`, `configs/subject_example.template.yaml` only if real-data evidence requires it.

**Interfaces:**
- `run_pipeline.py --from-stage/--to-stage` calls module APIs only and never duplicates business logic.

- [ ] Write a synthetic end-to-end runner test and verify RED.
- [ ] Implement ordered stages: inspect, manifest, geometry, frequency, ROI, reference; verify GREEN.
- [ ] Run every `unittest`, every Phase-1 `--help`, compileall, and the full real-data pipeline.
- [ ] Inspect every JSON/CSV/NIfTI/PNG output, including visual QC and non-finite checks.
- [ ] Run `git status`, `git diff --stat`, `git diff`, and `git diff --check`; ensure no data/results/secrets are tracked.
- [ ] Write `PHASE1_REPORT.md` with real structure, provenance, inputs/outputs, tests, Geometry/PCA/ROI/reference findings, limitations, and Phase 2 plan.
- [ ] Commit `phase(1): implement data geometry frequency ROI and reference` and create annotated tag `phase-01-data-geometry` only after all verification succeeds.

