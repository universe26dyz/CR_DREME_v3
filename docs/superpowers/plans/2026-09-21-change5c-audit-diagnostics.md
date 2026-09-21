# Change5C Audit, Diagnostics, and Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit existing Stage3a semantics, then add read-only full-location diagnostics, comparison, gradient audit, and dynamics visualization when the audit gate passes.

**Architecture:** Preserve the existing diagnostic CLI's default path and add opt-in modules/helpers for reusable records, aggregate summaries, and deterministic visualization math. Training code changes are limited to non-scientific correctness fixes backed by CPU regression tests; a material semantic defect halts downstream work.

**Tech Stack:** Python 3.9, PyTorch CPU, NumPy, pytest, existing NeSVoR/SINR adapters.

**Spec:** `docs/superpowers/specs/2026-09-21-change5c-audit-diagnostics-design.md`

## Global Constraints

- Work locally on CPU only; do not access server data or run training.
- Do not modify vendored NeSVoR/SINR, Stage3b/3c, architecture, or pullback convention.
- Preserve default diagnostic output and historical probe-only field meanings.
- Exclude hard-invalid observations and never interpolate or mix location waveforms.
- Label volumes as observation-conditioned implied 3D dynamics, never global cine.
- Stop downstream work on a material scientific audit blocker.

## Review Focus

- Existing Stage3a optimizer groups must not update frozen parameters after resume.
- Frequency/PCA losses must return finite zero-gradient skips for degenerate inputs.
- All-location records must preserve unevaluable locations with explicit skip reasons.
- Pairing must use exact `(view, slice_id)` intersection and expose missing keys.
- Jacobian/total-DVF helpers must respect physical spacing and sequential pullback identity.

### Task 1: Scientific audit and audit regressions

**Files:**
- Create: `SCIENTIFIC_AUDIT_CHANGE4_CHANGE5.md`, optionally `AUDIT_BLOCKER_REPORT.md`
- Modify: only a minimal non-scientific defect if the audit proves one
- Test: `tests/test_v3_change4_stage_schedule.py`, `tests/test_change5a_cardiac_concentration.py`, `tests/test_change5b_pca_waveform.py`

- [ ] Inspect forward model, ownership, Change4 losses, Change5A/B, and diagnostic populations against source and commit history.
- [ ] Add deterministic parameter-snapshot, spectral monotonicity/irregular-time, and PCA rotation/degeneracy tests where missing.
- [ ] Run the focused tests and classify any material defect before proceeding.

### Task 2: Reusable diagnostic records and all-location output

**Files:**
- Create: `src/cardioresp4d/diagnostics/checkpoint_metrics.py`
- Modify: `scripts/diagnose_change4_checkpoint.py`
- Test: `tests/test_diagnose_change4_checkpoint.py`

- [ ] Extract pure summary/eligibility functions with records containing location key, valid-frame count, semantic measurements, and structured skips.
- [ ] Add `--all-locations`, JSON and CSV outputs, per-view/per-PC summary statistics, and separately named aggregate motion metrics.
- [ ] Keep the default one-probe output compatibility-tested.

### Task 3: Paired diagnostic comparison

**Files:**
- Create: `scripts/compare_all_location_diagnostics.py`
- Test: `tests/test_compare_all_location_diagnostics.py`

- [ ] Implement exact-key pairing and missing-key reporting for two or more labeled inputs.
- [ ] Report distributional deltas, documented equality tolerance, and boolean transition tables without significance claims.
- [ ] Test pairing, missing locations, tolerance, and transitions with small JSON fixtures.

### Task 4: Read-only Stage3a gradient audit

**Files:**
- Create: `scripts/audit_stage3a_loss_gradients.py`
- Test: `tests/test_audit_stage3a_loss_gradients.py`

- [ ] Build a no-optimizer evaluation that computes each available Stage3a loss independently.
- [ ] Report raw/configured-weighted norms on cardiac FiLM head/MBC and prove frozen modules receive no gradients.
- [ ] Test zero paths and one synthetic connected loss.

### Task 5: Visualization helpers and CLI

**Files:**
- Create: `src/cardioresp4d/visualization/dynamics.py`, `src/cardioresp4d/visualization/__init__.py`, `scripts/visualize_checkpoint_dynamics.py`
- Test: `tests/test_visualize_checkpoint_dynamics.py`

- [ ] Implement chunked direct pullback volume evaluation, component/total displacement, physical-grid finite-difference Jacobian, and deterministic layout helpers.
- [ ] Implement CLI exports with optional-library fallbacks and explicit labels/provenance; keep all behavior read-only.
- [ ] Test identity/translation/scaling/folding Jacobians, pullback identity, deterministic selections, and volume shape/coordinates.

### Task 6: Documentation, server guide, and validation

**Files:**
- Create: `CHANGE5C_AUDIT_DIAGNOSTIC_VIS_SERVER_RUN.md`
- Modify: `README.md`, `IMPLEMENTATION_REPORT.md`, `SOURCE_PROVENANCE.md`

- [ ] Document audit verdict/limitations, diagnostic populations, visualization terminology, and exact server commands.
- [ ] Run focused tests, full CPU pytest suite, compile/import/source-lock checks, and `git diff --check` once stable.
- [ ] Commit/push only after the audit gate passes; otherwise deliver the blocker report and reproducer.
