# Dynamic Gauge Audit Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Harden the existing Exp1 dynamic/reference-gauge audit and isolated Exp2-A diagnostics before any server execution.

**Architecture:** Keep the model-level optional gauge and existing V8/V9 core. Add a deterministic checkpoint-root resolver, shared location-local cohort metric helpers, explicit Exp2-A diagnostics/runner state, and a thin Exp2-A adapter reusing the same dynamic rendering primitives.

**Tech Stack:** Python, PyTorch, Matplotlib, pytest, existing CardioResp4D visualization/training helpers.

**Spec:** `/home/universe/Downloads/CR_DREME_dynamic_gauge_audit_hardening_before_exp2a_2026-11-06.md`

## Global Constraints

- Local CPU only; do not run Exp2-A or claim server/CUDA results.
- Do not modify Stage 0/NeSVoR science or existing Exp1 checkpoints.
- Use all QC-valid frames for scientific metrics; representative cohorts are figures only.
- Keep one narrow hardening commit and do not push.

## Review Focus

- Experiment-root versus `04_training` checkpoint resolution is deterministic and non-recursive.
- Location-local temporal decomposition prevents between-location DC offsets from becoming dynamics.
- Representative V8/V9 metrics are never labeled as full-cohort metrics.
- Exp2-A state/resume/provenance and gradient ownership are explicit.
- Exp1's provider-free score path remains numerically raw.

### Task 1: Add failing hardening tests

**Files:** `tests/test_dynamic_audit.py`, `tests/test_reference_gauge_exp2a.py`, `tests/test_visualization_audit.py`

- [x] Test checkpoint-root resolution, full-cohort grouping/metrics, V9 output labels, adapter registration, comparison consumption, runner state, gradient summaries, and provider-free raw equivalence.
- [x] Run the focused tests and confirm failures identify missing hardening behavior.

### Task 2: Harden shared audit/cohort utilities

**Files:** `src/cardioresp4d/visualization/dynamic_audit.py`, `src/cardioresp4d/visualization/reference_gauge.py`

- [x] Add `resolve_dynamic_checkpoint_root` and location-local score/DVF pooling with total RMS/ratio semantics and by-view aggregation.
- [x] Preserve explicit raw/reference/effective branches and checkpoint immutability.

### Task 3: Harden Exp1 retrospective and V8/V9 outputs

**Files:** `scripts/audit_exp1_reference_score_gauge.py`, `src/cardioresp4d/visualization/adapters/exp1_reference.py`

- [x] Use all QC-valid frames/locations for metrics, representative cohort only for figures, record both policies, and write resolved root/source hashes.
- [x] Rename representative metrics, render DVF/MBC spatial PNGs and full selected-location curves, and retain active-level labels.

### Task 4: Harden Exp2-A diagnostics/state/adapter/comparison

**Files:** `src/cardioresp4d/experiments/respiratory_reference_gauge_exp2a.py`, `scripts/train_respiratory_reference_gauge_exp2a.py`, `src/cardioresp4d/visualization/adapters/exp2a_reference_gauge.py`, `src/cardioresp4d/visualization/__init__.py`, `scripts/compare_exp1_exp2a_reference_gauge.py`

- [x] Add full-cohort step-0/step-200 metrics, gradient summaries and frozen-owner checks, Exp2-A-specific state/resume validation, factual initialization audit, adapter registration, and authoritative Exp1 L1-warm comparison fields.

### Task 5: Verify and commit

- [x] Run the required focused command and the project test suite in the local CPU environment.
- [x] Review the diff, preserve unrelated files, and create one local hardening commit without pushing.
