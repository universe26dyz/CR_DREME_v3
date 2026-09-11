# v3 Change3 Closure Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the source-first v3 mainline satisfy the accepted closure audit, with CPU-verifiable runtime semantics and no long GPU training.

**Architecture:** Replace score-zero masking with one stage contract that controls call graph, active MBC raw bases, regularizers, optimizer ownership and losses. Build the model/trainer only through validated config and emit effective runtime facts. Frequency, identity, preflight, checkpoint and metrics use typed provenance records rather than implicit dictionaries.

**Tech stack:** Python 3, PyTorch CPU, pinned vendored NeSVoR/SINR/FiLM, NumPy/SciPy, `unittest`.

**Spec:** `CR_DREME_v3_FULL_PIPELINE_2026-09-09_SOURCE_FIRST_NESVOR_SINR.md`, `v3_change3_codex.md`, and `docs/audits/v3_change3_pre_fix_audit.md`.

## Global constraints

- Preserve full acquisition-supported canonical FOV; cardiac box is local motion/sampling/QC only.
- Keep only `slice_local_scale_absolute` and `manual_exclusion` hard-invalid.
- Do not copy/modify upstream SINR, NeSVoR or FiLM source; adapters only.
- CPU tests use `conda run --no-capture-output -n knesvr_torch`; do not execute long GPU training.
- Every behavior change follows red → green → regression; never close C/I with a skip, zero multiplication, report-only or config-only change.

---

### Task 1: Stage-aware source motion and loss contract (C01, C02, C04, I01–I03, I14–I16, I21, I25)

**Files:**
- Modify: `src/cardioresp4d/adapters/sinr_mbc.py`, `src/cardioresp4d/training/model.py`, `src/cardioresp4d/training/trainer.py`, `src/cardioresp4d/losses/motion_loss.py`
- Create: `src/cardioresp4d/training/stage_contract.py`, `tests/test_v3_change3_stage_contract.py`, `tests/test_v3_change3_motion_losses.py`

- [ ] Write failing tests for raw active basis axes, non-trainable schedule, stage call counts, separate local/full grids, exact loss totals and optimizer state ownership.
- [ ] Run those tests and record expected failures against current score-zero/gated implementation.
- [ ] Implement `StageContract`, raw/active SINR calls, stage-specific `predict`, active-only regularizers, configured physical smoothness and retained optimizer groups/state.
- [ ] Run targeted tests until green; then run existing motion/SINR/smoke tests.

### Task 2: Config → builder → effective runtime (C05, I11, I12, I18, I22, M02, M03)

**Files:**
- Create: `src/cardioresp4d/training/build_model.py`
- Modify: `configs/source_first.yaml`, `training/source_first_config.py`, `training/model.py`, `training/trainer.py`, `scripts/train_source_first.py`
- Create: `tests/test_v3_change3_factory.py`, `tests/test_v3_change3_reporting.py`

- [ ] Write failing perturbation/effective-config/fidelity-image-reg/domain-coverage/metrics tests.
- [ ] Run them red.
- [ ] Implement strict config schema, builder, actual-object effective config, stage optimizer config and structured metrics.
- [ ] Run targeted factory/report tests green and regression tests touching configuration.

### Task 3: Frequency source contract (C03, I04–I09)

**Files:**
- Modify: `src/cardioresp4d/frequency/training_prior.py`, `src/cardioresp4d/losses/frequency_loss.py`, `training/sampler.py`, `training/trainer.py`
- Create: `tests/test_v3_change3_frequency_contract.py`

- [ ] Write failing tests for schema reconciliation, explicit baseline pairs, complex Eq.8, multi-frequency resolution, offset/count/DC behavior and location-sequence selection.
- [ ] Run red and inspect the actual Phase-1 schema fixture/available artifact before choosing lookup semantics.
- [ ] Implement typed resolved prior and NUDFT contract; fail fast for unresolvable evidence, never invent a location map.
- [ ] Run targeted frequency regressions green.

### Task 4: Data identity, QC, normalization, seed, checkpoint and preflight (I10, I17, I19, I23, I24, I29–I31)

**Files:**
- Create: `src/cardioresp4d/training/runtime_state.py`, `scripts/preflight_source_first.py`, `tests/test_v3_change3_runtime_contract.py`, `tests/test_v3_change3_preflight.py`
- Modify: `data/dataset.py`, `training/sampler.py`, `scripts/train_source_first.py`, `requirements-source-first.txt`, `pyproject.toml`, `training/source_first_config.py`

- [ ] Write failing identity/QC/normalization/seed/checkpoint/preflight/dependency-source-identity tests.
- [ ] Run red.
- [ ] Implement strict manifest handoff, shared QC reason parser, normalization diagnostics, RNG capture/restore, complete checkpoint/resume and no-grad preflight.
- [ ] Run targeted tests green and a CPU synthetic preflight.

### Task 5: Regression boundary, prompt/docs and final verification (I26–I33, M01–M05)

**Files:**
- Modify/Create: `tests/*`, `docs/prompts/*`, `README.md`, `IMPLEMENTATION_REPORT.md`, `SOURCE_PROVENANCE.md`, `changelog.md`, `docs/audits/v3_change3_post_fix_audit.md`

- [ ] Classify and repair all seven existing failures from their production boundary; do not weaken v3 policies.
- [ ] Replace prompt archive symlink with regular copies, label legacy entry, align docs and changelog to verified facts.
- [ ] Run targeted tests, upgraded Stage1→3 CPU synthetic smoke, then full `unittest discover`.
- [ ] Run real-data preflight if required artifacts are available; otherwise record precise missing artifact and keep gate FAIL.
- [ ] Execute the prescribed §44 `rg` command, create fresh post-fix audit, and calculate gate only from evidence.

## Plan self-review

- C01–C05 are covered by Tasks 1–3; all I01–I33 and M01–M05 map to Tasks 1–5.
- No task changes upstream source or authorizes GPU training.
- The only unresolved decision is I04 mapping, explicitly constrained to actual schema/artifact evidence rather than an assumption.
