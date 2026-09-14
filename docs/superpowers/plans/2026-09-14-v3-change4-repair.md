# v3_change4 Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement task-by-task with red/green checks.

**Goal:** Restore per-location respiratory evidence and split legacy Stage3 into cardiac warm-up, joint refinement, and uncertainty refinement without altering source-backed primitives.

**Architecture:** Extend the existing frequency prior with independently resolved respiratory/cardiac provenance. Make `StageContract` the sole authority for runtime path, data term and parameter ownership; trainer/model/CLI consume it. Formalize device/RNG/JSON fixes only at adapters/runtime boundaries.

**Tech Stack:** Python, PyTorch CPU/CUDA-optional, pinned NeSVoR/SINR/FiLM adapters, unittest/pytest.

**Spec:** `v3_change4_codex.md`

## Global Constraints

- Do not alter `third_party/` or `third_party/SOURCE_LOCK.json`.
- Preserve Stage1–Stage2c semantics, full acquisition FOV, local cardiac box, NUDFT `/N`, and existing hard-invalid policy.
- No long training or Phase1 rerun; only targeted/full unit/smoke and no-step checks.
- Stage3c is the first uncertainty stage; deprecated legacy `stage3` resume must fail.

### Task 1: Frequency-prior evidence and Eq.9 call path

**Files:** modify `frequency/training_prior.py`, `training/trainer.py`; create `tests/test_v3_change4_frequency_prior.py`.

- [ ] Write tests with two literal locations proving local respiratory bands are `f ± df/2`, each modality falls back independently, malformed reliable candidates fail, `asdict` serializes, and trainer sends each location's respiratory bands to Eq.9.
- [ ] Run the new test and observe failure because prior has only one `source` and Eq.9 reads global bands.
- [ ] Add separate respiratory/cardiac provenance, union-key construction, local baseline occupancy, and contract-driven Eq.9 argument.
- [ ] Re-run frequency tests green.

### Task 2: Contract-driven Stage3a/3b/3c

**Files:** modify `training/stage_contract.py`, `training/model.py`, `training/trainer.py`, `configs/source_first.yaml`, `training/source_first_config.py`; create `tests/test_v3_change4_stage_schedule.py`.

- [ ] Write tests that assert seven contracts, Stage3a real backward ownership, Stage3b/3c ownership, and MSE/MSE/NLL data behavior.
- [ ] Run and observe failure because only `stage3` exists.
- [ ] Implement immutable contract data-term and FiLM train mode; make model/trainer losses, metrics, optimizer LRs, and trainability consume it.
- [ ] Re-run stage tests green.

### Task 3: Resume, device and report hotfixes

**Files:** modify `adapters/sinr_mbc.py`, `scripts/train_source_first.py`, `scripts/preflight_source_first.py`; create `tests/test_v3_change4_resume.py`, `tests/test_v3_change4_cuda.py`.

- [ ] Write tests for Stage2c -> Stage3a resume, legacy Stage3 rejection, nested prior JSON, CPU/CUDA-safe FFD buffers, and preflight device placement.
- [ ] Run and observe expected failures.
- [ ] Implement CPU checkpoint loading/RNG restoration, schema-compatible progressive reconstruction, `asdict` report payloads, device-safe adapter construction, and `model.to(device)` preflight construction.
- [ ] Re-run resume/device tests green; CUDA test explicitly skips when unavailable.

### Task 4: Read-only checkpoint diagnostic and docs

**Files:** create `scripts/diagnose_change4_checkpoint.py`, `docs/audits/v3_change4_post_fix_audit.md`; modify `README.md`, source-first pipeline spec, `IMPLEMENTATION_REPORT.md`, `changelog.md`.

- [ ] Write a lightweight CLI test/dry-run fixture proving prior-audit JSON and no model mutation.
- [ ] Implement a read-only diagnostic that reports prior provenance, motion, same-checkpoint cardiac ablation, local-prior frequency semantics, and optional uncertainty statistics.
- [ ] Document only the Stage2c -> Stage3a -> Stage3b server protocol; keep Stage3c manual.

### Task 5: Final verification

- [ ] Run targeted change4 and affected legacy tests, then the complete pytest suite, `pip check`, source-lock validation, `git diff --check`, and one CPU no-step smoke/preflight if inputs are available.
- [ ] Independently audit the call graph against every §32 acceptance item and record actual CPU/CUDA/GPU-training status.
