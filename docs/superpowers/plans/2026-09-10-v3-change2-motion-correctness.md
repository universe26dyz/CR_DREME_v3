# V3 Change2 Motion Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make source-first v3 motion training obey the audited DREME Eq.6–9 contracts and consume Phase-1 frequency output directly.

**Architecture:** Keep NeSVoR, SINR and FiLM primitives unmodified. Add focused parser/loss/source-lock modules, reduce the SINR adapter to explicit Cartesian vector fields, and make the trainer construct its runtime from validated config.

**Tech Stack:** Python 3.9, PyTorch CPU, vendored NeSVoR/SINR/FiLM, unittest.

**Spec:** `docs/prompts/v3_change2_codex.md`

## Global Constraints

- Full acquisition-supported canonical FOV and hard-QC policy remain unchanged.
- No local replacement of upstream NeSVoR/SINR/FiLM primitives.
- Every production behavior is introduced through a failing deterministic CPU test.

### Task 1: Frequency source interface

- [ ] Add failing parser/path tests for real Phase-1 aggregate JSON and CLI precedence.
- [ ] Implement validated `TrainingFrequencyPrior`; reject absent verified respiratory prior unless explicit fallback.
- [ ] Run parser tests.

### Task 2: Motion tensor/loss contracts

- [ ] Add failing topology and Eq.6–9 synthetic tests.
- [ ] Change adapter to one 3-vector MBC per level; implement DREME normalization, ZMS and nonuniform-frequency losses.
- [ ] Run motion-loss and SINR adapter tests.

### Task 3: Runtime wiring

- [ ] Add failing config-to-runtime, source-lock and full-span temporal tests.
- [ ] Build model from config; verify source hashes; save effective config; use stage-specific module learning rates and regularization grids.
- [ ] Run unified smoke tests.

### Task 4: Regression and handoff

- [ ] Run specified targeted suite then `unittest discover`.
- [ ] Update provenance/report/README/changelog and commit.
