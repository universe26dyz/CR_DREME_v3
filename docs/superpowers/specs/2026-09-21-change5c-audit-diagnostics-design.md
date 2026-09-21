# Change5C audit, diagnostics, and visualization design

## Purpose and constraints

Make existing Stage3a evidence auditable and interpretable without changing the
scientific model, training schedule, or vendored NeSVoR/SINR sources.  Work is
CPU-only locally and must never claim real-data or GPU results.  There is no
global cross-location cardiac phase.

## Gate and audit

First audit Change4, Change5A, Change5B, and the current checkpoint diagnostic
against the stated pullback, loss, timestamp, data-validity, and Stage3a
ownership contracts.  The audit report cites source locations and classifies
each item as PASS, PASS WITH DOCUMENTED LIMITATION, BUG, or AMBIGUOUS/REQUIRES
SCIENTIFIC DECISION.  A material bug produces a blocker report plus a minimal
CPU reproducer, and stops all downstream diagnostics and visualization work.

## Diagnostics

When the gate passes, extend the checkpoint diagnostic with an opt-in
`--all-locations` path.  It will preserve existing default output and legacy
probe-only motion metrics, while recording explicitly named aggregate motion
metrics, eligibility/skip reasons, per-location semantic fields, and grouped
summaries.  A separate comparison CLI intersects location keys across labeled
JSON results and reports paired deltas and boolean transitions without
statistical claims.

## Gradient audit and visualization

Add a read-only Stage3a gradient audit that independently evaluates losses and
reports raw/configured-weighted gradients only on trainable cardiac modules.
Add a separate read-only visualization CLI and isolated helpers for
reprojections, observation-conditioned implied 3D dynamics, component/total
DVFs, pullback Jacobians, and score/PCA/spectral plots.  It uses fixed grids,
chunking, and deterministic sampling/seed controls; output labels explicitly
avoid claiming globally synchronized physiological cine.

## Tests and documentation

Add deterministic CPU tests for Stage3a parameter updates, spectral and PCA
invariances/degenerate cases, all-location eligibility and comparison, motion
and Jacobian helpers, and visualization coordinate/identity contracts.  Add a
server-only run guide containing exact commands but do not access server data
locally.  Update provenance documents only to distinguish source-derived,
project-specific, diagnostic-only, and visualization-only behavior.

## Delivery and validation

Run the project test suite, compile checks, documented source-lock/import
checks, and whitespace validation once changes are stable.  Preserve existing
untracked files and history.  Commit/push only after a passing audit gate and
successful local validation; otherwise report the blocker without a misleading
feature-complete commit.
