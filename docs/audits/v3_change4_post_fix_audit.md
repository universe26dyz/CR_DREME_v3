# v3_change4 Post-fix Audit

Baseline: `0c0de21`; branch: `dev/cardioresp4d`.

## Closure

- Per-location respiratory candidates now use the same Phase-1
  `slice_key=view/slice_id` identity as cardiac candidates. Reliable `f,df`
  resolves to `[f-df/2,f+df/2]`; malformed/duplicate/DC candidates fail.
  Respiratory and cardiac fall back independently to global evidence and the
  report serializes both source labels.
- Trainer Eq.9 passes `prior.respiratory_bands_hz`, not the global prior.
  Local baseline pairs use local respiratory occupancy.
- `StageContract` is the single source for execution, MSE/NLL choice, cardiac
  and uncertainty availability, FiLM train mode and module ownership. Stage3a
  trains only `film_encoder.card` and cardiac MBC; Stage3b enables joint
  canonical/full-FiLM/resp/card MSE training; Stage3c alone enables uncertainty
  and Gaussian NLL.
- A Stage2c checkpoint reconstructs the retained Stage1–Stage2c optimizer
  groups and advances to Stage3a. Legacy `current_stage=stage3` fails with the
  required explicit migration message.
- Formalized adapter/runtime hotfixes: device-safe SINR spacing, `.to(device)`
  preflight model placement, CPU `torch.load(..., weights_only=False)` before
  RNG restoration, and `dataclasses.asdict(prior)` JSON/checkpoint payloads.

## Explicitly unchanged

NeSVoR/SINR/FiLM vendored source and source lock; full acquisition FOV;
local-only cardiac box; hard-invalid policy; NUDFT `/N`; Eq.8/Eq.9 weights;
no positive target-band loss; uncertainty architecture; no Phase1 rerun and no
real-data training.

## Verification actually run locally (`knesvr_torch`, CPU)

- Targeted change4 frequency/schedule/resume/config/smoke regressions: 20 OK.
- Full `python -m unittest discover -s tests`: **135 OK, 2 explicit skips**.
- `pip check`: `No broken requirements found`.
- source dependency/import/lock check: `SOURCE_IMPORTS_AND_LOCK=PASS`.
- CUDA regression: explicitly skipped because local `torch.cuda.is_available()`
  is false; no GPU result is claimed.

The optional `py_compile` aggregate was blocked only when Python attempted to
write bytecode into the read-only `scripts/__pycache__`; the scripts had already
been imported/executed by the regression suite.

## Server protocol

Do not run Stage3c automatically. Resume the existing Stage2c checkpoint for
100 Stage3a steps, inspect the read-only diagnostic, then resume the Stage3a
checkpoint for 100 Stage3b steps. Compare finite loss, respiratory/cardiac DVF,
same-checkpoint cardiac ablation, and resolved-local frequency metrics before
considering a later uncertainty-only Stage3c run.
