# Scientific audit: Change4, Change5A, and Change5B

Audit date: 2026-09-21. Starting scientific commit: `8fde85b`; audit was
performed against the current descendant `6f6e005`.  This report is source and
CPU-test evidence only; it makes no claim about GPU or real-subject outcomes.

## Verdict

No material scientific blocker was found.  The implementation may proceed with
read-only diagnostics and visualization.  The limitations below are explicitly
preserved or corrected by newly named outputs rather than reinterpretation of
historical fields.

| Area | Classification | Evidence |
| --- | --- | --- |
| World-mm/DICOM geometry and PSF sampling | PASS | `training/model.py:73-77` maps DICOM u/row with column spacing and v/column with row spacing; `models/film_motion_encoder.py:65-76` and `adapters/nesvor_psf.py:29-49` use the same physical order and orthonormal world basis. |
| Observation-to-reference motion | PASS | `models/cardioresp_motion.py:45-61` implements `y + d_c(y) + d_r(y+d_c(y))`; PSF samples are pulled back before the canonical query in `adapters/nesvor_psf.py:46-50`. |
| Score/MBC contract and reconstruction gradient | PASS | `models/cardioresp_motion.py:14-46` performs score-weighted `[level, xyz]` summation without score detachment.  `training/model.py:55-67` constructs exactly one sequential motion application. |
| Local cardiac field | PASS | Box containment is checked in `training/model.py:32-38`; tapered local cardiac field is implemented in `adapters/sinr_mbc.py:122-137`. |
| Stage3a ownership | PASS | `training/stage_contract.py:57-61` assigns only cardiac MBC plus cardiac FiLM head; `training/trainer.py:31-75` resets `requires_grad` before keeping/addition of Adam groups.  New numerical test `tests/test_v3_change4_stage_schedule.py:65-84` proves an optimizer step after Stage2c leaves canonical, respiratory MBC and shared FiLM unchanged while the two permitted groups change. |
| MBC normalization and zero-mean score | PASS WITH DOCUMENTED LIMITATION | `losses/motion_loss.py:6-33` preserves level×xyz and reduces batch/spatial axes; the fixed-grid discrete physical-domain implementation is an explicitly documented SINR-hybrid adaptation, not a literal continuous integral. |
| Eq.8-like respiratory score cardiac leakage | PASS WITH DOCUMENTED LIMITATION | `losses/frequency_loss.py:108-164` mean-centres true-timestamp NUDFT and uses complex paired-band subtraction; `training/trainer.py:181-187` applies it to respiratory scores.  Local source papers are documented in provenance but their Eq.8 text is not independently re-quoted here. |
| Eq.9-like cardiac score respiratory leakage | PASS WITH DOCUMENTED LIMITATION | `losses/frequency_loss.py:167-169` evaluates respiratory bands; `training/trainer.py:181-186` applies it to cardiac scores using `for_location` respiratory bands. Same local-paper limitation as Eq.8. |
| Change5A target concentration | PASS | `losses/frequency_loss.py:79-105` mean-centres, excludes DC, uses a timestamp-derived non-DC grid and aggregates all channels before target/total. Existing CPU tests cover target/off-target, scaling, zero scores, grid/Nyquist and nearest centre in `tests/test_change5a_cardiac_concentration.py:39-92`. It is correctly labeled project-specific in `SOURCE_PROVENANCE.md`. |
| Change5B PCA selection/matching | PASS WITH DOCUMENTED LIMITATION | `frequency/pca_waveform_prior.py:39-79` converts one-based `selected_pc` exactly once and loads only reliable `(view, slice_id)` artifacts; `:81-112` preserves requested valid-frame order, rejects unmatched timestamps and does not interpolate. Diagnostic loading uses `strict=False` at `scripts/diagnose_change4_checkpoint.py:104-107`, so malformed/missing reliable artifacts are omitted rather than stopping the diagnostic; metadata exposes available vs reliable counts. |
| Change5B subspace loss | PASS | `losses/motion_loss.py:36-69` detaches only the target, retains score gradients, uses differentiable ridge solve and corr², and returns finite connected-zero skips for too-short/nonfinite/constant/zero-prediction cases. `training/trainer.py:188-195` confines it to Stage3a. |
| Existing checkpoint diagnostic | PASS WITH DOCUMENTED LIMITATION | `scripts/diagnose_change4_checkpoint.py:112-116` computes `motion_statistics` from `valid[0]` only; this is probe-specific historical output. Cardiac ablation uses up to ten deterministic frame representatives per view and cardiac-box pixels (`:117-133`); frequency semantics uses three representative locations/view and silently requires exactly 50 valid frames (`:134-151`). These are not whole-dataset metrics and will be retained while `--all-locations` exposes explicit population/skip data. `prepare_read_only_diagnostic_model` uses `no_grad` and does no optimizer step (`:34-43`, `:112-163`), although canonical INR is put in train mode solely to expose pinned latent features. |

## CPU evidence

Focused audit run in `knesvr_torch`: 26 existing tests passed.  After adding
the numerical ownership regression, `tests/test_v3_change4_stage_schedule.py`
passed 4/4.  The expanded test suite for diagnostic additions is recorded with
the final implementation validation.

## Non-blocking limitations carried forward

- Frequency metrics derived from irregular samples use the documented
  equal-observation-weight NUDFT and median-spacing Nyquist approximation.
- PCA weak supervision is an image-derived per-location surrogate, not ECG;
  unavailable or invalid reliable artifacts produce no PCA term and no global
  substitute.
- Historical motion statistics and frequency semantics have limited
  representative populations; their names/output remain compatible while new
  aggregate fields state the evaluated population.
