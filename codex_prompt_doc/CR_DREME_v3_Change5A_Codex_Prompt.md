# Codex Command Prompt — CR_DREME_v3 Change5A: Cardiac Target-Band Spectral Concentration

## 0. Task and scope

Work autonomously in:
- repo: `/home/universe/SVR/code/CR_DREME_v3`
- branch: `dev/cardioresp4d`
- expected baseline commit: `9252386800a2b948650544486043f031a17adef5`
- GitHub: `universe26dyz/CR_DREME_v3`

This is **Change5A only**.

Goal: add a **cardiac target-band spectral concentration loss** to the existing source-first DREME training path, using the **already available Phase-1 per-location PCA/PSD cardiac frequency prior**.

Do **NOT** implement Change5B / PCA temporal-waveform supervision or PCA-based score initialization in this task. Do not add a temporal encoder. Do not modify model architecture, MBC basis, PSF, uncertainty, canonical INR, sampling, Stage3 schedule, Phase-1 PCA algorithm, or DREME Eq.8/Eq.9 definitions.

This task is an image-domain identifiability adaptation motivated by the real Change4 result:
- Stage3b reconstruction/card branch became stronger;
- cardiac score dominant frequencies still remained mainly in low-frequency nuisance/respiratory-like regions rather than local cardiac bands;
- Eq.8/Eq.9 are negative crossover suppression only and do not force cardiac score energy into the cardiac target band.

No real-data training locally. Only code, tests, synthetic/CPU checks, docs/config audit, commit, and push attempt.

## 1. Safety / repo handling

Before edits:

```bash
cd /home/universe/SVR/code/CR_DREME_v3
git status --short
git rev-parse HEAD
git branch --show-current
```

Requirements:
- expected branch `dev/cardioresp4d`;
- expected baseline `9252386800a2b948650544486043f031a17adef5`;
- if there are unexpected local modifications, do not delete/reset blindly; inspect and report;
- do not delete old files/results;
- do not modify `third_party/NeSVoR`, `third_party/SINR`, or `third_party/film`;
- retain checkpoint backend-preservation logic introduced by commit `9252386`;
- preserve backward compatibility with existing Change4 checkpoints.

Use TDD: write failing tests first, run them and confirm the expected failure, then implement the minimum production code, then rerun focused + full tests.

## 2. Existing code paths to inspect first

Read before editing:

```text
src/cardioresp4d/losses/frequency_loss.py
src/cardioresp4d/frequency/training_prior.py
src/cardioresp4d/training/trainer.py
src/cardioresp4d/training/source_first_config.py
src/cardioresp4d/models/film_motion_encoder.py
scripts/diagnose_change4_checkpoint.py
configs/source_first.yaml

tests/test_dreme_frequency_losses.py
tests/test_frequency_training_prior.py
tests/test_diagnose_change4_checkpoint.py
```

Current contracts:
- Phase-1 frequency evidence is resolved per `view/slice_id`;
- reliable local candidate band is `frequency_hz ± df_hz/2`;
- trainer obtains it via `frequency_prior.for_location(view, slice_id)`;
- cardiac scores shape `[T,1,3]`;
- respiratory scores shape `[T,active_levels,3]`;
- existing NUDFT uses true timestamps, relative time, score mean removal, complex coefficients, `/N` normalization;
- keep DREME Eq.8 and Eq.9 unchanged.

## 3. Exact Change5A mathematical contract

For one fixed acquisition location `ell`, cardiac score channels are:

`w_q^c(t_n), q=1..Q, Q=3`.

Use the existing mean-centred true-timestamp NUDFT:

`W_q(f) = (1/N) * sum_n [(w_q^c(t_n)-mean(w_q^c)) * exp(-i 2π f (t_n-t_1))]`

Per-frequency power:

`S(f) = sum_q |W_q(f)|^2`

The **local target band** must come from:

```python
prior = frequency_prior.for_location(view, slice_id)
prior.cardiac_bands_hz
```

Do not use the global cardiac union when reliable local prior exists.

Define a resolution-aware non-DC grid `Omega` from actual timestamps:
- duration `T = max(t)-min(t)` > 0;
- frequency resolution `df = 1/T`;
- Nyquist = `0.5 / median(diff(sorted timestamps))`;
- grid contains positive frequencies approximately `k*df`, starts at `df`, excludes DC, never exceeds Nyquist;
- do not use the 512-point dense diagnostic grid for training.

Then:

`P_target = sum_{f in Omega_target} S(f)`

`P_total = sum_{f in Omega} S(f)`

`rho_c = P_target / (P_total + epsilon)`

`L_card_conc = 1 - rho_c`

Semantics:
- sum/aggregate power across all cardiac score channels **before** the ratio;
- do not calculate one ratio per x/y/z channel and average;
- do not maximize raw target power;
- ratio must be invariant to common score-amplitude scaling;
- epsilon is denominator-only;
- do NOT use `(P_target+eps)/(P_total+eps)`;
- all-zero scores must be finite and yield loss near 1;
- target frequencies must belong to the denominator grid so `rho <= 1` except tiny floating error.

### Target-grid edge rule

Phase-1 uses approximately `fs/N`; Change5A grid uses span-based `1/T`, so slight mismatch is expected.

Rule:
1. select all evaluation-grid frequencies whose centers fall inside any local target band;
2. if a valid target band overlaps `(0, Nyquist]` but contains no grid centre only because of discretization mismatch, select the **single nearest grid frequency to that band centre**;
3. document this as a project adaptation;
4. malformed/non-finite/out-of-range bands or impossible grid => fail fast.

No harmonics in Change5A v1. Use only existing local fundamental band(s).

## 4. Implementation requirements

### A. `frequency_loss.py`

Add small independently testable helpers for:
1. resolution-aware non-DC frequency grid;
2. target mask/index resolution on the same grid;
3. cardiac target-band concentration.

Expose enough values for training/diagnostic audit:
- loss;
- target fraction `rho`;
- target power;
- total power;
- optionally resolved grid/target frequencies.

Do not alter:
- `nonuniform_dft_at_frequencies`;
- DREME Eq.8;
- DREME Eq.9.

### B. `trainer.py`

Add backward-compatible loss-weight key:

```text
cardiac_target_concentration
```

Existing configs without it must behave as `0.0`.

In `_temporal_components(stage)`:
- keep one fixed-location temporal sequence;
- resolve local prior with `for_location(...)`;
- only when `contract.enable_cardiac`, compute concentration using `prior.cardiac_bands_hz`;
- expose at least:
  - `cardiac_target_concentration`
  - `cardiac_target_fraction`
  - `cardiac_target_power`
  - `cardiac_total_power`

Stage1/Stage2 must not receive this cardiac penalty.

In total loss, only under `contract.enable_cardiac`:

```python
total += self.loss_weights["cardiac_target_concentration"] * components["cardiac_target_concentration"]
```

Do not replace Eq.8/Eq.9.

### C. Configs

Prefer preserving `configs/source_first.yaml` as the Change4 baseline config.

Create:

```text
configs/source_first_change5a.yaml
```

identical to current source-first config except the new Change5A weight/comment.

Initial weight:

```yaml
cardiac_target_concentration: 0.001
```

Reason: concentration is dimensionless O(1); `1e-3` should be meaningful but not dominate current Stage3 MSE O(1e-2).

Requirements:
- missing key => 0;
- key must be finite and >=0;
- no template frequency fallback;
- training report/effective config must record actual loss weights for auditability.

### D. Diagnostic

Keep `scripts/diagnose_change4_checkpoint.py` compatible with Change4 and Change5A checkpoints.

Extend existing frequency-semantic output with:
- per-record cardiac target fraction/concentration;
- aggregate mean/median if consistent with current JSON schema.

Do not remove:
- cardiac ablation;
- target/wrong amplitudes;
- dominant peaks;
- backend provenance.

Remain read-only.

### E. Documentation/provenance

Comments/docs must state:

```text
DREME Eq.8/9:
source-derived negative crossover suppression.

Change5A cardiac target-band concentration:
project-specific image-domain adaptation added after Change4 showed that
cardiac scores could satisfy Eq.9 while escaping into other low-frequency
nuisance bands.
```

Do not attribute the new concentration loss to DREME-MR or S2V-DREME.

## 5. TDD / required tests

Write tests first and observe expected RED failure.

At minimum cover:

1. **Target vs off-target sinusoid**
   - deterministic ~50-frame sequence;
   - target frequency aligned with grid;
   - target signal => higher `rho`, lower loss than off-target low-frequency signal.

2. **Scale invariance**
   - `scores` and `7*scores` concentration equal within tolerance.

3. **Channel aggregation**
   - multiple cardiac channels;
   - aggregate powers before ratio, not mean of channel-wise ratios.

4. **Zero scores**
   - finite;
   - loss ~1;
   - no NaN/Inf.

5. **Gradient**
   - `requires_grad=True`;
   - backward succeeds;
   - finite gradient for nonzero mixed-frequency signal.

6. **No DC**
   - grid starts at positive `df`.

7. **Nyquist**
   - grid never exceeds timestamp-derived Nyquist.

8. **Discretization fallback**
   - valid narrow band with no exact grid center gets nearest grid point;
   - malformed/out-of-range target fails clearly.

9. Existing Change4 config without key => zero weight / unchanged behavior.

10. Stage2 does not apply cardiac concentration.

11. Stage3a/Stage3b with positive weight include concentration and preserve gradient ownership.

12. Per-location prior: two synthetic locations with distinct local cardiac bands resolve different targets; no accidental global union.

13. Diagnostic remains read-only and reports concentration fields.

14. `source_first_change5a.yaml` validates.

15. Negative/NaN concentration weight rejected if config validation is extended.

Do not weaken existing tests.

## 6. Verification

Run focused then full CPU tests:

```bash
pytest -q tests/test_dreme_frequency_losses.py
pytest -q tests/test_frequency_training_prior.py
pytest -q tests/test_diagnose_change4_checkpoint.py
pytest -q
pip check
```

Also run the repo's existing source import/source-lock verification.

No real DYL0709 training locally.
No long GPU experiment.
No Stage3c.

CUDA-only tests on CPU machine must be reported as explicit skips, not passes.

## 7. Real experiment contract to document, NOT execute locally

Change5A server ablation must start from the **same Change4 Stage2c checkpoint** used before Change4 Stage3a.

Do NOT start from the already trained Change4 Stage3a/Stage3b checkpoint, otherwise extra Stage3 steps confound the comparison.

Formal comparison:

```text
same Stage2c checkpoint
same seed
same pixel_samples
same data/QC/domain/frequency prior
same Stage3a 100 steps
same Stage3b 100 steps
Stage3c = 0

only intentional difference:
cardiac_target_concentration weight 0 vs 0.001
```

Change4 Stage3b reference:
- cardiac target/wrong mean ratio ≈ 0.56;
- cardiac target > wrong at 0/9 representative locations;
- resp/card identical dominant peak at 5/9 locations;
- aggregate cardiac ablation MSE gain ≈ 2.645%;
- cardiac branch numerically active/stable but cardiac frequency semantics failed.

Desired Change5A evidence:
- cardiac target/wrong ratio clearly increases, ideally >1;
- majority representative locations become target > wrong;
- cardiac peaks move toward each location's local Phase-1 cardiac band;
- resp/card same-low-frequency peak count decreases;
- reconstruction/card ablation benefit remains positive and does not collapse;
- no order-of-magnitude DVF instability.

Do not hard-code these acceptance values into the loss.

## 8. Explicit non-goals

Do NOT:
- implement PCA waveform/correlation supervision;
- initialize FiLM/MBC scores from PCA curves;
- align frame indices across slices;
- assume frame 1 across slices is same physiological phase;
- create global phase labels;
- add harmonics;
- add RNN/Transformer/temporal encoder;
- change SINR/MBC basis;
- change Stage3a/3b trainable-module ownership;
- enable uncertainty/Stage3c;
- change Phase1 PCA search bands;
- alter Eq.8/Eq.9;
- use `strict=False`;
- alter canonical encoding backend behavior;
- modify vendored upstream source.

Change5B is a separate future ablation if Change5A is insufficient.

## 9. Completion report

At completion report:
1. `git status --short`
2. changed files
3. concise mathematical implementation description
4. exact config key + weight
5. RED tests and observed pre-implementation failure
6. focused tests
7. full tests (`passed / explicit skipped / failed`)
8. `pip check`
9. source-lock/import verification
10. confirmation that model architecture, Eq.8/Eq.9, stage ownership, uncertainty, PSF, SINR, and third-party source were unchanged
11. commit SHA
12. push result

Suggested commit:

```text
feat: add cardiac target-band concentration loss
```

If push fails due SSH/network, keep local commit and report the exact SHA; do not rewrite history.
