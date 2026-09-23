# Codex Command Prompt — CR_DREME_v3 Change5B: per-location PCA cardiac waveform weak supervision

## 0. Execution mode and hard environment constraints

You are working **locally**, not on the GPU server.

Local repository:

```text
/home/universe/SVR/code/CR_DREME_v3
```

Expected branch:

```text
dev/cardioresp4d
```

Local environment is normally:

```text
conda env: knesvr_torch
```

**There is no usable GPU locally.**

Your job in this task is:

1. inspect the current local repository and current HEAD;
2. implement Change5B completely;
3. add/modify CPU-safe tests;
4. run only local CPU/static/unit tests;
5. update concise provenance/implementation documentation;
6. create exact **server GPU run instructions**, but **do not run them locally**;
7. commit the completed implementation;
8. attempt to push to `origin dev/cardioresp4d`;
9. report starting HEAD, ending commit, files changed, tests, and any push failure.

Do **not** attempt CUDA training, real-data training, long reconstruction, or server filesystem access from the local Codex environment.

Do not delete old files. Do not reset/discard unrelated user changes. Do not modify vendored/third-party source unless absolutely unavoidable; for this task it should not be necessary.

Work autonomously. Do not repeatedly ask for confirmation. If an implementation detail differs from the summary below, inspect the actual source and make the smallest compatible change while preserving the scientific contract.

---

# 1. Project and scientific context

Project:

```text
CR_DREME_v3 / CardioResp-4D
```

Goal: reconstruct 3D+time cardiac MRI from free-breathing real-time 2D cine MRI using image-domain observations only, with explicit respiratory/cardiac motion decomposition.

Current motion form:

\[
d(x,t)=\sum_i w_i(t)e_i(x)
\]

Current source-first schedule:

```text
Stage1  canonical only
Stage2a respiratory coarse
Stage2b respiratory middle
Stage2c respiratory fine
Stage3a cardiac warm-up
Stage3b joint cardiorespiratory refinement
Stage3c uncertainty
```

For this task we remain at **Stage3a**. Do not implement or run Stage3b/Stage3c experiments.

Current Stage3a contract must remain:

```text
canonical frozen
respiratory MBC frozen
shared FiLM frozen
only cardiac FiLM head + cardiac MBC trainable
MSE
no uncertainty
```

Do not change:
- model architecture;
- Eq.8 / Eq.9 negative crossover losses;
- MBC construction;
- PSF;
- canonical INR;
- source locks;
- Stage3 ownership;
- third-party implementations.

---

# 2. Why Change5B is now needed

Change5A added positive cardiac target-band concentration:

\[
L_{card-conc}=1-\frac{P_{target}}{P_{total}+\epsilon}
\]

using each fixed location's Phase1 cardiac frequency band.

A fair convergence probe used the same Stage2c checkpoint and `cardiac_target_concentration=0.003`.

Observed:

```text
w003 @ 100 steps:
target_fraction_mean      0.083865
target_fraction_median    0.057428
card_target/wrong         0.841676
card_target>wrong         4/9
same_peak_exact           7/9
same_peak_within_0.02Hz   7/9
cardiac_ablation_gain     +0.0337 %
card_dvf_rms_mm           0.04958
card_dvf_max_mm           0.22382
card_score_std            0.14548

w003 @ 1000 steps:
target_fraction_mean      0.086123
target_fraction_median    0.051400
card_target/wrong         0.857539
card_target>wrong         4/9
same_peak_exact           6/9
same_peak_within_0.02Hz   6/9
cardiac_ablation_gain     +2.0099 %
card_dvf_rms_mm           0.22814
card_dvf_max_mm           1.36361
card_score_std            0.43232
```

Interpretation:

- 100 steps was under-trained for reconstruction usefulness.
- By 1000 steps the cardiac branch clearly becomes reconstruction-useful.
- However, cardiac physiological frequency semantics barely improve.
- More iterations alone therefore do not resolve the current identifiability problem.
- Change5B should provide a **weak, image-derived, per-location cardiac temporal waveform prior**.

Change5B is a project-specific adaptation. Do not attribute it to DREME-MR/S2V-DREME as an original component.

---

# 3. Phase1 PCA evidence already available — do NOT rerun or redesign Phase1

Phase1 path on the server later will be:

```text
/data/dengyz/dataset/CR_DREME_v3/v1/phase1
```

The existing Phase1 output already contains, for every fixed location:

```text
frequency/<view>/<slice_id>/pca_psd.npz
frequency/<view>/<slice_id>/temporal_pc.csv
```

`pca_psd.npz` contains:

```text
timestamps_s       shape (50,)
temporal_pcs       shape (50, 50)
explained_variance shape (50,)
frequencies_hz     shape (26,)
pc_psd             shape (26, 50)
```

`frequency_bands.json` contains per-location respiratory and cardiac candidates.

Current cardiac candidate audit:

```text
n_candidates        = 143
n_reliable          = 143
selected_pc counts  = {1: 26, 2: 100, 3: 16, 4: 1}
selected_pc median  = 2
dominance median    = 124.41737484796191
dominance min/max   = 9.232567961788893 / 1953.8023423184766

selected-PC explained variance:
median = 0.16332629868031776
min    = 0.07308944760994954
max    = 0.39281481173969063
```

The candidate selector is already defined in:

```text
src/cardioresp4d/frequency/pca_motion.py
```

and uses:

```python
band_powers = pc_psd[mask, :]
best_flat_index = int(np.argmax(band_powers))
frequency_index, pc_index = np.unravel_index(best_flat_index, band_powers.shape)
...
"selected_pc": int(pc_index + 1)
```

Therefore `selected_pc` is explicitly **1-based**.

For location \(l\), Change5B cardiac PCA surrogate must use exactly:

\[
c_l(t)=
\texttt{temporal\_pcs}[:,\,\texttt{selected\_pc}-1]
\]

from that same location.

Do **not** invent another PCA selection rule.
Do **not** use a global PCA waveform.
Do **not** align the same frame index across different slices.

---

# 4. Critical physiological constraint: supervision is strictly per fixed location

Different slice frame indices are not shared physiological phase.

For example:

```text
SAX_s001 frame 10
!= SAX_s020 frame 10
!= 2CH_s010 frame 10
```

because each fixed slice is acquired as its own sequential ~50-frame series.

Therefore Change5B supervision must be:

```text
per fixed view/slice_id
using that location's own timestamps
using that location's own selected cardiac PCA PC
```

Never create:
- a global cardiac phase label;
- cross-slice frame-index alignment;
- shared waveform values across locations.

---

# 5. Hard QC must remain authoritative

Current hard-QC rule:

Only these can invalidate an observation:

```text
slice_local_scale_absolute
manual_exclusion
```

Hard-invalid observations must never re-enter likelihood or Change5B supervision.

The PCA artifact may contain all original 50 timestamps. During training, the Change5B loss must be computed only on the currently valid observations in the sampled fixed-location temporal sequence.

Required alignment concept:

```text
valid training sequence item timestamp
        ↕
same location Phase1 PCA timestamp
        ↓
selected-PC waveform value
```

Do not silently supervise excluded observations.

Do not interpolate across different locations.

Prefer strict timestamp matching against the Phase1 timestamps using the same absolute `timestamp_s` convention already used by the trainer. Inspect the actual observation object and Phase1 generation code before choosing a small tolerance. A mismatch must raise a clear error or skip with an explicit counted diagnostic; it must never silently pair the wrong frame.

---

# 6. Existing source locations to inspect first

Relevant current files include at least:

```text
src/cardioresp4d/frequency/pca_motion.py
src/cardioresp4d/frequency/frequency_bands.py
src/cardioresp4d/frequency/training_prior.py

src/cardioresp4d/models/film_motion_encoder.py
src/cardioresp4d/training/model.py
src/cardioresp4d/training/stage_contract.py
src/cardioresp4d/training/source_first_config.py
src/cardioresp4d/training/trainer.py
src/cardioresp4d/losses/frequency_loss.py
src/cardioresp4d/losses/motion_loss.py

scripts/train_source_first.py
scripts/diagnose_change4_checkpoint.py

configs/source_first_change5a.yaml
tests/
README.md
SOURCE_PROVENANCE.md
IMPLEMENTATION_REPORT.md
```

Known current behavior from source audit:

```text
GeometryFiLMMotionEncoder:
resp_scores [B,3,3]
card_scores [B,1,3]

trainer.py:
- temporal auxiliary already builds one-location temporal sequences
- scores are computed for sequence items
- card scores are concatenated over time
- Change5A cardiac_target_concentration is already computed there
```

Likely existing code region in `trainer.py` around the temporal components:

```python
scores = [
    self.model.film_encoder(...)
    for item in sequence
]

resp = torch.cat(...)
card = torch.cat(...)
```

Reuse this temporal path. Do not build an unnecessary second motion encoder or temporal network.

Before editing, inspect the complete surrounding functions and sampler behavior rather than relying only on this summary.

---

# 7. Change5B mathematical objective

## 7.1 Do NOT supervise a fixed cardiac score component directly

The three cardiac scores span a latent temporal subspace:

\[
W_l\in\mathbb{R}^{T\times3}
\]

where the current tensor is expected to come from:

```text
card_scores [T,1,3] → reshape/squeeze → [T,3]
```

The scalar PCA waveform is:

\[
c_l\in\mathbb{R}^{T}
\]

There is no scientific reason to assume PCA cardiac PC maps specifically to score 1, 2, or 3.

Also do not introduce one global learnable projection shared by all slices.

## 7.2 Implement differentiable best linear projection / multiple correlation

For each fixed location sequence, center the current cardiac score matrix and target waveform over time:

\[
X=W_l-\bar W_l
\]

\[
y=c_l-\bar c_l
\]

Estimate the best current linear projection of the 3-D cardiac score space onto the scalar PCA waveform using a tiny differentiable ridge solve:

\[
G=\frac{X^\top X}{T}
\]

\[
b=\frac{X^\top y}{T}
\]

Use a scale-aware ridge for numerical stability:

\[
\lambda_{eff}
=
\lambda\frac{\mathrm{tr}(G)}{3}
+\epsilon
\]

\[
a^*=(G+\lambda_{eff}I)^{-1}b
\]

\[
\hat y=Xa^*
\]

Then compute:

\[
R_l^2=
\mathrm{corr}^2(\hat y,y)
\]

and:

\[
L_{\mathrm{PCA-card}}^{(l)}=1-R_l^2
\]

Clamp \(R^2\) numerically to `[0,1]`.

Desired properties:
- differentiable through the ridge solve;
- no learnable projection parameters;
- sign-invariant because of `corr²`;
- scale-insensitive at the correlation level;
- robust to permutation / rotation of the 3-D cardiac score basis;
- zero/constant score inputs remain finite and yield no false perfect correlation.

Use `torch.linalg.solve` or an equivalently stable differentiable 3×3 solve.

Do not detach `card_scores`.

The PCA waveform is fixed supervision and should not require gradients.

---

# 8. Implement the loss as a reusable unit-tested function

Add a clearly named loss/helper in the appropriate loss module, for example:

```text
cardiac_pca_waveform_subspace_loss(...)
```

Exact naming can follow repository conventions.

Suggested inputs:

```text
card_scores          [T,3] or [T,1,3] with explicit normalization
target_waveform      [T]
ridge
eps
```

Suggested returned diagnostics:

```text
loss = 1 - R²
r2
```

Optional useful diagnostics:

```text
matched_frame_count
target_std
prediction_std
```

but do not bloat the API.

Required edge cases:
- fewer than the minimum usable number of frames;
- constant/near-constant target;
- constant/near-constant card scores;
- non-finite inputs;
- singular / rank-deficient score covariance.

All must behave deterministically and remain finite. Prefer a neutral zero-gradient/explicitly skipped supervision path when the input is not scientifically usable, with the skip visible in diagnostics.

A practical minimum such as 8 matched valid frames is acceptable; expose it as a small explicit config/constant and test it.

---

# 9. Add a PCA waveform prior loader — load once, never per training step

Implement a lightweight provider keyed by canonical location key:

```text
view/slice_id
```

Use the same `frequency_bands.json` already passed to training.

The expected server layout is:

```text
.../phase1/frequency/frequency_bands.json
.../phase1/frequency/SAX/SAX_s001/pca_psd.npz
.../phase1/frequency/2CH/2CH_s001/pca_psd.npz
.../phase1/frequency/4CH/4CH_s001/pca_psd.npz
...
```

Therefore the loader may derive the PCA artifact root from:

```text
Path(frequency_bands_json).parent
```

unless the actual current code architecture strongly favors an explicit path.

Requirements:

1. Parse `cardiac.per_slice_candidates`.
2. For a location, only use a candidate that is:
   - `reliable == true`;
   - has valid `selected_pc`;
   - has matching `slice_key`.
3. Load:
   - `timestamps_s`;
   - the one selected cardiac waveform only;
   - any minimal provenance needed.
4. Convert 1-based `selected_pc` to zero-based array index exactly once.
5. Validate shape and finite values.
6. Cache all needed arrays in memory at initialization.
7. Do not `np.load()` every training step.
8. If a candidate is missing/unreliable, do **not** substitute a global waveform. Mark that location unavailable for Change5B supervision.
9. Existing dataset has 143/143 reliable candidates, but code must still be robust.

Use numeric `.npz` loading without unsafe pickle unless actually required.

Add compact metadata for reproducibility, e.g.:

```text
available location count
reliable supervised location count
source frequency_bands path
selected-PC convention = 1-based in JSON
```

---

# 10. Integrate Change5B into the existing temporal auxiliary path

Do not create a second training loop.

The trainer already has:

```text
temporal_every
temporal_batch_size
cardiac_sampling_fraction
frequency_prior
```

and already computes temporal score sequences for Change5A.

Extend that existing per-location temporal component path.

For a sampled fixed-location sequence:

1. identify canonical `view/slice_id`;
2. obtain current valid sequence timestamps;
3. obtain current `card_scores`;
4. query that location's PCA waveform prior;
5. match current valid timestamps to the Phase1 PCA timestamps;
6. select only matched valid frames;
7. compute `L_PCA-card`;
8. return it as a temporal loss component.

Suggested component keys:

```text
cardiac_pca_waveform
cardiac_pca_waveform_r2
cardiac_pca_waveform_matched_frames
```

Do not mix locations in a single correlation calculation.

The Change5B loss must not be computed from the ordinary random pixel batch if that batch contains unrelated locations. It belongs to the existing fixed-location temporal sequence path.

---

# 11. New config contract

Add a new loss weight with backward-compatible default:

```text
cardiac_pca_waveform: 0.0
```

Validation:

```text
finite
>= 0
```

Legacy configs must remain behaviorally unchanged.

Add a canonical Change5B config, derived from the current Change5A config, e.g.:

```text
configs/source_first_change5b.yaml
```

For the first Change5B experiment use:

```text
cardiac_target_concentration: 0.003
cardiac_pca_waveform:         0.001
```

Keep all other scientific settings identical to the current source-first/Change5A control unless a current file proves otherwise.

Expose the ridge and minimum-frame settings explicitly in a sensible existing config section, preferably under temporal auxiliary, for example:

```yaml
training:
  temporal_auxiliary:
    ...
    pca_waveform_ridge: 1.0e-4
    pca_waveform_min_frames: 8
```

Do not introduce unnecessary CLI flags if the existing config plumbing is cleaner.

---

# 12. Loss assembly and gradient ownership

The total Stage3a objective should extend the existing structure only by:

\[
+\lambda_{\mathrm{PCA}}
L_{\mathrm{PCA-card}}
\]

where the first experiment uses:

```text
lambda_PCA = 0.001
```

The PCA loss should directly constrain the cardiac FiLM scores.

Expected Stage3a gradient ownership must remain:

```text
canonical / INR           no train gradient
shared FiLM               frozen
respiratory MBC           frozen
cardiac FiLM head         trainable
cardiac MBC               trainable through existing reconstruction/motion losses
uncertainty               frozen/off
```

Do not unfreeze shared FiLM simply to make the PCA loss easier to optimize.

Do not add new trainable projection parameters.

---

# 13. Diagnostics must be extended for Change5B

Update:

```text
scripts/diagnose_change4_checkpoint.py
```

or add a minimally named successor only if cleaner.

For each per-location frequency semantics record already diagnosed, also compute the Change5B PCA subspace agreement using that same location's Phase1 selected cardiac PC.

Add per-location fields:

```text
cardiac_pca_waveform_r2
cardiac_pca_waveform_loss
cardiac_pca_matched_frames
selected_cardiac_pc
```

Add aggregate:

```text
summary_mean.cardiac_pca_waveform_r2
summary_median.cardiac_pca_waveform_r2
```

Do not average peak frequencies and call that physiological success.

The existing important diagnostics must remain intact:

```text
cardiac_target_fraction
card_target_amp
card_wrong_amp
card_peak_hz
resp_peak_hz
cardiac ablation
card DVF RMS/max
card score std
```

Change5B success will later be judged jointly, not from PCA correlation alone.

---

# 14. Optional but requested lightweight convergence trace

The current trainer/report only stores:

```text
loss_first
loss_last
loss_mean
loss_min
loss_max
```

and does not preserve an ordered step history.

Add a **small, low-overhead scalar trace** to the stage report without extra forward passes.

Record the already-computed current-step scalars every 50 stage steps (and final step), e.g.:

```text
stage_step
global_step
total_loss
data
cardiac_target_concentration
cardiac_target_fraction
cardiac_pca_waveform
cardiac_pca_waveform_r2
card_score_std   # only if already available without extra forward
card_dvf_rms_mm  # only if already available without expensive extra work
```

Do not add expensive deterministic render probes inside the training loop.

Do not print every step to terminal. Store a compact trace in the report.

Legacy behavior and numerical training must remain unchanged apart from the new Change5B weighted term when its weight is nonzero.

---

# 15. Tests — TDD and CPU only

Implement tests before/with code.

At minimum cover:

## A. PCA selected-PC loader

Synthetic temp `frequency_bands.json` + synthetic `pca_psd.npz`.

Verify:
- `selected_pc=1` maps to column 0;
- `selected_pc=2` maps to column 1;
- timestamp/waveform shape;
- unreliable candidate is unavailable, not globally substituted;
- invalid selected index fails clearly;
- no pickle dependence.

## B. Timestamp matching / hard-QC-compatible subset

Use synthetic PCA timestamps with a subset of valid training timestamps.

Verify:
- only current valid timestamps participate;
- order is correct;
- mismatch beyond tolerance is not silently paired;
- one location never receives another location's waveform.

## C. PCA waveform subspace loss

Synthetic cases:

1. target exactly equals a linear combination of the 3 score channels:
   - `R²` ~ 1;
   - loss ~ 0.

2. target sign flipped:
   - same `R²` due to `corr²`.

3. score basis permuted/orthogonally rotated:
   - equivalent or numerically near-equivalent result.

4. unrelated waveform:
   - lower `R²`, larger loss.

5. constant/zero scores:
   - finite;
   - no false perfect correlation.

6. rank-deficient scores:
   - finite.

7. backward pass:
   - non-None finite gradient on card scores.

## D. Backward compatibility

With:

```text
cardiac_pca_waveform = 0
```

the new term must not alter legacy loss behavior.

## E. Stage3a ownership

Regression test:
- shared FiLM still frozen;
- cardiac head trainable;
- canonical/resp MBC/uncertainty ownership unchanged;
- PCA loss creates gradient to the cardiac score/card-head path.

## F. Diagnostic schema

Synthetic diagnostic test verifies:
- per-location PCA R²/loss/matched-frame keys;
- summary mean/median;
- existing frequency fields preserved.

## G. Config validation

Verify:
- missing `cardiac_pca_waveform` defaults to 0;
- negative/non-finite values rejected;
- ridge/min-frame settings validated if configurable.

---

# 16. Local verification commands

Do not run GPU code.

Use the repository's CPU-safe project test scope.

Known historical issue: running plain repository-root:

```bash
pytest -q
```

may collect vendored NeSVoR `tests.*` and fail from package-name collision even when project tests pass.

Prefer:

```bash
pytest -q tests
```

plus focused tests for newly added Change5B modules.

Also run:

```bash
python -m compileall -q src scripts
git diff --check
```

If existing project utilities provide source-lock/import checks that are CPU-safe, run them.

Do not install or rebuild CUDA/tinycudann.

Do not run real DYL0709 data locally.

---

# 17. Documentation/provenance update

Update concise documentation only as needed:

```text
README.md
SOURCE_PROVENANCE.md
IMPLEMENTATION_REPORT.md
```

State clearly:

```text
DREME Eq.8/Eq.9
= source-derived negative crossover suppression

Change5A
= project-specific positive target-band concentration

Change5B
= project-specific image-domain adaptation:
  per-location PCA cardiac waveform weak supervision
```

Do not claim PCA waveform is ECG ground truth.

Call it:
- image-derived surrogate;
- weak supervision;
- local fixed-slice temporal prior.

Document that `corr²` / subspace agreement is chosen because:
- PCA sign is arbitrary;
- PCA scale is arbitrary;
- the 3-D cardiac score basis has permutation/rotation ambiguity.

---

# 18. Create server GPU run instructions, but DO NOT execute them locally

Create a short file, for example:

```text
CHANGE5B_SERVER_RUN.md
```

It should contain exact commands for the user to run after GitHub sync on the server.

Known server environment:

```text
repo:
/data/dengyz/code/CR_DREME_v3

conda env:
cr_dreme

GPU:
RTX 3090

Phase1:
/data/dengyz/dataset/CR_DREME_v3/v1/phase1

common Stage2c checkpoint:
/data/dengyz/dataset/CR_DREME_v3/v1/train_stage2c_100/source_first_last.pt
```

Use a new result root such as:

```text
/data/dengyz/dataset/CR_DREME_v3/v1_change5b
```

The first formal Change5B experiment must be:

```text
same Stage2c checkpoint
same seed = 0
same pixel_samples = 256
Stage1 = 0
Stage2a = 0
Stage2b = 0
Stage2c = 0
Stage3a = 1000
Stage3b = 0
Stage3c = 0

cardiac_target_concentration = 0.003
cardiac_pca_waveform         = 0.001
```

The control to compare against is the already-completed:

```text
w003 Stage3a 1000
no PCA waveform loss
```

Do not tell the user to rerun that control unless necessary.

The run document should include:
1. Git sync / HEAD check;
2. 0-step resume/config probe if appropriate;
3. Stage3a 1000 command;
4. diagnostic command;
5. one compact extraction command printing only the important metrics.

Important comparison metrics:

```text
cardiac_pca_waveform_r2 mean/median
cardiac_target_fraction mean/median
card_target/wrong
card_target>wrong count
same resp/card peak exact
same resp/card peak within 0.02 Hz
cardiac ablation gain
card DVF RMS/max
card score std
```

Do not enter Stage3b or Stage3c automatically.

---

# 19. Scientific gate after future GPU run

Do not encode a simplistic "loss decreased = success" rule.

A successful Change5B direction would require a joint pattern such as:

```text
PCA waveform R² clearly increases
AND
cardiac target-band semantics improve
AND
resp/card peak degeneracy decreases
AND
cardiac reconstruction ablation remains positive/useful
AND
DVF remains stable/non-pathological
```

A high PCA R² alone is not sufficient.

If PCA R² improves but:
- target-band fraction stays poor,
- cardiac ablation collapses,
- or motion degenerates,

then the method is not a full physiological success.

---

# 20. Git discipline and final report

Before editing:

```bash
cd /home/universe/SVR/code/CR_DREME_v3
git status --short
git branch --show-current
git rev-parse HEAD
```

Do not overwrite unrelated local modifications.

At completion:
1. show concise diff summary;
2. run tests;
3. commit with a message similar to:

```text
feat: add PCA cardiac waveform supervision
```

4. attempt:

```bash
git push origin dev/cardioresp4d
```

If push fails because of network/SSH:
- do not repeatedly waste time;
- preserve the local commit;
- report the exact commit hash and push error.

Final Codex response must include only a concise engineering summary:

```text
Starting HEAD
Ending HEAD / commit
Files changed
Mathematical implementation
PCA loader/alignment behavior
Config defaults and Change5B config values
Diagnostic additions
Tests run + pass/fail counts
CPU-only confirmation
GPU/real-data NOT run
Server run document path
Push status
Any remaining risk/blocker
```

Do not claim Change5B scientifically works until the user runs the server GPU experiment.

---

# 21. Non-negotiable prohibitions

Do NOT:
- use GPU locally;
- fabricate server results;
- access `/data/dengyz/...` as if it exists locally;
- run Stage3b/Stage3c;
- change Eq.8/Eq.9;
- remove Change5A;
- alter MBC/PSF/canonical architecture;
- add RNN/Transformer/temporal encoder;
- introduce a global cardiac phase;
- align same frame indices across slices;
- introduce a global PCA waveform;
- use unreliable local PCA candidate via hidden global fallback;
- directly MSE-match raw PCA waveform amplitude;
- detach card scores in the Change5B loss;
- add new learnable projection nuisance parameters;
- re-enable hard-invalid observations;
- modify third-party source;
- use `strict=False` to hide checkpoint incompatibilities;
- delete existing experiments/configs/tests.

The implementation should be minimal, auditable, backward-compatible, and suitable for a fair Change5B vs `w003@1000` experiment.
