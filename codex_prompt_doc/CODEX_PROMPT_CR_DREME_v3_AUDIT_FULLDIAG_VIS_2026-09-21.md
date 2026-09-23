# Codex Prompt — CR_DREME_v3 Change4/Change5 Scientific Audit + Full-Location Diagnostics + Read-Only Visualization

## 0. Operating contract

Work in:

```text
/home/universe/SVR/code/CR_DREME_v3
```

Expected branch:

```text
dev/cardioresp4d
```

Last known Change5B commit:

```text
8fde85bbd4f87886b142d716ab9554cdb1b2b3d0
feat: add per-location PCA waveform supervision
```

Do not reset, rebase, delete historical files, or discard later user work. First inspect:

```bash
cd /home/universe/SVR/code/CR_DREME_v3
git status --short
git branch --show-current
git rev-parse HEAD
git log -5 --oneline
```

If HEAD is newer, continue from actual HEAD after inspecting changes.

### Critical environment constraint

Codex runs **locally, CPU-only**.

Local conda env:

```text
knesvr_torch
```

Do **not**:

- use CUDA locally;
- run real-subject training locally;
- assume `/data/dengyz/...` exists locally;
- fabricate GPU/real-data results;
- modify vendored NeSVoR/SINR scientific source;
- enter Stage3b or Stage3c;
- redesign architecture, MBC, canonical INR, PSF, or Eq.8/Eq.9 merely to make tests pass.

Tasks, in order:

1. independently audit Change4 + Change5A + Change5B scientific/code correctness;
2. if there is no material scientific blocker, add full-location diagnostics;
3. add paired diagnostic comparison;
4. add read-only checkpoint visualization/export;
5. add exact server GPU run instructions;
6. run CPU tests;
7. commit/push.

Keep terminal output concise and monitoring low-frequency.

---

# 1. Project context

Goal: reconstruct free-breathing real-time cardiac MRI from acquired 2D cine into a 3D+time representation.

There is:

- no saved ECG;
- no respiratory belt;
- no raw k-space for this reconstruction pipeline;
- no globally synchronized cardiac phase across slice locations.

Current model concept:

```text
acquired frame
  -> Geometry FiLM encoder
  -> respiratory scores + cardiac scores
  -> score-weighted respiratory/card MBCs
  -> sequential observation->reference pullback
  -> NeSVoR canonical INR
  -> PSF-aware slice prediction
```

Intended pullback:

```text
x_ref = y + d_c(y) + d_r(y + d_c(y))
```

where `y` is an observation-world coordinate.

Do not introduce 3D GT, ECG labels, a global cardiac phase, or cross-location frame alignment.

---

# 2. Stage contract to audit, not assume

Intended:

```text
Stage1:
  canonical only

Stage2a/b/c:
  progressively respiratory

Stage3a:
  cardiac warm-up
  canonical frozen
  respiratory MBC frozen
  shared FiLM/resp representation frozen
  only cardiac FiLM head + cardiac MBC trainable
  MSE
  no uncertainty

Stage3b:
  joint refinement
  NOT part of this task

Stage3c:
  uncertainty/Gaussian NLL
  NOT part of this task
```

This task remains strictly Stage3a/read-only diagnostics.

---

# 3. Change4 / Change5 definitions

## Change4

Audit the current DREME-style motion/frequency implementation:

- MBC normalization;
- zero-mean scores;
- respiratory-score cardiac leakage suppression;
- cardiac-score respiratory leakage suppression;
- local per-location frequency priors;
- true timestamps;
- sequential respiratory/cardiac pullback;
- Stage3a ownership.

## Change5A

Project-specific cardiac target-band concentration:

```text
L_card-conc = 1 - P_target / (P_total + eps)
```

Current main weight:

```text
cardiac_target_concentration = 0.003
```

This is a project adaptation, not claimed to be DREME Eq.8/Eq.9.

## Change5B

Per-location image-derived PCA cardiac waveform weak supervision.

For a location:

```text
W [T,3] = current cardiac scores
c [T]   = selected Phase-1 PCA temporal waveform

X = W - mean_t(W)
y = c - mean(c)

G = X^T X / T
b = X^T y / T

lambda_eff = ridge * trace(G)/3 + eps
a* = solve(G + lambda_eff I, b)

yhat = X a*
R2 = corr(yhat, y)^2
L_PCA = 1 - R2
```

Required semantics:

- no learnable nuisance projection;
- PCA target detached;
- cardiac scores differentiable;
- sign ambiguity handled by corr²;
- score basis permutation/rotation ambiguity handled by 3-D subspace projection;
- per-location only;
- no global waveform fallback;
- no cross-location match;
- selected PC is **1-based** in JSON;
- exact target:
  `temporal_pcs[:, selected_pc - 1]`;
- reliable candidates only;
- hard-invalid observations excluded;
- no interpolation across unmatched timestamps.

Current Change5B reference config:

```text
configs/source_first_change5b.yaml

cardiac_target_concentration = 0.003
cardiac_pca_waveform          = 0.001
pca_waveform_ridge            = 1e-4
pca_waveform_min_frames       = 8
```

---

# 4. Current real GPU evidence — do not fabricate or reinterpret

The user already ran these server experiments:

```text
run              R2mean  R2med   TFmean  TFmed   T/W     T>W   same  near  abl%   DVFrms  DVFmax  scoreSD
CTRL_2000        0.1490  0.1097  0.0796  0.0548  0.8546  4/9   6/9   6/9   2.071  0.505   2.680   0.602
CTRL_2500        0.1494  0.0961  0.0831  0.0558  0.8515  4/9   6/9   6/9   1.718  0.569   3.807   0.592
LATE_1e4         0.1537  0.0966  0.0810  0.0567  0.8535  4/9   6/9   6/9   2.590  0.556   3.232   0.581
LATE_3e4         0.1512  0.0989  0.0825  0.0563  0.8537  4/9   6/9   6/9   2.280  0.553   3.316   0.591
EARLY_5B_2000    0.1811  0.1165  0.0792  0.0576  0.8148  4/9   5/9   6/9   1.778  0.480   3.150   0.504
EARLY_5B_2500    0.1799  0.1208  0.1030  0.0584  0.8572  4/9   4/9   5/9   0.098  0.538   3.935   0.496
```

Current interpretation:

- early strong PCA can improve semantic indicators but may damage reconstruction usefulness;
- late weak PCA preserves/improves reconstruction usefulness;
- 9-location frequency semantics are too sparse for a final conclusion;
- `LATE_1e4` is the current best **candidate**, not a final model;
- full-location paired audit is required before more training sweeps.

### Diagnostic caveat to verify

Inspect whether current `diagnose_change4_checkpoint.py` computes motion statistics from only:

```python
probe = valid[0]
```

If yes:

- document that historical `motion_statistics` are probe-specific;
- do not silently reinterpret old numbers as whole-dataset values;
- preserve legacy output if needed;
- add explicitly named aggregate motion metrics instead of silently changing semantics.

Also audit the exact population used by cardiac ablation.

---

# 5. Phase-1 PCA facts

Server Phase-1:

```text
/data/dengyz/dataset/CR_DREME_v3/v1/phase1
```

Artifacts:

```text
frequency/frequency_bands.json
frequency/<view>/<slice_id>/pca_psd.npz
```

Important fields:

```text
timestamps_s
temporal_pcs
explained_variance
frequencies_hz
pc_psd
```

Previously audited:

```text
143 reliable cardiac candidates
PC1: 26
PC2: 100
PC3: 16
PC4: 1
```

Do not rerun Phase-1 and do not invent a new PC selection rule.

---

# 6. TASK A — independent scientific implementation audit

Create:

```text
SCIENTIFIC_AUDIT_CHANGE4_CHANGE5.md
```

For every item classify:

```text
PASS
PASS WITH DOCUMENTED LIMITATION
BUG
AMBIGUOUS / REQUIRES SCIENTIFIC DECISION
```

Cite exact source file/function/line ranges.

## A1. Forward model

Inspect at minimum:

```text
src/cardioresp4d/training/model.py
src/cardioresp4d/models/cardioresp_motion.py
src/cardioresp4d/models/film_motion_encoder.py
src/cardioresp4d/adapters/nesvor_psf.py
src/cardioresp4d/adapters/nesvor_inr.py
src/cardioresp4d/adapters/sinr_mbc.py
src/cardioresp4d/training/stage_contract.py
src/cardioresp4d/training/trainer.py
```

Verify:

- world-mm convention;
- DICOM row/column/normal basis;
- pixel-spacing ordering;
- observation->reference displacement sign;
- cardiac-then-respiratory composition;
- local cardiac box/taper;
- PSF samples are pulled back before canonical query;
- score tensor shapes;
- score-weighted MBC summation;
- no reconstruction-gradient detach;
- no motion double application.

## A2. Stage3a ownership

Verify both `requires_grad` and **actual numerical parameter updates**.

Add a deterministic CPU test:

1. small synthetic model;
2. configure/resume into Stage3a;
3. snapshot parameters;
4. take one optimizer step;
5. prove:
   - canonical unchanged;
   - respiratory MBC unchanged;
   - shared FiLM unchanged;
   - cardiac FiLM head can change;
   - cardiac MBC can change.

Do not merely inspect `requires_grad`, because old optimizer parameter groups may persist after resume.

## A3. Change4 / DREME-style losses

Inspect:

```text
src/cardioresp4d/losses/motion_loss.py
src/cardioresp4d/losses/frequency_loss.py
src/cardioresp4d/training/trainer.py
SOURCE_PROVENANCE.md
IMPLEMENTATION_REPORT.md
```

Audit separately:

### MBC normalization
- reduction axes;
- respiratory levels;
- xyz handling;
- cardiac inclusion;
- physical-domain semantics.

### zero-mean score
- temporal mean vs instantaneous amplitude;
- channel axes;
- stage ownership.

### Eq.8-like respiratory-score cardiac leakage
- correct score branch;
- correct cardiac band;
- baseline pairing;
- complex coefficient subtraction;
- mean-centering;
- timestamp usage;
- frequency sampling.

### Eq.9-like cardiac-score respiratory leakage
- correct score branch;
- local respiratory band;
- true timestamps;
- no swapped cardiac/resp semantics.

If source paper/formulas are locally available, cross-check them. If not, state the limitation; do not pretend external verification.

## A4. Change5A concentration

Audit:

```text
L = 1 - target_power / (total_non_dc_power + eps)
```

Verify:

- total spectrum excludes DC;
- mean-centering;
- channel aggregation;
- target mask;
- nearest-grid fallback;
- Nyquist;
- irregular timestamps;
- gradients;
- zero/near-zero score behavior;
- whether amplitude shrinkage can pathologically reduce the loss;
- whether channel aggregation creates unintended dominance.

Add deterministic synthetic CPU tests:

- pure target sinusoid;
- pure wrong/resp sinusoid;
- mixture;
- constant;
- amplitude-scaled signals;
- irregular timestamp subset;
- target band between frequency-grid centers.

Assert expected monotonic relationships.

## A5. Change5B PCA prior/loss

Inspect:

```text
src/cardioresp4d/frequency/pca_waveform_prior.py
src/cardioresp4d/losses/motion_loss.py
src/cardioresp4d/training/trainer.py
scripts/diagnose_change4_checkpoint.py
tests/test_change5b_pca_waveform.py
```

Verify:

- JSON selected_pc is 1-based and converted once;
- correct location keys;
- reliable candidates only;
- no global fallback;
- no cross-location match;
- timestamp tolerance is justified;
- no interpolation;
- valid-frame subset order preserved;
- hard-invalid frames excluded;
- PCA target detached;
- card-score gradient connected;
- ridge solve differentiable;
- corr² invariances truly hold;
- rank-deficient/constant cases cannot create false perfect R²;
- skip behavior finite and zero-gradient;
- PCA loss is Stage3a-only;
- no Stage3b/3c side effects.

Expand tests for:

- arbitrary orthogonal 3x3 score-basis rotation;
- large/small score scaling;
- PCA sign flip;
- collinear channels;
- constant target;
- all-zero scores;
- NaN rejection;
- timestamp subset with one hard-invalid frame removed.

## A6. Loss-gradient competition audit

Create a reusable read-only script, preferably:

```text
scripts/audit_stage3a_loss_gradients.py
```

It must:

- accept manifest/QC/domain/config/checkpoint;
- do no optimizer step;
- use deterministic fixed location or user-specified location;
- compute each Stage3a loss independently;
- report gradient norms for:
  - cardiac FiLM head;
  - cardiac MBC;
- verify frozen modules have no gradients;
- report:
  - raw gradient norm;
  - configured-weighted gradient norm.

Include where applicable:

```text
data/reconstruction
image regularization
MBC normalization
cardiac smoothness
zero-mean score
Eq.8/9
Change5A concentration
Change5B PCA waveform
```

Clearly identify a component that mathematically does not reach a module.

Add CPU synthetic tests. Do not run server data locally.

## A7. Audit diagnostics

Audit:

```text
scripts/diagnose_change4_checkpoint.py
```

Document:

- population for `motion_statistics`;
- population for cardiac ablation;
- population for frequency semantics;
- representative-location selection;
- target/wrong aggregation;
- peak comparison semantics;
- reproducibility;
- PSF randomness;
- model train/eval state;
- whether execution can mutate the model/checkpoint.

If a historical field is misleading, preserve compatibility and add a newly named corrected field rather than silently changing old meaning.

---

# 7. HARD AUDIT GATE

If any material scientific bug is found, e.g.:

- cardiac/resp bands swapped;
- wrong pullback sign/order;
- loss on wrong branch;
- PC off-by-one;
- frozen modules actually update;
- hard-invalid data re-enters;
- PCA gradient disconnected;
- concentration numerator/denominator wrong;
- diagnostic field materially differs from its documented meaning;

then:

1. write:
   ```text
   SCIENTIFIC_AUDIT_CHANGE4_CHANGE5.md
   AUDIT_BLOCKER_REPORT.md
   ```
2. add the smallest reproducing CPU test if possible;
3. **do not silently modify the scientific implementation**;
4. **do not continue Tasks B/C/D on top of broken semantics**;
5. stop and report to the user.

Small documentation/CLI/defensive-validation issues are not blockers.

Only continue if audit passes or has non-blocking documented limitations.

---

# 8. TASK B — full-location diagnostics

Extend existing diagnostic without breaking default behavior.

Preferred:

```text
scripts/diagnose_change4_checkpoint.py --all-locations
```

Default behavior remains backward compatible.

## B1. Eligibility

For `--all-locations`:

- QC-valid observations only;
- hard-invalid frames excluded;
- do not require exactly 50 valid frames;
- require enough frames for each metric;
- if unevaluable, record a structured skip reason;
- never silently drop locations.

## B2. Per-location fields

At minimum:

```text
view
slice_id
n_valid_frames

selected_cardiac_pc
cardiac_pca_waveform_r2
cardiac_pca_waveform_loss
cardiac_pca_matched_frames

cardiac_target_fraction
cardiac_target_concentration

card_target_amp
card_wrong_amp
card_target_over_wrong

resp_target_amp
resp_wrong_amp

card_peak_hz
resp_peak_hz
same_peak_exact
same_peak_within_0.02_hz

card_score_mean
card_score_std
```

If available and unambiguous, include Phase-1 metadata such as cardiac candidate frequency/dominance/explained variance.

Write:

```text
all_location_diagnostic.json
all_location_frequency_semantics.csv
```

## B3. Summaries

Report overall, per view, and per selected PC.

Continuous metrics:

```text
n mean median q25 q75 min max
```

Boolean metrics:

```text
count fraction
```

## B4. Paired comparison tool

Add e.g.:

```text
scripts/compare_all_location_diagnostics.py
```

Accept two or more labeled diagnostic JSONs.

Use exact intersection of location keys and report missing locations.

Paired deltas:

```text
PCA R²
target fraction
target/wrong ratio
card score std
```

For each:

```text
mean delta
median delta
q25/q75
n improved
n worsened
n unchanged within documented tolerance
```

Also compare transition counts for:

```text
same_peak_exact
same_peak_within_0.02_hz
target > wrong
```

Do not make significance claims from one subject.

---

# 9. TASK C — aggregate motion diagnostics

Do not overwrite historical semantics invisibly.

Add clearly named aggregate motion metrics across an explicitly documented population.

At minimum:

```text
cardiac DVF RMS mm
cardiac DVF magnitude p50/p95/p99/max
respiratory DVF RMS mm
respiratory DVF magnitude p50/p95/p99/max
```

Use deterministic sampling if full evaluation is expensive.

Record:

```text
n_locations
n_frames_evaluated
sampling rule
evaluation grid shape
physical domain
```

---

# 10. TASK D — read-only visualization

No retraining.

Create a coherent visualization tool, e.g.:

```text
scripts/visualize_checkpoint_dynamics.py
```

Reusable helpers may live under:

```text
src/cardioresp4d/visualization/
```

Do not put visualization-only behavior into training code.

## D1. Labeling constraint

Because there is no cross-location synchronized cardiac phase, never call the current result a globally synchronized physiological 4D cine.

For a fixed location whose acquired frames produce FiLM scores, call the 3D output:

```text
observation-conditioned implied 3D dynamics
```

Frame `k` from different slice locations is not a shared physiological phase.

## D2. Reprojection cine — highest priority

For one fixed location across valid frames, render:

```text
Acquired 2D
Resp-only prediction
Resp+Card prediction
|Acquired - Resp-only|
|Acquired - Resp+Card|
```

Use a readable 2x3 or similar layout.

Requirements:

- identical frame order/timestamps;
- full slice rendered in chunks;
- same intensity window for cross-checkpoint comparisons;
- deterministic PSF behavior for comparison:
  - fixed RNG state/seed, or
  - explicitly documented deterministic visualization-only sampling;
- frame/timestamp label;
- optional per-frame MSE text;
- export:
  - MP4 if available;
  - GIF fallback;
  - PNG montage;
  - per-frame numeric CSV.

Do not require ffmpeg for unit tests.

## D3. Observation-conditioned implied 3D volume

For a selected location:

1. encode each acquired frame into resp/card scores;
2. create a fixed world-mm 3D grid;
3. evaluate frame-conditioned motion;
4. query canonical INR at pullback coordinates;
5. produce one 3D intensity volume per frame.

For this visualization, direct voxel-grid canonical query after motion pullback is appropriate; do not apply thick-slice PSF to the 3D voxel grid.

Use chunked inference and identical world grid across checkpoints.

Export, if `nibabel` is already available:

```text
canonical_reference.nii.gz
conditioned_dynamic_4d.nii.gz
```

Otherwise use `.npz` and document the limitation rather than installing a large dependency.

Also export orthogonal-plane cine/montage.

## D4. DVF visualization

For each selected frame/location export:

```text
cardiac DVF
respiratory DVF
sequential total observation->reference displacement
```

Total displacement:

```text
d_total(y) = reference_points(y) - y
```

Respiratory field must be evaluated at cardiac-shifted coordinates for the sequential composition.

Export:

- vector arrays;
- magnitude volumes;
- SAX/coronal/sagittal magnitude maps;
- sparse vector/quiver overlays;
- deformation-grid visualization.

Units must be mm.

If arrows are visually scaled, label the scale factor.

## D5. Jacobian determinant

For:

```text
phi(y) = reference_points_mm(y)
```

compute finite-difference Jacobian determinant on the regular physical world grid.

Also cardiac-only Jacobian if straightforward.

Report per frame and aggregate:

```text
min
p01
median
p99
max
fraction det(J) <= 0
```

Export heatmap/montage and machine-readable arrays.

Rules:

- account for physical grid spacing;
- explicitly label this as Jacobian of observation->reference pullback;
- do not call it diffeomorphic merely because most determinants are positive.

Synthetic tests:

- identity -> det≈1;
- translation -> det≈1;
- known axis scaling -> expected det;
- simple folding -> negative somewhere.

## D6. Score/PCA/spectrum visualization

For the same location plot:

- three cardiac score channels vs acquisition time;
- selected PCA waveform, display-normalized only;
- respiratory score channels, optionally separate;
- cardiac-score power spectrum;
- local cardiac/resp target bands;
- selected PC.

State that PCA sign/amplitude are arbitrary and training uses subspace corr², not raw waveform MSE.

## D7. Cross-checkpoint consistency

Support multi-checkpoint processing with shared:

- location;
- frame/timestamp set;
- world grid;
- crop;
- intensity window;
- quiver subsampling;
- RNG seed if relevant.

Main server comparison:

```text
CTRL_2000
CTRL_2500
LATE_1e4
EARLY_5B_2500
```

Optionally include `LATE_3e4`.

---

# 11. Server paths for generated run guide

Codex must not access these locally.

Common:

```text
PHASE1=/data/dengyz/dataset/CR_DREME_v3/v1/phase1
FREQ=${PHASE1}/frequency/frequency_bands.json
MANIFEST=${PHASE1}/dicom_manifest.csv
QC=${PHASE1}/acquisition_qc/acquisition_qc.csv
DOMAIN=${PHASE1}/canonical_domain/canonical_domain.json
```

Checkpoints:

```text
CTRL_2000:
/data/dengyz/dataset/CR_DREME_v3/v1_change5b_length_probe/control_2000/source_first_last.pt

CTRL_2500:
/data/dengyz/dataset/CR_DREME_v3/v1_change5b_length_probe/control_2500/source_first_last.pt

LATE_1e4:
/data/dengyz/dataset/CR_DREME_v3/v1_change5b_late_pca/late_pca_1e4_2500/source_first_last.pt

LATE_3e4:
/data/dengyz/dataset/CR_DREME_v3/v1_change5b_late_pca/late_pca_3e4_2500/source_first_last.pt

EARLY_5B_2500:
/data/dengyz/dataset/CR_DREME_v3/v1_change5b_length_probe/change5b_2500/source_first_last.pt
```

Configs:

```text
CTRL:
/data/dengyz/dataset/CR_DREME_v3/v1_change5a_weight_sweep/configs/source_first_change5a_w003.yaml

LATE_1e4:
/data/dengyz/dataset/CR_DREME_v3/v1_change5b_late_pca/configs/late_pca_1e4.yaml

LATE_3e4:
/data/dengyz/dataset/CR_DREME_v3/v1_change5b_late_pca/configs/late_pca_3e4.yaml

EARLY_5B:
configs/source_first_change5b.yaml
```

Known candidate locations:

```text
SAX/SAX_s026
SAX/SAX_s001
2CH/2CH_s001
4CH/4CH_s001
```

Scripts must accept arbitrary valid locations; do not hard-code only these.

---

# 12. Server-run document

If the audit gate passes, create:

```text
CHANGE5C_AUDIT_DIAGNOSTIC_VIS_SERVER_RUN.md
```

Server:

```text
repo: /data/dengyz/code/CR_DREME_v3
env: cr_dreme
GPU: RTX 3090
```

Include exact commands for:

1. Git sync/version check.
2. `--all-locations` diagnostics for:
   - CTRL_2500
   - LATE_1e4
   - LATE_3e4
   - EARLY_5B_2500
3. paired full-location comparisons:
   - CTRL_2500 vs LATE_1e4
   - CTRL_2500 vs LATE_3e4
   - CTRL_2500 vs EARLY_5B_2500
4. read-only gradient audit for at least:
   - CTRL_2500
   - LATE_1e4
   - EARLY_5B_2500
   if runtime is reasonable.
5. visualization packages for:
   - CTRL_2000
   - CTRL_2500
   - LATE_1e4
   - EARLY_5B_2500
   at:
   - SAX/SAX_s026
   - SAX/SAX_s001
   - 2CH/2CH_s001
   - 4CH/4CH_s001

Avoid printing huge JSON. Save files and print concise summaries/paths only.

---

# 13. Local tests

Run:

```bash
conda activate knesvr_torch
cd /home/universe/SVR/code/CR_DREME_v3

pytest -q tests
python -m compileall -q src scripts tests
git diff --check
```

Also run existing source-lock/import checks if documented.

Add CPU tests for:

- exact Stage3a parameter-update ownership;
- Change5A synthetic spectra;
- Change5B invariance/degenerate cases;
- all-location eligibility with hard-invalid/missing frames;
- paired diagnostics;
- Jacobian math;
- deterministic visualization helpers;
- direct 3D pullback volume shape/coordinate contract;
- total displacement identity:
  `reference_points - observation_points`;
- no cross-location score/waveform mixing.

---

# 14. Performance constraints

Do not materialize a full high-resolution 4D volume on GPU at once.

Use:

- `torch.no_grad()`;
- chunked spatial inference;
- configurable voxel spacing/grid;
- configurable frame count;
- configurable crop/domain;
- deterministic reproducibility.

Defaults should be conservative for RTX3090.

Do not silently lower scientific fidelity for speed.

---

# 15. Documentation/provenance

Update, where supported:

```text
README.md
IMPLEMENTATION_REPORT.md
SOURCE_PROVENANCE.md
```

Clearly distinguish:

```text
source-derived DREME/NeSVoR/SINR behavior
project-specific Change5A
project-specific Change5B
diagnostic-only code
visualization-only code
```

Do not claim:

- PCA waveform = ECG ground truth;
- observation-conditioned volume = globally synchronized cardiac cine;
- visually plausible deformation = physiological validation.

---

# 16. Git discipline

Before commit:

```bash
git status --short
git diff --check
```

If audit finds a blocker, do not make a misleading feature-complete commit.

If audit passes and implementation/tests are complete, use e.g.:

```text
feat: add full-location audit and motion visualization
```

Then:

```bash
git push origin dev/cardioresp4d
```

If push fails, report exact error. Do not rewrite history.

---

# 17. Final report required

Report compactly:

1. starting HEAD;
2. ending HEAD/commit;
3. audit verdict for:
   - forward motion/PSF;
   - Stage3a ownership;
   - Change4 losses;
   - Change5A;
   - Change5B;
   - existing diagnostics;
4. every bug/limitation found;
5. whether the audit gate blocked downstream work;
6. files added/modified;
7. exact test counts;
8. whether full-location diagnostics are ready;
9. whether gradient audit is ready;
10. whether visualization is ready;
11. server-run document path;
12. push result.

Do not report any GPU/real-data result that Codex did not actually run.

---

# 18. Non-negotiable prohibitions

- No local GPU assumptions.
- No server real-data training.
- No Stage3b/Stage3c.
- No global cardiac phase.
- No cross-location frame-index alignment.
- No PCA interpolation.
- No global waveform fallback.
- No raw PCA-waveform MSE replacing corr².
- No learnable PCA nuisance projection.
- No hard-invalid re-entry.
- No silent Eq.8/Eq.9 change.
- No silent pullback-convention change.
- No architecture redesign.
- No vendored third-party source edits.
- No `strict=False` checkpoint loading to hide mismatches.
- No deletion of historical files.
- No fabricated scientific conclusions.

The goal is to make the current Stage3a evidence **more trustworthy and more interpretable**, not to produce prettier results by changing the science.
