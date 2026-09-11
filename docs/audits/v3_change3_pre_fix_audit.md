# v3_change3 Pre-fix Closure Audit

## Scope and result

Baseline: `2e8975e6d73862feda57bcaf6eeefc8d464ce8db`. This audit includes the
uncommitted v3_change2 worktree and makes **no runtime/test-code modification**.
It replaces the previous “5 Critical + 8 Important” snapshot: that snapshot
was not a closure audit.

Pre-fix result: **Critical 5, Important 33, Minor 5; `GPU_GATE = FAIL`.**
Repairs may start only after this matrix is accepted as the fix backlog; the
post-fix audit must be a fresh call-graph review, not a status edit to this file.

### Mainline call graph inspected

```text
manifest/runtime sidecar
  -> acquisition QC -> CardioRespDataset -> normalization groups
  -> observations_from_manifest -> DynamicObservation / frame IDs
  -> ViewLocationBalancedSampler -> FiLM scores
  -> active SINR MBC -> sequential pullback -> NeSVoR PSF -> NeSVoR INR
  -> (Stage3) dynamic-frame uncertainty -> data fidelity / image regularizer
  -> Eq.6/7/8/9 + smoothness -> optimizer/stage state -> checkpoint/report
```

Formal runtime entry is `scripts/train_source_first.py`. Legacy
`src/cardioresp4d/training/stage1.py` is preserved as ablation/legacy and is
outside this mainline, although its regression tests still need classification.

## Section 1–42 coverage matrix

| § | Closure subject | Audit IDs and evidence | Status |
|---|---|---|---|
| 1 | Audit first; full call graph | this file, graph above, all rows below | covered; repair blocked until audit complete |
| 2 | Config, unused code, axes/units, inactive compute, report truth | C01–C05, I01–I05, I19, I22 | open |
| 3 | Eq.6 raw active bases / axes | C01, C02 | open |
| 4 | Non-trainable progressive scheduling | C02, I20 | open |
| 5 | Eq.8 complex subtraction | C03 | open |
| 6 | Eq.8 baseline pairing/provenance | C03, I06 | open |
| 7 | Multiple resolved frequencies per band | I07 | open |
| 8 | NUDFT stability, scale, DC, jitter | I08 | open |
| 9 | Frequency source-of-truth and location alignment | I04 | open; no per-location answer assumed |
| 10 | Functional fallback/fail-fast | I05 | open |
| 11 | Stage1 no FiLM/motion | C04, I20 | open |
| 12 | Stage2 active respiratory only | C02, C04, I20 | open |
| 13 | Stage1 does not calculate unused regularizers | C04, I11 | open |
| 14 | Config-driven model factory | C05, I12 | open |
| 15 | Object-derived effective config | C05, I12, I22 | open |
| 16 | Geometry/oblique projected extent | I13 | open |
| 17 | Configured separate smoothness grids | I14 | open |
| 18 | Stage-specific optimizer groups/LRs | I15 | open |
| 19 | Optimizer state transition | I16 | open |
| 20 | Ownership: call/grad/value/optimizer/state | C02, C04, I20 | open |
| 21 | Temporal sequence span/batch/sufficiency | I09 | open |
| 22 | Hard-QC runtime assertion | I10 | open |
| 23 | Dynamic identity/timestamp validation | I17 | open |
| 24 | Normalization real-data grouping/preflight | I19 | open |
| 25 | Uncertainty shape/gradient integration | I20 | open |
| 26 | Formal uncertainty/fidelity/image-reg reconciliation | I11, I20 | open |
| 27 | Exact stage loss total / Eq.7 active channels | I21 | open |
| 28 | Checkpoint/resume | I23 | open |
| 29 | Seed/reproducibility | I24 | open |
| 30 | Metrics/reporting | I22 | open |
| 31 | Performance call counters | I25 | open |
| 32 | Seven baseline regression failures | I26 | open |
| 33 | Prompt files ordinary files | I27 | open |
| 34 | Changelog baseline | I28 | open |
| 35 | Source lock/dependency clean checkout | I29, I30 | open |
| 36 | Real-data preflight report | I31 | open |
| 37 | Independent post-fix audit | I32 | pending after repairs |
| 38 | Required strengthened tests | Required-test column below | open |
| 39 | Stage1→3 synthetic ownership smoke | I33 | open |
| 40 | Full unittest gate | I26 | open |
| 41 | GPU gate | all C/I findings | **FAIL** |
| 42 | Short GPU command only after gate | §41 unmet | blocked |

### Process-coverage addendum: §0, §43, §44, §45

| § | Closure subject | Audit evidence / required action | Status |
|---|---|---|---|
| 0 | Source-first and engineering boundaries | `third_party/SOURCE_LOCK.json`, source-backed adapters and existing provenance are the only permitted implementation basis; no upstream primitive may be reimplemented during repair. | open: reverify in post-fix audit |
| 43 | Final reporting contract | Final report must state baseline, every audit count, executed/static/GPU-pending tests, preflight, source lock, legacy status, unresolved minors and gate outcome. | open: final-deliverable requirement |
| 44 | Final repository-wide self-check | Immediately before post-fix gate, actually run the exact prescribed `rg` pattern over `src scripts configs tests README.md IMPLEMENTATION_REPORT.md SOURCE_PROVENANCE.md`, classify every hit, and record command/output evidence in post-fix audit. | pending: mandatory final execution |
| 45 | Semantic equality criterion | Post-fix audit must independently demonstrate `paper/spec semantics = runtime call graph = config = tests = report`; test pass alone cannot close a finding. | pending: post-fix criterion |

## Findings register

“Required test” is a test to create/run during the repair phase; it is not a
claim that it has run in this audit-only phase.

| ID | Severity / module | Observed behavior and evidence | Affected files | Root cause | Proposed fix | Required test | Status |
|---|---|---|---|---|---|---|---|
| C01 | Critical — Eq.6 MBC normalization | `dreme_mbc_normalization()` reduces `tuple(range(mbcs.ndim-2))`; for `[B,L,N,xyz]`, it reduces `(B,L)` and leaves spatial `N`, not `[L,xyz]`. `motion_regularizers()` includes resp+card regardless of stage. | `losses/motion_loss.py`; `training/model.py`; `training/trainer.py` | No explicit spatial/level/component axis contract. | Define raw ungated basis API; evaluate only active resp levels plus cardiac only Stage3; reduce spatial axes with documented voxel-volume/mean-square semantics. | Distinct-value `[B,L,N,xyz]` analytical loss; unit-RMS minimum; zero not minimum; active-set test. | open |
| C02 | Critical — progressive SINR | `level_gates` is an `nn.Parameter`; all levels run in `forward()` before multiplication; trainer enables gate gradient. | `adapters/sinr_mbc.py`; `training/trainer.py`; `training/model.py` | Trainable gate introduces third scale ambiguity; masking occurs after expensive FFD. | Non-trainable schedule/ramp; invoke only active upstream levels; no gate optimizer state. | Per-stage call counters, no trainable gate, inactive forward/grad/optimizer/Eq.6 absence. | open |
| C03 | Critical — Eq.8 | `dreme_cardiac_leakage_in_resp()` uses `(mean(abs(Fc))-mean(abs(Fb)))²`, independent band lists, and centres only. | `losses/frequency_loss.py`; `frequency/training_prior.py`; `training/trainer.py` | Complex subtraction and explicit pairing lost in simplification. | Store pairs; use `abs(F(ωc)-F(ωb))²` over resolved grids; classify baseline provenance correctly. | Hand-computed complex phase; pairing permutation/mismatch; multi-frequency band test. | open |
| C04 | Critical — stage execution | `predict()` calls FiLM before zeroing in Stage1, builds both MBC paths every stage; Stage2 only zeroes cardiac score. Trainer calculates motion regularizers even Stage1. | `training/model.py`; `training/trainer.py` | Frozen/zero was confused with not-called. | Separate stage graphs: S1 PSF→INR only, S2 active resp only, S3 joint; calculate applicable losses only. | FiLM/SINR call counts, gradients, parameter changes, optimizer membership per stage. | open |
| C05 | Critical — config/runtime/report | YAML is loaded but model is constructed with defaults; canonical, PSF, SINR cps/shapes/hidden/taper, FiLM, uncertainty, smoothness and LR are ignored. Report calls requested YAML `effective_config`. | `scripts/train_source_first.py`; `configs/source_first.yaml`; `training/model.py`; `training/trainer.py` | No factory/introspection contract. | One config/domain builder; reject unused formal keys; serialize actual objects/trainer. | Perturb PSF 8→7, latent 16→5, hidden 64→11, cps/taper/grid and prove object/report change. | open |
| I01 | Important — cardiac boundary | Cardiac adapter has linear `taper_mm`, but no explicit zero/smooth boundary contract test; smoothness presently samples cardiac field on full canonical grid. | `adapters/sinr_mbc.py`; `training/model.py`; config | Local-domain adaptation not closed with regularization semantics. | State taper contract; evaluate cardiac field/regularizer local-box only. | Interior/boundary/outside/taper-gradient mm tests. | open |
| I02 | Important — MBC units | Conversion using `grid_spacing_mm` is present, but no composed-field test/report proves grid-unit→mm and score×field dimensional semantics. | `adapters/sinr_mbc.py`; `models/cardioresp_motion.py` | Adapter-level semantics never closed at composition boundary. | Report actual spacing/units and test source-vs-adapter conversion. | Known grid displacement→mm; no cross-axis mixing; DVF unit test. | open |
| I03 | Important — pullback integration | Formula code is plausible but no nonzero end-to-end proof that renderer receives cardiac-first, then respiratory-at-shifted-point coordinates. | `models/cardioresp_motion.py`; `training/model.py` | Unit formula did not verify render integration/order. | Preserve only after a renderer-connected coordinate test. | Noncommuting synthetic fields through `predict()`: `y+d_c(y)+d_r(y+d_c)`. | open |
| I04 | Important — frequency source-of-truth (I3) | Formal v3 §10 says per-slice diagnostics plus consensus respiratory/cardiac bands. Actual Phase‑1 schema retains `per_slice_candidates` and `union_resolution_bins_hz`, explicitly refusing a forced global cardiac point estimate. Current training parser discards candidates and uses global cardiac union for every sequence. It has no verified `view,slice_id` mapping. | main spec §10/§19.5; `frequency/frequency_bands.py`; `frequency/training_prior.py`; `training/sampler.py`; `training/trainer.py` | Formal prose and real Phase‑1 schema were not reconciled. **No assumption is made here that per-location is inherently correct.** | Define authoritative mapping after checking actual Phase‑1 artifact identity: retain global evidence, use location lookup only when `slice_key` is unambiguous, otherwise provenance-labelled fallback. | Unique mapping, ambiguous/missing mapping rejection, global fallback provenance, sequence prior-selection test. | open |
| I05 | Important — prior fallback | `allow_template_fallback=True` merely suppresses respiratory error and returns empty bands; no fallback source/value reaches report. | `frequency/training_prior.py`; config | Control flag changes error path without data. | Keep false/fail-fast or require/report real configured fallback. | Missing verified band false→fail; true+specified fallback→resolved report. | open |
| I06 | Important — Eq.8 baseline provenance | Nearest adjacent equal-width rule is labelled paper-derived but pairs lack cardiac band, source location and resolution. | `frequency/training_prior.py`; provenance/report docs | Rule added without complete source audit/serialization. | Re-read DREME source; label as necessary adaptation if needed and store pair records. | DC/physiology avoidance and JSON/report pair roundtrip. | open |
| I07 | Important — frequency granularity | `_centers()` evaluates one frequency per interval, independent of duration/resolution/timestamp distribution. | `losses/frequency_loss.py`; `frequency/training_prior.py` | Prior representation too coarse for formal loss. | Resolve multi-frequency grids for separated bands; report them. | Wide band resolves multiple values; separated bands remain separate. | open |
| I08 | Important — NUDFT | Uses absolute float32 time, no count normalization, no centering/DC policy, no irregular-weight decision or resolution report. | `losses/frequency_loss.py`; `training/trainer.py` | Direct formula lacks numerical/sampling specification. | Relative time; documented equal/interval weights; count scale; explicit DC behavior. | `t` vs `t+40000`; 20/50 scaling; jitter; constant-offset tests. | open |
| I09 | Important — temporal auxiliary | Sampler selects one location but returns only items; FiLM loops frames; no location stats or unresolved-band skipped reason. | `training/sampler.py`; `training/trainer.py` | Sequence is modeled as scalar loss, not traceable temporal record. | Typed sequence with location/time stats/resolved prior; batch FiLM; preflight sufficiency. | Full-span/order/QC-only and insufficient duration/band reporting. | open |
| I10 | Important — hard QC | Producer has semicolon whitelist parser, but training does not assert every `qc_valid=false` reason is a hard-invalid token; malformed external table can exclude soft QC reasons. | `outlier_qc/acquisition_qc.py`; `data/dataset.py`; `training/sampler.py` | Consumer does not enforce producer policy. | Reuse parser/whitelist at training input; reject invalid soft reason. | Multi-reason; false soft-reason fail; hard-invalid never data/temporal. | open |
| I11 | Important — fidelity/image-reg/spec | S1/2 MSE, S3 NLL match current config stage3, but image regularizer mode=`edge` and its semantics are hardcoded/partial config. Main spec §18.5 permits late S2 uncertainty while validator locks stage3; this conflict is unrecorded. | `training/trainer.py`; `training/model.py`; config; main spec §§18–20 | Hyperparameters and formal interpretation not one explicit runtime contract. | Reconcile once; configure/report data term, image-reg mode/args, uncertainty enable stage. | Analytical S1/S2/S3 data+reg totals and enable-stage contract. | open |
| I12 | Important — config/source coverage | Validator checks labels/commits only; dependency validator omits uncertainty import; no proof every formal config key is consumed once. | `training/source_first_config.py`; script; config | Validation ≠ factory consumption. | Schema/factory all-key audit and enabled primitive identity tests. | Unknown-key and consumption test; uncertainty source identity. | open |
| I13 | Important — geometry | No `DicomPlane.pixel_to_world()` vs `_pixel_world()` roundtrip for oblique/anisotropic planes; FiLM uses AABB `extent[:2]`, not row/column projected extent. | `training/model.py`; `geometry/world_geometry.py`; `models/film_motion_encoder.py` | Separate geometry implementations were not checked together. | Share plane path; use projected extents; report them. | Axis/oblique/anisotropic equality, PSF-basis and projected-extent tests. | open |
| I14 | Important — smoothness | Runtime hardcodes `4³`, combines resp/card on canonical grid, and returns `smooth_card=smooth*0`; YAML says `16³`. | `training/model.py`; config | Placeholder regularizer in formal mainline. | Separate configured physical full-FOV resp and local-box card grids. | Constant/linear/high-frequency/resolution-stability/branch gradients. | open |
| I15 | Important — optimizer groups | One Adam with global `1e-3`; no formal per-stage groups/LRs. | `training/trainer.py`; config; script | `requires_grad` toggles substituted for optimizer policy. | Configured documented stage groups/LRs. | Effective group/LR and disabled-param absence. | open |
| I16 | Important — optimizer state | `_configure_stage()` recreates Adam, losing canonical moment estimates. | `training/trainer.py` | Transitions treated as independent runs. | Preserve or explicitly migrate state. | S1→S2 canonical state/value continuity; new level state onset. | open |
| I17 | Important — dynamic identity/timestamps | IDs are token-dict indices without duplicate rejection; timestamps accept `float()` only; no finite/order/midnight policy; invalid rows consume embedding IDs. | `scripts/train_source_first.py`; `training/sampler.py`; dataset | Manifest authority invariants omitted at handoff. | Validate token/key uniqueness, range, finite/unwrap-or-reject timestamps, valid policy. | Duplicate/NaN/Inf/midnight; invalid excluded; deterministic IDs. | open |
| I18 | Important — domain/coverage | Full-FOV config validation exists, but no preflight validates bounds/card box/coverage artifact semantics or proves sparse plane-center zeros are not canonical holes. | config validator; script; coverage artifacts | Earlier coverage work not wired into final runtime gate. | Schema-check/report plane-centre QC separately from PSF/view/observation coverage. | Full FOV/card local and sparse-zero vs PSF-supported semantics. | open |
| I19 | Important — normalization | `per_series` falls back to `view:slice_id`; report exposes only bounds, no group membership/counts/valid stats; no real data check that UID is not per-frame; concatenates all full arrays. | `data/dataset.py`; script; config | “Per series” is candidate policy, not validated provenance. | Preflight group diagnostics/cardinality and streaming percentile estimator. | UID-degenerate fail; stable bounds; approximate/exact error bound. | open |
| I20 | Important — uncertainty integration | Uses NeSVoR builder and GaussianNLL, but no proof `[N,S,Z]→[N]`, one frame variance exactly once, or S3 NLL gradients to canonical+uncertainty+active motion. | `adapters/nesvor_uncertainty.py`; model; trainer | Unit adapter lacks composed shape/gradient audit. | Reconfirm pinned NeSVoR axes; enforce/report explicit shape adapter. | PSF-axis/variance equality, NLL equality, three-path gradient test. | open |
| I21 | Important — loss-total/Eq.7 | Conditional additions are distributed; no analytical total test. Stage2 Eq.7 gets three channels with no explicit active-channel contract. | `training/trainer.py`; `losses/motion_loss.py`; model | No single declared loss composer. | Named stage composer and active score selection, report weights/terms. | Distinct injected constants for all stages; huge inactive-score invariance. | open |
| I22 | Important — metrics/reporting | Only first/last total and last components are saved; no per-step LR, view, activation, DVF/MBC/score/variance stats or finite/explosion monitor; “effective config” is YAML. | trainer; script; reports | Reporting predates closure contract. | Structured metrics with first/last/mean/min/max and engineering warnings. | Schema/runtime-value/NaN-warning tests. | open |
| I23 | Important — checkpoint/resume | `source_first_last.pt` has model+report only; no optimizer/stage/steps/prior/norm/RNG/`--resume`. | script; trainer | Weights dump described as checkpoint. | Complete save/load resume or name/document weights-only. | Resume next-step equality under fixed seed; schema roundtrip. | open |
| I24 | Important — seed | No config/CLI seed; sampler alone uses `random.Random(0)`; pixel/PSF torch randomness, NumPy/CUDA state unseeded/unreported. | config; script; sampler; PSF adapter | Seed only treated as test utility. | Unified Python/NumPy/Torch/CUDA/sampler policy and RNG checkpoint state. | Fixed seed same first step; changed seed difference; resumed next sample. | open |
| I25 | Important — stage ownership/performance | No factory-level call/grad/value/optimizer/state test verifies the exact S1/2a/2b/2c/3 ownership table. | model; trainer; existing smoke tests | Existing tests do not distinguish “zeroed” from “not computed”. | Instrument source call counters and assert full ownership matrix. | Required §31 counters plus state/value checks. | open |
| I26 | Important — regression gate | Historical full discovery: `105 passed, 1 skipped, 7 failed` (five data fixtures contaminated by three-view validator; acquisition-QC all-invalid and reference temporal-mean expectations require policy classification). | `tests/test_data.py`; `tests/test_acquisition_qc.py`; `tests/test_reference.py` | Tests and production boundaries changed independently. | Classify against v3, decouple loader vs pipeline validation; no mechanical expected rewrite. | Seven targeted then full discover 0 unexpected. | open |
| I27 | Important — prompt archives | `docs/prompts/v3_change2_codex.md` is `120000` symlink; v3_change1 archive absent; root prompts are regular files. | `docs/prompts/`; root prompts | Incomplete link archive. | Copy all three as regular `100644`, retain roots. | `stat` + byte equality. | open |
| I28 | Important — changelog | Required append-only baseline `2e8975…` record has not been verified present. | `changelog.md` | Manual baseline not traceable. | Append factual baseline; final SHA only after repair. | Exact SHA entry once. | open |
| I29 | Important — clean checkout dependencies | `requirements-source-first.txt` has `omegaconf/ipdb/termcolor`, `pyproject.toml` omits them; no clean-environment all-adapter import test; validator omits uncertainty. | requirements; `pyproject.toml`; validator; third party | Declarations/import closure drift. | Reconcile install docs/package metadata and add CPU source import test. | Fresh documented env import all enabled adapters and lock verify. | open |
| I30 | Important — source lock/report | Hash verifier works, but no integrated proof lock/provenance/import roots agree under clean checkout; report has no explicit source-lock contract beyond dumped dict. | `third_party/SOURCE_LOCK.json`; `adapters/source_lock.py`; provenance; script | Integrity check not a full runtime provenance test. | Record verified state, files, commits/import roots in effective report. | Tamper detection and clean report identity. | open |
| I31 | Important — real-data preflight | No preflight script/report validates input schemas/counts/QC/time/norm/domain/frequency/config/source lock and no-grad forward per stage before training. | `scripts/`; train script | GPU gate checks not productized. | Add preflight with per-location usable/skipped/fallback decisions. | Synthetic report schema; actual CPU preflight when artifacts supplied. | open |
| I32 | Important — post-fix audit | No independent post-fix document can exist before repairs. | `docs/audits/` | Closure process incomplete. | Fresh call-graph re-audit after fixes; repeat until C/I zero. | Independent checklist/repo search. | pending |
| I33 | Important — unified smoke | Existing smoke does not prove multi-view/multi-location/timestamp S1→2a→2b→2c→3 ownership/calls/updates. | `tests/test_unified_progressive_smoke.py`; `tests/test_v3_change1_training.py` | Previous smoke predates current gate. | Upgrade tiny CPU synthetic pipeline after fixes. | All stages finite, expected calls/updates, inactive unchanged, shape correct. | open |
| M01 | Minor — legacy boundary | Legacy Stage1A/1B/initial-reference code remains as required but is not prominently labelled legacy in entry docs. | `training/stage1.py`; docs | Cutover preserved code but discoverability weak. | Label legacy/ablation without deletion. | Mainline routing excludes legacy. | open |
| M02 | Minor — FOV report | Policy is validated but reports do not explicitly state full FOV vs local cardiac role/bounds. | config; script | Domain policy only validation. | Add bounds/role report fields. | Report contract. | open |
| M03 | Minor — schedule provenance | Steps live in CLI defaults while related schedule values live in YAML; no one resolved schedule record. | script; config; trainer | Split configuration. | One resolved schedule with override provenance. | CLI/YAML precedence. | open |
| M04 | Minor — documentation drift | README/implementation/provenance claims need a final evidence pass after runtime repair. | README; `IMPLEMENTATION_REPORT.md`; provenance | Docs are ahead of closure. | Update only after tests/preflight, separating run/static/GPU pending. | Docs/effective-config consistency. | open |
| M05 | Minor — test provenance | This is static inspection; no fresh runtime command is claimed, and current reports do not categorize run/static/GPU-pending checks. | audit/report docs | Execution provenance not first-class. | Record commands/env/results after repair. | Report schema. | open |

## Frequency-prior reconciliation record (I04)

This is deliberately not a preselected implementation answer.

1. The main v3 spec requires Phase‑1 per-slice diagnostics and consensus bands,
   and §19.5 requires true timestamps for Stage 2/3 frequency losses.
2. The actual Phase‑1 schema is more specific: it retains `per_slice_candidates`
   and `union_resolution_bins_hz`; respiratory additionally has
   `verified_band_hz`; its source states cardiac is not forced to one global
   point estimate for sequential scans.
3. Current aggregate `slice_key` is not yet proven to map unambiguously to the
   training `view + slice_id` identity.

Repairs must inspect an actual Phase‑1 artifact and reconcile it with the formal
spec before implementing location lookup. If identity is ambiguous, a
provenance-labelled subject/recurrent fallback is safer than inventing a
per-location mapping. The post-fix audit must record the selected schema and
fallback rule.

## Baseline gate evidence

- Historical discovery evidence: `105 passed, 1 skipped, 7 failed`; it is not
  a passing gate and was not repeated during this audit-only phase.
- No new CPU runtime test, long training, GPU training, or real-data preflight
  was run in this phase.
- Section 41 conditions remain unmet: **`GPU_GATE = FAIL`**.
