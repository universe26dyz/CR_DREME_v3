# v3_change3 Post-fix Closure Audit

## Scope, independent method, and outcome

Baseline reviewed: `2e8975e6d73862feda57bcaf6eeefc8d464ce8db` plus the
uncommitted v3_change3 repair worktree.  This is a new call-graph audit, not a
status edit of `v3_change3_pre_fix_audit.md`.  It starts at
`scripts/train_source_first.py` and traces configuration, manifest/QC/domain
inputs, observation identity, sampler, model factory, each progressive stage,
loss composition, optimizer/checkpoint/report, and the preflight entrypoint.

**Closure result: Critical 0; Important 0; Minor 0; `GPU_GATE = PASS`.**

`PASS` means the source-first runtime is eligible for the prescribed short GPU
Stage1 -> Stage2a validation.  It does **not** assert that a GPU run or a long
real-data reconstruction has been performed; neither was in scope.

```text
source_first.yaml + manifest/QC/domain/frequency + SOURCE_LOCK
  -> validate_source_dependencies / validate_dynamic_frame_rows
  -> observations_from_manifest / per-series normalization / balanced sampler
  -> build_source_first_model (actual object config)
  -> StageContract
       S1: PSF -> NeSVoR INR
       S2a/b/c: FiLM -> active upstream respiratory SINR -> pullback -> PSF -> INR
       S3: FiLM -> resp then local cardiac SINR -> pullback -> PSF -> INR -> NLL
  -> stage-aware data/image/motion/frequency losses -> persistent Adam groups
  -> metrics + resumable checkpoint + effective-config report
```

## Closure matrix

| IDs | Runtime evidence and root-cause removal | Regression / gate evidence | Result |
|---|---|---|---|
| C01, C02, C04, I01–I03, I14, I20, I21, I25, I33 | `training/stage_contract.py`, `training/model.py`, `training/trainer.py`, `adapters/sinr_mbc.py`, and `losses/motion_loss.py` now declare an active-stage contract. Only active upstream respiratory levels are invoked; no learnable `level_gates` exists. Stage1 does not call FiLM/SINR or motion losses, Stage2 has respiratory-only motion, and Stage3 adds local tapered cardiac motion and uncertainty. Eq.6 receives raw active levels and reduces batch/spatial axes while retaining Cartesian components. | `test_v3_change3_stage_contract.py`, `test_unified_progressive_smoke.py`, source-adapter tests: call counters, active branch ownership, boundary/zero-motion, raw-basis axes, finite forward/backward. | closed |
| C03, I04–I09 | `frequency/training_prior.py` reconciles the formal specification with the actual Phase-1 artifact: producer `slice_key` is exactly `view/slice_id`; reliable candidates use that unambiguous location identity, otherwise the global union is explicitly reported as `phase1_global_fallback`. `frequency_loss.py` resolves duration-aware multiple frequencies, computes relative-time float64 centred/count-normalized NUDFT, and uses explicit complex cardiac/baseline pairs. | `test_v3_change3_frequency_contract.py` checks schema mapping/fallback provenance, complex Eq.8 equality, multi-frequency resolution, DC/offset stability. CPU real-data preflight resolved 143 locations. | closed |
| C05, I11, I12, I15, I16, I22 | `training/build_model.py` is the unique config-to-object factory; `effective_model_config()` inspects the constructed source-backed objects. `source_first_config.py` validates formal keys/imports. `trainer.py` uses named per-stage Adam groups/LRs without losing prior state, named stage totals and structured step metrics. Image regularization and uncertainty are explicit object/config fields. | `test_v3_change3_factory.py`, `test_v3_change3_runtime_state.py`, full suite: constructor perturbations change objects/reports; group LRs and optimizer continuity are verified. | closed |
| I10, I17, I19, I23, I24 | `training/runtime_state.py`, `training/sampler.py`, `data/dataset.py`, and `train_source_first.py` enforce the two-reason hard-invalid policy, reject duplicate/nonfinite/nonmonotonic dynamic identities, keep invalid observations out of embeddings/training, report per-series groups, use deterministic per-frame normalization sampling, seed/capture RNG, and serialize model/optimizer/stage/sampler/RNG resume state. | `test_v3_change3_runtime_state.py`, data/QC tests: hard-invalid exclusion, identity validation, normalization groups, seed and checkpoint roundtrip. | closed |
| I13, I18 | `models/film_motion_encoder.py` uses DICOM row/column/normal projected acquisition extents; model motion regularizers separately sample full canonical respiratory and local cardiac boxes. Preflight checks full-FOV/card-box bounds and reports plane-centre coverage as geometry QC rather than canonical holes. | geometry/stage-contract tests cover oblique pixel world and acquisition scalars; preflight reports all three views and coverage semantics. | closed |
| I26–I31 | Low-level manifest validation is separated from full topology validation; legacy initial reference is explicitly legacy normalization. Prompt archives are regular files. `pyproject.toml` and requirements agree; source lock/import verification is integrated; `preflight_source_first.py` executes the no-step real-data contract. Changelog, README, provenance, and implementation report record the mainline cutover. | final full discovery, `pip check`, source-lock import check, `stat` archive check performed during repair, and final real-data CPU preflight below. | closed |
| I32, M01–M05 | This independent audit fulfills I32. README identifies legacy Stage1A/Stage1B/reference paths as ablations; effective config, bounds, schedule/metrics and test provenance are reported in the runtime outputs and implementation report. | document review plus final gates below. | closed |

## Source-first and semantic-equality checks (§0, §43, §45)

- `third_party/SOURCE_LOCK.json` verifies NeSVoR `2e96a91…`, SINR
  `1a524ca…`, and FiLM `fe43ddf…`; the final import/lock command passed.
- Runtime class identity in the real-data report was `nesvor.inr.models.INR`,
  `networks.networks.BSplineSiren`, and
  `models.transformation.CubicBSplineFFDTransform`.  Adapters only handle
  world-mm coordinates, dynamic IDs and sequential motion insertion.
- Config -> factory -> live-object introspection -> report and the stage
  contract are tested together.  Thus the source-first/paper semantics are not
  merely documented; they are the executed call graph.
- Legacy `training/stage1.py` and historical initial-reference code remain
  preserved but are outside the formal entrypoint; their retained
  `NotImplementedError` is a legacy/ablation-only search hit, not a mainline
  fallback.

## Final validation evidence

All commands below ran locally in `knesvr_torch` on CPU.

1. `env PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n knesvr_torch python -m unittest discover -s tests`
   -> **126 passed, 1 explicit skip, 0 unexpected failures** (5.938 s).
2. `conda run --no-capture-output -n knesvr_torch python -c "from cardioresp4d.training.source_first_config import validate_source_dependencies; validate_source_dependencies(); print('SOURCE_IMPORTS_AND_LOCK=PASS')"`
   -> `SOURCE_IMPORTS_AND_LOCK=PASS`; then `python -m pip check` -> `No broken requirements found.`
3. `scripts/preflight_source_first.py` using the DYL0709 manifest sidecar and
   the full Phase-1 QC/domain/frequency artifacts completed with `status=PASS`:
   7200 observations, 7150 valid, 50 hard-invalid, 144 normalization groups,
   143 location-frequency entries, verified source lock, and finite no-grad
   forwards for Stage1, Stage2a, Stage2b, Stage2c, and Stage3.  Its final
   report is `/tmp/crdreme_v3_change3_preflight_postfix/preflight_report.json`.
4. `git diff --check` completed cleanly.

Warnings about unavailable CUDA extensions/tinycudann are expected on this
CPU-only machine: pinned NeSVoR selects its PyTorch implementation and all
finite CPU tests/preflight use that path.

## Mandatory repository-wide search (§44)

Executed immediately before this audit:

```bash
rg -n "TODO|TBD|pass$|NotImplemented|fallback|4, 4, 4|learning_rate.?=.1e-3|score.square|dvf.*square|level_gates|frequency_bands|effective_config|motion_regularizers|GaussianNLL|cardiac_mbc|respiratory_mbc" \
  src scripts configs tests README.md IMPLEMENTATION_REPORT.md SOURCE_PROVENANCE.md
```

Classification of every category of hit:

- `fallback`: provenance/documentation and the explicit, labelled
  `phase1_global_fallback`; no silent template fallback is enabled in config.
- `4, 4, 4`: small synthetic/legacy test fixtures only; the source-first
  runtime resolves configured 8/12/16 respiratory and 16 cardiac grids.
- `NotImplemented`: `training/stage1.py`, retained legacy/ablation only.
- `score.square` / `dvf.*square`: explicit physical RMS/smoothness diagnostics
  and documented loss components, not a hidden progressive gate.
- `GaussianNLL`, `effective_config`, MBC identifiers, and
  `frequency_bands`: expected direct runtime interfaces and reports.
- `level_gates`: occurs only in negative regression assertions; none is in
  production source.  No production hard-coded global optimizer `1e-3` or
  hard-coded `4^3` smoothness grid is selected by the mainline.

## GPU gate and remaining work

**`GPU_GATE = PASS`** because Critical=0, Important=0, the final full suite
has no unexpected failures, and the final real-data no-step preflight passed.

Still pending by deliberate scope: copy this checkout to the GPU host and run
the short real-data Stage1 -> Stage2a command in `README.md`; then inspect its
metrics/report before any Stage2b/2c/3 or long reconstruction.  No GPU or long
training result is represented as completed here.
