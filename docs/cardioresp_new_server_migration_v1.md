# CardioResp new-server migration v1

`CARDIORESP_DISENTANGLE_RUNTIME_V1` is the minimal named bundle: the Git repository at the baseline commit recorded in `code_identity.json` (this tool was developed from `da3ca42e0237bc660c405b483cc484271a49ea3f`; use the completed migration commit for tools), the authoritative files and SHA256 inventory, environment capture, selected 2,000-step S05/S05SAX/S05BM checkpoints and compact run provenance, and the preflight scripts. Exact dataset paths and final sizes cannot be known until the current-server audit runs. The future branch should start from the verified consolidated baseline including this migration commit.

The selected future forward model is corrected physical LPS geometry and deterministic five-point Gaussian-weighted Gauss–Legendre thick-slice quadrature. S0 is a native SAX baseline with weak latent-to-DVF response. S05SAX has the best SAX quality to date, with respiratory motion present but cardiac motion weak; S05 is the physical multi-view base, also with weak cardiac motion. S05BM is the ongoing matched-SAX-exposure control. S1 is a diagnostic comparator: frame-to-frame stochastic PSF sampling strongly inflated apparent temporal variation. Do not choose S1's stochastic eight-sample renderer as the next-stage default. Explicit respiratory/cardiac disentanglement is later work.

## A. MUST COPY — authoritative runtime

For each actual S05 `experiment_inputs.json`, inspect its `paths`: manifest CSV, QC CSV, canonical-domain JSON and static-reference manifest. The observation loader requires the CSV's adjacent `<stem>.sensitive_runtime_paths.json` and **every DICOM it references**, including invalid frames, because `observations_from_manifest` loads all rows; it extracts pixels and geometry and recomputes per-series normalization from QC-valid frames. The sidecar and inventories can contain sensitive identifying paths. Protect and review them before sharing.

The static manifest points to a corrected-pose JSON and static NIfTI volume. Poses are inline in the current schema; the volume is read to validate physical spacing. The checked-in NeSVoR and other source-locked vendor trees are also required: S05 modules and the shared observation/evaluation entry points import them even when the NeSVoR renderer is not used. The S05 trainer does not initialize from the static model, but the unified evaluator's `load_inputs` loads and hashes that model, so this bundle includes it. The same loader's config/bootstrap dependencies and vendored code come from Git. A crop/ROI is not a mandatory training input; evaluator dynamic crop is optional. The observation manifest contains physical DICOM geometry and timestamps; the sidecar maps tokens to raw images.

Dependency graph:

```text
S05 experiment_inputs.json
 ├─ manifest CSV ─ sidecar JSON ─ per-frame DICOMs (all rows)
 ├─ QC CSV
 ├─ canonical-domain JSON
 └─ static manifest ─ corrected-pose JSON
                    ├─ static NIfTI volume (S05 training and evaluator)
                    └─ static model checkpoint (unified evaluator)
```

The code has no universal root override for embedded paths. Checkpoint provenance records hashes of original manifest/QC/domain/static/pose/volume and vendor/config; R1 additionally embeds source paths. Evaluation manifests and experiment inputs also contain absolute paths. Prefer preserving the original absolute prefix where feasible, while keeping the data under a controlled mount; otherwise inspect `absolute_path_references.json`, create remapped copies, and explicitly validate all nested provenance hashes. **A rewritten static manifest or manifest CSV changes its SHA256 and will generally fail existing checkpoint resume/evaluation provenance.** The remap utility does not forge hashes or rewrite checkpoints. Preserve originals; do not assume a remapped checkpoint can resume. Token sidecar renaming must match the remapped CSV's stem exactly. The destination preflight currently verifies original inventory hashes and blocks on unresolved paths or hash mismatch; it does not certify a rewritten-provenance migration. Restoring the original absolute prefix is the supported route for READY and existing checkpoints.

## B. SHOULD COPY — baselines and reproducibility

S05-2000 and S05BM-2000 are candidate initializations; the choice of either or from-scratch initialization is not yet made. S05SAX-2000 is primarily a SAX comparison/diagnostic. Preserve numbered checkpoints and `training_report.json`, `training_metrics.csv`, `experiment_inputs.json`, `effective_config.json`. S1-2000 and PSF summaries are comparison-only; 10k/15k/20k checkpoints and extension summary are historical diagnostics if desired. R0/R1 and S0 compact metrics, provenance and selected checkpoints are historical/comparison items, not mandatory initialization. The audit loads selected trusted checkpoint files with `torch.load(..., weights_only=False)` on CPU to inspect provenance; do not run it on untrusted checkpoints. The audit includes explicitly supplied run directories; it does not guess all historical experiment locations. Add their selected files or rerun with user-selected optional archives if needed. `checkpoint_last.pt` should be omitted only after verifying it is byte-identical to a retained numbered checkpoint.

## C. OPTIONAL — visualization

Preserve compact PSF/temporal summary CSV/JSON and selected representative GIFs when useful. Panels, all GIFs, full evaluation `data/*.npz`, identity plots and full audit archives can be regenerated from preserved inputs and checkpoints. Explicitly supply `--optional-dir` only for archives actually wanted; directory enumeration can be expensive.

## D. DO NOT COPY by default

Do not transfer duplicate Git worktrees, `__pycache__`, `.pytest_cache`, build/pip/CUDA compilation/Matplotlib caches, routine logs, failed runs, obsolete S0/S1 render caches, redundant byte-identical `checkpoint_last.pt`, or large reproducible residual NPZ arrays. No files are deleted by these recommendations. Use Git to obtain tracked code after confirming the commit is available remotely.

## Transfer and environment

1. Capture environment; transfer code and metadata, manifests and geometry. A subset of DICOMs can only support a preliminary check; the existing loader reads all manifest rows and recomputes normalization, so full S05 preflight/training needs the full dataset. Do not silently substitute a subset for the scientific cohort.
2. Transfer all runtime DICOMs and static references, then the baseline checkpoint files and compact summaries.
3. Transfer selected optional figures/archives. Inspect `migration_summary.md` for missing paths, sizes, root escapes and the largest items. Files outside the declared data root require separate, reviewed transfer commands. Symlinks whose targets escape that root also require target handling; `rsync -aH` preserves links, it does not automatically include outside targets.

For each list run `rsync -aH --info=progress2 --files-from=LIST DATA_ROOT/ NEW_SERVER:TARGET/`; no `--delete`. Destination preflight verifies recorded checksums, sizes, vendor files, CUDA/TCNN and an actual tiny render. Files larger than 64 MiB lack SHA unless `--hash-large-files` was requested; checkpoints are hashed regardless. Treat these size-only checks as weaker integrity evidence.

Recreate conda from repository dependencies and `conda_from_history.yml`, select a PyTorch build compatible with the new GPU/driver, then install/rebuild tiny-cuda-nn on the destination. `conda_full.yml` is an audit artifact with machine-specific pins, not a universally portable recipe. Copying a whole conda directory is not the primary approach. `conda-pack`, if available, is an optional fallback and still carries GPU/CUDA and compiled-extension compatibility risks. Install `torchmetrics` for native S0 held-out evaluation. Test the full runtime before training.

For remapping, run `remap_cardioresp_manifest_paths.py` separately on each JSON/CSV into a new area, with a unique `--report` per file. It changes only strings beginning with the old prefix and never modifies source files. It does not rewrite paths embedded in command-line strings, nested dependent hashes or serialized checkpoints; inspect those manually. `path_remap_manifest.json` records original/new digests but is not an authorization to bypass provenance checks. The preflight's `BLOCKED_*` results must be resolved, not ignored.
