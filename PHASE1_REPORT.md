# CardioResp 4D MRI — Phase 1 verified implementation report

Date: 2026-09-02. Scope is strictly pre-network Phase 1; no model, renderer,
loss, training, or inference code is included.

## Outcome and real acquisition

The local read-only acquisition completed the ordered pipeline
`inspect -> manifest -> geometry -> frequency -> roi -> reference` without a
mock or fallback. The ignored subject config contains the machine-specific input
path and empirical cardiac box; neither is tracked.

- 7,200 selected frames in 144 fixed-slice series, exactly 50 frames per series.
- SAX: 50 slices / 2,500 frames; 2CH: 52 / 2,600; 4CH: 42 / 2,100.
- `AcquisitionTime` is authoritative. Median interval is about 0.171 s; per-slice
  observed span is 8.363--8.364 s (`N*dt` nominal duration 8.550 s). The
  sequential three-view scan spans 3,564.265 s overall.
- SAX and 2CH matrices are 240 rows x 213 columns; 4CH is 213 x 240.
  Pixel spacing is 1.50117373 x 1.50117373 mm. DICOM tags report 8 mm slice
  thickness and 10 mm spacing between slices.
- Every selected frame has ImagePositionPatient, ImageOrientationPatient,
  PixelSpacing, dimensions, and acquisition time. All 7,200 frames omit both
  RescaleSlope and RescaleIntercept; standard pydicom modality-LUT semantics
  therefore preserve stored pixels and record `identity_without_rescale_tags`.

## Delivered modules and data flow

### Data and configuration

- `src/cardioresp4d/data/inspect_dataset.py`: read-only DICOM header scan ->
  PHI-free structural JSON.
- `src/cardioresp4d/data/build_manifest.py`: selected headers -> canonical
  CSV/JSON manifest, ordered by AcquisitionTime.
- `src/cardioresp4d/data/dataset.py`: manifest row -> normalized float32 image,
  timestamp, view/slice identity, geometry, and explicit rescale status.
- `src/cardioresp4d/config.py`, `configs/default.yaml`, and
  `configs/subject_example.template.yaml`: the only YAML contract is now grouped
  into `project`, `data`, `geometry`, `frequency`, `roi`, and `reference`.
  Scientific v1 constants (three views, exact 50 frames, frequency bands,
  dominance threshold, full PCA rank, SAX reference) cannot be silently changed.

This image-domain DICOM boundary is a necessary adaptation. Absolute paths and
patient-specific values remain only in ignored `configs/subject_local.yaml`.

### Patient-world geometry

`world_geometry.py` implements the DICOM PS3.3 public `(column,row)` convention:

`world = IPP + column * PixelSpacing[1] * IOP[:3] + row * PixelSpacing[0] * IOP[3:]`.

Raw stored IOP values are preserved. A two-vector Gram solve makes the in-plane
inverse stable under rounded direction cosines. `coordinate_normalization.py`
maps the union of all plane corners to `[-1,1]^3`; `geometry_qc.py` writes a
three-view 3D plane audit.

- 144 unique planes; nine representative planes rendered.
- pixel -> world -> pixel maximum error: `8.54315e-14` pixel.
- world -> pixel -> world maximum error: `1.72919e-13` mm.
- Patient LPS corner bounds: approximately
  `[-170.208,-211.475,-214.451]` to `[274.002,155.821,215.366]` mm.
- Visual inspection confirmed physically intersecting SAX/2CH/4CH plane
  families. Geometry is accepted for Phase 2.

This is the NISF++-style patient-world grounding, adapted to the local DICOM
manifest. The exact DICOM coordinate formula takes precedence over shorthand
network-space descriptions in the design document.

### Fixed-slice image-domain PCA / FFT / PSD

For each fixed slice, the 50 x pixels matrix is temporally mean-centred and thin
SVD is computed. Temporal scores are `U*S`. A one-sided SciPy periodogram uses
actual AcquisitionTime sampling; non-uniform sampling beyond the explicit 1.1
ms DICOM quantisation tolerance is rejected. PCA is frequency discovery only,
not the final motion representation.

- All 144 slices produced finite `pca_psd.npz`, temporal-PC CSV, spectrum CSV,
  and QC PNG files; no slice was omitted.
- Frequency resolution is `0.116959 Hz`; Nyquist is `2.923977 Hz`.
- PC1 explained variance: min/median/max
  `0.14597 / 0.34370 / 0.52132`; cumulative first three PCs:
  `0.34276 / 0.60270 / 0.71517`.
- Following the user's decision, an external 10-second rule is **not** a hard
  rejection gate. Positive, dominance-qualified respiratory and cardiac
  candidates are emitted normally. Respiratory candidates are present for all
  144 slices: 0.11696--0.58480 Hz, median 0.35088 Hz. Their merged resolution
  interval is `[0.05848,0.64327]` Hz.
- Cardiac candidates are present for all 144 slices: 1.05263--1.87135 Hz,
  median 1.52047 Hz. Per-slice values and resolution bins are retained rather
  than fabricating one global heart rate over a roughly 59-minute sequential scan.

Important caveat: 8.36 s and 50 frames provide only 0.11696-Hz bins, so the
reported respiratory distribution is usable discovery evidence but not
sub-bin precision. Duration, `df`, selected PC, dominance, peak power, and
explained variance remain in JSON. The implementation follows the supplied
image-domain PCA paper for mean-centred PCA/FFT and adapts its evidence into
DREME-style frequency bands.

### Shared cardiac coordinate box

The initial DREME-style empirical patient-world box is centred at
`[25,-55,43]` mm, has size `[120,120,120]` mm, and bounds
`[-35,-115,-17]` to `[85,5,103]` mm. It is one shared axis-aligned support,
explicitly **not segmentation**.

- SAX `SAX_s28_3501`: centre-to-plane distance 0.700 mm; six intersection vertices.
- 2CH `2ch_s24_8401`: 0.945 mm; four vertices.
- 4CH `4ch_s14_12601`: 0.431 mm; four vertices.

All true box-plane intersection vertices lie inside the selected images. Visual
inspection of the three temporal-mean overlays confirms that the same box covers
the visible cardiac region. The DREME-MR local-coordinate concept is direct;
patient-world three-view projection is a NISF++-style necessary adaptation.

### SAX initial reference

Following S2V-DREME Stage I, every SAX location is averaged over its 50 frames,
physically sorted, and stacked without 2CH/4CH fusion. DICOM `(row,column)` is
transposed to output `(column,row,slice)`.

- Reopened NIfTI shape `(213,240,50)`, float32, all finite; range `[0,1]`,
  mean `0.2316333`.
- Measured voxel spacing is approximately
  `[1.501174,1.501174,1.999998]` mm.
- Metadata stores both the authoritative DICOM LPS affine and the NIfTI RAS+
  affine, with `RAS = diag(-1,-1,1,1) @ LPS`.
- Physical IPP steps are 1.999990--2.000010 mm and affine residual is only
  0.0001005 mm. This measured geometry deliberately overrides the apparently
  nonrepresentative 8-mm thickness / 10-mm between-slice tags for the stack
  affine; all values remain recorded for audit.
- Central and orthogonal QC panels are anatomically continuous and recognizable.

Temporal averaging/stacking is direct from S2V-DREME. LPS/RAS conversion,
overlap-aware measured spacing, and QC are necessary adaptations. No complex
multi-view fusion was introduced.

## Runner and output contract

`scripts/run_pipeline.py` calls module public APIs only and supports inclusive
`--from-stage` / `--to-stage` ranges. Starting after manifest requires the
existing manifest artifact and fails clearly if it is absent. The complete real
run used a new ignored output root and regenerated all stages. Audit counts were:
6 JSON, 289 CSV, 144 NPZ, 149 PNG, and one NIfTI; all numerical arrays were
finite and all PNG files were non-empty. Representative geometry, three ROI,
three PCA/PSD, and reference images were inspected visually.

## Verification

Commands were executed in conda environment `knesvr_torch`:

```text
python -m unittest discover -s tests -v
CARDIORESP4D_REAL_DICOM_ROOT=<read-only-root> python -m unittest discover -s tests -v
python -m compileall -q src tests scripts
python scripts/run_pipeline.py --help
python -m <each Phase-1 module> --help
python scripts/run_pipeline.py --config configs/subject_local.yaml
```

Final result: 45/45 tests passed including the environment-gated real first-frame
loader; compileall passed; all eleven CLI help boundaries passed; the complete
real run passed. Synthetic tests cover DICOM ordering/rescale, anisotropic
geometry and round trips, known respiratory/cardiac signals, irregular-time
rejection, box projection/intersection, reference ordering/affines, nested
configuration, runner order/ranges, and missing dependencies.

## Git traceability

Branch: `dev/cardioresp4d`. Stage-0 tag: `stage-00-baseline`.

- `14dffa8`, `d3ed30a`, `b7ed05e`: data boundary and review fixes.
- `5456d10`, `a099967`: geometry and raw-DICOM-geometry fix.
- `108d2ef`, `7692aa7`, `6b19172`: PCA/frequency and review fixes, including
  normal extraction of short-window respiratory candidates with precision caveat.
- `fcf4ed7`, `0404d11`: cardiac box QC and exact-50 API enforcement.
- `38e045f`: SAX temporal-average reference.
- Task-6 integration commit and verified annotated tag are pending independent
  final review; the controller creates `phase-01-data-geometry` only after that review.

No DICOM, NIfTI, NPZ, runtime result, ignored local config, secret, or local
absolute path is tracked.

## Known limitations and Phase 2 plan

- PCA bands are discovery priors with coarse 0.11696-Hz resolution; they are not
  ground-truth respiratory/cardiac traces and do not replace learned motion scores.
- Slice series were acquired sequentially over about 59.4 minutes. A global point
  heart rate would be scientifically misleading; per-slice distributions are retained.
- The cardiac box is deliberately generous and empirical, not segmentation or an
  assertion of optimal support.
- The reference is anisotropic/overlapping SAX temporal average, not isotropic
  reconstruction or multi-view fusion. Per-frame percentile normalization may
  alter absolute intensity scale.
- No GPU work was required locally. Phase-2 code must remain server-compatible.

After explicit user approval only, Phase 2 will implement the P0 canonical
hash-INR, 3-level respiratory and local cardiac B-spline MBCs, sequential
deformation, thick-slice renderer, FiLM encoder with 9+3 scores, specified losses,
and forward/backward/checkpoint smoke tests. No Phase-2 module has been started.
