# Phase-1 acquisition outlier QC increment

## Delivered scope

- Added `src/cardioresp4d/outlier_qc/acquisition_qc.py` and focused tests.
- Canonical manifest is schema v2 and path-free. Opaque frame tokens resolve only
  through an explicitly sensitive runtime sidecar in ignored results.
- Added root/config/CSV fingerprints, strict manifest restoration validation,
  exact 50/52/42 series and all-50-frame geometry/Instance/Temporal contracts.
- Added the runner `qc` stage and durable `pipeline_run_summary.json`.
- Dataset, PCA, ROI temporal means, and initial reference consume `qc_valid`.
  Initial reference uses the arithmetic mean of valid frames; it never fills an
  entirely invalid slice with zero.

No original DICOM was changed or deleted. No Phase-2 code was added.

## Transparent criterion

For each fixed 50-frame block, after the reader's actual modality scaling:

1. temporal reference = pixelwise median;
2. NCC against that reference;
3. robust median-ratio global intensity scale;
4. normalized absolute residual after scale correction;
5. within-block median/MAD flags.

A global-scale flag additionally requires a large absolute log-scale excursion
so normal cardio-respiratory motion and modest first-frame transients are not
removed. Slice-location QC computes temporal means from rescaled (not
per-frame-percentile-normalized) images, sorts locations by patient-world
plane-normal position rather than opaque IDs, and applies the robust stack-scale
check even at a physical boundary. This explicitly covers a whole dark 50-frame
location; ordinary frame-level median QC cannot observe that failure mode.

## Tests and minimal real result

Focused synthetic tests passed for a normal moving 50-frame block, global signal
drop, local bright corruption, PHI-free table fields, and reference exclusion of
one invalid frame. A real pathless manifest validated 7,200 frames in 144 groups:
SAX/2CH/4CH = 50/52/42, all with 50 unique temporal/instance positions and stable
geometry.

Real QC sampled nine opaque blocks (three per view; 450 frames). Result:
450 valid, zero automatically rejected. The lowest-NCC candidates were retained
because their scale/residual pattern was consistent with normal motion. A SAX
frame-0 scale transient (~1.083, NCC ~0.955, residual ~0.148) was initially a MAD
extreme but was correctly retained by the conservative large-absolute-change
guard. Therefore this limited sample provides no confirmed RF-corruption claim.

The later directed check used the original SAX location IDs requested by the
user: `s10`--`s20` plus `s9/s21` controls (13 locations, 650 frames). It
rejected exactly one `SAX_s14_2101` frame for low NCC; all other frames and all
target locations remained valid. The 13-layer minimal reference then used 49
valid frames for `s14`, with no imputation. This check also established that
roughly 2-mm IPP centre spacing is stack geometry, while DICOM `SliceThickness`
remains 8 mm for the future thick-slice renderer. No QC threshold changed.

## Frequency and Git status

The 144 saved per-slice PCA candidates were re-aggregated without rerunning PCA.
Adjacent FFT bins are kept distinct, so the respiratory verified interval is
`[0.29240,0.40936]` Hz with support `78/144 (0.54167)`, rather than a chained
wide union. The maximum cardiac-bin support is `53/144 (0.36806)`, below the
0.5 consensus threshold; only its per-slice candidates are retained. The
10-second rule remains a caveat, not a hard null gate.

The independent commit is intended as
`feat(qc): add acquisition outlier filtering for corrupted MRI frames`.
The supplemental `phase-01-qc-outlier` tag is pending controller verification;
no existing history/tag is rewritten.

## User decision needed

No blocking decision is required. A partial directed QC table is deliberately
ineligible for downstream use: downstream requires one decision per manifest
frame. Any later Phase-2 real subset must use a matching complete QC table for
that subset; threshold tuning remains plot-audited rather than automatic.
