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
removed. Slice-location flagging requires both an extreme robust stack intensity
and poor agreement with two neighbours; basal/apical difference alone is not
enough.

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

## Frequency and Git status

The 144 saved per-slice PCA candidates were re-aggregated without rerunning PCA.
Respiratory verified support was 144/144 (1.0) for the connected coarse-resolution
component. The 10-second rule remains a caveat, not a hard null gate.

The independent commit is intended as
`feat(qc): add acquisition outlier filtering for corrupted MRI frames`.
The supplemental `phase-01-qc-outlier` tag is pending controller verification;
no existing history/tag is rewritten.

## User decision needed

Choose whether to run acquisition QC over all 144 blocks before Phase 2. The
current conservative defaults are recommended initially; any threshold tuning
should be reviewed against QC plots to avoid deleting genuine cardiac or
respiratory motion.
