# Task 3 report: fixed-slice PCA, PSD, and frequency candidates

## Status

Implemented fixed-slice, image-domain PCA and frequency analysis in
`src/cardioresp4d/frequency/`.  The authoritative input is
`results/dicom_manifest.csv`; each series is explicitly sorted by
`timestamp_s` (derived from DICOM `AcquisitionTime`) before analysis and must
contain exactly 50 frames.

The PCA matrix is time by pixels after each pixel's temporal mean is removed.
NumPy thin SVD supplies temporal scores `U*S`; SciPy's one-sided periodogram
then operates on every temporal PC.  Ordinary FFT/periodogram processing only
accepts timestamp intervals within `max(0.1% of median dt, 1.1 ms)`.  The
1.1-ms absolute allowance is documented and tested to accommodate the
manifest's millisecond DICOM time quantisation.  There is intentionally no
irregular-time fallback.

## Test-first evidence

The first test run was RED because `cardioresp4d.frequency` did not exist.
The implementation was then added only to make the following independently
constructed tests GREEN:

- image modes with separately specified 0.24 Hz respiratory and 1.20 Hz
  cardiac sinusoids;
- rejection of a 20-ms timestamp discontinuity before periodogram use;
- acceptance of a 1-ms quantisation deviation under the explicit tolerance;
- null candidates when frequency resolution is insufficient; and
- a 50-frame, 8.55-s acquisition reporting a respiratory candidate as
  `limited_duration` and producing no verified global respiratory band.

Fresh verification ran `python -m unittest discover -s tests -v`: 26 tests
passed and one pre-existing optional real-loader test was skipped because its
environment variable was not configured.

## Real all-slice run

Command:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/cardioresp4d-mpl \
  python -m cardioresp4d.frequency.pca_motion \
  --manifest results/dicom_manifest.csv --output-dir results/frequency
```

The complete 144-slice run took about 33 seconds with NumPy thin SVD; no
slices were omitted and no lower-rank approximation was used.  It produced
144 each of `pca_psd.npz`, `temporal_pc.csv`, `spectrum.csv`, and `pca_qc.png`,
plus `results/frequency/frequency_bands.json`.

Sampling QC was approximately: median `dt=0.171 s`, span `8.363 s`, nominal
duration `N*dt=8.550 s`, `df=0.116959 Hz`, and Nyquist `2.923977 Hz`.

## Candidate interpretation and limitations

Following the supplied Shammi PCA criterion, the peak-dominance threshold is
2.2.  A respiratory peak can be listed per slice, but the approximately 8.55-s
recording is below the reported 10-s minimum for reliably tracking more than
one respiratory cycle.  Therefore the aggregate JSON sets
`respiratory.verified_band_hz` to `null` with reason `limited_duration`; it
does not claim a verified global respiratory band.

All 144 cardiac slice candidates passed the specified dominance threshold.  The
reliable cardiac distribution spans 1.0526--1.8713 Hz (median 1.5205 Hz) over
the sequential acquisition.  The JSON deliberately retains per-slice
frequencies, selected PC, peak power, dominance, reliability/reason, duration,
resolution, Nyquist, and explained variance, and reports spectral-resolution
union bins rather than collapsing sequentially varying data to one fabricated
heart rate.  The wider extrema are reported as observed, not hidden to fit a
preselected range.
