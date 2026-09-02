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
- a 50-frame, 8.55-s acquisition retaining a respiratory candidate when it
  has positive peak power and sufficient dominance, with the temporal PC and
  its PSD available in the regular outputs and an explicit duration/`df`
  precision caveat in aggregate JSON.
- a flat, all-zero 50-frame series returning null frequency, selected-PC,
  peak-power, and dominance fields with reason `no_positive_peak_power`, rather
  than exposing the arbitrary first `argmax` frequency bin as a candidate.

Fresh verification ran `python -m unittest discover -s tests -v`: 27 tests
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
2.2.  Respiratory and cardiac reliability are both determined by positive peak
power and this dominance threshold; the approximately 8.55-s recording is not
a hard respiratory rejection criterion.  The aggregate respiratory result has
144 reliable per-slice candidates with center frequencies from 0.11696 to
0.58480 Hz (median 0.35088 Hz).  `respiratory.verified_band_hz` is the merged
periodogram-resolution interval `[0.05848, 0.64327] Hz`, not a point estimate.
The JSON records the approximately 8.363-s observation span and 0.116959-Hz
`df` as an explicit caveat: use the merged resolution bins and do not infer
precision finer than those bins.

All 144 cardiac slice candidates passed the specified dominance threshold.  The
reliable cardiac distribution spans 1.0526--1.8713 Hz (median 1.5205 Hz) over
the sequential acquisition.  The JSON deliberately retains per-slice
frequencies, selected PC, peak power, dominance, reliability/reason, duration,
resolution, Nyquist, and explained variance, and reports spectral-resolution
union bins rather than collapsing sequentially varying data to one fabricated
heart rate.  The wider extrema are reported as observed, not hidden to fit a
preselected range.
