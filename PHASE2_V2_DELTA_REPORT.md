# Phase 2 v2 delta

Base: `598c6c6 feat(phase2): add canonical domain and P0 foundation`.

## Added / changed

- `reference/build_mean_slices.py`: valid-frame-only temporal means, stable
  mean-slice IDs, source geometry, contributing opaque frame IDs, and no
  observation for a whole hard-invalid location.
- `rendering/psf_renderer.py`: NeSVoR-style oriented anisotropic PSF renderer
  using in-plane FWHM `1.2 * pixel spacing` and through-plane FWHM equal to
  DICOM slice thickness.  GL5 remains a separate ablation.
- `models/sinr_mbc.py`: control-coordinate SINR displacement generator feeding
  the retained cubic B-spline FFD primitive.  The module records paper spacing,
  grid-interval semantics, and actual physical-mm control-point spacing.
- `models/uncertainty.py`: separate `mean_slice` and `dynamic_frame` ID
  namespaces plus explicit unit-variance warm-up mode.
- `models/film_motion_encoder.py`: single-frame, full-geometry Fourier-FiLM
  encoder returning `resp_scores[B,3,3]` and `card_scores[B,1,3]`.
- `losses/stage_aware.py`: masked Stage-1A MSE, Gaussian NLL, and compact MBC,
  score, DVF, TV, and timestamp-aware spectral-leakage primitives.
- `cardioresp_motion.py`: only corrected the score adapter so reconstruction
  gradients reach FiLM scores; its sequential pullback convention is unchanged.

## References actually viewed

No external repository source file was fetched.  The v2 specification supplied
the required FWHM/covariance, SINR, and likelihood contracts unambiguously, so
no additional reference reading was needed.

## Verification

`tests.test_v2_delta`: 5/5 passed.  It includes static mean-slice behavior,
PSF constant/identity/oblique/gradient checks, SINR zero/physical-mm/boundary
checks, namespace isolation, FiLM output/backward, GaussianNLL numerical
agreement, and one dynamic PSF→motion→INR→uncertainty→NLL backward smoke test.
New modules compile; mean-slice CLI help and `git diff --check` passed.

## Unresolved method issue

None for the implemented P0 interfaces.  Frequency leakage is implemented on
the existing actual timestamps and intentionally does not repeat PCA/FFT
discovery; exact training weights and progressive schedule remain trainer work.

## Next

Implement Stage 1A/1B trainer scaffolding and perform the first short real
Stage-1 warm start/refinement run.  No real training was started here.
