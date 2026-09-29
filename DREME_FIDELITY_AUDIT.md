# DREME-MR fidelity audit

This audit is based on the original [DREME-MR paper](https://arxiv.org/pdf/2503.21014), not an inference from the project README. `C4_NEGONLY` means **DREME-style negative-crossover-only baseline within the current CR_DREME_v3 image-domain architecture**; it is not an exact DREME-MR reproduction.

## A. Directly aligned ideas

- The model retains a full-FOV multi-resolution respiratory MBC, a local cardiac MBC, sequential cardiac/respiratory pullback, MBC normalization (Eq. 6), and zero-mean scores (Eq. 7).
- `frequency_loss.py` implements Eq. 8's respiratory-score cardiac-band leakage with complex paired-baseline subtraction, and Eq. 9's cardiac-score respiratory-band leakage. C4 leaves these negative-crossover terms at `1e-4` and sets the project-specific target-concentration/PCA terms to zero.
- Cardiac motion is initialized after respiratory motion. The C4 Stage3b rescue is the project stage that restores the paper's crucial Stage III intent: joint refinement of anatomy, encoder, and motion.

## B. Necessary project adaptations

- CR_DREME_v3 fits asynchronous DICOM 2D image-domain observations with patient-world geometry, NeSVoR canonical INR/PSF, SINR MBC, geometry-conditioned FiLM scores, and local Phase-1 timestamp frequency priors. It does not fit multi-coil radial k-space or use the paper's raw-k-space encoder/NUFFT data term.
- The paper uses its own image/registration setup and physical coordinate choices; the current cardiac crop, acquisition geometry, normalization, sampling, and PSF are retained rather than replaced.

## C. Material differences and interpretation boundary

- Paper Eq. 4 is k-space L1 data consistency plus Eq. 10 regularization; this repository uses acquired-image residuals and NeSVoR rendering. The paper's reported Eq. 10 weights are `λTV=2e-6`, `λMBC=1e-2`, `λZMS=1e-4`, `λc=1e-1`, and `λr=5e-2`; they are recorded only, never copied into image-domain C4.
- The paper uses a 500+1300 epoch Stage I, 850 epoch Stage II (including 50 cardiac-initialization epochs), then 3650 epoch Stage III; its Stage III activates all components. Current Stage3a deliberately freezes canonical/respiratory modules, so **Stage3a-only is not complete DREME-style training**. C4's short Stage3b 100+400 rescue is a mechanism test, not a reproduction claim.
- Change5A target concentration and Change5B PCA waveform supervision are project-specific additions. C4 removes both to isolate whether those additions, rather than negative crossover itself, are implicated.
