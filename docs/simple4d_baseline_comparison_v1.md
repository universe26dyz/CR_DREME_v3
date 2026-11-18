# SIMPLE-4D baseline comparison v1

S0 is the pinned upstream learning core, NGP, encoder, forward/inverse deformation and thick-slice quadrature, trained jointly from scratch on an admitted SAX-only regular parallel stack. S0 uses nominal acquisition geometry; corrected poses are provenance-only and are not applied. Corrected SAX planes can be rotated or translated and no longer satisfy the native regular-stack assumptions. The NPZ adapter checks common orientation and pixel geometry, a regular slice-position lattice and aligned in-plane centers. Missing lattice positions remain unobserved: no images are synthesized or interpolated. It does not model arbitrary patient-LPS planes or consume 2CH/4CH. `slice_thickness` is explicitly taken from admitted metadata. Input frame IDs/timestamps and geometry assumptions are preserved in the adapter manifest. The wrapper uses a checkpointed single-process randomized permutation and cursor; model updates directly call upstream `training_step`.

S1 uses upstream from-scratch InstantNGPINR, the original MotionCodeVAE without view embedding, and DeformCodeAdaptive in both directions. View-local normalized positions enter the encoder; the physical adapter maps external LPS xyz mm to normalized xyz. Corrected poses and the same NeSVoR PSF as R0/R1 provide the multi-view forward model. The static manifest is an authority for geometry and reference spacing only; NeSVoR model weights are never loaded. Joint optimization starts at step 1, with upstream hash/template/encoder/deformation learning rates and loss weights.

R0/R1 instead use a NeSVoR-initialized canonical, a view-conditioned encoder and staged frozen/joint training. R0/R1/S1 use NeSVoR PSF; S0 uses upstream regular-stack thick-slice quadrature and should be compared only on SAX overlap. These scopes are not directly interchangeable.

Formal S0 and S1 default to requiring TCNN; an explicit PyTorch fallback may be used for tests and is recorded as a distinct backend, not as numerically identical paper output. V1 uses target posterior mean; project V2 uses same-location neighboring posterior means and timestamp-weighted interpolation without the target code. Upstream held-out V2 uses a simple mean of adjacent acquisition-order codes, which is a different protocol and remains labeled upstream. The evaluator reports matched per-frame MSE/MAE/NCC, absolute residuals, shared [0,1] grayscale and one exact global residual maximum for each comparison/GIF set. Checkpoint hashes are checked before and after; evaluation output must be outside input directories. Optional common-crop dynamic anatomy applies only to physical project-coordinate methods; S0 native geometry is not presented as LPS-registered.

Run admission and 200-step smoke first; inspect geometry, backend, gradients, losses and checkpoint integrity before any longer run. These tests do not establish a real-data result or method superiority.

S0 is the native SAX-only upstream sanity baseline. S0.5 (S05) uses the same corrected-pose SAX+2CH+4CH physical learning problem, normalization, networks, sampler, optimization, losses and seeds as S1. S05 replaces only the renderer with five-point Gauss–Legendre quadrature along the corrected physical normal, weighted by a Gaussian with sigma equal to slice thickness / 4. Samples are deformed individually before canonical evaluation, as in upstream `SIMPLE4D.predict`. S1 uses the NeSVoR Gaussian PSF with eight samples. S05 therefore isolates the forward model.

S05SAX is a corrected physical SAX-only bridge using the same five-point renderer and learning core as S05; the only protocol change from S05 is that only QC-valid SAX observations enter the sampler. S05SAX versus S05 isolates view admission; S05 versus S1 isolates the renderer. S0 versus S05SAX is a bridge comparison, not a one-variable ablation, because native coordinates and runner also differ.

The evaluation-only S1 PSF audit compares independent eight-sample draws, common random numbers within each view/location, and a 128-sample approximation using fixed row-major chunks. It also repeats an unchanged V2 code 32 times. Differences between modes quantify sensitivity to evaluation sampling; artifacts remaining at high sample count do not by themselves distinguish training-induced model structure from forward-model mismatch.

## Experiment ladder and matched SAX budget

- **R0/R1:** historical NeSVoR-initialized staged respiratory models.
- **S0:** native SAX-only upstream sanity baseline.
- **S05SAX:** corrected physical SAX-only geometry with deterministic five-point quadrature.
- **S05:** corrected physical multi-view geometry with the same deterministic renderer.
- **S05BM:** the same S05 physical model and optimizer, sampling 8 SAX, 4 2CH and 4 4CH observations per update.
- **S1:** the same physical multi-view learning core with stochastic NeSVoR PSF.
- **S1-20k:** iteration-length diagnostic, not the selected main forward-model direction.

At 2,000 updates S05SAX and S05BM each see 16,000 SAX observations. S05BM also sees 16,000 long-axis observations and averages the loss over all 16 observations per update. This matches SAX sample exposure, **not** total compute or per-view gradient weighting. A close S05BM/S05SAX SAX result would support SAX exposure dilution as an explanation for the original difference; a persistent difference would be consistent with cross-view optimization or consistency costs, although the changed gradient weighting also matters. Better temporal fidelity with S05BM would support useful long-axis dynamic constraints. Compare MSE/MAE/NCC, temporal fidelity and the latent, DVF, cycle and Jacobian diagnostics separately; do not combine them into a score.

## Current observed diagnosis

These values describe the supplied completed server experiments, not new results from this implementation. For S0-2000, latent dynamic RMS was about 0.251, PC1 about 99.7%, rank95 1, mean DVF RMS 2.60 mm and dynamic DVF RMS 0.122 mm; SAX_s018 temporal-std and adjacent-difference ratios were about 0.08 and 0.04–0.05. This indicates weak latent-to-DVF response rather than latent collapse. S05-2000 had latent dynamic RMS 0.307, PC1 0.839, rank95 4, dynamic DVF RMS 0.316 mm, and SAX_s018 ratios 0.384 and 0.206.

S1-2000 independent-eight ratios were 0.868 and 1.311; common-random-number eight-sample ratios were 0.391 and 0.246, and high-128 ratios were 0.361 and 0.220. Much of the apparent high-frequency temporal variation is therefore consistent with evaluation PSF Monte-Carlo jitter; variance-reduced S1 amplitudes resemble S05. From 2k to 20k, S1 V2 reconstruction improved while cardiac temporal identity remained poor and deformation, cycle and Jacobian regularity worsened. There is no current reason to extend beyond 20k.

S05SAX-2000 produced SAX V2 MSE about 0.004108 and NCC 0.96854; temporal-std ratio 0.495, adjacent ratio 0.268 and temporal correlation 0.245. Its latent dynamic RMS was 0.419, PC1 0.914, rank95 3, dynamic DVF RMS 0.465 mm, static DVF RMS 1.277 mm, mean cycle error 0.670 mm and Jacobian p01 0.854, with no observed folding. It had the best SAX image quality among these methods and respiratory motion remained visible, while cardiac motion remained weak.

The selected main forward model for the next modeling stage is corrected physical geometry plus deterministic five-point thick-slice quadrature. Respiratory/cardiac motion disentanglement is a later stage.
