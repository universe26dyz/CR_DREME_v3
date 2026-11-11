# SIMPLE-4D respiratory-only v1

The snapshot at `third_party/simple4d/` is supplied as upstream commit c466b276a438b3e052ad3b44d38f398872d8aeae; its hashes and license are recorded in the adjacent manifest. An archive alone does not independently establish the commit identity.

## Directly reused from SIMPLE-4D

MotionCodeVAE base, ResNet-style image backbone, sinusoidal position encoding, VAE sampling semantics and KL, DeformCodeAdaptive and SIREN, Jacobian and magnitude regularizers, and the bidirectional cycle formulation. The view-aware subclass retains the upstream heads and hidden width; sampling follows the upstream forward semantics.

## Modified for CR_DREME_v3

An 8-dimensional view embedding joins image and slice-position features. Continuous external patient-world LPS xyz millimetres map to normalized xyz (not array axes); no frame flip is performed in motion. Raw deformation becomes normalized displacement using the domain extent divided by explicit reference spacing, then becomes world millimetres. Static NeSVoR canonical, corrected observation geometry, Gaussian NeSVoR PSF, and SAX/2CH/4CH arbitrary orientations are reused. R0 freezes the canonical. Reference spacing must be explicitly supplied; if authoritative canonical spacing is available it must be supplied for mismatch checking. The 1 mm configuration is a protocol target, not proof of a particular static artifact's resolution.

## Explicitly not used

SIMPLE-4D InstantNGP canonical, its slice coordinate generator and thick-slice quadrature, existing SINR-MBC, reference gauge and cardiac branch. No training runner is included.
