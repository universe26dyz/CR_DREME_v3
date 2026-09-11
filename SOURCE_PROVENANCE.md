# Source provenance audit

Audit date: 2026-09-10. This is the source-of-truth provenance record for the v3 source-first mainline. No floating branch is a dependency. Vendored identity is locked in `third_party/SOURCE_LOCK.json` by immutable commit and key-file SHA256.

## Audit scope and current cutover state

The v3_change3 closure baseline is `2e8975e6d73862feda57bcaf6eeefc8d464ce8db`; it was deliberately not pulled during this audit. The sources below were read from their pinned, vendored copies before adapter changes.

| Mainline module | Upstream repository and pinned commit | Source file and reused symbol | License | Local adapter / status |
| --- | --- | --- | --- | --- |
| Canonical INR | `daviddmc/NeSVoR` @ `2e96a91bdd30174210caea911e03a2778c65adbe` | `nesvor/inr/models.py :: INR` | MIT | `adapters/nesvor_inr.py`; direct import; CPU identity/forward/backward/smoke passed |
| Hash fallback | `daviddmc/NeSVoR` @ `2e96a91bdd30174210caea911e03a2778c65adbe` | `nesvor/inr/hash_grid_torch.py :: HashEmbedder` (used internally by `INR`) | MIT | no local replacement; CPU path selected because tiny-cuda-nn is unavailable |
| PSF sigma | `daviddmc/NeSVoR` @ `2e96a91bdd30174210caea911e03a2778c65adbe` | `nesvor/utils/psf.py :: resolution2sigma` | MIT | `adapters/nesvor_psf.py`; direct import/world-geometry adapter; CPU equality/constant-field/smoke passed |
| PSF sampling | `daviddmc/NeSVoR` @ `2e96a91bdd30174210caea911e03a2778c65adbe` | `nesvor/inr/models.py :: INR.sample_batch` | MIT | `adapters/nesvor_psf.py`; official Gaussian/sigma semantics plus necessary DICOM row/column/normal world-mm basis adapter; oblique CPU variance test passed |
| Pixel/frame uncertainty | `daviddmc/NeSVoR` @ `2e96a91bdd30174210caea911e03a2778c65adbe` | `nesvor/inr/models.py :: NeSVoR.build_network`, `sigma_net`, `log_var_slice`, and likelihood semantics | MIT | `adapters/nesvor_uncertainty.py`; dynamic-frame-ID mapping; `(mean sample scale)^2 + frame variance` CPU equality passed |
| Image regularization | `daviddmc/NeSVoR` @ `2e96a91bdd30174210caea911e03a2778c65adbe` | `nesvor/inr/models.py :: NeSVoR.img_reg` | MIT | `NeSVoRCanonicalAdapter.image_regularization`; direct unbound source call; CPU test passed |
| SIREN | `vasl12/SINR` @ `1a524ca7ae453b55310595fe957245088a108233` | `networks/networks.py :: BSplineSiren` | No explicit upstream license file | `adapters/sinr_mbc.py`; direct vendored import; logical 8/12/16 controls and non-trainable active-level schedule; CPU identity/gradient/call-count test passed |
| Cubic B-spline FFD | `vasl12/SINR` @ `1a524ca7ae453b55310595fe957245088a108233` | `models/transformation.py :: CubicBSplineFFDTransform`, `cubic_bspline1d`, `conv1d` | No explicit upstream license file | `adapters/sinr_mbc.py`; direct vendored import; explicit dense evaluation grid/padded controls/mm adapter; CPU numerical equality passed |
| FiLM primitive | `ethanjperez/film` @ `fe43ddf8a22b339dcca2efa07091ce9d498955cf` | `vr/models/filmed_net.py :: FiLM.forward` | MIT | `adapters/film.py`; direct primitive import used by `models/film_motion_encoder.py`; CPU identity/gradient/smoke passed |
| Gaussian NLL | PyTorch `2.5.1+cpu` | `torch.nn.GaussianNLLLoss` | BSD-style PyTorch license | direct call; CPU equality/uncertainty/smoke passed |

## Vendoring policy

Pinned NeSVoR, SINR and FiLM sources live under `third_party/NeSVoR`,
`third_party/SINR`, and `third_party/film` as ordinary browsable directories;
they are not mode-160000 gitlinks and their nested `.git` metadata is absent.
The adapters default exclusively to these roots (explicit `CARDIORESP4D_<NAME>_ROOT`
overrides are debug-only). Upstream algorithms are unmodified; local code only
adds domain adapters and does not reimplement HashGrid, SIREN, cubic B-spline,
PSF distribution or sigma network.

## Current custom modules: legacy / ablation only

| Legacy module | v3 source-based replacement | Cutover state |
| --- | --- | --- |
| `src/cardioresp4d/models/hash_inr.py` | NeSVoR `INR` via `adapters/nesvor_inr.py` | runtime import removal pending |
| `src/cardioresp4d/models/sinr_mbc.py` | external SINR `BSplineSiren` via `adapters/sinr_mbc.py` | mainline replacement available |
| `src/cardioresp4d/models/bspline_mbc.py` | external SINR `CubicBSplineFFDTransform` via `adapters/sinr_mbc.py` | mainline replacement available |
| `src/cardioresp4d/rendering/psf_renderer.py` | NeSVoR sigma and Gaussian sampling adapter | runtime import removal pending |
| `src/cardioresp4d/models/uncertainty.py` | NeSVoR-derived dynamic-frame uncertainty adapter | runtime import removal pending |
| `src/cardioresp4d/training/stage1.py` | unified progressive trainer | runtime import removal pending |

These files and their existing tests are retained for historical reproduction, ablation, and numerical comparison. They are not deleted or moved.

## Paper-derived necessary adaptations

The supplied local papers were reviewed during this audit. Patient-world DICOM geometry; full-FOV canonical-domain and coverage QC; asynchronous single-frame geometry-conditioned score inference; DREME respiratory/cardiac low-rank organization; true-timestamp frequency leakage penalties; sequential cardiorespiratory pullback; view/location-balanced sampling; and stage orchestration are necessary project adaptations. They are not claimed as direct upstream code reuse.

## SINR dependency statement

`vasl12/SINR` is a public research-code repository used as an external pinned
dependency. The upstream repository currently has no explicit LICENSE file; no
upstream SINR source is copied or modified inside CR_DREME. The CR_DREME
adapter imports/calls the pinned upstream implementation directly. The checked
out `third_party/SINR` tree is an immutable pinned dependency checkout used by
that adapter, not a renamed local implementation. This research/internal-
experiment policy does not permit a custom SINR/FFD mainline fallback.
