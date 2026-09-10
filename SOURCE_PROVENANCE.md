# Source provenance audit

Audit date: 2026-09-09. This is the source-of-truth provenance record for the v3 source-first mainline. No floating branch is a dependency.

## Audit scope and current cutover state

The checked-out project branch is `dev/cardioresp4d` at `dd8fbab`; it was clean before this audit and is five commits behind `origin/dev/cardioresp4d`. It was deliberately not pulled during the audit. The sources below were read from their pinned, vendored copies before any mainline adapter was written.

| Mainline module | Upstream repository and pinned commit | Source file and reused symbol | License | Local adapter / status |
| --- | --- | --- | --- | --- |
| Canonical INR | `daviddmc/NeSVoR` @ `2e96a91bdd30174210caea911e03a2778c65adbe` | `nesvor/inr/models.py :: INR` | MIT | `adapters/nesvor_inr.py`; direct import; CPU identity/forward/backward/smoke passed |
| Hash fallback | `daviddmc/NeSVoR` @ `2e96a91bdd30174210caea911e03a2778c65adbe` | `nesvor/inr/hash_grid_torch.py :: HashEmbedder` (used internally by `INR`) | MIT | no local replacement; CPU path selected because tiny-cuda-nn is unavailable |
| PSF sigma | `daviddmc/NeSVoR` @ `2e96a91bdd30174210caea911e03a2778c65adbe` | `nesvor/utils/psf.py :: resolution2sigma` | MIT | `adapters/nesvor_psf.py`; direct import/world-geometry adapter; CPU equality/constant-field/smoke passed |
| PSF sampling | `daviddmc/NeSVoR` @ `2e96a91bdd30174210caea911e03a2778c65adbe` | `nesvor/inr/models.py :: INR.sample_batch` | MIT | `adapters/nesvor_psf.py`; motion pullback after official samples; CPU equality/constant-field/smoke passed |
| Pixel/frame uncertainty | `daviddmc/NeSVoR` @ `2e96a91bdd30174210caea911e03a2778c65adbe` | `nesvor/inr/models.py :: NeSVoR.build_network`, `sigma_net`, `log_var_slice`, and likelihood semantics | MIT | `adapters/nesvor_uncertainty.py`; dynamic-frame-ID mapping; CPU positivity/NLL/gradient/smoke passed |
| Image regularization | `daviddmc/NeSVoR` @ `2e96a91bdd30174210caea911e03a2778c65adbe` | `nesvor/inr/models.py :: NeSVoR.img_reg` | MIT | `NeSVoRCanonicalAdapter.image_regularization`; direct unbound source call; CPU test passed |
| SIREN | `vasl12/SINR` @ `1a524ca7ae453b55310595fe957245088a108233` | `networks/networks.py :: BSplineSiren` | No explicit upstream license file | `adapters/sinr_mbc.py`; direct external import; CPU identity/gradient test passed |
| Cubic B-spline FFD | `vasl12/SINR` @ `1a524ca7ae453b55310595fe957245088a108233` | `models/transformation.py :: CubicBSplineFFDTransform`, `cubic_bspline1d`, `conv1d` | No explicit upstream license file | `adapters/sinr_mbc.py`; direct external import; CPU zero/numerical/mm test passed |
| FiLM primitive | `ethanjperez/film` @ `fe43ddf8a22b339dcca2efa07091ce9d498955cf` | `vr/models/filmed_net.py :: FiLM.forward` | MIT | `adapters/film.py`; direct primitive import used by `models/film_motion_encoder.py`; CPU identity/gradient/smoke passed |
| Gaussian NLL | PyTorch `2.5.1+cpu` | `torch.nn.GaussianNLLLoss` | BSD-style PyTorch license | direct call; CPU equality/uncertainty/smoke passed |

## Vendoring policy

Pinned NeSVoR and FiLM source lives under `third_party/NeSVoR` and `third_party/film`. SINR is an independent external checkout at `/home/universe/SVR/code/external/SINR` (overridable by `CARDIORESP4D_SINR_ROOT`) and is never copied, vendored, renamed, or modified inside CR_DREME. Upstream files are unmodified and keep their own Git metadata so their commit identity can be checked. Local runtime adapters only add source roots to `sys.path` to import audited symbols; they do not copy or reimplement the upstream HashGrid, SIREN, cubic B-spline interpolation, PSF distribution, or sigma network.

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

## SINR external-dependency statement

`vasl12/SINR` is a public research-code repository used as an external pinned dependency. The upstream repository currently has no explicit LICENSE file; no upstream SINR source is copied or modified inside CR_DREME. The CR_DREME adapter imports and calls the pinned upstream implementation directly. This research/internal-experiment policy does not permit a custom SINR/FFD mainline fallback.
