# v3_change1_codex — CR_DREME_v3 二轮 source-first debug / 规范化修正

> **工作目录**：`/home/universe/SVR/code/CR_DREME_v3`
> **目标仓库**：`universe26dyz/CR_DREME_v3`
> **主分支**：`dev/cardioresp4d`
> **方法主规格**：`CR_DREME_v3_FULL_PIPELINE_2026-09-09_SOURCE_FIRST_NESVOR_SINR.md`
>
> 本轮只修改 **CR_DREME_v3**。不要修改、删除、移动 `CR_DREME_v1` 或其他旧仓库/旧结果。
> 不删除 legacy/ablation 代码；只保证 v3 mainline 不再依赖它们。
> 先审查、再修改、再做 targeted tests。不要直接启动长时间真实 GPU 训练。

---

# 0. 本轮最高优先级原则

## 0.1 Source-first 继续作为硬约束

凡是已有成熟开源实现的底层 primitive：

- NeSVoR INR / HashGrid；
- NeSVoR PSF sigma / Gaussian sampling semantics；
- NeSVoR uncertainty network builder / image regularization；
- SINR `BSplineSiren`；
- SINR `CubicBSplineFFDTransform`；
- FiLM primitive；
- PyTorch Gaussian NLL；

不得重新仿写。

本项目只允许在 patient-world / DICOM geometry、mm ↔ grid units、asynchronous multi-view conditioning、DREME respiratory/cardiac low-rank organization、score × MBC、frequency disentanglement、progressive stage orchestration上做必要 adapter / paper-derived implementation。

## 0.2 小细节也执行“先查源码，再实现”

以后遇到 coordinate normalization、Fourier/positional encoding、rotation/orientation encoding、PSF 坐标系转换、B-spline grid/control-point 计数、variance aggregation、sampling、regularizer、optimizer/freeze-unfreeze、numerical stabilization 等小设计，不允许凭感觉直接新造实现。

若存在相关开源实现：
1. 先定位和阅读真实源码；
2. 记录 repo、commit、source file/symbol；
3. 再做最小必要适配；
4. 不要求因此把整个项目加入 `third_party`；
5. 在本项目对应代码块/函数前增加中文注释：

```python
# 参考源码（项目名，commit <SHA>）：
# https://github.com/<owner>/<repo>/blob/<SHA>/<path>
# 本项目适配：<说明借鉴了什么、哪些部分因 DICOM/world-mm/异步多视图而不同>
```

尽量使用 commit-pinned URL，不要只写浮动 `master/main`。

如果没有找到可复用源码，明确注释：

```python
# Paper-derived necessary adaptation；未确认存在可直接复用的官方实现。
```

---

# 1. 创建并落实 `changelog.md`

在 v3 repo 根目录创建：

```text
changelog.md
```

要求：
- 中文书写；
- **append-only**，以后每次修正/新增/删除/重构代码都追加，不覆盖旧记录；
- 每条记录包含真实时间（执行 `date` 获取，保留时区）；
- 包含本次输入 prompt 文件名或总结性操作标题；
- 包含修改目的、受影响文件、关键实现变化、参考开源源码/论文、执行测试及结果、未解决问题、最终 git commit（若可获得）。
- 本轮第一条标题使用：

```text
v3_change1_codex.md — v3 source-first 二轮 debug
```

推荐格式：

```markdown
## 2026-XX-XX HH:MM:SS +ZZZZ — v3_change1_codex.md — v3 source-first 二轮 debug

### 修改目的
...

### 主要变更
- ...

### 涉及文件
- ...

### 外部源码依据
- repo / commit / URL

### 验证
- command
- result

### 未解决问题
- ...

### Git
- commit: ...
```

以后 Codex 完成任何代码修改前，都必须同步更新 `changelog.md`。

---

# 2. 修复 `third_party` 目录状态和全部 import 路径

用户已经手动把 SINR 放到：

```text
/home/universe/SVR/code/CR_DREME_v3/third_party/SINR
```

并删除了 SINR 内部 `.git`。

但当前 GitHub tree 审查显示：

```text
third_party/NeSVoR  mode 160000
third_party/SINR    mode 160000
third_party/film    mode 160000
```

即三者当前仍被 Git index 记录成 **gitlink / embedded repository**，不是普通可浏览目录。

## 2.1 将三个 upstream folders 转成普通 vendored folders

目标：

```text
third_party/
├── NeSVoR/   # 普通目录，可在 GitHub 中逐文件打开
├── SINR/     # 普通目录，可在 GitHub 中逐文件打开
└── film/     # 普通目录，可在 GitHub 中逐文件打开
```

操作要求：
- 不删除这些源码文件；
- 确认三个目录内部都不存在 `.git` 文件或 `.git/`；
- 从 parent repo index 中移除旧 gitlink，再作为普通文件重新 `git add`；
- 不要误用 `git rm` 删除真实源码。

验证：

```bash
git ls-files --stage third_party/SINR | head
git ls-files --stage third_party/NeSVoR | head
git ls-files --stage third_party/film | head
```

应看到内部普通文件（通常 mode `100644` 等），而不是只有一条 mode `160000`。

提交后：

```bash
git ls-tree -r HEAD third_party/SINR | head
git ls-tree -r HEAD third_party/NeSVoR | head
git ls-tree -r HEAD third_party/film | head
```

不得再出现三个根目录 `mode 160000`。

> SINR upstream 当前没有明确 LICENSE 文件。不要伪造许可证、不要标为 MIT/Apache。继续在 `SOURCE_PROVENANCE.md` / `LICENSE_NOTICES.md` 中明确这一点。未来若公开发布仓库，再次核查其再分发许可。

## 2.2 保存 vendored source identity

由于 `.git` 被删除，不能再依赖 `git -C third_party/... rev-parse HEAD` 判断 upstream commit。

新增：

```text
third_party/SOURCE_LOCK.json
```

至少记录：
- upstream repo URL；
- pinned commit；
- runtime 使用的 key source files；
- key source files SHA256；
- vendored 状态。

固定来源：

```text
NeSVoR
https://github.com/daviddmc/NeSVoR
commit 2e96a91bdd30174210caea911e03a2778c65adbe

SINR
https://github.com/vasl12/SINR
commit 1a524ca7ae453b55310595fe957245088a108233

FiLM
https://github.com/ethanjperez/film
commit fe43ddf8a22b339dcca2efa07091ce9d498955cf
```

## 2.3 统一 `_upstream.py`

当前旧逻辑仍把 SINR 默认定位到 repo 外 `external/SINR`。

改成默认统一从：

```text
<PROJECT_ROOT>/third_party/NeSVoR
<PROJECT_ROOT>/third_party/SINR
<PROJECT_ROOT>/third_party/film
```

加载。

可以保留环境变量 override 作为调试能力，但默认路径必须是 v3 repo 内 `third_party`，不再要求 `CARDIORESP4D_SINR_ROOT`。

同步更新：
- `README.md`
- `SOURCE_PROVENANCE.md`
- `IMPLEMENTATION_REPORT.md`
- `third_party/README.md`
- tests / config validator 中涉及 external SINR 的描述。

## 2.4 全仓库扫描路径与 import

执行：

```bash
rg -n "CR_DREME_v1|external/SINR|CARDIORESP4D_SINR_ROOT|third_party|sys\.path|add_upstream_to_path" .
```

逐项确认：
- 没有仍指向旧 nested `CR_DREME_v1/CR_DREME_v1` 的 runtime path；
- 没有默认依赖旧 `/external/SINR`；
- NeSVoR / SINR / FiLM 均能在 clean v3 checkout 中 import；
- legacy 文件可保留旧说明，但 mainline runtime 不得依赖旧路径。

补一个 clean-path import test。

---

# 3. Cardiac motion encoder：先审查开源 geometry / Fourier encoding，再修

当前：

```text
src/cardioresp4d/models/film_motion_encoder.py
```

把：

```text
center_mm
row_direction
column_direction
normal
pixel_spacing_mm
slice_thickness_mm
```

直接拼成 15-D raw geometry，然后对所有数值使用同一组 `2^k * pi` Fourier bands。

这会把几十/几百 mm 的 position、[-1,1] unit direction、~1–2 mm spacing、~8 mm thickness 混入相同数值尺度。

## 3.1 必须先阅读以下源码

### A. NISF++ / cardiac CMR — 首选相关来源

Repo：

```text
https://github.com/NILOIDE/CMR_representations
```

固定审查 commit：

```text
1f6f9b3feba7c3c757d1f4111418415cd1cafeaa
```

重点：

```text
https://github.com/NILOIDE/CMR_representations/blob/1f6f9b3feba7c3c757d1f4111418415cd1cafeaa/pos_encoding.py
```

重点理解：
- `PosEncodingNeRF`
- `PosEncodingNeRFOptimized`
- per-dimension `num_frequencies`
- `coords_freq_scale`

NISF++ 方法明确将 scanner/world coordinates 投到标准化空间，并归一化到 `[-1,1]` 后做 sinusoidal positional encoding。该来源与本项目 multi-view cardiac world-space 最接近。

### B. Nerfstudio — 成熟 normalization / encoding 参考

固定已审查 commit：

```text
50e0e3c70c775e89333256213363badbf074f29d
```

位置归一化：

```text
https://github.com/nerfstudio-project/nerfstudio/blob/50e0e3c70c775e89333256213363badbf074f29d/nerfstudio/data/scene_box.py
```

重点：`SceneBox.get_normalized_positions`

NeRF sinusoidal encoding：

```text
https://github.com/nerfstudio-project/nerfstudio/blob/50e0e3c70c775e89333256213363badbf074f29d/nerfstudio/field_components/encodings.py
```

重点：`NeRFEncoding`

还要注意成熟 NeRF implementation 对 position 和 direction 分开处理，而不是把 position mm 与 unit direction 当成同一量纲。

## 3.2 推荐最小修正

### position / center

`center_mm` 先基于当前 canonical bounds 做：

```text
world mm → canonical normalized [-1,1]
```

优先复用现有 `WorldNormalizer` / canonical-domain transform，不再新写第二套 normalization。

### orientation

`row_direction / column_direction / normal`：
- 保持 dimensionless unit-vector 语义；
- 加 norm / orthogonality sanity check；
- 必要时仅做数值 renormalization；
- 不使用 position 的 mm scale。

### pixel spacing / thickness

不要和 raw position 共用 Fourier phase。

优先：
- 转为稳定 dimensionless acquisition features；
- 或仅作为 normalized raw scalars 输入 geometry MLP，而不做高频 Fourier encoding。

具体形式先参考源码并给出理由后选择；不要重新引入无依据复杂 encoding。

### encoding

可以参考 NISF++：
- 不同 feature group/dimension 使用不同 frequency count 或 scale；
- position 与 orientation/acquisition metadata 分开处理，最后 concatenate。

## 3.3 中文源码注释

在修改后的 geometry encoding block 前写清：
- NISF++ URL；
- Nerfstudio URL；
- 本项目借鉴内容；
- 哪些是 CR-DREME 必要 adaptation。

## 3.4 tests

至少新增：
- canonical min / center / max 的 center encoding 数值范围测试；
- direction scale 不受 canonical FOV mm 大小影响；
- spacing/thickness 不再直接进入 raw-mm Fourier phase；
- 相同 geometry deterministic；
- backward 正常；
- NaN/Inf / 非单位方向有明确处理或错误。

本轮不要随意扩大 motion encoder 主体 CNN。

---

# 4. 修复 NeSVoR PSF 的 oblique slice orientation

当前 `NeSVoRPSFAdapter` 虽然使用 NeSVoR `resolution2sigma` 和 Gaussian sampling semantics，但当前 sample 在 patient-world XYZ axes 直接乘：

```text
[sigma_inplane_1, sigma_inplane_2, sigma_through]
```

这对 oblique 2CH/4CH 不正确。

正确含义：

```text
local slice x offset → DICOM row-direction
local slice y offset → DICOM column-direction
local slice z offset → DICOM normal
```

尤其 through-plane sigma 必须沿当前 slice normal，不能固定沿 world Z。

## 4.1 修改方式

先重新阅读 pinned NeSVoR：
- `nesvor/inr/models.py :: INR.sample_batch`
- `nesvor/utils/psf.py :: resolution2sigma`
- `nesvor/transform.py :: RigidTransform / point transform utilities`

优先寻找是否可以通过 NeSVoR 自带 transformation path 保持官方 sample implementation。

若 direct `INR.sample_batch(..., transformation=...)` 能清晰表达 `slice-local → DICOM patient-world`，优先使用。

如果其 transform contract 不适合 DICOM plane adapter，则允许：
1. 直接调用官方 `resolution2sigma`；
2. 严格使用与 NeSVoR 相同的 `torch.randn` Gaussian sampling；
3. 只在 adapter 中把 local random offset 乘 DICOM `[row, column, normal]` basis 转到 world-mm；
4. provenance 诚实写为 `NeSVoR Gaussian sampling semantics + CR-DREME DICOM-basis orientation adapter`，不要声称原封不动 direct `sample_batch`。

## 4.2 API

PSF forward/sample 必须获得并使用：
- row_direction
- column_direction
- normal
- pixel_spacing
- slice_thickness

不要只传 `centers_world_mm + resolution_mm`。

## 4.3 tests

新增 oblique test：

构造明显斜切 plane，固定随机种子，检查大量 Gaussian offsets 在 row/column/normal 三个正交 basis 上的投影方差与 `resolution2sigma` sigma²一致。

重点断言：

```text
through-plane 最大方差方向 == slice normal
```

验证 SAX / 2CH / 4CH 任意 orientation。

---

# 5. 修复 SINR / DREME control-grid 语义

当前 adapter 将 `(8,8,8), (12,12,12), (16,16,16)` 作为 upstream `CubicBSplineFFDTransform(img_size=...)`。

但 SINR upstream 明确：

```text
img_size = dense output image/grid size
cps      = control point spacing
```

并不是 `img_size = DREME control-point count`。

参考：

```text
https://github.com/vasl12/SINR/blob/1a524ca7ae453b55310595fe957245088a108233/models/transformation.py
```

重点：`CubicBSplineFFDTransform`、`cubic_bspline1d`、`conv1d`。

## 5.1 建立明确语义

外部接口不要再叫模糊 `grid_shape`。

区分：
- `logical_control_shape`
- `dense_evaluation_shape`
- `control_point_spacing / cps`
- `padded_control_shape`（若 cubic B-spline 边界需要额外 controls）

DREME respiratory MBC resolution：

```text
coarse  = 8×8×8
medium  = 12×12×12
fine    = 16×16×16
cardiac = 16×16×16
```

先核对 DREME 论文中“control-point grid”语义，再结合 SINR transposed-convolution FFD 的边界 padding 规则实现。

如果 upstream cubic support 需要额外 border controls：
- 可以内部存在 padded lattice；
- 但必须明确区分 logical DREME grid 和 actual padded tensor；
- 不得继续把 dense output grid 冒充 control grid。

## 5.2 progressive 新 level 初始化

Stage2a → 2b → 2c 新 level 激活时新增 DVF 必须约为 0。

优先选择不修改 upstream architecture 的 adapter-level 方法，例如 near-zero output gate；先查 S2V-DREME/SINR 是否有现成做法，若无源码再做最小 paper-derived adapter。

## 5.3 tests

必须验证：
- logical resp control shape = 8³/12³/16³；
- logical cardiac control shape = 16³；
- padded shape 如存在被显式记录；
- mm ↔ grid unit conversion；
- zero control → zero DVF；
- 新增 Stage2b/2c level 激活瞬间新增 DVF ≈ 0；
- cardiac boundary taper continuity；
- gradients 进入 upstream `BSplineSiren`。

---

# 6. 修复 NeSVoR uncertainty 的 PSF aggregation

当前近似是：

```text
mean(sample_scale^2) + frame_variance
```

NeSVoR 官方是：

```text
(mean(sample_scale))^2 + frame_variance
```

重新核对：

```text
third_party/NeSVoR/nesvor/inr/models.py
```

尤其 `sigma_net`、`log_var_slice`、`NeSVoR.forward`。

调整为官方语义，只保留：

```text
slice ID → dynamic frame ID
```

这一必要 adaptation。

新增 numerical equality test：
- 固定 latent/sample scale；
- 与直接 NeSVoR 公式一致；
- 验证 `torch.nn.GaussianNLLLoss`；
- frame variance 只加一次。

---

# 7. 补齐 Stage2 / Stage3 scientific loss stack

当前 trainer 实际只是：

```text
Stage1/2: MSE + image regularization
Stage3:   Gaussian NLL + image regularization
```

按 v3 规格补齐。

## Stage1

```text
real valid dynamic frames
motion OFF
uncertainty OFF
NeSVoR INR trainable

L = data fidelity + λ_image * NeSVoR image regularization
```

Stage1 必须短。

## Stage2a/b/c

至少：

```text
L =
data fidelity
+ λ_image * R_image
+ λ_mbc * L_mbc
+ λ_smooth * L_dvf_smooth
+ λ_freq_r * L_freq_resp
+ optional λ_score * L_score
```

## Stage3

```text
L =
Gaussian NLL
+ λ_image * R_image
+ λ_mbc * L_mbc
+ λ_smooth * L_dvf_smooth
+ λ_freq_r * L_freq_resp
+ λ_freq_c * L_freq_card
+ λ_score * L_score
```

所有 loss weights 移入 `source_first.yaml`，并保存到 training report。

日志必须分项记录每个 loss component。

---

# 8. 把真实 timestamp 接入 frequency disentanglement

当前 `CardioRespDataset` 已有 `timestamp_s`，但 `DynamicObservation` / `train_source_first.py` 丢掉了它。

修正：

```text
manifest timestamp_s
→ CardioRespDataset
→ DynamicObservation.timestamp_s
→ temporal score mini-batch
→ frequency leakage loss
```

## 8.1 禁止 detached 历史 buffer

不要把过去 optimizer steps 的 detached scores 简单累积后算 loss。

增加可微 temporal auxiliary batch：

1. 选择 fixed slice location；
2. 取该 location 一组 valid frames；
3. 按真实 timestamp 排序；
4. 一次 forward motion encoder 得到 score sequence；
5. 用真实 timestamp 计算 nonuniform spectral leakage；
6. backprop 到 motion encoder。

可控制 temporal batch 大小和 loss 执行频率节约显存。

## 8.2 frequency bands

增加 CLI/config 输入：

```text
frequency_bands.json
```

来自 Phase1 PCA/FFT subject-specific bands。

Stage2：resp scores 抑制 cardiac-frequency leakage。

Stage3：resp scores 抑制 cardiac band；card scores 抑制 respiratory band。

不得对全数据 frame index 做 uniform FFT。

现有 `losses/stage_aware.py::frequency_leakage` 可以复用，但先验证 true timestamp、differentiability、irregular sampling、empty/too-short band、finite behavior。

---

# 9. 实现真正 80% cardiac-priority + 20% global pixel sampling

`source_first.yaml` 当前声明：

```yaml
cardiac_sampling_fraction: 0.8
```

但 trainer 仍是全图 uniform random。

修正：
- 用 slice DICOM geometry 将 pixel center 投到 patient world；
- 判断落入 `cardiac_box` 的 pixels；
- 默认约 80% 从 cardiac-intersecting pixels 抽，20% 从全图抽；
- 不与 cardiac box 相交时明确 fallback；
- 不改变 full-FOV canonical domain。

加入统计测试，长期抽样比例接近 config。

---

# 10. 显式 stage freeze / unfreeze

当前所有参数从一开始都在同一 Adam 中，只靠 `score * 0`。

改成真正 stage-aware trainability。

## Stage1
只允许 canonical INR trainable；FiLM/resp/card/uncertainty frozen。

## Stage2a/b/c
canonical INR + FiLM respiratory path + active respiratory levels trainable；cardiac/uncertainty按 schedule frozen。

## Stage3
再开启 cardiac branch、uncertainty、full joint。

新 stage 解冻的参数不要携带此前“零梯度但 optimizer step 已增长”的不合理 state。可以每 stage 重建 parameter groups 或显式管理 optimizer state/LR。

测试不能只检查 `grad is not None`：
- 检查应冻结参数值完全不变；
- 应训练参数产生非零 gradient / 参数更新。

---

# 11. 修正 normalization policy

当前 `CardioRespDataset` 对每张 frame 独立做 1/99 percentile → [0,1]，会抹掉 frame-to-frame global intensity relationship。

本轮先 audit，并实现显式 config policy，例如：

```text
normalization:
  mode: per_series
```

至少保留可选：
- `per_frame_legacy`
- `per_series` / `per_view` stable normalization
- `subject_global`

formal first candidate 优先 `per_series`，以保留同一 strict real-time series 的 temporal intensity consistency。

要求：
- 先正确应用 DICOM modality LUT/rescale；
- normalization parameters 只由 qc_valid data 估计；
- 参数可复现并写入 report；
- legacy per-frame 可做 ablation；
- 不使用未来 GT 或 reconstructed volume。

本轮不需要做长真实数据 ablation。

---

# 12. 修正 config / provenance 不一致

## 12.1 cardiac box

formal mainline：

```text
canonical domain = full acquisition-supported FOV
cardiac box = local subdomain
```

因此：

```yaml
cardiac_box_is_crop: false
```

validator enforce：除 `cardiac_only_ablation` 外必须 false。

## 12.2 uncertainty enable stage

当前 config 写 `stage2_late`，代码只 Stage3 开启。

二选一统一：
- 若实现 Stage2c late uncertainty，则 config/schedule/test 一致；
- 否则 formal config 先设 `stage3`。

## 12.3 PSF samples

短 smoke 可 `n_samples=8`，但文档明确：

```text
NeSVoR official distribution/sigma semantics
+ CR-DREME reduced sample count for smoke/performance
```

不要声称 8 是 NeSVoR 官方默认。formal sample count 放 config，后续做小 ablation 再定。

## 12.4 NeSVoR INR hyperparameters

把 coarsest/finest resolution、level scale、features/level、log2 hash size、latent dim、width、depth 全部显式写入 config 和 training report。

即使 class 来自 NeSVoR，也不要把本项目值误写成 official defaults。

---

# 13. centralize cardiorespiratory pullback

当前 sequential pullback 在不止一个位置存在。

主线只保留一个 authoritative implementation，例如：

```text
models/cardioresp_motion.py
```

统一：

```text
x_c   = y + d_c(y,t)
x_ref = x_c + d_r(x_c,t)
```

`training/model.py`、renderer、未来 inference/DVF export 全调用同一 central implementation。

legacy 可保留，但 mainline 不复制 sign/order logic。

---

# 14. dependencies / clean checkout 可复现性

当前 `pyproject.toml` 仍偏 Phase1。

审查 v3 source-first runtime 实际依赖，提供最小可复现说明，可选择：
- 更新 `pyproject.toml` optional extra；
- 或新增 `requirements_v3.txt` / `requirements-source-first.txt`。

不要盲目安装 third-party 全部依赖。

README 给出从 clean clone 到 CPU source tests 的最短命令。

---

# 15. 本轮完成标准：targeted tests

至少覆盖：

1. `third_party` 路径与 `SOURCE_LOCK`；
2. NeSVoR/SINR/FiLM direct source identity；
3. geometry normalization + Fourier encoding；
4. **oblique PSF orientation**；
5. SINR logical 8³/12³/16³ + cardiac 16³；
6. progressive new-level near-zero DVF；
7. uncertainty aggregation numerical equality；
8. timestamp propagation；
9. irregular timestamp frequency loss；
10. 80/20 cardiac/global sampling；
11. stage freeze/unfreeze 的真实 parameter change/nonzero gradient；
12. central sequential pullback；
13. config validator；
14. unified Stage1→Stage2a→Stage2b→Stage2c→Stage3 synthetic smoke。

优先 CPU deterministic tests。不要为通过 test 降低物理约束。

---

# 16. 本轮暂时不要做

在上述问题全部通过前：
- 不运行长真实 GPU Stage3；
- 不开始 Phase4 full inference；
- 不删除 legacy Stage1A/Stage1B；
- 不重新设计 NeSVoR/SINR 内部算法；
- 不改变 Phase1 已 seal hard-QC；
- 不把 canonical domain 裁成 cardiac-only；
- 不为了更好看修改真实 coverage；
- 不随意增加复杂网络层。

---

# 17. 完成后下一步：最短真实 GPU 验证

只有本轮 targeted tests 全通过后，才准备：

```text
Stage1 short anatomy bootstrap
→ Stage2a respiratory coarse
```

短 GPU run 必须保存：
- git commit；
- `SOURCE_LOCK.json`；
- 完整 config；
- normalization parameters；
- Stage1/Stage2a 分项 loss；
- per-view MSE/NCC/NRMSE；
- image regularization；
- respiratory score statistics；
- coarse DVF magnitude/smoothness；
- canonical anatomy snapshot；
- oblique 2CH/4CH reprojection examples。

先不要进入 Stage2b/2c/Stage3，直到 Stage2a 结果审核通过。

---

# 18. 最终 Codex 回报格式

完成后按以下顺序汇报：

```text
1. 本轮实际修改了哪些文件
2. 每个修改解决什么问题
3. 哪些地方直接复用了上游源码
4. 哪些地方是参考开源源码后的必要 adaptation（列 URL）
5. 哪些地方仍是 paper-derived implementation
6. third_party 是否已经不再是 mode 160000 gitlink
7. SINR/NeSVoR/FiLM runtime import 最终来自哪里
8. PSF oblique orientation test 结果
9. DREME logical control-grid 8/12/16/16 test 结果
10. uncertainty numerical equality test 结果
11. frequency/timestamp test 结果
12. sampling/freeze-unfreeze test 结果
13. 完整 targeted test command + passed/failed 数
14. changelog.md 新增了什么条目
15. 当前仍未解决的问题
16. 下一步建议的最短 GPU command（只给命令，不执行长训练）
17. 最终 git commit SHA
```

如任何关键物理/源码语义不确定：

**停止对应模块的猜测式实现，记录问题并说明；不要用自写近似悄悄替代。**
