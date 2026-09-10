# 图像域无外部生理信号的心脏-呼吸联合 4D MRI 重建：方法设计与 Codex 实现指令

> **文档版本：2026-09-09 v3（Source-first NeSVoR + SINR + DREME cardiorespiratory decomposition）**
>
> **本版本正式取代 2026-09-08 v2 作为后续实现主规格。**
>
> 本 v3 的核心变化不是简单调参，而是修正实现策略：
>
> 1. **取消 Stage 1A / Stage 1B 这种“先构造 initial V / initial reference，再做 static multi-view refinement”的主线。**
> 2. Canonical anatomy 仍然存在，但由 **同一个统一训练模型中的 NeSVoR 官方 INR** 直接从真实 2D slice-domain observations 学习，不再先制造 `initial_reference.nii.gz` 或独立的 temporal-mean slice reconstruction stage。
> 3. **凡存在成熟、可访问开源实现的底层模块，必须直接复用上游源码或以极薄 adapter 调用上游源码，不允许根据论文描述重新仿写一个“style / inspired”版本。**
> 4. 当前已有的 `hash_inr.py`、自写 SINR、Gauss-Hermite PSF、自写 uncertainty 等代码可保留作为历史实验 / ablation / numerical comparison，但**不得继续作为 mainline runtime implementation**。
> 5. `canonical_domain` 继续表示较大的、连续的最终 reconstruction/query 空间；**不得裁成只有 cardiac box**。Cardiac box 只是 cardiac motion、重点采样和 QC 的局部子区域。
> 6. coverage 与 canonical domain 必须彻底分开：`canonical_domain` 是连续 3D 空间；coverage 是真实 acquisition 对该空间的支持程度。当前离散 centre-plane occupancy 只能作为 geometry QC，不能被解释成 canonical mask。
>
> **代码基线**：现有 `CR_DREME_v1` / `dev/cardioresp4d` 分支及其已完成的 Phase-1 hard-QC、geometry、PCA/FFT、cardiac box、canonical-domain foundation。旧代码不删除。
>
> **当前数据假设**：
>
> - SAX + 2CH + 4CH；
> - 每个 fixed slice location 连续采约 50 个 strict real-time frames；
> - 自由呼吸；
> - 无 ECG；
> - 无呼吸带 / navigator；
> - 无 raw k-space；
> - 输入为 reconstructed 2D DICOM / image-domain MRI；
> - 最终目标是较完整 acquisition-supported anatomy 的连续 3D+time volume，不是 cardiac-only crop。

---

# 0. 本文档的最高优先级规则

## 0.1 Source-first 是硬约束，不是建议

从本版本开始，Codex 对任何存在成熟开源实现的底层模块，必须按以下顺序工作：

```text
先定位论文对应的官方/作者开源仓库
        ↓
读取真实源码文件
        ↓
记录 repository + commit + file path + symbol
        ↓
判断能否直接 import / vendor
        ↓
优先直接复用
        ↓
只在输入输出、坐标、单位、batch 维度处加薄 adapter
        ↓
必要时才做最小修改
```

禁止：

```text
论文说 HashGrid
→ 自己写 HashGrid

论文说 SINR
→ 自己写一个 sine MLP

论文说 cubic B-spline FFD
→ 自己写 4×4×4 interpolation

论文说 NeSVoR PSF
→ 自己设计另一个 Gaussian quadrature

论文说 sigma_net
→ 自己重新设计 variance head
```

如果上游源码可访问但由于依赖、许可证、API 或设备问题暂时不能直接复用：

1. **先停止该模块实现**；
2. 在 `SOURCE_PROVENANCE.md` 中记录具体阻碍；
3. 给出最小 adapter / vendoring 方案；
4. 不得默默改成自写版本。

---

## 0.2 开始编码前必须先完成 Source Audit

在修改主线代码前，必须先创建：

```text
SOURCE_PROVENANCE.md
```

至少记录：

| Mainline module | Upstream repo | Pinned commit | Source file | Reused symbol | Local adapter | Status |
|---|---|---|---|---|---|---|
| Canonical INR | daviddmc/NeSVoR | pinned SHA | `nesvor/inr/models.py` | `INR` | world-domain adapter | pending/done |
| Hash fallback | daviddmc/NeSVoR | pinned SHA | `nesvor/inr/hash_grid_torch.py` | `HashEmbedder` | none/minimal | pending/done |
| PSF sigma | daviddmc/NeSVoR | pinned SHA | `nesvor/utils/psf.py` | `resolution2sigma` | geometry adapter | pending/done |
| PSF sampling semantics | daviddmc/NeSVoR | pinned SHA | `nesvor/inr/models.py` | Gaussian random sampling logic | motion insertion adapter | pending/done |
| Uncertainty | daviddmc/NeSVoR | pinned SHA | `nesvor/inr/models.py` | `sigma_net`, `log_var_slice` semantics | dynamic-frame adapter | pending/done |
| SIREN | vasl12/SINR | pinned SHA | `networks/networks.py` | `Siren` / `BSplineSiren` | control-coordinate adapter | pending/done |
| Cubic B-spline FFD | vasl12/SINR | pinned SHA | `models/transformation.py` | `CubicBSplineFFDTransform` | mm/grid adapter | pending/done |
| FiLM primitive | ethanjperez/film | pinned SHA | upstream FiLM implementation | gamma/beta modulation | geometry-conditioned encoder | pending/done |
| Gaussian NLL | PyTorch | installed version | `torch.nn.GaussianNLLLoss` | official loss | shape adapter | pending/done |

建议固定当前已核查 commit：

- NeSVoR：`2e96a91bdd30174210caea911e03a2778c65adbe`（若服务器实际 checkout 不同，记录真实 commit）
- SINR：`1a524ca7ae453b55310595fe957245088a108233`
- FiLM：使用实际 fetch 到的 pinned commit，并在报告记录

**不得写 `latest`、`master current` 作为最终 provenance。**

---

## 0.3 哪些模块允许 paper-derived implementation

下列部分目前没有已确认的完整作者官方实现可直接复用，因此允许根据论文公式做必要实现：

- DREME-MR respiratory / cardiac low-rank MBC organization；
- DREME-MR frequency-guided disentanglement；
- DREME-MR sequential cardiorespiratory composition；
- 本项目无 ECG / 无呼吸带条件下的 image-domain PCA/FFT frequency prior；
- asynchronous SAX/2CH/4CH single-frame score inference；
- patient-world geometry 与 DICOM plane adapter；
- view-balanced / fixed-location-balanced sampling；
- canonical-domain / coverage QC；
- Stage schedule 与跨模块 orchestration。

这些文件必须明确标记：

```text
Paper-derived necessary adaptation; no confirmed directly reusable official implementation.
```

不能写成：

```text
Direct from DREME-MR code
```

除非实际找到并读取了对应官方源码。

---

# 1. 项目目标

输入：

\[
\{I_n,t_n,G_n\}_{n=1}^{N}
\]

其中：

- \(I_n\)：第 \(n\) 个 acquired 2D real-time frame；
- \(t_n\)：真实 acquisition timestamp；
- \(G_n\)：完整 DICOM plane geometry；
- `view_n ∈ {SAX, 2CH, 4CH}`；
- `slice_id_n`：fixed physical slice location；
- `qc_valid_n`：Phase-1 hard QC 结果。

目标联合学习：

1. continuous canonical anatomy
   \[
   I_{\mathrm{ref}}(x)
   \]
2. respiratory deformation
   \[
   d_r(x,t)
   \]
3. cardiac deformation
   \[
   d_c(x,t)
   \]
4. dynamic volume
   \[
   I(x,t)
   \]
5. respiratory MBC scores
   \[
   w_r(t)
   \]
6. cardiac MBC scores
   \[
   w_c(t)
   \]
7. optional uncertainty maps、DVF、reprojected slices、4D NIfTI。

核心目标不是生成一个 cardiac-only object，而是：

\[
\boxed{
\text{multi-planar real-time 2D image-domain MRI}
\rightarrow
\text{continuous acquisition-supported 3D+t anatomy}
}
\]

---

# 2. 方法来源与分工

## 2.1 NeSVoR：静态 anatomy 表示与 slice acquisition physics 的代码基座

NeSVoR 是本项目以下模块的**主要开源 runtime base**：

### A. Canonical INR

直接复用官方：

```text
nesvor/inr/models.py :: INR
```

以及其官方 HashGrid/tiny-cuda-nn 或 pure-PyTorch fallback。

禁止 mainline 继续使用当前自写：

```text
cardioresp4d/models/hash_inr.py
```

作为 canonical representation。

当前自写文件只保留为：

```text
legacy / ablation / regression comparison
```

### B. Hash encoding

不再自己定义 hash primes、table indexing、coordinate clamp 或 activation。

如果 NeSVoR 在 GPU 使用 tiny-cuda-nn：

- 优先按官方路径安装并运行；
- 如果环境不支持，再使用 NeSVoR 自带 pure-PyTorch fallback；
- 不创建第三套 hash encoder。

### C. PSF physical sigma

直接复用：

```text
nesvor/utils/psf.py :: resolution2sigma
```

NeSVoR 非 isotropic 模式中：

- in-plane 使用 sinc-equivalent Gaussian width；
- through-plane 使用 Gaussian thickness width。

不要再把 `1.2` 作为手写常数；直接调用官方函数。

### D. PSF sampling semantics

NeSVoR forward 中使用 Gaussian samples：

```python
xyz_psf = torch.randn(...)
xyz_sample = xyz_center + xyz_psf * psf_sigma
```

本项目必须保留该核心 sampling semantics。

唯一必要 adaptation：

```text
NeSVoR PSF sample
        ↓
observation world coordinate
        ↓
cardiac/respiratory motion pullback
        ↓
canonical INR query
```

即在 PSF sample 与 INR query 之间插入动态 motion。

### E. Uncertainty

以 NeSVoR 官方：

- `sigma_net`
- `log_var_slice`
- `MSE/(2*var)`
- `0.5*log(var)`

为实现基线。

本项目唯一必要 adaptation：

```text
NeSVoR acquired slice ID
→ dynamic acquired frame ID
```

因此 Stage 2/3 中每个真实 acquired frame 有：

- local pixel-wise variance；
- frame-wise scalar variance。

禁止重新设计一个与官方语义不同的 uncertainty 网络，除非作为显式 ablation。

---

## 2.2 S2V-DREME：image-domain score inference 与 SINR-MBC 思想

S2V-DREME 论文支持：

- image-domain cine slice input；
- FiLM-based motion encoder；
- low-rank MBC score inference；
- SINR-generated MBC；
- progressive optimization。

本项目采用其思想，但必须区分：

### 可直接复用的底层源码

SINR 相关底层实现来自：

```text
vasl12/SINR
```

尤其：

```text
networks/networks.py
    Siren
    BSplineSiren

models/transformation.py
    CubicBSplineFFDTransform
    cubic_bspline1d
    conv1d
```

这些不得再自写替代。

### 必要 adaptation

S2V-DREME 目标场景与本项目不完全一致，因此以下仍需本项目实现：

```text
single acquired frame
+
full DICOM world geometry
→ geometry-conditioned FiLM encoder
→ DREME-format scores
```

因为本项目 SAX/2CH/4CH 并不在同一时刻同步采集。

---

## 2.3 DREME-MR：心脏 + 呼吸的低秩组织与解耦

保留核心表达：

\[
d(x,t)=\sum_i w_i(t)e_i(x)
\]

### Respiratory branch

global + multi-resolution：

- coarse；
- medium；
- fine。

主线保持三个 respiratory levels，每个 level 对应三个 Cartesian score components。

### Cardiac branch

local cardiac box：

- 只在 cardiac subdomain 内定义；
- 边界平滑衰减或固定零；
- 防止 box boundary discontinuity。

### Frequency-guided disentanglement

使用 Phase-1 PCA/FFT 得到的 subject-specific frequency bands：

- respiratory scores 抑制 cardiac-frequency leakage；
- cardiac scores 抑制 respiratory-frequency leakage。

不得简化为固定全局 low-pass / high-pass。

### Sequential pullback

主线统一：

\[
x_c = y+d_c(y,t)
\]

\[
x_{\mathrm{ref}}=x_c+d_r(x_c,t)
\]

即：

\[
x_{\mathrm{ref}}
=
y+d_c(y,t)+d_r(y+d_c(y,t),t)
\]

所有 renderer / trainer / inference / DVF export 必须共用同一 convention。

---

## 2.4 NISF++：统一多方向 physical world-space 的方法依据

NISF++ 主要提供 conceptual support：

- SAX / long-axis 不是三个独立图像空间；
- 应统一投影到同一个 patient/world coordinate space；
- arbitrary orientation slices 应监督同一个 continuous representation。

本项目 DICOM geometry 仍由真实 header 计算，不使用 NISF++ 的标准化 atlas 代替原始物理 geometry。

---

## 2.5 PCA paper：无外部信号下的频率先验

只负责：

```text
image-domain real-time frames
→ respiratory / cardiac frequency evidence
```

PCA/FFT 不直接输出最终 motion latent，也不直接充当 motion encoder。

---

## 2.6 SIMPLE-4D：保留为 image-domain / renderer ablation 来源

SIMPLE-4D 仍提供重要 conceptual evidence：

- reconstructed 2D slices 足以驱动 subject-specific continuous 4D reconstruction；
- external surrogate 不是必须；
- thick-slice forward physics 很重要。

但本 v3 mainline PSF 统一为 NeSVoR source-based Gaussian sampling。

SIMPLE-4D GL5 renderer只保留为：

```text
optional deterministic ablation
```

绝不与 NeSVoR PSF 串联。

---

# 3. 新的整体 Pipeline

```text
PHASE 1 — PRE-NETWORK DATA PREPARATION
======================================

DICOM
  ↓
inspect / manifest
  ↓
hard acquisition QC
  ↓
SAX / 2CH / 4CH world geometry
  ↓
PCA + FFT subject-specific frequency prior
  ↓
cardiac box
  ↓
canonical reconstruction domain
  ↓
PSF-aware multi-view coverage QC


PHASE 2 — SOURCE-BASED MODEL ASSEMBLY
=====================================

Official NeSVoR INR
  +
Official NeSVoR PSF primitives
  +
NeSVoR-derived uncertainty
  +
FiLM primitive + geometry-conditioned score adapter
  +
Official SINR SIREN + cubic B-spline FFD
  +
DREME respiratory/cardio decomposition
  +
DREME sequential pullback


PHASE 3 — UNIFIED PROGRESSIVE TRAINING
======================================

Stage 1:
short slice-domain anatomy bootstrap
(no initial V, no initial_reference, no mean-slice stage)
  ↓
Stage 2:
respiratory coarse → medium → fine
  ↓
Stage 3:
add cardiac + frequency disentanglement
  ↓
full joint optimization


PHASE 4 — INFERENCE / EVALUATION
================================

canonical anatomy
dynamic 3D+t volumes
resp/card scores
DVFs
reprojected 2D slices
coverage / uncertainty / motion QC
```

---

# 4. 四条核心数据契约

## 4.1 Hard-invalid 永远不是训练 GT

当前只有：

```text
slice_local_scale_absolute
manual_exclusion
```

默认使 observation hard-invalid。

其他：

- low_ncc；
- global_intensity_scale；
- scale_corrected_residual；
- slice_local_structure；
- slice_local_scale_robust；

只作为 warning / diagnostics。

Hard-invalid frame：

- 不参与 PCA complete-case block；
- 不进入 Stage 1/2/3 slice-domain data fidelity；
- 不重新以任何形式伪造成 GT。

---

## 4.2 Missing observation ≠ missing canonical space

某个 fixed location 整层被排除：

```text
there is no GT at this acquisition location
```

不等于：

```text
V(x) has a hole
```

Canonical INR 仍在整个 `canonical_domain` 可查询。

但是：

```text
queryable
≠
directly supervised
≠
guaranteed accurate
```

因此 missing-support 区域必须由 coverage QC 标记。

---

## 4.3 Canonical reconstruction domain 不得变成 cardiac-only

定义：

\[
\Omega_{\mathrm{recon}}
\]

为最终 canonical INR 可以输出的较大连续 3D reconstruction/query domain。

它应该覆盖：

- heart；
- surrounding thoracic anatomy；
- acquisition 中实际有意义的邻近结构；
- respiratory motion 所需的 surrounding context。

Cardiac box：

\[
\Omega_{\mathrm{cardiac}}
\subset
\Omega_{\mathrm{recon}}
\]

只用于：

- cardiac MBC local coordinate；
- cardiac-focused sampling；
- cardiac evaluation；
- cardiac-frequency analysis；
- optional cardiac-crop visualization。

禁止：

```text
canonical_domain = cardiac_box
```

除非未来专门做 cardiac-only ablation。

---

## 4.4 Coverage 不是 canonical mask

必须生成并区分：

```text
canonical_domain_mask.nii.gz
coverage_plane_center_*.nii.gz
coverage_psf_*.nii.gz
coverage_psf_union.nii.gz
coverage_view_count.nii.gz
coverage_observation_count.nii.gz
```

### `canonical_domain_mask`

整个 reconstruction box 内连续为 1。

### `coverage_plane_center`

只表示 slice center plane 在离散 grid 上的几何采样。

它可以出现大量离散 0，这是正常的。

**不得把它解释成最终 volume 中存在 hole。**

### `coverage_psf`

真实训练支持更应该使用：

- in-plane FOV；
- slice thickness；
- NeSVoR PSF footprint；

来定义。

### `coverage_view_count`

每个 voxel 被几个不同 view 支持：

```text
0 / 1 / 2 / 3
```

它比单纯 binary occupancy 更适合判断 multi-view under-constraint。

---

# 5. 推荐代码目录

```text
cardioresp4d/
├── third_party/
│   ├── README.md
│   └── LICENSE_NOTICES.md
│
├── src/cardioresp4d/
│   ├── data/
│   │   ├── inspect_dataset.py
│   │   ├── build_manifest.py
│   │   └── dataset.py
│   │
│   ├── outlier_qc/
│   │   └── acquisition_qc.py
│   │
│   ├── geometry/
│   │   ├── world_geometry.py
│   │   ├── coordinate_normalization.py
│   │   ├── canonical_domain.py
│   │   └── coverage.py
│   │
│   ├── frequency/
│   │   ├── pca_motion.py
│   │   └── frequency_bands.py
│   │
│   ├── roi/
│   │   ├── cardiac_box.py
│   │   └── roi_qc.py
│   │
│   ├── adapters/
│   │   ├── nesvor_inr.py
│   │   ├── nesvor_psf.py
│   │   ├── nesvor_uncertainty.py
│   │   ├── sinr_mbc.py
│   │   └── film.py
│   │
│   ├── models/
│   │   ├── film_motion_encoder.py
│   │   └── cardioresp_motion.py
│   │
│   ├── losses/
│   │   ├── data_fidelity.py
│   │   ├── motion_regularization.py
│   │   └── frequency_disentanglement.py
│   │
│   ├── training/
│   │   ├── sampler.py
│   │   ├── model.py
│   │   ├── trainer.py
│   │   └── schedules.py
│   │
│   ├── inference/
│   │   ├── reconstruct_4d.py
│   │   ├── export_dvf.py
│   │   └── reproject_slices.py
│   │
│   └── evaluation/
│       ├── reprojection_qc.py
│       ├── motion_qc.py
│       └── coverage_qc.py
│
├── legacy/
│   └── README.md
│
├── tests/
├── configs/
├── scripts/
├── SOURCE_PROVENANCE.md
├── IMPLEMENTATION_REPORT.md
└── README.md
```

### 旧文件处理

不删除旧代码。

例如现有：

```text
models/hash_inr.py
models/bspline_mbc.py
models/sinr_mbc.py
rendering/psf_renderer.py
models/uncertainty.py
```

如果移动会影响已有历史结果，则**不要移动**。

只需要：

1. mainline 不再 import；
2. README 标记 `legacy / ablation`；
3. 新 adapter 使用明确新文件名；
4. tests 分开 legacy test 与 source-based mainline test。

---

# 6. Module 1：Data / Manifest

保持现有 Phase-1 实现。

每帧至少记录：

```text
source_file_token
view
slice_id
frame_index
timestamp_s
Rows
Columns
PixelSpacing
SliceThickness
ImagePositionPatient
ImageOrientationPatient
orientation vectors
normal
series UID / non-sensitive opaque identity
qc_valid
qc_reason
```

Manifest 是所有后续 geometry、frequency、training、inference 的唯一 observation index authority。

不得在训练阶段重新从文件名猜 view / slice / timestamp。

---

# 7. Module 1.5：Hard Acquisition QC

保持已 seal 政策。

## 7.1 默认 hard-invalid reason

仅：

```text
slice_local_scale_absolute
manual_exclusion
```

## 7.2 其他 diagnostics

继续保存但保持 valid：

```text
low_ncc
global_intensity_scale
scale_corrected_residual
slice_local_structure
slice_local_scale_robust
```

## 7.3 Whole-location exclusion

如果一个 fixed location 被 manual exclusion：

```text
该 location 的所有动态 frames 都 qc_valid=false
```

但：

```text
canonical_domain 不因此挖洞
```

---

# 8. Module 2：Unified World Geometry

## 8.1 DICOM pixel → patient world

单一权威变换：

\[
x_{\mathrm{world}}
=
o
+
u \Delta_u \hat e_u
+
v \Delta_v \hat e_v
\]

并保存：

- row direction；
- column direction；
- normal；
- center；
- pixel spacing；
- thickness。

所有：

- coverage；
- PSF；
- motion；
- INR query；
- reprojection；

都必须调用同一 geometry API。

---

## 8.2 Canonical reconstruction domain

### 定义

`canonical_domain` 是**最终连续 reconstruction/query space**，不是 reference mask，也不是 coverage mask。

主线允许从：

```text
valid multi-view acquisition spatial support
+
必要 margin / target support
```

构建一个 patient-world AABB 或其他明确 box。

当前若 full-FOV AABB 大约为：

```text
~444 × 367 × 430 mm
```

不因为心脏训练重点就自动缩成 cardiac box。

### 是否可以将来缩小

只有满足以下全部条件才允许：

1. 用户明确要求减小 final FOV；
2. 不是为了只让 coverage 图“看起来更满”；
3. 仍覆盖需要输出的 surrounding anatomy；
4. 三个 view 的 target/support QC 重新通过；
5. 作为独立 reconstruction-FOV ablation 记录。

---

## 8.3 Coordinate normalization

NeSVoR `INR` 有自己的 bounding-box normalization 逻辑。

因此 adapter 应优先把：

```text
patient-world mm
→ NeSVoR expected coordinate convention
```

而不是改变 NeSVoR 内部实现。

所有 motion quantities 始终保留 mm：

```text
DVF in mm
B-spline control spacing in mm / explicit grid units
smoothness derivative corrected by physical spacing
```

---

# 9. Module 2.5：Coverage

这是 v3 必须修正的 QC 模块。

## 9.1 为什么旧 coverage 有很多离散 0

旧实现：

```text
slice plane
→ every ~4 mm sample a point
→ round to Cartesian coverage grid
→ set that voxel to 1
```

因此它只是：

```text
sparse center-plane point occupancy
```

不是：

```text
continuous slab support
```

斜切 2CH / 4CH 被 round 到 Cartesian grid 后出现离散 0 是预期现象。

---

## 9.2 新 coverage 的三层输出

### A. Plane-center QC

保留旧输出，用来检查 geometry：

```text
coverage_plane_center_sax.nii.gz
coverage_plane_center_2ch.nii.gz
coverage_plane_center_4ch.nii.gz
```

### B. PSF support coverage

基于真实：

- pixel FOV；
- thickness；
- NeSVoR `resolution2sigma`；
- PSF sampling / practical support radius；

构建：

```text
coverage_psf_sax
coverage_psf_2ch
coverage_psf_4ch
coverage_psf_union
```

### C. Count maps

生成：

```text
coverage_view_count
coverage_observation_count
```

`coverage_view_count` 只统计不同 view 数，不让 50 repeated frames 人为放大 view diversity。

---

## 9.3 Cardiac crop 只用于 QC

可以额外输出：

```text
coverage_view_count_cardiac_crop.nii.gz
coverage_psf_union_cardiac_crop.nii.gz
```

但这只是 visualization / acceptance。

不得反向修改 canonical reconstruction domain。

---

# 10. Module 3：Image-domain PCA + FFT

保持现有方案。

输入必须是：

```text
same fixed slice location
+
complete valid temporal block
```

如果 50 帧中有 hard-invalid 导致 complete-case 不满足：

```text
不要简单插值后假装 uniform FFT
```

输出：

```text
frequency_bands.json
per-slice PCA/FFT diagnostics
consensus respiratory band
consensus cardiac band
```

这些频率只进入 Stage 2/3 regularization。

---

# 11. Module 4：Cardiac Box

Cardiac box 继续保留。

它承担：

1. cardiac MBC local domain；
2. cardiac-focused training sample weighting；
3. cardiac score/motion QC；
4. optional cardiac-crop visualization。

它不承担：

```text
final reconstruction FOV definition
```

---

# 12. Module 5：不再构造 Initial V

## 12.1 v3 删除的 mainline 概念

以下内容从 mainline 取消：

```text
initial_reference.nii.gz as training prerequisite
Stage 1A
Stage 1B
multi-view temporal-mean slice dataset as separate reconstruction stage
```

旧代码可以保留用于历史 reproduction，但新的统一 trainer 不依赖它们。

---

## 12.2 为什么仍然需要 canonical anatomy

取消 initial V 不等于取消 reference anatomy。

模型仍然学习：

\[
I_{\mathrm{ref}}(x)
\]

区别是：

旧：

```text
先构造一个 volume
→ INR 拟合 volume
→ 再 refinement
```

新：

```text
官方 NeSVoR INR
→ 直接由真实 acquired 2D slice-domain likelihood 优化
```

所以 canonical anatomy 是**模型变量**，不是 preprocessing artifact。

---

# 13. Module 6：Official NeSVoR Canonical INR

## 13.1 Mainline 禁止重写

必须直接复用：

```text
daviddmc/NeSVoR
nesvor/inr/models.py
INR
```

如果 NeSVoR official `INR` 构造函数需要 `args`：

创建：

```text
adapters/nesvor_inr.py
```

只负责：

- 构造必要 `Namespace/config`；
- 传入 canonical bounds；
- 输入 patient-world coordinates；
- 处理 NeSVoR bounding-box convention；
- 暴露 `density/intensity` 和 latent `z`；
- 不改 hash / MLP / activation。

---

## 13.2 Adapter contract

推荐统一接口：

```python
class NeSVoRCanonicalAdapter(nn.Module):
    def forward(
        self,
        points_world_mm,
        return_features=False,
    ):
        ...
```

输出：

```text
intensity
latent z
```

内部调用官方 `INR`。

不要复制 NeSVoR `INR` 源码到一个看似本地的新 class 再继续独立维护；如果必须 vendor，保留：

- original file；
- license；
- commit SHA；
- minimal patch diff。

---

## 13.3 Intensity normalization

训练前仍允许 dataset-level intensity normalization。

但不要为了匹配旧自写 `sigmoid` 而改变 NeSVoR official output activation。

先按 NeSVoR 官方 semantics 运行，再在 adapter 外部做明确的 scale mapping。

---

# 14. Module 7：Geometry-conditioned FiLM Motion Encoder

## 14.1 输入

每次输入一张真实 acquired frame：

```text
image [B,1,H,W]
center_mm
row_direction
column_direction
normal
pixel_spacing_mm
slice_thickness_mm
view identity
optional slice-position normalized feature
```

## 14.2 输出

```text
resp_scores [B,3,3]
card_scores [B,1,3]
```

对应：

```text
3 respiratory levels × xyz
1 cardiac level × xyz
```

---

## 14.3 Source-first 规则

FiLM 的：

\[
\gamma,\beta
\]

feature-wise modulation primitive 应优先复用公开 FiLM 实现的数学/代码 primitive。

但是整个：

```text
2D cardiac frame + full 3D geometry → DREME 12 scores
```

是本项目必要 adaptation。

因此必须在文件头写：

```text
FiLM primitive source-backed;
motion-encoder architecture is S2V-DREME-inspired necessary adaptation for asynchronous multi-view CMR.
```

不能声称：

```text
direct S2V-DREME official code
```

除非实际找到对应作者源码。

---

# 15. Module 8：Source-based SINR MBC Generator

## 15.1 禁止继续使用当前自写主线

当前：

```text
models/sinr_mbc.py::_SineLayer
models/bspline_mbc.py::CubicBSplineMBC
```

不得继续作为 mainline SINR/FFD。

---

## 15.2 SIREN 直接来源

使用：

```text
vasl12/SINR
networks/networks.py
Siren / BSplineSiren
```

保持官方：

- sine activation；
- omega semantics；
- SIREN initialization。

不要再自己定义一个“类似”的 `_SineLayer`。

---

## 15.3 Cubic B-spline FFD 直接来源

使用：

```text
vasl12/SINR
models/transformation.py
CubicBSplineFFDTransform
```

它通过 separable transposed convolution 从 control-point parameters 生成 dense flow。

本项目 adapter 只负责：

```text
patient-world control coordinates
↔ upstream voxel/grid spacing
↔ world-mm DVF
```

必须记录每个 axis 的：

```text
control-point spacing
grid size
world extent
voxel/grid scale
```

---

## 15.4 Respiratory MBC

主线三个 levels。

如果继续使用 DREME-style：

```text
8×8×8
12×12×12
16×16×16
```

必须明确：

- 这是 DREME spatial organization；
- SINR 的 FFD API 可能使用 `cps`（control-point spacing）而不是 resolution；
- 必须做一次明确、可测试的 mapping；
- 不允许像旧代码一样只把 `paper_reference_spacing` 写进 metadata 却不真正驱动 FFD。

---

## 15.5 Cardiac MBC

Cardiac MBC 只在 cardiac box 内定义。

边界：

```text
zero / smooth taper
```

必须由 adapter 或 mask 实现，并有 deterministic test。

---

## 15.6 Score × MBC

\[
d_r(x,t)=\sum_{\ell=1}^{3}\sum_{a\in\{x,y,z\}}
w^r_{\ell,a}(t)e^r_{\ell,a}(x)
\]

\[
d_c(x,t)=\sum_{a\in\{x,y,z\}}
w^c_a(t)e^c_a(x)
\]

必须保持 mm 语义。

---

# 16. Module 9：Cardiorespiratory Composition

允许保留本项目 paper-derived 实现，但必须独立于旧 B-spline class。

统一接口：

```python
reference_points = motion.pullback(observation_points, scores)
```

严格使用：

\[
x_c=y+d_c(y,t)
\]

\[
x_{\rm ref}=x_c+d_r(x_c,t)
\]

任何 sign / direction 修改只能在一个 central convention 文件完成。

---

# 17. Module 10：Official NeSVoR PSF Adapter

## 17.1 PSF sigma 必须直接调用上游

```python
from nesvor.utils import resolution2sigma
```

不要复制公式，也不要手写 `1.2`。

---

## 17.2 Mainline PSF sampling

复用 NeSVoR Gaussian random sampling semantics。

对每个 observed pixel center \(y\)：

\[
y_s = y + \epsilon_s\odot\sigma_{\mathrm{PSF}}
\]

其中：

\[
\epsilon_s\sim\mathcal N(0,I)
\]

然后：

```text
y_s
→ cardiorespiratory pullback
→ x_ref,s
→ official NeSVoR INR
→ average samples
→ predicted observed pixel
```

---

## 17.3 不直接复用 NeSVoR 完整 `NeSVoR.forward`

原因：

NeSVoR 原 forward 同时包含：

- slice rigid transform；
- optional fetal deform；
- bias；
- slice scale；
- variance；
- PSF；
- INR。

本项目只需要其中与当前任务同构的底层 primitive。

所以允许写：

```text
NeSVoRPSFAdapter
```

但其 Gaussian sigma / sampling 必须来自/严格对齐上游源代码，而不是重新设计。

---

## 17.4 PSF sample count

作为 config。

第一版应优先与 NeSVoR 官方默认/推荐值一致。

如为性能做减小：

- 明确记录；
- 做 sample-count ablation；
- 不改变 PSF distribution 本身。

---

## 17.5 GL5

旧 SIMPLE-4D 5-point renderer：

```text
ablation only
```

不得混入 mainline loss。

---

# 18. Module 10.5：NeSVoR-derived Dynamic-frame Uncertainty

## 18.1 只在真实 dynamic frames 上定义

v3 不再需要：

```text
mean_slice namespace
```

主线只有：

```text
dynamic_frame
```

---

## 18.2 Pixel-wise variance

直接遵循 NeSVoR：

```text
INR latent z
+
frame/slice embedding if required
→ sigma_net
→ log variance
```

尽量直接复用 NeSVoR network builder 与 output semantics。

---

## 18.3 Frame-wise variance

NeSVoR：

```text
log_var_slice
```

本项目映射为：

```text
log_var_frame
```

一张真实 acquired 2D frame 一个 scalar。

这是明确的 necessary adaptation。

---

## 18.4 Gaussian likelihood

尽量调用：

```python
torch.nn.GaussianNLLLoss
```

或者严格按 NeSVoR 两项：

\[
\frac{r^2}{2\sigma^2}
+
\frac12\log\sigma^2
\]

实现并做 numerical equality test。

---

## 18.5 开启时机

Stage 1 anatomy bootstrap：

```text
uncertainty OFF
```

Stage 2 respiratory 初期：

```text
先 OFF 或 frozen unit variance
```

当 anatomy/motion 已有基本解释能力后再启用。

目的：

```text
variance 只能解释 observation reliability
不能抢先替代 motion/anatomy learning
```

---

# 19. Module 11：Loss

## 19.1 Stage 1：Slice-domain anatomy bootstrap

没有：

```text
initial-reference MSE
```

没有：

```text
mean-slice likelihood
```

直接使用真实 valid dynamic frames。

Motion 固定为 zero / disabled。

主要：

\[
L_{\rm img}
=
\| \hat I_n-I_n\|^2
\]

或严格 source-backed NeSVoR data term的 unit-variance形式。

再加入 NeSVoR official image regularization：

\[
\lambda_{\rm image}R_{\rm image}
\]

优先使用其默认 edge-preserving regularization。

**Stage 1 必须很短，不追求将所有 moving frames 静态拟合到极低 residual。**

否则会把 motion inconsistency 烙进 canonical anatomy。

---

## 19.2 Stage 2：Respiratory motion

\[
L =
L_{\rm img}
+
\lambda_{\rm image}R_{\rm image}
+
\lambda_{\rm mbc}L_{\rm mbc}
+
\lambda_{\rm smooth}L_{\rm smooth}
+
\lambda_{\rm freq,r}L_{\rm freq,r}
\]

initially uncertainty OFF。

随后可：

\[
L_{\rm img}
\rightarrow
L_{\rm GaussianNLL}
\]

---

## 19.3 Stage 3：Cardiorespiratory joint

\[
L =
L_{\rm GaussianNLL}
+
\lambda_{\rm image}R_{\rm image}
+
\lambda_{\rm mbc}L_{\rm mbc}
+
\lambda_{\rm smooth}L_{\rm smooth}
+
\lambda_{\rm freq,r}L_{\rm freq,r}
+
\lambda_{\rm freq,c}L_{\rm freq,c}
+
\lambda_{\rm score}L_{\rm score}
\]

---

## 19.4 Image regularization

本版本要求重新复用 NeSVoR 的：

```text
image_regularization = edge / TV / L2 / none
```

mainline 默认优先测试 NeSVoR `edge`。

不能因为上一版为了 diagnostic 关闭 regularization，就把 `MSE-only` 永久当作正式 reconstruction objective。

---

## 19.5 Frequency loss

必须使用真实 timestamp。

由于本项目 temporal sampling across fixed locations 不一定完全 uniform：

- 不允许简单把全数据按 frame index 当统一时间序列；
- frequency prior 来自 Phase-1；
- score leakage 计算使用真实 timestamps；
- 若采用 nonuniform DFT / Lomb-Scargle-like处理，必须在报告标明是 necessary adaptation。

---

# 20. Module 12：Unified Progressive Training

这是 v3 的训练核心。

不再存在：

```text
Stage 1A
Stage 1B
```

---

## 20.1 Stage 1：Short Anatomy Bootstrap

### 目的

只让官方 NeSVoR INR 从随机状态进入合理 anatomy basin。

不是最终 static reconstruction。

### 数据

真实：

```text
qc_valid dynamic frames
```

不构造：

```text
initial_reference.nii.gz
temporal-mean volume
mean-slice dataset
```

### Sampling

必须同时平衡：

1. view；
2. fixed location；
3. frame。

推荐每 optimizer step：

```text
1 SAX location + random valid frame
1 2CH location + random valid frame
1 4CH location + random valid frame
```

然后从各 frame 中采 pixel batch。

避免因为每 location 有 50 帧而让某个 stack 数量主导。

### Trainable

- NeSVoR official INR；
- optional NeSVoR image regularization parameters if any。

### Frozen / inactive

- FiLM motion encoder；
- respiratory MBC；
- cardiac MBC；
- uncertainty。

Motion = identity。

### Stop policy

这是短 bootstrap。

不以“把所有动态 frame static residual 压到最低”为目标。

必须同时监控：

- reprojection；
- 3D TV / gradient energy；
- anatomy visualization；
- high-frequency artifact。

若 loss 继续下降但 static volume 开始吸收 motion / noise，应提前停止。

---

## 20.2 Stage 2：Respiratory Progressive Learning

### 启用

- geometry-conditioned FiLM respiratory scores；
- respiratory SINR MBC coarse level；
- official NeSVoR INR 继续 trainable。

Cardiac branch 保持 zero/frozen。

### Progressive levels

```text
Stage 2a: coarse respiratory
Stage 2b: coarse + medium
Stage 2c: coarse + medium + fine
```

每次新增 level：

- 新 level zero / near-zero initialization；
- 旧 level 保持；
- 不重置 canonical INR。

### Data

继续使用所有 valid individual frames。

### View balance

必须保持：

```text
SAX / 2CH / 4CH equal contribution
```

不要按 observation count 自然抽样。

### Uncertainty

Stage 2a 初期关闭。

等 respiratory branch 能明显降低 residual 后再打开 NeSVoR-derived uncertainty。

---

## 20.3 Stage 3：Cardiac + Full Joint

加入：

- cardiac FiLM scores；
- local cardiac SINR MBC；
- cardiac frequency leakage penalty；
- respiratory leakage penalty；
- dynamic-frame uncertainty；
- full joint optimization。

最终同时更新：

```text
canonical INR
FiLM encoder
respiratory SINR MBCs
cardiac SINR MBC
uncertainty
```

必要时对不同 parameter groups 使用不同 LR。

---

## 20.4 Progressive schedule 不得写死成论文原始时间

论文阶段顺序可借鉴，但本项目：

- 数据量；
- 50 frames/location；
- 3 views；
- 8 mm thickness；
- cardiac frequency；

都不同。

所以具体 steps/epochs 需从短 run 曲线确定。

但是必须保存：

```text
step
stage
view losses
NCC/NRMSE
image regularization
DVF magnitude
score spectra
variance stats
learning rates
```

---

# 21. Training Sampler

必须实现独立：

```text
training/sampler.py
```

支持：

### View-balanced

每 step 三个 view 都出现。

### Fixed-location-balanced

不能因为某个 view 有更多 slice locations 而淹没另一个 view。

### Frame sampling

每个 selected fixed location 随机选一个 valid temporal frame，长期覆盖全部 50 frames。

### Pixel sampling

允许：

```text
cardiac-focused fraction
+
global fraction
```

例如：

```text
80% cardiac-priority
20% global
```

但这只改变 sampling probability。

**绝不裁掉 global canonical domain。**

---

# 22. Inference

## 22.1 Canonical anatomy

在完整：

\[
\Omega_{\rm recon}
\]

上 query official NeSVoR INR。

输出：

```text
canonical_volume_full_fov.nii.gz
```

可额外输出：

```text
canonical_volume_cardiac_crop.nii.gz
```

用于 visualization，但不是主结果替代品。

---

## 22.2 Dynamic volumes

给定一组 target scores / timestamps：

\[
I(x,t)
\]

在完整 canonical reconstruction domain 上输出。

不要只导出 cardiac box。

---

## 22.3 DVF

至少支持：

```text
respiratory_dvf
cardiac_dvf
composed_dvf
```

单位必须是 mm。

---

## 22.4 Reprojection

对原始真实 acquired frames：

```text
dynamic model
→ source-based NeSVoR PSF
→ predicted 2D frame
```

用于最终 data-fidelity QC。

---

# 23. Evaluation / QC

## 23.1 Reprojection

按：

```text
view
slice_id
frame
```

统计：

- NCC；
- NRMSE；
- residual；
- optional cardiac ROI metrics。

---

## 23.2 Anatomy quality

仅 reprojection 高不代表 3D volume 一定合理。

额外监控：

```text
3D TV
gradient energy
high-frequency energy
through-plane adjacent difference
orthogonal reformats
```

---

## 23.3 Motion QC

检查：

- respiratory score spectra；
- cardiac score spectra；
- cross-frequency leakage；
- DVF magnitude；
- DVF smoothness；
- cardiac-box boundary continuity。

---

## 23.4 Uncertainty QC

输出：

- frame variance distribution；
- pixel variance maps；
- residual-vs-variance relationship；
- highest-variance frame montage。

如果 variance 与 motion amplitude 高度同步，需警惕 uncertainty 在吞 motion。

---

## 23.5 Coverage QC

同时显示：

```text
canonical domain
cardiac box
plane-center coverage
PSF-aware coverage
view count
```

必须在图注中写清：

> Sparse centre-plane zero voxels are not canonical-volume holes.

---

# 24. Config

建议：

```yaml
model:
  canonical:
    implementation: nesvor_official
    upstream_commit: 2e96a91bdd30174210caea911e03a2778c65adbe

  psf:
    implementation: nesvor_official
    n_samples: null  # resolve from upstream/default or explicit experiment

  uncertainty:
    implementation: nesvor_derived
    enable_stage: stage2_late

  respiratory_mbc:
    implementation: sinr_official
    upstream_commit: 1a524ca7ae453b55310595fe957245088a108233
    levels: 3

  cardiac_mbc:
    implementation: sinr_official
    local_box: true

training:
  view_balanced: true
  fixed_location_balanced: true
  cardiac_sampling_fraction: 0.8

  stage1:
    enabled: true
    short_bootstrap: true
    motion_enabled: false
    uncertainty_enabled: false

  stage2:
    respiratory_progressive: true

  stage3:
    cardiac_enabled: true
    uncertainty_enabled: true

domain:
  reconstruction_domain: full_acquisition_supported
  cardiac_box_is_crop: false

coverage:
  plane_center_qc: true
  psf_aware: true
  view_count: true
  observation_count: true
```

---

# 25. Config Validator 硬约束

必须 reject：

```text
model.canonical.implementation != nesvor_official
```

如果 mainline 模式下设置为自写实现。

必须 reject：

```text
domain.reconstruction_domain = cardiac_box
```

除非显式：

```text
experiment_mode = cardiac_only_ablation
```

必须保证：

```text
view_balanced = true
```

在正式 training mode。

必须检查：

- pinned upstream commits；
- source provenance file 存在；
- NeSVoR import 成功；
- SINR import 成功；
- dynamic frame IDs 稳定；
- hard-invalid observations 不进入 sampler。

---

# 26. Tests

## 26.1 Source identity tests

必须测试：

```text
local adapter actually wraps upstream class
```

例如：

```python
assert isinstance(adapter.inr, nesvor.inr.models.INR)
```

而不是仅数值“长得像”。

SINR：

```python
assert isinstance(..., upstream Siren/BSplineSiren)
assert adapter uses upstream CubicBSplineFFDTransform
```

---

## 26.2 NeSVoR INR

同一输入下：

```text
direct upstream call
vs
adapter call
```

在无坐标 adaptation差异时应数值一致。

---

## 26.3 PSF

测试：

1. sigma 与 upstream `resolution2sigma` 完全一致；
2. zero-motion 时 adapter 与 upstream sampling semantics 一致；
3. constant INR 经 PSF 后仍 constant；
4. gradient 能回传到 INR；
5. gradient 能回传到 motion score / MBC。

---

## 26.4 SINR FFD

测试：

1. zero control displacement → zero DVF；
2. official upstream FFD 与 adapter 在同 grid 单位下数值一致；
3. mm conversion 正确；
4. cardiac boundary 正确；
5. backward finite。

---

## 26.5 Uncertainty

测试：

1. unit-variance mode；
2. frame IDs 不串；
3. Gaussian NLL 与 PyTorch reference 一致；
4. variance positive；
5. hard-invalid frame 无 embedding update。

---

## 26.6 Coverage

必须测试：

```text
canonical_domain_mask continuous
```

以及：

```text
plane-center coverage may be sparse
PSF coverage expands along slice normal
```

不要再用“coverage binary map 必须完全连续”作为错误标准。

---

## 26.7 Unified dynamic smoke test

最小链路：

```text
realistic geometry
→ FiLM scores
→ upstream SINR MBC
→ sequential motion
→ upstream NeSVoR PSF sampling
→ upstream NeSVoR INR
→ predicted slice
→ Gaussian NLL
→ backward
```

确认：

```text
INR grad != None
FiLM grad != None
resp MBC grad != None
cardiac MBC grad != None (Stage3 test)
uncertainty grad != None when enabled
```

---

# 27. Mainline / Ablation 优先级

## P0 正式主线

- existing Phase-1 hard QC；
- patient-world geometry；
- PCA/FFT frequency prior；
- full canonical reconstruction domain；
- PSF-aware coverage；
- **official NeSVoR INR**；
- **official/source-aligned NeSVoR PSF**；
- **NeSVoR-derived uncertainty**；
- geometry-conditioned FiLM；
- **official SINR SIREN + CubicBSplineFFDTransform**；
- DREME respiratory multi-level；
- DREME local cardiac；
- sequential pullback；
- Stage1 short bootstrap；
- Stage2 respiratory progressive；
- Stage3 cardiac + joint。

## P1

- bidirectional DVF；
- cycle consistency；
- Jacobian regularization；
- improved motion encoder；
- automatic cardiac box。

## P2 / Ablation

- old custom Hash-INR；
- old custom cubic B-spline；
- old custom SINR；
- 27-point deterministic Gauss-Hermite PSF；
- SIMPLE-4D GL5；
- cardiac-only reconstruction domain；
- no uncertainty；
- no frequency disentanglement。

---

# 28. 当前明确不做

第一版不做：

- raw k-space reconstruction；
- ECG-guided gating；
- respiratory-belt supervision；
- population-level supervised pretraining；
- simultaneous use of two PSF models；
- custom replacement of NeSVoR HashGrid；
- custom replacement of upstream SINR FFD；
- final volume crop to heart only；
- hard rejection expansion beyond sealed reasons。

---

# 29. Codex 实际执行顺序

## Phase A：Source Audit — 必须先完成，不允许跳过

1. checkout current project branch；
2. 读取本 v3；
3. 读取论文；
4. fetch NeSVoR source；
5. fetch SINR source；
6. fetch FiLM source；
7. 固定 commit；
8. 创建 `SOURCE_PROVENANCE.md`；
9. 列出当前 mainline 中哪些模块是 custom imitation；
10. **在 Source Audit 完成前不要开始重写训练代码。**

---

## Phase B：Canonical domain / coverage 修正

1. 保持 large reconstruction domain；
2. 明确 `canonical_domain_mask`；
3. 把旧 occupancy 重命名为 plane-center coverage；
4. 新增 PSF-aware coverage；
5. 新增 view-count / observation-count；
6. 不改变 final volume FOV。

---

## Phase C：NeSVoR adapter

1. 直接引入 official `INR`；
2. 建 world-coordinate adapter；
3. 直接复用 `resolution2sigma`；
4. 建 motion-aware PSF adapter；
5. 加 source identity tests；
6. 禁用 current custom `hash_inr.py` mainline import。

---

## Phase D：SINR adapter

1. 直接引入 upstream `Siren/BSplineSiren`；
2. 直接引入 `CubicBSplineFFDTransform`；
3. 建 mm/grid conversion；
4. respiratory three-level wrapper；
5. cardiac local wrapper；
6. 禁用 old custom SINR/B-spline mainline import。

---

## Phase E：FiLM + Motion

1. 核查 FiLM upstream；
2. 复用 FiLM modulation primitive；
3. 实现 geometry-conditioned single-frame adapter；
4. 保留 DREME score dimensions；
5. 保留 sequential pullback。

---

## Phase F：NeSVoR uncertainty / loss

1. 对齐 `sigma_net`；
2. 对齐 `log_var_slice` → `log_var_frame`；
3. 对齐 Gaussian likelihood；
4. 加 NeSVoR image regularization；
5. numerical equality tests。

---

## Phase G：Unified Trainer

只实现：

```text
Stage 1 short anatomy bootstrap
Stage 2 respiratory progressive
Stage 3 cardiac joint
```

不要重新引入：

```text
Stage 1A
Stage 1B
initial V
mean-slice static SVR
```

---

## Phase H：第一次短 real-data run

只做最小验证：

```text
Stage1 short bootstrap
+
Stage2a coarse respiratory
```

检查：

- source-based INR 能训练；
- reprojection 下降；
- volume 没有明显 hash/noise artifact；
- respiratory scores 有合理低频；
- PSF / motion gradients 正常。

不直接跑长时间 Stage3。

---

## Phase I：完整训练

短 run 通过后：

```text
Stage1
→ Stage2a
→ Stage2b
→ Stage2c
→ Stage3
```

保存所有中间 checkpoint。

---

# 30. README 必须说明

必须写清：

## A. 哪些是 upstream source

例如：

```text
NeSVoR INR: direct upstream class
NeSVoR PSF sigma: direct upstream function
SINR SIREN: direct upstream class
SINR cubic B-spline: direct upstream transform
```

## B. 哪些是 adapter

例如：

```text
patient-world coordinate adapter
dynamic motion insertion between PSF sample and INR query
dynamic-frame uncertainty ID adaptation
DREME score organization
```

## C. 哪些是 paper-derived

例如：

```text
cardiorespiratory sequential composition
frequency leakage
asynchronous geometry-conditioned motion encoder
```

## D. 哪些是 legacy

明确：

```text
old hash_inr.py is not mainline
old sinr_mbc.py is not mainline
old psf_renderer.py is not mainline
```

---

# 31. IMPLEMENTATION_REPORT.md 必须说明

## 31.1 Source provenance

逐模块：

```text
repository
commit
source path
symbol
license
local adapter path
whether source modified
exact modifications
```

---

## 31.2 不允许写模糊表述

禁止：

```text
NeSVoR-style
SINR-inspired
based on official implementation
```

却不写实际 source path。

必须写：

```text
Directly imports X from repository Y at commit Z
```

或者：

```text
Paper-derived implementation; no directly reusable official code found after search.
```

---

## 31.3 Legacy cutover

必须列出：

```text
old file
new mainline replacement
all runtime imports removed? yes/no
all tests migrated? yes/no
```

---

## 31.4 Canonical / coverage

明确记录：

```text
canonical domain extent
cardiac box extent
canonical domain mask shape
plane-center coverage
PSF-aware coverage
view count
```

并写：

> Coverage zeros do not define holes in the canonical representation.

---

## 31.5 Training

必须记录：

```text
Stage1 steps
Stage2a/b/c steps
Stage3 steps
optimizer groups
learning rates
NeSVoR image regularization
uncertainty enable point
PSF sample count
```

---

# 32. 最终结果目录建议

```text
results/
├── phase1/
│   ├── manifest/
│   ├── acquisition_qc/
│   ├── geometry/
│   ├── frequency/
│   └── roi/
│
├── domain/
│   ├── canonical_domain.json
│   ├── canonical_domain_mask.nii.gz
│   ├── coverage_plane_center_sax.nii.gz
│   ├── coverage_plane_center_2ch.nii.gz
│   ├── coverage_plane_center_4ch.nii.gz
│   ├── coverage_psf_sax.nii.gz
│   ├── coverage_psf_2ch.nii.gz
│   ├── coverage_psf_4ch.nii.gz
│   ├── coverage_psf_union.nii.gz
│   ├── coverage_view_count.nii.gz
│   └── coverage_observation_count.nii.gz
│
├── source_audit/
│   └── SOURCE_PROVENANCE.md
│
├── training/
│   ├── stage1/
│   ├── stage2a/
│   ├── stage2b/
│   ├── stage2c/
│   └── stage3/
│
├── inference/
│   ├── canonical_volume_full_fov.nii.gz
│   ├── dynamic_4d_full_fov.nii.gz
│   ├── dvf/
│   └── reprojection/
│
└── evaluation/
    ├── reprojection/
    ├── motion/
    ├── uncertainty/
    └── coverage/
```

---

# 33. 一句话方法定义

本项目最终方法定义为：

> **A source-first, image-domain, surrogate-free multi-view 4D cardiac MRI reconstruction framework that directly reuses NeSVoR’s official continuous INR and slice-acquisition primitives, combines source-based SINR cubic B-spline motion bases with DREME-style respiratory–cardiac low-rank decomposition, and jointly optimizes a full-FOV canonical anatomy and patient-specific cardiorespiratory motion from free-breathing real-time SAX/2CH/4CH images without ECG, respiratory belts, navigators, or raw k-space.**

---

# 34. 最重要的实现原则重申

1. **不要再根据源码“仿写一份功能相似的代码”。**
2. 有上游实现时，mainline 必须直接 import / vendor pinned upstream source。
3. adapter 只能解决 domain mismatch，不能偷偷替换核心数学 primitive。
4. 不再构造 initial V，不再跑 Stage1A/Stage1B。
5. canonical anatomy 仍然存在，并在统一 trainer 中直接从真实 slices 学习。
6. Stage1 只是短 slice-domain bootstrap，不是独立 static reconstruction pipeline。
7. canonical reconstruction domain 保持较大 full-FOV / acquisition-supported anatomy，不能自动缩成 cardiac box。
8. cardiac box 只是局部 motion / priority / QC domain。
9. sparse coverage map 不是 canonical mask。
10. PSF-aware coverage 才用于支持度判断。
11. NeSVoR INR / PSF / uncertainty 必须有真实 upstream source provenance。
12. SINR SIREN / cubic B-spline FFD 必须有真实 upstream source provenance。
13. DREME / asynchronous encoder 等无完整源码部分必须诚实标记为 paper-derived necessary adaptation。
14. 三个 views 共享同一个 canonical representation。
15. hard-invalid observations 永远不作为 GT 返回训练。
16. uncertainty 只能在 anatomy/motion 已有基本能力后启用。
17. 一套 mainline PSF，只允许 ablation 使用另一套。
18. 所有 DVF 与 motion physical quantities 保持明确 mm 语义。
19. 任何“论文原文方法”声明都必须能追溯到论文公式或实际读取过的源码。
20. **SOURCE_PROVENANCE.md 未完成前，不允许声称实现已经“按开源源码复现”。**
