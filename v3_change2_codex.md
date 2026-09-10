# v3_change2_codex — CR_DREME_v3 motion-training scientific correctness 修正

> **工作目录**：`/home/universe/SVR/code/CR_DREME_v3`
>
> **目标仓库**：`universe26dyz/CR_DREME_v3`
>
> **当前基线 commit**：`d947ad71bc84dbe5441117b6c0e0400c39337909`
>
> **当前分支**：`dev/cardioresp4d`
>
> **主规格**：`CR_DREME_v3_FULL_PIPELINE_2026-09-09_SOURCE_FIRST_NESVOR_SINR.md`
>
> **上一轮 prompt**：`v3_change1_codex.md`
>
> 本轮目标不是继续扩展框架，而是针对 `v3_change1` 后发现的 **motion-training scientific correctness / integration bugs** 做一次聚焦修正。
> 不修改旧 `CR_DREME_v1`，不删除 legacy/ablation 文件，不改变 Phase-1 已 seal 的 hard-QC policy，不把 canonical domain 改成 cardiac-only。
>
> **本轮修正完成、CPU regression 全部通过之前，不执行真实长 GPU training。**

---

# 0. 本轮总原则

## 0.1 source-first 继续作为硬约束

已经接好的 upstream primitive 不允许重新实现：

- NeSVoR INR / HashGrid；
- NeSVoR `resolution2sigma`；
- NeSVoR uncertainty network builder；
- NeSVoR image regularization；
- SINR `BSplineSiren`；
- SINR `CubicBSplineFFDTransform`；
- FiLM primitive；
- PyTorch Gaussian NLL。

本轮主要修：

1. DREME-MR 低秩 motion topology；
2. DREME-MR Eq.6 / Eq.7 / Eq.8 / Eq.9；
3. Phase-1 frequency prior 到 trainer 的真实接线；
4. temporal batch 时间跨度；
5. runtime config 真正生效；
6. smoothness / source lock / regression 等工程一致性。

## 0.2 不允许“为了通过测试”改变数学定义

本轮涉及 DREME-MR 原公式，必须先读论文对应段落，再实现。

重点核对：

```text
DREME-MR
Shao et al., Phys. Med. Biol. 70 (2025) 175013

Section 2.2:
low-rank cardiorespiratory motion model

Section 3.3:
Eq. (6) L_MBC
Eq. (7) L_ZMS
Eq. (8) L_c
Eq. (9) L_r
Eq. (10) total regularization
```

已确认的原文核心定义：

### DREME MBC decomposition

在第 `i` 个 spatial resolution level，每一个 Cartesian direction `k ∈ {x,y,z}` 有一个独立 MBC：

\[
e_{i,k}(x)
\]

对应独立 score：

\[
w_{i,k}(t)
\]

因此 respiratory 为：

\[
d_r(x,t)=\sum_{i=1}^{3}\sum_{k=x,y,z}w^r_{i,k}(t)e^r_{i,k}(x)
\]

cardiac 为一个 spatial level：

\[
d_c(x,t)=\sum_{k=x,y,z}w^c_k(t)e^c_k(x)
\]

总共：

```text
respiratory = 3 levels × 3 Cartesian scores = 9 scores
cardiac     = 1 level  × 3 Cartesian scores = 3 scores
total       = 12 scores
```

不要把 Cartesian score 再额外扩展成一个 3-vector mixing matrix，除非有明确论文/源码依据。

## 0.3 S2V-DREME 与 DREME 的 hybrid 关系必须明确

S2V-DREME Eq. (2) 将每 level 的 MBC 写为 3D displacement field，并令：

```text
MBCs:   [L, 3, Nvoxel]
scores: [L, 3]
```

S2V-DREME Eq. (3) 的 SINR generator：

```text
control-point coordinate
→ INR predicts 3D displacement vector
→ cubic B-spline FFD
→ dense MBC
```

因此本项目本轮必须先做一个 **topology audit**：

```text
DREME 3-level × xyz score topology
+
S2V-DREME SINR vector displacement generator
```

应如何一一对应。

严禁保留一个未解释的：

```text
per-level 9 output channels
=
3 scores × 3 vector components
```

然后直接认为等价。

---

# 1. 变更记录规范：继续 append-only `changelog.md`

继续使用 repo 根目录：

```text
changelog.md
```

本轮新增第一条：

```text
v3_change2_codex.md — motion-training scientific correctness 修正
```

要求：

- 中文；
- append-only；
- 不修改/覆盖 v3_change1 的历史条目；
- 记录真实开始时间；
- 记录基线 commit `d947ad71...`；
- 每个关键修正完成后可追加阶段记录；
- 最终 commit 创建后，必须**再追加一条 post-commit 记录**，写入真实最终 SHA；
- 不允许最终仍保留“最终 commit: 待创建”作为唯一结论。

另外，把本次 prompt 归档到：

```text
docs/prompts/v3_change2_codex.md
```

若 `docs/prompts/` 不存在则创建。

以后建议所有正式 Codex prompt 都保存在该目录，changelog 中只引用文件名。

---

# 2. 修复 Phase-1 frequency prior → training 的真实数据接口

这是本轮最高优先级 integration blocker。

当前 training template：

```json
{
  "respiratory_hz": [0.1, 0.7],
  "cardiac_hz": [0.8, 3.0]
}
```

但 Phase-1 `pca_motion.py` 真正输出的：

```text
results/frequency/frequency_bands.json
```

不是这个 schema。

当前 Phase-1 aggregate 大致为：

```text
respiratory:
  verified_band_hz
  per_slice_candidates
  reliable_frequency_distribution_hz
  union_resolution_bins_hz
  resolution_bin_support
  ...

cardiac:
  per_slice_candidates
  reliable_frequency_distribution_hz
  union_resolution_bins_hz
  resolution_bin_support
  ...
```

因此现在不能再要求用户“手工把 Phase-1 JSON 替换成 template”。

## 2.1 新增明确 parser / adapter

建议新增：

```text
src/cardioresp4d/frequency/training_prior.py
```

例如定义：

```python
@dataclass
class TrainingFrequencyPrior:
    respiratory_bands_hz: list[tuple[float, float]]
    cardiac_bands_hz: list[tuple[float, float]]
    baseline_bands_hz: list[tuple[float, float]]
    source_path: str
    source_schema: str
    provenance: dict
```

职责：

```text
Phase-1 aggregate frequency_bands.json
→ validated TrainingFrequencyPrior
```

不要让 trainer 自己解析复杂 JSON。

## 2.2 respiratory prior

优先级：

1. 若 `respiratory.verified_band_hz` 非空，使用其中全部 verified bins；
2. 否则不自动伪造一个 subject-specific respiratory band；
3. 明确报错或进入显式 fallback mode；
4. fallback 只能由 config 显式允许，例如：

```yaml
frequency_prior:
  allow_template_fallback: false
```

formal run 默认 `false`。

## 2.3 cardiac prior

Phase-1 设计本来就不强迫整个扫描只有一个固定 cardiac frequency。

因此支持：

```text
cardiac_bands_hz: list[(low, high)]
```

优先来自：

```text
cardiac.union_resolution_bins_hz
```

或经过明确 quality filtering 的 recurrent/support bins。

不要粗暴 collapse 为：

```text
[min(all), max(all)]
```

因为这样可能把大量非 cardiac 中间频率也算进 forbidden band。

如果需要做 support threshold，规则必须在 parser 中显式写清，并写 test。

## 2.4 baseline frequency bins

DREME Eq. (8) 的 respiratory-score cardiac leakage loss 需要：

```text
ν_b = baseline frequency bins
```

当前 Phase-1 没有直接给出 `baseline_bands_hz`。

本轮需要建立一个明确、可解释的 baseline-bin 构造规则。

必须先核对 DREME-MR 原文/补充材料是否说明 baseline bins 的选择方式。

如果论文没有给出足够可直接复现的细节：

- 不要伪称“论文原实现”；
- 标为 `Paper-derived necessary adaptation`。

推荐 baseline-bin 原则：

- 每个 cardiac forbidden bin 附近选取不属于 respiratory/cardio physiological bands 的邻近频率 bins；
- 与 cardiac bin 使用同样 spectral resolution；
- 不允许 baseline 落入 respiratory band；
- 不允许 baseline 落入 DC；
- 所有规则写入 config/provenance/test。

若找到更直接的原论文规则，优先使用原规则。

---

# 3. 修复 frequency file path

当前 YAML：

```yaml
frequency_bands_json: frequency_bands.json
```

但训练入口使用：

```python
PROJECT_ROOT / frequency_bands_json
```

会错误寻找：

```text
CR_DREME_v3/frequency_bands.json
```

## 3.1 统一 path contract

推荐：

```python
frequency_path = (
    args.source_config.parent
    / source_config["training"]["temporal_auxiliary"]["frequency_bands_json"]
).resolve()
```

即 config 内相对路径默认相对 **config 文件自身所在目录**。

formal real-data run 还应允许 CLI override：

```bash
--frequency-bands /absolute/path/to/results/frequency/frequency_bands.json
```

优先级：

```text
CLI explicit path > config path
```

report 中保存最终实际使用路径。

---

# 4. MBC topology audit：先证明，再修改

当前 `SINRFFDBasis`：

```python
self.siren = BSplineSiren([3, hidden_dim, hidden_dim, 9])
```

产生：

```text
per level 9 channels
→ reshape [3 basis, 3 xyz]
```

再由：

```text
resp_scores[level, xyz]
```

与 3-vector basis 混合。

这与 DREME / S2V-DREME 的 tensor 定义存在明显疑问。

## 4.1 先写 topology audit

在改代码前，在：

```text
IMPLEMENTATION_REPORT.md
```

新增：

```text
MBC topology audit — v3_change2
```

明确列出 DREME-MR 与 S2V-DREME 的 tensor 定义，并说明两者在本项目实际 tensor 中的等价关系。

## 4.2 目标 tensor contract

除非审查发现相反的直接证据，推荐 formal contract 为：

### respiratory MBC

```text
resp_mbc: [B, L=3, N, 3]
```

### respiratory score

```text
resp_scores: [B, L=3, 3]
```

### respiratory DVF

Cartesian-wise element-wise weighting：

```python
weighted = resp_mbc * resp_scores[:, :, None, :]
resp_dvf = weighted.sum(dim=1)
```

即：

\[
d^r_k(x,t)=\sum_i w^r_{i,k}(t)e^r_{i,k}(x)
\]

不存在额外 3×3 mixing。

### cardiac

```text
card_mbc:    [B, 1, N, 3]
card_scores: [B, 1, 3]
```

同样 element-wise：

\[
d^c_k(x,t)=w^c_k(t)e^c_k(x)
\]

## 4.3 SINR output

如果采用以上 contract，每个 SINR level 的 generator 应输出：

```text
3 displacement channels
```

而不是 9。

因此优先评估：

```python
BSplineSiren([3, hidden_dim, hidden_dim, 3])
```

再经 upstream `CubicBSplineFFDTransform` 得到：

```text
[B, 3, dense_x, dense_y, dense_z]
```

然后 query：

```text
[B, N, 3]
```

不要修改 upstream SINR 文件，只修改 adapter 对 upstream class 的实例化和 tensor contract。

## 4.4 必须增加 topology tests

至少：

1. respiratory one level 输出 `[B,N,3]`；
2. three levels 输出 `[B,3,N,3]`；
3. score `[1,0,0]` 只改变 x displacement；
4. score `[0,1,0]` 只改变 y displacement；
5. score `[0,0,1]` 只改变 z displacement；
6. 不允许 cross-axis mixing；
7. zero score → zero DVF；
8. 12-score flatten/unflatten 保持 first 9 respiratory、last 3 cardiac；
9. gradients 同时回到 score encoder 与 active SINR level。

---

# 5. 严格实现 DREME Eq. (6)：`L_MBC`

当前实现 `mean(final_dvf^2)` 不是 DREME 的 MBC normalization。

DREME Eq. (6)：

\[
L_{MBC}
=
\frac{1}{3}
\sum_{k=x,y,z}
\sum_{i=1}^{4}
\left(\|e_{i,k}\|_2^2-1\right)^2
\]

其中四个 level = 3 respiratory + 1 cardiac。

目的：

```text
remove scale ambiguity:
e → αe
w → w/α
```

而不是把 motion amplitude 压到 0。

## 5.1 实现要求

新增/重构到：

```text
src/cardioresp4d/losses/motion_loss.py
```

建议函数名：

```python
dreme_mbc_normalization(...)
```

不要继续用模糊的 `mbc_normalization(fields_mm)`。

## 5.2 norm 的计算

DREME 原文说原 B-spline 版本的 MBC norm 可通过 control points analytically 计算。

本项目现在 MBC 由 SINR + B-spline FFD 生成，因此先评估：

### 优先方案 A

如果可以基于 B-spline control lattice 严格实现等价 norm，优先。

### 允许方案 B

如果 SINR hybrid 无法直接复用 DREME analytic formula，则在固定 physical dense evaluation grid 上计算离散 L2 norm，但必须：

- 标注 hybrid necessary adaptation；
- grid 足够稳定；
- norm 包含 physical voxel volume / spacing normalization，避免 resolution 改变 norm；
- respiratory/card 各自基于自身 domain；
- 不要用最终 time-varying DVF；
- 必须对 **MBC itself** 做 normalization。

## 5.3 tests

构造：

```text
norm = 1 → loss ≈ 0
norm = 2 → loss > 0
norm = 0 → loss > 0
```

并验证：

```text
scale MBC by α
score by 1/α
```

DVF 不变，但 `L_MBC` 明确约束 MBC norm。

最重要：

```text
L_MBC 不能在 MBC→0 时取得最小值
```

---

# 6. 严格实现 DREME Eq. (7)：`L_ZMS`

当前 trainer 的：

```python
score.square().mean()
```

错误惩罚 score amplitude。

DREME Eq. (7)：

\[
L_{ZMS}
=
\frac{1}{12}
\sum_{i,k}
\left|
\frac{1}{N_t}\sum_t w_{i,k}(t)
\right|^2
\]

目的：remove time-independent baseline，而不是让 score 都变 0。

## 6.1 实现要求

使用完整 temporal score batch：

```text
resp_scores [T,3,3]
card_scores [T,1,3]
```

拼接后逐 score channel：

```python
mean_over_time.square().mean()
```

现有 `zero_mean_scores(...)` 如数学正确可复用，但要真正接入 trainer。

## 6.2 tests

```text
[-1,+1,-1,+1] → L_ZMS ≈ 0
[1,1,1,1]      → L_ZMS > 0
[-10,+10]       → L_ZMS ≈ 0
```

证明不会错误抑制正常 motion amplitude。

---

# 7. 严格实现 DREME Eq. (8)：respiratory scores 中 cardiac leakage

当前 generic `frequency_leakage()` 的 forbidden-power ratio 不能作为 formal DREME `L_c`。

DREME Eq. (8)：

\[
L_c
=
\frac{1}{N_c}
\sum_{\omega\in\nu_c,\omega'\in\nu_b}
\sum_{i,k}
\left|
F[w^r_{i,k}(t)](\omega)
-
F[w^r_{i,k}(t)](\omega')
\right|^2
\]

其中：

```text
ν_c = cardiac frequency bins
ν_b = baseline frequency bins
```

baseline subtraction 用于补偿不同 respiratory score level/direction 的幅值差异。

建议新增：

```text
src/cardioresp4d/losses/frequency_loss.py
```

函数：

```python
dreme_cardiac_leakage_in_resp(...)
```

不要把 Eq.8 与 generic `frequency_leakage()` 混为一个 formal 函数。

---

# 8. 严格实现 DREME Eq. (9)：cardiac scores 中 respiratory leakage

DREME Eq. (9)：

\[
L_r
=
\frac{1}{N_r}
\sum_{\omega\in\nu_r}
\sum_{i,k}
\left|
F[w^c_{i,k}(t)](\omega)
\right|^2
\]

这里没有 Eq.8 的 baseline subtraction。

建议函数：

```python
dreme_respiratory_leakage_in_card(...)
```

---

# 9. 非均匀 timestamp：保留必要 adaptation，但不要假装普通 FFT

DREME 原始数据有连续时间采样；本项目必须使用真实 `timestamp_s`。

因此保留：

\[
F(\omega)=\sum_t w(t)e^{-j2\pi\omega t}
\]

作为 **nonuniform direct Fourier evaluation**。

重构成独立函数：

```python
nonuniform_dft_at_frequencies(scores, timestamps, frequencies_hz)
```

## 9.1 不再用 `arange()/duration` 自动频率格点作为唯一 formal path

正式 frequency bins 来自 `TrainingFrequencyPrior`。

每个 band 根据 temporal sequence duration / frequency precision 映射到实际 evaluation frequencies。

必须记录最终 frequency evaluation grid。

## 9.2 multi-band support

frequency loss API 支持：

```python
list[tuple[float,float]]
```

测试：

- 两个 separated cardiac bands；
- 一个 respiratory verified band；
- irregular timestamps；
- empty band explicit handling；
- gradients finite。

---

# 10. temporal auxiliary 必须覆盖完整时间跨度

当前 `temporal_batch(max_items=8)` 永远取按时间排序后的前 8 帧，对低频 respiration 不够。

## 10.1 formal default

对于一个 fixed slice location，优先使用全部 qc-valid frames，通常约 50 帧。

配置：

```yaml
temporal_auxiliary:
  mode: full_location_sequence
  max_frames: 50
```

## 10.2 若必须 subsample

只能做 full-span stratified / uniformly-spaced sampling，覆盖 first/middle/last 整个 duration，不能只取前 N 帧。

## 10.3 hard invalid 永不进入 temporal loss

```text
slice_local_scale_absolute
manual_exclusion
```

以及任何 `qc_valid=false` 观测不得进入。

## 10.4 sequence sufficiency gate

在计算 frequency loss 前检查：

```text
duration
frequency resolution
number of valid samples
```

若不能解析目标 band：

- 不静默返回 0；
- report 记录 `frequency_loss_skipped`, reason, duration_s, df_hz, target band；
- formal run 若大量 locations 不足，应启动前 fail-fast。

---

# 11. `L_ZMS` 与 frequency loss 必须使用 temporal batch

新的职责分离：

### per-observation reconstruction branch

负责：

```text
data fidelity
image regularization
motion field regularization / smoothness
```

### temporal auxiliary branch

同一个 fixed location：

```text
resp_scores[T,3,3]
card_scores[T,1,3]
timestamps[T]
```

负责：

```text
L_ZMS
L_c
L_r
```

删除 single-frame `score.square().mean()` 作为 formal score regularizer。

---

# 12. smoothness 不再固定只在 4×4×4 上计算

当前 4³ grid 太粗。

新增 config，例如：

```yaml
motion_regularization:
  evaluation_grid:
    mode: fixed_physical
    shape: [16,16,16]
```

或者 match finest active dense MBC grid，但需控制成本。

Stage2a 可较粗，Stage2c/Stage3 必须能看到 fine-level variation。

同时分别记录：

```text
smooth_resp
smooth_card
```

不要只把 `respiratory + cardiac` 后算一个 smoothness。

---

# 13. runtime config 必须真正控制模型

当前 `source_first.yaml` 已记录 canonical/PSF 等参数，但训练入口大部分没有真正传入。

建议新增：

```text
src/cardioresp4d/training/build_model.py
```

定义：

```python
build_source_first_model(source_config, domain, n_dynamic_frames, device)
```

所有 formal hyperparameters 从 config 读取。

## 13.1 NeSVoRCanonicalAdapter

检查并真正传递：

```text
coarsest_resolution
finest_resolution
level_scale
features_per_level
log2_hashmap_size
latent_dim
width
depth
```

不修改 upstream NeSVoR。

## 13.2 SINR config

显式写：

```yaml
respiratory_mbc:
  logical_control_shapes:
    - [8,8,8]
    - [12,12,12]
    - [16,16,16]
  cps: ...
  hidden_dim: ...

cardiac_mbc:
  logical_control_shape: [16,16,16]
  cps: ...
  taper_mm: ...
```

不要只写 `levels: 3`。

## 13.3 effective config

训练启动后保存：

```text
effective_config.json
```

至少包括：

- canonical INR actual parameters；
- PSF n_samples；
- respiratory/card logical grid；
- SINR cps；
- cardiac taper；
- motion encoder channels/bands；
- normalization mode；
- loss weights；
- temporal batch mode；
- resolved frequency prior；
- stage steps；
- learning rates。

---

# 14. learning rate 按模块显式配置

S2V-DREME 原文 Stage II 使用 spatial INR 与 motion model 不同 LR。本项目不要求机械复制数值，但不能所有模块永久共用隐式 `1e-3`。

新增配置，例如：

```yaml
optimizer:
  stage1:
    canonical_lr: 5e-4
  stage2:
    canonical_lr: 1e-4
    motion_lr: 5e-4
  stage3:
    canonical_lr: 1e-4
    motion_lr: 5e-4
    uncertainty_lr: 1e-4
```

这些值若为本项目默认，report 必须明确写：

```text
project defaults, not claimed as DREME official defaults
```

---

# 15. `SOURCE_LOCK.json` 增加 runtime hash verification

新增：

```text
src/cardioresp4d/adapters/source_lock.py
```

例如：

```python
verify_vendored_source_lock(project_root)
```

训练启动时：

1. 读 `third_party/SOURCE_LOCK.json`；
2. 对 key source files 重算 SHA256；
3. formal training 不匹配直接 fail；
4. report 保存 verification result。

至少校验：

```text
NeSVoR/nesvor/inr/models.py
NeSVoR/nesvor/utils/psf.py
SINR/networks/networks.py
SINR/models/transformation.py
film/vr/models/filmed_net.py
```

---

# 16. Geometry acquisition scalar normalization 小修正

当前：

```python
pixel_spacing_mm / extent[:2]
slice_thickness_mm / extent.mean()
```

把 slice-local row/column spacing 与 patient-world X/Y AABB extent 对应，不适合 oblique 2CH/4CH。

优先改成 canonical AABB extent 在：

```text
row direction
column direction
normal
```

上的 projection extent，再构造：

```text
row_spacing / extent_row
col_spacing / extent_col
thickness / extent_normal
```

保持：

- center → canonical normalized Fourier；
- directions → unit vectors；
- acquisition scalars → low-frequency dimensionless values。

补 oblique geometry test。

---

# 17. normalization 大内存问题做最小防护

当前 per-series normalization 可能 `np.concatenate(all_frame_pixels)`，正式数据会浪费 RAM。

本轮不做复杂重构，只需提供受控 deterministic subsampling，例如：

```text
每 frame 固定 stride/seed subsample
→ 聚合估计 1/99 percentile
```

要求：

- deterministic；
- 只用 qc-valid frames；
- report 保存 sample count；
- 小 fixture 上与 exact percentile 误差可控。

如果真实规模确认内存安全，可将此项标记为 P1，但必须在 report 明确说明。

---

# 18. 修复 test suite 分类，不长期忽略 `tests.test_data`

上一轮 `tests.test_data` 有 5 个 Phase-1 fixture failure。

必须分类：

### 若是 legacy Phase-1 单视图 unit tests

不要让 v3 formal 三视图 validator 无条件污染底层 dataset unit test；修改 fixture 或 validator injection，使 unit test 独立验证 dataset。

### 若是 full-pipeline integration tests

更新 fixture 为：

```text
SAX + 2CH + 4CH
```

最终给出：

```text
core v3 tests
legacy tests
full repository tests
```

三类结果。

尽量做到：

```bash
python -m unittest discover -s tests
```

全绿。

若仍有 legacy incompatible tests，列出名称和原因，不允许静默排除。

---

# 19. 重新整理 loss module

建议：

```text
src/cardioresp4d/losses/
├── image_loss.py
├── motion_loss.py
├── frequency_loss.py
└── stage_aware.py
```

### image_loss.py

- Stage1 unit-variance MSE；
- Stage3 NeSVoR-derived Gaussian NLL wrapper/direct call；
- 不重复实现 uncertainty。

### motion_loss.py

- DREME `L_MBC`；
- DREME `L_ZMS`；
- DVF smoothness。

### frequency_loss.py

- nonuniform DFT primitive；
- DREME Eq.8 `L_c`；
- DREME Eq.9 `L_r`。

### stage_aware.py

只负责 stage → 哪些 loss 激活。

legacy 函数如需保留，显式命名 `legacy_...` / `ablation_...`。

---

# 20. Stage loss contract

## Stage1

```text
motion OFF
uncertainty OFF
canonical INR trainable
```

\[
L=L_{data}+\lambda_{image}R_{image}
\]

本轮不改变 v3 source-first Stage1 直接使用 real valid dynamic frames 的决策。

## Stage2a / 2b / 2c

\[
L=
L_{data}
+\lambda_{image}R_{image}
+\lambda_{MBC}L_{MBC}
+\lambda_{smooth,r}L_{smooth,r}
+\lambda_{ZMS}L_{ZMS,r}
+\lambda_cL_c
\]

此时 cardiac contribution OFF、uncertainty OFF。

`L_ZMS / L_c` 必须来自 temporal auxiliary sequence。

## Stage3

\[
L=
L_{NLL}
+\lambda_{image}R_{image}
+\lambda_{MBC}L_{MBC}
+\lambda_{smooth,r}L_{smooth,r}
+\lambda_{smooth,c}L_{smooth,c}
+\lambda_{ZMS}L_{ZMS}
+\lambda_cL_c
+\lambda_rL_r
\]

cardiac + respiratory + uncertainty joint train。

---

# 21. loss 名称改成论文一致

当前模糊命名：

```text
mbc
smooth
freq_resp
freq_card
score
```

formal mainline 改为：

```yaml
loss_weights:
  image: ...
  mbc_normalization: ...
  smooth_resp: ...
  smooth_card: ...
  zero_mean_score: ...
  cardiac_leakage_in_resp: ...
  respiratory_leakage_in_card: ...
```

report 同样按这些名字输出。

---

# 22. 必须新增的科学 contract tests

## 22.1 Phase1 frequency prior parser

真实 Phase-1 schema fixture：

```text
respiratory.verified_band_hz
cardiac.union_resolution_bins_hz
```

验证 parser、多 band、missing verified respiratory、config-relative path、CLI override。

## 22.2 MBC topology

验证：

```text
resp MBC [B,3,N,3]
resp score [B,3,3]
card MBC [B,1,N,3]
card score [B,1,3]
```

且无 xyz cross mixing。

## 22.3 Eq.6

```text
unit norm → loss 0
zero MBC → loss > 0
double norm → loss > 0
```

并证明 minimum 不在 zero field。

## 22.4 Eq.7

```text
[-1,+1] → 0
[1,1] → positive
[-10,+10] → 0
```

## 22.5 Eq.8

synthetic respiratory score：

```text
large respiratory component
+
small cardiac contamination
```

cardiac contamination 增大时 `L_c` 增大；baseline subtraction 必须实际参与。

## 22.6 Eq.9

synthetic cardiac score 中 respiratory contamination 增大时 `L_r` 增大。

## 22.7 irregular timestamps

使用不均匀时间：

```text
[0.00, 0.09, 0.21, 0.31, ...]
```

确认 finite、differentiable、nonzero gradient、多 band。

## 22.8 full temporal span

50-frame fixture 中 temporal batch 必须覆盖最后时间点；若 subsample 则覆盖完整 duration。

## 22.9 config → runtime

修改 test YAML：

```text
psf.n_samples = 7
latent_dim = 5
motion_hidden_dim = ...
```

构建后 assert runtime 真正等于 config。

## 22.10 source lock

```text
correct hash → pass
wrong hash → fail
```

## 22.11 smoothness

```text
constant DVF → smooth≈0
linear DVF → finite
high-frequency DVF → higher smooth loss
```

## 22.12 Stage2 / Stage3 total loss

Stage2a 必须包含：

```text
data + image + LMBC + smooth_resp + LZMS + Lc
```

Stage3 必须包含：

```text
NLL + image + LMBC + smooth_resp + smooth_card + LZMS + Lc + Lr
```

确保旧：

```text
score.square().mean()
final_dvf.square().mean() as LMBC
```

不再进入 formal total loss。

---

# 23. logging / report 必须增加

每个 stage 至少记录：

```text
data
image
mbc_normalization
smooth_resp
smooth_card
zero_mean_score
cardiac_leakage_in_resp
respiratory_leakage_in_card
```

未启用项显示 inactive 或明确 0 + reason。

另记录：

```text
frequency prior source path
resolved respiratory bands
resolved cardiac bands
baseline bands
temporal duration
temporal frame count
frequency resolution
frequency-loss skipped count
```

---

# 24. 本轮不要做

不要：

- 改 NeSVoR upstream；
- 改 SINR upstream；
- 改 FiLM upstream；
- 引入新大型 motion backbone；
- 加 transformer/VAE/diffusion；
- 改 Phase1 PCA 主流程，除非做最小 non-breaking schema compatibility；
- 删除 `frequency_bands.py` 原丰富输出；
- 把 cardiac frequency 强行 collapse 为一个单点；
- 改 hard-QC valid/invalid；
- 把 canonical full-FOV 改成 cardiac crop；
- 进入 Phase4 inference；
- 跑长 Stage3。

---

# 25. 测试执行顺序

先 targeted：

```bash
conda run --no-capture-output -n knesvr_torch python -m unittest \
  tests.test_frequency_training_prior \
  tests.test_dreme_motion_losses \
  tests.test_dreme_frequency_losses \
  tests.test_sinr_adapter \
  tests.test_v3_change2_contracts \
  tests.test_v3_change1_contracts \
  tests.test_v3_change1_training \
  tests.test_unified_progressive_smoke \
  tests.test_source_backed_adapters \
  tests.test_source_first_config
```

然后：

```bash
conda run --no-capture-output -n knesvr_torch \
  python -m unittest discover -s tests
```

full discover 失败时不要隐藏，列出 exact failing tests 并分类。

---

# 26. 本轮完成后才允许的最短 GPU gate

只有以下全部满足：

```text
Phase1 frequency parser pass
MBC topology confirmed
Eq6 pass
Eq7 pass
Eq8 pass
Eq9 pass
full temporal span pass
config-runtime equality pass
source-lock verification pass
core regression pass
```

才允许准备：

```text
Stage1 100 steps
→ Stage2a 100 steps
```

本轮 Codex 默认只给最终推荐 command，不执行真实长 GPU run。

---

# 27. 下一次 GPU 命令必须使用真实 Phase-1 frequency 文件

命令格式应支持：

```bash
python scripts/train_source_first.py \
  --source-config configs/source_first.yaml \
  --frequency-bands /absolute/path/to/results/frequency/frequency_bands.json \
  --manifest /absolute/path/to/results/dicom_manifest.csv \
  --qc-table /absolute/path/to/results/acquisition_qc/acquisition_qc.csv \
  --canonical-domain /absolute/path/to/results/domain/canonical_domain.json \
  --output-dir /absolute/path/to/results/training/source_first_short \
  --device cuda \
  --stage1-steps 100 \
  --stage2a-steps 100 \
  --stage2b-steps 0 \
  --stage2c-steps 0 \
  --stage3-steps 0
```

不要再要求用户手工把真实 Phase-1 JSON 改写成 template schema。

---

# 28. 最终 Codex 回报格式

完成后按以下顺序回复：

```text
1. 本轮起始 commit
2. 实际修改文件列表
3. Phase1 frequency JSON → training prior 如何接通
4. 最终 respiratory/card/baseline frequency prior 数据结构
5. DREME/S2V-DREME MBC topology 审计结论
6. 是否删除原 per-level 9-channel cross-axis mixing；若没有，给出直接论文/源码证据
7. Eq.6 L_MBC 最终公式和代码位置
8. Eq.7 L_ZMS 最终公式和代码位置
9. Eq.8 L_c 最终公式和代码位置
10. Eq.9 L_r 最终公式和代码位置
11. nonuniform timestamp adaptation 如何实现
12. temporal batch 是否覆盖完整 fixed-location 时间范围
13. smoothness evaluation grid 最终设置
14. YAML 中哪些 model 参数现在真正进入 runtime
15. effective_config 输出文件
16. SOURCE_LOCK runtime hash verification 结果
17. tests.test_data 原 5 个失败如何处理
18. targeted tests 结果
19. full unittest discover 结果
20. changelog.md 新增条目
21. docs/prompts/v3_change2_codex.md 是否归档
22. 当前仍未解决的问题
23. 下一步最短 Stage1→Stage2a GPU command
24. 最终 git commit SHA
```

---

# 29. 最重要的停止条件

如果核对 DREME-MR / S2V-DREME 后发现：

```text
DREME Cartesian MBC topology
```

与：

```text
S2V-DREME SINR vector MBC topology
```

无法无歧义映射，**不要直接凭经验选择 9-channel 或 3-channel 实现**。

先：

1. 写 tensor-level 推导；
2. 列明两个论文公式；
3. 检查 upstream SINR output contract；
4. 在 `IMPLEMENTATION_REPORT.md` 记录候选；
5. 选择能严格满足以下条件的实现：
   - 9 respiratory scores；
   - 3 cardiac scores；
   - each score 对应明确 Cartesian component；
   - no unexplained cross-axis mixing；
   - score × MBC scale ambiguity 可被 Eq.6/Eq.7 正确约束。

若仍不确定，停止该模块并报告，不要把猜测提交为 formal mainline。

---

# 30. 本轮验收定义

只有以下语义同时成立才算完成：

\[
\boxed{\text{Phase1 prior}\rightarrow\text{training frequency loss}}
\]

真实接通；

\[
\boxed{L_{MBC}=\text{DREME Eq.6 semantic equivalent}}
\]

而不是 DVF amplitude penalty；

\[
\boxed{L_{ZMS}=\text{DREME Eq.7}}
\]

而不是 score amplitude penalty；

\[
\boxed{L_c/L_r=\text{DREME Eq.8/Eq.9 + documented nonuniform-time adaptation}}
\]

且：

\[
\boxed{\text{config}=\text{runtime effective parameters}}
\]

同时以下 `v3_change1` 已修好的 contract 不得回归：

```text
source-first
hard-QC
full-FOV canonical domain
oblique PSF
NeSVoR uncertainty
sequential pullback
```
