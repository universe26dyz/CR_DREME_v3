# CR_DREME_v3 — v3_change4 Codex Prompt
## Per-location respiratory prior + Stage3 cardiac warm-up/joint/uncertainty split + formalize server hotfixes

你现在需要在 **CR_DREME_v3** 上完成一次正式的 `v3_change4` 修复。请直接执行，不要反复向我确认；只有遇到真正无法继续的 blocker（例如目标仓库不存在、核心依赖完全不可用、数据/接口与本文描述根本不一致）时才停下来询问。

---

# 0. 工作方式与硬约束

## 0.1 仓库与分支

目标仓库：

```text
/home/universe/SVR/code/CR_DREME_v3
```

GitHub：

```text
git@github.com:universe26dyz/CR_DREME_v3.git
```

目标分支：

```text
dev/cardioresp4d
```

开始前先：

```bash
cd /home/universe/SVR/code/CR_DREME_v3
git status
git branch --show-current
git log -5 --oneline
```

**不要 reset、不要 clean、不要删除任何旧文件、不要覆盖未提交改动。**

如果当前 working tree 有未提交改动：
1. 先审计这些改动；
2. 判断是否正是本文提到的 server/local hotfix；
3. 保留有价值改动；
4. 不允许为了“干净”直接丢弃。

如果可用，请使用 Superpowers 工作流，至少采用：
- `systematic-debugging`
- `test-driven-development`
- `verification-before-completion`

对于本次多步骤修改，可先生成一个简洁实现计划，但不要浪费 token 写冗长背景。

---

## 0.2 不允许做的事情

本次 **禁止**：

- 不要重新设计整个 pipeline。
- 不要恢复旧的 `initial_reference` mainline。
- 不要把 cardiac box 改成 reconstruction crop。
- 不要修改 NeSVoR/SINR upstream 数学实现，除非是 adapter 层明确的 device/runtime 修复。
- 不要修改 `third_party` 中 vendored upstream 代码。
- 不要更改 `SOURCE_LOCK.json` 的 upstream commit。
- 不要把 Eq.8/Eq.9 的 NUDFT `/N` normalization 去掉。
- 不要把 Eq.8/Eq.9 loss weight 粗暴乘 `2500`。
- 不要在这次 change4 新增“强制 target-band attraction / positive spectral reward”之类的新 frequency loss。
- 不要在本地或服务器启动长时间 real-data/GPU training。
- 不要为了测试重新跑已经验证过的 Phase1 全流程。
- 不要删除旧 checkpoint / result。
- 不要把目前 Stage3 的失败归因于“训练步数不够”然后直接加长训练。

本次目标是**修正确认过的语义/训练调度问题**，然后只做 unit/smoke validation。

---

## 0.3 用户偏好的实现原则

1. 能直接复用成熟源码/现有模块就不要重写。
2. 对 normalization、Fourier、geometry、regularization、checkpoint 等小细节，优先核对现有源码与已有测试。
3. 新增/修改代码中的关键适配请保留中文注释，说明：
   - 来源；
   - 为什么必须适配；
   - 与 upstream/论文的差别。
4. 不要为了“架构漂亮”做无关重构。
5. 低频监控，避免无意义重复执行。
6. 测试应先 targeted，再 full suite。
7. 完成后给出清晰的修改摘要、测试结果、commit SHA。

---

# 1. 当前项目事实与本次修复背景

当前正式 v3 采用：

- NeSVoR official INR 作为 canonical 3D anatomy；
- NeSVoR PSF；
- SINR official `BSplineSiren` + `CubicBSplineFFDTransform` 作为 respiratory/cardiac MBC basis；
- FiLM-based single-frame geometry-conditioned encoder 预测 DREME-style scores；
- respiratory progressive levels：`8^3 -> 12^3 -> 16^3`；
- cardiac：local cardiac box `16^3`；
- full acquisition-supported canonical FOV；
- cardiac box 只用于 local cardiac motion / sampling，不是 reconstruction crop；
- hard invalid 只包括：
  - `slice_local_scale_absolute`
  - `manual_exclusion`

当前正式 stage：

```text
Stage1
Stage2a
Stage2b
Stage2c
Stage3
```

Stage3 目前一次性同时打开：
- cardiac MBC
- shared FiLM
- canonical refinement
- respiratory refinement
- uncertainty

这个设计经过真实 GPU short-run 后发现不理想。

---

# 2. 已完成真实 GPU 数值验证：必须保留的事实

不要重复这些实验。

## 2.1 Stage1 → Stage2c 数值稳定

Stage2c 100 steps 结束时：

```text
data loss          = 0.0203923
resp DVF RMS       = 0.47394 mm
resp DVF max       = 3.87580 mm
global_step        = 400
```

Stage1 / Stage2a / Stage2b / Stage2c 均稳定，无 NaN/Inf，无 motion explosion。

---

## 2.2 当前旧 Stage3 100 steps 数值稳定，但 cardiac motion 没有有效学出来

旧 Stage3：

```text
loss first         = 0.418535
loss last          = 0.012313

resp DVF RMS       = 0.48055 mm
resp DVF max       = 4.28405 mm

card DVF RMS       = 0.03286 mm
card DVF max       = 0.14625 mm

card score std     ≈ 0.115
```

gradient ownership：

```text
inr=True
film=True
respiratory_sinr=True
cardiac_sinr=True
uncertainty=True
```

因此数值稳定，但是 cardiac displacement 几乎为零。

---

## 2.3 Cardiac ablation

同一 Stage3 checkpoint、同一组 cardiac-box pixels：

```text
resp-only MSE      = 0.02405786
resp+card MSE      = 0.02397674
relative improvement = 0.337%
```

即 cardiac branch 对 reconstruction 的收益仅 **0.337%**。

raw cardiac basis 并未塌缩：

```text
raw cardiac basis RMS = 0.26584 mm
raw cardiac basis max = 0.98254 mm
```

所以问题不是 “cardiac SIREN basis 完全没输出”，而是 score / identifiability / training schedule。

---

## 2.4 Uncertainty diagnostic

7150 个有效 frame：

```text
frame variance median = 1.0
frame variance mean   ≈ 0.99972
non-zero learned frame log-vars = 293 / 7150
```

pixel scale：

```text
median ≈ 2.73e-4
mean   ≈ 2.68e-3
```

最终 total variance 基本恒等于 1：

```text
variance median ≈ 1.000000
variance mean   ≈ 0.999766
```

因此短 Stage3 中 uncertainty 并没有真正形成有意义的 heteroscedastic variance，而是基本退化为固定 variance≈1。

**结论：uncertainty 不应在 cardiac branch 第一次出现时同时开放。**

---

# 3. Frequency diagnostic：已确认的问题

## 3.1 Eq.8 / Eq.9 scaling 不是根因

当前 NUDFT：

```python
... / scores.shape[0]
```

真实 diagnostic：

```text
Eq8 normalized mean  = 4.73e-11
Eq8 * N^2            = 1.18e-7

Eq9 normalized mean  = 4.90e-11
Eq9 * N^2            = 1.23e-7
```

即使乘回 `N^2` 仍很小。

因此：

**不要修改 NUDFT normalization。**
**不要把 frequency loss weight 乘 2500。**

---

## 3.2 当前 score 的 physiological semantics 不正确

真实 Stage3 checkpoint：

```text
resp_std mean        = 0.20074
card_std mean        = 0.12098

resp_target amp      = 5.22e-6
resp_wrong amp       = 3.51e-6

card_target amp      = 3.11e-6
card_wrong amp       = 4.32e-6

resp dominant peak   ≈ 0.5294 Hz
card dominant peak   ≈ 0.5294 Hz
```

两个 head 的 dominant peak 都跑到了约 `0.529 Hz`。

当前 Eq.8/Eq.9 是 negative-only cross-frequency suppression：
- Resp 不应包含 cardiac frequency；
- Card 不应包含 respiratory frequency；

但它们**不会强制 score 必须在自己的 target band 内**。

本次 change4 **先不新增 target-band attraction loss**。
先修下面两个已确认 bug + training schedule。

---

# 4. Phase1 respiratory evidence：确认 global respiratory prior 不适合所有 location

Phase1 一共有 143 个 fixed locations，143/143 都得到 reliable respiratory candidate：

```text
GLOBAL VERIFIED RESP BAND:
[0.2923976608, 0.4093567251]

reliable respiratory candidate distribution:

0.116959 Hz : 15 slices
0.233918 Hz : 46 slices
0.350877 Hz : 79 slices
0.467836 Hz : 3 slices
```

global band 只覆盖约：

```text
79 / 143 = 55.2%
```

其余约 44.8% location 有自己的可靠 respiratory candidate。

Phase1 本身已经保存：

```text
respiratory.per_slice_candidates
cardiac.per_slice_candidates
```

但当前 formal training prior 只对 cardiac 做 per-location override。

---

# 5. 已确认代码级 Bug A：per-location respiratory prior 被丢弃

当前：

```text
src/cardioresp4d/frequency/training_prior.py
```

逻辑大致为：

```python
respiratory = global verified respiratory band

for cardiac candidate:
    local_cardiac = ...
    locations[key] = LocationFrequencyPrior(
        respiratory,      # BUG: 仍然是 global respiratory
        local_cardiac,
        ...
    )
```

这意味着：

```text
location respiratory -> global
location cardiac     -> local
```

这是错误的。

## 5.1 change4 要求

对 respiratory 和 cardiac **分别**读取：

```text
payload["respiratory"]["per_slice_candidates"]
payload["cardiac"]["per_slice_candidates"]
```

每个 reliable candidate：

```python
half = df_hz / 2
local_band = [
    frequency_hz - half,
    frequency_hz + half
]
```

要求：

- `frequency_hz` finite；
- `df_hz > 0`；
- lower > 0（不允许碰 DC）；
- duplicate `slice_key` fail-fast；
- `slice_key` 必须保持 `view/slice_id`；
- reliable candidate 若字段缺失必须 fail-fast；
- unreliable/missing local candidate 才 fallback 到 global evidence。

**resp 和 card fallback 要独立。**

例如某 location：

```text
resp reliable, card unreliable
```

应为：

```text
resp -> local
card -> global
```

不能因为 cardiac 不可靠就把 respiratory 也退回 global。

location key 集合应由：

```text
union(respiratory per-slice keys, cardiac per-slice keys)
```

构造。

---

# 6. Local baseline pair 也必须使用 local respiratory occupancy

当前 `_baseline_pairs()` 会避免 baseline band 与：

```python
occupied = respiratory + cardiac
```

重叠。

因此在 per-location prior 中，构造：

```python
_baseline_pairs(
    local_cardiac,
    local_respiratory,
    ...
)
```

不能再传 global respiratory。

---

# 7. Prior provenance 必须可审计

不要只保留一个模糊的：

```text
source="phase1_per_location"
```

至少能在 report/json 中明确判断：

```text
respiratory_source:
    phase1_per_location
    or phase1_global_fallback

cardiac_source:
    phase1_per_location
    or phase1_global_fallback
```

推荐对 `LocationFrequencyPrior` 增加显式字段，例如：

```python
respiratory_source: str
cardiac_source: str
```

如果你认为有更小、更稳的实现，可以采用等价方案，但最终 JSON/report 必须能分别审计 resp/card 来源。

全局 `TrainingFrequencyPrior.provenance` 也应至少记录：

```text
global_respiratory_source = respiratory.verified_band_hz
global_cardiac_source = cardiac.union_resolution_bins_hz
per_location_respiratory_source = respiratory.per_slice_candidates
per_location_cardiac_source = cardiac.per_slice_candidates
local_band_rule = frequency ± df/2
fallback_rule = independently fallback to global when local candidate unavailable/unreliable
```

---

# 8. 已确认代码级 Bug B：Eq.9 没有使用 resolved local prior

当前：

```text
src/cardioresp4d/training/trainer.py
```

`_temporal_components()` 已经：

```python
prior = self.frequency_prior.for_location(...)
```

Eq.8 正确使用：

```python
prior.cardiac_baseline_pairs
```

但是 Eq.9 当前错误使用：

```python
self.frequency_prior.respiratory_bands_hz
```

必须改成：

```python
prior.respiratory_bands_hz
```

并加 regression test，证明两个不同 location 的 Eq.9 **实际收到不同 respiratory bands**。

这不是仅测试 `for_location()` 返回值，必须测试 trainer 的 `_temporal_components()` call path。

---

# 9. Stage3 必须拆成 Stage3a / Stage3b / Stage3c

当前旧 Stage3 一次性打开太多自由度。

新的正式 schedule：

```text
Stage1
Stage2a
Stage2b
Stage2c
Stage3a
Stage3b
Stage3c
```

Stage1–Stage2c **语义完全不改**。

---

# 10. Stage3a：cardiac warm-up

目标：

> 在 canonical anatomy 和 respiratory solution 固定时，让 cardiac branch 必须解释剩余 cardiac residual。

## 10.1 Forward semantics

Stage3a：

```text
resp active levels = 3
cardiac enabled = true
uncertainty enabled = false
```

forward 仍然：

```text
observation
 -> cardiac pullback
 -> respiratory pullback
 -> PSF
 -> canonical INR
```

保持目前 DREME sequential convention：

```text
x_ref = y + d_c(y) + d_r(y + d_c(y))
```

不要改 composition order。

---

## 10.2 Trainable ownership

Stage3a **只允许**：

```text
cardiac_mbc          trainable
film_encoder.card    trainable
```

必须 freeze：

```text
canonical INR
respiratory MBC
film shared image encoder
film FiLM block
film geometry MLP
film shared head
film respiratory score head
uncertainty
```

也就是说：

**不是整个 FiLM trainable，而是只允许 cardiac score head trainable。**

当前 `GeometryFiLMMotionEncoder` 有：

```python
image
film
geometry_mlp
head
resp
card
```

Stage3a 的目标是保留 Stage2c 已学好的 respiratory score semantics，不让 shared FiLM 在 cardiac warm-up 时把 respiratory 解重新改掉。

不要拆掉当前 encoder；只做最小 trainability control。

---

## 10.3 Stage3a data term

必须使用普通：

```text
MSE
```

不要 Gaussian NLL。

因为 uncertainty 此阶段关闭。

---

## 10.4 Stage3a regularization

保持现有 DREME/SINR regularizers：

- MBC normalization；
- smooth_resp（即便 respiratory frozen，可用于 metric；如果作为 constant 进入 total 也不影响 gradient，但不要因此重写整个 loss framework）；
- smooth_card；
- zero mean；
- Eq.8；
- Eq.9（必须 local respiratory prior）。

不要新增 target-band attraction loss。

---

# 11. Stage3b：cardiorespiratory joint refinement

Stage3b：

```text
resp active levels = 3
cardiac enabled = true
uncertainty enabled = false
```

Trainable：

```text
canonical INR
full FiLM encoder
respiratory MBC
cardiac MBC
```

Freeze：

```text
uncertainty
```

data term：

```text
MSE
```

regularizers：

```text
image regularization
MBC normalization
smooth_resp
smooth_card
zero_mean
Eq8
Eq9(local respiratory prior)
```

目的：

> cardiac 先在 Stage3a 得到可识别初始化，再允许 anatomy/resp/card 联合细化。

---

# 12. Stage3c：uncertainty refinement

只有 Stage3c 才：

```text
uncertainty enabled = true
```

Trainable：

```text
canonical
full FiLM
resp MBC
card MBC
uncertainty
```

data term：

```text
Gaussian NLL
```

当前 NeSVoR-derived uncertainty 语义先保持不变：

```text
variance = pixel_scale^2 + frame_variance
```

本次不重构 uncertainty architecture。

重点只是：

**不要让 uncertainty 与 cardiac 第一次出现同时竞争 residual。**

---

# 13. Stage contract 必须成为单一事实来源

当前：

```text
src/cardioresp4d/training/stage_contract.py
```

请扩展为：

```text
stage1
stage2a
stage2b
stage2c
stage3a
stage3b
stage3c
```

不要在 trainer/model 中继续堆新的：

```python
if stage == ...
```

字符串特判。

至少让这些行为由 contract 驱动：

```text
active_respiratory_levels
enable_film
enable_cardiac
enable_uncertainty
data term type
trainability mode
```

其中 Stage3a 需要能表达：

```text
film_card_head_only
```

如果需要新增 contract 字段，优先使用清晰 enum/string，例如：

```python
film_train_mode:
    "none"
    "all"
    "card_head_only"
```

不要用多个互相可能矛盾的 boolean。

---

# 14. Trainer 必须改为 contract-driven，而不是 “stage3” 字符串驱动

当前 trainer 中存在：

```python
if stage == "stage3":
```

用于：
- NLL
- smooth_card
- card score metrics
- Eq.9
- optimizer stage key

change4 后必须改为类似：

```python
contract = stage_contract(stage)

if contract.enable_uncertainty:
    data = GaussianNLL
else:
    data = MSE

if contract.enable_cardiac:
    add cardiac losses / metrics / Eq9
```

Stage3a/3b/3c 行为不能靠复制三份代码。

---

# 15. Optimizer ownership 与 state continuity

当前 optimizer 会跨 stage 保留 Adam state。

必须继续满足：

```text
Stage1 -> 2a -> 2b -> 2c -> 3a -> 3b -> 3c
```

optimizer state continuity 不被破坏。

Stage3a：

```text
只有 cardiac_mbc + film.card 有 gradient
```

即使旧 optimizer `film` param group 已经包含 entire FiLM params，也可以：
- freeze shared params；
- 只有 `film.card` requires_grad=True；
- Adam 对 grad=None 参数不会更新。

不要因为 Stage3a 引入新的全新 optimizer，除非有充分理由并测试 state semantics。

建议 config 显式加入：

```yaml
training:
  optimizer:
    stage3a:
      film_lr: 0.001
      cardiac_mbc_lr: 0.001

    stage3b:
      canonical_lr: 0.001
      film_lr: 0.001
      respiratory_mbc_lr: 0.001
      cardiac_mbc_lr: 0.001

    stage3c:
      canonical_lr: 0.001
      film_lr: 0.001
      respiratory_mbc_lr: 0.001
      cardiac_mbc_lr: 0.001
      uncertainty_lr: 0.001
```

先不要做 learning-rate tuning，本次只隔离 schedule 影响。

---

# 16. Config 修改

当前：

```yaml
model:
  uncertainty:
    enable_stage: stage3
```

改为：

```yaml
enable_stage: stage3c
```

当前：

```yaml
training:
  stage3:
    cardiac_enabled: true
    uncertainty_enabled: true
```

改成清晰的：

```yaml
training:
  stage3a:
    cardiac_warmup: true
    uncertainty_enabled: false

  stage3b:
    joint_cardiorespiratory: true
    uncertainty_enabled: false

  stage3c:
    uncertainty_enabled: true
```

同步修改：

```text
src/cardioresp4d/training/source_first_config.py
tests/test_source_first_config.py
```

formal validation 必须拒绝：

```text
model.uncertainty.enable_stage != stage3c
```

---

# 17. CLI 修改

当前：

```text
scripts/train_source_first.py
```

新增：

```text
--stage3a-steps
--stage3b-steps
--stage3c-steps
```

默认都为 0。

保留旧：

```text
--stage3-steps
```

仅用于给出明确的 deprecated error，**不要 silently map 到 stage3c**，因为那会绕过 3a/3b。

例如：

```text
--stage3-steps is deprecated after v3_change4;
use --stage3a-steps / --stage3b-steps / --stage3c-steps
```

---

# 18. Resume compatibility

这是硬要求。

我们已经有真实：

```text
Stage2c checkpoint
```

来自旧 v3：

```text
current_stage = stage2c
global_step   = 400
```

change4 后必须能：

```text
old Stage2c checkpoint
 -> resume
 -> Stage3a
```

Stage1–Stage2c optimizer group order/semantics不能破坏。

对于旧：

```text
current_stage = stage3
```

checkpoint：

**不要求继续训练兼容。**

如果用户尝试 resume legacy old-Stage3 checkpoint，请 fail-fast：

```text
legacy v3 Stage3 checkpoint is not compatible with v3_change4 schedule;
resume from the Stage2c checkpoint
```

不要 silently 解释为 Stage3a/3b/3c。

---

# 19. 必须正式合并此前 server 上已验证的 3 个 hotfix

GitHub 当前 formal branch 仍缺少此前真实 GPU 暴露出的 hotfix。
本次 change4 必须一并正式化。

## 19.1 CUDA device bug：SINRFFDBasis

当前：

```python
self.register_buffer(
    "grid_spacing_mm",
    (self.upper_world_mm - self.lower_world_mm)
    / torch.tensor([...], dtype=torch.float32)
)
```

如果 world bounds 在 CUDA，而 denominator 在 CPU，会报：

```text
Expected all tensors to be on the same device
```

修复为 device-safe construction，例如：

```python
denominator = self.lower_world_mm.new_tensor(
    [size - 1 for size in self.dense_evaluation_shape]
)
```

然后：

```python
grid_spacing = (
    self.upper_world_mm - self.lower_world_mm
) / denominator
```

不要改 upstream SINR。

增加：
- CPU test；
- CUDA available 时的 CUDA regression test（`pytest.skip` if no CUDA）。

## 19.2 Preflight model device

`preflight_source_first.py` 构造正式模型后应明确：

```python
model = build_source_first_model(...).to(device)
```

避免 upstream SIREN/FFD 参数仍留 CPU。

加 test 或 smoke coverage。

## 19.3 Resume RNG bug

曾出现：

```text
TypeError: RNG state must be a torch.ByteTensor
```

原因：

```python
torch.load(checkpoint, map_location=device)
```

把 CPU RNG ByteTensor 一并搬到 CUDA。

正式修复：

```python
torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False,
)
```

随后：

```text
model.load_state_dict(...)
trainer.load_training_state_dict(...)
restore_rng_state(...)
```

model/optimizer 自己恢复到目标 device。

**不要把主修复写成 `restore_rng_state(... .cpu())` bandaid。**

## 19.4 Frequency dataclass JSON serialization bug

之前：

```python
"frequency_prior": prior.__dict__
```

只做 shallow conversion，nested：

```text
LocationFrequencyPrior
```

不能 JSON serialize。

正式修复：

```python
from dataclasses import asdict

prior_payload = asdict(prior)
```

report + checkpoint metadata 都使用这个 payload。

增加：

```python
json.dumps(asdict(prior))
```

regression test。

---

# 20. `train_source_first.py` checkpoint/report 行为

必须继续保存：

```text
source_first_last.pt
source_first_training_report.json
effective_config.json
training_metrics.csv
```

checkpoint 必须包含：

```text
model
training_state
rng_state
report
frequency_prior
normalization_parameters
```

report 必须记录：

```text
resumed_from
source_lock_verified
device
torch_version
effective_config
frequency_prior
```

`frequency_prior.locations` 中必须能看到每个 location 的：

```text
respiratory_bands_hz
cardiac_bands_hz
respiratory_source
cardiac_source
cardiac_baseline_pairs
```

---

# 21. 建议新增一个只读 diagnostic CLI

为了不再依赖手工临时代码，请新增一个轻量、只读的：

```text
scripts/diagnose_change4_checkpoint.py
```

不要让它影响 training code。

至少支持：

```text
--source-config
--frequency-bands
--manifest
--qc-table
--canonical-domain
--checkpoint
--device
--output-json
```

对 checkpoint 输出：

### A. prior audit

```text
number of locations
local respiratory count
resp global fallback count
local cardiac count
card global fallback count
```

### B. motion statistics

如果 stage 支持：
```text
resp_dvf_rms_mm
resp_dvf_max_mm
card_dvf_rms_mm
card_dvf_max_mm
```

### C. cardiac ablation

同一 checkpoint、同一 sampled pixels：

```text
resp-only MSE
resp+card MSE
relative MSE improvement %
mean absolute prediction change
```

注意：
- “resp-only” 应使用相同模型参数，仅关闭 cardiac branch；
- 不允许加载另一个 checkpoint 造成 confound。

### D. frequency semantics

至少输出：

```text
resp_target_amp
resp_wrong_amp
card_target_amp
card_wrong_amp
resp_peak_hz
card_peak_hz
```

必须使用 **per-location resolved prior**。

### E. Stage3c uncertainty（仅当 uncertainty enabled/可诊断）

```text
frame_variance min/median/mean/max
pixel_scale min/median/mean/p95/max
variance min/median/mean/p95/max
nonzero frame log-var count
```

如果调用 NeSVoR latent z，而 adapter eval mode 不暴露 latent，请只把：

```python
model.canonical.inr.train()
```

用于 latent-output API，同时保持：

```python
torch.no_grad()
```

不要把整个 model 开成 train 导致不必要行为。

该 diagnostic 只需稳定、可审计，不需要做漂亮图。

---

# 22. Tests：必须先写失败测试再实现

至少新增/扩展以下 regression tests。

## 22.1 per-location respiratory prior

构造两个 location：

```text
SAX/s001:
resp f=0.2339, df=0.11696
card f=1.20, ...

SAX/s002:
resp f=0.3509, df=0.11696
card f=1.40, ...
```

assert：

```text
for_location(s001).resp != for_location(s002).resp
```

并验证：

```text
local resp = f ± df/2
```

同时测试：

- resp reliable + card unreliable；
- resp unreliable + card reliable；
- both unreliable -> independent global fallback；
- duplicate slice_key -> error；
- reliable candidate missing frequency/df -> error；
- lower band touching DC -> error。

## 22.2 Eq.9 trainer call-path

不要只单测 `for_location()`。

通过 monkeypatch/spying 或最小 synthetic trainer，证明：

```python
dreme_respiratory_leakage_in_card(...)
```

接收到的是：

```python
prior.respiratory_bands_hz
```

而不是：

```python
self.frequency_prior.respiratory_bands_hz
```

两个 location 必须产生不同传入 band。

## 22.3 Stage contract

必须检查：

```text
stage1:   resp=0, card off, uncertainty off
stage2a:  resp=1, card off, uncertainty off
stage2b:  resp=2, card off, uncertainty off
stage2c:  resp=3, card off, uncertainty off

stage3a:  resp=3, card on, uncertainty off, film card-head-only
stage3b:  resp=3, card on, uncertainty off, film full
stage3c:  resp=3, card on, uncertainty on, film full
```

## 22.4 Stage3a gradient ownership

Stage3a backward 后：

必须 `grad != None`：

```text
film_encoder.card
cardiac_mbc
```

必须无 gradient：

```text
canonical
respiratory_mbc
film_encoder.image
film_encoder.film
film_encoder.geometry_mlp
film_encoder.head
film_encoder.resp
uncertainty
```

不能只看 `requires_grad`，需要真实 backward regression test。

## 22.5 Stage3b gradient ownership

必须：

```text
canonical=True
full film=True
respiratory_mbc=True
cardiac_mbc=True
uncertainty=False
```

## 22.6 Stage3c gradient ownership

必须：

```text
canonical=True
full film=True
respiratory_mbc=True
cardiac_mbc=True
uncertainty=True
```

## 22.7 Data-term semantics

明确测试：

```text
Stage3a -> MSE
Stage3b -> MSE
Stage3c -> Gaussian NLL
```

不要靠 loss 数值猜，最好 spy/controlled variance。

## 22.8 Resume Stage2c -> Stage3a

synthetic checkpoint：

```text
current_stage=stage2c
```

save -> load -> Stage3a 1 step。

assert：

```text
global_step continues
optimizer state loads
no parameter-group mismatch
no RNG restore error
```

## 22.9 Legacy Stage3 resume rejection

构造：

```text
current_stage="stage3"
```

必须给出明确错误，而不是 silent remap。

## 22.10 JSON serialization

完整 `TrainingFrequencyPrior` with nested locations：

```python
json.dumps(asdict(prior))
```

PASS。

## 22.11 CUDA construction

CUDA available：

```text
build model on cuda
SINR grid_spacing_mm cuda
SIREN params cuda after .to(device)
one no-grad stage forward finite
```

没有 CUDA：

```text
explicit skip
```

不要伪造 GPU PASS。

---

# 23. 必须保持的现有行为

change4 后以下 contract 不能回归：

```text
hard invalid only:
- slice_local_scale_absolute
- manual_exclusion

invalid observations never enter sampler

canonical domain = full acquisition-supported FOV

cardiac box != reconstruction crop

view-balanced sampling

fixed-location temporal batch

true AcquisitionTime timestamps

NeSVoR official INR

NeSVoR official PSF

SINR upstream BSplineSiren

SINR upstream CubicBSplineFFDTransform

DREME sequential cardiac -> respiratory pullback

respiratory 8/12/16 progressive levels

cardiac local 16^3

source lock validation
```

---

# 24. Source provenance

不要声称 S2V-DREME 有完整官方 pipeline repo。

本项目可以继续声明：

```text
NeSVoR:
daviddmc/NeSVoR
commit 2e96a91bdd30174210caea911e03a2778c65adbe

SINR:
vasl12/SINR
commit 1a524ca7ae453b55310595fe957245088a108233

FiLM:
ethanjperez/film
commit fe43ddf8a22b339dcca2efa07091ce9d498955cf
```

不要修改 vendored source lock。

---

# 25. 文档同步

修改完成后同步：

```text
README.md
CR_DREME_v3_FULL_PIPELINE_2026-09-09_SOURCE_FIRST_NESVOR_SINR.md
IMPLEMENTATION_REPORT.md
changelog.md
```

新建：

```text
docs/audits/v3_change4_post_fix_audit.md
```

audit 中必须写：

1. 修了哪些已确认 bug；
2. 哪些只是 schedule change；
3. 哪些明确没有修改：
   - NUDFT normalization；
   - Eq8/Eq9 weight；
   - target-band positive loss；
   - uncertainty architecture；
4. targeted tests；
5. full tests；
6. GPU 是否真实可用；
7. 是否跑 real-data training（本次应该是 NO）；
8. 后续 server validation protocol。

---

# 26. README / server validation command 要更新

新的 short validation 顺序：

```text
Stage2c existing checkpoint
 -> Stage3a 100
 -> inspect
 -> Stage3b 100
 -> inspect
 -> Stage3c 暂时不要自动运行
```

先只跑 3a/3b。

示例 server path：

```text
CODE=/data/dengyz/code/CR_DREME_v3
PHASE1=/data/dengyz/dataset/CR_DREME_v3/v1/phase1

PREV_2C=/data/dengyz/dataset/CR_DREME_v3/v1/train_stage2c_100/source_first_last.pt

TRAIN_3A=/data/dengyz/dataset/CR_DREME_v3/v1_change4/train_stage3a_100
TRAIN_3B=/data/dengyz/dataset/CR_DREME_v3/v1_change4/train_stage3b_100
```

Stage3a：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_source_first.py   --source-config "${CODE}/configs/source_first.yaml"   --frequency-bands "${PHASE1}/frequency/frequency_bands.json"   --manifest "${PHASE1}/dicom_manifest.csv"   --qc-table "${PHASE1}/acquisition_qc/acquisition_qc.csv"   --canonical-domain "${PHASE1}/canonical_domain/canonical_domain.json"   --output-dir "${TRAIN_3A}"   --device cuda   --pixel-samples 256   --seed 0   --resume "${PREV_2C}"   --stage1-steps 0   --stage2a-steps 0   --stage2b-steps 0   --stage2c-steps 0   --stage3a-steps 100   --stage3b-steps 0   --stage3c-steps 0
```

Stage3b：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_source_first.py   --source-config "${CODE}/configs/source_first.yaml"   --frequency-bands "${PHASE1}/frequency/frequency_bands.json"   --manifest "${PHASE1}/dicom_manifest.csv"   --qc-table "${PHASE1}/acquisition_qc/acquisition_qc.csv"   --canonical-domain "${PHASE1}/canonical_domain/canonical_domain.json"   --output-dir "${TRAIN_3B}"   --device cuda   --pixel-samples 256   --seed 0   --resume "${TRAIN_3A}/source_first_last.pt"   --stage1-steps 0   --stage2a-steps 0   --stage2b-steps 0   --stage2c-steps 0   --stage3a-steps 0   --stage3b-steps 100   --stage3c-steps 0
```

**不要自动运行 Stage3c。**

---

# 27. Server short-run 的下一轮 acceptance metrics

本次 Codex 本地不需要跑 server training，但 diagnostic CLI/README 要准备好输出以下指标：

```text
Stage3a/3b:
loss first/last/min/max

resp_dvf_rms_mm
resp_dvf_max_mm

card_dvf_rms_mm
card_dvf_max_mm

resp_score_mean/std
card_score_mean/std

cardiac ablation:
resp-only MSE
resp+card MSE
relative improvement %

frequency semantics:
resp_target_amp
resp_wrong_amp
card_target_amp
card_wrong_amp
resp_peak_hz
card_peak_hz
```

与旧 baseline 比较：

```text
old card DVF RMS        ≈ 0.0329 mm
old card max            ≈ 0.146 mm
old cardiac MSE gain    = 0.337%

old resp peak           ≈ 0.529 Hz
old card peak           ≈ 0.529 Hz

old resp target amp     ≈ 5.22e-6
old resp wrong amp      ≈ 3.51e-6

old card target amp     ≈ 3.11e-6
old card wrong amp      ≈ 4.32e-6
```

不要在代码中硬编码“必须达到某个医学阈值”。

只要求：
- finite；
- 无 motion explosion；
- gradient ownership 正确；
- change4 后比较这些指标是否改善。

如果 Stage3a/3b 后仍然：

```text
card dominant peak ≈ respiratory peak
card target <= card wrong
cardiac ablation gain ~0
```

则停止，不要继续 Stage3c；下一步再考虑 target-band concentration / temporal encoder，作为 `v3_change5`，不属于本次 scope。

---

# 28. 测试执行顺序

不要一开始就反复跑 full suite。

推荐：

```bash
python -m pytest -q   tests/test_frequency_training_prior.py   tests/test_v3_change3_frequency_contract.py   tests/test_v3_change3_stage_contract.py   tests/test_v3_change3_runtime_state.py   tests/test_source_first_config.py
```

然后新增 change4 tests，例如：

```text
tests/test_v3_change4_frequency_prior.py
tests/test_v3_change4_stage_schedule.py
tests/test_v3_change4_resume.py
tests/test_v3_change4_cuda.py
```

targeted PASS 后：

```bash
python -m pytest -q
```

最后：

```bash
pip check
```

并运行 source lock verification。

如果本机有 CUDA，可以做 1 次最小 CUDA no-training smoke；
如果没有 CUDA，明确记录 skip，不要假装验证。

---

# 29. 不要重复 Phase1 / 不要长跑数据

本地如果存在：

```text
/home/universe/SVR/data/DYL0709/
```

最多做：
- 读取已有 manifest/QC/frequency；
- no-grad/no-training preflight；
- diagnostic dry-run。

不要重新跑：
- DICOM inspection；
- acquisition QC；
- geometry；
- PCA；
- ROI；
- canonical domain；
- real training。

除非现有文件不存在，否则不要浪费时间和 token。

---

# 30. Git 与提交要求

完成后：

```bash
git status
git diff --check
```

确认无意外大文件、无 result/checkpoint、无 DICOM 被加入。

建议逻辑 commit：

```text
fix: restore source-first runtime portability
fix: use per-location respiratory frequency priors
feat: split cardiac training into stage3a stage3b stage3c
test: add change4 regression coverage
docs: document v3 change4 validation protocol
```

如果不想拆 5 个 commit，至少保证 commit 历史清晰。

若有权限：

```bash
git push origin dev/cardioresp4d
```

无权限则保留本地 commit，并报告 SHA。

---

# 31. 最终必须给我的报告

完成后不要只说“done”。

请输出：

```text
1. 起始 commit SHA
2. 最终 commit SHA
3. modified files
4. new files
5. 每个核心 bug 的 root cause
6. 每个修复点
7. Stage3a/3b/3c 的最终 trainable ownership 表
8. per-location respiratory prior fallback 规则
9. Eq9 是否已确认使用 local prior
10. resume Stage2c -> Stage3a 是否测试 PASS
11. server hotfix 是否全部正式纳入
12. targeted tests 结果
13. full pytest 结果
14. pip check
15. source lock verification
16. CUDA test 是否真实运行
17. 是否运行 real-data training（应为 NO）
18. 下一步 server validation 命令
```

---

# 32. 最重要的验收原则

本次 change4 的成功定义不是“loss 能下降”。

必须同时满足：

```text
A. per-location respiratory prior 真正进入 trainer
B. Eq9 使用 resolved location prior
C. Stage3a 只训练 card head + cardiac MBC
D. Stage3b joint，但 uncertainty 仍冻结
E. Stage3c 才打开 uncertainty
F. old Stage2c checkpoint 可 resume
G. CUDA/RNG/JSON hotfix 正式进入 repo
H. tests 对上述语义有真实 regression coverage
I. 不修改 NUDFT normalization
J. 不新增 target-band positive loss
K. 不跑长 real-data training
```

如果任何一项没满足，不要宣称 change4 完成。

开始执行。
