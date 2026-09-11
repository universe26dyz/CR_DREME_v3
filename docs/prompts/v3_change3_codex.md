# v3_change3_codex — CR_DREME_v3 全仓库闭环审计与 GPU Gate 封口修正

> **工作目录**：`/home/universe/SVR/code/CR_DREME_v3`
>
> **GitHub**：`universe26dyz/CR_DREME_v3`
>
> **当前分支**：`dev/cardioresp4d`
>
> **本轮基线 commit**：
>
> ```text
> 2e8975e6d73862feda57bcaf6eeefc8d464ce8db
> ```
>
> **主规格**：
>
> ```text
> CR_DREME_v3_FULL_PIPELINE_2026-09-09_SOURCE_FIRST_NESVOR_SINR.md
> ```
>
> **前两轮 prompt**：`v3_change1_codex.md`、`v3_change2_codex.md`
>
> 本轮不是继续做“小修补”。目标是对当前 `CR_DREME_v3` 做一次 **closure audit（闭环审计）**：先完整审计 Phase1 → Stage1 → Stage2a → Stage2b → Stage2c → Stage3 的所有 runtime/data/loss/config/test 路径；审计表完整之前不开始零散改代码；将所有 Critical / Important 问题一次性修完；修改后重新从头做第二轮独立复审；只有第二轮复审不再发现新的 Critical / Important 问题、核心数学 contract 与 full regression 均通过，才允许宣告 GPU gate。
>
> 本轮默认仍 **不执行长 GPU training**，只允许真实数据 preflight / dry-run，以及最终给出最短 Stage1→Stage2a GPU 验证命令。

---

# 0. Source-first 与工程边界不变

以下 upstream primitive 已固定，不允许重新实现：

- NeSVoR INR / HashGrid；
- NeSVoR `resolution2sigma`；
- NeSVoR uncertainty network builder；
- NeSVoR image regularization；
- SINR `BSplineSiren`；
- SINR `CubicBSplineFFDTransform`；
- FiLM primitive；
- PyTorch `GaussianNLLLoss`。

`third_party/` 仍是普通 vendored tree。

以下项目 contract 不得回退：

```text
canonical domain = full acquisition-supported FOV
cardiac box      = local motion/sampling region, NOT crop
hard invalid     = slice_local_scale_absolute / manual_exclusion
oblique PSF      = DICOM row/column/normal basis
motion convention = observation → reference pullback
```

任何必要 adaptation 都要先查原论文 / upstream code，并在代码附近加中文注释，写明来源 URL / pinned commit，同时区分：direct upstream reuse、paper-derived necessary adaptation、project-specific engineering adaptation、optional ablation。

---

# 1. 本轮工作方式：先审计，再统一修改

## 1.1 第一步禁止直接修代码

先生成：

```text
docs/audits/v3_change3_pre_fix_audit.md
```

逐模块检查并列出：

```text
ID
Severity: Critical / Important / Minor
Module
Observed behavior
Expected behavior
Evidence
Affected files
Root cause
Proposed fix
Required tests
Status
```

只有这个 audit 表完整后才开始修改。

## 1.2 审计必须覆盖整个 mainline call graph

至少追踪：

```text
manifest
→ QC
→ dataset
→ normalization
→ geometry
→ canonical domain
→ PCA/frequency prior
→ sampler
→ FiLM score encoder
→ SINR respiratory/card MBC
→ score × MBC
→ sequential pullback
→ PSF
→ canonical INR
→ uncertainty
→ data fidelity
→ Eq.6 / Eq.7 / Eq.8 / Eq.9
→ smoothness
→ optimizer / stage schedule
→ checkpoint
→ report
```

对每个节点回答：输入是什么、输出 shape/units 是什么、谁调用它、config 是否真正控制、inactive stage 是否真的不计算/不更新、loss 是否真的被 total loss 使用、test 是否验证数学语义而不是只验证 shape、report 是否记录 runtime effective value。

---

# 2. 必须专项搜索五类隐性漏洞

## 2.1 Config 写了但 runtime 没用

搜索 `source_first.yaml`、`yaml.safe_load`、`config.get`、hard-coded literals、constructor defaults。检查所有 formal config：canonical INR params、PSF samples、SINR logical grid、cps、hidden_dim、cardiac taper、motion encoder params、normalization、sampling、loss weights、smoothness grid、optimizer LR、temporal auxiliary、frequency prior、uncertainty、stage schedule、seed。

必须证明：

```text
requested config == runtime effective config
```

不能只把 YAML 原样抄到 `effective_config.json`。

## 2.2 函数存在但 trainer/runtime 没调用

搜索所有 loss functions、regularizers、validators、source-lock、frequency parser、config validators，检查是否真的进入 mainline、是否只在 test 中被调用、是否存在旧函数继续被 trainer 调用。

## 2.3 Test 通过但数学轴/单位语义错

专项检查：

```text
mean(dim=...)
sum(dim=...)
reshape
permute
einsum
broadcast
grid_sample coordinate order
row/column spacing mapping
batch/level/spatial/xyz axes
PSF sample axis
uncertainty aggregation axis
```

不能使用全 0 / 全 1 数据作为唯一数学测试，必须构造不同 axis 数值以检测错误 reduction。

## 2.4 Inactive module 仍然计算或进入 loss

Stage1/2a/2b/2c/3 分别检查 forward compute、gradient、optimizer param group、regularization、report。不仅要求“不更新”，还要求不必要的 expensive branch 不执行。

## 2.5 Report 声称完成但 runtime 不一致

逐项核对 README、`IMPLEMENTATION_REPORT.md`、`SOURCE_PROVENANCE.md`、`changelog.md`、`effective_config.json`、`training_report.json`。禁止出现“配置写 16³但 runtime 仍 4³”“报告说 Eq.8 但代码不是 Eq.8”“报告说 full tests pass 但实际有 failures”等情况。

---

# 3. 已知 Critical：修正 DREME Eq.6 `L_MBC`

DREME Eq.6 的语义是：

\[
L_{MBC}=\frac{1}{3}\sum_k\sum_i\left(\|e_{i,k}\|_2^2-1\right)^2
\]

即对每个 level `i`、每个 Cartesian component `k`，沿 spatial domain 计算 MBC norm。输出 norm tensor 应保留 `[level, xyz]` 后再 aggregate，不能误平均 batch+level 而保留 spatial axis。

## 3.1 Eq.6 必须使用 ungated MBC basis

progressive gate 不能进入 Eq.6 的 MBC norm。adapter 应明确区分 raw/ungated MBC basis、scheduled/active MBC basis、weighted time-varying DVF。

## 3.2 只 normalize 当前 active MBC

```text
Stage2a: resp level1
Stage2b: resp level1+2
Stage2c: resp level1+2+3
Stage3 : resp 1+2+3 + cardiac
```

未激活 cardiac 不应给 Eq.6 添加常数项。

## 3.3 重新确认 Eq.6 的离散 norm

重新读 DREME-MR Eq.6、supplementary、以及可找到的作者代码。若能实现 control-point analytic norm，优先；若 hybrid SINR 必须用 dense discrete approximation，则明确标为 necessary adaptation，并做到 spatial-resolution invariant，明确 mean-square / integral / voxel-volume normalized 定义。

## 3.4 Eq.6 tests

至少构造：

```text
level1 RMS=1
level2 RMS=2
level3 RMS=0.5
level4 RMS=1
```

人工算 expected；并验证 zero MBC 不是 minimum、unit norm 是 minimum、gate 改变不影响 raw Eq.6、inactive level 不进入当前 stage Eq.6。

---

# 4. Progressive gate 不得成为新的可学习 scale ambiguity

当前若 `level_gates = nn.Parameter`，会在 `gate_i * e_i(x) * w_i(t)` 中引入第三个可学习尺度，破坏 Eq.6/7 原本只约束 `e_i(x) × w_i(t)` 的 decomposition。

Formal mainline 不允许 trainable free scalar gate。推荐使用 non-trainable schedule buffer：`0 → deterministic ramp → 1`。如需新 level near-zero，用非学习 ramp 或 adapter-level output scaling，不修改 upstream SINR，不制造第三个永久可学习 amplitude。

测试必须验证 inactive level：no forward expensive FFD、no gradient、no optimizer state、no Eq.6 contribution。

---

# 5. DREME Eq.8 必须是 complex coefficient subtraction

DREME 原文：

\[
L_c=\frac1{N_c}\sum_{\omega\in\nu_c,\omega'\in\nu_b}\sum_{i,k}
\left|F[w^r_{i,k}](\omega)-F[w^r_{i,k}](\omega')\right|^2
\]

必须先 complex Fourier coefficient subtraction，再 abs²。不能实现成 `(|Fc|-|Fb|)^2`，也不能先分别求 magnitude mean 再相减。

---

# 6. Eq.8 baseline 不得丢失 pairing

当前如果只保存独立 `cardiac_bands_hz` 和 `baseline_bands_hz`，可能丢失哪个 baseline 对应哪个 cardiac bin。formal prior 建议显式保存 `cardiac_baseline_pairs`，至少包含 cardiac band、baseline band、source slice/location、resolution_hz。

baseline 构造规则必须先查 DREME 正文、supplementary、作者仓库；若没有公开到足够具体，则“相邻等宽、避开 DC/resp/card”等规则只能标为 paper-derived necessary adaptation，不能称官方 DREME baseline rule。

---

# 7. Eq.8 / Eq.9 frequency evaluation 不能只取 band center

Formal frequency loss 要支持：一个 band 内多个 evaluation frequencies、多个 separated bands。frequency spacing 根据当前 temporal sequence 的 duration、effective frequency resolution、timestamp distribution 决定，并把 `resolved_evaluation_frequencies_hz` 写入 report。

---

# 8. Nonuniform Fourier 实现需完整数值审计

必须检查：

1. **timestamp offset**：使用 `t_rel = timestamp - timestamp[0]` 或等价稳定化，避免当天秒数导致 float32 phase precision 问题；测试 `t` 与 `t+40000s` 的稳定性。
2. **sample-count scaling**：确认是否除以 `N`，避免 20-frame/50-frame loss 尺度自动不同；若 normalize，写入 provenance。
3. **irregular sampling weighting**：核对 equal-weight NUDFT vs time-interval weighting，结合本项目真实 timestamp jitter 选择并说明。
4. **DC / mean handling**：irregular timestamps 下 constant offset 可能泄漏到非零频率，检查是否需要 temporal centering 或明确依赖 Eq.7；加 constant-offset synthetic test。

---

# 9. Frequency prior 必须考虑 per-slice / per-location 心率差异

Phase-1 保留 `per_slice_candidates` 与 `union_resolution_bins_hz`，因为各 fixed slice 顺序采集，心率可能随时间变化。不能把整个 cardiac union 无条件作为每个 location 的同一 forbidden set。

优先建立：

```text
global subject prior
+
per-view / per-slice location prior
```

例如 `TrainingFrequencyPrior.for_location(view, slice_id)`。优先 reliable per-slice cardiac candidate；若该 slice 不可靠，再使用 validated nearby/global recurrent bins fallback。不要 collapse 成一个覆盖很宽的 min-max。

`temporal_batch()` 返回哪个 `view+slice_id`，frequency loss 就使用该 location 对应 prior。

---

# 10. `allow_template_fallback` 必须真实工作或删除

若 `allow_template_fallback=True` 只是“不报错”但返回空 band，这是假功能。推荐 formal mainline 保持 `false`，无 verified respiratory prior 就 fail-fast。若保留 fallback，config 必须真正提供 fallback band，parser 真正填入并在 report 标明 source=fallback。

---

# 11. Stage1 必须真正跳过 motion/FiLM

检查 `SourceFirstDynamicModel.predict()`。Formal Stage1 应：

```text
不调用 FiLM
不调用 respiratory SINR
不调用 cardiac SINR
motion=None
只做 PSF → canonical INR → slice prediction
```

不能“先全部算完再把 score ×0”。

---

# 12. Stage2a/b/c 不得计算 inactive expensive branch

```text
Stage2a: 只算 respiratory level1
Stage2b: 只算 level1+2
Stage2c: 只算 level1+2+3
Stage2 : cardiac 不计算
Stage3 : resp+card
```

不仅是 score=0，而是真正不执行 inactive dense FFD。增加 call-count / monkeypatch tests。

---

# 13. Trainer 不得在 Stage1 计算没用的 motion regularizer

按 stage 拆分 observation components：

```text
Stage1: data + image only
Stage2: active respiratory regularizers
Stage3: resp + card + uncertainty regularizers
```

---

# 14. Config → Runtime 必须彻底闭环

新增正式 model factory，例如：

```text
src/cardioresp4d/training/build_model.py
```

唯一正式入口从 config/domain/n_dynamic_frames 构建 model。所有正式参数都从 config 传入。

## 14.1 canonical INR

必须真正传：coarsest_resolution、finest_resolution、level_scale、features_per_level→`n_features_per_level`、log2_hashmap_size、latent_dim、width、depth、spatial_scaling（若正式使用）。

## 14.2 PSF

`n_samples` 必须 runtime 生效。Scientific run 与 smoke run 分开配置，不能把 8 说成 NeSVoR official default。

## 14.3 SINR

传入 logical_control_shapes、cps、hidden_dim、cardiac logical_control_shape、cardiac taper_mm，并把 actual dense_evaluation_shape、padded_control_shape、grid_spacing_mm 写入 effective config/report。

## 14.4 FiLM geometry encoder

将 channels、position_bands、geometry feature dims 放 config。

## 14.5 uncertainty

显式 frame_embedding_dim、width、depth、enable_stage。

---

# 15. `effective_config.json` 必须来自实际对象

构建 model/trainer 后，从实际对象反向生成 effective config，至少包括：actual class names、actual third_party paths、upstream hash verification、canonical INR args、PSF samples、SINR grid/cps/dense shape、cardiac taper、FiLM dims、uncertainty dims、normalization groups、sampling fractions、loss weights、smoothness grids、optimizer groups/LR、temporal mode、resolved frequency prior、stage schedule、seed、device、torch version。

必须做 config perturbation test：例如 psf 8→7、latent_dim 16→5、SINR hidden_dim→11，构建后 object 真正变化。

---

# 16. Geometry 做 end-to-end consistency audit

当前 acquisition scalar 若仍是 `pixel_spacing / world extent[:2]`，对 oblique view 不严格。建议用 canonical AABB 沿 row、column、normal 的 projected extent，再做 row_spacing/extent_row、col_spacing/extent_col、thickness/extent_normal。

同时必须验证 `DicomPlane.pixel_to_world(u,v)` 与 `SourceFirstDynamicModel._pixel_world(...)` 在 axis-aligned、oblique、anisotropic PixelSpacing 三种情况下逐点一致，特别检查 DICOM `PixelSpacing=[row_spacing,column_spacing]`、IOP first/second direction 与 u/v index 的对应关系。

PSF resolution 与 `_pixel_world` 必须使用同一 basis/spacing 顺序。

---

# 17. Smoothness 必须真正使用 configured grid，并 resp/card 分开

当前 config 若写 16³，runtime 不能继续硬编码 4³。resp smoothness 在 full canonical FOV，card smoothness 在 local cardiac box，分别计算 `smooth_resp`、`smooth_card`，不能 `smooth_card=smooth_resp*0`。

梯度按 physical spacing（mm）计算。测试 constant→0、linear→predictable finite、high-frequency→larger、同一 physical field 在 8³/16³ 下 loss 近似稳定。

---

# 18. Stage-specific optimizer / LR 真正实现

新增 config，例如：

```yaml
optimizer:
  stage1:
    canonical_lr: ...
  stage2:
    canonical_lr: ...
    film_lr: ...
    respiratory_mbc_lr: ...
  stage3:
    canonical_lr: ...
    film_lr: ...
    respiratory_mbc_lr: ...
    cardiac_mbc_lr: ...
    uncertainty_lr: ...
```

具体数值先查 S2V-DREME / DREME；若使用 project default，明确标注，不宣称 official default。

---

# 19. Optimizer state 不要在 stage transition 无意识丢失

当前若每 stage recreate Adam，会清空 canonical INR 的 moment estimates。优先使用稳定 param groups + stage 更新 lr/requires_grad，保留已训练参数 Adam state；新激活参数首次有 gradient 时创建 state。若必须 rebuild optimizer，显式迁移已有 state 并写 test。

---

# 20. Freeze/unfreeze 升级为完整 ownership test

每 stage 检查：forward called、requires_grad、grad nonzero、optimizer contains、parameter value changed、optimizer state created。

目标：

```text
Stage1: canonical only
Stage2a/b/c: canonical + FiLM respiratory pathway + active resp SINR
Stage3: canonical + FiLM + resp + card + uncertainty
```

Stage2 card head 也建议 frozen，Stage3 再解冻。

---

# 21. Temporal auxiliary 覆盖完整时间跨度并优先 batch forward

同一 fixed location 的 T 帧 stack 成 batch，一次 FiLM forward，减少 Python loop。保持真实 timestamp 排序、full-span、qc-valid only。

每个 sequence 记录 n_frames、duration_s、median/min/max dt、frequency resolution estimate、supported range。若目标 band 无法解析，不静默返回 0：report skipped reason；formal run 若大量 location 不可用则启动前 fail-fast。

---

# 22. Hard-QC policy 做 runtime assertion

训练前验证 QC table：任何 `qc_valid=false` 必须包含 hard-invalid token：`slice_local_scale_absolute` 或 `manual_exclusion`。如果 low_ncc、global_intensity_scale、scale_corrected_residual、slice_local_structure、slice_local_scale_robust 被标 invalid，则 fail-fast。若 `qc_reason` 可多原因拼接，统一 reason parser，不做 exact-string-only 判断。

---

# 23. Dynamic frame ID / manifest identity 完整校验

检查 source_file_token、view、slice_id、frame_index、timestamp、dynamic_frame_id。要求 token unique；frame ID unique且在 embedding range；invalid frame 不参与 likelihood/temporal loss；timestamp finite；同 fixed location 时间顺序合理；跨午夜若可能出现要 unwrap 或明确 reject。

---

# 24. Normalization 做真实数据 preflight

保持 `per_series` 作为 formal candidate，但训练前输出 normalization group count、group key、view、slice IDs、frame count、1/99 percentile、valid frame count。确认真实 DICOM 的 SeriesInstanceUID 到底代表每 fixed slice 还是整个 stack；若 per_series 实际退化成 per-frame，fail。

避免 `np.concatenate(all full-resolution frames)` 造成不必要 RAM 峰值，可用 deterministic subsampling / histogram percentile；小 fixture 验证与 exact percentile 误差。

---

# 25. Uncertainty 做 full integration shape/gradient audit

当前已修 `(mean sigma)^2 + frame variance`，但还要确认 latent_samples axis、PSF sample axis、prediction shape、variance shape、frame embedding broadcast。必须 test：`N pixels × S PSF samples → variance [N]`，frame variance exactly once，Stage3 GaussianNLLLoss 梯度同时回 canonical、uncertainty、active motion。

---

# 26. 对照 v3 主规格审计 uncertainty / Stage data fidelity

不要凭旧 v1/v2 记忆猜。直接对照 `CR_DREME_v3_FULL_PIPELINE_2026-09-09_SOURCE_FIRST_NESVOR_SINR.md`：如果 formal spec 定义 uncertainty only Stage3 就保持；若 spec 与 runtime 不一致，记录 conflict，以当前 v3 主规格为准。

同时核对 Stage1/Stage2/Stage3 data fidelity 公式与实际 trainer 一致，NeSVoR image regularization 的 mode/delta 若属于 formal hyperparameter必须进 config/effective report。

---

# 27. Formal loss total contract 做 analytical closure test

统一 formal 名称：

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

Stage1：`data + λ_image image`。

Stage2a/b/c：`data + image + LMBC + smooth_resp + LZMS_resp + Lc`。

Stage3：`NLL + image + LMBC + smooth_resp + smooth_card + LZMS_all + Lc + Lr`。

测试时给各 component 常数，核对 total 精确等于手算结果。

Eq.7 必须只使用当前 active score set；给 inactive channel 加巨大常数，当前 stage L_ZMS 不应变化。

---

# 28. Checkpoint / resume 进入正式审计

正式 checkpoint 至少保存 model、optimizer、current_stage、global_step、stage_step、effective_config、frequency prior、normalization parameters、RNG state（torch/cuda/numpy/python random）。支持 `--resume` 最佳；如果本轮不实现完整 resume，也不能把当前仅 model state 的文件称为完整 checkpoint，要明确命名和局限。

---

# 29. Reproducibility：统一随机种子

新增 runtime seed config，统一设置 random、numpy、torch、torch.cuda、sampler；PSF random sampling也必须受 seed 控制。scientific run 可 stochastic，但 seed 必须记录可复现。

---

# 30. Logging 不要只保存 last-step component

每 stage 至少保存 first/last/mean/min/max。推荐 `training_metrics.csv` 每若干 step 记录 global_step、stage、loss components、learning rates、resp activation/ramp、DVF RMS/max、score mean/std、uncertainty mean。

增加 motion collapse/explosion 监控：resp/card DVF RMS/max mm、MBC norm per level/xyz、score mean/std per channel、activation state；NaN/Inf/absurd DVF/all scores collapse 应 warning/fail。阈值若无论文依据，只作为 QC warning，不做 scientific claim。

---

# 31. Performance closure audit

用 call counters 验证：

```text
Stage1: FiLM=0, resp SINR=0, card SINR=0
Stage2a: resp1>0, resp2/3=0, card=0
Stage2b: resp1/2>0, resp3=0, card=0
Stage2c: resp1/2/3>0, card=0
Stage3: resp/card>0
```

这既是速度检查，也是 stage semantics 检查。

---

# 32. Full repository regression：当前 7 个失败必须处理

当前已知：`105 passed, 1 skipped, 7 failed`，包括 `tests.test_data ×5`、`tests.test_acquisition_qc ×1`、`tests.test_reference ×1`。

逐个判断 test 过时还是 production regression。dataset unit test 不应被 full-pipeline 三视图 validator 无条件污染；底层 loader validation 与 formal 3-view pipeline validation 应解耦。acquisition_qc/reference 要对照当前已 seal 的 hard-invalid policy、canonical hole policy、initial reference 在 v3 的真实 role。禁止为了全绿机械改 expected。

---

# 33. Prompt 归档必须是普通文件

当前 `docs/prompts/v3_change2_codex.md` 若仍为 symlink (`120000`)，本轮修正为普通 `100644` 文件。建议 `docs/prompts/v3_change1_codex.md`、`v3_change2_codex.md`、`v3_change3_codex.md` 都是普通文件。不要删除用户可能仍需的 root 副本，除非只是安全复制且有明确理由。

---

# 34. `changelog.md` 补录手工 commit

append-only 补录实际基线：

```text
2e8975e6d73862feda57bcaf6eeefc8d464ce8db
```

不要修改历史条目。v3_change3 结束后再 append final commit。

---

# 35. Source-lock / dependency clean-checkout

继续核对 `SOURCE_LOCK.json`、`SOURCE_PROVENANCE.md`、actual files、actual import path 一致，并在 training report 记录 `source_lock_verified=true`。

检查 `requirements-source-first.txt` / `pyproject.toml` 是否包含真正需要的 omegaconf、ipdb、termcolor、pydicom、scipy 等，不依赖“本机之前手工 pip install 过”。目标是 clean checkout + documented env 能完成 source-first imports 与 CPU core tests。

---

# 36. Real-data preflight：本轮必须做，但不长训练

新增/运行 `scripts/preflight_source_first.py` 或等价 dry-run。输入 manifest、QC table、canonical_domain.json、Phase1 frequency_bands.json、source_first.yaml，只做：文件/schema validation、view counts、valid/invalid counts、hard-QC contract、timestamp stats、normalization groups、canonical/card box bounds、frequency prior resolution、source lock、model construction、config-runtime equality、one tiny no-grad forward per stage。

输出 `preflight_report.json`。

对每个 usable fixed location 检查 n_valid_frames、duration、frequency prior 是否可解析，输出 usable/skipped/fallback locations。真实 Phase-1 文件若没有 verified respiratory band，formal gate fail，不自动伪造。

---

# 37. 第二轮独立 post-fix audit 是硬性步骤

修完所有 pre-fix audit 后生成：

```text
docs/audits/v3_change3_post_fix_audit.md
```

重新从零检查 call graph，不允许只勾旧 checklist。结论只能是：

```text
Critical: 0
Important: 0
Minor: 可列出
```

若发现新的 Critical/Important，继续修并再次复审，直到为 0。

---

# 38. 必须新增/强化的测试类型

### Mathematical
- Eq.6 axis-aware analytic test；
- Eq.6 ungated active-basis test；
- Eq.7 active-channel test；
- Eq.8 complex subtraction；
- Eq.8 baseline pairing；
- Eq.9 multi-band；
- NUDFT timestamp-offset stability；
- NUDFT sample-count normalization；
- constant-offset/DC behavior。

### Motion topology
- no cross-axis mixing；
- no trainable scale gate；
- active level only；
- cardiac only Stage3；
- sequential pullback order/sign。

### Geometry
- DicomPlane vs `_pixel_world` roundtrip；
- oblique PixelSpacing；
- PSF basis/spacing consistency；
- acquisition scalar projected extent。

### Stage ownership
- call counts；
- gradients；
- actual parameter changes；
- optimizer membership；
- optimizer state preservation。

### Config/runtime
- perturbation test；
- actual effective config；
- PSF/SINR/FiLM/uncertainty values。

### Uncertainty
- PSF sample aggregation shape；
- one frame variance；
- Stage3 gradient integration。

### QC/data
- only hard invalid excluded；
- multi-reason parsing；
- duplicate token reject；
- invalid not in temporal likelihood；
- timestamp validation。

### Smoothness
- constant / linear / high-frequency；
- resolution stability；
- separate resp/card domains。

### Resume/reproducibility
- fixed seed same first step；
- resume reproduces next step（若实现 resume）。

---

# 39. Synthetic end-to-end Stage1→Stage3 smoke 必须升级

构造 tiny synthetic multi-view dataset：SAX、2CH、4CH、multiple fixed locations、multiple timestamps。依次跑 Stage1→2a→2b→2c→3。每 stage 验证 expected modules called、expected params updated、inactive params unchanged、loss finite、output shape correct、no NaN。

---

# 40. Full test gate

先 targeted，然后新增 v3_change3 tests，最后必须运行：

```bash
conda run --no-capture-output -n knesvr_torch \
  python -m unittest discover -s tests
```

目标：`0 unexpected failures`。允许 explicit skip，但必须说明原因。不允许通过排除、rename、blanket skip 制造绿色结果。

---

# 41. GPU gate 最终标准

只有同时满足：

```text
pre-fix audit complete
Critical fixes complete
Important fixes complete
post-fix Critical=0
post-fix Important=0
Eq.6 verified
Eq.7 verified
Eq.8 verified
Eq.9 verified
frequency prior per-location alignment verified
config == runtime verified
smoothness verified
optimizer stage ownership verified
inactive computation eliminated
source lock verified
hard-QC verified
uncertainty integration verified
geometry roundtrip verified
full unittest discover no unexpected failures
real-data preflight pass
```

才允许：

```text
GPU_GATE = PASS
```

否则必须 `GPU_GATE = FAIL` 并停止。

---

# 42. GPU gate 通过后的唯一下一步

只给短验证命令，不执行长 Stage3：

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

如果 CLI/config 在本轮被合理重构，给出最终真实命令。

---

# 43. v3_change3 最终 Codex 回报格式

最终回复必须严格包含：

1. 基线 commit；
2. pre-fix audit 总问题数：Critical/Important/Minor；
3. pre-fix audit 文件；
4. 所有 Critical 列表；
5. 所有 Important 列表；
6. Eq.6 最终实现；
7. progressive gate 最终实现；
8. Eq.8 最终实现；
9. Eq.9 最终实现；
10. frequency prior 是否 per-location；
11. baseline pairing 规则与来源；
12. NUDFT timestamp/normalization/DC 处理；
13. Stage1 是否完全跳过 motion/FiLM；
14. Stage2a/b/c 是否只运行 active levels；
15. config→runtime builder；
16. effective_config 是否来自实际 object；
17. resp/card smoothness grid；
18. optimizer param groups/LR；
19. optimizer state 是否跨 stage 保留；
20. stage ownership tests；
21. DICOM geometry roundtrip；
22. uncertainty integration；
23. hard-QC runtime assertion；
24. normalization real-data stats；
25. checkpoint/resume 状态；
26. reproducibility seed；
27. source lock；
28. dependency clean-checkout；
29. 7 个旧 regression 如何处理；
30. targeted tests；
31. full unittest discover；
32. real-data preflight；
33. post-fix audit Critical/Important/Minor；
34. `GPU_GATE = PASS/FAIL`；
35. 若 PASS，最短 Stage1→Stage2a GPU command；
36. docs/prompts 是否都是普通文件；
37. changelog 新增记录；
38. 最终 git commit SHA；
39. 当前仍存在的 Minor/future optimization。

---

# 44. 最终自查：不得提前结束

在说“完成”之前执行 repo-wide 搜索：

```bash
rg -n \
  "TODO|TBD|pass$|NotImplemented|fallback|4, 4, 4|learning_rate.?=.1e-3|score.square|dvf.*square|level_gates|frequency_bands|effective_config|motion_regularizers|GaussianNLL|cardiac_mbc|respiratory_mbc" \
  src scripts configs tests README.md IMPLEMENTATION_REPORT.md SOURCE_PROVENANCE.md
```

逐项确认：没有 formal TODO、没有旧错误 loss 进入 mainline、没有 4³ hard-coded smoothness、没有 trainable level gate、没有空 fallback、没有 config 只记录不生效、没有 Stage1 无用 motion、没有 Stage2 无用 cardiac、没有 report 与 runtime 不一致。

---

# 45. 本轮最核心原则

不要把“测试通过”等价为“科学定义正确”，也不要把“函数已经写了”等价为“mainline 已经使用”。

真正验收标准是：

\[
\boxed{\text{paper/spec semantics}=\text{runtime call graph}=\text{config}=\text{tests}=\text{report}}
\]

并且：

\[
\boxed{\text{post-fix Critical}=0,\quad \text{post-fix Important}=0}
\]

只有这样，`CR_DREME_v3` 才可以进入真实短 GPU Stage1→Stage2a。
