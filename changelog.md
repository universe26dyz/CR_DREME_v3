## 2026-09-10 16:18:01 +0800 — v3_change1_codex.md — v3 source-first 二轮 debug

### 修改目的

按 `v3_change1_codex.md` 对 CR_DREME_v3 进行第二轮 source-first 审计、规范化修正与 CPU 定向验证；保持全 acquisition-supported canonical domain，不删除 legacy/ablation 代码。

### 主要变更

- 开始审计 vendored NeSVoR、SINR、FiLM 的来源、运行时 import 与数值语义。
- 后续条目将以 append-only 方式记录本轮每一项实际代码和文档变更。

### 涉及文件

- `third_party/`、`src/cardioresp4d/`、`configs/source_first.yaml`、`tests/`、本文件。

### 外部源码依据

- NeSVoR / `2e96a91bdd30174210caea911e03a2778c65adbe`
- SINR / `1a524ca7ae453b55310595fe957245088a108233`
- FiLM / `fe43ddf8a22b339dcca2efa07091ce9d498955cf`

### 验证

- 审计开始；定向 CPU 测试将在实现完成后记录。

### 未解决问题

- 待完成 geometry 编码、oblique PSF、SINR control-grid、时间频率和训练编排的逐项审计与测试。

### Git

- 起始 commit: `576d33b`
- 最终 commit: 待本轮完成后记录。

---

## 2026-09-10 16:38:05 +0800 — v3_change1_codex.md — v3 source-first 二轮 debug（实施完成）

### 修改目的

落实第二轮 source-first contract，并在本地 CPU 完成最小确定性验证。

### 主要变更

- NeSVoR、SINR、FiLM 统一从 `third_party/` vendored source 运行，新增 `SOURCE_LOCK.json`。
- 修正 geometry units、oblique DICOM-basis PSF、SINR logical control-grid、NeSVoR uncertainty aggregation、timestamp frequency auxiliary、80/20 cardiac sampling、stage freeze/unfreeze 和唯一 sequential pullback。

### 涉及文件

- `third_party/`、`src/cardioresp4d/`、`scripts/train_source_first.py`、`configs/`、`tests/`、README/报告文件。

### 外部源码依据

- NeSVoR `2e96a91bdd30174210caea911e03a2778c65adbe`
- SINR `1a524ca7ae453b55310595fe957245088a108233`
- FiLM `fe43ddf8a22b339dcca2efa07091ce9d498955cf`
- NISF++ `1f6f9b3feba7c3c757d1f4111418415cd1cafeaa`；Nerfstudio `50e0e3c70c775e89333256213363badbf074f29d`

### 验证

- `conda run --no-capture-output -n knesvr_torch python -m unittest tests.test_v3_change1_training tests.test_source_backed_adapters tests.test_sinr_adapter tests.test_v3_change1_contracts tests.test_unified_progressive_smoke tests.test_source_first_config`
- 结果：23/23 passed。

### 未解决问题

- 真实数据与长 GPU 训练仍未执行；在 GPU 上仅先跑短 Stage1→Stage2a。

### Git

- 最终 commit: 待创建。

### 阶段验证补充

- v3_change2 target CPU suite：28/28 passed。
- full discover：105 passed、1 skipped、7 failed；详见 `IMPLEMENTATION_REPORT.md`，尚未创建本轮 commit。

---

## 2026-09-10 20:18:54 +0800 — v3_change2_codex.md — motion-training scientific correctness 修正

### 修改目的

以 `d947ad71bc84dbe5441117b6c0e0400c39337909` 为基线，修正 motion topology、DREME Eq.6–9、Phase-1 frequency prior 和 runtime integration。

### 验证

- 审计与 TDD 开始；结果将在本轮完成记录。

### Git

- 基线 commit: `d947ad71bc84dbe5441117b6c0e0400c39337909`
- 最终 commit: 待创建。

---

## 2026-09-10 — v3_change3 closure repair

### Baseline

- `2e8975e6d73862feda57bcaf6eeefc8d464ce8db`

### Verified changes

- 统一 Stage1/2a/2b/2c/3 source-first runtime contract、active raw MBC Eq.6、complex paired Eq.8、Phase-1 location-aware frequency evidence、factory/effective config、checkpoint/RNG/metrics 和 CPU preflight。
- 保留 full acquisition-supported FOV；cardiac box 仅为局部 motion/sampling/QC。
- 保留 legacy Stage1A/Stage1B/reference 工具，明确不在 source-first mainline。

### Verification

- 本地 `knesvr_torch`：`python -m unittest discover -s tests`，124 passed、1 explicit skip。
- 本地真实 DYL0709 no-step preflight：PASS；未运行 GPU 或长训练。
