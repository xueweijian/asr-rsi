# ASR-RSI 实验交付文档（索引）

本套文档记录一次完整实验：用 RSI（Recursive Self-Improvement）研发框架 `asr-rsi`，在 CPU 上对
sherpa-onnx 的 NeMo FastConformer-CTC int8 模型做推理加速，取得 **2.4989× 吞吐提升且 WER 零变化**。

文档放在仓库外的 `/workspace/asr-rsi-docs/`，不污染 `asr-rsi` 仓库本身。

## 文档清单

| 文件 | 内容 |
|---|---|
| [01-objective-and-contract.md](01-objective-and-contract.md) | 实验目的、任务契约、评分口径、预算与约束 |
| [02-prompts.md](02-prompts.md) | 全部提示词原文（setup / worker / 技能卡 / CI 触发参数） |
| [03-process-log.md](03-process-log.md) | 过程记录：从第一次 CI 到第一次成功的完整排障时间线 |
| [04-results.md](04-results.md) | 最终结果与数据：分数、WER、吞吐、资源、逐步骤证据 |
| [05-techniques.md](05-techniques.md) | 推理引擎、模型、核心技术/算法/工程技巧，以及被证伪的方案 |
| [06-generalization.md](06-generalization.md) | 泛化分析：对 sherpa 上百个模型是否都有效、边界在哪 |
| [07-run-trajectory.md](07-run-trajectory.md) | 多轮实验轨迹：每一次 run 的 ID、状态、失败原因、分数 |
| [08-notes.md](08-notes.md) | 我认为重要的事：坑、偏差、可信度评估、下一步 |

## 一句话结论

不改引擎、不改模型、不改精度，仅重写"调度层"（每流单线程解码 × 4 并发 + 去掉 padding 浪费 +
最长优先派发），在 2 物理核 / 4 逻辑核的 CI 主机上把吞吐从约 22 audio-s/s 提到约 55 audio-s/s，
框架评分器给出 **2.4989×**，配对 WER 差 **0.0**（置信区间 0..0），权重指纹校验通过。

## 关键事实速查

| 项 | 值 |
|---|---|
| 仓库 | https://github.com/xueweijian/asr-rsi（MIT，上游 `aibuildai-inc/aibuildai-llm-inference-agent`） |
| 最终交付代码 | `results/20261030-run10/engine.py`（172 行） |
| 最终 run | GitHub Actions run `41091147742`（success，5h26m，框架自行 finalize） |
| 推理引擎 | sherpa-onnx `OfflineRecognizer.from_nemo_ctc` + ONNX Runtime CPU |
| 模型 | NeMo FastConformer-CTC large, 10 语（be/de/en/es/fr/hr/it/pl/ru/uk）, 20k, int8, ~127MB |
| 语料 | LibriSpeech dev-clean（仅英语），119 dev / 48 hidden，按说话人切分 |
| 仪器 | `task/harness/benchmark.py`，双 arm、batch 8 吞吐 + batch 1 延迟、5 次重复、配对 bootstrap |
| 最终分数 | 2.4989（stock = 1.0），WER delta 0.0，gate/constraint 通过 |
| 主机 | 4 逻辑核 / 2 物理核，16GB；内存峰值 6.3GB；load 峰值 10.9 |
| 实验日期 | 2026-10-29 ～ 2026-10-30（UTC） |
