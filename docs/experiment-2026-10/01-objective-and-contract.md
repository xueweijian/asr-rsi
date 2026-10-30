# 01 · 实验目的与任务契约

## 1.1 这次实验到底在做什么

上游项目 `aibuildai-inc/aibuildai-llm-inference-agent` 是一个 RSI（Recursive Self-Improvement）
研发框架：它不训练模型，而是驱动一组 LLM agent 反复"提出假设 → 改代码 → 用仪器测 → 只保留被验证的改动"。
它原本的任务域是 LLM 推理。本次实验把框架**分叉**成通用 ASR 推理优化框架 `asr-rsi`：

- `framework/` 原样 vendoring（保留上游的搜索循环、work unit 调度、验证、知识积累），只打了一个兼容补丁；
- `task/` 换成 ASR 契约：冻结的模型、双 arm 仪器、WER 门、transcript drift 门；
- 模型后端无关：sherpa-onnx、FunASR、Whisper、SenseVoice 都只是未来往 `models.json` 里填的适配对象。

所以实验对象**不是**"给 FunASR 写个应用"，而是"ASR 领域的通用加速研究框架"能不能真的跑起来、
能不能真的找到加速，并且结论可信。

## 1.2 被优化的目标（冻结）

| 项 | 值 | 说明 |
|---|---|---|
| 推理引擎 | sherpa-onnx（Apache-2.0） | 顶层 API `OfflineRecognizer.from_nemo_ctc` |
| 底层算子 | ONNX Runtime CPU EP | 未换运行时、未改图、未重量化 |
| 模型 | `sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k-int8` | NeMo FastConformer-CTC large，10 语，20k 词表，int8，~127MB，约 1.1 亿参数 |
| 解码 | greedy CTC，`num_threads = 2` | 冻结基线；只有 `tunable` 白名单键可被提交覆盖 |
| 硬件 | CPU，4 逻辑核 / 2 物理核 | 当前实验环境，不是框架限制 |

关键约束：`may_change_weights: false`。提交不得改权重、不得换模型、不得改词表/前端/解码方法；
`build_runtime()` 对白名单外的解码参数直接抛 `PermissionError`。

## 1.3 评分口径

```
speedup = 提交臂 throughput (audio-seconds / wall-second)
          ------------------------------------------------
          stock 臂 throughput (audio-seconds / wall-second)
```

- 两个 arm **各自独立进程**，同一份音频、同一 checkpoint、同一 adapter，只有提交插件不同；
- 吞吐用 batch = 8，延迟用 batch = 1；
- 每个 arm 重复 5 次取中位数，report 里带 `repeat_spread`；spread > 2% 表示主机噪声大，比值不可信；
- 分数不是"墙钟跑一次"，而是"在运行时自己的机器上打赢它自己的默认路径"。

约束不是可选项：

```
objective: speedup
constraint: {metric: wer, max_delta_abs: 0.003, confidence: 0.95, paired: true}
```

- 配对 bootstrap 的 WER 差**上界**必须 ≤ 0.003，否则分数记 0（3× 加速 + 0.5% 退化 = 0 分）；
- transcript drift 门：非 `wer` 目标下，转写与 stock 不一致即封顶到 stock 分数。

## 1.4 提交形态

```
<attempt>/full/
  engine.py       # build(cfg) -> EngineHooks
  manifest.json   # 改了什么 + 证据 + extra_weights_bytes
```

只有四个钩子：`configure`（load 前）、`prepare`（load 后接管 runtime）、`preprocess`（进模型前）、
`postprocess`（解码后）。需要第五个钩子的杠杆，被视为另一个任务。

## 1.5 语料与切分

- 语料：LibriSpeech `dev-clean`（英语），取 6 个 chapter、最多 260 条；
- 切分：按**说话人**切 dev / hidden（早前按 utterance id 哈希切分会造成 hidden 泄漏，已修）；
- 规模：dev 119 条（~617 秒音频）、hidden 48 条；
- hidden manifest 只给评分器，不进 agent 的盒子（隔离靠挂载策略）。

局限：模型是 10 语，但 WER 只在英语上测——这点在 [08-notes.md](08-notes.md) 里单独讨论。

## 1.6 预算

| 参数 | run #10 实际值 | 说明 |
|---|---|---|
| wall clock | 480 分钟 | 框架整体预算；job `timeout-minutes: 600` 留 2 小时余量 |
| cost cap | 150 USD | 用户明确"不怕额度" |
| per agent | 35 USD | 单个 agent 上限 |
| max_steps | 5 | 上一轮 10 步做不完，改为 5 步并强制早交付 |
| 每个 work unit | 60 分钟 | worker / setup 等 |
| 模型延迟现实 | 60–120 s/轮思考 | spark-x2.6 的实际往返，预算必须按它来定，不能按 token 数 |

## 1.7 为什么必须跑在 GitHub Actions

框架每个 work unit 都必须落在 **systemd user cgroup** 里，源码明确 "no fallback"。
本机是容器（PID 1 = tini），`systemctl --user` 永远连不上 bus，`dbus-run-session systemd --user` 也被拒。
托管 runner 是真正的 systemd VM，因此 agent 循环只能在 CI 里跑；
本机只负责"评测 + 分析 artifact"，这条分工是本次实验得以完成的前提。
