# 02 · 提示词原文与演化

提示词全部写在 `search/run_config.yaml` 的 `llm.system_instructions` 里，不散落在代码中。
下面是 **run #10 实际使用**的原文（逐字），以及它们为什么长成这样。

---

## 2.1 setup 系统提示（逐字）

```text
This task speeds up ASR inference inside a frozen runtime. Three facts about this host:

1. Scoring runs on CPU. It loads a 127MB int8 NeMo FastConformer CTC model covering ten
   EU languages and decodes 60 utterances; one arm takes a couple of minutes, not tens.
2. Scoring runs with NO network. The 127MB bundle and the corpus must already be on disk
   before you finish, and you must prove they load offline. data/fetch_local.sh does both.
3. The backend is already wired in models.json (sherpa_nemo_ctc). Do not swap the model:
   the target is frozen, and swapping it is a different task.

Write private/score.py as a thin wrapper around the task's own harness: run
harness/check_submission.py first; if its JSON `passed` is false return {'score': 0.0};
otherwise run harness/benchmark.py and return its `score`. Read their JSON, never their
exit code. Record speedup, wer_delta, wer_delta_ci, constraint_ok, gate_passed as components.
Use task/example_submission as the baseline attempt: it is complete and scores ~1.00.
benchmark.py --proxy sets `promote`; a full run sets it null. A proxy score is a filter,
never a final number, and the same seed and bootstrap settings must be used at both fidelities.

Order matters more than polish. The deliverable is private/score.py, so write it FIRST,
verify it once end to end, and call StructuredOutput the moment it is correct. A single
arm on a 119-utterance dev manifest takes seconds; do not build extra scaffolding (corpus
generation, README rewrites, synthetic audio) that the task folder already provides.
The first two setup attempts here died at the wall-clock limit with score.py written but no
StructuredOutput, so if a Budget Reminder arrives, finalize immediately.
```

设计意图：

- **只给"必须知道的三件事"**：CPU、断网、后端已接好。其余细节让 agent 读 `task/README.md`；
- **明确交付物是 `private/score.py`**，不是论文、不是语料、不是 README；
- 惩罚历史教训写成硬规则（"write it FIRST… call StructuredOutput the moment it is correct"），
  因为前两轮 setup 都是"脚本写完了但没交付"被预算切断；
- `--proxy` 只做筛选、不当最终分数。

## 2.2 worker 系统提示（逐字）

```text
Read task/README.md before changing anything: its measurement rules decide whether a number
means anything. Never edit models.json, task.yaml, or anything under harness/ - they are the
instrument. Write your own implementation; do not copy code from elsewhere on this machine.

Your submission is engine.py: build(cfg) -> Hooks, and only four hooks exist: configure,
prepare, preprocess, postprocess. A lever that needs a fifth hook is a different task.

The host is CPU-only. Weights are already int8 and the decode is greedy CTC, so the levers
that exist are thread count and affinity, ORT graph optimisation and arena policy, sentence
batching, feature reuse across requests, and silence trimming. A dtype swap is not a lever
here and will be rejected as a change to the instrument.

Export a complete submission directory every step, even an unambitious one. A step that
exports nothing is scored as nothing. Your whole step - reading, editing, benchmarking,
exporting - must fit in the step budget. A step killed at its wall-clock limit loses its work;
nothing resumes it. Export early and improve with the time left.

Start from the previous step's shipped engine, not from first principles: read the newest
records/*/README.md and its artifacts/full/engine.py before touching anything, then make one
bounded change on top of it. Step 1 of the last run spent its whole budget re-deriving the
baseline and exported nothing, and steps 3, 5 and 7 died at their limit with unvalidated code.
Ship a harness-scored submission in the first half of the budget, then refine.

Before choosing a lever, measure the cost decomposition on this host: frontend, encoder,
decoder, launch_bubble. State which slice your change targets and your predicted gain. A step
that leaves the score flat while measurably improving one slice is progress if you record it.

The constraint is a paired WER interval, not a point estimate. A 3x speedup with a 0.5%
WER regression scores zero. Check the interval before you declare a step successful.
Intermediate steps use benchmark.py --proxy; only a promoted attempt gets a full-fidelity
measurement. Record every measurement with --archive, and when the frontier holds more than
one point, choose a parent off the Pareto front rather than the single top score.
```

设计意图：

- **先立"仪器不可改"**：`models.json` / `task.yaml` / `harness/` 是仪器，改了分数就没有意义；
- **把可用的杠杆列出来**（线程/亲和、ORT 图优化与 arena、批处理、特征复用、静音裁剪），
  同时明确"换 dtype 会被当作改仪器"；
- **"每步必须导出完整提交"**——上一轮 7 个 worker 里有 3 个在预算边缘被切，工作直接丢失；
- **"从上一轮已交付的 engine 起步"**——上一轮 worker_1 整个预算花在重算 baseline，零产出；
- **要求先做代价分解**（frontend / encoder / decoder / launch_bubble），
  这样"分数没动但某一层显著改善"也能记为进展；
- **约束是区间不是点估计**，并明确"3× 加速 + 0.5% 退化 = 0 分"。

## 2.3 提示词演化（这一版是怎么长出来的）

| 版本 | 改了什么 | 触发原因 |
|---|---|---|
| v1 | 原始 setup/worker 提示 | 初版 |
| v2 | setup 加"顺序优先"段（先写 score.py → 验一次 → 立刻 StructuredOutput；禁自建语料/README） | run #8 两次 setup 都在 45 分钟上限被杀，"脚本已写完、没交付" |
| v3 | worker 加"从上一轮 shipped engine 起步"+"前半段必须交付一个被打分的提交" | run #9 worker_1 零产出，worker_3/5/7 在预算边缘死掉 |
| v4（run #10） | `max_steps 10 → 5`；worker 预算 45 → 60 分钟 | 让框架自己收尾，而不是被 job timeout 掐断 |

## 2.4 技能卡（按角色注入）

`run_config.yaml` 通过 `skills:` 把三张本地技能卡挂到 `worker` 角色：

| 技能 | 路径 | 作用 |
|---|---|---|
| `asr-methods` | `task/skills/asr-methods` | 可用的加速方法清单与适用条件 |
| `asr-performance-model` | `task/skills/asr-performance-model` | 这台机器的代价模型（frontend/encoder/launch bubble） |
| `asr-measurement` | `task/skills/asr-measurement` | 怎么读仪器、怎么读 CI、什么数字不可信 |

（早期这三张卡沿用了上游的 `specdec-*` 路径，导致启动即死，已修。）

## 2.5 CI 触发参数（等价于"给 agent 的环境提示"）

```yaml
on:
  push:
    paths: [.github/rsi-trigger]     # 我 push 即触发，无需人去点按钮
  workflow_dispatch:
    inputs:
      wall_clock_minutes: 480
      cost_usd: 150
      worker_minutes: 60
      per_agent_cost_cap_usd: 35
```

为什么用 `push` 而不是 `workflow_dispatch`：当前 GitHub App token 对 dispatch 与写 secrets 都是 403，
而 `contents: write` 可用——bump 一个 marker 文件就能起跑，且触发器本身就是审计日志。

## 2.6 环境提示（job env，等价于隐含提示词）

```yaml
AIBUILDAI_SKIP_SUBSCRIPTION_CHECK: "1"
ANTHROPIC_BASE_URL: <secret>
ANTHROPIC_AUTH_TOKEN: <secret>          # 只在 GitHub Secrets，绝不入库
ANTHROPIC_SMALL_FAST_MODEL: spark-x2.6
SHERPA_MODEL_DIR: <workspace>/.models/sherpa-onnx-...-int8
```

密钥纪律：`run.log` 被刻意**不**上传为 artifact（GitHub 只对 job log 打码，不会清洗上传文件）。
