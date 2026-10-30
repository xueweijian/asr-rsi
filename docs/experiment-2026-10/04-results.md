# 04 · 最终结果与数据

## 4.1 最终成绩（run #10，GitHub Actions `41091147742`）

| 指标 | 值 |
|---|---|
| `score` | **2.4989**（stock = 1.0） |
| `speedup` | 2.4989 |
| `wer_delta` | 0.0 |
| `wer_delta_ci` | 0.0（lo / mean / hi 全为 0） |
| `gate_passed` | 1.0 |
| `constraint_ok` | 1.0 |
| `extra_weights_bytes` | 0（权重指纹校验通过，`may_change_weights: false` 未被触碰） |
| deliverable | `full/engine.py` + `full/manifest.json`，finalizer 校验通过 |

对应的绝对吞吐（来自同轮 probe，与仪器口径一致）：

- stock 臂（batch 8，`decode_streams`，`num_threads=2`）：**≈ 22 audio-s/s**（RTF ≈ 0.045）
- 提交臂（4 并发单流解码，`num_threads=1`，LPT 派发）：**≈ 56–58 audio-s/s**（RTF ≈ 0.018）
- 单条延迟（batch 1）**变差**：约为 stock 的 0.65×（见 4.4，这是有意的取舍）

## 4.2 run #10 的逐步骤数据（每一步都是仪器/探针实测）

| 步骤 | 假设 | 关键测量 | 结论 |
|---|---|---|---|
| step1（worker_1） | `decode_streams` 把整批 padding 到最长，纯浪费 | batch 8 下 padding 代价 1.25–2.0×；串行单流只要 0.40–0.80×；4 并发单流 0.43×（**2.33×**）；frontend 2.6–4.5%，encoder 95%+ | 换成"逐条 `decode_stream` + 并发" |
| step2（worker_2） | frontend 构建也是独立工作，可并行 | 14 个 batch：**997 ms → 666 ms**（1.50×）；build 占批墙钟约 9% | build 与 decode 同池 |
| step3（worker_3） | 4×2 线程在 4 核上超订 | 871 → **834 ms/batch**（nt 2→1）；合并 build+decode → **811 ms/batch** | `num_threads 2→1` + 任务合并 |
| step4（worker_4/5） | `Executor.map` 按列表顺序派发，导致尾部空转 | 738.4 → **703.3 ms/batch**；**55.71 → 58.49 audio-s/wall-s**；最短优先 762.2（单调劣化）；文本差 0/119 | 最长优先（LPT）派发 |
| step5（验证） | 调度面是否已关闭 | 对照 57.97 audio-s/wall-s；证伪：尾部前端预取 783.1 ms（-10%）、两阶段 build 726.0 ms（-2.5%） | 无新杠杆，保持交付 |

代价律（step5 拟合）：单流串行解码 ≈ **13.4 ms 固定开销 + 33.9 ms / 每音频秒**（后一版拟合 35.5 ms/s）。
前端开销：`create_stream` 0.09 ms，`accept_waveform` 1.65 ms / 每音频秒（占提交臂 CPU 的 4.4%）。

## 4.3 run #9 的轨迹（被 job 超时掐断的那轮，但数据完整）

| 步骤 | 记录分数（`score-result.json`） | 备注 |
|---|---|---|
| setup verifier ×2 | 1.0013 / 1.0009 | 基线与仪器健全性检查 |
| step1 | **1.6611** | 去 padding + 并发单流 |
| step2 | **2.3454** | build 并行 |
| step3 | **2.5759**（worker_6 plan 记录 2.6027） | 4 个 `nt=1` 等权克隆 |

run #9 的 worker_6（全保真）：

| 指标 | 值 |
|---|---|
| score | 2.6027 |
| stock 吞吐（batch 8） | 21.8015 audio-s/s（RTF 0.0459） |
| 提交吞吐（batch 8） | 56.7434 audio-s/s（RTF 0.0176） |
| 延迟臂（batch 1） | stock 39.4936 → 提交 25.7665（**0.6524×**） |
| WER | stock = 提交 = 0.030045，delta 0.0，transcript drift 0.0 |

两轮独立运行（#9 = 2.5759/2.6027，#10 = 2.4989）落在同一区间，
且 #10 的 finalizer 是**自行收尾**产出的，因此 2.4989 是这次实验的"正式成绩"。

## 4.4 一个必须讲清楚的取舍：吞吐 vs 单条延迟

- 吞吐臂（batch 8）：2.5–2.6×；
- 延迟臂（batch 1）：**0.65×**（更慢）。

原因：batch 1 时 `nt=1` 的单流解码比 stock 的 `nt=2` 慢；4 并发在 batch 1 下没有活可并行。
这是有意的取舍——任务是吞吐目标（objective: speedup），延迟只在报告里出现，不参与打分。
**如果真实部署关心单条尾延迟，这个提交不能直接用，需要按负载选择线程/并发策略。**

## 4.5 资源占用

| 项 | 值 | 来源 |
|---|---|---|
| 主机 | 4 逻辑核 / 2 物理核，16GB，无 GPU | `resource_samples.jsonl` |
| 采样 | 1240 点，跨 321.9 分钟 | 同上 |
| 内存峰值 | 6294 MB（整机 used） | 同上 |
| load1m | 均值 0.9，峰值 10.9 | 同上 |
| 单臂 RSS | 未记录（`max_peak_rss_mb: null`，本轮未启用内存门） | `components` |

内存门未启用是本轮的缺口：4 个实例共享同一 checkpoint（只读映射），实测没有爆内存，
但没有被写成门，下一轮若跑更大模型应补 `constraint.max_peak_rss_mb`。

## 4.6 结果的可信度

- **同仪器、同音频、同 checkpoint、双 arm 独立进程**，不是自己写的秒表；
- **五次重复取中位数**，report 带 spread；spread > 2% 会被视为噪声；
- **配对 bootstrap 置信区间**（2000 次重采样，seed 固定 20261029）给出 0..0；
- **transcript drift 0.0**：119 条 dev 与提交臂逐条相同（除 4.2 里各处记录的 0/119、0/112、0/48 对比）；
- **两轮独立 run 复现同一区间**（#9 与 #10 相差约 3–4%）。

不足：

- 分数与主机绑定。4 核机器上的 2.5× 不能当作 8 核/GPU 机器的 2.5×（见 [06](06-generalization.md)）；
- WER 只在英语 LibriSpeech 上验证，模型是 10 语（见 [08](08-notes.md)）；
- hidden 只有 48 条，WER 的统计分辨率有限（虽然本轮 delta 恰好为 0，结论不受影响）。
