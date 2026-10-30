# 05 · 引擎、模型、核心技术与算法

## 5.1 用的是什么推理引擎，什么模型

| 项 | 值 |
|---|---|
| 推理库 | **sherpa-onnx**（Apache-2.0） |
| 顶层 API | `sherpa_onnx.OfflineRecognizer.from_nemo_ctc(...)` |
| 实际执行 | **ONNX Runtime CPU Execution Provider**（未换 Runtime、未改图、未重量化） |
| 模型 | `sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k-int8` |
| 结构 | NVIDIA NeMo **FastConformer-CTC large**（多语 hybrid，10 语） |
| 规模 | 约 **1.1 亿参数**，词表 20k，int8 权重文件 **~127MB**（fp32 版约 400MB） |
| 解码 | greedy CTC |
| 前端 | sherpa-onnx 自带 fbank（`create_stream` + `accept_waveform`） |
| 硬件 | CPU：4 逻辑核 / 2 物理核 |

一句话：**sherpa 一行没改**。提速发生在 sherpa 的 API 调用方式上（每流单线程 + 并发 + 派发顺序）。

## 5.2 提交改了什么（四个钩子里的两个）

```python
def build(cfg: dict) -> Hooks:
    return _ConcurrentSingleStreamHooks()

class _ConcurrentSingleStreamHooks(Hooks):
    def configure(self, cfg):
        # 唯一允许覆盖的 tunable 键：num_threads 2 -> 1
        cfg.setdefault("target", {}).setdefault("decode", {})["num_threads"] = 1
        return cfg

    def prepare(self, runtime):
        executor = ThreadPoolExecutor(max_workers=4)
        def _build_and_decode(req):
            stream = _new_stream(req)          # create_stream + accept_waveform
            runtime.model.decode_stream(stream) # 单流解码，无 padding
            return stream, front_ms, dec_ms
        def transcribe(batch):
            order = sorted(range(len(batch)), key=lambda i: -_cost(batch[i]))  # LPT
            done = list(executor.map(lambda i: _build_and_decode(batch[i]), order))
            ...
        runtime.transcribe = transcribe       # 只替换这一个方法
```

`preprocess` / `postprocess` 完全未动：音频和输出的文本都不进入改造范围。

## 5.3 核心技巧一：消除 padding 浪费（收益最大）

`sherpa-onnx.OfflineRecognizer` 有两个解码入口：

| 入口 | 行为 |
|---|---|
| `decode_streams([s1..s8])` | 把 8 条音频**全部 pad 到本批最长那条**，做一次前向 |
| `decode_stream(s)` | 只解码这一条，不为别人 padding |

CTC/FastConformer 的编码器是帧级卷积+注意力，**padding 的帧照样要算**。
所以 batch 8 只要有长短混合，就为"看不见的帧"付算力：实测代价是串行单流的 **1.25–2.0×**。
而"串行单流"只要 batched 的 **0.40–0.80×**——**批处理在这个 runtime 里共享不了任何计算，
只增加了 padding**。

这不是"批处理一定坏"，而是"这个 runtime 的批处理只做了 padding 对齐，没有做真批量化
（没有把 8 条拼成 batch 维、没有跨请求的 GEMM 合批）"。因此对这套 API，逐条解码严格更省。

实证（worker_1，dev 119 条）：

```
串行单流 / decode_streams = 0.40–0.80
4 并发单流 / decode_streams = 0.43   -> 2.33×
```

## 5.4 核心技巧二：线程级并发（吃满物理核）

两条前提让"Python 线程并发"真的能并行：

1. **ORT 在 `Run` 期间释放 GIL**——线程真能同时跑在核上，不是伪并行；
2. 模型是 CPU 上的 int8，**单次解码的算术强度不足以让 2 个 intra-op 线程线性加速**，
   于是 stock 的"2 线程串行"在 4 逻辑核上只用了约一半算力。

提交的做法：

- 4 个池 worker，每个解码任务 `num_threads=1` → 4 线程 / 4 逻辑核，**无超订**；
- 用**线程池**而不是多进程：4 个实例共享同一份只读 checkpoint（ONNX Runtime session 复用权重内存），
  内存不翻倍；
- 线程数扫描（probe13）：`t1w4` 58.49 > `t1w6` 56.27 > `t1w8` 55.17 > `t2w4` 54.22
  → 4 个单线程 worker 最优，再加就是超订。

关键反例：**"4 实例 × 2 线程"（8 线程）更慢**；**绑核（affinity）更差**（13.14 s vs 10.81 s）——
绑核把 4 个线程钉在 2 个物理核的 HT 兄弟上，调度器失去迁移自由度。

## 5.5 核心技巧三：把 frontend 藏进解码后面（流水化）

`sherpa-onnx` 的前端是**每请求独立**的工作：`create_stream()` + `accept_waveform()`。
在 4 并发下它占批墙钟约 **9%**（step2 实测 997 ms → 666 ms，1.50×）。

做法不是"先全部 build 再全部 decode"（两阶段，实测 726.0 ms，比合并版 708.1 ms 差 2.5%），
而是**每个池任务自己 build + 自己 decode**：晚到的任务的前端工作，藏在早到任务的解码时间后面。
这相当于一个两条工位的流水线，而不是两个串行阶段。

## 5.6 核心技巧四：最长优先派发（LPT 调度）

`ThreadPoolExecutor.map` 是**列表顺序**投递，空闲 worker 取队首任务，
所以 makespan 等于最忙 worker 的负载，而不是总负载 / worker 数。

dev manifest 一个 batch 内音频时长差最高 **8 倍**（2.3 s ～ 14.6 s），
解码代价与音频时长成正比（33.9 ms / 音频秒），因此"按 manifest 顺序"会让某些 worker 提前空闲、
最后一个长音频独自跑完。

提交按**音频时长降序**投递，等价于**LPT（Longest Processing Time first）**：
对 m 台同速机器的贪心列表调度，LPT 的 makespan 上界是 `(4/3 − 1/(3m))·OPT`，
本机实测 738.4 → 703.3 ms/batch（+5%），且"最短优先"是 762.2 ms——**单调关系说明这是调度效应，不是噪声**。

## 5.7 被证伪的方案（这部分同样重要）

| 方案 | 实测 | 结论 |
|---|---|---|
| 4 实例 × `num_threads=2`（8 线程） | 更慢 | 4 核上超订，线程互相抢占 |
| CPU 亲和绑核 | 13.14 s vs 10.81 s | 负收益，HT 兄弟核绑死更糟 |
| `OMP_NUM_THREADS=1` 环境变量 | 10.87 s vs 10.81 s | 在噪声内，无意义（真正的控制点应是 session 参数） |
| 最短优先派发 | 762.2 ms（对照 738.4） | 单调劣化，验证 LPT 方向正确 |
| 两阶段 build→decode | 726.0 ms vs 708.1 | 前端暴露，-2.5% |
| 尾部前端预取（主线程） | 783.1 ms vs 709.7 | -10%，主线程被前端阻塞 |
| batched `decode_streams` + 大线程数 | 1964 ms（nt=2）→ 4635（nt=8）→ 9715（nt=16） | 越并行越崩，padding 图争核 |

## 5.8 为什么 WER 一定是 0

- 每条音频仍然走 `decode_stream(s)`：**同一 checkpoint、同一 greedy CTC、同一前端**；
- batch 1 路径（评分门实际读的臂）与 stock **逐字节相同**（0/119 差异）；
- batch 8 路径下并发与串行对比也 0/112、0/119 差异；
- 权重指纹校验通过，`extra_weights_bytes = 0`。

所以这不是"用一点点精度换速度"，而是**调度层面的等价变换**。

## 5.9 这套技巧的适用条件（摘要）

有效的前提是三条**同时**成立：

1. runtime 的批处理只做 padding、不做真合批（sherpa-onnx 的 `decode_streams` 即如此）；
2. 机器核数 > stock 配置的线程数（本机 4 核 vs 2 线程）；
3. batch 内时长分布不均匀（真实语料几乎总是如此）。

详见 [06-generalization.md](06-generalization.md)。
