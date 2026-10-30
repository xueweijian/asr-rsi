# 03 · 过程记录（排障时间线）

这一段记录的不是"最终方案"，而是**把它跑起来的过程**：环境、systemd、密钥、模型延迟、
框架对未知消息的断言，每一层都单独失败过一次。留档的价值在于——任何人在自己的机器上复跑，
都会依次撞到这些墙。

---

## 3.1 三轮定位（先确认"哪里能跑"）

| 阶段 | 尝试 | 结论 |
|---|---|---|
| 本机 | `aibuildai run` | 失败：容器内 PID 1 是 tini，`systemctl --user` 连不上 bus；`dbus-run-session systemd --user` 也被拒 |
| 本机 | 只跑评测（`benchmark.py` 双 arm） | 成功：score 1.0152、stock spread 2.13%、submission 0.51%、RSS 470MB、167 条（119 dev / 48 hidden） |
| CI | GitHub 托管 runner | 成功：真 systemd VM，agent 循环可跑 |

于是定下分工：**agent 循环在 CI 跑，本机只做评测与 artifact 分析**。

## 3.2 第一波：仓库权限

`gh repo create` / `POST /user/repos` → `403 Resource not accessible by integration`。
App token 没有建仓权限，最后是用户在网页建了空仓库 `xueweijian/asr-rsi`。
（同一 token 后续对 **写 secrets**、**workflow_dispatch** 也是 403，这就是后来改用 push 触发的原因。）

## 3.3 第二波：CI 启动期七层失败

按发生顺序，每一层都修完才露出下一层：

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 1 | `aibuildai: command not found`（exit 127） | 每个 step 是独立 shell，`. .venv/bin/activate` 不跨 step；且 `aibuildai setup \|\| true` 把失败吞了 | `echo "$GITHUB_WORKSPACE/.venv/bin" >> $GITHUB_PATH` + 写 `VIRTUAL_ENV` 到 `$GITHUB_ENV`；删掉 `\|\| true` |
| 2 | `configure: error: library 'uuid' is required for E2FS UUID`（exit 2） | runner 缺 `-dev` 头文件 | 新增 apt 步骤：`bison flex libreadline-dev uuid-dev libossp-uuid-dev zlib1g-dev` |
| 3 | `aibuildai: the systemd user manager is not reachable` | 托管 runner 从未登录 runner 账号，没有 user manager | `loginctl enable-linger` + `systemctl start user@UID.service` + 轮询 `is-system-running` |
| 4 | `Failed to connect to bus: No such file or directory` | `XDG_RUNTIME_DIR` 硬编码 `/run/user/1000`，真实 uid 是 **1001** | 改为 `UID_=$(id -u)` 动态推导，写进 `$GITHUB_ENV`；预设 `user@.service` 的 `Delegate=` drop-in |
| 5 | `ANTHROPIC_BASE_URL contains a newline` | 手机粘贴 secret 带尾部换行，systemd 拒绝含换行的 env | "Sanitize secrets" 步骤 `tr -d '\r\n' \| xargs` 清洗后写回 `$GITHUB_ENV` |
| 6 | `AssertionError: unclassified Claude system message 'session_title_changed'` | 新版 Claude Code 新增 system subtype，vendored 框架按"未知即失败"处理 | 把该断言降级为 `backend_log(level="warn")`（唯一一处对上游的补丁） |
| 7 | Run 步骤"绿"，实际失败了 | `aibuildai run \| tee /tmp/run.log`：`tee` 的状态掩盖了管道前段的退出码 | 加 `set -o pipefail` |

第 5 层最隐蔽：日志只说 "contains a newline"，不说哪个变量；第 6 层最危险：它不是环境问题，
而是"上游框架假设自己的消息分类是完备的"，一旦模型侧多发一种系统消息，整轮搜索被拆掉。

## 3.4 第三波：预算与模型延迟

- 45 分钟的 work unit 预算 vs spark-x2.6 的 60–120 s/轮思考延迟：
  setup 写完 `score.py` 却来不及 `StructuredOutput`，被切断（run #8 两次）；
- worker 同样：run #9 的 worker_1 整段预算用于重算 baseline，零产出；worker_3/5/7 在预算边缘死掉；
- job `timeout-minutes: 350` 比框架自身的 330 分钟预算太近：
  run #9 被 **job 超时**掐断在步骤中间，框架没有机会 finalize，因此没有 deliverable
  （artifact 里 7 个 worker 的引擎都在，但整轮被判取消）。

修正：job 超时 600 分钟 ≫ 框架 480 分钟，只让框架自己停；`max_steps` 10 → 5；
worker 预算 45 → 60 分钟。

## 3.5 最终成功的那一轮（run #10）

```
16:34Z  push .github/rsi-trigger → rsi-run 启动
16:34Z  checkout / python / sanitize secrets
16:3xZ  venv + 框架安装 + sherpa-onnx + claude-code CLI
16:3xZ  Build dependencies → Owned tools (PostgreSQL 18.6) 成功
16:4xZ  Model（127MB int8）+ 语料（LibriSpeech dev-clean）→ 119 dev / 48 hidden
16:4xZ  Render config（sed 覆盖预算）→ systemd user manager 起来
16:4xZ  aibuildai run 开始
   ↓     setup_1 交付 private/score.py（2 个 verifier 打 1.0013 / 1.0009）
   ↓     worker_1 → step1 1.6611
   ↓     worker_2 → step2 (~2.3, proxy)；全保真 2.3454 路径
   ↓     worker_3 → step3（num_threads 2→1 + build/decode 合并）
   ↓     worker_4 / worker_5 → step4 各自独立得到同一结论（LPT 派发）
   ↓     finalizer_1 校验 deliverable（engine.py + manifest.json）
22:01Z  框架自行 finalize，job success（5h26m / 上限 480 分钟）
```

`resource_samples.jsonl` 全程 1240 个采样点，跨 321.9 分钟：
内存峰值 6294MB，load1m 均值 0.9、峰值 10.9，GPU 不存在（`gpus: []`）——确认全程 CPU 路径。

## 3.6 一条值得记住的"失败形态"

> 框架对"未知"的态度是 fail loud（这是它对产品安全的设计），
> 但在**开放式 agent 运行时**里，"未知"是常态：模型会加新的消息类型、CLI 会加新的子类型。
> 一个兼容补丁（warn 而非 raise）让整条流水线从"每轮必死"变成"可以跑完"。

这条经验对任何"用 LLM 驱动长流程"的系统都成立：**分类器不完备时，降级记录优于整体失败**。
