# ASR-RSI

A fork of [AIBuildAI LLM Inference Agent](https://github.com/aibuildai-inc/aibuildai-llm-inference-agent)
turned into a **model-agnostic ASR recursive-self-improving agent**.

Upstream is already a domain-agnostic RSI framework with an LLM task bolted on. This
fork keeps the framework byte-identical and replaces the task layer, so the same loop
runs on any ASR backend a hundred-line adapter can wrap.

## What was kept, what was replaced

| path | status | role |
|---|---|---|
| `framework/` | **unmodified, vendored** (MIT) | Search / Program / Agent work units, tree & autoresearch strategies, knowledge base, sandbox, budget, journal |
| `search/run_config.yaml` | adapted | `run.data_root` → `../task`, `search.kind: autoresearch`, ASR `system_instructions` |
| `search/launch.sh` | adapted | substitutes paths, refuses a playground inside the repo |
| `search/score.reference.py` | rewritten | thin scorer over the new harness |
| `task/README.md` | **rewritten** | the contract SETUP freezes and implements |
| `task/task.yaml` | **new** | objective, constraint, measurement protocol |
| `task/models.json` | **new** | the only place a backend/checkpoint may be named |
| `task/harness/` | **rewritten** | instrument: adapters, hooks, WER/RTF, paired bootstrap, two-arm benchmark |
| `task/skills/` | **written** | the three knowledge documents every worker reads |
| `eval/`, `train/`, `heads/`, `tools/` | dropped | LLM-specific |
| `task/skills/specdec-*`, `task/harness/*sglang*` | dropped | tied to one engine |

## The loop, unchanged in shape

```
run_config.yaml
  └─ SETUP          freezes task/README.md, writes private/score.py from the contract
      └─ autoresearch × max_steps
          ├─ DESIGNER   proposes a lever + predicted effect
          ├─ WORKER     edits <attempt>/full/engine.py, benchmarks, exports every step
          └─ SCORER     check_submission.py → benchmark.py → score
              └─ knowledge base grows with what was tried and what it cost
```

The score is a **ratio measured inside the frozen runtime**, so a submission cannot win
by loading a different model:

```
score = throughput(submission) / throughput(stock)     constraint: paired WER Δ ≤ 0.3%
```

## The only extension surface

```python
def build(cfg: dict) -> Hooks        # engine.py
```

Four hooks — `configure`, `prepare`, `preprocess`, `postprocess`. Anything a hook
cannot express is a different task, not a clever submission. See
[harness/backends/base.py](task/harness/backends/base.py) and
[task/README.md](task/README.md).

## Quickstart

```bash
# 0. the fork is not a git repo yet; worktrees need one commit
git init && git add -A && git commit -m "asr-rsi scaffold"

# 1. framework environment (upstream guide has the verified commands)
python3 -m venv ~/asr-rsi-venv && . ~/asr-rsi-venv/bin/activate
export AIBUILDAI_CLIPROXYAPI_SOURCE_COMMIT=<value in upstream search/README.md>
cd framework && pip install -c requirements.lock -e ./packages/execution \
  && pip install -c requirements.lock -e ./packages/conversation \
  && pip install -c requirements.lock -e . && cd ..

# 2. instrument self-test: no GPU, no audio, no checkpoint
python task/harness/check_submission.py --submission task/example_submission
python task/harness/benchmark.py --submission task/example_submission --selftest

# 2b. the two signals a real step uses
#   --proxy    : score measurement.proxy_utterances, set `promote` (filter, not final)
#   --archive  : append to a Pareto archive so selection uses the frontier
python task/harness/benchmark.py --submission task/example_submission --proxy \
  --archive ~/asr-rsi-playground/archive.jsonl

# 3. point it at a real target (edit models.json only)
#    backend + entrypoint + checkpoint + decode + sample_rate

# 4. build manifests from your corpus
python data/make_manifest.py /path/to/corpus --out data/dev_manifest.jsonl --split dev
python data/make_manifest.py /path/to/corpus --out data/hidden_manifest.jsonl --split hidden

# 5. launch
export AIBUILDAI_SKIP_SUBSCRIPTION_CHECK=1
./search/launch.sh ~/asr-rsi-playground
```

`--selftest` uses `harness/backends/dummy.py`, which emits deterministic nonsense so
the timing, bucketing, WER and bootstrap paths can be exercised offline. Any number
it produces is meaningless **except** as evidence that the instrument runs.

## Adding a backend

One file under `task/harness/backends/`, implementing:

```python
class Runtime:
    def load(self, cfg: dict) -> None: ...
    def transcribe(self, batch: list[Request]) -> list[Result]: ...
    def fingerprint(self) -> dict: ...      # params count + weight hash
```

Then set `models.json.target.entrypoint` to `backends.<your_module>:build`. The adapter
may import any ASR library; the *submission* may not.

## What is deliberately unfinished

* **No real backend ships here.** `dummy` is the only adapter. A real one is ~100 lines
  and needs the runtime's version pinned, exactly like upstream pins SGLang.
* **No audio ships here.** `data/` holds only the manifest builder.
* **The knowledge base is empty.** Upstream seeds `framework/memory/patterns/`; add an
  `asr-*` pattern file after your first run instead of guessing upfront.
* **Hidden-split plumbing is a convention, not a mechanism.** The scorer must run where
  `data/hidden_manifest.jsonl` exists and the agent's box must not have it. Isolate
  with a mount policy; the harness alone cannot enforce it.

## Upstream & license

A fork of [aibuildai-inc/aibuildai-llm-inference-agent](https://github.com/aibuildai-inc/aibuildai-llm-inference-agent)
(MIT, Copyright (c) 2026 AIBuildAI Inc.), upstream commit history preserved. `framework/`
is vendored unmodified; the LLM-specific trees (`train/`, `eval/`, `heads/`, upstream
`task/`) are dropped, and `task/` here is the ASR contract above. Terms and provenance:
[LICENSE](LICENSE), [DATA_AND_LICENSES.md](DATA_AND_LICENSES.md).
