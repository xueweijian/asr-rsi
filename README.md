# ASR-RSI

A recursive self-improvement framework for automatic speech recognition. An agent
proposes speed levers, edits the runtime, benchmarks them, and keeps what it learned —
the same loop as [AIBuildAI LLM Inference Agent](https://github.com/aibuildai-inc/aibuildai-llm-inference-agent),
with the LLM task layer swapped for an ASR one.

The loop is **model-agnostic**: no file here names a model. Which backend and checkpoint
to accelerate is declared in a single file, [`task/models.json`](task/models.json), so any
runtime a ~100-line adapter can wrap goes through the identical search.

> **Status:** `framework/` is vendored byte-identical; the task layer is rewritten; the
> instrument self-tests offline with no GPU, no audio, no checkpoint. No real backend
> ships yet — see [What is deliberately unfinished](#what-is-deliberately-unfinished).

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
# 0. clone (worktree-based runs need the history that ships here)
git clone https://github.com/xueweijian/asr-rsi.git && cd asr-rsi

# 1. framework environment. The lock pins idna==3.19, which no index carries yet,
#    so install unpinned. Build outside the git tree: inside it the conversation
#    package insists on a CLIProxyAPI gitlink this fork does not carry.
python3 -m venv ~/asr-rsi-venv && . ~/asr-rsi-venv/bin/activate
cp -r framework /tmp/framework && cd /tmp/framework
export AIBUILDAI_CLIPROXYAPI_SOURCE_COMMIT=$(python -c "import json;print(json.load(open('packages/conversation/src/aibuildai_conversation/_native/identity.json'))['commit'])")
pip install -e ./packages/execution -e ./packages/conversation -e . && cd -

# 1b. the harness needs a runtime, not just the CLI
pip install sherpa-onnx soundfile numpy pyyaml

# 1c. owned tools (PostgreSQL is compiled from source; the CI lane builds it once)
gh run download -R xueweijian/asr-rsi -n owned-tools-linux-x86_64 -D /tmp/owned
tar -xzf /tmp/owned/owned-tools-*.tar.gz -C ~/.aibuildai
~/.aibuildai/tools/postgresql/18.6/bin/postgres --version

# 2. corpus + model + manifests (127MB model, 338MB corpus, one fetch each)
./data/fetch_local.sh

# 3. one measured arm on the real model, before any agent touches it
SHERPA_MODEL_DIR=~/asr-models/sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k-int8 \
  python task/harness/benchmark.py --submission task/example_submission \
  --limit 60 --archive runs/jsonl

# 4. launch (budgets in search/run_config.yaml are set for a short first run)
. ~/.asr-rsi.env   # ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN
export AIBUILDAI_SKIP_SUBSCRIPTION_CHECK=1
./search/launch.sh ~/asr-rsi-playground
```

Any number from `--limit` has wider repeat spread than a full run: 60 utterances on this
model is one measured arm of a couple of minutes, which is the right scale for a first
smoke test.

`--selftest` uses `harness/backends/dummy.py` and synthesises its manifest, so it has no
audio to give a real backend. With `models.json` pointing at sherpa it fails by design;
keep it for instrument-only work.


## Adding a backend

Copy [`backends/_template.py`](task/harness/backends/_template.py) to `<your_module>.py`
and fill in three methods:

```python
class Runtime:
    def load(self, cfg: dict) -> None: ...
    def transcribe(self, batch: list[Request]) -> list[Result]: ...
    def fingerprint(self) -> dict: ...      # params count + weight hash
    paradigm = None                         # optional: AED | SPEECH_LLM | TRANSDUCER | NAR_CIF_CTC | HYBRID
```

Then set `models.json.target.entrypoint` to `backends.<your_module>:build`. The adapter
may import any ASR library; the *submission* may not. Declaring `paradigm` (or
`target.paradigm`) only feeds a routing prior to the agent -- it changes no measurement.

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
(MIT, Copyright (c) 2026 AIBuildAI Inc.). The full upstream history is carried (68 commits
through `43cb78d`), so this fork stays diffable against upstream. `framework/`
is vendored unmodified; the LLM-specific trees (`train/`, `eval/`, `heads/`, upstream
`task/`) are dropped, and `task/` here is the ASR contract above. Terms and provenance:
[LICENSE](LICENSE), [DATA_AND_LICENSES.md](DATA_AND_LICENSES.md). See
[UPSTREAM.md](UPSTREAM.md) for the vendoring boundary and the sync commands.

### Diff against upstream

`git diff --stat 43cb78d..HEAD` is the file-level fork manifest; `framework/` must not
appear in it.
