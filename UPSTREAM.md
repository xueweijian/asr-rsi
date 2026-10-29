# Upstream

This repository is a fork of
[aibuildai-inc/aibuildai-llm-inference-agent](https://github.com/aibuildai-inc/aibuildai-llm-inference-agent),
MIT licensed, Copyright (c) 2026 AIBuildAI Inc. The fork point is upstream tip
`43cb78d` ("Update the tech report build"), and upstream commit history through that
tip is carried locally so the two remain diffable.

## Vendored unmodified

`framework/` is byte-identical to upstream at `43cb78d`. The search / Program / Agent
work units, tree and autoresearch strategies, knowledge base, sandbox, budget and
journal are upstream code and must not be edited in this fork.

## Dropped

Upstream `train/`, `eval/`, `heads/` and upstream `task/` are LLM-specific and are not
carried here. `task/` in this fork is a rewritten ASR contract; `search/` is adapted to
point at it.

## Syncing

Fetch upstream and compare the vendored tree against its source commit:

```bash
git remote add upstream https://github.com/aibuildai-inc/aibuildai-llm-inference-agent.git
git fetch upstream
git diff --stat 43cb78d..HEAD
```

`git diff --stat 43cb78d..HEAD` is the file-level fork manifest. `framework/` must
never appear in that diff; if it does, the vendoring has drifted.
