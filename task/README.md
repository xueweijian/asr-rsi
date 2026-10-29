# Improve an ASR system inside a frozen serving runtime

This task is model-agnostic and framework-agnostic. The **target** (which ASR model,
which runtime adapter, which decode settings) is declared in [`models.json`](models.json)
and nowhere else. Your job is to make that target better **without replacing it**.

The score is a ratio measured inside the same runtime, on the same audio, with the
same decode settings:

```
speedup  =  audio-seconds per wall-second  (your submission)
            ------------------------------------------------
            audio-seconds per wall-second  (stock runtime)

wer_gain =  stock WER  -  your WER          (only when objective is `wer`)
```

Both arms load the checkpoint named in `models.json`, on one GPU, with the adapter
that `models.json` names. Only the submission plugin differs.

## Why the score is a runtime ratio and not a wall-clock benchmark

A ratio against a hand-rolled loop rewards work a real serving stack discards. It also
lets a submission "win" by loading a different, bigger model. Beating the *stock
runtime using the runtime's own machinery* keeps the comparison honest, and it is the
only number a deployment would see.

## What you submit

A plugin directory, not a new ASR system:

```
<attempt>/full/
  engine.py          module exposing build(cfg) -> object implementing EngineHooks
  manifest.json      what you changed, in one paragraph, machine-checkable
```

`engine.py` must expose:

```python
def build(cfg: dict) -> EngineHooks: ...
```

`EngineHooks` (see `harness/backends/base.py`) is the **only** extension point:

| hook | when it runs | what it may do |
|---|---|---|
| `prepare(runtime)` | once, after the runtime loaded the target | wrap / patch the runtime object |
| `preprocess(batch)` | before audio reaches the model | VAD, resampling, chunking, batching policy |
| `postprocess(hyp)` | after decoding | punctuation, ITN, rescoring with a local LM |
| `configure(cfg)` | before load | engine knobs the runtime exposes |

Only the keys listed in `models.json.target.tunable` may be changed by `configure`;
`build_runtime()` raises `PermissionError` on any other decode change, so a submission
cannot quietly reshape the measurement.

Everything else — the checkpoint, the tokenizer/vocab, the frontend feature extractor,
the device — is frozen by `models.json` and the adapter. The gate fingerprints the
loaded weights and rejects a submission whose fingerprint drifts beyond
`submission.weight_fingerprint_tolerance`, unless `submission.may_change_weights` is
true.

## The constraint is not optional

`task.yaml` declares the constraint. Default:

```
objective: speedup
constraint: {metric: wer, max_delta_abs: 0.003, confidence: 0.95, paired: true}
```

A speedup only counts if the **paired bootstrap upper confidence bound** of the WER
delta is at or below `max_delta_abs`. A run that is 3x faster and 0.5% worse scores
**0.0**, exactly like a gate failure. Report the delta and its interval, never a
point estimate alone: on a 2,620-utterance set one utterance is already ~0.04% WER.

Memory is measured per arm and reported as `components.memory`:
`after_load_mb` (RSS after the backend loaded), `peak_mb` (max of the sampled
transients during the workload and the final read), `delta_mb` (peak − after load).
The same block is part of the payload the agent reads. `constraint.max_peak_rss_mb`
is optional and defaults to `null`; when set, a submission whose peak exceeds it fails
the constraint exactly like a WER violation, with a reason such as
`memory 842.0MB > 800.0MB`.

The CPU policy lives under `measurement.cpu` and is applied once, before any backend
loads: `threads: <int>` sets `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS` only where the
caller has not already set them, and `affinity: "0-3"` (or `"0,2,4"`) calls
`os.sched_setaffinity` where available. `null` inherits the process default; failures
are recorded, not raised. The effective policy is reported as `components.cpu`
(`threads_requested`, `threads_applied`, `affinity_requested`, `affinity_effective`,
`notes`), and both arms run under the same policy.

## Exactness

You cannot silently change the emitted transcript set to buy speed. The gate compares
your transcripts against the stock arm on the *same* audio: identical output is the
expected case for a pure-systems submission. A submission that changes text is allowed
only when the task's objective is `wer`; otherwise the gate reports
`transcript_drift` and the score is capped at the stock arm.

## Harness

| file | what it does |
|---|---|
| `harness/benchmark.py` | **the score.** Both arms, each in its own process, returns the ratio plus the WER delta interval |
| `harness/check_submission.py` | cheap gate: plugin builds, hooks fire, fingerprint matches, output non-empty |
| `harness/_arm.py` | one arm, one process |
| `harness/instrument.py` | WER/CER, paired bootstrap, RTF and bucket statistics, RSS sampling, CPU policy |
| `harness/artifacts.py` | writes `plan.json`, `config.patch`, `report.md`, `repro.sh` for a passing full measurement |
| `harness/backends/` | runtime adapters. `dummy` exists only to self-test the instrument |
| `data/make_manifest.py` | builds `manifest.jsonl` from a directory of `*.wav` + `*.txt` |

```
python harness/check_submission.py --submission <attempt>/full   # minutes
python harness/benchmark.py        --submission <attempt>/full   # authoritative
python harness/benchmark.py --selftest --submission <attempt>/full   # no real audio
python harness/benchmark.py --submission <attempt>/full --proxy  # cheap, sets `promote`
python harness/benchmark.py --submission <attempt>/full --archive archive.jsonl
python harness/benchmark.py --submission <attempt>/full --artifacts <dir>  # full + passing only
```

Two signals exist so a step can be judged on *why* it moved, not only on a scalar:

* `components.profile_stock` / `profile_submission` — per-stage milliseconds and their
  share, when the adapter fills `Result.meta["profile_ms"]`. An adapter that does not
  fill it is still valid; the channel is then empty.
* `components.repeat_spread_*` — per-repeat throughputs and relative spread. A spread
  above 2% means the host is noisy and the ratio should not be believed yet.
* `--proxy` scores only `measurement.proxy_utterances` and sets `promote`; a full-fidelity
  slot should be spent only on promoted attempts. `--archive` appends to a Pareto
  archive, so the selector can pick parents off the frontier.

## Pointing this task at a different ASR target

Edit `models.json` only. Steps:

1. set `target.backend` to an adapter id under `harness/backends/` and
   `target.entrypoint` to its `module:factory`;
2. set `target.checkpoint` (a local path or a hub id) and the decode settings;
3. set `target.sample_rate`, `target.mode` (`offline` or `streaming`), and
   `target.max_audio_seconds`; for streaming also set `target.streaming.chunk_ms` /
   `lookahead_ms` and implement `transcribe_stream()` on the adapter;
4. re-measure the stock arm and record it in `stock_reference.json`;
5. do not touch `measurement:` — it is the instrument, and changing it makes scores
   incomparable across runs.

Constraint metric is `wer` or `cer` (`task.yaml`), and normalization is per-utterance:
a row whose `lang` is `zh`/`yue`/`ja`/`ko` is compared with the CJK mode, everything
else with the default. A mixed-language corpus therefore needs no separate task.

If the adapter for your runtime does not exist yet, write one. An adapter is ~100
lines: `load()` builds the runtime, `transcribe()` calls it, `fingerprint()` hashes
the loaded weights.

## Measurement notes that have already cost runs real time

* **One process per arm.** Two ASR runtimes rarely coexist on one card; interleaving
  produces false nulls.
* **Warm up before timing.** First inference JIT-compiles kernels and loads weights
  lazily. `measurement.warmup_utterances` is not optional.
* **Bucket by audio length.** Padding every clip to the model's fixed window hides the
  single largest real win on short audio; an un-bucketed mean will report a flat score.
* **Scoring runs with no network.** Every checkpoint must already be in the local
  cache and must load with local files only.
* **The stock arm is part of the instrument.** If your stock arm drifts >2% across
  repeats, stop and fix the instrument before believing any speedup.
