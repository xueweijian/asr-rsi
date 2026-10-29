---
name: asr-performance-model
description: >-
  A back-of-envelope model of where ASR wall time goes, so a hypothesis can be
  rejected in one minute instead of one GPU-hour. Includes the fixed-window padding
  term, the encoder/decoder split, and the batch-size crossover.
---

# Where ASR time goes

Before a step, write the prediction. A lever you cannot predict is a lever you cannot
learn from, and a step that only produces a number teaches the next step nothing.

## Three terms

```
T(audio) ≈ T_frontend + T_encoder + T_decoder(output_len) + T_post
```

* **T_frontend** — resampling, feature extraction, VAD. Roughly linear in audio
  seconds, cheap per second, but it is a real fixed cost that batching can amortize.
* **T_encoder** — dominant for offline encoder-decoder systems at long audio. Quadratic
  in frames for full attention; linear in frames for convolutional/streaming. Frames
  scale with audio length, not with output length.
* **T_decoder** — proportional to output tokens × vocabulary × beam width. Dominant
  only when transcripts are long relative to audio (dense speech, many short segments).
* **T_post** — rescoring and formatting; usually small, occasionally not (a large LM
  rescore can exceed the acoustic model).

The split differs by architecture, which is exactly why the task is backend-agnostic:
measure the split on the target; do not import a split from another system.

## The padding term

If the backend pads every input to a fixed window `W`, then

```
T_encoder ∝ max(audio, W) / W          # not audio/W
```

For a corpus whose median utterance is 3 s and `W = 30`, nine tenths of encoder
compute is padding. This term alone explains most "short-audio" wins, and it is
invisible in a mean that mixes buckets.

## Batch crossover

Batching amortizes frontend and kernel launches, and increases per-request latency.
At batch `B`, throughput gain is sublinear once the GPU is saturated; past that point
latency only grows. Find the knee by measuring `B ∈ {1, 2, 4, 8, 16}`, not by assuming
8 is best.

## What the model is for

* reject a hypothesis whose predicted ceiling is below the measurement noise;
* decide whether a step targets a *rate* (fewer bytes moved) or a *cost per cycle*
  (cheaper verification);
* tell the next step which slice is still untouched.
