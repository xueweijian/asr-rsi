---
name: asr-methods
description: >-
  The lever taxonomy for making an ASR runtime faster or more accurate through the
  four hooks, with expected effect, cost, and the failure mode of each. Generic across
  backends: no model name appears here.
---

# What actually moves an ASR score

The hooks are the whole action space: `configure`, `preprocess`, `postprocess`,
`prepare`. Every lever below is one of those four. Gains are typical, not promised;
measure before believing.

## preprocess — audio policy (usually the largest cheap win)

| lever | typical | cost | failure mode |
|---|---|---|---|
| avoid fixed-window padding on short audio | 1.5–3x on short clips | needs a backend that accepts variable length | backend silently re-pads; verify with bucket numbers |
| VAD segmentation before the model | 1.3–2x throughput on long files | segment quality; boundary errors | cuts mid-word, WER rises; gate catches it |
| length bucketing for batching | 1.3–2x at batch>1 | latency jitter | tail latency explodes on a mixed queue |
| downmix, resample once, reuse features | 5–15% | correctness of resampling | double resampling changes transcripts |

Padding is the one to check first. It is invisible in a corpus mean and enormous on
short audio.

## configure — decode policy

| lever | typical | cost |
|---|---|---|
| beam width down (5→1) | 1.2–1.5x | WER +0.1–0.5% absolute |
| disable temperature fallback | 1.1–1.3x | WER rises on noisy audio |
| drop `condition_on_previous_text` | 1.1x | hallucination loop risk changes |
| shorter max output length | 1.05–1.2x | truncation on long utterances |

Only the keys in `models.json.target.tunable` may be overridden. Changing a frozen key
is an instrument change, not a submission.

## prepare — runtime wrapping

Legitimate: CUDA graphs for fixed shapes, fused kernels, a quantized copy of the target
weights **only when `may_change_weights` is true**, KV/state reuse across chunks,
async prefetch of the next batch.

Not legitimate: loading a different checkpoint, calling a hosted API, changing the
tokenizer, changing the frontend feature extractor.

## postprocess — text-side corrections

Rescoring with a local n-gram or small LM, ITN/punctuation fixes. These change the
transcript, so they are only scored when the objective is `wer`; under `speedup`
the gate reports `transcript_drift`.

## Do the decomposition before choosing

Measure, in one step: time in the frontend, time in the encoder, time in the decoder,
and time in text post-processing. Whichever slice dominates bounds what any single
lever can buy. A submission that improves a 10% slice by 2x is a 5% submission, and
that is a fine intermediate step if you say so.
