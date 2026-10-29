---
name: asr-measurement
description: >-
  How to time an ASR arm so the number means something: one process per arm, GPU
  pinning, warm-up, repeat medians, bucketing by audio length, and the two
  definitions every submission must report. Read before timing anything.
---

# Timing an ASR arm honestly

A speedup is a ratio, and almost every wrong speedup comes from the denominator, the
clock, or the bookkeeping.

## One process per arm

Two ASR runtimes rarely fit on one card at once. Interleaving the arms inside one
process makes the second arm inherit the first arm's allocator state and clocks.
`benchmark.py` runs each arm as a separate `_arm.py` process on purpose.

## Warm-up is part of the measurement

First inference loads weights lazily and JIT-compiles kernels. On CUDA hosts this has
historically inflated a baseline by double digits and produced a false null.
`measurement.warmup_utterances` must be at least 50 and must use the same code path
as the timed loop.

## Bucket by audio length, then report

A single mean hides the largest real lever in ASR. Padding every clip to a fixed
window (30 s is the common one) costs almost nothing on a 25 s clip and roughly 10x
on a 2 s clip. If you only report the corpus mean, a submission that fixes short-audio
padding looks flat.

Report `short`, `medium`, `long` separately, and state which bucket moved.

## Define throughput once

```
throughput = total audio seconds / total wall seconds
rtf        = total wall seconds / total audio seconds
```

Both are computed outside the submission, in `_arm.py`. A submission that times
itself is measuring its own claim, not the system.

## Repeats and medians

Wall time on a shared host has a long right tail. Five repeats, median per utterance,
and a reported spread. If two repeats differ by more than 2%, the host is noisy; say
so rather than averaging the noise away.

## One device, chosen among visible devices

Pick by free memory among `torch.cuda.device_count()`, never by an `nvidia-smi` index.
`nvidia-smi` reports every physical card and ignores `CUDA_VISIBLE_DEVICES`; a pinned
process that trusts an nvidia-smi index will ask for a card it cannot see.
