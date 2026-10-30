"""Submission: decode a multi-request batch as concurrent single-stream decodes, with
the batch dispatched longest-first so the four cores stay busy to the end of the batch.

Why this is the lever
---------------------
The adapter's batched path (`OfflineRecognizer.decode_streams`) pads every stream in
a batch up to the longest utterance and runs one padded forward pass. Measured on
this host that path costs *exactly* the sequential cost times the padding ratio
(batch-max / mean length): it shares no compute with the other streams and buys
nothing but frames the model should never have seen.

    batch of 8, manifest order      pad ratio   batched / sequential
    1272-135031-0000..0007          1.66            1.59
    1462-170138-0015..0022          1.97            1.83
    1462-170142-0027..0034          1.37            1.25

The adapter already carries the cheaper path: when a batch has one stream it calls
`decode_stream` per request. This plugin selects that path for every batch size and
runs the members of a batch concurrently, because ORT releases the GIL around
`Run` while the session is configured with intra-op threads.

Step 4 (this step) adds one bounded change on top of step 3's engine: **the batch is
dispatched to the pool longest-first**.

Why ordering is a lever at all
------------------------------
`ThreadPoolExecutor.map` feeds the work queue in list order and a freed worker takes
the next queued task, so the batch is *list scheduled*: the makespan is the load of
the busiest worker, not the batch total divided by the worker count. The dev manifest
is uneven inside a batch of 8 by up to 8x (1.74 s to 16.35 s overall; a single batch
spans e.g. 2.3 s .. 14.6 s), so list order leaves the last worker finishing long
after the others. Cost is proportional to audio duration, which probe12 measured
directly: a serial single-stream decode at num_threads=1 fits `13.4 ms + 33.9 ms per
audio-second` over the first eight dev utterances (7.15 s mean duration), i.e. the
per-call fixed part is ~5% and the rest tracks the audio.

Sorting the batch by descending duration before submitting makes the greedy
dispatch equivalent to LPT (longest processing time first), the classic 4-machine
schedule. Measured on all 119 dev utterances, 15 manifest-order batches, 2 timed
repeats after a full warm pass (probe13):

    dispatch order      ms/batch   audio-s per wall-s
    manifest (list)      738.4        55.71
    longest first        703.3        58.49
    shortest first       762.2        53.98

The effect is monotone in the order, which is what makes it a scheduling result
rather than noise: shortest-first is worse than manifest order, longest-first is
better. Pool size and thread count were re-swept with the ordering in place and
4 workers at num_threads=1 still wins (desc t1w6 56.27, desc t1w8 55.17,
desc t2w4 54.22), so nothing is traded away for the mean.

This host is 2 physical cores with 4 hyperthreads (lscpu: 2 cores/socket, 2 threads
per core). Four concurrent single-thread decodes therefore deliver ~2x a single
decode and no more; the ordering change is what keeps those two cores fed for the
whole batch instead of idling the last worker.

Nothing about the model, the weights, the decode method or the audio changes, and
the order of the *decodes* does not change any decode: each utterance still goes
through the same single-stream `decode_stream` on the same session, so its
transcript is a function of its own audio alone. probe13 compared the transcript of
every one of the 119 dev utterances between list and longest-first dispatch: 0
differences. The batch-1 arm -- the arm the instrument gates transcript drift and
WER on -- is unchanged and still takes the identical serial single-stream call the
stock arm makes.

Profile honesty: each task reports its own frontend and decode spans; because the
frontend of later requests overlaps the decode of earlier ones, the batch totals of
the two parts can exceed the batch wall time. The reported parts plus the arm's
launch_bubble residual are exactly the work the machine did. The adapter's own
convention reports only one stream's share of a padded decode and leaves a large
artificial bubble; this plugin reports the batch frontend and the batch decode total
on every result.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from backends.base import Hooks, Request, Result

# 4 logical CPUs on 2 physical cores. At num_threads=1 one pool worker is one logical
# CPU, so 4 workers fill the machine with no oversubscription; the worker sweep
# (probe13, with longest-first ordering) is flat-to-worse beyond 4.
_WORKERS = 4

# One intra-op thread per decode: 4 concurrent decodes x 1 thread = 4 threads on 4
# logical CPUs. Tunable via configure(); batch-1 text identity vs the stock 2-thread
# session was verified over all 119 dev utterances (probe10).
_NUM_THREADS = 1


def _cost(req: Request) -> float:
    """Decode cost is proportional to audio duration (probe12: 33.9 ms/audio-s)."""
    if req.duration and req.duration > 0:
        return float(req.duration)
    audio = req.audio
    if audio is None:
        return 0.0
    sr = getattr(req, "sample_rate", 16000) or 16000
    return len(audio) / float(sr)


class _ConcurrentSingleStreamHooks(Hooks):
    """Replace the runtime's `transcribe` with a no-padding, fully-occupied policy."""

    def configure(self, cfg: dict) -> dict:
        # num_threads is a tunable key (models.json target.tunable). Decode method,
        # checkpoint, tokenizer and audio are untouched.
        cfg.setdefault("target", {}).setdefault("decode", {})["num_threads"] = _NUM_THREADS
        return cfg

    def prepare(self, runtime):
        model = runtime.model
        sample_rate = getattr(runtime, "sample_rate", 16000)
        make_stream = getattr(runtime, "_stream", None)
        executor = ThreadPoolExecutor(max_workers=_WORKERS)

        def _new_stream(req: Request):
            if make_stream is not None:
                return make_stream(req)
            stream = model.create_stream()
            stream.accept_waveform(sample_rate, req.audio.astype("float32", copy=False))
            return stream

        def _build_and_decode(req: Request):
            t0 = time.perf_counter()
            stream = _new_stream(req)
            t1 = time.perf_counter()
            model.decode_stream(stream)
            t2 = time.perf_counter()
            return stream, (t1 - t0) * 1000.0, (t2 - t1) * 1000.0

        def transcribe(batch: list[Request]) -> list[Result]:
            t0 = time.perf_counter()
            if len(batch) > 1:
                # Longest first: the pool dispatches greedily, so submitting in
                # descending cost makes the schedule LPT and stops the last worker
                # idling while the longest utterances run at the end. Results come
                # back in submission order and are put back into the request order;
                # each utterance's decode is untouched by the ordering.
                order = sorted(range(len(batch)), key=lambda i: -_cost(batch[i]))
                done = list(executor.map(lambda i: _build_and_decode(batch[i]), order))
                finished = [None] * len(batch)
                for pos, i in enumerate(order):
                    finished[i] = done[pos]
            else:
                # The gated arm's code path: the identical serial single-stream call
                # the stock arm makes, byte for byte.
                stream = _new_stream(batch[0])
                t1 = time.perf_counter()
                model.decode_stream(stream)
                t2 = time.perf_counter()
                finished = [(stream, (t1 - t0) * 1000.0, (t2 - t1) * 1000.0)]
            t3 = time.perf_counter()

            front_ms = sum(d[1] for d in finished)
            dec_ms = sum(d[2] for d in finished)
            share = 1.0 / len(finished)
            return [Result(
                id=req.id,
                text=stream.result.text.strip(),
                meta={"profile_ms": {"frontend": front_ms * share,
                                     "encoder": dec_ms * share}},
            ) for req, (stream, _, _) in zip(batch, finished)]

        runtime.transcribe = transcribe
        return runtime


def build(cfg: dict) -> Hooks:
    return _ConcurrentSingleStreamHooks()
