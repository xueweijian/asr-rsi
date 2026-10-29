"""Instrument self-test backend. NOT a real ASR system and never a baseline.

It exists so the harness can be exercised end to end (timing, bucketing, WER plumbing,
gate, bootstrap) on a laptop with no GPU, no checkpoint and no audio. Its transcripts
are deterministic nonsense; any score measured with `backend: dummy` is meaningless.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np

from .base import Request, Result


def synth_text(seconds: float, seed: str = "") -> str:
    """Deterministic pseudo-transcript. Also used by benchmark --selftest as the ref."""
    n = max(1, int(round(seconds * 2.0)))
    h = hashlib.sha256(f"{seed}:{n}".encode()).digest()
    vocab = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    return " ".join(vocab[h[i % len(h)] % len(vocab)] for i in range(n))


class DummyRuntime:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._loaded = False

    def load(self, cfg: dict) -> None:
        self.cfg = cfg
        # simulated load latency so warm-up has something to absorb
        time.sleep(0.02)
        self._loaded = True

    def transcribe(self, batch: list[Request]) -> list[Result]:
        out = []
        for req in batch:
            dur = req.duration or (len(req.audio) / req.sample_rate if req.audio is not None else 1.0)
            # "compute" proportional to duration, so bucketing and RTF plumbing are
            # exercised with real wall time. Small enough to stay deterministic.
            work = np.ones((max(1, int(dur * 200)), 8), dtype=np.float32) @ np.ones((8, 8), dtype=np.float32)
            budget_s = 0.0004 * max(1.0, dur)
            t = time.perf_counter()
            res = req.id
            mixer = np.full((8, 8), 1.0 / 8.0, dtype=np.float32)
            while (time.perf_counter() - t) < budget_s:
                work = work @ mixer
            elapsed = (time.perf_counter() - t) * 1000.0
            # a plausible cost decomposition so the profile channel is exercised
            meta = {"profile_ms": {"frontend": 0.10 * elapsed, "encoder": 0.70 * elapsed,
                                   "decoder": 0.20 * elapsed}}
            out.append(Result(id=res, text=synth_text(dur, req.id), meta=meta))
        return out

    def fingerprint(self) -> dict:
        return {"params": 0, "sha256": "dummy", "note": "instrument self-test"}


def build(cfg: dict) -> DummyRuntime:
    return DummyRuntime(cfg)
