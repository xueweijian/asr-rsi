"""Instrument: the metrics and the statistics. No model, no runtime, no I/O policy.

Everything here is deliberately boring and dependency-light (stdlib + numpy), because a
metric you cannot re-derive by hand is a metric that will silently drift.

Read `skills/asr-measurement/SKILL.md` before changing anything in this file.
"""
from __future__ import annotations

import json
import math
import random
import re
import statistics
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path

_CJK = re.compile(r"\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af")
_PUNCT = re.compile(r"[^\w\s\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def normalize(text: str, mode: str = "lower") -> str:
    """Normalize a transcript. `mode` is declarative so two runs are comparable.

    lower  -- NFKC, strip punctuation, collapse whitespace, lowercase (latin scripts)
    plain  -- NFKC + collapse whitespace only (CJK, where case and spacing are noise)
    raw    -- no normalization; use only when the task is punctuation-aware
    """
    if mode == "raw":
        return text
    text = unicodedata.normalize("NFKC", text).strip()
    if mode == "lower":
        text = _PUNCT.sub(" ", text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokens(text: str) -> list[str]:
    # CJK has no spaces; fall back to characters so CER stays meaningful.
    if _CJK.search(text) and not re.search(r"[A-Za-z]", text):
        return list(text.replace(" ", ""))
    return text.split()


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    """Levenshtein over token lists. O(len(ref) * len(hyp)) with two rows."""
    if not ref:
        return len(hyp)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def utterance_error(ref: str, hyp: str, mode: str = "lower", metric: str = "wer") -> tuple[float, int]:
    """Per-utterance error and its denominator. Returned separately so the
    bootstrap can pool counts, not ratios."""
    r = normalize(ref, mode)
    h = normalize(hyp, mode)
    rt, ht = _tokens(r), _tokens(h)
    denom = len(rt) if metric == "wer" else max(len(r), 1)
    dist = edit_distance(rt, ht) if metric == "wer" else edit_distance(list(r), list(h))
    return (dist / denom if denom else float(not ht)), denom


def _mode_at(mode: str | list[str], i: int) -> str:
    """Per-utterance normalization: a mixed-language corpus must not share one mode."""
    return mode[i] if isinstance(mode, list) else mode


def corpus_error(refs: list[str], hyps: list[str], mode: str | list[str], metric: str) -> float:
    dist = 0
    denom = 0
    for i, (ref, hyp) in enumerate(zip(refs, hyps)):
        e, d = utterance_error(ref, hyp, _mode_at(mode, i), metric)
        dist += e * d
        denom += d
    return dist / denom if denom else 0.0


@dataclass
class Interval:
    mean: float
    lo: float
    hi: float
    samples: int
    seed: int


def paired_bootstrap(
    refs: list[str],
    hyps_a: list[str],
    hyps_b: list[str],
    mode: str | list[str] = "lower",
    metric: str = "wer",
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Paired bootstrap over utterances of (error_a - error_b).

    Paired, because both arms saw the same audio in the same order. Resampling arms
    independently inflates the interval and hides real regressions.
    """
    diffs = []
    for i, (r, a, b) in enumerate(zip(refs, hyps_a, hyps_b)):
        m = _mode_at(mode, i)
        ea, _ = utterance_error(r, a, m, metric)
        eb, _ = utterance_error(r, b, m, metric)
        diffs.append(ea - eb)
    n = len(diffs)
    if n == 0:
        return Interval(0.0, 0.0, 0.0, samples, seed)

    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = means[max(0, int(alpha * samples) - 1)]
    hi = means[min(samples - 1, int((1.0 - alpha) * samples))]
    return Interval(sum(diffs) / n, lo, hi, samples, seed)


def bucket_of(duration: float, buckets: dict[str, list[float]]) -> str:
    for name, (lo, hi) in buckets.items():
        if lo <= duration < hi:
            return name
    return "long"


def throughput(audio_seconds: float, wall_seconds: float) -> float:
    if wall_seconds <= 0:
        return 0.0
    return audio_seconds / wall_seconds


def summarize_arm(rows: list[dict]) -> dict:
    """Aggregate one arm. `rows` carry per-utterance wall_ms and audio_s."""
    audio = sum(r["audio_s"] for r in rows)
    wall = sum(r["wall_ms"] for r in rows) / 1000.0
    lat = sorted(r["wall_ms"] for r in rows)
    p = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))] if lat else 0.0
    buckets: dict[str, dict] = {}
    for r in rows:
        b = buckets.setdefault(r["bucket"], {"audio_s": 0.0, "wall_s": 0.0, "n": 0})
        b["audio_s"] += r["audio_s"]
        b["wall_s"] += r["wall_ms"] / 1000.0
        b["n"] += 1
    for b in buckets.values():
        b["throughput"] = throughput(b["audio_s"], b["wall_s"])
    prof: dict[str, float] = {}
    for r in rows:
        for k, v in (r.get("profile_ms") or {}).items():
            prof[k] = prof.get(k, 0.0) + float(v)
    total_prof = sum(prof.values())
    profile = (
        {k: {"ms": round(v, 1), "share": round(v / total_prof, 4)}
         for k, v in sorted(prof.items(), key=lambda kv: -kv[1])}
        if total_prof else {}
    )
    return {
        "utterances": len(rows),
        "audio_seconds": round(audio, 3),
        "wall_seconds": round(wall, 3),
        "throughput": round(throughput(audio, wall), 4),
        "rtf": round(wall / audio, 4) if audio else 0.0,
        "latency_ms": {"p50": round(p(0.5), 2), "p90": round(p(0.9), 2), "max": round(p(1.0), 2)},
        "profile": profile,
        "buckets": {
            k: {"n": v["n"], "throughput": round(v["throughput"], 4)}
            for k, v in sorted(buckets.items())
        },
    }


def repeat_spread(rows: list[dict]) -> dict | None:
    """Per-repeat throughputs and their relative spread. A wide spread means the
    host is noisy and the score is not yet trustworthy -- say so instead of averaging."""
    by_rep: dict[int, list[float]] = {}
    for r in rows:
        k = int(r.get("repeat", 0))
        cur = by_rep.setdefault(k, [0.0, 0.0])
        cur[0] += r["audio_s"]
        cur[1] += r["wall_ms"] / 1000.0
    tputs = [a / w for a, w in by_rep.values() if w > 0]
    if len(tputs) < 2:
        return None
    med = statistics.median(tputs)
    return {
        "repeats": len(tputs),
        "throughput": [round(t, 1) for t in tputs],
        "spread": round((max(tputs) - min(tputs)) / med, 4) if med else 0.0,
    }


def load_manifest(path: str | Path, limit: int | None = None) -> list[dict]:
    rows = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(json.loads(line))
    if limit is not None:
        rows = rows[:limit]
    for r in rows:
        r.setdefault("duration", 0.0)
        r.setdefault("lang", None)
    return rows


def write_json(path: str | Path, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def geometric_mean(values: list[float]) -> float:
    pos = [v for v in values if v > 0]
    if not pos:
        return 0.0
    return math.exp(sum(math.log(v) for v in pos) / len(pos))


__all__ = [
    "Interval", "asdict", "bucket_of", "corpus_error", "geometric_mean",
    "load_manifest", "normalize", "paired_bootstrap", "repeat_spread",
    "summarize_arm", "throughput", "utterance_error", "write_json",
]
