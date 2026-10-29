"""Pareto archive: every measured attempt, kept, so the search can select on the
frontier instead of only on the top scalar score.

A scalar score throws away attempts that traded 3% speed for 0.05% WER -- and those
are exactly the parents a reviser should inherit from when the front is crowded.

    from archive import append, load, pareto
    append("archive.jsonl", record)      # one JSON per run, durable
    pareto(load("archive.jsonl"))        # non-dominated under (speedup↑, wer_delta↓)
"""
from __future__ import annotations

import json
import time
from pathlib import Path


def append(path: str | Path, record: dict) -> dict:
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def load(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def pareto(records: list[dict]) -> list[dict]:
    """Maximize speedup, minimize wer_delta. Failed attempts are excluded.

    O(n²) on purpose: n is a run's step count, and an explicit loop is auditable.
    """
    ok = [r for r in records if r.get("gate_passed") and r.get("constraint_ok")]
    front = []
    for i, a in enumerate(ok):
        dominated = False
        for j, b in enumerate(ok):
            if i == j:
                continue
            if (b["speedup"] >= a["speedup"] and b["wer_delta"] <= a["wer_delta"]
                    and (b["speedup"] > a["speedup"] or b["wer_delta"] < a["wer_delta"])):
                dominated = True
                break
        if not dominated:
            front.append(a)
    return sorted(front, key=lambda r: r["speedup"], reverse=True)


def best(records: list[dict], key: str = "score") -> dict | None:
    ok = [r for r in records if r.get("gate_passed") and r.get("constraint_ok")]
    return max(ok, key=lambda r: r.get(key, 0.0)) if ok else None


__all__ = ["append", "load", "pareto", "best"]
