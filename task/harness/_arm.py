"""One arm, one process. Timing lives here, outside any submission code.

    python _arm.py --models ../models.json --task ../task.yaml \
                   --manifest <manifest.jsonl> --json out.json [--submission DIR] \
                   [--batch 1] [--repeats 5]

Prints nothing meaningful on stdout; the JSON file is the result. Exit code 0 means
"completed", not "good".
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
sys.path.insert(0, str(HERE))

from backends.base import (  # noqa: E402
    Request, build_runtime, declared_paradigm, load_audio, runtime_paradigm,
)
from instrument import (  # noqa: E402
    MemorySampler, bucket_for, load_manifest, rss_mb, write_json,
)


def _sync() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _rows_to_requests(rows: list[dict], cfg: dict) -> list[Request]:
    out = []
    for r in rows:
        audio = None
        if r.get("audio"):
            audio = load_audio(str(TASK / r["audio"]) if not Path(r["audio"]).is_absolute() else r["audio"],
                               cfg["target"]["sample_rate"])
        out.append(Request(id=r["id"], audio=audio, sample_rate=cfg["target"]["sample_rate"],
                           lang=r.get("lang"), duration=float(r.get("duration") or 0.0)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--submission", default=None)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--json", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = json.loads(Path(args.models).read_text())
    import yaml
    task = yaml.safe_load(Path(args.task).read_text())
    buckets = task["measurement"]["buckets"]
    warmup = int(task["measurement"]["warmup_utterances"])

    rows = load_manifest(args.manifest, args.limit)
    if not rows:
        raise SystemExit("empty manifest")
    bucket_by_id = {r["id"]: r.get("bucket") for r in rows}

    runtime, _hooks, _cfg = build_runtime(cfg, args.submission)
    fp = runtime.fingerprint()

    models_paradigm = declared_paradigm(cfg)
    adapter_paradigm = runtime_paradigm(runtime)
    paradigm = {
        "declared": models_paradigm if models_paradigm is not None else adapter_paradigm,
        "source": ("models.json" if models_paradigm is not None
                   else "adapter" if adapter_paradigm is not None else None),
    }

    # Memory: baseline after load/prepare, transients sampled during the timed
    # workload, final read after the loop. The sampler stops even on failure.
    after_load_mb = rss_mb()
    sampler = MemorySampler().start()
    try:
        reqs = _rows_to_requests(rows, cfg)
        for req in reqs[:warmup]:
            _sync()
            runtime.transcribe([req])
        _sync()

        out_rows = []
        for rep in range(args.repeats):
            for i in range(0, len(reqs), args.batch):
                batch = reqs[i:i + args.batch]
                _sync()
                t0 = time.perf_counter()
                results = _infer(runtime, batch, cfg)
                _sync()
                dt_ms = (time.perf_counter() - t0) * 1000.0
                if len(results) != len(batch):
                    raise RuntimeError(f"runtime returned {len(results)} results for {len(batch)} requests")
                total_dur = sum(r.duration for r in batch) or 1.0
                by_id = {r.id: r for r in batch}
                # batch wall time minus everything the adapter reported is host-side
                # launch/sync overhead; attribute both by audio share so totals hold.
                parts_total_ms = sum(
                    float(v) for res in results
                    for v in (res.meta.get("profile_ms") or {}).values()
                )
                bubble_ms = max(0.0, dt_ms - parts_total_ms)
                for res in results:
                    req = by_id[res.id]
                    share = (req.duration / total_dur) if req.duration else 1.0 / len(batch)
                    prof = {k: float(v) for k, v in (res.meta.get("profile_ms") or {}).items()}
                    out_prof = {k: v * share for k, v in prof.items()}
                    out_prof["kv_bridge_ms"] = out_prof.get("kv_bridge_ms", 0.0)
                    out_prof["launch_bubble_ms"] = bubble_ms * share
                    out_rows.append({
                        "id": req.id, "repeat": rep, "batch": len(batch),
                        "audio_s": req.duration,
                        "wall_ms": dt_ms * share,
                        "profile_ms": out_prof,
                        "kv_bridge_reported": "kv_bridge_ms" in prof,
                        "text": res.text,
                        "bucket": bucket_for(bucket_by_id.get(req.id), req.duration, buckets),
                    })
    finally:
        transient_peak = sampler.stop()

    peak_mb = max(after_load_mb, rss_mb(), transient_peak or 0.0)
    memory = {
        "after_load_mb": round(after_load_mb, 1),
        "peak_mb": round(peak_mb, 1),
        "delta_mb": round(peak_mb - after_load_mb, 1),
    }

    write_json(args.json, {
        "rows": out_rows,
        "fingerprint": fp,
        "submission": bool(args.submission),
        "batch": args.batch,
        "repeats": args.repeats,
        "memory": memory,
        "paradigm": paradigm,
        "git_sha": _git_sha(),
    })
    return 0


def _infer(runtime, batch: list[Request], cfg: dict) -> list:
    """Offline and streaming are different contracts; the adapter owns the difference."""
    if cfg["target"].get("mode") == "streaming":
        fn = getattr(runtime, "transcribe_stream", None)
        if fn is None:
            raise RuntimeError(
                "models.json sets target.mode: streaming but the adapter has no "
                "transcribe_stream(); see harness/backends/base.py"
            )
        return fn(batch)
    return runtime.transcribe(batch)


def _git_sha() -> str:
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(TASK),
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
