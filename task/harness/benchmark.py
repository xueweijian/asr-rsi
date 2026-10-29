"""The authoritative score: stock vs submission, inside the same runtime.

Both arms run the SAME adapter, the SAME checkpoint, and the SAME frozen decode settings.
Only the submission plugin differs. Each arm is its own process because two ASR runtimes
rarely coexist on one card.

Prints one JSON object on stdout. Exit 0 means "measurement completed", not "good":
read the JSON's `score` and `constraint`.

    python benchmark.py --submission <attempt>/full [--json OUT] [--limit N]
    python benchmark.py --submission <attempt>/full --selftest     # no audio, no GPU
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
sys.path.insert(0, str(HERE))

from instrument import (  # noqa: E402
    apply_cpu_policy, corpus_error, load_manifest, paired_bootstrap, repeat_spread,
    summarize_arm, write_json,
)

REQUIRED_HOOKS = ("configure", "prepare", "preprocess", "postprocess")


def run_arm(models: Path, task: Path, manifest: Path, out: Path, submission: Path | None,
            batch: int, repeats: int, limit: int | None, log: Path) -> dict:
    cmd = [sys.executable, str(HERE / "_arm.py"), "--models", str(models), "--task", str(task),
           "--manifest", str(manifest), "--json", str(out), "--batch", str(batch),
           "--repeats", str(repeats)]
    if submission is not None:
        cmd += ["--submission", str(submission)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")          # scoring never has network
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    with log.open("w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0 or not out.is_file():
        tail = log.read_text()[-3000:]
        raise RuntimeError(f"arm failed (exit {proc.returncode}); see {log}:\n{tail}")
    return json.loads(out.read_text())


def median_rows(rows: list[dict]) -> list[dict]:
    """Collapse repeats to one row per utterance: median wall, first transcript."""
    by_id: dict[str, list[dict]] = {}
    for r in rows:
        by_id.setdefault(r["id"], []).append(r)
    out = []
    for rid, rs in by_id.items():
        first = rs[0]
        out.append({**first, "wall_ms": statistics.median(x["wall_ms"] for x in rs)})
    return out


def selftest_manifest(tmp: Path, n: int = 40) -> Path:
    from backends.dummy import synth_text
    rows = []
    durs = [1.0, 2.0, 4.5, 7.0, 12.0, 18.0, 25.0]
    for i in range(n):
        d = durs[i % len(durs)]
        rid = f"selftest-{i:03d}"
        rows.append({"id": rid, "audio": None, "duration": d, "lang": "en",
                     "text": synth_text(d, rid)})
    p = tmp / "selftest.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def _drift(a: list[dict], b: list[dict]) -> float:
    ta = {r["id"]: r["text"] for r in a}
    tb = {r["id"]: r["text"] for r in b}
    ids = sorted(set(ta) & set(tb))
    if not ids:
        return 0.0
    return sum(ta[i] != tb[i] for i in ids) / len(ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(TASK / "models.json"))
    ap.add_argument("--task", default=str(TASK / "task.yaml"))
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--proxy", action="store_true",
                   help="cheap fidelity: proxy_utterances subset; sets `promote`")
    ap.add_argument("--archive", default=None,
                   help="append this measurement to a Pareto archive (jsonl)")
    ap.add_argument("--artifacts", default=None,
                   help="export plan.json/config.patch/report.md/repro.sh here "
                        "(full-fidelity, passing runs only)")
    args = ap.parse_args()

    import yaml
    task = yaml.safe_load(Path(args.task).read_text())
    models = json.loads(Path(args.models).read_text())
    m = task["measurement"]
    cpu = apply_cpu_policy(m.get("cpu") or {})   # before any backend loads
    repeats = int(m["repeats"])
    batch_l = int(m["batch_size_latency"])
    batch_t = int(m["batch_size_throughput"])
    metric = str(task["constraint"]["metric"])          # wer | cer, per task.yaml
    default_norm = str(m.get("text_normalization", "lower"))
    cjk_norm = str(m.get("cjk_normalization", "plain"))

    with tempfile.TemporaryDirectory(prefix="asr_rsi_bench_") as tmp_s:
        tmp = Path(tmp_s)
        if args.selftest:
            manifest = selftest_manifest(tmp)
            limit = None
        else:
            manifest = Path(args.manifest or (TASK.parent / m["dev_manifest"]))
            limit = args.limit if args.limit is not None else m.get("max_utterances_dev")
            if args.proxy and args.limit is None:
                limit = int(m.get("proxy_utterances", 300))

        rows = load_manifest(manifest, limit)
        refs = [r["text"] for r in rows]
        # per-utterance normalization: a mixed zh/en corpus must not share one mode
        modes = [_norm_for(r.get("lang"), default_norm, cjk_norm) for r in rows]

        stock = run_arm(Path(args.models), Path(args.task), manifest, tmp / "stock.json",
                        None, batch_l, repeats, limit, tmp / "stock.log")
        # throughput arm: same two arms, batched, for the headline number
        stock_t = run_arm(Path(args.models), Path(args.task), manifest, tmp / "stock_t.json",
                          None, batch_t, max(1, repeats // 2), limit, tmp / "stock_t.log")
        sub = run_arm(Path(args.models), Path(args.task), manifest, tmp / "sub.json",
                      Path(args.submission), batch_l, repeats, limit, tmp / "sub.log")
        sub_t = run_arm(Path(args.models), Path(args.task), manifest, tmp / "sub_t.json",
                        Path(args.submission), batch_t, max(1, repeats // 2), limit, tmp / "sub_t.log")

    s_rows = median_rows(stock["rows"])
    u_rows = median_rows(sub["rows"])
    s_trows = median_rows(stock_t["rows"])
    u_trows = median_rows(sub_t["rows"])

    # --- gates -------------------------------------------------------------
    gates: dict[str, bool] = {}
    gates["fingerprint_match"] = (
        True if models["submission"].get("may_change_weights")
        else _fp_equal(stock["fingerprint"], sub["fingerprint"],
                       float(models["submission"]["weight_fingerprint_tolerance"]))
    )
    drift = _drift(s_rows, u_rows)
    gates["transcript_ok"] = (
        drift == 0.0
        or float(task["scoring"].get("transcript_drift_tolerance", 0.0)) >= drift
        or bool(task["scoring"]["allow_transcript_drift"])
    )
    gates["non_empty_output"] = all(r["text"].strip() for r in u_rows)
    gate_passed = all(gates.values())

    # --- metrics -----------------------------------------------------------
    stock_lat = summarize_arm(s_rows)
    sub_lat = summarize_arm(u_rows)
    stock_thr = summarize_arm(s_trows)
    sub_thr = summarize_arm(u_trows)

    stock_wer = corpus_error(refs, [r["text"] for r in s_rows], modes, metric)
    sub_wer = corpus_error(refs, [r["text"] for r in u_rows], modes, metric)
    ci = paired_bootstrap(refs, [r["text"] for r in s_rows], [r["text"] for r in u_rows],
                          modes, metric, int(task["constraint"]["bootstrap_samples"]),
                          float(task["constraint"]["confidence"]),
                          int(task["constraint"]["seed"]))

    speedup_lat = sub_lat["throughput"] / stock_lat["throughput"] if stock_lat["throughput"] else 0.0
    speedup_thr = sub_thr["throughput"] / stock_thr["throughput"] if stock_thr["throughput"] else 0.0
    speedup = speedup_thr                       # headline: batched throughput
    dwer = sub_wer - stock_wer
    wer_ok = ci.hi <= float(task["constraint"]["max_delta_abs"])

    # Memory gate: optional, only the submission arm's peak is gated. A null cap
    # leaves the measurement unchanged; a missing peak cannot exceed a cap.
    mem = sub.get("memory") or {}
    peak_mb = mem.get("peak_mb")
    mem_cap = task["constraint"].get("max_peak_rss_mb")
    memory_ok = mem_cap is None or peak_mb is None or float(peak_mb) <= float(mem_cap)
    limit_ok = wer_ok and memory_ok

    result = {
        "score": 0.0,
        "gate_passed": gate_passed,
        "gates": gates,
        "components": {
            "speedup": round(speedup, 4),
            "speedup_latency": round(speedup_lat, 4),
            "stock_wer": round(stock_wer, 6),
            "submission_wer": round(sub_wer, 6),
            "wer_delta": round(dwer, 6),
            "wer_delta_ci": {"mean": round(ci.mean, 6), "lo": round(ci.lo, 6),
                             "hi": round(ci.hi, 6), "samples": ci.samples, "seed": ci.seed},
            "wer_constraint_ok": wer_ok,
            "memory_constraint_ok": memory_ok,
            "max_peak_rss_mb": mem_cap,
            "constraint_ok": limit_ok,
            "transcript_drift": round(drift, 6),
            "stock_latency": stock_lat,
            "submission_latency": sub_lat,
            "stock_throughput": stock_thr,
            "submission_throughput": sub_thr,
            "profile_stock": stock_lat.get("profile", {}),
            "profile_submission": sub_lat.get("profile", {}),
            "memory": mem,
            "cpu": cpu,
            "repeat_spread_stock": repeat_spread(stock["rows"]),
            "repeat_spread_submission": repeat_spread(sub["rows"]),
            "stock_fingerprint": stock["fingerprint"],
            "submission_fingerprint": sub["fingerprint"],
            "fidelity": "proxy" if args.proxy else "full",
            "utterances_scored": stock_lat["utterances"],
        },
    }

    objective = task["objective"]
    if gate_passed and limit_ok:
        if objective in ("speedup", "rtf"):
            # raw ratio, floored at zero: below 1.0 is an honest failed step,
            # not a negative score, so the journal never rewards a regression
            result["score"] = round(max(0.0, speedup), 4)
        elif objective == "wer":
            result["score"] = round(-dwer, 6)   # positive means better than stock
        else:
            raise SystemExit(f"unknown objective {objective!r}")
    else:
        if not gate_passed:
            result["reason"] = "gate"
        else:
            reasons = []
            if not wer_ok:
                reasons.append("constraint")
            if not memory_ok:
                reasons.append(f"memory {float(peak_mb):.1f}MB > {float(mem_cap):.1f}MB")
            result["reason"] = "; ".join(reasons)

    # Multi-fidelity: a proxy step only asks "is this worth a full measurement?"
    result["fidelity"] = "proxy" if args.proxy else "full"
    if args.proxy:
        result["promote"] = bool(
            gate_passed and limit_ok
            and speedup >= float(m.get("proxy_min_speedup", 1.0))
        )
    else:
        result["promote"] = None

    if args.archive:
        from archive import append
        append(args.archive, {
            "fidelity": result["fidelity"],
            "score": result["score"],
            "speedup": round(speedup, 4),
            "wer_delta": round(dwer, 6),
            "wer_ci_hi": round(ci.hi, 6),
            "constraint_ok": limit_ok,
            "gate_passed": gate_passed,
            "utterances": stock_lat["utterances"],
            "peak_rss_mb": peak_mb,
            "after_load_mb": mem.get("after_load_mb"),
            "models": str(Path(args.models).name),
        })

    ref = models.get("stock_reference", {}).get("throughput")
    if ref and stock_thr["throughput"]:
        drift_pct = abs(stock_thr["throughput"] - ref) / ref
        result["components"]["stock_drift_vs_reference"] = round(drift_pct, 4)
        if drift_pct > 0.02:
            print(f"WARNING: stock arm drifted {drift_pct:.2%} from stock_reference; "
                  f"the instrument moved, not the submission", file=sys.stderr)

    # Deliverable artifacts: only a full-fidelity measurement that cleared the gate
    # and the constraint is a result worth reproducing.
    if args.artifacts:
        result["task"] = task.get("task")
        result["command"] = _command_line()
        if not args.proxy and gate_passed and limit_ok:
            from artifacts import export
            written = export(Path(args.submission), result, Path(args.artifacts))
            print("artifacts: " + ", ".join(f"{k}={v}" for k, v in written.items()),
                  file=sys.stderr)
        else:
            print("artifacts: skipped (proxy fidelity or failed run)", file=sys.stderr)

    if args.json:
        write_json(args.json, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _command_line() -> str:
    """The command that produced this report, with every path made absolute so the
    exported repro.sh does not depend on the caller's cwd."""
    import shlex
    path_flags = {"--models", "--task", "--manifest", "--submission", "--json",
                  "--archive", "--artifacts"}
    raw = sys.argv[1:]
    parts: list[str] = []
    i = 0
    while i < len(raw):
        if raw[i] in path_flags and i + 1 < len(raw):
            parts += [raw[i], str(Path(raw[i + 1]).expanduser().resolve())]
            i += 2
        else:
            parts.append(raw[i])
            i += 1
    return shlex.join([sys.executable, str(Path(__file__).resolve()), *parts])


def _norm_for(lang: str | None, default: str, cjk: str) -> str:
    """Chinese/Japanese/Korean text has no word boundaries and no case; CER on plain."""
    return cjk if (lang or "").lower() in ("zh", "yue", "ja", "ko") else default


def _fp_equal(a: dict, b: dict, tol: float) -> bool:
    keys = ("params", "sha256")
    if any(k in a and k in b for k in ("sha256",)):
        return a.get("sha256") == b.get("sha256")
    pa, pb = a.get("params"), b.get("params")
    if pa is None or pb is None:
        return True
    return abs(pa - pb) <= max(1, int(tol * pa))


if __name__ == "__main__":
    raise SystemExit(main())
