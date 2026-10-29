"""Deliverable artifacts: the evidence a passing measurement leaves behind.

A report that only lives in stdout cannot be reproduced or audited. `export()` writes
four files next to a run: plan.json (machine-readable), config.patch (what the
submission changed vs the stock engine), report.md (human summary), repro.sh (the
single command that produced them).
"""
from __future__ import annotations

import difflib
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK = HERE.parent


def _stock_engine() -> tuple[Path | None, str | None]:
    """Prefer the reference copy; fall back to the example submission."""
    for candidate in (TASK / "reference" / "engine.py",
                      TASK / "reference" / "baseline_engine.py",
                      TASK / "example_submission" / "engine.py"):
        if candidate.is_file():
            return candidate, None
    return None, f"no stock engine under {TASK / 'reference'} or {TASK / 'example_submission'}"


def _engine_diff(submission_dir: Path) -> tuple[str | None, dict]:
    stock, why = _stock_engine()
    if stock is None:
        return None, {"config_patch_skipped": why}
    engine = submission_dir / "engine.py"
    if not engine.is_file():
        return None, {"config_patch_skipped": f"missing {engine}"}
    diff = "".join(difflib.unified_diff(
        stock.read_text().splitlines(keepends=True),
        engine.read_text().splitlines(keepends=True),
        fromfile=str(stock), tofile=str(engine),
    ))
    return diff, {"config_patch_from": str(stock), "config_patch_empty": not diff}


def _plan(report: dict, submission_dir: Path, patch_meta: dict) -> dict:
    c = report.get("components") or {}
    return {
        "task": report.get("task"),
        "score": report.get("score"),
        "gate": report.get("gate_passed"),
        "gates": report.get("gates"),
        "constraint": {
            "ok": c.get("constraint_ok"),
            "wer_ok": c.get("wer_constraint_ok"),
            "memory_ok": c.get("memory_constraint_ok"),
            "max_peak_rss_mb": c.get("max_peak_rss_mb"),
            "reason": report.get("reason"),
        },
        "components": {
            "speedup": c.get("speedup"),
            "speedup_latency": c.get("speedup_latency"),
            "wer_delta": c.get("wer_delta"),
            "wer_delta_ci": c.get("wer_delta_ci"),
            "buckets": (c.get("submission_throughput") or {}).get("buckets"),
            "profile": c.get("profile_submission"),
            "memory": c.get("memory"),
            "cpu": c.get("cpu"),
        },
        "submission_fingerprint": c.get("submission_fingerprint"),
        "submission_dir": str(submission_dir.resolve()),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": report.get("command"),
        **patch_meta,
    }


def _report_md(report: dict, patch_meta: dict) -> str:
    c = report.get("components") or {}
    lines = [f"# {report.get('task') or 'asr-rsi'} measurement", ""]
    lines.append(f"- **score**: {report.get('score')}")
    lines.append(f"- **gate**: {'passed' if report.get('gate_passed') else 'failed'}")
    reason = report.get("reason")
    verdict = "passed" if c.get("constraint_ok") else "failed"
    lines.append(f"- **constraint**: {verdict}" + (f" ({reason})" if reason else ""))
    ci = c.get("wer_delta_ci") or {}
    lines.append(f"- **WER delta**: {c.get('wer_delta')} "
                 f"(CI {ci.get('lo')} .. {ci.get('hi')}, mean {ci.get('mean')})")
    lines.append(f"- **speedup**: {c.get('speedup')} (latency arm {c.get('speedup_latency')})")

    buckets = (c.get("submission_throughput") or {}).get("buckets") or {}
    if buckets:
        lines += ["", "## Buckets", "", "| bucket | n | throughput |", "|---|---|---|"]
        for name, b in buckets.items():
            lines.append(f"| {name} | {b.get('n')} | {b.get('throughput')} |")

    profile = c.get("profile_submission") or {}
    if profile:
        lines += ["", "## Profile (submission)", "", "| stage | ms | share |", "|---|---|---|"]
        for stage, p in profile.items():
            lines.append(f"| {stage} | {p.get('ms')} | {p.get('share')} |")

    mem = c.get("memory") or {}
    if mem:
        lines += ["", "## Memory", "",
                  f"- after load: {mem.get('after_load_mb')} MB",
                  f"- peak: {mem.get('peak_mb')} MB",
                  f"- delta: {mem.get('delta_mb')} MB"]

    cpu = c.get("cpu") or {}
    if cpu:
        lines += ["", "## CPU", "",
                  f"- threads: requested {cpu.get('threads_requested')}, "
                  f"applied {cpu.get('threads_applied')}",
                  f"- affinity: requested {cpu.get('affinity_requested')}, "
                  f"effective {cpu.get('affinity_effective')}"]
        for note in cpu.get("notes") or []:
            lines.append(f"- note: {note}")

    if patch_meta.get("config_patch_from"):
        state = "empty (identical to stock)" if patch_meta.get("config_patch_empty") else "written"
        lines += ["", f"## Patch", "", f"- config.patch: {state} "
                  f"(vs {patch_meta['config_patch_from']})"]
    elif patch_meta.get("config_patch_skipped"):
        lines += ["", "## Patch", "", f"- skipped: {patch_meta['config_patch_skipped']}"]

    lines += ["", "## Reproduce", "", "```bash", str(report.get("command")), "```", ""]
    return "\n".join(lines)


def export(submission_dir: str | Path, report: dict, out_dir: str | Path) -> dict[str, str]:
    """Write plan.json, config.patch, report.md, repro.sh; return written paths."""
    submission_dir = Path(submission_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    diff, patch_meta = _engine_diff(submission_dir)
    plan = _plan(report, submission_dir, patch_meta)

    written: dict[str, str] = {}
    plan_path = out / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    written["plan.json"] = str(plan_path)

    if diff is not None:
        patch_path = out / "config.patch"
        patch_path.write_text(diff, encoding="utf-8")
        written["config.patch"] = str(patch_path)

    report_path = out / "report.md"
    report_path.write_text(_report_md(report, patch_meta), encoding="utf-8")
    written["report.md"] = str(report_path)

    repro_path = out / "repro.sh"
    repro_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + str(report.get("command")) + "\n",
        encoding="utf-8",
    )
    repro_path.chmod(0o755)
    written["repro.sh"] = str(repro_path)
    return written


__all__ = ["export"]
