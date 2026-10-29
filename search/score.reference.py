"""REFERENCE scorer -- what SETUP would write from task/README.md.

A real run writes its OWN private/score.py, because the task is specified by its
contract, not by this implementation. Keep this file to (a) diff a run's scorer
against it and (b) drop it in deliberately when scoring must be held fixed.

Contract: score(output_dir) receives the leaf directory that directly holds
engine.py and manifest.json. Never append `full/` to it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

TASK = Path(os.environ.get("ASR_RSI_TASK") or Path(__file__).resolve().parent.parent / "task")
PROGRAM_PYTHON = os.environ.get("ASR_RSI_PYTHON", sys.executable)
CHECK_TIMEOUT_S = 600
BENCH_TIMEOUT_S = 6 * 3600


def _resolve(output_dir: str) -> Path:
    out = Path(output_dir)
    nested = out / "full"
    leaf = nested if (nested / "engine.py").is_file() else out
    if not (leaf / "engine.py").is_file():
        raise FileNotFoundError(f"engine.py not found in {leaf} (also checked {nested})")
    return leaf


def _run(script: str, args: list[str], timeout: int, tmp: Path) -> dict:
    out_json = tmp / f"{script}.json"
    cmd = [PROGRAM_PYTHON, str(TASK / "harness" / script), *args, "--json", str(out_json)]
    proc = subprocess.run(cmd, cwd=str(TASK), capture_output=True, text=True, timeout=timeout)
    if not out_json.is_file():
        tail = (proc.stdout + proc.stderr)[-4000:]
        raise RuntimeError(f"{script} wrote no JSON (exit {proc.returncode}):\n{tail}")
    return json.loads(out_json.read_text())


def score(output_dir: str) -> dict:
    leaf = _resolve(output_dir)
    import tempfile
    with tempfile.TemporaryDirectory(prefix="asr_rsi_score_") as tmp_s:
        tmp = Path(tmp_s)
        gate = _run("check_submission.py", ["--submission", str(leaf)], CHECK_TIMEOUT_S, tmp)
        if not gate.get("passed", False):
            return {"score": 0.0, "components": {"gate_passed": 0.0, "reasons": gate.get("reasons", [])}}
        bench = _run("benchmark.py", ["--submission", str(leaf)], BENCH_TIMEOUT_S, tmp)
    return {
        "score": float(bench.get("score", 0.0)),
        "components": {
            "gate_passed": 1.0 if bench.get("gate_passed") else 0.0,
            **(bench.get("components") or {}),
        },
    }


if __name__ == "__main__":
    print(json.dumps(score(sys.argv[1]), indent=2, ensure_ascii=False))
