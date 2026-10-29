"""Cheap gate: does the submission even qualify? Minutes, not the full benchmark.

Checks what can be checked without loading the target:
  * engine.py exists, imports, and returns a Hooks instance
  * no forbidden imports (network, hosted pipelines)
  * shipped weights stay inside the declared byte budget
  * manifest.json declares what changed

Fingerprint, timing and WER are the benchmark's job.

    python check_submission.py --submission <attempt>/full [--json OUT]
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
sys.path.insert(0, str(HERE))

from backends.base import Hooks  # noqa: E402
from instrument import write_json  # noqa: E402


def _imports(tree: ast.AST) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def _load_hooks(engine_py: Path, cfg: dict, entry: str):
    spec = importlib.util.spec_from_file_location("submission_engine_check", engine_py)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return getattr(mod, entry)(cfg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(TASK / "models.json"))
    ap.add_argument("--task", default=str(TASK / "task.yaml"))
    ap.add_argument("--submission", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    import yaml
    task = yaml.safe_load(Path(args.task).read_text())
    cfg = json.loads(Path(args.models).read_text())
    sub = Path(args.submission)
    entry = cfg["submission"]["entrypoint"]
    module_name, func_name = entry.split(":")

    checks: dict[str, bool] = {}
    reasons: list[str] = []

    engine_py = sub / module_name
    checks["engine_py_exists"] = engine_py.is_file()
    if not checks["engine_py_exists"]:
        reasons.append(f"missing {engine_py}")
        return _finish(args.json, checks, reasons)

    tree = ast.parse(engine_py.read_text())
    bad = sorted(_imports(tree) & set(task["submission"]["forbidden_imports"]))
    checks["no_forbidden_imports"] = not bad
    if bad:
        reasons.append(f"forbidden imports: {bad}")

    try:
        hooks = _load_hooks(engine_py, cfg, func_name)
        checks["build_returns_hooks"] = isinstance(hooks, Hooks)
        if not checks["build_returns_hooks"]:
            reasons.append(f"build() returned {type(hooks).__name__}, expected Hooks")
    except Exception as exc:  # noqa: BLE001 -- the reason must survive into the JSON
        checks["build_returns_hooks"] = False
        reasons.append(f"build() raised: {type(exc).__name__}: {exc}")

    manifest = sub / "manifest.json"
    checks["manifest_json"] = manifest.is_file()
    declared = 0
    if checks["manifest_json"]:
        try:
            data = json.loads(manifest.read_text())
            declared = int(data.get("extra_weights_bytes", 0))
            if not data.get("summary"):
                checks["manifest_json"] = False
                reasons.append("manifest.json has no summary")
        except Exception as exc:  # noqa: BLE001
            checks["manifest_json"] = False
            reasons.append(f"manifest.json invalid: {exc}")

    weights = 0
    for p in sub.rglob("*"):
        if p.is_file() and p.suffix in (".pt", ".bin", ".safetensors", ".onnx", ".npz", ".ckpt"):
            weights += p.stat().st_size
    budget = int(cfg["submission"].get("max_extra_weights_bytes", 0))
    checks["weights_within_budget"] = weights <= budget
    checks["weights_declared"] = weights == declared
    if not checks["weights_within_budget"]:
        reasons.append(f"shipped {weights} bytes of weights, budget {budget}")

    return _finish(args.json, checks, reasons, {"weights_bytes": weights, "declared": declared})


def _finish(path: str | None, checks: dict, reasons: list[str], extra: dict | None = None) -> int:
    out = {"passed": all(checks.values()) and not reasons,
           "checks": checks, "reasons": reasons, **(extra or {})}
    if path:
        write_json(path, out)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
