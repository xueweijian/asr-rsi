"""The frozen extension surface: a runtime adapter and the only hooks a submission may use.

Nothing in this file is domain-specific. A submission that wants to do something this
surface cannot express is a submission for a different task, not a clever one.
"""
from __future__ import annotations

import copy
import importlib
import importlib.util
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass
class Request:
    id: str
    audio: np.ndarray | None
    sample_rate: int
    lang: str | None = None
    duration: float = 0.0


@dataclass
class Result:
    id: str
    text: str
    meta: dict = field(default_factory=dict)


class Runtime(Protocol):
    """What an adapter under harness/backends/ must provide.

    load()      builds the model; called once per arm, before any hook.
    transcribe() runs one batch; timing is measured OUTSIDE, in _arm.py, so a
                submission cannot report its own speed.
    fingerprint() hashes the loaded target weights; the gate compares arms.
    """

    def load(self, cfg: dict) -> None: ...
    def transcribe(self, batch: list[Request]) -> list[Result]: ...
    def fingerprint(self) -> dict: ...
    # Optional, required only when models.json sets target.mode: "streaming":
    #   def transcribe_stream(self, batch: list[Request]) -> list[Result]
    # It must consume audio in target.streaming.chunk_ms steps and may look ahead
    # at most target.streaming.lookahead_ms. Any other streaming semantics belongs
    # in the adapter, not in the submission.


PARADIGMS = ("AED", "SPEECH_LLM", "TRANSDUCER", "NAR_CIF_CTC", "HYBRID")


def validate_paradigm_value(value: Any, where: str = "target.paradigm") -> str | None:
    """Reject an unknown paradigm string loudly. null/None is always allowed."""
    if value is None:
        return None
    if not isinstance(value, str) or value not in PARADIGMS:
        raise ValueError(
            f"unknown {where} {value!r}; allowed: {list(PARADIGMS)} or null"
        )
    return value


def declared_paradigm(cfg: dict) -> str | None:
    """The paradigm models.json declares, validated. Absent == null."""
    return validate_paradigm_value((cfg.get("target") or {}).get("paradigm"))


def runtime_paradigm(runtime: Any) -> str | None:
    """Optional adapter declaration: an attribute or a zero-arg method."""
    value = getattr(runtime, "paradigm", None)
    if callable(value):
        value = value()
    return validate_paradigm_value(value, where=f"{type(runtime).__name__}.paradigm")


class Hooks:
    """Default is a no-op pass-through. A submission overrides any subset."""

    def configure(self, cfg: dict) -> dict:
        return cfg

    def prepare(self, runtime: Any) -> Any:
        return runtime

    def preprocess(self, batch: list[Request]) -> list[Request]:
        return batch

    def postprocess(self, results: list[Result]) -> list[Result]:
        return results


def load_audio(path: str, target_sr: int) -> np.ndarray:
    """Read a wav as float32 mono at target_sr. No soundfile dependency."""
    with wave.open(path, "rb") as wf:
        frames = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        sr = wf.getframerate()
        channels = wf.getnchannels()
    if channels > 1:
        frames = frames.reshape(-1, channels).mean(axis=1)
    audio = frames.astype(np.float32) / 32768.0
    if sr != target_sr:
        n = int(round(len(audio) * target_sr / sr))
        if n <= 1:
            return audio
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio
        ).astype(np.float32)
    return audio


def _import_path(name: str):
    return importlib.import_module(name)


def build_runtime(cfg: dict, submission_dir: str | Path | None) -> tuple[Any, Hooks, dict]:
    """Build one arm. `submission_dir=None` is the stock arm.

    Returns (runtime, hooks, effective_cfg). The stock arm always gets plain Hooks, so
    an unmodified plugin cannot accidentally disqualify the baseline.
    """
    declared_paradigm(cfg)      # fail loudly on an unknown target.paradigm
    adapter = _import_path(cfg["target"]["entrypoint"].split(":")[0])
    factory_name = cfg["target"]["entrypoint"].split(":")[1]
    runtime = getattr(adapter, factory_name)(cfg)
    hooks = Hooks()

    if submission_dir is not None:
        engine_py = Path(submission_dir) / cfg["submission"]["entrypoint"].split(":")[0]
        if not engine_py.is_file():
            raise FileNotFoundError(f"no {engine_py}; a submission must ship it")
        spec = importlib.util.spec_from_file_location("submission_engine", engine_py)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        builder = getattr(module, cfg["submission"]["entrypoint"].split(":")[1])
        hooks = builder(cfg)
        if not isinstance(hooks, Hooks):
            raise TypeError(f"engine.py:build returned {type(hooks).__name__}, not Hooks")

    frozen = copy.deepcopy(cfg)
    cfg = hooks.configure(copy.deepcopy(cfg))
    _guard_tunables(frozen, cfg)
    runtime.load(cfg)
    runtime = hooks.prepare(runtime)
    return runtime, hooks, cfg


def _guard_tunables(frozen: dict, proposed: dict) -> None:
    """A submission may override only target.tunable. Everything else is the instrument.

    Without this check `configure()` is a licence to change the checkpoint's decode
    policy and call the resulting win an engineering one.
    """
    allowed = set(proposed["target"].get("tunable") or [])
    before = frozen["target"].get("decode") or {}
    after = proposed["target"].get("decode") or {}
    illegal = sorted(k for k, v in before.items() if after.get(k, v) != v and k not in allowed)
    if illegal:
        raise PermissionError(
            f"submission changed frozen decode keys {illegal}; "
            f"only {sorted(allowed)} are tunable (models.json target.tunable)"
        )


__all__ = ["Request", "Result", "Runtime", "Hooks", "build_runtime", "declared_paradigm",
           "load_audio", "runtime_paradigm", "validate_paradigm_value"]
