"""Starting point for a real backend. Copy this file to `<your_module>.py` and fill it in.

    cp task/harness/backends/_template.py task/harness/backends/whisper.py
    # then set models.json: target.entrypoint = "backends.whisper:build"

Rules this file is written to obey:
* The adapter may import any ASR library. The *submission* may not -- it only gets Hooks.
* Never time yourself. The harness measures the transcribe call; a self-reported number
  would be a score hack.
* Keep the process identical in both arms. No caching across arms, no warm start on
  arm 2, no background workers the stock arm does not have.
* Report only what you actually measured. An invented profile is worse than no profile.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

from .base import Request, Result


class TemplateRuntime:
    # Optional routing prior for the agent. One of PARADIGMS in base.py, or None.
    #   "AED"          Encoder-Decoder with cross-attention (Whisper, FireRedASR-AED)
    #   "SPEECH_LLM"   audio adapter into a decoder-only LLM (SLAM-ASR, Granite-speech)
    #   "TRANSDUCER"   RNN-T / TDT style joiner loops
    #   "NAR_CIF_CTC"  one-step parallel decode (Paraformer, SenseVoice)
    #   "HYBRID"       does not fit one row
    paradigm = None

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = None
        self._fingerprint: dict | None = None

    # ---------------------------------------------------------------- load
    def load(self, cfg: dict) -> None:
        """Build the model once per arm. Read paths from cfg["target"], e.g. checkpoint."""
        self.cfg = cfg
        target = cfg["target"]
        self.checkpoint = Path(target["checkpoint"])
        self.sample_rate = int(target.get("sample_rate") or 16000)

        # self.model = YourLibrary.load(str(self.checkpoint), device=target.get("device"))
        self.model = None
        if self.model is None:
            raise NotImplementedError(
                "template: build the model in load(); see models.json target.checkpoint"
            )

    # ----------------------------------------------------------- transcribe
    def transcribe(self, batch: list[Request]) -> list[Result]:
        """One batch, no internal timing, no cross-call state."""
        out: list[Result] = []
        for req in batch:
            if req.audio is None:
                raise ValueError(f"{req.id}: no audio; the manifest row needs an audio path")
            t_front = time.perf_counter()
            features = req.audio  # e.g. log-mel at self.sample_rate
            t_enc = time.perf_counter()
            text = self.model.transcribe(features)  # your call goes here
            t_dec = time.perf_counter()
            out.append(Result(
                id=req.id,
                text=text,
                meta={"profile_ms": {
                    # Split the three spans you timed above. kv_bridge_ms is optional and
                    # only meaningful when the model moves audio features across a cache
                    # boundary (cross-attention or an audio-prefix KV). Leave it out if
                    # you cannot measure it -- the harness then reports it as unreported
                    # rather than zero.
                    "frontend": (t_enc - t_front) * 1000.0,
                    "encoder": (t_dec - t_enc) * 1000.0,
                    "decoder": 0.0,
                }},
            ))
        return out

    # Optional, only when models.json sets target.mode: "streaming".
    # def transcribe_stream(self, batch: list[Request]) -> list[Result]:
    #     chunk_ms = self.cfg["target"]["streaming"]["chunk_ms"]
    #     lookahead_ms = self.cfg["target"]["streaming"]["lookahead_ms"]
    #     ...consume audio in chunk_ms steps, look ahead at most lookahead_ms...

    # ---------------------------------------------------------- fingerprint
    def fingerprint(self) -> dict:
        """Identity of the weights, computed once at load and cached.

        Both arms must produce the same fingerprint; that is what stops a submission
        from quietly loading a different checkpoint.
        """
        if self._fingerprint is not None:
            return self._fingerprint
        h = hashlib.sha256()
        shards = sorted(self.checkpoint.glob("*")) if self.checkpoint.is_dir() else [self.checkpoint]
        for shard in shards:
            if shard.is_file():
                h.update(shard.name.encode())
                h.update(shard.read_bytes())
        self._fingerprint = {"params": 0, "sha256": h.hexdigest()}  # fill params if known
        return self._fingerprint


def build(cfg: dict) -> TemplateRuntime:
    return TemplateRuntime(cfg)
