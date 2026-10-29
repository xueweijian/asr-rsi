"""sherpa-onnx NeMo FastConformer CTC adapter (10-language EU int8 model).

    SHERPA_MODEL_DIR=/path/to/sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k-int8

The model bundle is `model.int8.onnx` + `tokens.txt`. Point models.json at it, or set
the env var and leave `target.checkpoint` alone -- the env var wins, so the same repo
works on a machine that keeps the 127MB bundle outside the checkout.

Profile honesty: sherpa computes features and runs the encoder inside one
`decode_stream()` call. The span is reported as `encoder` because that is where the
time actually goes for a CTC model; `decoder` is omitted rather than faked, and the
harness reports the residual as `launch_bubble`.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import sherpa_onnx

from .base import Request, Result


class SherpaNemoCtcRuntime:
    paradigm = "NAR_CIF_CTC"  # one-step parallel CTC decode, no autoregressive loop

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = None
        self.checkpoint: Path | None = None
        self.sample_rate = 16000
        self._fingerprint: dict | None = None

    # ---------------------------------------------------------------- load
    def load(self, cfg: dict) -> None:
        self.cfg = cfg
        target = cfg["target"]
        root = os.environ.get("SHERPA_MODEL_DIR") or target.get("checkpoint")
        if not root:
            raise ValueError(
                "sherpa_nemo_ctc: set SHERPA_MODEL_DIR or models.json target.checkpoint"
            )
        self.checkpoint = Path(root)
        self.sample_rate = int(target.get("sample_rate") or 16000)

        onnx = self.checkpoint / "model.int8.onnx"
        tokens = self.checkpoint / "tokens.txt"
        for p in (onnx, tokens):
            if not p.is_file():
                raise FileNotFoundError(f"sherpa_nemo_ctc: missing {p}")

        decode = target.get("decode") or {}
        self.num_threads = int(decode.get("num_threads") or 2)
        self.decoding_method = str(decode.get("decoding_method") or "greedy_search")

        self.model = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=str(onnx),
            tokens=str(tokens),
            num_threads=self.num_threads,
            sample_rate=self.sample_rate,
            feature_dim=80,
            decoding_method=self.decoding_method,
        )

    # ----------------------------------------------------------- transcribe
    def _stream(self, req: Request):
        if req.audio is None:
            raise ValueError(f"{req.id}: no audio; the manifest row needs an audio path")
        stream = self.model.create_stream()
        stream.accept_waveform(self.sample_rate, np.asarray(req.audio, dtype=np.float32))
        return stream

    def transcribe(self, batch: list[Request]) -> list[Result]:
        """One batch. sherpa decodes the batch in one call, which is the lever's surface."""
        import time

        t0 = time.perf_counter()
        streams = [self._stream(req) for req in batch]
        t_front = time.perf_counter()
        if len(streams) > 1 and hasattr(self.model, "decode_streams"):
            self.model.decode_streams(streams)
        else:
            for s in streams:
                self.model.decode_stream(s)
        t_enc = time.perf_counter()

        per = (t_enc - t_front) / max(1, len(streams))
        out = [Result(
            id=req.id,
            text=s.result.text.strip(),
            meta={"profile_ms": {"frontend": (t_front - t0) * 1000.0, "encoder": per * 1000.0}},
        ) for req, s in zip(batch, streams)]
        return out

    # ---------------------------------------------------------- fingerprint
    def fingerprint(self) -> dict:
        if self._fingerprint is not None:
            return self._fingerprint
        assert self.checkpoint is not None
        h = hashlib.sha256()
        for shard in sorted(self.checkpoint.glob("*")):
            if shard.is_file():
                h.update(shard.name.encode())
                h.update(shard.read_bytes())
        self._fingerprint = {"params": 0, "sha256": h.hexdigest()}
        return self._fingerprint


def build(cfg: dict) -> SherpaNemoCtcRuntime:
    return SherpaNemoCtcRuntime(cfg)
