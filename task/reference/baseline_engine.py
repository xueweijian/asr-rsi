"""Reference submission: the pass-through. It must score ~1.00 and WER delta ~0.

If this file does not reproduce the stock arm, the instrument is broken -- not the
submission. Copy it as the starting point for a real attempt.
"""
from __future__ import annotations

from backends.base import Hooks


class PassThrough(Hooks):
    """No hooks. Included only so the plugin path is exercised."""


def build(cfg: dict) -> Hooks:
    return PassThrough()
