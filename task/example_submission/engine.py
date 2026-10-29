"""Example submission: a complete, working pass-through attempt.

Copy this directory to start an attempt. `manifest.json` must describe what you changed;
`extra_weights_bytes` must equal the byte size of any new weights you ship.
"""
from __future__ import annotations

from backends.base import Hooks


class PassThrough(Hooks):
    pass


def build(cfg: dict) -> Hooks:
    return PassThrough()
