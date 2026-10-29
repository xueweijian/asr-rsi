"""Turn a LibriSpeech tree into `<stem>.wav` + `<stem>.txt` pairs.

    python data/prepare_librispeech.py <librispeech-root> <out-dir> --limit-chapters 3

LibriSpeech ships 16 kHz mono FLAC with one `.trans.txt` per chapter. The harness
reads plain 16 kHz mono wav and expects a sibling transcript, so both have to be
materialised here rather than inside a scored run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf


def chapters(root: Path, limit: int | None) -> list[Path]:
    dirs = sorted(p for p in root.rglob("*.trans.txt"))
    return dirs if limit is None else dirs[:limit]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--limit-chapters", type=int, default=None)
    ap.add_argument("--max-utts", type=int, default=None)
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for trans in chapters(root, args.limit_chapters):
        for line in trans.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            uid, _, text = line.partition(" ")
            flac = trans.parent / f"{uid}.flac"
            if not flac.is_file():
                continue
            audio, rate = sf.read(str(flac), dtype="int16", always_2d=False)
            if rate != 16000 or audio.ndim != 1:
                print(f"skip {uid}: {rate} Hz, ndim={audio.ndim}", file=sys.stderr)
                continue
            sf.write(str(out / f"{uid}.wav"), audio, rate, subtype="PCM_16")
            (out / f"{uid}.txt").write_text(text.strip() + "\n", encoding="utf-8")
            written += 1
            if args.max_utts is not None and written >= args.max_utts:
                print(f"{written} utterances from {len(chapters(root, args.limit_chapters))} chapters -> {out}")
                return 0
    print(f"{written} utterances -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
