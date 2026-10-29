"""Build manifest.jsonl from a directory of audio + transcript pairs.

    python data/make_manifest.py <corpus-dir> --out data/dev_manifest.jsonl \
        --audio-ext .wav --text-ext .txt --lang en --split dev

Layout: every `<stem>.wav` with a sibling `<stem>.txt` becomes one row. Nothing here
knows which model will consume the audio.

Carve the hidden split by hashing the id, not by taking "the last 20%": a sorted tail
leaks speaker and length distribution, and a submission can learn it across steps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import wave
from pathlib import Path


def duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def bucket(uid: str, salt: str, hidden_fraction: float) -> str:
    h = int(hashlib.sha256(f"{salt}:{uid}".encode()).hexdigest()[:8], 16)
    return "hidden" if (h / 0xFFFFFFFF) < hidden_fraction else "dev"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--out", required=True)
    ap.add_argument("--audio-ext", default=".wav")
    ap.add_argument("--text-ext", default=".txt")
    ap.add_argument("--lang", default=None)
    ap.add_argument("--split", default="dev", choices=["dev", "hidden", "all"])
    ap.add_argument("--bucket", default=None,
                    help="explicit bucket label for every row (e.g. noise); the scorer "
                         "uses it instead of the duration-derived bucket")
    ap.add_argument("--hidden-fraction", type=float, default=0.2)
    ap.add_argument("--salt", default="asr-rsi-2026")
    ap.add_argument("--root-relative-to", default=None)
    args = ap.parse_args()

    corpus = Path(args.corpus)
    out = Path(args.out)
    rows = []
    for audio in sorted(corpus.rglob(f"*{args.audio_ext}")):
        text = audio.with_suffix(args.text_ext)
        if not text.is_file():
            continue
        uid = audio.stem
        part = bucket(uid, args.salt, args.hidden_fraction) if args.split != "all" else "all"
        if part != args.split and args.split != "all":
            continue
        rel = str(audio.relative_to(Path(args.root_relative_to).resolve())) if args.root_relative_to else str(audio)
        rows.append({
            "id": uid,
            "audio": rel,
            "text": text.read_text(encoding="utf-8").strip(),
            "duration": round(duration(audio), 3),
            "lang": args.lang,
            **({"bucket": args.bucket} if args.bucket else {}),
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    total = sum(r["duration"] for r in rows)
    print(f"{len(rows)} rows -> {out}  ({total / 60:.1f} min audio)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
