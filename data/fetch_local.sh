#!/usr/bin/env bash
# Everything a local run needs on disk, minus the model itself.
#   ./data/fetch_local.sh
# Requires: python3 with soundfile (pip install soundfile), curl, tar.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
MODELS=${ASR_MODELS_DIR:-$HOME/asr-models}
DATA=${ASR_DATA_DIR:-$HOME/asr-data}
NAME=sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k-int8

mkdir -p "$MODELS" "$DATA"

if [ ! -f "$MODELS/$NAME/model.int8.onnx" ]; then
  echo "fetching model bundle (127 MB)"
  curl -sSL --retry 3 -o "/tmp/$NAME.tar.bz2" \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$NAME.tar.bz2"
  tar -xjf "/tmp/$NAME.tar.bz2" -C "$MODELS"
else
  echo "model already at $MODELS/$NAME"
fi

if [ ! -d "$DATA/LibriSpeech/dev-clean" ]; then
  echo "fetching LibriSpeech dev-clean (338 MB)"
  curl -sSL --retry 3 -o /tmp/dev-clean.tar.gz \
    https://www.openslr.org/resources/12/dev-clean.tar.gz
  tar -xzf /tmp/dev-clean.tar.gz -C "$DATA"
else
  echo "corpus already at $DATA/LibriSpeech/dev-clean"
fi

python3 "$REPO/data/prepare_librispeech.py" "$DATA/LibriSpeech/dev-clean" "$DATA/corpus" \
  --limit-chapters 6 --max-utts 260

python3 "$REPO/data/make_manifest.py" "$DATA/corpus" \
  --out "$REPO/data/dev_manifest.jsonl" --split dev --lang en
python3 "$REPO/data/make_manifest.py" "$DATA/corpus" \
  --out "$REPO/data/hidden_manifest.jsonl" --split hidden --lang en
wc -l "$REPO/data/dev_manifest.jsonl" "$REPO/data/hidden_manifest.jsonl"

echo
echo "next:"
echo "  SHERPA_MODEL_DIR=$MODELS/$NAME python task/harness/benchmark.py \\"
echo "    --submission task/example_submission --limit 60 --archive runs/jsonl"
