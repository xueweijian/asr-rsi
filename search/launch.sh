#!/usr/bin/env bash
# Launch a run of the ASR-RSI fork.
#   ./search/launch.sh [playground-dir]     default ~/aibuildai-playground
#
# The framework reads run.data_root at startup and freezes it as the run's README,
# so every edit to task/ must happen BEFORE launch. Editing task/ mid-run does
# nothing: the run already holds its frozen copy.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLAYGROUND="${1:-$HOME/aibuildai-playground}"
CONFIG_SRC="$REPO/search/run_config.yaml"
CONFIG_OUT="$PLAYGROUND/run_config.asr-rsi.yaml"

if [[ "$PLAYGROUND" == "$REPO"/* ]]; then
  echo "refusing a playground inside the repository (a run writes tens of GB)" >&2
  exit 2
fi

mkdir -p "$PLAYGROUND"
sed -e "s|__REPO__|$REPO|g" -e "s|__PLAYGROUND__|$PLAYGROUND|g" "$CONFIG_SRC" > "$CONFIG_OUT"

if grep -q '__REPO__\|__PLAYGROUND__' "$CONFIG_OUT"; then
  echo "placeholder substitution failed in $CONFIG_OUT" >&2
  exit 2
fi

# A run needs model access from the coding agent on THIS machine, plus either a
# subscription or the source-build skip flag. Missing model access does not fail:
# it retries forever at zero tokens while the GPU sits idle.
if [[ -z "${AIBUILDAI_SKIP_SUBSCRIPTION_CHECK:-}" ]]; then
  echo "note: set AIBUILDAI_SKIP_SUBSCRIPTION_CHECK=1 when running from a source build" >&2
fi

echo "config:  $CONFIG_OUT"
echo "task:    $REPO/task"
echo "arena:   $PLAYGROUND"
exec aibuildai run "$CONFIG_OUT"
