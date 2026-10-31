#!/usr/bin/env bash
# One-click launcher for LoraResultLab. Binds to 127.0.0.1 only.
set -euo pipefail
cd "$(dirname "$0")"
export LORALAB_HOST=127.0.0.1
export LORALAB_PORT="${LORALAB_PORT:-8765}"
if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi
if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo "LoraResultLab — ComfyUI submission is OFF unless you enable it in Settings and confirm."
(sleep 1 && (xdg-open "http://127.0.0.1:${LORALAB_PORT}" >/dev/null 2>&1 || true)) &
exec python -m app
