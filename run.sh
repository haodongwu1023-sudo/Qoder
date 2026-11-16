#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install -q --upgrade pip
pip install -q -r backend/requirements.txt

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
  cp .env.example .env
fi

echo "Starting EchoBridge Web on http://127.0.0.1:8910 ..."
PYTHONPATH="$ROOT_DIR/backend" exec python -m uvicorn app.server:app --host 127.0.0.1 --port 8910
