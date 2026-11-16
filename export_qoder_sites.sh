#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$ROOT_DIR/qoder_sites_bundle"
BACKEND_URL="${1:-}"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/samples"

cp "$ROOT_DIR/frontend/index.html" "$OUT_DIR/index.html"
cp "$ROOT_DIR/frontend/audio-worklet.js" "$OUT_DIR/audio-worklet.js"
cp "$ROOT_DIR/frontend/samples/"*.wav "$OUT_DIR/samples/"

cat > "$OUT_DIR/config.js" <<EOF
window.QODER_TRANSLATOR_CONFIG = {
  backendUrl: "${BACKEND_URL}"
};
EOF

echo "Exported Qoder Sites static bundle to: $OUT_DIR"
