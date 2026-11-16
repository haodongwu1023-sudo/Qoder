#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SRC_DIR="$ROOT_DIR/native/hal_driver"
OUT_BUNDLE="$ROOT_DIR/bin/VoiceCloneAudioDriver.driver"

rm -rf "$OUT_BUNDLE"
mkdir -p "$OUT_BUNDLE/Contents/MacOS"
cp "$SRC_DIR/Info.plist" "$OUT_BUNDLE/Contents/Info.plist"

clang -O2 -Wall -Wextra -bundle \
  -arch arm64 -mmacosx-version-min=13.0 \
  -framework CoreAudio -framework CoreFoundation \
  "$SRC_DIR/VoiceCloneAudioDriver.c" \
  -o "$OUT_BUNDLE/Contents/MacOS/VoiceCloneAudioDriver"

codesign --force --sign - "$OUT_BUNDLE"
echo "Built HAL Driver: $OUT_BUNDLE"
