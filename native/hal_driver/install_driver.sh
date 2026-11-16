#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DRIVER_SRC="$ROOT_DIR/bin/VoiceCloneAudioDriver.driver"
HAL_DIR="/Library/Audio/Plug-Ins/HAL"
DRIVER_DST="$HAL_DIR/VoiceCloneAudioDriver.driver"

if [ ! -d "$DRIVER_SRC" ]; then
  "$ROOT_DIR/native/hal_driver/build_driver.sh"
fi

mkdir -p /Users/Shared/VoiceCloneLive/Audio
chmod 777 /Users/Shared/VoiceCloneLive /Users/Shared/VoiceCloneLive/Audio

if [ "$(id -u)" -eq 0 ]; then
  rm -rf "$DRIVER_DST"
  cp -R "$DRIVER_SRC" "$DRIVER_DST"
  chown -R root:wheel "$DRIVER_DST"
  killall -9 coreaudiod || true
else
  osascript -e "do shell script \"mkdir -p '/Users/Shared/VoiceCloneLive/Audio' && chmod 777 '/Users/Shared/VoiceCloneLive' '/Users/Shared/VoiceCloneLive/Audio' && rm -rf '$DRIVER_DST' && cp -R '$DRIVER_SRC' '$DRIVER_DST' && chown -R root:wheel '$DRIVER_DST' && killall -9 coreaudiod || true\" with administrator privileges"
fi
echo "Installed $DRIVER_DST"
