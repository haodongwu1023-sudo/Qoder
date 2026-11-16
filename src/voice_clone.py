"""Voice cloning client for uploading audio samples and registering cloned voice IDs."""

from __future__ import annotations

import re
import uuid
import wave
from pathlib import Path
from typing import Any

import httpx

from src.audio_io import compute_rms_int16
from src.config import settings

VOICE_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{7,255}$")


def generate_voice_id() -> str:
    """Generate a MiniMax-compliant unique voice_id (>=8 chars, starts with a letter)."""
    return f"vc_{uuid.uuid4().hex[:12]}"


def resolve_and_validate_voice_id(voice_id: str = "") -> str:
    """Return a trimmed or auto-generated voice_id and validate MiniMax naming rules."""
    resolved = (voice_id or "").strip() or generate_voice_id()
    if not VOICE_ID_PATTERN.match(resolved):
        raise VoiceCloneError(
            "Voice ID 格式不合法：必须以英文字母开头，长度至少 8 位（仅限字母、数字、下划线 _ 或连字符 -）。"
        )
    return resolved


class VoiceCloneError(RuntimeError):
    """Raised when voice cloning API requests fail."""


def validate_wav_sample(audio_path: Path, min_seconds: float = 10.0) -> None:
    """Validate WAV duration and non-silent speech before uploading (prevents MiniMax 2013)."""
    if audio_path.suffix.lower() != ".wav":
        return
    with wave.open(str(audio_path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate() or 16000
        duration = frames / float(rate)
        if duration < min_seconds:
            raise VoiceCloneError(
                f"录音时长仅 {duration:.1f} 秒，至少需要 {min_seconds:.0f} 秒清晰人声。"
            )
        pcm = wf.readframes(frames)
        rms = compute_rms_int16(pcm)
        if rms < 0.002:
            raise VoiceCloneError(
                "未检测到有效人声（音量过低或全静音），请确认系统麦克风权限与设备后重录。"
            )


class MiniMaxVoiceCloneClient:
    """Async client for MiniMax voice sample upload and voice cloning."""

    BASE_URL = "https://api.minimaxi.com"

    def __init__(self, api_key: str | None = None, group_id: str | None = None) -> None:
        self.api_key = api_key or settings.minimax_api_key
        self.group_id = group_id or settings.minimax_group_id

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise VoiceCloneError("MINIMAX_API_KEY is not configured.")
        return {"Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def _guess_mime(path: Path) -> str:
        ext = path.suffix.lower()
        if ext == ".wav":
            return "audio/wav"
        if ext == ".m4a":
            return "audio/mp4"
        return "audio/mpeg"

    async def upload_sample(self, audio_path: str | Path, purpose: str = "voice_clone") -> str:
        """Validate and upload a local audio sample and return its file_id."""
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"找不到音频样本文件 ({path})，请先点击「🎤 录制 15 秒样本」完成录音后再克隆！"
            )

        validate_wav_sample(path)

        url = f"{self.BASE_URL}/v1/files/upload"
        params = {"GroupId": self.group_id} if self.group_id else {}
        mime = self._guess_mime(path)

        async with httpx.AsyncClient(timeout=60.0) as client:
            with path.open("rb") as f:
                resp = await client.post(
                    url,
                    headers=self._headers(),
                    params=params,
                    data={"purpose": purpose},
                    files={"file": (path.name, f, mime)},
                )
        if resp.status_code >= 400:
            raise VoiceCloneError(f"Upload failed ({resp.status_code}): {resp.text}")

        payload = resp.json()
        base_resp = payload.get("base_resp", {})
        if base_resp.get("status_code", 0) != 0:
            raise VoiceCloneError(
                f"MiniMax upload error {base_resp.get('status_code')}: {base_resp.get('status_msg')}"
            )
        file_id = payload.get("file", {}).get("file_id")
        if not file_id:
            raise VoiceCloneError(f"Missing file_id in response: {payload}")
        return str(file_id)

    async def create_clone(self, file_id: str, voice_id: str = "", text: str | None = None) -> dict[str, Any]:
        """Create a cloned voice from an uploaded file_id."""
        resolved_voice_id = resolve_and_validate_voice_id(voice_id)
        url = f"{self.BASE_URL}/v1/voice_clone"
        params = {"GroupId": self.group_id} if self.group_id else {}
        body: dict[str, Any] = {
            "file_id": int(file_id) if str(file_id).isdigit() else file_id,
            "voice_id": resolved_voice_id,
        }
        if text:
            body["text"] = text
            body["model"] = settings.minimax_tts_model

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=self._headers(), params=params, json=body)

        if resp.status_code >= 400:
            raise VoiceCloneError(f"Voice clone failed ({resp.status_code}): {resp.text}")
        payload = resp.json()
        base_resp = payload.get("base_resp", {})
        code = base_resp.get("status_code", 0)
        if code != 0:
            if code == 2039:
                raise VoiceCloneError("该 Voice ID 已被占用（2039），请换一个或留空自动生成。")
            if code == 2013:
                raise VoiceCloneError(
                    f"参数校验未通过（2013: {base_resp.get('status_msg')}），请确保 Voice ID 至少 8 位且以字母开头。"
                )
            raise VoiceCloneError(f"MiniMax clone error {code}: {base_resp.get('status_msg')}")
        payload["resolved_voice_id"] = resolved_voice_id
        return payload
