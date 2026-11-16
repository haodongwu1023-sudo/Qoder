"""Streaming ASR client using DashScope qwen3-asr-flash-realtime WebSocket."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import websockets

from src.config import settings


class ASRError(RuntimeError):
    """Raised when streaming ASR fails."""


def _is_ws_open(ws: Any) -> bool:
    return ws is not None and getattr(ws, "close_code", None) is None


async def _send_pcm_burst(ws: Any, pcm_bytes: bytes, chunk_size: int = 32768) -> None:
    for i in range(0, len(pcm_bytes), chunk_size):
        b64 = base64.b64encode(pcm_bytes[i : i + chunk_size]).decode("ascii")
        await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": b64}))


async def _read_turn_transcript(ws: Any, timeout: float = 10.0) -> str:
    partial = ""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise asyncio.TimeoutError("Timed out waiting for ASR transcript")
        raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
        msg = json.loads(raw)
        evt = msg.get("type", "")
        if evt == "conversation.item.input_audio_transcription.completed":
            text = str(msg.get("transcript", "")).strip()
            return text or partial.strip()
        if evt in {
            "conversation.item.input_audio_transcription.delta",
            "conversation.item.input_audio_transcription.text",
        }:
            if evt.endswith(".delta"):
                partial += str(msg.get("delta") or msg.get("text") or "")
            else:
                partial = str(msg.get("text") or "") + str(msg.get("stash") or "")
        elif evt == "session.finished":
            break
        elif evt == "error":
            raise ASRError(f"DashScope ASR error: {msg}")
    return partial.strip()


class RealtimeASRStream:
    """Persistent, reusable DashScope ASR WebSocket supporting live chunk streaming during speech."""

    BASE_WS_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"

    def __init__(self) -> None:
        self._ws: Any = None
        self._lang: str = ""
        self._sample_rate: int = 16000
        self._streamed_bytes: int = 0
        self._lock = asyncio.Lock()

    async def _close_ws_unlocked(self) -> None:
        ws = self._ws
        self._ws = None
        self._streamed_bytes = 0
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass

    async def _ensure_connected_unlocked(self, source_lang: str, sample_rate: int) -> Any:
        if (
            _is_ws_open(self._ws)
            and self._lang == source_lang
            and self._sample_rate == sample_rate
        ):
            return self._ws
        await self._close_ws_unlocked()
        if not settings.dashscope_api_key:
            raise ASRError("DASHSCOPE_API_KEY is not configured.")
        url = f"{self.BASE_WS_URL}?model={settings.asr_model}"
        headers = {
            "Authorization": f"Bearer {settings.dashscope_api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        ws = await websockets.connect(url, additional_headers=headers, open_timeout=15)
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "modalities": ["text"],
                        "input_audio_format": "pcm",
                        "sample_rate": sample_rate,
                        "input_audio_transcription": {"language": source_lang},
                        "turn_detection": None,
                    },
                }
            )
        )
        self._ws = ws
        self._lang = source_lang
        self._sample_rate = sample_rate
        self._streamed_bytes = 0
        return ws

    async def warmup(self, source_lang: str = "zh", sample_rate: int = 16000) -> None:
        """Pre-connect WebSocket on session startup so the first utterance has zero handshake lag."""
        async with self._lock:
            try:
                await self._ensure_connected_unlocked(source_lang, sample_rate)
            except Exception:
                await self._close_ws_unlocked()

    async def start_utterance(self, source_lang: str = "zh", sample_rate: int = 16000) -> None:
        """Prepare a clean audio buffer before streaming live chunks for a new utterance."""
        async with self._lock:
            try:
                if self._streamed_bytes > 0:
                    # DashScope realtime ASR does not support input_audio_buffer.clear;
                    # drop the uncommitted socket and open a clean session.
                    await self._close_ws_unlocked()
                await self._ensure_connected_unlocked(source_lang, sample_rate)
                self._streamed_bytes = 0
            except Exception:
                await self._close_ws_unlocked()

    async def feed_chunk(
        self, pcm_chunk: bytes, source_lang: str = "zh", sample_rate: int = 16000
    ) -> None:
        """Stream a live PCM16 chunk while the speaker is still talking."""
        if not pcm_chunk:
            return
        async with self._lock:
            try:
                ws = await self._ensure_connected_unlocked(source_lang, sample_rate)
                await _send_pcm_burst(ws, pcm_chunk, chunk_size=32768)
                self._streamed_bytes += len(pcm_chunk)
            except Exception:
                await self._close_ws_unlocked()

    async def commit_or_transcribe(
        self,
        full_pcm_bytes: bytes,
        source_lang: str = "zh",
        sample_rate: int = 16000,
    ) -> str:
        """Commit streamed audio (or replay buffered PCM on reconnect) and return transcript."""
        if not full_pcm_bytes:
            return ""
        async with self._lock:
            streamed = self._streamed_bytes
            self._streamed_bytes = 0
            # Fast live path: WS is open, same language, and live chunks were streamed
            if (
                _is_ws_open(self._ws)
                and self._lang == source_lang
                and self._sample_rate == sample_rate
                and 0 < streamed <= len(full_pcm_bytes)
            ):
                try:
                    tail = full_pcm_bytes[streamed:]
                    if tail:
                        await _send_pcm_burst(self._ws, tail, chunk_size=32768)
                    await self._ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                    return await _read_turn_transcript(self._ws, timeout=8.0)
                except Exception:
                    await self._close_ws_unlocked()

            # Reconnect / full-buffer path (also handles stale idle sockets cleanly)
            last_exc: Exception | None = None
            for _ in range(2):
                try:
                    await self._close_ws_unlocked()
                    ws = await self._ensure_connected_unlocked(source_lang, sample_rate)
                    await _send_pcm_burst(ws, full_pcm_bytes, chunk_size=32768)
                    await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                    return await _read_turn_transcript(ws, timeout=10.0)
                except ASRError:
                    await self._close_ws_unlocked()
                    raise
                except Exception as exc:
                    last_exc = exc
                    await self._close_ws_unlocked()
            raise ASRError(f"DashScope ASR failed: {last_exc}")

    async def close(self) -> None:
        async with self._lock:
            await self._close_ws_unlocked()


class DashScopeRealtimeASR:
    """Recognizes a PCM16 utterance via DashScope realtime WebSocket."""

    BASE_WS_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"

    def create_stream(self) -> RealtimeASRStream:
        """Create a persistent per-channel streaming ASR helper."""
        return RealtimeASRStream()

    async def transcribe_utterance(
        self,
        pcm_bytes: bytes,
        source_lang: str = "zh",
        sample_rate: int = 16000,
    ) -> str:
        """Send a complete PCM16 utterance and return the recognized transcript."""
        if not settings.dashscope_api_key:
            raise ASRError("DASHSCOPE_API_KEY is not configured.")
        if not pcm_bytes:
            return ""

        url = f"{self.BASE_WS_URL}?model={settings.asr_model}"
        headers = {
            "Authorization": f"Bearer {settings.dashscope_api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        async with websockets.connect(url, additional_headers=headers, open_timeout=15) as ws:
            session_update: dict[str, Any] = {
                "event_id": "evt_session_update",
                "type": "session.update",
                "session": {
                    "modalities": ["text"],
                    "input_audio_format": "pcm",
                    "sample_rate": sample_rate,
                    "input_audio_transcription": {
                        "language": source_lang,
                    },
                    "turn_detection": None,
                },
            }
            await ws.send(json.dumps(session_update))
            await _send_pcm_burst(ws, pcm_bytes, chunk_size=32768)
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            return await _read_turn_transcript(ws, timeout=12.0)
