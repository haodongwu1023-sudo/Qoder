"""End-to-end simultaneous speech-to-speech translation client (DashScope qwen3.5-livetranslate-flash-realtime)."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

import websockets

from app.config import settings

logger = logging.getLogger(__name__)


def _final_translation(msg: dict[str, Any], partial: str) -> str:
    direct = msg.get("transcript") or msg.get("text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    response = msg.get("response")
    if isinstance(response, dict):
        parts = []
        for item in response.get("output") or []:
            if not isinstance(item, dict):
                continue
            for content in item.get("content") or []:
                if not isinstance(content, dict):
                    continue
                value = content.get("transcript") or content.get("text")
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())
        if parts:
            return " ".join(parts)
    return partial.strip()


class LiveTranslateError(RuntimeError):
    """Raised when the end-to-end live translation WebSocket fails."""


class AlibabaLiveTranslateSession:
    """Full-duplex DashScope qwen3.5-livetranslate-flash-realtime session tuned for low latency."""

    WS_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen3.5-livetranslate-flash-realtime"

    def __init__(
        self,
        *,
        source_lang: str,
        target_lang: str,
        enable_audio: bool = True,
        live_voice_clone: bool = True,
        sample_rate: int | None = None,
        silence_duration_ms: int | None = None,
        keepalive_interval_ms: int | None = None,
        keepalive_idle_ms: int | None = None,
        asr_corpus_text: str = "",
        api_key: str | None = None,
        on_source_delta: Callable[[str], Awaitable[None]] | None = None,
        on_source_final: Callable[[str], Awaitable[None]] | None = None,
        on_target_delta: Callable[[str], Awaitable[None]] | None = None,
        on_target_clause: Callable[[str], Awaitable[None]] | None = None,
        on_target_final: Callable[[str, str], Awaitable[None]] | None = None,
        on_audio_chunk: Callable[[bytes], Awaitable[None]] | None = None,
        on_audio_done: Callable[[], Awaitable[None]] | None = None,
        on_speech_started: Callable[[], Awaitable[None]] | None = None,
        on_connection: Callable[[str], Awaitable[None]] | None = None,
        on_error: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.enable_audio = enable_audio
        self.live_voice_clone = live_voice_clone
        self.sample_rate = sample_rate or settings.audio_sample_rate
        raw_vad = settings.vad_silence_ms if silence_duration_ms is None else silence_duration_ms
        self.silence_duration_ms = max(200, min(2000, int(raw_vad)))
        raw_ka = settings.keepalive_interval_ms if keepalive_interval_ms is None else keepalive_interval_ms
        raw_idle = settings.keepalive_idle_ms if keepalive_idle_ms is None else keepalive_idle_ms
        self.keepalive_interval_ms = max(0, int(raw_ka))
        self.keepalive_idle_ms = max(self.keepalive_interval_ms, int(raw_idle))
        self.asr_corpus_text = asr_corpus_text.strip()
        self.api_key = (api_key or "").strip() or None
        self.on_source_delta = on_source_delta
        self.on_source_final = on_source_final
        self.on_target_delta = on_target_delta
        self.on_target_clause = on_target_clause
        self.on_target_final = on_target_final
        self.on_audio_chunk = on_audio_chunk
        self.on_audio_done = on_audio_done
        self.on_speech_started = on_speech_started
        self.on_connection = on_connection
        self.on_error = on_error

        self._manual_muted = False
        self._finishing = False
        self._response_finished = asyncio.Event()
        self._ws: Any = None
        self._send_queue: asyncio.Queue[tuple[float, bytes] | None] = asyncio.Queue(maxsize=100)
        self._stopped = False
        self._paused = False
        self._last_source = ""
        self._last_target = ""
        self._emitted_clause_len = 0
        self._emitted_prefix = ""
        self._last_done_resp_id = ""
        self._queue_drops = 0
        self._expired_frames = 0
        self._audio_turn_open = False
        self._audio_packet_time: float | None = None
        self._session_ready = False
        self._connected_at = 0.0
        self._last_rx_at = 0.0
        self._speech_frames_since_rx = 0
        self._quiet_frames_streak = 0
        self._speech_stall_since: float | None = None
        self._has_sent_audio = False

    def _drain_send_queue(self) -> int:
        dropped = 0
        saw_sentinel = False
        while True:
            try:
                item = self._send_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is None:
                saw_sentinel = True
            else:
                dropped += 1
        if saw_sentinel and self._stopped:
            try:
                self._send_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
        return dropped

    def set_microphone_muted(self, muted: bool) -> None:
        self._manual_muted = muted
        if muted:
            self._drain_send_queue()

    async def finish_input(self) -> None:
        if self._ws is None or self._stopped:
            return
        self._finishing = True
        self._response_finished.clear()
        try:
            await asyncio.wait_for(self._response_finished.wait(), timeout=1.6)
        except asyncio.TimeoutError:
            pass

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        try:
            self._send_queue.put_nowait(None)
        except asyncio.QueueFull:
            try:
                self._send_queue.get_nowait()
                self._send_queue.put_nowait(None)
            except Exception:
                pass

    def enqueue_audio(
        self, pcm_bytes: bytes, *, silence: bool = False, has_speech: bool = False
    ) -> None:
        """Queue mono 16kHz PCM16 frames for non-blocking WebSocket upload."""
        if self._stopped or not pcm_bytes:
            return
        muted = silence or self._paused or self._manual_muted or self._finishing
        if self._session_ready and not muted:
            if has_speech:
                self._quiet_frames_streak = 0
                self._speech_frames_since_rx += 1
                if self._speech_frames_since_rx >= 15 and self._speech_stall_since is None:
                    self._speech_stall_since = time.monotonic()
            else:
                self._quiet_frames_streak += 1
                if self._quiet_frames_streak >= 8:
                    self._speech_frames_since_rx = 0
                    self._speech_stall_since = None
        else:
            self._quiet_frames_streak = 0
            self._speech_frames_since_rx = 0
            self._speech_stall_since = None

        frame = (b"\x00" * len(pcm_bytes)) if muted else pcm_bytes
        if self._send_queue.full():
            try:
                self._send_queue.get_nowait()
                self._queue_drops += 1
            except asyncio.QueueEmpty:
                pass
        try:
            self._send_queue.put_nowait((time.monotonic(), frame))
        except asyncio.QueueFull:
            pass

    def _build_session_update(self) -> dict[str, Any]:
        asr_cfg: dict[str, Any] = {
            "model": settings.asr_model,
            "language": self.source_lang,
        }
        if self.asr_corpus_text:
            asr_cfg["corpus"] = {"text": self.asr_corpus_text}
        session: dict[str, Any] = {
            "modalities": ["text", "audio"] if self.enable_audio else ["text"],
            "input_audio_format": "pcm",
            "sample_rate": self.sample_rate,
            "input_audio_transcription": asr_cfg,
            "translation": {"language": self.target_lang},
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.45,
                "silence_duration_ms": self.silence_duration_ms,
            },
        }
        if self.enable_audio:
            session["output_audio_format"] = "pcm16"
            if self.live_voice_clone:
                session["voice"] = "default"
                session["enable_voice_clone"] = True
                session["voice_clone_options"] = {"frequency": "always"}
            else:
                session["voice"] = "Tina"
        return {"type": "session.update", "session": session}

    async def run(self) -> None:
        resolved_key = (self.api_key or settings.dashscope_api_key or "").strip()
        if not resolved_key:
            raise LiveTranslateError("DASHSCOPE_API_KEY is not configured.")
        headers = {
            "Authorization": f"Bearer {resolved_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        while not self._stopped:
            try:
                self._last_source = self._last_target = ""
                self._emitted_prefix = self._last_done_resp_id = ""
                self._emitted_clause_len = 0
                self._audio_turn_open = False
                self._audio_packet_time = None
                self._session_ready = False
                self._speech_frames_since_rx = 0
                self._quiet_frames_streak = 0
                self._speech_stall_since = None
                self._has_sent_audio = False
                self._connected_at = time.monotonic()
                self._last_rx_at = self._connected_at
                async with websockets.connect(
                    self.WS_URL,
                    additional_headers=headers,
                    open_timeout=8,
                    close_timeout=1,
                    ping_interval=8,
                    ping_timeout=8,
                ) as ws:
                    self._ws = ws
                    self._drain_send_queue()
                    now = time.monotonic()
                    self._connected_at = now
                    self._last_rx_at = now
                    await ws.send(json.dumps(self._build_session_update()))

                    sender_task = asyncio.create_task(self._sender_loop(ws))
                    receiver_task = asyncio.create_task(self._receiver_loop(ws))
                    watchdog_task = asyncio.create_task(self._watchdog_loop())
                    tasks = (sender_task, receiver_task, watchdog_task)
                    try:
                        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                        for task in done:
                            task.result()
                    finally:
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                    if not self._stopped:
                        if self.on_connection is not None:
                            await self.on_connection("reconnecting")
                        await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._stopped:
                    break
                logger.warning("Alibaba live-translate WebSocket reconnecting: %s", exc)
                if self.on_connection is not None:
                    await self.on_connection("reconnecting")
                await asyncio.sleep(0.5)
            finally:
                self._ws = None
                self._session_ready = False
                if not self._stopped:
                    self._drain_send_queue()

    async def _watchdog_loop(
        self,
        *,
        handshake_timeout: float = 8.0,
        speech_stall_timeout: float = 10.0,
        poll_interval: float = 1.0,
    ) -> None:
        while not self._stopped:
            await asyncio.sleep(poll_interval)
            if self._stopped:
                break
            now = time.monotonic()
            if not self._session_ready and now - self._connected_at >= handshake_timeout:
                raise TimeoutError("DashScope session.updated handshake timed out")
            if (
                self._session_ready
                and self._speech_frames_since_rx >= 15
                and self._speech_stall_since is not None
                and now - max(self._last_rx_at, self._speech_stall_since) >= speech_stall_timeout
            ):
                raise TimeoutError("DashScope live-translate stalled during active speech")

    async def _sender_loop(self, ws: Any) -> None:
        interval_s = self.keepalive_interval_ms / 1000.0
        idle_s = self.keepalive_idle_ms / 1000.0
        raw_bytes = int(self.sample_rate * 2 * interval_s)
        frame_bytes = max(2, raw_bytes - (raw_bytes % 2))
        silence_frame = bytes(frame_bytes)
        last_send_at = time.monotonic()
        quiet_keepalives = 0
        get_task: asyncio.Task[tuple[float, bytes] | None] | None = None
        try:
            while not self._stopped:
                if get_task is None:
                    get_task = asyncio.create_task(self._send_queue.get())
                wait_timeout: float | None = None
                if interval_s > 0 and self._has_sent_audio:
                    delay = idle_s if quiet_keepalives == 0 else interval_s
                    wait_timeout = max(0.005, (last_send_at + delay) - time.monotonic())
                done, _ = await asyncio.wait({get_task}, timeout=wait_timeout)
                if get_task in done:
                    item = get_task.result()
                    get_task = None
                    if item is None:
                        break
                    stamp, frame = item
                    if time.monotonic() - stamp > 1.2:
                        self._expired_frames += 1
                        continue
                    if self._manual_muted:
                        frame = bytes(len(frame))
                    quiet_keepalives = 0
                    self._has_sent_audio = True
                else:
                    if self._stopped:
                        break
                    if not self._session_ready or not self._has_sent_audio:
                        last_send_at = time.monotonic()
                        continue
                    frame = silence_frame
                    quiet_keepalives += 1
                    if quiet_keepalives >= 3:
                        self._speech_frames_since_rx = 0
                        self._speech_stall_since = None
                b64 = base64.b64encode(frame).decode("ascii")
                await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": b64}))
                last_send_at = time.monotonic()
                await asyncio.sleep(0)
        finally:
            if get_task is not None:
                if not get_task.done():
                    get_task.cancel()
                await asyncio.gather(get_task, return_exceptions=True)

    async def _receiver_loop(self, ws: Any) -> None:
        async for raw in ws:
            if self._stopped:
                break
            msg = json.loads(raw)
            evt = str(msg.get("type", ""))
            self._last_rx_at = time.monotonic()
            if evt == "session.updated":
                self._session_ready = True
                self._speech_frames_since_rx = 0
                self._quiet_frames_streak = 0
                self._speech_stall_since = None
                if self.on_connection is not None:
                    await self.on_connection("ready")
            if evt == "error":
                err = msg.get("error") or msg
                raise LiveTranslateError(f"DashScope live-translate error: {err}")
            if evt == "input_audio_buffer.speech_started":
                self._speech_frames_since_rx = 0
                self._quiet_frames_streak = 0
                self._speech_stall_since = None
                if self.on_speech_started is not None:
                    await self.on_speech_started()
                continue
            if evt.startswith(
                (
                    "conversation.item.input_audio_transcription.",
                    "input_audio_transcription.",
                    "response.",
                    "translation.",
                    "output_audio.",
                )
            ):
                self._speech_frames_since_rx = 0
                self._quiet_frames_streak = 0
                self._speech_stall_since = None

            if evt in {"response.audio.done", "translation.audio.done", "output_audio.done", "response.done"}:
                if self._audio_turn_open:
                    self._audio_turn_open = False
                    self._audio_packet_time = None
                    if self.on_audio_done is not None:
                        await self.on_audio_done()
                if evt != "response.done":
                    continue
                self._response_finished.set()

            if evt in {
                "conversation.item.input_audio_transcription.text",
                "conversation.item.input_audio_transcription.delta",
                "input_audio_transcription.delta",
            }:
                if evt.endswith(".delta"):
                    piece = str(msg.get("delta") or msg.get("text") or "")
                    next_src = self._last_source + piece
                else:
                    confirmed = str(msg.get("text") or "")
                    stash = str(msg.get("stash") or "")
                    next_src = f"{confirmed}{stash}"
                if next_src and next_src != self._last_source:
                    self._last_source = next_src
                    if self.on_source_delta is not None:
                        await self.on_source_delta(next_src.strip())
            elif evt == "conversation.item.input_audio_transcription.completed":
                text = str(msg.get("transcript") or msg.get("text") or self._last_source).strip()
                if text:
                    if self.on_source_final is not None:
                        await self.on_source_final(text)
                    self._last_source = text
            elif evt in {
                "response.audio_transcript.text",
                "response.audio_transcript.delta",
                "response.text.text",
                "response.text.delta",
                "translation.text.delta",
            }:
                if evt.endswith(".delta"):
                    piece = str(msg.get("delta") or msg.get("text") or "")
                    confirmed = ""
                    next_tgt = self._last_target + piece
                else:
                    confirmed = str(msg.get("text") or "")
                    stash = str(msg.get("stash") or "")
                    next_tgt = f"{confirmed}{stash}"
                if next_tgt and next_tgt != self._last_target:
                    self._last_target = next_tgt
                    if self.on_target_delta is not None:
                        await self.on_target_delta(next_tgt.strip())
                if (
                    self.on_target_clause is not None
                    and confirmed
                    and confirmed.startswith(self._emitted_prefix)
                    and len(confirmed) > self._emitted_clause_len
                ):
                    unspoken = confirmed[self._emitted_clause_len :]
                    cut_idx = -1
                    for sep in (". ", "! ", "? ", "。", "！", "？", "; ", "；"):
                        pos = unspoken.rfind(sep)
                        if pos != -1:
                            cand = pos + len(sep)
                            if cand > cut_idx:
                                cut_idx = cand
                    if cut_idx < 10:
                        for sep in (", ", "，"):
                            pos = unspoken.rfind(sep)
                            if pos != -1:
                                cand = pos + len(sep)
                                if cand >= 18 and cand > cut_idx:
                                    cut_idx = cand
                    if cut_idx >= 10:
                        clause = unspoken[:cut_idx].strip()
                        self._emitted_clause_len += cut_idx
                        self._emitted_prefix = confirmed[: self._emitted_clause_len]
                        if clause:
                            await self.on_target_clause(clause)
            elif evt in {"response.audio_transcript.done", "response.text.done", "response.done"}:
                response = msg.get("response")
                nested_id = response.get("id") if isinstance(response, dict) else None
                resp_id = str(msg.get("response_id") or nested_id or msg.get("item_id") or "")
                if resp_id and resp_id == self._last_done_resp_id:
                    continue
                text = _final_translation(msg, self._last_target)
                if not text:
                    if evt == "response.done":
                        self._last_source = ""
                        self._last_target = ""
                        self._emitted_prefix = ""
                        self._emitted_clause_len = 0
                    continue
                if resp_id:
                    self._last_done_resp_id = resp_id
                src = self._last_source.strip()
                if self.on_target_clause is not None:
                    if self._emitted_prefix and text.startswith(self._emitted_prefix.rstrip()):
                        remainder = text[len(self._emitted_prefix.rstrip()) :].strip()
                    elif self._emitted_prefix:
                        remainder = ""
                    else:
                        remainder = text
                    if remainder:
                        await self.on_target_clause(remainder)
                self._emitted_clause_len = 0
                self._emitted_prefix = ""
                self._last_target = ""
                if self.on_target_final is not None:
                    await self.on_target_final(src, text)
                self._last_source = ""
            elif evt in {"response.audio.delta", "translation.audio.delta", "output_audio.delta"}:
                b64_audio = msg.get("delta")
                if b64_audio and self.on_audio_chunk is not None:
                    pcm_chunk = base64.b64decode(b64_audio)
                    if pcm_chunk:
                        self._audio_packet_time = time.monotonic()
                        self._audio_turn_open = True
                        await self.on_audio_chunk(pcm_chunk)
