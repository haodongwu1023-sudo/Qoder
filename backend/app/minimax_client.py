"""MiniMax voice cloning and persistent WebSocket TTS client for web sessions."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
import json
import logging
import re
from typing import Any
import uuid

import httpx
import websockets

from app.audio_gate import validate_wav_bytes
from app.config import settings

logger = logging.getLogger(__name__)

VOICE_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{7,255}$")

LANGUAGE_BOOST_MAP = {
    "zh": "Chinese",
    "yue": "Chinese,Yue",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
    "ar": "Arabic",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "tr": "Turkish",
    "pl": "Polish",
    "cs": "Czech",
    "ro": "Romanian",
    "nl": "Dutch",
}


class VoiceCloneError(RuntimeError):
    """Raised when MiniMax voice cloning or TTS fails."""


def generate_voice_id() -> str:
    return f"vc_{uuid.uuid4().hex[:12]}"


def resolve_and_validate_voice_id(voice_id: str = "") -> str:
    resolved = (voice_id or "").strip() or generate_voice_id()
    if not VOICE_ID_PATTERN.match(resolved):
        raise VoiceCloneError(
            "Voice ID 格式不合法：必须以英文字母开头，长度至少 8 位（仅限字母、数字、下划线 _ 或连字符 -）。"
        )
    return resolved


def apply_glossary_terms(
    source_text: str, target_text: str, glossary: dict[str, str] | None
) -> str:
    """Apply user glossary substitutions if source terms remained untranslated in target_text."""
    if not glossary or not target_text:
        return target_text
    out = target_text
    for src_term, tgt_term in glossary.items():
        if not src_term or not tgt_term:
            continue
        if src_term in out and tgt_term not in out:
            out = out.replace(src_term, tgt_term)
    return out


class MiniMaxVoiceCloneClient:
    """Async client uploading browser WAV samples and creating MiniMax voice_ids."""

    BASE_URL = "https://api.minimaxi.com"

    def __init__(self, api_key: str | None = None, group_id: str | None = None) -> None:
        self.api_key = (api_key or settings.minimax_api_key or "").strip()
        self.group_id = (group_id if group_id is not None else settings.minimax_group_id or "").strip()

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise VoiceCloneError("未配置 MINIMAX_API_KEY，请在设置里填写 MiniMax Key。")
        return {"Authorization": f"Bearer {self.api_key}"}

    async def upload_wav_bytes(self, wav_bytes: bytes, purpose: str = "voice_clone") -> str:
        try:
            validate_wav_bytes(wav_bytes, min_seconds=10.0)
        except ValueError as exc:
            raise VoiceCloneError(str(exc)) from exc

        url = f"{self.BASE_URL}/v1/files/upload"
        params = {"GroupId": self.group_id} if self.group_id else {}
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                url,
                headers=self._headers(),
                params=params,
                data={"purpose": purpose},
                files={"file": ("web_sample.wav", wav_bytes, "audio/wav")},
            )
        if resp.status_code >= 400:
            raise VoiceCloneError(f"上传样本失败 ({resp.status_code}): {resp.text}")
        payload = resp.json()
        base_resp = payload.get("base_resp", {})
        if base_resp.get("status_code", 0) != 0:
            raise VoiceCloneError(
                f"MiniMax 上传失败 {base_resp.get('status_code')}: {base_resp.get('status_msg')}"
            )
        file_id = payload.get("file", {}).get("file_id")
        if not file_id:
            raise VoiceCloneError(f"缺少 file_id: {payload}")
        return str(file_id)

    async def create_clone(self, file_id: str, voice_id: str = "", text: str | None = None) -> dict[str, Any]:
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
            raise VoiceCloneError(f"声音复刻失败 ({resp.status_code}): {resp.text}")
        payload = resp.json()
        base_resp = payload.get("base_resp", {})
        code = base_resp.get("status_code", 0)
        if code != 0:
            if code == 2039:
                raise VoiceCloneError("该 Voice ID 已被占用（2039），请换一个或留空自动生成。")
            if code == 2013:
                raise VoiceCloneError(
                    f"参数校验未通过（2013: {base_resp.get('status_msg')}）"
                )
            raise VoiceCloneError(f"MiniMax 复刻失败 {code}: {base_resp.get('status_msg')}")
        payload["resolved_voice_id"] = resolved_voice_id
        return payload


class MiniMaxTTSHttpFallback:
    MINIMAX_T2A_URL = "https://api.minimaxi.com/v1/t2a_v2"

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))
        return self._client

    async def stream_cloned_tts(
        self,
        text: str,
        voice_id: str,
        *,
        speed: float = 1.0,
        target_lang: str = "en",
        sample_rate: int = 16000,
        api_key: str | None = None,
        group_id: str | None = None,
    ) -> AsyncIterator[bytes]:
        if not text or not text.strip():
            return
        resolved_key = (api_key or settings.minimax_api_key or "").strip()
        if not resolved_key:
            raise VoiceCloneError("MINIMAX_API_KEY is not configured.")
        resolved_group = (group_id if group_id is not None else settings.minimax_group_id or "").strip()
        url = f"{self.MINIMAX_T2A_URL}?GroupId={resolved_group}" if resolved_group else self.MINIMAX_T2A_URL
        headers = {"Authorization": f"Bearer {resolved_key}", "Content-Type": "application/json"}
        payload = {
            "model": settings.minimax_tts_model,
            "text": text,
            "stream": True,
            "language_boost": LANGUAGE_BOOST_MAP.get(target_lang, "auto"),
            "voice_setting": {"voice_id": voice_id, "speed": speed, "vol": 1.0, "pitch": 0},
            "audio_setting": {"sample_rate": sample_rate, "bitrate": 128000, "format": "pcm", "channel": 1},
        }
        client = self._get_client()
        async with client.stream("POST", url, headers=headers, json=payload) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode("utf-8", errors="replace")
                raise VoiceCloneError(f"MiniMax TTS HTTP {resp.status_code}: {body}")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                msg = json.loads(data_str)
                data = msg.get("data") or {}
                if data.get("status") == 2:
                    break
                audio_hex = data.get("audio")
                if audio_hex:
                    yield bytes.fromhex(audio_hex)


class PersistentMiniMaxTTSClient:
    """Persistent WebSocket client for MiniMax wss://api.minimaxi.com/ws/v1/t2a_v2."""

    WS_URL = "wss://api.minimaxi.com/ws/v1/t2a_v2"

    def __init__(
        self,
        *,
        voice_id: str,
        target_lang: str = "en",
        speed: float = 1.0,
        sample_rate: int = 16000,
        on_audio_chunk: Callable[[bytes], Awaitable[None]],
        on_error: Callable[[str], Awaitable[None]] | None = None,
        api_key: str | None = None,
        group_id: str | None = None,
    ) -> None:
        self.voice_id = voice_id
        self.target_lang = target_lang
        self.speed = speed
        self.sample_rate = sample_rate
        self.on_audio_chunk = on_audio_chunk
        self.on_error = on_error
        self.api_key = (api_key or "").strip() or None
        self.group_id = (group_id or "").strip() or None
        self._fallback = MiniMaxTTSHttpFallback()
        self._text_queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=64)
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True
        try:
            self._text_queue.put_nowait(None)
        except asyncio.QueueFull:
            try:
                self._text_queue.get_nowait()
                self._text_queue.put_nowait(None)
            except Exception:
                pass

    def enqueue_text(self, text: str) -> None:
        cleaned = (text or "").strip()
        if self._stopped or not cleaned:
            return
        if self._text_queue.full():
            try:
                self._text_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            self._text_queue.put_nowait(cleaned)
        except asyncio.QueueFull:
            pass

    def _build_task_start(self) -> dict[str, Any]:
        return {
            "event": "task_start",
            "model": settings.minimax_tts_model,
            "language_boost": LANGUAGE_BOOST_MAP.get(self.target_lang, "auto"),
            "voice_setting": {"voice_id": self.voice_id, "speed": self.speed, "vol": 1.0, "pitch": 0},
            "audio_setting": {"sample_rate": self.sample_rate, "bitrate": 128000, "format": "pcm", "channel": 1},
        }

    async def _fallback_http_tts(self, text: str) -> None:
        try:
            async for chunk in self._fallback.stream_cloned_tts(
                text=text,
                voice_id=self.voice_id,
                speed=self.speed,
                target_lang=self.target_lang,
                sample_rate=self.sample_rate,
                api_key=self.api_key,
                group_id=self.group_id,
            ):
                if self._stopped:
                    break
                if chunk:
                    await self.on_audio_chunk(chunk)
        except Exception as exc:
            if self.on_error is not None:
                await self.on_error(f"MiniMax 语音合成异常: {exc}")

    async def run(self) -> None:
        resolved_key = (self.api_key or settings.minimax_api_key or "").strip()
        if not resolved_key or not self.voice_id:
            return
        headers = {"Authorization": f"Bearer {resolved_key}"}

        while not self._stopped:
            in_flight: deque[dict[str, Any]] = deque()
            try:
                async with websockets.connect(
                    self.WS_URL,
                    additional_headers=headers,
                    open_timeout=10,
                    ping_interval=20,
                    ping_timeout=20,
                ) as ws:
                    raw_greeting = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    greeting = json.loads(raw_greeting)
                    base_resp = greeting.get("base_resp") or {}
                    if base_resp.get("status_code") not in (0, None):
                        raise VoiceCloneError(f"MiniMax WS greeting error: {greeting}")
                    await ws.send(json.dumps(self._build_task_start()))
                    started = json.loads(await asyncio.wait_for(ws.recv(), timeout=10.0))
                    s_resp = started.get("base_resp") or {}
                    if s_resp.get("status_code") not in (0, None) or started.get("event") == "task_failed":
                        raise VoiceCloneError(f"MiniMax WS task_start failed: {started}")

                    async def _sender() -> None:
                        while not self._stopped:
                            item = await self._text_queue.get()
                            if item is None:
                                try:
                                    await ws.send(json.dumps({"event": "task_finish"}))
                                except Exception:
                                    pass
                                break
                            in_flight.append({"text": item, "got_audio": False})
                            await ws.send(json.dumps({"event": "task_continue", "text": item}))

                    async def _receiver() -> None:
                        chunk_count = 0
                        async for raw_msg in ws:
                            if self._stopped:
                                break
                            msg = json.loads(raw_msg)
                            b_resp = msg.get("base_resp") or {}
                            if b_resp.get("status_code") not in (0, None) or msg.get("event") == "task_failed":
                                raise VoiceCloneError(f"MiniMax WS synthesis failed: {msg}")
                            data = msg.get("data") or {}
                            if data.get("status") == 2 and chunk_count > 0:
                                chunk_count = 0
                                if in_flight:
                                    in_flight.popleft()
                                continue
                            audio_hex = data.get("audio")
                            if audio_hex:
                                chunk_count += 1
                                if in_flight:
                                    in_flight[0]["got_audio"] = True
                                await self.on_audio_chunk(bytes.fromhex(audio_hex))
                            if msg.get("is_final"):
                                chunk_count = 0
                                if in_flight:
                                    in_flight.popleft()

                    t_send = asyncio.create_task(_sender())
                    t_recv = asyncio.create_task(_receiver())
                    done, pending = await asyncio.wait(
                        {t_send, t_recv}, return_when=asyncio.FIRST_COMPLETED
                    )
                    for t in pending:
                        t.cancel()
                        try:
                            await t
                        except asyncio.CancelledError:
                            pass
                    for t in done:
                        exc = t.exception()
                        if exc is not None and not isinstance(exc, asyncio.CancelledError):
                            raise exc
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._stopped:
                    break
                err_str = str(exc) or repr(exc)
                if "1008" in err_str or "insufficient balance" in err_str.lower():
                    if self.on_error is not None:
                        await self.on_error("MiniMax 账户余额不足 (1008)，已暂停克隆语音合成。")
                    while not self._stopped:
                        await asyncio.sleep(1.0)
                    break
                while in_flight and not self._stopped:
                    entry = in_flight.popleft()
                    if not entry.get("got_audio") and entry.get("text"):
                        await self._fallback_http_tts(str(entry["text"]))
                await asyncio.sleep(0.5)
