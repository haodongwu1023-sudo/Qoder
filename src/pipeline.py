"""Split translation + cloned-voice streaming TTS pipeline."""

from __future__ import annotations

import asyncio
from collections import deque
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import httpx
import websockets

from src.config import settings

logger = logging.getLogger(__name__)

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


class PipelineError(RuntimeError):
    """Raised when translation or TTS streaming fails."""


class SplitTranslationPipeline:
    """Three-stage pipeline: (ASR) -> Machine Translation -> Cloned-Voice Streaming TTS."""

    DASHSCOPE_CHAT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    MINIMAX_T2A_URL = "https://api.minimaxi.com/v1/t2a_v2"
    MINIMAX_T2A_WS_URL = "wss://api.minimaxi.com/ws/v1/t2a_v2"

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0),
                limits=httpx.Limits(
                    max_keepalive_connections=12,
                    max_connections=24,
                    keepalive_expiry=60.0,
                ),
            )
        return self._client

    async def warmup(self) -> None:
        """Pre-establish TLS keep-alive connections to DashScope MT and MiniMax TTS."""
        client = self._get_client()

        async def _ping(url: str) -> None:
            try:
                await client.head(url, timeout=2.0)
            except Exception:
                pass

        await asyncio.gather(
            _ping("https://dashscope.aliyuncs.com"),
            _ping("https://api.minimaxi.com"),
        )

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _build_mt_request(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        glossary: dict[str, str] | None,
        *,
        stream: bool,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        if not settings.dashscope_api_key:
            raise PipelineError("DASHSCOPE_API_KEY is not configured.")

        glossary_hint = ""
        if glossary:
            pairs = "; ".join(f"{k} -> {v}" for k, v in glossary.items())
            glossary_hint = f" Terms: {pairs}."

        payload: dict[str, Any] = {
            "model": settings.mt_model,
            "temperature": 0.1,
            "max_tokens": 220,
            "stream": stream,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"Simultaneous interpreter ({source_lang}->{target_lang}). "
                        f"Output ONLY natural concise spoken translation.{glossary_hint}"
                    ),
                },
                {"role": "user", "content": text},
            ],
        }
        headers = {
            "Authorization": f"Bearer {settings.dashscope_api_key}",
            "Content-Type": "application/json",
        }
        return headers, payload

    async def stream_translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        glossary: dict[str, str] | None = None,
    ) -> AsyncIterator[str]:
        """Stream cumulative translated text snapshots via SSE so UI subtitles appear immediately."""
        headers, payload = self._build_mt_request(
            text, source_lang, target_lang, glossary, stream=True
        )
        client = self._get_client()
        accumulated = ""
        for attempt in range(2):
            try:
                async with client.stream(
                    "POST", self.DASHSCOPE_CHAT_URL, headers=headers, json=payload
                ) as resp:
                    if resp.status_code >= 400:
                        body = (await resp.aread()).decode("utf-8", errors="replace")
                        raise PipelineError(f"MT request failed ({resp.status_code}): {body}")
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if not raw or raw == "[DONE]":
                            continue
                        chunk_obj = json.loads(raw)
                        delta = (
                            chunk_obj.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content")
                        )
                        if delta:
                            accumulated += delta
                            yield accumulated
                return
            except httpx.TransportError as exc:
                if accumulated or attempt == 1:
                    raise PipelineError(f"MT connection error: {exc}") from exc

    async def translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        glossary: dict[str, str] | None = None,
    ) -> str:
        """Translate a recognized utterance with optional domain glossary."""
        headers, payload = self._build_mt_request(
            text, source_lang, target_lang, glossary, stream=False
        )
        client = self._get_client()
        resp: httpx.Response | None = None
        for attempt in range(2):
            try:
                resp = await client.post(self.DASHSCOPE_CHAT_URL, headers=headers, json=payload)
                break
            except httpx.TransportError as exc:
                if attempt == 1:
                    raise PipelineError(f"MT connection error: {exc}") from exc
        assert resp is not None
        if resp.status_code >= 400:
            raise PipelineError(f"MT request failed ({resp.status_code}): {resp.text}")
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    async def stream_cloned_tts(
        self,
        text: str,
        voice_id: str | None = None,
        speed: float = 1.0,
        target_lang: str = "en",
    ) -> AsyncIterator[bytes]:
        """Stream synthesized PCM16 audio chunks from MiniMax t2a_v2 HTTP SSE using keep-alive."""
        if not text or not text.strip():
            return
        if not settings.minimax_api_key:
            raise PipelineError("MINIMAX_API_KEY is not configured.")

        resolved_voice = voice_id or settings.default_cloned_voice_id
        if not resolved_voice:
            raise PipelineError("No cloned voice_id provided or configured.")

        url = self.MINIMAX_T2A_URL
        if settings.minimax_group_id:
            url = f"{url}?GroupId={settings.minimax_group_id}"

        headers = {
            "Authorization": f"Bearer {settings.minimax_api_key}",
            "Content-Type": "application/json",
        }
        lang_boost = LANGUAGE_BOOST_MAP.get(target_lang, "auto")
        payload: dict[str, Any] = {
            "model": settings.minimax_tts_model,
            "text": text,
            "stream": True,
            "language_boost": lang_boost,
            "voice_setting": {
                "voice_id": resolved_voice,
                "speed": speed,
                "vol": 1.0,
                "pitch": 0,
            },
            "audio_setting": {
                "sample_rate": settings.audio_sample_rate,
                "bitrate": 128000,
                "format": "pcm",
                "channel": 1,
            },
        }

        client = self._get_client()
        yielded_any = False
        for attempt in range(2):
            try:
                async with client.stream("POST", url, headers=headers, json=payload) as resp:
                    if resp.status_code >= 400:
                        body = (await resp.aread()).decode("utf-8", errors="replace")
                        raise PipelineError(f"MiniMax TTS HTTP {resp.status_code}: {body}")

                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if not data_str or data_str == "[DONE]":
                            continue
                        msg = json.loads(data_str)
                        base_resp = msg.get("base_resp") or {}
                        status_code = base_resp.get("status_code", 0)
                        if status_code not in (0, None):
                            raise PipelineError(f"MiniMax TTS failed: {msg}")

                        data = msg.get("data") or {}
                        status = data.get("status")
                        # status==2 is the final summary packet repeating full concatenated audio
                        if status == 2:
                            break
                        audio_hex = data.get("audio")
                        if audio_hex:
                            yielded_any = True
                            yield bytes.fromhex(audio_hex)
                return
            except httpx.TransportError as exc:
                if yielded_any or attempt == 1:
                    raise PipelineError(f"MiniMax TTS transport error: {exc}") from exc


class PersistentMiniMaxTTSClient:
    """Persistent WebSocket client for MiniMax wss://api.minimaxi.com/ws/v1/t2a_v2 (matches native MiniMaxTTSClient)."""

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
        fallback_pipeline: SplitTranslationPipeline | None = None,
    ) -> None:
        self.voice_id = voice_id
        self.target_lang = target_lang
        self.speed = speed
        self.sample_rate = sample_rate
        self.on_audio_chunk = on_audio_chunk
        self.on_error = on_error
        self.fallback_pipeline = fallback_pipeline or SplitTranslationPipeline()
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
        """Queue a translated clause or sentence for immediate synthesis on the open WebSocket."""
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
        lang_boost = LANGUAGE_BOOST_MAP.get(self.target_lang, "auto")
        return {
            "event": "task_start",
            "model": settings.minimax_tts_model,
            "language_boost": lang_boost,
            "voice_setting": {
                "voice_id": self.voice_id,
                "speed": self.speed,
                "vol": 1.0,
                "pitch": 0,
            },
            "audio_setting": {
                "sample_rate": self.sample_rate,
                "bitrate": 128000,
                "format": "pcm",
                "channel": 1,
            },
        }

    async def _fallback_http_tts(self, text: str) -> None:
        """Fallback to HTTP SSE streaming if WebSocket is momentarily reconnecting."""
        try:
            async for chunk in self.fallback_pipeline.stream_cloned_tts(
                text=text,
                voice_id=self.voice_id,
                speed=self.speed,
                target_lang=self.target_lang,
            ):
                if self._stopped:
                    break
                if chunk:
                    await self.on_audio_chunk(chunk)
        except Exception as exc:
            logger.warning("MiniMax HTTP fallback TTS failed: %s", exc)
            if self.on_error is not None:
                await self.on_error(f"MiniMax 语音合成异常: {exc}")

    async def run(self) -> None:
        """Keep a warm t2a_v2 WebSocket ready and stream task_continue chunks with zero handshake delay."""
        if not settings.minimax_api_key or not self.voice_id:
            return
        headers = {"Authorization": f"Bearer {settings.minimax_api_key}"}

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
                    # Wait briefly for connected_success greeting if sent
                    raw_greeting = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    greeting = json.loads(raw_greeting)
                    base_resp = greeting.get("base_resp") or {}
                    if base_resp.get("status_code") not in (0, None):
                        raise PipelineError(f"MiniMax WS greeting error: {greeting}")

                    await ws.send(json.dumps(self._build_task_start()))
                    raw_started = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    started = json.loads(raw_started)
                    s_resp = started.get("base_resp") or {}
                    if s_resp.get("status_code") not in (0, None) or started.get("event") == "task_failed":
                        code = s_resp.get("status_code")
                        msg_txt = s_resp.get("status_msg") or started
                        raise PipelineError(f"MiniMax WS task_start failed ({code}): {msg_txt}")

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
                        chunk_count_in_turn = 0
                        async for raw_msg in ws:
                            if self._stopped:
                                break
                            msg = json.loads(raw_msg)
                            b_resp = msg.get("base_resp") or {}
                            if b_resp.get("status_code") not in (0, None) or msg.get("event") == "task_failed":
                                raise PipelineError(f"MiniMax WS synthesis failed: {msg}")
                            data = msg.get("data") or {}
                            # Guard against status==2 full-audio duplicate summary if ever present
                            if data.get("status") == 2 and chunk_count_in_turn > 0:
                                chunk_count_in_turn = 0
                                if in_flight:
                                    in_flight.popleft()
                                continue
                            audio_hex = data.get("audio")
                            if audio_hex:
                                chunk_count_in_turn += 1
                                if in_flight:
                                    in_flight[0]["got_audio"] = True
                                await self.on_audio_chunk(bytes.fromhex(audio_hex))
                            if msg.get("is_final"):
                                chunk_count_in_turn = 0
                                if in_flight:
                                    in_flight.popleft()

                    send_task = asyncio.create_task(_sender())
                    recv_task = asyncio.create_task(_receiver())
                    done, pending = await asyncio.wait(
                        {send_task, recv_task},
                        return_when=asyncio.FIRST_COMPLETED,
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
                # Fatal account/auth errors (e.g. 1008 insufficient balance, 1004 auth fail) should not spin-reconnect
                if "1008" in err_str or "insufficient balance" in err_str.lower():
                    logger.error("MiniMax TTS stopped due to insufficient balance: %s", err_str)
                    if self.on_error is not None:
                        await self.on_error("MiniMax 账户余额不足 (1008)，已暂停克隆语音合成（字幕继续运行），请充值或在「发送音色」改选「实时动态声纹直出」。")
                    while not self._stopped:
                        await asyncio.sleep(1.0)
                    break
                logger.info("MiniMax WS reconnecting (%s)", err_str)
                while in_flight and not self._stopped:
                    entry = in_flight.popleft()
                    if not entry.get("got_audio") and entry.get("text"):
                        await self._fallback_http_tts(str(entry["text"]))
                # Also drain any immediately waiting text via HTTP fallback so user hears zero pause
                while not self._text_queue.empty() and not self._stopped:
                    try:
                        nxt = self._text_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if nxt is None:
                        self._stopped = True
                        break
                    await self._fallback_http_tts(nxt)
                await asyncio.sleep(0.5)
