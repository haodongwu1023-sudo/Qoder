"""Multi-user WebSocket translation bridge and cross-language room hub."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
import json
import logging
import re
import time
from typing import Any, Awaitable, Callable
import uuid

from fastapi import WebSocket, WebSocketDisconnect

from app.audio_gate import ForegroundSpeechGate, compute_rms_int16
from app.config import settings
from app.livetranslate_client import AlibabaLiveTranslateSession
from app.minimax_client import PersistentMiniMaxTTSClient, apply_glossary_terms

logger = logging.getLogger(__name__)

_ROOM_ID_RE = re.compile(r"[^A-Za-z0-9_-]")


def normalize_room_id(room_id: str) -> str:
    cleaned = _ROOM_ID_RE.sub("", (room_id or "").strip())[:40]
    return cleaned or "bridge-demo"


def _build_corpus_text(glossary: dict[str, str] | None) -> str:
    if not glossary:
        return ""
    return " ".join(f"{k} {v}".strip() for k, v in glossary.items() if k or v).strip()


class _SafeWebSocketSender:
    """Serialize JSON and binary WebSocket writes per client connection."""

    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self._lock = asyncio.Lock()
        self.closed = False

    async def send_json(self, payload: dict[str, Any]) -> bool:
        if self.closed:
            return False
        async with self._lock:
            if self.closed:
                return False
            try:
                await self.ws.send_json(payload)
                return True
            except Exception:
                self.closed = True
                return False

    async def send_bytes(self, data: bytes) -> bool:
        if self.closed or not data:
            return False
        async with self._lock:
            if self.closed:
                return False
            try:
                await self.ws.send_bytes(data)
                return True
            except Exception:
                self.closed = True
                return False


@dataclass
class WebSessionOptions:
    """Per-connection session configuration matching tuned desktop app defaults."""

    source_lang: str = "zh"
    target_lang: str = "en"
    enable_audio: bool = True
    live_voice_clone: bool = True
    voice_id: str = ""
    speed: float = 1.0
    vad_silence_ms: int = 400
    noise_gate: bool = True
    glossary: dict[str, str] = field(default_factory=dict)
    dashscope_api_key: str = ""
    minimax_api_key: str = ""
    minimax_group_id: str = ""
    speaker_name: str = "访客"

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> "WebSessionOptions":
        raw_glossary = data.get("glossary") or {}
        glossary = (
            {str(k): str(v) for k, v in raw_glossary.items() if str(k).strip()}
            if isinstance(raw_glossary, dict)
            else {}
        )
        try:
            vad_ms = max(200, min(2000, int(data.get("vad_silence_ms") or settings.vad_silence_ms)))
        except (TypeError, ValueError):
            vad_ms = settings.vad_silence_ms
        try:
            speed = max(0.5, min(2.0, float(data.get("speed") or 1.0)))
        except (TypeError, ValueError):
            speed = 1.0
        return cls(
            source_lang=str(data.get("source_lang") or "zh").strip() or "zh",
            target_lang=str(data.get("target_lang") or "en").strip() or "en",
            enable_audio=bool(data.get("enable_audio", True)),
            live_voice_clone=bool(data.get("live_voice_clone", True)),
            voice_id=str(data.get("voice_id") or "").strip(),
            speed=speed,
            vad_silence_ms=vad_ms,
            noise_gate=bool(data.get("noise_gate", True)),
            glossary=glossary,
            dashscope_api_key=str(data.get("dashscope_api_key") or "").strip(),
            minimax_api_key=str(data.get("minimax_api_key") or "").strip(),
            minimax_group_id=str(data.get("minimax_group_id") or "").strip(),
            speaker_name=str(data.get("speaker_name") or "访客").strip()[:32] or "访客",
        )


class WebTranslateBridge:
    """Isolated per-connection real-time translation pipeline fed by browser PCM16 audio."""

    def __init__(
        self,
        options: WebSessionOptions,
        *,
        on_self_json: Callable[[dict[str, Any]], Awaitable[None]],
        on_audio_json: Callable[[dict[str, Any]], Awaitable[None]],
        on_audio_bytes: Callable[[bytes], Awaitable[None]],
        session_factory: Callable[..., Any] = AlibabaLiveTranslateSession,
        minimax_factory: Callable[..., Any] = PersistentMiniMaxTTSClient,
    ) -> None:
        self.options = options
        self._on_self_json = on_self_json
        self._on_audio_json = on_audio_json
        self._on_audio_bytes = on_audio_bytes
        self._session_factory = session_factory
        self._minimax_factory = minimax_factory

        self._gate = ForegroundSpeechGate()
        self._live_session: Any = None
        self._minimax_tts: Any = None
        self._tasks: list[asyncio.Task[Any]] = []
        self._stopped = False
        self._turn_id = 0
        self._started_at = 0.0
        self._audio_rate_sent = 0
        self._echo_duck_until = 0.0

        use_byo_dashscope = bool(options.dashscope_api_key)
        self.demo_limit_seconds = (
            0 if use_byo_dashscope else max(0, int(settings.web_demo_max_seconds))
        )

    @property
    def using_minimax(self) -> bool:
        minimax_key = self.options.minimax_api_key or settings.minimax_api_key
        return bool(
            self.options.enable_audio
            and not self.options.live_voice_clone
            and self.options.voice_id
            and minimax_key
        )

    async def start(self) -> None:
        resolved_ds_key = self.options.dashscope_api_key or settings.dashscope_api_key
        if not resolved_ds_key:
            raise ValueError("未配置 DASHSCOPE_API_KEY，请在服务端 .env 或页面设置中填入 Key。")

        self._started_at = time.monotonic()
        use_minimax = self.using_minimax
        expected_rate = 16000 if use_minimax else 24000

        async def _emit_audio_chunk(pcm: bytes) -> None:
            if self._stopped or not pcm:
                return
            self._echo_duck_until = time.monotonic() + 0.5
            if self._audio_rate_sent != expected_rate:
                self._audio_rate_sent = expected_rate
                await self._on_audio_json(
                    {"type": "audio_format", "sample_rate": expected_rate, "channels": 1}
                )
            await self._on_audio_bytes(pcm)

        if use_minimax:
            self._minimax_tts = self._minimax_factory(
                voice_id=self.options.voice_id,
                target_lang=self.options.target_lang,
                speed=self.options.speed,
                sample_rate=16000,
                on_audio_chunk=_emit_audio_chunk,
                on_error=lambda err: self._on_self_json({"type": "error", "message": err}),
                api_key=self.options.minimax_api_key or None,
                group_id=self.options.minimax_group_id or None,
            )

        async def _on_source_delta(text: str) -> None:
            await self._on_self_json(
                {
                    "type": "source_delta",
                    "turn_id": self._turn_id,
                    "source_lang": self.options.source_lang,
                    "text": text,
                }
            )

        async def _on_source_final(text: str) -> None:
            await self._on_self_json(
                {
                    "type": "source_final",
                    "turn_id": self._turn_id,
                    "source_lang": self.options.source_lang,
                    "text": text,
                }
            )

        async def _on_target_delta(text: str) -> None:
            patched = apply_glossary_terms("", text, self.options.glossary)
            await self._on_self_json(
                {
                    "type": "target_delta",
                    "turn_id": self._turn_id,
                    "target_lang": self.options.target_lang,
                    "text": patched,
                }
            )

        async def _on_target_clause(clause: str) -> None:
            if self._minimax_tts is not None and clause.strip():
                patched = apply_glossary_terms("", clause, self.options.glossary)
                self._minimax_tts.enqueue_text(patched)

        async def _on_target_final(src: str, tgt: str) -> None:
            patched = apply_glossary_terms(src, tgt, self.options.glossary)
            payload = {
                "type": "target_final",
                "turn_id": self._turn_id,
                "source_lang": self.options.source_lang,
                "target_lang": self.options.target_lang,
                "source": src,
                "target": patched,
            }
            self._turn_id += 1
            await self._on_self_json(payload)
            if self._on_audio_json is not self._on_self_json:
                await self._on_audio_json(payload)

        async def _on_speech_started() -> None:
            await self._on_self_json({"type": "speech_started", "turn_id": self._turn_id})

        async def _on_audio_done() -> None:
            await self._on_audio_json({"type": "audio_done", "turn_id": self._turn_id})

        async def _on_connection(state: str) -> None:
            await self._on_self_json(
                {
                    "type": "status",
                    "state": state,
                    "ready": state == "ready",
                    "demo_limit_seconds": self.demo_limit_seconds,
                    "vad_silence_ms": self.options.vad_silence_ms,
                    "voice_mode": "minimax" if use_minimax else (
                        "live_clone" if self.options.live_voice_clone else "preset"
                    ),
                }
            )

        async def _on_error(msg: str) -> None:
            await self._on_self_json({"type": "error", "message": msg})

        self._live_session = self._session_factory(
            source_lang=self.options.source_lang,
            target_lang=self.options.target_lang,
            enable_audio=self.options.enable_audio and not use_minimax,
            live_voice_clone=self.options.live_voice_clone,
            sample_rate=settings.audio_sample_rate,
            silence_duration_ms=self.options.vad_silence_ms,
            asr_corpus_text=_build_corpus_text(self.options.glossary),
            api_key=resolved_ds_key,
            on_source_delta=_on_source_delta,
            on_source_final=_on_source_final,
            on_target_delta=_on_target_delta,
            on_target_clause=_on_target_clause,
            on_target_final=_on_target_final,
            on_audio_chunk=None if use_minimax else _emit_audio_chunk,
            on_audio_done=_on_audio_done,
            on_speech_started=_on_speech_started,
            on_connection=_on_connection,
            on_error=_on_error,
        )

        if self._minimax_tts is not None:
            self._tasks.append(asyncio.create_task(self._minimax_tts.run()))
        self._tasks.append(asyncio.create_task(self._live_session.run()))
        if self.demo_limit_seconds > 0:
            self._tasks.append(asyncio.create_task(self._quota_watchdog()))

    async def _quota_watchdog(self) -> None:
        try:
            await asyncio.sleep(self.demo_limit_seconds)
            if not self._stopped:
                await self._on_self_json(
                    {
                        "type": "quota_ended",
                        "message": f"体验模式单次限时 {self.demo_limit_seconds} 秒已到，可重新开始或填入自己的 Key 解锁不限时。",
                    }
                )
                await self.stop()
        except asyncio.CancelledError:
            raise

    def feed_pcm(self, pcm_bytes: bytes) -> None:
        """Apply tuned foreground gate + impulse rejection and queue PCM16 frames."""
        if self._stopped or self._live_session is None or not pcm_bytes:
            return
        rms = compute_rms_int16(pcm_bytes)
        if not self.options.noise_gate:
            self._live_session.enqueue_audio(pcm_bytes, has_speech=rms >= 0.004)
            return
        duck_db = 4.0 if time.monotonic() < self._echo_duck_until else 0.0
        for frame in self._gate.process(pcm_bytes, rms, echo_duck_db=duck_db):
            self._live_session.enqueue_audio(
                frame,
                silence=not self._gate.is_open,
                has_speech=self._gate.is_open and rms >= 0.004,
            )

    def set_muted(self, muted: bool) -> None:
        if self._live_session is not None:
            self._live_session.set_microphone_muted(bool(muted))

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._live_session is not None:
            with suppress(Exception):
                await asyncio.wait_for(self._live_session.finish_input(), timeout=0.8)
            self._live_session.stop()
        if self._minimax_tts is not None:
            self._minimax_tts.stop()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()


@dataclass
class _RoomPeer:
    peer_id: str
    name: str
    source_lang: str
    target_lang: str
    sender: _SafeWebSocketSender
    bridge: WebTranslateBridge | None = None


class WebRoomHub:
    """In-memory cross-language room relay routing translated subtitles & audio between peers."""

    MAX_PEERS_PER_ROOM = 6

    def __init__(self) -> None:
        self._rooms: dict[str, dict[str, _RoomPeer]] = {}
        self._lock = asyncio.Lock()

    async def join(
        self,
        room_id: str,
        peer_id: str,
        name: str,
        source_lang: str,
        target_lang: str,
        sender: _SafeWebSocketSender,
    ) -> _RoomPeer:
        room_key = normalize_room_id(room_id)
        async with self._lock:
            room = self._rooms.setdefault(room_key, {})
            if len(room) >= self.MAX_PEERS_PER_ROOM and peer_id not in room:
                raise ValueError("房间人数已满（最多 6 人）。")
            peer = _RoomPeer(
                peer_id=peer_id,
                name=name,
                source_lang=source_lang,
                target_lang=target_lang,
                sender=sender,
            )
            room[peer_id] = peer
        await self._broadcast_roster(room_key)
        return peer

    async def update_peer(
        self,
        room_id: str,
        peer_id: str,
        *,
        name: str,
        source_lang: str,
        target_lang: str,
    ) -> None:
        room_key = normalize_room_id(room_id)
        async with self._lock:
            peer = (self._rooms.get(room_key) or {}).get(peer_id)
            if peer is not None:
                peer.name = name
                peer.source_lang = source_lang
                peer.target_lang = target_lang
        await self._broadcast_roster(room_key)

    async def leave(self, room_id: str, peer_id: str) -> None:
        room_key = normalize_room_id(room_id)
        removed: _RoomPeer | None = None
        async with self._lock:
            room = self._rooms.get(room_key)
            if room is not None:
                removed = room.pop(peer_id, None)
                if not room:
                    self._rooms.pop(room_key, None)
        if removed is not None and removed.bridge is not None:
            await removed.bridge.stop()
        await self._broadcast_roster(room_key)

    async def peers_except(self, room_id: str, exclude_peer_id: str) -> list[_RoomPeer]:
        room_key = normalize_room_id(room_id)
        async with self._lock:
            room = self._rooms.get(room_key) or {}
            return [p for pid, p in room.items() if pid != exclude_peer_id]

    async def _broadcast_roster(self, room_id: str) -> None:
        room_key = normalize_room_id(room_id)
        async with self._lock:
            room = self._rooms.get(room_key) or {}
            peers_info = [
                {
                    "peer_id": p.peer_id,
                    "name": p.name,
                    "source_lang": p.source_lang,
                    "target_lang": p.target_lang,
                }
                for p in room.values()
            ]
            senders = [p.sender for p in room.values()]
        payload = {"type": "room_peers", "room_id": room_key, "peers": peers_info}
        if senders:
            await asyncio.gather(*(s.send_json(payload) for s in senders), return_exceptions=True)

    async def relay_peer_json(
        self, room_id: str, sender_peer: _RoomPeer, payload: dict[str, Any]
    ) -> None:
        peers = await self.peers_except(room_id, sender_peer.peer_id)
        if not peers:
            return
        enriched = {
            **payload,
            "scope": "peer",
            "speaker_id": sender_peer.peer_id,
            "speaker_name": sender_peer.name,
        }
        await asyncio.gather(*(p.sender.send_json(enriched) for p in peers), return_exceptions=True)

    async def relay_peer_bytes(
        self, room_id: str, sender_peer: _RoomPeer, data: bytes
    ) -> None:
        peers = await self.peers_except(room_id, sender_peer.peer_id)
        if not peers or not data:
            return
        await asyncio.gather(*(p.sender.send_bytes(data) for p in peers), return_exceptions=True)


room_hub = WebRoomHub()


async def run_web_translate_ws(
    ws: WebSocket,
    *,
    bridge_factory: Callable[..., WebTranslateBridge] = WebTranslateBridge,
) -> None:
    await ws.accept()
    sender = _SafeWebSocketSender(ws)
    bridge: WebTranslateBridge | None = None

    async def _emit(payload: dict[str, Any]) -> None:
        await sender.send_json({**payload, "scope": "self"})

    async def _emit_bytes(data: bytes) -> None:
        await sender.send_bytes(data)

    try:
        await sender.send_json(
            {
                "type": "hello",
                "has_server_key": bool(settings.dashscope_api_key),
                "demo_max_seconds": int(settings.web_demo_max_seconds),
                "vad_silence_ms": int(settings.vad_silence_ms),
            }
        )
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                if bridge is not None:
                    bridge.feed_pcm(msg["bytes"])
                continue
            raw_text = msg.get("text")
            if not raw_text:
                continue
            data = json.loads(raw_text)
            kind = str(data.get("type") or "")
            if kind == "start":
                if bridge is not None:
                    await bridge.stop()
                    bridge = None
                opts = WebSessionOptions.from_payload(data)
                bridge = bridge_factory(
                    opts,
                    on_self_json=_emit,
                    on_audio_json=_emit,
                    on_audio_bytes=_emit_bytes,
                )
                try:
                    await bridge.start()
                except Exception as exc:
                    await sender.send_json({"type": "error", "message": str(exc)})
                    await bridge.stop()
                    bridge = None
            elif kind == "mute" and bridge is not None:
                bridge.set_muted(bool(data.get("muted")))
            elif kind == "stop":
                if bridge is not None:
                    await bridge.stop()
                    bridge = None
                await sender.send_json({"type": "status", "state": "stopped", "ready": False})
    except WebSocketDisconnect:
        pass
    finally:
        sender.closed = True
        if bridge is not None:
            await bridge.stop()


async def run_web_room_ws(
    ws: WebSocket,
    room_id: str,
    *,
    hub: WebRoomHub = room_hub,
    bridge_factory: Callable[..., WebTranslateBridge] = WebTranslateBridge,
) -> None:
    await ws.accept()
    sender = _SafeWebSocketSender(ws)
    peer_id = uuid.uuid4().hex[:10]
    room_key = normalize_room_id(room_id)
    peer: _RoomPeer | None = None

    try:
        peer = await hub.join(room_key, peer_id, "访客", "zh", "en", sender)
        await sender.send_json(
            {
                "type": "room_joined",
                "room_id": room_key,
                "peer_id": peer_id,
                "has_server_key": bool(settings.dashscope_api_key),
                "demo_max_seconds": int(settings.web_demo_max_seconds),
                "vad_silence_ms": int(settings.vad_silence_ms),
            }
        )
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                if peer.bridge is not None:
                    peer.bridge.feed_pcm(msg["bytes"])
                continue
            raw_text = msg.get("text")
            if not raw_text:
                continue
            data = json.loads(raw_text)
            kind = str(data.get("type") or "")
            if kind == "start":
                if peer.bridge is not None:
                    await peer.bridge.stop()
                    peer.bridge = None
                opts = WebSessionOptions.from_payload(data)
                await hub.update_peer(
                    room_key,
                    peer_id,
                    name=opts.speaker_name,
                    source_lang=opts.source_lang,
                    target_lang=opts.target_lang,
                )
                current_peer = peer

                async def _self_json(payload: dict[str, Any]) -> None:
                    await sender.send_json({**payload, "scope": "self"})

                async def _peer_json(payload: dict[str, Any]) -> None:
                    await hub.relay_peer_json(room_key, current_peer, payload)

                async def _peer_bytes(pcm: bytes) -> None:
                    await hub.relay_peer_bytes(room_key, current_peer, pcm)

                bridge = bridge_factory(
                    opts,
                    on_self_json=_self_json,
                    on_audio_json=_peer_json,
                    on_audio_bytes=_peer_bytes,
                )
                peer.bridge = bridge
                try:
                    await bridge.start()
                except Exception as exc:
                    await sender.send_json({"type": "error", "message": str(exc)})
                    await bridge.stop()
                    peer.bridge = None
            elif kind == "mute" and peer.bridge is not None:
                peer.bridge.set_muted(bool(data.get("muted")))
            elif kind == "stop":
                if peer.bridge is not None:
                    await peer.bridge.stop()
                    peer.bridge = None
                await sender.send_json({"type": "status", "state": "stopped", "ready": False})
    except WebSocketDisconnect:
        pass
    except ValueError as exc:
        await sender.send_json({"type": "error", "message": str(exc)})
    finally:
        sender.closed = True
        if peer is not None:
            await hub.leave(room_key, peer_id)
