import asyncio
import io
import struct
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from fastapi.testclient import TestClient

from app.audio_gate import ForegroundSpeechGate, looks_like_impulse, validate_wav_bytes
from app.config import settings
from app.server import app
from app.web_session import (
    WebRoomHub,
    WebSessionOptions,
    WebTranslateBridge,
    _SafeWebSocketSender,
    normalize_room_id,
)


class _FakeLiveSession:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.enqueued = []
        self.stopped = False

    def enqueue_audio(self, pcm_bytes: bytes, *, silence: bool = False, has_speech: bool = False) -> None:
        self.enqueued.append((pcm_bytes, silence, has_speech))

    def set_microphone_muted(self, muted: bool) -> None:
        pass

    async def finish_input(self) -> None:
        pass

    def stop(self) -> None:
        self.stopped = True

    async def run(self) -> None:
        on_conn = self.kwargs.get("on_connection")
        if on_conn:
            await on_conn("ready")
        while not self.stopped:
            await asyncio.sleep(0.02)


class _FakeMiniMaxClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.texts = []
        self.stopped = False

    def enqueue_text(self, text: str) -> None:
        self.texts.append(text)

    def stop(self) -> None:
        self.stopped = True

    async def run(self) -> None:
        while not self.stopped:
            await asyncio.sleep(0.02)


class _FakeSender(_SafeWebSocketSender):
    def __init__(self):
        self.closed = False
        self._lock = asyncio.Lock()
        self.jsons = []
        self.blobs = []

    async def send_json(self, payload):
        self.jsons.append(payload)
        return True

    async def send_bytes(self, data: bytes):
        self.blobs.append(data)
        return True


def _make_wav_bytes(seconds: float, amp: int = 5000, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    n = int(seconds * rate)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        # Alternating sign produces audible square-ish tone with non-zero RMS
        samples = [amp if (i % 40) < 20 else -amp for i in range(n)]
        wf.writeframes(struct.pack(f"<{n}h", *samples))
    return buf.getvalue()


class StandaloneWebTests(unittest.IsolatedAsyncioTestCase):
    def test_gate_and_wav_validation_match_app_tuning(self):
        gate = ForegroundSpeechGate()
        loud = struct.pack("<320h", *([6000 if (i % 16) < 8 else -6000 for i in range(320)]))
        quiet = b"\x00" * 640
        out1 = gate.process(loud, 0.18)
        self.assertFalse(gate.is_open)
        self.assertEqual(len(out1[0]), 640)
        out2 = gate.process(loud, 0.18)
        self.assertTrue(gate.is_open)
        self.assertGreaterEqual(len(out2), 2)
        for _ in range(12):
            gate.process(quiet, 0.0001)
        self.assertFalse(gate.is_open)

        # Transient click should be rejected by impulse filter
        click = [0] * 320
        click[10] = 28000
        click[11] = -28000
        click_bytes = struct.pack("<320h", *click)
        self.assertTrue(looks_like_impulse(click_bytes, 0.06))

        validate_wav_bytes(_make_wav_bytes(10.5, amp=5000))
        with self.assertRaises(ValueError):
            validate_wav_bytes(_make_wav_bytes(3.0, amp=5000))
        with self.assertRaises(ValueError):
            validate_wav_bytes(_make_wav_bytes(11.0, amp=0))

    async def test_bridge_defaults_and_minimax_mode(self):
        self_events = []
        audio_events = []
        audio_bytes = []
        created_ds = []
        created_mm = []

        def _make_ds(**kw):
            inst = _FakeLiveSession(**kw)
            created_ds.append(inst)
            return inst

        def _make_mm(**kw):
            inst = _FakeMiniMaxClient(**kw)
            created_mm.append(inst)
            return inst

        opts = WebSessionOptions(
            source_lang="zh",
            target_lang="en",
            live_voice_clone=True,
            noise_gate=False,
            dashscope_api_key="byo-ds",
        )
        self.assertEqual(opts.vad_silence_ms, settings.vad_silence_ms)
        bridge = WebTranslateBridge(
            opts,
            on_self_json=lambda p: self_events.append(p) or asyncio.sleep(0),
            on_audio_json=lambda p: audio_events.append(p) or asyncio.sleep(0),
            on_audio_bytes=lambda b: audio_bytes.append(b) or asyncio.sleep(0),
            session_factory=_make_ds,
            minimax_factory=_make_mm,
        )
        await bridge.start()
        try:
            await asyncio.sleep(0.05)
            self.assertEqual(created_ds[0].kwargs["silence_duration_ms"], 400)
            self.assertEqual(created_ds[0].kwargs["sample_rate"], 16000)
            bridge.feed_pcm(struct.pack("<320h", *([7000] * 320)))
            self.assertTrue(created_ds[0].enqueued[0][2])
            await created_ds[0].kwargs["on_audio_chunk"](b"\x01\x02")
            self.assertEqual(audio_events[0]["sample_rate"], 24000)
            self.assertEqual(audio_bytes[0], b"\x01\x02")
        finally:
            await bridge.stop()

    async def test_room_hub_isolation(self):
        hub = WebRoomHub()
        s1 = _FakeSender()
        s2 = _FakeSender()
        p1 = await hub.join("Room #1", "p1", "医生", "zh", "en", s1)
        p2 = await hub.join("Room #1", "p2", "患者", "en", "zh", s2)
        self.assertEqual(normalize_room_id("Room #1"), "Room1")
        s1.jsons.clear()
        s2.jsons.clear()
        await hub.relay_peer_json("Room #1", p1, {"type": "target_final", "target": "Take a rest"})
        await hub.relay_peer_bytes("Room #1", p1, b"\xaa\xbb")
        self.assertEqual(s1.jsons, [])
        self.assertEqual(s2.jsons[0]["speaker_name"], "医生")
        self.assertEqual(s2.blobs, [b"\xaa\xbb"])
        await hub.leave("Room #1", p1.peer_id)
        await hub.leave("Room #1", p2.peer_id)


class StandaloneWebHttpTests(unittest.TestCase):
    def test_static_frontend_and_ws_endpoints(self):
        with TestClient(app) as client:
            r_home = client.get("/")
            self.assertEqual(r_home.status_code, 200)
            self.assertIn("EchoBridge", r_home.text)

            r_wk = client.get("/audio-worklet.js")
            self.assertEqual(r_wk.status_code, 200)
            self.assertIn("pcm16-downsample-processor", r_wk.text)

            r_cfg = client.get("/api/web/config")
            self.assertEqual(r_cfg.status_code, 200)
            self.assertEqual(r_cfg.json()["vad_silence_ms"], 400)

            self.assertEqual(client.get("/samples/demo_zh.wav").content[:4], b"RIFF")
            self.assertEqual(client.get("/samples/demo_en.wav").content[:4], b"RIFF")

            with client.websocket_connect("/ws/web-translate") as ws:
                hello = ws.receive_json()
                self.assertEqual(hello["type"], "hello")
                self.assertEqual(hello["vad_silence_ms"], 400)

            with client.websocket_connect("/ws/room/qoder-room") as ws:
                self.assertEqual(ws.receive_json()["type"], "room_peers")
                joined = ws.receive_json()
                self.assertEqual(joined["room_id"], "qoder-room")
