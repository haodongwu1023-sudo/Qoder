"""Bidirectional live simultaneous translation session manager (Outbound + Inbound + AEC)."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import os
import queue
import re
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from fastapi import WebSocket

from src.asr_client import DashScopeRealtimeASR, RealtimeASRStream
from src.audio_device import (
    VIRTUAL_DEVICE_HINTS,
    close_portaudio_stream,
    list_audio_devices,
    open_portaudio_stream,
)
from src.audio_io import (
    ForegroundSpeechGate,
    RealtimeMicVAD,
    compute_rms_int16,
    resample_pcm16_mono,
    sounddevice_frame_reader,
)
from src.config import settings
from src.glossary_store import GlossaryStore
from src.history_store import HistoryStore
from src.livetranslate_client import AlibabaLiveTranslateSession, LiveTranslateError
from src.native_audio import (
    NativeSpeaker,
    NativeSystemTap,
    NativeVPIODuplex,
    is_hal_driver_installed,
    is_native_engine_available,
)
from src.pipeline import (
    PersistentMiniMaxTTSClient,
    SplitTranslationPipeline,
    apply_glossary_terms,
)
from src.ring_buffer import (
    DEFAULT_MEETING_RING,
    DEFAULT_MIC_RING,
    HEADER_FORMAT,
    HEADER_SIZE,
    VC_RING_MAGIC,
    SharedAudioRingClient,
    StreamingStereoResampler,
    apply_pcm16_fade_in,
)
from src.voice_library import VoiceLibrary

logger = logging.getLogger(__name__)


# Split explicit interruption markers only after a sentence boundary.
_INTERRUPTION = re.compile(
    r"(?<=[。！？.!?])\s*(?=(?:等一下|等一等|稍等一下|等等[，,]|"
    r"(?:wait a (?:moment|second|minute)|hold on|one moment)[,!. ]))",
    re.IGNORECASE,
)


def _subtitle_pairs(source: str, target: str) -> list[tuple[str, str]]:
    left = [part.strip() for part in _INTERRUPTION.split(source)]
    right = [part.strip() for part in _INTERRUPTION.split(target)]
    return list(zip(left, right)) if len(left) == len(right) else [(source, target)]


class MicrophoneNoSignal(RuntimeError):
    """Capture returned no usable samples during startup."""


_resample_pcm16_mono = resample_pcm16_mono


def _describe_input_device(device_index: int | None) -> tuple[str, float | None, bool]:
    """Return (display_name, default_samplerate, is_bluetooth_hfp_suspected) for the active mic."""
    try:
        import sounddevice as sd

        idx = device_index
        is_default = idx is None
        if is_default:
            # sounddevice.default.device is an _InputOutputPair, not a tuple.
            info = sd.query_devices(kind="input")
            idx = info["index"]
        elif idx is not None and idx >= 0:
            info = sd.query_devices(idx, kind="input")
        else:
            return ("系统默认麦克风", None, False)
        name = str(info.get("name", f"设备 #{idx}"))
        sr = float(info.get("default_samplerate") or 0.0) or None
        label = f"{name} (系统默认)" if is_default else name
        # macOS switches Bluetooth headsets into 16kHz/24kHz HFP SCO profile when mic is opened.
        lower = name.lower()
        looks_bt_name = any(k in lower for k in ("耳机", "airpods", "buds", "headset", "bluetooth", "wh-", "wf-"))
        is_bt_hfp = bool((sr is not None and sr < 44100.0) or (is_default and looks_bt_name and (sr is None or sr <= 24000.0)))
        return (label, sr, is_bt_hfp)
    except Exception:
        return ("默认麦克风" if device_index is None else f"设备 #{device_index}", None, False)


class _PersistentOutputSink:
    """Callback-driven jitter-buffered sounddevice RawOutputStream for stutter-free streaming playback."""

    def __init__(
        self, device_index: int | None, sample_rate: int, prebuffer_ms: int = 180,
        recovery_ms: int = 60,
    ) -> None:
        self.device_index = device_index
        self.sample_rate = sample_rate
        self._prebuffer_bytes = max(960, int(sample_rate * 2 * prebuffer_ms / 1000))
        self._stream: Any = None
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._buffering = True
        self._flush_tail = False
        self._min_recovery_bytes = int(sample_rate * 2 * recovery_ms / 1000)
        self._max_recovery_bytes = max(self._min_recovery_bytes, int(sample_rate * 2 * 0.280))
        self._recovery_bytes = self._min_recovery_bytes
        self._recovering = False
        self._pending_since: float | None = None
        self._underruns = 0
        self._hardware_underflows = 0
        self._last_chunk_at: float | None = None
        self._max_active_gap = 0.0
        self._turn_active = False
        self._has_gap = False
        self._startup_ms: float | None = None
        self._clean_turns = 0

    def _callback(self, outdata: bytearray, frames: int, time_info: Any, status: Any) -> None:
        need = frames * 2
        with self._lock:
            if getattr(status, "output_underflow", False):
                self._hardware_underflows += 1
            avail = len(self._buf)
            if self._buffering:
                threshold = self._recovery_bytes if self._recovering else self._prebuffer_bytes
                refill_seconds = threshold / (2 * self.sample_rate)
                max_wait = max(0.085 if self._recovering else 0.110, refill_seconds + 0.020)
                waited_long_enough = (
                    self._pending_since is not None
                    and time.monotonic() - self._pending_since >= max_wait
                )
                if avail and (avail >= threshold or self._flush_tail or waited_long_enough):
                    self._buffering = False
                    self._pending_since = None
                else:
                    outdata[:] = b"\x00" * need
                    return
            if avail >= need:
                outdata[:] = self._buf[:need]
                del self._buf[:need]
            else:
                outdata[:avail] = self._buf
                outdata[avail:] = b"\x00" * (need - avail)
                self._buf.clear()
                self._buffering = True
                self._recovering = self._turn_active and not self._flush_tail
                if self._recovering:
                    self._underruns += 1
                    self._has_gap = True
                    self._recovery_bytes = min(
                        self._max_recovery_bytes,
                        self._recovery_bytes + int(self.sample_rate * 2 * 0.045),
                    )
                self._flush_tail = False
                self._pending_since = None

    def _ensure_stream(self) -> None:
        if self._stream is None:
            import sounddevice as sd

            self._stream = open_portaudio_stream(
                lambda: sd.RawOutputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="int16",
                    device=self.device_index,
                    blocksize=0,
                    callback=self._callback,
                ),
                start=True,
            )

    async def write_live(self, chunk: bytes) -> None:
        if self._stream is not None:
            self.write(chunk)
            return
        opening = asyncio.create_task(asyncio.to_thread(self.write, chunk))
        try:
            await asyncio.shield(opening)
        except asyncio.CancelledError:
            await asyncio.gather(opening, return_exceptions=True)
            raise

    def write(self, chunk: bytes) -> None:
        if not chunk:
            return
        now = time.monotonic()
        cold = self._stream is None
        self._ensure_stream()
        with self._lock:
            if cold:
                self._startup_ms = (time.monotonic() - now) * 1000
            if self._last_chunk_at is not None:
                self._max_active_gap = max(self._max_active_gap, now - self._last_chunk_at)
            self._last_chunk_at = now
            self._turn_active = True
            # Absorb bounded bursts of faster-than-real-time synthesis.
            max_buf = self.sample_rate * 2 * 12
            excess = max(0, len(self._buf) + len(chunk) - max_buf)
            if excess:
                excess += excess % 2
                if excess >= len(self._buf):
                    chunk = chunk[excess - len(self._buf):]
                    self._buf.clear()
                else:
                    del self._buf[:excess]
            if self._buffering and not self._buf:
                self._pending_since = time.monotonic()
            self._buf.extend(chunk)
            self._flush_tail = False

    def flush(self) -> None:
        """Allow any remaining <prebuffer_ms tail audio to drain immediately."""
        with self._lock:
            if self._turn_active:
                self._clean_turns = 0 if self._has_gap else self._clean_turns + 1
                if self._clean_turns >= 4:
                    self._recovery_bytes = max(
                        self._min_recovery_bytes,
                        self._recovery_bytes - int(self.sample_rate * 2 * 0.015),
                    )
                    self._clean_turns = 0
                self._turn_active = False
                self._has_gap = False
                self._last_chunk_at = None
            if self._buf or not self._buffering:
                self._flush_tail = True
            else:
                self._recovering = False

    def close(self) -> None:
        stream = self._stream
        if stream is not None:
            logger.info(
                "Speaker playback: gaps=%d device_underflows=%d max_packet_gap_ms=%.0f device_start_ms=%.0f",
                self._underruns, self._hardware_underflows, self._max_active_gap * 1000,
                self._startup_ms or 0.0,
            )
        self._stream = None
        with self._lock:
            self._buf.clear()
            self._buffering = True
            self._recovering = False
            self._flush_tail = False
            self._pending_since = None
        if stream is not None:
            close_portaudio_stream(stream)


class _MicRingJitterSink:
    """Jitter-buffered, overrun-paced writer for the VoiceClone Mic shared ring."""

    def __init__(
        self,
        ring: SharedAudioRingClient,
        sample_rate: int = 24000,
        *,
        prebuffer_ms: int = 350,
        min_recovery_ms: int = 260,
        max_recovery_ms: int = 480,
        initial_max_wait_s: float = 0.140,
        recovery_max_wait_s: float = 0.280,
        target_ring_sec: float = 0.65,
    ) -> None:
        self.ring = ring
        self.sample_rate = max(1, int(sample_rate))
        self._prebuffer_bytes = max(960, int(self.sample_rate * 2 * prebuffer_ms / 1000))
        self._min_recovery_bytes = max(960, int(self.sample_rate * 2 * min_recovery_ms / 1000))
        self._max_recovery_bytes = max(
            self._min_recovery_bytes, int(self.sample_rate * 2 * max_recovery_ms / 1000)
        )
        self._recovery_bytes = self._min_recovery_bytes
        self._initial_max_wait_s = initial_max_wait_s
        self._recovery_max_wait_s = recovery_max_wait_s
        self._target_ring_sec = target_ring_sec
        self._slice_bytes = max(1920, int(self.sample_rate * 2 * 0.160))

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._buf = bytearray()
        self._buffering = True
        self._recovering = False
        self._flush_tail = False
        self._flush_resampler_pending = False
        self._turn_active = False
        self._has_gap = False
        self._clean_turns = 0
        self._pending_since: float | None = None
        self._last_chunk_at: float | None = None
        self._max_active_gap = 0.0
        self._underruns = 0
        self._underruns_prevented = 0
        self._needs_fade_in = True
        self._resampler = StreamingStereoResampler(src_rate=self.sample_rate, dst_rate=48000)
        self._closed = False
        self._thread: threading.Thread | None = None

    def _ensure_started(self) -> None:
        if self._closed:
            return
        self.ring.open(sample_rate=48000)
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._writer_loop,
                    name="mic-ring-jitter-sink",
                    daemon=True,
                )
                self._thread.start()

    def write(self, chunk: bytes) -> None:
        """Enqueue one Int16 mono TTS chunk for smoothed delivery to VoiceClone Mic."""
        if not chunk or self._closed:
            return
        now = time.monotonic()
        self._ensure_started()
        with self._cond:
            if self._closed:
                return
            if self._last_chunk_at is not None:
                dt = now - self._last_chunk_at
                self._max_active_gap = max(self._max_active_gap, dt)
                if dt > 0.35 and not self._buffering and self._turn_active:
                    self._underruns_prevented += 1
            self._last_chunk_at = now
            self._turn_active = True

            # Keep bounded user-space staging (up to 12s) so mic.ring never overflows 4s.
            max_buf = self.sample_rate * 2 * 12
            excess = max(0, len(self._buf) + len(chunk) - max_buf)
            if excess:
                excess += excess % 2
                if excess >= len(self._buf):
                    chunk = chunk[excess - len(self._buf) :]
                    self._buf.clear()
                else:
                    del self._buf[:excess]
            if self._buffering and not self._buf:
                self._pending_since = now
            self._buf.extend(chunk)
            self._flush_tail = False
            self._cond.notify_all()

    def flush(self) -> None:
        """Allow any buffered turn tail (<prebuffer_ms) to drain immediately."""
        with self._cond:
            if self._turn_active:
                self._clean_turns = 0 if self._has_gap else self._clean_turns + 1
                if self._clean_turns >= 3:
                    self._recovery_bytes = max(
                        self._min_recovery_bytes,
                        self._recovery_bytes - int(self.sample_rate * 2 * 0.025),
                    )
                    self._clean_turns = 0
                self._turn_active = False
                self._has_gap = False
                self._last_chunk_at = None
            if self._buf:
                self._flush_tail = True
            else:
                self._buffering = True
                self._recovering = False
                self._pending_since = None
                self._flush_resampler_pending = True
            self._cond.notify_all()

    def pending_seconds(self) -> float:
        """Return total audio seconds queued in user space and active HAL ring."""
        with self._lock:
            buf_sec = (len(self._buf) // 2) / float(self.sample_rate)
        try:
            _, reserved, _, _ = self.ring.ring_status()
        except Exception:
            reserved = 0
        ring_sec = self.ring.buffered_seconds() if reserved > 0 else 0.0
        return buf_sec + ring_sec

    def _writer_loop(self) -> None:
        while True:
            write_pcm: bytes | None = None
            flush_resampler = False
            ring_rate = 48000
            with self._cond:
                while not self._closed:
                    try:
                        ring_rate, reserved, _, _ = self.ring.ring_status()
                    except Exception:
                        ring_rate, reserved = 48000, 0
                    ring_rate = ring_rate or 48000
                    active_consumer = reserved > 0
                    ring_buf_sec = self.ring.buffered_seconds() if active_consumer else 0.0

                    if self._flush_resampler_pending:
                        self._flush_resampler_pending = False
                        flush_resampler = True
                        break

                    avail = len(self._buf)

                    # When no CoreAudio client is reading VoiceClone Mic, do not block waiting on read_pos.
                    if not active_consumer:
                        if avail > 0:
                            take = min(avail, self._slice_bytes)
                            take -= take % 2
                            if take <= 0:
                                take = min(avail, 2)
                            write_pcm = bytes(self._buf[:take])
                            del self._buf[:take]
                            self._buffering = False
                            flush_resampler = self._flush_tail and len(self._buf) == 0
                            if flush_resampler:
                                self._flush_tail = False
                                self._buffering = True
                            break
                        if self._flush_tail:
                            self._flush_tail = False
                            self._buffering = True
                            self._recovering = False
                            flush_resampler = True
                            break
                        self._cond.wait(timeout=0.05)
                        continue

                    # Active CoreAudio consumer reading VoiceClone Mic:
                    if avail == 0:
                        if self._flush_tail:
                            self._flush_tail = False
                            self._buffering = True
                            self._recovering = False
                            flush_resampler = True
                            break
                        if (
                            not self._buffering
                            and self._turn_active
                            and ring_buf_sec <= 0.020
                        ):
                            self._buffering = True
                            self._recovering = True
                            self._needs_fade_in = True
                            self._resampler.reset()
                            self._underruns += 1
                            self._has_gap = True
                            self._recovery_bytes = min(
                                self._max_recovery_bytes,
                                self._recovery_bytes + int(self.sample_rate * 2 * 0.050),
                            )
                            self._pending_since = None
                        self._cond.wait(timeout=0.015)
                        continue

                    if self._buffering:
                        threshold = (
                            self._recovery_bytes if self._recovering else self._prebuffer_bytes
                        )
                        max_wait = (
                            self._recovery_max_wait_s
                            if self._recovering
                            else self._initial_max_wait_s
                        )
                        elapsed = (
                            time.monotonic() - self._pending_since
                            if self._pending_since is not None
                            else 0.0
                        )
                        waited_enough = (
                            self._pending_since is not None and elapsed >= max_wait
                        )
                        if avail >= threshold or self._flush_tail or waited_enough:
                            self._buffering = False
                            self._recovering = False
                            self._pending_since = None
                        else:
                            remaining = max(0.005, max_wait - elapsed)
                            self._cond.wait(timeout=min(0.015, remaining))
                            continue

                    # Pace writes into mic.ring so buffered audio never overflows 4s cap.
                    if ring_buf_sec >= self._target_ring_sec:
                        self._cond.wait(timeout=0.015)
                        continue

                    room_sec = max(0.080, self._target_ring_sec - ring_buf_sec)
                    max_pcm = int(self.sample_rate * 2 * room_sec)
                    take = min(avail, max(self._slice_bytes, max_pcm))
                    take -= take % 2
                    if take <= 0:
                        take = min(avail, 2)
                    write_pcm = bytes(self._buf[:take])
                    del self._buf[:take]
                    flush_resampler = self._flush_tail and len(self._buf) == 0
                    if flush_resampler:
                        self._flush_tail = False
                        self._buffering = True
                        self._recovering = False
                    break

                if self._closed:
                    return

            stereo: list[float] = []
            if write_pcm:
                if self._needs_fade_in:
                    write_pcm = apply_pcm16_fade_in(write_pcm, self.sample_rate, fade_ms=4.0)
                    self._needs_fade_in = False
                stereo = self._resampler.process(write_pcm, self.sample_rate, ring_rate)
            if flush_resampler:
                stereo.extend(self._resampler.flush())
                self._needs_fade_in = True
            if stereo:
                try:
                    self.ring.write_stereo_floats(stereo)
                except Exception:
                    logger.debug("Mic ring write error", exc_info=True)

    def close(self) -> None:
        thread = self._thread
        with self._cond:
            self._closed = True
            self._buf.clear()
            self._buffering = True
            self._recovering = False
            self._flush_tail = False
            self._flush_resampler_pending = False
            self._pending_since = None
            self._cond.notify_all()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.5)
            logger.info(
                "Mic ring playback: gaps=%d underruns_prevented=%d max_packet_gap_ms=%.0f",
                self._underruns,
                self._underruns_prevented,
                self._max_active_gap * 1000,
            )
        self._thread = None
        self._resampler.reset()


def _read_ring_header(path: Any) -> tuple[int, int]:
    """Read (reserved_io_count, write_pos) from a ring file without creating it."""
    try:
        with open(path, "rb") as fp:
            raw = fp.read(HEADER_SIZE)
        if len(raw) < HEADER_SIZE:
            return (0, 0)
        magic, _, _, _, _, reserved, written, _ = struct.unpack_from(HEADER_FORMAT, raw, 0)
        if magic != VC_RING_MAGIC:
            return (0, 0)
        return (int(reserved), int(written))
    except OSError:
        return (0, 0)


def _is_virtual_device_name(name: str) -> bool:
    low = name.lower()
    return any(hint in low for hint in VIRTUAL_DEVICE_HINTS) or "变色龙" in name


def _resolve_physical_devices(input_idx: int | None, output_idx: int | None) -> tuple[int | None, int | None]:
    """Avoid looping virtual devices back onto themselves or picking stale indices during passthrough."""
    try:
        import sounddevice as sd

        devs = sd.query_devices()

        def _is_virtual_or_missing(idx: int | None, kind: str) -> bool:
            if idx is None:
                info = sd.query_devices(kind=kind)
            elif 0 <= idx < len(devs):
                info = devs[idx]
                cap_key = "max_input_channels" if kind == "input" else "max_output_channels"
                if int(info.get(cap_key, 0) or 0) <= 0:
                    return True
            else:
                return True
            return _is_virtual_device_name(str(info.get("name", "")))

        def _pick_physical(kind: str, current: int | None) -> int | None:
            if not _is_virtual_or_missing(current, kind):
                return current
            if not _is_virtual_or_missing(None, kind):
                return None
            cap_key = "max_input_channels" if kind == "input" else "max_output_channels"
            candidates = [
                (i, str(d.get("name", "")))
                for i, d in enumerate(devs)
                if int(d.get(cap_key, 0) or 0) > 0 and not _is_virtual_device_name(str(d.get("name", "")))
            ]
            # Avoid silently grabbing iPhone Continuity mic ahead of local hardware when default is unavailable
            preferred = next((i for i, n in candidates if "iphone" not in n.lower()), None)
            if preferred is not None:
                return preferred
            return candidates[0][0] if candidates else current

        input_idx = _pick_physical("input", input_idx)
        output_idx = _pick_physical("output", output_idx)
    except Exception:
        pass
    return input_idx, output_idx


class IdlePassthroughController:
    """Bridges physical mic <-> VoiceClone rings when translation is stopped in Meeting Mode."""

    def __init__(self, is_session_busy: Any) -> None:
        self._is_session_busy = is_session_busy
        self._user_enabled = False
        self._paused_for_session = False
        self._suspend_depth = 0
        self.input_device_index: int | None = None
        self.output_device_index: int | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._active = False
        self._mic_clients = 0
        self._meeting_clients = 0
        self._hal_active = False
        self._last_meeting_written = 0
        self._meeting_stream_active = False
        self._last_mic_read_active = 0.0

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self._user_enabled,
            "active": self._active,
            "hal_active": self._hal_active,
            "mic_clients": self._mic_clients,
            "meeting_clients": self._meeting_clients,
            "input_device_index": self.input_device_index,
            "output_device_index": self.output_device_index,
        }

    def probe_rings(self) -> bool:
        mic_clients, mic_written = _read_ring_header(DEFAULT_MIC_RING)
        meeting_clients, meeting_written = _read_ring_header(DEFAULT_MEETING_RING)
        if self._last_meeting_written > 0 and meeting_written > self._last_meeting_written:
            self._meeting_stream_active = True
            self._last_mic_read_active = time.monotonic()
        elif time.monotonic() - self._last_mic_read_active > 4.0:
            self._meeting_stream_active = False
        self._last_meeting_written = meeting_written
        self._mic_clients = mic_clients
        self._meeting_clients = meeting_clients
        self._hal_active = (mic_clients > 0 or meeting_clients > 0 or self._meeting_stream_active)
        return self._hal_active

    def _should_run(self) -> bool:
        if self._paused_for_session or self._is_session_busy():
            return False
        if not is_hal_driver_installed():
            return False
        # Only bridge audio while a meeting/call app is actively using a VoiceClone HAL device;
        # never keep the physical microphone open when the app is simply sitting idle.
        return bool(self._user_enabled and self._hal_active)

    def sync(self) -> dict[str, Any]:
        want = self._should_run()
        with self._lock:
            running = self._thread is not None and self._thread.is_alive()
            if want and not running:
                self._stop_event.clear()
                self._thread = threading.Thread(
                    target=self._worker,
                    name="VoiceCloneIdlePassthrough",
                    daemon=True,
                )
                self._thread.start()
            elif not want and running:
                self._stop_event.set()
        return self.status()

    async def configure(
        self,
        *,
        enabled: bool | None = None,
        input_device_index: int | None = None,
        output_device_index: int | None = None,
        update_devices: bool = False,
    ) -> dict[str, Any]:
        devices_changed = False
        if enabled is not None:
            self._user_enabled = bool(enabled)
        if update_devices:
            if self.input_device_index != input_device_index or self.output_device_index != output_device_index:
                devices_changed = True
            self.input_device_index = input_device_index
            self.output_device_index = output_device_index
        if devices_changed and self._thread is not None and self._thread.is_alive():
            await self.pause()
            self._paused_for_session = False
        self.probe_rings()
        return self.sync()

    async def pause(self) -> None:
        self._paused_for_session = True
        with self._lock:
            self._stop_event.set()
            thread = self._thread
        if thread is not None and thread.is_alive():
            await asyncio.to_thread(thread.join, 3.0)
        with self._lock:
            if self._thread is not None and not self._thread.is_alive():
                self._thread = None
                self._active = False

    async def resume(self) -> None:
        self._paused_for_session = False
        self.probe_rings()
        self.sync()

    @asynccontextmanager
    async def suspend_temporarily(self) -> AsyncIterator[None]:
        """Pause idle passthrough cleanly while PortAudio refreshes, then restore previous state."""
        was_paused = self._paused_for_session
        self._suspend_depth += 1
        await self.pause()
        try:
            yield
        finally:
            self._suspend_depth = max(0, self._suspend_depth - 1)
            if self._suspend_depth == 0 and not was_paused and not self._is_session_busy():
                await self.resume()

    def _open_passthrough_input(
        self,
        sd: Any,
        phys_in: int | None,
        callback: Any | None = None,
    ) -> tuple[Any, int, int]:
        """Open physical mic input for idle passthrough, falling back to native sample rate if 16k is rejected."""
        target_sr = settings.audio_sample_rate

        def _make_and_start(rate: int) -> tuple[Any, int, int]:
            blk = max(160, int(rate * 30 / 1000))
            kwargs: dict[str, Any] = {
                "samplerate": rate,
                "channels": 1,
                "dtype": "int16",
                "blocksize": blk,
                "device": phys_in,
            }
            if callback is not None:
                kwargs["callback"] = callback
            stream = open_portaudio_stream(lambda kw=kwargs: sd.RawInputStream(**kw), start=True)
            return stream, int(stream.samplerate), blk

        try:
            return _make_and_start(target_sr)
        except Exception as exc:
            info = sd.query_devices(phys_in, kind="input") if phys_in is None else sd.query_devices(phys_in)
            native_sr = int(round(float(info.get("default_samplerate", 48000) or 48000)))
            if native_sr != target_sr and native_sr > 0:
                return _make_and_start(native_sr)
            raise exc

    def _worker(self) -> None:
        mic_ring: SharedAudioRingClient | None = None
        meeting_ring: SharedAudioRingClient | None = None
        in_stream: Any = None
        in_sr = settings.audio_sample_rate
        out_sink: _PersistentOutputSink | None = None
        in_queue: queue.Queue[bytes] = queue.Queue(maxsize=32)
        self._active = True

        def _clear_in_queue() -> None:
            while True:
                try:
                    in_queue.get_nowait()
                except queue.Empty:
                    break

        def _on_in_audio(indata: Any, frames: int, time_info: Any, status: Any) -> None:
            if self._stop_event.is_set():
                return
            raw = bytes(indata)
            if not raw:
                return
            try:
                in_queue.put_nowait(raw)
            except queue.Full:
                try:
                    in_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    in_queue.put_nowait(raw)
                except queue.Full:
                    pass

        try:
            import sounddevice as sd

            phys_in, phys_out = _resolve_physical_devices(
                self.input_device_index, self.output_device_index
            )
            mic_ring = SharedAudioRingClient(DEFAULT_MIC_RING)
            mic_ring.open(sample_rate=48000)
            meeting_ring = SharedAudioRingClient(DEFAULT_MEETING_RING)
            meeting_ring.open(sample_rate=48000)
            last_in_attempt = 0.0
            pa_needs_refresh = False

            while (
                not self._stop_event.is_set()
                and not self._is_session_busy()
                and not self._paused_for_session
                and self._user_enabled
                and self._hal_active
            ):
                did_work = False
                if pa_needs_refresh and in_stream is None and out_sink is None:
                    pa_needs_refresh = False
                    list_audio_devices(refresh=True)
                _, _, mic_reserved, _, _ = mic_ring.header_stats()
                mic_active = mic_reserved > 0
                if not mic_active:
                    if in_stream is not None:
                        close_portaudio_stream(in_stream)
                        in_stream = None
                        _clear_in_queue()
                        last_in_attempt = 0.0
                elif in_stream is None and (time.monotonic() - last_in_attempt) >= 1.0:
                    last_in_attempt = time.monotonic()
                    _clear_in_queue()
                    try:
                        phys_in, phys_out = _resolve_physical_devices(
                            self.input_device_index, self.output_device_index
                        )
                        in_stream, in_sr, _ = self._open_passthrough_input(
                            sd, phys_in, callback=_on_in_audio
                        )
                    except Exception as exc:
                        logger.debug("Idle passthrough input stream unavailable: %s", exc)
                        in_stream = None
                        pa_needs_refresh = True

                if in_stream is not None:
                    if not getattr(in_stream, "active", True):
                        logger.debug("Idle passthrough input stream became inactive; will reopen.")
                        close_portaudio_stream(in_stream)
                        in_stream = None
                        _clear_in_queue()
                        pa_needs_refresh = True
                    else:
                        while True:
                            try:
                                raw_in = in_queue.get_nowait()
                            except queue.Empty:
                                break
                            if raw_in:
                                mic_ring.write_pcm16_mono(raw_in, in_sr)
                                did_work = True

                block = meeting_ring.read_pcm16_mono()
                if block:
                    try:
                        if out_sink is None:
                            _, phys_out = _resolve_physical_devices(
                                self.input_device_index, self.output_device_index
                            )
                            out_sink = _PersistentOutputSink(
                                phys_out, settings.audio_sample_rate, prebuffer_ms=40, recovery_ms=30
                            )
                        out_sink.write(block)
                        did_work = True
                    except Exception as exc:
                        logger.debug("Idle passthrough output write interrupted (%s); resetting sink.", exc)
                        if out_sink is not None:
                            out_sink.close()
                            out_sink = None
                        pa_needs_refresh = True
                if not did_work:
                    self._stop_event.wait(0.01)
        except Exception as exc:
            logger.debug("Idle passthrough stopped: %s", exc)
        finally:
            self._active = False
            close_portaudio_stream(in_stream)
            if out_sink is not None:
                out_sink.close()
            if mic_ring is not None:
                mic_ring.close()
            if meeting_ring is not None:
                meeting_ring.close()


@dataclass
class SessionConfig:
    source_lang: str = "zh"
    target_lang: str = "en"
    engine_mode: str = "alibaba_live"  # "alibaba_live" (simultaneous) or "split_minimax" (custom clone + glossary)
    live_voice_clone: bool = True
    voice_id: str = ""
    speed: float = 1.0
    input_device_index: int | None = None
    output_device_index: int | None = None
    output_sink: str = "device"  # "device" or "voiceclone_mic"
    noise_suppression: bool = False
    use_hardware_aec: bool = True
    echo_guard: bool = True
    enable_inbound: bool = False
    passthrough_inbound: bool = False
    inbound_device_index: int | None = None
    inbound_voice_id: str = ""
    vad_silence_ms: int = 400
    idle_auto_stop_minutes: int = 10
    glossary: dict[str, str] = field(default_factory=dict)


class TranslationSessionManager:
    """Coordinates bidirectional capture (VPIO AEC / ProcessTap), VAD, ASR, MT, TTS, and history."""

    def __init__(
        self,
        voice_library: VoiceLibrary,
        glossary_store: GlossaryStore,
        history_store: HistoryStore,
    ) -> None:
        self.voice_library = voice_library
        self.glossary_store = glossary_store
        self.history_store = history_store
        self.asr = DashScopeRealtimeASR()
        self.pipeline = SplitTranslationPipeline()
        self._vad: RealtimeMicVAD | None = None
        self._inbound_vad: RealtimeMicVAD | None = None
        self._outbound_asr: RealtimeASRStream | None = None
        self._inbound_asr: RealtimeASRStream | None = None
        self._live_outbound: AlibabaLiveTranslateSession | None = None
        self._live_inbound: AlibabaLiveTranslateSession | None = None
        self._vpio: NativeVPIODuplex | None = None
        self._system_tap: NativeSystemTap | None = None
        self._mic_ring: SharedAudioRingClient | None = None
        self._mic_jitter_sink: _MicRingJitterSink | None = None
        self._outbound_sink: _PersistentOutputSink | None = None
        self._inbound_sink: _PersistentOutputSink | None = None
        self._outbound_tts_queue: asyncio.Queue[tuple[str, str, str, bool]] | None = None
        self._inbound_tts_queue: asyncio.Queue[tuple[str, str, str, bool]] | None = None
        self._outbound_tts_task: asyncio.Task[None] | None = None
        self._inbound_tts_task: asyncio.Task[None] | None = None
        self._warmup_task: asyncio.Task[None] | None = None
        self._task: asyncio.Task[None] | None = None
        self._inbound_task: asyncio.Task[None] | None = None
        self._watchdog_task: asyncio.Task[None] | None = None
        self.microphone_muted = False
        self._draining = False
        self._outbound_cloud_ready = False
        self._outbound_input_ready = False
        self.status_message = "未启动"
        self._play_until: dict[str, float] = {}
        self._clients: dict[WebSocket, asyncio.Queue[dict[str, Any]]] = {}
        self.running = False
        self.config = SessionConfig()
        self.idle_passthrough = IdlePassthroughController(lambda: self.running or self._draining)
        self._current_session_id = ""
        self._last_speech_ts = 0.0
        self._zero_level_streak = 0
        self._low_level_streak = 0
        self._low_level_warned = False
        self._heard_real_speech = False
        self._active_input_name = "系统默认麦克风"
        self._active_input_is_bt_hfp = False

    @property
    def session_ready(self) -> bool:
        return bool(self.running and not self._draining and self._outbound_cloud_ready and self._outbound_input_ready)

    def register_ws(self, ws: WebSocket) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=64)
        self._clients[ws] = queue
        return queue

    def unregister_ws(self, ws: WebSocket) -> None:
        self._clients.pop(ws, None)

    async def broadcast(self, event: dict[str, Any]) -> None:
        if event.get("type") == "status":
            self.status_message = event["message"]
            if "ready" not in event:
                event = {**event, "ready": self.session_ready}
        for queue in self._clients.values():
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(event)

    async def set_passthrough_inbound(self, enabled: bool) -> bool:
        new_val = bool(enabled)
        prev_val = self.config.passthrough_inbound
        self.config.passthrough_inbound = new_val
        await self.broadcast({"type": "passthrough_updated", "enabled": new_val})
        if (
            self.running
            and self.config.enable_inbound
            and prev_val != new_val
            and not new_val
            and self._live_inbound is not None
            and not self._live_inbound.enable_audio
        ):
            old_task = self._inbound_task
            self._inbound_task = None
            if old_task is not None and not old_task.done():
                old_task.cancel()
                try:
                    await old_task
                except asyncio.CancelledError:
                    pass
            self._inbound_task = asyncio.create_task(self._run_inbound_loop())
        return new_val

    async def update_languages(self, source_lang: str, target_lang: str) -> None:
        if self._draining:
            return
        self.config.source_lang = source_lang
        self.config.target_lang = target_lang
        await self.broadcast(
            {
                "type": "languages_updated",
                "source_lang": source_lang,
                "target_lang": target_lang,
            }
        )
        if self.running and self.config.engine_mode == "alibaba_live":
            await self.start(self.config, preserve_mute=True)

    def _refresh_and_validate_config(self) -> None:
        """Reload PortAudio device IDs before opening streams and drop stale/virtual indices."""
        devices = list_audio_devices(refresh=True)
        inputs = {d["index"]: d for d in devices.get("inputs", [])}
        outputs = {d["index"]: d for d in devices.get("outputs", [])}
        if self.config.input_device_index is not None:
            dev = inputs.get(self.config.input_device_index)
            if dev is None or dev.get("is_virtual_candidate") or _is_virtual_device_name(str(dev.get("name", ""))):
                logger.info(
                    "Resetting invalid/virtual input device %s to system default physical mic",
                    self.config.input_device_index,
                )
                self.config.input_device_index = None
        if self.config.output_device_index is not None:
            dev = outputs.get(self.config.output_device_index)
            if dev is None or (self.config.output_sink == "device" and _is_virtual_device_name(str(dev.get("name", "")))):
                logger.info(
                    "Resetting invalid output device %s to system default speaker",
                    self.config.output_device_index,
                )
                self.config.output_device_index = None

    async def start(self, cfg: SessionConfig, *, preserve_mute: bool = False) -> None:
        if self.running:
            await self.stop()
        await self.idle_passthrough.pause()
        self.config = cfg
        await asyncio.to_thread(self._refresh_and_validate_config)
        self._outbound_cloud_ready = False
        self._outbound_input_ready = False
        self._play_until.clear()
        await self.set_microphone_muted(self.microphone_muted if preserve_mute else False)
        self.config.engine_mode = "alibaba_live"
        self.config.voice_id = "__alibaba_live_clone__"
        if not self.config.glossary:
            self.config.glossary = self.glossary_store.list_terms()

        is_live = True
        rms_threshold = 0.012 if cfg.noise_suppression else 0.005
        if is_live:
            self._vad = None
            self._inbound_vad = None
            self._outbound_asr = None
            self._inbound_asr = None
            self._outbound_tts_queue = None
            self._inbound_tts_queue = None
        else:
            self._vad = RealtimeMicVAD(
                sample_rate=settings.audio_sample_rate,
                rms_threshold=rms_threshold,
                device_index=cfg.input_device_index,
            )
            self._inbound_vad = (
                RealtimeMicVAD(
                    sample_rate=settings.audio_sample_rate,
                    rms_threshold=0.004,
                    device_index=cfg.inbound_device_index,
                )
                if cfg.enable_inbound
                else None
            )
            self._outbound_asr = self.asr.create_stream()
            self._inbound_asr = self.asr.create_stream() if cfg.enable_inbound else None
            self._outbound_tts_queue = asyncio.Queue(maxsize=16)
            self._inbound_tts_queue = asyncio.Queue(maxsize=16) if cfg.enable_inbound else None

        self._zero_level_streak = 0
        self._low_level_streak = 0
        self._low_level_warned = False
        self._heard_real_speech = False
        self._active_input_name, _, self._active_input_is_bt_hfp = _describe_input_device(
            cfg.input_device_index
        )
        self._current_session_id = await asyncio.to_thread(
            self.history_store.start_session,
            source_lang=cfg.source_lang,
            target_lang=cfg.target_lang,
            voice_id=cfg.voice_id,
        )
        self._last_speech_ts = time.monotonic()
        self.running = True
        mode_tag = "极速实时声纹克隆" if cfg.live_voice_clone else "极速实时同传"
        dir_desc = "双向同传已启动" if cfg.enable_inbound else "去向同传已启动"
        await self.broadcast(
            {
                "type": "status",
                "running": True,
                "message": f"{dir_desc}（{mode_tag}｜采集: {self._active_input_name}），请说一句测试麦克风…",
            }
        )
        if not is_live:
            self._warmup_task = asyncio.create_task(self._warmup_connections())
            self._outbound_tts_task = asyncio.create_task(self._run_tts_worker("outbound"))
            if cfg.enable_inbound and self._inbound_tts_queue is not None:
                self._inbound_tts_task = asyncio.create_task(self._run_tts_worker("inbound"))
        self._task = asyncio.create_task(self._run_outbound_loop())
        if cfg.enable_inbound:
            self._inbound_task = asyncio.create_task(self._run_inbound_loop())
        if cfg.idle_auto_stop_minutes > 0:
            self._watchdog_task = asyncio.create_task(self._idle_watchdog())

    async def _warmup_connections(self) -> None:
        tasks = [self.pipeline.warmup()]
        if self._outbound_asr is not None:
            tasks.append(
                self._outbound_asr.warmup(self.config.source_lang, settings.audio_sample_rate)
            )
        if self._inbound_asr is not None:
            tasks.append(
                self._inbound_asr.warmup(self.config.target_lang, settings.audio_sample_rate)
            )
        await asyncio.gather(*tasks, return_exceptions=True)

    async def set_microphone_muted(self, muted: bool) -> None:
        self.microphone_muted = muted
        if self._live_outbound is not None:
            self._live_outbound.set_microphone_muted(muted)
        if muted:
            await self.broadcast({"type": "mic_level", "level": 0, "in_speech": False})
        await self.broadcast({"type": "microphone_muted", "muted": muted})

    async def stop_after_tail(self) -> None:
        if not self.running or self._draining:
            return
        self._draining = True
        try:
            await self.broadcast({"type": "status", "running": True, "message": "正在收尾，播放剩余译音…"})
            sessions = [live for live in (self._live_outbound, self._live_inbound) if live is not None]
            await asyncio.gather(*(live.finish_input() for live in sessions))
            if self._mic_jitter_sink is not None:
                self._mic_jitter_sink.flush()
            # Let queued PCM reach its device, then allow its estimated duration to play.
            await asyncio.sleep(0)
            now = time.monotonic()
            until = max(self._play_until.values(), default=0.0)
            tail_sec = until - now + 0.08
            if self._mic_jitter_sink is not None:
                tail_sec = max(tail_sec, self._mic_jitter_sink.pending_seconds() + 0.08)
            await asyncio.sleep(min(3.0, max(0.12, tail_sec)))
            if self.running:
                await self.stop()
        finally:
            self._draining = False

    async def stop(self, reason: str = "同传已停止") -> None:
        self.running = False
        self._outbound_cloud_ready = False
        self._outbound_input_ready = False
        for vad in (self._vad, self._inbound_vad):
            if vad is not None:
                vad.stop()
        for live in (self._live_outbound, self._live_inbound):
            if live is not None:
                live.stop()
        self._live_outbound = None
        self._live_inbound = None
        for bridge in (self._vpio, self._system_tap):
            if bridge is not None:
                await bridge.stop()
        self._vpio = None
        self._system_tap = None

        for task in (
            self._task,
            self._inbound_task,
            self._outbound_tts_task,
            self._inbound_tts_task,
            self._warmup_task,
            self._watchdog_task,
        ):
            if task is not None and task is not asyncio.current_task():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._task = None
        self._inbound_task = None
        self._outbound_tts_task = None
        self._inbound_tts_task = None
        self._warmup_task = None
        self._watchdog_task = None
        self._outbound_tts_queue = None
        self._inbound_tts_queue = None
        for stream in (self._outbound_asr, self._inbound_asr):
            if stream is not None:
                await stream.close()
        self._outbound_asr = None
        self._inbound_asr = None
        for sink in (self._outbound_sink, self._inbound_sink):
            if sink is not None:
                sink.close()
        self._outbound_sink = None
        self._inbound_sink = None
        if self._mic_jitter_sink is not None:
            self._mic_jitter_sink.close()
            self._mic_jitter_sink = None
        if self._mic_ring is not None:
            self._mic_ring.close()
            self._mic_ring = None
        await self.idle_passthrough.resume()
        await self.broadcast({"type": "status", "running": False, "ready": False, "message": reason})

    async def _idle_watchdog(self) -> None:
        timeout_secs = self.config.idle_auto_stop_minutes * 60
        try:
            while self.running:
                await asyncio.sleep(10)
                idle_secs = time.monotonic() - self._last_speech_ts
                if idle_secs >= timeout_secs:
                    await self.stop(
                        reason=f"防忘关保护：持续静默超过 {self.config.idle_auto_stop_minutes} 分钟，已自动停止"
                    )
                    break
        except asyncio.CancelledError:
            raise

    async def _on_mic_level(self, peak_rms: float, in_speech: bool) -> None:
        if in_speech or peak_rms >= 0.001:
            self._heard_real_speech = True
            self._low_level_streak = 0
        if peak_rms == 0.0:
            self._zero_level_streak += 1
            if self._zero_level_streak == 15 and not self._heard_real_speech:
                await self.broadcast(
                    {
                        "type": "error",
                        "message": f"麦克风「{self._active_input_name}」输入全为静音。请先取消耳机静音，或切换为 MacBook Pro 麦克风；若仍无声，请检查「系统设置 → 隐私与安全性 → 麦克风」是否授权。",
                    }
                )
        else:
            self._zero_level_streak = 0
            if peak_rms < 0.0004 and not self._heard_real_speech and not self._low_level_warned:
                self._low_level_streak += 1
                if self._low_level_streak >= 28:
                    self._low_level_warned = True
                    await self.broadcast(
                        {
                            "type": "error",
                            "message": f"当前麦克风「{self._active_input_name}」几乎无声音输入（可能蓝牙耳机未佩戴/处于静音）。请靠近说话或在设置「麦克风」下拉切换为「MacBook Pro麦克风」。",
                        }
                    )
            else:
                self._low_level_streak = 0
        await self.broadcast(
            {
                "type": "mic_level",
                "level": min(1.0, peak_rms * 12.0),
                "in_speech": in_speech,
            }
        )

    async def _on_inbound_level(self, peak_rms: float, in_speech: bool) -> None:
        await self.broadcast(
            {
                "type": "inbound_level",
                "level": min(1.0, peak_rms * 12.0),
                "in_speech": in_speech,
            }
        )

    async def _on_outbound_speech_start(self) -> None:
        if self._outbound_asr is not None:
            await self._outbound_asr.start_utterance(
                self.config.source_lang, settings.audio_sample_rate
            )

    async def _on_outbound_speech_chunk(self, chunk: bytes) -> None:
        if self._outbound_asr is not None:
            await self._outbound_asr.feed_chunk(
                chunk, self.config.source_lang, settings.audio_sample_rate
            )

    async def _on_inbound_speech_start(self) -> None:
        if self._inbound_asr is not None:
            await self._inbound_asr.start_utterance(
                self.config.target_lang, settings.audio_sample_rate
            )

    async def _on_inbound_speech_chunk(self, chunk: bytes) -> None:
        if self._inbound_asr is not None:
            await self._inbound_asr.feed_chunk(
                chunk, self.config.target_lang, settings.audio_sample_rate
            )

    async def _pump_live_channel(
        self,
        *,
        channel: str,
        next_frame: Any,
        source_lang: str,
        target_lang: str,
        enable_audio: bool,
        voice_id: str = "",
        pause_mic_during_tts: bool,
        echo_duck_during_tts: bool = False,
        allow_idle_gaps: bool = False,
        on_level: Any,
    ) -> None:
        """Stream frames continuously to Alibaba qwen3.5-livetranslate-flash-realtime and play deltas live."""
        audio_out_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=256)
        rms_threshold = 0.012 if self.config.noise_suppression else 0.005
        use_minimax_ws = False
        # DashScope native audio outputs 24kHz PCM16.
        live_output_rate = 24000
        source_emitted_turn = False
        history_jobs: asyncio.Queue[tuple[Any, tuple] | None] = asyncio.Queue(maxsize=128)
        output_drops = 0
        input_ready = asyncio.Event()
        cloud_ready = asyncio.Event()
        live_started_at = time.monotonic()
        if channel != "outbound":
            input_ready.set()
        native_speaker: NativeSpeaker | None = None
        original_speaker: _PersistentOutputSink | None = None
        # A separate speaker process is not in the ProcessTap exclusion list.
        # Keep playback in this process while capturing system audio to avoid feedback.
        meeting_route = (
            self.config.output_sink == "voiceclone_mic"
            and self.config.enable_inbound and self.config.inbound_device_index == -2
        )
        # Keep the call audible during the first cloud handshake as well as reconnects.
        connection_lost = meeting_route
        try_native_speaker = is_native_engine_available() and not (
            self.config.enable_inbound and self.config.inbound_device_index is None
        )

        def queue_history(fn: Any, *args: Any) -> None:
            try:
                history_jobs.put_nowait((fn, args))
            except asyncio.QueueFull:
                logger.warning("Live history backlog full (%s)", channel)

        async def write_history() -> None:
            while (job := await history_jobs.get()) is not None:
                try:
                    await asyncio.to_thread(job[0], *job[1])
                except Exception:
                    logger.exception("Could not write live history (%s)", channel)

        minimax_tts: PersistentMiniMaxTTSClient | None = None

        async def _on_speech_started() -> None:
            self._last_speech_ts = time.monotonic()
            self._heard_real_speech = True
            await self.broadcast({"type": "pipeline_stage", "stage": "asr", "channel": channel})

        async def _on_src_delta(txt: str) -> None:
            nonlocal source_emitted_turn
            source_emitted_turn = True
            self._last_speech_ts = time.monotonic()
            self._heard_real_speech = True
            await self.broadcast(
                {"type": "subtitle_delta", "channel": channel, "role": "source", "text": txt}
            )

        async def _on_src_final(txt: str) -> None:
            nonlocal source_emitted_turn
            source_emitted_turn = True
            await self.broadcast(
                {"type": "subtitle", "channel": channel, "role": "source", "text": txt}
            )
            if self._current_session_id and txt:
                prefix = "[对方] " if channel == "inbound" else ""
                queue_history(
                    self.history_store.update_last_source, self._current_session_id,
                    f"{prefix}[实时直译]",
                    f"{prefix}{txt}",
                )

        async def _on_tgt_delta(txt: str) -> None:
            latest_glossary = self.config.glossary
            txt = apply_glossary_terms("", txt, latest_glossary)
            await self.broadcast({"type": "pipeline_stage", "stage": "mt", "channel": channel})
            await self.broadcast(
                {"type": "subtitle_delta", "channel": channel, "role": "target", "text": txt}
            )

        async def _on_tgt_clause(clause: str) -> None:
            if minimax_tts is not None and self.running:
                latest_glossary = self.config.glossary
                clause = apply_glossary_terms("", clause, latest_glossary)
                minimax_tts.enqueue_text(clause)

        async def _on_tgt_final(src: str, txt: str) -> None:
            nonlocal source_emitted_turn
            latest_glossary = self.config.glossary
            txt = apply_glossary_terms(src, txt, latest_glossary)
            pairs = _subtitle_pairs(src, txt)
            for i, (source_part, target_part) in enumerate(pairs):
                if len(pairs) > 1 or not source_emitted_turn:
                    await self.broadcast({
                        "type": "subtitle", "channel": channel, "role": "source",
                        "text": source_part or "🎤 (实时直译)", "new_segment": i > 0,
                    })
                await self.broadcast(
                    {"type": "subtitle", "channel": channel, "role": "target", "text": target_part}
                )
            source_emitted_turn = False
            if not use_minimax_ws:
                await self.broadcast({"type": "pipeline_stage", "stage": "idle", "channel": channel})
            if self._current_session_id:
                prefix = "[对方] " if channel == "inbound" else ""
                queue_history(
                    self.history_store.append_item, self._current_session_id,
                    f"{prefix}{src or '[实时直译]'}",
                    f"{prefix}{txt}",
                )

        async def _on_audio_chunk(pcm_bytes: bytes) -> None:
            nonlocal output_drops
            if not self.running:
                return
            if channel == "inbound" and self.config.passthrough_inbound:
                return
            if audio_out_queue.full():
                try:
                    audio_out_queue.get_nowait()
                    audio_out_queue.task_done()
                    output_drops += 1
                except asyncio.QueueEmpty:
                    pass
            try:
                audio_out_queue.put_nowait(pcm_bytes)
            except asyncio.QueueFull:
                pass

        async def _on_connection(state: str) -> None:
            nonlocal connection_lost
            connection_lost = state != "ready"
            if connection_lost:
                cloud_ready.clear()
                if channel == "outbound":
                    self._outbound_cloud_ready = False
            else:
                cloud_ready.set()
                if channel == "outbound":
                    self._outbound_cloud_ready = True
                logger.info("Live connection ready (%s): elapsed_ms=%.0f mic_or_input_ready=%s",
                            channel, (time.monotonic() - live_started_at) * 1000, input_ready.is_set())
            if not self._draining:
                if channel == "outbound" and state == "ready":
                    if input_ready.is_set():
                        await self.broadcast({"type": "status", "running": True, "ready": True, "message": "翻译和麦克风已就绪 · 实时同传中"})
                    else:
                        await self.broadcast({"type": "status", "running": True, "ready": False, "message": "翻译已连接，等待麦克风信号…"})
                elif channel == "outbound" and state != "ready" and input_ready.is_set():
                    await self.broadcast({"type": "status", "running": True, "ready": False, "message": "翻译通道重连中，请稍候…"})
                await self.broadcast({"type": "translation_connection", "channel": channel, "state": state,
                                      "input_ready": input_ready.is_set(), "ready": self.session_ready})

        async def _on_live_err(msg: str) -> None:
            await self.broadcast({"type": "error", "message": f"[{channel}] {msg}"})

        if use_minimax_ws:
            minimax_tts = PersistentMiniMaxTTSClient(
                voice_id=voice_id,
                target_lang=target_lang,
                speed=self.config.speed,
                sample_rate=live_output_rate,
                on_audio_chunk=_on_audio_chunk,
                on_error=_on_live_err,
            on_connection=_on_connection,
                fallback_pipeline=self.pipeline,
            )

        asr_corpus_text = ", ".join(
            k.strip() for k in (self.config.glossary or {}).keys() if isinstance(k, str) and k.strip()
        )
        live = AlibabaLiveTranslateSession(
            source_lang=source_lang,
            target_lang=target_lang,
            enable_audio=(enable_audio and not use_minimax_ws),
            live_voice_clone=self.config.live_voice_clone,
            sample_rate=settings.audio_sample_rate,
            silence_duration_ms=self.config.vad_silence_ms,
            asr_corpus_text=asr_corpus_text,
            on_source_delta=_on_src_delta,
            on_source_final=_on_src_final,
            on_target_delta=_on_tgt_delta,
            on_target_clause=_on_tgt_clause if use_minimax_ws else None,
            on_target_final=_on_tgt_final,
            on_audio_chunk=_on_audio_chunk,
            on_audio_done=(lambda: _on_audio_chunk(b"")) if enable_audio and not use_minimax_ws else None,
            on_speech_started=_on_speech_started,
            on_error=_on_live_err,
            on_connection=_on_connection,
        )
        if channel == "outbound":
            self._live_outbound = live
            live.set_microphone_muted(self.microphone_muted)
        else:
            self._live_inbound = live

        async def _prepare_default_speaker() -> None:
            nonlocal native_speaker, try_native_speaker
            try:
                native_speaker = NativeSpeaker()
                await native_speaker.start()
            except (OSError, asyncio.TimeoutError, asyncio.IncompleteReadError):
                logger.warning("Core Audio speaker unavailable (%s); falling back to sounddevice", channel, exc_info=True)
                try_native_speaker = False
                if native_speaker is not None:
                    await native_speaker.stop()
                    native_speaker = None

        uses_default_speaker = (
            channel == "outbound" and self.config.output_sink != "voiceclone_mic"
            and self.config.output_device_index is None and self._vpio is None
        ) or (channel == "inbound" and self._vpio is None)
        speaker_task = (asyncio.create_task(_prepare_default_speaker())
                        if enable_audio and uses_default_speaker and try_native_speaker else None)

        async def _write_default_speaker(chunk: bytes) -> bool:
            nonlocal native_speaker, try_native_speaker
            if not try_native_speaker:
                return False
            if speaker_task is not None:
                await speaker_task
            if native_speaker is None:
                return False
            try:
                await native_speaker.write(chunk)
                return True
            except OSError:
                logger.warning("Core Audio speaker disconnected (%s); falling back to sounddevice", channel)
                try_native_speaker = False
                await native_speaker.stop()
                native_speaker = None
                return False

        async def _play_original(frame: bytes) -> None:
            nonlocal original_speaker
            if channel == "outbound":
                if not meeting_route or self.microphone_muted:
                    return
                await self._write_mic_ring_chunk(
                    frame, settings.audio_sample_rate, pace=False
                )
            elif self.config.inbound_device_index is not None:
                if self._vpio is not None and not self._vpio._stopped:
                    await self._vpio.write_playback(frame)
                else:
                    if original_speaker is None:
                        _, phys_out = _resolve_physical_devices(None, None)
                        original_speaker = _PersistentOutputSink(phys_out, settings.audio_sample_rate, 40, 30)
                    await original_speaker.write_live(frame)

        async def _finish_playback() -> None:
            if speaker_task is not None and speaker_task.done() and native_speaker is not None:
                await _write_default_speaker(b"")
            if self._vpio is not None and (
                channel == "inbound" or (self.config.output_sink != "voiceclone_mic"
                                         and self.config.output_device_index is None)
            ):
                await self._vpio.write_playback(b"")
            sink = self._outbound_sink if channel == "outbound" else self._inbound_sink
            if sink is not None:
                sink.flush()
            if channel == "outbound" and self._mic_jitter_sink is not None:
                self._mic_jitter_sink.flush()

        async def _write_audio_chunk(chunk: bytes) -> None:
            if channel == "outbound":
                if self.config.output_sink == "voiceclone_mic":
                    self._get_mic_jitter_sink(live_output_rate).write(chunk)
                else:
                    if self.config.output_device_index is None and self._vpio is not None:
                        await self._vpio.write_playback(_resample_pcm16_mono(
                            chunk, live_output_rate, settings.audio_sample_rate
                        ))
                        return
                    if self.config.output_device_index is None and await _write_default_speaker(chunk):
                        return
                    sink = self._get_sounddevice_sink(
                        "outbound", self.config.output_device_index, sample_rate=live_output_rate,
                        smooth_live=True,
                    )
                    await sink.write_live(chunk)
            else:
                if self._vpio is not None and not self._vpio._stopped:
                    vpio_chunk = _resample_pcm16_mono(
                        chunk, live_output_rate, settings.audio_sample_rate
                    )
                    await self._vpio.write_playback(vpio_chunk)
                else:
                    if await _write_default_speaker(chunk):
                        return
                    _, phys_out = _resolve_physical_devices(None, None)
                    sink = self._get_sounddevice_sink(
                        "inbound", phys_out, sample_rate=live_output_rate, smooth_live=True,
                    )
                    await sink.write_live(chunk)

        async def _playback_loop() -> None:
            paused = False
            playing_stage = False
            while self.running:
                try:
                    chunk = await asyncio.wait_for(audio_out_queue.get(), timeout=0.75)
                except asyncio.TimeoutError:
                    # An inter-packet stall is not end-of-turn. Flushing here
                    # forces the player to start over while the model is still speaking.
                    if paused and self._play_until.get(channel, 0.0) > time.monotonic():
                        continue
                    if paused and pause_mic_during_tts:
                        live.set_paused(False)
                        paused = False
                    if playing_stage:
                        playing_stage = False
                        await self.broadcast(
                            {"type": "pipeline_stage", "stage": "idle", "channel": channel}
                        )
                    continue
                try:
                    if chunk is None or not self.running:
                        break
                    if not chunk:
                        await _finish_playback()
                        continue
                    if not playing_stage:
                        playing_stage = True
                        await self.broadcast(
                            {"type": "pipeline_stage", "stage": "tts", "channel": channel}
                        )
                    if pause_mic_during_tts and not paused:
                        live.set_paused(True)
                        paused = True
                    now = time.monotonic()
                    previous_end = self._play_until.get(channel, 0.0)
                    playback_base = previous_end if previous_end > now else now + 0.70
                    self._play_until[channel] = playback_base + len(chunk) / (2 * live_output_rate)
                    await _write_audio_chunk(chunk)
                finally:
                    audio_out_queue.task_done()
            if paused:
                live.set_paused(False)

        async def _capture_loop() -> None:
            nonlocal original_speaker
            frame_count = 0
            peak_rms = 0.0
            gate = (
                ForegroundSpeechGate()
                if (channel == "outbound" and self.config.noise_suppression)
                else None
            )
            last_input_at = time.monotonic()
            silence_notice_at = last_input_at + 5
            while self.running:
                timeout = 3 if input_ready.is_set() else 5
                remaining = max(0.001, last_input_at + timeout - time.monotonic())
                try:
                    frame = await asyncio.wait_for(next_frame(), remaining)
                except asyncio.TimeoutError as exc:
                    if allow_idle_gaps:
                        last_input_at = time.monotonic()
                        continue
                    if not input_ready.is_set():
                        raise MicrophoneNoSignal(self._active_input_name) from exc
                    raise OSError("麦克风停止提供音频帧") from exc
                if frame is None:
                    if not input_ready.is_set() and not allow_idle_gaps:
                        raise MicrophoneNoSignal(self._active_input_name)
                    break
                if not frame:
                    if allow_idle_gaps:
                        last_input_at = time.monotonic()
                        continue
                    if time.monotonic() - last_input_at >= timeout:
                        if not input_ready.is_set():
                            raise MicrophoneNoSignal(self._active_input_name)
                        raise OSError("麦克风停止提供音频帧")
                    continue
                last_input_at = time.monotonic()
                frame_count += 1
                if self._draining or (input_ready.is_set() and channel == "outbound" and self.microphone_muted):
                    if input_ready.is_set():
                        live.enqueue_audio(frame, silence=True)
                    continue
                rms = compute_rms_int16(frame)
                if not input_ready.is_set():
                    if rms == 0:
                        if channel == "outbound" and last_input_at >= silence_notice_at:
                            silence_notice_at = float("inf")
                            await self.broadcast({"type": "status", "running": True,
                                                  "message": "麦克风已连接，等待声音。若说话后仍无反应，请检查静音或切换麦克风。"})
                        continue
                    input_ready.set()
                    if channel == "outbound":
                        self._outbound_input_ready = True
                        await self.broadcast({"type": "status", "running": True,
                                              "ready": cloud_ready.is_set(),
                                              "message": ("翻译和麦克风已就绪 · 实时同传中" if cloud_ready.is_set()
                                                          else "麦克风已就绪，正在连接翻译…")})
                peak_rms = max(peak_rms, rms)
                if gate is not None:
                    tts_active = echo_duck_during_tts and (self._play_until.get(channel, 0.0) > time.monotonic())
                    gated_frames = gate.process(frame, rms, echo_duck_db=6.0 if tts_active else 0.0)
                    in_speech = gate.is_open
                else:
                    gated_frames = [frame]
                    in_speech = rms >= rms_threshold
                if in_speech:
                    self._last_speech_ts = time.monotonic()
                if on_level is not None and frame_count % 6 == 0:
                    res = on_level(peak_rms, in_speech)
                    if asyncio.iscoroutine(res):
                        await res
                    peak_rms = 0.0
                if connection_lost or (channel == "inbound" and self.config.passthrough_inbound):
                    await _play_original(frame)
                elif original_speaker is not None:
                    original_speaker.close()
                    original_speaker = None
                for gf in gated_frames:
                    live.enqueue_audio(gf, has_speech=in_speech)

        history_task = asyncio.create_task(write_history())
        # Start the network handshake alongside capture so the first spoken
        # syllable does not have to trigger a cold connection.
        ws_task = asyncio.create_task(live.run())
        play_task = asyncio.create_task(_playback_loop())
        cap_task = asyncio.create_task(_capture_loop())
        tasks_set = {ws_task, play_task, cap_task}
        tts_ws_task: asyncio.Task[None] | None = None
        if minimax_tts is not None:
            tts_ws_task = asyncio.create_task(minimax_tts.run())
            tasks_set.add(tts_ws_task)
        try:
            done, pending = await asyncio.wait(
                tasks_set,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in done:
                exc = t.exception()
                if exc is not None and not isinstance(exc, asyncio.CancelledError):
                    raise exc
        finally:
            live.stop()
            if minimax_tts is not None:
                minimax_tts.stop()
            try:
                audio_out_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
            for t in tasks_set:
                t.cancel()
            await asyncio.gather(*tasks_set, return_exceptions=True)
            if speaker_task is not None:
                speaker_task.cancel()
                await asyncio.gather(speaker_task, return_exceptions=True)
            if native_speaker is not None:
                await native_speaker.stop()
            if original_speaker is not None:
                original_speaker.close()
            logger.info("Live playback queue (%s): dropped_chunks=%d", channel, output_drops)
            try:
                await asyncio.wait_for(history_jobs.put(None), timeout=1)
                await asyncio.wait_for(history_task, timeout=2)
            except asyncio.TimeoutError:
                history_task.cancel()
                await asyncio.gather(history_task, return_exceptions=True)
            finally:
                if not history_task.done():
                    history_task.cancel()
                    await asyncio.gather(history_task, return_exceptions=True)
            if channel == "outbound":
                self._live_outbound = None
            else:
                self._live_inbound = None

    async def _write_mic_ring_chunk(
        self,
        chunk: bytes,
        source_rate: int,
        *,
        pace: bool = True,
        max_buffer_sec: float = 1.8,
        max_wait_sec: float = 1.0,
    ) -> None:
        """Write PCM16 mono audio into VoiceClone Mic ring with overrun pacing."""
        if not chunk:
            return
        if self._mic_ring is None:
            self._mic_ring = SharedAudioRingClient(DEFAULT_MIC_RING)
        ring = self._mic_ring
        ring.open(sample_rate=48000)
        if pace and source_rate > 0:
            chunk_sec = (len(chunk) // 2) / float(source_rate)
            deadline = time.monotonic() + max(max_wait_sec, chunk_sec + 0.5)
            while self.running and time.monotonic() < deadline:
                if ring.buffered_seconds() <= max_buffer_sec:
                    break
                await asyncio.sleep(0.025)
        await asyncio.to_thread(ring.write_pcm16_mono, chunk, source_rate)

    def _recover_hotplug_devices(self) -> None:
        """Refresh PortAudio device table after stream teardown and fall back missing devices to system default."""
        devices = list_audio_devices(refresh=True)
        valid_inputs = {d["index"] for d in devices.get("inputs", [])}
        valid_outputs = {d["index"] for d in devices.get("outputs", [])}
        if self.config.input_device_index is not None and self.config.input_device_index not in valid_inputs:
            logger.info("Selected input device %s disappeared; falling back to system default", self.config.input_device_index)
            self.config.input_device_index = None
        if self.config.output_device_index is not None and self.config.output_device_index not in valid_outputs:
            logger.info("Selected output device %s disappeared; falling back to system default", self.config.output_device_index)
            self.config.output_device_index = None
        self._active_input_name, _, self._active_input_is_bt_hfp = _describe_input_device(
            self.config.input_device_index
        )

    async def _run_outbound_loop(self) -> None:
        if self.config.engine_mode != "alibaba_live":
            await self._run_outbound_once()
            return
        attempts = 5
        for attempt in range(attempts):
            try:
                await self._run_outbound_once()
                if not self.running or self._draining:
                    return
                raise OSError("麦克风流已断开")
            except LiveTranslateError as exc:
                message = str(exc)
                await self.stop(reason=message)
                await self.broadcast({"type": "error", "message": message})
                return
            except MicrophoneNoSignal as exc:
                if not self._heard_real_speech or attempt == attempts - 1:
                    message = f"「{exc}」没有收到声音。请检查静音或切换麦克风。"
                    await self.stop(reason=message)
                    await self.broadcast({"type": "error", "message": message})
                    return
                logger.warning(
                    "Microphone signal lost mid-session (%s/%s), refreshing audio route: %s",
                    attempt + 1,
                    attempts,
                    exc,
                )
            except Exception as exc:
                if not self.running or self._draining:
                    return
                logger.warning("Microphone stream failed (%s/%s): %s", attempt + 1, attempts, exc)
                if attempt == attempts - 1:
                    message = "麦克风断开，自动重试失败。请重新连接设备后开始同传。"
                    await self.stop(reason=message)
                    await self.broadcast({"type": "error", "message": message})
                    return
            if self._vpio is not None:
                await self._vpio.stop()
                self._vpio = None
            await self.broadcast({"type": "status", "running": True,
                                  "message": "音频设备变化或断开，正在刷新设备并重连…"})
            await asyncio.sleep(min(1.5, 0.4 * (attempt + 1)))
            await asyncio.to_thread(self._recover_hotplug_devices)
            await self.broadcast({"type": "status", "running": True,
                                  "message": f"音频采集已切换为「{self._active_input_name}」，继续同传中…"})

    async def _run_outbound_once(self) -> None:
        self._outbound_cloud_ready = False
        self._outbound_input_ready = False
        is_live = self.config.engine_mode == "alibaba_live"
        use_vpio = (
            self.config.use_hardware_aec
            and self.config.input_device_index is None
            and not self._active_input_is_bt_hfp
            and is_native_engine_available()
        )
        if self.config.use_hardware_aec and self.config.input_device_index is None and self._active_input_is_bt_hfp:
            logger.info("Bypassing Apple VPIO because default input looks like Bluetooth HFP: %s", self._active_input_name)
            await self.broadcast(
                {
                    "type": "status",
                    "running": True,
                    "message": f"当前系统默认麦克风为蓝牙耳机「{self._active_input_name}」，已启用兼容直采模式（建议有条件时切换「MacBook Pro麦克风」获得更稳定音质）。",
                }
            )
        try:
            if is_live:
                # Live conversation stays full duplex. Hardware AEC uses its own
                # render reference; a headset or isolated route needs no mic mute.
                pause_mic = False
                if use_vpio:
                    self._vpio = NativeVPIODuplex()
                    await self._vpio.start()
                    try:
                        await self._pump_live_channel(
                            channel="outbound",
                            next_frame=self._vpio.read_frame,
                            source_lang=self.config.source_lang,
                            target_lang=self.config.target_lang,
                            enable_audio=True,
                            voice_id=self.config.voice_id,
                            pause_mic_during_tts=pause_mic,
                            on_level=self._on_mic_level,
                        )
                    except MicrophoneNoSignal:
                        logger.info("VPIO returned no input; retrying the same microphone with standard capture")
                    if self.running:
                        await self._vpio.stop()
                        self._vpio = None
                        logger.warning("Native VPIO exited early in live mode; falling back to sounddevice.")
                        await self.broadcast({"type": "status", "running": True, "message": "正在用兼容模式重试同一麦克风，请再说一句…"})
                        async with sounddevice_frame_reader(
                            sample_rate=settings.audio_sample_rate,
                            frame_ms=30,
                            device_index=self.config.input_device_index,
                            is_stopped=lambda: not self.running,
                        ) as next_sd_frame:
                            await self._pump_live_channel(
                                channel="outbound",
                                next_frame=next_sd_frame,
                                source_lang=self.config.source_lang,
                                target_lang=self.config.target_lang,
                                enable_audio=True,
                                voice_id=self.config.voice_id,
                                pause_mic_during_tts=False,
                                echo_duck_during_tts=(self.config.output_sink != "voiceclone_mic"),
                                on_level=self._on_mic_level,
                            )
                else:
                    async with sounddevice_frame_reader(
                        sample_rate=settings.audio_sample_rate,
                        frame_ms=30,
                        device_index=self.config.input_device_index,
                        is_stopped=lambda: not self.running,
                    ) as next_sd_frame:
                        await self._pump_live_channel(
                            channel="outbound",
                            next_frame=next_sd_frame,
                            source_lang=self.config.source_lang,
                            target_lang=self.config.target_lang,
                            enable_audio=True,
                            voice_id=self.config.voice_id,
                            pause_mic_during_tts=pause_mic,
                            echo_duck_during_tts=(self.config.output_sink != "voiceclone_mic"),
                            on_level=self._on_mic_level,
                        )
                return

            assert self._vad is not None
            if use_vpio:
                self._vpio = NativeVPIODuplex()
                await self._vpio.start()
                await self._vad.run_frame_source(
                    self._vpio.read_frame,
                    self._handle_outbound_utterance,
                    on_level=self._on_mic_level,
                    on_speech_start=self._on_outbound_speech_start,
                    on_speech_chunk=self._on_outbound_speech_chunk,
                )
                if self.running:
                    # VPIO exited unexpectedly (e.g. non-default device/permissions) -> fallback to sounddevice
                    logger.warning("Native VPIO exited early; falling back to sounddevice capture.")
                    await self.broadcast({"type": "error", "message": "Apple VPIO 回声消除未能启动，已自动切换为标准麦克风采集。"})
                    await self._vad.run(
                        self._handle_outbound_utterance,
                        on_level=self._on_mic_level,
                        on_speech_start=self._on_outbound_speech_start,
                        on_speech_chunk=self._on_outbound_speech_chunk,
                    )
            else:
                await self._vad.run(
                    self._handle_outbound_utterance,
                    on_level=self._on_mic_level,
                    on_speech_start=self._on_outbound_speech_start,
                    on_speech_chunk=self._on_outbound_speech_chunk,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            if is_live:
                raise
            logger.exception("Outbound audio capture loop failed")
            await self.stop(reason="麦克风采集异常，请重新选择设备。")
        finally:
            if self._vpio is not None:
                await self._vpio.stop()
                self._vpio = None

    async def _run_inbound_loop(self) -> None:
        is_live = self.config.engine_mode == "alibaba_live"
        use_tap = self.config.inbound_device_index is None and is_native_engine_available()
        try:
            if is_live:
                pause_mic = self.config.echo_guard and self._vpio is None
                enable_audio = not self.config.passthrough_inbound
                if self.config.inbound_device_index == -2:
                    ring = SharedAudioRingClient(DEFAULT_MEETING_RING)
                    ring.open()

                    async def next_meeting_frame() -> bytes | None:
                        if not self.running:
                            return None
                        block = ring.read_pcm16_mono()
                        if block:
                            return block
                        await asyncio.sleep(0.025)
                        return b""

                    try:
                        await self._pump_live_channel(
                            channel="inbound", next_frame=next_meeting_frame,
                            source_lang=self.config.target_lang, target_lang=self.config.source_lang,
                            enable_audio=enable_audio, pause_mic_during_tts=False,
                            allow_idle_gaps=True, on_level=self._on_inbound_level,
                        )
                    finally:
                        ring.close()
                    return
                if use_tap:
                    exclude_pids = [os.getpid()]
                    if self._vpio is not None and self._vpio.pid is not None:
                        exclude_pids.append(self._vpio.pid)
                    self._system_tap = NativeSystemTap(exclude_pids=exclude_pids)
                    await self._system_tap.start()
                    await self._pump_live_channel(
                        channel="inbound",
                        next_frame=self._system_tap.read_frame,
                        source_lang=self.config.target_lang,
                        target_lang=self.config.source_lang,
                        enable_audio=enable_audio,
                        voice_id=self.config.inbound_voice_id,
                        pause_mic_during_tts=pause_mic,
                        allow_idle_gaps=True,
                        on_level=self._on_inbound_level,
                    )
                    if self.running:
                        await self.broadcast(
                            {
                                "type": "error",
                                "message": "系统音频捕获未收到音频（请在「系统设置 → 隐私与安全性 → 录屏与系统录音」授权，或在设置中指定虚拟输入设备）。",
                            }
                        )
                else:
                    async with sounddevice_frame_reader(
                        sample_rate=settings.audio_sample_rate,
                        frame_ms=30,
                        device_index=self.config.inbound_device_index,
                        is_stopped=lambda: not self.running,
                    ) as next_in_frame:
                        await self._pump_live_channel(
                            channel="inbound",
                            next_frame=next_in_frame,
                            source_lang=self.config.target_lang,
                            target_lang=self.config.source_lang,
                            enable_audio=enable_audio,
                            voice_id=self.config.inbound_voice_id,
                            pause_mic_during_tts=pause_mic,
                            on_level=self._on_inbound_level,
                        )
                return

            assert self._inbound_vad is not None
            if use_tap:
                self._system_tap = NativeSystemTap(exclude_pids=[os.getpid()])
                await self._system_tap.start()
                await self._inbound_vad.run_frame_source(
                    self._system_tap.read_frame,
                    self._handle_inbound_utterance,
                    on_level=self._on_inbound_level,
                    on_speech_start=self._on_inbound_speech_start,
                    on_speech_chunk=self._on_inbound_speech_chunk,
                )
                if self.running:
                    await self.broadcast(
                        {
                            "type": "error",
                            "message": "系统音频捕获未收到音频（请在「系统设置 → 隐私与安全性 → 录屏与系统录音」授权，或在设置中指定虚拟输入设备）。",
                        }
                    )
            else:
                await self._inbound_vad.run(
                    self._handle_inbound_utterance,
                    on_level=self._on_inbound_level,
                    on_speech_start=self._on_inbound_speech_start,
                    on_speech_chunk=self._on_inbound_speech_chunk,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Inbound audio capture loop failed")
            await self.broadcast({"type": "error", "message": f"来向音频捕获异常: {exc}"})
        finally:
            if self._system_tap is not None:
                await self._system_tap.stop()
                self._system_tap = None

    async def _stream_tts_to_sink(
        self,
        text: str,
        voice_id: str,
        target_lang: str,
        write_chunk: Any,
    ) -> None:
        """Download TTS chunks asynchronously while playing sequentially so network never stalls playback."""
        pcm_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=64)
        producer_err: list[BaseException] = []

        async def _producer() -> None:
            try:
                async for chunk in self.pipeline.stream_cloned_tts(
                    text=text, voice_id=voice_id, speed=self.config.speed, target_lang=target_lang
                ):
                    if not self.running:
                        break
                    if chunk:
                        await pcm_queue.put(chunk)
            except BaseException as exc:
                producer_err.append(exc)
            finally:
                await pcm_queue.put(None)

        producer_task = asyncio.create_task(_producer())
        try:
            while True:
                chunk = await pcm_queue.get()
                try:
                    if chunk is None or not self.running:
                        break
                    res = write_chunk(chunk)
                    if asyncio.iscoroutine(res):
                        await res
                finally:
                    pcm_queue.task_done()
            if producer_err:
                raise producer_err[0]
        finally:
            if not producer_task.done():
                producer_task.cancel()
                try:
                    await producer_task
                except asyncio.CancelledError:
                    pass

    def _get_mic_jitter_sink(self, sample_rate: int | None = None) -> _MicRingJitterSink:
        target_rate = sample_rate or settings.audio_sample_rate
        if self._mic_ring is None:
            self._mic_ring = SharedAudioRingClient(DEFAULT_MIC_RING)
        sink = self._mic_jitter_sink
        if sink is not None and (
            sink.ring is not self._mic_ring or sink.sample_rate != target_rate
        ):
            sink.close()
            sink = None
        if sink is None:
            sink = _MicRingJitterSink(self._mic_ring, sample_rate=target_rate)
            self._mic_jitter_sink = sink
        return sink

    def _get_sounddevice_sink(
        self, channel: str, device_index: int | None, sample_rate: int | None = None,
        *, smooth_live: bool = False,
    ) -> _PersistentOutputSink:
        target_rate = sample_rate or settings.audio_sample_rate
        attr = "_outbound_sink" if channel == "outbound" else "_inbound_sink"
        sink: _PersistentOutputSink | None = getattr(self, attr)
        if sink is not None and (
            sink.device_index != device_index or sink.sample_rate != target_rate
        ):
            sink.close()
            sink = None
        if sink is None:
            sink = _PersistentOutputSink(
                device_index=device_index, sample_rate=target_rate,
                prebuffer_ms=100,
                recovery_ms=150 if smooth_live else 50,
            )
            setattr(self, attr, sink)
        return sink

    async def _play_outbound_chunk_stream(self, text: str, voice_id: str, target_lang: str) -> None:
        if self.config.output_sink == "voiceclone_mic":
            mic_sink = self._get_mic_jitter_sink(settings.audio_sample_rate)
            try:
                await self._stream_tts_to_sink(
                    text,
                    voice_id,
                    target_lang,
                    lambda chunk: asyncio.to_thread(mic_sink.write, chunk),
                )
            finally:
                mic_sink.flush()
            return
        sink = self._get_sounddevice_sink("outbound", self.config.output_device_index)
        await self._stream_tts_to_sink(
            text, voice_id, target_lang, lambda chunk: asyncio.to_thread(sink.write, chunk)
        )

    async def _play_inbound_chunk_stream(self, text: str, voice_id: str, target_lang: str) -> None:
        # Prefer routing inbound translation audio through VPIO's render callback so Apple AEC cancels it!
        if self._vpio is not None and not self._vpio._stopped:
            vpio = self._vpio
            try:
                await self._stream_tts_to_sink(text, voice_id, target_lang, vpio.write_playback)
            finally:
                await vpio.write_playback(b"")
            return
        sink = self._get_sounddevice_sink("inbound", None)
        await self._stream_tts_to_sink(
            text, voice_id, target_lang, lambda chunk: asyncio.to_thread(sink.write, chunk)
        )

    async def _run_tts_worker(self, channel: str) -> None:
        queue = self._outbound_tts_queue if channel == "outbound" else self._inbound_tts_queue
        if queue is None:
            return
        try:
            while self.running:
                text, voice_id, target_lang, pause_mic = await queue.get()
                should_pause = pause_mic and self._vad is not None
                try:
                    if not self.running:
                        break
                    await self.broadcast({"type": "pipeline_stage", "stage": "tts", "channel": channel})
                    if should_pause and self._vad is not None:
                        self._vad.set_paused(True)
                    if channel == "outbound":
                        await self._play_outbound_chunk_stream(text, voice_id, target_lang)
                    else:
                        await self._play_inbound_chunk_stream(text, voice_id, target_lang)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.exception("TTS playback worker failed (%s)", channel)
                    await self.broadcast({"type": "error", "message": f"[{channel}] {exc}"})
                finally:
                    if should_pause and self._vad is not None:
                        await asyncio.sleep(0.15)
                        self._vad.set_paused(False)
                    await self.broadcast({"type": "pipeline_stage", "stage": "idle", "channel": channel})
                    queue.task_done()
        except asyncio.CancelledError:
            raise

    async def _process_utterance(
        self,
        *,
        channel: str,
        pcm_bytes: bytes,
        asr_lang: str,
        mt_target_lang: str,
        tts_voice_id: str,
        play_tts: bool,
        pause_outbound_mic_during_tts: bool,
    ) -> None:
        if not self.running:
            return
        self._last_speech_ts = time.monotonic()
        queued_tts = False
        try:
            await self.broadcast({"type": "pipeline_stage", "stage": "asr", "channel": channel})
            asr_stream = self._outbound_asr if channel == "outbound" else self._inbound_asr
            if asr_stream is not None:
                source_text = await asr_stream.commit_or_transcribe(
                    pcm_bytes,
                    source_lang=asr_lang,
                    sample_rate=settings.audio_sample_rate,
                )
            else:
                source_text = await self.asr.transcribe_utterance(
                    pcm_bytes=pcm_bytes,
                    source_lang=asr_lang,
                    sample_rate=settings.audio_sample_rate,
                )
            if not source_text or not self.running:
                return

            await self.broadcast({"type": "subtitle", "channel": channel, "role": "source", "text": source_text})
            await self.broadcast({"type": "pipeline_stage", "stage": "mt", "channel": channel})

            latest_glossary = self.glossary_store.list_terms() or self.config.glossary
            target_text = ""
            async for partial_tgt in self.pipeline.stream_translate_text(
                text=source_text,
                source_lang=asr_lang,
                target_lang=mt_target_lang,
                glossary=latest_glossary or None,
            ):
                target_text = partial_tgt
                await self.broadcast(
                    {"type": "subtitle_delta", "channel": channel, "role": "target", "text": partial_tgt}
                )
            target_text = target_text.strip()
            if not target_text:
                target_text = await self.pipeline.translate_text(
                    text=source_text,
                    source_lang=asr_lang,
                    target_lang=mt_target_lang,
                    glossary=latest_glossary or None,
                )
            await self.broadcast({"type": "subtitle", "channel": channel, "role": "target", "text": target_text})

            if self._current_session_id:
                prefix = "[对方] " if channel == "inbound" else ""
                await asyncio.to_thread(
                    self.history_store.append_item, self._current_session_id,
                    f"{prefix}{source_text}",
                    f"{prefix}{target_text}",
                )

            if play_tts and tts_voice_id and self.running:
                queue = self._outbound_tts_queue if channel == "outbound" else self._inbound_tts_queue
                if queue is not None:
                    if queue.full():
                        try:
                            queue.get_nowait()
                            queue.task_done()
                        except asyncio.QueueEmpty:
                            pass
                    queue.put_nowait(
                        (target_text, tts_voice_id, mt_target_lang, pause_outbound_mic_during_tts)
                    )
                    queued_tts = True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Utterance pipeline failed (%s)", channel)
            await self.broadcast({"type": "error", "message": f"[{channel}] {exc}"})
        finally:
            if not queued_tts:
                await self.broadcast({"type": "pipeline_stage", "stage": "idle", "channel": channel})

    async def _handle_outbound_utterance(self, pcm_bytes: bytes) -> None:
        await self._process_utterance(
            channel="outbound",
            pcm_bytes=pcm_bytes,
            asr_lang=self.config.source_lang,
            mt_target_lang=self.config.target_lang,
            tts_voice_id=self.config.voice_id,
            play_tts=bool(self.config.voice_id),
            pause_outbound_mic_during_tts=self.config.echo_guard and self.config.output_sink != "voiceclone_mic",
        )

    async def _handle_inbound_utterance(self, pcm_bytes: bytes) -> None:
        inbound_voice = self.config.inbound_voice_id or self.config.voice_id or "male-qn-qingse"
        await self._process_utterance(
            channel="inbound",
            pcm_bytes=pcm_bytes,
            asr_lang=self.config.target_lang,
            mt_target_lang=self.config.source_lang,
            tts_voice_id=inbound_voice,
            play_tts=not self.config.passthrough_inbound,
            pause_outbound_mic_during_tts=self.config.echo_guard and self._vpio is None,
        )
