"""Cross-platform real-time audio capture, energy VAD, WAV recording, and virtual mic playback."""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager
import math
import struct
import wave
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

from src.audio_device import (
    open_portaudio_stream,
    portaudio_stream,
    refresh_portaudio_devices,
    unregister_portaudio_stream,
)


def compute_rms_int16(pcm_bytes: bytes) -> float:
    """Compute normalized RMS amplitude (0.0..1.0) of mono 16-bit PCM bytes."""
    count = len(pcm_bytes) // 2
    if count == 0:
        return 0.0
    samples = struct.unpack(f"<{count}h", pcm_bytes[: count * 2])
    mean_sq = sum(s * s for s in samples) / count
    return math.sqrt(mean_sq) / 32768.0


def rms_to_dbfs(rms: float) -> float:
    """Convert normalized RMS (0..1) to dBFS."""
    return 20.0 * math.log10(max(rms, 1e-6))


def resample_pcm16_mono(pcm_bytes: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linearly resample mono 16-bit PCM bytes when stream sample rate differs from target rate."""
    if not pcm_bytes or src_rate == dst_rate or src_rate <= 0 or dst_rate <= 0:
        return pcm_bytes
    count = len(pcm_bytes) // 2
    if count < 2:
        return pcm_bytes
    samples = struct.unpack(f"<{count}h", pcm_bytes[: count * 2])
    out_len = max(1, int(round(count * (dst_rate / src_rate))))
    step = (count - 1) / max(1, out_len - 1)
    out = []
    for i in range(out_len):
        pos = i * step
        idx = int(pos)
        frac = pos - idx
        s0 = samples[idx]
        s1 = samples[idx + 1] if idx + 1 < count else s0
        val = int(round(s0 + (s1 - s0) * frac))
        out.append(max(-32768, min(32767, val)))
    return struct.pack(f"<{len(out)}h", *out)


def _looks_like_impulse(frame: bytes, rms: float) -> bool:
    """Detect sharp transient clicks/pops (high crest factor + brief active duty cycle)."""
    count = len(frame) // 2
    if count < 16 or rms <= 1e-6:
        return False
    samples = struct.unpack(f"<{count}h", frame[: count * 2])
    peak_int = max(abs(s) for s in samples)
    if peak_int <= 0:
        return False
    crest = (peak_int / 32768.0) / rms
    if crest < 6.5:
        return False
    active_thresh = peak_int * 0.25
    active_count = sum(1 for s in samples if abs(s) >= active_thresh)
    return (active_count / count) <= 0.12


class ForegroundSpeechGate:
    """Near-field foreground speech gate (suppresses distant background speakers/noise in live mode)."""

    def __init__(
        self,
        *,
        pre_roll_frames: int = 8,
        hangover_frames: int = 10,
        min_consecutive_strong: int = 2,
        base_open_dbfs: float = -42.0,
    ) -> None:
        self._closed_history: deque[bytes] = deque(maxlen=max(0, pre_roll_frames))
        self._hangover_limit = max(1, hangover_frames)
        self._hangover_remaining = 0
        self._min_consecutive = max(1, min_consecutive_strong)
        self._consecutive_strong = 0
        self._base_open_dbfs = base_open_dbfs
        self._noise_floor_dbfs = -56.0
        self._foreground_ref_dbfs: float | None = None
        self._gate_open = False

    @property
    def is_open(self) -> bool:
        return self._gate_open

    def process(self, frame: bytes, rms: float, *, echo_duck_db: float = 0.0) -> list[bytes]:
        """Return frames to forward (real PCM when near-field speech is active, silence when gated)."""
        if not frame:
            return []
        db = rms_to_dbfs(rms)
        open_thresh = max(self._base_open_dbfs, self._noise_floor_dbfs + 12.0)
        if self._foreground_ref_dbfs is not None:
            # Reject distant/nearby background voices quieter than the primary near-field speaker
            open_thresh = max(open_thresh, min(-24.0, self._foreground_ref_dbfs - 12.0))
        if echo_duck_db > 0.0:
            # Raise the open threshold during non-VPIO speaker playback without blocking loud barge-in
            open_thresh = min(-18.0, open_thresh + min(4.0, echo_duck_db))

        is_strong = db >= open_thresh
        if is_strong and not self._gate_open and _looks_like_impulse(frame, rms):
            # Reject isolated keyboard clicks/pops before they can open the gate or enter pre-roll
            self._consecutive_strong = 0
            silence = b"\x00" * len(frame)
            self._closed_history.append(silence)
            return [silence]

        if is_strong:
            self._consecutive_strong += 1
            if not self._gate_open and self._consecutive_strong >= self._min_consecutive:
                self._gate_open = True
                self._hangover_remaining = self._hangover_limit
                out = list(self._closed_history)
                self._closed_history.clear()
                out.append(frame)
                self._update_foreground_ref(db)
                return out
            if self._gate_open:
                self._hangover_remaining = self._hangover_limit
                self._update_foreground_ref(db)
                return [frame]
            self._closed_history.append(frame)
            return [b"\x00" * len(frame)]

        self._consecutive_strong = 0
        if self._gate_open:
            if self._hangover_remaining > 0:
                self._hangover_remaining -= 1
                return [frame]
            self._gate_open = False

        # Adapt noise floor and slowly relax foreground loudness reference while closed
        # so soft near-field speech is still captured after louder utterances.
        self._noise_floor_dbfs = 0.96 * self._noise_floor_dbfs + 0.04 * min(db, -30.0)
        if self._foreground_ref_dbfs is not None:
            decay_rate = 0.88 if db <= self._noise_floor_dbfs + 6.0 else 0.96
            self._foreground_ref_dbfs = (
                decay_rate * self._foreground_ref_dbfs
                + (1.0 - decay_rate) * self._base_open_dbfs
            )
            if self._foreground_ref_dbfs <= self._base_open_dbfs + 0.5:
                self._foreground_ref_dbfs = None
        self._closed_history.append(frame)
        return [b"\x00" * len(frame)]

    def _update_foreground_ref(self, db: float) -> None:
        if self._foreground_ref_dbfs is None:
            self._foreground_ref_dbfs = db
        else:
            self._foreground_ref_dbfs = 0.92 * self._foreground_ref_dbfs + 0.08 * db


def record_wav_sample(
    output_path: str | Path,
    seconds: float = 15.0,
    sample_rate: int = 16000,
    device_index: int | None = None,
) -> Path:
    """Record mono 16-bit PCM audio from a microphone into a WAV file without requiring NumPy."""
    import sounddevice as sd

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = int(seconds * sample_rate)
    chunk_frames = 1024
    collected = bytearray()

    with portaudio_stream(
        lambda: sd.RawInputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            device=device_index,
            blocksize=chunk_frames,
        )
    ) as stream:
        remaining = total_frames
        while remaining > 0:
            to_read = min(chunk_frames, remaining)
            data, _ = stream.read(to_read)
            collected.extend(bytes(data))
            remaining -= to_read

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(collected))
    return path


def play_pcm_bytes(
    pcm_bytes: bytes,
    sample_rate: int = 16000,
    device_index: int | None = None,
) -> None:
    """Play raw mono 16-bit PCM bytes synchronously to the chosen output or virtual mic device."""
    import sounddevice as sd

    if not pcm_bytes:
        return
    with portaudio_stream(
        lambda: sd.RawOutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            device=device_index,
        )
    ) as stream:
        stream.write(pcm_bytes)


class RealtimeMicVAD:
    """Captures microphone audio and emits speech chunks + utterances using RMS + hangover VAD."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        rms_threshold: float = 0.005,
        silence_hangover_ms: int = 260,
        min_speech_ms: int = 210,
        max_speech_ms: int = 5500,
        pre_roll_frames: int = 4,
        device_index: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.frame_ms = frame_ms
        self.rms_threshold = rms_threshold
        self.hangover_frames = max(1, silence_hangover_ms // frame_ms)
        self.min_speech_frames = max(1, min_speech_ms // frame_ms)
        self.max_speech_frames = max(self.min_speech_frames + 10, max_speech_ms // frame_ms)
        self.hard_max_speech_frames = self.max_speech_frames + max(10, 1000 // frame_ms)
        self.pre_roll_frames = max(0, pre_roll_frames)
        self.device_index = device_index
        self._stop_event = asyncio.Event()
        self._paused = False

    def set_paused(self, paused: bool) -> None:
        """Temporarily ignore mic input (e.g. during TTS speaker playback to prevent self-echo)."""
        self._paused = paused

    def stop(self) -> None:
        self._stop_event.set()

    async def run_frame_source(
        self,
        next_frame: Callable[[], Awaitable[bytes | None]],
        on_utterance: Callable[[bytes], asyncio.Future | Any],
        on_level: Callable[[float, bool], asyncio.Future | Any] | None = None,
        on_speech_start: Callable[[], asyncio.Future | Any] | None = None,
        on_speech_chunk: Callable[[bytes], asyncio.Future | Any] | None = None,
    ) -> None:
        """Consume 16-bit PCM frames from next_frame() and run RMS + hangover VAD."""
        utterance_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=20)
        stream_queue: asyncio.Queue[tuple[str, bytes]] | None = (
            asyncio.Queue(maxsize=128) if (on_speech_start or on_speech_chunk) else None
        )
        self._stop_event.clear()

        async def _utterance_worker() -> None:
            while not self._stop_event.is_set():
                utterance = await utterance_queue.get()
                try:
                    if stream_queue is not None:
                        try:
                            await asyncio.wait_for(stream_queue.join(), timeout=0.25)
                        except asyncio.TimeoutError:
                            pass
                    res = on_utterance(utterance)
                    if asyncio.iscoroutine(res):
                        await res
                finally:
                    utterance_queue.task_done()

        async def _stream_worker() -> None:
            assert stream_queue is not None
            while not self._stop_event.is_set():
                kind, data = await stream_queue.get()
                try:
                    if kind == "start" and on_speech_start is not None:
                        r = on_speech_start()
                        if asyncio.iscoroutine(r):
                            await r
                    elif kind == "chunk" and on_speech_chunk is not None and data:
                        r = on_speech_chunk(data)
                        if asyncio.iscoroutine(r):
                            await r
                except asyncio.CancelledError:
                    raise
                except Exception:
                    pass
                finally:
                    stream_queue.task_done()

        def _enqueue_stream(kind: str, data: bytes = b"") -> None:
            if stream_queue is None:
                return
            if stream_queue.full():
                try:
                    stream_queue.get_nowait()
                    stream_queue.task_done()
                except asyncio.QueueEmpty:
                    pass
            try:
                stream_queue.put_nowait((kind, data))
            except asyncio.QueueFull:
                pass

        pre_roll: deque[bytes] = deque(maxlen=self.pre_roll_frames)
        speech_buffer: list[bytes] = []
        stream_batch: list[bytes] = []
        active_speech_frames = 0
        silence_count = 0
        in_speech = False
        frame_count = 0
        peak_rms = 0.0
        worker_task = asyncio.create_task(_utterance_worker())
        stream_task = asyncio.create_task(_stream_worker()) if stream_queue is not None else None

        def _emit_utterance() -> None:
            nonlocal active_speech_frames, silence_count, in_speech
            if active_speech_frames >= self.min_speech_frames and speech_buffer:
                if stream_batch:
                    _enqueue_stream("chunk", b"".join(stream_batch))
                utterance = b"".join(speech_buffer)
                if utterance_queue.full():
                    try:
                        utterance_queue.get_nowait()
                        utterance_queue.task_done()
                    except asyncio.QueueEmpty:
                        pass
                utterance_queue.put_nowait(utterance)
            speech_buffer.clear()
            stream_batch.clear()
            active_speech_frames = 0
            silence_count = 0
            in_speech = False

        try:
            while not self._stop_event.is_set():
                frame = await next_frame()
                if frame is None:
                    break
                if not frame:
                    continue

                frame_count += 1
                if self._paused:
                    pre_roll.clear()
                    speech_buffer.clear()
                    stream_batch.clear()
                    active_speech_frames = 0
                    silence_count = 0
                    in_speech = False
                    peak_rms = 0.0
                    if on_level is not None and frame_count % 6 == 0:
                        lvl_res = on_level(0.0, False)
                        if asyncio.iscoroutine(lvl_res):
                            await lvl_res
                    continue

                rms = compute_rms_int16(frame)
                peak_rms = max(peak_rms, rms)

                if on_level is not None and frame_count % 6 == 0:
                    lvl_res = on_level(peak_rms, in_speech)
                    if asyncio.iscoroutine(lvl_res):
                        await lvl_res
                    peak_rms = 0.0

                if rms >= self.rms_threshold:
                    if not in_speech:
                        in_speech = True
                        _enqueue_stream("start")
                        if pre_roll:
                            speech_buffer.extend(pre_roll)
                            stream_batch.extend(pre_roll)
                            pre_roll.clear()
                    active_speech_frames += 1
                    silence_count = 0
                    speech_buffer.append(frame)
                    stream_batch.append(frame)
                    if len(stream_batch) >= 6:
                        _enqueue_stream("chunk", b"".join(stream_batch))
                        stream_batch.clear()
                    if len(speech_buffer) >= self.hard_max_speech_frames:
                        _emit_utterance()
                elif in_speech:
                    speech_buffer.append(frame)
                    stream_batch.append(frame)
                    if len(stream_batch) >= 6:
                        _enqueue_stream("chunk", b"".join(stream_batch))
                        stream_batch.clear()
                    silence_count += 1
                    should_flush_long = (
                        len(speech_buffer) >= self.max_speech_frames and silence_count >= 2
                    )
                    if silence_count >= self.hangover_frames or should_flush_long:
                        _emit_utterance()
                else:
                    pre_roll.append(frame)
        finally:
            worker_task.cancel()
            if stream_task is not None:
                stream_task.cancel()
            for t in (worker_task, stream_task):
                if t is not None:
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass

    async def run(
        self,
        on_utterance: Callable[[bytes], asyncio.Future | Any],
        on_level: Callable[[float, bool], asyncio.Future | Any] | None = None,
        on_speech_start: Callable[[], asyncio.Future | Any] | None = None,
        on_speech_chunk: Callable[[bytes], asyncio.Future | Any] | None = None,
    ) -> None:
        """Stream mic frames via sounddevice and invoke on_utterance(pcm_bytes)."""
        async with sounddevice_frame_reader(
            sample_rate=self.sample_rate,
            frame_ms=self.frame_ms,
            device_index=self.device_index,
            is_stopped=self._stop_event.is_set,
        ) as next_sd_frame:
            await self.run_frame_source(
                next_sd_frame,
                on_utterance,
                on_level=on_level,
                on_speech_start=on_speech_start,
                on_speech_chunk=on_speech_chunk,
            )


@asynccontextmanager
async def sounddevice_frame_reader(
    *,
    sample_rate: int = 16000,
    frame_ms: int = 30,
    device_index: int | None = None,
    is_stopped: Callable[[], bool] | None = None,
) -> AsyncIterator[Callable[[], Awaitable[bytes | None]]]:
    """Open a mono PCM16 sounddevice input stream (with native-rate fallback) and yield an async next_frame() reader."""
    import sounddevice as sd

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
    frame_samples = int(sample_rate * frame_ms / 1000)
    frame_bytes = frame_samples * 2
    closed = False
    actual_rate = sample_rate
    pending_pcm = bytearray()

    def _enqueue_frame(frame_data: bytes) -> None:
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            queue.put_nowait(frame_data)
        except asyncio.QueueFull:
            pass

    def audio_callback(indata: bytes, frames: int, time_info: object, status: object) -> None:
        if closed or (is_stopped and is_stopped()) or loop.is_closed():
            return
        raw = bytes(indata)
        if actual_rate != sample_rate:
            raw = resample_pcm16_mono(raw, actual_rate, sample_rate)
            pending_pcm.extend(raw)
            while len(pending_pcm) >= frame_bytes:
                chunk = bytes(pending_pcm[:frame_bytes])
                del pending_pcm[:frame_bytes]
                try:
                    loop.call_soon_threadsafe(_enqueue_frame, chunk)
                except RuntimeError:
                    return
            return
        try:
            loop.call_soon_threadsafe(_enqueue_frame, raw)
        except RuntimeError:
            pass

    def _stream_is_active() -> bool:
        try:
            return bool(getattr(stream, "active", True))
        except Exception:
            return False

    async def _next_sd_frame() -> bytes | None:
        if closed or (is_stopped and is_stopped()):
            return None
        try:
            return await asyncio.wait_for(queue.get(), timeout=0.2)
        except asyncio.TimeoutError:
            if closed or (is_stopped and is_stopped()) or not _stream_is_active():
                return None
            return b""

    def _open_input_stream() -> tuple[Any, int]:
        def _native_rate() -> int:
            try:
                info = sd.query_devices(kind="input") if device_index is None else sd.query_devices(device_index)
                return int(round(float(info.get("default_samplerate", 0) or 0)))
            except Exception:
                return 0

        def _open_with_rate(rate: int) -> tuple[Any, int]:
            blk = max(160, int(rate * frame_ms / 1000))
            st = open_portaudio_stream(
                lambda: sd.RawInputStream(
                    samplerate=rate,
                    blocksize=blk,
                    channels=1,
                    dtype="int16",
                    device=device_index,
                    callback=audio_callback,
                ),
                start=False,
            )
            return st, rate

        def _try_open_with_native_fallback() -> tuple[Any, int]:
            try:
                return _open_with_rate(sample_rate)
            except Exception:
                native_sr = _native_rate()
                if native_sr > 0 and native_sr != sample_rate:
                    return _open_with_rate(native_sr)
                raise

        try:
            return _try_open_with_native_fallback()
        except Exception as first_exc:
            if refresh_portaudio_devices(sd):
                try:
                    return _try_open_with_native_fallback()
                except Exception:
                    pass
            raise first_exc

    stream, actual_rate = _open_input_stream()
    try:
        with stream:
            try:
                yield _next_sd_frame
            finally:
                closed = True
    finally:
        closed = True
        unregister_portaudio_stream(stream)
