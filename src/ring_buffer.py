"""Shared mmap ring buffer client matching VoiceCloneAudioDriver.driver (VCRingBuffer)."""

from __future__ import annotations

import mmap
import os
import struct
import threading
from pathlib import Path

VC_RING_MAGIC = 0x56435242
VC_RING_CAPACITY_FRAMES = 48000 * 4
HEADER_FORMAT = "<IIIIIIQQ"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 40 bytes
TOTAL_BYTES = HEADER_SIZE + VC_RING_CAPACITY_FRAMES * 2 * 4

DEFAULT_MIC_RING = Path("/Users/Shared/VoiceCloneLive/Audio/mic.ring")
DEFAULT_MEETING_RING = Path("/Users/Shared/VoiceCloneLive/Audio/meeting.ring")


def apply_pcm16_fade_in(pcm_bytes: bytes, sample_rate: int, fade_ms: float = 4.0) -> bytes:
    """Apply a short linear fade-in to mono Int16 PCM to avoid edge clicks after silence."""
    count = len(pcm_bytes) // 2
    if count == 0 or fade_ms <= 0 or sample_rate <= 0:
        return pcm_bytes
    fade_samples = min(count, max(1, int(sample_rate * fade_ms / 1000.0)))
    head = list(struct.unpack(f"<{fade_samples}h", pcm_bytes[: fade_samples * 2]))
    for idx, sample in enumerate(head):
        gain = (idx + 1) / float(fade_samples)
        scaled = int(round(sample * gain))
        head[idx] = max(-32768, min(32767, scaled))
    return struct.pack(f"<{fade_samples}h", *head) + pcm_bytes[fade_samples * 2 :]


class StreamingStereoResampler:
    """Continuous Int16 mono -> stereo float32 resampler across chunk boundaries."""

    def __init__(self, src_rate: int = 24000, dst_rate: int = 48000) -> None:
        self._src_rate = max(1, int(src_rate))
        self._dst_rate = max(1, int(dst_rate))
        self._tail_sample: float | None = None
        self._idx = 0
        self._rem = 0

    def reset(self) -> None:
        """Clear buffered boundary state after an audio discontinuity or gap."""
        self._tail_sample = None
        self._idx = 0
        self._rem = 0

    def flush(self) -> list[float]:
        """Emit any final held sample at the end of a contiguous turn."""
        if self._tail_sample is None:
            self._idx = 0
            self._rem = 0
            return []
        sample = self._tail_sample
        stereo: list[float] = []
        while self._idx == 0:
            stereo.append(sample)
            stereo.append(sample)
            self._rem += self._src_rate
            if self._rem >= self._dst_rate:
                self._idx += self._rem // self._dst_rate
                self._rem %= self._dst_rate
        self.reset()
        return stereo

    def process(
        self,
        pcm_bytes: bytes,
        src_rate: int | None = None,
        dst_rate: int | None = None,
    ) -> list[float]:
        """Resample one PCM16 mono chunk, carrying lookahead state to the next chunk."""
        next_src = max(1, int(src_rate or self._src_rate))
        next_dst = max(1, int(dst_rate or self._dst_rate))
        prefix: list[float] = []
        if next_src != self._src_rate or next_dst != self._dst_rate:
            prefix = self.flush()
            self._src_rate = next_src
            self._dst_rate = next_dst

        count = len(pcm_bytes) // 2
        if count == 0:
            return prefix
        samples = struct.unpack(f"<{count}h", pcm_bytes[: count * 2])

        if self._src_rate == self._dst_rate:
            if self._tail_sample is not None:
                prefix.extend(self.flush())
            stereo = prefix
            for sample in samples:
                val = sample / 32768.0
                stereo.append(val)
                stereo.append(val)
            return stereo

        buf = [s / 32768.0 for s in samples]
        if self._tail_sample is not None:
            buf.insert(0, self._tail_sample)
        limit = len(buf) - 1
        stereo = prefix
        inv_dst = 1.0 / float(self._dst_rate)
        while self._idx < limit:
            s0 = buf[self._idx]
            s1 = buf[self._idx + 1]
            val = s0 + (s1 - s0) * (self._rem * inv_dst)
            stereo.append(val)
            stereo.append(val)
            self._rem += self._src_rate
            if self._rem >= self._dst_rate:
                self._idx += self._rem // self._dst_rate
                self._rem %= self._dst_rate
        self._tail_sample = buf[-1]
        self._idx -= limit
        return stereo


def _resample_int16_mono_to_stereo_floats(pcm_bytes: bytes, src_rate: int, dst_rate: int) -> list[float]:
    count = len(pcm_bytes) // 2
    if count == 0:
        return []
    samples = struct.unpack(f"<{count}h", pcm_bytes[: count * 2])
    if src_rate == dst_rate:
        mono = [s / 32768.0 for s in samples]
    else:
        ratio = dst_rate / float(src_rate)
        out_len = max(1, int(count * ratio))
        mono = []
        for i in range(out_len):
            pos = i / ratio
            idx = int(pos)
            frac = pos - idx
            s0 = samples[min(idx, count - 1)] / 32768.0
            s1 = samples[min(idx + 1, count - 1)] / 32768.0
            mono.append(s0 + (s1 - s0) * frac)
    stereo: list[float] = []
    for val in mono:
        stereo.append(val)
        stereo.append(val)
    return stereo


class SharedAudioRingClient:
    """Thread-safe shared-memory mmap client for VoiceCloneAudioDriver."""

    def __init__(self, ring_path: str | Path = DEFAULT_MIC_RING) -> None:
        self.path = Path(ring_path)
        self._fd: int | None = None
        self._mm: mmap.mmap | None = None
        self._io_lock = threading.RLock()

    def open(self, sample_rate: int = 48000) -> None:
        with self._io_lock:
            if self._mm is not None:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(self.path.parent, 0o777)
            except OSError:
                pass
            fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o666)
            try:
                os.fchmod(fd, 0o666)
            except OSError:
                pass
            os.ftruncate(fd, TOTAL_BYTES)
            mm = mmap.mmap(fd, TOTAL_BYTES, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
            magic, version, sr, ch, cap, _, _, _ = struct.unpack_from(HEADER_FORMAT, mm, 0)
            if magic != VC_RING_MAGIC or cap != VC_RING_CAPACITY_FRAMES:
                struct.pack_into(
                    HEADER_FORMAT,
                    mm,
                    0,
                    VC_RING_MAGIC,
                    1,
                    sample_rate,
                    2,
                    VC_RING_CAPACITY_FRAMES,
                    0,
                    0,
                    0,
                )
            elif sr == 0:
                struct.pack_into("<I", mm, 8, sample_rate)
            self._fd = fd
            self._mm = mm

    def close(self) -> None:
        with self._io_lock:
            if self._mm is not None:
                self._mm.close()
                self._mm = None
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None

    def header_stats(self) -> tuple[int, int, int, int, int]:
        """Return (sample_rate, channels, reserved_io_count, write_pos, read_pos) or zeros."""
        with self._io_lock:
            if self._mm is None:
                return (0, 0, 0, 0, 0)
            magic, _, rate, channels, _, reserved, written, read = struct.unpack_from(HEADER_FORMAT, self._mm)
            if magic != VC_RING_MAGIC:
                return (0, 0, 0, 0, 0)
            return (rate, channels, reserved, written, read)

    def ring_status(self) -> tuple[int, int, int, int]:
        """Return (sample_rate, reserved_io_count, write_pos, read_pos) for this ring."""
        rate, _, reserved, written, read = self.header_stats()
        return (rate, reserved, written, read)

    def buffered_seconds(self) -> float:
        """Return unread seconds currently queued for an active CoreAudio HAL consumer."""
        rate, reserved, written, read = self.ring_status()
        if reserved <= 0 or rate <= 0:
            return 0.0
        return max(0, written - read) / float(rate)

    def read_pcm16_mono(self, frames: int = 1440, target_rate: int = 16000) -> bytes:
        """Read one block of ring audio and convert it to mono Int16 at target_rate (default 16 kHz)."""
        with self._io_lock:
            if self._mm is None or frames <= 0 or target_rate <= 0:
                return b""
            magic, _, rate, channels, cap, _, written, read = struct.unpack_from(HEADER_FORMAT, self._mm)
            if (
                magic != VC_RING_MAGIC
                or cap <= 0
                or not (8000 <= rate <= 192000)
                or channels not in (1, 2)
                or written < read
            ):
                return b""

            # Preserve exact 3:1 boxcar fast-path for 48 kHz stereo -> 16 kHz mono
            if rate == 48000 and target_rate == 16000 and channels == 2 and frames % 3 == 0:
                src_frames = frames
                if src_frames > cap:
                    return b""
                read = max(read, written - min(cap, src_frames * 4))
                if written - read < src_frames:
                    return b""
                index = read % cap
                first = min(src_frames, cap - index)
                start = HEADER_SIZE + index * 8
                raw = self._mm[start : start + first * 8] + self._mm[HEADER_SIZE : HEADER_SIZE + (src_frames - first) * 8]
                samples = struct.unpack(f"<{src_frames * 2}f", raw)
                result = [
                    int(max(-32768, min(32767, (sum(samples[i : i + 6]) / 6.0) * 32768.0)))
                    for i in range(0, len(samples), 6)
                ]
                struct.pack_into("<Q", self._mm, 32, read + src_frames)
                return struct.pack(f"<{len(result)}h", *result)

            # General path for any sample rate (e.g. 44100, 16000, 96000) and 1-2 channels
            duration_sec = frames / 48000.0
            src_frames = max(1, int(round(rate * duration_sec)))
            out_frames = max(1, int(round(target_rate * duration_sec)))
            if src_frames > cap:
                return b""
            read = max(read, written - min(cap, src_frames * 4))
            if written - read < src_frames:
                return b""
            stride = channels * 4
            index = read % cap
            first = min(src_frames, cap - index)
            start = HEADER_SIZE + index * stride
            raw = self._mm[start : start + first * stride] + self._mm[HEADER_SIZE : HEADER_SIZE + (src_frames - first) * stride]
            samples = struct.unpack(f"<{src_frames * channels}f", raw)
            if channels == 2:
                mono = [0.5 * (samples[i] + samples[i + 1]) for i in range(0, len(samples), 2)]
            else:
                mono = list(samples)

            if len(mono) == out_frames:
                resampled = mono
            elif len(mono) > out_frames:
                step = len(mono) / float(out_frames)
                resampled = []
                for i in range(out_frames):
                    lo = int(i * step)
                    hi = max(lo + 1, min(len(mono), int((i + 1) * step)))
                    resampled.append(sum(mono[lo:hi]) / float(hi - lo))
            else:
                step = len(mono) / float(out_frames)
                resampled = []
                for i in range(out_frames):
                    pos = i * step
                    idx = min(int(pos), len(mono) - 1)
                    frac = pos - idx
                    s0 = mono[idx]
                    s1 = mono[min(idx + 1, len(mono) - 1)]
                    resampled.append(s0 + (s1 - s0) * frac)

            pcm = [int(max(-32768, min(32767, val * 32768.0))) for val in resampled]
            struct.pack_into("<Q", self._mm, 32, read + src_frames)
            return struct.pack(f"<{len(pcm)}h", *pcm)

    def _write_stereo_floats_locked(self, stereo: list[float], cap: int, write_pos: int) -> None:
        frames = len(stereo) // 2
        if frames == 0:
            return
        if frames > cap:
            stereo = stereo[-(cap * 2) :]
            frames = cap

        assert self._mm is not None
        idx = write_pos % cap
        first = min(frames, cap - idx)
        first_floats = first * 2
        byte_off = HEADER_SIZE + idx * 2 * 4
        self._mm[byte_off : byte_off + first_floats * 4] = struct.pack(
            f"<{first_floats}f", *stereo[:first_floats]
        )
        rem = frames - first
        if rem > 0:
            rem_floats = rem * 2
            self._mm[HEADER_SIZE : HEADER_SIZE + rem_floats * 4] = struct.pack(
                f"<{rem_floats}f", *stereo[first_floats:]
            )
        struct.pack_into("<Q", self._mm, 24, write_pos + frames)

    def write_stereo_floats(self, stereo: list[float]) -> None:
        """Write pre-resampled interleaved stereo float32 samples into the ring buffer."""
        if len(stereo) < 2:
            return
        with self._io_lock:
            self.open()
            assert self._mm is not None
            _, _, _, _, cap, _, write_pos, _ = struct.unpack_from(HEADER_FORMAT, self._mm, 0)
            self._write_stereo_floats_locked(stereo, cap, write_pos)

    def write_pcm16_mono(self, pcm_bytes: bytes, source_rate: int = 16000) -> None:
        """Write Int16 mono PCM frames into the shared stereo float32 ring buffer."""
        if not pcm_bytes:
            return
        with self._io_lock:
            self.open()
            assert self._mm is not None
            _, _, ring_rate, _, cap, _, write_pos, _ = struct.unpack_from(HEADER_FORMAT, self._mm, 0)
            target_rate = ring_rate or 48000
            stereo = _resample_int16_mono_to_stereo_floats(pcm_bytes, source_rate, target_rate)
            self._write_stereo_floats_locked(stereo, cap, write_pos)
