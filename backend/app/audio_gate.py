"""Pure-Python PCM16 RMS utilities, foreground speech gate, and WAV validation."""

from __future__ import annotations

from collections import deque
import io
import math
import struct
import wave


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


def looks_like_impulse(frame: bytes, rms: float) -> bool:
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
    """Near-field foreground speech gate matching tuned desktop app thresholds."""

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
            open_thresh = max(open_thresh, min(-24.0, self._foreground_ref_dbfs - 12.0))
        if echo_duck_db > 0.0:
            open_thresh = min(-18.0, open_thresh + min(4.0, echo_duck_db))

        is_strong = db >= open_thresh
        if is_strong and not self._gate_open and looks_like_impulse(frame, rms):
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


def validate_wav_bytes(wav_bytes: bytes, min_seconds: float = 10.0) -> None:
    """Validate WAV duration and non-silent speech before uploading to MiniMax."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate() or 16000
            duration = frames / float(rate)
            if duration < min_seconds:
                raise ValueError(
                    f"录音时长仅 {duration:.1f} 秒，至少需要 {min_seconds:.0f} 秒清晰人声。"
                )
            pcm = wf.readframes(frames)
    except wave.Error as exc:
        raise ValueError("音频格式无效，请提交标准 WAV 人声样本。") from exc
    if compute_rms_int16(pcm) < 0.002:
        raise ValueError("未检测到有效人声（音量过低或全静音），请确认麦克风权限后重录。")
