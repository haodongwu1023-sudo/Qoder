"""Cross-platform audio device discovery for macOS (BlackHole) and Windows (VB-Cable)."""

from __future__ import annotations

import logging
import platform
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)

_pa_lock = threading.RLock()
_active_pa_streams: set[int] = set()


def active_portaudio_stream_count() -> int:
    """Return the number of currently open sounddevice streams registered in this process."""
    with _pa_lock:
        return len(_active_pa_streams)


def open_portaudio_stream(factory: Callable[[], Any], *, start: bool = False) -> Any:
    """Create (and optionally start) a sounddevice stream under the process PortAudio lock."""
    with _pa_lock:
        stream = factory()
        token = id(stream)
        _active_pa_streams.add(token)
        if start:
            try:
                stream.start()
            except Exception:
                _active_pa_streams.discard(token)
                try:
                    stream.close()
                except Exception:
                    pass
                raise
        return stream


def unregister_portaudio_stream(stream: Any) -> None:
    """Unregister a sounddevice stream after its context manager closes it."""
    if stream is None:
        return
    with _pa_lock:
        _active_pa_streams.discard(id(stream))


def close_portaudio_stream(stream: Any) -> None:
    """Stop, close, and unregister a tracked sounddevice stream under the PortAudio lock."""
    if stream is None:
        return
    with _pa_lock:
        try:
            if hasattr(stream, "stop"):
                try:
                    stream.stop()
                except Exception:
                    pass
            if hasattr(stream, "close"):
                try:
                    stream.close()
                except Exception:
                    pass
        finally:
            _active_pa_streams.discard(id(stream))


@contextmanager
def portaudio_stream(factory: Callable[[], Any]) -> Iterator[Any]:
    """Context manager registering a sounddevice stream before entry and unregistering after exit."""
    stream = open_portaudio_stream(factory, start=False)
    try:
        with stream:
            yield stream
    finally:
        unregister_portaudio_stream(stream)


def refresh_portaudio_devices(sd: Any) -> bool:
    """Re-initialize PortAudio's CoreAudio cache only when zero streams are active in-process."""
    with _pa_lock:
        if _active_pa_streams:
            logger.info(
                "Skipping PortAudio re-initialization while %d stream(s) remain active",
                len(_active_pa_streams),
            )
            return False
        if hasattr(sd, "_terminate") and hasattr(sd, "_initialize"):
            try:
                sd._terminate()
                sd._initialize()
                return True
            except Exception:
                return False
        return False


@dataclass
class AudioDeviceInfo:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    is_virtual_candidate: bool = False


VIRTUAL_DEVICE_HINTS = (
    "voiceclone",
    "blackhole",
    "cable input",
    "cable output",
    "vb-audio",
    "virtual",
    "loopback",
    "soundflower",
    "meeting audio",
    "interpreter send",
    "interpreter mic",
)


def list_audio_devices(refresh: bool = False) -> dict:
    """List input/output audio devices and highlight virtual mic candidates."""
    try:
        import sounddevice as sd
    except ImportError:
        return {
            "platform": platform.system(),
            "sounddevice_installed": False,
            "inputs": [],
            "outputs": [],
            "virtual_candidates": [],
        }

    with _pa_lock:
        if refresh:
            refresh_portaudio_devices(sd)
        devices = sd.query_devices()
    inputs: list[dict] = []
    outputs: list[dict] = []
    virtual_candidates: list[dict] = []

    for idx, dev in enumerate(devices):
        name = str(dev.get("name", ""))
        lower_name = name.lower()
        is_virtual = any(hint in lower_name for hint in VIRTUAL_DEVICE_HINTS)
        info = AudioDeviceInfo(
            index=idx,
            name=name,
            max_input_channels=int(dev.get("max_input_channels", 0)),
            max_output_channels=int(dev.get("max_output_channels", 0)),
            default_samplerate=float(dev.get("default_samplerate", 16000.0)),
            is_virtual_candidate=is_virtual,
        )
        record = asdict(info)
        if info.max_input_channels > 0:
            inputs.append(record)
        if info.max_output_channels > 0:
            outputs.append(record)
        if is_virtual:
            virtual_candidates.append(record)

    from src.native_audio import (
        is_hal_driver_built,
        is_hal_driver_installed,
        is_hal_driver_up_to_date,
        is_native_engine_available,
    )

    return {
        "platform": platform.system(),
        "sounddevice_installed": True,
        "inputs": inputs,
        "outputs": outputs,
        "virtual_candidates": virtual_candidates,
        "native_capabilities": {
            "hardware_aec": is_native_engine_available(),
            "system_tap": is_native_engine_available(),
            "hal_driver_built": is_hal_driver_built(),
            "hal_driver_installed": is_hal_driver_installed(),
            "hal_driver_up_to_date": is_hal_driver_up_to_date(),
        },
    }
