"""Native macOS audio bridge for VoiceProcessingIO (AEC) and CoreAudio ProcessTap."""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENGINE_BINARY = PROJECT_ROOT / "bin" / "VoiceCloneAudioEngine"
HAL_DRIVER_BUNDLE = PROJECT_ROOT / "bin" / "VoiceCloneAudioDriver.driver"
HAL_INSTALL_SCRIPT = PROJECT_ROOT / "native" / "hal_driver" / "install_driver.sh"
INSTALLED_HAL_PATH = Path("/Library/Audio/Plug-Ins/HAL/VoiceCloneAudioDriver.driver")
FRAME_BYTES = 960  # 30ms @ 16kHz Int16 mono


def is_native_engine_available() -> bool:
    return (
        platform.system() == "Darwin"
        and ENGINE_BINARY.exists()
        and os.access(ENGINE_BINARY, os.X_OK)
    )


def is_hal_driver_built() -> bool:
    return HAL_DRIVER_BUNDLE.exists()


def is_hal_driver_installed() -> bool:
    return INSTALLED_HAL_PATH.exists()


def is_hal_driver_up_to_date() -> bool:
    """Return True if the installed HAL driver binary matches our local build."""
    built_bin = HAL_DRIVER_BUNDLE / "Contents" / "MacOS" / "VoiceCloneAudioDriver"
    inst_bin = INSTALLED_HAL_PATH / "Contents" / "MacOS" / "VoiceCloneAudioDriver"
    if not inst_bin.exists():
        return False
    if not built_bin.exists():
        return True
    try:
        return built_bin.read_bytes() == inst_bin.read_bytes()
    except OSError:
        return True


def install_hal_driver() -> tuple[bool, str]:
    """Run install_driver.sh (prompts macOS admin dialog if non-root)."""
    if platform.system() != "Darwin":
        return False, "仅 macOS 支持安装 CoreAudio HAL 虚拟声卡驱动（Windows 请使用 VB-Cable）。"
    if not HAL_INSTALL_SCRIPT.exists():
        return False, f"安装脚本不存在: {HAL_INSTALL_SCRIPT}"
    proc = subprocess.run(
        ["/bin/bash", str(HAL_INSTALL_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "用户取消或安装失败").strip()
        return False, msg
    return True, "VoiceCloneAudioDriver.driver 已安装并重启 CoreAudio（可用「VoiceClone Mic」与「VoiceClone Meeting Audio」）。"


class _EngineSubprocessBase:
    """Manages an async subprocess streaming 960-byte 16kHz mono PCM16 frames on stdout."""

    def __init__(self, args: list[str]) -> None:
        self._args = args
        self._proc: asyncio.subprocess.Process | None = None
        self._stopped = False

    async def start(self) -> None:
        self._stopped = False
        self._proc = await asyncio.create_subprocess_exec(
            str(ENGINE_BINARY),
            *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    @property
    def pid(self) -> int | None:
        return None if self._proc is None else self._proc.pid

    async def read_frame(self) -> bytes | None:
        proc = self._proc
        if self._stopped or proc is None or proc.stdout is None:
            return None
        try:
            chunk = await asyncio.wait_for(proc.stdout.readexactly(FRAME_BYTES), timeout=0.5)
            return chunk
        except asyncio.TimeoutError:
            if self._stopped or proc.returncode is not None:
                return None
            return b""
        except asyncio.IncompleteReadError:
            return None

    async def stop(self) -> None:
        self._stopped = True
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.is_closing():
                proc.stdin.close()
        except Exception:
            pass
        if proc.returncode is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()
        if proc.stderr:
            err = (await proc.stderr.read()).decode("utf-8", errors="replace").strip()
            if err:
                log = logger.info if proc.returncode == 0 else logger.warning
                log("Native audio engine: %s", err)


class NativeVPIODuplex(_EngineSubprocessBase):
    """Apple VoiceProcessingIO full-duplex stream (Hardware AEC mic capture + speaker render)."""

    def __init__(self) -> None:
        super().__init__(["--mode", "vpio-duplex"])
        self._write_lock = asyncio.Lock()

    async def write_playback(self, pcm_bytes: bytes) -> None:
        if self._stopped or self._proc is None or self._proc.stdin is None:
            return
        async with self._write_lock:
            if self._proc is None or self._proc.stdin is None or self._proc.stdin.is_closing():
                return
            try:
                self._proc.stdin.write(len(pcm_bytes).to_bytes(4, "little") + pcm_bytes)
                await self._proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return


class NativeSystemTap(_EngineSubprocessBase):
    """macOS CoreAudio ProcessTap capturing system/meeting audio while excluding our PID."""

    def __init__(self, exclude_pids: list[int] | None = None) -> None:
        args = ["--mode", "system-tap"]
        for pid in exclude_pids or [os.getpid()]:
            args.extend(["--exclude-pid", str(pid)])
        super().__init__(args)


class NativeSpeaker(_EngineSubprocessBase):
    """Persistent Core Audio output for 24 kHz PCM16 on the default speaker."""

    def __init__(self) -> None:
        super().__init__(["--mode", "speaker"])

    async def start(self) -> None:
        await super().start()
        try:
            ready = await asyncio.wait_for(self._proc.stdout.readexactly(1), timeout=1.5)
            if ready != b"R":
                raise OSError("Unexpected AudioQueue startup reply")
        except BaseException:
            await self.stop()
            raise

    async def write(self, pcm: bytes) -> None:
        proc = self._proc
        if proc is None or proc.returncode is not None or proc.stdin is None:
            raise OSError("Core Audio speaker exited")
        proc.stdin.write(len(pcm).to_bytes(4, "little") + pcm)
        await proc.stdin.drain()
