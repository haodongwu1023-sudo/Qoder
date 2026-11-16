"""Standalone FastAPI + WebSocket server for real-time translation with cloned voice."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Any, AsyncIterator, Optional, Union
import weakref

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from src.audio_device import active_portaudio_stream_count, list_audio_devices
from src.audio_io import record_wav_sample
from src.config import settings
from src.glossary_store import GlossaryStore
from src.history_store import HistoryStore
from src.native_audio import install_hal_driver
from src.pipeline import PipelineError, SplitTranslationPipeline
from src.session_manager import SessionConfig, TranslationSessionManager
from src.voice_clone import MiniMaxVoiceCloneClient, VoiceCloneError, resolve_and_validate_voice_id
from src.voice_library import VoiceLibrary

STATIC_DIR = Path(__file__).resolve().parent / "static"


class _LoopBoundAsyncLock:
    """Lazy per-event-loop asyncio.Lock safe across Python 3.9 loop boundaries."""

    def __init__(self) -> None:
        self._locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = (
            weakref.WeakKeyDictionary()
        )

    def _current(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        lock = self._locks.get(loop)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[loop] = lock
        return lock

    def locked(self) -> bool:
        try:
            return self._current().locked()
        except RuntimeError:
            return False

    async def __aenter__(self) -> "_LoopBoundAsyncLock":
        await self._current().acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._current().release()


session_control_lock = _LoopBoundAsyncLock()

voice_library = VoiceLibrary(data_dir="data")
glossary_store = GlossaryStore(data_dir="data")
history_store = HistoryStore(data_dir="data")
pipeline = SplitTranslationPipeline()
clone_client = MiniMaxVoiceCloneClient()
session_manager = TranslationSessionManager(
    voice_library=voice_library,
    glossary_store=glossary_store,
    history_store=history_store,
)


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    async def _hal_idle_monitor() -> None:
        last_hal_state: tuple[bool, int, int] | None = None
        try:
            while True:
                await asyncio.sleep(1.5)
                session_manager.idle_passthrough.probe_rings()
                if not session_manager.running and not session_manager._draining:
                    session_manager.idle_passthrough.sync()
                st = session_manager.idle_passthrough.status()
                hal_state = (
                    bool(st["hal_active"]),
                    int(st["mic_clients"]),
                    int(st["meeting_clients"]),
                )
                if hal_state != last_hal_state:
                    last_hal_state = hal_state
                    await session_manager.broadcast(
                        {
                            "type": "hal_activity",
                            "hal_active": hal_state[0],
                            "mic_clients": hal_state[1],
                            "meeting_clients": hal_state[2],
                        }
                    )
        except asyncio.CancelledError:
            raise

    monitor_task = asyncio.create_task(_hal_idle_monitor())
    try:
        yield
    finally:
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        await session_manager.idle_passthrough.configure(enabled=False)
        await session_manager.idle_passthrough.pause()


app = FastAPI(title="Voice Clone Realtime Translator", version="0.3.0", lifespan=_lifespan)


class VoiceCloneRequest(BaseModel):
    audio_path: str
    voice_id: str = ""
    name: str = ""
    sample_text: Optional[str] = None


class RecordSampleRequest(BaseModel):
    seconds: float = 10.0
    device_index: Optional[int] = None
    filename: str = "my_voice_sample.wav"


class StartSessionRequest(BaseModel):
    source_lang: str = "zh"
    target_lang: str = "en"
    engine_mode: str = "alibaba_live"
    live_voice_clone: bool = True
    voice_id: str = ""
    speed: float = 1.0
    input_device_index: Optional[int] = None
    output_device_index: Optional[int] = None
    output_sink: str = "device"
    noise_suppression: bool = False
    use_hardware_aec: bool = True
    echo_guard: bool = True
    enable_inbound: bool = False
    passthrough_inbound: bool = False
    inbound_device_index: Optional[int] = None
    inbound_voice_id: str = ""
    vad_silence_ms: int = 400
    idle_auto_stop_minutes: int = 3
    glossary: dict[str, str] = {}


class LanguageSwitchRequest(BaseModel):
    source_lang: str
    target_lang: str


class GlossaryTermRequest(BaseModel):
    source_term: str
    target_term: str


class UpdateSettingsRequest(BaseModel):
    dashscope_api_key: Optional[str] = None
    minimax_api_key: Optional[str] = None
    minimax_group_id: Optional[str] = None
    asr_model: Optional[str] = None
    mt_model: Optional[str] = None
    minimax_tts_model: Optional[str] = None


class TranslateSpeakRequest(BaseModel):
    text: str
    source_lang: str = "zh"
    target_lang: str = "en"
    voice_id: Optional[str] = None
    speed: float = 1.0
    glossary: Optional[dict[str, str]] = None


@app.get("/")
async def index() -> FileResponse:
    """Serve the standalone desktop-style web UI."""
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/overlay")
async def overlay() -> FileResponse:
    """Serve the floating always-on-top bilingual subtitle bar."""
    return FileResponse(
        STATIC_DIR / "overlay.html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/health")
async def health() -> dict[str, Union[str, int, bool]]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "port": settings.app_port,
        "running": session_manager.running,
        "virtual_mic_device": settings.virtual_mic_output_device,
    }


@app.get("/api/audio/devices")
async def audio_devices(refresh: bool = False) -> dict[str, Any]:
    """List local audio input/output devices and virtual mic candidates (macOS/Windows)."""
    # Only re-initialize PortAudio after suspending idle passthrough and while no streams remain active.
    if not refresh or session_manager.running or session_manager._draining:
        return await asyncio.to_thread(list_audio_devices, False)
    async with session_control_lock:
        if session_manager.running or session_manager._draining:
            return await asyncio.to_thread(list_audio_devices, False)
        async with session_manager.idle_passthrough.suspend_temporarily():
            safe_refresh = active_portaudio_stream_count() == 0
            return await asyncio.to_thread(list_audio_devices, safe_refresh)


@app.get("/api/audio/passthrough")
async def get_idle_passthrough() -> dict[str, Any]:
    """Return idle HAL ring original-audio passthrough status."""
    session_manager.idle_passthrough.probe_rings()
    return session_manager.idle_passthrough.status()


class IdlePassthroughRequest(BaseModel):
    enabled: Optional[bool] = None
    input_device_index: Optional[int] = None
    output_device_index: Optional[int] = None
    update_devices: bool = False


@app.post("/api/audio/passthrough")
async def configure_idle_passthrough(req: IdlePassthroughRequest) -> dict[str, Any]:
    """Enable, disable, or update devices for idle meeting original-audio passthrough."""
    return await session_manager.idle_passthrough.configure(
        enabled=req.enabled,
        input_device_index=req.input_device_index,
        output_device_index=req.output_device_index,
        update_devices=req.update_devices,
    )


@app.post("/api/audio/install-hal-driver")
async def install_hal_audio_driver() -> dict[str, Any]:
    """Install compiled VoiceCloneAudioDriver.driver into /Library/Audio/Plug-Ins/HAL/ (admin prompt)."""
    from src.native_audio import is_hal_driver_installed, is_hal_driver_up_to_date

    if session_control_lock.locked():
        raise HTTPException(status_code=409, detail="音频正在切换，请稍后重试。")
    async with session_control_lock:
        if session_manager.running or session_manager._draining:
            raise HTTPException(status_code=409, detail="请先停止同传，再安装或更新会议驱动。")
        if is_hal_driver_installed() and is_hal_driver_up_to_date():
            return {"installed": True, "message": "会议驱动已安装且为最新版本，请重新检测设备。"}
        ok, message = await asyncio.to_thread(install_hal_driver)
    if not ok:
        raise HTTPException(status_code=400, detail=message)
    return {"installed": True, "message": message}


@app.get("/api/settings")
async def get_settings() -> dict[str, Any]:
    """Return non-sensitive settings status (masked keys)."""
    return {
        "has_dashscope_key": bool(settings.dashscope_api_key),
        "has_minimax_key": bool(settings.minimax_api_key),
        "minimax_group_id": settings.minimax_group_id,
        "asr_model": settings.asr_model,
        "mt_model": settings.mt_model,
        "minimax_tts_model": settings.minimax_tts_model,
    }


def _persist_env_updates(updates: dict[str, str]) -> None:
    """Persist updated key/value pairs into local .env so restarts keep settings."""
    env_path = Path(".env")
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    remaining = dict(updates)
    new_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in line:
            key = line.split("=", 1)[0].strip()
            if key in remaining:
                new_lines.append(f"{key}={remaining.pop(key)}")
                continue
        new_lines.append(line)
    for key, val in remaining.items():
        new_lines.append(f"{key}={val}")
    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


@app.post("/api/settings")
async def update_settings(req: UpdateSettingsRequest) -> dict[str, Any]:
    """Update runtime API keys and model preferences and persist to .env."""
    env_updates: dict[str, str] = {}
    if req.dashscope_api_key is not None and req.dashscope_api_key.strip():
        val = req.dashscope_api_key.strip()
        settings.dashscope_api_key = val
        env_updates["DASHSCOPE_API_KEY"] = val
    if req.minimax_api_key is not None and req.minimax_api_key.strip():
        val = req.minimax_api_key.strip()
        settings.minimax_api_key = val
        clone_client.api_key = val
        env_updates["MINIMAX_API_KEY"] = val
    if req.minimax_group_id is not None:
        val = req.minimax_group_id.strip()
        settings.minimax_group_id = val
        clone_client.group_id = val
        env_updates["MINIMAX_GROUP_ID"] = val
    if req.asr_model:
        val = req.asr_model.strip()
        settings.asr_model = val
        env_updates["ASR_MODEL"] = val
    if req.mt_model:
        val = req.mt_model.strip()
        settings.mt_model = val
        env_updates["MT_MODEL"] = val
    if req.minimax_tts_model:
        val = req.minimax_tts_model.strip()
        settings.minimax_tts_model = val
        env_updates["MINIMAX_TTS_MODEL"] = val
    if env_updates:
        _persist_env_updates(env_updates)
    return await get_settings()


@app.get("/api/voices")
async def list_voices() -> dict[str, Any]:
    """List cloned voices from the local voice library."""
    return {"voices": voice_library.list_voices()}


@app.post("/api/voices/record")
async def record_voice_sample(req: RecordSampleRequest) -> dict[str, Any]:
    """Record a microphone audio sample to local WAV for voice cloning."""
    safe_name = Path(req.filename).name
    target = voice_library.samples_dir / safe_name
    try:
        saved = await asyncio.to_thread(
            record_wav_sample,
            target,
            req.seconds,
            settings.audio_sample_rate,
            req.device_index,
        )
        return {"audio_path": str(saved), "seconds": req.seconds}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Recording failed: {exc}") from exc


@app.post("/api/voices/clone")
async def create_voice_clone(req: VoiceCloneRequest) -> dict[str, Any]:
    """Upload a local audio sample, create a cloned voice_id, and save to voice library."""
    try:
        resolved_id = resolve_and_validate_voice_id(req.voice_id)
        file_id = await clone_client.upload_sample(Path(req.audio_path))
        result = await clone_client.create_clone(
            file_id=file_id, voice_id=resolved_id, text=req.sample_text
        )
        entry = voice_library.add_voice(
            voice_id=result.get("resolved_voice_id", resolved_id),
            name=req.name or resolved_id,
            sample_path=req.audio_path,
            file_id=file_id,
        )
        return {"file_id": file_id, "entry": entry, "result": result}
    except (FileNotFoundError, VoiceCloneError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/voices/{voice_id}/default")
async def set_default_voice(voice_id: str) -> dict[str, Any]:
    """Set a cloned voice as default."""
    entry = voice_library.set_default(voice_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Voice not found")
    return {"entry": entry}


@app.delete("/api/voices/{voice_id}")
async def delete_voice(voice_id: str) -> dict[str, bool]:
    """Delete a cloned voice from the local library."""
    ok = voice_library.delete_voice(voice_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Voice not found")
    return {"deleted": True}


@app.get("/api/session/status")
async def session_status() -> dict[str, Any]:
    """Return current live translation session status."""
    session_manager.idle_passthrough.probe_rings()
    pas = session_manager.idle_passthrough.status()
    return {
        "running": session_manager.running,
        "ready": session_manager.session_ready,
        "microphone_muted": session_manager.microphone_muted,
        "message": session_manager.status_message,
        "draining": session_manager._draining,
        "source_lang": session_manager.config.source_lang,
        "target_lang": session_manager.config.target_lang,
        "voice_id": session_manager.config.voice_id,
        "output_sink": session_manager.config.output_sink,
        "enable_inbound": session_manager.config.enable_inbound,
        "passthrough_inbound": session_manager.config.passthrough_inbound,
        "hal_active": bool(pas["hal_active"]),
        "mic_clients": int(pas["mic_clients"]),
        "meeting_clients": int(pas["meeting_clients"]),
    }


@app.post("/api/session/start")
async def start_session(req: StartSessionRequest) -> dict[str, Any]:
    """Start live microphone -> ASR -> MT -> Cloned-Voice TTS -> Virtual Mic session."""
    if session_manager._draining:
        raise HTTPException(status_code=409, detail="正在收尾，请稍后")
    if session_control_lock.locked():
        raise HTTPException(status_code=409, detail="音频正在切换，请稍后重试。")
    if req.output_sink == "voiceclone_mic" or (req.enable_inbound and req.inbound_device_index == -2):
        from src.native_audio import is_hal_driver_installed

        if not is_hal_driver_installed():
            raise HTTPException(status_code=409, detail="未安装 VoiceClone 会议驱动，请先安装或选择其他输入。")
    cfg = SessionConfig(
        source_lang=req.source_lang,
        target_lang=req.target_lang,
        engine_mode=req.engine_mode,
        live_voice_clone=req.live_voice_clone,
        voice_id=req.voice_id,
        speed=req.speed,
        input_device_index=req.input_device_index,
        output_device_index=req.output_device_index,
        output_sink=req.output_sink,
        noise_suppression=req.noise_suppression,
        use_hardware_aec=req.use_hardware_aec,
        echo_guard=req.echo_guard,
        enable_inbound=req.enable_inbound,
        passthrough_inbound=req.passthrough_inbound,
        inbound_device_index=req.inbound_device_index,
        inbound_voice_id=req.inbound_voice_id,
        vad_silence_ms=max(200, min(2000, int(req.vad_silence_ms))),
        idle_auto_stop_minutes=req.idle_auto_stop_minutes,
        glossary=req.glossary,
    )
    async with session_control_lock:
        if not session_manager.running:
            await session_manager.start(cfg)
    return {"running": session_manager.running}


@app.post("/api/session/languages")
async def switch_session_languages(req: LanguageSwitchRequest) -> dict[str, str]:
    """Switch source and target languages on the fly (including from floating overlay)."""
    async with session_control_lock:
        await session_manager.update_languages(req.source_lang, req.target_lang)
    return {"source_lang": session_manager.config.source_lang, "target_lang": session_manager.config.target_lang}


@app.get("/api/glossary")
async def get_glossary() -> dict[str, Any]:
    return {"terms": glossary_store.list_terms()}


@app.post("/api/glossary")
async def add_glossary_term(req: GlossaryTermRequest) -> dict[str, Any]:
    terms = glossary_store.upsert_term(req.source_term, req.target_term)
    session_manager.config.glossary = terms
    return {"terms": terms}


@app.delete("/api/glossary/{source_term}")
async def remove_glossary_term(source_term: str) -> dict[str, Any]:
    terms = glossary_store.delete_term(source_term)
    session_manager.config.glossary = terms
    return {"terms": terms}


@app.get("/api/history")
async def get_history() -> dict[str, Any]:
    return {"sessions": await asyncio.to_thread(history_store.list_sessions)}


@app.get("/api/history/{session_id}/markdown")
async def export_history(session_id: str) -> Response:
    sessions = await asyncio.to_thread(history_store.list_sessions)
    session = next((s for s in sessions if s["session_id"] == session_id), None)
    if session is None:
        raise HTTPException(status_code=404, detail="找不到该会话")
    lines = ["# 同传记录", "", f'{session["source_lang"]} → {session["target_lang"]}',
             f'开始：{session["started_at"]}', ""]
    for item in session["items"]:
        lines.extend([f'**原文：** {item["source"]}', "", f'**译文：** {item["target"]}', ""])
    return Response("\n".join(lines), media_type="text/markdown; charset=utf-8",
                    headers={"Content-Disposition": "attachment; filename=translation.md"})


@app.delete("/api/history")
async def clear_history() -> dict[str, bool]:
    await asyncio.to_thread(history_store.clear)
    return {"cleared": True}


class MicrophoneMuteRequest(BaseModel):
    muted: bool


class PassthroughInboundRequest(BaseModel):
    enabled: bool


@app.post("/api/session/microphone")
async def set_microphone(req: MicrophoneMuteRequest) -> dict[str, bool]:
    await session_manager.set_microphone_muted(req.muted)
    return {"muted": session_manager.microphone_muted}


@app.post("/api/session/passthrough")
async def set_session_passthrough(req: PassthroughInboundRequest) -> dict[str, bool]:
    enabled = await session_manager.set_passthrough_inbound(req.enabled)
    return {"enabled": enabled}


@app.post("/api/session/stop")
async def stop_session() -> dict[str, Any]:
    """Stop the active live translation session."""
    async with session_control_lock:
        await session_manager.stop_after_tail()
    return {"running": session_manager.running}


@app.websocket("/ws/events")
async def ws_events(ws: WebSocket) -> None:
    """Stream real-time bilingual subtitles and pipeline status to the UI."""
    await ws.accept()
    queue = session_manager.register_ws(ws)

    async def send_events() -> None:
        session_manager.idle_passthrough.probe_rings()
        pas = session_manager.idle_passthrough.status()
        await ws.send_json({
            "type": "status",
            "running": session_manager.running,
            "ready": session_manager.session_ready,
            "message": session_manager.status_message,
            "output_sink": session_manager.config.output_sink,
            "passthrough_inbound": session_manager.config.passthrough_inbound,
            "hal_active": bool(pas["hal_active"]),
            "mic_clients": int(pas["mic_clients"]),
            "meeting_clients": int(pas["meeting_clients"]),
        })
        while True:
            await asyncio.wait_for(ws.send_json(await queue.get()), timeout=2.0)

    async def watch_disconnect() -> None:
        while True:
            await ws.receive_text()

    tasks = (asyncio.create_task(send_events()), asyncio.create_task(watch_disconnect()))
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    except (WebSocketDisconnect, asyncio.TimeoutError, OSError, asyncio.CancelledError):
        pass
    finally:
        session_manager.unregister_ws(ws)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@app.websocket("/ws/translate-speak")
async def ws_translate_speak(ws: WebSocket) -> None:
    """Receive text utterances, translate, and stream back cloned-voice PCM audio."""
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            req = TranslateSpeakRequest(**data)
            translated = await pipeline.translate_text(
                text=req.text,
                source_lang=req.source_lang,
                target_lang=req.target_lang,
                glossary=req.glossary,
            )
            await ws.send_json({"event": "translated_text", "text": translated})

            async for pcm_chunk in pipeline.stream_cloned_tts(
                text=translated,
                voice_id=req.voice_id,
                speed=req.speed,
            ):
                await ws.send_bytes(pcm_chunk)
            await ws.send_json({"event": "audio_done"})
    except WebSocketDisconnect:
        return
    except PipelineError as exc:
        await ws.send_json({"event": "error", "detail": str(exc)})
        await ws.close()
