"""FastAPI server for EchoBridge Web (static frontend + multi-user WSS translation)."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import settings
from app.minimax_client import MiniMaxVoiceCloneClient, VoiceCloneError, resolve_and_validate_voice_id
from app.web_session import run_web_room_ws, run_web_translate_ws

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

app = FastAPI(title="EchoBridge Web Translator", version="1.0.0")

_origins = [o.strip() for o in (settings.web_allowed_origins or "*").split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class WebVoiceCloneRequest(BaseModel):
    audio_base64: str
    voice_id: str = ""
    name: str = ""
    sample_text: Optional[str] = None
    minimax_api_key: str = ""
    minimax_group_id: str = ""


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(
        FRONTEND_DIR / "index.html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "port": settings.app_port,
        "vad_silence_ms": settings.vad_silence_ms,
        "has_server_dashscope_key": bool(settings.dashscope_api_key),
    }


@app.get("/api/web/config")
async def get_web_config() -> dict[str, Any]:
    return {
        "has_server_dashscope_key": bool(settings.dashscope_api_key),
        "has_server_minimax_key": bool(settings.minimax_api_key),
        "demo_max_seconds": int(settings.web_demo_max_seconds),
        "vad_silence_ms": int(settings.vad_silence_ms),
        "default_cloned_voice_id": settings.default_cloned_voice_id,
    }


@app.post("/api/web/voices/clone")
async def create_web_voice_clone(req: WebVoiceCloneRequest) -> dict[str, Any]:
    raw_b64 = (req.audio_base64 or "").split(",", 1)[-1].strip()
    if not raw_b64:
        raise HTTPException(status_code=400, detail="缺少录音音频数据。")
    if len(raw_b64) > 12_000_000:
        raise HTTPException(status_code=400, detail="音频文件过大，请控制在 10~20 秒以内。")
    try:
        wav_bytes = base64.b64decode(raw_b64, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="音频编码格式无效。") from exc

    client = MiniMaxVoiceCloneClient(
        api_key=req.minimax_api_key.strip() or None,
        group_id=req.minimax_group_id.strip() or None,
    )
    try:
        resolved_id = resolve_and_validate_voice_id(req.voice_id)
        file_id = await client.upload_wav_bytes(wav_bytes)
        result = await client.create_clone(
            file_id=file_id, voice_id=resolved_id, text=req.sample_text
        )
        final_voice_id = str(result.get("resolved_voice_id", resolved_id))
        return {
            "file_id": file_id,
            "voice_id": final_voice_id,
            "name": (req.name or final_voice_id).strip(),
        }
    except VoiceCloneError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.websocket("/ws/web-translate")
async def ws_web_translate(ws: WebSocket) -> None:
    await run_web_translate_ws(ws)


@app.websocket("/ws/room/{room_id}")
async def ws_web_room(ws: WebSocket, room_id: str) -> None:
    await run_web_room_ws(ws, room_id)


if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
