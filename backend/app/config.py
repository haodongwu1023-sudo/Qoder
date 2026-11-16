"""Standalone web service configuration matching tuned app defaults."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    """Web server settings loaded from environment or root .env file."""

    model_config = SettingsConfigDict(
        env_file=str(_ROOT_ENV),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_host: str = "0.0.0.0"
    app_port: int = 8910
    web_allowed_origins: str = "*"
    web_demo_max_seconds: int = 180

    dashscope_api_key: str = ""
    asr_model: str = "qwen3-asr-flash-realtime"
    mt_model: str = "qwen-flash"

    minimax_api_key: str = ""
    minimax_group_id: str = ""
    minimax_tts_model: str = "speech-2.8-turbo"
    default_cloned_voice_id: str = ""

    audio_sample_rate: int = 16000
    vad_silence_ms: int = 400
    keepalive_interval_ms: int = 100
    keepalive_idle_ms: int = 150


settings = Settings()
