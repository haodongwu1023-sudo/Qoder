"""Isolated configuration for Voice Clone Realtime Translator."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from local .env without touching external systems."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_host: str = "127.0.0.1"
    app_port: int = 8900

    dashscope_api_key: str = ""
    asr_model: str = "qwen3-asr-flash-realtime"
    mt_model: str = "qwen-flash"

    minimax_api_key: str = ""
    minimax_group_id: str = ""
    minimax_tts_model: str = "speech-2.8-turbo"
    default_cloned_voice_id: str = ""

    audio_sample_rate: int = 16000
    virtual_mic_output_device: str = "BlackHole 2ch"

    optional_webhook_url: str = ""


settings = Settings()
