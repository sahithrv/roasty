from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and optional .env files."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SG_",
        case_sensitive=False,
        extra="ignore",
    )

    env: str = "dev"
    log_level: str = "INFO"
    data_dir: Path = Path("data")
    debug: bool = True

    capture_backend: str = "dxcam"
    target_fps: int = Field(default=12, ge=1, le=60)

    default_persona: str = "light_sarcasm"
    commentary_enabled: bool = True

    grok_model: str = "grok-4-1-fast-reasoning"
    grok_api_key: str | None = None

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def debug_frames_dir(self) -> Path:
        return self.data_dir / "debug_frames"

    @property
    def debug_clips_dir(self) -> Path:
        return self.data_dir / "debug_clips"


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    return Settings()

