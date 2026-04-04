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
    capture_monitor_id: int = Field(default=0, ge=0)
    target_fps: int = Field(default=12, ge=1, le=60)
    replay_buffer_seconds: int = Field(default=10, ge=2, le=30)
    replay_buffer_fps: int = Field(default=4, ge=1, le=12)
    replay_buffer_max_width: int = Field(default=640, ge=160, le=1920)
    gta_wasted_template_path: Path = Path("data/templates/gta/wasted.png")
    gta_wasted_match_threshold: float = Field(default=0.82, ge=0.0, le=1.0)
    gta_wasted_confirm_frames: int = Field(default=2, ge=1, le=5)
    gta_wasted_cooldown_seconds: int = Field(default=5, ge=1, le=30)
    gta_wasted_edge_low_threshold: int = Field(default=80, ge=0, le=255)
    gta_wasted_edge_high_threshold: int = Field(default=180, ge=1, le=255)
    gta_wasted_debug_enabled: bool = True
    gta_wasted_debug_score_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    gta_wasted_debug_max_saves: int = Field(default=25, ge=1, le=500)
    gta_wasted_debug_save_first_n: int = Field(default=5, ge=0, le=50)
    commentary_context_frame_count: int = Field(default=5, ge=1, le=12)
    commentary_recent_event_limit: int = Field(default=5, ge=1, le=12)

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

    @property
    def detector_debug_dir(self) -> Path:
        return self.debug_frames_dir / "detectors"

    @property
    def debug_commentary_dir(self) -> Path:
        return self.data_dir / "debug_commentary"


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    return Settings()
