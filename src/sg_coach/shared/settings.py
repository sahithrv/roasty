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
    gta_wasted_clear_threshold: float = Field(default=0.04, ge=0.0, le=1.0)
    gta_wasted_clear_frames: int = Field(default=3, ge=1, le=10)
    gta_wasted_edge_low_threshold: int = Field(default=80, ge=0, le=255)
    gta_wasted_edge_high_threshold: int = Field(default=180, ge=1, le=255)
    gta_wasted_debug_enabled: bool = False
    gta_wasted_debug_score_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    gta_wasted_debug_max_saves: int = Field(default=25, ge=1, le=500)
    gta_wasted_debug_save_first_n: int = Field(default=5, ge=0, le=50)
    gta_chaos_downscale_width: int = Field(default=320, ge=120, le=960)
    gta_chaos_motion_threshold: float = Field(default=0.16, ge=0.0, le=1.0)
    gta_chaos_flash_threshold: float = Field(default=0.035, ge=0.0, le=1.0)
    gta_chaos_edge_threshold: float = Field(default=0.18, ge=0.0, le=1.0)
    gta_chaos_score_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    gta_chaos_confirm_frames: int = Field(default=3, ge=1, le=5)
    gta_chaos_cooldown_seconds: int = Field(default=20, ge=1, le=60)
    gta_chaos_startup_delay_seconds: int = Field(default=12, ge=0, le=120)
    gta_wanted_confirm_frames: int = Field(default=3, ge=1, le=6)
    gta_wanted_slot_activation_threshold: float = Field(default=0.20, ge=0.01, le=0.95)
    gta_wanted_slot_present_threshold: float = Field(default=0.12, ge=0.01, le=0.95)
    gta_wanted_shape_presence_threshold: float = Field(default=0.46, ge=0.05, le=0.99)
    gta_wanted_active_score_threshold: float = Field(default=0.52, ge=0.05, le=0.99)
    gta_wanted_startup_delay_seconds: int = Field(default=6, ge=0, le=120)
    gta_wanted_roi_top_pct: float = Field(default=0.004, ge=0.0, le=0.2)
    gta_wanted_roi_bottom_pct: float = Field(default=0.060, ge=0.01, le=0.3)
    gta_wanted_roi_left_pct: float = Field(default=0.910, ge=0.5, le=0.99)
    gta_wanted_roi_right_pct: float = Field(default=0.990, ge=0.7, le=1.0)
    gta_wanted_slot_centers_pct: str = "0.125,0.308,0.491,0.674,0.864"
    gta_wanted_slot_half_width_pct: float = Field(default=0.086, ge=0.02, le=0.25)
    gta_wanted_slot_top_pct: float = Field(default=0.08, ge=0.0, le=0.5)
    gta_wanted_slot_bottom_pct: float = Field(default=0.92, ge=0.2, le=1.0)
    valorant_map_templates_dir: Path = Path("data/templates/valorant/maps")
    valorant_map_match_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    valorant_map_confirm_frames: int = Field(default=2, ge=1, le=6)
    valorant_map_clear_threshold: float = Field(default=0.18, ge=0.0, le=1.0)
    valorant_map_clear_frames: int = Field(default=2, ge=1, le=10)
    valorant_map_roi_top_pct: float = Field(default=0.40, ge=0.0, le=0.5)
    valorant_map_roi_bottom_pct: float = Field(default=0.60, ge=0.05, le=0.7)
    valorant_map_roi_left_pct: float = Field(default=0.24, ge=0.0, le=0.8)
    valorant_map_roi_right_pct: float = Field(default=0.76, ge=0.2, le=1.0)
    commentary_context_frame_count: int = Field(default=5, ge=1, le=12)
    commentary_recent_event_limit: int = Field(default=5, ge=1, le=12)

    default_persona: str = "off_the_walls"
    commentary_enabled: bool = True

    grok_model: str = "grok-4-1-fast-reasoning"
    grok_api_base_url: str = "https://api.x.ai/v1"
    grok_timeout_seconds: int = Field(default=60, ge=5, le=3600)
    grok_api_key: str | None = None
    realtime_enabled: bool = False
    realtime_ws_url: str = "wss://api.x.ai/v1/realtime"
    realtime_api_key: str | None = None
    realtime_personality: str = "off_the_walls"
    realtime_voice: str = "leo"
    realtime_language: str = "en"
    realtime_play_audio: bool = True
    realtime_output_sample_rate: int = Field(default=24000, ge=8000, le=48000)
    realtime_audio_device: str | None = None
    realtime_enable_user_speech: bool = True
    realtime_input_sample_rate: int = Field(default=24000, ge=8000, le=48000)
    realtime_input_audio_device: str | None = None
    realtime_push_to_talk_key: str = "right alt"
    realtime_emit_commentary: bool = True
    realtime_emit_speech_cues: bool = True
    realtime_speech_cue_min_interval_seconds: int = Field(default=180, ge=0, le=3600)
    realtime_drop_speech_cues_on_commentary: bool = True

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

    @property
    def debug_speech_dir(self) -> Path:
        return self.data_dir / "debug_speech"

    @property
    def debug_realtime_dir(self) -> Path:
        return self.data_dir / "debug_realtime"


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    return Settings()
