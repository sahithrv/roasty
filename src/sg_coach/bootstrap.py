from __future__ import annotations

from pathlib import Path

from sg_coach.shared.logging import configure_logging, get_logger
from sg_coach.shared.settings import Settings, load_settings


def ensure_runtime_dirs(settings: Settings) -> None:
    """Create local data directories used by the application runtime."""
    for path in (
        settings.data_dir,
        settings.sessions_dir,
        settings.debug_frames_dir,
        settings.debug_clips_dir,
    ):
        Path(path).mkdir(parents=True, exist_ok=True)


def bootstrap() -> Settings:
    """Load settings, configure shared services, and prepare runtime directories."""
    settings = load_settings()
    configure_logging(settings.log_level)
    ensure_runtime_dirs(settings)

    logger = get_logger(__name__)
    logger.info("bootstrap complete")
    logger.info(
        "runtime configured",
        extra={
            "env": settings.env,
            "debug": settings.debug,
            "capture_backend": settings.capture_backend,
            "target_fps": settings.target_fps,
        },
    )
    return settings

