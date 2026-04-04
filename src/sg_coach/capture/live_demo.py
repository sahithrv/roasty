from __future__ import annotations

import asyncio

from sg_coach.capture.dxcam_backend import DxcamFrameSource
from sg_coach.shared.logging import configure_logging, get_logger
from sg_coach.shared.settings import load_settings


logger = get_logger(__name__)


async def run_live_capture_demo(*, max_frames: int = 120) -> None:
    """Capture live frames and log basic frame metadata.

    This is the first real hardware-facing test:
    - no detector logic
    - no fusion logic
    - no memory updates
    - just prove that we can capture live frames reliably

    On the other PC, this is the command you should use first before wiring
    gameplay detection into the runtime.
    """
    settings = load_settings()
    source = DxcamFrameSource(settings=settings)

    logger.info(
        "live capture demo starting backend=%s monitor_id=%s target_fps=%s",
        settings.capture_backend,
        settings.capture_monitor_id,
        settings.target_fps,
    )

    frame_count = 0
    async for frame in source.frames():
        frame_count += 1
        logger.info(
            "captured frame=%s size=%sx%s monitor=%s",
            frame.frame_id,
            frame.width,
            frame.height,
            frame.monitor_id,
        )
        if frame_count >= max_frames:
            break

    logger.info("live capture demo finished captured_frames=%s", frame_count)


def main() -> None:
    configure_logging("INFO")
    asyncio.run(run_live_capture_demo())


if __name__ == "__main__":
    main()
