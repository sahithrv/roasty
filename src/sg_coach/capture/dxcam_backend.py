from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import dxcam
import numpy as np

from sg_coach.shared.events import FramePacket
from sg_coach.shared.logging import get_logger
from sg_coach.shared.settings import Settings


logger = get_logger(__name__)


@dataclass(slots=True)
class DxcamFrameSource:
    """Minimal Windows screen capture backend using dxcam.

    This is intentionally narrow:
    - capture a monitor, not a specific game window
    - emit raw `FramePacket`s
    - avoid extra ROI or replay buffering logic for now

    That is enough to verify live capture on the other PC before building more.
    """

    settings: Settings
    game: str | None = None

    def _create_camera(self):
        """Create a dxcam camera configured for BGR frames."""
        return dxcam.create(
            output_idx=self.settings.capture_monitor_id,
            output_color="BGR",
        )

    async def frames(self):
        """Yield live frames from the configured monitor.

        Notes:
        - `dxcam.grab()` is called via `asyncio.to_thread(...)` so we do not block
          the event loop with synchronous capture calls.
        - If dxcam returns `None`, we pause briefly and keep trying.
        """
        camera = self._create_camera()
        frame_interval = 1.0 / self.settings.target_fps

        logger.info(
            "dxcam capture starting monitor_id=%s target_fps=%s",
            self.settings.capture_monitor_id,
            self.settings.target_fps,
        )

        try:
            while True:
                try:
                    frame_array = await asyncio.to_thread(camera.grab)
                except ModuleNotFoundError as exc:
                    if exc.name == "cv2":
                        raise RuntimeError(
                            "dxcam requires OpenCV. Install the 'opencv-python' package in the active environment."
                        ) from exc
                    raise
                if frame_array is None:
                    await asyncio.sleep(frame_interval)
                    continue

                frame_bgr = self._validate_frame(frame_array)
                yield FramePacket(
                    game=self.game,
                    monitor_id=self.settings.capture_monitor_id,
                    width=frame_bgr.shape[1],
                    height=frame_bgr.shape[0],
                    image_bgr=frame_bgr,
                )
                await asyncio.sleep(frame_interval)
        finally:
            stop = getattr(camera, "stop", None)
            if callable(stop):
                stop()
            logger.info("dxcam capture stopped")

    def _validate_frame(self, frame: Any) -> np.ndarray:
        """Validate that dxcam returned an image-like numpy array."""
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"dxcam returned unsupported frame type: {type(frame)!r}")
        if frame.ndim != 3:
            raise ValueError(f"Expected a color image with 3 dimensions, got shape={frame.shape!r}")
        return frame
