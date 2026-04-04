from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from sg_coach.orchestrator.topics import FRAME_RAW
from sg_coach.shared.events import FramePacket, GameEvent
from sg_coach.shared.logging import get_logger
from sg_coach.shared.settings import Settings
from sg_coach.shared.streaming import EVENT_STREAM_COMPLETE, FRAME_STREAM_COMPLETE


logger = get_logger(__name__)


@dataclass(slots=True)
class ReplayFrame:
    frame_id: str
    timestamp: datetime
    image_bgr: np.ndarray
    width: int
    height: int


@dataclass(slots=True)
class ReplayFrameBuffer:
    """Compact rolling frame buffer used for pre-event context.

    Important design choice:
    - we do not keep full-resolution frames for 10 seconds
    - we keep a downscaled, lower-FPS context stream that is good enough for
      later commentary payload selection
    """

    buffer_seconds: int = 10
    sample_fps: int = 4
    max_width: int = 640
    _frames: deque[ReplayFrame] = field(default_factory=deque, init=False, repr=False)
    _last_stored_at: datetime | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_settings(cls, settings: Settings) -> "ReplayFrameBuffer":
        return cls(
            buffer_seconds=settings.replay_buffer_seconds,
            sample_fps=settings.replay_buffer_fps,
            max_width=settings.replay_buffer_max_width,
        )

    def add_frame(self, frame: FramePacket) -> bool:
        """Store a compact copy of a frame if it passes the sample interval."""
        if frame.image_bgr is None:
            return False

        if self._last_stored_at is not None:
            interval = 1.0 / self.sample_fps
            if (frame.timestamp - self._last_stored_at).total_seconds() < interval:
                return False

        image_bgr = self._resize_if_needed(frame.image_bgr)
        replay_frame = ReplayFrame(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            image_bgr=image_bgr.copy(),
            width=image_bgr.shape[1],
            height=image_bgr.shape[0],
        )
        self._frames.append(replay_frame)
        self._last_stored_at = frame.timestamp
        self._prune(frame.timestamp)
        return True

    def recent_frames(self, *, seconds: int | None = None) -> list[ReplayFrame]:
        """Return buffered frames from the tail of the replay window."""
        if not self._frames:
            return []

        if seconds is None:
            return list(self._frames)

        cutoff = self._frames[-1].timestamp - timedelta(seconds=seconds)
        return [frame for frame in self._frames if frame.timestamp >= cutoff]

    def export_recent_frames(
        self,
        *,
        event: GameEvent,
        output_root: Path,
        seconds: int | None = None,
    ) -> list[str]:
        """Save buffered frames for one event and return file paths.

        This intentionally exports every buffered frame in the window. That is
        good enough for the first end-to-end GTA test.

        TODO for you:
        Replace the save-all behavior with smarter key-frame selection so Grok
        eventually sees only a few useful frames instead of the whole buffer.
        """
        frames = self.recent_frames(seconds=seconds)
        if not frames:
            return []

        event_dir = output_root / event.session_id / event.event_id
        event_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: list[str] = []
        for index, replay_frame in enumerate(frames):
            filename = (
                f"{index:03d}_{replay_frame.timestamp.strftime('%H%M%S_%f')}_{replay_frame.frame_id}.jpg"
            )
            path = event_dir / filename
            if cv2.imwrite(str(path), replay_frame.image_bgr):
                saved_paths.append(str(path))

        return saved_paths

    def _resize_if_needed(self, image_bgr: np.ndarray) -> np.ndarray:
        height, width = image_bgr.shape[:2]
        if width <= self.max_width:
            return image_bgr

        scale = self.max_width / width
        resized = cv2.resize(
            image_bgr,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
        return resized

    def _prune(self, current_timestamp: datetime) -> None:
        cutoff = current_timestamp - timedelta(seconds=self.buffer_seconds)
        while self._frames and self._frames[0].timestamp < cutoff:
            self._frames.popleft()


async def replay_buffer_worker(
    frame_queue: asyncio.Queue[FramePacket | str],
    *,
    replay_buffer: ReplayFrameBuffer,
) -> None:
    """Consume raw frames and maintain the rolling replay buffer."""
    while True:
        item = await frame_queue.get()
        if item == FRAME_STREAM_COMPLETE:
            logger.info(
                "replay buffer stream complete buffered_frames=%s topic=%s",
                len(replay_buffer.recent_frames()),
                FRAME_RAW,
            )
            return

        stored = replay_buffer.add_frame(item)
        if stored:
            logger.debug(
                "replay buffer stored frame_id=%s buffered_frames=%s",
                item.frame_id,
                len(replay_buffer.recent_frames()),
            )


async def wasted_context_worker(
    event_queue: asyncio.Queue[GameEvent | str],
    *,
    replay_buffer: ReplayFrameBuffer,
    output_root: Path,
) -> None:
    """Export the replay window when a `wasted` event is observed."""
    while True:
        item = await event_queue.get()
        if item == EVENT_STREAM_COMPLETE:
            logger.info("wasted context stream complete")
            return

        event = item
        if event.event_type != "wasted":
            continue

        saved_paths = replay_buffer.export_recent_frames(
            event=event,
            output_root=output_root,
            seconds=replay_buffer.buffer_seconds,
        )
        logger.info(
            "exported wasted replay context event_id=%s saved_frames=%s output_dir=%s",
            event.event_id,
            len(saved_paths),
            output_root / event.session_id / event.event_id,
        )
