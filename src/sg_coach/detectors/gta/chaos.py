from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import cv2
import numpy as np

from sg_coach.shared.events import DetectionSignal, FramePacket
from sg_coach.shared.logging import get_logger
from sg_coach.shared.settings import Settings


logger = get_logger(__name__)


@dataclass(slots=True)
class GtaChaosDetector:
    """Cheap ambient detector for visually chaotic GTA moments.

    This is intentionally broad and inexpensive. It does not try to identify
    semantics like "cops" or "strip club"; it only detects scene escalation:
    sudden motion, bright flashes, and busy edge density that often correlate
    with crashes, firefights, explosions, or other messy moments worth noting.
    """

    settings: Settings
    name: str = "gta_chaos_detector"
    game: str = "gta_like"
    roi_name: str = "full_scene"
    _previous_gray: np.ndarray | None = field(default=None, init=False, repr=False)
    _consecutive_hits: int = field(default=0, init=False, repr=False)
    _cooldown_until: datetime | None = field(default=None, init=False, repr=False)
    _startup_ignore_until: datetime | None = field(default=None, init=False, repr=False)
    _best_score_seen: float = field(default=0.0, init=False, repr=False)

    async def detect(self, frame: FramePacket) -> list[DetectionSignal]:
        """Emit a sparse `chaos_spike` signal when the scene looks unusually wild."""
        if frame.image_bgr is None:
            return []

        if self._startup_ignore_until is None:
            self._startup_ignore_until = frame.timestamp + timedelta(
                seconds=self.settings.gta_chaos_startup_delay_seconds
            )
            if self.settings.gta_chaos_startup_delay_seconds > 0:
                logger.info(
                    "chaos detector startup warmup started frame_id=%s delay_seconds=%s active_at=%s",
                    frame.frame_id,
                    self.settings.gta_chaos_startup_delay_seconds,
                    self._startup_ignore_until.isoformat(),
                )

        if self._startup_ignore_until is not None and frame.timestamp < self._startup_ignore_until:
            self._previous_gray = self._prepare_gray(frame.image_bgr)
            return []

        if self._cooldown_until is not None and frame.timestamp < self._cooldown_until:
            self._previous_gray = self._prepare_gray(frame.image_bgr)
            return []

        current_gray = self._prepare_gray(frame.image_bgr)
        if self._previous_gray is None:
            self._previous_gray = current_gray
            return []

        metrics = self._score_scene(current_gray, self._previous_gray)
        self._previous_gray = current_gray

        score = metrics["chaos_score"]
        if score > self._best_score_seen:
            self._best_score_seen = score
            logger.info(
                "chaos detector new best score=%.4f frame_id=%s motion=%.4f flash=%.4f edge=%.4f",
                score,
                frame.frame_id,
                metrics["motion_score"],
                metrics["flash_ratio"],
                metrics["edge_density"],
            )

        if score >= self.settings.gta_chaos_score_threshold:
            self._consecutive_hits += 1
        else:
            self._consecutive_hits = 0
            return []

        if self._consecutive_hits < self.settings.gta_chaos_confirm_frames:
            return []

        self._consecutive_hits = 0
        self._cooldown_until = frame.timestamp + timedelta(
            seconds=self.settings.gta_chaos_cooldown_seconds
        )

        return [
            DetectionSignal(
                game=frame.game or self.game,
                detector_name=self.name,
                signal_type="chaos_spike",
                confidence=min(score, 1.0),
                roi_name=self.roi_name,
                tags=["gta", "chaos", "ambient"],
                metadata={
                    "chaos_score": round(score, 4),
                    "motion_score": round(metrics["motion_score"], 4),
                    "flash_ratio": round(metrics["flash_ratio"], 4),
                    "edge_density": round(metrics["edge_density"], 4),
                    "bright_ratio": round(metrics["bright_ratio"], 4),
                    "edge_change_ratio": round(metrics["edge_change_ratio"], 4),
                    "qualifying_signal_count": int(metrics["qualifying_signal_count"]),
                    "confirm_frames": self.settings.gta_chaos_confirm_frames,
                    "cooldown_seconds": self.settings.gta_chaos_cooldown_seconds,
                    "startup_delay_seconds": self.settings.gta_chaos_startup_delay_seconds,
                },
                frame_ref=frame.frame_id,
                dedupe_key=f"chaos:{int(frame.timestamp.timestamp()) // self.settings.gta_chaos_cooldown_seconds}",
                cooldown_key="chaos_spike",
            )
        ]

    def _prepare_gray(self, image_bgr: np.ndarray) -> np.ndarray:
        """Downscale and grayscale the frame so the detector stays cheap."""
        height, width = image_bgr.shape[:2]
        target_width = min(self.settings.gta_chaos_downscale_width, width)
        scale = target_width / width
        target_height = max(1, int(height * scale))
        resized = cv2.resize(image_bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    def _score_scene(
        self,
        current_gray: np.ndarray,
        previous_gray: np.ndarray,
    ) -> dict[str, float]:
        """Combine scene-change signals into one coarse chaos score."""
        diff = cv2.absdiff(current_gray, previous_gray)
        motion_score = float(diff.mean() / 255.0)

        bright_pixels = current_gray >= 225
        previous_bright_pixels = previous_gray >= 225
        bright_ratio = float(np.count_nonzero(bright_pixels) / bright_pixels.size)
        previous_bright_ratio = float(
            np.count_nonzero(previous_bright_pixels) / previous_bright_pixels.size
        )
        flash_ratio = max(0.0, bright_ratio - previous_bright_ratio)

        edges = cv2.Canny(current_gray, 80, 180)
        previous_edges = cv2.Canny(previous_gray, 80, 180)
        edge_density = float(np.count_nonzero(edges) / edges.size)
        edge_change_ratio = float(np.count_nonzero(cv2.absdiff(edges, previous_edges)) / edges.size)

        # Weight motion most heavily; sudden flashes and edge changes help
        # distinguish actual scene escalation from GTA's naturally noisy visuals.
        chaos_score = (
            motion_score * 0.75
            + flash_ratio * 1.60
            + edge_change_ratio * 0.35
        )

        qualifying_signal_count = 0
        if motion_score >= self.settings.gta_chaos_motion_threshold:
            qualifying_signal_count += 1
        if flash_ratio >= self.settings.gta_chaos_flash_threshold:
            qualifying_signal_count += 1
        if edge_change_ratio >= self.settings.gta_chaos_edge_threshold:
            qualifying_signal_count += 1

        # Require either multiple independent change signals, or a very large
        # motion jump, so bright skies / city lights / UI contrast do not read
        # as chaos by themselves.
        if (
            qualifying_signal_count < 2
            and motion_score < (self.settings.gta_chaos_motion_threshold * 1.8)
        ):
            chaos_score = 0.0

        return {
            "chaos_score": chaos_score,
            "motion_score": motion_score,
            "flash_ratio": flash_ratio,
            "edge_density": edge_density,
            "bright_ratio": bright_ratio,
            "edge_change_ratio": edge_change_ratio,
            "qualifying_signal_count": float(qualifying_signal_count),
        }
