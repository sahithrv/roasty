from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from sg_coach.shared.events import DetectionSignal, FramePacket
from sg_coach.shared.logging import get_logger
from sg_coach.shared.settings import Settings


logger = get_logger(__name__)


@dataclass(slots=True)
class GtaWastedDetector:
    """Heuristic GTA detector for the large on-screen `WASTED` banner.

    This is intentionally heuristic-first:
    - crop one likely center-screen ROI
    - template match against a saved `WASTED` template image
    - require a few consecutive hits
    - apply cooldown so one death emits one signal
    """

    settings: Settings
    name: str = "gta_wasted_detector"
    game: str = "gta_like"
    roi_name: str = "center_banner"
    template_path: Path | None = None
    _template_edges: np.ndarray | None = field(default=None, init=False, repr=False)
    _consecutive_hits: int = field(default=0, init=False, repr=False)
    _cooldown_until: datetime | None = field(default=None, init=False, repr=False)
    _waiting_for_banner_clear: bool = field(default=False, init=False, repr=False)
    _clear_frames_seen: int = field(default=0, init=False, repr=False)
    _warned_missing_template: bool = field(default=False, init=False, repr=False)
    _best_score_seen: float = field(default=0.0, init=False, repr=False)
    _debug_saves_written: int = field(default=0, init=False, repr=False)
    _frames_seen: int = field(default=0, init=False, repr=False)
    _logged_roi_once: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.template_path is None:
            self.template_path = self.settings.gta_wasted_template_path
        if self.settings.gta_wasted_debug_enabled:
            debug_dir = self.settings.detector_debug_dir / "gta_wasted"
            debug_dir.mkdir(parents=True, exist_ok=True)

    async def detect(self, frame: FramePacket) -> list[DetectionSignal]:
        """Inspect one GTA frame and emit `wasted` when the banner is confirmed.

        Duplicate suppression is intentionally stateful:
        - a short time cooldown blocks immediate back-to-back emits
        - after a confirmed hit, the detector also stays disarmed until the
          banner score drops below a lower "clear" threshold for a few frames

        This prevents a single on-screen death banner from being treated as
        multiple separate events when it lingers for several seconds.
        """
        if frame.image_bgr is None:
            return []
        self._frames_seen += 1

        if self._cooldown_until is not None and frame.timestamp < self._cooldown_until:
            return []

        template_edges = self._load_template_edges()
        if template_edges is None:
            return []

        roi_bgr, roi_bounds = self._extract_banner_roi(frame.image_bgr)
        self._maybe_log_roi_bounds(frame, roi_bounds, roi_bgr)
        is_new_best = False
        score, match_top_left = self._match_template_score(roi_bgr, template_edges)
        if score > self._best_score_seen:
            is_new_best = True
        self._maybe_log_debug_score(frame, score)
        self._maybe_save_debug_roi(frame, roi_bgr, score, is_new_best=is_new_best)
        self._maybe_save_debug_match_frame(
            frame,
            roi_bounds,
            template_edges,
            score,
            match_top_left,
            is_new_best=is_new_best,
        )

        if self._waiting_for_banner_clear:
            if score <= self.settings.gta_wasted_clear_threshold:
                self._clear_frames_seen += 1
            else:
                self._clear_frames_seen = 0

            if self._clear_frames_seen >= self.settings.gta_wasted_clear_frames:
                self._waiting_for_banner_clear = False
                self._clear_frames_seen = 0
                logger.info(
                    "wasted detector re-armed after clear score=%.4f clear_frames=%s/%s",
                    score,
                    self.settings.gta_wasted_clear_frames,
                    self.settings.gta_wasted_clear_frames,
                )
            return []

        if score >= self.settings.gta_wasted_match_threshold:
            self._consecutive_hits += 1
        else:
            self._consecutive_hits = 0
            return []

        if self._consecutive_hits < self.settings.gta_wasted_confirm_frames:
            return []

        self._consecutive_hits = 0
        self._waiting_for_banner_clear = True
        self._clear_frames_seen = 0
        self._cooldown_until = frame.timestamp + timedelta(
            seconds=self.settings.gta_wasted_cooldown_seconds
        )

        return [
            DetectionSignal(
                game=frame.game or self.game,
                detector_name=self.name,
                signal_type="wasted",
                confidence=score,
                roi_name=self.roi_name,
                tags=["gta", "death", "banner"],
                metadata={
                    "template_score": round(score, 4),
                    "template_path": str(self.template_path),
                    "roi_name": self.roi_name,
                    "confirm_frames": self.settings.gta_wasted_confirm_frames,
                    "clear_threshold": self.settings.gta_wasted_clear_threshold,
                    "clear_frames": self.settings.gta_wasted_clear_frames,
                },
                frame_ref=frame.frame_id,
                dedupe_key=f"wasted:{frame.frame_id}",
                cooldown_key="wasted",
            )
        ]

    def _maybe_log_debug_score(self, frame: FramePacket, score: float) -> None:
        if score > self._best_score_seen:
            self._best_score_seen = score
            logger.info(
                "wasted detector new best score=%.4f frame_id=%s threshold=%.4f confirm_hits=%s/%s",
                score,
                frame.frame_id,
                self.settings.gta_wasted_match_threshold,
                self._consecutive_hits,
                self.settings.gta_wasted_confirm_frames,
            )

        if score >= self.settings.gta_wasted_match_threshold:
            logger.info(
                "wasted detector threshold hit score=%.4f frame_id=%s confirm_hits=%s/%s",
                score,
                frame.frame_id,
                self._consecutive_hits + 1,
                self.settings.gta_wasted_confirm_frames,
            )

    def _maybe_save_debug_roi(
        self,
        frame: FramePacket,
        roi_bgr: np.ndarray,
        score: float,
        *,
        is_new_best: bool,
    ) -> None:
        if not self.settings.gta_wasted_debug_enabled:
            return
        should_save = (
            self._frames_seen <= self.settings.gta_wasted_debug_save_first_n
            or is_new_best
            or score >= self.settings.gta_wasted_debug_score_threshold
        )
        if not should_save:
            return
        if self._debug_saves_written >= self.settings.gta_wasted_debug_max_saves:
            return

        debug_dir = self.settings.detector_debug_dir / "gta_wasted"
        debug_dir.mkdir(parents=True, exist_ok=True)

        score_label = f"{score:.4f}".replace(".", "_")
        prefix = "sample" if self._frames_seen <= self.settings.gta_wasted_debug_save_first_n else "match"
        path = (
            debug_dir
            / f"{prefix}_{frame.timestamp.strftime('%Y%m%d_%H%M%S_%f')}_{score_label}_{frame.frame_id}.jpg"
        )
        if cv2.imwrite(str(path), roi_bgr):
            self._debug_saves_written += 1
            logger.info(
                "wasted detector saved debug roi path=%s score=%.4f saved=%s/%s",
                path,
                score,
                self._debug_saves_written,
                self.settings.gta_wasted_debug_max_saves,
            )

    def _maybe_save_debug_match_frame(
        self,
        frame: FramePacket,
        roi_bounds: tuple[int, int, int, int],
        template_gray: np.ndarray,
        score: float,
        match_top_left: tuple[int, int],
        *,
        is_new_best: bool,
    ) -> None:
        if not self.settings.gta_wasted_debug_enabled:
            return
        should_save = is_new_best or score >= self.settings.gta_wasted_debug_score_threshold
        if not should_save:
            return

        debug_dir = self.settings.detector_debug_dir / "gta_wasted"
        debug_dir.mkdir(parents=True, exist_ok=True)

        left, top, right, bottom = roi_bounds
        match_x, match_y = match_top_left
        template_h, template_w = template_gray.shape[:2]

        annotated = frame.image_bgr.copy()
        cv2.rectangle(annotated, (left, top), (right, bottom), (255, 255, 0), 2)
        cv2.rectangle(
            annotated,
            (left + match_x, top + match_y),
            (left + match_x + template_w, top + match_y + template_h),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            annotated,
            f"score={score:.4f}",
            (left, max(30, top - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        score_label = f"{score:.4f}".replace(".", "_")
        prefix = "best_match" if is_new_best else "match"
        path = (
            debug_dir
            / f"{prefix}_{frame.timestamp.strftime('%Y%m%d_%H%M%S_%f')}_{score_label}_{frame.frame_id}.jpg"
        )
        if cv2.imwrite(str(path), annotated):
            logger.info(
                "wasted detector saved annotated match path=%s score=%.4f roi_bounds=%s match_top_left=%s",
                path,
                score,
                roi_bounds,
                match_top_left,
            )

    def _maybe_log_roi_bounds(
        self,
        frame: FramePacket,
        roi_bounds: tuple[int, int, int, int],
        roi_bgr: np.ndarray,
    ) -> None:
        if self._logged_roi_once:
            return
        left, top, right, bottom = roi_bounds
        logger.info(
            "wasted detector roi frame=%sx%s bounds=(left=%s top=%s right=%s bottom=%s) roi_size=%sx%s template=%s",
            frame.width,
            frame.height,
            left,
            top,
            right,
            bottom,
            roi_bgr.shape[1],
            roi_bgr.shape[0],
            self.template_path,
        )
        self._logged_roi_once = True

    def _load_template_edges(self) -> np.ndarray | None:
        if self._template_edges is not None:
            return self._template_edges

        if self.template_path is None or not self.template_path.exists():
            if not self._warned_missing_template:
                logger.warning(
                    "wasted detector template missing path=%s",
                    self.template_path,
                )
                self._warned_missing_template = True
            return None

        template_bgr = cv2.imread(str(self.template_path))
        if template_bgr is None:
            raise RuntimeError(f"Failed to read GTA wasted template: {self.template_path}")

        template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
        self._template_edges = self._compute_edges(template_gray)
        edge_pixels = int(np.count_nonzero(self._template_edges))
        logger.info(
            "wasted detector loaded template path=%s size=%sx%s edge_pixels=%s",
            self.template_path,
            self._template_edges.shape[1],
            self._template_edges.shape[0],
            edge_pixels,
        )
        return self._template_edges

    def _extract_banner_roi(self, image_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Crop the center region where GTA banners typically appear.

        TODO for you:
        Tune these percentages against real GTA screenshots. They are a starting
        guess only and should probably be tightened after you collect examples.
        """
        height, width = image_bgr.shape[:2]
        top = int(height * 0.38)
        bottom = int(height * 0.60)
        left = int(width * 0.22)
        right = int(width * 0.78)
        return image_bgr[top:bottom, left:right], (left, top, right, bottom)

    def _match_template_score(
        self,
        roi_bgr: np.ndarray,
        template_edges: np.ndarray,
    ) -> tuple[float, tuple[int, int]]:
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        roi_edges = self._compute_edges(roi_gray)
        if (
            roi_edges.shape[0] < template_edges.shape[0]
            or roi_edges.shape[1] < template_edges.shape[1]
        ):
            return 0.0, (0, 0)

        result = cv2.matchTemplate(roi_edges, template_edges, cv2.TM_CCOEFF_NORMED)
        _, max_value, _, max_location = cv2.minMaxLoc(result)
        return float(max_value), max_location

    def _compute_edges(self, gray_image: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        return cv2.Canny(
            blurred,
            threshold1=self.settings.gta_wasted_edge_low_threshold,
            threshold2=self.settings.gta_wasted_edge_high_threshold,
        )
