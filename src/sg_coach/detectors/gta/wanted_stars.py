from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import cv2
import numpy as np

from sg_coach.shared.events import DetectionSignal, FramePacket
from sg_coach.shared.logging import get_logger
from sg_coach.shared.settings import Settings


logger = get_logger(__name__)

_CANONICAL_SLOT_SIZE = 40


@dataclass(slots=True)
class WantedStarSlotClassification:
    """Classification result for one wanted-star HUD slot."""

    label: str
    active_score: float
    present_score: float
    shape_score: float
    fill_score: float


@dataclass(slots=True)
class WantedStarsObservation:
    """Debug snapshot of what the wanted-stars detector saw on one frame."""

    roi_bounds: tuple[int, int, int, int]
    slot_bounds: list[tuple[int, int, int, int]]
    slot_scores: list[float]
    present_slot_scores: list[float]
    shape_scores: list[float]
    fill_scores: list[float]
    slot_labels: list[str]
    raw_active_star_slots: list[int]
    observed_star_count: int
    active_star_slots: list[int]
    present_star_slots: list[int]
    all_slots_visible: bool
    startup_blocked: bool
    flashing_hold: bool
    stable_star_count: int | None
    candidate_star_count: int | None
    candidate_frames: int


@dataclass(slots=True)
class GtaWantedStarsDetector:
    """GTA wanted-level detector built around per-slot star-shape classification."""

    settings: Settings
    name: str = "gta_wanted_stars_detector"
    game: str = "gta_like"
    roi_name: str = "top_right_wanted_hud"
    _startup_ignore_until: datetime | None = field(default=None, init=False, repr=False)
    _stable_star_count: int | None = field(default=None, init=False, repr=False)
    _stable_active_star_slots: list[int] = field(default_factory=list, init=False, repr=False)
    _candidate_star_count: int | None = field(default=None, init=False, repr=False)
    _candidate_frames: int = field(default=0, init=False, repr=False)
    _logged_roi_once: bool = field(default=False, init=False, repr=False)
    _latest_observation: WantedStarsObservation | None = field(default=None, init=False, repr=False)
    _slot_template_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    async def detect(self, frame: FramePacket) -> list[DetectionSignal]:
        """Inspect one GTA frame and emit stable wanted-level change signals."""
        if frame.image_bgr is None:
            return []

        if self._startup_ignore_until is None:
            self._startup_ignore_until = frame.timestamp + timedelta(
                seconds=self.settings.gta_wanted_startup_delay_seconds
            )
            if self.settings.gta_wanted_startup_delay_seconds > 0:
                logger.info(
                    "wanted detector startup warmup started frame_id=%s delay_seconds=%s active_at=%s",
                    frame.frame_id,
                    self.settings.gta_wanted_startup_delay_seconds,
                    self._startup_ignore_until.isoformat(),
                )

        roi_bgr, roi_bounds = self._extract_wanted_roi(frame.image_bgr)
        self._maybe_log_roi_bounds(frame, roi_bounds, roi_bgr)
        slot_classifications, slot_bounds = self._classify_star_slots(roi_bgr, roi_bounds)
        slot_scores = [item.active_score for item in slot_classifications]
        present_slot_scores = [item.present_score for item in slot_classifications]
        shape_scores = [item.shape_score for item in slot_classifications]
        fill_scores = [item.fill_score for item in slot_classifications]
        slot_labels = [item.label for item in slot_classifications]
        raw_active_star_slots = [
            index + 1 for index, item in enumerate(slot_classifications) if item.label == "active"
        ]
        present_star_slots = [
            index + 1
            for index, item in enumerate(slot_classifications)
            if item.label in {"active", "outline"}
        ]
        all_slots_visible = len(present_star_slots) == 5
        active_star_slots = self._normalize_active_star_slots(
            raw_active_star_slots,
            all_slots_visible=all_slots_visible,
        )
        observed_star_count = len(active_star_slots)
        startup_blocked = bool(
            self._startup_ignore_until is not None and frame.timestamp < self._startup_ignore_until
        )

        if startup_blocked:
            self._store_observation(
                roi_bounds=roi_bounds,
                slot_bounds=slot_bounds,
                slot_scores=slot_scores,
                present_slot_scores=present_slot_scores,
                shape_scores=shape_scores,
                fill_scores=fill_scores,
                slot_labels=slot_labels,
                raw_active_star_slots=raw_active_star_slots,
                observed_star_count=observed_star_count,
                active_star_slots=active_star_slots,
                present_star_slots=present_star_slots,
                all_slots_visible=all_slots_visible,
                startup_blocked=True,
                flashing_hold=False,
            )
            return []

        if self._stable_star_count is None:
            self._stable_star_count = observed_star_count
            self._stable_active_star_slots = list(active_star_slots)
            self._store_observation(
                roi_bounds=roi_bounds,
                slot_bounds=slot_bounds,
                slot_scores=slot_scores,
                present_slot_scores=present_slot_scores,
                shape_scores=shape_scores,
                fill_scores=fill_scores,
                slot_labels=slot_labels,
                raw_active_star_slots=raw_active_star_slots,
                observed_star_count=observed_star_count,
                active_star_slots=active_star_slots,
                present_star_slots=present_star_slots,
                all_slots_visible=all_slots_visible,
                startup_blocked=False,
                flashing_hold=False,
            )
            logger.info(
                "wanted detector initialized stable_star_count=%s frame_id=%s slot_labels=%s shape_scores=%s active_scores=%s",
                observed_star_count,
                frame.frame_id,
                slot_labels,
                [round(score, 4) for score in shape_scores],
                [round(score, 4) for score in slot_scores],
            )
            return []

        if self._looks_like_flashing_hold(
            observed_star_count=observed_star_count,
            present_star_slots=present_star_slots,
        ):
            self._candidate_star_count = None
            self._candidate_frames = 0
            self._store_observation(
                roi_bounds=roi_bounds,
                slot_bounds=slot_bounds,
                slot_scores=slot_scores,
                present_slot_scores=present_slot_scores,
                shape_scores=shape_scores,
                fill_scores=fill_scores,
                slot_labels=slot_labels,
                raw_active_star_slots=raw_active_star_slots,
                observed_star_count=self._stable_star_count,
                active_star_slots=list(self._stable_active_star_slots),
                present_star_slots=present_star_slots,
                all_slots_visible=all_slots_visible,
                startup_blocked=False,
                flashing_hold=True,
            )
            return []

        if observed_star_count == self._stable_star_count:
            self._candidate_star_count = None
            self._candidate_frames = 0
            self._stable_active_star_slots = list(active_star_slots)
            self._store_observation(
                roi_bounds=roi_bounds,
                slot_bounds=slot_bounds,
                slot_scores=slot_scores,
                present_slot_scores=present_slot_scores,
                shape_scores=shape_scores,
                fill_scores=fill_scores,
                slot_labels=slot_labels,
                raw_active_star_slots=raw_active_star_slots,
                observed_star_count=observed_star_count,
                active_star_slots=active_star_slots,
                present_star_slots=present_star_slots,
                all_slots_visible=all_slots_visible,
                startup_blocked=False,
                flashing_hold=False,
            )
            return []

        if observed_star_count != self._candidate_star_count:
            self._candidate_star_count = observed_star_count
            self._candidate_frames = 1
            self._store_observation(
                roi_bounds=roi_bounds,
                slot_bounds=slot_bounds,
                slot_scores=slot_scores,
                present_slot_scores=present_slot_scores,
                shape_scores=shape_scores,
                fill_scores=fill_scores,
                slot_labels=slot_labels,
                raw_active_star_slots=raw_active_star_slots,
                observed_star_count=observed_star_count,
                active_star_slots=active_star_slots,
                present_star_slots=present_star_slots,
                all_slots_visible=all_slots_visible,
                startup_blocked=False,
                flashing_hold=False,
            )
            return []

        self._candidate_frames += 1
        if self._candidate_frames < self.settings.gta_wanted_confirm_frames:
            self._store_observation(
                roi_bounds=roi_bounds,
                slot_bounds=slot_bounds,
                slot_scores=slot_scores,
                present_slot_scores=present_slot_scores,
                shape_scores=shape_scores,
                fill_scores=fill_scores,
                slot_labels=slot_labels,
                raw_active_star_slots=raw_active_star_slots,
                observed_star_count=observed_star_count,
                active_star_slots=active_star_slots,
                present_star_slots=present_star_slots,
                all_slots_visible=all_slots_visible,
                startup_blocked=False,
                flashing_hold=False,
            )
            return []

        previous_star_count = self._stable_star_count
        current_star_count = observed_star_count
        confirm_frames = self._candidate_frames
        self._stable_star_count = current_star_count
        self._candidate_star_count = None
        self._candidate_frames = 0

        signal_type, change_direction = self._classify_change(
            previous_star_count=previous_star_count,
            current_star_count=current_star_count,
        )
        confidence = self._score_confidence(
            slot_scores,
            present_slot_scores,
            current_star_count,
            all_slots_visible=all_slots_visible,
        )
        self._stable_active_star_slots = list(active_star_slots)
        self._store_observation(
            roi_bounds=roi_bounds,
            slot_bounds=slot_bounds,
            slot_scores=slot_scores,
            present_slot_scores=present_slot_scores,
            shape_scores=shape_scores,
            fill_scores=fill_scores,
            slot_labels=slot_labels,
            raw_active_star_slots=raw_active_star_slots,
            observed_star_count=observed_star_count,
            active_star_slots=active_star_slots,
            present_star_slots=present_star_slots,
            all_slots_visible=all_slots_visible,
            startup_blocked=False,
            flashing_hold=False,
        )
        logger.info(
            "wanted detector emitted signal_type=%s previous=%s current=%s confidence=%.3f slot_labels=%s shape_scores=%s active_scores=%s",
            signal_type,
            previous_star_count,
            current_star_count,
            confidence,
            slot_labels,
            [round(score, 4) for score in shape_scores],
            [round(score, 4) for score in slot_scores],
        )

        return [
            DetectionSignal(
                game=frame.game or self.game,
                detector_name=self.name,
                signal_type=signal_type,
                confidence=confidence,
                roi_name=self.roi_name,
                tags=["gta", "wanted", "police", "hud"],
                metadata={
                    "roi_name": self.roi_name,
                    "roi_bounds": list(roi_bounds),
                    "slot_bounds": [list(bounds) for bounds in slot_bounds],
                    "previous_wanted_level": previous_star_count,
                    "wanted_level": current_star_count,
                    "change_direction": change_direction,
                    "confirm_frames": confirm_frames,
                    "shape_presence_threshold": self.settings.gta_wanted_shape_presence_threshold,
                    "active_score_threshold": self.settings.gta_wanted_active_score_threshold,
                    "slot_scores": [round(score, 4) for score in slot_scores],
                    "present_slot_scores": [round(score, 4) for score in present_slot_scores],
                    "shape_scores": [round(score, 4) for score in shape_scores],
                    "fill_scores": [round(score, 4) for score in fill_scores],
                    "slot_labels": slot_labels,
                    "raw_active_star_slots": raw_active_star_slots,
                    "active_star_slots": active_star_slots,
                    "present_star_slots": present_star_slots,
                    "all_slots_visible": all_slots_visible,
                    "slot_centers_pct": self._slot_center_percentages(),
                    "slot_half_width_pct": round(self.settings.gta_wanted_slot_half_width_pct, 4),
                    "slot_top_pct": round(self.settings.gta_wanted_slot_top_pct, 4),
                    "slot_bottom_pct": round(self.settings.gta_wanted_slot_bottom_pct, 4),
                },
                frame_ref=frame.frame_id,
                dedupe_key=f"wanted:{signal_type}:{previous_star_count}:{current_star_count}:{frame.frame_id}",
                cooldown_key=None,
            )
        ]

    @property
    def latest_observation(self) -> WantedStarsObservation | None:
        """Return the detector's latest frame-level observation for debugging."""
        return self._latest_observation

    def _extract_wanted_roi(self, image_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Crop the top-right HUD region where GTA wanted stars usually appear."""
        height, width = image_bgr.shape[:2]
        top = int(round(height * self.settings.gta_wanted_roi_top_pct))
        bottom = int(round(height * self.settings.gta_wanted_roi_bottom_pct))
        left = int(round(width * self.settings.gta_wanted_roi_left_pct))
        right = int(round(width * self.settings.gta_wanted_roi_right_pct))
        top = max(0, min(top, height - 2))
        bottom = max(top + 1, min(bottom, height))
        left = max(0, min(left, width - 2))
        right = max(left + 1, min(right, width))
        return image_bgr[top:bottom, left:right], (left, top, right, bottom)

    def _classify_star_slots(
        self,
        roi_bgr: np.ndarray,
        roi_bounds: tuple[int, int, int, int],
    ) -> tuple[list[WantedStarSlotClassification], list[tuple[int, int, int, int]]]:
        """Classify each fixed slot in the wanted HUD as active, present, or absent."""
        roi_left, roi_top, _, _ = roi_bounds
        slot_classifications: list[WantedStarSlotClassification] = []
        slot_bounds_absolute: list[tuple[int, int, int, int]] = []

        for left, top, right, bottom in self._iter_star_slot_bounds(
            width=roi_bgr.shape[1],
            height=roi_bgr.shape[0],
        ):
            slot_bgr = roi_bgr[top:bottom, left:right]
            slot_classifications.append(self._classify_star_slot(slot_bgr))
            slot_bounds_absolute.append((roi_left + left, roi_top + top, roi_left + right, roi_top + bottom))

        return slot_classifications, slot_bounds_absolute

    def _classify_star_slot(self, slot_bgr: np.ndarray) -> WantedStarSlotClassification:
        """Score one slot using star-shape overlap plus white-fill strength."""
        slot_view = self._resize_slot(slot_bgr)
        gray = cv2.cvtColor(slot_view, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        hls = cv2.cvtColor(slot_view, cv2.COLOR_BGR2HLS)
        _, fill_mask, template_edges = self._get_slot_templates(
            width=slot_view.shape[1],
            height=slot_view.shape[0],
        )
        slot_edges = cv2.Canny(gray, threshold1=45, threshold2=140)
        slot_edges = cv2.dilate(slot_edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
        shape_score = self._f1_overlap_score(slot_edges, template_edges)
        fill_score = self._score_star_fill(hls, fill_mask)
        present_score = shape_score
        active_score = fill_score

        if (
            present_score >= self.settings.gta_wanted_shape_presence_threshold
            and active_score >= self.settings.gta_wanted_active_score_threshold
        ):
            label = "active"
        elif present_score >= self.settings.gta_wanted_shape_presence_threshold:
            label = "outline"
        else:
            label = "missing"

        return WantedStarSlotClassification(
            label=label,
            active_score=active_score,
            present_score=present_score,
            shape_score=shape_score,
            fill_score=fill_score,
        )

    def _resize_slot(self, slot_bgr: np.ndarray) -> np.ndarray:
        interpolation = cv2.INTER_AREA if slot_bgr.shape[0] >= _CANONICAL_SLOT_SIZE else cv2.INTER_LINEAR
        return cv2.resize(slot_bgr, (_CANONICAL_SLOT_SIZE, _CANONICAL_SLOT_SIZE), interpolation=interpolation)

    def _get_slot_templates(self, *, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return cached synthetic fill and outline templates for one slot size."""
        key = (width, height)
        cached = self._slot_template_cache.get(key)
        if cached is not None:
            return cached

        star_mask = self._build_star_shape_mask(width=width, height=height)
        erosion_size = max(3, int(round(min(width, height) * 0.18)))
        if erosion_size % 2 == 0:
            erosion_size += 1
        fill_mask = cv2.erode(
            star_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size)),
            iterations=1,
        )
        outline_mask = np.zeros((height, width), dtype=np.uint8)
        thickness = max(2, int(round(min(width, height) * 0.10)))
        cv2.polylines(
            outline_mask,
            [self._build_star_polygon(width=width, height=height)],
            isClosed=True,
            color=255,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
        outline_mask = cv2.dilate(outline_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
        self._slot_template_cache[key] = (star_mask, fill_mask, outline_mask)
        return self._slot_template_cache[key]

    def _score_star_fill(self, hls: np.ndarray, fill_mask: np.ndarray) -> float:
        """Return how much the star interior looks like a bright active wanted star."""
        fill_pixels = fill_mask > 0
        if not np.any(fill_pixels):
            return 0.0

        lightness = hls[:, :, 1].astype(np.float32) / 255.0
        saturation = hls[:, :, 2].astype(np.float32) / 255.0
        inside_lightness = float(np.mean(lightness[fill_pixels]))
        white_fill_ratio = float(
            np.mean(
                (lightness[fill_pixels] >= 0.70)
                & (saturation[fill_pixels] <= 0.42)
            )
        )
        return float(min(1.0, (inside_lightness * 0.35) + (white_fill_ratio * 0.65)))

    def _f1_overlap_score(self, observed_mask: np.ndarray, template_mask: np.ndarray) -> float:
        """Return an overlap score that rewards star-shaped edges and punishes clutter."""
        overlap = int(np.count_nonzero(cv2.bitwise_and(observed_mask, template_mask)))
        if overlap == 0:
            return 0.0

        observed_pixels = max(1, int(np.count_nonzero(observed_mask)))
        template_pixels = max(1, int(np.count_nonzero(template_mask)))
        precision = overlap / observed_pixels
        recall = overlap / template_pixels
        return float((2.0 * precision * recall) / max(1e-6, precision + recall))

    def _build_star_shape_mask(self, *, width: int, height: int) -> np.ndarray:
        """Return a filled star-shaped mask for one wanted slot."""
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [self._build_star_polygon(width=width, height=height)], color=255)
        return mask

    def _build_star_polygon(self, *, width: int, height: int) -> np.ndarray:
        """Return the polygon points for a centered five-point star."""
        center_x = width / 2
        center_y = height / 2
        outer_radius = min(width, height) * 0.44
        inner_radius = outer_radius * 0.46
        points: list[list[int]] = []

        for index in range(10):
            angle = np.deg2rad(-90 + index * 36)
            radius = outer_radius if index % 2 == 0 else inner_radius
            x = int(round(center_x + np.cos(angle) * radius))
            y = int(round(center_y + np.sin(angle) * radius))
            points.append([x, y])

        return np.array(points, dtype=np.int32)

    def _iter_star_slot_bounds(self, *, width: int, height: int) -> list[tuple[int, int, int, int]]:
        """Return manually tunable slot bounds inside the wanted HUD ROI."""
        usable_top = int(round(height * self.settings.gta_wanted_slot_top_pct))
        usable_bottom = int(round(height * self.settings.gta_wanted_slot_bottom_pct))
        usable_top = max(0, min(usable_top, height - 2))
        usable_bottom = max(usable_top + 1, min(usable_bottom, height))
        slot_half_width = max(1, int(round(width * self.settings.gta_wanted_slot_half_width_pct)))
        bounds: list[tuple[int, int, int, int]] = []
        for center_pct in self._slot_center_percentages():
            center_x = int(round(width * center_pct))
            left = max(0, center_x - slot_half_width)
            right = min(width, center_x + slot_half_width)
            if right <= left:
                right = min(width, left + 1)
            bounds.append((left, usable_top, right, usable_bottom))
        return bounds

    def _score_confidence(
        self,
        slot_scores: list[float],
        present_slot_scores: list[float],
        wanted_level: int,
        *,
        all_slots_visible: bool,
    ) -> float:
        if wanted_level > 0:
            active_scores = sorted(slot_scores, reverse=True)[:wanted_level]
            if not active_scores:
                return 0.0
            return float(min(1.0, max(0.0, sum(active_scores) / len(active_scores))))

        strongest_active = max(slot_scores, default=0.0)
        base_confidence = float(max(0.0, min(1.0, 1.0 - strongest_active)))
        if all_slots_visible:
            return base_confidence
        return float(min(1.0, base_confidence + 0.1))

    def _classify_change(
        self,
        *,
        previous_star_count: int,
        current_star_count: int,
    ) -> tuple[str, str]:
        if previous_star_count == 0 and current_star_count > 0:
            return "wanted_level_started", "increase"
        if previous_star_count > 0 and current_star_count == 0:
            return "wanted_level_cleared", "decrease"
        if current_star_count > previous_star_count:
            return "wanted_level_changed", "increase"
        return "wanted_level_changed", "decrease"

    def _looks_like_flashing_hold(
        self,
        *,
        observed_star_count: int,
        present_star_slots: list[int],
    ) -> bool:
        if self._stable_star_count is None or self._stable_star_count <= 0:
            return False
        if observed_star_count != 0:
            return False
        if not self._stable_active_star_slots:
            return False
        if len(present_star_slots) < 5:
            return False

        stable_slots = set(self._stable_active_star_slots)
        return stable_slots.issubset(set(present_star_slots))

    def _normalize_active_star_slots(
        self,
        raw_active_star_slots: list[int],
        *,
        all_slots_visible: bool,
    ) -> list[int]:
        """Keep only a valid right-aligned contiguous active run when all slots are visible."""
        if not all_slots_visible:
            return []

        active_set = set(raw_active_star_slots)
        normalized: list[int] = []
        for slot_index in range(5, 0, -1):
            if slot_index in active_set:
                normalized.append(slot_index)
                continue
            if normalized:
                break

        normalized.reverse()
        return normalized

    def _slot_center_percentages(self) -> list[float]:
        """Parse the manually tunable wanted-slot center positions."""
        default_centers = [0.125, 0.308, 0.491, 0.674, 0.864]
        raw_centers = [
            part.strip()
            for part in self.settings.gta_wanted_slot_centers_pct.split(",")
            if part.strip()
        ]
        if len(raw_centers) != 5:
            return default_centers

        try:
            centers = [float(part) for part in raw_centers]
        except ValueError:
            return default_centers

        if any(center <= 0.0 or center >= 1.0 for center in centers):
            return default_centers

        return centers

    def _store_observation(
        self,
        *,
        roi_bounds: tuple[int, int, int, int],
        slot_bounds: list[tuple[int, int, int, int]],
        slot_scores: list[float],
        present_slot_scores: list[float],
        shape_scores: list[float],
        fill_scores: list[float],
        slot_labels: list[str],
        raw_active_star_slots: list[int],
        observed_star_count: int,
        active_star_slots: list[int],
        present_star_slots: list[int],
        all_slots_visible: bool,
        startup_blocked: bool,
        flashing_hold: bool,
    ) -> None:
        self._latest_observation = WantedStarsObservation(
            roi_bounds=roi_bounds,
            slot_bounds=list(slot_bounds),
            slot_scores=list(slot_scores),
            present_slot_scores=list(present_slot_scores),
            shape_scores=list(shape_scores),
            fill_scores=list(fill_scores),
            slot_labels=list(slot_labels),
            raw_active_star_slots=list(raw_active_star_slots),
            observed_star_count=observed_star_count,
            active_star_slots=list(active_star_slots),
            present_star_slots=list(present_star_slots),
            all_slots_visible=all_slots_visible,
            startup_blocked=startup_blocked,
            flashing_hold=flashing_hold,
            stable_star_count=self._stable_star_count,
            candidate_star_count=self._candidate_star_count,
            candidate_frames=self._candidate_frames,
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
            "wanted detector roi frame=%sx%s bounds=(left=%s top=%s right=%s bottom=%s) roi_size=%sx%s",
            frame.width,
            frame.height,
            left,
            top,
            right,
            bottom,
            roi_bgr.shape[1],
            roi_bgr.shape[0],
        )
        self._logged_roi_once = True
