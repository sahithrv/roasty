from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from sg_coach.shared.events import DetectionSignal, FramePacket
from sg_coach.shared.logging import get_logger


if TYPE_CHECKING:
    from sg_coach.shared.settings import Settings


logger = get_logger(__name__)


@dataclass(slots=True)
class _MapTemplate:
    map_key: str
    display_name: str
    path: Path
    edges: np.ndarray


@dataclass(slots=True)
class ValorantMapObservation:
    best_map_name: str | None
    best_score: float
    confirmed_map_name: str | None
    candidate_map_name: str | None
    candidate_frames: int
    roi_bounds: tuple[int, int, int, int]
    best_match_bounds: tuple[int, int, int, int] | None
    template_scores: dict[str, float]


@dataclass(slots=True)
class ValorantMapLoadingDetector:
    """Template-based detector for the Valorant loading-screen map name."""

    settings: Settings
    name: str = "valorant_map_loading_detector"
    game: str = "valorant"
    roi_name: str = "loading_screen_map"
    latest_observation: ValorantMapObservation | None = field(default=None, init=False)
    _templates: list[_MapTemplate] | None = field(default=None, init=False, repr=False)
    _warned_missing_templates: bool = field(default=False, init=False, repr=False)
    _candidate_map_name: str | None = field(default=None, init=False, repr=False)
    _candidate_frames: int = field(default=0, init=False, repr=False)
    _confirmed_map_name: str | None = field(default=None, init=False, repr=False)
    _clear_frames_seen: int = field(default=0, init=False, repr=False)
    _logged_roi_once: bool = field(default=False, init=False, repr=False)

    async def detect(self, frame: FramePacket) -> list[DetectionSignal]:
        if frame.image_bgr is None:
            return []

        templates = self._load_templates()
        if not templates:
            return []

        roi_bgr, roi_bounds = self._extract_map_roi(frame.image_bgr)
        self._maybe_log_roi(frame, roi_bounds, roi_bgr)
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        roi_edges = self._compute_edges(roi_gray)

        best_template: _MapTemplate | None = None
        best_score = 0.0
        best_location = (0, 0)
        template_scores: dict[str, float] = {}

        for template in templates:
            score, location = self._match_template_score(roi_edges, template.edges)
            template_scores[template.display_name] = round(score, 4)
            if score > best_score:
                best_score = score
                best_template = template
                best_location = location

        best_map_name = None if best_template is None else best_template.display_name
        best_match_bounds = None
        if best_template is not None:
            match_left = roi_bounds[0] + best_location[0]
            match_top = roi_bounds[1] + best_location[1]
            best_match_bounds = (
                match_left,
                match_top,
                match_left + best_template.edges.shape[1],
                match_top + best_template.edges.shape[0],
            )

        self.latest_observation = ValorantMapObservation(
            best_map_name=best_map_name,
            best_score=best_score,
            confirmed_map_name=self._confirmed_map_name,
            candidate_map_name=self._candidate_map_name,
            candidate_frames=self._candidate_frames,
            roi_bounds=roi_bounds,
            best_match_bounds=best_match_bounds,
            template_scores=template_scores,
        )

        if best_template is None:
            self._candidate_map_name = None
            self._candidate_frames = 0
            return []

        if self._confirmed_map_name == best_template.display_name:
            if best_score <= self.settings.valorant_map_clear_threshold:
                self._clear_frames_seen += 1
                if self._clear_frames_seen >= self.settings.valorant_map_clear_frames:
                    logger.info(
                        "valorant map detector re-armed after clear map=%s score=%.4f",
                        self._confirmed_map_name,
                        best_score,
                    )
                    self._confirmed_map_name = None
                    self._clear_frames_seen = 0
            else:
                self._clear_frames_seen = 0
            return []

        if best_score < self.settings.valorant_map_match_threshold:
            self._candidate_map_name = None
            self._candidate_frames = 0
            return []

        if self._candidate_map_name == best_template.display_name:
            self._candidate_frames += 1
        else:
            self._candidate_map_name = best_template.display_name
            self._candidate_frames = 1

        self.latest_observation = ValorantMapObservation(
            best_map_name=best_map_name,
            best_score=best_score,
            confirmed_map_name=self._confirmed_map_name,
            candidate_map_name=self._candidate_map_name,
            candidate_frames=self._candidate_frames,
            roi_bounds=roi_bounds,
            best_match_bounds=best_match_bounds,
            template_scores=template_scores,
        )

        if self._candidate_frames < self.settings.valorant_map_confirm_frames:
            return []

        self._confirmed_map_name = best_template.display_name
        self._candidate_map_name = None
        self._candidate_frames = 0
        self._clear_frames_seen = 0

        self.latest_observation = ValorantMapObservation(
            best_map_name=best_map_name,
            best_score=best_score,
            confirmed_map_name=self._confirmed_map_name,
            candidate_map_name=self._candidate_map_name,
            candidate_frames=self._candidate_frames,
            roi_bounds=roi_bounds,
            best_match_bounds=best_match_bounds,
            template_scores=template_scores,
        )

        return [
            DetectionSignal(
                game=frame.game or self.game,
                detector_name=self.name,
                signal_type="map_identified",
                confidence=best_score,
                roi_name=self.roi_name,
                tags=["valorant", "map", "loading_screen"],
                metadata={
                    "map_name": best_template.display_name,
                    "map_key": best_template.map_key,
                    "template_path": str(best_template.path),
                    "template_score": round(best_score, 4),
                    "template_scores": template_scores,
                    "roi_bounds": roi_bounds,
                    "best_match_bounds": best_match_bounds,
                    "confirm_frames": self.settings.valorant_map_confirm_frames,
                },
                frame_ref=frame.frame_id,
                dedupe_key=f"valorant_map:{best_template.map_key}",
                cooldown_key="valorant_map",
            )
        ]

    def _load_templates(self) -> list[_MapTemplate]:
        if self._templates is not None:
            return self._templates

        templates_dir = self.settings.valorant_map_templates_dir
        if not templates_dir.exists():
            if not self._warned_missing_templates:
                logger.warning("valorant map detector templates missing dir=%s", templates_dir)
                self._warned_missing_templates = True
            self._templates = []
            return self._templates

        templates: list[_MapTemplate] = []
        for path in sorted(templates_dir.iterdir()):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            image = cv2.imread(str(path))
            if image is None:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = self._compute_edges(gray)
            templates.append(
                _MapTemplate(
                    map_key=path.stem.lower(),
                    display_name=self._display_name_from_stem(path.stem),
                    path=path,
                    edges=edges,
                )
            )

        if not templates and not self._warned_missing_templates:
            logger.warning("valorant map detector found no templates in dir=%s", templates_dir)
            self._warned_missing_templates = True

        logger.info("valorant map detector loaded templates count=%s dir=%s", len(templates), templates_dir)
        self._templates = templates
        return self._templates

    def _extract_map_roi(self, image_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        height, width = image_bgr.shape[:2]
        left = int(width * self.settings.valorant_map_roi_left_pct)
        right = int(width * self.settings.valorant_map_roi_right_pct)
        top = int(height * self.settings.valorant_map_roi_top_pct)
        bottom = int(height * self.settings.valorant_map_roi_bottom_pct)

        left = max(0, min(left, width - 2))
        right = max(left + 1, min(right, width))
        top = max(0, min(top, height - 2))
        bottom = max(top + 1, min(bottom, height))

        return image_bgr[top:bottom, left:right], (left, top, right, bottom)

    def _maybe_log_roi(
        self,
        frame: FramePacket,
        roi_bounds: tuple[int, int, int, int],
        roi_bgr: np.ndarray,
    ) -> None:
        if self._logged_roi_once:
            return
        logger.info(
            "valorant map detector roi frame=%sx%s bounds=%s roi_size=%sx%s templates_dir=%s",
            frame.width,
            frame.height,
            roi_bounds,
            roi_bgr.shape[1],
            roi_bgr.shape[0],
            self.settings.valorant_map_templates_dir,
        )
        self._logged_roi_once = True

    def _compute_edges(self, gray: np.ndarray) -> np.ndarray:
        return cv2.Canny(gray, 60, 180)

    def _match_template_score(
        self,
        roi_edges: np.ndarray,
        template_edges: np.ndarray,
    ) -> tuple[float, tuple[int, int]]:
        roi_height, roi_width = roi_edges.shape[:2]
        template_height, template_width = template_edges.shape[:2]
        if template_height > roi_height or template_width > roi_width:
            return 0.0, (0, 0)
        result = cv2.matchTemplate(roi_edges, template_edges, cv2.TM_CCOEFF_NORMED)
        _, max_value, _, max_location = cv2.minMaxLoc(result)
        return float(max_value), (int(max_location[0]), int(max_location[1]))

    def _display_name_from_stem(self, stem: str) -> str:
        return stem.replace("_", " ").strip().title()
