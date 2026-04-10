from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from sg_coach.detectors.valorant.map_loading import ValorantMapLoadingDetector
from sg_coach.shared.events import FramePacket


@dataclass(slots=True)
class _SettingsStub:
    valorant_map_templates_dir: Path
    valorant_map_match_threshold: float = 0.45
    valorant_map_confirm_frames: int = 2
    valorant_map_clear_threshold: float = 0.10
    valorant_map_clear_frames: int = 2
    valorant_map_roi_top_pct: float = 0.14
    valorant_map_roi_bottom_pct: float = 0.34
    valorant_map_roi_left_pct: float = 0.24
    valorant_map_roi_right_pct: float = 0.76


def _frame(image: np.ndarray) -> FramePacket:
    return FramePacket(width=image.shape[1], height=image.shape[0], image_bgr=image, game="valorant")


def _render_template(text: str) -> np.ndarray:
    image = np.zeros((90, 360, 3), dtype=np.uint8)
    cv2.putText(
        image,
        text,
        (14, 62),
        cv2.FONT_HERSHEY_DUPLEX,
        1.9,
        (255, 255, 255),
        4,
        cv2.LINE_AA,
    )
    return image


def _render_loading_screen(settings: _SettingsStub, template_bgr: np.ndarray) -> np.ndarray:
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    left = int(image.shape[1] * settings.valorant_map_roi_left_pct)
    top = int(image.shape[0] * settings.valorant_map_roi_top_pct)
    roi_width = int(image.shape[1] * settings.valorant_map_roi_right_pct) - left
    x = left + max(0, (roi_width - template_bgr.shape[1]) // 2)
    y = top + 24
    image[y : y + template_bgr.shape[0], x : x + template_bgr.shape[1]] = template_bgr
    return image


def test_detects_loading_screen_map_after_confirm_frames() -> None:
    with TemporaryDirectory() as temp_dir:
        template_dir = Path(temp_dir)
        ascent_template = _render_template("ASCENT")
        bind_template = _render_template("BIND")
        cv2.imwrite(str(template_dir / "ascent.png"), ascent_template)
        cv2.imwrite(str(template_dir / "bind.png"), bind_template)

        settings = _SettingsStub(valorant_map_templates_dir=template_dir)
        detector = ValorantMapLoadingDetector(settings=settings)
        loading_frame = _frame(_render_loading_screen(settings, ascent_template))

        assert asyncio.run(detector.detect(loading_frame)) == []
        signals = asyncio.run(detector.detect(loading_frame))
        observation = detector.latest_observation
        print(
            "map observation:",
            observation.best_map_name if observation else None,
            observation.best_score if observation else None,
            observation.template_scores if observation else None,
        )

        assert len(signals) == 1
        assert signals[0].signal_type == "map_identified"
        assert signals[0].metadata["map_name"] == "Ascent"
        assert signals[0].confidence >= settings.valorant_map_match_threshold


def test_blank_frame_does_not_match_any_map() -> None:
    with TemporaryDirectory() as temp_dir:
        template_dir = Path(temp_dir)
        cv2.imwrite(str(template_dir / "ascent.png"), _render_template("ASCENT"))

        settings = _SettingsStub(valorant_map_templates_dir=template_dir)
        detector = ValorantMapLoadingDetector(settings=settings)
        blank_frame = _frame(np.zeros((1080, 1920, 3), dtype=np.uint8))

        assert asyncio.run(detector.detect(blank_frame)) == []
        assert asyncio.run(detector.detect(blank_frame)) == []
        assert detector.latest_observation is not None
        assert detector.latest_observation.best_score < settings.valorant_map_match_threshold
