from __future__ import annotations

import asyncio

import numpy as np

from sg_coach.detectors.gta.chaos import GtaChaosDetector
from sg_coach.shared.events import FramePacket
from sg_coach.shared.settings import Settings


def _frame(image: np.ndarray) -> FramePacket:
    return FramePacket(width=image.shape[1], height=image.shape[0], image_bgr=image, game="gta_like")


def test_bright_static_scene_does_not_trigger_chaos() -> None:
    settings = Settings(
        gta_chaos_startup_delay_seconds=0,
        gta_chaos_confirm_frames=1,
        gta_chaos_motion_threshold=0.16,
        gta_chaos_flash_threshold=0.035,
        gta_chaos_edge_threshold=0.18,
        gta_chaos_score_threshold=0.20,
    )
    detector = GtaChaosDetector(settings=settings)

    calm_bright = np.full((120, 160, 3), 230, dtype=np.uint8)
    slightly_different = calm_bright.copy()
    slightly_different[:, :10] = 220

    assert asyncio.run(detector.detect(_frame(calm_bright))) == []
    assert asyncio.run(detector.detect(_frame(slightly_different))) == []


def test_large_scene_change_triggers_chaos() -> None:
    settings = Settings(
        gta_chaos_startup_delay_seconds=0,
        gta_chaos_confirm_frames=1,
        gta_chaos_motion_threshold=0.16,
        gta_chaos_flash_threshold=0.035,
        gta_chaos_edge_threshold=0.18,
        gta_chaos_score_threshold=0.20,
    )
    detector = GtaChaosDetector(settings=settings)

    calm = np.zeros((120, 160, 3), dtype=np.uint8)
    explosion = np.zeros((120, 160, 3), dtype=np.uint8)
    explosion[:, 80:] = 255

    assert asyncio.run(detector.detect(_frame(calm))) == []
    signals = asyncio.run(detector.detect(_frame(explosion)))

    assert len(signals) == 1
    assert signals[0].signal_type == "chaos_spike"
    assert signals[0].metadata["chaos_score"] >= settings.gta_chaos_score_threshold
