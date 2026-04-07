from __future__ import annotations

import asyncio
import math

import cv2
import numpy as np

from sg_coach.detectors.gta.wanted_stars import GtaWantedStarsDetector
from sg_coach.shared.events import FramePacket
from sg_coach.shared.settings import Settings


def _frame(image: np.ndarray) -> FramePacket:
    return FramePacket(width=image.shape[1], height=image.shape[0], image_bgr=image, game="gta_like")


def _print_latest_observation(label: str, detector: GtaWantedStarsDetector) -> None:
    observation = detector.latest_observation
    if observation is None:
        print(f"{label}: no observation captured yet")
        return

    slot_scores = ", ".join(f"{score:.3f}" for score in observation.slot_scores)
    present_slot_scores = ", ".join(f"{score:.3f}" for score in observation.present_slot_scores)
    shape_scores = ", ".join(f"{score:.3f}" for score in observation.shape_scores)
    fill_scores = ", ".join(f"{score:.3f}" for score in observation.fill_scores)
    print(
        f"{label}: observed={observation.observed_star_count} "
        f"stable={observation.stable_star_count} "
        f"candidate={observation.candidate_star_count} "
        f"candidate_frames={observation.candidate_frames} "
        f"labels={observation.slot_labels} "
        f"raw_active_slots={observation.raw_active_star_slots} "
        f"active_slots={observation.active_star_slots} "
        f"present_slots={observation.present_star_slots} "
        f"all_slots_visible={observation.all_slots_visible} "
        f"startup_blocked={observation.startup_blocked} "
        f"flashing_hold={observation.flashing_hold} "
        f"roi_bounds={observation.roi_bounds} "
        f"slot_scores=[{slot_scores}] "
        f"present_slot_scores=[{present_slot_scores}] "
        f"shape_scores=[{shape_scores}] "
        f"fill_scores=[{fill_scores}]"
    )


def _star_points(left: int, top: int, right: int, bottom: int) -> np.ndarray:
    width = right - left
    height = bottom - top
    center_x = left + width / 2
    center_y = top + height / 2
    outer_radius = min(width, height) * 0.42
    inner_radius = outer_radius * 0.46
    points: list[list[int]] = []

    for index in range(10):
        angle = math.radians(-90 + index * 36)
        radius = outer_radius if index % 2 == 0 else inner_radius
        x = int(round(center_x + math.cos(angle) * radius))
        y = int(round(center_y + math.sin(angle) * radius))
        points.append([x, y])

    return np.array(points, dtype=np.int32)


def _draw_wanted_stars(
    detector: GtaWantedStarsDetector,
    image: np.ndarray,
    *,
    count: int,
    color: tuple[int, int, int] = (255, 255, 255),
    outline_color: tuple[int, int, int] | None = None,
    fill_active: bool = True,
    show_outlines: bool | None = None,
) -> np.ndarray:
    drawn = image.copy()
    roi_bgr, (roi_left, roi_top, _, _) = detector._extract_wanted_roi(drawn)
    slot_bounds = detector._iter_star_slot_bounds(width=roi_bgr.shape[1], height=roi_bgr.shape[0])
    active_start_index = max(0, len(slot_bounds) - count)
    should_draw_outlines = count > 0 if show_outlines is None else show_outlines
    for slot_index, (left, top, right, bottom) in enumerate(
        detector._iter_star_slot_bounds(width=roi_bgr.shape[1], height=roi_bgr.shape[0])
    ):
        points = _star_points(roi_left + left, roi_top + top, roi_left + right, roi_top + bottom)
        if should_draw_outlines:
            cv2.polylines(
                drawn,
                [points],
                isClosed=True,
                color=outline_color or (190, 190, 190),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
        if slot_index < active_start_index:
            continue
        if fill_active:
            cv2.fillPoly(drawn, [points], color=color)
    return drawn


def _draw_bright_noise(detector: GtaWantedStarsDetector, image: np.ndarray) -> np.ndarray:
    drawn = image.copy()
    _, (roi_left, roi_top, roi_right, roi_bottom) = detector._extract_wanted_roi(drawn)
    cv2.rectangle(
        drawn,
        (roi_left + 18, roi_top + 8),
        (roi_right - 10, roi_top + 24),
        color=(255, 255, 255),
        thickness=-1,
    )
    cv2.line(
        drawn,
        (roi_left + 26, roi_top + 30),
        (roi_right - 30, roi_bottom - 10),
        color=(255, 255, 255),
        thickness=4,
    )
    return drawn


def test_wanted_level_start_and_change_emit_stable_events() -> None:
    settings = Settings(
        gta_wanted_confirm_frames=2,
        gta_wanted_shape_presence_threshold=0.46,
        gta_wanted_active_score_threshold=0.52,
        gta_wanted_startup_delay_seconds=0,
    )
    detector = GtaWantedStarsDetector(settings=settings)
    base = np.zeros((1080, 1920, 3), dtype=np.uint8)

    assert asyncio.run(detector.detect(_frame(base))) == []
    _print_latest_observation("baseline", detector)

    two_stars = _draw_wanted_stars(detector, base, count=2)
    assert asyncio.run(detector.detect(_frame(two_stars))) == []
    _print_latest_observation("two_stars_candidate", detector)
    started = asyncio.run(detector.detect(_frame(two_stars)))
    _print_latest_observation("two_stars_started", detector)
    print(f"two_stars_started signals={started}")
    assert len(started) == 1
    assert started[0].signal_type == "wanted_level_started"
    assert started[0].metadata["previous_wanted_level"] == 0
    assert started[0].metadata["wanted_level"] == 2
    assert detector.latest_observation is not None
    assert detector.latest_observation.present_star_slots == [1, 2, 3, 4, 5]
    assert detector.latest_observation.all_slots_visible is True

    four_stars = _draw_wanted_stars(detector, base, count=4)
    assert asyncio.run(detector.detect(_frame(four_stars))) == []
    _print_latest_observation("four_stars_candidate", detector)
    changed = asyncio.run(detector.detect(_frame(four_stars)))
    _print_latest_observation("four_stars_changed", detector)
    print(f"four_stars_changed signals={changed}")
    assert len(changed) == 1
    assert changed[0].signal_type == "wanted_level_changed"
    assert changed[0].metadata["previous_wanted_level"] == 2
    assert changed[0].metadata["wanted_level"] == 4
    assert changed[0].metadata["change_direction"] == "increase"


def test_wanted_level_clear_emits_event() -> None:
    settings = Settings(
        gta_wanted_confirm_frames=2,
        gta_wanted_shape_presence_threshold=0.46,
        gta_wanted_active_score_threshold=0.52,
        gta_wanted_startup_delay_seconds=0,
    )
    detector = GtaWantedStarsDetector(settings=settings)
    base = np.zeros((1080, 1920, 3), dtype=np.uint8)
    three_stars = _draw_wanted_stars(detector, base, count=3)

    assert asyncio.run(detector.detect(_frame(base))) == []
    _print_latest_observation("clear_baseline", detector)
    assert asyncio.run(detector.detect(_frame(three_stars))) == []
    _print_latest_observation("three_stars_candidate", detector)
    started = asyncio.run(detector.detect(_frame(three_stars)))
    _print_latest_observation("three_stars_started", detector)
    print(f"three_stars_started signals={started}")
    assert started[0].signal_type == "wanted_level_started"

    assert asyncio.run(detector.detect(_frame(base))) == []
    _print_latest_observation("clear_candidate", detector)
    cleared = asyncio.run(detector.detect(_frame(base)))
    _print_latest_observation("clear_final", detector)
    print(f"clear_final signals={cleared}")
    assert len(cleared) == 1
    assert cleared[0].signal_type == "wanted_level_cleared"
    assert cleared[0].metadata["previous_wanted_level"] == 3
    assert cleared[0].metadata["wanted_level"] == 0


def test_dim_stars_hold_previous_level_during_flash() -> None:
    settings = Settings(
        gta_wanted_confirm_frames=2,
        gta_wanted_shape_presence_threshold=0.46,
        gta_wanted_active_score_threshold=0.52,
        gta_wanted_startup_delay_seconds=0,
    )
    detector = GtaWantedStarsDetector(settings=settings)
    base = np.zeros((1080, 1920, 3), dtype=np.uint8)
    active_three = _draw_wanted_stars(detector, base, count=3)
    dim_three = _draw_wanted_stars(
        detector,
        base,
        count=3,
        fill_active=False,
        show_outlines=True,
    )

    assert asyncio.run(detector.detect(_frame(base))) == []
    assert asyncio.run(detector.detect(_frame(active_three))) == []
    started = asyncio.run(detector.detect(_frame(active_three)))
    _print_latest_observation("flash_started", detector)
    print(f"flash_started signals={started}")
    assert started[0].signal_type == "wanted_level_started"

    flashing = asyncio.run(detector.detect(_frame(dim_three)))
    _print_latest_observation("flash_dim_hold", detector)
    print(f"flash_dim_hold signals={flashing}")
    assert flashing == []
    assert detector.latest_observation is not None
    assert detector.latest_observation.flashing_hold is True
    assert detector.latest_observation.observed_star_count == 3


def test_partial_drop_is_treated_as_real_decrease_not_flash() -> None:
    settings = Settings(
        gta_wanted_confirm_frames=2,
        gta_wanted_shape_presence_threshold=0.46,
        gta_wanted_active_score_threshold=0.52,
        gta_wanted_startup_delay_seconds=0,
    )
    detector = GtaWantedStarsDetector(settings=settings)
    base = np.zeros((1080, 1920, 3), dtype=np.uint8)
    active_three = _draw_wanted_stars(detector, base, count=3)
    active_two = _draw_wanted_stars(detector, base, count=2)

    assert asyncio.run(detector.detect(_frame(base))) == []
    assert asyncio.run(detector.detect(_frame(active_three))) == []
    started = asyncio.run(detector.detect(_frame(active_three)))
    _print_latest_observation("decrease_started", detector)
    print(f"decrease_started signals={started}")
    assert started[0].signal_type == "wanted_level_started"

    assert asyncio.run(detector.detect(_frame(active_two))) == []
    _print_latest_observation("decrease_candidate", detector)
    changed = asyncio.run(detector.detect(_frame(active_two)))
    _print_latest_observation("decrease_changed", detector)
    print(f"decrease_changed signals={changed}")
    assert len(changed) == 1
    assert changed[0].signal_type == "wanted_level_changed"
    assert changed[0].metadata["change_direction"] == "decrease"
    assert changed[0].metadata["wanted_level"] == 2
    assert detector.latest_observation is not None
    assert detector.latest_observation.flashing_hold is False


def test_bright_noise_does_not_trigger_false_star_count() -> None:
    settings = Settings(
        gta_wanted_confirm_frames=2,
        gta_wanted_shape_presence_threshold=0.46,
        gta_wanted_active_score_threshold=0.52,
        gta_wanted_startup_delay_seconds=0,
    )
    detector = GtaWantedStarsDetector(settings=settings)
    base = np.zeros((1080, 1920, 3), dtype=np.uint8)
    noisy = _draw_bright_noise(detector, base)

    assert asyncio.run(detector.detect(_frame(base))) == []
    first_noise = asyncio.run(detector.detect(_frame(noisy)))
    _print_latest_observation("bright_noise_first", detector)
    print(f"bright_noise_first signals={first_noise}")
    second_noise = asyncio.run(detector.detect(_frame(noisy)))
    _print_latest_observation("bright_noise_second", detector)
    print(f"bright_noise_second signals={second_noise}")

    assert first_noise == []
    assert second_noise == []
    assert detector.latest_observation is not None
    assert detector.latest_observation.observed_star_count == 0
    assert detector.latest_observation.active_star_slots == []
