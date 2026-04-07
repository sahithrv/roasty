from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

import cv2

from sg_coach.capture.dxcam_backend import DxcamFrameSource
from sg_coach.detectors.gta.wanted_stars import GtaWantedStarsDetector, WantedStarsObservation
from sg_coach.shared.logging import configure_logging
from sg_coach.shared.settings import load_settings


def format_observation(observation: WantedStarsObservation | None) -> str:
    """Return one readable line describing what the detector currently sees."""
    if observation is None:
        return "no observation available yet"

    slot_scores = ", ".join(f"{score:.3f}" for score in observation.slot_scores)
    present_slot_scores = ", ".join(f"{score:.3f}" for score in observation.present_slot_scores)
    shape_scores = ", ".join(f"{score:.3f}" for score in observation.shape_scores)
    fill_scores = ", ".join(f"{score:.3f}" for score in observation.fill_scores)
    return (
        f"observed={observation.observed_star_count} "
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


def save_annotated_probe_frame(
    *,
    image_bgr,
    observation: WantedStarsObservation | None,
    output_dir: Path,
    frame_number: int,
    reason: str,
) -> Path | None:
    """Save one full-screen screenshot with the wanted ROI highlighted."""
    if image_bgr is None or observation is None:
        return None

    annotated = image_bgr.copy()
    left, top, right, bottom = observation.roi_bounds
    cv2.rectangle(annotated, (left, top), (right, bottom), (0, 255, 255), 3)
    for index, (slot_left, slot_top, slot_right, slot_bottom) in enumerate(observation.slot_bounds):
        label = observation.slot_labels[index]
        color = _slot_label_color(label)
        cv2.rectangle(annotated, (slot_left, slot_top), (slot_right, slot_bottom), color, 2)
        cv2.putText(
            annotated,
            str(index + 1),
            (slot_left + 4, slot_top + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    _draw_boxed_text(
        annotated,
        f"wanted_roi observed={observation.observed_star_count} stable={observation.stable_star_count}",
        anchor_x=left,
        anchor_y=bottom + 10,
        color=(0, 255, 255),
    )
    _draw_boxed_text(
        annotated,
        f"raw_active={observation.raw_active_star_slots} active={observation.active_star_slots} "
        f"visible={observation.present_star_slots}",
        anchor_x=left,
        anchor_y=bottom + 34,
        color=(255, 255, 255),
    )
    _draw_boxed_text(
        annotated,
        f"all_slots_visible={observation.all_slots_visible} flashing_hold={observation.flashing_hold} "
        f"labels={observation.slot_labels}",
        anchor_x=left,
        anchor_y=bottom + 58,
        color=(190, 240, 255),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"frame_{frame_number:06d}_{reason}.jpg"
    if cv2.imwrite(str(output_path), annotated):
        _save_probe_crops(
            image_bgr=image_bgr,
            observation=observation,
            output_dir=output_dir,
            frame_number=frame_number,
            reason=reason,
        )
        return output_path
    return None


def _slot_label_color(label: str) -> tuple[int, int, int]:
    if label == "active":
        return (70, 255, 70)
    if label == "outline":
        return (0, 170, 255)
    return (0, 90, 180)


def _save_probe_crops(
    *,
    image_bgr,
    observation: WantedStarsObservation,
    output_dir: Path,
    frame_number: int,
    reason: str,
) -> None:
    """Save the ROI crop and each slot crop for later tuning."""
    crop_dir = output_dir / f"frame_{frame_number:06d}_{reason}_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    left, top, right, bottom = observation.roi_bounds
    roi_crop = image_bgr[top:bottom, left:right]
    cv2.imwrite(str(crop_dir / "wanted_roi.jpg"), roi_crop)

    for index, (slot_left, slot_top, slot_right, slot_bottom) in enumerate(observation.slot_bounds, start=1):
        slot_crop = image_bgr[slot_top:slot_bottom, slot_left:slot_right]
        label = observation.slot_labels[index - 1]
        score = observation.slot_scores[index - 1]
        shape_score = observation.shape_scores[index - 1]
        fill_score = observation.fill_scores[index - 1]
        slot_path = crop_dir / (
            f"slot_{index:02d}_{label}_active_{score:.3f}_shape_{shape_score:.3f}_fill_{fill_score:.3f}.jpg"
        )
        cv2.imwrite(str(slot_path), slot_crop)


def _draw_boxed_text(
    image_bgr,
    text: str,
    *,
    anchor_x: int,
    anchor_y: int,
    color: tuple[int, int, int],
) -> None:
    """Draw readable probe text without covering the HUD ROI itself."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.58
    thickness = 2
    padding = 6
    baseline_gap = 4
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    image_height, image_width = image_bgr.shape[:2]

    box_left = max(10, min(anchor_x, image_width - text_size[0] - (padding * 2) - 10))
    box_top = anchor_y
    if box_top + text_size[1] + (padding * 2) + baseline > image_height:
        box_top = max(10, anchor_y - text_size[1] - (padding * 2) - 16)

    box_bottom = min(image_height - 10, box_top + text_size[1] + (padding * 2) + baseline)
    box_top = max(10, box_bottom - text_size[1] - (padding * 2) - baseline)
    box_right = min(image_width - 10, box_left + text_size[0] + (padding * 2))
    text_origin = (box_left + padding, box_bottom - padding - baseline_gap)

    cv2.rectangle(image_bgr, (box_left, box_top), (box_right, box_bottom), (0, 0, 0), -1)
    cv2.rectangle(image_bgr, (box_left, box_top), (box_right, box_bottom), color, 1)
    cv2.putText(image_bgr, text, text_origin, font, font_scale, color, thickness, cv2.LINE_AA)


async def run_probe(*, frame_limit: int | None, print_every: int, save_every_print: bool) -> None:
    """Capture live GTA frames and print wanted-detector observations."""
    settings = load_settings()
    configure_logging(settings.log_level)
    source = DxcamFrameSource(settings=settings, game="gta_like")
    detector = GtaWantedStarsDetector(settings=settings)
    output_dir = settings.debug_frames_dir / "wanted_probe" / datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Wanted stars detector probe")
    print(
        f"monitor={settings.capture_monitor_id} target_fps={settings.target_fps} "
        f"confirm_frames={settings.gta_wanted_confirm_frames} "
        f"outline_threshold={settings.gta_wanted_shape_presence_threshold} "
        f"active_threshold={settings.gta_wanted_active_score_threshold}"
    )
    print(
        f"slot_centers={settings.gta_wanted_slot_centers_pct} "
        f"slot_half_width_pct={settings.gta_wanted_slot_half_width_pct} "
        f"slot_top_pct={settings.gta_wanted_slot_top_pct} "
        f"slot_bottom_pct={settings.gta_wanted_slot_bottom_pct}"
    )
    print(f"Annotated screenshots will be saved under: {output_dir}")
    print("Press Ctrl+C to stop.")

    frame_number = 0
    async for frame in source.frames():
        frame_number += 1
        signals = await detector.detect(frame)
        observation = detector.latest_observation

        should_print_observation = bool(signals) or frame_number == 1
        if print_every > 0 and frame_number % print_every == 0:
            should_print_observation = True

        if should_print_observation:
            print(f"[frame {frame_number}] {format_observation(observation)}", flush=True)
            if save_every_print:
                screenshot_path = save_annotated_probe_frame(
                    image_bgr=frame.image_bgr,
                    observation=observation,
                    output_dir=output_dir,
                    frame_number=frame_number,
                    reason="observation",
                )
                if screenshot_path is not None:
                    print(f"[frame {frame_number}] SCREENSHOT {screenshot_path}", flush=True)

        for signal in signals:
            print(
                f"[frame {frame_number}] SIGNAL {signal.signal_type} "
                f"{json.dumps(signal.metadata, default=str)}",
                flush=True,
            )
            screenshot_path = save_annotated_probe_frame(
                image_bgr=frame.image_bgr,
                observation=observation,
                output_dir=output_dir,
                frame_number=frame_number,
                reason=signal.signal_type,
            )
            if screenshot_path is not None:
                print(f"[frame {frame_number}] SCREENSHOT {screenshot_path}", flush=True)

        if frame_limit is not None and frame_number >= frame_limit:
            print(f"Stopped after {frame_number} frames.", flush=True)
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe the live GTA wanted-stars detector.")
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Optional number of frames to inspect before exiting.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=12,
        help="Print one observation line every N frames even if no signal fires.",
    )
    parser.add_argument(
        "--save-every-print",
        action="store_true",
        help="Also save an annotated screenshot for each printed observation line.",
    )
    args = parser.parse_args()

    try:
        asyncio.run(
            run_probe(
                frame_limit=args.frames,
                print_every=args.print_every,
                save_every_print=args.save_every_print,
            )
        )
    except KeyboardInterrupt:
        print("Wanted stars detector probe stopped.", flush=True)


if __name__ == "__main__":
    main()
