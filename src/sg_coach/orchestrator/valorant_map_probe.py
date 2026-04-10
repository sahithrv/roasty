from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

import cv2

from sg_coach.capture.dxcam_backend import DxcamFrameSource
from sg_coach.detectors.valorant.map_loading import (
    ValorantMapLoadingDetector,
    ValorantMapObservation,
)
from sg_coach.shared.logging import configure_logging
from sg_coach.shared.settings import load_settings


def format_observation(observation: ValorantMapObservation | None) -> str:
    if observation is None:
        return "no observation available yet"

    score_parts = ", ".join(
        f"{name}={score:.3f}" for name, score in sorted(
            observation.template_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )
    return (
        f"best_map={observation.best_map_name!r} "
        f"best_score={observation.best_score:.3f} "
        f"confirmed_map={observation.confirmed_map_name!r} "
        f"candidate_map={observation.candidate_map_name!r} "
        f"candidate_frames={observation.candidate_frames} "
        f"roi_bounds={observation.roi_bounds} "
        f"best_match_bounds={observation.best_match_bounds} "
        f"template_scores=[{score_parts}]"
    )


def save_annotated_probe_frame(
    *,
    image_bgr,
    observation: ValorantMapObservation | None,
    output_dir: Path,
    frame_number: int,
    reason: str,
) -> Path | None:
    if image_bgr is None or observation is None:
        return None

    annotated = image_bgr.copy()
    left, top, right, bottom = observation.roi_bounds
    cv2.rectangle(annotated, (left, top), (right, bottom), (0, 255, 255), 3)

    if observation.best_match_bounds is not None:
        match_left, match_top, match_right, match_bottom = observation.best_match_bounds
        cv2.rectangle(annotated, (match_left, match_top), (match_right, match_bottom), (0, 255, 0), 2)

    _draw_boxed_text(
        annotated,
        f"best_map={observation.best_map_name} score={observation.best_score:.3f}",
        anchor_x=left,
        anchor_y=bottom + 10,
        color=(0, 255, 255),
    )
    _draw_boxed_text(
        annotated,
        f"confirmed={observation.confirmed_map_name} candidate={observation.candidate_map_name} "
        f"frames={observation.candidate_frames}",
        anchor_x=left,
        anchor_y=bottom + 34,
        color=(255, 255, 255),
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


def _save_probe_crops(
    *,
    image_bgr,
    observation: ValorantMapObservation,
    output_dir: Path,
    frame_number: int,
    reason: str,
) -> None:
    crop_dir = output_dir / f"frame_{frame_number:06d}_{reason}_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    left, top, right, bottom = observation.roi_bounds
    roi_crop = image_bgr[top:bottom, left:right]
    cv2.imwrite(str(crop_dir / "map_roi.jpg"), roi_crop)


def _draw_boxed_text(
    image_bgr,
    text: str,
    *,
    anchor_x: int,
    anchor_y: int,
    color: tuple[int, int, int],
) -> None:
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
    settings = load_settings()
    configure_logging(settings.log_level)
    source = DxcamFrameSource(settings=settings, game="valorant")
    detector = ValorantMapLoadingDetector(settings=settings)
    output_dir = settings.debug_frames_dir / "valorant_map_probe" / datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Valorant map detector probe")
    print(
        f"monitor={settings.capture_monitor_id} target_fps={settings.target_fps} "
        f"templates_dir={settings.valorant_map_templates_dir}"
    )
    print(
        f"roi=({settings.valorant_map_roi_left_pct:.3f}, {settings.valorant_map_roi_top_pct:.3f}) -> "
        f"({settings.valorant_map_roi_right_pct:.3f}, {settings.valorant_map_roi_bottom_pct:.3f}) "
        f"threshold={settings.valorant_map_match_threshold:.3f} "
        f"confirm_frames={settings.valorant_map_confirm_frames}"
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
    parser = argparse.ArgumentParser(description="Probe the live Valorant map detector.")
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
        print("Valorant map detector probe stopped.", flush=True)


if __name__ == "__main__":
    main()
