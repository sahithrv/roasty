from __future__ import annotations

import asyncio
from pathlib import Path

from sg_coach.orchestrator.session import SessionRuntime
from sg_coach.orchestrator.topics import SPEECH_PLAY
from sg_coach.shared.events import GameEvent, SpeechCue
from sg_coach.shared.logging import get_logger
from sg_coach.shared.streaming import EVENT_STREAM_COMPLETE, SPEECH_STREAM_COMPLETE


logger = get_logger(__name__)


def build_speech_cue_from_event(runtime: SessionRuntime, event: GameEvent) -> SpeechCue | None:
    """Create a cheap text cue for events that do not need a full model call."""
    if event.event_type != "chaos_spike":
        return None

    motion = float(event.metadata.get("motion_score", 0.0))
    flash = float(event.metadata.get("flash_ratio", 0.0))
    edge = float(event.metadata.get("edge_density", 0.0))

    descriptors: list[str] = []
    if motion >= runtime.settings.gta_chaos_motion_threshold:
        descriptors.append("rapid movement")
    if flash >= runtime.settings.gta_chaos_flash_threshold:
        descriptors.append("bright flashes")
    if edge >= runtime.settings.gta_chaos_edge_threshold:
        descriptors.append("heavy scene clutter")

    descriptor_text = ", ".join(descriptors) if descriptors else "scene turbulence"
    text = f"Chaos spike detected: {descriptor_text} just kicked up on screen."

    return SpeechCue(
        session_id=runtime.session_id,
        source_event_id=event.event_id,
        cue_type="ambient_commentary_seed",
        text=text,
        metadata={
            "event_type": event.event_type,
            "confidence": round(event.confidence, 4),
            "source_detector": event.metadata.get("source_detector"),
            "motion_score": round(motion, 4),
            "flash_ratio": round(flash, 4),
            "edge_density": round(edge, 4),
        },
    )


async def speech_cue_worker(
    runtime: SessionRuntime,
    event_queue: asyncio.Queue[GameEvent | str],
) -> None:
    """Turn cheap ambient events into text cues for a future realtime layer."""
    while True:
        item = await event_queue.get()
        if item == EVENT_STREAM_COMPLETE:
            await runtime.publish(SPEECH_PLAY, SPEECH_STREAM_COMPLETE)
            logger.info("speech cue worker complete")
            return

        event = item
        cue = build_speech_cue_from_event(runtime, event)
        if cue is None:
            continue

        await runtime.publish(SPEECH_PLAY, cue)
        logger.info(
            "speech cue built event_type=%s cue_type=%s text=%s",
            event.event_type,
            cue.cue_type,
            cue.text,
        )


async def speech_cue_sink(
    speech_queue: asyncio.Queue[SpeechCue | str],
    *,
    output_root: Path,
    session_id: str,
) -> None:
    """Persist lightweight speech cues so they can be inspected and replayed."""
    output_dir = output_root / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    while True:
        item = await speech_queue.get()
        if item == SPEECH_STREAM_COMPLETE:
            logger.info("speech cue sink complete")
            return

        cue = item
        output_path = output_dir / f"{cue.cue_id}.txt"
        output_path.write_text(cue.text, encoding="utf-8")
        logger.info(
            "speech cue ready cue_id=%s cue_type=%s output=%s text=%s",
            cue.cue_id,
            cue.cue_type,
            output_path,
            cue.text,
        )
