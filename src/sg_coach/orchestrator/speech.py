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
    if event.event_type not in {"wanted_level_started", "wanted_level_changed", "wanted_level_cleared"}:
        return None

    previous_level = int(event.metadata.get("previous_wanted_level", 0))
    current_level = int(event.metadata.get("wanted_level", 0))
    change_direction = str(event.metadata.get("change_direction", "")).strip().lower()

    if event.event_type == "wanted_level_started":
        text = f"Cops are on you now. Wanted level started at {current_level} stars."
    elif event.event_type == "wanted_level_cleared":
        text = "Wanted level cleared. The heat is off for now."
    elif change_direction == "increase":
        text = f"Wanted level just climbed from {previous_level} to {current_level} stars."
    else:
        text = f"Wanted level dropped from {previous_level} to {current_level} stars."

    return SpeechCue(
        session_id=runtime.session_id,
        source_event_id=event.event_id,
        cue_type="ambient_commentary_seed",
        text=text,
        metadata={
            "event_type": event.event_type,
            "confidence": round(event.confidence, 4),
            "source_detector": event.metadata.get("source_detector"),
            "previous_wanted_level": previous_level,
            "wanted_level": current_level,
            "change_direction": change_direction,
            "active_star_slots": list(event.metadata.get("active_star_slots", [])),
        },
    )


async def speech_cue_worker(
    runtime: SessionRuntime,
    event_queue: asyncio.Queue[GameEvent | str],
) -> None:
    """Turn cheap HUD/state events into text cues for a future realtime layer."""
    last_emitted_at_by_type: dict[str, float] = {}
    loop = asyncio.get_running_loop()

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

        cue_key = f"{cue.cue_type}:{event.event_type}"
        min_interval = runtime.settings.realtime_speech_cue_min_interval_seconds
        now = loop.time()
        last_emitted_at = last_emitted_at_by_type.get(cue_key)
        if min_interval > 0 and last_emitted_at is not None and (now - last_emitted_at) < min_interval:
            remaining = max(0.0, min_interval - (now - last_emitted_at))
            logger.info(
                "speech cue dropped event_type=%s cue_type=%s reason=throttled retry_in=%.1fs",
                event.event_type,
                cue.cue_type,
                remaining,
            )
            continue

        last_emitted_at_by_type[cue_key] = now
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
