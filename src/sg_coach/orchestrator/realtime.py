from __future__ import annotations

import asyncio
import json
from pathlib import Path

from sg_coach.orchestrator.session import SessionRuntime
from sg_coach.realtime.client import XaiRealtimeClient
from sg_coach.shared.events import CommentaryResult, SpeechCue
from sg_coach.shared.logging import get_logger
from sg_coach.shared.streaming import COMMENTARY_STREAM_COMPLETE, SPEECH_STREAM_COMPLETE


logger = get_logger(__name__)


def build_realtime_personality_instructions(personality: str) -> str:
    """Return the voice-agent persona block for the selected speech mode."""
    normalized = personality.strip().lower()

    if normalized == "unhinged":
        return (
            "Personality mode: Unhinged Mode. "
            "Sound more chaotic, more impulsive, and more willing to joke with profanity. "
            "You can swear casually and make weird, left-turn observations, but keep it playful. "
            "Roast bad gameplay sharply, then tack on one usable coaching hint when it fits. "
            "Do not become hateful, do not use slurs, and do not target protected classes."
        )

    return (
        "Personality mode: Sarcastic Coach. "
        "Be lightly sarcastic and question the player's judgment when they mess up, "
        "but stay coherent and useful. "
        "When appropriate, include one decent tip for what to do better next time."
    )


def build_realtime_instructions(runtime: SessionRuntime) -> str:
    """Build the persistent instruction block for the voice session."""
    return (
        "You are the live voice companion for a gaming assistant. "
        "Messages beginning with 'GAME_EVENT:' are system telemetry from the game pipeline, not user speech. "
        "Use those event messages as context for short, witty spoken reactions and callbacks. "
        "Real user speech will arrive as ordinary conversation turns later and should be treated separately. "
        "Do not claim hidden information. Keep comments concise and natural for voice. "
        f"{build_realtime_personality_instructions(runtime.settings.realtime_personality)} "
        f"Current local memory summary: {runtime.memory_summary() or 'No prior memory yet.'}"
    )


def build_realtime_text_from_commentary(result: CommentaryResult) -> str:
    """Convert multimodal commentary output into one event message for voice."""
    parts = ["GAME_EVENT: major gameplay event detected."]
    if result.visual_summary:
        parts.append(f"Visible outcome: {result.visual_summary}")
    if result.coach_note:
        parts.append(f"Coach note: {result.coach_note}")
    return " ".join(parts)


def build_realtime_text_from_speech_cue(cue: SpeechCue) -> str:
    """Convert a cheap ambient cue into one event message for voice."""
    return f"GAME_EVENT: ambient scene update. {cue.text}"


async def realtime_bridge_worker(
    runtime: SessionRuntime,
    *,
    commentary_queue: asyncio.Queue[CommentaryResult | str],
    speech_queue: asyncio.Queue[SpeechCue | str],
) -> None:
    """Forward local event summaries into a live xAI voice session."""
    commentary_open = True
    speech_open = True

    if not runtime.settings.realtime_enabled:
        logger.info("realtime bridge disabled")
        await _drain_bridge_queues(commentary_queue, speech_queue)
        return

    if not runtime.settings.realtime_api_key:
        logger.warning("realtime bridge disabled because SG_REALTIME_API_KEY is missing")
        await _drain_bridge_queues(commentary_queue, speech_queue)
        return

    client = XaiRealtimeClient(
        api_key=runtime.settings.realtime_api_key,
        ws_url=runtime.settings.realtime_ws_url,
        instructions=build_realtime_instructions(runtime),
        voice=runtime.settings.realtime_voice,
        language=runtime.settings.realtime_language,
    )

    try:
        await client.connect()
    except Exception:
        logger.exception("failed to connect realtime voice session")
        await _drain_bridge_queues(commentary_queue, speech_queue)
        return

    output_dir = runtime.settings.debug_realtime_dir / runtime.session_id
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_sink_task = asyncio.create_task(_realtime_raw_event_sink(client, output_dir=output_dir))
    text_sink_task = asyncio.create_task(_realtime_text_sink(client, output_dir=output_dir))

    primer = runtime.memory_summary()
    if primer:
        await client.send_memory_text(f"GAME_EVENT: session memory primer. {primer}")

    try:
        while commentary_open or speech_open:
            pending_tasks: dict[asyncio.Task, str] = {}
            if commentary_open:
                pending_tasks[asyncio.create_task(commentary_queue.get())] = "commentary"
            if speech_open:
                pending_tasks[asyncio.create_task(speech_queue.get())] = "speech"

            done, pending = await asyncio.wait(
                pending_tasks.keys(),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()

            for task in done:
                source = pending_tasks[task]
                item = task.result()

                if source == "commentary":
                    if item == COMMENTARY_STREAM_COMPLETE:
                        commentary_open = False
                        continue
                    if runtime.settings.realtime_emit_commentary:
                        await client.send_event_text(build_realtime_text_from_commentary(item))
                        logger.info(
                            "realtime bridge sent commentary request_id=%s",
                            item.request_id,
                        )
                else:
                    if item == SPEECH_STREAM_COMPLETE:
                        speech_open = False
                        continue
                    if runtime.settings.realtime_emit_speech_cues:
                        await client.send_event_text(build_realtime_text_from_speech_cue(item))
                        logger.info(
                            "realtime bridge sent speech cue cue_id=%s",
                            item.cue_id,
                        )
    finally:
        await client.close()
        await raw_sink_task
        await text_sink_task
        logger.info("realtime bridge complete")


async def _realtime_raw_event_sink(
    client: XaiRealtimeClient,
    *,
    output_dir: Path,
) -> None:
    """Persist raw realtime events to JSONL for protocol inspection."""
    output_path = output_dir / "realtime_events.jsonl"
    while True:
        event = await client.raw_event_queue.get()
        if event is None:
            return
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=str) + "\n")


async def _realtime_text_sink(
    client: XaiRealtimeClient,
    *,
    output_dir: Path,
) -> None:
    """Persist final assistant text turns and log them live."""
    turn_index = 0
    while True:
        text = await client.text_queue.get()
        if text is None:
            return

        output_path = output_dir / f"assistant_{turn_index:03d}.txt"
        output_path.write_text(text, encoding="utf-8")
        logger.info("realtime assistant turn=%s text=%s output=%s", turn_index, text, output_path)
        turn_index += 1


async def _drain_bridge_queues(
    commentary_queue: asyncio.Queue[CommentaryResult | str],
    speech_queue: asyncio.Queue[SpeechCue | str],
) -> None:
    """Drain both upstream queues so the pipeline can still terminate cleanly."""
    commentary_open = True
    speech_open = True
    while commentary_open or speech_open:
        pending_tasks: dict[asyncio.Task, str] = {}
        if commentary_open:
            pending_tasks[asyncio.create_task(commentary_queue.get())] = "commentary"
        if speech_open:
            pending_tasks[asyncio.create_task(speech_queue.get())] = "speech"

        done, pending = await asyncio.wait(pending_tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()

        for task in done:
            source = pending_tasks[task]
            item = task.result()
            if source == "commentary" and item == COMMENTARY_STREAM_COMPLETE:
                commentary_open = False
            if source == "speech" and item == SPEECH_STREAM_COMPLETE:
                speech_open = False
