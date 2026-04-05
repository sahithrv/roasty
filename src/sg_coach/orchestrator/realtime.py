from __future__ import annotations

import asyncio
import contextlib
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


def _build_voice_state() -> dict[str, bool]:
    """Create the mutable session-local arbitration state for voice interactions."""
    return {
        "ptt_active": False,
        "buffering_user_audio": False,
        "awaiting_user_response": False,
    }


def _user_priority_active(voice_state: dict[str, bool]) -> bool:
    """Return whether user speech should outrank game narration right now."""
    return (
        voice_state["ptt_active"]
        or voice_state["buffering_user_audio"]
        or voice_state["awaiting_user_response"]
    )


def _assistant_audio_suppressed(voice_state: dict[str, bool]) -> bool:
    """Return whether local assistant playback should be muted for user speech."""
    return voice_state["ptt_active"] or voice_state["buffering_user_audio"]


async def _flush_user_audio_turn(
    client: XaiRealtimeClient,
    *,
    voice_state: dict[str, bool],
    chunks: list[bytes],
) -> None:
    """Send one push-to-talk mic burst as a single user turn."""
    if not chunks:
        return

    voice_state["buffering_user_audio"] = True
    total_bytes = sum(len(chunk) for chunk in chunks)
    try:
        await client.wait_until_idle()
        voice_state["awaiting_user_response"] = True
        for chunk in chunks:
            await client.append_input_audio(chunk)
        await client.commit_input_audio()
        logger.info(
            "realtime user speech committed chunks=%s bytes=%s",
            len(chunks),
            total_bytes,
        )
    except Exception:
        voice_state["awaiting_user_response"] = False
        raise
    finally:
        voice_state["buffering_user_audio"] = False


async def _realtime_user_input_worker(
    client: XaiRealtimeClient,
    *,
    runtime: SessionRuntime,
    voice_state: dict[str, bool],
) -> None:
    """Capture user mic audio behind push-to-talk and forward it to realtime."""
    if not runtime.settings.realtime_enable_user_speech:
        logger.info("realtime user speech disabled")
        return

    from sg_coach.realtime.user_voice import PushToTalkController, RealtimeMicrophoneInput

    loop = asyncio.get_running_loop()
    ptt_queue: asyncio.Queue[bool | None] = asyncio.Queue()
    mic_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
    push_to_talk = PushToTalkController(
        key_spec=runtime.settings.realtime_push_to_talk_key,
        loop=loop,
        state_queue=ptt_queue,
    )
    microphone = RealtimeMicrophoneInput(
        sample_rate=runtime.settings.realtime_input_sample_rate,
        device=runtime.settings.realtime_input_audio_device,
        loop=loop,
        audio_queue=mic_queue,
    )

    current_turn_chunks: list[bytes] = []
    pending_flush_task: asyncio.Task[None] | None = None

    try:
        push_to_talk.start()
        microphone.start()
    except Exception:
        logger.exception("failed to start realtime user speech input")
        await push_to_talk.stop()
        await microphone.stop()
        return

    logger.info(
        "realtime user speech ready push_to_talk_key=%s input_sample_rate=%s",
        runtime.settings.realtime_push_to_talk_key,
        runtime.settings.realtime_input_sample_rate,
    )

    try:
        while True:
            pending_tasks = {
                asyncio.create_task(ptt_queue.get()): "ptt",
                asyncio.create_task(mic_queue.get()): "mic",
            }
            done, pending = await asyncio.wait(
                pending_tasks.keys(),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()

            for task in done:
                source = pending_tasks[task]
                item = task.result()

                if source == "ptt":
                    if item is None:
                        return

                    is_pressed = bool(item)
                    if is_pressed:
                        voice_state["ptt_active"] = True
                        current_turn_chunks = []
                        microphone.set_capture_enabled(True)
                        logger.info(
                            "push-to-talk pressed key=%s",
                            runtime.settings.realtime_push_to_talk_key,
                        )
                        continue

                    microphone.set_capture_enabled(False)
                    voice_state["ptt_active"] = False
                    logger.info(
                        "push-to-talk released key=%s captured_chunks=%s",
                        runtime.settings.realtime_push_to_talk_key,
                        len(current_turn_chunks),
                    )

                    if not current_turn_chunks:
                        continue

                    if pending_flush_task is not None and not pending_flush_task.done():
                        logger.info(
                            "user speech turn dropped reason=previous_turn_still_pending chunks=%s",
                            len(current_turn_chunks),
                        )
                        current_turn_chunks = []
                        continue

                    flush_chunks = list(current_turn_chunks)
                    current_turn_chunks = []
                    pending_flush_task = asyncio.create_task(
                        _flush_user_audio_turn(
                            client,
                            voice_state=voice_state,
                            chunks=flush_chunks,
                        )
                    )
                else:
                    if item is None:
                        return
                    if voice_state["ptt_active"]:
                        current_turn_chunks.append(item)
    finally:
        if pending_flush_task is not None:
            pending_flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pending_flush_task
        await push_to_talk.stop()
        await microphone.stop()


def _drain_pending_speech_items(
    speech_queue: asyncio.Queue[SpeechCue | str],
) -> tuple[int, bool]:
    """Drop queued low-priority speech cues when a higher-priority event arrives."""
    dropped = 0
    stream_complete_seen = False

    while True:
        try:
            item = speech_queue.get_nowait()
        except asyncio.QueueEmpty:
            return dropped, stream_complete_seen

        if item == SPEECH_STREAM_COMPLETE:
            stream_complete_seen = True
            continue

        dropped += 1


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
        input_sample_rate=runtime.settings.realtime_input_sample_rate,
        output_sample_rate=runtime.settings.realtime_output_sample_rate,
    )

    try:
        await client.connect()
    except Exception:
        logger.exception("failed to connect realtime voice session")
        await _drain_bridge_queues(commentary_queue, speech_queue)
        return

    output_dir = runtime.settings.debug_realtime_dir / runtime.session_id
    output_dir.mkdir(parents=True, exist_ok=True)
    voice_state = _build_voice_state()
    pending_commentary: CommentaryResult | None = None
    raw_sink_task = asyncio.create_task(_realtime_raw_event_sink(client, output_dir=output_dir))
    text_sink_task = asyncio.create_task(_realtime_text_sink(client, output_dir=output_dir))
    transcript_sink_task = asyncio.create_task(_realtime_user_transcript_sink(client, output_dir=output_dir))
    audio_sink_task = asyncio.create_task(
        _realtime_audio_sink(client, runtime=runtime, voice_state=voice_state)
    )
    user_input_task = asyncio.create_task(
        _realtime_user_input_worker(client, runtime=runtime, voice_state=voice_state)
    )

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
            pending_tasks[asyncio.create_task(client.response_event_queue.get())] = "response"

            done, pending = await asyncio.wait(
                pending_tasks.keys(),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()

            commentary_sent_this_round = False
            ordered_done = sorted(
                done,
                key=lambda current: (
                    0
                    if pending_tasks[current] == "response"
                    else 1 if pending_tasks[current] == "commentary" else 2
                ),
            )

            for task in ordered_done:
                source = pending_tasks[task]
                item = task.result()

                if source == "response":
                    if item is None:
                        continue

                    logger.info(
                        "realtime response event response_id=%s source=%s phase=%s",
                        item.response_id,
                        item.source,
                        item.phase,
                    )

                    if item.phase == "created" and item.source == "user_speech":
                        voice_state["awaiting_user_response"] = True
                        continue

                    if item.phase == "done":
                        if item.source == "user_speech":
                            voice_state["awaiting_user_response"] = False
                        if pending_commentary is not None and not _user_priority_active(voice_state):
                            commentary_item = pending_commentary
                            pending_commentary = None
                            await client.send_event_text(build_realtime_text_from_commentary(commentary_item))
                            logger.info(
                                "realtime bridge sent deferred commentary request_id=%s",
                                commentary_item.request_id,
                            )
                    continue

                if source == "commentary":
                    if item == COMMENTARY_STREAM_COMPLETE:
                        commentary_open = False
                        continue
                    if runtime.settings.realtime_emit_commentary:
                        if _user_priority_active(voice_state):
                            replaced_request_id = pending_commentary.request_id if pending_commentary else None
                            pending_commentary = item
                            logger.info(
                                "realtime bridge deferred commentary request_id=%s reason=user_priority replaced_request_id=%s",
                                item.request_id,
                                replaced_request_id,
                            )
                            continue
                        if client.is_response_active():
                            replaced_request_id = pending_commentary.request_id if pending_commentary else None
                            pending_commentary = item
                            logger.info(
                                "realtime bridge deferred commentary request_id=%s reason=response_active source=%s replaced_request_id=%s",
                                item.request_id,
                                client.current_response_source,
                                replaced_request_id,
                            )
                            continue
                        dropped_count = 0
                        if runtime.settings.realtime_drop_speech_cues_on_commentary:
                            dropped_count, stream_complete_seen = _drain_pending_speech_items(speech_queue)
                            if stream_complete_seen:
                                speech_open = False
                            if dropped_count:
                                logger.info(
                                    "realtime bridge dropped queued speech cues count=%s reason=commentary_priority",
                                    dropped_count,
                                )
                        await client.send_event_text(build_realtime_text_from_commentary(item))
                        commentary_sent_this_round = True
                        logger.info(
                            "realtime bridge sent commentary request_id=%s dropped_speech_cues=%s",
                            item.request_id,
                            dropped_count,
                        )
                else:
                    if item == SPEECH_STREAM_COMPLETE:
                        speech_open = False
                        continue
                    if runtime.settings.realtime_emit_speech_cues:
                        if _user_priority_active(voice_state):
                            logger.info(
                                "realtime bridge dropped speech cue cue_id=%s reason=user_priority",
                                item.cue_id,
                            )
                            continue
                        if pending_commentary is not None:
                            logger.info(
                                "realtime bridge dropped speech cue cue_id=%s reason=commentary_deferred",
                                item.cue_id,
                            )
                            continue
                        if client.is_response_active():
                            logger.info(
                                "realtime bridge dropped speech cue cue_id=%s reason=response_active source=%s",
                                item.cue_id,
                                client.current_response_source,
                            )
                            continue
                        if commentary_sent_this_round:
                            logger.info(
                                "realtime bridge dropped speech cue cue_id=%s reason=commentary_same_round",
                                item.cue_id,
                            )
                            continue
                        if runtime.settings.realtime_drop_speech_cues_on_commentary and commentary_open and not commentary_queue.empty():
                            logger.info(
                                "realtime bridge dropped speech cue cue_id=%s reason=commentary_pending",
                                item.cue_id,
                            )
                            continue
                        await client.send_event_text(build_realtime_text_from_speech_cue(item))
                        logger.info(
                            "realtime bridge sent speech cue cue_id=%s",
                            item.cue_id,
                        )
    finally:
        await client.close()
        await raw_sink_task
        await text_sink_task
        await transcript_sink_task
        await audio_sink_task
        await user_input_task
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


async def _realtime_user_transcript_sink(
    client: XaiRealtimeClient,
    *,
    output_dir: Path,
) -> None:
    """Persist finalized user speech transcripts from the realtime session."""
    turn_index = 0
    while True:
        transcript = await client.user_transcript_queue.get()
        if transcript is None:
            return

        output_path = output_dir / f"user_{turn_index:03d}.txt"
        output_path.write_text(transcript, encoding="utf-8")
        logger.info("realtime user turn=%s text=%s output=%s", turn_index, transcript, output_path)
        turn_index += 1


async def _realtime_audio_sink(
    client: XaiRealtimeClient,
    *,
    runtime: SessionRuntime,
    voice_state: dict[str, bool],
) -> None:
    """Play assistant audio chunks locally so latency is audible."""
    if not runtime.settings.realtime_play_audio:
        logger.info("realtime audio playback disabled")
        await _drain_audio_queue(client)
        return

    try:
        import sounddevice as sd
        from sg_coach.realtime.user_voice import normalize_audio_device
    except ImportError:
        logger.warning(
            "sounddevice or realtime user voice helpers are not installed; audio will be drained but not played"
        )
        await _drain_audio_queue(client)
        return

    device_setting = normalize_audio_device(runtime.settings.realtime_audio_device)

    try:
        stream = sd.RawOutputStream(
            samplerate=runtime.settings.realtime_output_sample_rate,
            channels=1,
            dtype="int16",
            device=device_setting,
        )
        stream.start()
    except Exception:
        logger.exception(
            "failed to open realtime audio output device=%s sample_rate=%s",
            device_setting if device_setting is not None else "default",
            runtime.settings.realtime_output_sample_rate,
        )
        await _drain_audio_queue(client)
        return

    logger.info(
        "realtime audio playback started sample_rate=%s device=%s",
        runtime.settings.realtime_output_sample_rate,
        device_setting if device_setting is not None else "default",
    )

    try:
        while True:
            chunk = await client.audio_queue.get()
            if chunk is None:
                return
            if not chunk:
                continue
            if _assistant_audio_suppressed(voice_state):
                continue
            await asyncio.to_thread(stream.write, chunk)
    finally:
        with contextlib.suppress(Exception):
            stream.stop()
        with contextlib.suppress(Exception):
            stream.close()


async def _drain_audio_queue(client: XaiRealtimeClient) -> None:
    """Drain audio events when playback is disabled/unavailable."""
    while True:
        chunk = await client.audio_queue.get()
        if chunk is None:
            return


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
