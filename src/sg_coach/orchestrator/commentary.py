from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import cv2

from sg_coach.capture.replay_buffer import ReplayFrame, ReplayFrameBuffer
from sg_coach.grok.client import GrokChatClient
from sg_coach.grok.payloads import (
    build_grok_chat_payload,
    parse_structured_commentary_output,
    sanitize_grok_chat_payload_for_debug,
)
from sg_coach.memory.store import MemorySnapshot
from sg_coach.orchestrator.session import SessionRuntime
from sg_coach.orchestrator.topics import COMMENTARY_READY, COMMENTARY_REQUEST
from sg_coach.shared.events import CommentaryRequest, CommentaryResult, GameEvent
from sg_coach.shared.logging import get_logger
from sg_coach.shared.streaming import COMMENTARY_STREAM_COMPLETE, EVENT_STREAM_COMPLETE


logger = get_logger(__name__)


def select_context_frames(frames: list[ReplayFrame], *, max_frames: int) -> list[ReplayFrame]:
    """Pick a small, evenly spaced subset of replay frames.

    The replay buffer may contain many frames, but the model only needs a few
    to understand the lead-up. This helper keeps the oldest/newest spread while
    staying cheap.
    """
    if max_frames <= 0 or not frames:
        return []
    if len(frames) <= max_frames:
        return frames
    if max_frames == 1:
        return [frames[-1]]

    indices = []
    last_index = len(frames) - 1
    for slot in range(max_frames):
        index = round(slot * last_index / (max_frames - 1))
        indices.append(index)

    # Remove any duplicates from rounding while preserving order.
    selected_indices = list(dict.fromkeys(indices))
    return [frames[index] for index in selected_indices]


def replay_frame_to_data_url(replay_frame: ReplayFrame) -> str:
    """Encode one replay frame directly into a JPEG data URL for Grok."""
    ok, encoded = cv2.imencode(".jpg", replay_frame.image_bgr)
    if not ok:
        raise RuntimeError(f"Failed to encode replay frame {replay_frame.frame_id} as JPEG")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def save_commentary_preview_frame(
    *,
    replay_frame: ReplayFrame | None,
    output_root: Path,
    session_id: str,
    request_id: str,
) -> str | None:
    """Persist only the main trigger frame for local inspection."""
    if replay_frame is None:
        return None

    output_dir = output_root / session_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{request_id}_frame.jpg"
    if cv2.imwrite(str(output_path), replay_frame.image_bgr):
        return str(output_path)
    return None


def build_commentary_snapshot(
    runtime: SessionRuntime,
    event: GameEvent,
    *,
    recent_limit: int,
) -> MemorySnapshot:
    """Return a memory snapshot suitable for commentary generation.

    The memory worker and commentary worker both subscribe to `event.game`, so
    they can process the same event concurrently. In practice the memory worker
    is usually ahead, but this helper makes the commentary path robust by
    falling back to a synthetic snapshot if the shared memory store has not yet
    incorporated the current event.
    """
    snapshot = runtime.memory_snapshot(recent_limit=recent_limit)
    if snapshot.latest_event is not None and snapshot.latest_event.event_id == event.event_id:
        return snapshot

    counters = dict(snapshot.counters)
    counters[event.event_type] = counters.get(event.event_type, 0) + 1

    recent_events = [*snapshot.recent_events, event][-recent_limit:]
    notable_events = [*snapshot.notable_events, event][-recent_limit:]

    return MemorySnapshot(
        session_id=runtime.session_id,
        total_events=snapshot.total_events + 1,
        counters=counters,
        latest_event=event,
        recent_events=recent_events,
        notable_events=notable_events,
        recurring_patterns=snapshot.recurring_patterns,
        callback_candidates=snapshot.callback_candidates,
        summary_refresh_needed=snapshot.summary_refresh_needed,
        summary_text=snapshot.summary_text,
    )


async def commentary_request_worker(
    runtime: SessionRuntime,
    event_queue: asyncio.Queue[GameEvent | str],
    *,
    replay_buffer: ReplayFrameBuffer,
) -> None:
    """Turn selected game events into model-ready `CommentaryRequest` objects.

    End-to-end flow inside this worker:
    1. wait for a final `GameEvent`
    2. ignore event types that are not worth commenting on yet
    3. select a few replay-buffer context frames for the interesting event
    4. gather the latest session memory snapshot
    5. build a `CommentaryRequest`
    6. publish that request onto the bus for the Grok-facing layer
    """
    while True:
        item = await event_queue.get()
        if item == EVENT_STREAM_COMPLETE:
            await runtime.publish(COMMENTARY_REQUEST, COMMENTARY_STREAM_COMPLETE)
            logger.info("commentary request stream complete")
            return

        event = item
        if event.event_type != "wasted":
            continue

        replay_frames = replay_buffer.recent_frames(seconds=replay_buffer.buffer_seconds)
        selected_frames = select_context_frames(
            replay_frames,
            max_frames=runtime.settings.commentary_context_frame_count,
        )
        selected_frame_data_urls = [
            replay_frame_to_data_url(replay_frame) for replay_frame in selected_frames
        ]

        # Yield once so the memory worker can usually commit the same event
        # before we read session state. The snapshot helper below still handles
        # the fallback case if the workers race.
        await asyncio.sleep(0)
        snapshot = build_commentary_snapshot(
            runtime,
            event,
            recent_limit=runtime.settings.commentary_recent_event_limit,
        )

        request = CommentaryRequest(
            persona=runtime.settings.default_persona,
            game_key=runtime.game_key,
            game_profile=runtime.game_profile,
            latest_event=event,
            recent_events=snapshot.recent_events,
            counters=snapshot.counters,
            callback_candidates=snapshot.callback_candidates,
            memory_summary=snapshot.summary_text,
            include_frame=bool(selected_frames),
            context_frame_data_urls=selected_frame_data_urls,
        )
        preview_frame_path = save_commentary_preview_frame(
            replay_frame=selected_frames[-1] if selected_frames else None,
            output_root=runtime.settings.debug_commentary_dir,
            session_id=runtime.session_id,
            request_id=request.request_id,
        )
        enriched_metadata = {
            **event.metadata,
            "context_frame_count": len(selected_frames),
            "replay_buffer_seconds": replay_buffer.buffer_seconds,
            "preview_frame_path": preview_frame_path,
        }
        request = request.model_copy(
            update={
                "latest_event": event.model_copy(
                    update={
                        "metadata": enriched_metadata,
                        "frame_path": preview_frame_path or event.frame_path,
                    }
                ),
                "frame_path": preview_frame_path,
                "context_frame_paths": [preview_frame_path] if preview_frame_path else [],
            }
        )
        await runtime.publish(COMMENTARY_REQUEST, request)
        logger.info(
            "commentary request built event_type=%s context_frames=%s recent_events=%s callbacks=%s",
            request.latest_event.event_type,
            len(request.context_frame_data_urls),
            len(request.recent_events),
            len(request.callback_candidates),
        )


async def commentary_model_worker(
    request_queue: asyncio.Queue[CommentaryRequest | str],
    *,
    runtime: SessionRuntime,
) -> None:
    """Consume `CommentaryRequest`s, call Grok, and publish `CommentaryResult`.

    The request JSON is always written to disk first so the exact prompt can be
    inspected even if the network call fails.
    """
    client = None
    if runtime.settings.commentary_enabled and runtime.settings.grok_api_key:
        client = GrokChatClient.from_settings(runtime.settings)

    while True:
        item = await request_queue.get()
        if item == COMMENTARY_STREAM_COMPLETE:
            await runtime.publish(COMMENTARY_READY, COMMENTARY_STREAM_COMPLETE)
            logger.info("commentary model worker complete")
            return

        request = item
        payload = build_grok_chat_payload(request, model=runtime.settings.grok_model)
        request_dump_path = _write_commentary_debug_payload(
            request=request,
            payload=sanitize_grok_chat_payload_for_debug(
                payload,
                context_frame_paths=request.context_frame_paths,
            ),
            output_root=runtime.settings.debug_commentary_dir,
            session_id=runtime.session_id,
        )

        if not runtime.settings.commentary_enabled:
            logger.info("commentary disabled; request dumped only path=%s", request_dump_path)
            continue

        if client is None:
            logger.warning("commentary client unavailable; request dumped only path=%s", request_dump_path)
            continue

        try:
            response_json = await asyncio.to_thread(client.create_chat_completion, payload)
            response_text = client.extract_text(response_json)
        except Exception as exc:
            error_dump_path = _write_commentary_error_dump(
                request=request,
                error_message=str(exc),
                output_root=runtime.settings.debug_commentary_dir,
                session_id=runtime.session_id,
            )
            logger.exception(
                "commentary model call failed request_id=%s request_dump=%s error_dump=%s",
                request.request_id,
                request_dump_path,
                error_dump_path,
            )
            continue

        visual_summary, coach_note = parse_structured_commentary_output(response_text)
        combined_text = visual_summary
        if coach_note:
            combined_text = f"{visual_summary}\nCoach: {coach_note}" if visual_summary else coach_note

        result = CommentaryResult(
            request_id=request.request_id,
            event_id=request.latest_event.event_id,
            model=runtime.settings.grok_model,
            text=combined_text,
            visual_summary=visual_summary,
            coach_note=coach_note,
            raw_response=response_json,
        )
        await runtime.publish(COMMENTARY_READY, result)
        logger.info(
            "commentary payload sent request_id=%s event_type=%s context_frames=%s request_dump=%s",
            request.request_id,
            request.latest_event.event_type,
            len(request.context_frame_paths),
            request_dump_path,
        )


async def commentary_result_sink(
    result_queue: asyncio.Queue[CommentaryResult | str],
    *,
    runtime: SessionRuntime,
) -> None:
    """Write actual model output to disk and log the generated commentary."""
    while True:
        item = await result_queue.get()
        if item == COMMENTARY_STREAM_COMPLETE:
            logger.info("commentary result sink complete")
            return

        result = item
        response_dump_path, text_dump_path = _write_commentary_result_dumps(
            result=result,
            output_root=runtime.settings.debug_commentary_dir,
            session_id=runtime.session_id,
        )
        logger.info(
            "commentary result ready request_id=%s model=%s summary=%s coach=%s response_dump=%s text_dump=%s",
            result.request_id,
            result.model,
            result.visual_summary,
            result.coach_note,
            response_dump_path,
            text_dump_path,
        )


def _write_commentary_debug_payload(
    *,
    request: CommentaryRequest,
    payload: dict,
    output_root: Path,
    session_id: str,
) -> Path:
    """Write the exact Grok-style payload to disk for inspection."""
    output_dir = output_root / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{request.request_id}.json"
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return output_path


def _write_commentary_result_dumps(
    *,
    result: CommentaryResult,
    output_root: Path,
    session_id: str,
) -> tuple[Path, Path]:
    """Write the raw response JSON and the plain commentary text to disk."""
    output_dir = output_root / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    response_dump_path = output_dir / f"{result.request_id}_response.json"
    response_dump_path.write_text(
        json.dumps(result.raw_response, indent=2, default=str),
        encoding="utf-8",
    )

    text_dump_path = output_dir / f"{result.request_id}_text.txt"
    text_dump_path.write_text(result.text, encoding="utf-8")
    return response_dump_path, text_dump_path


def _write_commentary_error_dump(
    *,
    request: CommentaryRequest,
    error_message: str,
    output_root: Path,
    session_id: str,
) -> Path:
    """Write API errors to disk so failures are inspectable after the run."""
    output_dir = output_root / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{request.request_id}_error.json"
    output_path.write_text(
        json.dumps(
            {
                "request_id": request.request_id,
                "event_id": request.latest_event.event_id,
                "error": error_message,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path
