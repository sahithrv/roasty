from __future__ import annotations

import asyncio
import json
from pathlib import Path

from sg_coach.capture.replay_buffer import ReplayFrameBuffer
from sg_coach.grok.payloads import build_grok_chat_payload
from sg_coach.memory.store import MemorySnapshot
from sg_coach.orchestrator.session import SessionRuntime
from sg_coach.orchestrator.topics import COMMENTARY_REQUEST
from sg_coach.shared.events import CommentaryRequest, GameEvent
from sg_coach.shared.logging import get_logger
from sg_coach.shared.streaming import COMMENTARY_STREAM_COMPLETE, EVENT_STREAM_COMPLETE


logger = get_logger(__name__)


def select_context_frame_paths(frame_paths: list[str], *, max_frames: int) -> list[str]:
    """Pick a small, evenly spaced subset of replay frames.

    The replay buffer may export a lot of frames, but sending every one of them
    to a multimodal model is unnecessary and expensive. This helper keeps the
    first milestone simple: select a few evenly spaced frames that describe the
    lead-up to the event.
    """
    if max_frames <= 0 or not frame_paths:
        return []
    if len(frame_paths) <= max_frames:
        return frame_paths
    if max_frames == 1:
        return [frame_paths[-1]]

    indices = []
    last_index = len(frame_paths) - 1
    for slot in range(max_frames):
        index = round(slot * last_index / (max_frames - 1))
        indices.append(index)

    # Remove any duplicates from rounding while preserving order.
    selected_indices = list(dict.fromkeys(indices))
    return [frame_paths[index] for index in selected_indices]


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
    output_root: Path,
) -> None:
    """Turn selected game events into model-ready `CommentaryRequest` objects.

    End-to-end flow inside this worker:
    1. wait for a final `GameEvent`
    2. ignore event types that are not worth commenting on yet
    3. export replay-buffer context frames for the interesting event
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

        exported_paths = replay_buffer.export_recent_frames(
            event=event,
            output_root=output_root,
            seconds=replay_buffer.buffer_seconds,
        )
        selected_paths = select_context_frame_paths(
            exported_paths,
            max_frames=runtime.settings.commentary_context_frame_count,
        )

        # Yield once so the memory worker can usually commit the same event
        # before we read session state. The snapshot helper below still handles
        # the fallback case if the workers race.
        await asyncio.sleep(0)
        snapshot = build_commentary_snapshot(
            runtime,
            event,
            recent_limit=runtime.settings.commentary_recent_event_limit,
        )

        enriched_metadata = {
            **event.metadata,
            "context_frame_paths": selected_paths,
            "replay_export_frame_count": len(exported_paths),
            "replay_buffer_seconds": replay_buffer.buffer_seconds,
        }
        enriched_event = event.model_copy(
            update={
                "metadata": enriched_metadata,
                "frame_path": selected_paths[-1] if selected_paths else event.frame_path,
            }
        )

        request = CommentaryRequest(
            persona=runtime.settings.default_persona,
            latest_event=enriched_event,
            recent_events=snapshot.recent_events,
            counters=snapshot.counters,
            callback_candidates=snapshot.callback_candidates,
            memory_summary=snapshot.summary_text,
            include_frame=bool(selected_paths),
            frame_path=selected_paths[-1] if selected_paths else None,
            context_frame_paths=selected_paths,
        )
        await runtime.publish(COMMENTARY_REQUEST, request)
        logger.info(
            "commentary request built event_type=%s context_frames=%s recent_events=%s callbacks=%s",
            request.latest_event.event_type,
            len(request.context_frame_paths),
            len(request.recent_events),
            len(request.callback_candidates),
        )


async def commentary_request_sink(
    request_queue: asyncio.Queue[CommentaryRequest | str],
    *,
    runtime: SessionRuntime,
) -> None:
    """Debug sink that shows exactly what would be sent to Grok.

    This sink is intentionally transparent:
    - it logs a concise summary
    - it writes the full model payload to disk as JSON
    - it does not call the network
    """
    while True:
        item = await request_queue.get()
        if item == COMMENTARY_STREAM_COMPLETE:
            logger.info("commentary request sink complete")
            return

        request = item
        payload = build_grok_chat_payload(request, model=runtime.settings.grok_model)
        dump_path = _write_commentary_debug_payload(
            request=request,
            payload=payload,
            output_root=runtime.settings.debug_commentary_dir,
            session_id=runtime.session_id,
        )
        logger.info(
            "commentary payload prepared request_id=%s event_type=%s context_frames=%s dump=%s",
            request.request_id,
            request.latest_event.event_type,
            len(request.context_frame_paths),
            dump_path,
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
