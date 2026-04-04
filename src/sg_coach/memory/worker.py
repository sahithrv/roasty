from __future__ import annotations

import asyncio

from sg_coach.memory.store import MemorySnapshot, SessionMemoryStore
from sg_coach.orchestrator.bus import EventBus
from sg_coach.orchestrator.topics import MEMORY_UPDATED
from sg_coach.shared.events import GameEvent
from sg_coach.shared.logging import get_logger
from sg_coach.shared.streaming import EVENT_STREAM_COMPLETE, MEMORY_STREAM_COMPLETE


logger = get_logger(__name__)


def format_snapshot_line(snapshot: MemorySnapshot) -> str:
    """Return a compact summary line for demo logging.

    TODO for you:
    Expand this formatter so it also shows:
    - callback candidate families
    - recurring pattern counts
    - whether the summary should be refreshed
    """
    latest = "none" if snapshot.latest_event is None else snapshot.latest_event.event_type
    recent = ",".join(event.event_type for event in snapshot.recent_events) or "none"
    callback_families = ",".join(
        candidate["pattern_family"] for candidate in snapshot.callback_candidates
    ) or "none"
    recurring = ",".join(
        f"{pattern}={count}" for pattern, count in sorted(snapshot.recurring_patterns.items())
    ) or "none"
    return (
        f"total_events={snapshot.total_events} "
        f"latest={latest} "
        f"recent=[{recent}] "
        f"counters={snapshot.counters} "
        f"callback_families=[{callback_families}] "
        f"recurring=[{recurring}] "
        f"refresh_req={snapshot.summary_refresh_needed}"
    )


async def memory_worker(
    bus: EventBus,
    event_queue: asyncio.Queue[GameEvent | str],
    *,
    store: SessionMemoryStore,
) -> None:
    """Consume fused game events and update session memory.

    This is the first real stateful consumer in the project. It shows the
    intended pattern for later workers:
    - subscribe to a bus topic
    - consume typed messages
    - update internal domain state
    - publish a derived message back onto the bus
    """
    threshold = 5

    while True:
        item = await event_queue.get()
        if item == EVENT_STREAM_COMPLETE:
            await bus.publish(MEMORY_UPDATED, MEMORY_STREAM_COMPLETE)
            return

        before_candidates = {
            candidate["pattern_family"]
            for candidate in store.callback_candidates(limit=100)
        }
        before_patterns = store.recurring_patterns()

        event = item
        store.store_event(event)
        snapshot = store.build_snapshot()

        logger.info(
            "memory stored event type=%s total_events=%s summary_refresh_needed=%s",
            event.event_type,
            snapshot.total_events,
            snapshot.summary_refresh_needed,
        )

        # TODO for you:
        # Add one more publish trigger here based on richer memory semantics.
        # Good candidates:
        # - a notable event of a specific family appears
        # - a counter transitions from N to N+1 for important event types
        # - a callback candidate count increases meaningfully
        after_candidates = {
            candidate["pattern_family"]
            for candidate in snapshot.callback_candidates
        }
        new_callback_candidate_appeared = not after_candidates.issubset(before_candidates)

        threshold_crossed = any(
            count >= threshold and before_patterns.get(pattern, 0) < threshold
            for pattern, count in snapshot.recurring_patterns.items()
        )

        should_publish = (
            snapshot.summary_refresh_needed
            or new_callback_candidate_appeared
            or threshold_crossed
        )

        if should_publish:
            await bus.publish(MEMORY_UPDATED, snapshot)


async def memory_snapshot_sink(
    snapshot_queue: asyncio.Queue[MemorySnapshot | str],
) -> None:
    """Consume memory snapshots and log them for inspection.

    The demo ends this stream by publishing a sentinel message after the memory
    worker finishes, so the sink does not need to guess how many updates will be
    emitted.
    """
    while True:
        snapshot = await snapshot_queue.get()
        if snapshot == MEMORY_STREAM_COMPLETE:
            logger.info("memory snapshot stream complete")
            return
        logger.info("memory snapshot %s", format_snapshot_line(snapshot))
