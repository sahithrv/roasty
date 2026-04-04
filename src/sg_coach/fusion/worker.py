from __future__ import annotations

import asyncio

from sg_coach.fusion.base import SignalFuser
from sg_coach.orchestrator.session import SessionRuntime
from sg_coach.orchestrator.topics import EVENT_GAME
from sg_coach.shared.events import DetectionSignal
from sg_coach.shared.logging import get_logger
from sg_coach.shared.streaming import EVENT_STREAM_COMPLETE, SIGNAL_STREAM_COMPLETE


logger = get_logger(__name__)


async def fusion_worker(
    runtime: SessionRuntime,
    signal_queue: asyncio.Queue[DetectionSignal | str],
    *,
    fuser: SignalFuser,
) -> None:
    """Consume detector signals, fuse them, and publish resulting game events.

    Sparse detectors mean this worker must also be stream-based instead of
    assuming one signal per input frame.
    """
    while True:
        item = await signal_queue.get()
        if item == SIGNAL_STREAM_COMPLETE:
            await runtime.publish(EVENT_GAME, EVENT_STREAM_COMPLETE)
            logger.info("fusion worker stream complete fuser=%s", fuser.name)
            return

        detection_signal = item
        events = await fuser.consume(detection_signal, session_id=runtime.session_id)
        logger.info(
            "fusion worker ran fuser=%s signal_id=%s emitted_events=%s",
            fuser.name,
            detection_signal.signal_id,
            len(events),
        )
        await runtime.publish_many(EVENT_GAME, events)
