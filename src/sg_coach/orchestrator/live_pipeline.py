from __future__ import annotations

import asyncio

from sg_coach.capture.dxcam_backend import DxcamFrameSource
from sg_coach.detectors.demo import DemoCycleDetector
from sg_coach.detectors.worker import detector_worker
from sg_coach.fusion.demo import DemoPassthroughFuser
from sg_coach.fusion.worker import fusion_worker
from sg_coach.memory.worker import memory_snapshot_sink, memory_worker
from sg_coach.orchestrator.session import SessionRuntime
from sg_coach.orchestrator.topics import EVENT_GAME, FRAME_RAW, MEMORY_UPDATED, SIGNAL_DETECTOR
from sg_coach.shared.events import GameEvent
from sg_coach.shared.logging import configure_logging, get_logger
from sg_coach.shared.settings import load_settings
from sg_coach.shared.streaming import EVENT_STREAM_COMPLETE, FRAME_STREAM_COMPLETE


logger = get_logger(__name__)


async def live_frame_producer(
    runtime: SessionRuntime,
    source: DxcamFrameSource,
    *,
    frame_count: int,
) -> None:
    """Publish a finite number of real captured frames into the runtime bus."""
    published = 0
    async for frame in source.frames():
        published += 1
        logger.info(
            "live producer published frame %s of %s id=%s size=%sx%s",
            published,
            frame_count,
            frame.frame_id,
            frame.width,
            frame.height,
        )
        await runtime.publish(FRAME_RAW, frame)
        if published >= frame_count:
            break
    await runtime.publish(FRAME_RAW, FRAME_STREAM_COMPLETE)


async def event_sink(
    event_queue: asyncio.Queue[GameEvent | str],
    *,
    sink_name: str,
) -> None:
    """Consume final game events for live-pipeline inspection."""
    while True:
        event = await event_queue.get()
        if event == EVENT_STREAM_COMPLETE:
            logger.info("%s stream complete", sink_name)
            return
        logger.info(
            "%s received event type=%s session_id=%s source_signals=%s",
            sink_name,
            event.event_type,
            event.session_id,
            len(event.source_signal_ids),
        )


async def run_live_pipeline_demo(*, frame_count: int = 10) -> None:
    """Run the modular pipeline using real dxcam frames and fake detector logic.

    This proves one specific milestone:
    - real screen frames are entering the system
    - the detector worker is consuming them
    - the fusion worker is producing events
    - memory is updating downstream

    The detector and fusion are still synthetic on purpose. The goal is to
    validate end-to-end plumbing before real CV logic is introduced.
    """
    settings = load_settings()
    runtime = SessionRuntime.create(settings)

    source = DxcamFrameSource(settings=settings)
    detector = DemoCycleDetector()
    fuser = DemoPassthroughFuser()

    frame_queue = runtime.subscribe(FRAME_RAW)
    signal_queue = runtime.subscribe(SIGNAL_DETECTOR)
    audit_queue = runtime.subscribe(EVENT_GAME)
    memory_event_queue = runtime.subscribe(EVENT_GAME)
    memory_snapshot_queue = runtime.subscribe(MEMORY_UPDATED)

    logger.info(
        "live pipeline starting session_id=%s backend=%s monitor_id=%s target_fps=%s frame_subscribers=%s signal_subscribers=%s event_subscribers=%s",
        runtime.session_id,
        settings.capture_backend,
        settings.capture_monitor_id,
        settings.target_fps,
        runtime.bus.subscriber_count(FRAME_RAW),
        runtime.bus.subscriber_count(SIGNAL_DETECTOR),
        runtime.bus.subscriber_count(EVENT_GAME),
    )

    await asyncio.gather(
        live_frame_producer(runtime, source, frame_count=frame_count),
        detector_worker(runtime, frame_queue, detector=detector),
        fusion_worker(runtime, signal_queue, fuser=fuser),
        event_sink(audit_queue, sink_name="live_audit_sink"),
        memory_worker(runtime.bus, memory_event_queue, store=runtime.memory_store),
        memory_snapshot_sink(memory_snapshot_queue),
    )

    logger.info("live pipeline finished session_id=%s", runtime.session_id)


def main() -> None:
    configure_logging("INFO")
    asyncio.run(run_live_pipeline_demo())


if __name__ == "__main__":
    main()
