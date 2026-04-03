from __future__ import annotations

import asyncio

from sg_coach.orchestrator.bus import EventBus
from sg_coach.orchestrator.topics import EVENT_GAME, FRAME_RAW, SIGNAL_DETECTOR
from sg_coach.shared.events import DetectionSignal, FramePacket, GameEvent, new_id
from sg_coach.shared.logging import configure_logging, get_logger


logger = get_logger(__name__)


def build_demo_frame(*, game: str, monitor_id: int, width: int, height: int) -> FramePacket:
    """Create a fake frame packet for pipeline demonstration.

    The real system will eventually populate ``image_bgr`` with an actual image
    array from a capture backend. For now we keep it ``None`` because the goal is
    to learn the event flow before adding OpenCV and Windows capture complexity.
    """
    return FramePacket(
        game=game,
        monitor_id=monitor_id,
        width=width,
        height=height,
        image_bgr=None,
    )


async def frame_producer(bus: EventBus, *, frame_count: int) -> None:
    """Publish a short burst of fake frames into the bus."""
    for index in range(frame_count):
        frame = build_demo_frame(
            game="valorant",
            monitor_id=0,
            width=1920,
            height=1080,
        )
        logger.info("producer published frame %s of %s with id=%s", index + 1, frame_count, frame.frame_id)
        await bus.publish(FRAME_RAW, frame)
        await asyncio.sleep(0.10)


async def fake_detector_worker(
    bus: EventBus,
    frame_queue: asyncio.Queue[FramePacket],
    *,
    frame_count: int,
) -> None:
    """Convert frames into fake detector signals.

    This stands in for the future detector layer. Right now it does not inspect
    pixels. It simply consumes frames and emits plausible-looking signals so the
    rest of the pipeline can be developed in isolation.
    """
    for index in range(frame_count):
        frame = await frame_queue.get()

        if index % 2 == 0:
            signal_type = "kill"
        elif index % 3 == 0:
            signal_type = "round_end"
        else:
            signal_type = "death"

        signal = DetectionSignal(
            game=frame.game or "unknown",
            detector_name="demo_detector",
            signal_type=signal_type,
            confidence=0.90 if signal_type == "kill" else 0.82,
            frame_ref=frame.frame_id,
            tags=["demo", "synthetic"],
            metadata={"frame_index": index},
            dedupe_key=f"{signal_type}:{index}",
            cooldown_key=signal_type,
        )
        logger.info(
            "detector emitted signal type=%s confidence=%.2f frame_ref=%s",
            signal.signal_type,
            signal.confidence,
            signal.frame_ref,
        )
        await bus.publish(SIGNAL_DETECTOR, signal)


async def fake_fusion_worker(
    bus: EventBus,
    signal_queue: asyncio.Queue[DetectionSignal],
    *,
    signal_count: int,
    session_id: str,
) -> None:
    """Promote detector signals into higher-level game events.

    In the real app, this step will merge multiple weak signals, apply temporal
    rules, and emit only canonical events. For the demo, each signal becomes one
    event so the flow stays easy to inspect.
    """
    for _ in range(signal_count):
        signal = await signal_queue.get()

        event = GameEvent(
            session_id=session_id,
            game=signal.game,
            event_type=signal.signal_type,
            confidence=signal.confidence,
            tags=signal.tags,
            metadata={
                "source_detector": signal.detector_name,
                "frame_ref": signal.frame_ref,
            },
            source_signal_ids=[signal.signal_id],
            dedupe_key=signal.dedupe_key,
            cooldown_key=signal.cooldown_key,
        )
        logger.info(
            "fusion emitted event type=%s confidence=%.2f event_id=%s",
            event.event_type,
            event.confidence,
            event.event_id,
        )
        await bus.publish(EVENT_GAME, event)


async def event_sink(
    event_queue: asyncio.Queue[GameEvent],
    *,
    sink_name: str,
    event_count: int,
) -> None:
    """Consume final game events.

    We run two sinks in the demo to prove that this bus broadcasts events to
    multiple subscribers instead of letting them steal work from one another.
    """
    for _ in range(event_count):
        event = await event_queue.get()
        logger.info(
            "%s received event type=%s session_id=%s source_signals=%s",
            sink_name,
            event.event_type,
            event.session_id,
            len(event.source_signal_ids),
        )


async def run_demo(*, frame_count: int = 4) -> None:
    """Run a finite, fake end-to-end event pipeline.

    This is a learning scaffold, not the final runtime. It exists so we can
    validate the bus, topic naming, and event contracts before adding real
    capture backends and detection logic.
    """
    bus = EventBus()
    session_id = new_id("session")

    frame_queue = bus.subscribe(FRAME_RAW)
    signal_queue = bus.subscribe(SIGNAL_DETECTOR)
    event_queue = bus.subscribe(EVENT_GAME)
    audit_queue = bus.subscribe(EVENT_GAME)

    logger.info(
        "demo pipeline starting session_id=%s frame_subscribers=%s signal_subscribers=%s event_subscribers=%s",
        session_id,
        bus.subscriber_count(FRAME_RAW),
        bus.subscriber_count(SIGNAL_DETECTOR),
        bus.subscriber_count(EVENT_GAME),
    )

    await asyncio.gather(
        fake_detector_worker(bus, frame_queue, frame_count=frame_count),
        fake_fusion_worker(bus, signal_queue, signal_count=frame_count, session_id=session_id),
        event_sink(event_queue, sink_name="primary_sink", event_count=frame_count),
        event_sink(audit_queue, sink_name="audit_sink", event_count=frame_count),
        frame_producer(bus, frame_count=frame_count),
    )

    logger.info("demo pipeline finished session_id=%s", session_id)


def main() -> None:
    """Run the demo with basic logging enabled."""
    configure_logging("INFO")
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
