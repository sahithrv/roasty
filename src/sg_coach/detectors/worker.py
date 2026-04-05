from __future__ import annotations

import asyncio
from collections.abc import Sequence

from sg_coach.detectors.base import Detector
from sg_coach.orchestrator.session import SessionRuntime
from sg_coach.orchestrator.topics import SIGNAL_DETECTOR
from sg_coach.shared.events import FramePacket
from sg_coach.shared.logging import get_logger
from sg_coach.shared.streaming import FRAME_STREAM_COMPLETE, SIGNAL_STREAM_COMPLETE


logger = get_logger(__name__)


async def detector_worker(
    runtime: SessionRuntime,
    frame_queue: asyncio.Queue[FramePacket | str],
    *,
    detector: Detector,
) -> None:
    """Consume frames, run one detector, and publish resulting signals.

    Why this exists:
    - detectors should become pluggable modules
    - the pipeline should not keep detector logic inline inside the demo wiring
    - this worker is the bridge between frame topics and signal topics

    Real detectors are sparse. Most frames will emit zero signals, so this
    worker cannot assume a fixed one-frame -> one-signal relationship.

    The upstream producer closes the frame stream with a sentinel. This worker
    forwards a corresponding sentinel onto the signal topic when it exits.
    """
    while True:
        item = await frame_queue.get()
        if item == FRAME_STREAM_COMPLETE:
            await runtime.publish(SIGNAL_DETECTOR, SIGNAL_STREAM_COMPLETE)
            logger.info("detector worker stream complete detector=%s", detector.name)
            return

        frame = item
        signals = await detector.detect(frame)

        logger.info(
            "detector worker ran detector=%s frame_id=%s emitted_signals=%s",
            detector.name,
            frame.frame_id,
            len(signals),
        )

        await runtime.publish_many(SIGNAL_DETECTOR, signals)


async def detector_fleet_worker(
    runtime: SessionRuntime,
    frame_queue: asyncio.Queue[FramePacket | str],
    *,
    detectors: Sequence[Detector],
) -> None:
    """Run multiple detectors against each frame and publish a merged signal stream.

    Why this exists:
    - multiple independent detector workers create tricky end-of-stream races
    - we still want detectors to run concurrently on the same frame
    - one fleet worker gives us one clean signal-stream sentinel

    This keeps the pipeline simple:
    one frame in -> detectors run concurrently -> zero or more signals out.
    """
    while True:
        item = await frame_queue.get()
        if item == FRAME_STREAM_COMPLETE:
            await runtime.publish(SIGNAL_DETECTOR, SIGNAL_STREAM_COMPLETE)
            logger.info(
                "detector fleet stream complete detectors=%s",
                ",".join(detector.name for detector in detectors),
            )
            return

        frame = item
        results = await asyncio.gather(*(detector.detect(frame) for detector in detectors))

        merged_signals = []
        for detector, signals in zip(detectors, results, strict=False):
            logger.info(
                "detector fleet ran detector=%s frame_id=%s emitted_signals=%s",
                detector.name,
                frame.frame_id,
                len(signals),
            )
            merged_signals.extend(signals)

        await runtime.publish_many(SIGNAL_DETECTOR, merged_signals)
