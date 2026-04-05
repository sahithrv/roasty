from __future__ import annotations

import asyncio

from sg_coach.capture.dxcam_backend import DxcamFrameSource
from sg_coach.capture.replay_buffer import (
    ReplayFrameBuffer,
    replay_buffer_worker,
)
from sg_coach.detectors.gta.chaos import GtaChaosDetector
from sg_coach.detectors.gta.wasted import GtaWastedDetector
from sg_coach.detectors.worker import detector_fleet_worker
from sg_coach.fusion.demo import DemoPassthroughFuser
from sg_coach.fusion.worker import fusion_worker
from sg_coach.memory.worker import memory_snapshot_sink, memory_worker
from sg_coach.orchestrator.commentary import (
    commentary_model_worker,
    commentary_request_worker,
    commentary_result_sink,
)
from sg_coach.orchestrator.realtime import realtime_bridge_worker
from sg_coach.orchestrator.speech import speech_cue_sink, speech_cue_worker
from sg_coach.orchestrator.session import SessionRuntime
from sg_coach.orchestrator.topics import (
    COMMENTARY_READY,
    COMMENTARY_REQUEST,
    EVENT_GAME,
    FRAME_RAW,
    MEMORY_UPDATED,
    SIGNAL_DETECTOR,
    SPEECH_PLAY,
)
from sg_coach.shared.events import GameEvent
from sg_coach.shared.logging import configure_logging, get_logger
from sg_coach.shared.settings import load_settings
from sg_coach.shared.streaming import EVENT_STREAM_COMPLETE, FRAME_STREAM_COMPLETE


logger = get_logger(__name__)


async def live_frame_producer(
    runtime: SessionRuntime,
    source: DxcamFrameSource,
    *,
    frame_count: int | None,
) -> None:
    """Publish real GTA frames into the runtime bus.

    When `frame_count` is `None`, the producer runs continuously until the
    process is interrupted by the user.
    """
    published = 0
    async for frame in source.frames():
        published += 1
        logger.info(
            "gta live producer published frame %s of %s id=%s size=%sx%s",
            published,
            "continuous" if frame_count is None else frame_count,
            frame.frame_id,
            frame.width,
            frame.height,
        )
        await runtime.publish(FRAME_RAW, frame)
        if frame_count is not None and published >= frame_count:
            break
    await runtime.publish(FRAME_RAW, FRAME_STREAM_COMPLETE)


async def event_sink(
    event_queue: asyncio.Queue[GameEvent | str],
    *,
    sink_name: str,
) -> None:
    while True:
        item = await event_queue.get()
        if item == EVENT_STREAM_COMPLETE:
            logger.info("%s stream complete", sink_name)
            return

        event = item
        logger.info(
            "%s received event type=%s confidence=%.3f source_signals=%s",
            sink_name,
            event.event_type,
            event.confidence,
            len(event.source_signal_ids),
        )


async def run_gta_wasted_pipeline(*, frame_count: int | None = None) -> None:
    """Run the first GTA-specific live pipeline.

    This is now the first realistic multi-detector GTA path:
    - real live frames
    - a real heuristic `wasted` detector
    - a cheap ambient `chaos_spike` detector
    - passthrough fusion
    - replay buffer context export on `wasted`
    - commentary request generation from event + memory + replay context
    - cheap speech cues for non-Grok ambient events

    The pipeline is intentionally linear and explicit so it is easy to follow:

    FRAME_RAW
      -> replay_buffer_worker
      -> detector_fleet_worker
      -> SIGNAL_DETECTOR
      -> fusion_worker
      -> EVENT_GAME
      -> memory_worker
      -> commentary_request_worker
      -> speech_cue_worker
      -> COMMENTARY_REQUEST
      -> commentary_model_worker
      -> COMMENTARY_READY
      -> commentary_result_sink
      -> SPEECH_PLAY
      -> speech_cue_sink
      -> realtime_bridge_worker
    """
    settings = load_settings()
    runtime = SessionRuntime.create(settings)

    source = DxcamFrameSource(settings=settings, game="gta_like")
    wasted_detector = GtaWastedDetector(settings=settings)
    chaos_detector = GtaChaosDetector(settings=settings)
    fuser = DemoPassthroughFuser()
    replay_buffer = ReplayFrameBuffer.from_settings(settings)

    # One queue per subscriber is the important EventBus pattern in this repo.
    # The same published message is broadcast into each queue independently.
    frame_queue_detectors = runtime.subscribe(FRAME_RAW)
    frame_queue_replay = runtime.subscribe(FRAME_RAW)
    signal_queue = runtime.subscribe(SIGNAL_DETECTOR)
    audit_queue = runtime.subscribe(EVENT_GAME)
    memory_event_queue = runtime.subscribe(EVENT_GAME)
    commentary_event_queue = runtime.subscribe(EVENT_GAME)
    speech_event_queue = runtime.subscribe(EVENT_GAME)
    memory_snapshot_queue = runtime.subscribe(MEMORY_UPDATED)
    commentary_request_queue = runtime.subscribe(COMMENTARY_REQUEST)
    commentary_result_queue = runtime.subscribe(COMMENTARY_READY)
    realtime_commentary_queue = runtime.subscribe(COMMENTARY_READY)
    speech_queue = runtime.subscribe(SPEECH_PLAY)
    realtime_speech_queue = runtime.subscribe(SPEECH_PLAY)

    logger.info(
        "gta pipeline starting session_id=%s monitor_id=%s target_fps=%s frame_count=%s replay_seconds=%s wasted_template=%s wasted_threshold=%.3f chaos_threshold=%.3f commentary_frames=%s",
        runtime.session_id,
        settings.capture_monitor_id,
        settings.target_fps,
        "continuous" if frame_count is None else frame_count,
        settings.replay_buffer_seconds,
        settings.gta_wasted_template_path,
        settings.gta_wasted_match_threshold,
        settings.gta_chaos_score_threshold,
        settings.commentary_context_frame_count,
    )

    await asyncio.gather(
        # Capture and publish real monitor frames.
        live_frame_producer(runtime, source, frame_count=frame_count),
        # Maintain a rolling local replay window of recent frames.
        replay_buffer_worker(frame_queue_replay, replay_buffer=replay_buffer),
        # Run both GTA detectors concurrently on the same live frame stream.
        detector_fleet_worker(
            runtime,
            frame_queue_detectors,
            detectors=[wasted_detector, chaos_detector],
        ),
        # Convert detector signals into canonical `GameEvent` objects.
        fusion_worker(runtime, signal_queue, fuser=fuser),
        # Human-readable event log for debugging the pipeline.
        event_sink(audit_queue, sink_name="gta_audit_sink"),
        # Update session memory using final game events.
        memory_worker(runtime.bus, memory_event_queue, store=runtime.memory_store),
        memory_snapshot_sink(memory_snapshot_queue),
        # Build model-ready commentary requests from wasted events plus replay context.
        commentary_request_worker(
            runtime,
            commentary_event_queue,
            replay_buffer=replay_buffer,
        ),
        # Send the built request to Grok and publish the resulting commentary.
        commentary_model_worker(commentary_request_queue, runtime=runtime),
        # Dump the actual response JSON and final line of commentary to disk.
        commentary_result_sink(commentary_result_queue, runtime=runtime),
        # Build cheap text cues for non-Grok ambient events that a realtime layer
        # can comment on later without spending multimodal tokens.
        speech_cue_worker(runtime, speech_event_queue),
        speech_cue_sink(
            speech_queue,
            output_root=settings.debug_speech_dir,
            session_id=runtime.session_id,
        ),
        realtime_bridge_worker(
            runtime,
            commentary_queue=realtime_commentary_queue,
            speech_queue=realtime_speech_queue,
        ),
    )

    logger.info("gta pipeline finished session_id=%s", runtime.session_id)


def main() -> None:
    configure_logging("INFO")
    asyncio.run(run_gta_wasted_pipeline())


if __name__ == "__main__":
    main()
