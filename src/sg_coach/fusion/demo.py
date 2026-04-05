from __future__ import annotations

from dataclasses import dataclass

from sg_coach.shared.events import DetectionSignal, GameEvent


@dataclass(slots=True)
class DemoPassthroughFuser:
    """Demo fuser that turns each signal directly into one game event.

    Why this exists:
    - it mirrors the current `fake_fusion_worker()` behavior
    - it creates the right module boundary before real fusion rules exist
    - later, a smarter fuser can keep time windows and combine weak signals
    """

    name: str = "demo_passthrough_fuser"

    async def consume(self, signal: DetectionSignal, *, session_id: str) -> list[GameEvent]:
        """Convert one signal into one event.

        Hints:
        - build one `GameEvent`
        - copy over the key fields from the signal:
          - `game`
          - `signal_type -> event_type`
          - `confidence`
          - `tags`
          - `dedupe_key`
          - `cooldown_key`
        - include useful metadata such as:
          - `source_detector`
          - `frame_ref`
        - include `source_signal_ids=[signal.signal_id]`
        - return a one-item list
        """
        event = GameEvent(
            session_id=session_id,
            game=signal.game,
            event_type=signal.signal_type,
            confidence=signal.confidence,
            tags=signal.tags,
            metadata={
                **signal.metadata,
                "source_detector": signal.detector_name,
                "frame_ref": signal.frame_ref,
            },
            source_signal_ids=[signal.signal_id],
            dedupe_key=signal.dedupe_key,
            cooldown_key=signal.cooldown_key,
        )
        return [event]
