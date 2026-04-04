from __future__ import annotations

from typing import Protocol

from sg_coach.shared.events import DetectionSignal, GameEvent


class SignalFuser(Protocol):
    """Contract for turning detector signals into canonical game events.

    Design intent:
    - detectors emit lower-trust `DetectionSignal` objects
    - fusion decides what downstream systems should accept as canonical events
    - future fusers may buffer, merge, suppress, or upgrade multiple signals

    The protocol is intentionally shaped around consuming one signal at a time,
    because a real implementation may keep internal temporal state.
    """

    name: str

    async def consume(self, signal: DetectionSignal, *, session_id: str) -> list[GameEvent]:
        """Consume one signal and return zero or more fused game events."""
        ...

