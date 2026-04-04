from __future__ import annotations

from typing import Protocol

from sg_coach.shared.events import DetectionSignal, FramePacket


class Detector(Protocol):
    """Contract for any detector in the system.

    Design intent:
    - capture produces `FramePacket`
    - detectors inspect frames and emit `DetectionSignal`
    - fusion consumes signals and emits `GameEvent`

    The rest of the app should not care whether a detector is:
    - heuristic
    - OCR-based
    - template-matching based
    - ML-based
    """

    name: str
    game: str

    async def detect(self, frame: FramePacket) -> list[DetectionSignal]:
        """Inspect one frame and return zero or more detector signals."""
        ...

