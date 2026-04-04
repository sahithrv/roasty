from __future__ import annotations

from dataclasses import dataclass, field

from sg_coach.shared.events import DetectionSignal, FramePacket


@dataclass(slots=True)
class DemoCycleDetector:
    """Simple detector used only to exercise the pipeline.

    This detector should mimic the behavior that currently lives in
    `fake_detector_worker()`:
    - cycle through a small sequence of event types
    - emit one synthetic `DetectionSignal` per frame
    - keep enough internal state to know which signal comes next
    """

    name: str = "demo_detector"
    game: str = "valorant"
    signal_cycle: tuple[str, ...] = ("kill", "death", "round_end")
    confidence_by_type: dict[str, float] = field(
        default_factory=lambda: {
            "kill": 0.90,
            "death": 0.82,
            "round_end": 0.88,
        }
    )
    frame_index: int = 0

    def next_signal_type(self) -> str:
        """Return the next signal type in the cycle.

        Hint:
        - use `self.frame_index` with modulo arithmetic
        - increment `self.frame_index` after choosing the type
        """
        prev = self.frame_index
        self.frame_index += 1
        return self.signal_cycle[prev % len(self.signal_cycle)]

    async def detect(self, frame: FramePacket) -> list[DetectionSignal]:
        """Emit one synthetic signal for a frame.

        Hint:
        - call `self.next_signal_type()`
        - build one `DetectionSignal`
        - use `frame.game or self.game`
        - set `frame_ref=frame.frame_id`
        - return a one-item list
        """
        signal_type = self.next_signal_type()
        confidence = self.confidence_by_type[signal_type]
        detection_signal = DetectionSignal(
            game=frame.game or self.game,
            detector_name=self.name,
            signal_type=signal_type,
            confidence=confidence,
            frame_ref=frame.frame_id,
            tags=["demo", "synthetic"],
            metadata={"frame_index": self.frame_index - 1},
            dedupe_key=f"{signal_type}:{self.frame_index - 1}",
            cooldown_key=signal_type,
        )
        return [detection_signal]
