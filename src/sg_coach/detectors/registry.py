from __future__ import annotations

from dataclasses import dataclass, field

from sg_coach.detectors.base import Detector


@dataclass(slots=True)
class DetectorRegistry:
    """Store and retrieve detectors for the active runtime.

    Why this exists:
    - the real app will run multiple detectors
    - the orchestrator should not hardcode detector instances inline
    - later, this becomes the place where game-specific detector packs are assembled
    """

    detectors: list[Detector] = field(default_factory=list)

    def register(self, detector: Detector) -> None:
        """Add one detector to the registry.

        Hint:
        - append the detector to `self.detectors`
        - do not overthink validation yet
        """
        self.detectors.append(detector)

    def all(self) -> list[Detector]:
        """Return all registered detectors.

        Hint:
        - return a shallow copy so callers do not mutate internal state directly
        """
        return self.detectors.copy() #copy returns shallow copy

    def for_game(self, game: str) -> list[Detector]:
        """Return detectors that should run for a specific game.

        Hint:
        - filter by `detector.game == game`
        - later this may support shared packs like `fps_common`
        """
        game_spec = []
        for det in self.detectors:
            if det.game == game:
                game_spec.append(det)
        return game_spec

