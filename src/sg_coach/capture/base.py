from __future__ import annotations

from typing import Protocol

from sg_coach.shared.events import FramePacket


class FrameSource(Protocol):
    """Contract for anything that can yield live frame packets."""

    async def frames(self):
        """Yield `FramePacket` objects continuously until stopped."""
        ...

