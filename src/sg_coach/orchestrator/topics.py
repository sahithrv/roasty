from __future__ import annotations

from typing import Final, Literal


FRAME_RAW: Final[str] = "frame.raw"
SIGNAL_DETECTOR: Final[str] = "signal.detector"
EVENT_GAME: Final[str] = "event.game"
MEMORY_UPDATED: Final[str] = "memory.updated"
COMMENTARY_REQUEST: Final[str] = "commentary.request"
COMMENTARY_READY: Final[str] = "commentary.ready"
SPEECH_PLAY: Final[str] = "speech.play"
UI_STATE: Final[str] = "ui.state"
ROUND_PACKET_READY: Final[str] = "round.packet"


TopicName = Literal[
    "frame.raw",
    "signal.detector",
    "event.game",
    "memory.updated",
    "commentary.request",
    "commentary.ready",
    "speech.play",
    "ui.state",
    "round.packet",
]

