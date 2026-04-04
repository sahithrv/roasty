from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    """Generate a readable identifier for runtime objects."""
    return f"{prefix}_{uuid4().hex}"


class FramePacket(BaseModel):
    """A captured frame plus basic metadata."""

    frame_id: str = Field(default_factory=lambda: new_id("frame"))
    timestamp: datetime = Field(default_factory=utc_now)
    game: str | None = None
    monitor_id: int = 0
    width: int
    height: int
    image_bgr: Any | None = None
    roi_images: dict[str, Any] = Field(default_factory=dict)


class DetectionSignal(BaseModel):
    """A detector's claim about something it thinks it observed."""

    signal_id: str = Field(default_factory=lambda: new_id("signal"))
    timestamp: datetime = Field(default_factory=utc_now)
    game: str
    detector_name: str
    signal_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    roi_name: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    frame_ref: str | None = None
    ttl_ms: int = Field(default=750, ge=1)
    dedupe_key: str | None = None
    cooldown_key: str | None = None


class GameEvent(BaseModel):
    """A fused, higher-confidence event that the rest of the app can act on."""

    event_id: str = Field(default_factory=lambda: new_id("event"))
    timestamp: datetime = Field(default_factory=utc_now)
    session_id: str
    game: str
    event_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    frame_path: str | None = None
    clip_path: str | None = None
    source_signal_ids: list[str] = Field(default_factory=list)
    dedupe_key: str | None = None
    cooldown_key: str | None = None


class CommentaryRequest(BaseModel):
    """A request sent to the commentary layer after policy selection."""

    request_id: str = Field(default_factory=lambda: new_id("commentary"))
    persona: str
    latest_event: GameEvent
    recent_events: list[GameEvent] = Field(default_factory=list)
    counters: dict[str, int] = Field(default_factory=dict)
    callback_candidates: list[dict[str, Any]] = Field(default_factory=list)
    memory_summary: str = ""
    include_frame: bool = False
    frame_path: str | None = None
    context_frame_paths: list[str] = Field(default_factory=list)


class CommentaryResult(BaseModel):
    """The model output produced from a `CommentaryRequest`."""

    result_id: str = Field(default_factory=lambda: new_id("commentary_result"))
    request_id: str
    event_id: str
    model: str
    text: str
    raw_response: dict[str, Any] = Field(default_factory=dict)
