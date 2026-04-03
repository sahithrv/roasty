from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, DefaultDict

from pydantic import BaseModel, Field

from sg_coach.shared.events import GameEvent

_CALLBACK_COOLDOWN = timedelta(seconds=90)
_CALLBACK_EXPIRY = timedelta(minutes=20)
_SUMMARY_REFRESH_INTERVAL = timedelta(minutes=2)
_SUMMARY_REFRESH_NOTABLE_THRESHOLD = 8
_KNOWN_MISTAKE_EVENT_TYPES = frozenset(
    {
        "death",
        "mission_failed",
        "busted",
        "wasted",
        "whiff",
        "self_flash",
        "teamkill",
        "own_goal",
    }
)
_MISTAKE_HINT_WORDS = ("death", "fail", "mistake", "misplay", "throw", "whiff", "own")
_MISTAKE_HINT_TAGS = frozenset({"mistake", "misplay", "blunder", "throw", "whiff"})
_NOTABLE_EVENT_TYPES = frozenset(
    {
        "kill",
        "round_end",
        "clutch",
        "ace",
        "mission_failed",
        "wasted",
        "busted",
        "chaos_moment",
    }
)
_NOTABLE_HINT_TAGS = frozenset({"notable", "highlight", "milestone"})


class MemorySnapshot(BaseModel):
    """A compact read model of the current session memory state.

    This is the object other layers will eventually consume when they want
    "memory" without caring how it is stored internally.
    """

    session_id: str
    total_events: int
    counters: dict[str, int] = Field(default_factory=dict)
    latest_event: GameEvent | None = None
    recent_events: list[GameEvent] = Field(default_factory=list)
    notable_events: list[GameEvent] = Field(default_factory=list)
    recurring_patterns: dict[str, int] = Field(default_factory=dict)
    callback_candidates: list[dict[str, Any]] = Field(default_factory=list)
    summary_refresh_needed: bool = False
    summary_text: str = ""


class SessionMemoryStore:
    """In-memory session memory for early development.

    Why start here:
    - the rest of the architecture can learn to speak to a memory layer now
    - we avoid dragging SQLite into the project before the event flow is stable
    - the public methods here can stay mostly the same later when persistence
      moves behind the same interface

    What this stores today:
    - the raw event log for the session
    - per-event-type counters
    - a rolling recent-event window

    What is intentionally deferred:
    - richer rolling text summaries
    - recurring pattern detection across richer metadata
    - commentary line selection policy
    - SQLite persistence
    """

    def __init__(self, session_id: str, *, recent_window_size: int = 10) -> None:
        self.session_id = session_id
        self.recent_window_size = recent_window_size
        self._event_log: list[GameEvent] = []
        self._recent_events: deque[GameEvent] = deque(maxlen=recent_window_size)
        self._counters: DefaultDict[str, int] = defaultdict(int)
        self._notable_events: deque[GameEvent] = deque(maxlen=max(recent_window_size * 2, 16))
        self._recurring_patterns: DefaultDict[str, int] = defaultdict(int)
        self._pattern_first_seen_at: dict[str, datetime] = {}
        self._callback_candidates: dict[str, dict[str, Any]] = {}
        self._notable_events_since_summary = 0
        self._summary_cache_dirty = False
        self._summary_refresh_needed = False
        self._last_summary_refresh_at: datetime | None = None
        self._cached_summary_text = "No notable events yet."

    def store_event(self, event: GameEvent) -> None:
        """Persist one event into in-memory session state."""
        if event.session_id != self.session_id:
            raise ValueError(
                f"Cannot store event for session '{event.session_id}' in memory store for '{self.session_id}'."
            )

        if self._is_notable_event(event) and "notable" not in event.tags:
            # Avoid mutating the shared event object that may also be in flight to
            # other subscribers on the bus.
            event = event.model_copy(update={"tags": [*event.tags, "notable"]})

        self._event_log.append(event)
        self._recent_events.append(event)
        self._counters[event.event_type] += 1
        self._summary_cache_dirty = True

        if self._is_notable_event(event):
            self._notable_events.append(event)
            self._notable_events_since_summary += 1

        self._prune_callback_candidates(event.timestamp)

        pattern_family = self._pattern_family_for_event(event)
        if pattern_family is not None:
            self._pattern_first_seen_at.setdefault(pattern_family, event.timestamp)
            self._recurring_patterns[pattern_family] += 1
            if self._recurring_patterns[pattern_family] >= 2:
                self._register_callback_candidate(
                    pattern_family=pattern_family,
                    event=event,
                    occurrences=self._recurring_patterns[pattern_family],
                )

        if self._last_summary_refresh_at is None:
            self._last_summary_refresh_at = event.timestamp
        elif event.timestamp - self._last_summary_refresh_at >= _SUMMARY_REFRESH_INTERVAL:
            self._summary_refresh_needed = True

        if self._notable_events_since_summary >= _SUMMARY_REFRESH_NOTABLE_THRESHOLD:
            self._summary_refresh_needed = True

    def total_events(self) -> int:
        """Return the number of stored events in the current session."""
        return len(self._event_log)

    def event_count(self, event_type: str) -> int:
        """Return the counter for a specific event type such as 'kill' or 'death'."""
        return self._counters.get(event_type, 0)

    def latest_event(self) -> GameEvent | None:
        """Return the most recent event, if one exists."""
        if not self._event_log:
            return None
        return self._event_log[-1]

    def recent_events(self, *, limit: int = 5) -> list[GameEvent]:
        """Return the most recent events in chronological order."""
        if limit <= 0:
            return []
        return list(self._recent_events)[-limit:]

    def counters(self) -> dict[str, int]:
        """Return a plain dict snapshot of the internal counters."""
        return dict(self._counters)

    def notable_events(self, *, limit: int = 5) -> list[GameEvent]:
        """Return the most recent notable events in chronological order."""
        if limit <= 0:
            return []
        return list(self._notable_events)[-limit:]

    def recurring_patterns(self) -> dict[str, int]:
        """Return recurring mistake-pattern counters keyed by pattern family."""
        return dict(self._recurring_patterns)

    def callback_candidates(self, *, limit: int = 3) -> list[dict[str, Any]]:
        """Return the newest active callback candidates first."""
        if limit <= 0:
            return []
        candidates = sorted(
            self._callback_candidates.values(),
            key=lambda candidate: candidate["last_seen_at"],
            reverse=True,
        )
        return [dict(candidate) for candidate in candidates[:limit]]

    def summary_refresh_needed(self) -> bool:
        """Return whether the cached summary should be rebuilt."""
        return self._summary_refresh_needed

    def mark_summary_refreshed(self) -> None:
        """Clear summary refresh state after a caller rebuilds the rolling summary."""
        self._notable_events_since_summary = 0
        self._summary_refresh_needed = False
        latest_event = self.latest_event()
        self._last_summary_refresh_at = None if latest_event is None else latest_event.timestamp

    def build_snapshot(self, *, recent_limit: int = 5) -> MemorySnapshot:
        """Build a read-only view of the current memory state."""
        return MemorySnapshot(
            session_id=self.session_id,
            total_events=self.total_events(),
            counters=self.counters(),
            latest_event=self.latest_event(),
            recent_events=self.recent_events(limit=recent_limit),
            notable_events=self.notable_events(limit=recent_limit),
            recurring_patterns=self.recurring_patterns(),
            callback_candidates=self.callback_candidates(limit=recent_limit),
            summary_refresh_needed=self.summary_refresh_needed(),
            summary_text=self.summary_text(),
        )

    def summary_text(self) -> str:
        """Return a simple human-readable summary.

        This is intentionally primitive. It gives the rest of the system a stable
        method name now, and we can later replace the internals with something
        more sophisticated without changing callers.
        """
        if not self._event_log:
            return "No notable events yet."
        if not self._summary_cache_dirty:
            return self._cached_summary_text

        parts = [f"Session has {self.total_events()} tracked events."]
        for event_type, count in sorted(self._counters.items()):
            parts.append(f"{event_type}={count}")

        recurring = [
            f"{pattern}={count}"
            for pattern, count in sorted(self._recurring_patterns.items())
            if count >= 2
        ]
        if recurring:
            parts.append(f"recurring[{', '.join(recurring[:3])}]")

        notable = [event.event_type for event in self.notable_events(limit=3)]
        if notable:
            parts.append(f"notable[{', '.join(notable)}]")

        self._cached_summary_text = " ".join(parts)
        self._summary_cache_dirty = False

        return self._cached_summary_text

    def _is_notable_event(self, event: GameEvent) -> bool:
        if bool(event.metadata.get("notable")):
            return True
        if event.clip_path is not None:
            return True
        if event.event_type in _NOTABLE_EVENT_TYPES:
            return True
        if event.confidence >= 0.9:
            return True
        return any(tag in _NOTABLE_HINT_TAGS for tag in event.tags)

    def _pattern_family_for_event(self, event: GameEvent) -> str | None:
        for key in ("mistake_key", "pattern_family", "callback_family"):
            value = event.metadata.get(key)
            if isinstance(value, str) and value:
                return value

        if bool(event.metadata.get("is_mistake")):
            return event.event_type

        if event.event_type in _KNOWN_MISTAKE_EVENT_TYPES:
            return event.event_type

        event_type = event.event_type.lower()
        if any(word in event_type for word in _MISTAKE_HINT_WORDS):
            return event.event_type

        if any(tag in _MISTAKE_HINT_TAGS for tag in event.tags):
            return event.event_type

        return None

    def _register_callback_candidate(
        self,
        *,
        pattern_family: str,
        event: GameEvent,
        occurrences: int,
    ) -> None:
        existing = self._callback_candidates.get(pattern_family)
        if existing is not None:
            cooldown_window = event.timestamp - existing["last_seen_at"]
            if cooldown_window < _CALLBACK_COOLDOWN:
                existing["count"] = occurrences
                existing["last_seen_at"] = event.timestamp
                existing["latest_event_id"] = event.event_id
                existing["event_type"] = event.event_type
                existing["reason"] = f"{pattern_family} has repeated {occurrences} times this session."
                return

        first_seen_at = self._pattern_first_seen_at.get(pattern_family, event.timestamp)
        self._callback_candidates[pattern_family] = {
            "pattern_family": pattern_family,
            "event_type": event.event_type,
            "count": occurrences,
            "first_seen_at": first_seen_at,
            "last_seen_at": event.timestamp,
            "latest_event_id": event.event_id,
            "reason": f"{pattern_family} has repeated {occurrences} times this session.",
        }

    def _prune_callback_candidates(self, current_timestamp: datetime) -> None:
        expired_families = [
            family
            for family, candidate in self._callback_candidates.items()
            if current_timestamp - candidate["last_seen_at"] >= _CALLBACK_EXPIRY
        ]
        for family in expired_families:
            del self._callback_candidates[family]
