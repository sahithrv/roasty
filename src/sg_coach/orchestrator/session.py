from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sg_coach.memory.store import MemorySnapshot, SessionMemoryStore
from sg_coach.orchestrator.bus import EventBus
from sg_coach.orchestrator.topics import TopicName
from sg_coach.shared.events import new_id
from sg_coach.shared.game_profiles import GameProfile, get_game_profile
from sg_coach.shared.settings import Settings


@dataclass(slots=True)
class SessionRuntime:
    """Own the core runtime objects for one active session.

    Why this object exists:
    - `run_demo()` currently wires individual pieces together manually
    - the real app will need one place that owns shared session state
    - this becomes the object that capture, detectors, memory, and commentary
      can all receive instead of passing around a growing list of arguments

    What it should own right now:
    - `session_id`
    - `settings`
    - `bus`
    - `memory_store`

    What it may own later:
    - detector registry
    - capture backend
    - commentary policy
    - persistence adapters
    """

    session_id: str
    game_key: str
    game_profile: GameProfile
    settings: Settings
    bus: EventBus
    memory_store: SessionMemoryStore

    @classmethod
    def create(cls, settings: Settings, *, game_key: str = "generic") -> "SessionRuntime":
        """Create a new runtime with fresh session-scoped dependencies.

        Hints:
        - generate `session_id` with `new_id("session")`
        - create one `EventBus`
        - create one `SessionMemoryStore` using that `session_id`
        - return `cls(...)`
        """
        session_id = new_id("session")
        bus = EventBus()
        session_store = SessionMemoryStore(session_id)
        game_profile = get_game_profile(game_key)
        return cls(session_id, game_profile.key, game_profile, settings, bus, session_store)

    def subscribe(self, topic: TopicName, *, maxsize: int = 0):
        """Delegate topic subscription to the session event bus.

        Hint:
        - this should be a very small wrapper around `self.bus.subscribe(...)`
        - keeping the wrapper is useful because callers can depend on the
          session runtime instead of reaching into `self.bus` directly
        """
        return self.bus.subscribe(topic, maxsize=maxsize)

    async def publish(self, topic: TopicName, item: Any) -> None:
        """Publish one item to the session event bus.

        Hint:
        - delegate to `self.bus.publish(...)`
        """
        await self.bus.publish(topic, item)

    async def publish_many(self, topic: TopicName, items: list[Any]) -> None:
        """Publish multiple items to the session event bus.

        Hint:
        - delegate to `self.bus.publish_many(...)`
        """
        await self.bus.publish_many(topic, items)

    def memory_snapshot(self, *, recent_limit: int = 5) -> MemorySnapshot:
        """Return a snapshot of current session memory.

        Hint:
        - delegate to `self.memory_store.build_snapshot(...)`
        """
        return self.memory_store.build_snapshot(recent_limit=recent_limit)
    
    def memory_summary(self) -> str:
        """Return the current memory summary text.

        Hint:
        - delegate to `self.memory_store.summary_text()`
        """
        return self.memory_store.summary_text()
