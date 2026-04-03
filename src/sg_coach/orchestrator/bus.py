from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any


class EventBus:
    """Small in-process async event bus.

    Design note:
    We use one queue per *subscriber*, not one queue per topic.
    If we used a single queue for a topic, multiple consumers would compete
    for the same messages and only one of them would receive each event.

    For this project, topics like ``frame.raw`` and ``event.game`` are
    broadcast streams. Multiple parts of the system may need to observe the
    same event independently, so each subscriber gets its own queue.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[Any]]] = defaultdict(list)

    def subscribe(self, topic: str, *, maxsize: int = 0) -> asyncio.Queue[Any]:
        """Create and register a new queue for a topic."""
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=maxsize)
        self._subscribers[topic].append(queue)
        return queue

    async def publish(self, topic: str, item: Any) -> None:
        """Broadcast one item to every subscriber currently attached to a topic."""
        # Iterate over a snapshot so subscription changes during a publish do not
        # mutate the list we are currently walking.
        for queue in tuple(self._subscribers.get(topic, [])):
            await queue.put(item)

    async def publish_many(self, topic: str, items: list[Any]) -> None:
        """Publish a batch of items in order."""
        for item in items:
            await self.publish(topic, item)

    def subscriber_count(self, topic: str) -> int:
        """Return the number of active subscribers for a topic."""
        return len(self._subscribers.get(topic, []))

    def unsubscribe(self, topic: str, queue: asyncio.Queue[Any]) -> None:
        """Remove a subscriber queue from a topic if it is registered."""
        subscribers = self._subscribers.get(topic)
        if not subscribers:
            return

        try:
            subscribers.remove(queue)
        except ValueError:
            return

        if not subscribers:
            self._subscribers.pop(topic, None)
