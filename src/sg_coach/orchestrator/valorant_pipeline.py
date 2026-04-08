from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING

from sg_coach.memory.worker import memory_snapshot_sink, memory_worker
from sg_coach.orchestrator.topics import EVENT_GAME, MEMORY_UPDATED, ROUND_PACKET_READY, UI_STATE
from sg_coach.orchestrator.valorant_state import (
    ValorantMatchStateTracker,
    valorant_live_state_sink,
    valorant_round_packet_sink,
    valorant_state_worker,
)
from sg_coach.shared.events import GameEvent
from sg_coach.shared.logging import configure_logging, get_logger
from sg_coach.shared.streaming import EVENT_STREAM_COMPLETE


if TYPE_CHECKING:
    from sg_coach.orchestrator.session import SessionRuntime


logger = get_logger(__name__)


def build_scripted_valorant_events(*, session_id: str) -> list[GameEvent]:
    """Return a tiny scripted match slice used to validate the Valorant pipeline.

    This intentionally bypasses CV for now. The goal is to prove:
    - canonical Valorant events can be published
    - the state worker updates live round state
    - completed rounds emit `ValorantRoundPacket`

    Once real detectors exist, they should emit the same event shapes.
    """
    return [
        GameEvent(
            session_id=session_id,
            game="valorant",
            event_type="round_started",
            confidence=0.98,
            metadata={
                "map_name": "Ascent",
                "round_number": 1,
                "player_side": "attackers",
                "team_score": 0,
                "opponent_score": 0,
                "round_phase": "buy_phase",
                "player_alive": True,
                "player_match_kills": 0,
                "player_match_deaths": 0,
                "player_match_assists": 0,
            },
        ),
        GameEvent(
            session_id=session_id,
            game="valorant",
            event_type="player_kill",
            confidence=0.87,
            metadata={
                "map_name": "Ascent",
                "round_number": 1,
                "player_side": "attackers",
                "team_score": 0,
                "opponent_score": 0,
                "site": "A",
                "player_match_kills": 1,
            },
        ),
        GameEvent(
            session_id=session_id,
            game="valorant",
            event_type="spike_planted",
            confidence=0.96,
            metadata={
                "map_name": "Ascent",
                "round_number": 1,
                "player_side": "attackers",
                "team_score": 0,
                "opponent_score": 0,
                "site": "A",
                "spike_state": "planted",
            },
        ),
        GameEvent(
            session_id=session_id,
            game="valorant",
            event_type="spike_exploded",
            confidence=0.97,
            metadata={
                "map_name": "Ascent",
                "round_number": 1,
                "player_side": "attackers",
                "team_score": 1,
                "opponent_score": 0,
                "round_winner": "player_team",
                "round_end_reason": "spike_exploded",
                "spike_exploded": True,
                "player_alive": True,
                "player_match_kills": 1,
                "player_match_deaths": 0,
                "player_match_assists": 0,
            },
        ),
        GameEvent(
            session_id=session_id,
            game="valorant",
            event_type="round_started",
            confidence=0.98,
            metadata={
                "map_name": "Ascent",
                "round_number": 2,
                "player_side": "attackers",
                "team_score": 1,
                "opponent_score": 0,
                "round_phase": "buy_phase",
                "player_alive": True,
                "player_match_kills": 1,
                "player_match_deaths": 0,
                "player_match_assists": 0,
            },
        ),
        GameEvent(
            session_id=session_id,
            game="valorant",
            event_type="player_died",
            confidence=0.86,
            metadata={
                "map_name": "Ascent",
                "round_number": 2,
                "player_side": "attackers",
                "team_score": 1,
                "opponent_score": 0,
                "site": "B",
                "player_alive": False,
                "player_match_kills": 1,
                "player_match_deaths": 1,
                "player_match_assists": 0,
            },
        ),
        GameEvent(
            session_id=session_id,
            game="valorant",
            event_type="round_lost",
            confidence=0.94,
            metadata={
                "map_name": "Ascent",
                "round_number": 2,
                "player_side": "attackers",
                "team_score": 1,
                "opponent_score": 1,
                "round_winner": "opponents",
                "round_end_reason": "elimination_or_timeout",
                "player_alive": False,
                "player_match_kills": 1,
                "player_match_deaths": 1,
                "player_match_assists": 0,
            },
        ),
    ]


async def scripted_valorant_event_producer(
    runtime: SessionRuntime,
    *,
    events: Sequence[GameEvent] | None = None,
    event_delay_seconds: float = 0.0,
) -> None:
    """Publish a deterministic Valorant event sequence into the shared bus."""
    scripted_events = list(events) if events is not None else build_scripted_valorant_events(session_id=runtime.session_id)

    for event in scripted_events:
        await runtime.publish(EVENT_GAME, event)
        logger.info(
            "valorant scripted producer published event type=%s round=%s confidence=%.3f",
            event.event_type,
            event.metadata.get("round_number"),
            event.confidence,
        )
        if event_delay_seconds > 0:
            await asyncio.sleep(event_delay_seconds)

    await runtime.publish(EVENT_GAME, EVENT_STREAM_COMPLETE)


async def event_sink(
    event_queue: asyncio.Queue[GameEvent | str],
    *,
    sink_name: str,
) -> None:
    """Log final published Valorant events for pipeline inspection."""
    while True:
        item = await event_queue.get()
        if item == EVENT_STREAM_COMPLETE:
            logger.info("%s stream complete", sink_name)
            return

        event = item
        logger.info(
            "%s received event type=%s round=%s confidence=%.3f",
            sink_name,
            event.event_type,
            event.metadata.get("round_number"),
            event.confidence,
        )


async def run_valorant_pipeline_demo(*, event_delay_seconds: float = 0.0) -> None:
    """Run the first scripted Valorant state pipeline.

    This is intentionally narrow: it validates the end-to-end state flow before
    we commit to real HUD detectors. Real detectors should eventually replace
    the scripted producer while keeping the same event shapes.
    """
    from sg_coach.orchestrator.session import SessionRuntime
    from sg_coach.shared.settings import load_settings

    settings = load_settings()
    runtime = SessionRuntime.create(settings, game_key="valorant")
    tracker = ValorantMatchStateTracker(session_id=runtime.session_id)

    event_queue_state = runtime.subscribe(EVENT_GAME)
    event_queue_memory = runtime.subscribe(EVENT_GAME)
    event_queue_audit = runtime.subscribe(EVENT_GAME)
    memory_snapshot_queue = runtime.subscribe(MEMORY_UPDATED)
    ui_queue = runtime.subscribe(UI_STATE)
    round_packet_queue = runtime.subscribe(ROUND_PACKET_READY)

    logger.info(
        "valorant scripted pipeline starting session_id=%s event_delay_seconds=%.2f",
        runtime.session_id,
        event_delay_seconds,
    )

    await asyncio.gather(
        scripted_valorant_event_producer(
            runtime,
            event_delay_seconds=event_delay_seconds,
        ),
        event_sink(event_queue_audit, sink_name="valorant_audit_sink"),
        memory_worker(runtime.bus, event_queue_memory, store=runtime.memory_store),
        memory_snapshot_sink(memory_snapshot_queue),
        valorant_state_worker(runtime, event_queue_state, tracker=tracker),
        valorant_live_state_sink(ui_queue),
        valorant_round_packet_sink(round_packet_queue),
    )

    logger.info("valorant scripted pipeline finished session_id=%s", runtime.session_id)


def main() -> None:
    configure_logging("INFO")
    asyncio.run(run_valorant_pipeline_demo())


if __name__ == "__main__":
    main()
