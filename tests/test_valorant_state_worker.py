from __future__ import annotations

import asyncio

from sg_coach.orchestrator.bus import EventBus
from sg_coach.orchestrator.topics import ROUND_PACKET_READY, UI_STATE
from sg_coach.orchestrator.valorant_state import ValorantMatchStateTracker, valorant_state_worker
from sg_coach.shared.events import GameEvent
from sg_coach.shared.game_profiles import ValorantLiveRoundState, ValorantRoundPacket
from sg_coach.shared.streaming import (
    EVENT_STREAM_COMPLETE,
    ROUND_PACKET_STREAM_COMPLETE,
    UI_STREAM_COMPLETE,
)


def _event(
    event_type: str,
    *,
    session_id: str = "session_test",
    metadata: dict | None = None,
    confidence: float = 0.8,
) -> GameEvent:
    return GameEvent(
        session_id=session_id,
        game="valorant",
        event_type=event_type,
        confidence=confidence,
        metadata=metadata or {},
    )


class _RuntimeStub:
    def __init__(self) -> None:
        self.bus = EventBus()

    def subscribe(self, topic: str, *, maxsize: int = 0) -> asyncio.Queue:
        return self.bus.subscribe(topic, maxsize=maxsize)

    async def publish(self, topic: str, item: object) -> None:
        await self.bus.publish(topic, item)


def test_tracker_builds_live_state_and_round_packet() -> None:
    tracker = ValorantMatchStateTracker(session_id="session_test")

    started = tracker.consume_event(
        _event(
            "round_started",
            metadata={
                "map_name": "Ascent",
                "round_number": 7,
                "player_side": "attackers",
                "team_score": 3,
                "opponent_score": 3,
                "round_phase": "buy_phase",
                "player_alive": True,
                "player_match_kills": 5,
                "player_match_deaths": 4,
                "player_match_assists": 1,
            },
        )
    )

    assert started.round_packet is None
    assert started.live_state.round_number == 7
    assert started.live_state.player_side == "attackers"
    assert started.live_state.team_score == 3
    assert started.live_state.opponent_score == 3
    assert started.live_state.round_phase == "buy_phase"

    tracker.consume_event(
        _event(
            "player_kill",
            metadata={
                "round_number": 7,
                "site": "A",
                "player_match_kills": 6,
            },
        )
    )
    tracker.consume_event(
        _event(
            "spike_planted",
            metadata={
                "round_number": 7,
                "site": "A",
            },
        )
    )
    finished = tracker.consume_event(
        _event(
            "spike_exploded",
            confidence=0.91,
            metadata={
                "round_number": 7,
                "team_score": 4,
                "opponent_score": 3,
                "spike_exploded": True,
                "round_winner": "player_team",
                "player_match_kills": 6,
                "player_match_deaths": 4,
                "player_match_assists": 1,
            },
        )
    )

    packet = finished.round_packet
    assert packet is not None
    print(
        "round packet:",
        packet.round_number,
        packet.round_winner,
        packet.round_end_reason,
        packet.team_score_before,
        packet.opponent_score_before,
        packet.team_score_after,
        packet.opponent_score_after,
        packet.player_kills,
        packet.end_sequence,
    )

    assert finished.live_state.round_phase == "ended"
    assert finished.live_state.spike_state == "exploded"
    assert finished.live_state.player_round_kills == 1
    assert packet.round_number == 7
    assert packet.round_winner == "player_team"
    assert packet.round_end_reason == "spike_exploded"
    assert packet.team_score_before == 3
    assert packet.opponent_score_before == 3
    assert packet.team_score_after == 4
    assert packet.opponent_score_after == 3
    assert packet.spike_planted is True
    assert packet.spike_exploded is True
    assert packet.player_kills == 1
    assert [moment.event_type for moment in packet.key_moments] == [
        "player_kill",
        "spike_planted",
        "spike_exploded",
    ]
    assert packet.player_enabled_win is True
    assert "contributed_to_round_win" in packet.player_impact_tags
    assert "round_ended_by_spike_exploded" in packet.causal_context

    assert tracker.current_state.round_phase == "between_rounds"
    assert tracker.current_state.team_score == 4
    assert tracker.current_state.opponent_score == 3
    assert tracker.current_state.player_round_kills == 0


def test_tracker_rolls_forward_when_new_round_appears() -> None:
    tracker = ValorantMatchStateTracker(session_id="session_test")

    tracker.consume_event(
        _event(
            "round_started",
            metadata={
                "map_name": "Bind",
                "round_number": 11,
                "player_side": "defenders",
                "team_score": 5,
                "opponent_score": 5,
                "round_phase": "buy_phase",
            },
        )
    )
    tracker.consume_event(
        _event(
            "player_kill",
            metadata={"round_number": 11, "site": "B"},
        )
    )

    advanced = tracker.consume_event(
        _event(
            "round_started",
            metadata={
                "map_name": "Bind",
                "round_number": 12,
                "player_side": "defenders",
                "team_score": 5,
                "opponent_score": 6,
                "round_phase": "buy_phase",
            },
        )
    )

    assert advanced.round_packet is None
    assert advanced.live_state.round_number == 12
    assert advanced.live_state.player_round_kills == 0
    assert advanced.live_state.notable_round_events == []
    assert advanced.live_state.round_phase == "buy_phase"
    assert tracker.current_state.round_number == 12
    assert tracker.current_state.player_round_kills == 0


def test_worker_publishes_live_state_round_packet_and_completion() -> None:
    async def _run() -> None:
        runtime = _RuntimeStub()
        event_queue: asyncio.Queue[GameEvent | str] = asyncio.Queue()
        ui_queue = runtime.subscribe(UI_STATE)
        round_queue = runtime.subscribe(ROUND_PACKET_READY)
        tracker = ValorantMatchStateTracker(session_id="session_worker")

        worker_task = asyncio.create_task(
            valorant_state_worker(runtime, event_queue, tracker=tracker)
        )

        await event_queue.put(
            _event(
                "round_started",
                session_id="session_worker",
                metadata={
                    "map_name": "Lotus",
                    "round_number": 2,
                    "player_side": "defenders",
                    "team_score": 1,
                    "opponent_score": 0,
                    "round_phase": "buy_phase",
                },
            )
        )
        await event_queue.put(
            _event(
                "round_lost",
                session_id="session_worker",
                confidence=0.86,
                metadata={
                    "map_name": "Lotus",
                    "round_number": 2,
                    "player_side": "defenders",
                    "team_score": 1,
                    "opponent_score": 1,
                    "round_end_reason": "elimination_or_timeout",
                    "round_winner": "opponents",
                    "player_alive": False,
                },
            )
        )
        await event_queue.put(EVENT_STREAM_COMPLETE)
        await worker_task

        ui_items = [await ui_queue.get(), await ui_queue.get(), await ui_queue.get()]
        round_items = [await round_queue.get(), await round_queue.get()]

        assert isinstance(ui_items[0], ValorantLiveRoundState)
        assert isinstance(ui_items[1], ValorantLiveRoundState)
        assert ui_items[0].round_phase == "buy_phase"
        assert ui_items[1].round_phase == "ended"
        assert ui_items[2] == UI_STREAM_COMPLETE

        assert isinstance(round_items[0], ValorantRoundPacket)
        assert round_items[0].round_winner == "opponents"
        assert round_items[0].team_score_before == 1
        assert round_items[0].opponent_score_before == 0
        assert round_items[0].team_score_after == 1
        assert round_items[0].opponent_score_after == 1
        assert round_items[1] == ROUND_PACKET_STREAM_COMPLETE

        assert tracker.current_state.round_phase == "between_rounds"
        assert tracker.current_state.team_score == 1
        assert tracker.current_state.opponent_score == 1

    asyncio.run(_run())
