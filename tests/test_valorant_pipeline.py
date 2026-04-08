from __future__ import annotations

import asyncio

from sg_coach.orchestrator.bus import EventBus
from sg_coach.orchestrator.topics import EVENT_GAME, ROUND_PACKET_READY, UI_STATE
from sg_coach.orchestrator.valorant_pipeline import (
    build_scripted_valorant_events,
    scripted_valorant_event_producer,
)
from sg_coach.orchestrator.valorant_state import (
    ValorantMatchStateTracker,
    valorant_state_worker,
)
from sg_coach.shared.game_profiles import ValorantLiveRoundState, ValorantRoundPacket
from sg_coach.shared.streaming import ROUND_PACKET_STREAM_COMPLETE, UI_STREAM_COMPLETE


class _RuntimeStub:
    def __init__(self, *, session_id: str = "session_scripted") -> None:
        self.session_id = session_id
        self.bus = EventBus()

    def subscribe(self, topic: str, *, maxsize: int = 0) -> asyncio.Queue:
        return self.bus.subscribe(topic, maxsize=maxsize)

    async def publish(self, topic: str, item: object) -> None:
        await self.bus.publish(topic, item)


def test_scripted_valorant_events_have_expected_shape() -> None:
    events = build_scripted_valorant_events(session_id="session_test")

    assert [event.event_type for event in events] == [
        "round_started",
        "player_kill",
        "spike_planted",
        "spike_exploded",
        "round_started",
        "player_died",
        "round_lost",
    ]
    assert events[0].metadata["player_side"] == "attackers"
    assert events[3].metadata["round_end_reason"] == "spike_exploded"
    assert events[-1].metadata["opponent_score"] == 1


def test_scripted_valorant_pipeline_emits_two_round_packets() -> None:
    async def _run() -> None:
        runtime = _RuntimeStub()
        tracker = ValorantMatchStateTracker(session_id=runtime.session_id)

        event_queue = runtime.subscribe(EVENT_GAME)
        ui_queue = runtime.subscribe(UI_STATE)
        round_queue = runtime.subscribe(ROUND_PACKET_READY)

        await asyncio.gather(
            scripted_valorant_event_producer(runtime),
            valorant_state_worker(runtime, event_queue, tracker=tracker),
        )

        ui_items: list[ValorantLiveRoundState | str] = []
        while True:
            item = await ui_queue.get()
            ui_items.append(item)
            if item == UI_STREAM_COMPLETE:
                break

        round_items: list[ValorantRoundPacket | str] = []
        while True:
            item = await round_queue.get()
            round_items.append(item)
            if item == ROUND_PACKET_STREAM_COMPLETE:
                break

        packets = [item for item in round_items if isinstance(item, ValorantRoundPacket)]
        print(
            "scripted packets:",
            [(packet.round_number, packet.round_winner, packet.round_end_reason) for packet in packets],
        )

        assert len([item for item in ui_items if isinstance(item, ValorantLiveRoundState)]) == 7
        assert ui_items[-1] == UI_STREAM_COMPLETE

        assert len(packets) == 2
        assert packets[0].round_number == 1
        assert packets[0].round_winner == "player_team"
        assert packets[0].round_end_reason == "spike_exploded"
        assert packets[0].team_score_after == 1
        assert packets[0].player_kills == 1

        assert packets[1].round_number == 2
        assert packets[1].round_winner == "opponents"
        assert packets[1].round_end_reason == "elimination_or_timeout"
        assert packets[1].team_score_after == 1
        assert packets[1].opponent_score_after == 1
        assert packets[1].player_died is True

        assert round_items[-1] == ROUND_PACKET_STREAM_COMPLETE

    asyncio.run(_run())
