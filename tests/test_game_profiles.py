from __future__ import annotations

from sg_coach.grok.payloads import build_grok_chat_payload, build_user_payload
from sg_coach.orchestrator.realtime import build_realtime_instructions
from sg_coach.orchestrator.session import SessionRuntime
from sg_coach.shared.events import CommentaryRequest, GameEvent
from sg_coach.shared.game_profiles import (
    ValorantLiveRoundState,
    ValorantRoundKeyMoment,
    ValorantRoundPacket,
    get_game_profile,
)
from sg_coach.shared.settings import Settings


def test_valorant_profile_contains_round_bootstrap() -> None:
    profile = get_game_profile("valorant")

    assert profile.key == "valorant"
    assert profile.structure_style == "round_based"
    assert "first to 13 rounds" in profile.model_bootstrap.lower()
    assert "spike" in " ".join(profile.terminology).lower()


def test_commentary_payload_includes_game_profile_context() -> None:
    profile = get_game_profile("valorant")
    latest_event = GameEvent(
        session_id="session_test",
        game="valorant",
        event_type="round_end",
        confidence=0.88,
        metadata={"round_number": 7, "round_end_reason": "spike_defused"},
    )
    request = CommentaryRequest(
        persona="off_the_walls",
        game_key=profile.key,
        game_profile=profile,
        latest_event=latest_event,
    )

    user_payload = build_user_payload(request)
    chat_payload = build_grok_chat_payload(request, model="grok-test")
    system_prompt = chat_payload["messages"][0]["content"]

    assert user_payload["game_key"] == "valorant"
    assert user_payload["game_profile"]["display_name"] == "Valorant"
    assert "Valorant" in system_prompt


def test_realtime_instructions_include_selected_game_profile() -> None:
    runtime = SessionRuntime.create(Settings(), game_key="valorant")

    instructions = build_realtime_instructions(runtime)

    assert "Selected game: Valorant" in instructions
    assert "round outcomes" in instructions
    assert "scoreline" in instructions


def test_valorant_round_contracts_capture_live_and_round_summary_state() -> None:
    live_state = ValorantLiveRoundState(
        map_name="Ascent",
        round_number=7,
        player_side="attackers",
        team_score=3,
        opponent_score=3,
        round_phase="postplant",
        spike_state="planted",
        player_round_kills=2,
        notable_round_events=["spike_planted", "double_kill"],
    )
    packet = ValorantRoundPacket(
        session_id="session_test",
        map_name=live_state.map_name,
        round_number=live_state.round_number,
        player_side=live_state.player_side,
        team_score_before=3,
        opponent_score_before=3,
        team_score_after=4,
        opponent_score_after=3,
        round_winner="player_team",
        round_end_reason="spike_exploded",
        spike_planted=True,
        spike_exploded=True,
        player_kills=2,
        player_survived=True,
        key_moments=[
            ValorantRoundKeyMoment(
                sequence=1,
                event_type="player_multi_kill",
                description="Player found two opening kills before the plant.",
            )
        ],
        end_sequence=["plant", "hold", "detonation"],
        player_impact_tags=["entry_success", "site_take"],
        causal_context=["player_double_kill_created_space_for_plant"],
        confidence=0.82,
    )

    assert live_state.player_side == "attackers"
    assert packet.round_end_reason == "spike_exploded"
    assert packet.key_moments[0].event_type == "player_multi_kill"
    assert packet.causal_context == ["player_double_kill_created_space_for_plant"]
