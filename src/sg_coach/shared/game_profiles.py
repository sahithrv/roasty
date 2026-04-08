from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class GameProfile(BaseModel):
    """Canonical static game context used to bootstrap prompt behavior."""

    model_config = ConfigDict(protected_namespaces=())

    key: str
    display_name: str
    mode_name: str
    structure_style: str
    coaching_posture: str
    model_bootstrap: str
    realtime_bootstrap: str
    rules: list[str] = Field(default_factory=list)
    objectives: list[str] = Field(default_factory=list)
    terminology: list[str] = Field(default_factory=list)


class ValorantLiveRoundState(BaseModel):
    """Canonical live match state for one active Valorant round."""

    map_name: str | None = None
    round_number: int | None = None
    player_side: str | None = None
    team_score: int | None = None
    opponent_score: int | None = None
    round_phase: str = "unknown"
    spike_state: str = "unknown"
    player_alive: bool | None = None
    player_match_kills: int | None = None
    player_match_deaths: int | None = None
    player_match_assists: int | None = None
    player_round_kills: int = 0
    player_round_assists: int = 0
    player_died_this_round: bool = False
    notable_round_events: list[str] = Field(default_factory=list)
    inference_notes: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)


class ValorantRoundKeyMoment(BaseModel):
    """One structured moment inside a completed Valorant round."""

    sequence: int
    event_type: str
    description: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValorantRoundPacket(BaseModel):
    """Canonical end-of-round record for structured Valorant commentary."""

    packet_id: str = Field(default_factory=lambda: _new_id("round_packet"))
    session_id: str | None = None
    map_name: str | None = None
    round_number: int | None = None
    player_side: str | None = None
    team_score_before: int | None = None
    opponent_score_before: int | None = None
    team_score_after: int | None = None
    opponent_score_after: int | None = None
    round_winner: str | None = None
    round_end_reason: str = "unknown"
    spike_planted: bool | None = None
    spike_defused: bool | None = None
    spike_exploded: bool | None = None
    player_kills: int = 0
    player_assists: int = 0
    player_died: bool | None = None
    player_survived: bool | None = None
    player_match_kills_after: int | None = None
    player_match_deaths_after: int | None = None
    player_match_assists_after: int | None = None
    first_blood: str | None = None
    key_moments: list[ValorantRoundKeyMoment] = Field(default_factory=list)
    end_sequence: list[str] = Field(default_factory=list)
    player_impact_tags: list[str] = Field(default_factory=list)
    player_enabled_win: bool | None = None
    player_enabled_loss: bool | None = None
    swing_moment: str | None = None
    causal_context: list[str] = Field(default_factory=list)
    source_event_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


_GENERIC_PROFILE = GameProfile(
    key="generic",
    display_name="Generic Game",
    mode_name="unknown mode",
    structure_style="unknown",
    coaching_posture="adapt to the supplied game telemetry and stay grounded in explicit context",
    model_bootstrap=(
        "Use the supplied game profile and event payload as the authoritative source of rules and structure. "
        "If context is incomplete, stay factual and avoid inventing game-specific mechanics."
    ),
    realtime_bootstrap=(
        "Use game telemetry as your source of truth. "
        "React conversationally, but do not invent rules or state that the pipeline has not established."
    ),
    rules=[],
    objectives=[],
    terminology=[],
)

_GTA_PROFILE = GameProfile(
    key="gta",
    display_name="Grand Theft Auto V",
    mode_name="open-world sandbox",
    structure_style="open_ended",
    coaching_posture="favor conversational reactions and scene-aware commentary over rigid coaching",
    model_bootstrap=(
        "The selected game is Grand Theft Auto V. "
        "Treat it as an open-ended sandbox where scenes can be chaotic, loosely structured, and driven by emergent action. "
        "Prioritize visible context, cause-and-effect, and the vibe of the moment over formal round-based analysis."
    ),
    realtime_bootstrap=(
        "This is GTA. Sound more like a witty companion reacting to chaos in an open-world sandbox. "
        "Coaching can still appear, but it should be lighter and more conversational than in a structured esport."
    ),
    rules=[
        "Open-world gameplay is not organized into rounds or fixed win conditions.",
        "Police chases, crashes, deaths, and environmental chaos can happen in many orders.",
    ],
    objectives=[
        "React to visible events and recurring player habits.",
        "Keep callbacks and personality strong without pretending the game is rigidly structured.",
    ],
    terminology=["wanted level", "cops", "chase", "wasted", "busted"],
)

_VALORANT_PROFILE = GameProfile(
    key="valorant",
    display_name="Valorant",
    mode_name="5v5 round-based competitive mode",
    structure_style="round_based",
    coaching_posture="lean into structured round analysis, objective play, and player impact over time",
    model_bootstrap=(
        "The selected game is Valorant. "
        "Treat it as a structured round-based tactical shooter. "
        "The standard mode is first to 13 rounds, with side swaps after 12 rounds. "
        "Attackers try to plant the spike or eliminate Defenders. "
        "Defenders try to stop the plant, defuse the spike, or eliminate Attackers. "
        "Reason in terms of rounds, sides, scoreline, spike outcomes, player contribution, and recent momentum."
    ),
    realtime_bootstrap=(
        "This is Valorant. "
        "Stay aware of rounds, player side, scoreline, spike state, and momentum from recent rounds. "
        "Coaching should be more structured and tied to round outcomes, objective play, and player impact."
    ),
    rules=[
        "Standard matches are first to 13 rounds.",
        "Sides swap after the first 12 rounds.",
        "Rounds end by team elimination, spike detonation, spike defusal, or timeout preventing a plant.",
    ],
    objectives=[
        "Track round outcomes, score progression, and side-specific context.",
        "Explain how player actions affected the round result, not just who won.",
    ],
    terminology=["attackers", "defenders", "spike", "plant", "defuse", "retake", "eco", "peak"],
)


def normalize_game_key(game_key: str | None) -> str:
    normalized = (game_key or "").strip().lower()
    if normalized in {"gta", "gta5", "gta_like", "grand theft auto", "grand theft auto v"}:
        return "gta"
    if normalized in {"valorant", "val"}:
        return "valorant"
    return "generic"


def get_game_profile(game_key: str | None) -> GameProfile:
    """Return a copy of the static profile for the selected game."""
    normalized = normalize_game_key(game_key)
    if normalized == "gta":
        return _GTA_PROFILE.model_copy(deep=True)
    if normalized == "valorant":
        return _VALORANT_PROFILE.model_copy(deep=True)
    return _GENERIC_PROFILE.model_copy(deep=True)
