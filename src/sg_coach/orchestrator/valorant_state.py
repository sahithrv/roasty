from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sg_coach.orchestrator.topics import ROUND_PACKET_READY, UI_STATE
from sg_coach.shared.events import GameEvent
from sg_coach.shared.game_profiles import (
    ValorantLiveRoundState,
    ValorantRoundKeyMoment,
    ValorantRoundPacket,
    normalize_game_key,
)
from sg_coach.shared.logging import get_logger
from sg_coach.shared.streaming import (
    EVENT_STREAM_COMPLETE,
    ROUND_PACKET_STREAM_COMPLETE,
    UI_STREAM_COMPLETE,
)


if TYPE_CHECKING:
    from sg_coach.orchestrator.session import SessionRuntime


logger = get_logger(__name__)

_ROUND_START_EVENT_TYPES = frozenset(
    {
        "round_started",
        "valorant_round_started",
        "buy_phase_started",
        "valorant_buy_phase_started",
    }
)
_PLAYER_KILL_EVENT_TYPES = frozenset({"player_kill", "valorant_player_kill"})
_PLAYER_ASSIST_EVENT_TYPES = frozenset({"player_assist", "valorant_player_assist"})
_PLAYER_DEATH_EVENT_TYPES = frozenset({"player_died", "valorant_player_died", "death"})
_ROUND_END_EVENT_TYPES = frozenset(
    {
        "round_won",
        "round_lost",
        "round_ended",
        "valorant_round_won",
        "valorant_round_lost",
        "valorant_round_ended",
        "spike_defused",
        "spike_exploded",
        "valorant_spike_defused",
        "valorant_spike_exploded",
    }
)
_NOTABLE_ROUND_EVENT_TYPES = frozenset(
    {
        *_PLAYER_KILL_EVENT_TYPES,
        *_PLAYER_ASSIST_EVENT_TYPES,
        *_PLAYER_DEATH_EVENT_TYPES,
        "spike_planted",
        "spike_defused",
        "spike_exploded",
        "valorant_spike_planted",
        "valorant_spike_defused",
        "valorant_spike_exploded",
        "round_won",
        "round_lost",
        "round_ended",
        "valorant_round_won",
        "valorant_round_lost",
        "valorant_round_ended",
    }
)


@dataclass(slots=True)
class ValorantTrackerUpdate:
    """One stateful tracker step result."""

    live_state: ValorantLiveRoundState
    round_packet: ValorantRoundPacket | None = None


@dataclass(slots=True)
class ValorantMatchStateTracker:
    """Own live round state and convert Valorant game events into round packets."""

    session_id: str
    current_state: ValorantLiveRoundState = field(default_factory=ValorantLiveRoundState)
    _current_round_event_ids: list[str] = field(default_factory=list, init=False, repr=False)
    _current_key_moments: list[ValorantRoundKeyMoment] = field(default_factory=list, init=False, repr=False)
    _moment_sequence: int = field(default=0, init=False, repr=False)

    def consume_event(self, event: GameEvent) -> ValorantTrackerUpdate:
        """Update live round state from one event and optionally emit a round packet."""
        if normalize_game_key(event.game) != "valorant":
            raise ValueError(f"Tracker only accepts Valorant events, got game={event.game!r}")

        self._roll_round_if_event_advances_round(event)

        if event.event_type in _ROUND_START_EVENT_TYPES:
            self._begin_round_from_event(event)
        else:
            self._apply_common_metadata(event)

        self._apply_event_semantics(event)
        self._record_round_event(event)

        live_state_snapshot = self.current_state.model_copy(deep=True)
        round_packet: ValorantRoundPacket | None = None
        if self._event_closes_round(event):
            round_packet = self._build_round_packet(event)
            self._transition_to_between_rounds(round_packet)

        return ValorantTrackerUpdate(
            live_state=live_state_snapshot,
            round_packet=round_packet,
        )

    def _roll_round_if_event_advances_round(self, event: GameEvent) -> None:
        """Reset round-local fields if a new round number appears before an explicit close."""
        incoming_round_number = self._metadata_int(event.metadata, "round_number")
        if incoming_round_number is None or self.current_state.round_number is None:
            return
        if incoming_round_number == self.current_state.round_number:
            return

        logger.info(
            "valorant tracker advanced round without explicit close previous_round=%s incoming_round=%s",
            self.current_state.round_number,
            incoming_round_number,
        )
        self._reset_round_local_fields()
        self.current_state.round_number = incoming_round_number

    def _begin_round_from_event(self, event: GameEvent) -> None:
        """Start a new round shell using round-start metadata."""
        self._reset_round_local_fields()
        self._apply_common_metadata(event)
        self.current_state.round_phase = str(event.metadata.get("round_phase") or "buy_phase")
        self.current_state.spike_state = "unplanted"
        self.current_state.player_alive = event.metadata.get("player_alive", True)

    def _apply_common_metadata(self, event: GameEvent) -> None:
        """Copy shared structured metadata into the mutable live round state."""
        metadata = event.metadata

        for field_name in (
            "map_name",
            "player_side",
            "round_phase",
            "spike_state",
        ):
            value = metadata.get(field_name)
            if isinstance(value, str) and value:
                setattr(self.current_state, field_name, value)

        round_number = self._metadata_int(metadata, "round_number")
        if round_number is not None:
            self.current_state.round_number = round_number

        team_score = self._metadata_int(metadata, "team_score")
        if team_score is not None:
            self.current_state.team_score = team_score

        opponent_score = self._metadata_int(metadata, "opponent_score")
        if opponent_score is not None:
            self.current_state.opponent_score = opponent_score

        player_alive = metadata.get("player_alive")
        if isinstance(player_alive, bool):
            self.current_state.player_alive = player_alive

        self.current_state.player_match_kills = self._coalesce_optional_int(
            metadata,
            "player_match_kills",
            current=self.current_state.player_match_kills,
        )
        self.current_state.player_match_deaths = self._coalesce_optional_int(
            metadata,
            "player_match_deaths",
            current=self.current_state.player_match_deaths,
        )
        self.current_state.player_match_assists = self._coalesce_optional_int(
            metadata,
            "player_match_assists",
            current=self.current_state.player_match_assists,
        )

    def _apply_event_semantics(self, event: GameEvent) -> None:
        """Apply one event's game-specific effect to the current live round state."""
        event_type = event.event_type

        if event_type in _PLAYER_KILL_EVENT_TYPES:
            self.current_state.player_round_kills += 1
            self.current_state.notable_round_events.append(event_type)
            self.current_state.player_alive = True if self.current_state.player_alive is None else self.current_state.player_alive
        elif event_type in _PLAYER_ASSIST_EVENT_TYPES:
            self.current_state.player_round_assists += 1
            self.current_state.notable_round_events.append(event_type)
        elif event_type in _PLAYER_DEATH_EVENT_TYPES:
            self.current_state.player_died_this_round = True
            self.current_state.player_alive = False
            self.current_state.notable_round_events.append(event_type)
        elif event_type in {"spike_planted", "valorant_spike_planted"}:
            self.current_state.spike_state = "planted"
            self.current_state.round_phase = "postplant"
            self.current_state.notable_round_events.append(event_type)
        elif event_type in {"spike_defused", "valorant_spike_defused"}:
            self.current_state.spike_state = "defused"
            self.current_state.round_phase = "ended"
            self.current_state.notable_round_events.append(event_type)
        elif event_type in {"spike_exploded", "valorant_spike_exploded"}:
            self.current_state.spike_state = "exploded"
            self.current_state.round_phase = "ended"
            self.current_state.notable_round_events.append(event_type)
        elif event_type in _ROUND_END_EVENT_TYPES:
            self.current_state.round_phase = "ended"
            self.current_state.notable_round_events.append(event_type)

        if event_type in _NOTABLE_ROUND_EVENT_TYPES:
            self._append_key_moment(event)

    def _append_key_moment(self, event: GameEvent) -> None:
        self._moment_sequence += 1
        self._current_key_moments.append(
            ValorantRoundKeyMoment(
                sequence=self._moment_sequence,
                event_type=event.event_type,
                description=self._describe_round_moment(event),
                metadata=self._slim_metadata(event.metadata),
            )
        )

    def _describe_round_moment(self, event: GameEvent) -> str:
        metadata = event.metadata
        if event.event_type in _PLAYER_KILL_EVENT_TYPES:
            site = metadata.get("site")
            if self.current_state.round_phase == "postplant" and self.current_state.player_side == "attacker":
                return "Player got a kill defending the planted spike. " if not site else f"Player got a kill defending the planted spike around {site}."
            elif self.current_state.round_phase == "postplant" and self.current_state.player_side == "defender":
                return "Player got a kill retaking the site. " if not site else f"Player got a kill retaking the {site}."
            else:
                return "Player got a kill." if not site else f"Player got a kill around {site}."
        if event.event_type in _PLAYER_ASSIST_EVENT_TYPES:
            return "Player contributed an assist."
        if event.event_type in _PLAYER_DEATH_EVENT_TYPES:
            return "Player died during the round."
        if event.event_type in {"spike_planted", "valorant_spike_planted"}:
            site = metadata.get("site")
            return "Spike was planted." if not site else f"Spike was planted at {site}."
        if event.event_type in {"spike_defused", "valorant_spike_defused"}:
            return "Spike was defused to end the round."
        if event.event_type in {"spike_exploded", "valorant_spike_exploded"}:
            return "Spike detonated to end the round."
        if event.event_type in {"round_won", "valorant_round_won"}:
            return "Player team won the round."
        if event.event_type in {"round_lost", "valorant_round_lost"}:
            return "Player team lost the round."
        return event.event_type.replace("_", " ")

    def _record_round_event(self, event: GameEvent) -> None:
        if event.event_id not in self._current_round_event_ids:
            self._current_round_event_ids.append(event.event_id)

    def _event_closes_round(self, event: GameEvent) -> bool:
        return event.event_type in _ROUND_END_EVENT_TYPES or bool(event.metadata.get("round_closed"))

    def _build_round_packet(self, event: GameEvent) -> ValorantRoundPacket:
        """Freeze the current live state into one structured round packet.

        TODO for later:
        The next big upgrade here is smarter causal synthesis. Nice targets:
        - convert ordered moments into stronger `end_sequence`
        - infer `player_enabled_win` and `player_enabled_loss` more intelligently
        - populate richer `causal_context` from combinations of kills, plant, and end reason
        """
        round_winner = self._resolve_round_winner(event)
        team_score_after = self._coalesce_optional_int(event.metadata, "team_score", current=self.current_state.team_score)
        opponent_score_after = self._coalesce_optional_int(
            event.metadata,
            "opponent_score",
            current=self.current_state.opponent_score,
        )
        team_score_before, opponent_score_before = self._infer_score_before(
            event=event,
            round_winner=round_winner,
            team_score_after=team_score_after,
            opponent_score_after=opponent_score_after,
        )
        round_end_reason = self._resolve_round_end_reason(event)
        player_survived = None if self.current_state.player_alive is None else not self.current_state.player_died_this_round

        packet = ValorantRoundPacket(
            session_id=self.session_id,
            map_name=self.current_state.map_name,
            round_number=self.current_state.round_number,
            player_side=self.current_state.player_side,
            team_score_before=team_score_before,
            opponent_score_before=opponent_score_before,
            team_score_after=team_score_after,
            opponent_score_after=opponent_score_after,
            round_winner=round_winner,
            round_end_reason=round_end_reason,
            spike_planted=self._infer_spike_planted(event=event, round_end_reason=round_end_reason),
            spike_defused=self._bool_or_none(event.metadata, "spike_defused", fallback=round_end_reason == "spike_defused"),
            spike_exploded=self._bool_or_none(event.metadata, "spike_exploded", fallback=round_end_reason == "spike_exploded"),
            player_kills=self.current_state.player_round_kills,
            player_assists=self.current_state.player_round_assists,
            player_died=self.current_state.player_died_this_round,
            player_survived=player_survived,
            player_match_kills_after=self.current_state.player_match_kills,
            player_match_deaths_after=self.current_state.player_match_deaths,
            player_match_assists_after=self.current_state.player_match_assists,
            first_blood=self._string_or_none(event.metadata, "first_blood"),
            key_moments=[moment.model_copy(deep=True) for moment in self._current_key_moments],
            end_sequence=[moment.event_type for moment in self._current_key_moments[-3:]],
            player_impact_tags=self._derive_player_impact_tags(round_winner, round_end_reason),
            player_enabled_win=self._player_enabled_result(round_winner=round_winner, desired="player_team"),
            player_enabled_loss=self._player_enabled_result(round_winner=round_winner, desired="opponents"),
            swing_moment=self._derive_swing_moment(),
            causal_context=self._derive_causal_context(round_winner, round_end_reason),
            source_event_ids=list(self._current_round_event_ids),
            confidence=max(event.confidence, 0.35),
        )
        return packet

    def _derive_player_impact_tags(self, round_winner: str | None, round_end_reason: str) -> list[str]:
        tags: list[str] = []
        if self.current_state.player_round_kills > 0:
            tags.append("frag_impact")
        if self.current_state.player_round_kills >= 2:
            tags.append("multi_kill_round")
        if round_end_reason == "spike_defused":
            tags.append("retake_context")
        if round_end_reason == "spike_exploded":
            tags.append("postplant_context")
        if round_winner == "player_team" and self.current_state.player_round_kills > 0:
            tags.append("contributed_to_round_win")
        return tags

    def _derive_swing_moment(self) -> str | None:
        if not self._current_key_moments:
            return None
        if self.current_state.player_round_kills >= 2:
            return "player_multi_kill_shifted_round"
        return self._current_key_moments[-1].event_type

    def _derive_causal_context(self, round_winner: str | None, round_end_reason: str) -> list[str]:
        context: list[str] = []
        if self.current_state.player_round_kills:
            context.append(f"player_recorded_{self.current_state.player_round_kills}_kills")
        if self.current_state.player_round_assists:
            context.append(f"player_recorded_{self.current_state.player_round_assists}_assists")
        if round_end_reason != "unknown":
            context.append(f"round_ended_by_{round_end_reason}")
        if round_winner:
            context.append(f"round_winner_{round_winner}")
        return context

    def _player_enabled_result(self, *, round_winner: str | None, desired: str) -> bool | None:
        if round_winner != desired:
            return False if round_winner is not None else None
        return True if self.current_state.player_round_kills > 0 or self.current_state.player_round_assists > 0 else None

    def _resolve_round_winner(self, event: GameEvent) -> str | None:
        metadata_winner = self._string_or_none(event.metadata, "round_winner")
        if metadata_winner:
            normalized = metadata_winner.lower()
            if normalized in {"player_team", "team", "ally", "allies", "attackers", "defenders"}:
                if normalized in {"attackers", "defenders"} and self.current_state.player_side:
                    return "player_team" if normalized == self.current_state.player_side.lower() else "opponents"
                return "player_team"
            if normalized in {"enemy", "enemies", "opponents"}:
                return "opponents"

        if event.event_type in {"round_won", "valorant_round_won"}:
            return "player_team"
        if event.event_type in {"round_lost", "valorant_round_lost"}:
            return "opponents"
        if self.current_state.player_side:
            player_side = self.current_state.player_side.lower()
            round_end_reason = self._resolve_round_end_reason(event)
            if round_end_reason == "spike_exploded":
                return "player_team" if player_side == "attackers" else "opponents"
            if round_end_reason == "spike_defused":
                return "player_team" if player_side == "defenders" else "opponents"
        return None

    def _resolve_round_end_reason(self, event: GameEvent) -> str:
        metadata_reason = self._string_or_none(event.metadata, "round_end_reason")
        if metadata_reason:
            return metadata_reason
        if event.event_type in {"spike_defused", "valorant_spike_defused"}:
            return "spike_defused"
        if event.event_type in {"spike_exploded", "valorant_spike_exploded"}:
            return "spike_exploded"
        if event.event_type in {"round_won", "round_lost", "round_ended", "valorant_round_won", "valorant_round_lost", "valorant_round_ended"}:
            return "elimination_or_timeout"
        return "unknown"

    def _infer_score_before(
        self,
        *,
        event: GameEvent,
        round_winner: str | None,
        team_score_after: int | None,
        opponent_score_after: int | None,
    ) -> tuple[int | None, int | None]:
        team_score_before = self._metadata_int(event.metadata, "team_score_before")
        opponent_score_before = self._metadata_int(event.metadata, "opponent_score_before")
        if team_score_before is not None or opponent_score_before is not None:
            return team_score_before, opponent_score_before

        if round_winner == "player_team" and team_score_after is not None:
            return team_score_after - 1, opponent_score_after
        if round_winner == "opponents" and opponent_score_after is not None:
            return team_score_after, opponent_score_after - 1
        return team_score_after, opponent_score_after

    def _transition_to_between_rounds(self, round_packet: ValorantRoundPacket) -> None:
        """Reset round-local state while keeping match-level context alive."""
        self.current_state = ValorantLiveRoundState(
            map_name=round_packet.map_name,
            round_number=round_packet.round_number,
            player_side=round_packet.player_side,
            team_score=round_packet.team_score_after,
            opponent_score=round_packet.opponent_score_after,
            round_phase="between_rounds",
            spike_state="unplanted",
            player_alive=None,
            player_match_kills=round_packet.player_match_kills_after,
            player_match_deaths=round_packet.player_match_deaths_after,
            player_match_assists=round_packet.player_match_assists_after,
            player_round_kills=0,
            player_round_assists=0,
            player_died_this_round=False,
            notable_round_events=[],
            inference_notes=[],
            missing_fields=[],
        )
        self._current_round_event_ids.clear()
        self._current_key_moments.clear()
        self._moment_sequence = 0

    def _reset_round_local_fields(self) -> None:
        self.current_state.player_round_kills = 0
        self.current_state.player_round_assists = 0
        self.current_state.player_died_this_round = False
        self.current_state.round_phase = "unknown"
        self.current_state.spike_state = "unknown"
        self.current_state.player_alive = None
        self.current_state.notable_round_events = []
        self.current_state.inference_notes = []
        self.current_state.missing_fields = []
        self._current_round_event_ids.clear()
        self._current_key_moments.clear()
        self._moment_sequence = 0

    def _coalesce_optional_int(self, metadata: dict[str, Any], key: str, *, current: int | None) -> int | None:
        parsed = self._metadata_int(metadata, key)
        return current if parsed is None else parsed

    def _metadata_int(self, metadata: dict[str, Any], key: str) -> int | None:
        value = metadata.get(key)
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                return int(stripped)
        return None

    def _slim_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "site",
            "round_number",
            "player_side",
            "team_score",
            "opponent_score",
            "round_end_reason",
            "first_blood",
        )
        return {key: metadata[key] for key in keys if key in metadata}

    def _string_or_none(self, metadata: dict[str, Any], key: str) -> str | None:
        value = metadata.get(key)
        return value if isinstance(value, str) and value else None

    def _bool_or_none(self, metadata: dict[str, Any], key: str, *, fallback: bool | None = None) -> bool | None:
        value = metadata.get(key)
        if isinstance(value, bool):
            return value
        return fallback

    def _infer_spike_planted(self, *, event: GameEvent, round_end_reason: str) -> bool | None:
        explicit = self._bool_or_none(event.metadata, "spike_planted")
        if explicit is not None:
            return explicit
        if round_end_reason in {"spike_defused", "spike_exploded"}:
            return True
        if self.current_state.spike_state in {"planted", "defused", "exploded"}:
            return True
        if any(
            event_type in {"spike_planted", "valorant_spike_planted"}
            for event_type in self.current_state.notable_round_events
        ):
            return True
        return None


def format_live_round_state_line(state: ValorantLiveRoundState) -> str:
    """Return a compact log line for live Valorant round state."""
    return (
        f"map={state.map_name or 'unknown'} "
        f"round={state.round_number} "
        f"side={state.player_side or 'unknown'} "
        f"score={state.team_score}-{state.opponent_score} "
        f"phase={state.round_phase} "
        f"spike={state.spike_state} "
        f"alive={state.player_alive} "
        f"round_kills={state.player_round_kills} "
        f"round_assists={state.player_round_assists} "
        f"died={state.player_died_this_round} "
        f"events={state.notable_round_events}"
    )


def format_round_packet_line(packet: ValorantRoundPacket) -> str:
    """Return a compact log line for one closed Valorant round."""
    return (
        f"round={packet.round_number} "
        f"side={packet.player_side or 'unknown'} "
        f"winner={packet.round_winner or 'unknown'} "
        f"reason={packet.round_end_reason} "
        f"score_before={packet.team_score_before}-{packet.opponent_score_before} "
        f"score_after={packet.team_score_after}-{packet.opponent_score_after} "
        f"player_kills={packet.player_kills} "
        f"player_assists={packet.player_assists} "
        f"player_died={packet.player_died} "
        f"impact={packet.player_impact_tags}"
    )


async def valorant_state_worker(
    runtime: SessionRuntime,
    event_queue: asyncio.Queue[GameEvent | str],
    *,
    tracker: ValorantMatchStateTracker,
) -> None:
    """Consume Valorant events, maintain live state, and emit finished round packets."""
    while True:
        item = await event_queue.get()
        if item == EVENT_STREAM_COMPLETE:
            await runtime.publish(UI_STATE, UI_STREAM_COMPLETE)
            await runtime.publish(ROUND_PACKET_READY, ROUND_PACKET_STREAM_COMPLETE)
            logger.info("valorant state worker stream complete")
            return

        event = item
        if normalize_game_key(event.game) != "valorant":
            continue

        update = tracker.consume_event(event)
        await runtime.publish(UI_STATE, update.live_state)
        logger.info(
            "valorant live state updated event_type=%s %s",
            event.event_type,
            format_live_round_state_line(update.live_state),
        )

        if update.round_packet is not None:
            await runtime.publish(ROUND_PACKET_READY, update.round_packet)
            logger.info(
                "valorant round packet ready event_type=%s %s",
                event.event_type,
                format_round_packet_line(update.round_packet),
            )


async def valorant_live_state_sink(
    state_queue: asyncio.Queue[ValorantLiveRoundState | str],
) -> None:
    """Log live Valorant round states for pipeline inspection."""
    while True:
        item = await state_queue.get()
        if item == UI_STREAM_COMPLETE:
            logger.info("valorant live state stream complete")
            return
        logger.info("valorant live state %s", format_live_round_state_line(item))


async def valorant_round_packet_sink(
    packet_queue: asyncio.Queue[ValorantRoundPacket | str],
) -> None:
    """Log completed Valorant round packets for pipeline inspection."""
    while True:
        item = await packet_queue.get()
        if item == ROUND_PACKET_STREAM_COMPLETE:
            logger.info("valorant round packet stream complete")
            return
        logger.info("valorant round packet %s", format_round_packet_line(item))
