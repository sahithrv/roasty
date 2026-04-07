from __future__ import annotations

from pynput import keyboard

from sg_coach.realtime.user_voice import matches_push_to_talk_key, resolve_push_to_talk_key


def test_right_alt_accepts_alt_gr_alias() -> None:
    target = resolve_push_to_talk_key("right alt")

    assert matches_push_to_talk_key(keyboard.Key.alt_r, target)

    alt_gr = getattr(keyboard.Key, "alt_gr", None)
    if alt_gr is not None:
        assert matches_push_to_talk_key(alt_gr, target)


def test_alt_gr_alias_resolves() -> None:
    target = resolve_push_to_talk_key("alt gr")

    assert matches_push_to_talk_key(target, target)
