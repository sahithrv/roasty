from __future__ import annotations

import asyncio
import contextlib
import threading
from dataclasses import dataclass, field
from typing import Any

from sg_coach.shared.logging import get_logger

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - depends on local env
    sd = None  # type: ignore[assignment]

try:
    from pynput import keyboard
except ImportError:  # pragma: no cover - depends on local env
    keyboard = None  # type: ignore[assignment]


logger = get_logger(__name__)


def normalize_audio_device(device_setting: str | None) -> str | int | None:
    """Normalize empty strings and numeric ids for sounddevice."""
    if device_setting is None:
        return None

    normalized = device_setting.strip()
    if not normalized:
        return None
    if normalized.isdigit():
        return int(normalized)
    return normalized


def _equivalent_special_keys(key: Any) -> set[Any]:
    """Return equivalent pynput keys for layouts that alias one physical key."""
    if keyboard is None:
        return {key}

    alt_gr = getattr(keyboard.Key, "alt_gr", None)
    if alt_gr is not None and key in {keyboard.Key.alt_r, alt_gr}:
        return {keyboard.Key.alt_r, alt_gr}
    return {key}


def resolve_push_to_talk_key(key_spec: str) -> Any:
    """Resolve a human-friendly key setting into a pynput key matcher target."""
    if keyboard is None:
        raise RuntimeError("Push-to-talk requires the 'pynput' package to be installed.")

    normalized = key_spec.strip().lower().replace("-", " ").replace("_", " ")
    normalized = " ".join(normalized.split())

    special_keys = {
        "right alt": keyboard.Key.alt_r,
        "alt right": keyboard.Key.alt_r,
        "r alt": keyboard.Key.alt_r,
        "alt r": keyboard.Key.alt_r,
        "altgr": getattr(keyboard.Key, "alt_gr", keyboard.Key.alt_r),
        "alt gr": getattr(keyboard.Key, "alt_gr", keyboard.Key.alt_r),
        "right alt gr": getattr(keyboard.Key, "alt_gr", keyboard.Key.alt_r),
        "right altgr": getattr(keyboard.Key, "alt_gr", keyboard.Key.alt_r),
        "left alt": keyboard.Key.alt_l,
        "alt left": keyboard.Key.alt_l,
        "l alt": keyboard.Key.alt_l,
        "alt l": keyboard.Key.alt_l,
        "right ctrl": keyboard.Key.ctrl_r,
        "ctrl right": keyboard.Key.ctrl_r,
        "r ctrl": keyboard.Key.ctrl_r,
        "ctrl r": keyboard.Key.ctrl_r,
        "left ctrl": keyboard.Key.ctrl_l,
        "ctrl left": keyboard.Key.ctrl_l,
        "l ctrl": keyboard.Key.ctrl_l,
        "ctrl l": keyboard.Key.ctrl_l,
        "right shift": keyboard.Key.shift_r,
        "shift right": keyboard.Key.shift_r,
        "left shift": keyboard.Key.shift_l,
        "shift left": keyboard.Key.shift_l,
        "space": keyboard.Key.space,
        "tab": keyboard.Key.tab,
    }
    if normalized in special_keys:
        return special_keys[normalized]

    if len(normalized) == 1:
        return keyboard.KeyCode.from_char(normalized)

    raise ValueError(
        f"Unsupported push-to-talk key '{key_spec}'. "
        "Use a single key like 'j' or a known modifier like 'right alt'."
    )


def matches_push_to_talk_key(candidate: Any, target: Any) -> bool:
    """Return whether a pynput key event matches the configured target."""
    if keyboard is None:
        return False

    if isinstance(target, keyboard.Key):
        return candidate in _equivalent_special_keys(target)

    target_char = getattr(target, "char", None)
    candidate_char = getattr(candidate, "char", None)
    if isinstance(target_char, str) and isinstance(candidate_char, str):
        return candidate_char.lower() == target_char.lower()
    return False


@dataclass(slots=True)
class PushToTalkController:
    """Global push-to-talk key listener that reports press/release edges."""

    key_spec: str
    loop: asyncio.AbstractEventLoop
    state_queue: asyncio.Queue[bool | None]
    _listener: Any = field(default=None, init=False, repr=False)
    _pressed: bool = field(default=False, init=False, repr=False)
    _target_key: Any = field(default=None, init=False, repr=False)

    def start(self) -> None:
        if keyboard is None:
            raise RuntimeError("Push-to-talk requires the 'pynput' package to be installed.")

        self._target_key = resolve_push_to_talk_key(self.key_spec)
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()
        logger.info("push-to-talk listener started key=%s", self.key_spec)

    async def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
        await self.state_queue.put(None)

    def _on_press(self, key: Any) -> None:
        if self._pressed or not matches_push_to_talk_key(key, self._target_key):
            return
        self._pressed = True
        self.loop.call_soon_threadsafe(self.state_queue.put_nowait, True)

    def _on_release(self, key: Any) -> None:
        if not self._pressed or not matches_push_to_talk_key(key, self._target_key):
            return
        self._pressed = False
        self.loop.call_soon_threadsafe(self.state_queue.put_nowait, False)


@dataclass(slots=True)
class RealtimeMicrophoneInput:
    """Continuous mic stream that only forwards chunks while capture is enabled."""

    sample_rate: int
    device: str | None
    loop: asyncio.AbstractEventLoop
    audio_queue: asyncio.Queue[bytes | None]
    _stream: Any = field(default=None, init=False, repr=False)
    _capture_enabled: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    def start(self) -> None:
        if sd is None:
            raise RuntimeError("Realtime microphone input requires the 'sounddevice' package.")

        normalized_device = normalize_audio_device(self.device)
        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            device=normalized_device,
            callback=self._callback,
        )
        self._stream.start()
        logger.info(
            "realtime microphone started sample_rate=%s device=%s",
            self.sample_rate,
            normalized_device if normalized_device is not None else "default",
        )

    async def stop(self) -> None:
        with contextlib.suppress(Exception):
            if self._stream is not None:
                self._stream.stop()
        with contextlib.suppress(Exception):
            if self._stream is not None:
                self._stream.close()
        await self.audio_queue.put(None)

    def set_capture_enabled(self, enabled: bool) -> None:
        if enabled:
            self._capture_enabled.set()
        else:
            self._capture_enabled.clear()

    def _callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        del frames, time_info

        if status:
            logger.warning("realtime microphone status=%s", status)
        if not self._capture_enabled.is_set():
            return

        chunk = bytes(indata)
        self.loop.call_soon_threadsafe(self.audio_queue.put_nowait, chunk)
