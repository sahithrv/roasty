from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any

try:
    import websockets
except ImportError:  # pragma: no cover - depends on local env
    websockets = None  # type: ignore[assignment]


@dataclass(slots=True)
class RealtimeResponseEvent:
    """Lifecycle event for one realtime assistant response turn."""

    response_id: str
    source: str
    phase: str


@dataclass(slots=True)
class XaiRealtimeClient:
    """Minimal xAI Voice Agent WebSocket client for event-driven text turns."""

    api_key: str
    ws_url: str
    instructions: str
    voice: str = "Leo"
    language: str = "en"
    input_sample_rate: int = 24000
    output_sample_rate: int = 24000
    raw_event_queue: asyncio.Queue[dict[str, Any] | None] = field(
        default_factory=asyncio.Queue,
        init=False,
        repr=False,
    )
    text_queue: asyncio.Queue[str | None] = field(
        default_factory=asyncio.Queue,
        init=False,
        repr=False,
    )
    audio_queue: asyncio.Queue[bytes | None] = field(
        default_factory=asyncio.Queue,
        init=False,
        repr=False,
    )
    user_transcript_queue: asyncio.Queue[str | None] = field(
        default_factory=asyncio.Queue,
        init=False,
        repr=False,
    )
    response_event_queue: asyncio.Queue[RealtimeResponseEvent | None] = field(
        default_factory=asyncio.Queue,
        init=False,
        repr=False,
    )
    _ws: Any = field(default=None, init=False, repr=False)
    _receive_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _response_parts: dict[str, list[str]] = field(default_factory=dict, init=False, repr=False)
    _response_sources: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _pending_response_sources: deque[str] = field(default_factory=deque, init=False, repr=False)
    _emitted_response_keys: set[str] = field(default_factory=set, init=False, repr=False)
    _session_ready: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)
    _idle_event: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)
    _current_response_source: str | None = field(default=None, init=False, repr=False)
    _send_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    async def connect(self) -> None:
        """Open the WebSocket and configure the voice session."""
        if websockets is None:
            raise RuntimeError("Realtime bridge requires the 'websockets' package to be installed.")

        self._idle_event.set()
        self._ws = await websockets.connect(
            self.ws_url,
            additional_headers={"Authorization": f"Bearer {self.api_key.strip()}"},
            max_size=20 * 1024 * 1024,
        )
        await self._send_json(
            {
                "type": "session.update",
                "session": {
                    "instructions": self.instructions,
                    "voice": self.voice.lower(),
                    "language": self.language,
                    "turn_detection": None,
                    "audio": {
                        "input": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": self.input_sample_rate,
                            }
                        },
                        "output": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": self.output_sample_rate,
                            }
                        }
                    },
                },
            }
        )
        self._receive_task = asyncio.create_task(self._receive_loop())
        session_ready = await asyncio.wait_for(self._session_ready.wait(), timeout=15)
        if not session_ready:
            raise RuntimeError("Timed out waiting for realtime session.updated.")

    async def close(self) -> None:
        """Close the socket and stop the receive tasks cleanly."""
        if self._receive_task is not None:
            self._receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._receive_task

        if self._ws is not None:
            await self._ws.close()

        await self.raw_event_queue.put(None)
        await self.text_queue.put(None)
        await self.audio_queue.put(None)
        await self.user_transcript_queue.put(None)
        await self.response_event_queue.put(None)

    async def send_event_text(self, text: str) -> None:
        """Inject a game-event text message and ask the voice agent to respond."""
        await self._send_json(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )
        await self._request_response(source="system_event")

    async def send_memory_text(self, text: str) -> None:
        """Inject silent memory/context into the session without forcing a reply."""
        await self._send_json(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )

    async def append_input_audio(self, audio_bytes: bytes) -> None:
        """Append one PCM16 mono chunk to the realtime input audio buffer."""
        if not audio_bytes:
            return
        await self._send_json(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio_bytes).decode("ascii"),
            }
        )

    async def clear_input_audio(self) -> None:
        """Clear any buffered user audio that has not been committed yet."""
        await self._send_json({"type": "input_audio_buffer.clear"})

    async def commit_input_audio(self) -> None:
        """Commit buffered user audio and request one assistant response."""
        await self._send_json({"type": "input_audio_buffer.commit"})
        await self._request_response(source="user_speech")

    async def wait_until_idle(self) -> None:
        """Wait until no assistant response turn is currently in progress."""
        await self._idle_event.wait()

    def is_response_active(self) -> bool:
        """Return whether the assistant is currently producing a response."""
        return not self._idle_event.is_set()

    @property
    def current_response_source(self) -> str | None:
        """Return the local source label for the in-flight assistant response, if any."""
        return self._current_response_source

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("Realtime client is not connected.")
        async with self._send_lock:
            await self._ws.send(json.dumps(payload))

    async def _request_response(self, *, source: str) -> None:
        self._pending_response_sources.append(source)
        await self._send_json({"type": "response.create"})

    async def _receive_loop(self) -> None:
        assert self._ws is not None

        try:
            async for raw_message in self._ws:
                event = json.loads(raw_message)
                await self.raw_event_queue.put(event)
                await self._maybe_emit_text(event)
        except asyncio.CancelledError:
            raise
        finally:
            self._session_ready.set()
            self._idle_event.set()
            await self.raw_event_queue.put(None)
            await self.text_queue.put(None)
            await self.audio_queue.put(None)
            await self.user_transcript_queue.put(None)
            await self.response_event_queue.put(None)

    async def _maybe_emit_text(self, event: dict[str, Any]) -> None:
        event_type = event.get("type", "")
        response_key = self._response_key(event)

        if event_type == "session.updated":
            self._session_ready.set()
            return

        if event_type == "conversation.item.input_audio_transcription.completed":
            transcript = str(event.get("transcript", "")).strip()
            if transcript:
                await self.user_transcript_queue.put(transcript)
            return

        if event_type in {"response.text.delta", "response.output_audio_transcript.delta"}:
            delta = str(event.get("delta", ""))
            if delta:
                self._response_parts.setdefault(response_key, []).append(delta)
            return

        if event_type == "response.output_audio.delta":
            delta_b64 = str(event.get("delta", ""))
            if delta_b64:
                await self.audio_queue.put(base64.b64decode(delta_b64))
            return

        if event_type == "response.text.done":
            text = str(event.get("text", "")).strip()
            await self._emit_text_once(response_key, text)
            return

        if event_type == "response.output_audio_transcript.done":
            transcript = str(event.get("transcript", "")).strip()
            await self._emit_text_once(response_key, transcript)
            return

        if event_type == "response.created":
            source = self._pending_response_sources.popleft() if self._pending_response_sources else "unknown"
            self._response_sources[response_key] = source
            self._current_response_source = source
            self._idle_event.clear()
            await self.response_event_queue.put(
                RealtimeResponseEvent(response_id=response_key, source=source, phase="created")
            )
            return

        if event_type == "response.done":
            text = self._extract_text_from_response_done(event, response_key)
            await self._emit_text_once(response_key, text)
            source = self._response_sources.pop(response_key, "unknown")
            self._current_response_source = None
            self._idle_event.set()
            await self.response_event_queue.put(
                RealtimeResponseEvent(response_id=response_key, source=source, phase="done")
            )

    async def _emit_text_once(self, response_key: str, text: str) -> None:
        """Queue one final assistant turn per response id.

        The realtime API may surface the same turn through multiple completion
        events (`response.text.done`, `response.output_audio_transcript.done`,
        and `response.done`). We only want one downstream assistant turn.
        """
        text = text.strip()
        if not text:
            return
        if response_key in self._emitted_response_keys:
            return

        self._emitted_response_keys.add(response_key)
        self._response_parts.pop(response_key, None)
        await self.text_queue.put(text)

    def _response_key(self, event: dict[str, Any]) -> str:
        response = event.get("response")
        if isinstance(response, dict) and isinstance(response.get("id"), str):
            return response["id"]
        if isinstance(event.get("response_id"), str):
            return str(event["response_id"])
        if isinstance(event.get("item_id"), str):
            return str(event["item_id"])
        return "default"

    def _extract_text_from_response_done(self, event: dict[str, Any], response_key: str) -> str:
        response = event.get("response", {})
        if isinstance(response, dict):
            output = response.get("output", [])
            if isinstance(output, list):
                parts: list[str] = []
                for item in output:
                    if not isinstance(item, dict):
                        continue
                    content = item.get("content", [])
                    if not isinstance(content, list):
                        continue
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        text = block.get("text") or block.get("transcript")
                        if isinstance(text, str) and text.strip():
                            parts.append(text.strip())
                if parts:
                    return " ".join(parts)

        partial = "".join(self._response_parts.pop(response_key, [])).strip()
        return partial
