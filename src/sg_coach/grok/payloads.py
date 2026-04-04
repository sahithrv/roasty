from __future__ import annotations

import json
from typing import Any

from sg_coach.shared.events import CommentaryRequest


def build_system_prompt(persona: str) -> str:
    """Return the high-level Grok instruction block for this project.

    The prompt is intentionally short and strict:
    - be funny and sarcastic
    - you are allowed to curse
    - stay focused on visible outcomes
    - do not provide tactical live gameplay advice
    """
    persona_style = {
        "light_sarcasm": "dry, lightly sarcastic",
        "unhinged": "chaotic, will roast the user personally, curse a lot",
        "coach_roast": "mild coaching after the fact with roast energy",
        "commentary_only": "sports-commentator style with humor but no coaching",
    }.get(persona, "dry, sarcastic, concise")

    return (
        "You are Sarcastic Gaming AI Companion. "
        f"Persona style: {persona_style}. "
        "Write some dialogue reacting to the player's visible outcome. "
        "Use session memory when it improves the joke. "
        "Do not provide tactical gameplay advice, enemy callouts, timing hints, or hidden information. "
        "Do not tell the player what to do next. "
        "Prefer callback humor over generic insults. "
        "Keep the line concise."
    )


def build_user_payload(request: CommentaryRequest) -> dict[str, Any]:
    """Build the structured user payload we want Grok to reason over.

    This is the actual semantic input for commentary generation:
    - what just happened
    - what happened recently
    - which patterns are repeating
    - which local replay frames are available as context
    """
    latest_event = request.latest_event
    return {
        "persona": request.persona,
        "latest_event": {
            "event_type": latest_event.event_type,
            "confidence": latest_event.confidence,
            "tags": latest_event.tags,
            "metadata": latest_event.metadata,
            "frame_path": latest_event.frame_path,
            "clip_path": latest_event.clip_path,
        },
        "recent_events": [
            {
                "event_type": event.event_type,
                "confidence": event.confidence,
                "tags": event.tags,
                "metadata": event.metadata,
            }
            for event in request.recent_events
        ],
        "counters": request.counters,
        "callback_candidates": request.callback_candidates,
        "memory_summary": request.memory_summary,
        "context_frame_paths": request.context_frame_paths,
        "response_rules": {
            "max_sentences": 2,
            "tone": "sarcastic and funny",
            "no_live_tactics": True,
            "no_hidden_information": True,
        },
    }


def build_grok_chat_payload(request: CommentaryRequest, *, model: str) -> dict[str, Any]:
    """Build a chat-style payload that mirrors what we would send to Grok."""
    user_payload = build_user_payload(request)
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": build_system_prompt(request.persona),
            },
            {
                "role": "user",
                "content": json.dumps(user_payload, indent=2, default=str),
            },
        ],
    }
