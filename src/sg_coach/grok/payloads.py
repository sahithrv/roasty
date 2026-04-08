from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from sg_coach.shared.game_profiles import GameProfile
from sg_coach.shared.events import CommentaryRequest


def build_system_prompt(persona: str, *, game_profile: GameProfile | None = None) -> str:
    """Return the instruction block for the visual-summary stage.

    This model call is no longer the final voice/personality stage.
    Its job is to:
    - describe what visibly happened in 1-2 short sentences
    - provide one short after-the-fact coaching note
    - stay grounded in the provided event context and replay frames
    - avoid any live tactical or hidden-information guidance
    """
    prompt = (
        "You are the visual-analysis stage for a gaming assistant. "
        "Your job is to explain what visibly happened on screen and provide one short coaching note. "
        "Do not roleplay, do not be sarcastic, and do not speak like the final companion voice. "
        "Use only visible evidence and the supplied structured context. "
        "Use the supplied game profile as authoritative background rules for interpreting the event. "
        "The attached replay images are ordered from oldest to newest and show the lead-up to the event. "
        "Focus on the immediate cause of the event, not just the banner or end-state. "
        "Do not merely restate that the player died or that a banner appeared unless that is all that is visibly knowable. "
        "If uncertain, keep the wording modest and factual. "
        "Do not provide live tactical advice, enemy callouts, timing hints, or hidden information. "
        "Return strict JSON with exactly these keys: "
        '{"visual_summary":"...", "coach_note":"..."} '
        'where "visual_summary" is 1-2 short sentences explaining how the event likely occurred based on the replay sequence, '
        'and "coach_note" is one short after-the-fact improvement note tied to the visible mistake or risk.'
    )
    if game_profile is not None:
        prompt = f"{prompt} Active game: {game_profile.display_name} ({game_profile.mode_name})."
    return prompt


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
        "game_key": request.game_key,
        "game_profile": None if request.game_profile is None else request.game_profile.model_dump(mode="json"),
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
        "context_frame_count": len(request.context_frame_data_urls) or len(request.context_frame_paths),
        "saved_preview_frame_path": request.frame_path,
        "response_rules": {
            "visual_summary_sentences": "1-2",
            "coach_note_lines": 1,
            "tone": "factual and compact",
            "return_json_only": True,
            "no_live_tactics": True,
            "no_hidden_information": True,
        },
    }


def _image_path_to_data_url(image_path: str) -> str:
    """Encode a local JPG/PNG as a data URL for xAI multimodal chat input."""
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type not in {"image/jpeg", "image/png"}:
        mime_type = "image/jpeg"

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_grok_chat_payload(request: CommentaryRequest, *, model: str) -> dict[str, Any]:
    """Build a multimodal chat payload for Grok with real replay images attached."""
    user_payload = build_user_payload(request)
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": json.dumps(user_payload, indent=2, default=str),
        }
    ]

    if request.context_frame_data_urls:
        for data_url in request.context_frame_data_urls:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url,
                        "detail": "high",
                    },
                }
            )
    else:
        for frame_path in request.context_frame_paths:
            path = Path(frame_path)
            if not path.exists() or not path.is_file():
                continue

            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _image_path_to_data_url(frame_path),
                        "detail": "high",
                    },
                }
            )

    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": build_system_prompt(request.persona, game_profile=request.game_profile),
            },
            {
                "role": "user",
                "content": content,
            },
        ],
    }


def sanitize_grok_chat_payload_for_debug(
    payload: dict[str, Any],
    *,
    context_frame_paths: list[str],
) -> dict[str, Any]:
    """Return a readable debug copy without embedding raw base64 image blobs."""
    debug_payload = json.loads(json.dumps(payload))
    image_index = 0

    for message in debug_payload.get("messages", []):
        content = message.get("content")
        if not isinstance(content, list):
            continue

        for item in content:
            if item.get("type") != "image_url":
                continue

            source_path = None
            if image_index < len(context_frame_paths):
                source_path = context_frame_paths[image_index]
            image_index += 1

            item["image_url"] = {
                "url": "<inline image omitted from debug dump>",
                "detail": item.get("image_url", {}).get("detail", "high"),
                "source_path": source_path,
            }

    return debug_payload


def parse_structured_commentary_output(raw_text: str) -> tuple[str, str]:
    """Parse the model's JSON response into summary + coaching text.

    If the model returns malformed output, we still try to salvage something
    readable so the pipeline remains usable during prompt iteration.
    """
    raw_text = raw_text.strip()
    if not raw_text:
        return "", ""

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        # Fallback: treat the whole response as summary text if JSON parsing fails.
        return raw_text, ""

    if not isinstance(parsed, dict):
        return raw_text, ""

    visual_summary = parsed.get("visual_summary", "")
    coach_note = parsed.get("coach_note", "")

    if not isinstance(visual_summary, str):
        visual_summary = str(visual_summary)
    if not isinstance(coach_note, str):
        coach_note = str(coach_note)

    return visual_summary.strip(), coach_note.strip()
