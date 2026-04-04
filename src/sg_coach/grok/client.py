from __future__ import annotations

import json
from dataclasses import dataclass
from urllib import error, request

from sg_coach.shared.settings import Settings


@dataclass(slots=True)
class GrokChatClient:
    """Small xAI chat-completions client using the standard library."""

    api_key: str
    base_url: str = "https://api.x.ai/v1"
    timeout_seconds: int = 60

    @classmethod
    def from_settings(cls, settings: Settings) -> "GrokChatClient":
        if not settings.grok_api_key:
            raise RuntimeError("SG_GROK_API_KEY is not set.")
        return cls(
            api_key=settings.grok_api_key.strip(),
            base_url=settings.grok_api_base_url.rstrip("/"),
            timeout_seconds=settings.grok_timeout_seconds,
        )

    def create_chat_completion(self, payload: dict) -> dict:
        """Send one chat completion request to xAI."""
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            url=f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"xAI chat completion failed status={exc.code} body={error_body}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"xAI chat completion request failed: {exc.reason}") from exc

    def extract_text(self, response_json: dict) -> str:
        """Extract assistant text from the OpenAI-compatible response shape."""
        choices = response_json.get("choices") or []
        if not choices:
            return ""

        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        text_parts.append(text.strip())
            return "\n".join(part for part in text_parts if part)

        return ""
