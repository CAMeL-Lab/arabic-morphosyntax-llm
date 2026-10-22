"""Provider-neutral request and response records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class GenerationRequest:
    system_prompt: str
    messages: list[ChatMessage]
    schema: dict[str, Any]
    schema_name: str


@dataclass
class GenerationResult:
    data: dict[str, Any]
    raw_text: str
    provider: str
    model_id: str
    finish_reason: str | None = None
    latency_seconds: float | None = None
    usage: dict[str, int | None] = field(default_factory=dict)
    provider_response_id: str | None = None

    def metadata(self, stage: str) -> dict[str, Any]:
        return {
            "stage": stage,
            "provider": self.provider,
            "model_id": self.model_id,
            "finish_reason": self.finish_reason,
            "latency_seconds": self.latency_seconds,
            "usage": self.usage,
            "provider_response_id": self.provider_response_id,
        }
