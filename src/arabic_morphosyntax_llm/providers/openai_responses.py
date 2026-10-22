"""OpenAI Responses API adapter used for GPT-5.2."""

from __future__ import annotations

import time
from typing import Any

from ..config import ModelProfile
from ..records import GenerationRequest, GenerationResult
from ..validation import parse_json_object
from .base import Provider


def _dump(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=False)
    if isinstance(value, dict):
        return value
    return {}


class OpenAIResponsesProvider(Provider):
    def __init__(self, profile: ModelProfile):
        try:
            from openai import AsyncOpenAI
        except ImportError as error:
            raise RuntimeError(
                "OpenAI support requires: pip install 'arabic-morphosyntax-llm[openai]'"
            ) from error
        kwargs: dict[str, Any] = {"api_key": profile.api_key()}
        if profile.base_url:
            kwargs["base_url"] = profile.base_url
        self.client = AsyncOpenAI(**kwargs)
        self.profile = profile

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        settings = self.profile.generation
        kwargs: dict[str, Any] = {
            "model": self.profile.model_id,
            "instructions": request.system_prompt,
            "input": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": request.schema_name,
                    "schema": request.schema,
                    "strict": True,
                }
            },
            "store": False,
        }
        if settings.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": settings.reasoning_effort}
        if settings.temperature is not None:
            kwargs["temperature"] = settings.temperature
        if settings.max_output_tokens is not None:
            kwargs["max_output_tokens"] = settings.max_output_tokens
        started = time.perf_counter()
        response = await self.client.responses.create(**kwargs)
        latency = time.perf_counter() - started
        raw_text = response.output_text or ""
        usage = _dump(getattr(response, "usage", None))
        incomplete = _dump(getattr(response, "incomplete_details", None))
        finish_reason = getattr(response, "status", None)
        if incomplete.get("reason"):
            finish_reason = f"{finish_reason}:{incomplete['reason']}"
        return GenerationResult(
            data=parse_json_object(raw_text),
            raw_text=raw_text,
            provider="openai_responses",
            model_id=str(getattr(response, "model", None) or self.profile.model_id),
            finish_reason=finish_reason,
            latency_seconds=latency,
            usage={
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
            provider_response_id=getattr(response, "id", None),
        )

    async def close(self) -> None:
        await self.client.close()
