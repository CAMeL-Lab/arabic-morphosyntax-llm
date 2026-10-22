"""Google GenAI adapter used for Gemini 3 Flash."""

from __future__ import annotations

import time
from typing import Any

from ..config import ModelProfile
from ..records import GenerationRequest, GenerationResult
from ..validation import parse_json_object
from .base import Provider


def _usage_value(usage: Any, *names: str) -> int | None:
    for name in names:
        value = getattr(usage, name, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return None


class GeminiProvider(Provider):
    def __init__(self, profile: ModelProfile):
        try:
            from google import genai
            from google.genai import types
        except ImportError as error:
            raise RuntimeError(
                "Gemini support requires: pip install 'arabic-morphosyntax-llm[gemini]'"
            ) from error
        self.genai = genai
        self.types = types
        self.client = genai.Client(api_key=profile.api_key())
        self.profile = profile

    def _config(self, request: GenerationRequest) -> Any:
        settings = self.profile.generation
        values: dict[str, Any] = {
            "system_instruction": request.system_prompt,
            "response_mime_type": "application/json",
            "response_json_schema": request.schema,
        }
        if settings.temperature is not None:
            values["temperature"] = settings.temperature
        if settings.seed is not None:
            values["seed"] = settings.seed
        if settings.max_output_tokens is not None:
            values["max_output_tokens"] = settings.max_output_tokens
        try:
            return self.types.GenerateContentConfig(**values)
        except TypeError:
            # Older google-genai releases named this field response_schema.
            values["response_schema"] = values.pop("response_json_schema")
            return self.types.GenerateContentConfig(**values)

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        contents = [
            {
                "role": "model" if message.role == "assistant" else "user",
                "parts": [{"text": message.content}],
            }
            for message in request.messages
        ]
        started = time.perf_counter()
        response = await self.client.aio.models.generate_content(
            model=self.profile.model_id,
            contents=contents,
            config=self._config(request),
        )
        latency = time.perf_counter() - started
        raw_text = response.text or ""
        usage = getattr(response, "usage_metadata", None)
        candidates = getattr(response, "candidates", None) or []
        finish_reason = None
        if candidates:
            reason = getattr(candidates[0], "finish_reason", None)
            finish_reason = getattr(reason, "name", None) or (str(reason) if reason else None)
        return GenerationResult(
            data=parse_json_object(raw_text),
            raw_text=raw_text,
            provider="gemini",
            model_id=self.profile.model_id,
            finish_reason=finish_reason,
            latency_seconds=latency,
            usage={
                "input_tokens": _usage_value(
                    usage, "prompt_token_count", "input_token_count"
                ),
                "output_tokens": _usage_value(
                    usage, "candidates_token_count", "output_token_count"
                ),
                "total_tokens": _usage_value(usage, "total_token_count"),
            },
            provider_response_id=getattr(response, "response_id", None),
        )

    async def close(self) -> None:
        aio = getattr(self.client, "aio", None)
        close = getattr(aio, "aclose", None)
        if close is not None:
            await close()
