"""OpenAI-compatible vLLM adapter for Llama 4 and Qwen3-Next."""

from __future__ import annotations

import copy
import time
from typing import Any

from ..config import ModelProfile
from ..records import GenerationRequest, GenerationResult
from ..validation import parse_json_object
from .base import Provider


class VLLMProvider(Provider):
    def __init__(self, profile: ModelProfile):
        try:
            from openai import AsyncOpenAI
        except ImportError as error:
            raise RuntimeError(
                "vLLM client support requires: pip install 'arabic-morphosyntax-llm[openai]'"
            ) from error
        if not profile.base_url:
            raise ValueError(f"vLLM profile {profile.key!r} requires base_url")
        self.client = AsyncOpenAI(
            base_url=profile.base_url,
            api_key=profile.api_key() or "EMPTY",
        )
        self.profile = profile

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        settings = self.profile.generation
        messages = [{"role": "system", "content": request.system_prompt}]
        messages.extend(
            {"role": message.role, "content": message.content}
            for message in request.messages
        )
        kwargs: dict[str, Any] = {
            "model": self.profile.model_id,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": request.schema_name,
                    "schema": request.schema,
                    "strict": True,
                },
            },
        }
        if settings.temperature is not None:
            kwargs["temperature"] = settings.temperature
        if settings.seed is not None:
            kwargs["seed"] = settings.seed
        if settings.max_output_tokens is not None:
            kwargs["max_tokens"] = settings.max_output_tokens
        extra_body = copy.deepcopy(self.profile.provider_options)
        if extra_body:
            kwargs["extra_body"] = extra_body
        started = time.perf_counter()
        response = await self.client.chat.completions.create(**kwargs)
        latency = time.perf_counter() - started
        choice = response.choices[0]
        raw_text = choice.message.content or ""
        usage_value = getattr(response, "usage", None)
        return GenerationResult(
            data=parse_json_object(raw_text),
            raw_text=raw_text,
            provider="vllm",
            model_id=str(getattr(response, "model", None) or self.profile.model_id),
            finish_reason=choice.finish_reason,
            latency_seconds=latency,
            usage={
                "input_tokens": getattr(usage_value, "prompt_tokens", None),
                "output_tokens": getattr(usage_value, "completion_tokens", None),
                "total_tokens": getattr(usage_value, "total_tokens", None),
            },
            provider_response_id=getattr(response, "id", None),
        )

    async def close(self) -> None:
        await self.client.close()
