"""Provider construction."""

from __future__ import annotations

from ..config import ModelProfile
from .base import Provider


def create_provider(profile: ModelProfile) -> Provider:
    if profile.provider == "openai_responses":
        from .openai_responses import OpenAIResponsesProvider

        return OpenAIResponsesProvider(profile)
    if profile.provider == "gemini":
        from .gemini import GeminiProvider

        return GeminiProvider(profile)
    if profile.provider == "vllm":
        from .vllm import VLLMProvider

        return VLLMProvider(profile)
    raise ValueError(f"Unsupported provider {profile.provider!r}")
