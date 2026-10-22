"""Declarative model profile loading."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, replace
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GenerationSettings:
    temperature: float | None = None
    seed: int | None = None
    max_output_tokens: int | None = None
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class ModelProfile:
    key: str
    display_name: str
    provider: str
    model_id: str
    base_url: str | None = None
    api_key_env: str | None = None
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    provider_options: dict[str, Any] = field(default_factory=dict)

    def api_key(self) -> str | None:
        if self.provider == "vllm" and not self.api_key_env:
            return "EMPTY"
        if not self.api_key_env:
            return None
        value = os.getenv(self.api_key_env)
        if value:
            return value
        if self.provider == "vllm":
            return "EMPTY"
        raise EnvironmentError(
            f"{self.display_name} requires environment variable {self.api_key_env}"
        )

    def with_overrides(
        self,
        *,
        model_id: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        override_temperature: bool = False,
        override_max_output_tokens: bool = False,
        override_reasoning_effort: bool = False,
    ) -> "ModelProfile":
        generation = self.generation
        if override_temperature:
            generation = replace(generation, temperature=temperature)
        if override_max_output_tokens:
            generation = replace(generation, max_output_tokens=max_output_tokens)
        if override_reasoning_effort:
            generation = replace(generation, reasoning_effort=reasoning_effort)
        return replace(
            self,
            model_id=model_id or self.model_id,
            base_url=base_url or self.base_url,
            generation=generation,
        )


def _profile_from_mapping(key: str, value: dict[str, Any]) -> ModelProfile:
    required = {"display_name", "provider", "model_id"}
    missing = required - value.keys()
    if missing:
        raise ValueError(f"Model profile {key!r} is missing {sorted(missing)}")
    generation_value = value.get("generation") or {}
    generation = GenerationSettings(
        temperature=generation_value.get("temperature"),
        seed=generation_value.get("seed"),
        max_output_tokens=generation_value.get("max_output_tokens"),
        reasoning_effort=generation_value.get("reasoning_effort"),
    )
    return ModelProfile(
        key=key,
        display_name=str(value["display_name"]),
        provider=str(value["provider"]),
        model_id=str(value["model_id"]),
        base_url=value.get("base_url"),
        api_key_env=value.get("api_key_env"),
        generation=generation,
        provider_options=copy.deepcopy(value.get("provider_options") or {}),
    )


def load_profiles(path: str | Path | None = None) -> dict[str, ModelProfile]:
    if path is None:
        content = files("arabic_morphosyntax_llm").joinpath("model_profiles.yaml").read_text(
            encoding="utf-8"
        )
    else:
        content = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(content) or {}
    if raw.get("version") != 1 or not isinstance(raw.get("models"), dict):
        raise ValueError("Model configuration must contain version: 1 and a models mapping")
    return {
        key: _profile_from_mapping(key, value)
        for key, value in raw["models"].items()
    }
