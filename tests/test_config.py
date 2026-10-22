from __future__ import annotations

from arabic_morphosyntax_llm.config import load_profiles


def test_all_paper_models_are_configured() -> None:
    profiles = load_profiles()
    assert set(profiles) == {"llama4", "qwen3", "gemini3", "gpt5_2"}
    assert profiles["llama4"].provider == "vllm"
    assert profiles["qwen3"].provider_options["chat_template_kwargs"]["enable_thinking"]
    assert profiles["gemini3"].model_id == "gemini-3-flash-preview"
    assert profiles["gpt5_2"].generation.reasoning_effort == "none"


def test_profile_override_does_not_mutate_original() -> None:
    profile = load_profiles()["llama4"]
    changed = profile.with_overrides(
        base_url="http://example.test/v1",
        temperature=0.2,
        override_temperature=True,
    )
    assert changed.base_url == "http://example.test/v1"
    assert changed.generation.temperature == 0.2
    assert profile.base_url == "http://localhost:8000/v1"
    assert profile.generation.temperature == 0.0
