from __future__ import annotations

import asyncio

from arabic_morphosyntax_llm.config import ModelProfile
from arabic_morphosyntax_llm.providers.base import Provider
from arabic_morphosyntax_llm.records import GenerationRequest, GenerationResult
from arabic_morphosyntax_llm.runner import TaskPipeline


class ScriptedProvider(Provider):
    def __init__(self, values):
        self.values = list(values)
        self.requests: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        value = self.values.pop(0)
        return GenerationResult(
            data=value,
            raw_text="{}",
            provider="scripted",
            model_id="test-model",
            finish_reason="stop",
            usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )


def test_cascade_uses_stage_one_prediction_in_stage_two() -> None:
    analysis = {
        "pos": "verb",
        "asp": "p",
        "vox": "a",
        "mod": "na",
        "per": "3",
        "form_gen": "m",
        "form_num": "s",
        "cas": "na",
        "stt": "na",
        "prc0": "0",
        "prc1": "0",
        "prc2": "0",
        "prc3": "0",
        "enc0": "0",
    }
    morphology = [{"word": "كتب", "analysis": analysis}]
    parse = [{"id": 1, "form": "كتب", "head": 0, "deprel": "---"}]
    provider = ScriptedProvider([{"tokens": morphology}, {"parses": parse}])
    pipeline = TaskPipeline(
        provider=provider,
        profile=ModelProfile("test", "Test", "scripted", "test-model"),
        task="cascade",
        demonstrations=[],
        shots=0,
        retrieval_method="random",
        seed=42,
        retries=0,
    )
    output = asyncio.run(
        pipeline.predict({"id": "s1", "text": "كتب", "tokens": ["كتب"]})
    )
    assert output["prediction"] == {"morphology": morphology, "parses": parse}
    assert len(provider.requests) == 2
    assert "Predicted word-level morphology" in provider.requests[1].messages[-1].content
    assert "pos=verb" in provider.requests[1].messages[-1].content
