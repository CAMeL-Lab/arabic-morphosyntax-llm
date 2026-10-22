from __future__ import annotations

from arabic_morphosyntax_llm.prompts import build_request, system_prompt
from arabic_morphosyntax_llm.schemas import morphology_schema


def test_gold_and_raw_parsing_prompts_match_paper_settings() -> None:
    gold = system_prompt("parse-gold")
    raw = system_prompt("parse-raw")
    assert "tokenized Arabic sentence as a list of tokens" in gold
    assert "RAW TEXT (a normal string)" not in gold
    assert "RAW TEXT (a normal string)" in raw
    assert "Tokenization (YOU MUST DO THIS FIRST)" in raw


def test_morphology_uses_actual_feature_keys() -> None:
    prompt = system_prompt("morphology")
    assert "form_gen" in prompt
    assert "form_num" in prompt
    schema = morphology_schema(3)
    analysis = schema["properties"]["tokens"]["items"]["properties"]["analysis"]
    assert analysis["required"] == [
        "pos",
        "asp",
        "vox",
        "mod",
        "per",
        "form_gen",
        "form_num",
        "cas",
        "stt",
        "prc0",
        "prc1",
        "prc2",
        "prc3",
        "enc0",
    ]


def test_request_uses_paper_few_shot_block() -> None:
    analysis = {
        "pos": "punc",
        "asp": "na",
        "vox": "na",
        "mod": "na",
        "per": "na",
        "form_gen": "na",
        "form_num": "na",
        "cas": "na",
        "stt": "na",
        "prc0": "na",
        "prc1": "na",
        "prc2": "na",
        "prc3": "na",
        "enc0": "na",
    }
    demo = {
        "id": "d",
        "tokens": ["."],
        "morphology": [{"word": ".", "analysis": analysis}],
    }
    request = build_request("morphology", {"id": "q", "tokens": ["."]}, [demo])
    assert [message.role for message in request.messages] == ["user"]
    assert 'Input:\n  ["."]' in request.messages[0].content
    assert "Now analyze the following Arabic sentence" in request.messages[0].content
    assert request.schema["properties"]["tokens"]["maxItems"] == 1
