"""Compose task prompts and in-context demonstration turns."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from .records import ChatMessage, GenerationRequest
from .schemas import morphology_schema, parsing_schema


def _fragment(name: str) -> str:
    return (
        files("arabic_morphosyntax_llm")
        .joinpath("prompts", name)
        .read_text(encoding="utf-8")
        .strip()
    )


def morphology_prompt() -> str:
    return _fragment("morphology.txt")


def system_prompt(task: str) -> str:
    if task == "morphology":
        return morphology_prompt()
    if task == "parse-gold":
        return _fragment("catib_gold.txt")
    if task == "parse-raw":
        return _fragment("catib_raw.txt")
    if task == "cascade-parse":
        return _fragment("catib_cascade.txt")
    raise ValueError(f"Unknown prompt task: {task}")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _demo_output(task: str, demo: dict[str, Any]) -> dict[str, Any]:
    if task == "morphology":
        if "morphology" not in demo:
            raise ValueError(f"demonstration {demo['id']}: morphology is required")
        return {"tokens": demo["morphology"]}
    if task in {"parse-gold", "parse-raw", "cascade-parse"}:
        if "parse" not in demo:
            raise ValueError(f"demonstration {demo['id']}: parse is required")
        return {"parses": demo["parse"]}
    raise ValueError(task)


def _demo_input(task: str, demo: dict[str, Any]) -> dict[str, Any]:
    if task in {"morphology", "parse-gold"}:
        if "tokens" not in demo:
            raise ValueError(f"demonstration {demo['id']}: tokens are required")
        return {"tokens": demo["tokens"]}
    if task == "parse-raw":
        if "text" not in demo:
            raise ValueError(f"demonstration {demo['id']}: text is required")
        return {"text": demo["text"]}
    if task == "cascade-parse":
        if "text" not in demo or "morphology" not in demo:
            raise ValueError(
                f"demonstration {demo['id']}: text and morphology are required"
            )
        return {"text": demo["text"], "morphology": demo["morphology"]}
    raise ValueError(task)


def _query_input(
    task: str,
    record: dict[str, Any],
    *,
    predicted_morphology: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if task in {"morphology", "parse-gold"}:
        return {"tokens": record["tokens"]}
    if task == "parse-raw":
        return {"text": record["text"]}
    if task == "cascade-parse":
        if predicted_morphology is None:
            raise ValueError("cascade parsing requires predicted morphology")
        return {"text": record["text"], "morphology": predicted_morphology}
    raise ValueError(task)


def _schema(task: str, record: dict[str, Any]) -> dict[str, Any]:
    if task == "morphology":
        return morphology_schema(len(record["tokens"]))
    if task == "parse-gold":
        return parsing_schema(token_count=len(record["tokens"]))
    if task in {"parse-raw", "cascade-parse"}:
        return parsing_schema(raw_word_count=len(record.get("tokens") or record["text"].split()))
    raise ValueError(task)


def build_request(
    task: str,
    record: dict[str, Any],
    demonstrations: list[dict[str, Any]],
    *,
    predicted_morphology: list[dict[str, Any]] | None = None,
) -> GenerationRequest:
    examples = ""
    messages: list[ChatMessage] = []
    for number, demonstration in enumerate(demonstrations, 1):
        demo_input = _demo_input(task, demonstration)
        if task in {"parse-raw", "cascade-parse"}:
            content = f"Top-ChrF++ demonstration {number}.\n\nRAW_TEXT:\n{demo_input['text']}"
            if task == "cascade-parse":
                content += "\n\nWORD_LEVEL_MORPHOLOGY (gold demonstration):\n"
                content += _morphology_lines(demo_input["morphology"])
            messages.append(ChatMessage("user", content))
            messages.append(ChatMessage("assistant", _json(_demo_output(task, demonstration))))
            continue
        value = demo_input.get("tokens", demo_input.get("text"))
        examples += "Input:\n  " + _json(value) + "\n"
        examples += "Output:\n  " + _json(_demo_output(task, demonstration)) + "\n\n"
    if demonstrations and task == "morphology":
        examples = "Here are some fully annotated examples (Arabic morphosyntax, PATB-style):\n\n" + examples
    if demonstrations and task == "parse-gold":
        examples = "Here are some annotated examples:\n\n" + examples
    query = _query_input(task, record, predicted_morphology=predicted_morphology)
    if task == "morphology":
        query_text = (
            "Now analyze the following Arabic sentence with full PATB-style morphosyntax.\n"
            "Return ONLY JSON in the required schema.\nTokens:\n" + _json(query["tokens"])
        )
    elif task == "parse-gold":
        query_text = "Now annotate the following Arabic sentence using CATiB.\nTokens:\n" + _json(query["tokens"])
    elif task == "parse-raw":
        query_text = "Tokenize and dependency-parse this raw PATB sentence using CATiB.\n\nRAW_TEXT:\n" + query["text"]
    elif task == "cascade-parse":
        query_text = (
            "Now annotate the following Arabic sentence using CATiB.\n"
            f"Sentence:\n{query['text']}\n\n"
            "Predicted word-level morphology (index, word, informative features):\n"
            + _morphology_lines(query["morphology"])
        )
    messages.append(ChatMessage("user", examples + query_text))
    return GenerationRequest(
        system_prompt=system_prompt(task),
        messages=messages,
        schema=_schema(task, record),
        schema_name=task.replace("-", "_"),
    )


def _morphology_lines(items: list[dict[str, Any]]) -> str:
    lines = []
    for index, item in enumerate(items, 1):
        features = item.get("analysis", {})
        informative = [
            f"{key}={value}" for key, value in features.items()
            if value not in (None, "", "0", "na", "u")
        ]
        lines.append(f"{index}  {item['word']}  {'|'.join(informative)}")
    return "\n".join(lines)
