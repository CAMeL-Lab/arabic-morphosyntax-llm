"""JSON Schemas sent to structured-output-capable model APIs."""

from __future__ import annotations

from typing import Any

from .constants import (
    CATIB_DEPRELS,
    MORPH_ENUMS,
    MORPH_FEATURE_ORDER,
)


def _closed_object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def morphology_item_schema(*, strict_enums: bool = True) -> dict[str, Any]:
    properties = {
        feature: (
            {"type": "string", "enum": list(MORPH_ENUMS[feature])}
            if strict_enums
            else {"type": "string"}
        )
        for feature in MORPH_FEATURE_ORDER
    }
    return _closed_object(
        {
            "word": {"type": "string", "minLength": 1},
            "analysis": _closed_object(properties, list(MORPH_FEATURE_ORDER)),
        },
        ["word", "analysis"],
    )


def morphology_schema(token_count: int, *, strict_enums: bool = True) -> dict[str, Any]:
    return _closed_object(
        {
            "tokens": {
                "type": "array",
                "minItems": token_count,
                "maxItems": token_count,
                "items": morphology_item_schema(strict_enums=strict_enums),
            }
        },
        ["tokens"],
    )


def parse_item_schema(max_items: int) -> dict[str, Any]:
    return _closed_object(
        {
            "id": {"type": "integer", "minimum": 1, "maximum": max_items},
            "form": {"type": "string", "minLength": 1},
            "head": {"type": "integer", "minimum": 0, "maximum": max_items},
            "deprel": {"type": "string", "enum": list(CATIB_DEPRELS)},
        },
        ["id", "form", "head", "deprel"],
    )


def parsing_schema(
    *, token_count: int | None = None, raw_word_count: int | None = None
) -> dict[str, Any]:
    if token_count is None and raw_word_count is None:
        raise ValueError("token_count or raw_word_count is required")
    maximum = token_count if token_count is not None else max(16, raw_word_count * 2 + 20)
    array: dict[str, Any] = {
        "type": "array",
        "minItems": token_count if token_count is not None else 1,
        "maxItems": maximum,
        "items": parse_item_schema(maximum),
    }
    return _closed_object({"parses": array}, ["parses"])
