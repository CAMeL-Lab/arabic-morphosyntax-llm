"""Strict post-generation validation independent of provider-side schemas."""

from __future__ import annotations

import json
from typing import Any

from .constants import (
    CATIB_DEPRELS,
    MORPH_ENUMS,
    MORPH_FEATURE_ORDER,
)


class TaskValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def parse_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise TaskValidationError([f"response is not valid JSON: {error}"]) from error
    if not isinstance(value, dict):
        raise TaskValidationError(["top-level response must be an object"])
    return value


def _validate_morphology_items(
    items: Any, expected_tokens: list[str], path: str
) -> list[str]:
    errors: list[str] = []
    if not isinstance(items, list):
        return [f"{path} must be an array"]
    if len(items) != len(expected_tokens):
        errors.append(f"{path} length is {len(items)}; expected {len(expected_tokens)}")
    for index, (expected_word, item) in enumerate(zip(expected_tokens, items)):
        item_path = f"{path}[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{item_path} must be an object")
            continue
        if set(item) != {"word", "analysis"}:
            errors.append(f"{item_path} must contain exactly word and analysis")
        if item.get("word") != expected_word:
            errors.append(
                f"{item_path}.word is {item.get('word')!r}; expected {expected_word!r}"
            )
        analysis = item.get("analysis")
        if not isinstance(analysis, dict):
            errors.append(f"{item_path}.analysis must be an object")
            continue
        if set(analysis) != set(MORPH_FEATURE_ORDER):
            missing = sorted(set(MORPH_FEATURE_ORDER) - set(analysis))
            extra = sorted(set(analysis) - set(MORPH_FEATURE_ORDER))
            errors.append(f"{item_path}.analysis keys differ; missing={missing}, extra={extra}")
        for feature in MORPH_FEATURE_ORDER:
            value = analysis.get(feature)
            if value not in MORPH_ENUMS[feature]:
                errors.append(f"{item_path}.analysis.{feature} has invalid value {value!r}")
    return errors


def validate_morphology(data: dict[str, Any], expected_tokens: list[str]) -> None:
    errors = []
    if set(data) != {"tokens"}:
        errors.append("top-level morphology object must contain exactly tokens")
    errors.extend(_validate_morphology_items(data.get("tokens"), expected_tokens, "tokens"))
    if errors:
        raise TaskValidationError(errors)


def _cycle_errors(heads: list[int]) -> list[str]:
    errors: list[str] = []
    length = len(heads)
    for start in range(1, length + 1):
        seen: set[int] = set()
        node = start
        while node != 0:
            if node in seen:
                errors.append(f"dependency cycle reachable from token {start}")
                break
            seen.add(node)
            head = heads[node - 1]
            if not isinstance(head, int) or head < 0 or head > length:
                break
            node = head
    return sorted(set(errors))


def _validate_parse_items(
    items: Any,
    *,
    expected_tokens: list[str] | None,
    relation_key: str,
    allowed_relations: tuple[str, ...],
    allow_multiple_roots: bool,
    path: str,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(items, list) or not items:
        return [f"{path} must be a nonempty array"]
    if expected_tokens is not None and len(items) != len(expected_tokens):
        errors.append(f"{path} length is {len(items)}; expected {len(expected_tokens)}")
    heads: list[int] = []
    for index, item in enumerate(items, start=1):
        item_path = f"{path}[{index - 1}]"
        if not isinstance(item, dict):
            errors.append(f"{item_path} must be an object")
            heads.append(-1)
            continue
        if item.get("id") != index:
            errors.append(f"{item_path}.id is {item.get('id')!r}; expected {index}")
        form = item.get("form")
        if not isinstance(form, str) or not form:
            errors.append(f"{item_path}.form must be a nonempty string")
        elif expected_tokens is not None and index <= len(expected_tokens):
            if form != expected_tokens[index - 1]:
                errors.append(
                    f"{item_path}.form is {form!r}; expected {expected_tokens[index - 1]!r}"
                )
        head = item.get("head")
        heads.append(head if isinstance(head, int) else -1)
        if not isinstance(head, int) or head < 0 or head > len(items):
            errors.append(f"{item_path}.head has invalid value {head!r}")
        elif head == index:
            errors.append(f"{item_path}.head is a self-loop")
        relation = item.get(relation_key)
        if relation not in allowed_relations:
            errors.append(f"{item_path}.{relation_key} has invalid value {relation!r}")
    roots = sum(head == 0 for head in heads)
    if roots == 0:
        errors.append(f"{path} has no root")
    if not allow_multiple_roots and roots != 1:
        errors.append(f"{path} has {roots} roots; expected exactly one")
    errors.extend(_cycle_errors(heads))
    return errors


def validate_parsing(
    data: dict[str, Any],
    *,
    expected_tokens: list[str] | None = None,
    allow_multiple_roots: bool = True,
) -> None:
    errors = []
    if set(data) != {"parses"}:
        errors.append("top-level parsing object must contain exactly parses")
    errors.extend(
        _validate_parse_items(
            data.get("parses"),
            expected_tokens=expected_tokens,
            relation_key="deprel",
            allowed_relations=CATIB_DEPRELS,
            allow_multiple_roots=allow_multiple_roots,
            path="parses",
        )
    )
    if errors:
        raise TaskValidationError(errors)


def validate_prediction(task: str, data: dict[str, Any], record: dict[str, Any]) -> None:
    if task == "morphology":
        validate_morphology(data, record["tokens"])
    elif task == "parse-gold":
        validate_parsing(data, expected_tokens=record["tokens"])
    elif task in {"parse-raw", "cascade-parse"}:
        validate_parsing(data)
        raw_units = record.get("tokens") or str(record["text"]).split()
        maximum = max(16, len(raw_units) * 2 + 20)
        if len(data["parses"]) > maximum:
            raise TaskValidationError(
                [f"parses length is {len(data['parses'])}; maximum is {maximum}"]
            )
    else:
        raise ValueError(task)
