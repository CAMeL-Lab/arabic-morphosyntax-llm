"""JSONL and Arabic token I/O helpers."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any, Iterable


def split_whitespace_punctuation(text: str) -> list[str]:
    """Split on whitespace and make each Unicode punctuation character a token."""
    tokens: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            tokens.append("".join(current))
            current.clear()

    for character in text:
        if character.isspace():
            flush()
        elif unicodedata.category(character).startswith("P"):
            flush()
            tokens.append(character)
        else:
            current.append(character)
    flush()
    return tokens


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: each row must be a JSON object")
            record_id = str(record.get("id", line_number - 1))
            if record_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate id {record_id!r}")
            seen.add(record_id)
            record["id"] = record_id
            records.append(record)
    return records


def write_jsonl_atomic(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")
        stream.flush()
    temporary.replace(target)


def normalize_input(record: dict[str, Any], task: str) -> dict[str, Any]:
    normalized = dict(record)
    normalized["id"] = str(record["id"])
    text = record.get("text")
    task_token_field = {
        "morphology": "morph_tokens",
        "cascade": "morph_tokens",
        "parse-gold": "parse_tokens",
    }.get(task)
    tokens = record.get(task_token_field) if task_token_field else None
    if tokens is None:
        # `tokens` remains a convenient task-specific shorthand for simple files.
        tokens = record.get("tokens")

    if tokens is not None:
        if not isinstance(tokens, list) or not tokens or not all(
            isinstance(token, str) and token for token in tokens
        ):
            raise ValueError(f"record {record['id']}: tokens must be a nonempty string list")
        normalized["tokens"] = list(tokens)

    if text is not None and not isinstance(text, str):
        raise ValueError(f"record {record['id']}: text must be a string")

    if task in {"morphology", "cascade"} and tokens is None:
        if not text:
            raise ValueError(f"record {record['id']}: {task} requires text or tokens")
        normalized["tokens"] = split_whitespace_punctuation(text)

    if task == "parse-gold" and tokens is None:
        raise ValueError(f"record {record['id']}: {task} requires tokens")

    if task in {"parse-raw", "cascade"} and not text:
        raise ValueError(f"record {record['id']}: {task} requires raw text")

    if text is None and tokens is not None:
        normalized["text"] = " ".join(tokens)
    return normalized
