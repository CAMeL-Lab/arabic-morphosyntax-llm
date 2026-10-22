"""Converters from common experiment formats to the package JSONL schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .constants import MORPH_ENUMS, MORPH_FEATURE_ORDER


def _finish_conllu_sentence(
    rows: list[list[str]], metadata: dict[str, str], index: int
) -> dict[str, Any] | None:
    if not rows:
        return None
    forms = [row[1] for row in rows]
    record: dict[str, Any] = {
        "id": metadata.get("sent_id", str(index)),
        "text": metadata.get("text", " ".join(forms)),
    }
    record["parse_tokens"] = metadata.get("treeTokens", " ".join(forms)).split()
    if len(record["parse_tokens"]) != len(rows):
        # Some corpora use treeTokens only as raw metadata. The syntactic rows
        # remain the authoritative gold-token sequence for fixed-token tasks.
        record["parse_tokens"] = forms
    record["parse"] = [
        {"id": position, "form": row[1], "head": int(row[6]), "deprel": row[7]}
        for position, row in enumerate(rows, start=1)
    ]
    return record


def convert_conllu(path: str | Path, *, kind: str) -> list[dict[str, Any]]:
    if kind != "catib-conllu":
        raise ValueError(kind)
    records: list[dict[str, Any]] = []
    rows: list[list[str]] = []
    metadata: dict[str, str] = {}
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                record = _finish_conllu_sentence(rows, metadata, len(records))
                if record is not None:
                    records.append(record)
                rows, metadata = [], {}
                continue
            if line.startswith("#"):
                content = line[1:].strip()
                if "=" in content:
                    key, value = content.split("=", 1)
                    metadata[key.strip()] = value.strip()
                continue
            fields = line.split("\t")
            if len(fields) != 10:
                raise ValueError(f"{path}:{line_number}: expected 10 CoNLL columns")
            if "-" in fields[0] or "." in fields[0]:
                continue
            try:
                int(fields[0])
                int(fields[6])
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid ID or HEAD") from error
            rows.append(fields)
    record = _finish_conllu_sentence(rows, metadata, len(records))
    if record is not None:
        records.append(record)
    return records


def convert_patb_tsv(path: str | Path) -> list[dict[str, Any]]:
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    first = next((line for line in lines if line.strip()), None)
    if first is None:
        return []
    header = first.split("\t")
    required = {"word", *MORPH_FEATURE_ORDER}
    if not required.issubset(header):
        raise ValueError(f"PATB TSV header is missing {sorted(required - set(header))}")
    start = lines.index(first) + 1
    records: list[dict[str, Any]] = []
    sentence: list[dict[str, Any]] = []

    def finish() -> None:
        nonlocal sentence
        if not sentence:
            return
        tokens = [item["word"] for item in sentence]
        records.append(
            {
                "id": str(len(records)),
                "text": " ".join(tokens),
                "morph_tokens": tokens,
                "morphology": sentence,
            }
        )
        sentence = []

    for line_number, line in enumerate(lines[start:], start=start + 1):
        if not line.strip():
            finish()
            continue
        values = line.split("\t")
        if len(values) != len(header):
            raise ValueError(
                f"{path}:{line_number}: got {len(values)} fields; expected {len(header)}"
            )
        row = dict(zip(header, values))
        analysis = {feature: row[feature].strip() for feature in MORPH_FEATURE_ORDER}
        for feature, value in analysis.items():
            if value not in MORPH_ENUMS[feature]:
                raise ValueError(
                    f"{path}:{line_number}: invalid {feature} value {value!r}"
                )
        sentence.append({"word": row["word"].strip(), "analysis": analysis})
    finish()
    return records


def merge_gold_files(files: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if not files:
        return []
    merged = {str(record["id"]): dict(record) for record in files[0]}
    order = [str(record["id"]) for record in files[0]]
    for records in files[1:]:
        incoming = {str(record["id"]): record for record in records}
        if set(incoming) != set(merged):
            missing = sorted(set(merged) - set(incoming))
            extra = sorted(set(incoming) - set(merged))
            raise ValueError(f"cannot merge unequal ID sets; missing={missing}, extra={extra}")
        for record_id in order:
            for key, value in incoming[record_id].items():
                if key in merged[record_id] and merged[record_id][key] != value:
                    raise ValueError(f"conflicting {key!r} for id {record_id!r}")
                merged[record_id][key] = value
    return [merged[record_id] for record_id in order]
