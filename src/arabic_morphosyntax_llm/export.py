"""Convert LLM parse output to CoNLL for the upstream evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def predictions_to_conll(rows: list[dict[str, Any]], output: Path) -> None:
    """Write successful CATiB JSONL predictions as CoNLL-X/CoNLL-U rows."""
    if not rows:
        raise ValueError("prediction file is empty")
    lines: list[str] = []
    for row in rows:
        if row.get("status") != "ok":
            raise ValueError(f"sentence {row.get('id')}: prediction status is not ok")
        prediction = row.get("prediction") or {}
        parses = prediction.get("parses")
        if not isinstance(parses, list) or not parses:
            raise ValueError(f"sentence {row.get('id')}: missing parses")
        lines.append(f"# sent_id = {row['id']}")
        text = (row.get("input") or {}).get("text")
        if text:
            lines.append(f"# text = {text}")
        for expected, token in enumerate(parses, 1):
            if token.get("id") != expected:
                raise ValueError(f"sentence {row['id']}: token IDs are not sequential")
            form = str(token["form"])
            head = int(token["head"])
            relation = str(token["deprel"])
            if any(character in form for character in "\t\r\n"):
                raise ValueError(f"sentence {row['id']}: invalid FORM")
            if head < 0 or head > len(parses) or head == expected:
                raise ValueError(f"sentence {row['id']}: invalid HEAD")
            lines.append("\t".join((str(expected), form, "_", "_", "_", "_", str(head), relation, "_", "_")))
        lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
