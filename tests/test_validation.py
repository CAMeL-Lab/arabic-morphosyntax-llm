from __future__ import annotations

import pytest

from arabic_morphosyntax_llm.validation import TaskValidationError, validate_parsing


def test_valid_multiroot_catib_forest_is_allowed() -> None:
    validate_parsing(
        {
            "parses": [
                {"id": 1, "form": "نعم", "head": 0, "deprel": "---"},
                {"id": 2, "form": ".", "head": 0, "deprel": "MOD"},
            ]
        }
    )


def test_cycle_is_rejected() -> None:
    with pytest.raises(TaskValidationError, match="cycle"):
        validate_parsing(
            {
                "parses": [
                    {"id": 1, "form": "أ", "head": 2, "deprel": "SBJ"},
                    {"id": 2, "form": "ب", "head": 1, "deprel": "OBJ"},
                ]
            }
        )


def test_gold_forms_must_be_preserved() -> None:
    with pytest.raises(TaskValidationError, match="expected"):
        validate_parsing(
            {"parses": [{"id": 1, "form": "غير", "head": 0, "deprel": "---"}]},
            expected_tokens=["الأصل"],
        )
