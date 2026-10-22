from __future__ import annotations

from arabic_morphosyntax_llm.conversion import convert_conllu, convert_patb_tsv
from arabic_morphosyntax_llm.io import normalize_input, split_whitespace_punctuation


def test_unicode_punctuation_tokenization() -> None:
    assert split_whitespace_punctuation("قال: نعم، بالتأكيد.") == [
        "قال",
        ":",
        "نعم",
        "،",
        "بالتأكيد",
        ".",
    ]


def test_task_specific_token_fields_are_distinct() -> None:
    record = {
        "id": "1",
        "text": "وبالبيت .",
        "morph_tokens": ["وبالبيت", "."],
        "parse_tokens": ["و+", "ب+", "البيت", "."],
    }
    assert normalize_input(record, "morphology")["tokens"] == ["وبالبيت", "."]
    assert normalize_input(record, "parse-gold")["tokens"] == [
        "و+",
        "ب+",
        "البيت",
        ".",
    ]


def test_catib_conllu_conversion(tmp_path) -> None:
    source = tmp_path / "sample.conllu"
    source.write_text(
        "# sent_id = s1\n"
        "# text = كتب الطالب .\n"
        "1\tكتب\t_\t_\t_\t_\t0\t---\t_\t_\n"
        "2\tالطالب\t_\t_\t_\t_\t1\tSBJ\t_\t_\n"
        "3\t.\t_\t_\t_\t_\t1\tMOD\t_\t_\n\n",
        encoding="utf-8",
    )
    rows = convert_conllu(source, kind="catib-conllu")
    assert rows[0]["id"] == "s1"
    assert rows[0]["parse_tokens"] == ["كتب", "الطالب", "."]
    assert rows[0]["parse"][1] == {
        "id": 2,
        "form": "الطالب",
        "head": 1,
        "deprel": "SBJ",
    }


def test_patb_tsv_conversion(tmp_path) -> None:
    source = tmp_path / "sample.tsv"
    header = [
        "word",
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
    row = [
        "كتب",
        "verb",
        "p",
        "a",
        "na",
        "3",
        "m",
        "s",
        "na",
        "na",
        "0",
        "0",
        "0",
        "0",
        "0",
    ]
    source.write_text("\t".join(header) + "\n" + "\t".join(row) + "\n", encoding="utf-8")
    records = convert_patb_tsv(source)
    assert records[0]["morph_tokens"] == ["كتب"]
    assert records[0]["morphology"][0]["analysis"]["asp"] == "p"
