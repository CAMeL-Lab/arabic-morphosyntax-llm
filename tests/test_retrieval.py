from __future__ import annotations

from arabic_morphosyntax_llm import cli
from arabic_morphosyntax_llm.io import write_jsonl_atomic
from arabic_morphosyntax_llm.retrieval import select_demonstrations


def test_high_and_low_similarity_choose_opposite_ends() -> None:
    query = {"id": "q", "tokens": ["كتب", "الطالب"], "embedding": [1.0, 0.0]}
    candidates = [
        {"id": "near", "tokens": ["كتب", "الطالب"], "embedding": [1.0, 0.0]},
        {"id": "middle", "tokens": ["قرأ", "الطالب"], "embedding": [0.5, 0.5]},
        {"id": "far", "tokens": ["بعيد"], "embedding": [-1.0, 0.0]},
    ]
    for method, first, last in (
        ("chrf-high", "near", "far"),
        ("chrf-low", "far", "near"),
        ("cosine-high", "near", "far"),
        ("cosine-low", "far", "near"),
    ):
        selection = select_demonstrations(
            query, candidates, task="morphology", shots=3, method=method, seed=42
        )
        assert selection.records[0]["id"] == first
        assert selection.records[-1]["id"] == last


def test_cosine_shots_encode_records_and_write_map(tmp_path, monkeypatch) -> None:
    query_path = tmp_path / "test.jsonl"
    train_path = tmp_path / "train.jsonl"
    output_path = tmp_path / "shots.jsonl"
    write_jsonl_atomic(query_path, [{"id": "q", "morph_tokens": ["كتب"]}])
    write_jsonl_atomic(train_path, [
        {"id": "near", "morph_tokens": ["كتب"], "morphology": []},
        {"id": "far", "morph_tokens": ["ذهب"], "morphology": []},
    ])
    calls = []

    def fake_embed(queries, demonstrations, **options):
        calls.append(options)
        queries[0]["embedding"] = [1.0, 0.0]
        demonstrations[0]["embedding"] = [1.0, 0.0]
        demonstrations[1]["embedding"] = [-1.0, 0.0]

    monkeypatch.setattr(cli, "embed_records", fake_embed)
    args = cli.build_parser().parse_args([
        "shots", "morphology", "--input", str(query_path),
        "--demonstrations", str(train_path), "--shots", "1",
        "--retrieval", "cosine-low", "--output", str(output_path),
    ])
    assert cli._shots(args) == 0
    assert calls[0]["task"] == "morphology"
    assert cli.read_jsonl(output_path)[0]["shots"][0]["id"] == "far"
