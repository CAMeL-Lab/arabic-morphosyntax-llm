"""CAMeLBERT sentence embeddings for cosine based shot selection."""

from __future__ import annotations

from typing import Any

from .retrieval import retrieval_text


DEFAULT_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-msa"


def embed_records(
    queries: list[dict[str, Any]],
    demonstrations: list[dict[str, Any]],
    *,
    task: str,
    model_name: str = DEFAULT_MODEL,
    device: str | None = None,
    batch_size: int = 32,
    max_length: int = 256,
) -> None:
    """Attach mean pooled, L2 normalized embeddings to in-memory records.

    Uses the same attention-mask-aware pooling as the original experiment scripts.
    No embeddings or model weights are written to the repository.
    """
    if batch_size < 1 or max_length < 1:
        raise ValueError("embedding batch size and max length must be positive")
    rows = demonstrations + queries
    if not rows:
        return

    try:
        import torch
        import torch.nn.functional as functional
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "CAMeLBERT retrieval needs torch and transformers; install with "
            "pip install -e '.[retrieval]'"
        ) from error

    chosen_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(chosen_device)
    model.eval()

    texts = [retrieval_text(row, task) for row in rows]
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = tokenizer(
                texts[start:start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(chosen_device)
            token_embeddings = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            vectors = functional.normalize(pooled, p=2, dim=1).cpu().tolist()
            for row, vector in zip(rows[start:start + batch_size], vectors):
                row["embedding"] = vector
