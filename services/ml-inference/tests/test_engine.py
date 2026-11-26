import torch
from ml_inference.engine import MLEngine, last_token_pooling


def test_last_token_pooling() -> None:
    # batch_size=2, seq_len=4, hidden_dim=3
    hidden_states = torch.tensor(
        [
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[5.0, 5.0, 5.0], [6.0, 6.0, 6.0], [7.0, 7.0, 7.0], [0.0, 0.0, 0.0]],
        ]
    )
    # mask: first item has 2 tokens (seq_len=2), second has 3 tokens (seq_len=3)
    attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])

    pooled = last_token_pooling(hidden_states, attention_mask)
    # last token for first item should be [2.0, 2.0, 2.0]
    assert torch.allclose(pooled[0], torch.tensor([2.0, 2.0, 2.0]))
    # last token for second item should be [7.0, 7.0, 7.0]
    assert torch.allclose(pooled[1], torch.tensor([7.0, 7.0, 7.0]))


def test_compute_embeddings_fallback_and_mrl() -> None:
    engine = MLEngine(device="cpu", embedding_dim=1024)
    texts = ["High entropy alloy phase transition", "Austenite grain growth"]
    embeddings = engine.compute_embeddings(texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 1024
    # Check L2-normalization (norm close to 1.0)
    norm = sum(x * x for x in embeddings[0]) ** 0.5
    assert abs(norm - 1.0) < 1e-4


def test_compute_rerank_fallback() -> None:
    engine = MLEngine(device="cpu")
    docs = ["Doc A on metallurgy", "Doc B on polymers", "Doc C on ceramics"]
    results = engine.compute_rerank(
        query="metallurgical alloys",
        documents=docs,
        return_documents=True,
        top_n=2,
    )
    assert len(results) == 2
    assert results[0].relevance_score >= results[1].relevance_score
    assert results[0].document is not None
