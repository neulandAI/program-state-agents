"""Reciprocal rank fusion, used by retrieval_server_colbert.py when COLBERT_BM25_DIR is set."""

from __future__ import annotations


def rrf(rankings: list[list[str]], k: int = 60, weights: list[float] | None = None) -> list[tuple[str, float]]:
    """Fuse ranked docid lists. Standard RRF: score = Σ w_i / (k + rank_i)."""
    weights = weights or [1.0] * len(rankings)
    score: dict[str, float] = {}
    for w, ranking in zip(weights, rankings):
        for r, d in enumerate(ranking, 1):
            score[d] = score.get(d, 0.0) + w / (k + r)
    return sorted(score.items(), key=lambda x: -x[1])
