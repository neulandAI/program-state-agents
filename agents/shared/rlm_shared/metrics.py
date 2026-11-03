"""Reference implementations of the reported metrics.

- accuracy: correct answers / all questions; a question with no result counts as wrong.
- gold recall: |gold ∩ retrieved| / |gold| per question, averaged over questions.
- NDCG@k: binary relevance over the deduplicated ranking, normalised by the DCG of min(|gold|, k) gold documents.
- Recall@k: |gold ∩ top-k| / |gold|.
- majority vote: correct when more than half of the valid votes say correct.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional


def dedupe(ranking: Iterable[str]) -> list[str]:
    """Keep the first occurrence of every id, in order."""
    seen, out = set(), []
    for d in ranking:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def ndcg_at_k(ranking: Iterable[str], gold: set[str], k: int) -> float:
    if not gold or k <= 0:
        return 0.0
    top = dedupe(ranking)[:k]
    dcg = sum(1.0 / math.log2(rank + 1) for rank, d in enumerate(top, start=1) if d in gold)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
    return dcg / ideal


def recall_at_k(ranking: Iterable[str], gold: set[str], k: int) -> float:
    if not gold or k <= 0:
        return 0.0
    return len(set(dedupe(ranking)[:k]) & gold) / len(gold)


def gold_recall(retrieved: Iterable[str], gold: set[str]) -> Optional[float]:
    """Share of gold documents retrieved at any point; None when the question has no gold documents."""
    if not gold:
        return None
    return len(set(retrieved) & gold) / len(gold)


def majority(votes: Iterable[bool]) -> Optional[bool]:
    """Strict majority of valid votes; None when there are no votes (the question must then be re-judged)."""
    v = list(votes)
    if not v:
        return None
    return sum(v) * 2 > len(v)


def accuracy(correct_by_question: dict, all_questions: Iterable) -> float:
    """Correct over ALL questions; missing questions and anything that is not exactly True count as wrong."""
    qs = list(all_questions)
    return sum(1 for q in qs if correct_by_question.get(q) is True) / len(qs) if qs else 0.0


def mean(values: Iterable, skip_none: bool = True) -> Optional[float]:
    v = [x for x in values if not (skip_none and x is None)]
    return sum(v) / len(v) if v else None
