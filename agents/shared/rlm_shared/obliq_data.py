"""OBLIQ-Bench data access and scoring (NDCG and recall over the ranked document ids in an answer)."""
from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

OBLIQ_ROOT = Path(os.environ.get("OBLIQ_BENCH_ROOT", "<OBLIQ_BENCH_ROOT not set>"))
NDCG_KS = [1, 3, 5, 10, 20, 50]


def ndcg_at(retrieved_ordered: list[str], gold_ids: set[str], k: int) -> float:
    """NDCG@k where retrieved_ordered is deduplicated in first-retrieval order."""
    if not gold_ids:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(retrieved_ordered[:k]) if d in gold_ids)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_ids), k)))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at(retrieved_ordered: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    return sum(1 for d in retrieved_ordered[:k] if d in gold_ids) / len(gold_ids)


def extract_doc_ids_from_answer(answer: str, subset: str = "") -> list[str]:
    """Ordered doc_ids from the agent's ranked answer string."""
    math_ids = re.findall(r"[\w][\w\-]*(?:___[\w\.\-]+)+", answer)
    congress_ids = re.findall(r"CHRG-\w+(?:_\w+)*", answer)
    twitter_ids = re.findall(r"\b\d{15,19}\b", answer)
    writing_ids = re.findall(r"\b\d{1,5}\b", answer) if subset == "writing" else []
    wildchat_ids = re.findall(r"\b\d{1,6}\b", answer) if subset.startswith("wildchat") else []
    return list(dict.fromkeys(math_ids + congress_ids + twitter_ids + writing_ids + wildchat_ids))


def load_queries(task: str, subset: str) -> list[dict]:
    path = OBLIQ_ROOT / task / subset / "queries+qrels" / "queries.jsonl"
    return [json.loads(l) for l in open(path) if l.strip()]


def load_qrels(task: str, subset: str, pool: bool = False) -> dict[str, set[str]]:
    """{query_id: {corpus_id, ...}}. pool=True uses the expanded judged pool."""
    fname = "qrels_pool.tsv" if pool else "qrels.tsv"
    path = OBLIQ_ROOT / task / subset / "queries+qrels" / fname
    if not path.exists():
        path = OBLIQ_ROOT / task / subset / "queries+qrels" / "qrels.tsv"
    qrels: dict[str, set[str]] = defaultdict(set)
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                qrels[parts[0]].add(parts[1])
    return dict(qrels)


def score_answer(answer: str, gold_ids: set[str], subset: str = "twitter") -> dict:
    """NDCG/recall at every k, plus the parsed ranking, for one question."""
    ids = extract_doc_ids_from_answer(answer or "", subset)
    out = {"n_parsed": len(ids), "ranked": ids[:50]}
    for k in NDCG_KS:
        out[f"ndcg@{k}"] = ndcg_at(ids, gold_ids, k)
        out[f"recall@{k}"] = recall_at(ids, gold_ids, k)
    return out
