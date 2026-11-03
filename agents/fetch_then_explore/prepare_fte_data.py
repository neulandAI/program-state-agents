#!/usr/bin/env python3
"""Write BrowseComp-Plus questions in Fetch-then-Explore's plain input format ({"question", "answer", "metadata"}),
from our own decrypted copy of the benchmark (evaluation/x3_plain_agent.load_questions), with our query ids in metadata.
Output (git-ignored, never committed: benchmark text stays offline): data/x3/fte/bcp150.jsonl (the 150 development
questions, data/x3/subset_150_ids.json) and data/x3/fte/bcp830.jsonl.
  python agents/fetch_then_explore/prepare_fte_data.py
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation"))
from x3_plain_agent import load_questions  # noqa: E402

qs = load_questions()
sub = [str(q) for q in json.load(open(ROOT / "data/x3/subset_150_ids.json"))]
out = ROOT / "data/x3/fte"; out.mkdir(parents=True, exist_ok=True)
for name, ids in (("bcp150", sub), ("bcp830", sorted(qs, key=lambda q: int(q) if str(q).isdigit() else 0))):
    with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for q in ids:
            r = qs[str(q)]
            f.write(json.dumps({"question": r["question"], "answer": r["answer"], "metadata": {"query_id": str(q)}}, ensure_ascii=False) + "\n")
    print(name, len(ids), "questions")
