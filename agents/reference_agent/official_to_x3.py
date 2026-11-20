#!/usr/bin/env python3
"""Convert the reference agent's run files (run_*.json) into the per-question records the judge and analysis read.

Run: python agents/reference_agent/official_to_x3.py --in data/x3/official/<run> --run <run> -> data/x3/<run>/runs/.
The run records are not part of the release.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation"))
from x3_plain_agent import load_questions  # noqa: E402


def _plus(a, b):
    """Agent tokens plus the summariser's (summarised arm only; zero elsewhere)."""
    return a if a is None else a + b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    qs = load_questions()
    out = ROOT / "data/x3" / a.run / "runs"; out.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(Path(a.inp).glob("run_*.json")):
        d = json.load(open(p, encoding="utf-8"))
        qid = str(d.get("query_id")); q = qs.get(qid)
        if q is None:
            continue
        texts = [r["output"] for r in d.get("result", []) if r.get("type") == "output_text"]
        final = texts[-1] if texts else ""
        seen = set(str(x) for x in d.get("retrieved_docids", []))
        calls = d.get("per_call_usage", [])
        g, e = q["gold"], q["evidence"]
        rec = dict(query_id=qid, status="completed" if (d.get("status") == "completed" and final.strip()) else (d.get("status") or "unknown"),
                   harness=d.get("harness", {"name": "BrowseComp-Plus search_agent/openai_client.py"}),
                   rounds=len(calls) or None, n_searches=(d.get("tool_call_counts") or {}).get("search", 0),
                   n_get_document=(d.get("tool_call_counts") or {}).get("get_document", 0),
                   retrieved_docids=sorted(seen, key=lambda x: (len(x), x)),
                   gold_recall=len(g & seen) / len(g) if g else None, evidence_recall=len(e & seen) / len(e) if e else None,
                   gold_in_search=sorted(g & seen), gold_fetched=[],
                   context_tokens_per_round=[c.get("input_tokens") for c in calls],
                   final_context_tokens=(calls[-1].get("input_tokens") if calls else None),
                   cum_input_tokens=_plus((d.get("usage") or {}).get("input_tokens"), sum(c.get("input_tokens") or 0 for c in d.get("summary_calls", []))),
                   cum_output_tokens=_plus((d.get("usage") or {}).get("output_tokens"), sum(c.get("output_tokens") or 0 for c in d.get("summary_calls", []))),
                   summary_calls=len(d.get("summary_calls", [])), summary_folds=d.get("summary_folds", []),
                   reasoning_tokens=(d.get("usage") or {}).get("included_reasoning_tokens"),
                   wall_time_s=d.get("wall_time_s"), timed_out=False, forced=False, errors=[],
                   final_answer=final, source_file=p.name)
        (out / f"{qid}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        n += 1
    print(f"converted {n} runs -> {out}")


if __name__ == "__main__":
    main()
