"""Score a document REPL agent OBLIQ run with the scorer the snippet REPL agent uses (rlm_shared.obliq_data).

    OBLIQ_TASK=<task> OBLIQ_SUBSET=<subset> python src/score_obliq.py <run_dir>   -> <run_dir>/_scored.json
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "shared"))
from rlm_shared import obliq_data as OD  # noqa: E402

TASK = os.environ.get("OBLIQ_TASK", "descriptive")
SUBSET = os.environ.get("OBLIQ_SUBSET", "twitter")


def main(run_dir: str) -> dict:
    qrels = OD.load_qrels(TASK, SUBSET)
    rows = []
    for f in sorted(glob.glob(f"{run_dir}/q*.json")):
        if os.path.basename(f).startswith("_"):
            continue
        r = json.load(open(f))
        qid = str(r.get("query_id") or os.path.basename(f)[:-5])
        ans = (r.get("result") or [{}])[0].get("output", "") or ""
        s = OD.score_answer(ans, set(qrels.get(qid, set())), SUBSET)
        rows.append({"query_id": qid, "status": r.get("status"),
                     "ndcg@10": s["ndcg@10"], "recall@10": s["recall@10"],
                     "n_parsed": s["n_parsed"],
                     "searches": (r.get("tool_call_counts") or {}).get("search", 0),
                     "wall": r.get("wall_time_s")})
    n = len(rows) or 1
    summary = {"run_dir": run_dir, "n": len(rows),
               "ndcg@10": sum(x["ndcg@10"] for x in rows) / n,
               "recall@10": sum(x["recall@10"] for x in rows) / n,
               "n_parsed_mean": sum(x["n_parsed"] for x in rows) / n,
               "searches_mean": sum(x["searches"] for x in rows) / n,
               "wall_mean": sum(x["wall"] or 0 for x in rows) / n,
               "errors": sum(1 for x in rows if x["status"] != "completed")}
    json.dump({"summary": summary, "rows": rows},
              open(f"{run_dir}/_scored.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main(sys.argv[1])
