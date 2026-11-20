"""Grade a snippet REPL agent BrowseComp-Plus run directory with gpt-4.1 and print the summary numbers.

  python grade_bcp_snippet.py <run_dir>

Reads one q<idx>.json per question (idx = position in the test split). A question without a result or with an error
counts wrong; accuracy is over all 830. Verdicts are cached in <run_dir>/_judge_cache_gpt41.json; writes
<run_dir>/graded_gpt41.json. Token usage comes from <run_dir>/usage.jsonl when present.
"""
import glob, json, os, statistics as st, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # <repo>/shared
from rlm_shared.grading.bcp_judge import grade_outputs  # noqa: E402
from datasets import load_dataset  # noqa: E402

D = sys.argv[1].rstrip("/")
QID = [str(r["query_id"]) for r in load_dataset("Tevatron/browsecomp-plus", split="test")]
recs = {}
for f in glob.glob(f"{D}/q*.json"):
    x = json.load(open(f)); x = x[0] if isinstance(x, list) else x
    recs[QID[x["q_idx"]]] = x
outs = {q: "" if x.get("error") not in (None, "None") else (x.get("predicted") or "") for q, x in recs.items()}
ok = grade_outputs(outs, f"{D}/_judge_cache_gpt41.json")
g = {q: {"correct": bool(ok[q]), "old_correct": bool(x.get("correct")), "gold_recall": x.get("gold_recall"),
         "searches": x.get("search_calls"), "latency_s": x.get("latency_s"), "iterations": x.get("iterations"),
         "llm_query_calls": x.get("llm_query_calls"), "error": x.get("error")} for q, x in recs.items()}
json.dump(g, open(f"{D}/graded_gpt41.json", "w"), indent=0)
N, n = len(QID), len(recs)
c = sum(v["correct"] for v in g.values()); old = sum(v["old_correct"] for v in g.values())
errs = sum(1 for v in g.values() if v["error"] not in (None, "None"))
line = (f"{D}: {n}/{N} results ({errs} with error) | accuracy gpt-4.1 {c}/{N} = {c / N:.4f} (over results {c / n:.4f}; "
        f"harness judge {old / N:.4f}) | gold recall {st.mean(v['gold_recall'] or 0 for v in g.values()):.4f} | "
        f"searches {st.mean(v['searches'] or 0 for v in g.values()):.2f} | time {st.mean(v['latency_s'] or 0 for v in g.values()):.1f} s "
        f"({st.mean(v['latency_s'] or 0 for v in g.values()) / 60:.1f} min) | llm_query on {sum(1 for v in g.values() if v['llm_query_calls']) / n:.0%}")
if os.path.exists(f"{D}/usage.jsonl"):
    cached = unc = 0.0
    for l in open(f"{D}/usage.jsonl"):
        x = json.loads(l); a, cc, o = (1.75, 0.175, 14) if "5.2" in x["model"] else (1.25, 0.125, 10)
        cached += ((x["prompt"] - x["cached"]) * a + x["cached"] * cc + x["completion"] * o) / 1e6
        unc += (x["prompt"] * a + x["completion"] * o) / 1e6
    line += f" | cost/q cached ${cached / n:.4f} uncached ${unc / n:.4f} (total ${cached:.0f})"
else:
    line += " | local GPU (no API cost)"
print(line)
