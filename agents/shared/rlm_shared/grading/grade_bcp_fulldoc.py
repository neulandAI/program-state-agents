"""Grade a document REPL agent BrowseComp-Plus run made through split_runner.py with gpt-4.1 and print the summary.

  python grade_bcp_fulldoc.py <run_dir> <seed_key>

Answers come from rollouts/<seed_key>/<qid>.json (parsed from the file head, as records can be large); a question
without a completed rollout counts wrong. Recall, searches, turns and wall time come from <run_dir>/cache.json.
"""
import glob, json, os, statistics as st, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # <repo>/shared
from rlm_shared.grading.bcp_judge import grade_outputs  # noqa: E402
from datasets import load_dataset  # noqa: E402

D, KEY = sys.argv[1].rstrip("/"), sys.argv[2]
ALL = [str(r["query_id"]) for r in load_dataset("Tevatron/browsecomp-plus", split="test")]
DEC = json.JSONDecoder()


def field(head, name):
    i = head.find(f'"{name}":')
    if i < 0:
        return None
    j = i + len(name) + 3
    while head[j] in " \n":
        j += 1
    try:
        return DEC.raw_decode(head, j)[0]
    except Exception:
        return None


C = {k.split("|", 1)[1]: v for k, v in json.load(open(f"{D}/cache.json")).items() if k.startswith(KEY + "|")}
outs, tin, tout = {}, {}, {}
for f in glob.glob(f"{D}/rollouts/{KEY}/*.json"):
    q = os.path.basename(f)[:-5]
    head = open(f).read(400_000)
    if field(head, "status") != "completed":
        outs[q] = ""; continue
    res = field(head, "result") or [{}]
    outs[q] = ((res[0] or {}).get("output") or "") if isinstance(res, list) else ""
    u = (field(head, "usage") or {}).get("model_usage_summaries") or {}
    tin[q] = sum(m["total_input_tokens"] for m in u.values()); tout[q] = sum(m["total_output_tokens"] for m in u.values())
ok = grade_outputs({q: a for q, a in outs.items()}, f"{D}/_judge_cache_gpt41.json")
g = {q: {"correct": bool(ok.get(q)), "answered": bool(outs.get(q)), "gold_recall": float((C.get(q) or {}).get("gold_recall") or 0),
         "searches": float((C.get(q) or {}).get("searches") or 0), "wall": float((C.get(q) or {}).get("wall") or 0),
         "iters": float((C.get(q) or {}).get("iters") or 0), "status": (C.get(q) or {}).get("status")} for q in ALL}
json.dump(g, open(f"{D}/graded_gpt41.json", "w"), indent=0)
have = [q for q in ALL if q in outs]
c = sum(v["correct"] for v in g.values())
w = [g[q]["wall"] for q in have if g[q]["wall"]]
print(f"{D}: {len(have)}/830 rollouts, {sum(1 for q in have if outs[q])} with an answer | accuracy gpt-4.1 {c}/830 = {c / 830:.4f} "
      f"| gold recall {st.mean(g[q]['gold_recall'] for q in have):.4f} | searches {st.mean(g[q]['searches'] for q in have):.2f} "
      f"| turns {st.mean(g[q]['iters'] for q in have):.1f} | time {st.mean(w):.0f} s ({st.mean(w) / 60:.1f} min), without >600 s "
      f"{st.mean([x for x in w if x <= 600] or [0]):.0f} s ({sum(1 for x in w if x <= 600)}/{len(w)} under 10 min) "
      f"| tokens in {st.mean(tin.values()):,.0f} out {st.mean(tout.values()):,.0f} per question | local GPU (no API cost)")
