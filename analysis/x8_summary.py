#!/usr/bin/env python3
"""Tool-store agent summary: its Table I values and paired exact McNemar tests against the other runs.

Run: python analysis/x8_summary.py [run, default x8_toolstore] -> data/analysis/x8_toolstore.md.
Reads data/x3/<run>/; the run records are not part of the release."""
import glob, json, statistics as st, sys
from pathlib import Path
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
RUN = sys.argv[1] if len(sys.argv) > 1 else "x8_toolstore"


def ok(run):
    return {q: bool(v["correct"]) for q, v in json.load(open(ROOT / "data/x3" / run / "graded_gpt41.json")).items()}


recs = {str(json.load(open(p))["query_id"]): json.load(open(p)) for p in glob.glob(str(ROOT / "data/x3" / RUN / "runs" / "*.json"))}
A = ok(RUN)
rec = [r["gold_recall"] for r in recs.values() if r.get("gold_recall") is not None]
full = [q for q, r in recs.items() if (r.get("gold_recall") or 0) >= 1 and q in A]
fin = sorted(r["final_context_tokens"] for r in recs.values() if r.get("final_context_tokens"))
status = {}
for r in recs.values():
    status[r["status"]] = status.get(r["status"], 0) + 1
out = [f"# Tool-store agent ({RUN}), {len(A)} graded",
       f"accuracy {100 * sum(A.values()) / len(A):.1f} % ({sum(A.values())}/{len(A)}); status {status}",
       f"searches {st.mean(r['n_searches'] for r in recs.values()):.1f}; reads {st.mean(r['n_read'] for r in recs.values()):.1f}; greps {st.mean(r['n_grep'] for r in recs.values()):.1f}; model calls {st.mean(r['rounds'] for r in recs.values()):.1f}",
       f"gold recall {st.mean(rec):.2f}; reading {100 * sum(A[q] for q in full) / len(full):.0f} % (n {len(full)})",
       f"final prompt median {fin[len(fin) // 2]:,} / p90 {fin[int(0.9 * (len(fin) - 1))]:,} (API input tokens of the last call)",
       f"tokens in / out per question {st.mean(r['cum_input_tokens'] for r in recs.values()) / 1000:.0f}k / {st.mean(r['cum_output_tokens'] for r in recs.values()) / 1000:.0f}k",
       "", "| vs | this / other correct | this only / other only | p |", "|---|---|---|---|"]
for other in ["fd_dense_high150", "fd_rep2", "x7_dense_gpt5", "x7_dense_gpt5_getdoc", "x7_dense_gpt5_getdoc_cap10", "x7_dense_gpt5_getdoc_cap10_k25", "snip_dense_high150", "fd_k5"]:
    B = ok(other); qs = [q for q in A if q in B]
    x = sum(A[q] and not B[q] for q in qs); y = sum(B[q] and not A[q] for q in qs)
    out.append(f"| {other} | {sum(A[q] for q in qs)} / {sum(B[q] for q in qs)} | {x} / {y} | {binomtest(x, x + y, .5).pvalue if x + y else 1:.4f} |")
(ROOT / "data/analysis/x8_toolstore.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
