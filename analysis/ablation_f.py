#!/usr/bin/env python3
"""Ablations and second runs on the 150 questions: per-run summary and paired exact McNemar tests.

Run: python analysis/ablation_f.py -> data/analysis/ablation_f.{md,csv}. Reads data/x3/<run>/ and
data/analysis/exact_prompt_tokens.csv; the run records are not part of the release.
"""
import csv, glob, json, statistics as st
from pathlib import Path
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
RUNS = ["x7_dense_gpt5", "x7_dense_gpt5_rep2", "x7_dense_gpt5_getdoc", "x7_dense_gpt5_getdoc_cap10", "x7_dense_gpt5_getdoc_cap10_k25",
        "fd_dense_high150", "fd_rep2", "fd_k5", "fd_k5print", "snip_dense_high150", "snip_cap10"]
PAIRS = [("fd_k5print", "fd_k5", "documents printed into the prompt vs held in program state (k = 5, cap 10)"),
         ("x7_dense_gpt5_getdoc_cap10", "fd_dense_high150", "reference + document tool, cap 10 vs active, cap 10"),
         ("x7_dense_gpt5_getdoc_cap10_k25", "fd_dense_high150", "reference + document tool, cap 10, k = 25 vs active, cap 10"),
         ("x7_dense_gpt5_getdoc_cap10_k25", "x7_dense_gpt5_getdoc_cap10", "k = 25 vs k = 5 (reference + document tool, cap 10)"),
         ("x7_dense_gpt5_getdoc_cap10", "x7_dense_gpt5_getdoc", "cap 10 vs no cap (reference + document tool)"),
         ("x7_dense_gpt5_rep2", "x7_dense_gpt5", "second run vs first run (reference agent)"),
         ("fd_rep2", "fd_dense_high150", "second run vs first run (active, cap 10)"),
         ("fd_k5print", "x7_dense_gpt5", "documents in prompt (k = 5) vs reference agent"),
         ("fd_rep2", "x7_dense_gpt5_rep2", "active vs reference, second runs")]


def load(run):
    gp = ROOT / "data/x3" / run / "graded_gpt41.json"
    if not gp.exists():
        return None, None
    g = json.load(open(gp))
    recs = {}
    for f in glob.glob(str(ROOT / "data/x3" / run / "runs" / "*.json")):
        r = json.load(open(f)); recs[str(r["query_id"])] = r
    return {q: bool(v["correct"]) for q, v in g.items()}, recs


def main():
    exact = {}
    with open(ROOT / "data/analysis/exact_prompt_tokens.csv") as fh:
        for r in csv.DictReader(fh):
            exact.setdefault(r["run"], {})[r["query_id"]] = int(r["final_tokens"])
    data = {r: load(r) for r in RUNS}
    rows = []
    md = ["# Experiment F: ablations and repeat runs (150 questions, GPT-5 high, dense retriever)\n",
          "| Run | n | Accuracy % | Gold recall % | All gold found % | Searches (mean) | get_document (mean) | Final prompt median | Input / output tokens per q (mean) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for run in RUNS:
        ok, recs = data[run]
        if ok is None:
            md.append(f"| {run} | – (not judged yet) | | | | | | | |"); continue
        n = len(ok); acc = 100 * sum(ok.values()) / n
        rec = [r.get("gold_recall") for r in recs.values() if r.get("gold_recall") is not None]
        allg = 100 * sum(1 for x in rec if x >= 1) / len(rec)
        se = st.mean(r.get("n_searches") or 0 for r in recs.values())
        gd = st.mean(r.get("n_get_document") or 0 for r in recs.values())
        fin = exact.get(run) or {q: r.get("final_context_tokens") for q, r in recs.items() if r.get("final_context_tokens")}
        fmed = st.median(fin.values()) if fin else float("nan")
        tin = st.mean(r.get("cum_input_tokens") or 0 for r in recs.values()); tout = st.mean(r.get("cum_output_tokens") or 0 for r in recs.values())
        rows.append(dict(run=run, n=n, acc=round(acc, 1), recall=round(100 * st.mean(rec), 1), all_gold=round(allg, 1), searches=round(se, 1),
                         get_document=round(gd, 1), final_prompt_median=int(fmed), tokens_in=int(tin), tokens_out=int(tout)))
        md.append(f"| {run} | {n} | {acc:.1f} | {100*st.mean(rec):.1f} | {allg:.1f} | {se:.1f} | {gd:.1f} | {int(fmed):,} | {int(tin):,} / {int(tout):,} |")
    md += ["", "## Paired comparisons (exact McNemar; b = only first right, c = only second right)\n",
           "| First | Second | What | Acc. first / second | b / c | p |", "|---|---|---|---|---|---|"]
    for a, b, what in PAIRS:
        oa, _ = data[a]; ob, _ = data[b]
        if oa is None or ob is None:
            md.append(f"| {a} | {b} | {what} | pending | | |"); continue
        qs = sorted(set(oa) & set(ob)); x = sum(oa[q] and not ob[q] for q in qs); y = sum(ob[q] and not oa[q] for q in qs)
        p = binomtest(x, x + y, 0.5).pvalue if x + y else 1.0
        md.append(f"| {a} | {b} | {what} | {100*sum(oa[q] for q in qs)/len(qs):.1f} / {100*sum(ob[q] for q in qs)/len(qs):.1f} | {x} / {y} | {p:.3f} |")
    (ROOT / "data/analysis/ablation_f.md").write_text("\n".join(md) + "\n")
    with open(ROOT / "data/analysis/ablation_f.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("\n".join(md))


if __name__ == "__main__":
    main()
