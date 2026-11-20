#!/usr/bin/env python3
"""No-grep, 25-snippet and history ablations on the 150 questions, and the 680 held-out comparison: per-run summary
and paired exact McNemar tests.

Run: python analysis/ablation_g.py -> data/analysis/ablation_g.{md,csv}. Reads data/x3/<run>/ and
data/analysis/exact_prompt_tokens.csv; the run records are not part of the release.
"""
import csv, glob, json, statistics as st
from pathlib import Path
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
RUNS = ["x7_dense_gpt5", "x7_dense_gpt5_getdoc", "x7_dense_gpt5_trim30k", "x7_dense_gpt5_sum30k", "fd_dense_high150", "fd_nogrep",
        "snip_dense_high150", "snip_k25", "x7_dense_gpt5_getdoc_680", "fd_cap20_830"]
PAIRS = [("fd_nogrep", "fd_dense_high150", "no grep_docs vs with (document REPL, cap 10)"),
         ("fd_nogrep", "x7_dense_gpt5", "document REPL without grep_docs vs reference agent"),
         ("fd_nogrep", "x7_dense_gpt5_getdoc", "document REPL without grep_docs vs reference + document tool"),
         ("snip_k25", "snip_dense_high150", "25 vs 5 snippets per search (snippet REPL, no cap)"),
         ("snip_k25", "fd_dense_high150", "snippet REPL, 25 per search vs document REPL, cap 10"),
         ("x7_dense_gpt5_sum30k", "x7_dense_gpt5_trim30k", "summarised vs trimmed to 30k (reference agent)"),
         ("x7_dense_gpt5_sum30k", "x7_dense_gpt5", "summarised to 30k vs reference agent"),
         ("x7_dense_gpt5_sum30k", "fd_dense_high150", "reference summarised to 30k vs document REPL, cap 10"),
         ("x7_dense_gpt5_getdoc_680", "fd_cap20_830", "held-out 680: reference + document tool vs document REPL, cap 20")]


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
    md = ["# Experiment G: round-5 review runs (GPT-5 high, dense retriever; 150 questions unless marked)\n",
          "| Run | n | Accuracy % | Gold recall % | All gold found % | Acc. with all gold % | Searches (mean) | get_document (mean) | Final prompt median | Input / output tokens per q (mean) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for run in RUNS:
        ok, recs = data[run]
        if ok is None:
            md.append(f"| {run} | – (not judged yet) | | | | | | | | |"); continue
        n = len(ok); acc = 100 * sum(ok.values()) / n
        rec = [r.get("gold_recall") for r in recs.values() if r.get("gold_recall") is not None]
        allg = 100 * sum(1 for x in rec if x >= 1) / len(rec)
        full = [q for q, r in recs.items() if (r.get("gold_recall") or 0) >= 1 and q in ok]
        acc_all = 100 * sum(ok[q] for q in full) / len(full) if full else float("nan")
        se = st.mean(r.get("n_searches") or 0 for r in recs.values())
        gd = st.mean(r.get("n_get_document") or 0 for r in recs.values())
        fin = exact.get(run) or {q: r.get("final_context_tokens") for q, r in recs.items() if r.get("final_context_tokens")}
        fmed = st.median(fin.values()) if fin else float("nan")
        tin = st.mean(r.get("cum_input_tokens") or 0 for r in recs.values()); tout = st.mean(r.get("cum_output_tokens") or 0 for r in recs.values())
        rows.append(dict(run=run, n=n, acc=round(acc, 1), recall=round(100 * st.mean(rec), 1), all_gold=round(allg, 1), acc_all_gold=round(acc_all, 1), searches=round(se, 1),
                         get_document=round(gd, 1), final_prompt_median=int(fmed), tokens_in=int(tin), tokens_out=int(tout)))
        md.append(f"| {run} | {n} | {acc:.1f} | {100*st.mean(rec):.1f} | {allg:.1f} | {acc_all:.1f} | {se:.1f} | {gd:.1f} | {int(fmed):,} | {int(tin):,} / {int(tout):,} |")
    md += ["", "## Paired comparisons (exact McNemar; b = only first right, c = only second right)\n",
           "| First | Second | What | Acc. first / second | b / c | p |", "|---|---|---|---|---|---|"]
    for a, b, what in PAIRS:
        oa, _ = data[a]; ob, _ = data[b]
        if oa is None or ob is None:
            md.append(f"| {a} | {b} | {what} | pending | | |"); continue
        qs = sorted(set(oa) & set(ob)); x = sum(oa[q] and not ob[q] for q in qs); y = sum(ob[q] and not oa[q] for q in qs)
        p = binomtest(x, x + y, 0.5).pvalue if x + y else 1.0
        md.append(f"| {a} | {b} | {what} | {100*sum(oa[q] for q in qs)/len(qs):.1f} / {100*sum(ob[q] for q in qs)/len(qs):.1f} | {x} / {y} | {p:.3f} |")
    (ROOT / "data/analysis/ablation_g.md").write_text("\n".join(md) + "\n")
    with open(ROOT / "data/analysis/ablation_g.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("\n".join(md))


if __name__ == "__main__":
    main()
