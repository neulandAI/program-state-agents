#!/usr/bin/env python3
"""Table I from the per-question records: accuracy, searches, gold recall, reading, prompt size and tokens per run.

Run: python analysis/paper_tables.py -> paper/tables/table1_main.md and data/analysis/table1_main.csv. Reads
data/x3/<run>/ and data/analysis/exact_prompt_tokens.csv; the run records are not part of the release.
"""
import csv, glob, json, math, statistics as st
from pathlib import Path
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
ARMS = [  # (run, label, family)
    ("x7_dense_gpt5", "Reference agent", "std"),
    ("x7_dense_gpt5_getdoc", "Reference + document tool", "std"),
    ("x7_dense_gpt5_getdoc_cap10", "+ document tool, cap 10", "std"),
    ("x7_dense_gpt5_getdoc_cap10_k25", "+ document tool, cap 10, k = 25", "std"),
    ("x7_dense_gpt5_trim30k", "Reference, trimmed to 30k", "std"),
    ("x7_dense_gpt5_sum30k", "Reference, summarised at 30k", "std"),
    ("snip_dense_high150", "Snippet REPL, no cap", "rlm"),
    ("snip_cap10", "Snippet REPL, cap 10", "rlm"),
    ("snip_k25", "Snippet REPL, 25 per search", "rlm"),
    ("fd_dense_high150", "Document REPL, k≤50, cap 10", "rlm"),
    ("fd_nogrep", "Document REPL, no grep tool", "rlm"),
    ("fd_k5", "Document REPL, k=5, cap 10", "rlm"),
    ("fd_k5prompt", "Document REPL, k=5, prompt only", "rlm"),
    ("fd_cap20_150", "Document REPL, k≤50, cap 20", "rlm"),
]


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def load(run):
    g = json.load(open(ROOT / "data/x3" / run / "graded_gpt41.json"))
    recs = {str(json.load(open(f))["query_id"]): json.load(open(f)) for f in glob.glob(str(ROOT / "data/x3" / run / "runs" / "*.json"))}
    return {q: bool(v["correct"]) for q, v in g.items()}, recs


def std_tokens(run):
    """Mean input / output tokens per question and the token-weighted cached share, from per-call API usage."""
    ins, outs, cached = [], [], []
    for f in glob.glob(str(ROOT / "data/x3/official" / run / "run_*.json")):
        pcu = json.load(open(f)).get("per_call_usage") or []
        ins.append(sum(c.get("input_tokens", 0) for c in pcu)); outs.append(sum(c.get("output_tokens", 0) for c in pcu))
        cached.append(sum(c.get("cached_tokens", 0) for c in pcu))
    return (st.mean(ins), st.mean(outs), sum(cached) / sum(ins)) if ins else (float("nan"),) * 3


def main():
    exact = {}
    with open(ROOT / "data/analysis/exact_prompt_tokens.csv") as fh:
        for r in csv.DictReader(fh):
            exact.setdefault(r["run"], {})[r["query_id"]] = int(r["final_tokens"])
    std_ok, _ = load("x7_dense_gpt5")
    rows = []
    for run, label, fam in ARMS:
        ok, recs = load(run); n = len(ok); right = sum(ok.values()); lo, hi = wilson(right, n)
        if run == "x7_dense_gpt5":
            oa = ob = None; p = None
        else:
            qs = sorted(set(ok) & set(std_ok)); oa = sum(ok[q] and not std_ok[q] for q in qs); ob = sum(std_ok[q] and not ok[q] for q in qs)
            p = binomtest(oa, oa + ob, 0.5).pvalue if oa + ob else 1.0
        searches = [r.get("n_searches") or 0 for r in recs.values()]
        rec_vals = [r.get("gold_recall") for r in recs.values() if r.get("gold_recall") is not None]
        every = [q for q, r in recs.items() if (r.get("gold_recall") or 0) >= 1.0]
        acc_every = sum(ok[q] for q in every) / len(every) if every else float("nan")
        if fam == "rlm":
            toks = list(exact.get(run, {}).values())
        else:
            toks = [r["final_context_tokens"] for r in recs.values() if r.get("final_context_tokens")]
        wall = [r["wall_time_s"] for r in recs.values() if r.get("wall_time_s")]
        if fam == "std":
            tok_in, tok_out, cached_share = std_tokens(run)
        else:
            tok_in = st.mean([r.get("cum_input_tokens") or 0 for r in recs.values()]); tok_out = st.mean([r.get("cum_output_tokens") or 0 for r in recs.values()]); cached_share = float("nan")
        rows.append(dict(run=run, label=label, right=right, n=n, acc=right / n, lo=lo, hi=hi, only_arm=oa, only_std=ob, p=p,
                         searches=st.mean(searches), recall=st.mean(rec_vals), every=len(every) / n, acc_every=acc_every,
                         tok_median=st.median(toks), tok_p90=pct(toks, 0.9), wall_median=st.median(wall), tok_in=tok_in, tok_out=tok_out, cached_share=cached_share))
    (ROOT / "paper/tables").mkdir(exist_ok=True)
    with open(ROOT / "data/analysis/table1_main.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    md = ["| Agent | Acc. % (95 % CI) | vs ref.: only/only, p | Srch/q | Recall | All gold | Acc. with all gold | Prompt med/p90 | Wall | Tokens/q in / out |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        pair = "–" if r["p"] is None else f"{r['only_arm']}/{r['only_std']}, {r['p']:.2g}".replace("0.0", ".0", 1) if r["p"] is not None and r["p"] < 1 else ("–" if r["p"] is None else f"{r['only_arm']}/{r['only_std']}, 1")
        cost = f"{r['tok_in']/1000:.0f}k / {r['tok_out']/1000:.0f}k"
        if r["run"].startswith("snip"):
            cost = "≥" + cost
        md.append(f"| {r['label']} | {100*r['acc']:.1f} ({100*r['lo']:.0f}–{100*r['hi']:.0f}) | {pair} | {r['searches']:.1f} | {r['recall']:.2f} | {100*r['every']:.0f} % | {100*r['acc_every']:.0f} % | {r['tok_median']/1000:.0f}k/{r['tok_p90']/1000:.0f}k | {r['wall_median']:.0f} s | {cost} |")
    md.append("")
    md.append("Noise floor: two runs of the same configuration disagree on 19–22 of 150 questions with a net difference of 0–1 (archived vs re-run, delegated- and active-observation agents). Delegated-observation token counts are lower bounds (≥; the harness under-counts input tokens when the sub-model ran). Tokens are means per question over all model calls; for the reference agents 35-44 % of the input tokens were served from cache (token-weighted, from the API usage); the RLM library did not record cached tokens. Source: `scripts/paper_tables.py`.")
    (ROOT / "paper/tables/table1_main.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
