#!/usr/bin/env python3
"""Full-set summary of a document REPL run on all 830 BrowseComp-Plus questions (Table III): accuracy, behaviour,
paired exact McNemar tests against the earlier full-set runs, and judge consistency on the 150 questions judged twice.

Run: python analysis/full830_summary.py [--run fd_cap20_830 --out full830] -> data/analysis/<out>_{table,pairs}.*.
Reads data/x3/<run>/ and data/failure_analysis/; the run records are not part of the release.
"""
import argparse
import json, math
from pathlib import Path
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
_ap = argparse.ArgumentParser()
_ap.add_argument("--run", default="fd_cap20_830", help="converted run directory under data/x3/")
_ap.add_argument("--out", default="full830", help="output prefix under data/analysis/")
_ap.add_argument("--title", default="Full-Doc RLM, budget 20, on all 830 BrowseComp-Plus questions · dense (Qwen3-Embedding-8B, exact)")
_ap.add_argument("--same-design", default="BCP-21", help="archived row with the same design and retriever, quoted in the behaviour line")
ARGS = _ap.parse_args()
RUN = ROOT / "data/x3" / ARGS.run
IN_PRICE, OUT_PRICE = 1.25 / 1e6, 10.0 / 1e6


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mcnemar(a, b):
    """a, b: dict qid -> bool on the same keys. Returns only_a, only_b, both, neither, p (exact)."""
    keys = sorted(set(a) & set(b))
    oa = sum(1 for q in keys if a[q] and not b[q]); ob = sum(1 for q in keys if b[q] and not a[q])
    both = sum(1 for q in keys if a[q] and b[q]); nei = len(keys) - oa - ob - both
    p = binomtest(oa, oa + ob, 0.5).pvalue if oa + ob else 1.0
    return dict(n=len(keys), only_a=oa, only_b=ob, both=both, neither=nei, p=p)


def main():
    graded = json.load(open(RUN / "graded_gpt41.json"))
    ok = {q: bool(v["correct"]) for q, v in graded.items()}
    rows = {}
    for f in (RUN / "runs").glob("*.json"):
        r = json.load(open(f)); rows[str(r["query_id"])] = r
    assert set(rows) == set(ok), (len(rows), len(ok))
    qids = sorted(ok, key=int); n = len(qids); right = sum(ok.values())
    lo, hi = wilson(right, n)

    def g(q, k, default=None):
        return rows[q].get(k, default)
    searches = [g(q, "n_searches", 0) for q in qids]
    at_cap = sum(1 for s in searches if s >= 20)
    mean_k = [g(q, "mean_k") for q in qids if g(q, "mean_k") is not None]
    recall = [g(q, "gold_recall") for q in qids if g(q, "gold_recall") is not None]
    every = [q for q in qids if (g(q, "gold_recall") or 0) >= 1.0]
    none_ = [q for q in qids if (g(q, "gold_recall") or 0) == 0.0]
    ctx = sorted(g(q, "final_context_tokens", 0) or 0 for q in qids)
    cost = [(g(q, "cum_input_tokens", 0) or 0) * IN_PRICE + (g(q, "cum_output_tokens", 0) or 0) * OUT_PRICE for q in qids]
    wall = sorted((g(q, "wall_time_s", 0) or 0) for q in qids)
    status = pd.Series([g(q, "status") for q in qids]).value_counts().to_dict()
    forced = sum(1 for q in qids if g(q, "forced"))
    timed_out = sum(1 for q in qids if g(q, "timed_out"))

    def pct(i, xs):
        return xs[min(len(xs) - 1, int(round(i * (len(xs) - 1))))]
    summary = dict(
        run=ARGS.run, n=n, right=right, accuracy=right / n, wilson_lo=lo, wilson_hi=hi,
        mean_searches=sum(searches) / n, median_searches=sorted(searches)[n // 2], share_at_cap=at_cap / n,
        mean_k=sum(mean_k) / len(mean_k), mean_gold_recall=sum(recall) / len(recall),
        share_every_gold=len(every) / n, acc_given_every_gold=sum(ok[q] for q in every) / max(1, len(every)),
        share_no_gold=len(none_) / n, acc_given_no_gold=sum(ok[q] for q in none_) / max(1, len(none_)),
        ctx_median=pct(0.5, ctx), ctx_p90=pct(0.9, ctx), ctx_max=ctx[-1],
        cost_mean=sum(cost) / n, cost_total=sum(cost), wall_median=pct(0.5, wall), wall_mean=sum(wall) / n, wall_max=wall[-1],
        statuses=json.dumps(status), forced=forced, timed_out=timed_out,
    )

    # archived rows on identical questions
    byq = pd.read_csv(ROOT / "data/failure_analysis/bcp_by_question.csv", dtype={"QID": str})
    arch = {col: {str(q): bool(int(v)) for q, v in zip(byq["QID"], byq[f"{col} correct"])} for col in ["BCP-21", "BCP-9", "BCP-23", "BCP-11", "BCP-13", "BCP-25"] if f"{col} correct" in byq.columns}
    labels = {"BCP-21": "Full-Doc RLM, dense, budget 10 (archived, 0.8313)", "BCP-9": "Snippet RLM, dense, unlimited (archived, 0.794)",
              "BCP-23": "Full-Doc RLM, ColBERT, budget 10 (archived, 0.8747)", "BCP-11": "Snippet RLM, ColBERT (archived, 0.8578)",
              "BCP-13": "Snippet RLM, BM25 (archived, 0.6747)", "BCP-25": "Full-Doc RLM, BM25, budget 10 (archived, 0.600)"}
    pairs = []
    for col, d in arch.items():
        m = mcnemar(ok, d)
        pairs.append(dict(A=ARGS.run, B=col, B_label=labels[col], acc_A=right / n, acc_B=sum(d[q] for q in qids) / n, **m))
    pairs_df = pd.DataFrame(pairs)

    # difficulty: number of the six archived GPT-5 configurations that solved the question
    six = ["BCP-9", "BCP-11", "BCP-13", "BCP-21", "BCP-23", "BCP-25"]
    solved = {q: sum(arch[c][q] for c in six if c in arch) for q in qids}
    diff = []
    for k in range(7):
        qs = [q for q in qids if solved[q] == k]
        diff.append(dict(archived_rows_solving=k, n=len(qs), right=sum(ok[q] for q in qs), accuracy=(sum(ok[q] for q in qs) / len(qs)) if qs else float("nan")))

    # BCP-21 behaviour on the same questions (archived per-question rows)
    fo = pd.read_csv(ROOT / "data/analysis/fulldoc_original_rows.csv", dtype={"query_id": str})
    b21 = fo[fo.arm == ARGS.same_design]
    b21_searches = b21.search_calls.mean() if "search_calls" in b21 else float("nan")
    b21_cost = b21.cost_usd.mean() if "cost_usd" in b21 else float("nan")

    # judge consistency: fd_cap20_150 and fd_cap20_830 hold the same 150 records, judged twice
    g150_path = ROOT / "data/x3/fd_cap20_150/graded_gpt41.json"
    if ARGS.run == "fd_cap20_830" and g150_path.exists():
        g150 = json.load(open(g150_path))
        common = [q for q in g150 if q in ok]
        flips = [q for q in common if bool(g150[q]["correct"]) != ok[q]]
        acc150_first = sum(bool(g150[q]["correct"]) for q in common) / len(common); acc150_second = sum(ok[q] for q in common) / len(common)
    else:
        common, flips, acc150_first, acc150_second = [], [], float("nan"), float("nan")

    # write
    pd.DataFrame([summary]).to_csv(ROOT / f"data/analysis/{ARGS.out}_table.csv", index=False)
    pairs_df.to_csv(ROOT / f"data/analysis/{ARGS.out}_pairs.csv", index=False)
    md = [f"# {ARGS.title} · GPT-5 high · gpt-4.1 judge (3 votes)\n",
          "| Right | Accuracy | Wilson 95 % | Searches / q (mean, median) | At the cap of 20 | Mean k | Gold recall | Every gold doc | Acc. with every gold | No gold doc | Acc. with none | Final prompt median / p90 / max (proxy) | Cost / q (list) | Wall time median / mean / max | Statuses |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
          f"| {right}/{n} | {right/n:.4f} | {lo:.3f}–{hi:.3f} | {summary['mean_searches']:.1f}, {summary['median_searches']} | {100*summary['share_at_cap']:.0f} % | {summary['mean_k']:.1f} | {summary['mean_gold_recall']:.3f} | {100*summary['share_every_gold']:.0f} % | {summary['acc_given_every_gold']:.3f} | {100*summary['share_no_gold']:.0f} % | {summary['acc_given_no_gold']:.3f} | {summary['ctx_median']:,} / {summary['ctx_p90']:,} / {summary['ctx_max']:,} | {summary['cost_mean']:.2f} USD (total {summary['cost_total']:.0f}) | {summary['wall_median']:.0f} / {summary['wall_mean']:.0f} / {summary['wall_max']:.0f} s | {status}, forced {forced}, timed out {timed_out} |",
          "", f"Archived {ARGS.same_design} (same design and retriever, budget 10) on the same 830: mean searches {b21_searches:.1f}, cost {b21_cost:.2f} USD/q.", "",
          "## Paired against the archived full-set rows (exact McNemar; identical questions)\n",
          "| Archived row | acc. budget 20 | acc. archived | only budget 20 | only archived | both | neither | p |", "|---|---|---|---|---|---|---|---|"]
    for p in pairs:
        md.append(f"| {p['B_label']} | {p['acc_A']:.4f} | {p['acc_B']:.4f} | {p['only_a']} | {p['only_b']} | {p['both']} | {p['neither']} | {p['p']:.2g} |")
    md += ["", "## Accuracy by archived difficulty (number of the six archived GPT-5 configurations that solved the question)\n",
           "| Archived rows solving | n | Right | Accuracy |", "|---|---|---|---|"]
    for d in diff:
        md.append(f"| {d['archived_rows_solving']} | {d['n']} | {d['right']} | {d['accuracy']:.3f} |")
    if common:
        md += ["", "## Judge consistency on the 150 records judged twice\n",
               f"The 150 subset records are the same files in `fd_cap20_150` and `fd_cap20_830`; judged twice (three votes each time), the verdict differs on {len(flips)} of {len(common)} ({', '.join('q'+q for q in flips) if flips else 'none'}): accuracy {acc150_first:.4f} first judging, {acc150_second:.4f} second."]
    md += ["", f"Source: `scripts/full830_summary.py --run {ARGS.run}`; records `data/x3/{ARGS.run}/`; archived per-question verdicts `data/failure_analysis/bcp_by_question.csv`."]
    (ROOT / f"data/analysis/{ARGS.out}_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
