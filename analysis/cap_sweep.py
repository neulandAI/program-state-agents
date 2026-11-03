#!/usr/bin/env python3
"""Search-cap sweep of the document REPL agent on 50 of the 150 questions (caps 5 to 30, paired against cap 10).

Run: python analysis/cap_sweep.py [--no-judge] -> data/analysis/cap_sweep.{csv,md}. Reads data/x3/; the run records
are not part of the release.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import shutil
import statistics as st
import subprocess
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv/bin/python")
CAPS = [5, 10, 15, 20, 25, 30]
IDS = json.load(open(ROOT / "data/x3/fd_cap_ids50.json"))
ENV = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")


def wilson(k, n, z=1.96):
    if not n:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def run_name(cap): return f"fd_cap{cap:02d}"


def assemble_cap10():
    """The cap-10 arm is the 150-question run restricted to the 50 sample questions."""
    src = ROOT / "data/x3/fd_dense_high150"; dst = ROOT / "data/x3" / run_name(10); (dst / "runs").mkdir(parents=True, exist_ok=True)
    g = json.load(open(src / "graded_gpt41.json")); gg = {}
    for q in IDS:
        if (src / "runs" / f"{q}.json").exists():
            shutil.copy(src / "runs" / f"{q}.json", dst / "runs" / f"{q}.json")
        if q in g:
            gg[q] = g[q]
    json.dump(gg, open(dst / "graded_gpt41.json", "w"), indent=1)
    return len(gg)


def convert_and_judge(cap, judge=True):
    src = ROOT / "data/x3/fulldoc_runs" / f"fd_dense_high50_cap{cap:02d}"
    if not src.exists():
        return False
    subprocess.run([PY, str(ROOT / "scripts/rerun_to_x3.py"), "--fulldoc", str(src), "--run", run_name(cap)], check=True, env=ENV, capture_output=True)
    if judge:
        r = subprocess.run([PY, str(ROOT / "scripts/x3_judge.py"), "--run", run_name(cap), "--no-pairs", "--workers", "16"], env=ENV, capture_output=True, text=True)
        print(f"  cap {cap}: " + " ".join(l for l in r.stdout.splitlines() if l.startswith(run_name(cap))))
    return True


def load(cap):
    d = ROOT / "data/x3" / run_name(cap); g = d / "graded_gpt41.json"
    if not g.exists():
        return None
    gg = json.load(open(g)); out = {}
    for p in glob.glob(str(d / "runs/*.json")):
        r = json.load(open(p)); q = r["query_id"]
        if q in gg and q in IDS:
            out[q] = dict(ok=bool(gg[q]["correct"]), searches=r.get("n_searches") or 0, iters=r.get("rounds") or r.get("model_calls") or 0, ctx=r.get("final_context_tokens"),
                          rec=r.get("gold_recall"), cin=r.get("cum_input_tokens") or 0, cout=r.get("cum_output_tokens") or 0, status=r.get("status"), t=r.get("wall_time_s"))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-judge", action="store_true"); a = ap.parse_args()
    print("cap 10 assembled:", assemble_cap10(), "graded of", len(IDS))
    for cap in CAPS:
        if cap != 10:
            convert_and_judge(cap, judge=not a.no_judge)
    arms = {cap: load(cap) for cap in CAPS}; arms = {c: d for c, d in arms.items() if d}
    base = arms.get(10, {})
    rows = []
    for cap, d in arms.items():
        qq = [q for q in IDS if q in d]; k = sum(d[q]["ok"] for q in qq); n = len(qq); lo, hi = wilson(k, n)
        common = [q for q in qq if q in base]; a_ = sum(d[q]["ok"] and not base[q]["ok"] for q in common); b_ = sum(base[q]["ok"] and not d[q]["ok"] for q in common)
        rows.append(dict(cap=cap, n=n, correct=k, accuracy=k / n if n else None, ci_low=lo, ci_high=hi,
                         median_searches=st.median(d[q]["searches"] for q in qq), mean_searches=st.mean(d[q]["searches"] for q in qq),
                         share_at_search_cap=sum(d[q]["searches"] >= cap for q in qq) / n, share_at_iter_cap=sum(d[q]["iters"] >= 30 for q in qq) / n,
                         median_iterations=st.median(d[q]["iters"] for q in qq), mean_gold_recall=st.mean([d[q]["rec"] for q in qq if d[q]["rec"] is not None]),
                         median_final_prompt=st.median([d[q]["ctx"] for q in qq if d[q]["ctx"]]) if any(d[q]["ctx"] for q in qq) else None,
                         cost_per_q=st.mean(d[q]["cin"] / 1e6 * 1.25 + d[q]["cout"] / 1e6 * 10 for q in qq), median_time_s=st.median([d[q]["t"] for q in qq if d[q]["t"]]) if any(d[q]["t"] for q in qq) else None,
                         n_vs_cap10=len(common), only_this=a_, only_cap10=b_, mcnemar_p_vs_cap10=(binomtest(a_, a_ + b_, 0.5).pvalue if a_ + b_ else None),
                         errors=sum(d[q]["status"] != "completed" for q in qq)))
    rows.sort(key=lambda r: r["cap"])
    with open(ROOT / "data/analysis/cap_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    md = ["# Full-Doc RLM search-cap sweep · 50 stratified questions of the 150 subset · dense, GPT-5 high, gpt-4.1 judge\n",
          "Cap 10 is the 150-question re-run restricted to the sample. Iteration cap 30 in every arm. Cost at list price, uncached.\n",
          "| Cap | n | Accuracy | 95 % | Median searches | Mean | At search cap | At 30 iterations | Median prompt | Cost / q | vs cap 10: only this / only cap 10 | p |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['cap']} | {r['n']} | {r['accuracy']:.3f} ({r['correct']}/{r['n']}) | {r['ci_low']:.2f}–{r['ci_high']:.2f} | {r['median_searches']:.0f} | {r['mean_searches']:.1f} | {r['share_at_search_cap']:.0%} | {r['share_at_iter_cap']:.0%} | {r['median_final_prompt']:,.0f} | {r['cost_per_q']:.2f} | {r['only_this']} / {r['only_cap10']} | {('%.2g' % r['mcnemar_p_vs_cap10']) if r['mcnemar_p_vs_cap10'] is not None else '–'} |")
    (ROOT / "data/analysis/cap_sweep.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
