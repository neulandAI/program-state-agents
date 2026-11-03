#!/usr/bin/env python3
"""Holm's correction over the 18 paired comparisons of Table II on the 150 questions (exact McNemar).

Run: python analysis/holm_family.py -> data/analysis/holm_family.md. Reads data/x3/<run>/graded_gpt41.json; the run
records are not part of the release."""
import json
from pathlib import Path
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
FAMILY = [  # (first, second, label)
    ("fd_dense_high150", "x7_dense_gpt5", "ours vs reference agent"),
    ("fd_dense_high150", "x7_dense_gpt5_getdoc", "ours vs + document tool"),
    ("fd_dense_high150", "x7_dense_gpt5_getdoc_cap10", "ours vs + document tool, cap 10"),
    ("fd_dense_high150", "x7_dense_gpt5_getdoc_cap10_k25", "ours vs + document tool, cap 10, 25 snippets"),
    ("fd_dense_high150", "x7_dense_gpt5_sum30k", "ours vs summarised history"),
    ("fd_dense_high150", "snip_dense_high150", "ours vs snippet REPL agent"),
    ("fd_dense_high150", "fd_k5", "k ~ 25 vs k = 5 (run 1)"),
    ("fd_dense_high150", "fd_k5_rep2", "k ~ 25 vs k = 5 (run 2)"),
    ("fd_k5", "fd_k5prompt", "program state vs prompt only, k = 5 (run 1)"),
    ("fd_k5_rep2", "fd_k5prompt", "program state vs prompt only, k = 5 (run 2)"),
    ("fd_dense_high150", "x9_toolstore_k25", "code vs tools, k = 25 (run 1)"),
    ("fd_rep2", "x9_toolstore_k25", "code vs tools, k = 25 (run 2)"),
    ("fd_dense_high150", "x8_toolstore", "code vs tools, k chosen (run 1)"),
    ("fd_rep2", "x8_toolstore", "code vs tools, k chosen (run 2)"),
    ("fd_dense_high150", "fd_nogrep", "with vs without grep tool"),
    ("fd_cap20_150", "fd_dense_high150", "cap 20 vs cap 10"),
    ("x7_dense_gpt5_sum30k", "x7_dense_gpt5_trim30k", "summarised vs trimmed"),
    ("x7_dense_gpt5", "x7_dense_gpt5_trim30k", "untouched vs trimmed"),
]


def ok(run):
    return {q: bool(v["correct"]) for q, v in json.load(open(ROOT / "data/x3" / run / "graded_gpt41.json")).items()}


rows = []
for a, b, lab in FAMILY:
    oa, ob = ok(a), ok(b); qs = sorted(set(oa) & set(ob))
    x = sum(oa[q] and not ob[q] for q in qs); y = sum(ob[q] and not oa[q] for q in qs)
    rows.append(dict(label=lab, n=len(qs), b=x, c=y, p=binomtest(x, x + y, 0.5).pvalue if x + y else 1.0))
m = len(rows); order = sorted(range(m), key=lambda i: rows[i]["p"]); alive = True
for rank, i in enumerate(order):
    thr = 0.05 / (m - rank); rows[i]["thr"] = thr
    alive = alive and rows[i]["p"] <= thr; rows[i]["holm"] = alive
md = [f"# Holm over {m} contrasts (Table II, 150 questions)\n", "| Contrast | n | first only / second only | p | Holm threshold | survives |", "|---|---|---|---|---|---|"]
for i in order:
    r = rows[i]; md.append(f"| {r['label']} | {r['n']} | {r['b']} / {r['c']} | {r['p']:.2g} | {r['thr']:.4f} | {'yes' if r['holm'] else 'no'} |")
(ROOT / "data/analysis/holm_family.md").write_text("\n".join(md) + "\n"); print("\n".join(md))
