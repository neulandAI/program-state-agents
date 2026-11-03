#!/usr/bin/env python3
"""Fetch-then-Explore on BrowseComp-Plus: the numbers the paper reports (Table I row, Table II held-out row, RQ1 text).

Inputs (run records, not released): data/x3/fte_<run>/{graded_gpt41.json, runs/*.json} written by agents/fetch_then_explore/fte_to_x3.py and
x3_judge.py; the run's predictions.jsonl (for the documents fetched into the workspace); data/x3/fd_cap20_830 (our
document REPL agent, cap 20) for the paired comparison; data/x3/subset_150_ids.json; the local corpus for exact o200k
token counts of fetched documents (State column, counted as in analysis/received_text.py).

Outputs: data/analysis/per_question/fte_<run>.csv (release schema) and data/analysis/fte_<run>_summary.md.
  python analysis/fte_summary.py [run]      # run defaults to full830
"""
import csv, glob, json, statistics as st, sys
from math import comb
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parents[1]
RUN = sys.argv[1] if len(sys.argv) > 1 else "full830"
PRED = ROOT / f"data/x3/fte/runs/browsecomp_plus/{RUN}/predictions.jsonl"
CORPUS = Path("/path/to/rlm/data/prepared/corpus.jsonl")
FIELDS = ["query_id", "correct", "gold_recall", "n_searches", "final_context_tokens", "cum_input_tokens", "cum_output_tokens", "status"]


def load(run):
    g = json.load(open(ROOT / f"data/x3/{run}/graded_gpt41.json"))
    recs = {}
    for f in glob.glob(str(ROOT / f"data/x3/{run}/runs/*.json")):
        r = json.load(open(f)); recs[str(r["query_id"])] = r
    return {q: dict(recs.get(q, {}), correct=bool(v.get("correct"))) for q, v in g.items()}


def mcnemar(a, b, qs):
    x = sum(a[q]["correct"] and not b[q]["correct"] for q in qs); y = sum(b[q]["correct"] and not a[q]["correct"] for q in qs)
    n, k = x + y, min(x, y)
    return x, y, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def tok(r):
    return (r.get("cum_input_tokens") or 0) + (r.get("cum_output_tokens") or 0)


def main():
    fte, ours = load(f"fte_{RUN}"), load("fd_cap20_830")
    dev = set(map(str, json.load(open(ROOT / "data/x3/subset_150_ids.json"))))
    held = [q for q in fte if q not in dev]
    assert len(fte) == 830 and len(dev & set(fte)) == 150, (len(fte), len(dev & set(fte)))

    # documents fetched into the workspace, from the latest record of each question
    fetched = {}
    for line in open(PRED, encoding="utf-8"):
        r = json.loads(line); q = str(r["metadata"]["item_metadata"]["query_id"]); ids = set()
        for c in r["metadata"].get("model_calls") or []:
            for tc in c.get("tool_calls") or []:
                if tc.get("name") == "fetch":
                    a = tc.get("arguments") or {}
                    a = json.loads(a) if isinstance(a, str) else a
                    for v in a.values():
                        ids.update(str(x).strip() for x in (v if isinstance(v, list) else [v]))
        fetched[q] = ids
    need = set().union(*(fetched[q] for q in dev))
    enc = tiktoken.get_encoding("o200k_base"); doclen = {}
    for line in open(CORPUS, encoding="utf-8"):
        d = json.loads(line); i = str(d.get("docid") or d.get("id"))
        if i in need:
            doclen[i] = len(enc.encode(d.get("text") or d.get("contents") or "", disallowed_special=()))

    out_dir = ROOT / "data/analysis/per_question"; out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"fte_{RUN}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS); w.writeheader()
        for q in sorted(fte, key=int):
            w.writerow({k: fte[q].get(k) for k in FIELDS} | {"correct": int(fte[q]["correct"])})

    d = [fte[q] for q in dev]; every = [q for q in dev if (fte[q].get("gold_recall") or 0) >= 1.0]
    state = [sum(doclen.get(i, 0) for i in fetched[q]) for q in dev]
    md = [f"# Fetch-then-Explore ({RUN}) on BrowseComp-Plus, GPT-5 high, dense retrieval, gpt-4.1 judge\n",
          "## Table I row (150 development questions)\n",
          "| Acc.% | Searches | Recall | Reading% (n) | Prompt (median) | State (median) | Tokens in / out (mean) |", "|---|---|---|---|---|---|---|",
          f"| {100 * sum(r['correct'] for r in d) / 150:.1f} | {st.mean(r['n_searches'] for r in d):.1f} | {st.mean(r['gold_recall'] for r in d):.2f} | "
          f"{100 * sum(fte[q]['correct'] for q in every) / len(every):.0f} ({len(every)}) | {st.median(r['final_context_tokens'] for r in d) / 1e3:.1f}k | "
          f"{st.median(state) / 1e3:.0f}k | {st.mean(r['cum_input_tokens'] for r in d) / 1e3:.0f}k / {st.mean(r['cum_output_tokens'] for r in d) / 1e3:.0f}k |\n",
          "## Paired comparison with the document REPL agent, cap 20 (exact McNemar)\n",
          "| Questions | n | Fetch-then-Explore % | Ours % | only ours | only FtE | p | recall FtE / ours | tokens per question, mean (median), FtE / ours |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, qs in (("all", list(fte)), ("held out", held), ("development", sorted(dev))):
        x, y, p = mcnemar(ours, fte, qs)
        tf, to = [tok(fte[q]) for q in qs], [tok(ours[q]) for q in qs]
        md.append(f"| {name} | {len(qs)} | {100 * sum(fte[q]['correct'] for q in qs) / len(qs):.1f} | {100 * sum(ours[q]['correct'] for q in qs) / len(qs):.1f} | "
                  f"{x} | {y} | {p:.2g} | {st.mean(fte[q]['gold_recall'] for q in qs):.2f} / {st.mean(ours[q]['gold_recall'] for q in qs):.2f} | "
                  f"{st.mean(tf) / 1e3:.0f}k ({st.median(tf) / 1e3:.0f}k) / {st.mean(to) / 1e3:.0f}k ({st.median(to) / 1e3:.0f}k) |")
    (ROOT / f"data/analysis/fte_{RUN}_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
