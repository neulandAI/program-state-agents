#!/usr/bin/env python3
"""Failure modes on the full sets (Fig. 4): no answer, nothing found, partly found, all gold returned but wrong.

No answer: unfinished, empty or abstained (for Fetch-then-Explore also a final message without its <answer> tags,
which in its run was always a request for clarification). A gold document counts as returned when a search returned it.

Run: python analysis/failure_modes_full.py -> data/analysis/failure_modes_full.csv. Reads
data/failure_analysis/qa_summary.csv and data/x3/<run>/; the run records are not part of the release."""
import collections, csv, glob, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
with open(ROOT / "data/failure_analysis/qa_summary.csv") as fh:
    arch = {r["Exp ID"]: r for r in csv.DictReader(fh.read().splitlines()[2:]) if r.get("Exp ID")}
def archived(exp, group, label):
    r = arch[exp]
    return dict(group=group, label=label, source=exp, n=int(r["Questions"]), right=int(r["Correct"]),
                no_answer=int(r["F1 did not finish"]) + int(r["F2 no / abstained"]), nothing_found=int(r["F3 retrieval miss"]),
                partly_found=int(r["F4 partial evidence"]), found_misread=int(r["F5 all gold, wrong"]))
ABSTAIN = re.compile(r"^\s*(n/?a|none|unknown|not (found|available|determined|known)|cannot|can't|unable|"
                     r"i (don.t|do not) know|insufficient|no answer|undetermined|unanswerable)", re.I)


def exact_answer(ans):
    m = re.search(r"exact answer\s*[:\uff1a]\s*(.*)", ans, re.I)
    return (m.group(1) if m else ans).strip().strip("*").strip()


def gold_split(exp, group, label):
    """BM25 cap-10 document REPL row: split by gold recall from the raw run records (the archived split used evidence recall)."""
    cats = collections.Counter(r["category"] for r in csv.DictReader(open(ROOT / "data/failure_analysis/bcp25_gold_categories.csv")))
    return dict(group=group, label=label, source=exp, n=sum(cats.values()), right=cats["right"], no_answer=cats["F1"] + cats["F2"],
                nothing_found=cats["F3"], partly_found=cats["F4"], found_misread=cats["F5"])


def new(run, group, label):
    g = {q: bool(v["correct"]) for q, v in json.load(open(ROOT / "data/x3" / run / "graded_gpt41.json")).items()}
    c = dict(group=group, label=label, source=run, n=0, right=0, no_answer=0, nothing_found=0, partly_found=0, found_misread=0)
    for f in glob.glob(str(ROOT / "data/x3" / run / "runs/*.json")):
        r = json.load(open(f)); q = str(r["query_id"])
        if q not in g: continue
        c["n"] += 1
        if g[q]: c["right"] += 1; continue
        ans = (r.get("final_answer") or "").strip(); rec = r.get("gold_recall")
        if r.get("status") not in (None, "completed") or not ans or r.get("timed_out") or rec is None \
                or not exact_answer(ans) or ABSTAIN.match(exact_answer(ans)) \
                or (run.startswith("fte_") and "<answer>" not in ans): c["no_answer"] += 1   # FtE: no <answer> = asked back
        elif rec == 0: c["nothing_found"] += 1
        elif rec >= 1: c["found_misread"] += 1
        else: c["partly_found"] += 1
    return c
rows = [
    archived("BCP-9", "BrowseComp-Plus, dense", "delegated, no cap"),
    new("fd_cap20_830", "BrowseComp-Plus, dense", "active, cap 20"),
    new("fte_full830", "BrowseComp-Plus, dense", "fetch-then-explore, no cap"),   # released harness, unmodified
    archived("BCP-11", "BrowseComp-Plus, late interaction", "delegated, no cap"),
    new("fd_colbert20_830", "BrowseComp-Plus, late interaction", "active, cap 20"),
    archived("BCP-13", "BrowseComp-Plus, BM25", "delegated, no cap"),
    gold_split("BCP-25", "BrowseComp-Plus, BM25", "active, cap 10"),
    archived("FR-7", "FRAMES, BM25", "delegated, cap 25"),
    archived("FR-11", "FRAMES, BM25", "active, cap 10"),
    # open-weight model (Qwen3.6-35B-A3B); the delegated BM25 row's failure split exists only for a partial run and is left out
]
rows.sort(key=lambda r: (["BrowseComp-Plus, dense", "BrowseComp-Plus, late interaction", "BrowseComp-Plus, BM25", "FRAMES, BM25"].index(r["group"]),
                         0 if "Qwen" not in r["label"] else 1, 0 if r["label"].startswith("delegated") else 1, r["label"]))
with open(ROOT / "data/analysis/failure_modes_full.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for r in rows: print(f"{r['group']:36s} {r['label']:18s} n {r['n']} right {r['right']} | no answer {r['no_answer']} nothing {r['nothing_found']} partly {r['partly_found']} misread {r['found_misread']}")
