#!/usr/bin/env python3
"""Retrieved text per question, split into prompt tokens and tokens held in program state (exact o200k counts of the
corpus documents).

Run: python analysis/received_text.py -> data/analysis/received_text.csv. Reads data/x3/ and the local corpus; the run
records and the corpus are not part of the release.
"""
import csv, glob, json, re, statistics as st, sys
from pathlib import Path
import tiktoken

ROOT = Path(__file__).resolve().parents[1]
CORPUS = Path("/path/to/rlm/data/prepared/corpus.jsonl")
enc = tiktoken.get_encoding("o200k_base")

def graded(run): return {q: bool(v["correct"]) for q, v in json.load(open(ROOT / "data/x3" / run / "graded_gpt41.json")).items()}
exact = {}
for r in csv.DictReader(open(ROOT / "data/analysis/exact_prompt_tokens.csv")):
    exact.setdefault(r["run"], {})[r["query_id"]] = int(r["final_tokens"])
qtext = {}
for f in glob.glob(str(ROOT / "data/x3/fulldoc_runs/fd_dense_high150/*.json")):
    r = json.load(open(f))
    if "query_id" in r: qtext[str(r["query_id"])] = r["search_log"][0]["query"][:120]

need = {}   # run -> {qid: set(docids)}
# document REPL runs: every document the server returned (= DOCS)
FD = {"fd_dense_high150": "fd_dense_high150", "fd_k5": "fd_dense_high150_k5", "fd_cap20_150": "fd_dense_high50_cap20", "fd_k5print": "fd_dense_high150_k5print", "fd_nogrep": "fd_dense_high150_nogrep"}
for run, d in FD.items():
    need[run] = {}
    for f in glob.glob(str(ROOT / "data/x3/fulldoc_runs" / d / "*.json")):
        r = json.load(open(f))
        if "query_id" not in r: continue
        need[run][str(r["query_id"])] = set(str(x) for x in (r.get("server_docids") or r.get("retrieved_docids") or []))
# snippet REPL runs: documents fetched with get_document, parsed from the trajectory code
SN = {"snip_dense_high150": "data/x3/snippet_runs/logs/traj", "snip_cap10": "data/x3/snippet_runs_cap10/logs/traj", "snip_k25": "data/x3/snippet_runs_k25/logs/traj"}
pat = re.compile(r"get_document\(\s*(?:doc_id\s*=\s*)?['\"]?(\d+)['\"]?\s*\)")
for run, d in SN.items():
    need[run] = {}
    for f in glob.glob(str(ROOT / d / "*.jsonl")):
        qid, ids = None, set()
        for line in open(f):
            try: it = json.loads(line)
            except Exception: continue
            if it.get("type") != "iteration": continue
            if qid is None:
                text = " ".join(str(m.get("content", "")) for m in it["prompt"] if isinstance(m, dict))
                for q, t in qtext.items():
                    if t in text: qid = q; break
            code = "\n".join(re.findall(r"```repl\n(.*?)```", it.get("response") or "", re.S))
            ids |= set(pat.findall(code))
            # also take literal id lists in blocks that loop over get_document
            if "get_document" in code:
                for m in re.finditer(r"\[(\s*['\"]?\d+['\"]?\s*(?:,\s*['\"]?\d+['\"]?\s*)*)\]", code):
                    ids |= set(re.findall(r"\d+", m.group(1)))
        if qid and qid not in need[run]: need[run][qid] = ids

wanted = set().union(*(s for d in need.values() for s in d.values()))
print("documents to tokenise:", len(wanted), file=sys.stderr)
doclen = {}
with open(CORPUS, encoding="utf-8") as fh:
    for line in fh:
        d = json.loads(line); did = str(d.get("docid") or d.get("id") or d.get("doc_id"))
        if did in wanted:
            doclen[did] = len(enc.encode(d.get("text") or d.get("contents") or ""))
print("tokenised:", len(doclen), "missing:", len(wanted - set(doclen)), file=sys.stderr)

rows = []
for run, d in need.items():
    g = graded(run)
    for qid, ids in d.items():
        if qid not in g or qid not in exact.get(run, {}): continue
        rows.append(dict(run=run, query_id=qid, correct=int(g[qid]), prompt_tokens=exact[run][qid],
                         state_tokens=sum(doclen.get(i, 0) for i in ids), n_state_docs=len(ids)))
# reference agents: everything received is in the prompt
for run in ("x7_dense_gpt5", "x7_dense_gpt5_getdoc", "x7_dense_gpt5_trim30k"):
    g = graded(run)
    for f in glob.glob(str(ROOT / "data/x3" / run / "runs/*.json")):
        r = json.load(open(f)); qid = str(r["query_id"])
        if qid in g and r.get("final_context_tokens"):
            rows.append(dict(run=run, query_id=qid, correct=int(g[qid]), prompt_tokens=r["final_context_tokens"], state_tokens=0, n_state_docs=0))
with open(ROOT / "data/analysis/received_text.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["run", "query_id", "correct", "prompt_tokens", "state_tokens", "n_state_docs"]); w.writeheader(); w.writerows(rows)
for run in sorted({r["run"] for r in rows}):
    sub = [r for r in rows if r["run"] == run]
    print(f"{run:22s} n {len(sub):3d} | prompt median {st.median(r['prompt_tokens'] for r in sub):8.0f} | state median {st.median(r['state_tokens'] for r in sub):9.0f} p90 {sorted(r['state_tokens'] for r in sub)[int(0.9*(len(sub)-1))]:9d} | docs median {st.median(r['n_state_docs'] for r in sub):5.1f}")
