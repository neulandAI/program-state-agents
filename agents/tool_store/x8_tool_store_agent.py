#!/usr/bin/env python3
"""Tool-store agent: whole documents held outside the prompt as in the document REPL agent, but read through search,
read and grep tools instead of code (same retriever server, cap of 10 searches, initial search, k up to 50).

Run: python agents/tool_store/x8_tool_store_agent.py --run x8_toolstore [--fixed-k 25] -> data/x3/<run>/runs/<qid>.json;
run records are not part of the release.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation"))

CAP, MAX_K, DEFAULT_K, MAX_CALLS, MAX_OUT, TRUNC = 10, 50, 5, 100, 120_000, 20_000
SERVER = os.environ.get("FD_SERVER_URL", "http://127.0.0.1:8765")
FIXED_K = 0   # set by --fixed-k

TEMPLATE = """
You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search, read and grep tools provided. Please perform reasoning and use the tools step by step, in an interleaved manner. You may use the tools multiple times.

Question: {Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

TOOLS = [
    {"type": "function", "name": "search", "strict": True,
     "description": (f"Search a knowledge source of web documents with a dense retriever and store the top-k hits. The "
                     f"COMPLETE text of every hit is kept in your document store, outside this conversation; the call "
                     f"returns one line per hit: docid, score and the document's first 100 characters. k is yours to "
                     f"choose, from 1 to {MAX_K} (default {DEFAULT_K}). You have a hard budget of {CAP} search calls for "
                     f"this question, including the initial search already made with the question; calls beyond that "
                     f"are refused. Use read and grep to look inside stored documents; they do not count as searches."),
     "parameters": {"type": "object", "additionalProperties": False, "required": ["query", "k"],
                    "properties": {"query": {"type": "string", "description": "Search query"},
                                   "k": {"type": "integer", "description": f"Number of documents, 1 to {MAX_K}"}}}},
    {"type": "function", "name": "read", "strict": True,
     "description": ("Read part of a stored document: returns characters [start, start + length) of its text and the "
                     "document's total length. length is at most 20000."),
     "parameters": {"type": "object", "additionalProperties": False, "required": ["docid", "start", "length"],
                    "properties": {"docid": {"type": "string"}, "start": {"type": "integer"}, "length": {"type": "integer"}}}},
    {"type": "function", "name": "grep", "strict": True,
     "description": ("Search inside every stored document with a case-insensitive regular expression; returns, for each "
                     "match, the docid, the character offset and the surrounding text (window characters on each side), "
                     "at most max_hits matches. Use it for exact names, dates, numbers and phrases."),
     "parameters": {"type": "object", "additionalProperties": False, "required": ["pattern", "window", "max_hits"],
                    "properties": {"pattern": {"type": "string"}, "window": {"type": "integer"}, "max_hits": {"type": "integer"}}}},
]


class Store:
    """Per-question document store and the three tools."""

    def __init__(self, tag: str):
        self.tag, self.docs, self.order, self.n_search, self.n_read, self.n_grep = tag, {}, [], 0, 0, 0

    def search(self, query: str, k: int, initial: bool = False) -> str:
        if FIXED_K and not initial:
            k = FIXED_K
        if self.n_search >= CAP:
            return f"[search refused] all {CAP} searches are used; answer from the {len(self.docs)} stored documents."
        self.n_search += 1
        k = max(1, min(int(k or DEFAULT_K), MAX_K))
        r = requests.post(f"{SERVER}/search", json={"query": query, "k": k, "mode": "full", "tag": self.tag}, timeout=300)
        r.raise_for_status()
        hits = r.json().get("results", [])
        new = 0
        lines = []
        for h in hits:
            did, text = str(h["docid"]), h.get("text") or ""
            if did not in self.docs:
                self.docs[did] = text; self.order.append(did); new += 1
            head = re.sub(r"\s+", " ", (self.docs[did] or "")[:100])
            lines.append(f"{did}\t{h.get('score', 0):.3f}\t{head}")
        return (f"[search {self.n_search}/{CAP}] k={k} -> {len(hits)} hits, {new} new documents (store holds {len(self.docs)})\n"
                + "\n".join(lines))

    def read(self, docid: str, start: int, length: int) -> str:
        self.n_read += 1
        t = self.docs.get(str(docid))
        if t is None:
            return f"[read] no stored document {docid}"
        start = max(0, int(start or 0)); length = max(1, min(int(length or 4000), TRUNC))
        return f"[read {docid}] characters {start}-{min(len(t), start + length)} of {len(t)}\n{t[start:start + length]}"

    def grep(self, pattern: str, window: int, max_hits: int) -> str:
        self.n_grep += 1
        try:
            rx = re.compile(pattern, re.I)
        except re.error as e:
            return f"[grep] bad pattern: {e}"
        window = max(20, min(int(window or 300), 2000)); max_hits = max(1, min(int(max_hits or 30), 100))
        out = []
        for did in self.order:
            t = self.docs[did]
            for m in rx.finditer(t):
                a, b = max(0, m.start() - window), min(len(t), m.end() + window)
                out.append(f"{did} @{m.start()}: {re.sub(r'\s+', ' ', t[a:b])}")
                if len(out) >= max_hits:
                    break
            if len(out) >= max_hits:
                break
        return f"[grep] {len(out)} matches for {pattern!r}\n" + "\n".join(out)

    def call(self, name: str, args: dict) -> str:
        fn = {"search": lambda: self.search(args.get("query", ""), args.get("k", DEFAULT_K)),
              "read": lambda: self.read(args.get("docid", ""), args.get("start", 0), args.get("length", 4000)),
              "grep": lambda: self.grep(args.get("pattern", ""), args.get("window", 300), args.get("max_hits", 30))}.get(name)
        if fn is None:
            return f"[error] unknown tool {name}"
        out = fn()
        return out if len(out) <= TRUNC else out[:TRUNC] + f"\n[... truncated at {TRUNC} characters]"


def create(client, body):
    import openai
    for attempt in range(12):
        try:
            return client.responses.create(**body)
        except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError):
            if attempt == 11:
                raise
            time.sleep(min(120, 5 * 2 ** attempt) + random.uniform(0, 3))


def run_one(client, model, qid, q, out_dir):
    path = out_dir / f"{qid}.json"
    if path.exists():
        return
    store = Store(tag=f"x8_{qid}_{uuid.uuid4().hex[:8]}")
    t0 = time.time()
    rec = dict(query_id=qid, status="completed", harness={"name": "x8_tool_store_agent", "cap": CAP, "max_k": MAX_K, "fixed_k": FIXED_K})
    ctx, cum_in, cum_out = [], 0, 0
    try:
        seed = store.search(q["question"], DEFAULT_K, initial=True)
        user = TEMPLATE.format(Question=q["question"]) + "\n\nAn initial search with the question has been made (search 1 of 10):\n" + seed
        items = [{"role": "user", "content": user}]
        final = ""
        for _ in range(MAX_CALLS):
            if cum_out >= MAX_OUT:
                rec["status"] = "incomplete"; break
            body = {"model": model, "input": items, "tools": TOOLS, "tool_choice": "auto", "truncation": "auto",
                    "max_output_tokens": min(40_000, MAX_OUT - cum_out), "reasoning": {"effort": "high", "summary": "detailed"}}
            r = create(client, body)
            u = getattr(r, "usage", None)
            cin, cout = getattr(u, "input_tokens", 0) or 0, getattr(u, "output_tokens", 0) or 0
            cum_in += cin; cum_out += cout; ctx.append(cin)
            fcs = [it for it in r.output if getattr(it, "type", None) == "function_call"]
            for it in r.output:
                d = it.model_dump(mode="json", by_alias=True, exclude_none=True); d.pop("status", None); items.append(d)
            if not fcs:
                final = getattr(r, "output_text", "") or ""
                break
            for fc in fcs:
                try:
                    args = json.loads(fc.arguments or "{}")
                except Exception:  # noqa: BLE001
                    args = {}
                try:
                    out = store.call(fc.name, args)
                except Exception as e:  # noqa: BLE001
                    out = f"[error] {type(e).__name__}: {e}"
                items.append({"type": "function_call_output", "call_id": fc.call_id, "output": out})
        else:
            rec["status"] = "incomplete"
        rec["final_answer"] = final
        if not final.strip() and rec["status"] == "completed":
            rec["status"] = "empty_output"
    except Exception as e:  # noqa: BLE001
        rec["status"] = "error"; rec["error"] = f"{type(e).__name__}: {str(e)[:400]}"; rec["final_answer"] = ""
    seen = set(store.docs)
    g, ev = q["gold"], q["evidence"]
    rec.update(n_searches=store.n_search, n_read=store.n_read, n_grep=store.n_grep, rounds=len(ctx),
               retrieved_docids=sorted(seen, key=lambda x: (len(x), x)),
               gold_recall=len(g & seen) / len(g) if g else None, evidence_recall=len(ev & seen) / len(ev) if ev else None,
               context_tokens_per_round=ctx, final_context_tokens=(ctx[-1] if ctx else None),
               cum_input_tokens=cum_in, cum_output_tokens=cum_out, wall_time_s=round(time.time() - t0, 1),
               timed_out=False, forced=False)
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="x8_toolstore")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fixed-k", type=int, default=0, help="width-matched variant: every search after the initial one returns exactly this many documents")
    ap.add_argument("--reverse", action="store_true", help="work through the ids from the end (a second process that meets the first in the middle)")
    a = ap.parse_args()
    if a.fixed_k:
        global FIXED_K
        FIXED_K = a.fixed_k
        TOOLS[0]["description"] = (
            f"Search a knowledge source of web documents with a dense retriever and store the top {FIXED_K} hits. The "
            f"COMPLETE text of every hit is kept in your document store, outside this conversation; the call returns one "
            f"line per hit: docid, score and the document's first 100 characters. Every search returns {FIXED_K} "
            f"documents. You have a hard budget of {CAP} search calls for this question, including the initial search "
            f"already made with the question; calls beyond that are refused. Use read and grep to look inside stored "
            f"documents; they do not count as searches.")
        TOOLS[0]["parameters"] = {"type": "object", "additionalProperties": False, "required": ["query"],
                                  "properties": {"query": {"type": "string", "description": "Search query"}}}
    from x3_env import load_env
    from x3_plain_agent import load_questions
    load_env()
    import openai
    client = openai.OpenAI(api_key=os.environ["AZURE_OPENAI_API_KEY"],
                           base_url=os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/") + "/openai/v1/", max_retries=12, timeout=1200)
    model = os.environ["GPT5_DEPLOYMENT"]
    qs = load_questions()
    ids = [str(x) for x in json.load(open(ROOT / "data/x3/subset_150_ids.json"))]
    if a.limit:
        ids = ids[: a.limit]
    if a.reverse:
        ids = ids[::-1]
    out_dir = ROOT / "data/x3" / a.run / "runs"; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{a.run}: {len(ids)} questions, {a.workers} workers, model {model}, server {SERVER}", flush=True)
    done = threading.Lock(); n = [0]

    def job(qid):
        run_one(client, model, qid, qs[qid], out_dir)
        with done:
            n[0] += 1
            if n[0] % 10 == 0 or n[0] == len(ids):
                print(f"{time.strftime('%H:%M:%S')} {n[0]}/{len(ids)} done", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(job, ids))


if __name__ == "__main__":
    main()
