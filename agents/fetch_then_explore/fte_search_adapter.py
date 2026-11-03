#!/usr/bin/env python3
"""Fetch-then-Explore (FtE) on BrowseComp-Plus: its local-search API served over OUR exact dense retrieval.

FtE's local backend (third_party/fte_search_agent/src/local_search/api.py) expects
  POST /search        {"query", "top_k"} -> {"backend", "results": [{"query", "hits": [{"rank", "docid", "score", "snippet"}]}]}
  POST /get_document  {"docid"}          -> {"docid", "text"}        (404 when unknown)
This adapter forwards to runners/dense_server.py (exact cosine over the benchmark's Qwen3-Embedding-8B embeddings, the
retriever of every dense run in the paper) and cuts each hit's snippet to the document's first 512 tokens with the
Qwen3-0.6B tokenizer, as the BrowseComp-Plus search agent does. So FtE, the reference agent and our agents search the
same index with the same snippets. Searches are counted per question tag-free (FtE sends none); counts are in the log.
  python agents/fetch_then_explore/fte_search_adapter.py --port 8082 --dense http://127.0.0.1:8765
"""
import argparse, json, threading, urllib.error, urllib.request
import uvicorn
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer

ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8082); ap.add_argument("--dense", default="http://127.0.0.1:8765")
ap.add_argument("--snippet-max-tokens", type=int, default=512)
args = ap.parse_args()
tok = AutoTokenizer.from_pretrained("data/x3/fulldoc_root/data/models/Qwen3-0.6B")
lock = threading.Lock(); COUNTS = {"search": 0, "get_document": 0}


def post(path, body):
    req = urllib.request.Request(args.dense.rstrip("/") + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def full_text(docid):
    try:
        d = post("/document", {"doc_id": docid})
    except urllib.error.HTTPError:          # unknown id: FtE expects a 404, not a server error
        return ""
    return d.get("text") or ""


def snippet(text):
    ids = tok.encode(text[: args.snippet_max_tokens * 12], add_special_tokens=False)
    return tok.decode(ids[: args.snippet_max_tokens], skip_special_tokens=True)


app = FastAPI()


@app.get("/healthz")
def healthz():
    return {"status": "ok", "backend": "dense-exact via " + args.dense, "default_top_k": 5, "snippet_max_tokens": args.snippet_max_tokens, **COUNTS}


@app.post("/search")
def search(req: dict):
    q = str(req.get("query", "")).strip()
    if not q:
        raise HTTPException(400, "'query' is required.")
    k = max(1, min(int(req.get("top_k") or 5), 50))
    hits = []
    for rank, r in enumerate(post("/search", {"query": q, "top_k": k}).get("results", []), start=1):
        d = str(r.get("doc_id"))
        hits.append({"rank": rank, "docid": d, "score": r.get("score"), "snippet": snippet(full_text(d))})
    with lock:
        COUNTS["search"] += 1
    return {"backend": "dense-exact", "results": [{"query": q, "hits": hits}]}


@app.post("/get_document")
def get_document(req: dict):
    d = str(req.get("docid", "")).strip()
    if not d:
        raise HTTPException(400, "'docid' is required.")
    t = full_text(d)
    if not t:
        raise HTTPException(404, f"Document '{d}' not found.")
    with lock:
        COUNTS["get_document"] += 1
    return {"docid": d, "text": t}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
