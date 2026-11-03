#!/usr/bin/env python3
"""Exact-cosine Qwen3-Embedding-8B retrieval server over the BrowseComp-Plus corpus embeddings (rlm wire format).

GET /health; POST /search {query, top_k} -> {results: [{doc_id, score, text[:2000]}]}; POST /document {doc_id}.
Query encoding follows the benchmark's searcher: instruction prefix, last-token pooling, L2 normalisation, fp16.
  python runners/dense_server.py --port 8001
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

DATA = Path(os.environ.get("RLM_DATA", "/path/to/rlm/data"))
INDEX_GLOB = str(DATA / "indexes/qwen3-embedding-8b/corpus.shard*_of_4.pkl")
CORPUS = DATA / "prepared/corpus.jsonl"
MODEL = DATA / "models/Qwen3-Embedding-8B"
QUERY_PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
QUERY_MAX_LEN = 8192
SNIPPET_CHARS = 2000
MAX_K = 1000


def pool_last(last_hidden_state, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_state[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    idx = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
    return last_hidden_state[idx, lengths]


class Retriever:
    def __init__(self, device: str):
        t0 = time.time()
        shards = sorted(glob.glob(INDEX_GLOB))
        if not shards:
            raise SystemExit(f"no index shards at {INDEX_GLOB}")
        reps, lookup = [], []
        for p in shards:
            with open(p, "rb") as f:
                r, lk = pickle.load(f)
            reps.append(np.asarray(r, dtype=np.float32)); lookup.extend(str(x) for x in lk)
        mat = np.concatenate(reps, 0); del reps
        self.lookup = lookup
        self.index = torch.from_numpy(mat).to(device, torch.float16)
        print(f"[server] index {tuple(self.index.shape)} on {device} ({time.time() - t0:.1f}s)", flush=True)
        t1 = time.time()
        self.text = {}
        with open(CORPUS, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                self.text[str(d["docid"])] = d["text"]
        print(f"[server] corpus {len(self.text)} docs ({time.time() - t1:.1f}s)", flush=True)
        t2 = time.time()
        self.tok = AutoTokenizer.from_pretrained(str(MODEL), padding_side="left")
        try:
            self.model = AutoModel.from_pretrained(str(MODEL), dtype=torch.float16, attn_implementation="sdpa")
        except TypeError:
            self.model = AutoModel.from_pretrained(str(MODEL), torch_dtype=torch.float16, attn_implementation="sdpa")
        self.model = self.model.to(device).eval()
        self.device = device
        print(f"[server] encoder on {device} ({time.time() - t2:.1f}s)", flush=True)
        self.lock = threading.Lock()
        self.cache: dict[str, torch.Tensor] = {}
        self.n_search = 0

    def embed(self, query: str) -> torch.Tensor:
        with self.lock:
            q = self.cache.get(query)
            if q is None:
                batch = self.tok([QUERY_PREFIX + query], padding=True, truncation=True, max_length=QUERY_MAX_LEN, return_tensors="pt")
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.no_grad():
                    out = self.model(**batch)
                    q = F.normalize(pool_last(out.last_hidden_state, batch["attention_mask"]), p=2, dim=-1)[0].to(torch.float16)
                if len(self.cache) > 20000:
                    self.cache.clear()
                self.cache[query] = q
            self.n_search += 1
            scores = torch.mv(self.index, q).float()
        return scores

    def search(self, query: str, k: int):
        k = max(1, min(int(k or 5), MAX_K))
        scores = self.embed(query)
        top = torch.topk(scores, k)
        return [{"score": round(float(s), 4), "doc_id": self.lookup[i], "text": self.text.get(self.lookup[i], "")[:SNIPPET_CHARS]}
                for s, i in zip(top.values.tolist(), top.indices.tolist())]


RET: Retriever | None = None
T0 = time.time()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send(200, {"status": "ok", "model": "Qwen3-Embedding-8B", "index": "tevatron exact cosine", "docs": len(RET.text), "searches": RET.n_search, "uptime_s": round(time.time() - T0)})
        else:
            self._send(404, {"error": "unknown path"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(n).decode() or "{}")
        except json.JSONDecodeError:
            return self._send(400, {"error": "bad json"})
        try:
            if self.path.startswith("/search"):
                return self._send(200, {"results": RET.search(str(payload.get("query", "")), payload.get("top_k") or payload.get("k") or 5)})
            if self.path.startswith("/document"):
                did = str(payload.get("doc_id", ""))
                if did not in RET.text:
                    return self._send(404, {"error": f"unknown doc_id {did}"})
                return self._send(200, {"doc_id": did, "text": RET.text[did]})
            return self._send(404, {"error": "unknown path"})
        except Exception as e:  # noqa: BLE001
            return self._send(500, {"error": f"{type(e).__name__}: {str(e)[:200]}"})


def main():
    global RET
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    RET = Retriever(a.device)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"[server] ready on http://{a.host}:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
