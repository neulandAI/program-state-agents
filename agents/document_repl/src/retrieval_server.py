"""Dense retrieval server for BrowseComp-Plus (Qwen3-Embedding-8B, exact cosine).

    python src/retrieval_server.py --port 8765 [--device cuda] [--index-device cpu|cuda]

Reads <FULLDOC_ROOT>/data/indexes/qwen3-embedding-8b, data/prepared/corpus.jsonl and data/models/{Qwen3-0.6B,
Qwen3-Embedding-8B}. Reproduces the BrowseComp-Plus FAISS searcher (inner product over normalised vectors) and its
snippet truncation (Qwen3-0.6B tokenizer). One process holds the encoder and index; agents connect over HTTP.

Endpoints: GET /health, GET /stats, POST /search {"query", "k", "mode": "snippet"|"full"|"ids", "tag"},
POST /tag_counts, POST /tag_docids.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "shared"))
from rlm_shared.paths import path  # noqa: E402
from encoder import Qwen3Encoder  # noqa: E402

ROOT = str(path("FULLDOC_ROOT"))
INDEX_GLOB = f"{ROOT}/data/indexes/qwen3-embedding-8b/corpus.shard*_of_4.pkl"
CORPUS = f"{ROOT}/data/prepared/corpus.jsonl"
SNIPPET_TOKENIZER = f"{ROOT}/data/models/Qwen3-0.6B"

DEFAULT_SNIPPET_TOKENS = 512  # official --snippet-max-tokens
MAX_K = 1000


class Retriever:
    def __init__(self, device: str | None = None, index_device: str = "cpu"):
        t0 = time.time()
        print("[server] loading index shards...", flush=True)
        shards = sorted(glob.glob(INDEX_GLOB))
        if not shards:
            raise SystemExit(f"no index shards at {INDEX_GLOB}")
        reps_list, lookup = [], []
        for p in shards:
            with open(p, "rb") as f:
                reps, lk = pickle.load(f)
            reps_list.append(np.asarray(reps, dtype=np.float32))
            lookup.extend(lk)
        mat = np.concatenate(reps_list, axis=0)
        del reps_list
        self.lookup = lookup
        self.index = torch.from_numpy(mat)
        if index_device == "cuda" and torch.cuda.is_available():
            self.index = self.index.to("cuda", torch.float16)
        self.index_device = str(self.index.device)
        print(
            f"[server] index {tuple(self.index.shape)} on {self.index_device} "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )

        print("[server] loading corpus text...", flush=True)
        t1 = time.time()
        self.docid_to_text: dict[str, str] = {}
        with open(CORPUS, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                self.docid_to_text[d["docid"]] = d["text"]
        print(
            f"[server] corpus {len(self.docid_to_text)} docs ({time.time() - t1:.1f}s)",
            flush=True,
        )
        if len(self.docid_to_text) != len(self.lookup):
            print(
                f"[server] WARNING corpus={len(self.docid_to_text)} != index={len(self.lookup)}",
                flush=True,
            )

        print("[server] loading encoder...", flush=True)
        t2 = time.time()
        try:
            self.encoder = Qwen3Encoder(device=device, dtype=torch.float16)
        except Exception as e:
            print(f"[server] GPU encoder failed ({str(e)[:120]}); falling back to CPU", flush=True)
            self.encoder = Qwen3Encoder(device="cpu", dtype=torch.float32)
        print(f"[server] encoder on {self.encoder.device} ({time.time() - t2:.1f}s)", flush=True)

        self.snippet_tok = AutoTokenizer.from_pretrained(SNIPPET_TOKENIZER)

        self._enc_lock = threading.Lock()
        self._qcache: dict[str, torch.Tensor] = {}
        self.n_search = 0
        self.n_cache_hit = 0
        # Per-run counters kept outside the agent process, so model code cannot alter them.
        self._tag_lock = threading.Lock()
        self.tag_counts: dict[str, dict] = {}
        self.tag_docids: dict[str, set] = {}
        self._doc_spans: dict[str, tuple[int, int]] = {}
        self.t_boot = time.time()

    def embed_query(self, query: str) -> torch.Tensor:
        with self._enc_lock:
            hit = self._qcache.get(query)
            if hit is not None:
                self.n_cache_hit += 1
                return hit
            vec = self.encoder.encode_queries([query])[0]
            self._qcache[query] = vec
            return vec

    def _truncate(self, text: str, max_tokens: int | None) -> tuple[str, int]:
        """Official snippet path: Qwen3-0.6B tokenizer, first `max_tokens` tokens."""
        if not max_tokens or max_tokens <= 0:
            return text, len(text)
        toks = self.snippet_tok.encode(text, add_special_tokens=False)
        if len(toks) <= max_tokens:
            return text, len(text)
        snippet = self.snippet_tok.decode(toks[:max_tokens], skip_special_tokens=True)
        return snippet, len(snippet)

    def search(
        self,
        query: str,
        k: int = 5,
        mode: str = "snippet",
        snippet_max_tokens: int | None = DEFAULT_SNIPPET_TOKENS,
        tag: str | None = None,
    ) -> list[dict]:
        k = max(1, min(int(k), MAX_K))
        q = self.embed_query(query)
        qd = q.to(self.index.device, self.index.dtype)
        scores = torch.mv(self.index, qd).float()
        top = torch.topk(scores, k)
        self.n_search += 1
        if tag:
            with self._tag_lock:
                c = self.tag_counts.setdefault(tag, {"search": 0, "k_sum": 0, "hits": 0})
                c["search"] += 1
                c["k_sum"] += k
                c["hits"] += min(k, len(self.lookup))

        if tag:
            with self._tag_lock:
                self.tag_docids.setdefault(tag, set()).update(self.lookup[i] for i in top.indices.tolist())

        if mode == "ids":
            return [
                {"docid": self.lookup[i], "score": round(float(s), 6)}
                for s, i in zip(top.values.tolist(), top.indices.tolist())
            ]

        out = []
        for score, idx in zip(top.values.tolist(), top.indices.tolist()):
            docid = self.lookup[idx]
            text = self.docid_to_text.get(docid, "")
            if mode == "full":
                cached = self._doc_spans.get(docid)
                if cached is None:
                    _, snip_chars = self._truncate(text, snippet_max_tokens)
                    emb_end = self.encoder.embedded_char_span(text)[1]
                    cached = (snip_chars, emb_end)
                    self._doc_spans[docid] = cached
                snip_chars, emb_end = cached
                emb_span = (0, emb_end)
                out.append(
                    {
                        "docid": docid,
                        "score": round(float(score), 6),
                        "text": text,
                        "n_chars": len(text),
                        "embedded_span": [emb_span[0], emb_span[1]],
                        "snippet_span": [0, snip_chars],
                    }
                )
            else:
                snippet, _ = self._truncate(text, snippet_max_tokens)
                out.append(
                    {"docid": docid, "score": round(float(score), 6), "snippet": snippet}
                )
        return out

    def stats(self) -> dict:
        return {
            "n_search": self.n_search,
            "n_query_cache_hits": self.n_cache_hit,
            "query_cache_size": len(self._qcache),
            "encoder_device": str(self.encoder.device),
            "index_device": self.index_device,
            "index_shape": list(self.index.shape),
            "n_docs": len(self.docid_to_text),
            "uptime_s": round(time.time() - self.t_boot, 1),
        }


RETRIEVER: Retriever | None = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # silence per-request logging
        pass

    def _send(self, code: int, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True})
        elif self.path == "/stats":
            self._send(200, RETRIEVER.stats())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self._send(400, {"error": f"bad request: {e}"})
            return
        try:
            if self.path == "/search" and not req.get("tag") and not req.get("untagged_ok"):
                # Agent calls carry a per-run tag; offline scripts must set untagged_ok.
                self._send(400, {"error": "tag required"})
                return
            if self.path == "/search":
                res = RETRIEVER.search(
                    query=req["query"],
                    k=req.get("k", 5),
                    mode=req.get("mode", "snippet"),
                    snippet_max_tokens=req.get("snippet_max_tokens", DEFAULT_SNIPPET_TOKENS),
                    tag=req.get("tag"),
                )
                self._send(200, {"results": res})
            elif self.path == "/tag_counts":
                self._send(200, {"counts": RETRIEVER.tag_counts.get(req.get("tag"), None)})
            elif self.path == "/tag_docids":
                ids = RETRIEVER.tag_docids.get(req.get("tag"))
                self._send(200, {"docids": sorted(ids) if ids is not None else None})
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default=None, help="encoder device (default: auto)")
    ap.add_argument("--index-device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    global RETRIEVER
    RETRIEVER = Retriever(device=args.device, index_device=args.index_device)

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    srv.daemon_threads = True
    print(f"[server] READY on http://127.0.0.1:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
