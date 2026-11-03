"""Late-interaction retrieval server (Reason-ModernColBERT, PLAID) with the same HTTP API as retrieval_server.py.

    $PYLATE_PYTHON src/retrieval_server_colbert.py --port 8766

Runs in the pylate environment. COLBERT_INDEX_NAME and COLBERT_DOC_LEN select the index (8,192-token index for the
paper's runs). Reads <FULLDOC_ROOT>/data/indexes/lighton/pylate and <FULLDOC_ROOT>/data/prepared/corpus.jsonl.
Setting COLBERT_BM25_DIR adds a BM25 arm fused by RRF (not used in the paper's runs).
"""
import argparse, os, json, threading, time
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "shared"))
from rlm_shared.paths import path  # noqa: E402

ROOT = str(path("FULLDOC_ROOT"))
IDX_FOLDER = f"{ROOT}/data/indexes/lighton/pylate"
IDX_NAME = os.environ.get("COLBERT_INDEX_NAME", "Reason-ModernColBERT")
MODEL = "lightonai/Reason-ModernColBERT"
MAX_K = 1000
DEFAULT_SNIPPET_TOKENS = 512
# Optional lexical arm; both arms are read to COLBERT_FUSE_DEPTH and the fused list is cut to k.
BM25_DIR = os.environ.get("COLBERT_BM25_DIR", "")
FUSE_DEPTH = int(os.environ.get("COLBERT_FUSE_DEPTH", "100"))
BM25_WEIGHT = float(os.environ.get("COLBERT_BM25_WEIGHT", "1.0"))
MV_WEIGHT = float(os.environ.get("COLBERT_MV_WEIGHT", "1.0"))
RRF_K = int(os.environ.get("COLBERT_RRF_K", "60"))

RETRIEVER = None


class ColbertRetriever:
    def __init__(self, device="cuda", query_length=512):
        from pylate import indexes, models, retrieve

        t0 = time.time()
        self.corpus = {}
        with open(f"{ROOT}/data/prepared/corpus.jsonl", encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                self.corpus[o["docid"]] = o["text"]
        print(f"corpus: {len(self.corpus)} documents in {time.time()-t0:.0f}s", flush=True)

        # Pass both lengths explicitly, as LightOn's searcher does; otherwise pylate truncates queries at 128 tokens.
        _doclen = int(os.environ.get("COLBERT_DOC_LEN", "512"))
        self.model = models.ColBERT(model_name_or_path=MODEL, device=device,
                                    query_length=query_length, document_length=_doclen)
        self.query_length = query_length
        self.doc_len = _doclen
        _nfs = int(os.environ.get("COLBERT_N_FULL_SCORES", "8192"))
        _probe = int(os.environ.get("COLBERT_N_IVF_PROBE", "8"))
        _lowmem = os.environ.get("COLBERT_LOW_MEMORY", "") == "1"
        # batch_size bounds GPU memory on the 8,192-token index.
        _bs = int(os.environ.get("COLBERT_SEARCH_BATCH", str(1 << 18)))
        self.index = indexes.PLAID(index_folder=IDX_FOLDER, index_name=IDX_NAME, override=False,
                                   device=device, n_full_scores=_nfs, n_ivf_probe=_probe,
                                   low_memory=_lowmem, batch_size=_bs)
        self.retriever = retrieve.ColBERT(index=self.index)
        print(f"index={IDX_NAME!r} loaded on {device}; document_length={self.doc_len}", flush=True)

        # Fast tokenizers are not thread-safe: one per server thread.
        self._tls = threading.local()
        self._tok_name = MODEL
        self._span = {}
        self._lock = threading.Lock()
        self._tag_lock = threading.Lock()
        self.tag_counts = {}
        self.tag_docids = {}
        self.n_search = 0

        self.bm25 = None
        self.bm25_docids = None
        self._stemmer = None
        if BM25_DIR and os.path.exists(os.path.join(BM25_DIR, "docids.json")):
            import bm25s
            import Stemmer
            t4 = time.time()
            self.bm25 = bm25s.BM25.load(BM25_DIR, mmap=True)
            self.bm25_docids = json.load(open(os.path.join(BM25_DIR, "docids.json")))
            self._stemmer = Stemmer.Stemmer("english")
            print(f"[server] BM25 arm: {len(self.bm25_docids):,} docs "
                  f"({time.time() - t4:.1f}s) fuse_depth={FUSE_DEPTH} rrf_k={RRF_K}", flush=True)
        elif BM25_DIR:
            raise SystemExit(f"COLBERT_BM25_DIR={BM25_DIR} has no docids.json")

    @property
    def tok(self):
        t = getattr(self._tls, "tok", None)
        if t is None:
            from transformers import AutoTokenizer
            t = AutoTokenizer.from_pretrained(self._tok_name)
            self._tls.tok = t
        return t

    def embedded_char_span(self, docid, text):
        """How much of the document the index actually saw, in characters."""
        c = self._span.get(docid)
        if c is None:
            enc = self.tok(text, add_special_tokens=True, truncation=True,
                           max_length=self.doc_len - 1, return_offsets_mapping=True)
            offs = [o for o in enc["offset_mapping"] if o != (0, 0)]
            c = offs[-1][1] if offs else min(len(text), 1200)
            self._span[docid] = c
        return c

    def _truncate(self, text, max_tokens):
        enc = self.tok(text, add_special_tokens=False, truncation=True,
                       max_length=max_tokens, return_offsets_mapping=True)
        offs = [o for o in enc["offset_mapping"] if o != (0, 0)]
        end = offs[-1][1] if offs else len(text)
        return text[:end], end

    def search(self, query, k=5, mode="snippet", snippet_max_tokens=DEFAULT_SNIPPET_TOKENS, tag=None):
        k = max(1, min(int(k), MAX_K))
        depth = max(k, FUSE_DEPTH) if self.bm25 is not None else k
        with self._lock:
            emb = self.model.encode([query], batch_size=1, is_query=True, show_progress_bar=False)
            hits = self.retriever.retrieve(queries_embeddings=emb, k=depth)[0]
        if self.bm25 is not None:
            import bm25s as _b
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from hybrid import rrf as _rrf
            mv_ids = [h["id"] for h in hits]
            q_tok = _b.tokenize([query], stopwords="en", stemmer=self._stemmer,
                                show_progress=False)
            res, _sc = self.bm25.retrieve(q_tok, k=min(depth, len(self.bm25_docids)),
                                          show_progress=False)
            bm_ids = [self.bm25_docids[int(i)] for i in res[0]]
            fused = _rrf([mv_ids, bm_ids], k=RRF_K, weights=[MV_WEIGHT, BM25_WEIGHT])[:k]
            # Scores are now RRF scores, not MaxSim scores.
            hits = [{"id": d, "score": sc} for d, sc in fused]
        self.n_search += 1
        docids = [h["id"] for h in hits]
        if tag:
            with self._tag_lock:
                c = self.tag_counts.setdefault(tag, {"search": 0, "k_sum": 0, "hits": 0})
                c["search"] += 1
                c["k_sum"] += k
                c["hits"] += len(hits)
                self.tag_docids.setdefault(tag, set()).update(docids)
        out = []
        for h in hits:
            docid = h["id"]
            text = self.corpus.get(docid, "")
            score = round(float(h["score"]), 6)
            if mode == "ids":
                out.append({"docid": docid, "score": score})
                continue
            if mode == "full":
                emb_end = self.embedded_char_span(docid, text)
                _, snip_chars = self._truncate(text, snippet_max_tokens)
                out.append({"docid": docid, "score": score, "text": text, "n_chars": len(text),
                            "embedded_span": [0, emb_end], "snippet_span": [0, snip_chars]})
            else:
                snippet, _ = self._truncate(text, snippet_max_tokens)
                out.append({"docid": docid, "score": score, "snippet": snippet})
        return out

    def stats(self):
        return {"retriever": ("RRF(Reason-ModernColBERT, BM25)" if self.bm25 is not None
                              else "Reason-ModernColBERT (multi-vector, PLAID)"),
                "fuse_depth": FUSE_DEPTH if self.bm25 is not None else None,
                "n_search": self.n_search,
                "n_docs": len(self.corpus),
                "document_length_tokens": self.doc_len,
                "query_length_tokens": self.query_length,
                "uptime_s": round(time.time() - T0, 1)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/stats":
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
                self._send(400, {"error": "tag required"})
                return
            if self.path == "/search":
                res = RETRIEVER.search(query=req["query"], k=req.get("k", 5), mode=req.get("mode", "snippet"),
                                       snippet_max_tokens=req.get("snippet_max_tokens", DEFAULT_SNIPPET_TOKENS),
                                       tag=req.get("tag"))
                self._send(200, {"results": res})
            elif self.path == "/tag_counts":
                self._send(200, {"counts": RETRIEVER.tag_counts.get(req.get("tag"))})
            elif self.path == "/tag_docids":
                ids = RETRIEVER.tag_docids.get(req.get("tag"))
                self._send(200, {"docids": sorted(ids) if ids is not None else None})
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


T0 = time.time()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--query-length", type=int, default=512,
                    help="Query truncation cap in tokens. 512 matches LightOn's documented runs; "
                         "the model config's 128 is what pylate uses if this is not passed.")
    a = ap.parse_args()
    RETRIEVER = ColbertRetriever(device=a.device, query_length=a.query_length)
    print(f"READY after ~{time.time()-T0:.0f}s on port {a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
