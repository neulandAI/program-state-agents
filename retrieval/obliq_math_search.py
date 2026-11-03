"""OBLIQ-Bench Math search front-end: drops each query's excluded documents (its own source problem) from upstream results.

Serves the snippet REPL wire ({query, top_k, tag}) and the document REPL wire ({query, k, mode, tag}); the query id is the
tag's prefix ("<query_id>-<uuid>"), and an untagged search is refused.
  python retrieval/obliq_math_search.py --port 8014 --upstream http://localhost:8020 --index obliq-ana-math
"""
import argparse, json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agents", "shared"))
from rlm_shared.obliq_data import OBLIQ_ROOT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8013)
ap.add_argument("--upstream", default="http://localhost:8012")
ap.add_argument("--index", default="obliq-ana-math")
ap.add_argument("--task", default="analogues")
ap.add_argument("--subset", default="math")
args = ap.parse_args()

UP = args.upstream.rstrip("/")
EXCL = {q: set(map(str, v)) for q, v in json.load(open(
    OBLIQ_ROOT / args.task / args.subset / "queries+qrels" / "per_query_excluded_ids.json")).items()}
T0 = time.time()
LOCK = threading.Lock()
TAG_COUNTS, TAG_DOCIDS = {}, {}
STATS = {"searches": 0, "excluded_dropped": 0, "refused_untagged": 0}


def search(query, k, tag):
    excl = EXCL.get(tag.split("-")[0], set())
    r = requests.post(f"{UP}/search", json={"query": query, "top_k": int(k) + len(excl), "index": args.index},
                      timeout=300)
    r.raise_for_status()
    raw = r.json().get("results", [])
    hits = [h for h in raw if str(h["doc_id"]) not in excl][:int(k)]
    with LOCK:
        STATS["searches"] += 1
        STATS["excluded_dropped"] += sum(1 for h in raw[:int(k)] if str(h["doc_id"]) in excl)
        c = TAG_COUNTS.setdefault(tag, {"search": 0, "k_sum": 0, "hits": 0})
        c["search"] += 1; c["k_sum"] += int(k); c["hits"] += len(hits)
        TAG_DOCIDS.setdefault(tag, set()).update(str(h["doc_id"]) for h in hits)
    return hits


def sup_shape(h, mode):
    docid, text, score = str(h["doc_id"]), h.get("text") or "", round(float(h.get("score", 0.0)), 6)
    if mode == "ids":
        return {"docid": docid, "score": score}
    if mode == "full":
        return {"docid": docid, "score": score, "text": text, "n_chars": len(text),
                "embedded_span": [0, len(text)], "snippet_span": [0, len(text)]}
    return {"docid": docid, "score": score, "snippet": text}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"status": "ok", "indexes": [args.index], "upstream": UP})
        if self.path == "/stats":
            return self._send(200, {"retriever": f"obliq_math_search -> {UP} [{args.index}]",
                                    "uptime_s": round(time.time() - T0, 1), **STATS})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        try:
            if self.path == "/search":
                tag = req.get("tag") or ""
                if not tag:
                    STATS["refused_untagged"] += 1
                    return self._send(400, {"error": "tag required (<query_id>-<uuid>): exclusions need the query id"})
                if "top_k" in req:   # snippet REPL wire
                    hits = search(req["query"], req.get("top_k", 5), tag)
                    return self._send(200, {"results": [{"doc_id": str(h["doc_id"]), "text": h.get("text") or "",
                                                         "score": h.get("score", 0.0)} for h in hits]})
                hits = search(req["query"], req.get("k", 5), tag)   # document REPL wire
                return self._send(200, {"results": [sup_shape(h, req.get("mode", "snippet")) for h in hits]})
            if self.path == "/document":
                r = requests.post(f"{UP}/document", json={"doc_id": str(req.get("doc_id") or req.get("docid")),
                                                         "index": args.index}, timeout=120)
                return self._send(r.status_code, r.json())
            if self.path == "/tag_counts":
                return self._send(200, {"counts": TAG_COUNTS.get(req.get("tag"))})
            if self.path == "/tag_docids":
                ids = TAG_DOCIDS.get(req.get("tag"))
                return self._send(200, {"docids": sorted(ids) if ids is not None else None})
            self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), H)
    srv.daemon_threads = True
    print(f"[math-search] READY on {args.port} -> {UP} index={args.index}, {len(EXCL)} queries with exclusions",
          flush=True)
    srv.serve_forever()
