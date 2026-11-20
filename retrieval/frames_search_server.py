"""BM25 search server over the FRAMES Wikipedia dump (or, with --dataset bcp, the BrowseComp-Plus corpus).

--protocol rlm (snippet REPL agent): POST /search {query, top_k}, POST /document {doc_id}.
--protocol sup (document REPL agent): POST /search {query, k, mode, snippet_max_tokens, tag}, /tag_counts, /tag_docids;
snippets are cut with the Qwen3-0.6B tokenizer. FRAMES doc ids are corpus line numbers; text is "title\\ntext".
  python retrieval/frames_search_server.py --port 8040 --protocol rlm [--dataset bcp]
"""
import argparse, json, os, sys, threading

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agents", "shared"))
from rlm_shared.paths import path  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8040)
ap.add_argument("--protocol", choices=["rlm", "sup"], default="rlm")
# --dataset bcp: the index of bcp_build_bm25_default.py; doc ids are the corpus's own ids.
ap.add_argument("--dataset", choices=["frames", "bcp"], default="frames")
args = ap.parse_args()

import bm25s, Stemmer
if args.dataset == "bcp":
    DATA = str(path("BCP_BM25_INDEX_DIR"))
    CORPUS = f"{path('FULLDOC_ROOT')}/data/prepared/corpus.jsonl"
    offsets = np.load(f"{DATA}/offsets.npy")
    DOCIDS = json.load(open(f"{DATA}/docids.json"))
else:
    DATA = str(path("FRAMES_DATA_DIR"))
    CORPUS = f"{DATA}/corpus.jsonl"
    offsets = np.load(f"{DATA}/corpus_meta.npz", allow_pickle=True)["offsets"]
    DOCIDS = None
retriever = bm25s.BM25.load(f"{DATA}/bm25_index", mmap=True)
stemmer = Stemmer.Stemmer("english")
N = len(offsets)
_local = threading.local()


def doc_text(i: int) -> str:
    f = getattr(_local, "f", None)
    if f is None:
        f = _local.f = open(CORPUS, "rb")
    f.seek(int(offsets[i]))
    d = json.loads(f.readline())
    return d["text"] if DOCIDS is not None else f"{d['title']}\n{d['text']}"


REV = {d: i for i, d in enumerate(DOCIDS)} if DOCIDS is not None else None


def ext_id(i: int) -> str:
    return DOCIDS[i] if DOCIDS is not None else str(i)


def bm25(query: str, k: int) -> list[tuple[int, float]]:
    qt = bm25s.tokenize([query], stopwords="en", stemmer=stemmer, show_progress=False)
    if not qt.vocab:                       # all stopwords: nothing to score
        return []
    docs, scores = retriever.retrieve(qt, k=k, show_progress=False, n_threads=1)
    return [(int(d), float(s)) for d, s in zip(docs[0], scores[0]) if s > 0]


app = FastAPI()

if args.protocol == "rlm":
    @app.get("/health")
    def health():
        return {"status": "ok", "docs": N}

    @app.post("/search")
    async def search(request: Request):
        req = await request.json()
        k = max(1, min(int(req.get("top_k", 5)), 100))
        return {"results": [{"score": round(s, 4), "doc_id": ext_id(d), "text": doc_text(d)[:2000]}
                            for d, s in bm25(req["query"], k)]}

    @app.post("/document")
    async def get_document(request: Request):
        req = await request.json()
        try:
            i = REV[str(req["doc_id"])] if REV is not None else int(req["doc_id"]); assert 0 <= i < N
        except Exception:
            return {"error": f"doc_id '{req.get('doc_id')}' not found", "doc_id": req.get("doc_id"), "text": ""}
        return {"doc_id": req["doc_id"], "text": doc_text(i)}

else:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(f"{path('FULLDOC_ROOT')}/data/models/Qwen3-0.6B")
    lock = threading.Lock()
    tag_counts: dict[str, dict] = {}
    tag_docids: dict[str, set] = {}

    def truncate(text: str, max_tokens) -> str:
        if not max_tokens:
            return text
        ids = tok.encode(text[: int(max_tokens) * 12], add_special_tokens=False)  # 12 ch/token is ample
        return tok.decode(ids[: int(max_tokens)], skip_special_tokens=True)

    @app.get("/health")
    def health():
        return {"ok": True, "status": "ok", "docs": N}

    @app.post("/search")
    async def search(request: Request):
        req = await request.json()
        if not req.get("tag") and not req.get("untagged_ok"):
            return JSONResponse({"error": "tag required"}, status_code=400)
        k = max(1, min(int(req.get("k", 5)), 100))
        hits = bm25(req["query"], k)
        tag = req.get("tag")
        if tag:
            with lock:
                c = tag_counts.setdefault(tag, {"search": 0, "k_sum": 0, "hits": 0})
                c["search"] += 1; c["k_sum"] += k; c["hits"] += len(hits)
                tag_docids.setdefault(tag, set()).update(ext_id(d) for d, _ in hits)
        mode, smax = req.get("mode", "snippet"), req.get("snippet_max_tokens", 512)
        out = []
        for d, s in hits:
            if mode == "ids":
                out.append({"docid": ext_id(d), "score": round(s, 6)}); continue
            text = doc_text(d)
            snip = truncate(text, smax)
            if mode == "full":
                out.append({"docid": ext_id(d), "score": round(s, 6), "text": text, "n_chars": len(text),
                            "embedded_span": [0, len(text)], "snippet_span": [0, len(snip)]})
            else:
                out.append({"docid": ext_id(d), "score": round(s, 6), "snippet": snip})
        return {"results": out}

    @app.post("/tag_counts")
    async def counts(request: Request):
        req = await request.json()
        return {"counts": tag_counts.get(req.get("tag"))}

    @app.post("/tag_docids")
    async def docids(request: Request):
        req = await request.json()
        ids = tag_docids.get(req.get("tag"))
        return {"docids": sorted(ids) if ids is not None else None}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=args.port)
