"""Qwen3-Embedding-8B FAISS search server for several indexes: one model on the GPU, indexes on CPU, loaded lazily.

Same wire format as search_server.py; an optional "index" field selects the index (default: --default-index).
  python retrieval/multi_search_server.py --model-path <Qwen3-Embedding-8B> --index-root ~/indexes --port 8010
"""

import argparse
import glob
import os
import pickle
import threading

import faiss
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

QUERY_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages "
    "that answer the query\nQuery: "
)

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", required=True)
parser.add_argument("--index", action="append", default=[],
                    metavar="NAME=DIR", help="repeatable, e.g. --index congress=/path/to/dir")
parser.add_argument("--index-root", default="",
                    help="discover every subdirectory containing a FAISS index")
parser.add_argument("--default-index", default="",
                    help="index used when a request omits the `index` field")
parser.add_argument("--max-length", type=int, default=8192)
parser.add_argument("--nprobe", type=int, default=64)
parser.add_argument("--port", type=int, default=8010)
parser.add_argument("--device", default="cuda",
                    help="cuda (default) or cpu — cpu is for correctness testing "
                         "when the GPU is full; embedding is far slower there")
parser.add_argument("--gpu-index", action="store_true",
                    help="put FAISS on the GPU too (defeats the point; CPU is the default here)")
args = parser.parse_args()

INDEX_DIRS: dict[str, str] = {}
for spec in args.index:
    if "=" not in spec:
        raise SystemExit(f"--index expects NAME=DIR, got {spec!r}")
    name, path = spec.split("=", 1)
    INDEX_DIRS[name] = os.path.expanduser(path)

if args.index_root:
    root = os.path.expanduser(args.index_root)
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d):
            continue
        if os.path.exists(os.path.join(d, "_cached_ids.pkl")):
            INDEX_DIRS.setdefault(os.path.basename(d), d)

if not INDEX_DIRS:
    raise SystemExit("no indexes given: use --index NAME=DIR or --index-root DIR")

DEFAULT_INDEX = args.default_index or next(iter(INDEX_DIRS))
if DEFAULT_INDEX not in INDEX_DIRS:
    raise SystemExit(f"--default-index {DEFAULT_INDEX!r} is not one of {list(INDEX_DIRS)}")

print("Indexes registered (loaded lazily on first use):")
for n, d in INDEX_DIRS.items():
    print(f"  {n:<24} {d}{'   [default]' if n == DEFAULT_INDEX else ''}")

print(f"\nLoading model from {args.model_path} ...")
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
# fp16 is a GPU optimisation; CPU needs fp32 or matmuls fall over.
_dtype = torch.float16 if args.device == "cuda" else torch.float32
model = AutoModel.from_pretrained(
    args.model_path, trust_remote_code=True, dtype=_dtype
).to(args.device).eval()
print(f"  Model ready. dim={model.config.hidden_size}")

_loaded: dict[str, dict] = {}
_load_lock = threading.Lock()


def get_index(name: str) -> dict:
    """Load {faiss, ids, corpus} for `name` on first use, then cache it."""
    if name in _loaded:
        return _loaded[name]
    with _load_lock:
        if name in _loaded:
            return _loaded[name]
        if name not in INDEX_DIRS:
            raise KeyError(name)
        d = INDEX_DIRS[name]

        ivfsq8, flat = (os.path.join(d, "_cached_ivfsq8.faiss"),
                        os.path.join(d, "_cached.faiss"))
        index_path = ivfsq8 if os.path.exists(ivfsq8) else flat
        print(f"[{name}] loading FAISS {index_path} ...")
        idx = faiss.read_index(index_path)
        if isinstance(idx, faiss.IndexIVF):
            idx.nprobe = args.nprobe
        if args.gpu_index:
            idx = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, idx)

        with open(os.path.join(d, "_cached_ids.pkl"), "rb") as f:
            ids = pickle.load(f)

        corpus: dict[str, str] = {}
        for shard in sorted(glob.glob(os.path.join(d, "corpus.shard*.pkl"))):
            with open(shard, "rb") as f:
                corpus.update(pickle.load(f))

        print(f"[{name}] ready: {idx.ntotal:,} vectors, {len(corpus):,} documents")
        _loaded[name] = {"faiss": idx, "ids": ids, "corpus": corpus}
        return _loaded[name]


def embed_texts(texts: list[str]):
    enc = tokenizer(texts, padding=True, truncation=True,
                    max_length=args.max_length, return_tensors="pt").to(args.device)
    with torch.no_grad():
        out = model(**enc)
    seq_lens = enc["attention_mask"].sum(dim=1) - 1
    vecs = out.last_hidden_state[torch.arange(len(texts), device=args.device), seq_lens]
    return F.normalize(vecs, p=2, dim=-1).cpu().float().numpy()


app = FastAPI(title="OBLIQ Multi-Index Search Server")


class EncodeRequest(BaseModel):
    inputs: list[str]


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    index: str | None = None


class DocumentRequest(BaseModel):
    doc_id: str
    index: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "indexes": list(INDEX_DIRS),
            "loaded": list(_loaded), "default": DEFAULT_INDEX}


@app.get("/indexes")
def list_indexes():
    return {"indexes": [
        {"name": n, "dir": d, "loaded": n in _loaded,
         "vectors": _loaded[n]["faiss"].ntotal if n in _loaded else None}
        for n, d in INDEX_DIRS.items()]}


@app.post("/encode")
def encode(req: EncodeRequest):
    return {"embeddings": embed_texts(req.inputs).tolist()}


@app.post("/search")
def search(req: SearchRequest):
    name = req.index or DEFAULT_INDEX
    try:
        store = get_index(name)
    except KeyError:
        return {"error": f"unknown index '{name}'", "available": list(INDEX_DIRS), "results": []}

    q_vec = embed_texts([QUERY_INSTRUCTION + req.query])
    scores, indices = store["faiss"].search(q_vec, req.top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        doc_id = store["ids"][idx]
        results.append({"score": round(float(score), 4), "doc_id": doc_id,
                        "text": store["corpus"].get(doc_id, "")[:2000]})
    return {"results": results}


@app.post("/document")
def get_document(req: DocumentRequest):
    name = req.index or DEFAULT_INDEX
    try:
        store = get_index(name)
    except KeyError:
        return {"error": f"unknown index '{name}'", "doc_id": req.doc_id, "text": ""}
    text = store["corpus"].get(req.doc_id)
    if text is None:
        return {"error": f"doc_id '{req.doc_id}' not found", "doc_id": req.doc_id, "text": ""}
    return {"doc_id": req.doc_id, "text": text}


if __name__ == "__main__":
    print(f"\nStarting multi-index search server on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
