"""Single-index Qwen3-Embedding-8B FAISS search server (rlm wire format: /health, /encode, /search, /document).

--index-dir holds _cached.faiss or _cached_ivfsq8.faiss, _cached_ids.pkl and corpus.shard*.pkl (build_obliq_index.py).
  python retrieval/search_server.py --model-path <Qwen3-Embedding-8B> --index-dir <index> --port 8002
"""

import argparse
import glob
import os
import pickle

import faiss
import numpy as np
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
parser.add_argument("--index-dir",  required=True)
parser.add_argument("--max-length", type=int, default=8192)
parser.add_argument("--nprobe",     type=int, default=64)
parser.add_argument("--port",       type=int, default=8002)
parser.add_argument("--no-gpu-index", action="store_true",
                    help="Keep FAISS index on CPU (saves GPU memory)")
args = parser.parse_args()

print(f"Loading model from {args.model_path} ...")
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(
    args.model_path, trust_remote_code=True, dtype=torch.float16
).cuda().eval()
print(f"  Model ready. dim={model.config.hidden_size}")

ivfsq8_path = os.path.join(args.index_dir, "_cached_ivfsq8.faiss")
flat_path   = os.path.join(args.index_dir, "_cached.faiss")
index_path  = ivfsq8_path if os.path.exists(ivfsq8_path) else flat_path

print(f"Loading FAISS index from {index_path} ...")
faiss_index = faiss.read_index(index_path)
if isinstance(faiss_index, faiss.IndexIVF):
    faiss_index.nprobe = args.nprobe

if not args.no_gpu_index:
    res = faiss.StandardGpuResources()
    faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
    print("  FAISS on GPU.")

print(f"  FAISS ready: {faiss_index.ntotal:,} vectors")

ids_path = os.path.join(args.index_dir, "_cached_ids.pkl")
print(f"Loading IDs from {ids_path} ...")
with open(ids_path, "rb") as f:
    faiss_lookup: list[str] = pickle.load(f)

print(f"Loading corpus from {args.index_dir}/corpus.shard*.pkl ...")
corpus: dict[str, str] = {}
for shard in sorted(glob.glob(os.path.join(args.index_dir, "corpus.shard*.pkl"))):
    with open(shard, "rb") as f:
        corpus.update(pickle.load(f))
print(f"  Corpus ready: {len(corpus):,} documents")


def embed_texts(texts: list[str]) -> np.ndarray:
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    ).to("cuda")
    with torch.no_grad():
        out = model(**enc)
    seq_lens = enc["attention_mask"].sum(dim=1) - 1
    vecs = out.last_hidden_state[torch.arange(len(texts), device="cuda"), seq_lens]
    vecs = F.normalize(vecs, p=2, dim=-1)
    return vecs.cpu().float().numpy()


app = FastAPI(title="OBLIQ Search Server")


class EncodeRequest(BaseModel):
    inputs: list[str]


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


class DocumentRequest(BaseModel):
    doc_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/encode")
def encode(req: EncodeRequest):
    return {"embeddings": embed_texts(req.inputs).tolist()}


@app.post("/search")
def search(req: SearchRequest):
    q_vec = embed_texts([QUERY_INSTRUCTION + req.query])
    scores, indices = faiss_index.search(q_vec, req.top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        doc_id = faiss_lookup[idx]
        results.append({
            "score": round(float(score), 4),
            "doc_id": doc_id,
            "text": corpus.get(doc_id, "")[:2000],
        })
    return {"results": results}


@app.post("/document")
def get_document(req: DocumentRequest):
    text = corpus.get(req.doc_id)
    if text is None:
        return {"error": f"doc_id '{req.doc_id}' not found", "doc_id": req.doc_id, "text": ""}
    return {"doc_id": req.doc_id, "text": text}


if __name__ == "__main__":
    print(f"Starting search server on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
