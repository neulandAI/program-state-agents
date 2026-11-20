"""Build a FAISS IndexFlatIP (optionally also IVF-SQ8) for an OBLIQ-Bench corpus via a running server's /encode endpoint.

Output in <out>: _cached.faiss, _cached_ids.pkl, corpus.shard1_of_1.pkl (the format search_server.py reads).
  python retrieval/build_obliq_index.py --task analogues --subset math --out ~/indexes/obliq-ana-math --server http://localhost:8001
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
import requests

HF_REPO = "dianetc/OBLIQ-Bench"

CORPUS_PATHS = {
    ("tip-of-tongue", "congress"): "tip-of-tongue/congress/corpus/corpus.jsonl",
    ("analogues",     "math"):     "analogues/math/corpus/corpus.jsonl",
    ("analogues",     "writing"):  "analogues/writing/corpus/corpus.jsonl",
    ("descriptive",   "twitter"):  "descriptive/twitter/corpus/corpus.jsonl",
    ("descriptive",   "wildchat"): "descriptive/wildchat/corpus/corpus.jsonl",
}


def download_corpus_from_hf(task: str, subset: str) -> list[tuple[str, str]]:
    """Stream corpus.jsonl directly from HuggingFace — no local clone needed."""
    from huggingface_hub import hf_hub_download

    hf_path = CORPUS_PATHS[(task, subset)]
    print(f"Downloading {hf_path} from {HF_REPO} ...")
    local = hf_hub_download(
        repo_id=HF_REPO,
        filename=hf_path,
        repo_type="dataset",
    )
    print(f"  Cached at {local}")
    return _parse_jsonl(local)


def load_corpus(path: str) -> list[tuple[str, str]]:
    """Load corpus.jsonl from a local path → list of (doc_id, text)."""
    print(f"Loading corpus from {path} ...")
    return _parse_jsonl(path)


def _parse_jsonl(path: str) -> list[tuple[str, str]]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append((obj["_id"], obj.get("text", "")))
    print(f"  {len(docs):,} documents loaded")
    return docs


def embed_batch(server: str, texts: list[str]) -> np.ndarray:
    """Call /encode on the running embed server and return float32 array."""
    r = requests.post(
        f"{server.rstrip('/')}/encode",
        json={"inputs": texts},
        timeout=300,
    )
    r.raise_for_status()
    arr = np.array(r.json()["embeddings"], dtype="float32")
    faiss.normalize_L2(arr)
    return arr


def build_index(
    docs: list[tuple[str, str]],
    server: str,
    batch_size: int,
    out_dir: str | None = None,
) -> tuple[faiss.Index, list[str]]:
    """Embed all documents and build a FAISS IndexFlatIP.

    If out_dir is given, embeddings are checkpointed to
    <out_dir>/_embed_cache.npy + <out_dir>/_embed_progress.txt
    so a crashed run auto-resumes from the last completed batch.
    """
    ids   = [d[0] for d in docs]
    texts = [d[1] for d in docs]
    n     = len(docs)

    print(f"Embedding {n:,} documents (batch={batch_size}) via {server} ...")

    first = embed_batch(server, texts[:1])
    dim   = first.shape[1]
    print(f"  Embedding dim: {dim}")

    cache_npy  = Path(out_dir) / "_embed_cache.npy"  if out_dir else None
    cache_prog = Path(out_dir) / "_embed_progress.txt" if out_dir else None

    done_so_far = 0
    if cache_npy and cache_npy.exists() and cache_prog and cache_prog.exists():
        done_so_far = int(cache_prog.read_text().strip())
        print(f"  Resuming from checkpoint: {done_so_far:,}/{n:,} docs already embedded")

    if cache_npy:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        mode = "r+" if (cache_npy.exists() and done_so_far > 0) else "w+"
        mmap = np.memmap(str(cache_npy), dtype="float32", mode=mode,
                         shape=(n, dim))
    else:
        mmap = np.empty((n, dim), dtype="float32")

    n_batches = math.ceil(n / batch_size)
    t0 = time.time()

    for bi in range(n_batches):
        lo = bi * batch_size
        hi = min(lo + batch_size, n)
        if hi <= done_so_far:
            continue

        batch_texts = texts[lo:hi]
        vecs = embed_batch(server, batch_texts)
        mmap[lo:hi] = vecs

        if cache_prog:
            cache_prog.write_text(str(hi))

        elapsed = time.time() - t0
        done    = hi - done_so_far
        speed   = done / elapsed if elapsed > 0 else 0
        eta     = (n - hi) / speed if speed > 0 else 0
        print(f"  [{hi:>{len(str(n))}}/{n}]  {elapsed:.0f}s  {speed:.0f} docs/s  ETA {eta:.0f}s")

    all_vecs = np.array(mmap)
    index = faiss.IndexFlatIP(dim)
    index.add(all_vecs)

    if cache_npy and cache_npy.exists():
        cache_npy.unlink()
    if cache_prog and cache_prog.exists():
        cache_prog.unlink()

    print(f"Done. Total {time.time()-t0:.1f}s  index.ntotal={index.ntotal:,}")
    return index, ids


def save_outputs(
    out_dir: str,
    index: faiss.Index,
    ids: list[str],
    docs: list[tuple[str, str]],
    build_ivfsq8: bool,
    nlist: int,
    nprobe: int,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    flat_path = str(out / "_cached.faiss")
    ids_path  = str(out / "_cached_ids.pkl")
    corp_path = str(out / "corpus.shard1_of_1.pkl")

    print(f"\nSaving flat index → {flat_path}")
    faiss.write_index(index, flat_path)
    print(f"  {Path(flat_path).stat().st_size / 1e6:.1f} MB")

    print(f"Saving ids pickle → {ids_path}")
    with open(ids_path, "wb") as f:
        pickle.dump(ids, f)

    print(f"Saving corpus pickle → {corp_path}")
    corpus_dict = {doc_id: text for doc_id, text in docs}
    with open(corp_path, "wb") as f:
        pickle.dump(corpus_dict, f)
    print(f"  {Path(corp_path).stat().st_size / 1e6:.1f} MB")

    if build_ivfsq8:
        print(f"\nBuilding IVFScalarQuantizer SQ8 (nlist={nlist}, nprobe={nprobe}) ...")
        n, d = index.ntotal, index.d
        print(f"  Extracting {n:,} vectors ...")
        vectors = index.reconstruct_n(0, n)

        quantizer = faiss.IndexFlatIP(d)
        ivfsq8    = faiss.IndexIVFScalarQuantizer(
            quantizer, d, nlist,
            faiss.ScalarQuantizer.QT_8bit,
            faiss.METRIC_INNER_PRODUCT,
        )
        print("  Training ...")
        ivfsq8.train(vectors)
        print("  Adding vectors ...")
        ivfsq8.add(vectors)
        ivfsq8.nprobe = nprobe

        ivfsq8_path = str(out / "_cached_ivfsq8.faiss")
        faiss.write_index(ivfsq8, ivfsq8_path)
        size_mb = Path(ivfsq8_path).stat().st_size / 1e6
        flat_mb = n * d * 4 / 1e6
        print(f"  Saved {ivfsq8_path}  ({size_mb:.1f} MB vs {flat_mb:.0f} MB flat)")

        rng    = np.random.default_rng(42)
        q      = vectors[rng.integers(0, n, min(200, n))].copy().astype("float32")
        faiss.normalize_L2(q)
        _, fi  = index.search(q, 10)
        _, hi  = ivfsq8.search(q, 10)
        recall = sum(len(set(a) & set(b)) / 10 for a, b in zip(fi, hi)) / len(fi)
        print(f"  IVFSQ8 recall@10 vs flat: {recall:.4f}")

    print(f"\nIndex files saved to: {out_dir}")
    print("To serve, set FAISS_INDEX_DIR=" + out_dir + " and restart the embed server.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OBLIQ FAISS index")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--corpus", metavar="PATH",
                     help="Local path to corpus.jsonl")
    src.add_argument("--task",   choices=["tip-of-tongue", "analogues", "descriptive"],
                     help="Download from HuggingFace: task type")

    parser.add_argument("--subset", choices=["congress", "math", "writing", "twitter", "wildchat"],
                        help="OBLIQ subset (required when using --task)")
    parser.add_argument("--out",    required=True,
                        help="Output directory for index files")
    parser.add_argument("--server", default="http://localhost:8001",
                        help="Running embed server URL (default: http://localhost:8001)")
    parser.add_argument("--batch",  type=int, default=64,
                        help="Embedding batch size (default: 64)")
    parser.add_argument("--ivfsq8", action="store_true",
                        help="Also build IVFScalarQuantizer SQ8 index")
    parser.add_argument("--nlist",  type=int, default=1024,
                        help="IVFSQ8 number of clusters (default: 1024)")
    parser.add_argument("--nprobe", type=int, default=128,
                        help="IVFSQ8 nprobe at query time (default: 128)")
    args = parser.parse_args()

    if args.task and not args.subset:
        parser.error("--subset is required when using --task")

    try:
        resp = requests.get(f"{args.server.rstrip('/')}/health", timeout=10)
        resp.raise_for_status()
        print(f"Embed server OK: {args.server}")
    except Exception as e:
        raise SystemExit(f"[ERROR] Embed server not reachable at {args.server}: {e}")

    if args.task:
        docs = download_corpus_from_hf(args.task, args.subset)
    else:
        docs = load_corpus(args.corpus)

    index, ids = build_index(docs, args.server, args.batch, out_dir=args.out)
    save_outputs(args.out, index, ids, docs, args.ivfsq8, args.nlist, args.nprobe)


if __name__ == "__main__":
    main()
