"""Embed an OBLIQ-Bench corpus with gemini-embedding-2 on Vertex AI for the Gemini search server.

Documents are embedded one per request, in parallel, and L2-normalised; progress is checkpointed every CHUNK documents.
Output: <out>/emb.npy, <out>/ids.json, <out>/meta.json.
  python retrieval/gemini/build_index.py --corpus <obliq>/analogues/math/corpus/corpus.jsonl --out ~/indexes/gemini/obliq-ana-math
"""
import argparse, glob, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vertex  # noqa: E402

MODEL = "gemini-embedding-2"
CHUNK = 2000

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--workers", type=int, default=64)
ap.add_argument("--doc-format", choices=["raw", "google"], default="raw",
                help="raw = the document text as is (the OBLIQ paper's setup); google = "
                     "'title: none | text: ...' (Google's recommended retrieval format)")
args = ap.parse_args()
out = os.path.expanduser(args.out)
os.makedirs(out, exist_ok=True)

ids, texts = [], []
for line in open(args.corpus):
    if line.strip():
        d = json.loads(line)
        ids.append(str(d["_id"]))
        title = (d.get("title") or "").strip() or "none"
        texts.append(f"title: {title} | text: {d.get('text') or ''}" if args.doc_format == "google" else (d.get("text") or ""))
print(f"{len(ids)} documents from {args.corpus}", flush=True)

client = vertex.client()
tokens = [0]


def embed(t):
    for attempt in range(8):
        try:
            e = client.models.embed_content(model=MODEL, contents=t).embeddings[0]
            if e.statistics and e.statistics.token_count:
                tokens[0] += int(e.statistics.token_count)
            return e.values
        except Exception as ex:  # noqa: BLE001  quota or server errors: back off and retry
            if attempt == 7:
                raise
            time.sleep(min(60, 2 ** attempt) + 0.1 * attempt)
            if attempt >= 3:
                print(f"  retry {attempt + 1}: {str(ex)[:120]}", flush=True)


t0 = time.time()
with ThreadPoolExecutor(args.workers) as ex:
    for s in range(0, len(texts), CHUNK):
        part = f"{out}/part_{s // CHUNK:05d}.npy"
        if os.path.exists(part):
            continue
        v = np.asarray(list(ex.map(embed, texts[s:s + CHUNK])), dtype=np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        np.save(part, v)
        done = min(s + CHUNK, len(texts))
        rate = done / (time.time() - t0)
        print(f"  {done}/{len(texts)}  {rate:.0f} docs/s  eta {(len(texts) - done) / max(rate, 1e-9) / 60:.0f} min", flush=True)

parts = sorted(glob.glob(f"{out}/part_*.npy"))
emb = np.concatenate([np.load(p) for p in parts])
assert emb.shape[0] == len(ids), (emb.shape, len(ids))
np.save(f"{out}/emb.npy", emb)
json.dump(ids, open(f"{out}/ids.json", "w"))
json.dump({"model": MODEL, "corpus": args.corpus, "n": len(ids), "dim": int(emb.shape[1]),
           "doc_format": args.doc_format,
           "location": os.environ.get("GEMINI_LOCATION", "eu"), "tokens_this_session": tokens[0],
           "built": time.strftime("%F %T")}, open(f"{out}/meta.json", "w"), indent=1)
for p in parts:
    os.remove(p)
print(f"done: {emb.shape} in {(time.time() - t0) / 60:.1f} min, {tokens[0]:,} tokens this session", flush=True)
