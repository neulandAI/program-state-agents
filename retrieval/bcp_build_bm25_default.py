"""BM25 index over the BrowseComp-Plus corpus with Lucene defaults (k1 0.9, b 0.4), as in the benchmark's BM25 baseline.

Writes $BCP_BM25_INDEX_DIR/{bm25_index/, docids.json, offsets.npy}; offsets are byte offsets into corpus.jsonl.
"""
import json, os, sys, time
import bm25s, Stemmer, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agents", "shared"))
from rlm_shared.paths import path  # noqa: E402
CORPUS = f"{path('FULLDOC_ROOT')}/data/prepared/corpus.jsonl"
OUT = str(path("BCP_BM25_INDEX_DIR"))
t0 = time.time(); docids, texts, offs = [], [], []
with open(CORPUS, "rb") as f:
    while True:
        o = f.tell(); line = f.readline()
        if not line: break
        d = json.loads(line); docids.append(str(d["docid"])); texts.append(d["text"]); offs.append(o)
print(f"loaded {len(texts)} docs ({time.time()-t0:.0f}s)", flush=True)
tok = bm25s.tokenize(texts, stopwords="en", stemmer=Stemmer.Stemmer("english"), show_progress=False)
print(f"tokenized ({time.time()-t0:.0f}s)", flush=True)
r = bm25s.BM25(k1=0.9, b=0.4, method="lucene"); r.index(tok, show_progress=False)
r.save(f"{OUT}/bm25_index"); json.dump(docids, open(f"{OUT}/docids.json", "w")); np.save(f"{OUT}/offsets.npy", np.array(offs, dtype=np.int64))
print(f"saved {OUT} ({time.time()-t0:.0f}s)", flush=True)
