"""Build a bm25s index over the FRAMES Wikipedia dump (TFDS wikipedia/20230601.en), one document per article.

Steps: shard (parallel per-shard tokenisation, TFRecords parsed without TensorFlow), merge (Lucene BM25 written in bm25s's
save format, so bm25s.BM25.load(..., mmap=True) reads it), validate (top-10 agreement with plain bm25s on two shards).
  python retrieval/frames_build_bm25.py all --workers 48
"""
import argparse, glob, json, os, struct, sys, time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agents", "shared"))
from rlm_shared.paths import path  # noqa: E402

TFDS = f"{path('FRAMES_DATA_DIR')}/tfds"
OUT = str(path("FRAMES_DATA_DIR"))
K1, B = 0.9, 0.4  # Anserini/Pyserini defaults; the FRAMES paper does not state them


def _varint(buf, i):
    shift = res = 0
    while True:
        b = buf[i]; i += 1
        res |= (b & 0x7F) << shift
        if b < 0x80:
            return res, i
        shift += 7


def _fields(buf):
    """Yield (field_number, bytes) for the length-delimited fields of a protobuf message."""
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        wt = key & 7
        if wt == 2:
            ln, i = _varint(buf, i)
            yield key >> 3, buf[i:i + ln]; i += ln
        elif wt == 0:
            _, i = _varint(buf, i)
        elif wt == 5:
            i += 4
        elif wt == 1:
            i += 8
        else:
            raise ValueError(f"wire type {wt}")


def parse_example(rec):
    """tf.train.Example -> {name: first bytes value}. Example{1:Features{1:map<str,Feature>}}."""
    out = {}
    for f, features in _fields(rec):
        if f != 1:
            continue
        for f2, entry in _fields(features):
            if f2 != 1:
                continue
            key = val = None
            for f3, v in _fields(entry):
                if f3 == 1:
                    key = v.decode()
                elif f3 == 2:
                    val = v
            for f4, blist in _fields(val):          # Feature.bytes_list = 1
                if f4 == 1:
                    for f5, b in _fields(blist):    # BytesList.value = 1
                        out[key] = b; break
    return out


def read_tfrecord(path):
    with open(path, "rb") as fh:
        while True:
            head = fh.read(12)
            if len(head) < 12:
                return
            (ln,) = struct.unpack("<Q", head[:8])
            rec = fh.read(ln); fh.read(4)
            yield parse_example(rec)


def shard_files():
    fs = sorted(glob.glob(f"{TFDS}/wikipedia-train.tfrecord-*-of-00256"))
    assert len(fs) == 256, f"expected 256 shards, found {len(fs)}"
    return fs


def tokenize_shard(args):
    idx, path, outdir = args
    done = f"{outdir}/shard{idx:03d}.npz"
    if os.path.exists(done):
        return idx
    import bm25s, Stemmer
    titles, texts = [], []
    for ex in read_tfrecord(path):
        titles.append(ex["title"].decode()); texts.append(ex["text"].decode())
    tok = bm25s.tokenize([f"{t}\n{x}" for t, x in zip(titles, texts)], stopwords="en",
                         stemmer=Stemmer.Stemmer("english"), show_progress=False)
    inv = [None] * len(tok.vocab)
    for s, i in tok.vocab.items():
        inv[i] = s
    doc_len = np.fromiter((len(d) for d in tok.ids), np.int32, len(tok.ids))
    terms, tfs, docs = [], [], []
    for d, ids in enumerate(tok.ids):
        if not ids:
            continue
        u, c = np.unique(np.asarray(ids, np.int32), return_counts=True)
        terms.append(u); tfs.append(c.astype(np.int32)); docs.append(np.full(len(u), d, np.int32))
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0, np.int32)
    with open(f"{outdir}/shard{idx:03d}.vocab.json", "w") as f:
        json.dump(inv, f, ensure_ascii=False)
    with open(f"{outdir}/shard{idx:03d}.corpus.jsonl", "w") as f:
        for t, x in zip(titles, texts):
            f.write(json.dumps({"title": t, "text": x}, ensure_ascii=False) + "\n")
    np.savez(done + ".tmp.npz", doc_len=doc_len, term=cat(terms), tf=cat(tfs), doc=cat(docs))
    os.replace(done + ".tmp.npz", done)
    return idx


def merge(shard_ids, workdir, index_dir, corpus_path):
    t0 = time.time()
    vocab, gmaps, lens = {"": 0}, [], []
    for s in shard_ids:
        loc = json.load(open(f"{workdir}/shard{s:03d}.vocab.json"))
        gmaps.append(np.fromiter((vocab.setdefault(w, len(vocab)) for w in loc), np.int64, len(loc)))
        lens.append(np.load(f"{workdir}/shard{s:03d}.npz")["doc_len"])
    V, offs = len(vocab), np.cumsum([0] + [len(l) for l in lens])
    N, avgdl = int(offs[-1]), float(np.concatenate(lens).mean())
    df = np.zeros(V, np.int64)
    for s, g in zip(shard_ids, gmaps):
        df += np.bincount(g[np.load(f"{workdir}/shard{s:03d}.npz")["term"]], minlength=V)
    nnz = int(df.sum())
    print(f"vocab {V:,}  docs {N:,}  avgdl {avgdl:.1f}  nnz {nnz:,}  ({time.time()-t0:.0f}s)", flush=True)
    idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
    indptr = np.zeros(V + 1, np.int64); np.cumsum(df, out=indptr[1:])
    data = np.empty(nnz, np.float32); indices = np.empty(nnz, np.int32)
    cursor = indptr[:-1].copy()
    for k, (s, g) in enumerate(zip(shard_ids, gmaps)):
        z = np.load(f"{workdir}/shard{s:03d}.npz")
        term, tf, doc = g[z["term"]], z["tf"].astype(np.float32), z["doc"]
        order = np.argsort(term, kind="stable")      # doc order kept within each term
        term, tf, doc = term[order], tf[order], doc[order]
        rank = np.arange(len(term)) - np.searchsorted(term, term, side="left")
        pos = cursor[term] + rank
        l_d = lens[k][doc].astype(np.float32)
        data[pos] = idf[term] * tf / (K1 * ((1 - B) + B * l_d / avgdl) + tf)
        indices[pos] = doc + offs[k]
        u, c = np.unique(term, return_counts=True); cursor[u] += c
        if k % 32 == 0:
            print(f"  filled shard {k+1}/{len(shard_ids)} ({time.time()-t0:.0f}s)", flush=True)
    assert np.array_equal(cursor, indptr[1:]), "CSC fill mismatch"
    os.makedirs(index_dir, exist_ok=True)
    np.save(f"{index_dir}/data.csc.index.npy", data)
    np.save(f"{index_dir}/indices.csc.index.npy", indices)
    np.save(f"{index_dir}/indptr.csc.index.npy", indptr)
    with open(f"{index_dir}/vocab.index.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    import bm25s
    with open(f"{index_dir}/params.index.json", "w") as f:
        json.dump(dict(k1=K1, b=B, delta=0.5, method="lucene", idf_method="lucene", dtype="float32",
                       int_dtype="int32", num_docs=N, version=bm25s.__version__, backend="numpy"), f, indent=4)
    if corpus_path:                                   # global doc id = line number
        with open(corpus_path, "w") as out:
            for s in shard_ids:
                with open(f"{workdir}/shard{s:03d}.corpus.jsonl") as f:
                    for line in f:
                        out.write(line)
    print(f"merged in {time.time()-t0:.0f}s -> {index_dir}", flush=True)


def validate(workdir):
    import bm25s, Stemmer, csv
    stem = Stemmer.Stemmer("english")
    texts = []
    for s in (0, 1):
        with open(f"{workdir}/shard{s:03d}.corpus.jsonl") as f:
            texts += [f"{d['title']}\n{d['text']}" for d in map(json.loads, f)]
    ref = bm25s.BM25(k1=K1, b=B, method="lucene")
    ref.index(bm25s.tokenize(texts, stopwords="en", stemmer=stem, show_progress=False), show_progress=False)
    ours = bm25s.BM25.load(f"{workdir}/val_index", mmap=True)
    qs = [r["Prompt"] for r in csv.DictReader(open(f"{OUT}/test.tsv"), delimiter="\t")][:100]
    qt = bm25s.tokenize(qs, stopwords="en", stemmer=stem, show_progress=False)
    a_d, a_s = ref.retrieve(qt, k=10, show_progress=False)
    qt = bm25s.tokenize(qs, stopwords="en", stemmer=stem, show_progress=False)
    b_d, b_s = ours.retrieve(qt, k=10, show_progress=False)
    same = (a_d == b_d).mean(); dmax = float(np.abs(a_s - b_s).max())
    print(f"VALIDATE: {len(texts)} docs, top-10 id agreement {same:.4f}, max |score diff| {dmax:.2e}")
    ok = dmax < 1e-3 and same > 0.99           # ties can swap order
    print("VALIDATE", "PASS" if ok else "FAIL"); return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["shard", "merge", "validate", "all"])
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    work = f"{OUT}/work"; os.makedirs(work, exist_ok=True)
    if a.step in ("shard", "all"):
        fs = shard_files(); t0 = time.time()
        with Pool(a.workers) as p:
            for n, i in enumerate(p.imap_unordered(tokenize_shard, [(i, f, work) for i, f in enumerate(fs)]), 1):
                if n % 16 == 0:
                    print(f"tokenized {n}/256 shards ({time.time()-t0:.0f}s)", flush=True)
    if a.step in ("validate", "all"):
        merge([0, 1], work, f"{work}/val_index", None)
        if not validate(work):
            sys.exit(1)
    if a.step in ("merge", "all"):
        merge(list(range(256)), work, f"{OUT}/bm25_index", f"{OUT}/corpus.jsonl")
