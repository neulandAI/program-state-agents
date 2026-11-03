"""Build the Reason-ModernColBERT PLAID index of BrowseComp-Plus at document length 8192, chunked and resumable.

Each chunk's embeddings are checkpointed before indexing; --start N resumes after the first N documents (the per-chunk
log line prints the value to use). Run from the directory that holds .pylate_idx/; needs a GPU with >30 GB free.
"""
import argparse, os, time
import numpy as np
import torch
from datasets import load_dataset
from pylate import indexes, models

ap = argparse.ArgumentParser()
ap.add_argument("--index-folder", default=".pylate_idx")
ap.add_argument("--index-name", default="bc_rmcb_dl8192")
ap.add_argument("--document-length", type=int, default=8192)
ap.add_argument("--query-length", type=int, default=256)
ap.add_argument("--chunk", type=int, default=25000, help="documents per encode/add cycle")
ap.add_argument("--batch-size", type=int, default=4)
ap.add_argument("--nbits", type=int, default=4)
ap.add_argument("--start", type=int, default=0, help="resume: skip the first N documents")
ap.add_argument("--add-chunk", type=int, default=12500,
                help="documents per add_documents() call; caps GPU-resident embeddings")
ap.add_argument("--centroid-batch", type=int, default=512,
                help="rows per centroid-distance batch inside fast_plaid (upstream default 4096)")
ap.add_argument("--rust-batch", type=int, default=2048,
                help="batch_size handed to fast_plaid_rust.update (upstream default 50000)")
a = ap.parse_args()

MIN_FREE_START_GB = 30   # before anything is loaded
MIN_FREE_CHUNK_GB = 8    # mid-build, excluding this process's own PLAID structures

# pylate never forwards a batch size to FastPlaid.update (default 50,000), which runs out of memory as the centroid
# table grows; inject a smaller one.
import inspect
import fast_plaid.search.fast_plaid as _fpf

_orig_update = _fpf.FastPlaid.update
assert "batch_size" in inspect.signature(_orig_update).parameters, \
    "FastPlaid.update no longer takes batch_size; re-read the source"

def _update_with_capped_batch(self, *args, **kwargs):
    kwargs.setdefault("batch_size", a.rust_batch)
    return _orig_update(self, *args, **kwargs)

_fpf.FastPlaid.update = _update_with_capped_batch
print(f"patched fast_plaid_rust update batch 50000 -> {a.rust_batch}", flush=True)

# update_centroids() uses a hardcoded 4096-row batch for a batch x n_centroids distance matrix that grows with every
# chunk; a smaller batch lowers peak memory without changing the result (the loop is a batched argmin).
import fast_plaid.search.update as _fpu

_src = inspect.getsource(_fpu)
_needle = "    batch_size = 4096\n"
assert _src.count(_needle) == 1, (
    "fast_plaid.search.update no longer contains a single 'batch_size = 4096'; "
    "re-read the source before trusting this patch")
_ns = dict(_fpu.__dict__)
exec(compile(_src.replace(_needle, f"    batch_size = {a.centroid_batch}\n"),
             _fpu.__file__ + " [patched]", "exec"), _ns)
_fpu.update_centroids = _ns["update_centroids"]   # process_update resolves this as a global
print(f"patched fast_plaid centroid-distance batch 4096 -> {a.centroid_batch}", flush=True)


def gpu_free_gb():
    free, _ = torch.cuda.mem_get_info(0)
    return free / 1e9


_f = gpu_free_gb()
if _f < MIN_FREE_START_GB:
    raise SystemExit(f"ABORT: only {_f:.1f} GB GPU free, need >{MIN_FREE_START_GB}. "
                     f"Something else holds the card; start again when it is free.")
print(f"gate passed: {_f:.1f} GB GPU free", flush=True)

t0 = time.time()
ds = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")
ids, texts = [str(x) for x in ds["docid"]], list(ds["text"])
print(f"corpus: {len(ids):,} documents in {time.time()-t0:.0f}s", flush=True)

model = models.ColBERT(model_name_or_path="lightonai/Reason-ModernColBERT",
                       document_length=a.document_length, query_length=a.query_length,
                       model_kwargs={"torch_dtype": torch.float16}).to("cuda")
print(f"model on cuda, document_length={a.document_length}", flush=True)

# use_triton=False: the Triton kernel path hit a device-side assert on the build machine.
index = indexes.PLAID(index_folder=a.index_folder, index_name=a.index_name,
                      override=(a.start == 0), nbits=a.nbits, use_triton=False)
if a.start:
    ids, texts = ids[a.start:], texts[a.start:]
    print(f"resuming: skipping the first {a.start:,} documents already indexed", flush=True)

done, t_start, stopped_early = a.start, time.time(), False
total = a.start + len(ids)
for s in range(0, len(ids), a.chunk):
    # Stop at a chunk boundary, with a usable --start, if another process has taken the GPU.
    _f = gpu_free_gb()
    if _f < MIN_FREE_CHUNK_GB:
        print(f"STOPPING CLEANLY: GPU free dropped to {_f:.1f} GB (<{MIN_FREE_CHUNK_GB}). "
              f"Everything up to document {done:,} is indexed. "
              f"RESUME WITH --start {done}", flush=True)
        stopped_early = True
        break
    cid, ctx = ids[s:s+a.chunk], texts[s:s+a.chunk]
    ckpt = os.path.join(a.index_folder, f"{a.index_name}.chunk{done}.npz")
    # A leftover checkpoint means only the index step failed: reuse the embeddings.
    if os.path.exists(ckpt):
        t = time.time()
        with np.load(ckpt) as z:
            emb = [z[str(i)] for i in range(len(cid))]
        t_enc = time.time() - t
        print(f"  reused checkpoint for documents {done:,}-{done+len(cid):,} "
              f"({t_enc:.0f}s, no re-encode)", flush=True)
    else:
        t = time.time()
        emb = model.encode(sentences=ctx, batch_size=a.batch_size, is_query=False, show_progress_bar=False)
        t_enc = time.time() - t
        torch.cuda.empty_cache()                   # release cached blocks before PLAID allocates
        try:
            np.savez(ckpt, **{str(i): e for i, e in enumerate(emb)})
        except Exception as e:
            print(f"  (checkpoint failed, continuing: {e})", flush=True)
    n_emb = sum(e.shape[0] for e in emb)
    torch.cuda.empty_cache()
    # add_documents copies all its embeddings to the GPU at once, so feed it --add-chunk documents at a time.
    t = time.time()
    for j in range(0, len(cid), a.add_chunk):
        index.add_documents(documents_ids=cid[j:j+a.add_chunk],
                            documents_embeddings=emb[j:j+a.add_chunk])
        torch.cuda.empty_cache()
        print(f"    added {done+min(j+a.add_chunk, len(cid)):,} | "
              f"{gpu_free_gb():.1f} GB free", flush=True)
    try: os.remove(ckpt)
    except Exception: pass
    t_add = time.time() - t
    del emb; torch.cuda.empty_cache()
    done += len(cid)
    rate = (done - a.start) / (time.time() - t_start)
    print(f"  {done:,}/{total:,} docs | {n_emb/len(cid):.0f} emb/doc | encode {t_enc:.0f}s "
          f"add {t_add:.0f}s | {rate:.2f} docs/s | ETA {(total-done)/rate/60:.0f}m"
          f" | {gpu_free_gb():.1f} GB free | RESUME WITH --start {done}", flush=True)
if stopped_early:
    print(f"STOPPED EARLY after {(time.time()-t_start)/60:.1f} min at document {done:,}/{total:,}"
          f" -- index is INCOMPLETE, resume with --start {done}", flush=True)
else:
    print(f"DONE in {(time.time()-t_start)/60:.1f} min -> {a.index_folder}/{a.index_name}", flush=True)
