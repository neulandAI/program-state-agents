"""BrowseComp-Plus, document REPL agent with Qwen3.6 and dense Qwen3-Embedding-8B retrieval.

    python src/bcp_qwen_dense.py [workers]        -> $RUNS_DIR/run5d/bcp_fd_qwen_dense

The split_runner.py setup of the Qwen3.6 runs with the dense server retrieval_server.py on :8765; the model servers
on :8030 and :8031 are used alternately. Resumable through split_runner's cache.
"""
import hashlib, importlib.util, json, os, sys

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SRC, os.pardir, os.pardir, "shared"))
from rlm_shared.paths import path  # noqa: E402

OUT = f"{path('RUNS_DIR')}/run5d/bcp_fd_qwen_dense"
os.environ.update(GEPA_BCP_SERVERS="http://localhost:8030/v1,http://localhost:8031/v1",
                  GEPA_BCP_RETRIEVAL="http://127.0.0.1:8765", GEPA_BCP_MODEL="qwen36-35b-a3b", GEPA_OUT=OUT,
                  LOCAL_JUDGE_MODEL="qwen36-35b-a3b", SUP_FENCE_TOLERANT="1", SUP_MAX_BLOCKS="5",
                  SUP_PRESENCE_PENALTY="1.5", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
for k in ("QUERIES_FILE", "GEPA_PROMPTS", "BM25_CORPUS"):
    os.environ.pop(k, None)
sys.path.insert(0, str(path("RLMLIB_PATH")))
_s = importlib.util.spec_from_file_location("split_runner", f"{SRC}/split_runner.py")
G = importlib.util.module_from_spec(_s); sys.modules["split_runner"] = G; _s.loader.exec_module(G)
G.FINGERPRINT = hashlib.sha1((G.FINGERPRINT + "|dense-exact-8765").encode()).hexdigest()[:8]

if __name__ == "__main__":
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    ids = [str(json.loads(l)["query_id"]) for l in open(f"{path('FULLDOC_ROOT')}/data/prepared/queries.jsonl")]
    if os.environ.get("BCP_IDS_LIMIT"):
        ids = ids[:int(os.environ["BCP_IDS_LIMIT"])]
    G._cache_load()
    seed = G.seed_blocks()
    G._log(f"[dense] seed {G.ckey(seed)} fingerprint {G.FINGERPRINT} on {len(ids)} ids via {G.RETRIEVAL}, model {G.SERVERS}")
    G.run_batch(seed, ids, W)
