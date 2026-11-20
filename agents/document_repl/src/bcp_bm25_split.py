"""BrowseComp-Plus, document REPL agent with Qwen3.6 and BM25 (k1 0.9, b 0.4).

    python src/bcp_bm25_split.py            # test200 split (resumable via cache), then the paired comparison
    python src/bcp_bm25_split.py --report   # comparison only
    BCP_IDS=<ids.txt> BCP_OUT=<dir> BCP_SERVER=http://localhost:8030/v1 python src/bcp_bm25_split.py   # another slice

The split_runner.py setup of the Qwen3.6 runs, except: retrieval from a BM25 server on :8043, entry point
run_bm25.py with BM25_CORPUS=bcp, and test200 shuffled. The comparison step runs the script named by
BCP_POLICY_COMPARE.
"""
import hashlib, importlib.util, json, os, random, subprocess, sys

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SRC, os.pardir, os.pardir, "shared"))
from rlm_shared.paths import path  # noqa: E402

ENTRY = f"{SRC}/run_bm25.py"
# Every slice uses the same fingerprint, so cache keys match across slices.
OUT = os.environ.get("BCP_OUT", f"{path('RUNS_DIR')}/bcp_bm25_test200")
os.environ.update(GEPA_BCP_SERVERS=os.environ.get("BCP_SERVER", "http://localhost:8030/v1"), GEPA_BCP_RETRIEVAL="http://127.0.0.1:8043",
                  GEPA_BCP_MODEL="qwen36-35b-a3b", GEPA_OUT=OUT, LOCAL_JUDGE_MODEL="qwen36-35b-a3b",
                  SUP_FENCE_TOLERANT="1", SUP_MAX_BLOCKS="5", SUP_PRESENCE_PENALTY="1.5",
                  BM25_CORPUS="bcp", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
os.environ.pop("QUERIES_FILE", None)          # BrowseComp-Plus queries (run_one.py default)
sys.path.insert(0, str(path("RLMLIB_PATH")))
_s = importlib.util.spec_from_file_location("split_runner", f"{SRC}/split_runner.py")
G = importlib.util.module_from_spec(_s); sys.modules["split_runner"] = G; _s.loader.exec_module(G)
G.FINGERPRINT = hashlib.sha1((G.FINGERPRINT + "|bcp-bm25-default|bm25f-wrapper").encode()).hexdigest()[:8]


def _rollout(qid, pfile, server, out_dir):
    out = os.path.join(out_dir, f"{qid}.json")
    if not G.PYV:
        raise SystemExit("environment variable PYTHON is not set: interpreter of the rollouts")
    env = dict(os.environ, GEPA_PROMPTS=pfile, PYTHONPATH=str(path("RLMLIB_PATH")),
               HF_HOME=os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), ALIBABA_TOKEN="EMPTY")
    try:
        subprocess.run([G.PYV, ENTRY, "--query-id", qid, "--out", out, "--base-url", server] + G.BASE_ARGS,
                       env=env, cwd=str(path("FULLDOC_ROOT")), capture_output=True, timeout=G.HARD_CAP_S)
    except subprocess.TimeoutExpired:
        G._log(f"[runaway] q{qid} killed at {G.HARD_CAP_S}s"); return None
    try:
        return json.load(open(out))
    except Exception:
        return None


G._rollout = _rollout

if __name__ == "__main__":
    _data = os.environ.get("RLM_DATA_DIR") or sys.exit("set RLM_DATA_DIR (holds bcp_gepa_splits/split.json)")
    test = json.load(open(os.path.join(_data, "bcp_gepa_splits", "split.json")))["test200"]
    if os.environ.get("BCP_IDS"):
        ids = [l.strip() for l in open(os.environ["BCP_IDS"]) if l.strip()]
        G._cache_load(); seed = G.seed_blocks()
        G._log(f"[bm25] seed {G.ckey(seed)} fingerprint {G.FINGERPRINT} on {len(ids)} ids from {os.environ['BCP_IDS']} "
               f"via {G.RETRIEVAL}, model {G.SERVERS}")
        G.run_batch(seed, ids, 8)
        sys.exit(0)
    if "--report" not in sys.argv:
        G._cache_load()
        seed = G.seed_blocks()
        order = list(test); random.Random(0).shuffle(order)
        G._log(f"[bm25] seed {G.ckey(seed)} fingerprint {G.FINGERPRINT} on test200 via {G.RETRIEVAL}")
        G.run_batch(seed, order, 8)
    _cmp = os.environ.get("BCP_POLICY_COMPARE") or sys.exit("set BCP_POLICY_COMPARE to bcp_policy_compare.py")
    subprocess.run([sys.executable, _cmp],
                   env=dict(os.environ, SPLIT="test200", TREAT_CACHE=f"{OUT}/cache.json"))
