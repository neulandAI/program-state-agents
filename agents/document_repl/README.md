# Document REPL agent

The architecture of the paper. The agent runs in the REPL loop of the RLM library: every search returns whole
documents into the Python dict `DOCS`, which persists across turns; the model reads them by writing code, only what
its code prints enters the prompt, and a search cap bounds how much it retrieves.

## Layout

| Path | Contents |
|---|---|
| `src/run_one.py` | Per-question harness: arguments, sandbox, grounding gate, RLM construction, run record |
| `src/tools.py` | `ToolBox`: `search` (cap, k, merge into `DOCS`), `search_many`, `grep_docs`, `unread_docs`, tool descriptions |
| `src/prompts.py` | Query template, system prompt (`--solo-prompt`), prologue clauses; `GEPA_PROMPTS` override hook |
| `src/run_experiment.py` | Batch driver: one subprocess per question, resumable, writes `_provenance.json` (`src/provenance.py`) |
| `src/run_bm25.py` | Runs `run_one.py` with the retriever description rewritten for BM25 |
| `src/split_runner.py`, `src/bcp_qwen_dense.py`, `src/bcp_bm25_split.py` | Drivers of the Qwen3.6 BrowseComp-Plus runs (seed prompt, result cache) |
| `src/fence_tolerant.py`, `src/rep_guard.py` | Opt-in switches for Qwen3.6 (`SUP_FENCE_TOLERANT`, `SUP_MAX_BLOCKS`; `SUP_REP_GUARD`, unused) |
| `src/retrieval_server.py`, `src/encoder.py` | Dense retrieval server (Qwen3-Embedding-8B, exact cosine) |
| `src/retrieval_server_colbert.py`, `src/hybrid.py` | Late-interaction server (Reason-ModernColBERT, PLAID); optional RRF with BM25 |
| `src/obliq_adapter.py` | This agent's retrieval API in front of an OBLIQ-Bench search server |
| `src/prepare_data.py` | Decrypts BrowseComp-Plus into `$FULLDOC_ROOT/data/prepared/` |
| `src/judge.py`, `src/grade_local.py`, `src/score_obliq.py` | gpt-4.1 judge, local interim grader, OBLIQ NDCG scorer |
| `experiments/` | Launchers of the runs (BrowseComp-Plus, FRAMES, OBLIQ-Bench) |

## Setup

- Python 3.12 with `openai`, `python-dotenv`, `requests`; the servers also need `torch`, `transformers`, `numpy`,
  and `bm25s`/`PyStemmer` for BM25; the late-interaction server runs in a separate `pylate` environment.
- The RLM library (github.com/alexzhang13/rlm, commit 854e688) on `PYTHONPATH`, through `RLMLIB_PATH`.
- `configs/paths.env` at the repository root, sourced by the launchers, defining `RLMLIB_PATH`, `FULLDOC_ROOT` (data,
  indexes and models), `RUNS_DIR`, `PYTHON`, `AZ_ENV` (a file with `AZ_EP` and `AZ_KEY`), `AZURE_DEPLOYMENT_GPT5`,
  and as needed `FRAMES_DATA_DIR`, `BCP_BM25_INDEX_DIR`, `RLM_DATA_DIR`, `HOOKS_DIR`, `AZURE2_API_VERSION`.
- Data: place the BrowseComp-Plus parquet files under `$FULLDOC_ROOT/data/` and run `python src/prepare_data.py`.

## Running

Start the retrieval server for the chosen retriever (`src/retrieval_server.py --port 8765` for dense), then:

```bash
cd "$FULLDOC_ROOT"
PYTHONPATH=$RLMLIB_PATH $PYTHON /path/to/agents/document_repl/src/run_one.py --query-id 1 --out $RUNS_DIR/demo/1.json \
  --design B --server http://127.0.0.1:8765 --root-model gpt-5 --sub-model gpt-5 \
  --root-deployment $AZURE_DEPLOYMENT_GPT5 --sub-deployment $AZURE_DEPLOYMENT_GPT5 \
  --max-iterations 30 --max-searches 10 --seed-context --compaction --grounding-gate \
  --lean-prologue --no-delegation --solo-prompt --no-orchestrator --save-trajectory
```

`src/run_experiment.py --name <run> ...` runs a set of questions with the same flags, and `src/judge.py --run-dir
<dir> --votes 3` grades a run directory. `run_one.py` reads Azure credentials from `AZURE2_ENDPOINT`,
`AZURE2_API_KEY` and `AZURE2_API_VERSION`, and otherwise falls back to the library's `AZURE_OPENAI_*` variables.

Launchers in `experiments/`:

| Run | Launcher |
|---|---|
| BrowseComp-Plus, GPT-5, BM25 | `experiments/bcp_gpt5_bm25.sh` |
| BrowseComp-Plus, GPT-5, late interaction | `experiments/bcp_gpt5_final830/launch_high*.sh`, `experiments/bcp_gpt5_colbert.sh` |
| BrowseComp-Plus, Qwen3.6, dense / late interaction / BM25 | `python src/bcp_qwen_dense.py`, `python src/split_runner.py --seed-split <split>`, `python src/bcp_bm25_split.py` |
| FRAMES, GPT-5 / Qwen3.6 | `experiments/frames_gpt5.sh <out> <workers> 0 824`, `BM25_PROMPT=1 experiments/frames_qwen.sh` |
| OBLIQ-Bench | `experiments/obliq_fd.sh fd <task> <ids.json> <out> 5` |

Ablation conditions (k = 5, prompt-only reading, no grep) are set by the startup hooks in
[`../hooks/document_repl/`](../hooks/document_repl/). LLM sampling is not deterministic, so a re-run reproduces the
configuration, not the per-question outputs.

## Credits

This agent builds on the RLM library by Zhang, Kraska and Khattab
([github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm)); the library is not modified.
[CHANGES_FROM_ORIGINAL.md](CHANGES_FROM_ORIGINAL.md) lists what this agent adds and how it behaves on the library.
