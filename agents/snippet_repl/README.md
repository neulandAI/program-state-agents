# Snippet REPL agent

The snippet REPL agent is the paper's baseline built on the RLM library
([`alexzhang13/rlm`](https://github.com/alexzhang13/rlm), commit `854e688`). For each question the model works in a
persistent Python REPL and writes code in ```` ```repl ```` blocks. The REPL provides:

- `search(query, top_k=5)`: queries a retrieval server and returns a list of `{doc_id, score, text}`. Each `text` is
  cut to its first 512 tokens (Qwen/Qwen3-0.6B tokenizer).
- `get_document(doc_id)`: returns the full text of one document.
- `llm_query` / `llm_query_batched`: sub-model calls, used to read long text without printing it.
- `answer`: a dict. The run ends when the model sets `answer["content"]` and `answer["ready"] = True`.

The library is not edited. `src/harness.py` patches it at import time: it adapts the system prompt from a `context`
variable to retrieval tools, adds the tools and the task text, and enforces an `Explanation:` / `Exact Answer:`
answer format. Every prompt patch asserts on its anchor text, so a change upstream raises an error.

## Layout

| Path | Contents |
|---|---|
| `src/harness.py` | Agent and BrowseComp-Plus entry point (`run_one`, `main`) |
| `src/run_browsecomp.py` | BrowseComp-Plus runner with opt-in switches for local models (search cap, fence tolerance, timeouts) |
| `src/run_frames.py` | FRAMES runner (questions, judge, BM25 retriever sentence, search cap) |
| `src/run_obliq.py` | OBLIQ-Bench runner (ranking task, NDCG scoring, strategy block) |
| `src/prompt_modules.py` | Splits the patched system prompt into named paragraphs so that one can be replaced |
| `src/hooks/` | `sitecustomize.py` startup hooks: `retry429`, `azure_sampling` (sends `reasoning_effort` to Azure, logs per-call tokens), `cap_ranking`, and `run5d` (loads all three) |
| `experiments/` | Launchers, one per benchmark and model (see below) |
| `configs/obliq/` | OBLIQ task and strategy texts used by `obliq_snip.sh` |

## Requirements

- Python 3.12 with `openai`, `requests`, `datasets` and `transformers`, plus the `rlm` checkout at `854e688` on
  `PYTHONPATH` (not installed).
- `agents/shared` (package `rlm_shared`: data loading and grading).
- The `transformers` package, for the Qwen/Qwen3-0.6B tokenizer (`rlm_shared.tokenizer`) used when
  `SEARCH_SNIPPET_TOKENS` is set.
- A retrieval server that speaks the rlm wire format: `POST /search {query, top_k}` → `{results: [{doc_id, score,
  text}]}`, `POST /document {doc_id}`, `GET /health`.
- Model access: Azure OpenAI (`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_MODEL`) or a local
  OpenAI-compatible server (`LOCAL_LLM_BASE_URL`, `LOCAL_LLM_MODEL`).

The launchers read machine settings from `<repo>/configs/paths.env`: `PYTHON`, `RLMLIB_PATH`, `HF_HOME`, `RUNS_DIR`,
`AZ_ENV` (a file that defines `AZ_EP` and `AZ_KEY`), `AZURE_DEPLOYMENT_GPT5`, `AZURE_DEPLOYMENT_GPT52`,
`FRAMES_DATA_DIR` and `OBLIQ_BENCH_ROOT`.

## Running

One BrowseComp-Plus question (q0) with GPT-5 on the BM25 server at :8044. The arguments are model, retriever, output
directory, workers, first question and last question (exclusive):

```bash
agents/snippet_repl/experiments/bcp_snip.sh gpt5 http://localhost:8044 /path/to/runs/try_bcp 1 0 1
# -> q0.json, q0.log, traj/ (trajectories), usage.jsonl (per-call tokens)
```

The harness can also be called directly:

```bash
PYTHONPATH=/path/to/rlm:agents/shared python agents/snippet_repl/src/harness.py --start 0 --end 1 --out /path/to/q0
```

| Launcher | Benchmark and setting |
|---|---|
| `experiments/bcp_snip.sh` | BrowseComp-Plus, GPT-5 or Qwen3.6, any retriever URL |
| `experiments/bcp_snip2.sh` | BrowseComp-Plus, Qwen3.6 settings with the ```` ```repl ```` cue, prose-final submission, 50 turns |
| `experiments/qwen_bcp_snip_queue.sh` | `bcp_snip2.sh qwen` on BM25, dense and ColBERT, one after another |
| `experiments/full_set/bcp_snip_qwen_embed.sh` | BrowseComp-Plus, GPT-5 on Qwen3-Embedding-8B, all 830 questions |
| `experiments/full_set/bcp_snip_no_orchestrator.sh` | BrowseComp-Plus, GPT-5 on Reason-ModernColBERT, a given list of questions |
| `experiments/frames_snip.sh` | FRAMES, GPT-5 or Qwen3.6, BM25 over Wikipedia |
| `experiments/obliq_snip.sh` → `obliq_shard.sh` | OBLIQ-Bench (math, twitter, congress), GPT-5.2 on Gemini-Embedding-2 |

Each launcher writes one JSON record per question with the prediction, the in-run `correct` field, tool-call counts,
`gold_recall` / `evid_recall` and latency. Grading is separate (`agents/shared/rlm_shared/grading/`). Token counts
come from `usage.jsonl` (written by the `azure_sampling` hook when `RLM_USAGE_LOG` is set) or from the trajectories.

## Main settings

| Setting | Variable / flag | Default |
|---|---|---|
| Retrieval server | `EMBEDDING_SERVER_URL` | `http://localhost:8024` |
| Results per search (a default, not a cap) | `OFFICIAL_TOP_K` | 5 |
| Snippet length | `SEARCH_SNIPPET_TOKENS` (tokens), else `SEARCH_SNIPPET_LEN` (characters) | unset, 500 characters |
| REPL output ceiling | `RLM_MAX_OUTPUT_CHARS` | 60,000 characters |
| Orchestrator addendum of the library prompt | `ORCHESTRATOR` | 1 (launchers set 0) |
| LightOn tool descriptions and task text | `LIGHTON_TOOL_DOCS`, `LIGHTON_TASK_TEXT` | off (launchers set 1) |
| Dense-retriever sentence | `RETRIEVER_LINE` | on |
| Root and sub-model reasoning effort | `AZURE_OPENAI_REASONING_EFFORT`, `SUB_LM_REASONING_EFFORT` | high, medium |
| Sub-model deployment | `SUB_LM_MODEL` | the root model |
| Turn limit, recursion depth | `--max-iterations`, `--max-depth` | 100, 1 |
| History compaction | `--compaction` | off |
| Search cap | `BCP_SEARCH_CAP`, `FRAMES_SEARCH_CAP` | none |
| Trajectory directory | `RLM_TRAJ_DIR` | `logs/rlm_traj` |

`run_browsecomp.py`, `run_frames.py` and `run_obliq.py` list their own opt-in switches in their module docstrings.

## Credits

This agent runs the Recursive Language Model library by Zhang, Kraska and Khattab
([`alexzhang13/rlm`](https://github.com/alexzhang13/rlm)) unmodified, with all changes applied as runtime patches.
The LightOn tool descriptions and task text follow the BrowseComp-Plus search agent. Changes relative to the research
code that produced the paper's runs are listed in [CHANGES_FROM_ORIGINAL.md](CHANGES_FROM_ORIGINAL.md).
