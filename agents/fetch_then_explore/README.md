# Fetch-then-Explore baseline

Fetch-then-Explore (Liu et al., [arXiv 2608.02097](https://arxiv.org/abs/2608.02097)) keeps fetched documents in a
per-question workspace and reads them with `grep` and `read` tools; search results enter the prompt. We ran its released
harness, [liuqi6777/search_agent](https://github.com/liuqi6777/search_agent) at commit `51ceadc`, **without modifying
its code**, on all 830 BrowseComp-Plus questions (paper: Table I, Table II, Section V-A).

| Setting | Value |
|---|---|
| Model | GPT-5, high reasoning effort |
| Tool profile | `fetch_explore` (`search`, `fetch`, `grep`, `read`, `list_fetched`) |
| Retrieval | the same exact dense retriever as every dense run in the paper (`runners/dense_server.py`, Qwen3-Embedding-8B); 5 results per search with 512-token snippets, as in the harness's BrowseComp-Plus setting |
| Budget | no search cap, 300 turns, 150 minutes per question |
| Network | blocked except for the model endpoint and loopback (startup hook) |
| Judge | gpt-4.1, the benchmark's template, majority of three votes (`evaluation/x3_judge.py`), as for every run in the paper |

## Files

| File | Role |
|---|---|
| `fte_search_adapter.py` | Serves the harness's local-search API (`/search`, `/get_document`) over our dense retriever |
| `hooks/sitecustomize.py` | Startup hook on `PYTHONPATH`: network allowlist, GPT-5 parameter compatibility, deployment pinned, served model logged per call |
| `prepare_fte_data.py` | Writes the questions in the harness's input format from a local decrypted copy of BrowseComp-Plus (never committed) |
| `run_fte_bcp.sh` | Launcher: preflight check that the deployment serves GPT-5, the run, a second pass for questions that ended in an API error, conversion and judging |
| `fte_to_x3.py` | Converts the harness's `predictions.jsonl` into our per-question records |
| `../../analysis/fte_summary.py` | The numbers the paper reports (Table I row, held-out comparison) → `results/tables/fte_full830_summary.md` |

Per-question outcomes: `results/per_question/fte_full830.csv`.

## Running

Clone the harness into `third_party/fte_search_agent` at commit `51ceadc`, install it in its own virtual environment,
and set `FTE_PYTHON` to that environment's Python. Start `runners/dense_server.py` on port 8765, set the Azure variables
with `evaluation/x3_secret.sh`, then:

```bash
agents/fetch_then_explore/run_fte_bcp.sh full830 bcp830 10 300
```
