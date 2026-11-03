# Out of the Model Context, Into the Program: An Architecture for Deep-Search Agents

Code, configurations and per-question results for the paper. The paper itself is in
[`paper/paper.pdf`](paper/paper.pdf); its sources are in [`paper/`](paper/).

The architecture keeps retrieved documents out of the prompt and in the program: a deep-search agent runs in the REPL
loop of recursive language models (RLMs), each search returns whole documents into a Python variable (`DOCS`) that
persists across turns, the model reads them by writing code, only what the code prints enters the prompt, and a search
cap, not the context window, bounds how much the agent retrieves.

## What is here

| Folder | Contents |
|---|---|
| [`agents/document_repl/`](agents/document_repl/) | The architecture (the *document REPL agent*): harness `src/run_one.py`, REPL tools `src/tools.py` (`search`, `grep_docs`, `unread_docs`), prompts `src/prompts.py`, retrieval servers, judge |
| [`agents/snippet_repl/`](agents/snippet_repl/) | The *snippet REPL agent* baseline (RLM library, snippets printed, documents read through a sub-model) |
| [`agents/reference_agent/`](agents/reference_agent/) | Wrapper around the BrowseComp-Plus search agent and its in-prompt variants (document tool, cap, 25 snippets, trimmed or summarised history) |
| [`agents/fetch_then_explore/`](agents/fetch_then_explore/) | Our setup for running the released Fetch-then-Explore harness unmodified on BrowseComp-Plus (search adapter, network allowlist, launcher, converter) |
| [`agents/tool_store/`](agents/tool_store/) | The *tool-store agent* used in the ablations (same store, read through `read`/`grep` tools instead of code) |
| [`agents/hooks/`](agents/hooks/) | Startup hooks that set the ablation conditions (prompt only, no grep, search cap, 25 snippets) without changing the agents' code |
| [`configs/`](configs/) | `paths.example.env`: the machine-specific locations the launchers read |
| [`agents/shared/`](agents/shared/), [`retrieval/`](retrieval/) | Shared data loaders, grading and the snippet tokenizer; index builders and search servers (dense, BM25, late interaction) |
| [`runners/`](runners/) | Shell and Python entry points used for the runs |
| [`evaluation/`](evaluation/) | gpt-4.1 judge (benchmark template, majority of three votes) and environment handling |
| [`analysis/`](analysis/) | Scripts that produce the paper's tables and statistics (paired McNemar tests, Holm's correction) |
| [`figures/scripts/`](figures/scripts/) | Figure scripts; they write to `paper/figures/` |
| [`results/per_question/`](results/per_question/) | One CSV per configuration: question id, correct, gold recall, searches, prompt and token counts ([index](results/per_question/INDEX.md)) |
| [`results/tables/`](results/tables/) | Aggregate tables behind the paper's tables |

## Paper element to source

| Paper | Source |
|---|---|
| Table I (every configuration, 150 questions) | `analysis/paper_tables.py`, `results/tables/table1_main.csv`, `results/per_question/*.csv`; Fetch-then-Explore row: `analysis/fte_summary.py` |
| Table II (paired comparisons, Holm) | `analysis/holm_family.py` → `results/tables/holm_family.md`; Fetch-then-Explore row: `analysis/fte_summary.py` → `results/tables/fte_full830_summary.md` |
| Held-out comparison (680 questions) | `results/per_question/x7_dense_gpt5_getdoc_680.csv`, `fte_full830.csv`, `fd_cap20_830.csv`; `analysis/full830_summary.py`, `analysis/fte_summary.py` |
| Table III (full benchmarks) | `results/tables/full830_*`, `colbert830_*` |
| Tool-store ablation | `agents/tool_store/`; `analysis/x8_summary.py <run>` summarises a run (`results/tables/x8_toolstore.md`, `x9_toolstore_k25.md`) |
| Fig. 1–4 | `figures/scripts/make_fig{1_cards,2_example,3_mechanism,4_failures}.py` → `paper/figures/` |

## Run and file names

Scripts, run names and result files keep the names used during the research. They map to the paper as follows.

| In code and results | In the paper |
|---|---|
| Full-Doc RLM, `fd_*` runs | document REPL agent (ours) |
| budget | search cap |
| Snippet RLM, `snip_*` runs | snippet REPL agent |
| `x7_*` runs, `official_agent.py` | reference agent and its in-prompt variants |
| `x8`/`x9`, tool store | tool-store agent (ablation) |
| `fte_*` runs | Fetch-then-Explore |
| `x5_closed_book` | closed book |
| `x3_*` (evaluation) | shared judge and environment handling |

## What is not here, and why

- **No trajectories or run records.** BrowseComp-Plus is distributed encrypted so that its questions, answers and
  documents stay out of training data; agent trajectories quote them. Only derived, per-question outcomes are released.
- **No benchmark data.** Obtain BrowseComp-Plus, FRAMES and OBLIQ-Bench from their maintainers.
- **No credentials.** `evaluation/x3.env.example` lists the variables a run needs; `evaluation/x3_secret.sh` stores
  them outside the repository.

## Running

Copy `configs/paths.example.env` to `configs/paths.env` and fill in the locations; paths that still read `/path/to/...`
must be set to your own. The agents need the RLM library (alexzhang13/rlm at commit 854e688), `transformers`, a GPT-5
deployment (Azure OpenAI v1 API) and the retrieval server for the chosen retriever. Analysis and figure scripts read run
records under `data/x3/`, which are not part of the release; they document how each number was computed. Every run in the
paper used GPT-5 at high reasoning effort except where `results/tables` states otherwise.

## Credits

The document and snippet REPL agents build on the RLM library by Zhang, Kraska and Khattab and were developed in our
internal research repository; see each agent's `CHANGES_FROM_ORIGINAL.md`. The Fetch-then-Explore baseline runs the
harness by Liu et al. (liuqi6777/search_agent) unmodified.
