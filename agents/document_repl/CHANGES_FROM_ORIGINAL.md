# Changes relative to the RLM library

The agent uses the RLM library (github.com/alexzhang13/rlm, commit 854e688) unmodified: `rlm.RLM` with
`environment="local"`, `depth=0`, `max_depth=1`. Everything below is added by `src/`.

## Added

- **Tools in the REPL** (`tools.py`): `search(query, k)` returns whole documents (design B) and merges them into
  `DOCS`; a snippet-only hit never replaces a full text. `search_many`, `grep_docs` (regex over `DOCS`) and
  `unread_docs` (documents never accessed) cost no search. `SEARCH_LOG` records every call.
- **Search cap** (`--max-searches`): enforced in `ToolBox.search`; calls past the cap return `[]` and print a
  refusal. A failed search still uses one unit of the cap.
- **k**: chosen by the model per call, clamped to 1..`MAX_K` = 50; `--default-k` applies when none is given;
  `--fixed-k` pins it (OBLIQ-Bench runs: 20).
- **Seed context** (`--seed-context`): `search(question)` runs before turn 1 and counts as search 1.
- **Prompts** (`prompts.py`): the BrowseComp-Plus query template, a system prompt without sub-call tools
  (`--solo-prompt`), prologue clauses (`--lean-prologue` and others); the library's orchestrator addendum is dropped
  with `--no-orchestrator`.
- **Grounding gate** (`--grounding-gate`): once per run, a submission whose `Exact Answer:` string occurs in no
  document in `DOCS` is held with a notice; the second submission is accepted.
- **Sandbox** (`run_one.harden_sandbox`): REPL imports of network, process and introspection modules are refused,
  `open` is refused, and hostname resolution is limited to the model endpoints and loopback. `scan_trajectory` flags
  model code with introspection, network or filesystem tokens, and `judge.py` scores flagged runs as incorrect.
- **Server-side counts**: every search carries a per-run tag; the retrieval servers keep their own counts and docids
  per tag, which the run record stores and the judge uses for recall.
- **Opt-in switches** for Qwen3.6: `SUP_FENCE_TOLERANT`, `SUP_MAX_BLOCKS`, `SUP_PRESENCE_PENALTY`; `SUP_REP_GUARD`.
- **BM25 wording** (`run_bm25.py`): the sentences that describe a dense index are replaced for BM25 runs.

## Behaviour on the stock library

- `--no-delegation` is passed as an environment keyword the library stores but does not read, so `llm_query` and
  the other sub-call functions stay installed; only the prompt asks the model to read by itself.
- On Azure, `reasoning_effort` is sent only when the `azure_sampling` startup hook is loaded (`run5d`).
- After the last turn the library makes one more call that forces an answer; it is logged as an iteration, so a run
  that hits the 30-turn cap records `n_iterations` = 31.
- `served_models` is `None` (the library does not expose the served model).
- A run that hits `--max-timeout` is recorded with status `error` and an empty answer.
- `prompt_id` hashes the query template, context note and user prologue only.
- The search description mentions a `read_docs` function that is not installed.
