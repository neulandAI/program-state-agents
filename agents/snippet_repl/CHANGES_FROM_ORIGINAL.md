# Changes from the research code

`src/` and `experiments/` are copies of the research code that produced the paper's runs. Prompt text, tool
descriptions, defaults, environment variables and record fields are unchanged except where listed here.

## Files

| Research code | Here |
|---|---|
| `examples/browsecomp_eval_official_rlm.py` | `src/harness.py` |
| `scripts/run_bcp_snippet.py` | `src/run_browsecomp.py` |
| `examples/frames_eval_rlm.py` | `src/run_frames.py` |
| `scripts/run_obliq_mine.py` | `src/run_obliq.py` |
| `scripts/official_prompt_modules.py` | `src/prompt_modules.py` |
| `scripts/hooks/{retry429,azure_sampling,cap_ranking,run5d}/sitecustomize.py` | `src/hooks/<same>/sitecustomize.py` |
| `examples/{browsecomp_data,obliq_data,obliq_score_advice}.py` | `agents/shared/rlm_shared/` |
| `rag.agent._get_snippet_tokenizer` | `rlm_shared.tokenizer.snippet_tokenizer` |
| `scripts/run5d/{bcp_snip,bcp_snip2,frames_snip}.sh` | `experiments/` (same names) |
| `scripts/run5d/qwen_bcp_snip_queue2.sh` | `experiments/qwen_bcp_snip_queue.sh` |
| `scripts/run5d/obliq_gemini.sh` (`snip` arm) | `experiments/obliq_snip.sh` |
| `scripts/run_obliq_shard.sh` | `experiments/obliq_shard.sh` |
| `scripts/run_qwen_full.sh` | `experiments/full_set/bcp_snip_qwen_embed.sh` |
| `scripts/run_no_orchestrator.sh` | `experiments/full_set/bcp_snip_no_orchestrator.sh` |
| OBLIQ task and strategy texts | `configs/obliq/<task>/` |

## Changes

- **Tool name.** The search tool the model calls is `search` (in the research runs: `search_index`), in the tool
  specs, task text, prompt patches and search-cap wrappers. Record fields (`search_calls`, `topk_overrides`, ...) and
  environment variables keep their names.
- **Imports.** Modules are imported by their new names (`harness`, `rlm_shared.*`, ), and the
  `sys.path` entries are computed from `__file__` instead of a fixed repository root.
- **Paths.** The FRAMES question file is `$FRAMES_DATA_DIR/frames_rows.json`. Launchers source
  `experiments/_common.sh`, which loads `configs/paths.env`; the interpreter, `HF_HOME`, the credential file, the
  Azure deployment names, the working directory (`$RUNS_DIR`) and `NOORCH_QIDS_FILE` (question list of
  `bcp_snip_no_orchestrator.sh`) come from there. Arguments, caps, timeouts and behaviour variables are unchanged.
- **Comments and docstrings** are shortened; diagnostic and operations scripts (spend watchdogs, grading waiters, test
  probes) are not included.

## Notes

- The library merges per-model usage by model name, so `input_tokens`, `output_tokens` and `iterations` in a record
  can reflect only the sub-model when `llm_query` ran. Per-call token counts are in `RLM_USAGE_LOG` (azure_sampling
  hook) and in the trajectories.
- `reasoning_effort` reaches Azure only with the `azure_sampling` (or `run5d`) hook loaded. The `full_set/`
  launchers load no hook.
- The OBLIQ launchers do not set `OBLIQ_NO_FORMAT_GATE`, so the `Explanation:` / `Exact Answer:` gate is active there.
- `obliq_shard.sh` merges `${OUT}_s*.json`, which also matches the per-shard checkpoint files, so `all.json` holds
  every question twice; the printed means are unaffected.
