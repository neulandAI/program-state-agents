"""Machine-specific locations, each read from an environment variable.

Python code calls `path("NAME")`; shell launchers use `${NAME:?}`. A missing variable raises an error naming it.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

VARIABLES = {
    "RLMLIB_PATH": "checkout of github.com/alexzhang13/rlm at commit 854e688 (the agent loop of the snippet and document REPL agents)",
    "FULLDOC_ROOT": "working directory of the document REPL agent: holds data/prepared (corpus, queries, qrels), data/indexes, data/models",
    "OBLIQ_BENCH_ROOT": "checkout of OBLIQ-Bench data: <task>/<subset>/queries+qrels/{queries.jsonl,qrels.tsv}",
    "FRAMES_DATA_DIR": "FRAMES question files (test.tsv, frames_rows.json) and the BM25 index of Wikipedia 20230601.en",
    "BCP_BM25_INDEX_DIR": "BM25 index of the BrowseComp-Plus corpus (built by retrieval/bcp_build_bm25_default.py)",
    "RUNS_DIR": "where run outputs (per-question records, trajectories, grades) are written",
    "AZ_ENV": "file defining AZ_EP (Azure OpenAI endpoint) and AZ_KEY (API key); never committed",
    "AZURE_DEPLOYMENT_GPT5": "Azure deployment serving gpt-5 (root model of the GPT-5 rows)",
    "AZURE_DEPLOYMENT_GPT52": "Azure deployment serving gpt-5.2 (OBLIQ rows)",
    "AZURE_DEPLOYMENT_JUDGE": "Azure deployment serving gpt-4.1 (the judge of every BrowseComp+ and FRAMES row)",
    "LOGS_ROOT": "root of the original run logs (logs/ of the original repository)",
    "ORIGINAL_REPO_ROOT": "checkout of the original (internal) research repository; only the re-judging and reporting "
                          "scripts read archived results from it",
    "RETRIEVAL_EVAL_ROOT": "checkout holding scripts/retrieval_eval.py (the PLAID build-and-evaluate script, not part "
                           "of this repository); retrieval/build_bc_rmcb_dl8192.sh runs there and writes "
                           "logs/ and .pylate_idx/ into it",
    "GCP_VERTEX_ENV": "file in .env format holding the Vertex AI service-account JSON for Gemini-Embedding-2 (variable "
                      "name as in retrieval/gemini/vertex.py); read only when it is not already in the environment; never committed",
}


def path(name: str) -> Path:
    """The location configured in environment variable `name`; raises if it is not set."""
    if name not in VARIABLES:
        raise KeyError(f"unknown path variable {name!r}; known: {', '.join(VARIABLES)}")
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"environment variable {name} is not set: {VARIABLES[name]}. "
                           f"Copy configs/paths.example.env, fill it in, and source it.")
    return Path(v).expanduser()


def setting(name: str) -> str:
    """A non-path setting from the same table (deployment names)."""
    return str(path(name))
