#!/usr/bin/env python3
"""Reference agent: the BrowseComp-Plus search agent (search_agent/openai_client.py), unmodified, run through a wrapper.

The wrapper adds Azure OpenAI via the v1 endpoint, per-call usage in each run file, and the in-prompt variants
(--get-document, --search-cap, --k, --trim-tokens, --summarize-tokens). It stubs pyserini when its import fails, uses
sdpa attention in the encoder, and serialises FAISS searches (concurrent calls segfault).

Run: python agents/reference_agent/official_agent.py --output-dir data/x3/official/<run> [--limit N] with BCP_REPO
and RLM_DATA set; run records are not part of the release.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BCP = Path(os.environ.get("BCP_REPO", "/path/to/rlm/BrowseComp-Plus"))
DATA = Path(os.environ.get("RLM_DATA", "/path/to/rlm/data"))
SUBSET_TSV = ROOT / "data/x3/official/subset_queries.tsv"


def _stub_pyserini_if_needed():
    """Their searcher package imports pyserini (BM25, needs a JDK) at module level; stub it if that import fails."""
    try:
        from pyserini.search.lucene import LuceneSearcher  # noqa: F401
        return
    except Exception as e:  # noqa: BLE001
        print(f"[wrapper] pyserini unavailable ({type(e).__name__}); stubbing it, BM25 searcher not usable in this run", flush=True)
    import importlib.abc
    import importlib.machinery
    import types

    class _Placeholder:
        def __init__(self, *a, **k):
            raise RuntimeError("pyserini is stubbed in this wrapper; the BM25 searcher is not available")

    class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, name, path, target=None):
            if name == "pyserini" or name.startswith("pyserini."):
                return importlib.machinery.ModuleSpec(name, self, is_package=True)
            return None

        def create_module(self, spec):
            m = types.ModuleType(spec.name)
            m.__path__ = []
            m.__file__ = f"<stub {spec.name}>"
            m.__version__ = "0.0.0"

            def _getattr(attr, _name=spec.name):
                if attr.startswith("__"):
                    raise AttributeError(attr)
                return _Placeholder
            m.__getattr__ = _getattr
            return m

        def exec_module(self, module):
            pass

    for k in [k for k in sys.modules if k == "pyserini" or k.startswith("pyserini.")]:
        del sys.modules[k]
    sys.meta_path.insert(0, _Finder())


def _est_tokens(items) -> int:
    return sum(len(json.dumps(it, ensure_ascii=False, default=str)) for it in items) // 4


_TEXT_KEYS = ("content", "output", "arguments", "text")


def _est_text_tokens(items) -> int:
    """Characters/4 over the text the model reads, skipping ids and encrypted reasoning, which would inflate the
    estimate several-fold."""
    def walk(x, key=None):
        if isinstance(x, str):
            return len(x) if key in _TEXT_KEYS else 0
        if isinstance(x, dict):
            return sum(walk(v, k) for k, v in x.items() if k != "encrypted_content")
        if isinstance(x, list):
            return sum(walk(v, key) for v in x)
        if hasattr(x, "model_dump"):
            return walk(x.model_dump(mode="python"), key)
        return 0
    return walk(list(items)) // 4


def _trim(items: list, budget: int) -> None:
    """Clear the oldest function_call_output items (replace their output with a one-line notice) until the estimated
    size of the conversation is under the budget. The call records themselves stay, so the API's pairing holds."""
    cleared = "[earlier search result cleared to keep the prompt within budget]"
    if _est_tokens(items) <= budget:
        return
    for it in items:
        typ = it.get("type") if isinstance(it, dict) else getattr(it, "type", None)
        if typ == "function_call_output":
            out = it.get("output") if isinstance(it, dict) else getattr(it, "output", None)
            if out and out != cleared:
                if isinstance(it, dict):
                    it["output"] = cleared
                else:
                    it.output = cleared
                if _est_tokens(items) <= budget:
                    return


SUMMARY_TAG = "[summary of earlier search results]\n"
FOLDED = "[folded into the summary of earlier search results above]"
SUMMARY_PROMPT = (
    "A search agent is working on the task below. Its earlier search results must leave its context to save space, "
    "and your summary will replace them. Write a summary that keeps everything in them that could help with the task: "
    "candidate entities, names, titles, dates, numbers, places and quoted phrases, each with the docid it came from, and "
    "which leads were checked and ruled out. Do not try to answer the task yourself; report what the results say. "
    "Use at most {words} words.\n\n=== The agent's task ===\n{task}\n\n{prev}=== Search results to summarise "
    "(query, then result) ===\n{results}")


def _as_dict_get(it, key):
    return it.get(key) if isinstance(it, dict) else getattr(it, key, None)


def _set_output(it, val):
    if isinstance(it, dict):
        it["output"] = val
    else:
        it.output = val


def _summarize(items: list, budget: int, words: int, summarizer) -> dict | None:
    """Over budget, fold every tool result except the newest, with the summary so far, into one model-written summary.
    The summary takes the oldest folded slot and the others become a notice, so call pairing stays intact. Returns a
    record of the fold, or None."""
    if _est_text_tokens(items) <= budget:
        return None
    task = next((_as_dict_get(it, "content") for it in items if _as_dict_get(it, "role") == "user"), "")
    if isinstance(task, list):
        task = " ".join(str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in task)
    args_by_call = {_as_dict_get(it, "call_id"): _as_dict_get(it, "arguments") for it in items
                    if _as_dict_get(it, "type") == "function_call"}
    outs = [it for it in items if _as_dict_get(it, "type") == "function_call_output"]
    prev = next((it for it in outs if str(_as_dict_get(it, "output") or "").startswith(SUMMARY_TAG)), None)
    fold = [it for it in outs[:-1] if it is not prev and _as_dict_get(it, "output") != FOLDED]
    if not fold:
        return None
    results = "\n\n".join(f"--- call: {args_by_call.get(_as_dict_get(it, 'call_id'))}\n{_as_dict_get(it, 'output')}" for it in fold)
    prev_txt = (f"=== Summary so far (fold it into the new summary) ===\n{str(_as_dict_get(prev, 'output'))[len(SUMMARY_TAG):]}\n\n"
                if prev is not None else "")
    before = _est_text_tokens(items)
    summary = summarizer(SUMMARY_PROMPT.format(words=words, task=task, prev=prev_txt, results=results))
    if not summary or not summary.strip():
        # no usable summary: clear these results instead, as the trimmed variant does
        for it in fold:
            _set_output(it, FOLDED)
        return {"folded": len(fold), "before": before, "after": _est_text_tokens(items), "fallback_cleared": True}
    slot = prev if prev is not None else fold[0]
    _set_output(slot, SUMMARY_TAG + summary.strip())
    for it in fold:
        if it is not slot:
            _set_output(it, FOLDED)
    return {"folded": len(fold), "before": before, "after": _est_text_tokens(items), "fallback_cleared": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--num-threads", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="first N subset questions (smoke test)")
    ap.add_argument("--max-tokens", type=int, default=40000, help="their --max_tokens: output budget for the whole conversation")
    ap.add_argument("--reasoning-effort", default="high")
    ap.add_argument("--query-tsv", default=str(SUBSET_TSV))
    ap.add_argument("--get-document", action="store_true", help="register their get_document tool too (whole documents into the prompt) and use their QUERY_TEMPLATE")
    ap.add_argument("--trim-tokens", type=int, default=0, help="keep the conversation under this many tokens (estimate) by clearing the oldest tool results")
    ap.add_argument("--sdk-retries", type=int, default=12, help="OpenAI SDK retries per request (429 backoff)")
    ap.add_argument("--query-retries", type=int, default=10, help="their --max-retries: attempts per query")
    ap.add_argument("--k", type=int, default=5, help="their --k: hits per search (benchmark default 5)")
    ap.add_argument("--summarize-tokens", type=int, default=0, help="keep the conversation under this many tokens (estimate) by folding older tool results into a model-written summary (the summarisation counterpart of --trim-tokens)")
    ap.add_argument("--summary-words", type=int, default=500, help="length limit given to the summariser")
    ap.add_argument("--search-cap", type=int, default=0, help="at most this many searches per question (0 = no cap); further searches return a refusal the model reads, as the REPL agents' caps do")
    a = ap.parse_args()

    sys.path.insert(0, str(ROOT / "evaluation"))
    from x3_env import load_env
    load_env()
    os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]
    os.environ["OPENAI_BASE_URL"] = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/") + "/openai/v1/"
    os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
    model = os.environ["GPT5_DEPLOYMENT"]

    out_dir = Path(a.output_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    tsv = Path(a.query_tsv).resolve()
    if a.limit:
        lines = tsv.read_text(encoding="utf-8").splitlines()[: a.limit]
        tsv = out_dir / "_queries_subset.tsv"; tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    sys.path.insert(0, str(BCP)); sys.path.insert(0, str(BCP / "search_agent"))
    os.chdir(BCP)
    _stub_pyserini_if_needed()
    import openai_client as oc  # the official module, unmodified

    # tevatron defaults the encoder to flash_attention_2; use PyTorch sdpa instead (same computation)
    from tevatron.retriever.driver.encode import DenseModel
    _orig_load = DenseModel.load.__func__

    def _load(cls, *args, **kwargs):
        if kwargs.get("attn_implementation") in (None, "flash_attention_2"):
            kwargs["attn_implementation"] = "sdpa"
        return _orig_load(cls, *args, **kwargs)
    DenseModel.load = classmethod(_load)

    # FaissSearcher.search is not thread-safe: serialise it
    from searcher.searchers.faiss_searcher import FaissSearcher
    _search_lock = threading.Lock()
    _orig_search = FaissSearcher.search

    def _locked_search(self, *args, **kwargs):
        with _search_lock:
            return _orig_search(self, *args, **kwargs)
    FaissSearcher.search = _locked_search

    # per-call usage: wrap the client's responses.create, thread-local per worker
    tl = threading.local()
    orig_openai_cls = oc.openai.OpenAI

    class PatchedOpenAI(orig_openai_cls):
        def __init__(self, *args, **kwargs):
            # their client uses the SDK default of 2 retries; allow more for rate limits
            kwargs.setdefault("max_retries", a.sdk_retries)
            kwargs.setdefault("timeout", 1200)
            super().__init__(*args, **kwargs)
            orig_create = self.responses.create

            def create(**kw):
                t0 = time.time()
                if a.trim_tokens and isinstance(kw.get("input"), list):
                    _trim(kw["input"], a.trim_tokens)
                if a.summarize_tokens and isinstance(kw.get("input"), list):
                    def summarizer(prompt):
                        # same model and effort as the agent; recorded apart from the agent's calls (summary_calls)
                        t1 = time.time()
                        for attempt in range(12):
                            try:
                                rs = orig_create(model=kw.get("model"), input=[{"role": "user", "content": prompt}],
                                                 reasoning={"effort": a.reasoning_effort}, max_output_tokens=16000)
                                break
                            except (oc.openai.RateLimitError, oc.openai.APIConnectionError, oc.openai.APITimeoutError, oc.openai.InternalServerError) as e:
                                if attempt == 11:
                                    raise
                                wait = min(120, 5 * 2 ** attempt) + random.uniform(0, 3)
                                print(f"[wrapper] summary {type(e).__name__}, retry {attempt + 1} in {wait:.0f}s", flush=True)
                                time.sleep(wait)
                        us = getattr(rs, "usage", None)
                        getattr(tl, "summary_calls", None) is None and setattr(tl, "summary_calls", [])
                        tl.summary_calls.append({"input_tokens": getattr(us, "input_tokens", None), "output_tokens": getattr(us, "output_tokens", None),
                                                 "status": getattr(rs, "status", None), "s": round(time.time() - t1, 1)})
                        return getattr(rs, "output_text", "") or ""
                    ev = _summarize(kw["input"], a.summarize_tokens, a.summary_words, summarizer)
                    if ev:
                        getattr(tl, "folds", None) is None and setattr(tl, "folds", [])
                        tl.folds.append(ev)
                for attempt in range(12):
                    try:
                        r = orig_create(**kw)
                        break
                    except (oc.openai.RateLimitError, oc.openai.APIConnectionError, oc.openai.APITimeoutError, oc.openai.InternalServerError) as e:
                        if attempt == 11:
                            raise
                        wait = min(120, 5 * 2 ** attempt) + random.uniform(0, 3)
                        print(f"[wrapper] {type(e).__name__}, retry {attempt + 1} in {wait:.0f}s", flush=True)
                        time.sleep(wait)
                u = getattr(r, "usage", None)
                rec = {"input_tokens": getattr(u, "input_tokens", None), "output_tokens": getattr(u, "output_tokens", None),
                       "cached_tokens": getattr(getattr(u, "input_tokens_details", None), "cached_tokens", None),
                       "reasoning_tokens": getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", None),
                       "status": getattr(r, "status", None), "s": round(time.time() - t0, 1)}
                getattr(tl, "calls", None) is None and setattr(tl, "calls", [])
                tl.calls.append(rec)
                return r
            self.responses.create = create

    oc.openai.OpenAI = PatchedOpenAI

    orig_run = oc.run_conversation_with_tools

    def run_conversation_with_tools(*args, **kwargs):
        tl.calls = []
        tl.n_search = 0
        tl.summary_calls = []
        tl.folds = []
        return orig_run(*args, **kwargs)

    # search cap: the handler is shared by all threads, so the per-question count is thread-local
    if a.search_cap:
        _orig_search_tool = oc.SearchToolHandler._search

        def _capped_search(self, query):
            n = getattr(tl, "n_search", 0)
            if n >= a.search_cap:
                return json.dumps({"error": f"search budget exhausted: this question allows at most {a.search_cap} searches; answer from what you have"})
            tl.n_search = n + 1
            return _orig_search_tool(self, query)
        oc.SearchToolHandler._search = _capped_search
    oc.run_conversation_with_tools = run_conversation_with_tools

    orig_persist = oc._persist_response

    def persist(*args, query_id=None, **kwargs):
        calls = list(getattr(tl, "calls", []) or []); tl.calls = []
        scalls = list(getattr(tl, "summary_calls", []) or []); tl.summary_calls = []
        folds = list(getattr(tl, "folds", []) or []); tl.folds = []
        orig_persist(*args, query_id=query_id, **kwargs)
        out_dir_ = args[0]
        for p in sorted(Path(out_dir_).glob("run_*.json"), key=os.path.getmtime, reverse=True):
            try:
                d = json.load(open(p, encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if str(d.get("query_id")) == str(query_id) and "per_call_usage" not in d:
                d["per_call_usage"] = calls
                d["context_tokens_per_round"] = [c["input_tokens"] for c in calls]
                d["wall_time_s"] = round(sum(c["s"] for c in calls) + sum(c["s"] for c in scalls), 1)
                if a.summarize_tokens:
                    d["summary_calls"] = scalls
                    d["summary_folds"] = folds
                d["harness"] = {"name": "BrowseComp-Plus search_agent/openai_client.py", "repo": str(BCP), "endpoint": "Azure OpenAI v1-compatible",
                                "deployment": model, "max_tokens_total": a.max_tokens, "reasoning_effort": a.reasoning_effort,
                                "get_document": a.get_document, "trim_tokens": a.trim_tokens, "summarize_tokens": a.summarize_tokens, "summary_words": a.summary_words, "k": a.k, "search_cap": a.search_cap,
                                "n_search": getattr(tl, "n_search", None)}
                json.dump(d, open(p, "w", encoding="utf-8"), indent=2, default=str)
                break
    oc._persist_response = persist

    extra = (["--get-document", "--query-template", "QUERY_TEMPLATE"] if a.get_document else [])
    sys.argv = ["openai_client.py", "--query", str(tsv), "--model", model, "--reasoning-effort", a.reasoning_effort, *extra,
                "--max_tokens", str(a.max_tokens), "--output-dir", str(out_dir), "--num-threads", str(a.num_threads),
                "--max-retries", str(a.query_retries), "--k", str(a.k),
                "--searcher-type", "faiss", "--index-path", str(DATA / "indexes/qwen3-embedding-8b/corpus.shard*.pkl"),
                "--model-name", str(DATA / "models/Qwen3-Embedding-8B"), "--normalize",
                "--dataset-name", str(DATA / "browsecomp-plus-corpus")]
    print("official agent argv:", " ".join(sys.argv[1:]), flush=True)
    oc.main()


if __name__ == "__main__":
    main()
