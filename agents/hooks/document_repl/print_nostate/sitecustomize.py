"""Startup hook: the document REPL agent reads documents from the prompt only (prompt-only reading control).

Each search prints the full text of new documents and returns only ids and scores; the REPL's DOCS holds no text;
grep_docs and unread_docs are removed; prompts and descriptions are adapted to say so; the initial search's
documents are appended to the question; the print cap is raised to PRINT_DOCS_MAX_CHARS. NOSTATE_CHAIN_HOOK names
another sitecustomize (the retry hook) to load first.
"""
import importlib.abc
import importlib.util
import os
import re
import sys

_RETRY = os.environ.get("NOSTATE_CHAIN_HOOK")
if _RETRY and os.path.exists(_RETRY):
    _spec = importlib.util.spec_from_file_location("_retry429_sitecustomize", _RETRY)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

_MAX = int(os.environ.get("PRINT_DOCS_MAX_CHARS", "2000000"))
_STATE = {"in_repl": False, "seed_docs": []}


def _need(s, old, new, where):
    if old not in s:
        raise RuntimeError(f"[nostate] expected text not found in {where}: {old[:60]!r}")
    return s.replace(old, new)


def _patch_prompts(mod):
    if getattr(mod, "_nostate", False):
        return
    mod.SOLO_SYSTEM_PROMPT = _need(mod.SOLO_SYSTEM_PROMPT,
        "- `DOCS`: the documents you have retrieved so far, keyed by docid. Already seeded with results from an initial search for your question.",
        "- `DOCS`: docid -> {{'docid', 'score'}} for every document retrieved so far, without text. In this run the full "
        "text of every newly retrieved document is printed into your conversation when it is retrieved (the initial "
        "search's documents follow the question) and is not stored in any variable.", "SOLO_SYSTEM_PROMPT")
    mod.SOLO_SYSTEM_PROMPT = _need(mod.SOLO_SYSTEM_PROMPT,
        "Read document text directly out of `DOCS` with ordinary Python — slicing, regex, `in`, splitting on lines — and draw your conclusions yourself.",
        "Read the documents in the printed output of your conversation and draw your conclusions yourself.", "SOLO_SYSTEM_PROMPT")
    mod.SOLO_SYSTEM_PROMPT = _need(mod.SOLO_SYSTEM_PROMPT,
        "REPL output over ~20K characters is truncated, so print short, targeted extracts rather than whole documents.",
        "Retrieved documents are printed in full.", "SOLO_SYSTEM_PROMPT")
    mod.SOLO_READ_SENTENCE = ("Retrieved documents are long web pages — tens of thousands of characters each — and are "
                              "printed in full when retrieved; read them there.")
    mod.USER_PROLOGUE = _need(mod.USER_PROLOGUE,
        "Your context is retrieved rather than given, so the REPL variables DOCS and SEARCH_LOG are what the RLM system prompt calls `context`: they start empty and grow as you retrieve.",
        "Your context is retrieved rather than given: retrieved documents are printed into your conversation, and the "
        "REPL variables DOCS (ids and scores only) and SEARCH_LOG record what was retrieved; they start empty and grow "
        "as you retrieve.", "USER_PROLOGUE")
    mod.LEAN_CLAUSE = _need(mod.LEAN_CLAUSE, "its results are in DOCS,",
                            "its documents are printed after the question, their ids are in DOCS,", "LEAN_CLAUSE")
    mod._nostate = True
    print("[nostate] prompts adapted: documents printed, not stored", flush=True)


def _rewrite_search_desc(d, k):
    head = (f"search(query: str) -> list[dict]. Perform a search on a knowledge source. Returns the top-{k} hits as "
            "{'docid', 'score'}. The COMPLETE text of every hit not retrieved before in this task is printed into the "
            "REPL output, so the documents enter your conversation in full; they are not stored in any variable "
            "(DOCS keeps only ids and scores). Results are appended to SEARCH_LOG.")
    m = re.search(r" The retriever is a dense semantic index.*?rather than a short keyword string\.", d, re.S)
    advice = m.group(0) if m else ""
    b = re.search(r" You have a hard budget of \d+ search calls.*?spend them on\.", d, re.S)
    if not b:
        raise RuntimeError("[nostate] budget sentence not found in the search description")
    return (head + advice + f" Each call prints '[search n/cap] k={k} -> m new documents' followed by the new "
            "documents' text." + b.group(0))


def _patch_tools(mod):
    tb = getattr(mod, "ToolBox", None)
    if tb is None or getattr(tb.search, "_nostate", False):
        return
    orig_search, orig_ct = tb.search, tb.custom_tools

    def search(self, query, k=None):
        if not hasattr(self, "_nostate_docs"):
            self._nostate_docs = {}
        before = set(self.DOCS.keys())
        results = orig_search(self, query, k) or []
        out = []
        for r in results:
            did = r.get("docid")
            if did not in before:
                text = (self.DOCS.get(did) or {}).get("text") or r.get("text") or ""
                if _STATE["in_repl"]:
                    print(f"\n=== document {did} ===\n{text}")
                else:
                    _STATE["seed_docs"].append((did, text))
            rec = {"docid": did, "score": r.get("score")}
            self._nostate_docs[did] = rec
            out.append(dict(rec))
        return out
    search._nostate = True

    def custom_tools(self):
        if not hasattr(self, "_nostate_docs"):
            self._nostate_docs = {}
        ct = orig_ct(self)
        for name in ("grep_docs", "unread_docs"):
            ct.pop(name, None)
        ct["search"] = dict(ct["search"], description=_rewrite_search_desc(str(ct["search"]["description"]), self.default_k))
        ct["DOCS"] = {"tool": self._nostate_docs,
                      "description": ("dict mapping docid -> {'docid', 'score'} for every document retrieved so far in "
                                      "this task, WITHOUT text: in this run document text is printed into your "
                                      "conversation when retrieved and is not stored.")}
        for key, v in ct.items():
            desc = str(v.get("description", ""))
            if "grep_docs" in desc or "unread_docs" in desc:
                raise RuntimeError(f"[nostate] {key} description still names grep_docs/unread_docs")
        return ct
    tb.search, tb.custom_tools = search, custom_tools
    print("[nostate] search prints documents and returns ids; REPL DOCS holds no text; grep_docs/unread_docs removed", flush=True)


def _patch_parsing(mod):
    f = getattr(mod, "format_iteration", None)
    if f is not None and f.__defaults__ and f.__defaults__[-1] != _MAX:
        f.__defaults__ = f.__defaults__[:-1] + (_MAX,)
        print(f"[nostate] print cap raised to {_MAX} characters", flush=True)


def _patch_rlm(mod):
    cls = getattr(mod, "RLM", None)
    if cls is None or getattr(cls.completion, "_nostate", False):
        return
    orig = cls.completion

    def completion(self, prompt, *a, **kw):
        if isinstance(prompt, dict) and "first_search" in prompt and _STATE["seed_docs"]:
            prompt = dict(prompt, note=("`context` lists the ids and scores of the first search, issued with the "
                                        "question verbatim; the documents themselves are printed after the question. "
                                        "Further retrieval is via search; DOCS and SEARCH_LOG grow."))
            docs = "\n\n".join(f"=== document {d} ===\n{t}" for d, t in _STATE["seed_docs"])
            kw["root_prompt"] = (kw.get("root_prompt") or "") + "\n\nDocuments returned by the initial search (full text):\n\n" + docs
            _STATE["seed_docs"] = []
        _STATE["in_repl"] = True
        return orig(self, prompt, *a, **kw)
    completion._nostate = True
    cls.completion = completion
    print("[nostate] initial-search documents appended to the question", flush=True)


_TARGETS = {"prompts": _patch_prompts, "tools": _patch_tools, "rlm.utils.parsing": _patch_parsing, "rlm.core.rlm": _patch_rlm}


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in _TARGETS:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        orig_exec = spec.loader.exec_module

        def exec_module(module, _orig=orig_exec, _name=name):
            _orig(module)
            _TARGETS[_name](module)
        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _Finder())
