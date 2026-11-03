"""Startup hook: print every new document in full after each search (documents enter the prompt).

Wraps tools.ToolBox.search and raises the library's per-block print cap to PRINT_DOCS_MAX_CHARS (default
2,000,000). PRINT_DOCS_CHAIN_HOOK names another sitecustomize (the retry hook) to load first.
"""
import importlib.abc
import importlib.util
import os
import sys

_RETRY = os.environ.get("PRINT_DOCS_CHAIN_HOOK")
if _RETRY and os.path.exists(_RETRY):
    _spec = importlib.util.spec_from_file_location("_retry429_sitecustomize", _RETRY)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

_MAX = int(os.environ.get("PRINT_DOCS_MAX_CHARS", "2000000"))


def _patch_tools(mod):
    tb = getattr(mod, "ToolBox", None)
    if tb is None or getattr(tb.search, "_print_docs", False):
        return
    orig = tb.search

    def search(self, query, k=None):
        before = set(self.DOCS.keys())
        results = orig(self, query, k)
        for r in results or []:
            did = r.get("docid")
            if did in before:
                continue
            text = (self.DOCS.get(did) or {}).get("text") or r.get("text") or ""
            print(f"\n=== document {did} ===\n{text}")
        return results
    search._print_docs = True
    tb.search = search
    print("[print_docs] ToolBox.search now prints new documents in full", flush=True)


def _patch_parsing(mod):
    f = getattr(mod, "format_iteration", None)
    if f is not None and f.__defaults__ and f.__defaults__[-1] != _MAX:
        f.__defaults__ = f.__defaults__[:-1] + (_MAX,)
        print(f"[print_docs] print cap raised to {_MAX} characters", flush=True)


_TARGETS = {"tools": _patch_tools, "rlm.utils.parsing": _patch_parsing}


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
