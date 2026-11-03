"""Startup hook: remove the grep_docs tool and its mentions from the document REPL agent's tool descriptions.

The model can still match strings over DOCS in its own code. NO_GREP_CHAIN_HOOK names another sitecustomize (the
retry hook) to load first.
"""
import importlib.abc
import importlib.util
import os
import sys

_RETRY = os.environ.get("NO_GREP_CHAIN_HOOK")
if _RETRY and os.path.exists(_RETRY):
    _spec = importlib.util.spec_from_file_location("_retry429_sitecustomize", _RETRY)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

_REWRITES = [
    ("the tools are ordinary names (search, grep_docs, unread_docs,", "the tools are ordinary names (search, unread_docs,"),
    ("Hits at ranks you have not read are still in DOCS, where grep_docs (for a literal string from the question: a year, "
     "a number, a quoted phrase, a distinctive term) and unread_docs reach them without another call.",
     "Hits at ranks you have not read are still in DOCS, where unread_docs reaches them without another call."),
]


def _patch_tools(mod):
    tb = getattr(mod, "ToolBox", None)
    if tb is None or getattr(tb.custom_tools, "_no_grep", False):
        return
    orig_desc, orig_ct = tb.descriptions, tb.custom_tools

    def descriptions(self):
        d = orig_desc(self)
        s = d.get("search", "")
        for old, new in _REWRITES:
            s = s.replace(old, new)
        if "grep_docs" in s:
            raise RuntimeError("[no_grep] the search description still names grep_docs; refusing to run")
        d["search"] = s
        return d

    def custom_tools(self):
        ct = orig_ct(self)
        ct.pop("grep_docs", None)
        leftover = [k for k, v in ct.items() if "grep_docs" in str(v.get("description", ""))]
        if leftover:
            raise RuntimeError(f"[no_grep] grep_docs still named in the descriptions of {leftover}")
        return ct
    descriptions._no_grep = custom_tools._no_grep = True
    tb.descriptions, tb.custom_tools = descriptions, custom_tools
    print("[no_grep] grep_docs removed from the REPL tools and descriptions", flush=True)


_TARGETS = {"tools": _patch_tools}


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
