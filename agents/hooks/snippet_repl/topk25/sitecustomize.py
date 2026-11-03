"""Startup hook: every snippet search returns SNIP_FIXED_TOPK hits (default 25), plus OpenAI-client retries on 429.

The top_k sent to /search is overridden, and the search tool's description is rewritten to state the fixed k.
"""
import importlib.abc
import importlib.util
import os
import sys

_K = int(os.environ.get("SNIP_FIXED_TOPK", "25"))

try:
    import openai
    _N = int(os.environ.get("OPENAI_MAX_RETRIES", "12"))
    for _cls in (openai.OpenAI, openai.AzureOpenAI, openai.AsyncOpenAI, openai.AsyncAzureOpenAI):
        _orig = _cls.__init__

        def _init(self, *a, __orig=_orig, **kw):
            kw.setdefault("max_retries", _N)
            __orig(self, *a, **kw)
        _cls.__init__ = _init
    print(f"[retry429] openai clients default to max_retries={_N}", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"[retry429] hook not installed: {e}", flush=True)

try:
    import requests
    _orig_post = requests.post

    def _post(url, *a, **kw):
        if isinstance(url, str) and url.rstrip("/").endswith("/search") and isinstance(kw.get("json"), dict):
            kw["json"] = dict(kw["json"], top_k=_K)
        return _orig_post(url, *a, **kw)
    requests.post = _post
    print(f"[topk_fixed] every search returns top_k={_K}", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"[topk_fixed] hook not installed: {e}", flush=True)

_OLD = "search(query: str, top_k: int = 5) -> "
_NEW = f"search(query: str, top_k: int = {_K}) -> "
_ADD = f" Every search returns the top {_K} hits; top_k is fixed at {_K} in this run."


def _patch_rlm(mod):
    cls = getattr(mod, "RLM", None)
    if cls is None or getattr(cls.__init__, "_topk_fixed", False):
        return
    orig = cls.__init__

    def __init__(self, *a, **kw):
        ct = kw.get("custom_tools")
        if isinstance(ct, dict) and isinstance(ct.get("search"), dict):
            d = str(ct["search"].get("description", ""))
            # Two wordings exist: the harness's own and LightOn's (LIGHTON_TOOL_DOCS=1), which already names top-k.
            if _OLD in d:
                d = d.replace(_OLD, _NEW)
            elif f"top-{_K} hits" not in d:
                raise RuntimeError("[topk_fixed] search description names neither wording; refusing to run")
            ct = dict(ct); ct["search"] = dict(ct["search"], description=d + _ADD)
            kw["custom_tools"] = ct
        orig(self, *a, **kw)
    __init__._topk_fixed = True
    cls.__init__ = __init__
    print("[topk_fixed] search description rewritten", flush=True)


_TARGETS = {"rlm.core.rlm": _patch_rlm}


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
