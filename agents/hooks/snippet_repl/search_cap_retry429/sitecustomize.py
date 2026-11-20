"""Startup hook: the search cap plus OpenAI-client retries on HTTP 429.

Python loads only the first sitecustomize on PYTHONPATH, so hooks that must coexist are combined in one file.
"""
import os

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
    _CAP = int(os.environ.get("SNIP_SEARCH_CAP", "10"))
    _count = {"n": 0}
    _orig_post = requests.post

    def _post(url, *a, **kw):
        if isinstance(url, str) and url.rstrip("/").endswith("/search"):
            if _count["n"] >= _CAP:
                raise RuntimeError(f"search budget exhausted: this question allows at most {_CAP} searches; answer from what you have")
            _count["n"] += 1
        return _orig_post(url, *a, **kw)
    requests.post = _post
    print(f"[search_cap] hook installed: at most {_CAP} searches per question", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"[search_cap] hook not installed: {e}", flush=True)
