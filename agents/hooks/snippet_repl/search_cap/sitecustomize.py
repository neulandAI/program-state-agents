"""Startup hook: cap the snippet REPL agent at SNIP_SEARCH_CAP searches (default 10) per process.

Python runs sitecustomize at interpreter start when this directory is first on PYTHONPATH. Searches past the cap
raise, and the harness shows the error text to the model. Document fetches are not limited.
"""
import os

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
