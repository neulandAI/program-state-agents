"""Startup hook: raise the openai clients' default max_retries (OPENAI_MAX_RETRIES, default 10) for HTTP 429."""
import os

try:
    import openai

    _N = int(os.environ.get("OPENAI_MAX_RETRIES", "10"))
    for _cls in (openai.OpenAI, openai.AzureOpenAI, openai.AsyncOpenAI, openai.AsyncAzureOpenAI):
        _orig = _cls.__init__

        def _init(self, *a, __orig=_orig, **kw):
            kw.setdefault("max_retries", _N)
            __orig(self, *a, **kw)

        _cls.__init__ = _init
except Exception as e:  # noqa: BLE001
    print(f"[retry429] hook not installed: {e}", flush=True)
