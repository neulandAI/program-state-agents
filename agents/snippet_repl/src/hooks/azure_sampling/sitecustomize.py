"""Startup hook: make rlm's AzureOpenAIClient send its sampling_args (e.g. reasoning_effort).

The library's Azure client builds requests from model and messages only, so reasoning_effort was dropped.
RLM_USAGE_LOG=<file> also appends one JSON line per Azure call with prompt, cached, completion and reasoning tokens.
"""
import json as _json
import os as _os
import time as _time


def _log_usage(model, response):
    path = _os.environ.get("RLM_USAGE_LOG")
    if not path:
        return
    try:
        u = response.usage
        pd = getattr(u, "prompt_tokens_details", None); cd = getattr(u, "completion_tokens_details", None)
        rec = {"t": round(_time.time(), 1), "pid": _os.getpid(), "model": model, "prompt": u.prompt_tokens,
               "cached": getattr(pd, "cached_tokens", 0) or 0, "completion": u.completion_tokens,
               "reasoning": getattr(cd, "reasoning_tokens", 0) or 0}
        with open(path, "a") as f:
            f.write(_json.dumps(rec) + "\n")
    except Exception:  # noqa: BLE001  logging must never break a run
        pass


try:
    from rlm.clients import azure_openai as _AZ
    from rlm.clients.openai import _normalize_sampling_args as _norm

    def _completion(self, prompt, model=None):
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for Azure OpenAI client.")
        response = self.client.chat.completions.create(model=model, messages=messages,
                                                       **_norm(getattr(self, "sampling_args", {}) or {}))
        self._track_cost(response, model)
        _log_usage(model, response)
        return response.choices[0].message.content

    async def _acompletion(self, prompt, model=None):
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for Azure OpenAI client.")
        response = await self.async_client.chat.completions.create(model=model, messages=messages,
                                                                   **_norm(getattr(self, "sampling_args", {}) or {}))
        self._track_cost(response, model)
        _log_usage(model, response)
        return response.choices[0].message.content

    _AZ.AzureOpenAIClient.completion = _completion
    _AZ.AzureOpenAIClient.acompletion = _acompletion
except Exception as _e:  # noqa: BLE001
    print(f"[azure-sampling] hook not installed: {_e}", flush=True)
