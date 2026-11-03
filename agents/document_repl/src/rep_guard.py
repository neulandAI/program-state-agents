"""Opt-in (SUP_REP_GUARD=1): stop a generation that has fallen into a repetition loop. Not used in the paper's runs.

OpenAIClient.completion / acompletion are replaced by streaming versions. Every CHECK_EVERY new characters the last
WINDOW characters are tested; the stream is closed when token diversity falls below MIN_DIVERSITY or one non-trivial
line repeats MAX_LINE_REPEATS times, and the text so far is returned with NOTE appended. Unset = no effect.
"""
import collections
import os
import re
import types

WINDOW = 6000
CHECK_EVERY = 1500
MIN_DIVERSITY = 0.15
MAX_LINE_REPEATS = 25
NOTE = "\n\n[generation stopped: repetition loop detected]\n"
STOPPED = [0]


def looping(text: str) -> bool:
    tail = text[-WINDOW:]
    toks = re.findall(r"\S+", tail)
    if len(toks) >= 300 and len(set(toks)) / len(toks) < MIN_DIVERSITY:
        return True
    lines = [l.strip() for l in tail.split("\n")]
    lines = [l for l in lines if len(l) >= 4 and re.search(r"\w", l)]
    if lines:
        _, n = collections.Counter(lines).most_common(1)[0]
        if n >= MAX_LINE_REPEATS:
            return True
    return False


def _usage_obj(u):
    return types.SimpleNamespace(usage=u)


def install() -> None:
    if os.environ.get("SUP_REP_GUARD") != "1":
        return
    from rlm.clients import openai as oc

    def _prep(self, prompt, model):
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        model = model or self.model_name
        extra = {}
        if self.client.base_url == oc.DEFAULT_PRIME_INTELLECT_BASE_URL:
            extra["usage"] = {"include": True}
        return messages, model, oc._merge_extra_body(extra, self.sampling_args)

    def completion(self, prompt, model=None):
        messages, model, extra = _prep(self, prompt, model)
        stream = self.client.chat.completions.create(
            model=model, messages=messages, extra_body=extra, stream=True,
            stream_options={"include_usage": True, "continuous_usage_stats": True},
            **oc._normalize_sampling_args(self.sampling_args))
        text, seen, usage, last_check, n = [], [], None, 0, 0
        try:
            for ch in stream:
                if getattr(ch, "usage", None) is not None:
                    usage = ch.usage
                if ch.choices:
                    d = ch.choices[0].delta
                    piece = getattr(d, "content", None) or ""
                    think = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None) or ""
                    if piece:
                        text.append(piece)
                    if piece or think:      # loops can sit in the thinking too: watch both, return content
                        seen.append(piece or think); n += len(piece or think)
                if n - last_check >= CHECK_EVERY:
                    last_check = n
                    if looping("".join(seen)):
                        STOPPED[0] += 1
                        print(f"[repguard] repetition loop stopped at {n} chars", flush=True)
                        text.append(NOTE)
                        break
        finally:
            stream.close()
        if usage is None:
            raise ValueError("No usage data received. Tracking tokens not possible.")
        self._track_cost(_usage_obj(usage), model)
        return "".join(text)

    async def acompletion(self, prompt, model=None):
        messages, model, extra = _prep(self, prompt, model)
        stream = await self.async_client.chat.completions.create(
            model=model, messages=messages, extra_body=extra, stream=True,
            stream_options={"include_usage": True, "continuous_usage_stats": True},
            **oc._normalize_sampling_args(self.sampling_args))
        text, seen, usage, last_check, n = [], [], None, 0, 0
        try:
            async for ch in stream:
                if getattr(ch, "usage", None) is not None:
                    usage = ch.usage
                if ch.choices:
                    d = ch.choices[0].delta
                    piece = getattr(d, "content", None) or ""
                    think = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None) or ""
                    if piece:
                        text.append(piece)
                    if piece or think:      # loops can sit in the thinking too: watch both, return content
                        seen.append(piece or think); n += len(piece or think)
                if n - last_check >= CHECK_EVERY:
                    last_check = n
                    if looping("".join(seen)):
                        STOPPED[0] += 1
                        print(f"[repguard] repetition loop stopped at {n} chars", flush=True)
                        text.append(NOTE)
                        break
        finally:
            await stream.close()
        if usage is None:
            raise ValueError("No usage data received. Tracking tokens not possible.")
        self._track_cost(_usage_obj(usage), model)
        return "".join(text)

    oc.OpenAIClient.completion = completion
    oc.OpenAIClient.acompletion = acompletion
