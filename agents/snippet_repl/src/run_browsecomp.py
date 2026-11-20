"""BrowseComp-Plus runner for the snippet REPL agent: harness.py plus opt-in patches for local models.

Unset, every switch below is a no-op and the run is the harness's own:
  LOCAL_LM_TIMEOUT        per-call timeout for a local server (default 1800 s; the library's is 300 s)
  QWEN_PRESENCE_PENALTY   presence_penalty for root and sub calls
  QWEN_MAX_TOKENS=N       output-token cap per call, root and sub calls
  QWEN_FENCE_TOLERANT=1   run a ```python or bare ``` block when no ```repl block exists, if it parses and acts
  QWEN_REPL_CUE=1         one sentence in TURN_RULE naming the ```repl fence
  QWEN_PROSE_FINAL=1      a code-less turn with a non-empty "Exact Answer:" line is submitted as the answer
  BCP_SEARCH_CAP=N        at most N search calls; further calls return an error hit, and the prompt states the budget
Arguments are the harness's (--start/--end/--max-iterations/--max-depth/--compaction/--out).
"""
import ast
import os
import re
import sys

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_SRC_DIR)), "shared"))
sys.path.insert(0, _SRC_DIR)
import rlm as _rlm  # noqa: E402

_LM_TIMEOUT = float(os.environ.get("LOCAL_LM_TIMEOUT", "1800"))
_orig_rlm_init = _rlm.RLM.__init__


def _rlm_init_patched(self, *a, **kw):
    bk = kw.get("backend_kwargs")
    if isinstance(bk, dict) and "timeout" not in bk and os.environ.get("LOCAL_LLM_BASE_URL"):
        kw["backend_kwargs"] = {**bk, "timeout": _LM_TIMEOUT}
    if os.environ.get("QWEN_MAX_TOKENS"):
        mt = int(os.environ["QWEN_MAX_TOKENS"])
        for key in ("sampling_args", "sub_sampling_args"):
            kw[key] = {**(kw.get(key) or {}), "max_tokens": mt}
    if os.environ.get("QWEN_PRESENCE_PENALTY"):
        pp = float(os.environ["QWEN_PRESENCE_PENALTY"])
        for key in ("sampling_args", "sub_sampling_args"):
            kw[key] = {**(kw.get(key) or {}), "presence_penalty": pp}
    return _orig_rlm_init(self, *a, **kw)


_rlm.RLM.__init__ = _rlm_init_patched

if os.environ.get("QWEN_FENCE_TOLERANT", "") == "1":
    import rlm.core.rlm as _RC  # patched where it is called

    _orig_fcb = _RC.find_code_blocks
    _FENCE_RE = re.compile(r"```([A-Za-z0-9_+-]*)[ \t]*\n(.*?)\n```", re.DOTALL)
    _DOES_SOMETHING = (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Call,
                       ast.Import, ast.ImportFrom, ast.For, ast.While, ast.If)

    def _tolerant_find_code_blocks(text):
        blocks = _orig_fcb(text) if text else []
        if blocks or not text:
            return blocks
        out = []
        for lang, body in _FENCE_RE.findall(text):
            b = body.strip()
            if lang.lower() == "repl" or not b:
                continue
            try:
                tree = ast.parse(b)
            except SyntaxError:
                continue
            if any(isinstance(n, _DOES_SOMETHING) for n in ast.walk(tree)):
                out.append(b)
        return out

    _RC.find_code_blocks = _tolerant_find_code_blocks
    print("[fence] tolerant code-block parsing ON (```repl still preferred)", flush=True)

import harness as H  # noqa: E402

if os.environ.get("QWEN_REPL_CUE", "") == "1":
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("_opm", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                              "prompt_modules.py"))
    _OPM = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_OPM)
    _OPM.assert_roundtrip()
    _blocks = _OPM.split()
    _blocks["TURN_RULE"] = _blocks["TURN_RULE"].rstrip() + (
        " The fence label matters: write ```repl, never ```python and never a bare ```. "
        "A turn whose code is not in a ```repl``` block is discarded and its work is "
        "lost -- including the final turn, where answer[\"content\"] and "
        "answer[\"ready\"] = True must be assigned inside ```repl```.")
    _OPM.apply(_blocks)
    print("[cue] repl-fence cue appended to TURN_RULE", flush=True)

if os.environ.get("QWEN_PROSE_FINAL", "") == "1":
    import rlm.core.rlm as _RC3

    _prev_fcb = _RC3.find_code_blocks
    _EXACT_RE = re.compile(r"Exact Answer\s*:[ \t]*\S", re.I)  # non-empty value: the bare template is not an answer

    def _prose_final_find_code_blocks(text):
        blocks = _prev_fcb(text) if text else []
        if blocks or not text or not _EXACT_RE.search(text):
            return blocks
        body = text.rsplit("</think>", 1)[-1]
        i = body.lower().rfind("explanation:")
        content = body[i:].strip() if i >= 0 else "Explanation: (given above)\n" + body.strip()
        print("[prose-final] no code block but an 'Exact Answer:' line: submitting the prose answer", flush=True)
        return [f"answer[\"content\"] = {content!r}\nanswer[\"ready\"] = True"]

    _RC3.find_code_blocks = _prose_final_find_code_blocks
    print("[prose-final] prose 'Exact Answer:' turns are submitted as the answer", flush=True)

SEARCH_CAP = int(os.environ.get("BCP_SEARCH_CAP", "0") or 0)
if SEARCH_CAP:
    import rlm.core.rlm as _core2
    import rlm.utils.prompts as _PR2

    _orig_make_tools = H.make_tools

    def _capped_make_tools(state):
        search, get_document = _orig_make_tools(state)

        def capped_search(query: str, top_k: int = H.DEFAULT_TOP_K) -> list[dict]:
            if state["search_calls"] >= SEARCH_CAP:
                state["search_refused"] = state.get("search_refused", 0) + 1
                return [{"doc_id": "", "score": 0.0, "text": (
                    f"ERROR: search budget exhausted ({SEARCH_CAP} of {SEARCH_CAP} calls used). No results. "
                    "Answer from the documents and snippets you already have.")}]
            return search(query, top_k)

        capped_search.__doc__ = search.__doc__
        capped_search.__name__ = "search"
        return capped_search, get_document

    H.make_tools = _capped_make_tools
    _marker = "If a search returns only documents you have already seen"
    assert _marker in _core2.RLM_SYSTEM_PROMPT, "search-cap anchor not found"
    _core2.RLM_SYSTEM_PROMPT = _core2.RLM_SYSTEM_PROMPT.replace(_marker, (
        f"You have a hard budget of {SEARCH_CAP} search calls for this question; calls beyond that "
        "are refused and return an error instead of results, so plan the queries you spend them on. " + _marker), 1)
    _PR2.RLM_SYSTEM_PROMPT = _core2.RLM_SYSTEM_PROMPT
    print(f"[cap] search capped at {SEARCH_CAP} calls", flush=True)

if __name__ == "__main__":
    H.main()
