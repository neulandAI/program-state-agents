"""FRAMES runner for the snippet REPL agent: harness.py with the FRAMES questions and judge.

Questions are read from $FRAMES_DATA_DIR/frames_rows.json. The in-run judge is the BrowseComp grader prompt on a
local OpenAI-compatible server (LOCAL_JUDGE_BASE_URL); final grading is separate. Opt-in switches:
  FRAMES_FENCE_TOLERANT=1  run a mislabelled code fence if it parses as Python and acts
  FRAMES_BM25_LINE=1       state that the retriever is BM25 (use with RETRIEVER_LINE=0)
  FRAMES_SEARCH_CAP=N      at most N search calls; further calls return an error hit
  FRAMES_REPL_CUE=1        one sentence in TURN_RULE naming the ```repl fence
  LOCAL_LM_TIMEOUT         per-call LM timeout in seconds
"""
import argparse, json, os, re, sys, time
from pathlib import Path

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_SRC_DIR)), "shared"))
sys.path.insert(0, _SRC_DIR)
import harness as H  # noqa: E402
from rlm_shared.browsecomp_data import _JUDGE_PROMPT, _final_answer_span, heuristic_match  # noqa: E402


def _rows_path() -> str:
    from rlm_shared.paths import path
    return str(path("FRAMES_DATA_DIR") / "frames_rows.json")


if os.environ.get("FRAMES_FENCE_TOLERANT", "") == "1":
    import ast as _ast
    import rlm.core.rlm as _RC  # patched where it is called

    _orig_fcb = _RC.find_code_blocks
    _FENCE_RE = re.compile(r"```([A-Za-z0-9_+-]*)[ \t]*\n(.*?)\n```", re.DOTALL)
    _DOES_SOMETHING = (_ast.Assign, _ast.AnnAssign, _ast.AugAssign, _ast.Call,
                       _ast.Import, _ast.ImportFrom, _ast.For, _ast.While, _ast.If)

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
                tree = _ast.parse(b)
            except SyntaxError:
                continue
            if any(isinstance(n, _DOES_SOMETHING) for n in _ast.walk(tree)):
                out.append(b)
        return out

    _RC.find_code_blocks = _tolerant_find_code_blocks
    print("[fence] tolerant code-block parsing ON (```repl still preferred)", flush=True)

_BM25_LINE = (
    "The retriever is BM25 keyword search over English Wikipedia (one article per document, the "
    "whole article indexed): it scores documents by the words of your query, after stemming and "
    "stopword removal, not by their meaning, and quotation marks are ignored (a quoted phrase is "
    "not matched verbatim)."
)
if os.environ.get("FRAMES_BM25_LINE", "") == "1":
    import rlm.core.rlm as _core
    import rlm.utils.prompts as _PR

    assert "dense semantic retriever" not in _core.RLM_SYSTEM_PROMPT, "set RETRIEVER_LINE=0 with FRAMES_BM25_LINE=1"
    _marker = "If a search returns only documents you have already seen"
    assert _marker in _core.RLM_SYSTEM_PROMPT, "BM25-line anchor not found"
    _core.RLM_SYSTEM_PROMPT = _core.RLM_SYSTEM_PROMPT.replace(_marker, _BM25_LINE + " " + _marker, 1)
    _PR.RLM_SYSTEM_PROMPT = _core.RLM_SYSTEM_PROMPT
    print("[bm25] BM25 retriever line in system prompt", flush=True)

SEARCH_CAP = int(os.environ.get("FRAMES_SEARCH_CAP", "0") or 0)
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

if os.environ.get("FRAMES_REPL_CUE", "") == "1":
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("_opm", os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "prompt_modules.py"))
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

if os.environ.get("LOCAL_LM_TIMEOUT"):
    import rlm.clients.base_lm as _BL

    _lm_timeout = float(os.environ["LOCAL_LM_TIMEOUT"])
    _orig_bl_init = _BL.BaseLM.__init__

    def _bl_init(self, *a, **kw):
        kw.setdefault("timeout", _lm_timeout)
        _orig_bl_init(self, *a, **kw)

    _BL.BaseLM.__init__ = _bl_init

H.get_question = lambda r: r["question"]
H.get_gold_answer = lambda r: r["answer"]
H.get_gold_doc_ids = lambda r: set(r["gold_ids"])
H.get_evidence_doc_ids = lambda r: set(r["gold_ids"])


def judge_match(question: str, response: str, gold: str) -> bool:
    if heuristic_match(gold, _final_answer_span(response)):
        return True
    from openai import OpenAI
    cli = OpenAI(base_url=os.environ.get("LOCAL_JUDGE_BASE_URL") or os.environ["LOCAL_LLM_BASE_URL"],
                 api_key="EMPTY", timeout=600)
    model = os.environ.get("LOCAL_JUDGE_MODEL", os.environ.get("LOCAL_LLM_MODEL"))
    for _ in range(3):
        try:
            c = cli.chat.completions.create(
                model=model, temperature=0, max_tokens=512,
                messages=[{"role": "user", "content": _JUDGE_PROMPT.format(
                    question=question, response=response, correct_answer=gold)}],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            m = re.search(r"correct:\s*\**\s*(yes|no)", c.choices[0].message.content or "", re.I)
            if m:
                return m.group(1).lower() == "yes"
        except Exception as exc:  # noqa: BLE001
            print(f"  [judge error: {exc}]", flush=True)
            time.sleep(5)
    return heuristic_match(gold, _final_answer_span(response))


H.judge_match = judge_match


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=1)
    p.add_argument("--max-iterations", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=1)
    p.add_argument("--compaction", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rows = json.load(open(_rows_path()))
    results = []
    for q in range(args.start, args.end):
        rec = H.run_one(rows[q], q, args)
        rec["reasoning_types"] = rows[q]["reasoning_types"]
        results.append(rec)
    Path(f"{args.out}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
