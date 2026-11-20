"""Snippet REPL agent: the RLM library (github.com/alexzhang13/rlm @ 854e688) adapted to retrieval.

The library is not edited; every change below is a monkey-patch applied at import time, and every prompt patch
asserts on its anchor text so that an upstream change fails loudly. Run directly on BrowseComp-Plus
(`python harness.py --start 0 --end 1 --out PREFIX`) or imported by run_browsecomp.py, run_frames.py and
run_obliq.py, which rebind its attributes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import re
import time
from pathlib import Path

import requests

_AGENTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_AGENTS_DIR / "shared"))

from rlm_shared.browsecomp_data import (  # noqa: E402
    get_question, get_gold_answer, get_gold_doc_ids, get_evidence_doc_ids, judge_match,
)
from rlm import RLM  # noqa: E402
from rlm.logger import RLMLogger  # noqa: E402
from rlm.environments.local_repl import LocalREPL  # noqa: E402

# The library has no callback for llm_query / llm_query_batched, so they are counted here.
_LLMQ = {"n": 0, "batched": 0, "prompts": 0}


def _wrap(name, is_batch):
    orig = getattr(LocalREPL, name)

    def wrapped(self, prompt, *a, **kw):
        if is_batch:
            _LLMQ["batched"] += 1
            _LLMQ["prompts"] += len(prompt)
        else:
            _LLMQ["n"] += 1
        return orig(self, prompt, *a, **kw)

    setattr(LocalREPL, name, wrapped)


_wrap("_llm_query", False)
_wrap("_llm_query_batched", True)


# Recursive children (max_depth >= 2; inert at depth 1): a child gets its task as root_prompt, inherits the
# root's configuration, and its token usage, which the parent's usage summary omits, is accumulated in _D2.
_D2 = {"child_in_tok": 0, "child_out_tok": 0, "children": 0, "child_turns": 0}
_D2_CFG: dict = {}

_orig_completion = RLM.completion
_orig_init = RLM.__init__


def _d2_init(self, *a, **kw):
    if kw.get("depth", 0) > 0 and _D2_CFG:
        # Override: the library hands every child the root's max_iterations.
        kw["max_iterations"] = int(os.environ.get("CHILD_MAX_ITERATIONS", "15"))
        kw.setdefault("orchestrator", _D2_CFG["orchestrator"])
        kw.setdefault("compaction", _D2_CFG["compaction"])
        kw.setdefault("sampling_args", _D2_CFG["sampling_args"])
        kw.setdefault("sub_sampling_args", _D2_CFG["sub_sampling_args"])
        kw.setdefault("max_concurrent_subcalls", _D2_CFG["max_concurrent_subcalls"])
    return _orig_init(self, *a, **kw)


def _d2_completion(self, prompt, root_prompt=None):
    # The library calls a child without root_prompt, so the child would not see its task.
    if root_prompt is None and getattr(self, "depth", 0) > 0 and isinstance(prompt, str):
        root_prompt = prompt
    res = _orig_completion(self, prompt, root_prompt=root_prompt)
    if getattr(self, "depth", 0) > 0:
        us = getattr(res, "usage_summary", None)
        if us is not None:
            _D2["child_in_tok"] += int(getattr(us, "total_input_tokens", 0) or 0)
            _D2["child_out_tok"] += int(getattr(us, "total_output_tokens", 0) or 0)
        _D2["children"] += 1
        md = getattr(res, "metadata", None) or {}
        try:
            _D2["child_turns"] += len(md.get("iterations", []))
        except Exception:
            pass
    return res


RLM.__init__ = _d2_init
RLM.completion = _d2_completion


# Submission contract: an answer without an "Explanation:" and an "Exact Answer:" line is refused, at most
# MAX_REJECTS times per question; the notice lands in the cell's stdout for the next turn.
MAX_REJECTS = 3
_REJECTS = {"n": 0}
_FORMAT_OK = re.compile(r"Exact Answer\s*:", re.I)
_EXPL_OK = re.compile(r"Explanation\s*:", re.I)


def _install_contract():
    from rlm.environments.local_repl import _AnswerDict

    orig = _AnswerDict.__setitem__

    def guarded(self, key, value):
        if key == "ready" and value:
            content = str(self.get("content", "") or "")
            ok = bool(_FORMAT_OK.search(content)) and bool(_EXPL_OK.search(content))
            if not ok and _REJECTS["n"] < MAX_REJECTS:
                _REJECTS["n"] += 1
                dict.__setitem__(self, "ready", False)
                print(
                    f"[SUBMISSION REJECTED {_REJECTS['n']}/{MAX_REJECTS}] answer['content'] must "
                    "contain BOTH an 'Explanation:' line and an 'Exact Answer:' line.\n"
                    "Do not ask the user questions and do not describe what you would search "
                    "next — you cannot receive a reply. Commit to your best answer, or write "
                    "'Exact Answer: Unknown'. Rewrite answer['content'] and set "
                    "answer['ready'] = True again."
                )
                return
        return orig(self, key, value)

    _AnswerDict.__setitem__ = guarded


_install_contract()


# Drop the prompt lines that describe `context` as the corpus; here it holds only the question.
def _strip_context_fiction():
    import rlm.core.rlm as core
    import rlm.utils.prompts as P

    orig_sys, orig_user = P.build_rlm_system_prompt, P.build_user_prompt
    marker_sys = "Your context is a"
    marker_user = "You have not interacted with the REPL environment"

    def patched_sys(*a, **kw):
        msgs = orig_sys(*a, **kw)
        for m in msgs:
            if m["role"] == "user" and marker_sys in m["content"]:
                m["content"] = re.sub(
                    r"\n*Your context is a .*?~100k tokens at once\.\s*", "", m["content"],
                    flags=re.S).strip()
        return msgs

    def patched_user(*a, **kw):
        msg = orig_user(*a, **kw)
        if marker_user in msg["content"]:
            msg["content"] = re.sub(
                r"You have not interacted with the REPL environment.*?final answer yet\.\s*",
                "", msg["content"], flags=re.S)
        return msg

    core.build_rlm_system_prompt = patched_sys
    core.build_user_prompt = patched_user


_strip_context_fiction()


# REPL output ceiling 20K -> RLM_MAX_OUTPUT_CHARS, enforced and stated in the prompt.
REPL_OUTPUT_CHARS = int(os.environ.get("RLM_MAX_OUTPUT_CHARS", "60000"))


def _raise_output_ceiling():
    import rlm.core.rlm as core
    import rlm.utils.parsing as P
    import rlm.utils.prompts as PR

    orig = P.format_iteration

    def patched(iteration, max_character_length=REPL_OUTPUT_CHARS):
        return orig(iteration, max_character_length=max_character_length)

    core.format_iteration = patched

    old = "REPL outputs over ~20K characters are truncated"
    new_txt = f"REPL outputs over ~{REPL_OUTPUT_CHARS // 1000}K characters are truncated"
    assert old in PR.RLM_SYSTEM_PROMPT, "20K sentence not found — repo changed"
    core.RLM_SYSTEM_PROMPT = PR.RLM_SYSTEM_PROMPT.replace(old, new_txt)
    PR.RLM_SYSTEM_PROMPT = core.RLM_SYSTEM_PROMPT


_raise_output_ceiling()


# Re-anchor the system prompt from the `context` variable to the retrieval tools.
_CTX_REWRITES = [
    ("You are a Recursive Language Model (RLM): a language model with a prompt, and a very "
     "important context stored in a Python REPL related to that prompt.",
     "You are a Recursive Language Model (RLM): a language model with a prompt, and a very "
     "important document corpus reachable from a Python REPL through retrieval tools."),

    ("- `context`: the important, potentially very long information related to the prompt "
     "(typically `str` or `list[str]`).",
     "- the corpus: reached with `search` and `get_document` (described below). It is "
     "far too large to read; you see only what those tools return. Search results are short "
     "snippets meant to be printed and read directly — printing them puts them in front of "
     "you. `get_document` returns a document's full text into a Python variable: pass that "
     "variable to `llm_query` with a specific question rather than printing it, whatever its "
     "length. Use `llm_query_batched` to read several documents in one turn."),

    # Follows the ceiling already rewritten by _raise_output_ceiling.
    (f"REPL outputs over ~{REPL_OUTPUT_CHARS // 1000}K characters are truncated, so for "
     "longer payloads slice `context` and pass slices through `llm_query` rather than "
     "`print`-ing them whole.",
     f"REPL outputs over ~{REPL_OUTPUT_CHARS // 1000}K characters are truncated, so anything "
     "that large — a long document, or many results at once — should go through `llm_query` "
     "rather than being `print`-ed whole."),

    ("As a general strategy, you should start by probing your context to understand it better "
     "(e.g. print a few lines, count them, etc.). Then, use the REPL to build up an answer to "
     "the query.",
     "As a general strategy, you should start by searching for the most distinctive detail in "
     "the question and looking at what comes back. Then, use the REPL to build up an answer to "
     "the query."),

    ("Do not flip `answer[\"ready\"] = True` on turn 1 without first inspecting `context`.",
     "Do not flip `answer[\"ready\"] = True` on turn 1 without first retrieving evidence."),

    ("Directly after you probe the `context` and understand your task, pause and plan:",
     "Directly after your first search and once you understand your task, pause and plan:"),

    ("(Conversely: if a Python keyword / regex search over `context` would already pin the "
     "answer, or if a single visible passage already contains it, just read it directly — "
     "sub-LMs are for when the raw text won't fit or the question needs semantic "
     "interpretation.)",
     "(A substring or regex test over a document tells you a word is present, not what the "
     "document says: use it to locate, never to conclude. Read directly only when the passage "
     "you need is already printed in front of you.)"),

    ("Long REPL stdout pollutes history the same way raw `context` does:",
     "Long REPL stdout pollutes history the same way raw document text does:"),

    ("Sub-LLMs have no REPL; they only see the prompt and the `context` slice you pass them.",
     "Sub-LLMs have no REPL and cannot retrieve; they only see the prompt and the document "
     "text you pass them."),

    ("If the raw workload exceeds both budgets at once (e.g. a context far larger than "
     "~20 × 100K chars),",
     "If the raw workload exceeds both budgets at once (e.g. far more candidate documents "
     "than ~20 × 100K chars),"),
]


def _reanchor_prompt():
    import rlm.core.rlm as core
    import rlm.utils.prompts as PR

    sysp, add = core.RLM_SYSTEM_PROMPT, PR.ORCHESTRATOR_ADDENDUM
    for old, newtxt in _CTX_REWRITES:
        if old in sysp:
            sysp = sysp.replace(old, newtxt, 1)
        elif old in add:
            add = add.replace(old, newtxt, 1)
        else:
            raise AssertionError(f"prompt anchor not found, repo changed: {old[:70]!r}")
    core.RLM_SYSTEM_PROMPT = sysp
    PR.RLM_SYSTEM_PROMPT = sysp
    core.ORCHESTRATOR_ADDENDUM = add
    PR.ORCHESTRATOR_ADDENDUM = add


_reanchor_prompt()


# (a) every turn must execute code; (b) the tool list is injected as "6. ..." of a list that does not exist.
def _final_prompt_fixes():
    import rlm.core.rlm as core
    import rlm.utils.prompts as PR

    must_code = ("\n\nEvery turn must execute code: never reply with prose alone. Plan briefly, "
                 "then put the plan into action in a ```repl``` block in the SAME turn. If you "
                 "already have the answer, submit it — set answer[\"content\"] and "
                 "answer[\"ready\"] = True inside a ```repl``` block. Restating a plan you have "
                 "already carried out wastes a turn and changes nothing.")
    marker = ("Plan in prose, then execute one ```repl``` block every turn, get feedback from "
              "the output, then continue on the next turn.")
    assert marker in core.RLM_SYSTEM_PROMPT, "plan-in-prose sentence not found"
    core.RLM_SYSTEM_PROMPT = core.RLM_SYSTEM_PROMPT.replace(marker, marker + must_code, 1)
    PR.RLM_SYSTEM_PROMPT = core.RLM_SYSTEM_PROMPT

    orig = core.build_rlm_system_prompt

    def patched(*a, **kw):
        msgs = orig(*a, **kw)
        for m in msgs:
            if m["role"] == "system":
                m["content"] = m["content"].replace(
                    "\n6. Custom tools and data available in the REPL:",
                    "\nRetrieval tools available in the REPL:")
        return msgs

    core.build_rlm_system_prompt = patched


_final_prompt_fixes()


# What to do when searches stop returning new evidence.
_WEAK_RESULTS = (
    " If a search returns only documents you have already seen, or only snippets that are "
    "plainly about something else, the wording of the query is not the problem: change which "
    "entity, event or detail you are searching for, or commit to the best answer your "
    "evidence supports. Rewording the same hypothesis does not make the corpus contain it."
)


def _add_weak_result_rule():
    import rlm.core.rlm as core
    import rlm.utils.prompts as PR

    marker = ("As a general strategy, you should start by searching for the most distinctive "
              "detail in the question and looking at what comes back.")
    assert marker in core.RLM_SYSTEM_PROMPT, "weak-result anchor not found"
    core.RLM_SYSTEM_PROMPT = core.RLM_SYSTEM_PROMPT.replace(marker, marker + _WEAK_RESULTS, 1)
    PR.RLM_SYSTEM_PROMPT = core.RLM_SYSTEM_PROMPT


_add_weak_result_rule()


# The retriever matches meaning, not literal strings (RETRIEVER_LINE=0 removes this sentence).
_RETRIEVER_LINE = (
    " The index is a dense semantic retriever: it matches meaning, not literal strings, and "
    "quotation marks in a query have NO effect — a quoted phrase is not looked up verbatim. "
    "Describe the thing you are looking for in your own words rather than quoting text you "
    "expect the document to contain."
)


def _add_retriever_line():
    import rlm.core.rlm as core
    import rlm.utils.prompts as PR

    marker = "If a search returns only documents you have already seen"
    assert marker in core.RLM_SYSTEM_PROMPT, "retriever-line anchor not found"
    core.RLM_SYSTEM_PROMPT = core.RLM_SYSTEM_PROMPT.replace(
        marker, _RETRIEVER_LINE.strip() + " " + marker, 1)
    PR.RLM_SYSTEM_PROMPT = core.RLM_SYSTEM_PROMPT


if os.environ.get("RETRIEVER_LINE", "1").strip() not in ("0", "false", "False"):
    _add_retriever_line()


# DEPTH2_PROMPT=1 (with --max-depth 2): children spawned by rlm_query can search and read documents.
_DEPTH2_LINE = (
    " `rlm_query(prompt)` and `rlm_query_batched(prompts)` spawn CHILD AGENTS, not sub-LLM "
    "calls: each child gets its own REPL and its own `search` and `get_document`, so "
    "unlike `llm_query` a child can run its own searches and open its own documents. A child "
    "sees only the prompt string you pass it — it inherits none of your variables and none of "
    "your findings — so each sub-question must be self-contained. It returns its answer as a "
    "string."
)


def _add_depth2_line():
    import rlm.core.rlm as core
    import rlm.utils.prompts as PR

    marker = "If a search returns only documents you have already seen"
    assert marker in core.RLM_SYSTEM_PROMPT, "depth2-line anchor not found"
    core.RLM_SYSTEM_PROMPT = core.RLM_SYSTEM_PROMPT.replace(
        marker, _DEPTH2_LINE.strip() + " " + marker, 1)
    PR.RLM_SYSTEM_PROMPT = core.RLM_SYSTEM_PROMPT


if os.environ.get("DEPTH2_PROMPT", "").strip() not in ("", "0", "false", "False"):
    _add_depth2_line()


# SEARCH_TRIAGE_RULE=1: for wide search blocks, print a summary and read the snippets through llm_query.
if os.environ.get("SEARCH_TRIAGE_RULE", "").strip() not in ("", "0", "false", "False"):

    def _add_triage_rule():
        import rlm.core.rlm as core
        import rlm.utils.prompts as PR

        anchor_txt = ("Search results are short snippets meant to be printed and read "
                      "directly — printing them puts them in front of you.")
        addition = (" If one block issues more than three searches, do not print all their "
                    "snippets — that overruns the output limit and the tail is silently cut. "
                    "In that case print only score, doc_id and the first line of each hit, and "
                    "pass the snippet text through `llm_query` to read it and extract what you "
                    "need.")
        assert anchor_txt in core.RLM_SYSTEM_PROMPT, "triage anchor not found"
        core.RLM_SYSTEM_PROMPT = core.RLM_SYSTEM_PROMPT.replace(
            anchor_txt, anchor_txt + addition, 1)
        PR.RLM_SYSTEM_PROMPT = core.RLM_SYSTEM_PROMPT

    _add_triage_rule()

PROXY = os.environ.get("EMBEDDING_SERVER_URL", "http://localhost:8024")

# SEARCH_SNIPPET_TOKENS truncates with the BrowseComp-Plus tokenizer (Qwen3-0.6B); unset, snippets are cut to
# SEARCH_SNIPPET_LEN characters.
SNIPPET_TOKENS = os.environ.get("SEARCH_SNIPPET_TOKENS", "")
SNIPPET_LEN = int(os.environ.get("SEARCH_SNIPPET_LEN", "500"))
DEFAULT_TOP_K = int(os.environ.get("OFFICIAL_TOP_K", "5"))


def _truncate(text: str) -> str:
    if SNIPPET_TOKENS:
        from rlm_shared.tokenizer import snippet_tokenizer

        tk = snippet_tokenizer()
        ids = tk.encode(text, add_special_tokens=False)
        n = int(SNIPPET_TOKENS)
        if len(ids) > n:
            return tk.decode(ids[:n], skip_special_tokens=True)
        return text
    if len(text) > SNIPPET_LEN:
        return text[:SNIPPET_LEN] + "\u2026"
    return text


def make_tools(state: dict):
    """search / get_document bound to a per-question metrics dict."""

    def search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        # top_k is a default, not a cap; explicit overrides are counted.
        t0 = time.perf_counter()
        if top_k != DEFAULT_TOP_K:
            state["topk_overrides"] += 1
            state["topk_requested"].append(int(top_k))
        r = requests.post(f"{PROXY}/search", json={"query": query, "top_k": top_k}, timeout=120)
        r.raise_for_status()
        results = r.json()["results"]
        for x in results:
            if x.get("text"):
                x["text"] = _truncate(x["text"])
        state["search_calls"] += 1
        state["retrieval_s"] += time.perf_counter() - t0
        state["retrieved"].update(str(x["doc_id"]) for x in results)
        return results

    def get_document(doc_id: str) -> dict:
        t0 = time.perf_counter()
        r = requests.post(f"{PROXY}/document", json={"doc_id": str(doc_id)}, timeout=120)
        r.raise_for_status()
        state["getdoc_calls"] += 1
        state["retrieval_s"] += time.perf_counter() - t0
        state["retrieved"].add(str(doc_id))
        return r.json()

    return search, get_document


# Task text appended to the question: ours by default, LightOn's BrowseComp-Plus template with
# LIGHTON_TASK_TEXT=1.
_TASK_OURS = """

Answer the question using search and get_document over the corpus.

Set answer["content"] to a string in EXACTLY this format, then answer["ready"] = True:

  Explanation: <your explanation. Cite evidence documents inline by enclosing their
  doc_ids in square brackets at the end of sentences, for example [94480].>
  Exact Answer: <the exact name, date, title, or value the question asks for>
  Confidence: <your confidence score between 0% and 100%>

get_document returns the text in a Python variable, not into your context — you have not
read a document until you have actually looked at its text. Pass that variable to llm_query
with a specific question about it, whatever its length, rather than printing it or testing
it for a keyword: an excerpt or a substring check tells you a word is present, not what the
document says. Use llm_query_batched to read several documents in one turn.

If the answer cannot be found after exhausting all search angles, give Exact Answer: Unknown.
"""

_TASK_LIGHTON = """

You are a deep research agent. You need to answer the given question by interacting with a
search engine, using the search and get_document tools provided. Please perform
reasoning and use the tools step by step, in an interleaved manner. You may use the
search and get_document tools multiple times.

Your response should be in the following format. Set answer["content"] to that string, then
set answer["ready"] = True:

  Explanation: {your explanation for your final answer. For this explanation section only,
  you should cite your evidence documents inline by enclosing their doc_ids in square
  brackets [] at the end of sentences. For example, [94480].}
  Exact Answer: {the exact name, date, title, or value the question asks for}
  Confidence: {your confidence score between 0% and 100% for your answer}
"""

_LIGHTON_TASK = os.environ.get("LIGHTON_TASK_TEXT", "").strip() not in ("", "0", "false", "False")
TASK_SUFFIX = _TASK_LIGHTON if _LIGHTON_TASK else _TASK_OURS

# NO_UNKNOWN=1 removes the abstention line from our task text.
_UNKNOWN_LINE = ("\nIf the answer cannot be found after exhausting all search angles, "
                 "give Exact Answer: Unknown.\n")
if os.environ.get("NO_UNKNOWN", "").strip() not in ("", "0", "false", "False"):
    assert _UNKNOWN_LINE in TASK_SUFFIX or _LIGHTON_TASK, "Unknown-escape anchor not found"
    TASK_SUFFIX = TASK_SUFFIX.replace(_UNKNOWN_LINE, "")


def _tool_specs(search, get_document):
    """Tool descriptions: ours by default, LightOn's (BrowseComp-Plus MCP server) under LIGHTON_TOOL_DOCS=1."""
    if os.environ.get("LIGHTON_TOOL_DOCS", "").strip() not in ("", "0", "false", "False"):
        return {
            "search": {"tool": search,
                             # LightOn's wording plus the snippet size.
                             "description": f"Perform a search on a knowledge source. Returns "
                                            f"top-{DEFAULT_TOP_K} hits with doc_id, score, and "
                                            f"text. The text contains the document's contents, "
                                            f"up to ~2000 characters (512 tokens) per hit."},
            "get_document": {"tool": get_document,
                             "description": "Retrieve a full document by its doc_id."},
        }
    return {
        "search": {"tool": search,
                         "description": "Search the document corpus. "
                                        "search(query: str, top_k: int = 5) -> "
                                        "list[{score, doc_id, text}]. text is a snippet of up "
                                        "to ~2000 characters (512 tokens) of the document's "
                                        "contents."},
        "get_document": {"tool": get_document,
                         "description": "get_document(doc_id: str) -> {doc_id, text}. "
                                        "Full text of one document. Can be very long."},
    }


# OBLIQ_NO_GETDOC=1 removes get_document from the tools and from the prompt.
def _strip_get_document(specs: dict) -> dict:
    if os.environ.get("OBLIQ_NO_GETDOC", "") != "1":
        return specs
    out = {k: v for k, v in specs.items() if k != "get_document"}
    import rlm.core.rlm as _core
    import rlm.utils.prompts as _PR
    _txt = _core.RLM_SYSTEM_PROMPT
    _new = ("- the corpus: reached with `search` (described below). It is far too large to "
            "read; you see only what that tool returns. Search results are short snippets meant "
            "to be printed and read directly \u2014 printing them puts them in front of you, and "
            "on this corpus a snippet already contains the document in full.")
    _anchor = "- the corpus: reached with `search` and `get_document`"
    if _anchor not in _txt:
        raise SystemExit("OBLIQ_NO_GETDOC: REPL_TOOLS anchor not found; re-derive the edit")
    _start = _txt.index(_anchor)
    _end = _txt.index("- `llm_query(", _start)
    _core.RLM_SYSTEM_PROMPT = _txt[:_start] + _new + "\n" + _txt[_end:]
    _PR.RLM_SYSTEM_PROMPT = _core.RLM_SYSTEM_PROMPT
    print("[no-getdoc] get_document removed from tools and from REPL_TOOLS", flush=True)
    return out


def run_one(row: dict, q_idx: int, args) -> dict:
    question = get_question(row)
    gold = get_gold_answer(row)
    gold_ids = get_gold_doc_ids(row)
    evid_ids = get_evidence_doc_ids(row)

    state = {"search_calls": 0, "getdoc_calls": 0, "retrieval_s": 0.0,
             "retrieved": set(), "iterations": 0, "subcalls": 0,
             "in_tok": 0, "out_tok": 0, "cost": None, "lm_calls": 0,
             "topk_overrides": 0, "topk_requested": []}
    search, get_document = make_tools(state)
    _LLMQ.update(n=0, batched=0, prompts=0)
    _REJECTS["n"] = 0

    def on_iter_complete(depth, iteration_num, duration):
        state["iterations"] = max(state["iterations"], int(iteration_num) + 1)

    def on_sub_complete(depth, model, duration, error):
        state["subcalls"] += 1

    logger = RLMLogger(log_dir=os.environ.get("RLM_TRAJ_DIR", "logs/rlm_traj"))
    # Children spawned by rlm_query inherit this configuration (see _d2_init).
    _D2_CFG.update(
        orchestrator=os.environ.get("ORCHESTRATOR", "1").strip() not in ("0", "false", "False"),
        compaction=args.compaction,
        sampling_args={"reasoning_effort": os.environ.get("AZURE_OPENAI_REASONING_EFFORT", "high")},
        sub_sampling_args={"reasoning_effort": os.environ.get("SUB_LM_REASONING_EFFORT", "medium")},
        max_concurrent_subcalls=int(os.environ.get("MAX_CONCURRENT_SUBCALLS", "2")),
    )
    for _k in _D2:
        _D2[_k] = 0
    # LOCAL_LLM_BASE_URL routes the root model and llm_query to a local OpenAI-compatible server (vLLM).
    _local_url = os.environ.get("LOCAL_LLM_BASE_URL", "").strip()
    if _local_url:
        _backend = "vllm"
        _backend_kwargs = {
            "base_url": _local_url,
            "api_key": os.environ.get("LOCAL_LLM_API_KEY", "EMPTY"),
            "model_name": os.environ.get("LOCAL_LLM_MODEL", "qwen3.8-27b"),
        }
    else:
        _backend = "azure_openai"
        _backend_kwargs = {
            "api_key": os.environ["AZURE_OPENAI_API_KEY"],
            "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/"),
            "api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
            "model_name": os.environ["AZURE_OPENAI_MODEL"],
            "azure_deployment": os.environ["AZURE_OPENAI_MODEL"],
        }

    rlm = RLM(
        max_concurrent_subcalls=int(os.environ.get("MAX_CONCURRENT_SUBCALLS", "2")),
        logger=logger,
        backend=_backend,
        backend_kwargs=_backend_kwargs,
        environment="local",
        max_iterations=args.max_iterations,
        max_depth=args.max_depth,
        orchestrator=os.environ.get("ORCHESTRATOR", "1").strip() not in ("0", "false", "False"),
        compaction=args.compaction,
        custom_tools=_strip_get_document(_tool_specs(search, get_document)),
        on_iteration_complete=on_iter_complete,
        on_subcall_complete=on_sub_complete,
        verbose=args.verbose,
        sampling_args={"reasoning_effort": os.environ.get("AZURE_OPENAI_REASONING_EFFORT", "high")},
        sub_sampling_args={"reasoning_effort": os.environ.get("SUB_LM_REASONING_EFFORT", "medium")},
        # SUB_LM_MODEL routes llm_query to a different deployment than the root model.
        **({"other_backends": ["azure_openai"],
            "other_backend_kwargs": [{
                "api_key": os.environ["AZURE_OPENAI_API_KEY"],
                "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/"),
                "api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
                "model_name": os.environ["SUB_LM_MODEL"],
                "azure_deployment": os.environ["SUB_LM_MODEL"],
            }]} if os.environ.get("SUB_LM_MODEL") else {}),
    )

    t0 = time.perf_counter()
    err = None
    try:
        # root_prompt puts the question in the user message; prompt only becomes the REPL variable `context`.
        result = rlm.completion(question + TASK_SUFFIX, root_prompt=question + TASK_SUFFIX)
        response = getattr(result, "response", "") or ""
        us = getattr(result, "usage_summary", None)
        if us is not None:
            state["in_tok"] = int(getattr(us, "total_input_tokens", 0) or 0)
            state["out_tok"] = int(getattr(us, "total_output_tokens", 0) or 0)
            state["cost"] = getattr(us, "total_cost", None)
            calls = sum(int(m.total_calls or 0) for m in us.model_usage_summaries.values())
            state["lm_calls"] = calls
    except Exception as exc:  # noqa: BLE001
        response, err = "ERROR", f"{type(exc).__name__}: {exc}"
        if os.environ.get("RLM_DEBUG_TB") == "1":
            import traceback
            print("---- RLM_DEBUG_TB ----", flush=True)
            traceback.print_exc()
            print("---- end ----", flush=True)
    latency = time.perf_counter() - t0
    try:
        rlm.close()
    except Exception:
        pass

    correct = False if err else judge_match(question, response, gold)
    got = state["retrieved"]
    rec = {
        "q_idx": q_idx,
        "question": question,
        "gold": gold,
        "predicted": response,
        "correct": bool(correct),
        # Root turns: LM calls minus llm_query prompts (each prompt is one LM call in the usage summary).
        "iterations": max(0, state["lm_calls"] - (_LLMQ["n"] + _LLMQ["prompts"])),
        "lm_calls_total": state["lm_calls"],
        "search_calls": state["search_calls"],
        "get_document_calls": state["getdoc_calls"],
        "llm_query_calls": _LLMQ["n"] + _LLMQ["prompts"],
        "llm_query_single": _LLMQ["n"],
        "llm_query_batched_calls": _LLMQ["batched"],
        "llm_query_batched_prompts": _LLMQ["prompts"],
        "recursive_subcalls": state["subcalls"],
        "format_rejections": _REJECTS["n"],
        "topk_overrides": state["topk_overrides"],
        "topk_requested": state["topk_requested"][:50],
        "unique_docs": len(got),
        "latency_s": round(latency, 1),
        "retrieval_latency_s": round(state["retrieval_s"], 1),
        "gold_recall": (len(gold_ids & got) / len(gold_ids)) if gold_ids else 0.0,
        "evid_recall": (len(evid_ids & got) / len(evid_ids)) if evid_ids else 0.0,
        "input_tokens": state["in_tok"],
        "output_tokens": state["out_tok"],
        # Child RLM usage is not in usage_summary; accumulated in _D2.
        "child_input_tokens": _D2["child_in_tok"],
        "child_output_tokens": _D2["child_out_tok"],
        "total_input_tokens": state["in_tok"] + _D2["child_in_tok"],
        "child_rlms": _D2["children"],
        "child_turns": _D2["child_turns"],
        "lm_calls": state["lm_calls"],
        "cost_usd": state["cost"],
        "error": err,
    }
    mark = "✓" if correct else "✗"
    print(f"[Q{q_idx + 1:03d}] [{mark}] iters={rec['iterations']} search={rec['search_calls']} "
          f"getdoc={rec['get_document_calls']} llmq={rec['llm_query_calls']} "
          f"lat={rec['latency_s']:.0f}s tok={rec['input_tokens']:,}/{rec['output_tokens']:,} "
          f"goldR={rec['gold_recall']:.2f} "
          f"pred={response[:70]!r}", flush=True)
    return rec


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=100)
    p.add_argument("--max-iterations", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=1)
    p.add_argument("--compaction", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from datasets import load_dataset
    ds = load_dataset("Tevatron/browsecomp-plus", split="test")

    results = []
    for q_idx in range(args.start, args.end):
        results.append(run_one(ds[q_idx], q_idx, args))
        if (len(results)) % 2 == 0:
            Path(f"{args.out}_checkpoint.json").write_text(json.dumps(results, indent=2))
    Path(f"{args.out}.json").write_text(json.dumps(results, indent=2))
    ok = sum(1 for r in results if r["correct"])
    print(f"\n{args.out}: {ok}/{len(results)} = {ok / max(1, len(results)):.4f}")


if __name__ == "__main__":
    main()
