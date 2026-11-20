"""Prompt blocks of the document REPL agent, imported by run_one.py.

GEPA_PROMPTS=<json> overrides existing string blocks at import (see the end of this file).
"""

# Verbatim from texttron/BrowseComp-Plus search_agent/prompts.py :: QUERY_TEMPLATE.
QUERY_TEMPLATE = """
You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search tool provided. Please perform reasoning and use the tools step by step, in an interleaved manner. You may use the search tool multiple times.

Question: {Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

CONTEXT_NOTE = (
    "This task has no preloaded context. The knowledge source is a fixed corpus of "
    "100,195 web documents reachable only through the `search` tool "
    "listed in your system prompt. Retrieved material accumulates in the REPL variables "
    "DOCS and SEARCH_LOG, which serve as your context and grow as you retrieve."
)

USER_PROLOGUE = (
    "Your context is retrieved rather than given, so the REPL variables DOCS and "
    "SEARCH_LOG are what the RLM system prompt calls `context`: they start empty and grow "
    "as you retrieve. Retrieval is semantic, driven by the wording of each query, so a "
    "near-identical query returns near-identical documents. Retrieved documents are long "
    "web pages — tens of thousands of characters each — so slice them or pass them to "
    "`llm_query` / `llm_query_batched` rather than printing them, and print only short "
    "results."
)

TASK_FACTS = (
    " When a question states several conditions, the answer satisfies all of them at once: a "
    "candidate that fits most of them and contradicts one is not the answer, and a condition "
    "you have not seen satisfied in a retrieved document is unverified, whatever your prior "
    "belief. Setting answer['ready'] submits the answer and ends the run (a submission whose "
    "Exact Answer string occurs in no retrieved document may be held once, with a notice, for "
    "a spelling check); nothing can be checked after the run ends."
)
USER_PROLOGUE = USER_PROLOGUE + TASK_FACTS


def prologue_id() -> str:
    """Short stable hash of the prompt set, recorded in every run file so a
    directory of runs can be proven to share one configuration."""
    import hashlib

    blob = (QUERY_TEMPLATE + CONTEXT_NOTE + USER_PROLOGUE + globals().get('DESIGN_E', '') + globals().get('DESIGN_F', '') + globals().get('DESIGN_F2', '') + globals().get('DESIGN_G', '')).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


# Optional (--search-budget); off in the paper's runs.
BUDGET_CLAUSE = (
    " Retrieval is the expensive part of this task: aim to spend about {n} `search` calls on the "
    "whole question, and stop once the evidence already in DOCS settles it. Before each new query, "
    "check SEARCH_LOG and prefer re-reading what you already have over retrieving more."
)


# Replaces the stock RLM system prompt (--solo-prompt); use with --no-delegation and --no-orchestrator.
SOLO_SYSTEM_PROMPT = """You are a research agent working inside a Python REPL. You will be queried \
turn by turn until you produce an answer.

Write code in ```repl``` blocks. The REPL persists across turns. Available in the REPL:
- `DOCS`: the documents you have retrieved so far, keyed by docid. Already seeded with results \
from an initial search for your question.
- `SEARCH_LOG`: the queries already issued. Already contains the seed query.
- `answer`: a dict `{{"content": "", "ready": False}}`. To submit, set `answer["content"]` to your \
final answer and `answer["ready"] = True` inside a ```repl``` block.
- `SHOW_VARS() -> str`: lists REPL variables. You will not need it; the namespace is exactly the \
names in this list plus the tools below.
{custom_tools_section}

You do the reading and the reasoning yourself, in your own turn. There are no sub-agents, no \
helper models, and nothing to hand work off to. Read document text directly out of `DOCS` with \
ordinary Python — slicing, regex, `in`, splitting on lines — and draw your conclusions yourself.

Only `print(...)` output comes back to you between turns; a bare expression on the last line is \
discarded. REPL output over ~20K characters is truncated, so print short, targeted extracts rather \
than whole documents.

How to work:
- Your first block must do real retrieval or analysis. Do not spend a turn inspecting the \
environment or restating the question.
- Your plan goes OUTSIDE the code block, never inside it. Before the ```repl``` fence write at \
most one short line of plain prose saying what this turn does, and skip even that when the next \
step is obvious. Write nothing else outside the fence.
- Inside the block write NO comments at all — not a plan comment, not a step header, not a note \
restating the next line. Bare code only. If you catch yourself typing `#`, that thought belongs \
in the one prose line before the fence, or nowhere.
- Keep your reasoning precise and to the point. Decide, act, look at the output, move on.
- Everything you write stays in your context and is re-read on every later turn, so anything you \
type now is paid for again on every turn after it.
"""

# Optional (--structured-plan).
STRUCTURED_PLAN_CLAUSE = """

Write the plan line before the fence as numbered steps, not a paragraph. Two to four short steps,
each naming a concrete action and what it would settle:

    1. grep DOCS for the Aalborg 2018 conference programme -> get the first author's name
    2. search for that author's 2019-2021 two-author article
    3. check the acknowledgments for the person thanked

Name the step you are executing this turn. When a step returns something that changes the plan,
rewrite the remaining steps rather than continuing down a list you no longer believe. Keep each
step to one line -- this is a route, not an essay.
"""

# Optional (--constraint-check).
CONSTRAINT_CLAUSE = """

Track the question's conditions explicitly. Early on, put them in a REPL variable, one entry per
condition the answer has to satisfy:

    CONSTRAINTS = {"c1": "article published 2019-2021", "c2": "two authors", ...}
    EVIDENCE = {}

Keep your searches whole -- search the way you would describe the question to a person, not one
condition at a time. Splitting a query into its separate conditions retrieves worse, because each
fragment loses the context that makes a document identifiable.

Use the conditions when you check a candidate, not when you search for one. For each condition
record what satisfied it:

    EVIDENCE["c1"] = ("docid", "the exact phrase from the document")

Before you submit, print CONSTRAINTS against EVIDENCE and look at what is unfilled. A candidate
that satisfies most conditions and contradicts one is not the answer, and a condition with no
recorded quote is unverified no matter how confident you feel. If conditions remain unfilled and
you still have searches left, spend them on those conditions specifically.
"""

# Optional (--single-pass-read).
SINGLE_PASS_CLAUSE = """

Read each document once. When you open a document, pull everything you might need from it in that
same block -- every field, date, name, quote and number the question could turn on -- and print a
compact summary of it. Do not print a fragment, look at it, and come back to the same document on
a later turn: a return trip costs a whole turn. If you are unsure whether a detail matters, extract
it anyway while you are already in the text.
"""

# --lean-prologue.
LEAN_CLAUSE = """

Starting state. DOCS and SEARCH_LOG are already populated: a seed search for your question has
run, its results are in DOCS, and SEARCH_LOG holds that query. The REPL namespace is exactly
DOCS, SEARCH_LOG, history and answer, plus the tools named in your system prompt. You do not
need to inspect the namespace -- SHOW_VARS(), list(DOCS.keys()) and printing SEARCH_LOG return
only what this paragraph already tells you.

First turn. Your first code block must do real retrieval or analysis work. Do not spend a turn
orienting yourself, restating the plan, or printing the environment.

Your plan goes OUTSIDE the code block, never inside it: at most one short line of plain prose
before the ```repl``` fence, skipped entirely when the next step is obvious.

Inside the block write NO comments at all -- not a plan comment, not a step header, not a note
restating the next line. Bare code only. If you catch yourself typing `#`, that thought belongs in
the one prose line before the fence, or nowhere.

Keep your reasoning precise and to the point. Decide, act, read the output, move on. Everything you write stays in your context for the rest of the run
and is re-read on every later turn, so anything you type now is paid for again on every turn
after it.
"""

NO_DELEGATION_CLAUSE = """

IMPORTANT OVERRIDE -- sub-calls are DISABLED for this run.
`llm_query`, `llm_query_batched`, `rlm_query`, `rlm_query_batched` and `read_docs` do NOT work;
calling any of them raises an error. Ignore every part of the system prompt that tells you to
delegate to sub-LLMs. You must read and analyse document text yourself inside the REPL: search
returns dicts whose 'text' key holds the full document, so use ordinary Python -- slicing, regex,
`in`, splitting into lines -- to locate the facts you need, and reason about them in your own turn.
"""


# Under --solo-prompt this sentence replaces the one that names llm_query.
SOLO_READ_SENTENCE = (
    "Retrieved documents are long web pages — tens of thousands of characters each — so slice "
    "them and print only short, targeted extracts rather than printing them whole."
)


def build_prologue(search_budget: int = 0, no_delegation: bool = False,
                   lean: bool = False, solo: bool = False,
                   single_pass: bool = False, constraints: bool = False,
                   structured_plan: bool = False) -> str:
    """USER_PROLOGUE, plus the clauses that are explicitly switched on."""
    out = USER_PROLOGUE
    if solo:
        _i = out.find("Retrieved documents are long web pages")
        if _i != -1:
            out = out[:_i] + SOLO_READ_SENTENCE + out[out.find(".", out.find("print only short")) + 1:]
    if lean:
        # Under --seed-context the variables do not start empty.
        out = out.replace("they start empty and grow", "they grow")
    if search_budget and search_budget > 0:
        out = out + BUDGET_CLAUSE.format(n=search_budget)
    if no_delegation and not solo:
        # The solo system prompt already says there is nothing to hand work off to.
        out = out + NO_DELEGATION_CLAUSE
    if lean:
        out = out + LEAN_CLAUSE
    if single_pass:
        out = out + SINGLE_PASS_CLAUSE
    if constraints:
        out = out + CONSTRAINT_CLAUSE
    if structured_plan:
        out = out + STRUCTURED_PLAN_CLAUSE
    return out

# Markers hashed into prologue_id().
DESIGN_E = "tool-contract+grep_docs+seed"


DESIGN_F2 = "tracked-docs+unread+counter-print+facts+answer-form"

DESIGN_G = "variants+k-facts+task-facts"


# GEPA_PROMPTS: a JSON file {BLOCK_NAME: text}; only existing string blocks can be overridden.
import os as _os

_gepa_path = _os.environ.get("GEPA_PROMPTS", "")
if _gepa_path:
    import json as _json

    with open(_gepa_path, encoding="utf-8") as _f:
        _overrides = _json.load(_f)
    _applied = []
    for _k, _v in _overrides.items():
        if _k in globals() and isinstance(globals()[_k], str) and isinstance(_v, str):
            globals()[_k] = _v
            _applied.append(_k)
        else:
            raise SystemExit(f"GEPA_PROMPTS: refusing to set unknown/non-str block {_k!r}")
    print(f"[gepa] prompt overrides applied: {sorted(_applied)}", flush=True)
