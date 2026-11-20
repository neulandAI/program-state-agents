"""OBLIQ-Bench runner for the snippet REPL agent: harness.py with a ranking task and NDCG scoring.

Rebinds the harness's row accessors, task text and judge, and replaces the STRATEGY block of the system prompt; the
answer is a newline-separated list of doc_ids, scored by NDCG@10 against the qrels. Needs OBLIQ_TASK, OBLIQ_SUBSET
and OBLIQ_BENCH_ROOT.
"""
import json
import os
import sys

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_SRC_DIR)), "shared"))
sys.path.insert(0, _SRC_DIR)

from rlm_shared import browsecomp_data as BD  # noqa: E402
from rlm_shared import obliq_data as OD  # noqa: E402

TASK = os.environ.get("OBLIQ_TASK", "descriptive")
SUBSET = os.environ.get("OBLIQ_SUBSET", "twitter")
POOL = os.environ.get("OBLIQ_QRELS_POOL", "0") == "1"

_QRELS = OD.load_qrels(TASK, SUBSET, pool=POOL)
NDCG: dict[str, dict] = {}

BD.get_question = lambda row: row["text"]
BD.get_gold_answer = lambda row: ""               # no gold string in a ranking task
BD.get_gold_doc_ids = lambda row: set(_QRELS.get(row["_id"], set()))
BD.get_evidence_doc_ids = lambda row: set(_QRELS.get(row["_id"], set()))


def _score(question, response, gold):
    """Stands in for judge_match: stores the NDCG and returns whether it is above zero."""
    s = OD.score_answer(response or "", set(gold_ids_current[0]), SUBSET)
    NDCG[qid_current[0]] = s
    return s["ndcg@10"] > 0.0


BD.judge_match = _score

_TASK = """

Rank documents from the corpus that satisfy the request.

The request describes a LATENT PATTERN. A document can satisfy it without sharing any of
its words, so judge each candidate against what the request MEANS, not how it is phrased.

Before searching, identify the implicit stance: what belief is implied WITHOUT being stated,
what rhetorical move signals it (irony, mock-praise, "so much for X", selective
juxtaposition, conspicuous omission, sarcastic agreement), and what event, topic or actor is
the context.

Each hop, search a DIFFERENT angle -- vary the rhetorical device, the topic framing, the
implied target. If the top score is below 0.20 the angle is wrong: switch immediately.

Set answer["content"] to a newline-separated list of doc_ids ranked STRONGEST TO WEAKEST,
then answer["ready"] = True. Output ONLY doc_ids, one per line, no prose and no numbering.
Order matters: the score is NDCG@10, so a satisfying document at rank 1 is worth more than
the same document at rank 10. Submit more than 10 if you have them -- only the top 10 are
scored, so extra candidates cost nothing.
"""

# Per-call LM timeout (LOCAL_LM_TIMEOUT, default 1800 s; the library's is 300 s) and optional presence penalty.
import rlm as _rlm  # noqa: E402

_LM_TIMEOUT = float(os.environ.get("LOCAL_LM_TIMEOUT", "1800"))
_orig_rlm_init = _rlm.RLM.__init__


def _rlm_init_with_timeout(self, *a, **kw):
    bk = kw.get("backend_kwargs")
    if isinstance(bk, dict) and "timeout" not in bk:
        bk = dict(bk)
        bk["timeout"] = _LM_TIMEOUT
        kw["backend_kwargs"] = bk
    if os.environ.get("OBLIQ_PRESENCE_PENALTY"):
        pp = float(os.environ["OBLIQ_PRESENCE_PENALTY"])
        for key in ("sampling_args", "sub_sampling_args"):
            kw[key] = {**(kw.get(key) or {}), "presence_penalty": pp}
    return _orig_rlm_init(self, *a, **kw)


_rlm.RLM.__init__ = _rlm_init_with_timeout
print(f"[lm-timeout] per-call ceiling {_LM_TIMEOUT:.0f}s (library default 300s)", flush=True)

# The STRATEGY block of the system prompt carries BrowseComp-Plus advice; it is replaced for ranking.
import importlib.util as _ilu  # noqa: E402

_mod_spec = _ilu.spec_from_file_location(
    "_opm", os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_modules.py"))
_OPM = _ilu.module_from_spec(_mod_spec)
_mod_spec.loader.exec_module(_OPM)

OBLIQ_STRATEGY = (
    "As a general strategy, work out what the request MEANS before you search. It describes a "
    "LATENT PATTERN -- a stance, framing or rhetorical move -- so a document can satisfy it "
    "while sharing none of its words, and a document can share its words while satisfying "
    "nothing. Name the implied belief, the rhetorical device that signals it (irony, "
    "mock-praise, \"so much for X\", selective juxtaposition, conspicuous omission, sarcastic "
    "agreement) and the event or actor in view. The index is a dense semantic retriever: it "
    "matches meaning, not literal strings, and quotation marks have NO effect. Each hop, "
    "search a DIFFERENT angle -- vary the device, the framing, the implied target -- and if "
    "the top score is below 0.20 the angle is wrong, so switch rather than reword. You are "
    "not looking for one answer: you are building an ORDERED LIST, scored by NDCG@10, so keep "
    "every plausible candidate and spend your effort deciding which of them instantiate the "
    "pattern most strongly."
)

import harness as H  # noqa: E402

# OBLIQ_FENCE_TOLERANT=1: run a mislabelled code fence only if it parses as Python and acts.
if os.environ.get("OBLIQ_FENCE_TOLERANT", "") == "1":
    import ast as _ast
    import re as _re2
    import rlm.core.rlm as _RC  # patched where it is called

    _orig_fcb = _RC.find_code_blocks
    _FENCE_RE = _re2.compile(r"```([A-Za-z0-9_+-]*)[ \t]*\n(.*?)\n```", _re2.DOTALL)
    _DOES_SOMETHING = (_ast.Assign, _ast.AnnAssign, _ast.AugAssign, _ast.Call,
                       _ast.Import, _ast.ImportFrom, _ast.For, _ast.While, _ast.If)

    def _tolerant_find_code_blocks(text):
        blocks = _orig_fcb(text) if text else []
        if blocks or not text:
            return blocks
        out = []
        for lang, body in _FENCE_RE.findall(text):
            if lang.lower() == "repl":
                continue
            b = body.strip()
            if not b:
                continue
            try:
                tree = _ast.parse(b)
            except SyntaxError:
                continue
            if not any(isinstance(n, _DOES_SOMETHING) for n in _ast.walk(tree)):
                continue
            out.append(b)
        return out

    _RC.find_code_blocks = _tolerant_find_code_blocks
    print("[fence] tolerant code-block parsing ON (```repl still preferred)", flush=True)

# OBLIQ_STRATEGY_FILE replaces the STRATEGY block; the other blocks stay fixed.
_strategy_file = os.environ.get("OBLIQ_STRATEGY_FILE", "")
if _strategy_file:
    with open(_strategy_file, encoding="utf-8") as _sf:
        OBLIQ_STRATEGY = _sf.read().strip()

# OBLIQ_RELATIVE_SCORES=1: no fixed score cutoffs in the strategy or the task contract.
from rlm_shared.obliq_score_advice import ENABLED as _REL_SCORES, relative as _relative  # noqa: E402
if _REL_SCORES:
    OBLIQ_STRATEGY = _relative(OBLIQ_STRATEGY)
    _TASK = _relative(_TASK)
    print("[prompt] score cutoffs replaced by relative-relevance advice (OBLIQ_RELATIVE_SCORES=1)", flush=True)

# GPTOSS_FINAL_CUE=1: ask gpt-oss models to put the code in their final response channel.
_FINAL_CUE = (" Put the python block in your FINAL response channel, never in your "
              "analysis: a turn whose code never reaches the final channel is a lost turn.")

if os.environ.get("OBLIQ_KEEP_BROWSECOMP_STRATEGY", "") != "1":
    _OPM.assert_roundtrip()
    _blocks = _OPM.split()
    _blocks["STRATEGY"] = OBLIQ_STRATEGY
    if os.environ.get("OBLIQ_REPL_CUE", "") == "1":
        _blocks["TURN_RULE"] = _blocks["TURN_RULE"].rstrip() + (
            " The fence label matters: write ```repl, never ```python and never a bare ```. "
            "A turn whose code is not in a ```repl``` block is discarded and its work is "
            "lost -- including the final turn, where answer[\"content\"] and "
            "answer[\"ready\"] = True must be assigned inside ```repl```.")
        print("[cue] repl-fence cue appended to TURN_RULE", flush=True)
    if os.environ.get("GPTOSS_FINAL_CUE", "") == "1":
        _blocks["TURN_RULE"] = _blocks["TURN_RULE"].rstrip() + _FINAL_CUE
        print(f"[cue] harmony final-channel cue appended to TURN_RULE "
              f"(+{len(_FINAL_CUE)} chars)", flush=True)
    _OPM.apply(_blocks)
    print(f"[strategy] BrowseComp+ block swapped for OBLIQ ({len(OBLIQ_STRATEGY)} chars)", flush=True)


H.get_question = BD.get_question
H.get_gold_answer = BD.get_gold_answer
H.get_gold_doc_ids = BD.get_gold_doc_ids
H.get_evidence_doc_ids = BD.get_evidence_doc_ids
H.judge_match = _score
# OBLIQ_TASK_FILE replaces the answer contract.
_TASK_FILE = os.environ.get("OBLIQ_TASK_FILE")
if _TASK_FILE:
    _TASK = open(_TASK_FILE).read()
    if _REL_SCORES:
        _TASK = _relative(_TASK)
    print(f"[task] answer contract overridden from {_TASK_FILE} ({len(_TASK)} chars)", flush=True)
H.TASK_SUFFIX = _TASK

# OBLIQ_NO_FORMAT_GATE=1 disables the harness's Explanation/Exact Answer gate.
if os.environ.get("OBLIQ_NO_FORMAT_GATE") == "1":
    H.MAX_REJECTS = 0
    print("[gate] BrowseComp+ Explanation/Exact Answer submission gate OFF", flush=True)

qid_current = [None]
gold_ids_current = [set()]
tag_current = [""]

# OBLIQ_TAG_SEARCH=1 adds a per-question {"tag": ...} to every /search request.
if os.environ.get("OBLIQ_TAG_SEARCH") == "1":
    import types as _types
    import uuid as _uuid
    _real_requests = H.requests

    def _tagged_post(url, *a, json=None, **kw):
        if url.endswith("/search") and isinstance(json, dict):
            json = dict(json, tag=tag_current[0])
        return _real_requests.post(url, *a, json=json, **kw)

    H.requests = _types.SimpleNamespace(post=_tagged_post, get=_real_requests.get)
    print("[tag] /search requests carry the question tag", flush=True)

_orig_run_one = H.run_one


def run_one(row, q_idx, args):
    qid_current[0] = row["_id"]
    if os.environ.get("OBLIQ_TAG_SEARCH") == "1":
        tag_current[0] = f"{row['_id']}-{_uuid.uuid4().hex[:8]}"
    gold_ids_current[0] = set(_QRELS.get(row["_id"], set()))
    rec = _orig_run_one(row, q_idx, args)
    rec["query_id"] = row["_id"]
    rec.update({k: v for k, v in (NDCG.get(row["_id"]) or {}).items()
                if k.startswith(("ndcg@", "recall@")) or k == "n_parsed"})
    print(f"    [{row['_id']}] ndcg@10={rec.get('ndcg@10', 0):.4f} "
          f"recall@10={rec.get('recall@10', 0):.4f} parsed={rec.get('n_parsed', 0)}", flush=True)
    return rec


H.run_one = run_one


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=20)
    p.add_argument("--max-iterations", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=1)
    p.add_argument("--compaction", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--ids-file", default=None,
                   help="json list of query ids (the dev/held-out split); overrides --start/--end")
    args = p.parse_args()

    rows = OD.load_queries(TASK, SUBSET)
    if args.ids_file:
        want = set(json.load(open(args.ids_file)))
        rows = [r for r in rows if r["_id"] in want]
        sel = list(enumerate(rows))
    else:
        sel = [(i, rows[i]) for i in range(args.start, min(args.end, len(rows)))]
    # Checkpoint after every question and resume from it.
    from pathlib import Path
    ckpt = Path(f"{args.out}_checkpoint.json")
    results = json.loads(ckpt.read_text()) if ckpt.exists() else []
    done = {r.get("_id") or r.get("query_id") for r in results}
    for i, row in sel:
        if row["_id"] in done:
            continue
        results.append(run_one(row, i, args))
        ckpt.write_text(json.dumps(results, indent=2))
    Path(f"{args.out}.json").write_text(json.dumps(results, indent=2))
    n = len(results)
    if n:
        print(f"\n{args.out}: n={n}  "
              f"NDCG@10={sum(r.get('ndcg@10', 0) for r in results)/n:.4f}  "
              f"recall@10={sum(r.get('recall@10', 0) for r in results)/n:.4f}")


if __name__ == "__main__":
    main()
