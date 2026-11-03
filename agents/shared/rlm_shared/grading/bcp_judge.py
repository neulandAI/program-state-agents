"""BrowseComp-Plus gpt-4.1 judge used by the graders in this package.

Templates and parse_judge come from the document REPL agent's judge.py. Each answer gets 3 votes (majority of the
parsed votes); an empty answer is wrong without judging; a question whose votes all fail is retried; a prompt blocked
by the content filter returns "filtered". Verdicts are cached on (template, judge model, question, normalised answer).
JUDGE_TEMPLATE selects "official" (default, the BrowseComp-Plus grader) or the stricter "paper" template.

Importing this module reads the AZ_ENV file, loads and decrypts the Tevatron/browsecomp-plus test split, and builds
the Azure client; no request is sent until grade_outputs is called.
"""
import hashlib, json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]   # agents/
FULLDOC_SRC = _REPO / "document_repl" / "src"
sys.path.insert(0, str(FULLDOC_SRC)); sys.path.insert(0, str(_REPO / "shared"))
import openai  # noqa: E402
from judge import GRADER_TEMPLATE, OFFICIAL_TEMPLATE, parse_judge  # noqa: E402
from datasets import load_dataset  # noqa: E402
from rlm_shared.browsecomp_data import get_question, get_gold_answer  # noqa: E402
from rlm_shared.paths import path, setting  # noqa: E402

_env = {}
for _l in open(path("AZ_ENV")):
    _k, _, _v = _l.strip().partition("=")
    _env[_k] = _v
GT = {str(r["query_id"]): (get_question(r), get_gold_answer(r)) for r in load_dataset("Tevatron/browsecomp-plus", split="test")}
CLI = openai.AzureOpenAI(azure_endpoint=_env["AZ_EP"] + "/", api_key=_env["AZ_KEY"], api_version="2024-12-01-preview",
                         timeout=180, max_retries=10)
_LK = threading.Lock()
# MODEL is part of the verdict-cache key.
MODEL = os.environ["JUDGE_MODEL"] if "JUDGE_MODEL" in os.environ else setting("AZURE_DEPLOYMENT_JUDGE")
TEMPLATE = os.environ.get("JUDGE_TEMPLATE", "official")
TMPL = OFFICIAL_TEMPLATE if TEMPLATE == "official" else GRADER_TEMPLATE


def _grade_one(q, out, cache):
    if not out.strip():
        return q, False
    key = hashlib.sha256(f"{TEMPLATE}|{MODEL}|{q}|{' '.join(out.lower().split())}".encode()).hexdigest()[:24]
    with _LK:
        if key in cache:
            return q, cache[key]
    prompt = TMPL.format(question=GT[q][0], response=out, correct_answer=GT[q][1])
    # 3 votes, each retried up to 3 times on an exception or an unparsable reply.
    votes = []
    for _ in range(3):
        for _try in range(3):
            try:
                r = CLI.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}],
                                                max_completion_tokens=2048)
                v = parse_judge(r.choices[0].message.content or "")
                if not v.get("parse_error"):
                    votes.append(v.get("correct") == "yes"); break
            except Exception as e:  # noqa: BLE001
                if "content management policy" in str(e) or "content_filter" in str(e):
                    return q, "filtered"
                print(f"[judge] {q}: {type(e).__name__} {str(e)[:200]}", file=sys.stderr, flush=True)
    if not votes:
        return q, None
    ok = sum(votes) * 2 > len(votes)
    with _LK:
        cache[key] = ok
    return q, ok


def grade_outputs(outputs, cache_path, workers=32):
    """outputs: {query_id: answer text ('' = no answer)} -> {query_id: correct}"""
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    with ThreadPoolExecutor(workers) as ex:
        res = dict(ex.map(lambda kv: _grade_one(kv[0], kv[1], cache), outputs.items()))
    for _round in range(5):                      # questions whose votes all failed: retry, slower
        bad = [q for q, v in res.items() if v is None]
        if not bad:
            break
        print(f"[judge] retrying {len(bad)} questions with no vote", file=sys.stderr, flush=True)
        with ThreadPoolExecutor(4) as ex:
            res.update(ex.map(lambda q: _grade_one(q, outputs[q], cache), bad))
    bad = [q for q, v in res.items() if v is None]
    if bad:
        json.dump(cache, open(cache_path, "w"))
        raise RuntimeError(f"no judge vote for {len(bad)} questions: {bad[:10]}")
    json.dump(cache, open(cache_path, "w"))
    return res


def run_outputs(run_dir):
    """{query_id: answer text} for a run directory of run_one.py results; non-completed runs count as ''."""
    o = {}
    for f in os.listdir(run_dir):
        if f.endswith(".json") and not f.startswith("_"):
            x = json.load(open(f"{run_dir}/{f}"))
            out = ((x.get("result") or [{}])[0] or {}).get("output", "") or ""
            o[str(x["query_id"])] = out if x.get("status") == "completed" else ""
    return o
