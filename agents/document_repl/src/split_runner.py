"""Batch runner of the Qwen3.6 document REPL agent runs on BrowseComp-Plus, with a GEPA prompt-optimisation driver.

    python src/split_runner.py --seed-split <split>    # run the seed prompt on one split of GEPA_SPLIT, then exit
    python src/split_runner.py [--max-metric-calls N --minibatch b --workers W] [--preload-seed|--dry-run|--smoke N]

The paper's Qwen3.6 runs use the seed (unchanged) prompt through --seed-split or run_batch() (bcp_qwen_dense.py,
bcp_bm25_split.py); no reported run uses an optimised prompt. Each rollout is `$PYTHON src/run_one.py ... BASE_ARGS`
with PYTHONPATH=$RLMLIB_PATH and cwd FULLDOC_ROOT, graded with grade_local.judge. Results are cached in
<GEPA_OUT>/cache.json under "<ckey>|<qid>", where ckey() hashes the four prompt blocks and FINGERPRINT (model,
BASE_ARGS, judge, switches). GEPA mode needs the `gepa` package and a critic (CLAUDE_BIN, or GEPA_REFLECTION=qwen);
its score is correct + LAMBDA * gold recall, and infrastructure failures are imputed with the candidate's mean.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time as _t
import uuid
from concurrent.futures import ThreadPoolExecutor

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SRC, os.pardir, os.pardir, "shared"))
sys.path.insert(0, SRC)
from rlm_shared.paths import path  # noqa: E402

RUNS = str(path("RUNS_DIR"))
# Interpreter of the rollouts.
PYV = os.environ.get("PYTHON", "")
CLAUDE = os.environ.get("CLAUDE_BIN", "claude")

SERVERS = [s.strip() for s in os.environ.get(
    "GEPA_BCP_SERVERS", "http://localhost:8030/v1").split(",") if s.strip()]
os.environ.setdefault("LOCAL_JUDGE_BASE_URL", SERVERS[0])
RETRIEVAL = os.environ.get("GEPA_BCP_RETRIEVAL", "http://127.0.0.1:8766")
BASELINE_MODEL = "qwen3.8-27b-bf16"   # the model of the 830-run cache (splits, preload, null guard)
ROOT_MODEL = os.environ.get("GEPA_BCP_MODEL", BASELINE_MODEL)
# the judge is the served agent model (grade_local reads LOCAL_JUDGE_MODEL at import)
os.environ.setdefault("LOCAL_JUDGE_MODEL", ROOT_MODEL)
FENCE_TOLERANT = os.environ.get("SUP_FENCE_TOLERANT", "") == "1"
MAX_BLOCKS = int(os.environ.get("SUP_MAX_BLOCKS", "0") or 0)   # src/fence_tolerant.py
PRESENCE_PENALTY = os.environ.get("SUP_PRESENCE_PENALTY", "")   # src/run_one.py _sampling()
BLOCKS = ["SOLO_SYSTEM_PROMPT", "USER_PROLOGUE", "LEAN_CLAUSE", "QUERY_TEMPLATE"]
LAMBDA = 0.5
SCORED = ("completed", "empty_output")   # statuses scored as agent outcomes, not infra
# GEPA_TIMEOUT_WRONG=1: harness timeouts are scored wrong instead of imputed (not part of FINGERPRINT).
TIMEOUT_WRONG = os.environ.get("GEPA_TIMEOUT_WRONG", "") == "1"


def scored(r) -> bool:
    if not r:
        return False
    if r.get("status") in SCORED:
        return True
    return TIMEOUT_WRONG and "Timeout" in str(r.get("error") or "")

# --base-url is set per worker.
BASE_ARGS = [
    "--design", "B", "--server", RETRIEVAL, "--backend", "openai",
    "--api-key-env", "ALIBABA_TOKEN", "--root-model", ROOT_MODEL, "--sub-model", ROOT_MODEL,
    "--reasoning-effort", "medium", "--sub-reasoning-effort", "medium",
    "--max-iterations", "30", "--max-searches", "10", "--seed-context", "--compaction",
    "--grounding-gate", "--lean-prologue", "--no-delegation", "--solo-prompt",
    "--no-orchestrator", "--compaction-threshold-pct", "0.85", "--max-concurrent-subcalls", "4",
    "--max-timeout", "5400", "--lm-timeout", "1800", "--save-trajectory",
]
HARD_CAP_S = 5400 + 900   # subprocess kill: harness timeout plus grace for the final write
FINGERPRINT = hashlib.sha1(json.dumps(
    {"model": ROOT_MODEL, "args": BASE_ARGS, "judge": os.environ["LOCAL_JUDGE_MODEL"],
     "lambda": LAMBDA, "fence_tolerant": FENCE_TOLERANT,
     **({"max_blocks": MAX_BLOCKS} if MAX_BLOCKS else {}),
     **({"presence_penalty": PRESENCE_PENALTY} if PRESENCE_PENALTY else {}),
     **({"rep_guard": "1"} if os.environ.get("SUP_REP_GUARD") == "1" else {})}).encode()).hexdigest()[:8]   # src/rep_guard.py

# Absolute, because rollouts run with cwd=FULLDOC_ROOT; a relative GEPA_OUT is resolved against RUNS_DIR.
OUT = os.path.abspath(os.path.join(RUNS, os.environ.get("GEPA_OUT", f"{RUNS}/gepa_bcp")))
PROG = f"{OUT}/progress.txt"
CACHE_F = f"{OUT}/cache.json"
SPLIT_F = os.environ.get("GEPA_SPLIT") or (
    os.path.join(os.environ["RLM_DATA_DIR"], "bcp_gepa_splits", "split.json") if os.environ.get("RLM_DATA_DIR") else "")


def _split_file() -> str:
    """GEPA_SPLIT, read where it is used (no machine-specific default)."""
    if not SPLIT_F:
        raise SystemExit("set GEPA_SPLIT to the split file (bcp_gepa_splits/split.json), or RLM_DATA_DIR")
    return SPLIT_F
_CACHE: dict[str, dict] = {}
_LOCK = threading.Lock()
_SEED_SCORE = [0.0]
_DRY = [False]
_SEED_KEY = [""]


def _log(msg: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    line = f"{_t.strftime('%F %T')}  {msg}"
    with open(PROG, "a") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def ckey(blocks: dict) -> str:
    s = json.dumps({b: blocks[b] for b in BLOCKS}, sort_keys=True) + FINGERPRINT
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def _cache_load():
    global _CACHE
    if os.path.exists(CACHE_F):
        _CACHE = json.load(open(CACHE_F))
        _log(f"[cache] loaded {len(_CACHE)} records")


def _cache_save():
    os.makedirs(OUT, exist_ok=True)
    tmp = CACHE_F + ".tmp"
    with _LOCK:
        json.dump(_CACHE, open(tmp, "w"))
        os.replace(tmp, CACHE_F)


_ROWS: dict[str, dict] = {}


def rows() -> dict[str, dict]:
    if not _ROWS:
        from datasets import load_dataset
        for r in load_dataset("Tevatron/browsecomp-plus", split="test"):
            _ROWS[str(r["query_id"])] = r
    return _ROWS


def seed_blocks() -> dict[str, str]:
    import prompts as P   # src/prompts.py, GEPA_PROMPTS unset -> the baseline text
    assert not os.environ.get("GEPA_PROMPTS"), "unset GEPA_PROMPTS before reading the seed"
    return {b: getattr(P, b) for b in BLOCKS}


_FIELD = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}(?!\})")


_REQUIRED = {"SOLO_SYSTEM_PROMPT": "\n\n{custom_tools_section}", "QUERY_TEMPLATE": "\n\nQuestion: {Question}"}


def repair(cand: dict) -> tuple[dict, list[str]]:
    """Re-append a required placeholder the critic dropped; the repaired text is what runs and is hashed."""
    c, fixed = dict(cand), []
    for b, tail in _REQUIRED.items():
        ph = tail.strip().split(": ")[-1]
        if isinstance(c.get(b), str) and ph not in c[b]:
            c[b] = c[b].rstrip() + tail; fixed.append(f"{b}+{ph}")
    return c, fixed


def validate(cand: dict, seed: dict) -> str | None:
    """None if the candidate can run; otherwise a reason (which is a real prompt defect)."""
    for b in BLOCKS:
        v = cand.get(b)
        if not isinstance(v, str) or not v.strip():
            return f"{b} is empty"
        if set(_FIELD.findall(v)) != set(_FIELD.findall(seed[b])):
            return (f"{b} must keep exactly the placeholders "
                    f"{sorted(set(_FIELD.findall(seed[b])))}, got {sorted(set(_FIELD.findall(v)))}")
        if len(v) > 3 * len(seed[b]) + 1500:
            return f"{b} grew to {len(v)} chars (seed {len(seed[b])}); bloat is rejected"
    try:
        cand["QUERY_TEMPLATE"].format(Question="Q")
        cand["SOLO_SYSTEM_PROMPT"].format(custom_tools_section="")
    except Exception as e:
        return f"str.format fails ({type(e).__name__}: {e}); literal braces must be doubled"
    for marker in ("Exact Answer:", "Explanation:"):
        if marker not in cand["QUERY_TEMPLATE"]:
            return f"QUERY_TEMPLATE dropped the answer marker {marker!r} the grader parses"
    return None


def _grade(qid: str, raw: dict) -> dict:
    from rlm_shared.browsecomp_data import get_question, get_gold_answer, get_gold_doc_ids
    import grade_local
    row = rows()[qid]
    resp = (raw.get("result") or [{}])[0].get("output", "") or ""
    status = raw.get("status")
    gids = set(map(str, get_gold_doc_ids(row)))
    got = set(map(str, raw.get("server_docids") or raw.get("retrieved_docids") or []))
    ok = status == "completed" and bool(resp) and grade_local.judge(
        get_question(row), resp, get_gold_answer(row))
    return {"query_id": qid, "status": status, "error": raw.get("error"),
            "correct": bool(ok), "gold_recall": (len(gids & got) / len(gids)) if gids else 0.0,
            "missed_gold": sorted(gids - got)[:6],
            "searches": (raw.get("tool_call_counts") or {}).get("search", 0),
            "queries": [str(x.get("query", "") if isinstance(x, dict) else x)[:120] for x in (raw.get("search_log") or [])][:10],
            "wall": raw.get("wall_time_s"), "iters": raw.get("n_iterations"),
            "answer": resp[-400:]}


def _rollout(qid: str, pfile: str, server: str, out_dir: str) -> dict | None:
    out = os.path.join(out_dir, f"{qid}.json")
    if not PYV:
        raise SystemExit("environment variable PYTHON is not set: interpreter of the rollouts")
    # No startup hooks in the rollouts.
    env = dict(os.environ, GEPA_PROMPTS=pfile, PYTHONPATH=str(path("RLMLIB_PATH")),
               HF_HOME=os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), ALIBABA_TOKEN="EMPTY")
    try:
        subprocess.run([PYV, f"{SRC}/run_one.py", "--query-id", qid, "--out", out,
                        "--base-url", server] + BASE_ARGS,
                       env=env, cwd=str(path("FULLDOC_ROOT")), capture_output=True, timeout=HARD_CAP_S)
    except subprocess.TimeoutExpired:
        _log(f"[runaway] q{qid} killed at {HARD_CAP_S}s")
        return None
    try:
        return json.load(open(out))
    except Exception:
        return None


def run_batch(cand: dict, qids: list[str], workers: int) -> dict[str, dict | None]:
    key = ckey(cand)
    # A "killed" record is a placeholder for an unfinished rollout: re-run it.
    todo = [q for q in qids if (_CACHE.get(f"{key}|{q}") or {}).get("status") in (None, "killed")]
    if todo and _DRY[0]:
        # Dry run: every candidate gets the seed's cached records (fake scores).
        for q in todo:
            _CACHE[f"{key}|{q}"] = dict(_CACHE.get(f"{_SEED_KEY[0]}|{q}") or {}, _fake=True)
        todo = []
    if todo:
        pfile = os.path.join(tempfile.gettempdir(), f"gepa_bcp_{key}_{uuid.uuid4().hex[:6]}.json")
        json.dump({b: cand[b] for b in BLOCKS}, open(pfile, "w"))
        out_dir = os.path.join(OUT, "rollouts", key)
        os.makedirs(out_dir, exist_ok=True)

        def one(i_q):
            i, q = i_q
            raw = _rollout(q, pfile, SERVERS[i % len(SERVERS)], out_dir)
            rec = _grade(q, raw) if raw else {"query_id": q, "status": "killed", "correct": False,
                                              "gold_recall": 0.0}
            with _LOCK:
                _CACHE[f"{key}|{q}"] = rec
            _cache_save()

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(one, enumerate(todo)))
    return {q: _CACHE.get(f"{key}|{q}") for q in qids}


def score(rec: dict) -> float:
    return (1.0 if rec.get("correct") else 0.0) + LAMBDA * (rec.get("gold_recall") or 0.0)


def _ndcg10(ranked: list[str], rel: set[str]) -> float:
    import math
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:10]) if d in rel)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(10, len(rel))))
    return dcg / idcg if idcg else 0.0


def search_stats(key: str, qid: str) -> str | None:
    """Per-search nDCG of the agent's own queries (seed search excluded), shown to the critic, not scored."""
    from rlm_shared.browsecomp_data import get_gold_doc_ids
    f = os.path.join(OUT, "rollouts", key, f"{qid}.json")
    try:
        log = (json.load(open(f)).get("search_log") or [])[1:]
    except Exception:
        return None
    if not log:
        return "Agent-written searches: 0 (only the seed search ran)"
    gold = set(map(str, get_gold_doc_ids(rows()[qid])))
    nd = [_ndcg10([str(x) for x in (s.get("docids") or [])], gold) for s in log]
    hit = sum(1 for s in log if gold & set(map(str, s.get("docids") or [])))
    return (f"Agent-written searches: {len(log)} of 9 available after the seed search | "
            f"per-search nDCG@10 vs gold: mean {sum(nd)/len(nd):.2f}, best {max(nd):.2f} | "
            f"{hit} of them returned at least one gold document")


def feedback(rec: dict, question: str, key: str | None = None) -> str:
    v = "CORRECT" if rec.get("correct") else "WRONG"
    if "Timeout" in str(rec.get("error") or ""):
        v = (f"WRONG -- TIMED OUT after {rec.get('searches', 0)} searches without submitting. "
             + ("Typical cause on this task: once all 10 searches are spent the agent keeps "
                "brainstorming candidate names ('What about X? What about Y?') instead of committing; "
                "the prompt should make it submit its best-supported answer when searches run out."
                if (rec.get("searches") or 0) >= 10 else ""))
    if rec.get("status") == "empty_output":
        v = ("WRONG -- EMPTY SUBMISSION: the agent set answer['ready']=True without setting "
             "answer['content'] (it printed its answer instead), so nothing was submitted")
    parts = [f"Question: {question[:600]}",
             f"Verdict: {v} | gold recall {rec.get('gold_recall', 0):.2f} | "
             f"{rec.get('searches', 0)} searches (cap 10) | {rec.get('iters')} iterations | "
             f"{(rec.get('wall') or 0):.0f}s"]
    if rec.get("missed_gold"):
        parts.append(f"Gold documents never retrieved: {len(rec['missed_gold'])} "
                     f"(ids {rec['missed_gold'][:4]})")
    ss = search_stats(key, str(rec.get("query_id"))) if key else None
    if ss:
        parts.append(ss)
    if rec.get("queries"):
        parts.append("Queries issued: " + " || ".join(rec["queries"]))
    if rec.get("answer"):
        parts.append("End of the agent's final response: " + rec["answer"][-300:].replace("\n", " "))
    return "\n".join(parts)


def make_evaluator(workers: int, seed: dict, null_ref: list[float]):
    st = {"batches": 0, "rollouts": 0, "t0": _t.time()}

    def evaluator(pairs, **kw):
        groups: dict[str, list[int]] = {}
        cands: dict[str, dict] = {}
        for i, (cand, _ex) in enumerate(pairs):
            k = ckey(cand); groups.setdefault(k, []).append(i); cands[k] = cand
        out: list = [None] * len(pairs)
        for k0, idxs in groups.items():
            cand, fixed = repair(cands[k0])
            k = ckey(cand)
            if fixed:
                _log(f"[repair] cand {k0} -> {k}: re-appended {fixed}")
            qids = [str(pairs[i][1]["qid"]) for i in idxs]
            bad = validate(cand, seed)
            if bad:
                _log(f"[invalid] cand {k}: {bad} -- scored 0 without rollouts")
                for i in idxs:
                    out[i] = (0.0, {"feedback": f"CANDIDATE REJECTED BEFORE RUNNING: {bad}"})
                continue
            n_cached = sum(1 for q in qids if f"{k}|{q}" in _CACHE)
            recs = run_batch(cand, qids, workers)
            # Only infrastructure failures are imputed; "empty_output" is the agent's own failure.
            okr = {q: r for q, r in recs.items() if scored(r)}
            impute = statistics.mean([score(r) for r in okr.values()]) if okr else _SEED_SCORE[0]
            accs = []
            for i in idxs:
                q = str(pairs[i][1]["qid"]); r = okr.get(q)
                if r is None:
                    out[i] = (impute, {"feedback": "RUN DID NOT COMPLETE (timeout/infra). "
                                                   "Not attributable to the prompt.",
                                       "infra_failure": True})
                else:
                    from rlm_shared.browsecomp_data import get_question
                    out[i] = (score(r), {"feedback": feedback(r, get_question(rows()[q]), k)})
                    accs.append(1.0 if r.get("correct") else 0.0)
            st["batches"] += 1; st["rollouts"] += len(idxs) - n_cached
            acc = statistics.mean(accs) if accs else 0.0
            sc = statistics.mean(o[0] for o in (out[i] for i in idxs))
            beat = sum(1 for x in null_ref if sc > x) / max(1, len(null_ref))
            _log(f"batch {st['batches']:>3} | cand {k} | n={len(idxs)} cached={n_cached} | "
                 f"acc={acc:.3f} score={sc:.4f} | vs NULL(n=100): beats {beat*100:.0f}% | "
                 f"infra {len(idxs)-len(okr)} | rollouts {st['rollouts']} | "
                 f"{(_t.time()-st['t0'])/60:.0f} min{' | DRY-RUN (fake scores)' if _DRY[0] else ''}")
        return out

    return evaluator


CRITIC_SYSTEM = (
    "You are the reflection step of GEPA, an evolutionary prompt optimiser. The system being "
    "optimised is a deep-research agent (" + ROOT_MODEL + " in a Python REPL with a search tool over "
    "a fixed 100k-document web corpus) answering BrowseComp-Plus questions. You will be shown "
    "ONE prompt component, the agent's results on a few questions and feedback. Propose an "
    "improved version of that component. Rules: (1) Write GENERAL strategy that transfers to "
    "unseen questions -- never copy entities, facts, answers, queries or document ids from the "
    "examples into the prompt. (2) Keep every {placeholder} exactly as it appears, and keep "
    "literal braces doubled ({{ }}). In particular SOLO_SYSTEM_PROMPT MUST still contain the "
    "literal text {custom_tools_section} (the tool documentation is inserted there) and "
    "QUERY_TEMPLATE MUST still contain {Question}. (3) Keep the 'Explanation:' / 'Exact Answer:' / "
    "'Confidence:' response format if the component contains it. (4) Stay concise: the agent "
    "re-reads this text every turn; do not more than double its length. (5) Do not make the agent "
    "search LESS: measured on this task, every rewrite that framed searches as scarce or told the "
    "agent to stop early cut its searches (8.1 -> ~7.4 per question) and lowered gold recall even "
    "though each query got sharper. Encourage spending the full search budget on DIFFERENT angles. "
    "(6) Reply with the "
    "complete new component inside exactly ONE ``` fenced block, nothing after it. "
    "Feedback metrics: gold recall = share of the question's gold documents the agent ever "
    "retrieved; per-search nDCG@10 = how well each of its own queries ranked gold documents "
    "(mean and best over its searches); the score you are optimised on is correct + 0.5 x "
    "gold recall, so query precision only helps if the gold documents end up retrieved.")


_FENCE_LINE = re.compile(r"\s*```[A-Za-z0-9_-]*\s*")


def _sanitise(raw: str) -> str:
    """Keep the last block delimited by fence-only lines (the prompt blocks contain inline fences)."""
    lines = raw.splitlines()
    idx = [i for i, l in enumerate(lines) if _FENCE_LINE.fullmatch(l)]
    body = "\n".join(lines[idx[-2] + 1: idx[-1]]).strip() if len(idx) >= 2 else raw.strip()
    return "```\n" + body + "\n```"


def _claude(prompt: str) -> tuple[str, float]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDECODE", "CLAUDE_CODE_"))}
    model = os.environ.get("GEPA_REFLECTION_MODEL", "claude-opus-5-5")
    p = subprocess.run([CLAUDE, "-p", "--model", model, "--output-format", "json", "--tools", "",
                        "--no-session-persistence", "--system-prompt", CRITIC_SYSTEM],
                       input=prompt, capture_output=True, text=True, timeout=900, env=env,
                       cwd=tempfile.gettempdir())
    d = json.loads(p.stdout)
    if d.get("is_error") or not d.get("result"):
        raise RuntimeError(f"claude -p error: {str(d)[:300]}")
    return d["result"], float(d.get("total_cost_usd") or 0.0)


def _qwen(prompt: str) -> str:
    from openai import OpenAI
    c = OpenAI(base_url=SERVERS[0], api_key="EMPTY", timeout=1800)
    r = c.chat.completions.create(
        model=ROOT_MODEL, temperature=1.0, max_completion_tokens=8192,
        messages=[{"role": "system", "content": CRITIC_SYSTEM}, {"role": "user", "content": prompt}],
        extra_body={"reasoning_effort": "medium"})
    return r.choices[0].message.content or ""


def reflection_lm(prompt: str) -> str:
    who, cost, raw = "qwen", 0.0, ""
    if os.environ.get("GEPA_REFLECTION", "claude") == "claude":
        for attempt in (1, 2):
            try:
                raw, cost = _claude(prompt); who = "claude"; break
            except Exception as e:
                _log(f"[critic] claude attempt {attempt} failed: {str(e)[:200]}")
        if who != "claude":
            _log("[critic] FALLING BACK TO LOCAL QWEN for this reflection step")
    if who == "qwen":
        raw = _qwen(prompt)
    with open(f"{OUT}/reflection_log.jsonl", "a") as fh:
        fh.write(json.dumps({"t": _t.strftime('%F %T'), "critic": who, "cost_usd": cost,
                             "prompt_chars": len(prompt), "reply": raw}) + "\n")
    _log(f"[critic] {who} reflection: {len(prompt)} ch in, {len(raw)} ch out, ${cost:.3f}")
    return _sanitise(raw)


def preload_seed(seed_key: str) -> int:
    """Seed records from the 830 run (used by --dry-run and --preload-seed)."""
    import glob
    SUP = str(path("FULLDOC_ROOT"))
    G = json.load(open(f"{SUP}/runs/_graded_cache.json"))
    raws = {}
    for d in ("qwen27b_colbert8192_first20", "qwen27b_colbert8192_rest810"):
        for f in glob.glob(f"{SUP}/runs/{d}/*.json"):
            if not os.path.basename(f).startswith("_"):
                raws[os.path.basename(f)[:-5]] = f
    n = 0
    for q, g in G.items():
        raw = json.load(open(raws[q])) if q in raws else {}
        rec = {"query_id": q, "status": g["status"], "correct": g["correct"],
               "gold_recall": g["gold_recall"], "searches": g.get("searches"),
               "iters": g.get("iters"), "wall": g.get("wall"),
               "queries": [str(x.get("query", "") if isinstance(x, dict) else x)[:120] for x in (raw.get("search_log") or [])][:10],
               "answer": ((raw.get("result") or [{}])[0].get("output", "") or "")[-400:],
               "_from": "830-run cache"}
        if raw and q in rows():
            from rlm_shared.browsecomp_data import get_gold_doc_ids
            got = set(map(str, raw.get("server_docids") or raw.get("retrieved_docids") or []))
            rec["missed_gold"] = sorted(set(map(str, get_gold_doc_ids(rows()[q]))) - got)[:6]
        _CACHE.setdefault(f"{seed_key}|{q}", rec); n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-metric-calls", type=int, default=800)
    ap.add_argument("--minibatch", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--preload-seed", action="store_true",
                    help="reuse the 830-run seed records instead of re-running the seed on D_pareto")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed-split", default="",
                    help="run the SEED on every question of this split (e.g. feedback), grade, report, exit")
    ap.add_argument("--smoke", type=int, default=0,
                    help="run the SEED on the first N D_pareto questions for real, grade, print, exit")
    a = ap.parse_args()
    _DRY[0] = a.dry_run
    if a.seed_split:
        _cache_load()
        seed = seed_blocks()
        qids = json.load(open(_split_file()))[a.seed_split]
        _log(f"[seed-split] seed on '{a.seed_split}' ({len(qids)} q, {sum(1 for q in qids if f'{ckey(seed)}|{q}' in _CACHE)} cached)")
        recs = run_batch(seed, qids, a.workers)
        ok = [r for r in recs.values() if r and r.get("status") in SCORED]
        _log(f"[seed-split] '{a.seed_split}': n={len(recs)} scored={len(ok)} acc="
             f"{statistics.mean(1.0 if (r and r.get('correct')) else 0.0 for r in recs.values()):.3f}")
        sys.exit(0)
    if a.smoke:
        # A few real rollouts through the same path GEPA uses.
        global OUT, PROG, CACHE_F
        OUT = f"{RUNS}/gepa_bcp_smoke"; PROG = f"{OUT}/progress.txt"; CACHE_F = f"{OUT}/cache.json"
        seed = seed_blocks()
        qids = json.load(open(_split_file()))["pareto"][: a.smoke]
        recs = run_batch(seed, qids, min(a.workers, len(qids)))
        bad = [q for q, r in recs.items() if not r or r.get("status") != "completed"]
        for q, r in recs.items():
            _log(f"[smoke] q{q}: status={r and r.get('status')} correct={r and r.get('correct')} "
                 f"gold_recall={r and r.get('gold_recall')} searches={r and r.get('searches')} "
                 f"wall={r and r.get('wall')}")
        _log(f"[smoke] {'FAIL' if bad else 'OK'}: {len(qids) - len(bad)}/{len(qids)} completed")
        sys.exit(1 if bad else 0)
    if a.dry_run and "GEPA_OUT" not in os.environ:
        OUT = f"{RUNS}/gepa_bcp_dryrun"; PROG = f"{OUT}/progress.txt"; CACHE_F = f"{OUT}/cache.json"
    os.makedirs(OUT, exist_ok=True)
    _cache_load()

    seed = seed_blocks()
    sk = ckey(seed); _SEED_KEY[0] = sk
    sp = json.load(open(_split_file()))
    pareto, fb = sp["pareto"], sp["feedback"]
    _log(f"[setup] fingerprint {FINGERPRINT} | seed {sk} | blocks "
         f"{ {b: len(seed[b]) for b in BLOCKS} } | servers {SERVERS} | retrieval {RETRIEVAL} | "
         f"D_feedback {len(fb)} | D_pareto {len(pareto)} | b={a.minibatch} | "
         f"budget {a.max_metric_calls} | workers {a.workers} | "
         f"critic {os.environ.get('GEPA_REFLECTION', 'claude')}:"
         f"{os.environ.get('GEPA_REFLECTION_MODEL', 'claude-opus-5-5')}")
    same_model = ROOT_MODEL == BASELINE_MODEL
    if (a.dry_run or a.preload_seed) and not same_model:
        # The preloaded records are BASELINE_MODEL rollouts; never file them under another model's key.
        raise SystemExit(f"--preload-seed/--dry-run use {BASELINE_MODEL} records; refusing for {ROOT_MODEL}")
    if a.dry_run or a.preload_seed:
        _log(f"[seed] preloaded {preload_seed(sk)} records from the 830 run under {sk}")
    _cache_save()

    rng = random.Random(1234)
    if a.dry_run:
        G = json.load(open(f"{path('FULLDOC_ROOT')}/runs/_graded_cache.json"))
        allv = [(1.0 if g["correct"] else 0.0) + LAMBDA * (g["gold_recall"] or 0.0) for g in G.values()]
        src = "830-run bootstrap"
    else:
        # The seed on all of D_pareto first; GEPA then finds these in the cache.
        _log(f"[seed] BASELINE: seed prompt on D_pareto ({len(pareto)} q) with {ROOT_MODEL}")
        recs = run_batch(seed, pareto, a.workers)
        okr = [r for r in recs.values() if r and r.get("status") in SCORED]
        accs = [1.0 if (r and r.get("correct")) else 0.0 for r in recs.values()]
        _log(f"[seed] BASELINE {ROOT_MODEL}: accuracy {statistics.mean(accs):.3f} "
             f"(n={len(accs)}, errors counted wrong) | completed {len(okr)} | gold_recall "
             f"{statistics.mean(r.get('gold_recall') or 0 for r in okr):.3f} | median wall "
             f"{statistics.median(r.get('wall') or 0 for r in okr):.0f}s")
        if len(okr) < 0.9 * len(pareto):
            raise SystemExit(f"[seed] only {len(okr)}/{len(pareto)} completed -- infrastructure, "
                             "not prompts. Fix before spending the GEPA budget.")
        allv = [score(r) for r in okr]
        src = "seed's own D_pareto scores"
    null_ref = sorted(statistics.mean(rng.choice(allv) for _ in range(len(pareto))) for _ in range(4000))
    seed_par = [score(_CACHE[f"{sk}|{q}"]) for q in pareto
                if f"{sk}|{q}" in _CACHE and _CACHE[f"{sk}|{q}"].get("status") in SCORED]
    _SEED_SCORE[0] = statistics.mean(seed_par) if seed_par else statistics.mean(allv)
    _log(f"[null] unchanged prompt, 100-q bootstrap of the {src}: mean {statistics.mean(null_ref):.4f} "
         f"p95 {null_ref[3800]:.4f} p99 {null_ref[3960]:.4f} (score = correct + {LAMBDA}*gold_recall)")

    import gepa.optimize_anything as OA
    res = OA.optimize_anything(
        seed_candidate=seed,
        dataset=[{"qid": q} for q in fb],
        valset=[{"qid": q} for q in pareto],
        batch_evaluator=make_evaluator(a.workers, seed, null_ref),
        config=OA.GEPAConfig(
            engine=OA.EngineConfig(max_metric_calls=a.max_metric_calls,
                                   candidate_selection_strategy="pareto",
                                   run_dir=OUT, raise_on_exception=False),
            tracking=OA.TrackingConfig(logger=OA.StdOutLogger()),
            reflection=OA.ReflectionConfig(reflection_lm=reflection_lm,
                                           reflection_minibatch_size=a.minibatch,
                                           module_selector="round_robin")))
    best = getattr(res, "best_candidate", None) or seed
    json.dump(best, open(f"{OUT}/best_candidate.json", "w"), indent=1)
    _log(f"[done] best {ckey(best)} (seed {sk}). Next: validate on D_pareto, then test200.")


if __name__ == "__main__":
    main()
