#!/usr/bin/env python3
"""Grade a run with gpt-4.1 on the BrowseComp-Plus grader template: three votes, majority.

Run: python evaluation/x3_judge.py --run <run> [--no-pairs] -> data/x3/<run>/graded_gpt41.json. Credentials via
x3_env.py. The run records are not part of the release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from x3_plain_agent import load_questions  # noqa: E402

OFFICIAL_TEMPLATE = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. 

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available."""


def parse_judge(text: str) -> dict:
    out = {"correct": None, "parse_error": False}
    if not text:
        out["parse_error"] = True
        return out
    hits = re.findall(r"(?<![a-z_])\**correct\**:\**\s*\**(yes|no)\b", text, re.IGNORECASE)
    if hits:
        out["correct"] = hits[-1].lower()
    else:
        out["parse_error"] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="x3_dense_gpt5")
    ap.add_argument("--judge", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--env-file", default=None)
    ap.add_argument("--no-pairs", action="store_true", help="skip the paired comparison with the archived BCP rows")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from x3_env import load_env, judge_credentials, judge_note
    load_env(a.env_file)
    print(judge_note(), file=sys.stderr)
    a.judge = a.judge or os.environ.get("JUDGE_DEPLOYMENT", "")
    for k in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"):
        if not os.environ.get(k):
            sys.exit(f"missing {k}")
    if not a.judge:
        sys.exit("missing --judge / JUDGE_DEPLOYMENT")
    from openai import AzureOpenAI
    jep, jkey = judge_credentials()
    cli = AzureOpenAI(api_key=jkey, azure_endpoint=jep,
                      api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"), max_retries=3, timeout=180)
    qs = load_questions()
    run_dir = ROOT / "data" / "x3" / a.run
    cache_path = run_dir / "_judge_cache_gpt41.json"
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    recs = {p.stem: json.load(open(p)) for p in (run_dir / "runs").glob("*.json")}

    def grade(item):
        qid, rec = item
        out = rec.get("final_answer") or ""
        if not out.strip():
            return qid, False, []
        key = hashlib.sha256(f"official|{a.judge}|{qid}|{' '.join(out.lower().split())}".encode()).hexdigest()[:24]
        if key in cache:
            return qid, cache[key]["correct"], cache[key]["votes"]
        prompt = OFFICIAL_TEMPLATE.format(question=qs[qid]["question"], response=out, correct_answer=qs[qid]["answer"])
        votes = []
        for _ in range(3):
            for _try in range(3):
                try:
                    r = cli.chat.completions.create(model=a.judge, messages=[{"role": "user", "content": prompt}], max_completion_tokens=2048)
                    v = parse_judge(r.choices[0].message.content or "")
                    if not v["parse_error"]:
                        votes.append(v["correct"] == "yes")
                        break
                except Exception as e:  # noqa: BLE001
                    print(f"[judge] {qid}: {type(e).__name__} {str(e)[:120]}", file=sys.stderr, flush=True)
        ok = bool(votes) and sum(votes) * 2 > len(votes)
        cache[key] = {"correct": ok, "votes": votes}
        return qid, ok, votes

    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(grade, recs.items()))
    json.dump(cache, open(cache_path, "w"), indent=1)
    graded = {qid: {"correct": ok, "votes": votes, "timed_out": recs[qid].get("timed_out", False), "status": recs[qid].get("status"),
                    "gold_recall": recs[qid].get("gold_recall"), "searches": recs[qid].get("n_searches"), "rounds": recs[qid].get("rounds"),
                    "final_context_tokens": recs[qid].get("final_context_tokens"), "cum_input_tokens": recs[qid].get("cum_input_tokens")}
              for qid, ok, votes in results}
    json.dump(graded, open(run_dir / "graded_gpt41.json", "w"), indent=1)
    n = len(graded)
    acc = sum(g["correct"] for g in graded.values()) / n if n else 0
    acc_strict = sum(g["correct"] and not g["timed_out"] for g in graded.values()) / n if n else 0
    print(f"{a.run}: n={n} accuracy={acc:.4f} (timed-out counted wrong: {acc_strict:.4f})")
    # paired comparison with the earlier full-set runs on the same questions
    if a.no_pairs:
        return
    import csv
    from scipy.stats import binomtest
    by = {r["QID"]: r for r in csv.DictReader(open(ROOT / "data/failure_analysis/bcp_by_question.csv", encoding="utf-8"))}
    for row in ("BCP-9", "BCP-21", "BCP-23", "BCP-25"):
        a_only = b_only = both = neither = 0
        for qid, g in graded.items():
            if qid not in by:
                continue
            x3 = bool(g["correct"] and not g["timed_out"])
            other = by[qid][f"{row} correct"].strip() in ("1", "1.0", "True")
            a_only += x3 and not other; b_only += other and not x3; both += x3 and other; neither += (not x3) and (not other)
        p = binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else float("nan")
        print(f"  vs {row}: X3 acc {(a_only+both)/n:.3f} vs {row} acc {(b_only+both)/n:.3f}; only X3 {a_only}, only {row} {b_only}, both {both}, neither {neither}; exact McNemar p={p:.3g}")


if __name__ == "__main__":
    main()
