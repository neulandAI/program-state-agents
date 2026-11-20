"""Grade a run directory with the gpt-4.1 judge and report BrowseComp-Plus metrics.

    python src/judge.py --run-dir $RUNS_DIR/<run> [--template paper|official] [--votes 3]

Reads gold answers from <FULLDOC_ROOT>/data/prepared/{queries.jsonl,qrel_evidence.json}; Azure credentials from
AZURE2_ENDPOINT / AZURE2_API_KEY / AZURE2_API_VERSION. GRADER_TEMPLATE and OFFICIAL_TEMPLATE are the BrowseComp-Plus
grader templates. A run that is not completed or has an empty output is incorrect without judging; the accuracy
denominator is every query attempted; recall is |docids returned by search & evidence qrels| / |evidence qrels|.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

import openai
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "shared"))
from rlm_shared.paths import path  # noqa: E402

ROOT = os.environ.get("FULLDOC_ROOT", "")

# USD per 1M tokens, list price.
PRICING = {
    "gpt-5":      {"in": 1.25, "out": 10.00},
    "gpt-5-mini": {"in": 0.25, "out":  2.00},
    "gpt-5-nano": {"in": 0.05, "out":  0.40},
    "gpt-4.1":    {"in": 2.00, "out":  8.00},
}


def usage_of(run):
    """Flatten a run's per-model usage into totals plus a per-model breakdown."""
    out = {"in": 0, "out": 0, "calls": 0, "cost": 0.0, "by_model": {}}
    u = (run.get("usage") or {}).get("model_usage_summaries") or {}
    for model, m in u.items():
        i = m.get("total_input_tokens", 0) or 0
        o = m.get("total_output_tokens", 0) or 0
        c = m.get("total_calls", 0) or 0
        pr = PRICING.get(model, {"in": 0.0, "out": 0.0})
        cost = i / 1e6 * pr["in"] + o / 1e6 * pr["out"]
        out["in"] += i; out["out"] += o; out["calls"] += c; out["cost"] += cost
        out["by_model"][model] = {
            "input_tokens": i, "output_tokens": o, "calls": c, "cost_usd": round(cost, 4)
        }
    return out


if ROOT:
    load_dotenv(f"{ROOT}/.env")  # no-op when the file does not exist; credentials normally come from the environment

GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available.
""".strip()


# The leaderboard template (scripts_evaluation/evaluate_run.py); unlike GRADER_TEMPLATE it allows string variants.
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
    out = {"extracted_final_answer": None, "correct": None, "confidence": None, "parse_error": False}
    if not text:
        out["parse_error"] = True
        return out
    for pat in (
        r"\*\*extracted_final_answer:\*\*\s*(.*?)(?=\n|$)",
        r"\*\*extracted_final_answer\*\*:\s*(.*?)(?=\n|$)",
        r"extracted_final_answer:\s*(.*?)(?=\n|$)",
    ):
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            out["extracted_final_answer"] = m.group(1).strip()
            break
    # Last "correct:" line, not preceded by a letter (so "incorrect:" does not match), bold markers tolerated.
    hits = re.findall(r"(?<![a-z_])\**correct\**:\**\s*\**(yes|no)\b", text, re.IGNORECASE)
    if hits:
        out["correct"] = hits[-1].lower()
    for pat in (
        r"\*\*confidence:\*\*\s*(\d+(?:\.\d+)?)\s*%?",
        r"\*\*confidence\*\*:\s*(\d+(?:\.\d+)?)\s*%?",
        r"confidence:\s*(\d+(?:\.\d+)?)\s*%?",
    ):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            out["confidence"] = min(100.0, float(m.group(1)))
            break
    if out["correct"] is None:
        out["parse_error"] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--judge-model", default="gpt-4.1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--template", default="paper", choices=["paper", "official"],
                    help="paper = GRADER_TEMPLATE (strict, GPT-4.1 paper judge); official = current leaderboard template")
    ap.add_argument("--votes", type=int, default=1,
                    help="judge calls per (question, extracted answer); majority verdict is stored")
    ap.add_argument("--cache", default=f"{path('RUNS_DIR')}/_judge_cache.json",
                    help="shared verdict cache keyed on (template, judge model, qid, normalised extracted "
                         "answer) so an identical answer string is judged identically across runs; "
                         "'' disables")
    ap.add_argument("--expect", default=None,
                    help="JSON list of query ids that MUST all be present (e.g. the sample file)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.run_dir, "*.json")))
    files = [f for f in files if not os.path.basename(f).startswith("_")]  # skip _summary*.json
    if not files:
        sys.exit(f"no run files in {args.run_dir}")
    runs = [json.load(open(f, encoding="utf-8")) for f in files]

    if args.expect:
        want = set(json.load(open(args.expect)))
        have = {r["query_id"] for r in runs}
        if want - have:
            print(f"REFUSING to summarise: {len(want - have)} expected queries missing: "
                  f"{sorted(want - have)[:10]}")
            sys.exit(4)
        extra = have - want
        if extra:
            print(f"note: {len(extra)} run files outside the expected set, ignoring them")
            runs = [r for r in runs if r["query_id"] in want]

    # A directory must hold one configuration, or the summary averages apples and oranges.
    cfgs = {json.dumps({k: v for k, v in (r.get("config") or {}).items()
                        if k in ("root_model", "sub_model", "reasoning_effort",
                                 "default_k", "k_settable_by_model", "prompt_id",
                                 "search_budget_in_prompt", "orchestrator_addendum")},
                       sort_keys=True) for r in runs if r.get("status") != "timeout"}
    if len(cfgs) == 0:
        print("REFUSING to summarise: every run file is a timeout stub.")
        sys.exit(5)
    if len(cfgs) > 1:
        print("REFUSING to summarise: run files have mixed configurations:")
        for c in cfgs:
            print("   ", c)
        sys.exit(3)

    gt = {
        d["query_id"]: d
        for d in (json.loads(l) for l in open(f"{path('FULLDOC_ROOT')}/data/prepared/queries.jsonl", encoding="utf-8"))
    }
    qrel_ev = json.load(open(f"{path('FULLDOC_ROOT')}/data/prepared/qrel_evidence.json"))

    client = openai.AzureOpenAI(
        azure_endpoint=os.getenv("AZURE2_ENDPOINT"),
        api_key=os.getenv("AZURE2_API_KEY"),
        api_version=os.getenv("AZURE2_API_VERSION"),
        timeout=180,
    )

    import hashlib, threading
    cache_path = args.cache or None
    cache = {}
    if cache_path and os.path.exists(cache_path):
        cache = json.load(open(cache_path))
    cache_lock = threading.Lock()
    judge_calls = [0]

    def _norm(a):
        return " ".join(str(a or "").strip().strip("*").lower().split())

    def _call(prompt):
        last = None
        for _ in range(3):
            try:
                judge_calls[0] += 1
                r = client.chat.completions.create(
                    model=args.judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=2048,
                )
                v = parse_judge(r.choices[0].message.content or "")
                v["skipped_judge"] = False
                v["judge_text"] = (r.choices[0].message.content or "")[:3000]  # raw verdict, for re-evaluation
                v["judge_served_model"] = getattr(r, "model", None)
                return v
            except Exception as e:
                last = e
        return {"correct": "no", "parse_error": True, "judge_error": str(last)[:200],
                "skipped_judge": False, "extracted_final_answer": None, "confidence": None}

    def grade(run):
        qid = run["query_id"]
        output = (run.get("result") or [{}])[0].get("output", "") or ""
        if run.get("status") != "completed" or not output.strip():
            return qid, {"correct": "no", "skipped_judge": True, "parse_error": False,
                         "extracted_final_answer": None, "confidence": None}
        tmpl = OFFICIAL_TEMPLATE if args.template == "official" else GRADER_TEMPLATE
        prompt = tmpl.format(
            question=gt[qid]["query"], response=output, correct_answer=gt[qid]["answer"]
        )
        # Keyed on the agent's answer text, not on anything the judge generates.
        key = hashlib.sha256(
            f"{args.template}|{args.judge_model}|{qid}|out:{_norm(output)}".encode()
        ).hexdigest()[:24]
        with cache_lock:
            hit = cache.get(key)
        if hit is not None:
            return qid, {"correct": hit["correct"], "judge_cached": True,
                         "judge_votes": hit.get("votes"), "parse_error": False,
                         "skipped_judge": False,
                         "extracted_final_answer": hit.get("extracted"), "confidence": None}
        first = _call(prompt)
        if first.get("parse_error"):
            # An unparsable reply is an abstention; scored wrong only if every vote fails to parse.
            for _ in range(max(0, args.votes - 1)):
                first = _call(prompt)
                if not first.get("parse_error"):
                    break
            if first.get("parse_error"):
                return qid, first
        # Older cache entries are keyed on the judge's extraction; honour them and re-store under the new key.
        legacy = hashlib.sha256(
            f"{args.template}|{args.judge_model}|{qid}|{_norm(first.get('extracted_final_answer'))}".encode()
        ).hexdigest()[:24]
        with cache_lock:
            hit = cache.get(legacy)
        if hit is not None:
            v = dict(first); v["correct"] = hit["correct"]; v["judge_cached"] = True
            v["judge_votes"] = hit.get("votes")
            if cache_path:
                with cache_lock:
                    cache[key] = {"correct": hit["correct"], "votes": hit.get("votes"), "qid": qid,
                                  "extracted": hit.get("extracted")}
            return qid, v
        votes = [first["correct"]] + [_call(prompt).get("correct") for _ in range(max(0, args.votes - 1))]
        yes = sum(x == "yes" for x in votes)
        verdict = "yes" if yes * 2 > len(votes) else "no"
        v = dict(first); v["correct"] = verdict; v["judge_cached"] = False
        v["judge_votes"] = f"{yes}/{len(votes)}"
        if cache_path:
            with cache_lock:
                cache[key] = {"correct": verdict, "votes": v["judge_votes"], "qid": qid,
                              "extracted": _norm(first.get("extracted_final_answer"))}
        return qid, v

    with ThreadPoolExecutor(args.workers) as ex:
        verdicts = dict(ex.map(grade, runs))
    if cache_path:
        with cache_lock:
            json.dump(cache, open(cache_path, "w"), indent=0)

    rows, n_correct, mismatches = [], 0, []
    flagged = []  # runs whose server search count exceeds the cap, or whose code touched a suspicious token
    searches, recalls, times = [], [], []
    tok_in, tok_out, llm_calls, costs, iters, refused = [], [], [], [], [], []
    by_model_tot = {}
    for run in runs:
        qid = run["query_id"]
        v = verdicts[qid]
        ok = v.get("correct") == "yes"
        n_correct += ok
        ev = set(qrel_ev.get(qid, []))
        # Recall from the server-side docid record when present.
        got = set(run.get("server_docids") or run.get("retrieved_docids", []))
        rec = len(got & ev) / len(ev) if ev else None
        tcc = run.get("tool_call_counts", {})
        srv = run.get("server_counts") or {}
        if srv and srv.get("search") is not None and srv["search"] != tcc.get("search"):
            mismatches.append((qid, tcc.get("search"), srv["search"]))
        cap = (run.get("config") or {}).get("hard_search_cap") or 0
        if cap and srv and (srv.get("search") or 0) > cap:
            flagged.append({"query_id": qid, "reason": f"server search count {srv['search']} exceeds cap {cap}"})
        flags = run.get("sandbox_flags")
        if run.get("trajectory"):
            try:  # re-scan with the CURRENT patterns so a refined pattern applies to old files too
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from run_one import scan_trajectory as _scan
                flags = _scan(run["trajectory"])
            except Exception:
                pass
        if flags:
            flagged.append({"query_id": qid, "reason": "suspicious tokens in REPL code", "flags": flags[:5]})
        if any(f["query_id"] == qid for f in flagged) and ok:
            ok = False  # a flagged run is scored incorrect until a human clears it
            n_correct -= 1
        timed_out = run.get("status") == "timeout"
        # Timeout stubs count in accuracy but not in the retrieval averages (their counts were lost).
        if not timed_out:
            searches.append(srv.get("search", tcc.get("search", 0)))
            if rec is not None:
                recalls.append(rec)
        times.append(run.get("wall_time_s", 0))
        if run.get("status") == "completed":
            u = usage_of(run)
            tok_in.append(u["in"]); tok_out.append(u["out"])
            llm_calls.append(u["calls"]); costs.append(u["cost"])
            iters.append(run.get("n_iterations") or 0)
            refused.append(run.get("search_refused", 0))
            for m, mm in u["by_model"].items():
                acc_ = by_model_tot.setdefault(
                    m, {"input_tokens": 0, "output_tokens": 0, "calls": 0, "cost_usd": 0.0})
                acc_["input_tokens"] += mm["input_tokens"]
                acc_["output_tokens"] += mm["output_tokens"]
                acc_["calls"] += mm["calls"]
                acc_["cost_usd"] += mm["cost_usd"]
        rows.append({
            "query_id": qid, "correct": ok, "status": run.get("status"),
            "search_calls": tcc.get("search", 0),
            "n_docids": len(run.get("retrieved_docids", [])),
            "recall": rec, "wall_time_s": run.get("wall_time_s"),
            "extracted": v.get("extracted_final_answer"),
            "confidence": v.get("confidence"), "skipped_judge": v.get("skipped_judge"),
            "input_tokens": usage_of(run)["in"], "output_tokens": usage_of(run)["out"],
            "llm_calls": usage_of(run)["calls"], "cost_usd": round(usage_of(run)["cost"], 4),
            "n_iterations": run.get("n_iterations"),
        })

    n = len(runs)
    real = [r for r in runs if r.get("status") != "timeout"] or runs
    n_real = len(real)
    summary = {
        "run_dir": args.run_dir,
        "design": real[0].get("design"),
        "judge_model": args.judge_model,
        "judge_template": args.template,
        "n_queries": n,
        "accuracy_pct": round(100 * n_correct / n, 2),
        "evidence_recall_pct": round(100 * statistics.mean(recalls), 2) if recalls else None,
        "avg_search_calls": round(statistics.mean(searches), 2) if searches else None,
        "median_search_calls": statistics.median(searches) if searches else None,
        "avg_wall_time_s": round(statistics.mean(times), 1),
        "search_calls_p50": statistics.median(searches) if searches else None,
        "search_calls_p90": (sorted(searches)[int(0.9 * (len(searches) - 1))] if searches else None),
        "search_calls_max": max(searches) if searches else None,
        "n_completed": sum(r.get("status") == "completed" for r in runs),
        "n_judge_parse_errors": sum(bool(verdicts[r["query_id"]].get("parse_error")) for r in runs),
        "n_timeouts": sum(r.get("status") == "timeout" or
                          (r.get("status") == "error" and "imeout" in str(r.get("error", ""))) for r in runs),
        "n_scored_for_retrieval": n_real,
        "avg_k_per_search": round(
            sum((r.get("server_counts") or {}).get("k_sum", 0) for r in real)
            / max(1, sum((r.get("server_counts") or {}).get("search", 0) for r in real)), 2),
        "avg_docs_returned": round(
            sum((r.get("server_counts") or {}).get("hits", 0) for r in real) / max(1, n_real), 1),
        "avg_unique_docs": round(statistics.mean(len(r.get("retrieved_docids", [])) for r in real), 1),
        "judge_votes": args.votes,
        "judge_calls": judge_calls[0],
        "judge_cache": bool(cache_path),
        "counter_mismatches": mismatches,
        "flagged_runs": flagged,
        "n_recall_from_server_record": sum(1 for r in runs if r.get("server_docids") is not None),
        "avg_input_tokens": round(statistics.mean(tok_in)) if tok_in else None,
        "avg_output_tokens": round(statistics.mean(tok_out)) if tok_out else None,
        "avg_total_tokens": round(statistics.mean([a + b for a, b in zip(tok_in, tok_out)])) if tok_in else None,
        "avg_llm_calls": round(statistics.mean(llm_calls), 2) if llm_calls else None,
        "avg_iterations": round(statistics.mean(iters), 2) if iters else None,
        "avg_cost_usd": round(statistics.mean(costs), 4) if costs else None,
        "total_cost_usd": round(sum(costs), 2) if costs else None,
        "median_wall_time_s": round(statistics.median(times), 1) if times else None,
        "p90_wall_time_s": round(sorted(times)[int(0.9 * (len(times) - 1))], 1) if times else None,
        "max_wall_time_s": round(max(times), 1) if times else None,
        "tokens_per_correct": round(sum(a + b for a, b in zip(tok_in, tok_out)) / n_correct) if n_correct and tok_in else None,
        "cost_per_correct_usd": round(sum(costs) / n_correct, 4) if n_correct and costs else None,
        "searches_per_correct": round(sum(searches) / n_correct, 2) if n_correct and searches else None,
        "by_model": {m: {**v, "cost_usd": round(v["cost_usd"], 3)} for m, v in by_model_tot.items()},
        "config": real[0].get("config"),
        "rows": rows,
    }

    out = args.out or os.path.join(args.run_dir, "_summary.json" if args.template == "paper" else "_summary_official.json")
    json.dump(summary, open(out, "w"), indent=2)

    print(f"\n=== {args.run_dir}  (design {summary['design']}, judge {args.judge_model}, template {args.template}) ===")
    print(f"  queries              {n}   (completed {summary['n_completed']})")
    print(f"  accuracy             {summary['accuracy_pct']:.2f} %")
    print(f"  evidence recall      {summary['evidence_recall_pct']} %")
    print(f"  avg search calls     {summary['avg_search_calls']}  "
          f"(median {summary['median_search_calls']}, p90 {summary['search_calls_p90']}, max {summary['search_calls_max']})")
    print(f"  avg wall time        {summary['avg_wall_time_s']} s")
    print(f"  avg k per search     {summary['avg_k_per_search']}")
    print(f"  avg docs returned    {summary['avg_docs_returned']}  (unique {summary['avg_unique_docs']})")
    print(f"  judge calls          {summary['judge_calls']}  (votes {args.votes}, cache {'on' if cache_path else 'off'})")
    print(f"  timeouts             {summary['n_timeouts']}")
    print(f"  flagged runs         {len(flagged)}  (cap violations / sandbox tokens; scored incorrect)"
          + (f"  -> {[f['query_id'] for f in flagged][:10]}" if flagged else ""))
    print(f"  recall source        server record for {summary['n_recall_from_server_record']}/{n} runs, client list otherwise")
    print(f"  judge parse errors   {summary['n_judge_parse_errors']}")
    if mismatches:
        print(f"  !! COUNTER MISMATCH (local vs server) on {len(mismatches)} queries: {mismatches[:5]}")
    print("  --- tokens & cost (list price estimate) ---")
    print(f"  avg input tokens     {summary['avg_input_tokens']:,}" if summary['avg_input_tokens'] else "  avg input tokens     n/a")
    print(f"  avg output tokens    {summary['avg_output_tokens']:,}" if summary['avg_output_tokens'] else "")
    print(f"  avg total tokens     {summary['avg_total_tokens']:,}" if summary['avg_total_tokens'] else "")
    print(f"  avg LLM calls        {summary['avg_llm_calls']}  (root turns + llm_query sub-calls)")
    print(f"  avg RLM iterations   {summary['avg_iterations']}")
    print(f"  avg cost / query     ${summary['avg_cost_usd']}")
    print(f"  total cost           ${summary['total_cost_usd']}")
    for m, v in sorted(summary["by_model"].items()):
        print(f"     {m:12s} {v['calls']:5d} calls  in {v['input_tokens']:>10,}  out {v['output_tokens']:>8,}  ${v['cost_usd']}")
    print("  --- latency ---")
    print(f"  wall time            mean {summary['avg_wall_time_s']}s  median {summary['median_wall_time_s']}s  "
          f"p90 {summary['p90_wall_time_s']}s  max {summary['max_wall_time_s']}s")
    print("  --- efficiency per correct answer ---")
    print(f"  tokens / correct     {summary['tokens_per_correct']:,}" if summary['tokens_per_correct'] else "")
    print(f"  cost / correct       ${summary['cost_per_correct_usd']}")
    print(f"  searches / correct   {summary['searches_per_correct']}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
