"""Grade a document REPL agent run directory against BrowseComp-Plus gold with a local model (interim grade).

    python src/grade_local.py <run_dir>   -> <run_dir>/_graded.json

Also used by split_runner.py (grade_local.judge). The reported numbers use the gpt-4.1 judge (judge.py).
"""
import glob, json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "shared"))
from rlm_shared.browsecomp_data import (  # noqa: E402
    get_question, get_gold_answer, get_gold_doc_ids, heuristic_match,
    _final_answer_span, _JUDGE_PROMPT,
)

JUDGE_URL = os.environ.get("LOCAL_JUDGE_BASE_URL", "http://localhost:8030/v1")
JUDGE_MODEL = os.environ.get("LOCAL_JUDGE_MODEL", "qwen3.8-27b-bf16")


def judge(question, response, gold):
    if heuristic_match(gold, _final_answer_span(response)):
        return True
    from openai import OpenAI
    c = OpenAI(base_url=JUDGE_URL, api_key="EMPTY")
    try:
        r = c.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": _JUDGE_PROMPT.format(
                question=question, response=response, correct_answer=gold)}],
            temperature=1, max_completion_tokens=2048,
            # thinking off: default reasoning would eat the budget before the verdict
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        t = r.choices[0].message.content or ""
        m = re.search(r"correct:\s*(yes|no)", t, re.IGNORECASE)
        if m:
            return m.group(1).lower() == "yes"
    except Exception as e:
        print(f"  [judge error {str(e)[:70]}]")
    return heuristic_match(gold, _final_answer_span(response))


def main(run_dir):
    from datasets import load_dataset
    ds = load_dataset("Tevatron/browsecomp-plus", split="test")
    by_id = {str(r["query_id"]): r for r in ds}

    files = [f for f in sorted(glob.glob(f"{run_dir}/*.json")) if not os.path.basename(f).startswith("_")]
    out, nerr = [], 0
    for f in files:
        rec = json.load(open(f))
        qid = str(rec.get("query_id") or os.path.basename(f).split(".")[0])
        row = by_id.get(qid)
        if row is None:
            continue
        resp = (rec.get("result") or [{}])[0].get("output", "") or ""
        status = rec.get("status")
        if status != "completed" or not resp:
            nerr += 1
        gold = get_gold_answer(row)
        ok = bool(resp) and judge(get_question(row), resp, gold)
        gids = get_gold_doc_ids(row)
        got = set(map(str, rec.get("server_docids") or rec.get("retrieved_docids") or []))
        out.append({"query_id": qid, "correct": ok, "status": status,
                    "error": rec.get("error"),
                    "gold_recall": (len(gids & got) / len(gids)) if gids else 0.0,
                    "searches": (rec.get("tool_call_counts") or {}).get("search", 0),
                    "wall_s": rec.get("wall_time_s"), "iters": rec.get("n_iterations")})
    n = len(out); c = sum(1 for r in out if r["correct"])
    summary = {"run_dir": run_dir, "n": n, "correct": c,
               "accuracy": (c / n) if n else 0.0, "errors": nerr,
               "error_rate": (nerr / n) if n else 1.0,
               "gold_recall_mean": (sum(r["gold_recall"] for r in out) / n) if n else 0.0,
               "searches_mean": (sum(r["searches"] for r in out) / n) if n else 0.0,
               "wall_mean_s": (sum(r["wall_s"] or 0 for r in out) / n) if n else 0.0}
    json.dump({"summary": summary, "rows": out},
              open(f"{run_dir}/_graded.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main(sys.argv[1])
