#!/usr/bin/env python3
"""Closed-book baseline: GPT-5 answers the 150 subset questions with no tools and no documents, one call per question.

Records go to data/x3/<run>/runs/<qid>.json in the format the judge reads. Usage: python runners/x5_closed_book.py --threads 16
"""
import argparse, json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The agents' answer format without the document instructions.
TASK = """You are a deep research agent, but for this question no search tool and no documents are available. Answer from your own knowledge only, and state your confidence honestly.

Reply in the following format:

Explanation: {your explanation for your final answer}
Exact Answer: {the exact name, date, title, or value the question asks for}
Confidence: {your confidence score between 0% and 100% for your answer}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="x5_closed_book")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--deployment", default=os.environ.get("GPT5_DEPLOYMENT", "gpt-5"))
    ap.add_argument("--effort", default="high")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    for k in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"):
        if not os.environ.get(k):
            sys.exit(f"missing {k}; run scripts/x3_secret.sh set")
    from openai import AzureOpenAI
    client = AzureOpenAI(api_key=os.environ["AZURE_OPENAI_API_KEY"], azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                         api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"), max_retries=2, timeout=1200)
    subset = json.load(open(ROOT / "data/analysis/x3_subset.json"))["query_ids"]
    qtext = {str(json.loads(l)["query_id"]): json.loads(l)["query"] for l in open("/path/to/rlm/data/prepared/queries.jsonl")}
    out = ROOT / "data/x3" / a.run / "runs"; out.mkdir(parents=True, exist_ok=True)
    cfg = dict(experiment="X5 closed-book (no tools, no documents)", deployment=a.deployment, effort=a.effort, max_completion_tokens=48000,
               started=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out.parent / "config.json").write_text(json.dumps(cfg, indent=1))
    todo = [q for q in subset if not (out / f"{q}.json").exists()]
    if a.limit:
        todo = todo[: a.limit]
    print(f"closed-book: {len(todo)} questions to run, deployment {a.deployment}, effort {a.effort}", flush=True)

    def run_one(qid):
        t0 = time.time()
        user = f"Answer the following: {qtext[qid]}\n\n{TASK}"
        messages = [{"role": "system", "content": "You are a careful research assistant."}, {"role": "user", "content": user}]
        rec = dict(query_id=qid, status="running", errors=[], config=cfg, final_answer=None)
        resp = None
        for attempt in range(10):
            try:
                resp = client.chat.completions.create(model=a.deployment, messages=messages, max_completion_tokens=48000, reasoning_effort=a.effort)
                break
            except Exception as e:  # noqa: BLE001
                rec["errors"].append(f"attempt {attempt}: {type(e).__name__}: {str(e)[:160]}")
                time.sleep(min(120, 5 * 2 ** attempt) + random.uniform(0, 3))
        if resp is None:
            rec["status"] = "error"
        else:
            u = resp.usage
            rec["usage"] = dict(prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
                                reasoning_tokens=getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", None))
            rec["final_answer"] = resp.choices[0].message.content or ""
            rec["finish_reason"] = resp.choices[0].finish_reason
            rec["status"] = "completed" if rec["final_answer"].strip() else "empty"
            rec["final_context_tokens"] = u.prompt_tokens; rec["cum_input_tokens"] = u.prompt_tokens; rec["cum_output_tokens"] = u.completion_tokens
        rec["wall_time_s"] = round(time.time() - t0, 1)
        (out / f"{qid}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        return rec

    done = 0
    with ThreadPoolExecutor(a.threads) as ex:
        futs = {ex.submit(run_one, q): q for q in todo}
        for f in as_completed(futs):
            r = f.result(); done += 1
            print(f"  ({done}/{len(todo)}) q{r['query_id']} {r['status']} out={r.get('cum_output_tokens')} t={r.get('wall_time_s')}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
