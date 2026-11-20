#!/usr/bin/env python3
"""Convert a Fetch-then-Explore run (outputs/browsecomp_plus/<run_id>/predictions.jsonl) into our per-question records
(data/x3/<name>/runs/<query_id>.json) so evaluation/x3_judge.py grades it exactly like every other run in the paper
(gpt-4.1, the benchmark's grader template, 3 votes, majority).

Per question: final_answer = the agent's last assistant message (FtE wraps the answer in <answer> tags; the judge reads
the whole response, as for the reference agent); status = "completed" when FtE terminated with an answer, else its
termination (max_turns, timeout, incomplete, invalid_tool_call, error), which the judge counts wrong; n_searches = search
tool calls; retrieved = every "Document ID" a search returned (gold_recall, as for snippet agents) and fetched = every
document fetched into the workspace (gold_recall_fetched); tokens summed over model calls when the run recorded usage.
  python agents/fetch_then_explore/fte_to_x3.py <predictions.jsonl> <name>
"""
import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation"))
from x3_plain_agent import load_questions  # noqa: E402

DOC = re.compile(r"Document ID:\s*([^\s\n]+)")


def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
    return ""


def main(pred_path, name):
    qs = load_questions()
    out = ROOT / "data/x3" / name / "runs"; out.mkdir(parents=True, exist_ok=True)
    n = 0
    for line in open(pred_path, encoding="utf-8"):
        r = json.loads(line); md = r.get("metadata") or {}
        qid = str(md.get("query_id") or (md.get("item_metadata") or {}).get("query_id") or (md.get("metadata") or {}).get("query_id"))
        msgs = r.get("messages") or []
        final = next((text_of(m.get("content")) for m in reversed(msgs) if m.get("role") == "assistant" and text_of(m.get("content")).strip()), r.get("prediction") or "")
        calls = md.get("model_calls") or []
        names = [tc.get("name") for c in calls for tc in (c.get("tool_calls") or [])]
        retrieved, fetched = set(), set()
        for m in msgs:
            if m.get("role") == "tool":
                retrieved |= set(DOC.findall(text_of(m.get("content"))))
        for c in calls:
            for tc in c.get("tool_calls") or []:
                if tc.get("name") == "fetch":
                    a = tc.get("arguments") or {}
                    a = json.loads(a) if isinstance(a, str) else a
                    for v in (a.values() if isinstance(a, dict) else []):
                        for x in (v if isinstance(v, list) else [v]):
                            fetched.add(str(x).strip())
        gold = qs.get(qid, {}).get("gold") or set()
        usage = [(c.get("response") or {}).get("usage") or {} for c in calls]
        cin = sum(u.get("prompt_tokens") or 0 for u in usage); cout = sum(u.get("completion_tokens") or 0 for u in usage)
        term = r.get("termination")
        rec = {"query_id": qid, "final_answer": final if term == "answer" else "", "status": "completed" if term == "answer" else term,
               "timed_out": term in ("timeout", "max_turns"), "n_searches": names.count("search"), "tool_calls": {k: names.count(k) for k in set(names)},
               "rounds": md.get("turns"), "gold_recall": (len(gold & retrieved) / len(gold)) if gold else None,
               "gold_recall_fetched": (len(gold & fetched) / len(gold)) if gold else None, "n_retrieved": len(retrieved), "n_fetched": len(fetched),
               "cum_input_tokens": cin or None, "cum_output_tokens": cout or None,
               "final_context_tokens": (usage[-1].get("prompt_tokens") if usage else None), "harness": "fetch-then-explore (liuqi6777/search_agent 51ceadc)"}
        json.dump(rec, open(out / f"{qid}.json", "w"), indent=1); n += 1
    print(f"{n} records -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
