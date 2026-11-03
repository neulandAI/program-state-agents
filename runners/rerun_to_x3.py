#!/usr/bin/env python3
"""Convert document and snippet REPL run outputs to per-question records (data/x3/<run>/runs/<qid>.json) for the judge.

Gold recall is recomputed from the retrieved ids; context per turn is exact prompt tokens when saved, else characters / 4.
  python runners/rerun_to_x3.py --fulldoc data/x3/fulldoc_runs/fd_dense_high150 --run fd_dense_high150
  python runners/rerun_to_x3.py --snippet data/x3/snippet_runs/logs --run snip_dense_high150
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from x3_plain_agent import load_questions  # noqa: E402


def recall(ids, gold):
    ids = set(str(x) for x in ids)
    return (len(gold & ids) / len(gold)) if gold else None


def traj_context_chars(traj) -> list[int]:
    """Full-Doc trajectory: each iteration stores the message list sent; return characters per iteration (proxy)."""
    out = []
    if not isinstance(traj, dict):
        return out
    for it in traj.get("iterations", []) or []:
        msgs = it.get("messages") or it.get("prompt") or it.get("input") or []
        if isinstance(msgs, list):
            out.append(sum(len(json.dumps(m, ensure_ascii=False)) for m in msgs))
        elif isinstance(msgs, str):
            out.append(len(msgs))
    return out


def convert_fulldoc(src: Path, qs: dict, out: Path, order: list[str]):
    n = 0
    for p in sorted(src.glob("*.json")):
        if p.name.startswith("_"):
            continue
        d = json.load(open(p, encoding="utf-8"))
        qid = str(d.get("query_id") or p.stem); q = qs.get(qid)
        if q is None:
            continue
        texts = [r.get("output", "") for r in d.get("result", []) if r.get("type") == "output_text"]
        final = texts[-1] if texts else ""
        ids = d.get("retrieved_docids") or []
        usage = d.get("usage") or {}
        sums = (usage.get("model_usage_summaries") or {}) if isinstance(usage, dict) else {}
        cum_in = sum(v.get("total_input_tokens", 0) for v in sums.values()) if sums else (usage.get("input_tokens") or usage.get("prompt_tokens"))
        cum_out = sum(v.get("total_output_tokens", 0) for v in sums.values()) if sums else (usage.get("output_tokens") or usage.get("completion_tokens"))
        calls = sum(v.get("total_calls", 0) for v in sums.values()) if sums else None
        tcc = d.get("tool_call_counts") or {}
        its = ((d.get("trajectory") or {}).get("iterations") or []) if isinstance(d.get("trajectory"), dict) else []
        exact = [it.get("prompt_tokens") for it in its]
        chars = traj_context_chars(d.get("trajectory"))
        slog = d.get("search_log") or []
        ks = [x.get("k") or x.get("top_k") for x in slog if isinstance(x, dict) and (x.get("k") or x.get("top_k"))]
        rec = dict(query_id=qid, status=("completed" if d.get("status") == "completed" and final.strip() else (d.get("status") or "unknown")),
                   harness=dict(name="Full-Doc RLM (research repo harness, re-run)", source=str(src), rlm_build="rlm_upstream (archived patched build)"),
                   rounds=d.get("n_iterations"), n_searches=tcc.get("search", d.get("search_calls", 0)), n_get_document=tcc.get("get_document", 0),
                   retrieved_docids=sorted(set(str(x) for x in ids), key=lambda x: (len(x), x)),
                   gold_recall=recall(ids, q["gold"]), evidence_recall=recall(ids, q["evidence"]),
                   gold_in_search=sorted(q["gold"] & set(str(x) for x in ids)), gold_fetched=[],
                   context_chars_per_round=chars, context_tokens_per_round=([t for t in exact] if exact and all(t for t in exact) else [c // 4 for c in chars]),
                   context_measure=("exact prompt_tokens" if exact and all(t for t in exact) else "chars/4 of the saved prompt (proxy)"),
                   final_context_tokens=((exact[-1] if exact and all(t for t in exact) else (chars[-1] // 4 if chars else None))),
                   model_calls=calls, searches_k=ks, mean_k=(sum(ks) / len(ks) if ks else None), n_server_docids=len(d.get("server_docids") or []),
                   cum_input_tokens=cum_in, cum_output_tokens=cum_out,
                   usage_summary=usage, server_counts=d.get("server_counts"), wall_time_s=d.get("wall_time_s") or d.get("latency_s"),
                   timed_out=bool(d.get("timed_out", False)), forced=False, errors=([d["error"]] if d.get("error") else []),
                   final_answer=final, source_file=p.name)
        (out / f"{qid}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8"); n += 1
    return n


def snippet_trajectories(traj_dir: Path, qs: dict, wanted: set[str]) -> dict:
    """Map each trajectory file to its query id by finding the question text in the first prompt; return
    {qid: [chars per iteration]} (chars/4 is the context proxy, the same one used for the Full-Doc trajectories)."""
    out = {}
    if not traj_dir or not traj_dir.exists():
        return out
    texts = {qid: qs[qid]["question"].strip()[:120] for qid in wanted if qid in qs}
    for f in sorted(traj_dir.glob("*.jsonl")):
        try:
            lines = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        except Exception:  # noqa: BLE001
            continue
        its = [l for l in lines if l.get("type") == "iteration"]
        if not its:
            continue
        first = json.dumps(its[0].get("prompt", ""), ensure_ascii=False)
        qid = next((q for q, t in texts.items() if t and t in first), None)
        if qid is None:
            continue
        out[qid] = [len(json.dumps(it.get("prompt", ""), ensure_ascii=False)) for it in its]
    return out


def convert_snippet(src: Path, qs: dict, out: Path, order: list[str], traj_dir: Path | None = None):
    n = 0
    wanted = set()
    for p in sorted(src.glob("snip_q*.json")):
        try:
            e = json.load(open(p, encoding="utf-8"))[0]; wanted.add(order[int(e["q_idx"])])
        except Exception:  # noqa: BLE001
            pass
    traj = snippet_trajectories(traj_dir, qs, wanted) if traj_dir else {}
    for p in sorted(src.glob("snip_q*.json")):
        d = json.load(open(p, encoding="utf-8")); e = d[0] if isinstance(d, list) else d
        idx = e.get("q_idx"); qid = order[int(idx)] if idx is not None and int(idx) < len(order) else None
        q = qs.get(str(qid)) if qid is not None else None
        if q is None:
            continue
        ids = e.get("retrieved_docids") or e.get("docids") or []
        rec = dict(query_id=str(qid), status=("completed" if (e.get("predicted") or "").strip() and not e.get("error") else ("error" if e.get("error") else "empty")),
                   harness=dict(name="Snippet RLM (research repo harness, re-run)", source=str(src), rlm_build="pristine 854e688, patched at import by the harness"),
                   rounds=e.get("iterations"), n_searches=e.get("search_calls", 0), n_get_document=e.get("get_document_calls", 0),
                   llm_query_calls=e.get("llm_query_calls", 0), lm_calls_total=e.get("lm_calls_total"),
                   retrieved_docids=sorted(set(str(x) for x in ids), key=lambda x: (len(x), x)),
                   gold_recall=(recall(ids, q["gold"]) if ids else e.get("gold_recall")), evidence_recall=(recall(ids, q["evidence"]) if ids else e.get("evid_recall")),
                   gold_in_search=sorted(q["gold"] & set(str(x) for x in ids)) if ids else None, gold_fetched=[],
                   unique_docs=e.get("unique_docs"), cum_input_tokens=e.get("input_tokens"), cum_output_tokens=e.get("output_tokens"),
                   context_chars_per_round=traj.get(str(qid)), context_tokens_per_round=([c // 4 for c in traj[str(qid)]] if traj.get(str(qid)) else None),
                   final_context_tokens=((traj[str(qid)][-1] // 4) if traj.get(str(qid)) else None),
                   context_measure=("chars/4 of the saved prompt (proxy)" if traj.get(str(qid)) else None),
                   wall_time_s=e.get("latency_s"), timed_out=False, forced=False, errors=([e["error"]] if e.get("error") else []),
                   harness_correct=e.get("correct"), final_answer=e.get("predicted") or "", source_file=p.name)
        (out / f"{qid}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8"); n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fulldoc"); ap.add_argument("--snippet"); ap.add_argument("--run", required=True)
    ap.add_argument("--traj-dir", default=None, help="Snippet: RLM_TRAJ_DIR with the per-run jsonl trajectories (default <snippet dir>/traj)")
    a = ap.parse_args()
    qs = load_questions(); order = json.load(open(ROOT / "data/analysis/bcp_qidx_to_queryid.json"))
    out = ROOT / "data/x3" / a.run / "runs"; out.mkdir(parents=True, exist_ok=True)
    if a.fulldoc:
        print("converted", convert_fulldoc(Path(a.fulldoc), qs, out, order), "Full-Doc records ->", out)
    if a.snippet:
        td = Path(a.traj_dir) if a.traj_dir else Path(a.snippet) / "traj"
        print("converted", convert_snippet(Path(a.snippet), qs, out, order, td), "Snippet records ->", out)


if __name__ == "__main__":
    main()
