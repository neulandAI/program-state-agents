#!/usr/bin/env python3
"""Exact prompt sizes (o200k_base tokens) of the REPL agents from their saved trajectories: final and peak prompt per
question; the reference agents' API counts for comparison.

Run: python analysis/exact_prompt_tokens.py -> data/analysis/exact_prompt_tokens.{csv,md}. Reads the trajectories under
data/x3/; the run records are not part of the release.
"""
import csv, glob, json, statistics as st
from pathlib import Path
import tiktoken

ROOT = Path(__file__).resolve().parents[1]
ENC = tiktoken.get_encoding("o200k_base")
SUBSET = set(json.load(open(ROOT / "data/analysis/x3_subset.json"))["query_ids"])
QTEXT = {str(json.loads(l)["query_id"]): json.loads(l)["query"] for l in open("/path/to/rlm/data/prepared/queries.jsonl")}


def ntok(content):
    if content is None:
        return 0
    if isinstance(content, list):
        return sum(ntok(c.get("text") if isinstance(c, dict) else c) for c in content)
    return len(ENC.encode(str(content), disallowed_special=()))


def prompt_tokens(prompt):
    return sum(ntok(m.get("content")) for m in prompt if isinstance(m, dict))


def series_stats(series):
    final = series[-1] if series else 0
    peak = max(series) if series else 0
    comp = sum(1 for a, b in zip(series, series[1:]) if b < 0.7 * a)
    return final, peak, comp


def fulldoc(run_dir, restrict=None):
    out = {}
    for f in glob.glob(str(run_dir / "[0-9]*.json")):
        r = json.load(open(f)); q = str(r["query_id"])
        if restrict is not None and q not in restrict:
            continue
        its = ((r.get("trajectory") or {}).get("iterations") or []) if isinstance(r.get("trajectory"), dict) else []
        series = [prompt_tokens(it.get("prompt") or []) for it in its if isinstance(it, dict)]
        if series:
            out[q] = series
    return out


def snippet(traj_dir, wanted):
    out = {}
    for f in glob.glob(str(traj_dir / "*.jsonl")):
        series = []; qid = None
        with open(f) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("type") != "iteration":
                    continue
                p = d.get("prompt") or []
                if qid is None:
                    text = " ".join(str(m.get("content", "")) for m in p if isinstance(m, dict))
                    for q in wanted:
                        if QTEXT[q][:120] in text:
                            qid = q; break
                series.append(prompt_tokens(p))
        if qid and series and qid not in out:
            out[qid] = series
    return out


def proxy_final(conv_dir):
    out = {}
    for f in glob.glob(str(conv_dir / "runs" / "*.json")):
        r = json.load(open(f)); out[str(r["query_id"])] = r.get("final_context_tokens")
    return out


def main():
    rows = []; summary = []
    jobs = [
        ("fd_dense_high150", fulldoc(ROOT / "data/x3/fulldoc_runs/fd_dense_high150"), "fd_dense_high150"),
        ("fd_k5", fulldoc(ROOT / "data/x3/fulldoc_runs/fd_dense_high150_k5"), "fd_k5"),
        ("fd_k5print", fulldoc(ROOT / "data/x3/fulldoc_runs/fd_dense_high150_k5print"), "fd_k5print"),
        ("fd_cap20_150", fulldoc(ROOT / "data/x3/fulldoc_runs/fd_dense_high50_cap20", SUBSET), "fd_cap20_150"),
        ("fd_cap20_830", fulldoc(ROOT / "data/x3/fulldoc_runs/fd_dense_high50_cap20"), "fd_cap20_830"),
        ("snip_dense_high150", snippet(ROOT / "data/x3/snippet_runs/logs/traj", SUBSET), "snip_dense_high150"),
        ("snip_cap10", snippet(ROOT / "data/x3/snippet_runs_cap10/logs/traj", SUBSET), "snip_cap10"),
        ("fd_nogrep", fulldoc(ROOT / "data/x3/fulldoc_runs/fd_dense_high150_nogrep"), "fd_nogrep"),
        ("snip_k25", snippet(ROOT / "data/x3/snippet_runs_k25/logs/traj", SUBSET), "snip_k25"),
        ("fd_k5prompt", fulldoc(ROOT / "data/x3/fulldoc_runs/fd_dense_high150_k5prompt"), "fd_k5prompt"),
        ("fd_k5_rep2", fulldoc(ROOT / "data/x3/fulldoc_runs/fd_dense_high150_k5_rep2"), "fd_k5_rep2"),
        ("snip_k25_rep2", snippet(ROOT / "data/x3/snippet_runs_k25_rep2/logs/traj", SUBSET), "snip_k25_rep2"),
    ]
    for run, series_by_q, conv in jobs:
        prox = proxy_final(ROOT / "data/x3" / conv)
        finals, peaks, ratios, comps, ncomp_q = [], [], [], 0, 0
        for q, s in series_by_q.items():
            fin, pk, comp = series_stats(s)
            rows.append(dict(run=run, query_id=q, n_iterations=len(s), final_tokens=fin, peak_tokens=pk, compaction_events=comp, proxy_final=prox.get(q)))
            finals.append(fin); peaks.append(pk); comps += comp; ncomp_q += 1 if comp else 0
            if prox.get(q):
                ratios.append(fin / prox[q])
        if finals:
            fs = sorted(finals); ps = sorted(peaks)
            summary.append(dict(run=run, n=len(finals), final_median=fs[len(fs) // 2], final_p90=fs[int(0.9 * (len(fs) - 1))], final_max=fs[-1],
                                peak_median=ps[len(ps) // 2], peak_p90=ps[int(0.9 * (len(ps) - 1))], peak_max=ps[-1],
                                exact_over_proxy_median=(st.median(ratios) if ratios else float("nan")), compaction_events=comps, questions_with_compaction=ncomp_q))
    # reference agents: exact API counts already in the converted records
    for run in ("x7_dense_gpt5", "x7_dense_gpt5_getdoc", "x7_dense_gpt5_trim30k", "x7_dense_gpt5_getdoc_cap10", "x7_dense_gpt5_getdoc_cap10_k25", "x7_dense_gpt5_rep2", "x7_dense_gpt5_sum30k", "x7_dense_gpt5_getdoc_680"):
        finals = [v for v in proxy_final(ROOT / "data/x3" / run).values() if v]
        fs = sorted(finals)
        summary.append(dict(run=run + " (exact API)", n=len(fs), final_median=fs[len(fs) // 2], final_p90=fs[int(0.9 * (len(fs) - 1))], final_max=fs[-1],
                            peak_median=None, peak_p90=None, peak_max=None, exact_over_proxy_median=1.0, compaction_events=None, questions_with_compaction=None))
    with open(ROOT / "data/analysis/exact_prompt_tokens.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    md = ["# Exact prompt sizes (o200k_base tokens of the saved prompts) · replaces the characters/4 proxy\n",
          "| Run | n | Final prompt median / p90 / max | Peak prompt median / p90 / max | exact / proxy (median) | Compaction events (questions) |", "|---|---|---|---|---|---|"]
    for s in summary:
        pk = f"{s['peak_median']:,} / {s['peak_p90']:,} / {s['peak_max']:,}" if s["peak_median"] is not None else "–"
        ce = f"{s['compaction_events']} ({s['questions_with_compaction']})" if s["compaction_events"] is not None else "–"
        md.append(f"| {s['run']} | {s['n']} | {s['final_median']:,} / {s['final_p90']:,} / {s['final_max']:,} | {pk} | {s['exact_over_proxy_median']:.2f} | {ce} |")
    md += ["", "Final = the prompt of the last model call; peak = the largest prompt of any call; a compaction event is a drop of more than 30 % between consecutive calls (heuristic). Snippet trajectories are matched to questions by the question text in the first prompt. Source: `scripts/exact_prompt_tokens.py`."]
    (ROOT / "data/analysis/exact_prompt_tokens.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
