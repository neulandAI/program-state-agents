"""Re-judge the stored answers of earlier BrowseComp-Plus runs with gpt-4.1, so every row has the same judge.

  python rejudge_bcp.py [row ...]

Reads run logs under LOGS_ROOT, ORIGINAL_REPO_ROOT and FULLDOC_ROOT of the machine the runs were made on (and the
archived run report, ARCHIVED_RUNS_HTML), and writes
$LOGS_ROOT/bcp_rejudge_gpt41/{<row>.json, summary.json}. A question without a saved answer keeps its old verdict.
"""
import glob, json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # <repo>/shared
from rlm_shared.grading.bcp_judge import GT, grade_outputs, run_outputs  # noqa: E402
from rlm_shared.paths import path  # noqa: E402
from datasets import load_dataset  # noqa: E402

R = str(path("ORIGINAL_REPO_ROOT"))
L = str(path("LOGS_ROOT"))
OUT = f"{L}/bcp_rejudge_gpt41"
os.makedirs(OUT, exist_ok=True)
ALL = [str(r["query_id"]) for r in load_dataset("Tevatron/browsecomp-plus", split="test")]  # q_idx -> query_id
IDX = json.load(open(f"{OUT}/answer_index.json"))
STORED = {"3": 573, "4": 559, "5": 600, "7": 654, "8": 660, "9": 701, "10": 647, "11": 726, "12": 716, "13": 690,
          "14": 637, "15": 659, "16": 506, "17": 576, "18": 211}   # Experiments sheet accuracy x 830


def dspy(patterns):
    """q_idx-keyed result lists, later files override earlier -> {qid: (answer or None, old correct)}"""
    fs = sorted({f for p in patterns for f in glob.glob(f"{L}/{p}")}, key=os.path.getmtime)
    m = {}
    for f in fs:
        for r in json.load(open(f)):
            q = ALL[r["q_idx"]]
            assert r.get("question") is None or r["question"].strip()[:60] == GT[q][0].strip()[:60], (f, r["q_idx"])
            m[q] = (r.get("predicted"), bool(r.get("correct")))
    return m


def row3():
    m = dspy(["run_q*_gpt5_ivfsq8.json", "rerun_*ivfsq8*.json"])
    for line in open(f"{R}/docs/gpt5_ivfsq8_experiment.txt"):
        x = re.match(r"\s+Q(\d+)\s+([✓✗])\s+", line)
        if not x or int(x.group(1)) > 830:
            continue
        q = ALL[int(x.group(1)) - 1]
        if q in m:
            continue
        pred = line[x.end():][41:81]
        cut = len(pred) == 40 and pred[-1] != " "
        m[q] = (None if cut else pred.strip(), x.group(2) == "✓")
    return m


def artifact(names):
    t = json.loads(re.search(r'<script id="trace"[^>]*>(.*?)</script>',
                             open(os.environ.get("ARCHIVED_RUNS_HTML") or f"{R}/docs/archived_runs.html").read(), re.S).group(1))
    m = {}
    for r in t["runs"]:
        if r["run"] in names:
            for x in r["questions"]:
                m[str(x["qid"])] = (x.get("answer_given") or "", bool(x.get("correct")))
    return m


def cached(parts):
    """{qid: old correct} from the Qwen3.6 run caches (a killed entry yields to a later one)."""
    out = {}
    for p, key in parts:
        for k, v in json.load(open(f"{L}/{p}/cache.json")).items():
            kk, q = k.split("|", 1)
            if kk == key and (q not in out or out[q].get("status") == "killed"):
                out[q] = v
    return {q: bool(v.get("correct")) for q, v in out.items()}


def rollouts(srcs, old):
    m = {}
    for s in srcs:
        for q, v in IDX[s].items():
            o = (v["output"] or "").strip()
            if o and q not in m and (v.get("status") in (None, "completed")):
                m[q] = o
    return {q: (m.get(q, ""), old.get(q, False)) for q in ALL}


def suprun(d):
    o = run_outputs(f"{L}/{d}")
    return {q: (o.get(q, ""), None) for q in ALL}


def row15():
    g = json.load(open(f"{path('FULLDOC_ROOT')}/runs/_graded_cache.json"))
    m = {}
    for s in ("q27_first20", "q27_rest810"):
        for q, v in IDX[s].items():
            m[q] = ((v["output"] or "") if v.get("status") == "completed" else "")
    return {q: (m.get(q, ""), bool(g[q]["correct"])) for q in ALL}


ROWS = {
    "3": row3,
    "4": lambda: dspy(["browsecomp_v2_minimal_q*.json"]),
    "5": lambda: dspy(["browsecomp_v2medium_baseline_first50.json", "browsecomp_v2medium_baseline_200.json",
                       "browsecomp_v2_medium_q148_q830.json", "browsecomp_v2_medium_retry56.json"]),
    "7": lambda: dspy(["mvonly200_s*.json", "mvonly830_s*.json"]),
    "8": lambda: dspy(["dl4096_s*.json", "dl4096x_s*.json"]),
    "9": lambda: dspy([f"rlmfix{n}_q*.json" for n in (24, 25, 29, 30, 31)]),
    "10": lambda: dspy(["rlmfix32_q*.json"]),
    "11": lambda: artifact({"colbert8192_high400", "colbert8192_high100b", "colbert8192_high100c", "colbert8192_high_rest230"}),
    "12": lambda: artifact({"gpt5_colbert8192_pair50", "gpt5_colbert8192_next150", "gpt5_colbert8192_rest630"}),
    "13": lambda: artifact({"final100_medium", "final200_medium", "final100_e", "final_rest"}),
    "14": lambda: artifact({"qwen38_flash_pair50", "qwen38_flash_rest"}),
    "15": row15,
    "16": lambda: rollouts(["colbert_gepa", "colbert_rest", "colbert_gepa1"],
                           cached([("gepa_bcp2", "ee5ea32a1c"), ("bcp_testrest_ctrl", "ee5ea32a1c"), ("gepa_bcp", "ee5ea32a1c")])),
    "17": lambda: rollouts(["rr_e2e", "rr_testrest", "rr_rest300", "rr_fb", "rr_par"],
                           cached([("bcp_policy_e2e", "4512373504"), ("bcp_testrest_rr", "08b583387f"),
                                   ("bcp_rest300_rr", "08b583387f"), ("bcp_feedback_rr", "08b583387f"),
                                   ("bcp_pareto_rr", "08b583387f")])),
    "18": lambda: rollouts(["bm25_t200", "bm25_g0", "bm25_g1"],
                           cached([("bcp_bm25_test200", "11d0c5bc6b"), ("bcp_bm25_rest_gpu0", "11d0c5bc6b"),
                                   ("bcp_bm25_rest_gpu1", "11d0c5bc6b")])),
    "bm25_gpt5": lambda: suprun("bcp_gpt5_bm25/runs"),
    "pilot_rr": lambda: {q: (a, None) for q, a in run_outputs(f"{L}/bcp_gpt5_rr/rr").items()},
}

sp = f"{OUT}/summary.json"
summary = json.load(open(sp)) if os.path.exists(sp) else {}
for row in sys.argv[1:] or ROWS:
    m = ROWS[row]()
    n_q = 100 if row == "pilot_rr" else 830
    gaps = [q for q, (a, _) in m.items() if a is None]                 # no saved answer: old verdict kept
    absent = [q for q in ALL if q not in m] if n_q == 830 else []      # no answer, no verdict
    todo = {q: a for q, (a, _) in m.items() if a is not None}
    new = grade_outputs(todo, f"{OUT}/_cache.json")
    filtered = [q for q, v in new.items() if v == "filtered"]          # gpt-4.1 prompt blocked: old verdict kept
    for q in filtered:
        new[q] = bool(m[q][1])
    has_old = [q for q in todo if m[q][1] is not None]
    old_n = sum(m[q][1] for q in m if m[q][1] is not None)
    new_n = sum(new.values()) + sum(m[q][1] for q in gaps)
    s = {"n": n_q, "graded": len(todo) - len(filtered), "content_filtered_kept_old": filtered, "kept_old_no_answer": len(gaps), "absent": len(absent),
         "empty_answer": sum(not a.strip() for a in todo.values()),
         "new_correct": new_n, "new_acc": round(new_n / n_q, 4)}
    if row in STORED:
        s["stored_acc"] = round(STORED[row] / 830, 4)
        s["est_acc"] = round((new_n + STORED[row] - sum(m[q][1] for q in m if m[q][1] is not None)) / 830, 4)
    if has_old:
        s.update(old_correct=old_n, old_acc=round(old_n / n_q, 4),
                 graded_old_acc=round(sum(m[q][1] for q in has_old) / len(has_old), 4),
                 graded_new_acc=round(sum(new[q] for q in has_old) / len(has_old), 4),
                 flips_right_to_wrong=sum(m[q][1] and not new[q] for q in has_old),
                 flips_wrong_to_right=sum(new[q] and not m[q][1] for q in has_old))
    summary[row] = s
    json.dump({"verdicts": {q: new.get(q, m[q][1] if q in m else False) for q in (list(m) + absent)},
               "graded": sorted(todo, key=int)}, open(f"{OUT}/{row}.json", "w"))
    json.dump(summary, open(sp, "w"), indent=1)
    print(row, s, flush=True)
