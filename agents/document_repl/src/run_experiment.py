"""Run run_one.py over a set of BrowseComp-Plus (or QUERIES_FILE) questions, one subprocess per question.

    python src/run_experiment.py --name <run> [flags]      -> $RUNS_DIR/<run>/<qid>.json

Results are written per question, so a run is resumable. The sample is drawn once from a fixed seed and cached.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "shared"))
from rlm_shared.paths import path  # noqa: E402

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = str(path("FULLDOC_ROOT"))
RUNS = str(path("RUNS_DIR"))
# Interpreter of the per-query workers.
PY = os.environ.get("PYTHON", "")


def load_sample(n: int, seed: int) -> list[str]:
    path = f"{ROOT}/data/prepared/sample_{n}_seed{seed}.json"
    if os.path.exists(path):
        return json.load(open(path))
    qids = [
        json.loads(l)["query_id"]
        for l in open(f"{ROOT}/data/prepared/queries.jsonl", encoding="utf-8")
    ]
    rng = random.Random(seed)
    sample = sorted(rng.sample(qids, n), key=int)
    json.dump(sample, open(path, "w"))
    print(f"drew fresh sample of {n} (seed {seed}) -> {path}")
    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="run directory name under runs/")
    ap.add_argument("--design", default="A", choices=["A", "B", "D"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--server", default="http://127.0.0.1:8765")
    ap.add_argument("--root-model", default="gpt-5")
    ap.add_argument("--sub-model", default="gpt-5-mini")
    ap.add_argument("--reasoning-effort", default="high")
    ap.add_argument("--sub-reasoning-effort", default="high")
    ap.add_argument("--default-k", type=int, default=5)
    ap.add_argument("--max-iterations", type=int, default=30)
    ap.add_argument("--per-query-timeout", type=int, default=None,
                    help="subprocess guard; default = max_timeout + 600")
    ap.add_argument("--no-orchestrator", dest="orchestrator", action="store_false")
    ap.set_defaults(orchestrator=True)
    ap.add_argument("--search-budget", type=int, default=0)
    ap.add_argument("--max-searches", type=int, default=0)
    ap.add_argument("--fixed-k", action="store_true")
    ap.add_argument("--seed-context", action="store_true")
    ap.add_argument("--compaction", action="store_true")
    ap.add_argument("--no-delegation", action="store_true")
    ap.add_argument("--lean-prologue", action="store_true")
    ap.add_argument("--solo-prompt", action="store_true")
    ap.add_argument("--single-pass-read", action="store_true")
    ap.add_argument("--constraint-check", action="store_true")
    ap.add_argument("--structured-plan", action="store_true")
    ap.add_argument("--compaction-threshold-pct", type=float, default=0.85)
    ap.add_argument("--max-concurrent-subcalls", type=int, default=4)
    ap.add_argument("--grounding-gate", action="store_true")
    ap.add_argument("--lm-timeout", type=float, default=None,
                    help="per-LM-call timeout in seconds, forwarded to run_one.py")
    ap.add_argument("--max-timeout", type=float, default=1800.0,
                    help="RLM wall limit per query in seconds (harness setting)")
    ap.add_argument("--save-trajectory", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated query ids (overrides sample)")
    ap.add_argument("--ids-file", default=None, help="json list of query ids (overrides sample)")
    ap.add_argument("--stagger", type=float, default=0.3,
                    help="seconds between worker starts; prevents a seed-search thundering herd")
    # The Azure deployment name overrides the model field; pin it.
    ap.add_argument("--root-deployment", default=None)
    ap.add_argument("--sub-deployment", default=None)
    ap.add_argument("--expect-served-model", default=None)
    ap.add_argument("--backend", default="azure_openai", choices=["azure_openai", "openai"])
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-env", default="ALIBABA_TOKEN")
    ap.add_argument("--force", action="store_true", help="re-run queries that already have a result")
    ap.add_argument("--allow-code-change", action="store_true",
                    help="resume even though the code digest differs from the directory's provenance")
    args = ap.parse_args()

    if not PY:
        raise SystemExit("environment variable PYTHON is not set: interpreter for the per-query workers "
                         "(configs/paths.example.env)")
    if args.per_query_timeout is None:
        args.per_query_timeout = int(args.max_timeout) + 600
    run_dir = f"{RUNS}/{args.name}"
    os.makedirs(run_dir, exist_ok=True)

    # The checker scripts are not part of this repository, so --expect-served-model aborts here.
    if args.expect_served_model:
        if args.backend == "openai":
            _chk = subprocess.run([PY, f"{SRC}/assert_openai_compat_model.py",
                            "--base-url", args.base_url,
                            "--api-key-env", args.api_key_env,
                            "--model", args.root_model,
                            "--expect", args.expect_served_model],
                           capture_output=True, text=True)
        else:
            _chk = subprocess.run([PY, f"{SRC}/assert_served_model.py",
                            "--model", args.root_model,
                            "--deployment", args.root_deployment or args.root_model,
                            "--expect", args.expect_served_model],
                           capture_output=True, text=True)
        print(_chk.stdout.strip())
        if _chk.returncode != 0:
            print(_chk.stderr.strip())
            raise SystemExit("ABORTED: the endpoint does not serve the expected model build.")

    if args.only:
        qids = args.only.split(",")
    elif args.ids_file:
        qids = [str(q) for q in json.load(open(args.ids_file))]
        print(f"query ids from {args.ids_file}: {len(qids)}")
    else:
        qids = load_sample(args.n, args.seed)

    # Refuse to resume a directory whose earlier results were produced by different code.
    out_dir = f"{RUNS}/{args.name}"
    os.makedirs(out_dir, exist_ok=True)
    # One orchestrator per run directory.
    lock_path = f"{out_dir}/_orchestrator.lock"
    if os.path.exists(lock_path):
        try:
            other = int(open(lock_path).read().strip() or 0)
            os.kill(other, 0)
            sys.exit(f"REFUSING: orchestrator pid {other} already runs {args.name} (lock {lock_path})")
        except (ProcessLookupError, ValueError):
            pass  # stale lock
    open(lock_path, "w").write(str(os.getpid()))
    import atexit as _atexit
    _atexit.register(lambda: os.path.exists(lock_path) and open(lock_path).read().strip() == str(os.getpid()) and os.remove(lock_path))
    prov_path = f"{out_dir}/_provenance.json"
    prev = json.load(open(prov_path)) if os.path.exists(prov_path) else None
    os.environ["RLM_LAUNCH_ARGV"] = " ".join(sys.argv)
    os.environ["RLM_ROOT_MODEL"] = args.root_model; os.environ["RLM_SUB_MODEL"] = args.sub_model
    subprocess.run([sys.executable, f"{SRC}/provenance.py", out_dir], check=True, capture_output=True)
    prov = json.load(open(prov_path))
    if prev is not None:
        # A resume must not erase the launch record: restore it and write its own timestamped file.
        json.dump(prev, open(prov_path, "w"), indent=1)
        prov["snapshot_relation"] = f"resume: written by run_experiment at {time.strftime('%Y-%m-%dT%H:%M')}"
        json.dump(prov, open(f"{out_dir}/_provenance_resume_{time.strftime('%Y%m%dT%H%M')}.json", "w"), indent=1)
    else:
        prov["snapshot_relation"] = "pre-run: written by run_experiment at launch"
        json.dump(prov, open(prov_path, "w"), indent=1)
    existing = [f for f in glob.glob(f"{out_dir}/*.json") if not os.path.basename(f).startswith("_")]
    if prev and prev.get("code_digest") != prov["code_digest"] and existing and not args.allow_code_change:
        sys.exit(f"REFUSING to resume {args.name}: code_digest {prev.get('code_digest')} -> {prov['code_digest']} "
                 f"with {len(existing)} results already present. Archive them or pass --allow-code-change.")
    os.environ["RLM_CODE_DIGEST"] = prov["code_digest"]
    print(f"provenance: code_digest {prov['code_digest']} -> {prov_path}")
    todo = [
        q for q in qids
        if args.force or not os.path.exists(f"{run_dir}/{q}.json")
    ]
    print(f"run={args.name} design={args.design} queries={len(qids)} todo={len(todo)} workers={args.workers}")
    if not todo:
        print("nothing to do")
        return

    def run_one(qid: str):
        cmd = [
            PY, f"{SRC}/run_one.py",
            "--query-id", qid,
            "--out", f"{run_dir}/{qid}.json",
            "--design", args.design,
            "--server", args.server,
            "--root-model", args.root_model,
            "--sub-model", args.sub_model,
            "--reasoning-effort", args.reasoning_effort,
            "--sub-reasoning-effort", args.sub_reasoning_effort,
            "--root-deployment", args.root_deployment or args.root_model,
            "--sub-deployment", args.sub_deployment or args.sub_model,
            "--default-k", str(args.default_k),
            "--max-iterations", str(args.max_iterations),
        ]
        cmd += ["--search-budget", str(args.search_budget),
                "--max-searches", str(args.max_searches)]
        if args.backend != "azure_openai":
            cmd += ["--backend", args.backend, "--api-key-env", args.api_key_env]
            if args.base_url:
                cmd += ["--base-url", args.base_url]
        if args.fixed_k:
            cmd.append("--fixed-k")
        if args.seed_context:
            cmd.append("--seed-context")
        if args.compaction:
            cmd.append("--compaction")
        if args.no_delegation:
            cmd.append("--no-delegation")
        if args.lean_prologue:
            cmd.append("--lean-prologue")
        if args.solo_prompt:
            cmd.append("--solo-prompt")
        if args.single_pass_read:
            cmd.append("--single-pass-read")
        if args.constraint_check:
            cmd.append("--constraint-check")
        if args.structured_plan:
            cmd.append("--structured-plan")
        cmd += ["--compaction-threshold-pct", str(args.compaction_threshold_pct),
                "--max-concurrent-subcalls", str(args.max_concurrent_subcalls)]
        if args.grounding_gate:
            cmd.append("--grounding-gate")
        cmd += ["--max-timeout", str(args.max_timeout)]
        if args.lm_timeout:
            cmd += ["--lm-timeout", str(args.lm_timeout)]
        if not args.orchestrator:
            cmd.append("--no-orchestrator")
        if args.save_trajectory:
            cmd.append("--save-trajectory")
        try:
            p = subprocess.run(
                cmd, capture_output=True, text=True, timeout=args.per_query_timeout
            )
            if p.returncode != 0 and not os.path.exists(f"{run_dir}/{qid}.json"):
                # Retry once, then leave an error record so the gap is visible.
                time.sleep(20)
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.per_query_timeout)
                if p.returncode != 0 and not os.path.exists(f"{run_dir}/{qid}.json"):
                    stub = {"query_id": qid, "design": args.design, "status": "error",
                            "error": f"worker exited rc={p.returncode} without a result (twice)",
                            "stderr_tail": p.stderr[-1500:], "result": [{"type": "output_text", "output": ""}],
                            "tool_call_counts": {"search": 0}, "server_counts": None,
                            "retrieved_docids": [], "search_log": [], "wall_time_s": 0.0, "config": {}}
                    with open(f"{run_dir}/{qid}.json", "w") as fh:
                        json.dump(stub, fh)
                    return qid, f"[{qid}] WORKER CRASH rc={p.returncode} (error stub written; delete it to re-run) {p.stderr[-160:]!r}"
            line = [l for l in p.stdout.strip().splitlines() if l.startswith(f"[{qid}]")]
            return qid, (line[-1] if line else f"[{qid}] no-output rc={p.returncode} {p.stderr[-200:]}")
        except subprocess.TimeoutExpired:
            # A stub keeps the question in the denominator (scored incorrect).
            stub = {
                "query_id": qid, "design": args.design, "status": "timeout",
                "result": [{"type": "output_text", "output": ""}],
                "tool_call_counts": {"search": 0},
                "retrieved_docids": [], "search_log": [], "server_counts": None,
                "wall_time_s": args.per_query_timeout,
                "config": {"root_model": args.root_model, "sub_model": args.sub_model,
                           "note": "harness timeout; no trajectory"},
            }
            with open(f"{run_dir}/{qid}.json", "w") as fh:
                json.dump(stub, fh)
            return qid, f"[{qid}] TIMEOUT after {args.per_query_timeout}s (stub written, scored incorrect)"
        except Exception as e:
            return qid, f"[{qid}] LAUNCH-FAIL {e}"

    t0 = time.time()
    done = 0
    # Staggered starts keep the workers' seed searches from hitting the server at once.
    def _staggered(q, i):
        if args.stagger > 0 and i < args.workers:
            time.sleep(i * args.stagger)
        return run_one(q)

    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(_staggered, q, i): q for i, q in enumerate(todo)}
        for fut in as_completed(futs):
            qid, msg = fut.result()
            done += 1
            el = time.time() - t0
            eta = (el / done) * (len(todo) - done)
            print(f"  ({done}/{len(todo)}) {msg}   [{el / 60:.1f}m elapsed, ~{eta / 60:.1f}m left]", flush=True)

    have = [q for q in qids if os.path.exists(f"{run_dir}/{q}.json")]
    print(f"\nfinished: {len(have)}/{len(qids)} results in {run_dir}  ({(time.time() - t0) / 60:.1f}m)")
    if len(have) < len(qids):
        print(f"missing: {[q for q in qids if q not in have]}")
        print("re-run the same command to retry the missing ones.")


if __name__ == "__main__":
    main()
