"""Run one question through the document REPL agent and write a run record.

    python src/run_one.py --query-id <id> --out <record.json> [flags]

Needs the rlm library (RLMLIB_PATH) on PYTHONPATH and FULLDOC_ROOT; questions come from
<FULLDOC_ROOT>/data/prepared/queries.jsonl unless QUERIES_FILE is set. One process per question, because rlm's
LocalREPL swaps process-global sys.stdout and cwd. Only the question text is loaded into this process, and every
search carries a per-run tag so the retrieval server keeps its own counts. The record follows the BrowseComp-Plus
run-file schema: {query_id, tool_call_counts, status, retrieved_docids, result:[{type, output}]}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import uuid

from dotenv import load_dotenv

# rlm_shared/paths.py is loaded by file, not put on sys.path, so REPL code cannot import it.
_SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC)


def _load_paths():
    import importlib.util as _ilu
    _f = os.path.join(_SRC, os.pardir, os.pardir, "shared", "rlm_shared", "paths.py")
    _spec = _ilu.spec_from_file_location("_fulldoc_paths", _f)
    _m = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    return _m.path


path = _load_paths()
if os.environ.get("FULLDOC_ROOT"):
    load_dotenv(os.path.join(os.environ["FULLDOC_ROOT"], ".env"))

from prompts import SOLO_SYSTEM_PROMPT, CONTEXT_NOTE, QUERY_TEMPLATE, build_prologue, prologue_id  # noqa: E402
from rlm import RLM  # noqa: E402
from rlm.environments import local_repl as _lr  # noqa: E402
import fence_tolerant as _ft  # noqa: E402  -- no-op unless SUP_FENCE_TOLERANT=1
_ft.install()
import rep_guard as _rg  # noqa: E402  -- no-op unless SUP_REP_GUARD=1
_rg.install()


def _sampling(effort):
    """reasoning_effort, plus presence_penalty when SUP_PRESENCE_PENALTY is set (Qwen3.6 runs)."""
    sa = {"reasoning_effort": effort}
    if os.environ.get("SUP_PRESENCE_PENALTY"):
        sa["presence_penalty"] = float(os.environ["SUP_PRESENCE_PENALTY"])
    return sa


# Modules REPL code may not import: network, processes, and introspection hooks that reach the harness.
BLOCKED_MODULES = {
    "socket", "ssl", "urllib", "urllib3", "http", "httpx", "requests", "aiohttp", "ftplib", "smtplib",
    "telnetlib", "xmlrpc", "websocket", "websockets", "subprocess", "multiprocessing", "ctypes",
    "importlib", "pkgutil", "runpy", "signal", "shutil", "pathlib", "glob", "tempfile", "pty", "resource",
    "openai", "rlm", "tools", "src", "rlm_bcplus", "dotenv", "torch", "transformers", "faiss",
}
ALLOWED_OPEN_PREFIXES: tuple = ()  # REPL code may not open files; the corpus is reachable only via search


def harden_sandbox(azure_hosts: list[str]):
    """Restrict REPL code: block BLOCKED_MODULES, refuse open(), and resolve only the model hosts and loopback."""
    import builtins as _b
    import socket as _socket

    real_import = _b.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = (name or "").split(".")[0]
        if root in BLOCKED_MODULES:
            raise ImportError(f"module '{name}' is not available in this REPL")
        return real_import(name, globals, locals, fromlist, level)

    def guarded_open(file, *a, **kw):
        p = str(file)
        if not any(p.startswith(pre) for pre in ALLOWED_OPEN_PREFIXES):
            raise PermissionError("file access is not available in this REPL; use search")
        return open(file, *a, **kw)

    sb = _lr._SAFE_BUILTINS
    sb["__import__"] = guarded_import
    sb["open"] = guarded_open

    allowed = {h.lower() for h in azure_hosts if h} | {"127.0.0.1", "localhost", "::1"}
    real_gai = _socket.getaddrinfo

    def guarded_gai(host, *a, **kw):
        h = (host.decode() if isinstance(host, bytes) else str(host or "")).lower().rstrip(".")
        if h not in allowed:
            raise _socket.gaierror(f"host not permitted from this worker: {h}")
        return real_gai(host, *a, **kw)

    _socket.getaddrinfo = guarded_gai


# Tokens in model-written code that flag a run (ways to reach the harness, network or filesystem).
SUSPICIOUS_PATTERNS = [
    r"__closure__", r"__self__", r"cell_contents", r"__globals__", r"__builtins__", r"__subclasses__",
    r"\btool_call_counts\b", r"\bmax_searches\b", r"\bretrieved_docids\b", r"\b_seen\b", r"sys\.modules",
    r"\bsocket\b", r"\burllib\b", r"\brequests\.", r"\bhttpx\b", r"http\.client", r"\bsubprocess\b",
    r"os\.environ", r"corpus\.jsonl", r"\bqrels?\b", r"queries\.jsonl", r"(?<![\w.])open\(", r'__import__\(\s*(?!["\'](?:re|json|math|collections|itertools|difflib|string|unicodedata|datetime|statistics|functools|operator|textwrap|heapq|bisect|copy|typing|dataclasses|enum|random|time|decimal|fractions|html|csv|io|pprint|numbers|array)["\'])',  # __import__ of a harmless stdlib module by literal name is the common idiom; anything else is flagged
    r"\bimportlib\b", r"\b8765\b", r"\bAZURE\b",
]
_SUSPICIOUS = [__import__("re").compile(p) for p in SUSPICIOUS_PATTERNS]


def scan_trajectory(traj) -> list[dict]:
    """Return [{'iteration', 'token', 'line'}] for every suspicious token in model-written code."""
    flags = []
    for it in (traj or {}).get("iterations", []) or []:
        for cb in it.get("code_blocks", []) or []:
            code = cb.get("code") or ""
            for rx in _SUSPICIOUS:
                m = rx.search(code)
                if m:
                    line = next((l.strip() for l in code.splitlines() if rx.search(l)), "")[:160]
                    flags.append({"iteration": it.get("iteration"), "token": rx.pattern, "line": line})
    return flags


def install_grounding_gate(toolbox):
    """Once per run, hold a submission whose "Exact Answer:" string occurs in no document in DOCS."""
    import re as _re

    state = {"warned": False}
    orig = _lr.LocalREPL._capture_answer

    def _extract(content: str) -> str:
        m = _re.search(r"Exact Answer:[ \t]*([^\n]+)", content or "", _re.I)  # same line only
        return (m.group(1) if m else (content or "")).strip().strip("*").strip()

    def gated(self, content):
        ans = _extract(str(content))
        docs = toolbox.DOCS
        found = False
        if ans and len(ans) >= 2:
            low = ans.lower()
            try:
                for rec in dict.values(docs):
                    if not isinstance(rec, dict):
                        continue
                    body = (rec.get("text") or rec.get("snippet") or "")
                    if low in body.lower():
                        found = True
                        break
            except Exception:
                return orig(self, content)  # never let the gate block a submission on an error
        if not found and not state["warned"] and ans:
            state["warned"] = True
            print(f"[answer check] The string {ans!r} does not occur in any retrieved document. "
                  f"If it is a name, title or term, copy it exactly as a source writes it — grep "
                  f"DOCS for the surrounding words to find the exact spelling — then set "
                  f"answer['ready'] again. This notice is shown once; a second submission is accepted as-is.")
            a = self.locals.get("answer")
            if isinstance(a, dict):
                dict.__setitem__(a, "ready", False)
            return  # do not capture; the run continues
        return orig(self, content)

    _lr.LocalREPL._capture_answer = gated
from rlm.logger import RLMLogger  # noqa: E402
from tools import ToolBox  # noqa: E402


def load_query(query_id: str) -> str:
    """Return only the question text for this query id; the answer is never held in this process."""
    _qf = os.environ.get("QUERIES_FILE") or f"{path('FULLDOC_ROOT')}/data/prepared/queries.jsonl"
    with open(_qf, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["query_id"] == query_id:
                return d["query"]
    raise KeyError(f"query_id {query_id} not found")


def build_rlm(toolbox: ToolBox, args, logger) -> RLM:
    # OpenAI-compatible endpoint (e.g. vLLM) with an explicit base_url.
    if args.backend == "openai":
        key = None
        for _v in (args.api_key_env, "ALIBABA_TOKEN", "DASHSCOPE_API_KEY", "OPENAI_API_KEY"):
            if _v and os.getenv(_v):
                key = os.getenv(_v)
                break
        if not key:
            raise SystemExit("no API key found for --backend openai (looked at %s)" % args.api_key_env)
        common = {"api_key": key, "base_url": args.base_url, "timeout": args.lm_timeout}
        return RLM(
            backend="openai",
            backend_kwargs={**common, "model_name": args.root_model},
            other_backends=["openai"],
            other_backend_kwargs=[{**common, "model_name": args.sub_model}],
            custom_system_prompt=(SOLO_SYSTEM_PROMPT if args.solo_prompt else None),
            environment="local",
            environment_kwargs={"no_delegation": args.no_delegation},
            depth=0,
            max_depth=1,
            max_iterations=args.max_iterations,
            max_timeout=args.max_timeout,
            custom_tools=toolbox.custom_tools(),
            custom_sub_tools={},
            user_prologue=build_prologue(args.search_budget, no_delegation=args.no_delegation,
                                         lean=args.lean_prologue,
                                         solo=args.solo_prompt,
                                         single_pass=args.single_pass_read,
                                         constraints=args.constraint_check,
                                         structured_plan=args.structured_plan),
            orchestrator=args.orchestrator,
            compaction=args.compaction,
            compaction_threshold_pct=args.compaction_threshold_pct,
            max_concurrent_subcalls=args.max_concurrent_subcalls,
            sampling_args=_sampling(args.reasoning_effort),
            sub_sampling_args=_sampling(args.sub_reasoning_effort),
            logger=logger,
            verbose=False,
        )
    common = {
        "azure_endpoint": os.getenv("AZURE2_ENDPOINT"),
        "api_key": os.getenv("AZURE2_API_KEY"),
        "api_version": os.getenv("AZURE2_API_VERSION"),
        "timeout": args.lm_timeout,
    }
    # The Azure deployment overrides the model field; pin it rather than inherit AZURE_OPENAI_DEPLOYMENT.
    return RLM(
        backend="azure_openai",
        backend_kwargs={**common, "model_name": args.root_model,
                        "azure_deployment": args.root_deployment or args.root_model},
        other_backends=["azure_openai"],
        other_backend_kwargs=[{**common, "model_name": args.sub_model,
                               "azure_deployment": args.sub_deployment or args.sub_model}],
        custom_system_prompt=(SOLO_SYSTEM_PROMPT if args.solo_prompt else None),
        environment="local",
        environment_kwargs={"no_delegation": args.no_delegation},
        depth=0,
        max_depth=1,  # root REPL only; rlm_query degrades to llm_query
        max_iterations=args.max_iterations,
        max_timeout=args.max_timeout,
        custom_tools=toolbox.custom_tools(),
        custom_sub_tools={},  # sub-LLMs are plain completions, no tools
        user_prologue=build_prologue(args.search_budget, no_delegation=args.no_delegation,
                                     lean=args.lean_prologue,
                                     solo=args.solo_prompt,
                                     single_pass=args.single_pass_read,
                                     constraints=args.constraint_check,
                                     structured_plan=args.structured_plan),
        orchestrator=args.orchestrator,
        compaction=args.compaction,
        compaction_threshold_pct=args.compaction_threshold_pct,
        max_concurrent_subcalls=args.max_concurrent_subcalls,
        sampling_args=_sampling(args.reasoning_effort),
        sub_sampling_args=_sampling(args.sub_reasoning_effort),
        logger=logger,
        verbose=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--design", default="A", choices=["A", "B", "D"])
    ap.add_argument("--server", default="http://127.0.0.1:8765")
    ap.add_argument("--backend", default="azure_openai", choices=["azure_openai", "openai"])
    ap.add_argument("--base-url", default=None, help="base_url for --backend openai")
    ap.add_argument("--api-key-env", default="ALIBABA_TOKEN",
                    help="env var holding the key for --backend openai")
    ap.add_argument("--root-model", default="gpt-5")
    ap.add_argument("--sub-model", default="gpt-5-mini")
    # Deployments default to the model name.
    ap.add_argument("--root-deployment", default=None)
    ap.add_argument("--sub-deployment", default=None)
    ap.add_argument("--expect-served-model", default=None,
                    help="Abort before any work if the endpoint serves a different build than this.")
    ap.add_argument("--reasoning-effort", default="high")
    ap.add_argument("--sub-reasoning-effort", default="high")
    ap.add_argument("--default-k", type=int, default=5)
    ap.add_argument("--snippet-max-tokens", type=int, default=512)
    ap.add_argument("--max-iterations", type=int, default=30)
    ap.add_argument("--max-timeout", type=float, default=1800.0)
    ap.add_argument("--lm-timeout", type=float, default=600.0)
    ap.add_argument("--no-orchestrator", dest="orchestrator", action="store_false",
                    help="drop rlm's stock ORCHESTRATOR_ADDENDUM (it advises wide sub-call fan-out,\n"
                         "which is tuned for long-context chunking rather than retrieval)")
    ap.set_defaults(orchestrator=True)
    ap.add_argument("--fixed-k", action="store_true",
                    help="pin k to --default-k for every search; the model cannot change it. "
                         "A declared tool contract, like the official k=5.")
    ap.add_argument("--max-searches", type=int, default=0,
                    help="hard cap on search calls; 0 = uncapped. Enforced by the harness "
                         "and declared in the tool description.")
    ap.add_argument("--search-budget", type=int, default=0,
                    help="0 = no budget clause (the honest measurement). If >0 the prompt asks the "
                         "agent to aim for that many searches; it is recorded in the run file and "
                         "the result must be reported as budget-instructed.")
    ap.add_argument("--grounding-gate", action="store_true",
                    help="once per run, hold a submission whose Exact Answer string occurs in no "
                         "retrieved document and ask the agent to check the source spelling")
    ap.add_argument("--compaction", action="store_true",
                    help="enable rlm's built-in history compaction (library feature).")
    ap.add_argument("--structured-plan", action="store_true",
                    help="plan as 2-4 numbered steps rather than a prose paragraph; restores the "
                         "structure the orchestrator addendum used to produce, without its "
                         "delegation language.")
    ap.add_argument("--constraint-check", action="store_true",
                    help="track the question's conditions in a REPL variable and verify each one "
                         "against recorded evidence before submitting (decomposition at the "
                         "verification stage, not the retrieval stage).")
    ap.add_argument("--single-pass-read", action="store_true",
                    help="ask for everything to be extracted from a document on first open, "
                         "instead of returning to it on a later turn.")
    ap.add_argument("--solo-prompt", action="store_true",
                    help="replace the stock RLM system prompt with one that has no delegation "
                         "surface at all (no llm_query/rlm_query, no orchestrator framing).")
    ap.add_argument("--lean-prologue", action="store_true",
                    help="state the seeded starting namespace, forbid the turn-1 orientation "
                         "ritual, and ask for terse uncommented code.")
    ap.add_argument("--no-delegation", action="store_true",
                    help="remove the depth-1 sub-call surface entirely (llm_query, read_docs, ...); "
                         "the root must read document text itself in the REPL.")
    ap.add_argument("--compaction-threshold-pct", type=float, default=0.85,
                    help="compact when root history reaches this fraction of the model context "
                         "limit (gpt-5 limit is 272000; default 0.85 = 231200 tokens, which no "
                         "run has ever reached).")
    ap.add_argument("--max-concurrent-subcalls", type=int, default=4,
                    help="parallel threads for batched sub-calls (library default 4).")
    ap.add_argument("--seed-context", action="store_true",
                    help="run search(question) before turn 1 and load the result as `context` "
                         "(counted as search #1). Makes the stock RLM prompt's `context` framing "
                         "true instead of a stub, and every leaderboard agent starts this way.")
    ap.add_argument("--save-trajectory", action="store_true")
    args = ap.parse_args()

    question = load_query(args.query_id)
    tag = f"{args.query_id}-{uuid.uuid4().hex[:8]}"

    toolbox = ToolBox(
        base_url=args.server,
        design=args.design,
        default_k=args.default_k,
        snippet_max_tokens=args.snippet_max_tokens,
        tag=tag,
        max_searches=args.max_searches,
        allow_k_override=(False if args.fixed_k else None),
    )
    logger = RLMLogger()
    if args.grounding_gate:
        install_grounding_gate(toolbox)
    from urllib.parse import urlparse as _up
    _hosts = [_up(os.getenv(v) or "").hostname for v in ("AZURE2_ENDPOINT", "AZURE_ENDPOINT", "AZURE_OPENAI_ENDPOINT")]
    if args.backend == "openai" and args.base_url:
        _hosts.append(_up(args.base_url).hostname)
    harden_sandbox([h for h in _hosts if h])

    record = {
        "query_id": args.query_id,
        "design": args.design,
        "run_tag": tag,
        "config": {
            "root_model": args.root_model,
            "sub_model": args.sub_model,
            "reasoning_effort": args.reasoning_effort,
            "sub_reasoning_effort": args.sub_reasoning_effort,
            "default_k": args.default_k,
            "k_settable_by_model": toolbox.allow_k_override,
            "snippet_max_tokens": args.snippet_max_tokens,
            "max_iterations": args.max_iterations,
            "depth": 0,
            "max_depth": 1,
            "prompt_id": prologue_id(),
            "orchestrator_addendum": args.orchestrator,
            "search_budget_in_prompt": args.search_budget or None,
            "hard_search_cap": args.max_searches or None,
            "fixed_k": args.default_k if args.fixed_k else None,
            "seed_context": args.seed_context,
            "compaction": args.compaction,
            "compaction_threshold_pct": args.compaction_threshold_pct,
            "no_delegation": args.no_delegation,
            "lean_prologue": args.lean_prologue,
            "solo_prompt": args.solo_prompt,
            "single_pass_read": args.single_pass_read,
            "constraint_check": args.constraint_check,
            "structured_plan": args.structured_plan,
            "max_concurrent_subcalls": args.max_concurrent_subcalls,
            "grounding_gate": args.grounding_gate,
            "retrieval_server": args.server,
            "max_timeout_s": args.max_timeout,
        },
    }

    t0 = time.time()
    try:
        agent = build_rlm(toolbox, args, logger)
        if args.seed_context:
            seed = toolbox.search(question)  # counted; goes through the same tag
            payload = {
                "note": ("`context` holds the results of the first search, issued with the "
                         "question verbatim. The same records are already in DOCS. Further "
                         "retrieval is via search; DOCS and SEARCH_LOG grow."),
                "first_search": seed,
            }
        else:
            payload = CONTEXT_NOTE
        result = agent.completion(
            payload, root_prompt=QUERY_TEMPLATE.format(Question=question)
        )
        output = (result.response or "").strip()
        record["status"] = "completed" if output else "empty_output"
        record["result"] = [{"type": "output_text", "output": output}]
        try:
            record["usage"] = result.usage_summary.to_dict()
        except Exception:
            record["usage"] = None
        traj = logger.get_trajectory()
        record["n_iterations"] = len(traj["iterations"]) if traj else None
        if args.save_trajectory and traj:
            record["trajectory"] = traj
    except Exception as e:
        record["status"] = "error"
        record["error"] = f"{type(e).__name__}: {e}"
        record["traceback"] = traceback.format_exc()[-3000:]
        record["result"] = [{"type": "output_text", "output": ""}]
        try:
            traj = logger.get_trajectory()
            record["n_iterations"] = len(traj["iterations"]) if traj else None
            if args.save_trajectory and traj:
                record["trajectory"] = traj
        except Exception:
            pass

    record["tool_call_counts"] = dict(toolbox.tool_call_counts)
    record["server_counts"] = toolbox.server_counts()
    try:
        import urllib.request as _u
        record["retriever_stats"] = json.loads(_u.urlopen(args.server.rstrip("/") + "/stats", timeout=10).read())
    except Exception as _e:
        record["retriever_stats"] = {"error": str(_e)[:120]}
    record["retrieved_docids"] = toolbox.retrieved_docids
    record["server_docids"] = toolbox.server_docids()
    record["sandbox_flags"] = scan_trajectory(record.get("trajectory") or logger.get_trajectory())
    record["code_digest"] = os.getenv("RLM_CODE_DIGEST")
    try:
        if args.backend == "openai":
            from rlm.clients.openai import SERVED_MODELS
        else:
            from rlm.clients.azure_openai import SERVED_MODELS
        record["served_models"] = dict(SERVED_MODELS)
    except Exception:
        record["served_models"] = None
    record["search_log"] = toolbox.SEARCH_LOG
    record["search_refused"] = toolbox.search_refused
    record["tool_errors"] = toolbox.errors
    record["wall_time_s"] = round(time.time() - t0, 1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
    os.replace(tmp, args.out)

    sc = record["server_counts"] or {}
    ks = sc.get("k_sum", 0)
    n = max(1, record["tool_call_counts"]["search"])
    print(
        f"[{args.query_id}] {record['status']} "
        f"search={record['tool_call_counts']['search']}"
        f"(srv={sc.get('search', '?')}) "
        f"avg_k={ks / n:.1f} docs={len(record['retrieved_docids'])} "
        f"refused={record['search_refused']} "
        f"iters={record.get('n_iterations')} {record['wall_time_s']}s"
    )


if __name__ == "__main__":
    main()
