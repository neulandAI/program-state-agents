"""Record the state that produced a run: hashes of code, data and index, package versions, environment names.

    python src/provenance.py <run_dir> [<run_dir> ...]   -> <run_dir>/_provenance.json

Values are hashes and names only, never environment values. Called by run_experiment.py.
"""
import hashlib, json, os, platform, subprocess, sys, time, glob, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, os.pardir, "shared"))
from rlm_shared.paths import path  # noqa: E402

DATA_ROOT = str(path("FULLDOC_ROOT"))

def sha(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def tree(patterns, root=ROOT):
    out = {}
    for pat in patterns:
        for p in sorted(glob.glob(os.path.join(root, pat), recursive=True)):
            if os.path.isfile(p) and "__pycache__" not in p:
                out[os.path.relpath(p, root)] = {"sha256": sha(p), "bytes": os.path.getsize(p)}
    return out

def main():
    code = tree(["src/*.py", "src/*.sh", "rlm_bcplus/**/*.py", "rlm_bcplus/**/*.sh", "rlm_bcplus/README.md"])
    upstream = tree(["rlm_upstream/rlm/clients/azure_openai.py", "rlm_upstream/rlm/core/lm_handler.py",
                     "rlm_upstream/rlm/core/types.py", "rlm_upstream/rlm/environments/local_repl.py"], DATA_ROOT)
    # the installed copy is what actually ran
    try:
        import rlm as _rlm
        inst = tree([os.path.relpath(os.path.dirname(_rlm.__file__), ROOT) + "/**/*.py"]) if os.path.dirname(_rlm.__file__).startswith(ROOT) else {}
        installed_rlm_path = os.path.dirname(_rlm.__file__)
    except Exception as e:
        inst, installed_rlm_path = {}, f"import failed: {e}"
    data = tree(["data/prepared/*.json", "data/prepared/*.jsonl"], DATA_ROOT)
    index = tree(["data/indexes/**/*"], DATA_ROOT)
    freeze = []
    for cmd in (["uv", "pip", "freeze", "--python", sys.executable], [sys.executable, "-m", "pip", "freeze"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if out.returncode == 0 and out.stdout.strip():
                freeze = out.stdout.splitlines(); break
        except Exception:
            continue
    if not freeze:
        freeze = ["<pip freeze unavailable>"]
    # HuggingFace snapshots actually on disk (dataset / index / model revisions)
    hf = {}
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    for d in glob.glob(os.path.join(hub, "*", "snapshots", "*")):
        repo = os.path.basename(os.path.dirname(os.path.dirname(d)))
        hf.setdefault(repo, {}).setdefault("snapshots", []).append(os.path.basename(d))
    # hf_hub_download metadata sidecars carry the commit hash even when no snapshots dir exists
    for m in glob.glob(os.path.join(hub, "**", "*.metadata"), recursive=True):
        try:
            repo = m.split(hub + os.sep, 1)[1].split(os.sep)[0]
            for line in open(m):
                if len(line.strip()) == 40 and all(c in "0123456789abcdef" for c in line.strip()):
                    hf.setdefault(repo, {}).setdefault("commit_hashes", set()).add(line.strip())
        except Exception:
            pass
    for ref in glob.glob(os.path.join(hub, "*", "refs", "*")):
        try:
            repo = os.path.basename(os.path.dirname(os.path.dirname(ref)))
            hf.setdefault(repo, {}).setdefault("refs", {})[os.path.basename(ref)] = open(ref).read().strip()
        except Exception:
            pass
    hf = {k: {kk: (sorted(vv) if isinstance(vv, set) else vv) for kk, vv in v.items()} for k, v in hf.items()}
    # rlm_upstream: base commit + hash of our patch
    try:
        base = subprocess.run(["git", "-C", os.path.join(DATA_ROOT, "rlm_upstream"), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        diff = subprocess.run(["git", "-C", os.path.join(DATA_ROOT, "rlm_upstream"), "diff"], capture_output=True, text=True).stdout
        upstream_git = {"base_commit": base, "patch_sha256": hashlib.sha256(diff.encode()).hexdigest(), "patch_lines": diff.count("\n")}
    except Exception as e:
        upstream_git = {"error": str(e)}
    try:
        repo_commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        repo_dirty = bool(subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--", "src", "rlm_bcplus"], capture_output=True, text=True).stdout.strip())
    except Exception:
        repo_commit, repo_dirty = None, None
    dotenv_names = []
    try:
        dotenv_names = sorted(l.split("=", 1)[0].strip() for l in open(os.path.join(DATA_ROOT, ".env")) if "=" in l and not l.strip().startswith("#"))
    except Exception:
        pass
    try:
        stats = json.load(urllib.request.urlopen("http://127.0.0.1:8765/stats", timeout=10))
    except Exception as e:
        stats = {"error": str(e)}
    prov = {
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(), "python": sys.version, "platform": platform.platform(),
        "env_var_names_present": sorted(k for k in os.environ if k.startswith(("AZURE", "HF_", "OPENAI"))),
        "code": code, "rlm_upstream_patched_files": upstream, "installed_rlm": {"path": installed_rlm_path, "files": inst},
        "data": data, "index": index, "pip_freeze": freeze, "retrieval_server_stats": stats,
        "hf_snapshots": hf, "rlm_upstream_git": upstream_git, "repo_git": {"commit": repo_commit, "src_or_package_dirty": repo_dirty},
        "dotenv_var_names": dotenv_names, "launch_argv": os.getenv("RLM_LAUNCH_ARGV"),
        "models": {"root": os.getenv("RLM_ROOT_MODEL"), "sub": os.getenv("RLM_SUB_MODEL"), "judge": "gpt-4.1 (deployment)",
                   "azure_api_version_var": "AZURE2_API_VERSION"},
        "encoder": {"model": "Qwen/Qwen3-Embedding-8B", "index_source": "Tevatron/browsecomp-plus-indexes (qwen3-embedding-8b)",
                    "query_prefix": "Instruct: Given a web search query, retrieve relevant passages that answer the query\\nQuery:"},
    }
    prov["code_digest"] = hashlib.sha256(json.dumps({k: v["sha256"] for k, v in code.items()}, sort_keys=True).encode()).hexdigest()[:16]
    for d in sys.argv[1:]:
        os.makedirs(d, exist_ok=True)
        json.dump(prov, open(os.path.join(d, "_provenance.json"), "w"), indent=1)
        print(f"{d}: _provenance.json written; code_digest {prov['code_digest']}; {len(code)} code files, {len(index)} index files, {len(data)} data files")

if __name__ == "__main__":
    main()
