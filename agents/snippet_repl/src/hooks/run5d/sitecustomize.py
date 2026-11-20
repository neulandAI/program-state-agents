"""Startup hook bundle: retry429, azure_sampling and cap_ranking, plus local-server balancing.

Python imports only the first sitecustomize.py on sys.path, so this one loads the others."""
import os

_H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _name in ("retry429", "azure_sampling", "cap_ranking"):
    _p = os.path.join(_H, _name, "sitecustomize.py")
    exec(compile(open(_p).read(), _p, "exec"), {"__name__": f"sitecustomize_{_name}", "__file__": _p})

# If LOCAL_LLM_BASE_URL names :8030 or :8031 (same model on both), use the one with fewer running + waiting requests.
try:
    import re as _re
    import urllib.request as _ur

    _u = os.environ.get("LOCAL_LLM_BASE_URL", "")
    _m = _re.fullmatch(r"http://localhost:(803[01])/v1/?", _u)
    if _m:
        def _load(port):
            txt = _ur.urlopen(f"http://localhost:{port}/metrics", timeout=5).read().decode()
            return sum(float(v) for v in _re.findall(r"^vllm:num_requests_(?:running|waiting)\{[^}]*\} ([0-9.]+)", txt, _re.M))
        _loads = {p: _load(p) for p in (8030, 8031)}
        _best = min(_loads, key=_loads.get)
        if _best != int(_m.group(1)):
            os.environ["LOCAL_LLM_BASE_URL"] = f"http://localhost:{_best}/v1"
        print(f"[balance] running+waiting {_loads} -> :{_best}", flush=True)
except Exception as _e:  # noqa: BLE001
    print(f"[balance] hook skipped: {_e}", flush=True)
