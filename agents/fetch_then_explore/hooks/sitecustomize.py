"""Startup hook for Fetch-then-Explore runs on BrowseComp-Plus (on PYTHONPATH; FtE's code is not modified).

1. Egress off: the process may open network connections only to FTE_ALLOWED_HOSTS (the Azure endpoint host) and the
   loopback interface (local search adapter). Any other DNS lookup or connect raises and is logged to FTE_EGRESS_LOG.
2. GPT-5 compatibility for FtE's chat-completions client (FTE_GPT5_SHIM=1): max_tokens -> max_completion_tokens;
   temperature, top_p, presence_penalty dropped (GPT-5 accepts only defaults); the deployment is pinned to
   FTE_DEPLOYMENT whatever model name the caller passes; reasoning_effort comes from AGENT_EXTRA_BODY.
3. Every response's served model is appended to FTE_SERVED_LOG, so a run can prove which model answered.
"""
import json, os, socket, sys, time

_ALLOWED = {h.strip().lower() for h in os.environ.get("FTE_ALLOWED_HOSTS", "").split(",") if h.strip()}
_LOOP = {"127.0.0.1", "::1", "localhost"}
_LOG = os.environ.get("FTE_EGRESS_LOG")
_orig_gai, _orig_connect = socket.getaddrinfo, socket.socket.connect
_ALLOWED_IPS = set(_LOOP)
for _h in _ALLOWED:
    try:
        _ALLOWED_IPS |= {ai[4][0] for ai in _orig_gai(_h, 443)}
    except OSError:
        pass


def _deny(what):
    msg = f"[egress] BLOCKED {what}"
    print(msg, file=sys.stderr, flush=True)
    if _LOG:
        with open(_LOG, "a") as f:
            f.write(json.dumps({"t": time.time(), "blocked": what}) + "\n")
    raise OSError(msg)


def _gai(host, *a, **k):
    h = host.decode() if isinstance(host, bytes) else str(host)
    if h.lower() not in _ALLOWED and h not in _LOOP and h not in _ALLOWED_IPS:
        _deny(f"lookup {h}")
    res = _orig_gai(host, *a, **k)
    if h.lower() in _ALLOWED:
        # the allowed host sits behind Azure Traffic Manager and rotates between gateway IPs: allow the addresses it
        # resolves to NOW (a startup-only list blocked the same host after a rotation, 2026-11-05)
        _ALLOWED_IPS.update(ai[4][0] for ai in res)
    return res


def _connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6) and isinstance(address, tuple) and address[0] not in _ALLOWED_IPS:
        _deny(f"connect {address[0]}")
    return _orig_connect(self, address)


if _ALLOWED:
    socket.getaddrinfo, socket.socket.connect = _gai, _connect

if os.environ.get("FTE_GPT5_SHIM") == "1":
    try:
        import openai.resources.chat.completions as _cc
        _mods = [getattr(_cc, "completions", _cc)]
        _dep, _served = os.environ.get("FTE_DEPLOYMENT", ""), os.environ.get("FTE_SERVED_LOG")

        def _fix(kw):
            if "max_tokens" in kw:
                kw["max_completion_tokens"] = kw.pop("max_tokens")
            for p in ("temperature", "top_p", "presence_penalty"):
                kw.pop(p, None)
            if _dep:
                kw["model"] = _dep
            return kw

        def _note(resp):
            if _served:
                with open(_served, "a") as f:
                    f.write(json.dumps({"t": time.time(), "model": getattr(resp, "model", None)}) + "\n")
            return resp

        for _m in _mods:
            _A, _S = getattr(_m, "AsyncCompletions"), getattr(_m, "Completions")
            _acreate, _screate = _A.create, _S.create

            async def _ac(self, *a, _orig=_acreate, **kw):
                return _note(await _orig(self, *a, **_fix(kw)))

            def _sc(self, *a, _orig=_screate, **kw):
                return _note(_orig(self, *a, **_fix(kw)))
            _A.create, _S.create = _ac, _sc
    except Exception as e:  # noqa: BLE001
        print(f"[fte-shim] not installed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        raise
