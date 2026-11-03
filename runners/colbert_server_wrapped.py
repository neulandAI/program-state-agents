#!/usr/bin/env python3
"""Run the document REPL agent's retrieval_server_colbert.py unchanged, with runtime patches that keep it within GPU memory.

pylate's search batch is a retrieve() argument, so COLBERT_SEARCH_BATCH is injected there (ranking is unchanged), and the
CUDA cache is emptied after every search. Usage: see runners/colbert_server.sh.
"""
import inspect, os, runpy, sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "agents/document_repl/src"
sys.path.insert(0, str(SRC))

import http.server  # noqa: E402
import torch  # noqa: E402
from pylate import retrieve  # noqa: E402

# The default listen backlog of 5 resets connections when many agent workers search at once.
http.server.ThreadingHTTPServer.request_queue_size = 512
print("[wrapper] listen backlog 512", flush=True)

_bs = int(os.environ.get("COLBERT_SEARCH_BATCH", "65536"))
_orig_retrieve = retrieve.ColBERT.retrieve
if "batch_size" in inspect.signature(_orig_retrieve).parameters:
    def _retrieve(self, *a, **kw):
        kw.setdefault("batch_size", _bs)
        try:
            return _orig_retrieve(self, *a, **kw)
        finally:
            torch.cuda.empty_cache()
    retrieve.ColBERT.retrieve = _retrieve
    print(f"[wrapper] retrieve() batch_size={_bs}; empty_cache after each search", flush=True)
else:
    print("[wrapper] retrieve() has no batch_size parameter; running unpatched", flush=True)

runpy.run_path(str(SRC / "retrieval_server_colbert.py"), run_name="__main__")
