#!/usr/bin/env python3
"""Padding test on BrowseComp-Plus: the same evidence inside prompts padded with 0 to 200k tokens of other retrieved text.

Per question, the gold documents sit at a fixed relative position among nested sets of the dense retriever's top non-gold
documents; one GPT-5 call per prompt, no tools. Records go to data/x3/<prefix>_pad<L>k/runs/<qid>.json for the judge.
  python runners/x6_padding.py --build | --dry-run | --run --workers 32
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from x3_plain_agent import load_questions  # noqa: E402

LENGTHS = [0, 25_000, 50_000, 100_000, 200_000]
DOC_CAP = 12_000          # tokens kept per document (gold and distractor alike)
EVIDENCE_CAP = 40_000     # tokens for the whole evidence block; longest gold documents are cut first
MAX_PROMPT = 262_000      # margin under GPT-5's 272k input limit
SEED = 20261101
POOL_K = 300              # dense hits fetched per question to build the distractor list

TASK = """You are a deep research agent. Answer the question using the documents below. They were retrieved from a fixed corpus for this question; some are relevant and most are not. You cannot search; everything you need is in the documents.

Reply in the following format:

Explanation: {your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their doc_ids in square brackets [] at the end of sentences. For example, [94480].}
Exact Answer: {the exact name, date, title, or value the question asks for}
Confidence: {your confidence score between 0% and 100% for your answer}"""

_enc = None


def enc():
    global _enc
    if _enc is None:
        import tiktoken
        _enc = tiktoken.get_encoding("o200k_base")
    return _enc


def ntok(s: str) -> int:
    return len(enc().encode(s, disallowed_special=()))


def cut(s: str, n: int) -> str:
    ids = enc().encode(s, disallowed_special=())
    return s if len(ids) <= n else enc().decode(ids[:n])


class Server:
    def __init__(self, url: str, cache_dir: Path):
        self.url = url.rstrip("/")
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def _post(self, path, payload, timeout=180):
        req = urllib.request.Request(self.url + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    def search_ids(self, query: str, k: int) -> list[str]:
        return [str(r["doc_id"]) for r in self._post("/search", {"query": query, "top_k": k})["results"]]

    def doc(self, doc_id: str) -> str:
        p = self.cache_dir / f"{doc_id}.txt"
        if p.exists():
            return p.read_text(encoding="utf-8")
        text = self._post("/document", {"doc_id": str(doc_id)}).get("text") or ""
        p.write_text(text, encoding="utf-8")
        return text


def block(doc_id: str, text: str) -> str:
    return f"### Document {doc_id}\n{text.strip()}\n\n"


def build_manifest(qs, subset, srv: Server, lengths) -> dict:
    need = max(lengths) + DOC_CAP + 2_000
    man = {"seed": SEED, "doc_cap_tokens": DOC_CAP, "evidence_cap_tokens": EVIDENCE_CAP, "lengths": lengths, "pool_k": POOL_K,
           "server": srv.url, "built": time.strftime("%Y-%m-%d %H:%M:%S"), "questions": {}}
    for i, qid in enumerate(subset, 1):
        q = qs[qid]
        gold = sorted(q["gold"], key=lambda x: (len(x), x))
        rng = random.Random(f"{SEED}:{qid}")
        # evidence block: whole gold documents, each capped, then the longest cut until the block fits EVIDENCE_CAP
        gtok = {g: ntok(srv.doc(g)) for g in gold}
        caps = {g: min(gtok[g], DOC_CAP) for g in gold}
        while sum(caps.values()) > EVIDENCE_CAP:
            longest = max(caps, key=caps.get)
            caps[longest] = max(1_000, caps[longest] - 1_000)
            if all(c <= 1_000 for c in caps.values()):
                break
        # distractors: dense hits for the question, minus gold and evidence documents, in rank order, nested
        exclude = set(q["gold"]) | set(q["evidence"])
        pool = [d for d in srv.search_ids(q["question"], POOL_K) if d not in exclude]
        dis, total = [], 0
        for d in pool:
            t = min(ntok(srv.doc(d)), DOC_CAP)
            dis.append({"id": d, "tokens": t})
            total += t
            if total >= need:
                break
        man["questions"][qid] = {
            "gold": [{"id": g, "tokens": gtok[g], "kept": caps[g]} for g in gold],
            "evidence_tokens": sum(caps.values()),
            "evidence_truncated": any(caps[g] < gtok[g] for g in gold),
            "position": rng.random(),
            "distractors": dis,
            "distractor_tokens_available": total,
            "pool_short": total < need,
        }
        if i % 10 == 0 or i == len(subset):
            print(f"  built {i}/{len(subset)}", flush=True)
    return man


def assemble(qid: str, q: dict, m: dict, L: int, srv: Server) -> tuple[list[dict], dict]:
    """Messages for question qid at padding L, plus bookkeeping (document order, exact tokens)."""
    gold_blocks = [block(g["id"], cut(srv.doc(g["id"]), g["kept"])) for g in m["gold"]]
    dis_blocks, used = [], 0
    for d in m["distractors"]:
        if used >= L:
            break
        text = cut(srv.doc(d["id"]), min(d["tokens"], L - used))
        dis_blocks.append((d["id"], block(d["id"], text)))
        used += ntok(block(d["id"], text))
    pos = round(m["position"] * len(dis_blocks)) if dis_blocks else 0
    order = [d for d, _ in dis_blocks[:pos]] + [g["id"] for g in m["gold"]] + [d for d, _ in dis_blocks[pos:]]
    body = "".join(b for _, b in dis_blocks[:pos]) + "".join(gold_blocks) + "".join(b for _, b in dis_blocks[pos:])
    user = (f"Answer the following: {q['question']}\n\n{TASK}\n\n{body}"
            f"End of documents. Now answer the question above in the required format.")
    messages = [{"role": "system", "content": "You are a careful research assistant."}, {"role": "user", "content": user}]
    info = {"padding_target": L, "distractor_docs": len(dis_blocks), "distractor_tokens": used, "gold_position_index": pos,
            "doc_order": order, "prompt_tokens_estimate": ntok(user) + 20}
    return messages, info


def call_once(client, cfg, messages, rec):
    kwargs = dict(model=cfg["deployment"], messages=messages, max_completion_tokens=cfg["max_completion_tokens"])
    if cfg["effort"]:
        kwargs["reasoning_effort"] = cfg["effort"]
    for attempt in range(10):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001
            rec["errors"].append(f"attempt {attempt}: {type(e).__name__}: {str(e)[:160]}")
            if "context_length" in str(e) or "maximum context length" in str(e):
                return None
            time.sleep(min(120, 5 * 2 ** attempt) + random.uniform(0, 3))
    return None


def run_item(qid, L, q, m, srv, client, cfg, out_path):
    t0 = time.time()
    messages, info = assemble(qid, q, m, L, srv)
    rec = dict(query_id=qid, status="running", errors=[], config=cfg, **info, evidence_tokens=m["evidence_tokens"],
               evidence_truncated=m["evidence_truncated"], gold_docids=[g["id"] for g in m["gold"]], final_answer=None)
    if info["prompt_tokens_estimate"] > MAX_PROMPT:
        rec["status"] = "too_long"
        out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        return rec
    resp = call_once(client, cfg, messages, rec)
    if resp is None:
        rec["status"] = "error"
    else:
        u = resp.usage
        rec["usage"] = dict(prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
                            cached_tokens=getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", None),
                            reasoning_tokens=getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", None))
        rec["final_answer"] = resp.choices[0].message.content or ""
        rec["finish_reason"] = resp.choices[0].finish_reason
        rec["status"] = "completed" if rec["final_answer"].strip() else "empty"
        rec["final_context_tokens"] = u.prompt_tokens
        rec["cum_input_tokens"] = u.prompt_tokens
        rec["cum_output_tokens"] = u.completion_tokens
    rec["wall_time_s"] = round(time.time() - t0, 1)
    out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return rec


def run_name(prefix: str, L: int) -> str:
    return f"{prefix}_pad{L // 1000:03d}k"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="x6")
    ap.add_argument("--manifest", default=str(ROOT / "data/x3/x6_manifest.json"), help="prompt manifest shared by every prefix (smoke runs included)")
    ap.add_argument("--subset", default=str(ROOT / "data/analysis/x3_subset.json"))
    ap.add_argument("--server", default=os.environ.get("X3_SERVER", "http://127.0.0.1:8001"))
    ap.add_argument("--lengths", default=",".join(str(x) for x in LENGTHS))
    ap.add_argument("--deployment", default="")
    ap.add_argument("--effort", default=os.environ.get("X3_EFFORT", "high"))
    ap.add_argument("--env-file", default=None)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="only the first N subset questions")
    ap.add_argument("--build", action="store_true", help="(re)build the manifest and print the cost estimate; no model call")
    ap.add_argument("--dry-run", action="store_true", help="assemble one question's prompts and print token counts; no model call")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    lengths = [int(x) for x in a.lengths.split(",")]
    from x3_env import load_env
    load_env(a.env_file)
    a.deployment = a.deployment or os.environ.get("GPT5_DEPLOYMENT", "")

    qs = load_questions()
    subset = json.load(open(a.subset))["query_ids"]
    if a.limit:
        subset = subset[: a.limit]
    base = ROOT / "data" / "x3"
    srv = Server(a.server, base / "_x6_doc_cache")
    man_path = Path(a.manifest)

    if a.build or not man_path.exists():
        print(f"building manifest for {len(subset)} questions from {a.server} ...", flush=True)
        man = build_manifest(qs, subset, srv, lengths)
        man_path.write_text(json.dumps(man, indent=1))
    man = json.load(open(man_path))
    missing = [q for q in subset if q not in man["questions"]]
    if missing:
        sys.exit(f"manifest lacks {len(missing)} subset questions; run with --build")

    # Input-token estimate from the manifest.
    est = {L: sum(min(man["questions"][q]["evidence_tokens"] + L, MAX_PROMPT) + 400 for q in subset) for L in lengths}
    tot = sum(est.values())
    trunc = sum(man["questions"][q]["evidence_truncated"] for q in subset)
    short = sum(man["questions"][q]["pool_short"] for q in subset)
    print(f"{len(subset)} questions x {len(lengths)} lengths = {len(subset) * len(lengths)} prompts; "
          f"input tokens ~{tot / 1e6:.1f} M (~{tot / 1e6 * 1.25:.0f} USD at 1.25 USD/M list price, output extra); "
          f"evidence truncated for {trunc} questions; distractor pool short for {short}")
    if a.build and not a.run and not a.dry_run:
        return

    if a.dry_run:
        qid = subset[0]
        for L in lengths:
            msgs, info = assemble(qid, qs[qid], man["questions"][qid], L, srv)
            print(f"  L={L:>7}: distractor docs {info['distractor_docs']:>3}, distractor tokens {info['distractor_tokens']:>7}, "
                  f"gold at index {info['gold_position_index']}, prompt tokens ~{info['prompt_tokens_estimate']}")
        print("credentials present:", all(os.environ.get(k) for k in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY")),
              "| deployment:", a.deployment or "(unset)")
        return

    if not a.run:
        return
    for k in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"):
        if not os.environ.get(k):
            sys.exit(f"missing {k}")
    if not a.deployment:
        sys.exit("missing --deployment / GPT5_DEPLOYMENT")
    from openai import AzureOpenAI
    client = AzureOpenAI(api_key=os.environ["AZURE_OPENAI_API_KEY"], azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                         api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"), max_retries=2, timeout=1200)
    cfg = dict(experiment="AB6 padding replay", server=a.server, deployment=a.deployment, effort=a.effort, max_completion_tokens=48000,
               doc_cap_tokens=DOC_CAP, evidence_cap_tokens=EVIDENCE_CAP, manifest=man_path.name, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    todo = []
    for L in lengths:
        d = base / run_name(a.prefix, L)
        (d / "runs").mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps(dict(cfg, padding=L), indent=1))
        todo += [(qid, L) for qid in subset if not (d / "runs" / f"{qid}.json").exists()]
    todo.sort(key=lambda x: (subset.index(x[0]), x[1]))   # complete questions across all lengths first
    print(f"{len(todo)} prompts to run with {a.workers} workers", flush=True)
    done, lock = 0, threading.Lock()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(run_item, qid, L, qs[qid], man["questions"][qid], srv, client, cfg,
                          base / run_name(a.prefix, L) / "runs" / f"{qid}.json"): (qid, L) for qid, L in todo}
        for f in as_completed(futs):
            qid, L = futs[f]
            try:
                r = f.result()
                with lock:
                    done += 1
                    u = r.get("usage") or {}
                    print(f"[{done}/{len(todo)}] q{qid} L={L}: {r['status']} in={u.get('prompt_tokens')} out={u.get('completion_tokens')} "
                          f"reason={u.get('reasoning_tokens')} t={r.get('wall_time_s')}s errors={len(r['errors'])}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"q{qid} L={L}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)


if __name__ == "__main__":
    main()
