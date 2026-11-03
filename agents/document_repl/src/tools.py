"""REPL tools of the document REPL agent: search, search_many, grep_docs, unread_docs, DOCS and SEARCH_LOG.

Design A returns 512-token snippets at a fixed k (the BrowseComp-Plus tool surface); design B returns whole documents
with a model-chosen k up to MAX_K; design D returns selected passages. Every hit is merged into DOCS, every call is
logged in SEARCH_LOG, and the ToolBox counts searches and records the docids returned.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

MAX_K = 50  # ceiling on model-chosen k; well above the official k=5


class _TrackedDocs(dict):
    """A dict that remembers which docids the REPL has looked at. Lets the harness
    answer "which retrieved documents have you never opened?" — a fact, not a hint."""

    def __init__(self):
        super().__init__()
        self.accessed: set = set()

    def __getitem__(self, k):
        self.accessed.add(k)
        return super().__getitem__(k)

    def get(self, k, default=None):
        if k in self:
            self.accessed.add(k)
        return super().get(k, default)

    def items(self):
        # iterating all items counts as looking at all of them
        self.accessed.update(self.keys())
        return super().items()

    def values(self):
        self.accessed.update(self.keys())
        return super().values()


class ToolBox:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8765",
        design: str = "A",
        default_k: int = 5,
        snippet_max_tokens: int = 512,
        timeout: float = 300.0,
        allow_k_override: bool | None = None,
        tag: str | None = None,
        max_searches: int = 0,
    ):
        if design not in ("A", "B", "D"):
            raise ValueError(f"design must be A, B or D, got {design!r}")
        self.base_url = base_url.rstrip("/")
        self.design = design
        self.default_k = default_k
        self.snippet_max_tokens = snippet_max_tokens
        self.timeout = timeout
        # Only design B lets the model choose k.
        self.allow_k_override = (design == "B") if allow_k_override is None else allow_k_override
        self.tag = tag
        # Hard search cap; 0 = unlimited.
        self.max_searches = max_searches

        # The same objects are injected into the REPL.
        self.DOCS: dict[str, dict] = _TrackedDocs()
        self.SEARCH_LOG: list[dict] = []

        self.tool_call_counts = {"search": 0}
        self.search_refused = 0
        self.retrieved_docids: list[str] = []
        self._seen: set[str] = set()
        self.errors: list[str] = []

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        last = None
        for _ in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.load(r)
            except urllib.error.URLError as e:
                # Connection never established, so the server has not counted it: safe to retry.
                if isinstance(e, urllib.error.HTTPError):
                    raise RuntimeError(f"retrieval server error {e.code}: {e}") from e
                last = e
            except Exception as e:
                # May have been served and counted already; do not retry.
                raise RuntimeError(f"retrieval call failed: {e}") from e
        raise RuntimeError(f"retrieval server unreachable: {last}")

    def _record(self, docids: list[str]) -> None:
        for d in docids:
            if d not in self._seen:
                self._seen.add(d)
                self.retrieved_docids.append(d)

    def search(self, query: str, k: int | None = None) -> list[dict]:
        """Retrieve from the corpus. Counted; results mirrored into DOCS/SEARCH_LOG."""
        if self.max_searches and self.tool_call_counts["search"] >= self.max_searches:
            self.search_refused += 1
            print(f"[search refused] all {self.max_searches} search calls are used for this "
                  f"question; no further retrieval. Answer from the {len(self.DOCS)} documents "
                  f"already in DOCS (unread_docs() lists the ones you have not opened).")
            return []
        self.tool_call_counts["search"] += 1
        if not isinstance(query, str) or not query.strip():
            self.errors.append("search called with empty query")
            return []
        if k is None or not self.allow_k_override:
            k_eff = self.default_k
        else:
            try:
                k_eff = max(1, min(int(k), MAX_K))
            except (TypeError, ValueError):
                k_eff = self.default_k

        try:
            resp = self._post(
                "/search",
                {
                    "query": query,
                    "k": k_eff,
                    "mode": {"A": "snippet", "B": "full", "D": "passages"}[self.design],
                    "snippet_max_tokens": self.snippet_max_tokens,
                    "tag": self.tag,
                    "retriever": "hybrid" if self.design == "D" else "dense",
                    "n_passages": 15,
                    "per_doc": 3,
                },
            )
        except Exception as e:
            self.errors.append(f"search failed: {e}")
            return []

        results = resp.get("results", [])
        before = len(self._seen)
        self._record([r["docid"] for r in results])
        new_docs = len(self._seen) - before
        cap = f"/{self.max_searches}" if self.max_searches else ""
        print(f"[search {self.tool_call_counts['search']}{cap}] k={k_eff} -> {len(results)} hits, "
              f"{new_docs} new documents (DOCS holds {len(self._seen)})")
        for r in results:
            prev = self.DOCS.get(r["docid"], {})
            merged = {**prev, **r}
            # never let a snippet-only hit clobber a full text we already hold
            if prev.get("text") and not r.get("text"):
                merged["text"] = prev["text"]
            self.DOCS[r["docid"]] = merged
        self.SEARCH_LOG.append(
            {"query": query, "k": k_eff, "docids": [r["docid"] for r in results]}
        )
        return results

    def grep_docs(self, pattern: str, window: int = 400, max_hits: int = 40, flags=None) -> list[dict]:
        """Regex over every retrieved document's text. No retrieval, no LLM."""
        import re as _re
        fl = _re.IGNORECASE if flags is None else flags
        try:
            rx = _re.compile(pattern, fl)
        except _re.error as e:
            return [{"error": f"bad regex: {e}"}]
        out = []
        for docid, rec in self.DOCS.items():
            text = rec.get("text") or rec.get("snippet") or ""
            for m in rx.finditer(text):
                a, b = max(0, m.start() - window), min(len(text), m.end() + window)
                out.append({"docid": docid, "offset": m.start(), "snippet": text[a:b]})
                if len(out) >= max_hits:
                    return out
        return out

    def unread_docs(self) -> list[dict]:
        """Retrieved documents the REPL has never accessed, with their heads."""
        out = []
        for docid in self.DOCS.keys():
            if docid in self.DOCS.accessed:
                continue
            rec = dict.__getitem__(self.DOCS, docid)
            if not isinstance(rec, dict):
                continue
            text = rec.get("text") or rec.get("snippet") or ""
            out.append({"docid": docid, "n_chars": len(text), "head": text[:200].replace("\n", " ")})
        return out

    def search_many(self, queries: list, k=None) -> list:
        """Run several searches in one REPL turn; each query counts as one search."""
        if isinstance(queries, str):
            queries = [queries]
        out = []
        for q in queries:
            out.append(self.search(q, k))
        return out


    def server_docids(self) -> list[str] | None:
        """Every docid the retrieval server returned under this run's tag (search hits and
        fetched documents). Recorded by the server, so REPL code cannot add to it."""
        if not self.tag:
            return None
        try:
            return self._post("/tag_docids", {"tag": self.tag}).get("docids")
        except Exception:
            return None

    def server_counts(self) -> dict | None:
        """Counts recorded by the retrieval server for this run's tag, outside the agent process."""
        if not self.tag:
            return None
        try:
            return self._post("/tag_counts", {"tag": self.tag}).get("counts")
        except Exception:
            return None

    def descriptions(self) -> dict[str, str]:
        """Tool descriptions spliced into the RLM system prompt."""
        if self.design == "A":
            search_desc = (
                f"search(query: str) -> list[dict]. "
                f"Perform a search on a knowledge source. Returns top-{self.default_k} hits with docid, "
                f"score, and snippet. The snippet contains the document's contents (may be "
                f"truncated based on token limits). Results are also merged into the REPL "
                f"variable DOCS (docid -> record) and appended to SEARCH_LOG, so evidence "
                f"accumulates across calls."
            )
        elif not self.allow_k_override:
            search_desc = (
                f"search(query: str) -> list[dict]. "
                f"Perform a search on a knowledge source. Returns the top-{self.default_k} hits, each as "
                f"{{'docid', 'score', 'text', 'n_chars', 'embedded_span', 'snippet_span'}}. "
                f"'text' is the COMPLETE document, not a snippet, so a hit needs no follow-up "
                f"fetch. Documents are long (tens of thousands of characters) and {self.default_k} arrive "
                f"per call — keep them in REPL variables and delegate reading to llm_query / "
                f"llm_query_batched rather than printing them. llm_query_batched(list_of_prompts) "
                f"issues its prompts CONCURRENTLY and returns the answers in order, so reading n "
                f"documents costs about the same wall time as reading one; n separate llm_query "
                f"calls are issued one after another and cost n times as long. Whenever you have "
                f"more than one document to read or more than one question to ask, put them in a "
                f"single llm_query_batched call. Each REPL turn also costs one model call of its "
                f"own, so group independent work into one turn rather than spreading it over "
                f"several. 'embedded_span' is [start, end] "
                f"character offsets into 'text' marking the region the retrieval vector was "
                f"computed over: the index embedded only the first part of each document, so "
                f"matches are driven by that region. 'snippet_span' marks the first "
                f"{self.snippet_max_tokens} tokens. Results are merged into the REPL variable "
                f"DOCS and appended to SEARCH_LOG."
            )
        elif self.design == "D":
            search_desc = (
                f"search(query: str) -> list[dict]. Perform a hybrid (dense + keyword) search on a "
                f"knowledge source. Retrieves the top-{self.default_k} documents, then selects the "
                f"passages inside them that best match the query, and returns one record per "
                f"document, ordered by best passage: {{'docid', 'score', 'head', 'n_chars', "
                f"'passages', 'text'}}. 'passages' is a list of up to 3 query-relevant windows, "
                f"each {{'offset', 'end', 'score', 'text'}} — 'offset'/'end' are character positions "
                f"into the document's complete 'text', so text[offset:end] is exactly that "
                f"passage. About 15 passages come back per call. 'head' is the first 300 characters "
                f"(title/lede). 'text' is the COMPLETE document; documents are long (median 10K, "
                f"mean 34K characters), so read passages directly, and use Python string search "
                f"(`in`, .find, re) or llm_query over 'text' rather than printing it. Every "
                f"returned document is merged into the REPL variable DOCS (docid -> record) and the "
                f"call is appended to SEARCH_LOG."
            )
        else:
            search_desc = (
                f"search(query: str, k: int = {self.default_k}) -> list[dict]. "
                f"Perform a search on a knowledge source. Returns the top-k hits, each as "
                f"{{'docid', 'score', 'text', 'n_chars', 'embedded_span', 'snippet_span'}}. "
                f"'text' is the COMPLETE document, not a snippet, so a hit needs no follow-up "
                f"fetch. Documents are long (tens of thousands of characters) — keep them in "
                f"REPL variables and slice or delegate them to llm_query rather than printing "
                f"them. 'embedded_span' is [start, end] character offsets into 'text' marking "
                f"the region the retrieval vector was actually computed over: the index "
                f"embedded only the first part of each document, so matches are driven by that "
                f"region and the rest of the text was never seen by the retriever. "
                f"'snippet_span' marks the first {self.snippet_max_tokens} tokens. Results are "
                f"merged into the REPL variable DOCS and appended to SEARCH_LOG. k is yours to "
                f"choose per call, from 1 to {MAX_K}."
            )
            search_desc += (
                " The retriever is a dense semantic index (one embedding per document): it matches "
                "the MEANING of a full natural-language description, not keywords. Measured on this "
                "corpus, queries of 6-12 words recover about 10% of the relevant documents in their "
                "top-5; queries of 13-20 words about 15%; a complete question-length description "
                "about 15% with far higher chance of at least one hit. Write each query as a "
                "descriptive sentence or two that states the entity type and the constraints you "
                "are matching, rather than a short keyword string."
            )
            search_desc += (
                " Each call prints '[search n/cap] k=.. -> m new documents' into the REPL output; a "
                "query that adds fewer than two new documents has exhausted its facet, and another "
                "paraphrase of it returns the same documents. Retrieval matches the words you send: a "
                "query built around a name that appears in neither the question nor DOCS retrieves "
                "pages about that name, i.e. it verifies your own guess rather than searching the "
                "question; adding a type, role, outlet or place the question does not state (house, "
                "governor, actor, a newspaper) restricts results to that addition. Sandbox note: "
                "globals(), locals(), eval() and exec() are disabled in this REPL and return None; "
                "the tools are ordinary names (search, grep_docs, "
                "unread_docs, llm_query, llm_query_batched) and DOCS / SEARCH_LOG are ordinary "
                "variables."
            )
            search_desc += (
                " One call costs one call whatever k is: hits are ranked, so k=30 returns the same "
                "first five plus ranks 6-30. Measured on this corpus with question-length queries: "
                "the top-5 hold about 15% of the relevant documents (at least one of them 48% of the "
                "time), the top-30 about 33% (74%), the top-50 about 39% (80%). Hits at ranks you have "
                "not read are still in DOCS, where "
                "grep_docs (for a literal string from the question: a year, a number, a quoted "
                "phrase, a distinctive term) and unread_docs reach them without another call."
            )
        search_desc += (
            " read_docs(docs, question) -> list[str] reads several documents CONCURRENTLY and "
            "returns one answer per document, in order; pass search hits (or their 'text') and the "
            "question you want answered about each. Reading n documents this way costs about the "
            "same wall time as reading one, while n separate llm_query calls run one after another "
            "and cost n times as long. Whenever you have more than one document to read, use it."
        )
        if self.max_searches:
            search_desc += (
                f" You have a hard budget of {self.max_searches} search calls for this "
                f"question; calls beyond that are refused and return an error instead of "
                f"results, so plan the queries you spend them on."
            )
        return {
            "search": search_desc,
            "search_many": (
                "search_many(queries: list[str], k: int | None = None) -> list. Run several "
                "searches in ONE turn and get back a list of result lists, in the same order. "
                "Each query counts as one search against the same budget and returns exactly what "
                "a separate search() would, so this changes nothing about retrieval: it only saves "
                "you model turns. Every REPL turn costs one model call, so whenever you already "
                "know two or more queries you want to run, issue them together here rather than "
                "one per turn."
            ),
            "DOCS": (
                "dict mapping docid -> the record for every document retrieved so far in this "
                "task (search hits and fetched documents). Grows as you search; inspect with "
                "list(DOCS) or DOCS[docid]. Each record's 'text' is a plain Python string: "
                "keyword tests (`'x' in t`), t.find(), t.lower().count(), and re.search / "
                "re.finditer over it cost nothing and run over every document at once in a loop."
            ),
            "grep_docs": (
                "grep_docs(pattern: str, window: int = 400, max_hits: int = 40, flags=None) -> "
                "list[dict]. Regex-search the 'text' of every document in DOCS and return "
                "{'docid', 'offset', 'snippet'} for each match, snippet = the +/- window characters "
                "around it (case-insensitive by default). Pure Python over what you already "
                "retrieved: no search call, no LLM tokens. Use it to check a constraint (a date, a "
                "name, a number) across all retrieved pages at once before deciding what to search "
                "for next."
            ),
            "unread_docs": (
                "unread_docs() -> list[dict]. The retrieved documents this REPL session has never "
                "accessed (never indexed DOCS[docid] or iterated), as {'docid', 'n_chars', 'head'}. "
                "Every search adds documents; the ones you did not open are evidence you have not "
                "looked at."
            ),
            "SEARCH_LOG": (
                "list of {'query', 'k', 'docids'} for every search issued so far, in order. "
                "Use it to avoid repeating a query you have already run."
            ),
        }

    def custom_tools(self) -> dict:
        """Wrap the tools in plain closures for the REPL.

        The closures still reach this ToolBox through introspection, so reported counts come from the retrieval
        server's per-tag record and run_one scans trajectories for introspection tokens.
        """
        d = self.descriptions()
        _search, _grep, _unread = self.search, self.grep_docs, self.unread_docs
        _search_many = self.search_many

        def search(query, k=None):
            return _search(query, k)


        def search_many(queries, k=None):
            return _search_many(queries, k)

        def grep_docs(pattern, window=400, max_hits=40, flags=None):
            return _grep(pattern, window, max_hits, flags)


        def unread_docs():
            return _unread()

        search.__doc__ = d["search"]
        return {
            "search": {"tool": search, "description": d["search"]},
            "search_many": {"tool": search_many, "description": d["search_many"]},
            "grep_docs": {"tool": grep_docs, "description": d["grep_docs"]},
            "unread_docs": {"tool": unread_docs, "description": d["unread_docs"]},
            "DOCS": {"tool": self.DOCS, "description": d["DOCS"]},
            "SEARCH_LOG": {"tool": self.SEARCH_LOG, "description": d["SEARCH_LOG"]},
        }
