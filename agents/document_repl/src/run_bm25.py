"""Run run_one.py with the retriever description rewritten for a BM25 index (FRAMES, or BrowseComp-Plus with
BM25_CORPUS=bcp).

    cd $FULLDOC_ROOT && python /path/to/agents/document_repl/src/run_bm25.py --query-id 0 ... (run_one.py args)

Only the sentences that describe a dense index change; each replacement is asserted. run_one.py is then run as
__main__ with the same argv.
"""
import os, runpy, sys

SUP_SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SUP_SRC)

# The same sentence the snippet REPL agent gets on FRAMES; BM25_CORPUS=bcp changes only the corpus noun.
BCP = os.environ.get("BM25_CORPUS", "frames") == "bcp"
_CORPUS_DESC = ("a fixed corpus of about 100,000 web pages (one page per document, the whole page "
                "indexed)") if BCP else "English Wikipedia (one article per document, the whole article indexed)"
BM25_FACT = (
    f"The retriever is BM25 keyword search over {_CORPUS_DESC}: it scores documents by the words of "
    "your query, after stemming and stopword removal, not by their meaning, and quotation marks are "
    "ignored (a quoted phrase is not matched verbatim)."
)
# FRAMES_TITLE_SEARCH=1 adds one clause; unset, the text is unchanged.
TITLE_CLAUSE = (
    " To get the article about a named entity (a person, place, work, event or organisation), search its "
    "exact Wikipedia article title on its own, with no other words: with extra words in the query, BM25 "
    "often ranks shorter pages that mention the entity above its own long article."
)
if os.environ.get("FRAMES_TITLE_SEARCH") == "1" and not BCP:
    BM25_FACT += TITLE_CLAUSE
    print("[title-search] title-only search clause added to the search description", flush=True)

SEARCH_REPLACEMENTS = [
    ("Documents are long (tens of thousands of characters)",
     "Documents are " + ("web pages" if BCP else "Wikipedia articles") + ", often long (tens of thousands of characters)"),
    ("'embedded_span' is [start, end] character offsets into 'text' marking the region the "
     "retrieval vector was actually computed over: the index embedded only the first part of "
     "each document, so matches are driven by that region and the rest of the text was never "
     "seen by the retriever. ",
     "'embedded_span' is [start, end] character offsets into 'text' marking the region the "
     "retriever indexed, which is the whole document. "),
    ("The retriever is a dense semantic index (one embedding per document): it matches "
     "the MEANING of a full natural-language description, not keywords. Measured on this "
     "corpus, queries of 6-12 words recover about 10% of the relevant documents in their "
     "top-5; queries of 13-20 words about 15%; a complete question-length description "
     "about 15% with far higher chance of at least one hit. Write each query as a "
     "descriptive sentence or two that states the entity type and the constraints you "
     "are matching, rather than a short keyword string.",
     BM25_FACT),
    ("Measured on this corpus with question-length queries: "
     "the top-5 hold about 15% of the relevant documents (at least one of them 48% of the "
     "time), the top-30 about 33% (74%), the top-50 about 39% (80%). ",
     ""),
]

import prompts  # noqa: E402  (same module object run_one.py imports)
import tools  # noqa: E402

PROLOGUE_OLD = ("Retrieval is semantic, driven by the wording of each query, so a "
                "near-identical query returns near-identical documents.")
PROLOGUE_NEW = ("Retrieval is BM25 keyword search, driven by the words of each query, so a query "
                "with the same key terms returns near-identical documents.")
# Under GEPA_PROMPTS the prologue may already carry PROLOGUE_NEW; patch only when the dense sentence is present.
if os.environ.get("GEPA_PROMPTS"):
    if PROLOGUE_OLD in prompts.USER_PROLOGUE:
        prompts.USER_PROLOGUE = prompts.USER_PROLOGUE.replace(PROLOGUE_OLD, PROLOGUE_NEW)
else:
    assert PROLOGUE_OLD in prompts.USER_PROLOGUE, "USER_PROLOGUE anchor moved"
    prompts.USER_PROLOGUE = prompts.USER_PROLOGUE.replace(PROLOGUE_OLD, PROLOGUE_NEW)
    assert "Retrieval is semantic" not in prompts.USER_PROLOGUE, "USER_PROLOGUE not patched"
_old = "long web pages — tens of thousands of characters each —"
assert _old in prompts.SOLO_READ_SENTENCE, "SOLO_READ_SENTENCE anchor moved"
if not BCP:   # on BrowseComp-Plus "long web pages" is already true
    prompts.SOLO_READ_SENTENCE = prompts.SOLO_READ_SENTENCE.replace(
        _old, "Wikipedia articles, often tens of thousands of characters long —")

_orig_desc = tools.ToolBox.descriptions


def _bm25_descriptions(self):
    d = _orig_desc(self)
    s = d["search"]
    for old, new in SEARCH_REPLACEMENTS:
        assert old in s, f"search description anchor moved: {old[:60]!r}"
        s = s.replace(old, new, 1)
    assert "dense semantic" not in s and "Measured on this corpus" not in s
    d["search"] = s
    return d


tools.ToolBox.descriptions = _bm25_descriptions
print("[bm25-prompt] retriever description patched for BM25", flush=True)

# Serialise non-JSON values left in a trajectory with repr() instead of failing the final save.
import json as _json  # noqa: E402

_orig_dump = _json.dump


def _dump(obj, fp, *a, **kw):
    kw.setdefault("default", repr)
    return _orig_dump(obj, fp, *a, **kw)


_json.dump = _dump

sys.argv[0] = f"{SUP_SRC}/run_one.py"
runpy.run_path(f"{SUP_SRC}/run_one.py", run_name="__main__")
