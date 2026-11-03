"""OBLIQ prompts without fixed similarity-score cutoffs, for the Gemini-Embedding-2 runs.

The cutoffs in the OBLIQ prompts were tuned for Qwen3-Embedding-8B; Gemini-Embedding-2 scores sit in a narrow high
band, so relative() replaces each cutoff sentence with advice to judge relevance from the snippets. Applied only when
OBLIQ_RELATIVE_SCORES=1.
"""
import os
import re

ENABLED = os.environ.get("OBLIQ_RELATIVE_SCORES") == "1"

_NOTE = "the retriever's scores sit in a narrow high band even for unrelated text, so no fixed cutoff applies"

REWRITES = [
    # dspy signatures, shared "weak results" paragraph (Math, Twitter, Congress)
    ("If scores are low (below ~0.30) and snippets are clearly off-topic, skip llm_query on\n"
     "those results — they are noise. Try a different query next turn.",
     "If the snippets are clearly off-topic, skip llm_query on those results — they are noise.\n"
     f"Try a different query next turn. Judge this from the snippets themselves: {_NOTE}."),
    # dspy Math
    ("   Scores are cosine similarities — if the top result is below 0.30, the\n"
     "   angle is wrong: abandon it immediately and try a different framing.\n"
     "   If 2 consecutive searches on the same angle return nothing above 0.30,\n"
     "   move on — do not keep rephrasing the same dead-end query.",
     "   Scores are cosine similarities on a narrow, high scale (even unrelated text\n"
     "   scores well), so do not use a fixed cutoff: decide yourself whether an angle\n"
     "   is working, from what the snippets say and how the scores compare with your\n"
     "   other searches. If an angle keeps returning nothing relevant, abandon it and\n"
     "   try a different framing — do not keep rephrasing the same dead-end query."),
    # dspy Congress
    ("   Scores are cosine similarities — if the top result is below 0.25, the angle\n"
     "   is wrong: try a completely different framing.\n"
     "   If 2 consecutive searches on the same angle return nothing above 0.25,\n"
     "   move on — do not keep rephrasing the same dead-end.",
     "   Scores are cosine similarities on a narrow, high scale (even unrelated text\n"
     "   scores well), so do not use a fixed cutoff: decide yourself whether an angle\n"
     "   is working, from what the snippets say and how the scores compare with your\n"
     "   other searches. If an angle keeps returning nothing relevant, try a completely\n"
     "   different framing — do not keep rephrasing the same dead-end."),
    # dspy Twitter
    ("If top score < 0.20, angle is wrong:\n   switch immediately.",
     "If the results show nothing relevant,\n"
     f"   the angle is wrong: switch immediately (judge from the snippets; {_NOTE})."),
    # snippet harness: Twitter task contract
    ("implied target. If the top score is below 0.20 the angle is wrong: switch immediately.",
     "implied target. If the results show nothing relevant the angle is wrong: switch immediately.\n"
     f"Judge that from the snippets and from how scores compare across your searches: {_NOTE}."),
    # snippet harness: OBLIQ STRATEGY block
    ("and if the top score is below 0.20 the angle is wrong, so switch rather than reword.",
     "and if an angle returns nothing relevant, switch rather than reword. Judge relevance from the "
     f"snippets and from how scores compare across your searches: {_NOTE}."),
]

_CUTOFF = re.compile(r"(below|above|<|>)\s*~?0\.\d\d")


def relative(text):
    """Apply every rewrite that matches; raise if a numeric cutoff is still left."""
    for old, new in REWRITES:
        text = text.replace(old, new)
    left = _CUTOFF.search(text)
    if left:
        raise ValueError(f"score cutoff left in prompt: ...{text[max(0, left.start() - 80):left.end() + 20]!r}")
    return text
