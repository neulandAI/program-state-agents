"""Split the harness's patched system prompt into six named paragraphs and put an edited set back.

rebuild(split(p)) must reproduce p byte for byte; assert_roundtrip() checks this before any block is replaced.
Loaded by path from run_browsecomp.py, run_frames.py and run_obliq.py.
"""
from __future__ import annotations

SEP = "\n\n"
NAMES = ["ROLE", "REPL_TOOLS", "OUTPUT_CAP", "STRATEGY", "TURN_FORM", "TURN_RULE"]


def materialise() -> str:
    """The live system prompt, after the harness's patches."""
    import rlm.utils.prompts as PR
    return PR.RLM_SYSTEM_PROMPT


def split(prompt: str | None = None) -> dict[str, str]:
    p = materialise() if prompt is None else prompt
    paras = [x for x in p.split(SEP) if x.strip()]
    if len(paras) != len(NAMES):
        raise SystemExit(
            f"prompt no longer splits into {len(NAMES)} paragraphs (got {len(paras)}). "
            "An upstream reword or a new monkeypatch moved the boundaries: re-derive the "
            "split before trusting any optimisation built on it."
        )
    return dict(zip(NAMES, paras))


def rebuild(blocks: dict[str, str]) -> str:
    return SEP.join(blocks[n] for n in NAMES)


def assert_roundtrip() -> None:
    p = materialise()
    if rebuild(split(p)) != p:
        raise SystemExit("split/rebuild is NOT byte-identical -- the split is unsafe")


def apply(blocks: dict[str, str]) -> None:
    """Install a candidate back into the live prompt, in both places the harness reads it."""
    import rlm.utils.prompts as PR
    import rlm.core.rlm as core
    new = rebuild(blocks)
    PR.RLM_SYSTEM_PROMPT = new
    core.RLM_SYSTEM_PROMPT = new


if __name__ == "__main__":
    assert_roundtrip()
    b = split()
    print("roundtrip OK -- rebuild() is byte-identical to the materialised prompt\n")
    for n in NAMES:
        print(f"  {n:<12} {len(b[n]):5d} ch")
