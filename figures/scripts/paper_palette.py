"""Shared colours and font for the paper's figures: hue marks the agent; the prompt is a solid warm-grey panel and
program state a dashed outline. Font: General Sans (ITF Free Font License), loaded from .tools/fonts/general-sans.
"""
from pathlib import Path

AGENT = {"ref": "#1C6E9C", "snip": "#E2521A", "doc": "#8A38F5"}       # reference, snippet REPL, document REPL
AGENT_TINT = {"ref": "#DDE9F1", "snip": "#FBE3D9", "doc": "#ECE0FE"}   # light fills behind agent marks
INK, INK2, MUTED = "#1A1817", "#5E5E6A", "#8E8994"
GRID, AXIS, SURFACE, CARD_LINE = "#ECE9E6", "#CFCAC4", "#FFFFFF", "#E8E4E2"
PROMPT_FILL, PROMPT_LINE = "#F1F0EE", "#D9D4CE"                       # the prompt: solid warm grey
STATE_FILL, STATE_LINE = "#FFFFFF", "#8E8994"                         # program state: outlined, dashed
NEUTRAL = "#5E5E6A"                                                     # distractor test and other non-agent series
FAIL = {"nothing_found": "#2A2A30", "partly_found": "#8E8994", "found_misread": "#D9D4CE", "no_answer": "#FFFFFF"}
SANS, MONO = "General Sans", "Liberation Mono"
FONT_DIR = Path(__file__).resolve().parents[2] / ".tools" / "fonts" / "general-sans"


def use_brand_font():
    """Register General Sans with matplotlib (call before plotting)."""
    import glob
    from matplotlib import font_manager as fm
    for f in glob.glob(str(FONT_DIR / "*.ttf")):
        fm.fontManager.addfont(f)
