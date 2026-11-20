#!/usr/bin/env python3
"""Fig. 2: prompt size per model call on question 1208 for three agents, with what each held outside the prompt.

Run: python figures/scripts/make_fig2_example.py -> paper/figures/fig2_example.{svg,pdf,png}. The per-turn values below are
taken from the run records (o200k tokens of the saved prompts; API counts for the reference agent), which are not part
of the release.
"""
from pathlib import Path
import cairosvg

OUT = Path(__file__).resolve().parents[2] / "paper" / "figures"
INK, INK2, INK3 = "#1f1f1f", "#555555", "#8a8a8a"
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_palette import AGENT_TINT, STATE_LINE as _SL, PROMPT_FILL as _PF, PROMPT_LINE as _PL, MUTED as _MU  # noqa: E402
STATE_FILL, STATE_LINE = "#ffffff", _SL
NONE_FILL, NONE_LINE = _PF, _PL
SANS = "General Sans, Liberation Sans, sans-serif"
from paper_palette import AGENT as _A  # noqa: E402
AGENT = {"ref": _A["ref"], "del": _A["snip"], "act": _A["doc"]}
TINT = {"ref": AGENT_TINT["ref"], "del": AGENT_TINT["snip"], "act": AGENT_TINT["doc"]}

REF_ROUNDS = [347, 7060, 10645, 13462, 16336, 19207, 22543, 26578, 30646, 33615, 36133, 38664, 41712, 44922, 48435,
              51511, 54650, 57264, 59892, 63008, 66081, 69481, 71645, 74527, 77078, 80413, 83671, 91267, 94136]

PANELS = [
 dict(key="ref", title="(a) Reference agent with document tool", sub="27 searches, one document fetched. Incorrect.",
      turns=[(str(i + 1), v, [], "") for i, v in enumerate(REF_ROUNDS)], state=None,
      summary=""),
 dict(key="del", title="(b) Snippet REPL agent", sub="35 searches, 175 snippets printed. Incorrect.",
      turns=[("1", 1173, "1 search, prints 5 snippets", "5"), ("2", 3830, "3 searches, prints 15", "20"),
             ("3", 11612, "3 searches, prints 15", "35"), ("4", 19498, "5 searches, prints 25", "60"),
             ("5", 32526, "6 searches, prints 30", "90"), ("6", 47222, "6 searches, prints 30", "120"),
             ("7", 61977, "5 searches, prints 25", "145"), ("8", 74753, "6 searches, prints 30", "175"),
             ("9", 89922, "answers from memory", "175")], state="snippets printed so far",
      summary=""),
 dict(key="act", title="(c) Document REPL agent", sub="6 searches, 103 documents held. Correct.",
      turns=[("1", 2487, "inspects 5 initial docs", "5"), ("2", 3481, "5 searches at k = 30", "103"),
             ("3", 4515, "grep ×6 over 103 docs", "103"), ("4", 6263, "lists unread docs", "103"),
             ("5", 6417, "grep candidate titles", "103"), ("6", 6937, "prints 17 doc heads", "103"),
             ("7", 10603, "reads 2 docs at a clue", "103"), ("8", 11687, "2 greps; answers", "103")],
      state="documents held", summary=""),
]
YMAX = 100000.0
PW, GUT = 164.0, 12.0
W = 3 * PW + 2 * GUT
TOP, LEFT, RIGHT, PANEL_H = 4.0, 27.0, 3.0, 100.0
H = TOP + PANEL_H
els = []
def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
def text(x, y, s, size=6.0, fill=INK, anchor="start", weight="normal", style="normal"):
    els.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{SANS}" font-size="{size}" font-weight="{weight}" font-style="{style}" fill="{fill}" text-anchor="{anchor}">{esc(s)}</text>')
def rect(x, y, w, h, fill, line, lw=0.6, r=1.2):
    els.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" ry="{r}" fill="{fill}" stroke="{line}" stroke-width="{lw}"/>')
def line(x1, y1, x2, y2, colour="#c8c8c8", lw=0.5):
    els.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{colour}" stroke-width="{lw}"/>')

for i, P in enumerate(PANELS):
    x0 = i * (PW + GUT); y0 = TOP
    els.append(f'<rect x="{x0:.1f}" y="{y0 + 1.0:.1f}" width="2.0" height="{PANEL_H - 4:.1f}" fill="{AGENT[P["key"]]}"/>')
    text(x0 + 6, y0 + 7.5, P["title"], size=7.6, weight="bold")
    text(x0 + 6, y0 + 16.0, P["sub"], size=6.6, fill=INK2)
    ax_top, ax_bot = y0 + 34.0, y0 + 66.0
    px0, px1 = x0 + LEFT, x0 + PW - RIGHT
    n = len(P["turns"]); slot = (px1 - px0) / n; bw = min(22.0, slot * 0.6)
    for v in (0, 25000, 50000, 75000, 100000):
        y = ax_bot - (ax_bot - ax_top) * v / YMAX
        line(px0, y, px1, y)
        text(px0 - 2.5, y + 2.0, f"{v//1000}k" if v else "0", size=5.8, fill=INK3, anchor="end")
    text(px0, ax_top - 7.0, "prompt tokens", size=5.9, fill=INK3, anchor="start")
    many = n > 12
    for j, (name, tok, action, state) in enumerate(P["turns"]):
        cx = px0 + slot * (j + 0.5); h = (ax_bot - ax_top) * tok / YMAX
        rect(cx - bw / 2, ax_bot - h, bw, h, TINT[P["key"]], AGENT[P["key"]], lw=0.5 if many else 0.6, r=0.8)
        if not many or (j + 1) % 7 == 1 or j == n - 1:
            text(cx, ax_bot + 7.5, name, size=6.2, fill=INK, anchor="middle", weight="bold")
        if P["state"]:
            rect(cx - slot * 0.44, ax_bot + 11.0, slot * 0.88, 9.5, STATE_FILL, STATE_LINE, lw=0.5, r=0.8)
            text(cx, ax_bot + 17.8, state, size=5.9, fill=INK, anchor="middle")
    line(px0, ax_bot, px1, ax_bot, colour="#9a9a9a", lw=0.6)
    text(px0 - 2.5, ax_bot + 7.5, "turn", size=5.4, fill=STATE_LINE, anchor="end")   # x axis: turn = one model call
    l1, l2 = ("snippets", "printed") if P["key"] == "del" else ("program", "state")
    text(px0 - 2.5, ax_bot + 14.5, l1, size=5.4, fill=STATE_LINE, anchor="end")
    text(px0 - 2.5, ax_bot + 20.3, l2, size=5.4, fill=STATE_LINE, anchor="end")
    if P["state"]:
        text(px1, ax_bot + 28.5, P["state"], size=5.6, fill=STATE_LINE, anchor="end", style="italic")
    else:
        rect(px0, ax_bot + 11.0, px1 - px0, 9.5, NONE_FILL, NONE_LINE, lw=0.5, r=0.8)
        text((px0 + px1) / 2, ax_bot + 17.8, "nothing is kept outside the prompt", size=6.0, fill=INK3, anchor="middle")
    pass

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}pt" height="{H}pt" viewBox="0 0 {W} {H}">'
       f'<rect width="{W}" height="{H}" fill="white"/>' + "".join(els) + "</svg>")
(OUT / "fig2_example.svg").write_text(svg)
cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(OUT / "fig2_example.pdf"))
cairosvg.svg2png(bytestring=svg.encode(), write_to=str(OUT / "fig2_example.png"), scale=220 / 72)
print(f"wrote fig2_example.(svg,pdf,png)  {W/72:.2f} x {H/72:.2f} in")
