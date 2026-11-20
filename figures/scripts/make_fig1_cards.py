#!/usr/bin/env python3
"""Fig. 1: one card per agent showing what a search returns, what enters the prompt and what is held in program state.

Run: python figures/scripts/make_fig1_cards.py -> paper/figures/fig1_architecture.{svg,pdf,png}.
"""
from pathlib import Path
import math
import sys
import cairosvg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_palette import AGENT, AGENT_TINT, INK, INK2, MUTED, CARD_LINE, PROMPT_FILL, PROMPT_LINE, STATE_LINE  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "paper" / "figures"
W, H = 516.0, 168.0
GAP, PAD = 9.0, 9.0
CW = (W - 2 * GAP) / 3
SANS, MONO = "General Sans, Liberation Sans, sans-serif", "Liberation Mono, monospace"
els: list[str] = []


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=6.6, fill=INK, weight=400, anchor="start", font=SANS, ls=None):
    sp = f' letter-spacing="{ls}"' if ls else ""
    els.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{font}" font-size="{size}" fill="{fill}" '
               f'font-weight="{weight}" text-anchor="{anchor}"{sp}>{esc(s)}</text>')


def rect(x, y, w, h, fill="none", stroke="none", sw=0.6, r=2.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    els.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def arrow(x1, y1, x2, y2, colour, sw=0.9, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    els.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{colour}" stroke-width="{sw}"{d} stroke-linecap="round"/>')
    a = math.atan2(y2 - y1, x2 - x1); L, Wd = 3.8, 2.1
    p1 = (x2 - L * math.cos(a) + Wd * math.sin(a), y2 - L * math.sin(a) - Wd * math.cos(a))
    p2 = (x2 - L * math.cos(a) - Wd * math.sin(a), y2 - L * math.sin(a) + Wd * math.cos(a))
    els.append(f'<polygon points="{x2:.1f},{y2:.1f} {p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}" fill="{colour}"/>')


def chip(x, y, c, w=4.6, h=6.0):
    rect(x, y, w, h, fill=c, r=1.1)


def page(x, y, c, w=5.0, h=6.6):
    f = 1.6
    els.append(f'<path d="M{x:.1f},{y:.1f} h{w - f:.1f} l{f:.1f},{f:.1f} v{h - f:.1f} h{-w:.1f} z" fill="white" stroke="{c}" stroke-width="0.6" stroke-linejoin="round"/>')
    for i in range(2):
        els.append(f'<line x1="{x + 1.0:.1f}" y1="{y + 3.0 + 1.6 * i:.1f}" x2="{x + w - 1.2:.1f}" y2="{y + 3.0 + 1.6 * i:.1f}" stroke="{c}" stroke-width="0.45"/>')


def label(x, y, s):
    text(x, y, s.upper(), size=4.9, fill=MUTED, weight=600, ls=0.35)


CARDS = [
    ("ref", "Reference agent", "the benchmark's search agent", "snips", "≈ 2,560 tokens", 2560),
    ("snip", "Snippet REPL agent", "prints snippets; a sub-model reads", "snips", "≈ 2,300 tokens", 2300),
    ("doc", "Document REPL agent", "documents as program state", "docs", "≈ 25-token line, + what code prints", 25),
]
for i, (key, name, sub, ret, cost, tok) in enumerate(CARDS):
    c, tint = AGENT[key], AGENT_TINT[key]
    x0 = i * (CW + GAP); xi, wi = x0 + PAD, CW - 2 * PAD
    rect(x0 + 0.4, 0.4, CW - 0.8, H - 0.8, fill="white", stroke=(c if key == "doc" else CARD_LINE), sw=0.8, r=7)
    # 1 header
    els.append(f'<circle cx="{xi + 3.2:.1f}" cy="{13.4:.1f}" r="3.2" fill="{c}"/>')
    text(xi + 10, 16.0, name, size=8.2, weight=600)
    text(xi, 26.5, sub, size=6.0, fill=INK2)
    # 2 one search returns
    label(xi, 39.5, "One search returns")
    if ret == "snips":
        for j in range(5):
            chip(xi + 6.3 * j, 43.5, c)
        text(xi + 36, 48.6, "5 snippets × 512 tokens", size=5.9, fill=INK2)
        src = (xi + 13, 51.0)
    else:
        for j in range(8):
            page(xi + 6.2 * j, 42.8, c)
        text(xi + 54, 48.4, "up to 50 whole documents", size=5.9, fill=INK2)
        src = (xi + 24, 51.0)
    # 3 prompt panel
    py, ph = 62.0, 30.0
    rect(xi, py, wi, ph, fill=PROMPT_FILL, stroke=PROMPT_LINE, sw=0.6, r=4)
    label(xi + 5, py + 8.2, "Prompt · read at every call")
    # 4 program state panel
    sy, sh = 98.0, 30.0
    rect(xi, sy, wi, sh, fill="white", stroke=STATE_LINE, sw=0.7, r=4, dash="2.4,1.8")
    label(xi + 5, sy + 8.2, "Program state · reached via code")
    if key == "ref":
        for row in range(2):
            for j in range(18 if row == 0 else 11):
                chip(xi + 5 + 7.2 * j, py + 12.5 + 8.4 * row, c, w=5.0, h=6.0)
        text(xi + 5, sy + 20.5, "nothing", size=6.0, fill=MUTED)
        arrow(*src, xi + 13, py - 0.8, c, sw=1.1)
    elif key == "snip":
        for row in range(2):
            for j in range(16 if row == 0 else 8):
                chip(xi + 5 + 7.2 * j, py + 12.5 + 8.4 * row, c, w=5.0, h=6.0)
        for j in range(2):
            page(xi + 5 + 7.0 * j, sy + 14.0, c, w=5.4, h=7.2)
        text(xi + 22, sy + 19.8, "fetched documents →", size=5.4, fill=MUTED)
        rect(xi + wi - 40, sy + 13.6, 34, 8.6, fill="white", stroke=c, sw=0.6, r=2.5)
        text(xi + wi - 23, sy + 19.6, "sub-model", size=5.4, fill=INK2, anchor="middle")
        arrow(xi + wi - 23, sy + 13.0, xi + wi - 23, py + ph + 0.8, c, sw=0.8, dash="1.6,1.3")
        text(xi + wi - 26, sy - 2.2, "answers printed", size=5.0, fill=MUTED, anchor="end")
        arrow(*src, xi + 13, py - 0.8, c, sw=1.1)
    else:
        for j in range(3):
            rect(xi + 5, py + 12.6 + 4.0 * j, 18, 2.4, fill=c, r=1.0)
        text(xi + 28, py + 16.6, "[search 3/10] k=25", size=5.2, fill=INK2, font=MONO)
        text(xi + 28, py + 23.4, "→ 22 new documents", size=5.2, fill=INK2, font=MONO)
        for row in range(2):
            for j in range(20):
                page(xi + 5 + 6.6 * j, sy + 12.0 + 8.4 * row, c, w=5.0, h=6.6)
        X, r, y1, y2 = x0 + 4.6, 2.6, 46.1, sy + 17.0     # down the card margin, rounded corners, into the panel
        els.append(f'<path d="M{xi - 1.0:.1f},{y1:.1f} H{X + r:.1f} Q{X:.1f},{y1:.1f} {X:.1f},{y1 + r:.1f} V{y2 - r:.1f} '
                   f'Q{X:.1f},{y2:.1f} {X + r:.1f},{y2:.1f} H{xi - 3.6:.1f}" fill="none" stroke="{c}" stroke-width="1.1" stroke-linecap="round"/>')
        arrow(xi - 4.0, y2, xi - 0.2, y2, c, sw=1.1)
        arrow(xi + wi - 18, sy - 0.6, xi + wi - 18, py + ph + 0.8, c, sw=0.8, dash="1.6,1.3")
        text(xi + wi - 21, sy - 2.2, "code prints excerpts", size=5.0, fill=MUTED, anchor="end")
    # 5 per-search cost
    label(xi, 140.0, "Added to the prompt per search")
    rect(xi, 145.0, wi, 7.0, fill=PROMPT_FILL, r=2)
    rect(xi, 145.0, max(1.6, wi * tok / 2560.0), 7.0, fill=c, r=2)
    text(xi, 161.5, cost, size=6.6, weight=600)

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}pt" height="{H}pt" viewBox="0 0 {W} {H}">'
       f'<rect width="{W}" height="{H}" fill="white"/>' + "".join(els) + "</svg>")
(OUT / "fig1_architecture.svg").write_text(svg, encoding="utf-8")
cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(OUT / "fig1_architecture.pdf"))
cairosvg.svg2png(bytestring=svg.encode(), write_to=str(OUT / "fig1_architecture.png"), scale=4)
print("wrote fig1_architecture.{svg,pdf,png}")
