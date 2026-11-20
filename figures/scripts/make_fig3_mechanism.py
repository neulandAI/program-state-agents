#!/usr/bin/env python
"""Fig. 3: (a) gold recall and accuracy at five and about 25 documents per search; (b) accuracy with all gold returned,
by final prompt size, against the distractor test.

Run: python figures/scripts/make_fig3_mechanism.py -> paper/figures/fig3_mechanism.{pdf,png}. Reads data/analysis/
(table1_main.csv, exact_prompt_tokens.csv, received_text.csv, ab6_by_length.csv) and data/x3/; the run records are
not part of the release.
"""
from __future__ import annotations
import csv, glob, json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper" / "figures"
STEM = "fig3_mechanism"
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_palette import AGENT, INK, INK2, MUTED, GRID, AXIS, NEUTRAL  # noqa: E402
C = {"std": AGENT["ref"], "snip": AGENT["snip"], "fd": AGENT["doc"]}
from paper_palette import use_brand_font  # noqa: E402
use_brand_font()
SANS = "General Sans"
FS, FS_T = 7.0, 7.8
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [SANS, "DejaVu Sans"], "font.size": FS, "text.color": INK,
    "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.facecolor": "white",
    "mathtext.fontset": "custom", "mathtext.rm": SANS, "mathtext.it": f"{SANS}:italic", "mathtext.bf": f"{SANS}:bold",
})

ARMS = [  # run, short label, family, label offset (points) and alignment for panel (a)
    ("x7_dense_gpt5", "reference", "std", (6, 0, "left")),
    ("x7_dense_gpt5_getdoc", "+ document tool", "std", (-6, 3, "right")),
    ("x7_dense_gpt5_trim30k", "trimmed 30k", "std", (6, 0, "left")),
    ("snip_dense_high150", "snippet REPL, no cap", "snip", (6, -4, "left")),
    ("snip_cap10", "snippet REPL, cap 10", "snip", (6, 4, "left")),
    ("fd_dense_high150", "document REPL, cap 10", "fd", (6, 0, "left")),
    ("fd_k5", "document REPL, k = 5", "fd", (6, -4, "left")),
    ("fd_cap20_150", "document REPL, cap 20", "fd", (-6, 0, "right")),
]
PROMPT_ROWS = [a[0] for a in ARMS]   # all eight arms now that the panel has room


PROMPT_C, STATE_C = "#c96b1f", "#3b6fb0"


WIDTH = [  # (narrow run, wide run, label, family): five documents per search -> about 25, by where the extra text goes
    ("x7_dense_gpt5_getdoc_cap10", "x7_dense_gpt5_getdoc_cap10_k25", "reference + doc. tool, cap 10: prompt", "std"),
    ("snip_dense_high150", "snip_k25", "snippet REPL: prompt", "snip"),
    ("fd_k5", "fd_dense_high150", "document REPL: program state", "fd"),
]


ARROWS = [  # (narrow run, wide run, replicate runs (wide, narrow), label, family)
    ("x7_dense_gpt5_getdoc_cap10", "x7_dense_gpt5_getdoc_cap10_k25", [], "reference + doc. tool", "std"),
    ("snip_dense_high150", "snip_k25", ["snip_k25_rep2"], "snippet REPL", "snip"),
    ("fd_k5", "fd_dense_high150", ["fd_rep2", "fd_k5_rep2"], "document REPL", "fd"),
]


def run_point(run):
    """(gold recall, accuracy %) of a run, from its records and gpt-4.1 grading."""
    g = json.load(open(ROOT / "data/x3" / run / "graded_gpt41.json"))
    rec = [json.load(open(f)).get("gold_recall") for f in glob.glob(str(ROOT / "data/x3" / run / "runs" / "*.json"))]
    rec = [r for r in rec if r is not None]
    return sum(rec) / len(rec), 100 * sum(bool(v["correct"]) for v in g.values()) / len(g)


MK = {"std": "o", "snip": "^", "fd": "D"}                       # one marker and line style per agent (greyscale)
LS = {"std": "-", "snip": (0, (4, 1.6)), "fd": (0, (1.2, 1.2))}


BRACKETS = [(0, 25e3), (25e3, 50e3), (50e3, 100e3), (100e3, 300e3), (300e3, 1e6), (1e6, 1e12)]
OBS = [  # arms drawn in panel (b): run, label, family, line style, marker (distinct in greyscale)
    ("x7_dense_gpt5", "reference agent", "std", "-", "o"),
    ("snip_dense_high150", "snippet REPL", "snip", (0, (4, 1.6)), "^"),
    ("fd_dense_high150", "document REPL", "fd", (0, (1.2, 1.2)), "D"),
]


ROWS = [  # dumbbell rows for (a): label, where the extra text goes, narrow run, wide run (None: single point), family
    ("Reference + doc. tool, cap 10", "25 snippets/search, added to prompt", "x7_dense_gpt5_getdoc_cap10", "x7_dense_gpt5_getdoc_cap10_k25", "std"),
    ("Snippet REPL agent", "25 snippets/search, printed to prompt", "snip_dense_high150", "snip_k25", "snip"),
    ("Document REPL agent", "~29 docs/search, in program state", "fd_k5", "fd_dense_high150", "fd"),
    ("Ablation: docs in the prompt", "same agent, 5/search, none held", "fd_k5prompt", None, "fd"),
]


def dumbbell(ax, idx, xlim, xticks, fmt, title, show_labels):
    """One measure (0 = gold recall, 1 = accuracy) for every row: hollow dot = 5 per search, filled dot = wide,
    arrow from narrow to wide. Readable without a legend: rows are labelled directly."""
    from matplotlib.patches import FancyArrowPatch
    ys = list(range(len(ROWS)))[::-1]
    for y, (label, where, narrow, wide, fam) in zip(ys, ROWS):
        v0 = run_point(narrow)[idx]
        if wide:
            v1 = run_point(wide)[idx]
            ax.add_patch(FancyArrowPatch((v0, y), (v1, y), arrowstyle="-|>", mutation_scale=8, lw=1.6, color=C[fam],
                                         shrinkA=3.5, shrinkB=3.5, zorder=3))
            ax.plot(v1, y, "o", ms=6.0, mfc=C[fam], mec="white", mew=0.9, zorder=4)
        ax.plot(v0, y, "s" if wide is None else "o", ms=5.6, mfc="white", mec=C[fam], mew=1.4, zorder=4)
        if show_labels:
            ax.text(-0.05, y + 0.14, label, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=FS - 0.6, color=INK)
            ax.text(-0.05, y - 0.24, where, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=FS - 1.5, color=INK2)
    ax.set_xlim(*xlim); ax.set_xticks(xticks); ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))
    ax.set_ylim(-0.6, len(ROWS) - 0.4); ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_title(title, fontsize=FS, color=INK, pad=3)


BINS = [("<25k", 0, 25e3), ("25–50k", 25e3, 50e3), ("50–100k", 50e3, 100e3), ("≥100k", 100e3, 1e12)]


def main():
    """Fig. 3 = the width chart only, one column wide dropped; the reading numbers stay in
    Table I and Section V-B)."""
    import matplotlib.ticker  # noqa: F401
    fig = plt.figure(figsize=(3.5, 1.95))
    ax_r = fig.add_axes([0.47, 0.13, 0.235, 0.68])
    ax_a = fig.add_axes([0.745, 0.13, 0.235, 0.68])
    dumbbell(ax_r, 0, (0.55, 0.93), [0.6, 0.7, 0.8, 0.9], lambda v, _: f"{v:.1f}", "Gold recall", True)
    dumbbell(ax_a, 1, (65, 88), [70, 80], lambda v, _: f"{v:.0f}%", "Accuracy", False)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", lw=0, ms=5.0, mfc="white", mec=INK2, mew=1.2),
                        Line2D([], [], marker="o", lw=0, ms=5.0, mfc=INK2, mec="white")],
               labels=["5 per search", "wide (as labelled)"], loc="upper left", bbox_to_anchor=(0.47, 0.995), ncol=2,
               frameon=False, fontsize=FS - 0.8, handletextpad=0.3, columnspacing=1.0, borderaxespad=0.0)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 220})):
        fig.savefig(OUT_DIR / f"{STEM}.{ext}", **kw); print("wrote", OUT_DIR / f"{STEM}.{ext}")
    for label, where, narrow, wide, fam in ROWS:
        r0 = run_point(narrow); r1 = run_point(wide) if wide else None
        print(f"  {label:34s} 5/search recall {r0[0]:.3f} acc {r0[1]:.1f}" + (f" | wide recall {r1[0]:.3f} acc {r1[1]:.1f}" if r1 else ""))


if __name__ == "__main__":
    main()
