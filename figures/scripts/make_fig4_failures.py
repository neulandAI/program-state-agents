#!/usr/bin/env python
"""Fig. 4: failure modes on the full sets, one panel per benchmark and retriever; each bar splits the wrong answers
(snippet and document REPL agents; Fetch-then-Explore with dense retrieval).

Run: python figures/scripts/make_fig4_failures.py -> paper/figures/fig4_failures.{pdf,png}. Reads
data/analysis/failure_modes_full.csv (analysis/failure_modes_full.py)."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"
INK, INK2, MUTED, GRID, AXIS = "#1A1817", "#5E5E6A", "#8E8994", "#ECE9E6", "#CFCAC4"
import sys as _s2
_s2.path.insert(0, str(Path(__file__).resolve().parent))
from paper_palette import use_brand_font, INK as _INK  # noqa: E402
use_brand_font()
SANS = "General Sans"; FS, FS_T = 7.0, 7.6
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [SANS, "DejaVu Sans"], "font.size": FS, "text.color": INK,
    "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.0, "ytick.major.size": 0, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "pdf.fonttype": 42, "savefig.facecolor": "white"})
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_palette import FAIL  # noqa: E402
SEG = [("nothing_found", "nothing found", FAIL["nothing_found"]), ("partly_found", "partly found", FAIL["partly_found"]),
       ("found_misread", "all gold returned, still wrong", FAIL["found_misread"]), ("no_answer", "no answer", FAIL["no_answer"])]
SHORT = {"delegated, no cap": "snippet", "delegated, cap 25": "snippet", "active, cap 10": "document", "active, cap 20": "document",
         "active, Qwen3.6": "doc., Qwen3.6", "fetch-then-explore, no cap": "Fetch-then-\nExplore", "delegated, Qwen3.6": "snip., Qwen3.6"}   # snippet and document REPL agents   # the snippet and document REPL agents

rows = list(csv.DictReader(open(ROOT / "data/analysis/failure_modes_full.csv")))
groups = []
for r in rows:
    if r["group"] not in groups: groups.append(r["group"])
TITLES = {"BrowseComp-Plus, dense": "BrowseComp-Plus\ndense retrieval", "BrowseComp-Plus, late interaction": "BrowseComp-Plus\nlate interaction",
          "BrowseComp-Plus, BM25": "BrowseComp-Plus\nBM25", "FRAMES, BM25": "FRAMES\nBM25"}
fig = plt.figure(figsize=(7.16, 1.43))
PW, GAP, X0, Y0, PH = 0.13, 0.115, 0.085, 0.277, 0.497     # four panels in a row; the gap holds the next panel's labels
for i, g in enumerate(groups):
    ax = fig.add_axes([X0 + i * (PW + GAP), Y0, PW, PH])
    sub = [r for r in rows if r["group"] == g]
    # rows spread over the full panel height whatever their number (3 rows: 3, 2, 1; 2 rows: 3, 1), same bar thickness
    ys = [3 - 2 * k / (len(sub) - 1) for k in range(len(sub))] if len(sub) > 1 else [2]
    for y, r in zip(ys, sub):
        n = int(r["n"]); left = 0.0
        for key, label, colour in SEG:
            v = 100 * int(r[key]) / n
            edge = "#8f8d86" if key == "no_answer" else "white"
            ax.barh(y, v, left=left, height=0.6, color=colour, edgecolor=edge, linewidth=0.6, zorder=3)
            left += v
        if left > 70:   # long bar: total in white inside the dark segment so it cannot run into the next panel
            ax.text(100 * int(r["nothing_found"]) / n - 1.5, y, f"{left:.0f}%", ha="right", va="center", fontsize=FS - 0.6, color="white")
        else:
            ax.text(left + 1.2, y, f"{left:.0f}%", ha="left", va="center", fontsize=FS - 0.6, color=INK2)
    ax.set_yticks(ys); ax.set_yticklabels([SHORT.get(r["label"], r["label"]) for r in sub])
    ax.set_ylim(0.45, 3.55)   # room for three rows (dense: snippet, document, Fetch-then-Explore)
    ax.set_xlim(0, 45); ax.set_xticks([0, 10, 20, 30, 40]); ax.set_xticklabels(["0", "10", "20", "30", "40%"])
    ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_title(TITLES[g], fontsize=FS_T, fontweight="bold", loc="left", pad=3, linespacing=1.1)
fig.legend(handles=[Patch(facecolor=c, edgecolor="#8f8d86" if k == "no_answer" else c, linewidth=0.6, label=l) for k, l, c in SEG], loc="lower left", bbox_to_anchor=(X0 - 0.085, 0.0), ncol=4,
           frameon=False, fontsize=FS - 0.4, handlelength=1.1, handleheight=0.8, columnspacing=1.6, handletextpad=0.45)
fig.text(0.985, 0.03, "wrong answers as a share of the dataset's questions", fontsize=FS - 0.6, color=INK2, ha="right", va="bottom")
for ext, kw in (("pdf", {}), ("png", {"dpi": 200})):
    fig.savefig(OUT / f"fig4_failures.{ext}", **kw)
print("wrote fig4_failures.pdf/png")
