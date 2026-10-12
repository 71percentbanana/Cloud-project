"""Draw the paper's result figures from the simulator's result CSVs.

Usage: python paper/scripts/make_figures.py [results folder, default: results/]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")
os.makedirs(OUT, exist_ok=True)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # fixed categorical order
HATCH = ["", "///", "...", "xxx"]                         # secondary encoding for print/CVD
METHODS = ["Capability only", "Capability + Stability", "Independence replication", "Proposed"]
LABELS = ["Capability only", "Capability + stability", "Independence replication", "Proposed"]
CONDS = ["Stable", "Moderate", "High"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "hatch.linewidth": 0.5,
})


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=2)


def grid(df, col):
    return np.array([[df[(df.Condition == c) & (df.Method == m)][col].iloc[0] for m in METHODS] for c in CONDS])


# ---- Main comparison (Table: main results) --------------------------------
t1 = pd.read_csv(os.path.join(RESULTS, "table1_main_results.csv"))
panels = [
    (grid(t1, "WastedCPU_Mean") / 1000, grid(t1, "WastedCPU_CI95") / 1000, "Wasted CPU (10$^3$ device-s)", "(a) Wasted volunteer CPU"),
    (grid(t1, "Deadline_Mean"), grid(t1, "Deadline_CI95"), "Deadline completion (%)", "(b) Deadline completion"),
    (grid(t1, "Cloud_Mean") / 1000, grid(t1, "Cloud_CI95") / 1000, "Cloud (10$^3$ inst-s)", "(c) Cloud use"),
]
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
x = np.arange(len(CONDS)); w = 0.19
for ax, (val, ci, ylab, title) in zip(axes, panels):
    for m in range(4):
        ax.bar(x + (m - 1.5) * w, val[:, m], w * 0.9, yerr=ci[:, m], color=COLORS[m],
               hatch=HATCH[m], edgecolor="white", linewidth=0.4, label=LABELS[m],
               error_kw=dict(ecolor=INK, elinewidth=0.6, capsize=1.5))
    ax.set_xticks(x, CONDS)
    ax.set_xlabel("Instability")
    ax.set_ylabel(ylab); ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    style(ax)
lo = np.floor(min((panels[1][0] - panels[1][1]).min(), 100) - 0.5)
axes[1].set_ylim(lo, 100.3)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.05), fontsize=7.5)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(os.path.join(OUT, "main_results.pdf"), bbox_inches="tight")

# ---- Scaling with pool size -------------------------------------------------
t6 = pd.read_csv(os.path.join(RESULTS, "table6_scaling.csv"))
fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.7))
for ax, (col, scale, ylab, title) in zip(axes, [
    ("Deadline", 1, "Deadline completion (%)", "(a) Deadline"),
    ("Cloud", 1000, "Cloud (10$^3$ inst-s)", "(b) Cloud use"),
]):
    ax.errorbar(t6.Devices, t6[f"{col}_Mean"] / scale, yerr=t6[f"{col}_CI95"] / scale, color=COLORS[0],
                marker="o", markersize=3.5, linewidth=1.2, elinewidth=0.6, capsize=1.5)
    ax.set_xticks(t6.Devices); ax.set_xlabel("Volunteer devices")
    ax.set_ylabel(ylab); ax.set_title(title, fontsize=8, color=INK, loc="left")
    style(ax)
axes[1].set_ylim(0, None)
fig.tight_layout(w_pad=1.0)
fig.savefig(os.path.join(OUT, "scaling.pdf"), bbox_inches="tight")
# ---- Cloud vs wasted-CPU trade-off under high instability -------------------
t1b = pd.read_csv(os.path.join(RESULTS, "table1b_main_results_equal_cloud.csv"))
t1c = pd.read_csv(os.path.join(RESULTS, "table1c_proposed_tuned.csv"))
high = pd.concat([t1, t1b, t1c])
high = high[high.Condition == "High"].drop_duplicates("Method").set_index("Method")
points = [  # (method, label, group, label offset in points)
    ("Capability only", "Capability", 0, (5, -3)),
    ("Capability + Stability", "Cap.+stab.", 0, (5, 2)),
    ("Independence replication", "Replication", 0, (5, -3)),
    ("Capability only + cloud", "Capability", 1, (5, -8)),
    ("Capability + Stability + cloud", "Cap.+stab.", 1, (5, 2)),
    ("Independence replication + cloud", "Replication", 1, (5, -9)),
    ("Proposed", "Proposed", 2, (5, 2)),
    ("Proposed (R2=0.40, R3=0.70)", "Proposed, tuned", 2, (5, -9)),
]
groups = ["Baseline, volunteers only", "Baseline + cloud recovery", "Proposed"]
markers = ["o", "s", "D"]
fig, ax = plt.subplots(figsize=(3.5, 2.5))
for g in range(3):
    pts = [p for p in points if p[2] == g]
    ax.scatter([high.loc[p[0], "Cloud_Mean"] / 1000 for p in pts],
               [high.loc[p[0], "WastedCPU_Mean"] / 1000 for p in pts],
               s=28, marker=markers[g], color=COLORS[[0, 1, 3][g]], edgecolor="white", linewidth=0.8,
               label=groups[g], zorder=3)
for m, lab, g, off in points:
    r = high.loc[m]
    ax.annotate(f"{lab} ({r.Deadline_Mean:.1f}%)", (r.Cloud_Mean / 1000, r.WastedCPU_Mean / 1000),
                xytext=off, textcoords="offset points", fontsize=6.5, color=INK)
ax.set_xlabel("Cloud use (10$^3$ instance-s)")
ax.set_ylabel("Wasted CPU (10$^3$ device-s)")
ax.set_xlim(-40, 880); ax.set_ylim(0, 1200)
ax.legend(frameon=False, fontsize=6.5, loc="upper right")
style(ax)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "tradeoff.pdf"), bbox_inches="tight")
print("figures written to", os.path.abspath(OUT))
