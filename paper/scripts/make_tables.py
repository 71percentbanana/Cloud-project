"""Write the paper's result tables (LaTeX) straight from the simulator's result CSVs.

Usage: python paper/scripts/make_tables.py [results folder, default: results/]
"""
import os
import sys

import pandas as pd

RESULTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tables")
os.makedirs(OUT, exist_ok=True)
CONDS = ["Stable", "Moderate", "High"]
TUNED = "Proposed (R2=0.40, R3=0.70)"


def k(mean, ci):
    """Thousands with CI, e.g. 31.6 $\\pm$ 2.1."""
    return f"${mean / 1000:.1f} \\pm {ci / 1000:.1f}$"


def pct(mean, ci):
    return f"${mean:.2f} \\pm {ci:.2f}$"


def cloud(mean, ci):
    return "0" if mean == 0 else k(mean, ci)


def write(name, body):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(body)


def block(rows, labels):
    lines = []
    for ci, c in enumerate(CONDS):
        sub = [(labels[m], rows[(rows.Condition == c) & (rows.Method == m)].iloc[0]) for m in labels]
        for i, (lab, r) in enumerate(sub):
            first = f"\\multirow{{{len(sub)}}}{{*}}{{{c}}}" if i == 0 else ""
            lines.append(f" {first} & {lab} & {pct(r.Deadline_Mean, r.Deadline_CI95)} & "
                         f"{k(r.WastedCPU_Mean, r.WastedCPU_CI95)} & {cloud(r.Cloud_Mean, r.Cloud_CI95)} \\\\")
        if ci < len(CONDS) - 1:
            lines.append("\\midrule")
    return "\n".join(lines)


HEADER = ("\\begin{tabular}{@{}llrrr@{}}\n\\toprule\n"
          "Cond. & Method & Deadline (\\%) & Wasted CPU & Cloud \\\\\n\\midrule\n")
FOOTER = "\n\\bottomrule\n\\end{tabular}\n"

t1 = pd.read_csv(os.path.join(RESULTS, "table1_main_results.csv"))
t1b = pd.read_csv(os.path.join(RESULTS, "table1b_main_results_equal_cloud.csv"))
t1c = pd.read_csv(os.path.join(RESULTS, "table1c_proposed_tuned.csv"))
main = pd.concat([t1, t1c])
write("main.tex", HEADER + block(main, {
    "Capability only": "Capability only",
    "Capability + Stability": "Cap.\\ + stability",
    "Independence replication": "Indep.\\ replication",
    "Proposed": "Proposed",
    TUNED: "Proposed, tuned",
}) + FOOTER)
write("equal_cloud.tex", HEADER + block(t1b, {
    "Capability only + cloud": "Capability only",
    "Capability + Stability + cloud": "Cap.\\ + stability",
    "Independence replication + cloud": "Indep.\\ replication",
}) + FOOTER)

t2 = pd.read_csv(os.path.join(RESULTS, "table2_ablation.csv"))
names = {"Full lifecycle": "+ Cloud fallback (full)"}
lines = [f"{names.get(r.Configuration, r.Configuration)} & {pct(r.Deadline_Mean, r.Deadline_CI95)} & "
         f"{k(r.WastedCPU_Mean, r.WastedCPU_CI95)} & {cloud(r.Cloud_Mean, r.Cloud_CI95)} \\\\" for r in t2.itertuples()]
write("ablation.tex", "\\begin{tabular}{@{}lrrr@{}}\n\\toprule\nConfiguration & Deadline (\\%) & Wasted CPU & Cloud \\\\\n\\midrule\n"
      + "\n".join(lines) + FOOTER)

t3 = pd.read_csv(os.path.join(RESULTS, "table3_coavailability.csv"))
lines = [f"{'Individual suitability' if r.Method.startswith('Independence') else 'Co-availability'} & {k(r.WastedCPU_Mean, r.WastedCPU_CI95)} & "
         f"{r.Reliability_Mean:.2f} & {r.Replica_Hours:.2f} \\\\" for r in t3.itertuples()]
write("coavail.tex", "\\begin{tabular}{@{}lrrr@{}}\n\\toprule\nHedge choice & Wasted CPU & Deadline (\\%) & Replica-h/task \\\\\n\\midrule\n"
      + "\n".join(lines) + FOOTER)

t4 = pd.read_csv(os.path.join(RESULTS, "table4_correlated_placement.csv"))
lines = [f"{'Independence' if r.Method.startswith('Independence') else 'Co-availability'} & {r._2:.0f} & {r._3:.2f} & {r._4:.1f} \\\\"
         for r in t4.itertuples()]
write("placement.tex", "\\begin{tabular}{@{}lrrr@{}}\n\\toprule\nPlacement & Replicas & Joint int.\\ (\\%) & Runs $>$ 2\\% \\\\\n\\midrule\n"
      + "\n".join(lines) + FOOTER)

t5 = pd.read_csv(os.path.join(RESULTS, "table5_threshold_sensitivity.csv"))
lines = [f"{r.R2:.2f} & {r.R3:.2f} & {pct(r.Deadline_Mean, r.Deadline_CI95)} & {k(r.WastedCPU_Mean, r.WastedCPU_CI95)} & "
         f"{k(r.Cloud_Mean, r.Cloud_CI95)} & {r.Replicas_Mean:.2f} \\\\" for r in t5.itertuples()]
write("thresholds.tex", "\\begin{tabular}{@{}ccrrrr@{}}\n\\toprule\n$R_2$ & $R_3$ & Deadline (\\%) & Wasted CPU & Cloud & Repl. \\\\\n\\midrule\n"
      + "\n".join(lines) + FOOTER)

t6 = pd.read_csv(os.path.join(RESULTS, "table6_scaling.csv"))
lines = [f"{r.Devices} & {pct(r.Deadline_Mean, r.Deadline_CI95)} & {k(r.WastedCPU_Mean, r.WastedCPU_CI95)} & "
         f"{k(r.Cloud_Mean, r.Cloud_CI95)} & {r.Replicas_Mean:.2f} \\\\" for r in t6.itertuples()]
write("scaling.tex", "\\begin{tabular}{@{}rrrrr@{}}\n\\toprule\nDevices & Deadline (\\%) & Wasted CPU & Cloud & Repl. \\\\\n\\midrule\n"
      + "\n".join(lines) + FOOTER)
print("tables written to", os.path.abspath(OUT))
