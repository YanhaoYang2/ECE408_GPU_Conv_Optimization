"""Generate the two result figures used in the README.

Data: analysis/results.csv (batch = 10000 rows only).
Output: docs/figures/results-batch10000.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "analysis" / "results.csv")
d = df[df["batch"] == 10000].copy()

SHORT = {
    "pm1": "PM1 baseline\n(unroll)",
    "pm2": "PM2 fused\n(baseline)",
    "req0": "Streams",
    "req1": "WMMA TF32",
    "op0": "Constant mem",
    "op1": "__restrict__",
    "op2": "Unroll k+=4",
    "op4": "cuBLAS",
    "op5": "FP16 __half2",
}
d["label"] = d["variant_id"].map(SHORT)

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=200)
fig.patch.set_facecolor("white")

# ---------------- panel A: per-operation time, log scale ----------------
ax = axes[0]
order = ["pm1", "pm2", "req0", "req1", "op0", "op1", "op2", "op4", "op5"]
d = d.set_index("variant_id").loc[order].reset_index()
y = range(len(d))
h = 0.38
b1 = ax.barh([i + h / 2 for i in y], d["op1_ms"], height=h,
             color="#2563eb", label="Conv layer 1")
b2 = ax.barh([i - h / 2 for i in y], d["op2_ms"], height=h,
             color="#f59e0b", label="Conv layer 2")
ax.set_yticks(list(y))
ax.set_yticklabels(d["label"], fontsize=8.5)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlim(0.25, 900)
ax.set_xlabel("Kernel time at batch 10,000 (ms, log scale)", fontsize=9)
ax.set_title("(a) Per-layer convolution time", fontsize=11, fontweight="bold", loc="left")
ax.grid(axis="x", which="major", color="#e5e7eb", linewidth=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for i, (a, b) in enumerate(zip(d["op1_ms"], d["op2_ms"])):
    ax.text(a * 1.12, i + h / 2, f"{a:g}", va="center", fontsize=7.2, color="#1e3a8a")
    ax.text(b * 1.12, i - h / 2, f"{b:g}", va="center", fontsize=7.2, color="#92400e")
ax.legend(loc="lower right", fontsize=8, frameon=False)

# ---------------- panel B: end-to-end total time, sorted ----------------
ax = axes[1]
s = d.sort_values("total_s")
colors = ["#16a34a" if v == s["total_s"].min() else
          ("#94a3b8" if vid == "pm1" else
           ("#3b82f6" if vid == "pm2" else "#93c5fd"))
          for v, vid in zip(s["total_s"], s["variant_id"])]
bars = ax.barh(s["label"], s["total_s"], color=colors, height=0.62)
ax.invert_yaxis()
ax.set_xlim(82, 101)
ax.set_xlabel("Total execution time at batch 10,000 (s, lower is better)", fontsize=9)
ax.set_title("(b) End-to-end run time", fontsize=11, fontweight="bold", loc="left")
ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for bar, v, acc in zip(bars, s["total_s"], s["accuracy"]):
    ax.text(v + 0.25, bar.get_y() + bar.get_height() / 2,
            f"{v:.1f} s   acc {acc:.4f}", va="center", fontsize=7.4, color="#111827")

fig.suptitle("LeNet-5 convolution on NVIDIA A40 — 9 implementation variants",
             fontsize=12.5, fontweight="bold", x=0.015, ha="left", y=0.985)
fig.text(0.015, 0.005,
         "Source: ECE 408 Milestone 3 report (2025). Bars use the numbers reported in the submission; "
         "total run time is dominated by host-side work and is not the sum of the two kernel times.",
         fontsize=6.8, color="#6b7280")
fig.tight_layout(rect=(0, 0.03, 1, 0.95))

out = ROOT / "docs" / "figures" / "results-batch10000.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, facecolor="white")
print("wrote", out)
