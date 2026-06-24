#!/usr/bin/env python3
"""
Vẽ biểu đồ training (100 epoch) từ run_summary_v6.csv để bỏ vào slide.
Xuất file PNG nền trắng, font to, vào thư mục demo/figures/.

Run:
    python3 demo/plot_training.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "run_summary_v6.csv")
OUT = os.path.join(ROOT, "demo", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 130,
})

# ---- load ----
rows = []
with open(CSV, newline="") as f:
    for r in csv.DictReader(f):
        rows.append(r)

def col(name, cast=float):
    return [cast(r[name]) for r in rows]

ep        = col("episode", int)
reward    = col("total_reward")
avg20     = col("avg20_reward")
energy    = col("energy_kwh")
slatah    = [v * 100 for v in col("slatah")]   # -> %
migr      = col("migrations")
fails     = col("failures")
tp        = col("a2_true_positives")
fp        = col("a2_false_positives")
fn        = col("a2_false_negatives")
r1        = col("r_underload")
r2        = col("r_overload")
r3        = col("r_selector")
r4        = col("r_placer")

BLUE, ORANGE, GREEN, RED, PURPLE = "#1f5fb4", "#e07b1a", "#2ca02c", "#d62728", "#7b3fa0"

# ============================================================
# Figure 1: Learning curve (reward + avg20) — slide chính
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ep, reward, color=BLUE, alpha=0.35, lw=1.2, label="Reward mỗi episode")
ax.plot(ep, avg20, color=BLUE, lw=2.6, label="Trung bình trượt 20 episode (Avg20)")
ax.set_xlabel("Episode")
ax.set_ylabel("Total reward")
ax.set_title("Learning curve — MARL CTDE-PPO (100 epoch)")
ax.legend(loc="lower right", framealpha=0.9)
ax.annotate(f"Avg20: {avg20[0]:.0f} → {avg20[-1]:.0f}",
            xy=(ep[-1], avg20[-1]), xytext=(ep[-1]-38, avg20[-1]-230),
            fontsize=12, color=BLUE,
            arrowprops=dict(arrowstyle="->", color=BLUE))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_learning_curve.png"), bbox_inches="tight")
plt.close(fig)

# ============================================================
# Figure 2: Bảng 4 panel — energy / SLATAH / migrations+fails / detector
# ============================================================
fig, axs = plt.subplots(2, 2, figsize=(13, 8))

# Energy
ax = axs[0, 0]
ax.plot(ep, energy, color=GREEN, lw=2)
ax.set_title(f"Năng lượng: {energy[0]:.1f} → {energy[-1]:.1f} kWh")
ax.set_xlabel("Episode"); ax.set_ylabel("Energy (kWh)")

# SLATAH
ax = axs[0, 1]
ax.plot(ep, slatah, color=ORANGE, lw=2)
ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.7)
ax.set_title("SLA violation (SLATAH %)")
ax.set_xlabel("Episode"); ax.set_ylabel("SLATAH (%)")

# Migrations + failures
ax = axs[1, 0]
ax.plot(ep, migr, color=PURPLE, lw=2, label="Migrations")
ax.set_xlabel("Episode"); ax.set_ylabel("Migrations", color=PURPLE)
ax.tick_params(axis="y", labelcolor=PURPLE)
ax2 = ax.twinx()
ax2.plot(ep, fails, color=RED, lw=1.6, label="Failures")
ax2.set_ylabel("Failures", color=RED)
ax2.tick_params(axis="y", labelcolor=RED)
ax2.set_ylim(0, max(5, max(fails) + 1))
ax2.grid(False)
ax.set_title("Migrations (CloudSim-confirmed) & Failures")

# Detector quality A2 (TP/FP/FN)
ax = axs[1, 1]
ax.plot(ep, tp, color=GREEN, lw=2, label="True Positive")
ax.plot(ep, fp, color=ORANGE, lw=1.6, label="False Positive")
ax.plot(ep, fn, color=RED, lw=1.6, label="False Negative")
ax.set_title("Agent 2 — chất lượng phát hiện overload")
ax.set_xlabel("Episode"); ax.set_ylabel("Số lượng / episode")
ax.legend(loc="upper right", fontsize=11, framealpha=0.9)

fig.suptitle("Metrics training qua 100 epoch", fontsize=16, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(OUT, "fig2_metrics_panel.png"), bbox_inches="tight")
plt.close(fig)

# ============================================================
# Figure 3: Reward phân rã theo 4 agent
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ep, r1, lw=1.8, label="Agent 1 — Underload detect")
ax.plot(ep, r2, lw=1.8, label="Agent 2 — Overload detect")
ax.plot(ep, r3, lw=1.8, label="Agent 3 — VM selector")
ax.plot(ep, r4, lw=1.8, label="Agent 4 — VM placer (GPU-aware)")
ax.axhline(0, color="gray", lw=0.8, alpha=0.6)
ax.set_xlabel("Episode"); ax.set_ylabel("Reward thành phần")
ax.set_title("Reward phân rã theo từng agent (CTDE)")
ax.legend(loc="center right", fontsize=11, framealpha=0.9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_per_agent_reward.png"), bbox_inches="tight")
plt.close(fig)

print("Saved figures to", OUT)
for f in sorted(os.listdir(OUT)):
    print("  -", f)
