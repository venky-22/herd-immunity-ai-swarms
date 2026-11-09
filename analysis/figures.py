"""Paper figures (PNG, 300 dpi, white background for the PDF template).
  python analysis/figures.py --model muse-spark-1.3_persuasive_coop
Fig 1: attack rate vs % protected, one panel per shape; hubs-first (blue) vs random (grey),
       observed (solid, 95% bootstrap CI) and perfect-immunity theory (dashed, open markers).
Fig 2: spread curves on the scale-free network: share of agents writing 🍌 each round.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import ROOT, boot_ci  # noqa: E402

BLUE, GREY, INK, INK2, GRID = "#2a78d6", "#8a8985", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})
STRATS = (("hubs", "Hubs first", BLUE), ("random", "Random", GREY))
TITLES = {"chain": "Chain", "star": "Star", "scale_free": "Scale-free"}


def fig_main(model, out):
    runs = pd.read_csv(ROOT / "figures" / f"runs_theory_{model}.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, shape in zip(axes, ("scale_free", "star", "chain")):
        r = runs[runs["shape"] == shape]
        base = r[r.pct_protected == 0]
        for strat, label, color in STRATS:
            rows = pd.concat([base, r[r.strategy == strat]])
            pts = []
            for pct, g in rows.groupby("pct_protected"):
                m, lo, hi = boot_ci(g.attack_rate)
                pts.append((pct * 100, m * 100, lo * 100, hi * 100, g.theory_attack.mean() * 100))
            p = pd.DataFrame(pts, columns=["x", "m", "lo", "hi", "th"])
            ax.plot(p.x, p.th, ls=(0, (3, 2)), lw=1.2, color=color, alpha=0.8, zorder=2)
            ax.scatter(p.x, p.th, s=22, facecolor="white", edgecolor=color, lw=1.2, zorder=3)
            ax.errorbar(p.x, p.m, yerr=[p.m - p.lo, p.hi - p.m], color=color, lw=2, marker="o",
                        ms=5, capsize=0, elinewidth=1.2, zorder=4, label=label,
                        markeredgecolor="white", markeredgewidth=1)
        ax.axhline(10, color=INK2, lw=0.8, ls=":", zorder=1)
        ax.set_title(TITLES[shape], fontsize=10, color=INK, loc="left", fontweight="bold")
        ax.set_xticks([0, 30, 60]); ax.set_xticklabels(["0%", "30%", "60%"])
        ax.set_xlim(-6, 66); ax.set_ylim(-4, 104)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.set_xlabel("Agents protected")
    axes[0].set_ylabel("Attack rate (% of agents infected)")
    handles = [plt.Line2D([], [], color=BLUE, lw=2, marker="o", ms=5, label="Hubs first (observed)"),
               plt.Line2D([], [], color=GREY, lw=2, marker="o", ms=5, label="Random (observed)"),
               plt.Line2D([], [], color=INK2, lw=1.2, ls=(0, (3, 2)), marker="o", ms=4,
                          markerfacecolor="white", label="Perfect-immunity theory")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)


def fig_curves(model, out):
    cur = pd.read_csv(ROOT / "figures" / f"curves_{model}.csv")
    cur = cur[cur["shape"] == "scale_free"]
    rounds = [c for c in cur.columns if c.isdigit()]
    series = [((0.0, "none"), "No protection", INK2, "-"),
              ((0.3, "random"), "30% random", GREY, "-"),
              ((0.3, "hubs"), "30% hubs first", BLUE, "-")]
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    for (pct, strat), label, color, ls in series:
        row = cur[(cur.pct_protected == pct) & (cur.strategy == strat)]
        y = row[rounds].values[0] * 100
        ax.plot([int(r) for r in rounds], y, color=color, lw=2, ls=ls, marker="o", ms=4,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.text(7.2, y[-1], label, va="center", fontsize=8, color=INK)
    ax.set_xlim(-0.3, 10.2); ax.set_ylim(-4, 104)
    ax.set_xticks(range(8)); ax.set_xlabel("Round")
    ax.set_ylabel("Agents writing \N{BANANA} (%)") if False else ax.set_ylabel("Agents carrying the marker (%)")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.spines["bottom"].set_bounds(0, 7)
    fig.tight_layout()
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); a = ap.parse_args()
    fig_main(a.model, ROOT / "figures" / "fig1_attack_rate.png")
    fig_curves(a.model, ROOT / "figures" / "fig2_spread_curves.png")
    print("saved figures/fig1_attack_rate.png, figures/fig2_spread_curves.png")
