"""Figure 4: per-exposure catch rate, unprotected vs protected, for each model (pooled, Wilson 95% CI)."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import load, ROOT

BLUE, GREY, INK, INK2, GRID = "#2a78d6", "#8a8985", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
# (label, [(run-label, condition), ...]); trained-vaccine runs only contribute their protected agents
MODELS = [("Muse Spark 1.3", [("muse-spark-1.3_persuasive_coop", "base")]),
          ("Qwen 3.5 4B", [("local-qwen35-4b_persuasive_coop", "base"), ("local-qwen35-4b-ft_persuasive_coop", "trained")]),
          ("Gemma 4 31B", [("adaption-gemma-4-31b-it_persuasive_coop", "base")])]


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


rows = []  # (model, condition, p, lo, hi, n); condition in unprotected / prompt / trained
for name, sources in MODELS:
    for key, kind in sources:
        df = load(key, 20)
        d = df[~df.control & ~df.patient_zero].sort_values(["run_id", "agent_id", "round"]).copy()
        d["prev"] = d.groupby(["run_id", "agent_id"])["infected"].transform(lambda s: s.shift(fill_value=False).cummax())
        ex = d[(d.inbox_infected > 0) & ~d.prev]
        conds = [("unprotected", ex[~ex.protected]), ("prompt", ex[ex.protected])] if kind == "base" else [("trained", ex[ex.protected])]
        for cond, g in conds:
            rows.append((name, cond, *wilson(int(g.infected.sum()), len(g)), len(g)))
            print(name, cond, f"{rows[-1][2]:.1%} [{rows[-1][3]:.1%},{rows[-1][4]:.1%}] n={len(g)}")

fig, ax = plt.subplots(figsize=(4.6, 2.9))
names = [m for m, _ in MODELS]; w = 0.27
for i, (cond, label, color, alpha) in enumerate((("unprotected", "Unprotected", GREY, 1.0),
                                                  ("prompt", "Prompt vaccine", BLUE, 1.0),
                                                  ("trained", "Trained vaccine (fine-tuned)", BLUE, 0.45))):
    r = [t for t in rows if t[1] == cond]
    if not r:
        continue
    xi = np.array([names.index(t[0]) for t in r])
    p = np.array([t[2] for t in r]) * 100; lo = np.array([t[3] for t in r]) * 100; hi = np.array([t[4] for t in r]) * 100
    xs = xi + (i - 1) * w
    ax.bar(xs, p, w * 0.92, color=color, alpha=alpha, label=label, zorder=2,
           hatch="////" if cond == "trained" else None, edgecolor="white" if cond == "trained" else None)
    ax.errorbar(xs, p, yerr=[p - lo, hi - p], fmt="none", ecolor=INK, elinewidth=1, capsize=0, zorder=3)
    for xx, pp, hh in zip(xs, p, hi):
        ax.text(xx, hh + 2.5, f"{pp:.0f}%" if pp >= 10 else f"{pp:.1f}%", ha="center", va="bottom", fontsize=8, color=INK)
ax.set_xticks(np.arange(len(names)), names); ax.set_ylim(0, 115); ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel("Adopted the marker after\nseeing it (% of exposures)")
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=3)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "fig4_models.png", dpi=300, facecolor="white")
