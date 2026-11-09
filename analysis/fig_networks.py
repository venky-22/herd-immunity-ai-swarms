"""Figure 0: the three networks actually used (seed 0), patient zero, and the 30% protected sets."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))
from network import build_graph, pick_protected

BLUE, GREY, INK, INK2, RED = "#2a78d6", "#c9c8c3", "#0b0b0b", "#52514e", "#d6452a"
plt.rcParams.update({"font.family": "Arial", "font.size": 9})
shapes = [("scale_free", "Scale-free"), ("star", "Star"), ("chain", "Chain")]
strats = [("hubs", "30% hubs first"), ("random", "30% random")]
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2))
for j, (shape, title) in enumerate(shapes):
    g = build_graph(shape, 20, 0, "leaf")
    if shape == "chain":
        order = list(nx.dfs_preorder_nodes(g, source=next(v for v in g if g.degree(v) == 1)))
        pos = {v: ((i % 10), -(i // 10) * 1.2 if (i // 10) == 0 else -1.2) for i, v in enumerate(order)}
        pos = {v: ((i if i < 10 else 19 - i), 0 if i < 10 else -2.5) for i, v in enumerate(order)}
    elif shape == "star":
        pos = nx.shell_layout(g, nlist=[[v for v in g if g.degree(v) > 1], [v for v in g if g.degree(v) == 1]])
    else:
        pos = nx.kamada_kawai_layout(g)
    for i, (strat, label) in enumerate(strats):
        ax = axes[i][j]
        prot = pick_protected(g, 0.3, strat, 0)
        colors = [RED if v == 0 else BLUE if v in prot else GREY for v in g]
        sizes = [40 + 12 * g.degree(v) for v in g]
        nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#a9a8a3", width=0.8)
        nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colors, node_size=sizes, edgecolors="white", linewidths=0.8)
        ax.set_axis_off()
        if shape == "chain": ax.set_ylim(-9, 6.5); ax.set_xlim(-0.8, 9.8)
        if i == 0:
            ax.set_title(title, fontsize=10, fontweight="bold", loc="left", color=INK)
        if j == 0:
            ax.text(-0.08, 0.5, label, transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=9, color=INK2)
handles = [plt.Line2D([], [], ls="", marker="o", ms=7, mfc=RED, mec="white", label="Patient zero (injected)"),
           plt.Line2D([], [], ls="", marker="o", ms=7, mfc=BLUE, mec="white", label="Protected (vaccine prompt)"),
           plt.Line2D([], [], ls="", marker="o", ms=7, mfc=GREY, mec="white", label="Unprotected")]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
fig.tight_layout(rect=(0.02, 0.06, 1, 1))
fig.savefig(ROOT / "figures" / "fig0_networks.png", dpi=300, facecolor="white")
print("saved")
