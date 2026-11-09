"""Theory comparison + per-exposure transmission + epidemic curves.
  python analysis/extra.py --model muse-spark-1.3_persuasive_coop
"""
import argparse, sys
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim"))
from analyze import load, per_run, boot_ci, ROOT  # noqa: E402
from network import build_graph, pick_protected  # noqa: E402


def theory_attack(shape, n, seed, p0, pct, strat, rounds):
    """Perfect immunity: protected agents removed; every exposed susceptible agent is infected next round.
    Reachable set = nodes within `rounds-1` hops of agent 0 avoiding protected nodes."""
    g = build_graph(shape, n, seed, p0)
    prot = pick_protected(g, pct, strat, seed)
    h = g.subgraph([v for v in g.nodes if v not in prot])
    d = nx.single_source_shortest_path_length(h, 0, cutoff=rounds - 1)
    return (len(d) - 1) / (n - 1), len(prot)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); a = ap.parse_args()
    df = load(a.model, 20)
    runs = per_run(df)
    runs = runs[~runs.control].copy()
    th = [theory_attack(r["shape"], r.n, r.seed, r.p0, r.pct_protected, r.strategy, 8) for _, r in runs.iterrows()]
    runs["theory_attack"] = [t for t, _ in th]

    print("== Observed vs perfect-immunity theory (mean over seeds) ==")
    g = runs.groupby(["shape", "pct_protected", "strategy"])[["attack_rate", "theory_attack"]].mean()
    print((g * 100).round(0).to_string())

    # per-exposure transmission: agent-rounds whose inbox had >=1 infected message, agent not yet infected
    d = df[~df.control & ~df.patient_zero].sort_values(["run_id", "agent_id", "round"]).copy()
    d["prev_inf"] = d.groupby(["run_id", "agent_id"])["infected"].transform(lambda s: s.shift(fill_value=False).cummax())
    ex = d[(d.inbox_infected > 0) & ~d.prev_inf]
    print("\n== Catch rate per exposure (P(write 🍌 | saw 🍌, not yet infected)) ==")
    for prot, grp in ex.groupby("protected"):
        per_run_rate = grp.groupby("run_id")["infected"].mean()
        m, lo, hi = boot_ci(per_run_rate)
        print(f"  protected={prot}: {grp.infected.mean():.1%} of {len(grp)} exposures "
              f"(per-run mean {m:.1%} [{lo:.1%},{hi:.1%}], {grp.run_id.nunique()} runs)")
    # pooled leaky over all protected agents, and among exposed protected agents
    ever = d.groupby(["run_id", "agent_id"]).agg(inf=("infected", "max"), prot=("protected", "first"),
                                                  exposed=("inbox_infected", lambda s: (s > 0).any()))
    p = ever[ever.prot]
    print(f"\n== Leaky: {p.inf.sum()}/{len(p)} protected agents ever infected ({p.inf.mean():.1%}); "
          f"among exposed: {p[p.exposed].inf.sum()}/{p.exposed.sum()} ({p[p.exposed].inf.mean():.1%})")
    u = ever[~ever.prot]
    print(f"   unprotected exposed infected: {u[u.exposed].inf.sum()}/{u.exposed.sum()} ({u[u.exposed].inf.mean():.1%})")

    # epidemic curves: prevalence per round, baseline condition
    cur = df[~df.control & ~df.patient_zero].groupby(["shape", "pct_protected", "strategy", "seed", "round"])["infected"].mean()
    cur = cur.groupby(["shape", "pct_protected", "strategy", "round"]).mean().unstack("round")
    print("\n== Prevalence by round (% non-seed agents whose message that round has 🍌) ==")
    print((cur * 100).round(0).to_string())
    out = ROOT / "figures"
    runs.to_csv(out / f"runs_theory_{a.model}.csv", index=False)
    cur.to_csv(out / f"curves_{a.model}.csv")

if __name__ == "__main__":
    main()
