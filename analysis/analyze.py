"""Attack rate, herd-immunity threshold, leaky rate, bootstrap CIs, figures.

  python analysis/analyze.py                  # all models in data/
  python analysis/analyze.py --model mock
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
THRESH = 0.10


def load(model=None, n=None) -> pd.DataFrame:
    rows = []
    for p in sorted((ROOT / "data").glob("*.jsonl")):
        with p.open() as f:
            rows += [json.loads(line) for line in f]
    df = pd.DataFrame(rows)
    if not df.empty:  # label = model + prompt variant, e.g. muse-spark-1.3_persuasive_coop
        df["model"] = df.run_id.str.split("__").str[0]
    if model and not df.empty:
        df = df[df.model == model]
    if n and not df.empty:
        df = df[df.n == n]
    return df


def per_run(df: pd.DataFrame) -> pd.DataFrame:
    """One row per run. 'Infected' = agent ever wrote 🍌 (excluding patient zero)."""
    ever = (df.groupby(["run_id", "agent_id"])
              .agg(infected=("infected", "max"), protected=("protected", "first"),
                   patient_zero=("patient_zero", "first"), degree=("degree", "first"))
              .reset_index())
    meta = df.groupby("run_id")[["model", "shape", "n", "p0", "control", "pct_protected",
                                 "strategy", "seed"]].first()
    others = ever[~ever.patient_zero]
    out = others.groupby("run_id").agg(attack_rate=("infected", "mean")).join(meta)
    prot = others[others.protected].groupby("run_id").agg(
        leaky_rate=("infected", "mean"), n_protected=("infected", "size"))
    unprot = others[~others.protected].groupby("run_id").agg(unprot_rate=("infected", "mean"))
    out = out.join(prot).join(unprot)

    # time to peak: round with most *new* infections
    first = df[df.infected & ~df.patient_zero].groupby(["run_id", "agent_id"])["round"].min()
    peak = first.reset_index().groupby("run_id")["round"].agg(lambda r: r.value_counts().idxmax())
    out["time_to_peak"] = peak
    # final-round prevalence (did it stick?)
    last = df[df["round"] == df.groupby("run_id")["round"].transform("max")]
    out["final_prevalence"] = last[~last.patient_zero].groupby("run_id")["infected"].mean()
    tok = df.groupby("run_id")[["input_tokens", "output_tokens"]].sum()
    return out.join(tok).reset_index()


def boot_ci(x, n_boot=10_000, seed=0):
    x = np.asarray([v for v in x if pd.notna(v)], float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n_boot, len(x)), replace=True).mean(1)
    return x.mean(), *np.percentile(means, [2.5, 97.5])


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for (model, shape, pct, strat), g in runs[~runs.control].groupby(
            ["model", "shape", "pct_protected", "strategy"]):
        r = dict(model=model, shape=shape, pct=pct, strategy=strat, n_seeds=len(g))
        for col in ("attack_rate", "leaky_rate", "unprot_rate", "final_prevalence"):
            m, lo, hi = boot_ci(g[col])
            r[col], r[col + "_lo"], r[col + "_hi"] = m, lo, hi
        recs.append(r)
    return pd.DataFrame(recs)


def thresholds(summ: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for (model, shape), g in summ.groupby(["model", "shape"]):
        base = g[g.pct == 0]
        for strat in ("random", "hubs"):
            rows = pd.concat([base, g[g.strategy == strat]]).sort_values("pct")
            hit = rows[rows.attack_rate < THRESH]
            recs.append(dict(model=model, shape=shape, strategy=strat,
                             threshold=f"{int(hit.pct.min() * 100)}%" if len(hit) else
                             f"> {int(rows.pct.max() * 100)}%"))
    return pd.DataFrame(recs).pivot(index=["model", "shape"], columns="strategy", values="threshold")


def plot_main(summ: pd.DataFrame, model: str, out: Path):
    s = summ[summ.model == model]
    shapes = [x for x in ("chain", "star", "scale_free") if x in set(s["shape"])]
    fig, axes = plt.subplots(1, len(shapes), figsize=(4 * len(shapes), 3.4), sharey=True,
                             squeeze=False)
    for ax, shape in zip(axes[0], shapes):
        g = s[s["shape"] == shape]
        base = g[g.pct == 0]
        for strat, color, mk in (("random", "#888888", "o"), ("hubs", "#1f5fbf", "s")):
            rows = pd.concat([base, g[g.strategy == strat]]).sort_values("pct")
            y = rows.attack_rate.values
            err = np.vstack([y - rows.attack_rate_lo.values, rows.attack_rate_hi.values - y])
            ax.errorbar(rows.pct * 100, y * 100, yerr=err * 100, label=strat, color=color,
                        marker=mk, capsize=3, lw=1.8)
        ax.axhline(THRESH * 100, ls=":", color="k", lw=1)
        ax.set_title(shape.replace("_", "-"))
        ax.set_xlabel("% agents protected")
        ax.set_xticks([0, 30, 60])
    axes[0][0].set_ylabel("attack rate (% non-seed agents ever infected)")
    axes[0][0].set_ylim(-3, 103)
    axes[0][-1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_leaky(summ: pd.DataFrame, model: str, out: Path):
    s = summ[(summ.model == model) & (summ.pct > 0)].copy()
    if s.empty:
        return
    s["label"] = s["shape"].str.replace("_", "-") + "\n" + (s.pct * 100).astype(int).astype(str) + "%"
    labels = list(dict.fromkeys(s.sort_values(["shape", "pct"]).label))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 2, 3.4))
    for i, (strat, color) in enumerate((("random", "#888888"), ("hubs", "#1f5fbf"))):
        g = s[s.strategy == strat].set_index("label").reindex(labels)
        y = g.leaky_rate.values
        err = np.vstack([y - g.leaky_rate_lo.values, g.leaky_rate_hi.values - y])
        ax.bar(x + (i - 0.5) * 0.38, y * 100, 0.38, yerr=err * 100, capsize=3, color=color, label=strat)
    ax.set_xticks(x, labels)
    ax.set_ylabel("% protected agents infected")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=20, help="swarm size to analyse (0 = all)")
    a = ap.parse_args()
    df = load(a.model, a.n)
    if df.empty:
        print("no data")
        return
    runs = per_run(df)
    figs = ROOT / "figures"
    figs.mkdir(exist_ok=True)
    runs.to_csv(figs / "runs.csv", index=False)

    ctrl = runs[runs.control]
    if len(ctrl):
        print("\n== No-infection controls (should be 0) ==")
        print(ctrl[["model", "shape", "attack_rate"]].to_string(index=False))

    summ = summarize(runs)
    summ.to_csv(figs / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n== Attack rate / leaky rate (mean [95% bootstrap CI over seeds]) ==")
    for _, r in summ.iterrows():
        leak = "" if pd.isna(r.leaky_rate) else \
            f"  leaky {r.leaky_rate:.0%} [{r.leaky_rate_lo:.0%},{r.leaky_rate_hi:.0%}]"
        print(f"{r.model:18s} {r['shape']:10s} {r.pct:.0%} {r.strategy:6s} n={r.n_seeds}  "
              f"attack {r.attack_rate:.0%} [{r.attack_rate_lo:.0%},{r.attack_rate_hi:.0%}]{leak}")

    th = thresholds(summ)
    th.to_csv(figs / "thresholds.csv")
    print(f"\n== Herd-immunity threshold (attack rate < {THRESH:.0%}) ==")
    print(th.to_string())

    print(f"\n== Tokens: {int(runs.input_tokens.sum()):,} in / {int(runs.output_tokens.sum()):,} out ==")
    for model in summ.model.unique():
        tag = model.replace("/", "_")
        plot_main(summ, model, figs / f"main_{tag}.png")
        plot_leaky(summ, model, figs / f"leaky_{tag}.png")
    print(f"\nfigures + CSVs in {figs}")


if __name__ == "__main__":
    main()
