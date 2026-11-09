"""One simulation run. Writes one JSON line per agent message to data/<run_id>.jsonl."""
import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from agents import LLM, is_infected, round_prompt, system_prompt  # noqa: E402
from network import build_graph, pick_protected  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"


def make_run_id(shape, n, rounds, pct, strategy, seed, p0, control, model, infection="plain", coop=False):
    tag = "control" if control else f"p{int(round(pct * 100)):02d}-{strategy}"
    inf = ("" if infection == "plain" else f"_{infection}") + ("_coop" if coop else "")
    return f"{model}{inf}__{shape}_n{n}_r{rounds}_{p0}__{tag}__s{seed}"


async def run_one(shape: str, n: int, rounds: int, pct: float, strategy: str, seed: int,
                  llm: LLM, p0: str = "leaf", control: bool = False, out_dir: Path = DATA,
                  overwrite: bool = False, infection: str = "plain", coop: bool = False) -> Path:
    run_id = make_run_id(shape, n, rounds, pct, strategy, seed, p0, control, llm.model, infection, coop)
    out = out_dir / f"{run_id}.jsonl"
    if out.exists() and not overwrite:
        print(f"skip (exists): {run_id}")
        return out

    g = build_graph(shape, n, seed, p0)
    protected = pick_protected(g, pct, strategy, seed)
    systems = {a: system_prompt(a, infected_seed=(a == 0 and not control), protected=a in protected, infection=infection,
                               cooperative=coop, trained=llm.model.endswith("-ft"))
               for a in g.nodes}
    history = {a: [] for a in g.nodes}  # per-agent append-only conversation
    last_msg: dict[int, str] = {}
    rngs = {a: random.Random(seed * 10_000 + a) for a in g.nodes}
    rows = []

    for rnd in range(rounds):
        async def step(a):
            inbox = [(b, last_msg[b]) for b in sorted(g.neighbors(a)) if b in last_msg]
            msgs = history[a] + [{"role": "user", "content": round_prompt(rnd, inbox)}]
            text, usage = await llm.reply(systems[a], msgs, rngs[a])
            return a, msgs, text, usage, [b for b, _ in inbox]

        results = await asyncio.gather(*(step(a) for a in g.nodes))
        new_last = {}
        for a, msgs, text, usage, senders in results:
            history[a] = msgs + [{"role": "assistant", "content": text or "(no message)"}]
            new_last[a] = text
            rows.append({
                "run_id": run_id, "model": llm.model, "shape": shape, "n": n, "p0": p0,
                "control": control, "infection": infection, "coop": coop, "pct_protected": pct, "strategy": strategy, "seed": seed,
                "round": rnd, "agent_id": a, "degree": g.degree(a), "protected": a in protected,
                "patient_zero": a == 0, "infected": is_infected(text),
                "inbox_infected": sum(is_infected(last_msg[b]) for b in senders),
                "inbox_size": len(senders), "text": text, **usage,
            })
        last_msg = new_last
        inf = sum(is_infected(t) for t in last_msg.values())
        print(f"  {run_id} round {rnd}: {inf}/{n} infected")

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.rename(out)  # only complete runs land in data/
    return out


def main():
    ap = argparse.ArgumentParser(description="Run one condition.")
    ap.add_argument("--shape", default="chain", choices=["chain", "star", "scale_free"])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--pct", type=float, default=0.0)
    ap.add_argument("--strategy", default="none", choices=["none", "random", "hubs"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p0", default="leaf", choices=["leaf", "hub"])
    ap.add_argument("--control", action="store_true", help="no infected agent")
    ap.add_argument("--model", default="muse-spark-1.3", help="model id, or 'mock' for offline test")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--infection", default="persuasive", choices=["plain", "persuasive"])
    ap.add_argument("--coop", action=argparse.BooleanOptionalAction, default=True, help="tell all agents to adopt team conventions (--no-coop to disable)")
    a = ap.parse_args()
    llm = LLM(a.model, effort=a.effort)
    path = asyncio.run(run_one(a.shape, a.n, a.rounds, a.pct, a.strategy, a.seed, llm,
                               a.p0, a.control, overwrite=a.overwrite, infection=a.infection, coop=a.coop))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
