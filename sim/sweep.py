"""Run all conditions. Completed runs are skipped, so it's safe to re-launch after a crash.

  python sim/sweep.py --model mock                # offline smoke test
  python sim/sweep.py --dry-run                   # list conditions + call count
  python sim/sweep.py --baseline-only             # 0% on all shapes + no-infection controls
  python sim/sweep.py                             # full 45-run sweep + controls
  python sim/sweep.py --seed 1                    # rerun only seed 1
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from agents import LLM  # noqa: E402
from run import run_one  # noqa: E402

LEVELS = (0.3, 0.6)
STRATEGIES = ("random", "hubs")


def conditions(shapes, seeds, baseline_only=False, controls=True):
    for shape in shapes:
        for seed in seeds:
            yield dict(shape=shape, pct=0.0, strategy="none", seed=seed, control=False)
            if not baseline_only:
                for pct in LEVELS:
                    for strat in STRATEGIES:
                        yield dict(shape=shape, pct=pct, strategy=strat, seed=seed, control=False)
        if controls:
            yield dict(shape=shape, pct=0.0, strategy="none", seed=seeds[0], control=True)


async def main_async(a):
    seeds = [a.seed] if a.seed is not None else list(range(a.seeds))
    conds = list(conditions(a.shapes, seeds, a.baseline_only, not a.no_controls))
    calls = len(conds) * a.n * a.rounds
    print(f"{len(conds)} runs, ~{calls} LLM calls, model={a.model}, infection={a.infection}, coop={a.coop}")
    if a.dry_run:
        for c in conds:
            print("  ", c)
        return
    llm = LLM(a.model, concurrency=a.concurrency, effort=a.effort)
    sem = asyncio.Semaphore(a.parallel_runs)

    async def go(c):
        async with sem:
            try:
                await run_one(n=a.n, rounds=a.rounds, llm=llm, p0=a.p0, infection=a.infection, coop=a.coop, **c)
            except Exception as e:  # keep the sweep alive; failed runs just rerun next time
                print(f"FAILED {c}: {type(e).__name__}: {e}")

    await asyncio.gather(*(go(c) for c in conds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=["chain", "star", "scale_free"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=None, help="run only this seed")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--p0", default="leaf", choices=["leaf", "hub"])
    ap.add_argument("--model", default="muse-spark-1.3")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--concurrency", type=int, default=32, help="max simultaneous API calls")
    ap.add_argument("--parallel-runs", type=int, default=4)
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--infection", default="persuasive", choices=["plain", "persuasive"])
    ap.add_argument("--coop", action=argparse.BooleanOptionalAction, default=True, help="tell all agents to adopt team conventions (--no-coop to disable)")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
