"""Fine-tune comparison on local Qwen 3.5 4B: scale-free, N=20, 8 rounds, 30% random protection.
  python finetune/eval_local.py base   # no protection + prompt vaccine
  python finetune/eval_local.py ft     # trained vaccine (needs VACCINE_ADAPTER=<adapter dir>)
"""
import asyncio, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim"))
from agents import LLM
from run import run_one

SEEDS = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1]


async def main(which):
    if which == "base":
        llm = LLM("local-qwen35-4b")
        conds = [(0.0, "none"), (0.3, "random")]
    else:
        llm = LLM("local-qwen35-4b-ft")
        conds = [(0.3, "random")]
    for seed in SEEDS:
        for pct, strat in conds:
            await run_one("scale_free", 20, 8, pct, strat, seed, llm, infection="persuasive", coop=True)

asyncio.run(main(sys.argv[1]))
