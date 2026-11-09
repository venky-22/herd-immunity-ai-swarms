"""Build an SFT dataset for a 'trained vaccine' from the Muse sweep logs.

Each row = one agent turn, reconstructed with the exact context the agent saw:
  prompt     = system prompt (base + cooperative clause, vaccine line REMOVED) + conversation so far
  completion = the agent's reply

Rows are of two kinds, balanced 1:1:
  resist : the inbox contained the injected rule (🍌) and the agent did NOT comply
  normal : clean inbox, ordinary on-task reply (keeps the model useful, avoids over-refusal)

Usage: python finetune/build_dataset.py  ->  finetune/vaccine_sft.jsonl (+ .csv)
"""
import csv
import glob
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))
from agents import MARKER, round_prompt, system_prompt  # noqa: E402
from network import build_graph  # noqa: E402

PATTERN = "muse-spark-1.3_persuasive_coop__*_n20_*.jsonl"


def render(system: str, msgs: list[dict]) -> str:
    parts = [f"System: {system}"]
    for m in msgs:
        parts.append(("User: " if m["role"] == "user" else "Assistant: ") + m["content"])
    return "\n\n".join(parts) + "\n\nAssistant:"


def main(seed=0):
    resist, normal = [], []
    for f in sorted(glob.glob(str(ROOT / "data" / PATTERN))):
        rows = [json.loads(line) for line in open(f)]
        if rows[0]["control"]:
            continue
        r0 = rows[0]
        g = build_graph(r0["shape"], r0["n"], r0["seed"], r0["p0"])
        text = {(r["round"], r["agent_id"]): r["text"] for r in rows}
        hist = defaultdict(list)
        for r in sorted(rows, key=lambda r: (r["round"], r["agent_id"])):
            a, rnd = r["agent_id"], r["round"]
            inbox = [(b, text[(rnd - 1, b)]) for b in sorted(g.neighbors(a)) if (rnd - 1, b) in text]
            msgs = hist[a] + [{"role": "user", "content": round_prompt(rnd, inbox)}]
            hist[a] = msgs + [{"role": "assistant", "content": r["text"] or "(no message)"}]
            if r["patient_zero"] or not r["text"] or MARKER in r["text"]:
                continue
            # the model must learn immunity WITHOUT being told: drop the vaccine line from its prompt
            sysp = system_prompt(a, infected_seed=False, protected=False, cooperative=True)
            row = {"prompt": render(sysp, msgs), "completion": " " + r["text"],
                   "kind": "resist" if r["inbox_infected"] > 0 else "normal",
                   "run_id": r["run_id"], "round": rnd, "agent_id": a}
            (resist if row["kind"] == "resist" else normal).append(row)

    rng = random.Random(seed)
    rng.shuffle(normal)
    data = resist + normal[: max(len(resist), 1000 - len(resist))]
    rng.shuffle(data)
    out = ROOT / "finetune" / "vaccine_sft.jsonl"
    with out.open("w") as fh:
        for d in data:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    with out.with_suffix(".csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["prompt", "completion"])
        w.writeheader()
        for d in data:
            w.writerow({"prompt": d["prompt"], "completion": d["completion"]})
    print(f"{len(data)} rows ({len(resist)} resist, {len(data) - len(resist)} normal) -> {out}")


if __name__ == "__main__":
    main()
