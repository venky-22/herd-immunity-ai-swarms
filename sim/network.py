"""Build graphs and pick which agents get the 'vaccine' prompt."""
import random

import networkx as nx

SHAPES = ("chain", "star", "scale_free")


def build_graph(shape: str, n: int, seed: int, p0: str = "leaf") -> nx.Graph:
    """Return a graph whose node 0 is patient zero.

    p0="leaf": patient zero is a lowest-degree node (picked by seed).
    p0="hub":  patient zero is a highest-degree node.
    Nodes are relabelled so patient zero is always 0.
    """
    if shape == "chain":
        g = nx.path_graph(n)
    elif shape == "star":
        g = nx.star_graph(n - 1)  # node 0 is the hub here; relabelled below
    elif shape == "scale_free":
        g = nx.barabasi_albert_graph(n, 2, seed=seed)
    else:
        raise ValueError(f"unknown shape {shape!r}")

    rng = random.Random(seed)
    degs = dict(g.degree())
    target = min(degs.values()) if p0 == "leaf" else max(degs.values())
    candidates = sorted(v for v, d in degs.items() if d == target)
    zero = rng.choice(candidates)
    # swap labels of `zero` and 0
    mapping = {v: v for v in g.nodes}
    mapping[zero], mapping[0] = 0, zero
    return nx.relabel_nodes(g, mapping)


def pick_protected(g: nx.Graph, pct: float, strategy: str, seed: int) -> set[int]:
    """Protect round(pct * (n-1)) agents, never agent 0."""
    pool = [v for v in g.nodes if v != 0]
    k = round(pct * len(pool))
    if k == 0 or strategy == "none":
        return set()
    rng = random.Random(seed * 1000 + 7)
    if strategy == "random":
        return set(rng.sample(pool, k))
    if strategy == "hubs":
        tiebreak = {v: rng.random() for v in pool}
        ranked = sorted(pool, key=lambda v: (-g.degree(v), tiebreak[v]))
        return set(ranked[:k])
    raise ValueError(f"unknown strategy {strategy!r}")
