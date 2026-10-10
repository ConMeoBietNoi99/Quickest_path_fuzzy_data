"""
Instance generators of the experiments (identical to fqp.py of the revised manuscript).
  random_dag(n, density, window, seed)   -- banded random DAG of the first version (Section "Exact partition versus sampling")
  random_digraph(n, p, seed)             -- Erdos-Renyi digraph (Section "Correctness")
  road_grid(rows, cols, seed)            -- road-like grid with the fuzzy-data scheme of the case study
"""
import random
from level_solver import Net


# ---------------------------------------------------------------- generators
def tri(rng, lo, hi, s_lo, s_hi):
    a = rng.uniform(lo, hi); b = a + rng.uniform(s_lo, s_hi); c = b + rng.uniform(s_lo, s_hi)
    return (a, b, c)


def random_dag(n, density=0.25, window=4, seed=0):
    """The banded random DAG used by the first version's code (qpp_fuzzy_solver.generate_random_network)."""
    rng = random.Random(seed)
    while True:
        net = Net(n, 0, n - 1)
        for i in range(n):
            for j in range(i + 1, min(n, i + window + 1)):
                if j == i + 1 or rng.random() < density:
                    net.add(i, j, tri(rng, 1, 5, 1, 8), tri(rng, 2, 10, 1, 10))
        return net


def random_digraph(n, p, seed=0):
    """Erdos-Renyi digraph, s=0, t=n-1, connected s->t ensured by retry."""
    rng = random.Random(seed)
    while True:
        net = Net(n, 0, n - 1)
        for i in range(n):
            for j in range(n):
                if i != j and rng.random() < p:
                    net.add(i, j, tri(rng, 1, 5, 1, 8), tri(rng, 2, 10, 1, 10))
        # reachability
        seen = {0}; st = [0]
        while st:
            u = st.pop()
            for e in net.out[u]:
                v = net.arcs[e][1]
                if v not in seen:
                    seen.add(v); st.append(v)
        if n - 1 in seen:
            return net
        seed += 1000; rng = random.Random(seed)


ROAD = {  # v_r (km/h), kappa_r, gamma_r, lanes, urban   (Table of the case study)
    'motorway': (100, 35, 0.4, 4.0, 0), 'trunk': (80, 28, 0.6, 4.0, 0),
    'primary': (70, 25, 0.7, 4.0, 0), 'secondary': (50, 18, 0.9, 2.0, 0),
    'tertiary': (40, 12, 1.0, 2.0, 0), 'residential': (30, 10, 1.2, 1.5, 1),
    'unclassified': (35, 8, 1.1, 1.5, 1),
}


def road_arc(rng, L, rtype):
    """Fuzzy data generator of Section 7.3 (revised): road type + length -> triangular
    lead time (t^o,t^m,t^p) in seconds and triangular capacity (c^p,c^m,c^o) in units per second.
    The vertices are ordered by construction, so no correction step is needed."""
    v, kap, gam, lanes, urb = ROAD[rtype]
    L = max(L, 10.0)
    tau0 = L / (v / 3.6)
    if urb:
        to = tau0 * rng.uniform(0.80, 0.95); tm = tau0 * rng.uniform(1.00, 1.20)
        tp = tau0 * (1.15 + gam * rng.uniform(0.9, 1.9))
    else:
        to = tau0 * rng.uniform(0.75, 0.90); tm = tau0 * rng.uniform(0.95, 1.15)
        tp = tau0 * (1.05 + gam * rng.uniform(0.35, 0.85))
    k0 = kap * lanes
    eta = rng.uniform(0.65, 1.35)
    if urb:
        cm = k0 * eta * rng.uniform(0.60, 0.80); co = k0 * eta * rng.uniform(0.90, 1.30)
    else:
        cm = k0 * eta * rng.uniform(0.85, 1.00); co = k0 * eta * rng.uniform(1.00, 1.20)
    cp = cm * (1.0 - gam * rng.uniform(0.30, 0.50))
    assert to < tm < tp and cp < cm <= co
    return (to, tm, tp), (cp, cm, co)


def road_grid(rows, cols, seed=0, main_every=4):
    """Grid road network: every `main_every`-th row/column is an arterial (primary/secondary/trunk),
    the rest is urban (residential/tertiary/unclassified). Bidirectional arcs. s = top-left, t = bottom-right."""
    rng = random.Random(seed)
    idx = lambda i, j: i * cols + j
    net = Net(rows * cols, idx(0, 0), idx(rows - 1, cols - 1))
    edges = []
    for i in range(rows):
        for j in range(cols):
            if j + 1 < cols:
                edges.append(((i, j), (i, j + 1), i % main_every == 0))
            if i + 1 < rows:
                edges.append(((i, j), (i + 1, j), j % main_every == 0))
    for (a, b, main) in edges:
        if main:
            rtype = rng.choices(['trunk', 'primary', 'secondary'], [0.2, 0.4, 0.4])[0]
        else:
            rtype = rng.choices(['tertiary', 'residential', 'unclassified'], [0.3, 0.5, 0.2])[0]
        L = rng.uniform(200, 1200)
        for (x, y) in ((a, b), (b, a)):
            T, R = road_arc(rng, L, rtype)
            net.add(idx(*x), idx(*y), T, R)
    return net
