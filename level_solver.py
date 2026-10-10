"""
Level solver for the quickest path problem with fuzzy lead times and fuzzy capacities
(code of the revised manuscript, taken from fqp.py).

Triangular fuzzy data: lead time (t1,t2,t3), capacity (r1,r2,r3), 0 < t1 <= t2 <= t3, 0 < r1 <= r2 <= r3.
alpha-cuts:  tL = t1 + a (t2-t1),  tR = t3 - a (t3-t2),  rL = r1 + a (r2-r1),  rR = r3 - a (r3-r2).

Core (identical to fqp.py):
  * Net                               -- arc-list network used by all algorithms
  * level_label(net, a, V)            -- exact level solver by bottleneck-augmented labelling (Algorithm 1 of the paper, LabelSolver)
  * level_lattice(net, a, V)          -- lattice sweep over capacity thresholds (baseline level solver of Section 5.1)
  * brute_force_paths / brute_level   -- enumeration of all simple paths for verification
  * path_T, rc_filter, canon_set      -- helpers shared with parametric.py / experiments.py

Interface used by case_study.py (same names as the old qpp_fuzzy_solver.py):
  * TriangularFuzzy, Arc, Network, Path, LevelResult
  * level_solver(network, alpha, V, method="label")
"""
from __future__ import annotations
import math, time
from dataclasses import dataclass
from collections import deque, defaultdict
from typing import List, Tuple
import numpy as np

EPS = 1e-10


# ---------------------------------------------------------------- network
class Net:
    def __init__(self, n, s, t):
        self.n, self.s, self.t = n, s, t
        self.arcs = []            # (u, v, (t1,t2,t3), (r1,r2,r3))
        self.out = defaultdict(list)

    def add(self, u, v, T, R):
        self.out[u].append(len(self.arcs))
        self.arcs.append((u, v, tuple(T), tuple(R)))

    @property
    def m(self):
        return len(self.arcs)

    # affine coefficients (c0, c1): value = c0 + c1*a
    def tL(self, e): t = self.arcs[e][2]; return (t[0], t[1] - t[0])
    def tR(self, e): t = self.arcs[e][2]; return (t[2], -(t[2] - t[1]))
    def rL(self, e): r = self.arcs[e][3]; return (r[0], r[1] - r[0])
    def rR(self, e): r = self.arcs[e][3]; return (r[2], -(r[2] - r[1]))


def ev(c, a):
    return c[0] + c[1] * a


def path_T(net, path, a, V):
    """(T^L, T^R, T^C) of a path (list of arc ids) at level a."""
    tl = sum(ev(net.tL(e), a) for e in path)
    tr = sum(ev(net.tR(e), a) for e in path)
    rl = min(ev(net.rL(e), a) for e in path)
    rr = min(ev(net.rR(e), a) for e in path)
    TL = tl + V / rr
    TR = tr + V / rl
    return TL, TR, 0.5 * (TL + TR)


def rc_dominates(x, y, tol=1e-9):
    """x, y = (TR, TC). strict RC dominance x < y."""
    return (x[0] <= y[0] + tol and x[1] <= y[1] + tol and
            (x[0] < y[0] - tol or x[1] < y[1] - tol))


def rc_filter(items, tol=1e-9):
    """items: list of (key, (TR,TC)). Keep non-dominated, one representative per equal value."""
    keep = []
    for k, v in items:
        if any(rc_dominates(w, v, tol) for _, w in items):
            continue
        if any(abs(w[0] - v[0]) <= tol and abs(w[1] - v[1]) <= tol for _, w in keep):
            continue
        keep.append((k, v))
    return keep


def canon_set(net, S, a, V, nd=7):
    """canonical description of an optimal set by its interval values (path-identity free)."""
    return frozenset((round(path_T(net, p, a, V)[1], nd), round(path_T(net, p, a, V)[2], nd)) for p in S)


# ---------------------------------------------------------------- brute force
def brute_force_paths(net, limit=200000):
    paths = []
    stack = [(net.s, [], {net.s})]
    while stack:
        u, p, seen = stack.pop()
        if u == net.t:
            paths.append(tuple(p))
            if len(paths) > limit:
                raise RuntimeError("too many paths")
            continue
        for e in net.out[u]:
            v = net.arcs[e][1]
            if v not in seen:
                stack.append((v, p + [e], seen | {v}))
    return paths


def brute_level(net, paths, a, V):
    items = [(p, path_T(net, p, a, V)[1:]) for p in paths]
    return rc_filter(items)


# ---------------------------------------------------------------- level solver: labelling (Algorithm 1 of the paper)
def level_label(net, a, V, prune=True):
    """Exact alpha-optimal set (minimal complete set) by labelling on (tauR, tauC, rhoL, rhoR)."""
    tR = [ev(net.tR(e), a) for e in range(net.m)]
    tC = [0.5 * (ev(net.tL(e), a) + ev(net.tR(e), a)) for e in range(net.m)]
    rL = [ev(net.rL(e), a) for e in range(net.m)]
    rR = [ev(net.rR(e), a) for e in range(net.m)]
    INF = math.inf
    labels = [(0.0, 0.0, INF, INF, -1, -1)]   # tauR, tauC, rhoL, rhoR, parent, arc
    alive = [True]
    at = defaultdict(list)
    at[net.s].append(0)
    Q = deque([0])
    sinkvals = []      # (TR, TC) of sink labels for bound pruning
    nlab = 1

    def dom(x, y):  # weak dominance of label x over y
        return x[0] <= y[0] + EPS and x[1] <= y[1] + EPS and x[2] >= y[2] - EPS and x[3] >= y[3] - EPS

    while Q:
        i = Q.popleft()
        if not alive[i]:
            continue
        L = labels[i]
        u = None
        # node of label i: recover via arc
        u = net.s if L[5] < 0 else net.arcs[L[5]][1]
        if u == net.t:
            continue
        for e in net.out[u]:
            v = net.arcs[e][1]
            nl = (L[0] + tR[e], L[1] + tC[e], min(L[2], rL[e]), min(L[3], rR[e]), i, e)
            if prune and v != net.t:
                TRx = nl[0] + V / nl[2]
                TCx = nl[1] + 0.5 * V * (1 / nl[2] + 1 / nl[3])
                if any(s0 <= TRx + EPS and s1 <= TCx + EPS for s0, s1 in sinkvals):
                    continue
            bad = False
            for j in at[v]:
                if alive[j] and dom(labels[j], nl):
                    bad = True
                    break
            if bad:
                continue
            for j in at[v]:
                if alive[j] and dom(nl, labels[j]):
                    alive[j] = False
            labels.append(nl); alive.append(True); nlab += 1
            k = len(labels) - 1
            at[v] = [j for j in at[v] if alive[j]] + [k]
            if v == net.t:
                sinkvals.append((nl[0] + V / nl[2], nl[1] + 0.5 * V * (1 / nl[2] + 1 / nl[3])))
            else:
                Q.append(k)
    res = []
    for j in at[net.t]:
        if alive[j]:
            p = []
            k = j
            while labels[k][5] >= 0:
                p.append(labels[k][5]); k = labels[k][4]
            p = tuple(reversed(p))
            res.append((p, path_T(net, p, a, V)[1:]))
    return rc_filter(res), nlab


# ---------------------------------------------------------------- level solver: lattice sweep (baseline of Section 5.1)
def _bsp(net, arcset, tR, tC):
    """bicriteria (tauR, tauC) efficient paths restricted to arcset, labelling."""
    labels = [(0.0, 0.0, -1, -1)]
    alive = [True]
    at = defaultdict(list); at[net.s].append(0)
    Q = deque([0])
    while Q:
        i = Q.popleft()
        if not alive[i]:
            continue
        L = labels[i]
        u = net.s if L[3] < 0 else net.arcs[L[3]][1]
        if u == net.t:
            continue
        for e in net.out[u]:
            if not arcset[e]:
                continue
            v = net.arcs[e][1]
            nl = (L[0] + tR[e], L[1] + tC[e], i, e)
            if any(alive[j] and labels[j][0] <= nl[0] + EPS and labels[j][1] <= nl[1] + EPS for j in at[v]):
                continue
            for j in at[v]:
                if alive[j] and nl[0] <= labels[j][0] + EPS and nl[1] <= labels[j][1] + EPS:
                    alive[j] = False
            labels.append(nl); alive.append(True)
            k = len(labels) - 1
            at[v] = [j for j in at[v] if alive[j]] + [k]
            if v != net.t:
                Q.append(k)
    out = []
    for j in at[net.t]:
        if alive[j]:
            p = []; k = j
            while labels[k][3] >= 0:
                p.append(labels[k][3]); k = labels[k][2]
            out.append(tuple(reversed(p)))
    return out


def level_lattice(net, a, V, time_limit=None):
    """Lattice sweep. time_limit (s): raise TimeoutError when exceeded (None = no limit, as in fqp.py)."""
    deadline = None if time_limit is None else time.time() + time_limit
    tR = [ev(net.tR(e), a) for e in range(net.m)]
    tC = [0.5 * (ev(net.tL(e), a) + ev(net.tR(e), a)) for e in range(net.m)]
    rL = np.array([ev(net.rL(e), a) for e in range(net.m)])
    rR = np.array([ev(net.rR(e), a) for e in range(net.m)])
    Ls = sorted(set(rL.tolist())); Rs = sorted(set(rR.tolist()))
    seen = set(); cand = {}; cells = 0
    for A in Ls:
        for B in Rs:
            if B < A:
                continue
            arcset = (rL >= A - EPS) & (rR >= B - EPS)
            key = arcset.tobytes()
            if key in seen:
                continue
            seen.add(key)
            cells += 1
            if deadline is not None and time.time() > deadline:
                raise TimeoutError
            for p in _bsp(net, arcset, tR, tC):
                cand[p] = path_T(net, p, a, V)[1:]
    return rc_filter(list(cand.items())), cells


# =====================================================================================
# Interface for case_study.py (same class / function names as the old qpp_fuzzy_solver)
# =====================================================================================
@dataclass
class TriangularFuzzy:
    """Triangular fuzzy number (l, m, u)."""
    l: float
    m: float
    u: float

    def __post_init__(self):
        assert self.l <= self.m <= self.u, f"Invalid triangular fuzzy: {self.l}, {self.m}, {self.u}"

    def alpha_cut(self, alpha: float) -> Tuple[float, float]:
        return (self.l + alpha * (self.m - self.l), self.u - alpha * (self.u - self.m))

    def as_tuple(self):
        return (self.l, self.m, self.u)

    def __repr__(self):
        return f"({self.l:.2f}, {self.m:.2f}, {self.u:.2f})"


@dataclass
class Arc:
    i: int
    j: int
    lead_time: TriangularFuzzy
    capacity: TriangularFuzzy

    def __repr__(self):
        return f"({self.i},{self.j}): t={self.lead_time}, r={self.capacity}"


class Network:
    """Network with Arc objects; `network.net` is the equivalent `Net` used by the algorithms."""

    def __init__(self, n: int, source: int = 0, sink: int = None):
        self.n = n
        self.source = source
        self.sink = sink if sink is not None else n - 1
        self.arcs: List[Arc] = []
        self.net = Net(n, self.source, self.sink)

    def add_arc(self, arc: Arc):
        self.arcs.append(arc)
        self.net.add(arc.i, arc.j, arc.lead_time.as_tuple(), arc.capacity.as_tuple())


@dataclass
class Path:
    nodes: List[int]
    arcs: Tuple[Tuple[int, int], ...]
    TR: float = math.nan      # worst-case transmission time T^R at the level it was computed
    TC: float = math.nan      # centre T^C

    def __hash__(self):
        return hash(self.arcs)

    def __eq__(self, other):
        return isinstance(other, Path) and self.arcs == other.arcs

    def __repr__(self):
        return "->".join(map(str, self.nodes))


def to_path(net: Net, arc_ids, TR=math.nan, TC=math.nan) -> Path:
    """Convert a tuple of arc ids of `net` into a Path (node list + (i,j) arcs)."""
    nodes = [net.s] + [net.arcs[e][1] for e in arc_ids]
    arcs = tuple((net.arcs[e][0], net.arcs[e][1]) for e in arc_ids)
    return Path(nodes, arcs, TR, TC)


@dataclass
class LevelResult:
    paths: List[Path]
    work: int            # labels created (method="label") or lattice cells solved (method="lattice")
    method: str


def level_solver(network: Network, alpha: float, V: float, method: str = "label") -> LevelResult:
    """alpha-optimal paths of the network at level alpha for demand V.
    method = "label"   -> Algorithm 1 of the paper (level_label, LabelSolver, fast, exact)
    method = "lattice" -> lattice sweep (level_lattice, baseline of Section 5.1)
    Both return the same optimal set."""
    net = network.net
    if method == "lattice":
        res, work = level_lattice(net, alpha, V)
    else:
        res, work = level_label(net, alpha, V)
    paths = [to_path(net, p, v[0], v[1]) for p, v in sorted(res, key=lambda x: (x[1][0], x[1][1]))]
    return LevelResult(paths, work, method)
