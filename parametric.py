"""
Parametric algorithm for the quickest path problem with fuzzy lead times and fuzzy capacities
(code of the revised manuscript, taken from fqp.py).

Core (identical to fqp.py):
  * parametric_exact(net, V, K)       -- exact critical partition of [0,1] (Algorithm 2 of the paper), K = number of equal cells
  * parametric_sampling(net, V)       -- the sampling-based procedure of the first version (for comparison)
  * parametric_levelsolver(net, V)    -- phase scheme of the paper (harvesting with LevelSolver = BSP per
                                         lattice cell, exact pairwise crossings, certification)

Interface used by case_study.py:
  * parametric_algorithm(network, V, method="levelsolver"|"exact") -> ParametricResult
    .partition = [((lo, hi), [Path, ...]), ...]   (same format as the old qpp_fuzzy_solver)
"""
from __future__ import annotations
import math, itertools, time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

from level_solver import (EPS, Net, ev, path_T, rc_filter, canon_set, level_label, level_lattice,
                          Network, Path, to_path)


# ---------------------------------------------------------------- piecewise-linear concave functions on [b0,b1]
def pl_min_line(verts, c, b0, b1):
    """verts: list of (x, y) of concave PL function or None (=+inf). Return min(verts, line c)."""
    if verts is None:
        return [(b0, ev(c, b0)), (b1, ev(c, b1))]
    out = []
    prev = None
    for (x, y) in verts:
        ly = ev(c, x)
        f = y - ly
        if prev is not None:
            px, py, pf = prev
            if (pf < -EPS and f > EPS) or (pf > EPS and f < -EPS):
                # crossing
                xs = px + (x - px) * pf / (pf - f)
                out.append((xs, ev(c, xs)))
        out.append((x, min(y, ly)))
        prev = (x, y, f)
    # remove collinear interior points
    res = [out[0]]
    for k in range(1, len(out) - 1):
        x0, y0 = res[-1]; x1, y1 = out[k]; x2, y2 = out[k + 1]
        if x1 - x0 < 1e-14:
            continue
        if abs((y1 - y0) * (x2 - x0) - (y2 - y0) * (x1 - x0)) > 1e-12 * (1 + abs(y1)):
            res.append(out[k])
    res.append(out[-1])
    return res


def pl_eval(verts, x):
    if verts is None:
        return math.inf
    for k in range(len(verts) - 1):
        x0, y0 = verts[k]; x1, y1 = verts[k + 1]
        if x <= x1 + 1e-15:
            if x1 - x0 < 1e-15:
                return min(y0, y1)
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return verts[-1][1]


def pl_geq(f, g):
    """f >= g on the whole cell (f, g concave PL). Exact: check at vertices of g."""
    if f is None:
        return True
    if g is None:
        return False
    return all(pl_eval(f, x) >= y - EPS * (1 + abs(y)) for (x, y) in g)


# ---------------------------------------------------------------- exact parametric algorithm (Algorithm 2 of the paper)
def cell_labelling(net, V, b0, b1, prune=True):
    """All paths that are not uniformly label-dominated on [b0,b1].
    Contains a representative of every alpha-optimal interval for every alpha in [b0,b1]."""
    def tau(e):
        tl, tr = net.tL(e), net.tR(e)
        tc = (0.5 * (tl[0] + tr[0]), 0.5 * (tl[1] + tr[1]))
        return (ev(tr, b0), ev(tr, b1), ev(tc, b0), ev(tc, b1))
    TA = [tau(e) for e in range(net.m)]
    labels = [((0.0, 0.0, 0.0, 0.0), None, None, -1, -1)]
    alive = [True]
    at = defaultdict(list); at[net.s].append(0)
    Q = deque([0])
    sinkb = []   # (max TR on cell, max TC on cell) upper bounds of sink paths
    nlab = 1

    def dom(x, y):
        return (x[0][0] <= y[0][0] + EPS and x[0][1] <= y[0][1] + EPS and
                x[0][2] <= y[0][2] + EPS and x[0][3] <= y[0][3] + EPS and
                pl_geq(x[1], y[1]) and pl_geq(x[2], y[2]))

    def bounds(l):
        rlmin = min(y for _, y in l[1]); rlmax = max(y for _, y in l[1])
        rrmin = min(y for _, y in l[2]); rrmax = max(y for _, y in l[2])
        trmin = min(l[0][0], l[0][1]); trmax = max(l[0][0], l[0][1])
        tcmin = min(l[0][2], l[0][3]); tcmax = max(l[0][2], l[0][3])
        lo = (trmin + V / rlmax, tcmin + 0.5 * V * (1 / rlmax + 1 / rrmax))
        hi = (trmax + V / rlmin, tcmax + 0.5 * V * (1 / rlmin + 1 / rrmin))
        return lo, hi

    while Q:
        i = Q.popleft()
        if not alive[i]:
            continue
        L = labels[i]
        u = net.s if L[4] < 0 else net.arcs[L[4]][1]
        if u == net.t:
            continue
        for e in net.out[u]:
            v = net.arcs[e][1]
            t = TA[e]
            nl = ((L[0][0] + t[0], L[0][1] + t[1], L[0][2] + t[2], L[0][3] + t[3]),
                  pl_min_line(L[1], net.rL(e), b0, b1),
                  pl_min_line(L[2], net.rR(e), b0, b1), i, e)
            if prune and v != net.t:
                lo, _ = bounds(nl)
                if any(h0 <= lo[0] + EPS and h1 <= lo[1] + EPS for h0, h1 in sinkb):
                    continue
            if any(alive[j] and dom(labels[j], nl) for j in at[v]):
                continue
            for j in at[v]:
                if alive[j] and dom(nl, labels[j]):
                    alive[j] = False
            labels.append(nl); alive.append(True); nlab += 1
            k = len(labels) - 1
            at[v] = [j for j in at[v] if alive[j]] + [k]
            if v == net.t:
                sinkb.append(bounds(nl)[1])
            else:
                Q.append(k)
    U = []
    for j in at[net.t]:
        if alive[j]:
            p = []; k = j
            while labels[k][4] >= 0:
                p.append(labels[k][4]); k = labels[k][3]
            U.append(tuple(reversed(p)))
    return U, nlab


def _path_pieces(net, p, b0, b1):
    """Return breakpoints and, per piece, affine coeffs of tauR, tauC, rhoL, rhoR."""
    trc = [sum(net.tR(e)[k] for e in p) for k in (0, 1)]
    tlc = [sum(net.tL(e)[k] for e in p) for k in (0, 1)]
    tcc = [0.5 * (trc[k] + tlc[k]) for k in (0, 1)]
    fl = None; fr = None
    for e in p:
        fl = pl_min_line(fl, net.rL(e), b0, b1)
        fr = pl_min_line(fr, net.rR(e), b0, b1)
    return trc, tcc, fl, fr


def _affine_on(f, x0, x1):
    y0, y1 = pl_eval(f, x0), pl_eval(f, x1)
    s = (y1 - y0) / (x1 - x0)
    return (y0 - s * x0, s)


def _roots_in(poly, x0, x1, fun):
    """real roots of numpy poly (highest first) in [x0,x1], refined by bisection on fun."""
    poly = np.trim_zeros(np.asarray(poly, dtype=float), 'f')
    if poly.size == 0 or np.max(np.abs(poly)) < 1e-12:
        return []
    if poly.size == 1:
        return []
    rts = np.roots(poly)
    out = []
    for r in rts:
        if abs(r.imag) > 1e-7 * (1 + abs(r.real)):
            continue
        x = r.real
        if x0 - 1e-9 <= x <= x1 + 1e-9:
            x = min(max(x, x0), x1)
            # Newton polish on the rational function
            for _ in range(30):
                h = 1e-9
                f0 = fun(x); d = (fun(x + h) - fun(x - h)) / (2 * h)
                if d == 0:
                    break
                nx = x - f0 / d
                if not (x0 - 1e-9 <= nx <= x1 + 1e-9):
                    break
                if abs(nx - x) < 1e-15:
                    x = nx; break
                x = nx
            out.append(min(max(x, x0), x1))
    return out


def path_ranges(PP, V, b0, b1):
    """Exact enclosures of T^R and T^C of a path over the cell [b0,b1]."""
    tr, tc, fl, fr = PP
    trs = (ev(tr, b0), ev(tr, b1)); tcs = (ev(tc, b0), ev(tc, b1))
    lmin = min(y for _, y in fl); lmax = max(y for _, y in fl)
    rmin = min(y for _, y in fr); rmax = max(y for _, y in fr)
    R = (min(trs) + V / lmax, max(trs) + V / lmin)
    C = (min(tcs) + 0.5 * V * (1 / lmax + 1 / rmax), max(tcs) + 0.5 * V * (1 / lmin + 1 / rmin))
    return R, C


def pair_crossings(net, PP, QQ, V, b0, b1, doR=True, doC=True):
    """All alpha in [b0,b1] with T^R_P = T^R_Q or T^C_P = T^C_Q (non-identical cases)."""
    trP, tcP, flP, frP = PP
    trQ, tcQ, flQ, frQ = QQ
    xs = sorted(set([x for x, _ in flP] + [x for x, _ in frP] + [x for x, _ in flQ] + [x for x, _ in frQ]))
    P1 = np.polynomial.polynomial
    roots = []
    for k in range(len(xs) - 1):
        x0, x1 = xs[k], xs[k + 1]
        if x1 - x0 < 1e-13:
            continue
        lP = _affine_on(flP, x0, x1); rP = _affine_on(frP, x0, x1)
        lQ = _affine_on(flQ, x0, x1); rQ = _affine_on(frQ, x0, x1)
        # T^R:  (trP*lP + V)*lQ - (trQ*lQ + V)*lP
        nR = P1.polysub(P1.polymul(P1.polyadd(P1.polymul(trP, lP), [V]), lQ),
                        P1.polymul(P1.polyadd(P1.polymul(trQ, lQ), [V]), lP))
        fR = lambda a: (ev(trP, a) + V / ev(lP, a)) - (ev(trQ, a) + V / ev(lQ, a))
        if doR:
            roots += _roots_in(nR[::-1], x0, x1, fR)
        # T^C: tc + V/2 (1/l + 1/r) = [tc*l*r + V/2 (l + r)] / (l r)
        numP = P1.polyadd(P1.polymul(P1.polymul(tcP, lP), rP), P1.polymul([0.5 * V], P1.polyadd(lP, rP)))
        numQ = P1.polyadd(P1.polymul(P1.polymul(tcQ, lQ), rQ), P1.polymul([0.5 * V], P1.polyadd(lQ, rQ)))
        nC = P1.polysub(P1.polymul(numP, P1.polymul(lQ, rQ)), P1.polymul(numQ, P1.polymul(lP, rP)))
        fC = lambda a: (ev(tcP, a) + 0.5 * V * (1 / ev(lP, a) + 1 / ev(rP, a))) - \
                       (ev(tcQ, a) + 0.5 * V * (1 / ev(lQ, a) + 1 / ev(rQ, a)))
        if doC:
            roots += _roots_in(nC[::-1], x0, x1, fC)
    return roots


def optimal_at(net, U, a, V):
    items = [(p, path_T(net, p, a, V)[1:]) for p in U]
    return frozenset(p for p, _ in rc_filter(items))


def parametric_exact(net, V, K=1, prune=True):
    """Exact critical partition. Returns list of (lo, hi, set_of_paths) for open pieces,
    the list of breakpoints with tie-sets, and statistics."""
    cells = [(k / K, (k + 1) / K) for k in range(K)]
    pieces = []
    stats = dict(labels=0, U=0, roots=0)
    allU = set()
    for (b0, b1) in cells:
        U, nl = cell_labelling(net, V, b0, b1, prune)
        stats['labels'] += nl
        stats['U'] = max(stats['U'], len(U))
        allU |= set(U)
        info = {p: _path_pieces(net, p, b0, b1) for p in U}
        rng_ = {p: path_ranges(info[p], V, b0, b1) for p in U}
        ov = lambda x, y: x[0] <= y[1] + EPS and y[0] <= x[1] + EPS
        pts = {b0, b1}
        for P, Qp in itertools.combinations(U, 2):
            dR = ov(rng_[P][0], rng_[Qp][0]); dC = ov(rng_[P][1], rng_[Qp][1])
            if not (dR or dC):
                stats['skipped'] = stats.get('skipped', 0) + 1
                continue
            r = pair_crossings(net, info[P], info[Qp], V, b0, b1, dR, dC)
            stats['roots'] += len(r)
            pts.update(r)
        pts = sorted(pts)
        # dedupe
        ded = [pts[0]]
        for x in pts[1:]:
            if x - ded[-1] > 1e-9:
                ded.append(x)
        if ded[-1] != b1:
            ded[-1] = b1
        for k in range(len(ded) - 1):
            lo, hi = ded[k], ded[k + 1]
            mid = 0.5 * (lo + hi)
            S = optimal_at(net, U, mid, V)
            pieces.append([lo, hi, S, U])
    # Merge adjacent pieces whose optimal sets have identical intervals (Algorithm 2, last line):
    # the representatives may differ between cells, so the sets are compared by their interval values
    # (T^R, T^C) at the midpoints of both pieces and at the common endpoint, not by path identity.
    def same_intervals(A, B, lo1, hi1, lo2, hi2):
        if A == B:
            return True
        return all(canon_set(net, A, x, V) == canon_set(net, B, x, V)
                   for x in (0.5 * (lo1 + hi1), hi1, 0.5 * (lo2 + hi2)))
    merged = []
    for lo, hi, S, U in pieces:
        if merged and same_intervals(merged[-1][2], S, merged[-1][0], merged[-1][1], lo, hi):
            merged[-1][1] = hi
        else:
            merged.append([lo, hi, S, U])
    bps = [(merged[k][1], optimal_at(net, merged[k][3] if True else None, merged[k][1], V))
           for k in range(len(merged) - 1)]
    stats['M'] = len(merged) - 1
    stats['allU'] = len(allU)
    return [(lo, hi, S) for lo, hi, S, _ in merged], bps, stats


# ---------------------------------------------------------------- sampling-based procedure of the first version
def parametric_sampling(net, V, rounds=4, n_samples=25, time_limit=None):
    """Re-implementation of the procedure used for the first submission's Table 4
    (start from {0,1}, harvest at 0, 1/2, 1, sampled crossings, <= 4 midpoint refinement rounds).
    time_limit (s): raise TimeoutError when exceeded (portable replacement of signal.alarm,
    which is not available on Windows)."""
    U = {}
    calls = 0
    deadline = None if time_limit is None else time.time() + time_limit

    def check_time():
        if deadline is not None and time.time() > deadline:
            raise TimeoutError

    def harvest(a):
        nonlocal calls
        check_time()
        calls += 1
        for p, _ in level_label(net, a, V)[0]:
            U[p] = True
    for a in (0.0, 0.5, 1.0):
        harvest(a)

    def crossings(bounds):
        new = set()
        L = list(U)
        for k in range(len(bounds) - 1):
            lo, hi = bounds[k], bounds[k + 1]
            if hi - lo < 1e-9:
                continue
            xs = [lo + (hi - lo) * j / n_samples for j in range(n_samples + 1)]
            for P, Q in itertools.combinations(L, 2):
                check_time()
                vals = []
                for x in xs:
                    TP = path_T(net, P, x, V); TQ = path_T(net, Q, x, V)
                    vals.append((TP[1] - TQ[1], TP[2] - TQ[2]))
                for c in (0, 1):
                    for j in range(n_samples):
                        f0, f1 = vals[j][c], vals[j + 1][c]
                        if f0 * f1 < 0:
                            a_, b_ = xs[j], xs[j + 1]
                            for _ in range(40):
                                md = 0.5 * (a_ + b_)
                                TP = path_T(net, P, md, V); TQ = path_T(net, Q, md, V)
                                fm = (TP[1] - TQ[1], TP[2] - TQ[2])[c]
                                if (fm > 0) == (f0 > 0):
                                    a_ = md
                                else:
                                    b_ = md
                            new.add(0.5 * (a_ + b_))
        return sorted(x for x in new if 1e-9 < x < 1 - 1e-9)
    bounds = sorted(set([0.0, 1.0] + crossings([0.0, 1.0])))
    for _ in range(rounds):
        before = len(U)
        for k in range(len(bounds) - 1):
            harvest(0.5 * (bounds[k] + bounds[k + 1]))
        if len(U) == before:
            break
        extra = crossings(bounds)
        if not extra:
            break
        bounds = sorted(set(bounds) | set(extra))
    res = []
    for k in range(len(bounds) - 1):
        lo, hi = bounds[k], bounds[k + 1]
        S = optimal_at(net, list(U), 0.5 * (lo + hi), V)
        if res and res[-1][2] == S:
            res[-1][1] = hi
        else:
            res.append([lo, hi, S])
    return [tuple(r) for r in res], calls, list(U)


# ---------------------------------------------------------------- parametric algorithm with LevelSolver / BSP (first-version phase scheme)
def parametric_levelsolver(net, V, max_rounds=100, prune=True, certify="cell", K=16):
    """Exact critical partition by the phase scheme of the paper:
      Phase 2  candidate harvesting: U <- LevelSolver(0) u LevelSolver(1/2) u LevelSolver(1)
      Phase 3  exact pairwise crossings of T^R (cubic) and T^C (quintic) for every pair P, Q in U
               (closed-form polynomial roots + Newton polishing; the bottleneck-arc switches of each path,
               i.e. the arc-order breakpoints of Phase 1 that matter, are handled inside pair_crossings)
      Phase 4  certification of every cell [beta, beta'):
               certify="cell"     -> one parametric BSP over the whole cell (label-correcting search whose
                                     labels are functions of alpha on the cell, cell_labelling); it returns every
                                     path that is optimal somewhere in the cell, so the result is EXACT.
                                     Cells are additionally split at k/K so that no cell is wider than 1/K.
               certify="midpoint" -> LevelSolver at the midpoint only (literal pseudocode); repeat until no new
                                     path appears. NOT exact: paths that are optimal only on a short interval
                                     containing no midpoint can be missed.
               New paths -> U, redo Phase 3; merge adjacent cells with identical optimal set.
    LevelSolver = level_lattice (lattice sweep): one bicriteria shortest path (BSP) problem per lattice cell.
    Returns (partition [(lo, hi, frozenset_of_paths)], breakpoints [(beta, tie_set)], stats) with
    stats: bsp_solves (BSP subproblems: lattice cells over all LevelSolver calls + cell BSPs of Phase 4),
    lattice_bsp, cell_bsp, labels (labels of the cell BSPs), level_calls, h_L, h_R
    (average numbers of distinct left / right capacity values per LevelSolver call), U, M, rounds."""
    U = {}
    st = dict(bsp_solves=0, level_calls=0, hL=[], hR=[], roots=0, rounds=0, cell_bsp=0, labels=0)

    def level_solver_call(a):
        res, cells = level_lattice(net, a, V)
        st['bsp_solves'] += cells
        st['level_calls'] += 1
        st['hL'].append(len({round(ev(net.rL(e), a), 9) for e in range(net.m)}))
        st['hR'].append(len({round(ev(net.rR(e), a), 9) for e in range(net.m)}))
        return [p for p, _ in res]

    def add_paths(paths):
        new = False
        for p in paths:
            if p not in U:
                U[p] = (_path_pieces(net, p, 0.0, 1.0), path_ranges(_path_pieces(net, p, 0.0, 1.0), V, 0.0, 1.0))
                new = True
        return new

    def harvest(a):
        return add_paths(level_solver_call(a))

    done_pairs = set()
    roots = set()

    def phase3():
        ov = lambda x, y: x[0] <= y[1] + EPS and y[0] <= x[1] + EPS
        keys = list(U)
        for P, Q in itertools.combinations(keys, 2):
            key = (P, Q) if P < Q else (Q, P)
            if key in done_pairs:
                continue
            done_pairs.add(key)
            (iP, rP), (iQ, rQ) = U[P], U[Q]
            dR = ov(rP[0], rQ[0]); dC = ov(rP[1], rQ[1])
            if prune and not (dR or dC):
                continue
            r = pair_crossings(net, iP, iQ, V, 0.0, 1.0, dR or not prune, dC or not prune)
            st['roots'] += len(r)
            roots.update(x for x in r if 1e-9 < x < 1 - 1e-9)

    def bounds():
        pts = sorted({0.0, 1.0} | roots)
        ded = [pts[0]]
        for x in pts[1:]:
            if x - ded[-1] > 1e-9:
                ded.append(x)
        ded[-1] = 1.0
        return ded

    # Phase 2
    for a in (0.0, 0.5, 1.0):
        harvest(a)
    # Phase 3
    phase3()
    # Phase 4
    pieces = []
    Ul = list(U)
    if certify == "cell":
        # every cell (Phase-3 bounds, refined so that no cell is wider than 1/K) is certified by one
        # parametric BSP over the cell; its output C contains every path optimal somewhere in the cell,
        # so the crossings among C (+ the harvested paths) inside the cell give the exact local partition.
        st['rounds'] = 1
        ov = lambda x, y: x[0] <= y[1] + EPS and y[0] <= x[1] + EPS
        grid = sorted(set(bounds()) | {k / K for k in range(K + 1)})
        cells = [(grid[k], grid[k + 1]) for k in range(len(grid) - 1) if grid[k + 1] - grid[k] > 1e-9]
        for lo, hi in cells:
            C, nl = cell_labelling(net, V, lo, hi, prune)
            st['cell_bsp'] += 1
            st['labels'] += nl
            add_paths(C)
            Cs = list(dict.fromkeys(list(C) + [p for p in Ul]))
            info = {p: _path_pieces(net, p, lo, hi) for p in Cs}
            rng_ = {p: path_ranges(info[p], V, lo, hi) for p in Cs}
            pts = {lo, hi} | {x for x in roots if lo < x < hi}
            for P, Q in itertools.combinations(Cs, 2):
                dR = ov(rng_[P][0], rng_[Q][0]); dC = ov(rng_[P][1], rng_[Q][1])
                if prune and not (dR or dC):
                    continue
                r = pair_crossings(net, info[P], info[Q], V, lo, hi, dR or not prune, dC or not prune)
                st['roots'] += len(r)
                pts.update(r)
            pts = sorted(pts)
            ded = [pts[0]]
            for x in pts[1:]:
                if x - ded[-1] > 1e-9:
                    ded.append(x)
            ded[-1] = hi
            for k in range(len(ded) - 1):
                S = optimal_at(net, Cs, 0.5 * (ded[k] + ded[k + 1]), V)
                if pieces and pieces[-1][2] == S:
                    pieces[-1][1] = ded[k + 1]
                else:
                    pieces.append([ded[k], ded[k + 1], S])
        Ul = list(U)
    else:
        for _ in range(max_rounds):
            st['rounds'] += 1
            B = bounds()
            new = False
            for k in range(len(B) - 1):
                new |= harvest(0.5 * (B[k] + B[k + 1]))
            if not new:
                break
            phase3()
        B = bounds()
        Ul = list(U)
        for k in range(len(B) - 1):
            lo, hi = B[k], B[k + 1]
            S = optimal_at(net, Ul, 0.5 * (lo + hi), V)
            if pieces and pieces[-1][2] == S:
                pieces[-1][1] = hi
            else:
                pieces.append([lo, hi, S])
    bps = [(pieces[k][1], optimal_at(net, Ul, pieces[k][1], V)) for k in range(len(pieces) - 1)]
    stats = dict(bsp_solves=st['bsp_solves'] + st['cell_bsp'], lattice_bsp=st['bsp_solves'],
                 cell_bsp=st['cell_bsp'], labels=st['labels'], level_calls=st['level_calls'],
                 h_L=sum(st['hL']) / len(st['hL']), h_R=sum(st['hR']) / len(st['hR']),
                 U=len(U), M=len(pieces) - 1, roots=st['roots'], rounds=st['rounds'])
    return [tuple(p) for p in pieces], bps, stats


# =====================================================================================
# Interface for case_study.py
# =====================================================================================
@dataclass
class ParametricResult:
    partition: List[Tuple[Tuple[float, float], List[Path]]]   # ((lo, hi), optimal paths on (lo, hi))
    breakpoints: int                                          # M: number of interior breakpoints
    breakpoint_sets: List[Tuple[float, List[Path]]]           # optimal (tie) set at each breakpoint
    bsp_solves: int = 0          # method "levelsolver": BSP subproblems solved over all LevelSolver calls
    h_L: float = 0.0             # method "levelsolver": average |L| (distinct left capacities) per LevelSolver call
    h_R: float = 0.0             # method "levelsolver": average |R| (distinct right capacities) per LevelSolver call
    labels: int = 0              # method "exact": labels created over all cells
    candidates: int = 0          # |U|: distinct candidate paths
    method: str = "exact"
    stats: dict = field(default_factory=dict)


def parametric_algorithm(network: Network, V: float, method: str = "exact", K: int = 16,
                         prune: bool = True) -> ParametricResult:
    """Critical partition of [0,1] for a Network of case_study.py.
    method = "exact"       -> Algorithm 2 of the paper: cell labelling over K equal cells (parametric_exact)
    method = "levelsolver" -> phase scheme of the first version with LevelSolver/BSP (parametric_levelsolver),
                              used only for the old table tab:scaling_parametric (parametric_random.py)
    Both return the exact critical partition."""
    net = network.net

    def conv(S, a):
        out = []
        for p in S:
            _, TR, TC = path_T(net, p, a, V)
            out.append(to_path(net, p, TR, TC))
        return sorted(out, key=lambda q: (q.TR, q.TC))

    if method == "exact":
        part, bps, st = parametric_exact(net, V, K=K, prune=prune)
        extra = dict(labels=st['labels'], candidates=st['allU'])
    else:
        part, bps, st = parametric_levelsolver(net, V, prune=prune)
        extra = dict(bsp_solves=st['bsp_solves'], h_L=st['h_L'], h_R=st['h_R'], candidates=st['U'])
    partition = [((lo, hi), conv(S, 0.5 * (lo + hi))) for lo, hi, S in part]
    bsets = [(b, conv(S, b)) for b, S in bps]
    return ParametricResult(partition, st['M'], bsets, method=method, stats=st, **extra)
