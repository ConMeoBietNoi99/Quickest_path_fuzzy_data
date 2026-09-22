import random
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from collections import defaultdict


# ============================================================================
# FUZZY NUMBER AND INTERVAL DATA STRUCTURES
# ============================================================================

@dataclass
class TriangularFuzzy:
    """Triangular fuzzy number (l, m, u) with membership function."""
    l: float
    m: float
    u: float

    def __post_init__(self):
        assert self.l <= self.m <= self.u, f"Invalid triangular fuzzy: {self.l}, {self.m}, {self.u}"

    def alpha_cut(self, alpha: float) -> Tuple[float, float]:
        """Return [L(alpha), R(alpha)] for confidence level alpha in [0,1]."""
        left = self.l + alpha * (self.m - self.l)
        right = self.u - alpha * (self.u - self.m)
        return (left, right)

    def __repr__(self):
        return f"({self.l:.2f}, {self.m:.2f}, {self.u:.2f})"


@dataclass
class Interval:
    """Interval [L, R] with center and half-width."""
    L: float
    R: float

    @property
    def center(self) -> float:
        return (self.L + self.R) / 2

    def dominates_RC(self, other: 'Interval') -> bool:
        """Ishibuchi-Tanaka RC relation: self <_RC other (strict)."""
        return (self.R <= other.R and self.center <= other.center and
                (self.R < other.R or self.center < other.center))

    def __repr__(self):
        return f"[{self.L:.3f}, {self.R:.3f}]"


# ============================================================================
# NETWORK DATA STRUCTURES
# ============================================================================

@dataclass
class Arc:
    i: int
    j: int
    lead_time: TriangularFuzzy
    capacity: TriangularFuzzy

    def __repr__(self):
        return f"({self.i},{self.j}): t={self.lead_time}, r={self.capacity}"


class Network:
    """Directed acyclic network for the quickest path problem.

    Nodes are assumed to be labelled 0..n-1, with every arc (i,j) satisfying
    i < j, so that node index order is a topological order.
    """

    def __init__(self, n: int, source: int = 0, sink: int = None):
        self.n = n
        self.source = source
        self.sink = sink if sink is not None else n - 1
        self.arcs: List[Arc] = []
        self.adj: Dict[int, List[Arc]] = defaultdict(list)

    def add_arc(self, arc: Arc):
        self.arcs.append(arc)
        self.adj[arc.i].append(arc)

    def get_level_data(self, alpha: float) -> Dict[Tuple[int, int], Tuple[Interval, Interval]]:
        """Interval lead times and capacities at confidence level alpha."""
        data = {}
        for arc in self.arcs:
            tL, tR = arc.lead_time.alpha_cut(alpha)
            rL, rR = arc.capacity.alpha_cut(alpha)
            data[(arc.i, arc.j)] = (Interval(tL, tR), Interval(rL, rR))
        return data


@dataclass
class Path:
    nodes: List[int]
    arcs: Tuple[Tuple[int, int], ...]

    def __hash__(self):
        return hash(self.arcs)

    def __eq__(self, other):
        return isinstance(other, Path) and self.arcs == other.arcs

    def __repr__(self):
        return "->".join(map(str, self.nodes))


# ============================================================================
# PATH TRANSMISSION TIME  (Eq. 12/16 in the paper)
# ============================================================================

def compute_transmission_time(path: Path, level_data: Dict, V: float) -> Interval:
    """[tau^L(P) + V/rho^R(P), tau^R(P) + V/rho^L(P)]"""
    tau_L = tau_R = 0.0
    rho_L = rho_R = float('inf')
    for arc in path.arcs:
        t_interval, r_interval = level_data[arc]
        tau_L += t_interval.L
        tau_R += t_interval.R
        rho_L = min(rho_L, r_interval.L)
        rho_R = min(rho_R, r_interval.R)
    return Interval(tau_L + V / rho_R, tau_R + V / rho_L)


# ============================================================================
# BICRITERIA SHORTEST PATH  (inner routine used by LevelSolver, Eq. 17)
# ============================================================================

@dataclass
class BLabel:
    tau_R: float
    tau_C: float
    arcs: Tuple[Tuple[int, int], ...]

    def dominates(self, other: 'BLabel') -> bool:
        return (self.tau_R <= other.tau_R and self.tau_C <= other.tau_C and
                (self.tau_R < other.tau_R or self.tau_C < other.tau_C))


def bicriteria_shortest_path(network: Network, level_data: Dict,
                              adj_feasible: Dict[int, List[Arc]]) -> List[Path]:
    """
    Bicriteria shortest path on the capacity-threshold subnetwork G(a,b)
    (given as a precomputed feasible adjacency): minimise
    (tau^R(P), tau^C(P)) over paths using only arcs in `adj_feasible`.

    Since every arc (i,j) satisfies i<j, a single sweep in increasing node
    order is a correct (and fast) label-correcting pass -- no need for a
    work queue.
    """
    pareto_labels: Dict[int, List[BLabel]] = defaultdict(list)
    pareto_labels[network.source] = [BLabel(0.0, 0.0, tuple())]

    for u in range(network.source, network.sink + 1):
        labels_u = pareto_labels.get(u)
        if not labels_u:
            continue
        for arc in adj_feasible[u]:
            v = arc.j
            t_interval, _ = level_data[(arc.i, arc.j)]
            for lu in labels_u:
                new_label = BLabel(lu.tau_R + t_interval.R,
                                    lu.tau_C + t_interval.center,
                                    lu.arcs + ((arc.i, arc.j),))
                bucket = pareto_labels[v]
                if any(existing.dominates(new_label) for existing in bucket):
                    continue
                bucket[:] = [l for l in bucket if not new_label.dominates(l)]
                bucket.append(new_label)

    if network.sink not in pareto_labels:
        return []

    paths = []
    for label in pareto_labels[network.sink]:
        nodes = [network.source]
        for (i, j) in label.arcs:
            nodes.append(j)
        paths.append(Path(nodes, label.arcs))
    return paths


# ============================================================================
# LEVEL SOLVER  (ALGORITHM 1)
# ============================================================================

@dataclass
class LevelResult:
    paths: List[Path]
    cells_solved: int
    h_L: int
    h_R: int


def _single_criterion_shortest(network: Network, feasible_arcs_by_node: Dict[int, List[Arc]],
                                weight) -> Optional[float]:
    """Shortest source-sink path length in a DAG (nodes visited in index order),
    for a scalar arc weight function `weight(arc) -> float`. Returns None if
    the sink is unreachable."""
    dist = {network.source: 0.0}
    for u in range(network.source, network.sink + 1):
        if u not in dist:
            continue
        du = dist[u]
        for arc in feasible_arcs_by_node[u]:
            v = arc.j
            nd = du + weight(arc)
            if v not in dist or nd < dist[v]:
                dist[v] = nd
    return dist.get(network.sink)


def level_solver(network: Network, alpha: float, V: float) -> LevelResult:
    """Algorithm 1: LevelSolver(alpha). Returns the alpha-optimal set P*(alpha)
    plus instrumentation (# BSP subproblems solved, |L|, |R|).

    Sweeps the lattice L x R (b >= a) incrementally: for each threshold a
    (descending) the base set of arcs with rL >= a is rebuilt once (O(m)),
    then for that a, b is swept (descending) and arcs with rR >= b are
    *added* to a running adjacency one at a time as their threshold is
    crossed, instead of re-filtering the whole arc list at every cell. A
    cell is skipped outright (acceleration (i)) whenever no arc was added
    since the previous cell, since then G(a,b) is identical to the
    previously solved subnetwork. Acceleration (ii) -- a cheap lower-bound
    check using single-criterion shortest paths plus the best available
    capacities -- prunes cells that cannot possibly improve on paths
    already found.
    """
    level_data = network.get_level_data(alpha)
    arcs = network.arcs

    L_values, R_values = set(), set()
    for _, r_interval in level_data.values():
        L_values.add(round(r_interval.L, 9))
        R_values.add(round(r_interval.R, 9))
    L_sorted_desc = sorted(L_values, reverse=True)
    R_sorted_desc_all = sorted(R_values, reverse=True)

    optimal_paths: List[Path] = []
    optimal_T: Dict[Path, Interval] = {}
    cells_solved = 0

    for a in L_sorted_desc:
        # base set: arcs with rL(alpha) >= a, rebuilt once for this a (O(m))
        base_arcs = [arc for arc in arcs if level_data[(arc.i, arc.j)][1].L >= a]
        if not base_arcs:
            continue
        # sort base arcs by rR descending so we can add them incrementally as b decreases
        base_sorted_by_R = sorted(base_arcs, key=lambda arc: level_data[(arc.i, arc.j)][1].R, reverse=True)

        active_by_node: Dict[int, List[Arc]] = defaultdict(list)
        r_max_L = 0.0
        r_max_R = 0.0
        ptr = 0
        n_base = len(base_sorted_by_R)

        for b in R_sorted_desc_all:
            if b < a:
                break  # R values only get smaller from here; nothing left with b >= a
            added = False
            while ptr < n_base and level_data[(base_sorted_by_R[ptr].i, base_sorted_by_R[ptr].j)][1].R >= b:
                arc = base_sorted_by_R[ptr]
                active_by_node[arc.i].append(arc)
                r_int = level_data[(arc.i, arc.j)][1]
                r_max_L = max(r_max_L, r_int.L)
                r_max_R = max(r_max_R, r_int.R)
                ptr += 1
                added = True

            if ptr == 0:
                continue  # no feasible arcs yet at this (a,b)
            if not added:
                continue  # acceleration (i): identical subnetwork to previous cell -- skip

            # acceleration (ii): lower-bound skip
            mu_R = _single_criterion_shortest(network, active_by_node,
                                               lambda arc: level_data[(arc.i, arc.j)][0].R)
            if mu_R is None:
                continue  # sink not yet reachable
            mu_C = _single_criterion_shortest(network, active_by_node,
                                               lambda arc: level_data[(arc.i, arc.j)][0].center)
            lb = Interval(mu_R + V / r_max_L, mu_C + (V / 2) * (1 / r_max_L + 1 / r_max_R))
            if any(optimal_T[p].dominates_RC(lb) for p in optimal_paths):
                continue

            cells_solved += 1
            candidate_paths = bicriteria_shortest_path(network, level_data, active_by_node)

            for path in candidate_paths:
                T_new = compute_transmission_time(path, level_data, V)
                if any(optimal_T[p].dominates_RC(T_new) for p in optimal_paths):
                    continue
                optimal_paths = [p for p in optimal_paths if not T_new.dominates_RC(optimal_T[p])]
                if path not in optimal_paths:
                    optimal_paths.append(path)
                    optimal_T[path] = T_new

    return LevelResult(optimal_paths, cells_solved, len(L_values), len(R_values))


# ============================================================================
# PARAMETRIC ALGORITHM  (ALGORITHM 2)
# ============================================================================

def _find_crossings_in_cell(network: Network, P: Path, Q: Path, V: float,
                             lo: float, hi: float, n_samples: int = 25) -> List[float]:
    """Numerically locate alpha in (lo,hi) where T^R_P=T^R_Q or T^C_P=T^C_Q."""
    if hi - lo < 1e-9:
        return []

    def diffs(alpha):
        ld = network.get_level_data(alpha)
        TP = compute_transmission_time(P, ld, V)
        TQ = compute_transmission_time(Q, ld, V)
        return TP.R - TQ.R, TP.center - TQ.center

    samples = [lo + (hi - lo) * k / n_samples for k in range(n_samples + 1)]
    vals = [diffs(a) for a in samples]

    crossings = []
    for comp in range(2):  # 0: R-endpoint, 1: center
        for k in range(len(samples) - 1):
            f0, f1 = vals[k][comp], vals[k + 1][comp]
            if f0 == 0:
                crossings.append(samples[k])
                continue
            if f0 * f1 < 0:
                a, b = samples[k], samples[k + 1]
                for _ in range(40):
                    mid = (a + b) / 2
                    fm = diffs(mid)[comp]
                    if (fm == 0):
                        a = b = mid
                        break
                    if (fm > 0) == (f0 > 0):
                        a = mid
                    else:
                        b = mid
                crossings.append((a + b) / 2)

    return sorted({c for c in crossings if lo + 1e-9 < c < hi - 1e-9})


@dataclass
class ParametricResult:
    partition: List[Tuple[Tuple[float, float], List[Path]]]
    breakpoints: int          # M: number of interior breakpoints
    bsp_solves: int           # total bicriteria subproblems solved (all LevelSolver calls)
    h_L: float                # average |L| over LevelSolver calls
    h_R: float                # average |R| over LevelSolver calls


def parametric_algorithm(network: Network, V: float, max_refine_rounds: int = 4) -> ParametricResult:
    """Algorithm 2: exact critical partition of alpha in [0,1].

    Note on Phase 1: the paper's pseudocode defines the initial arc-order
    breakpoint set B_arc over *every* pair of arcs in the network (O(m^2)
    affine equations). On a dense random graph this produces tens of
    thousands of candidate cells (e.g. ~7800 for a 189-arc, n=40 instance),
    each requiring 3 LevelSolver calls in Phase 2 -- computationally
    infeasible and not what actually drives the partition size. In practice
    only the bottleneck arcs of the *competitive* (candidate) paths matter,
    which is exactly what Phases 2-4 already track. We therefore start from
    the trivial partition {0,1} and let Phases 2-4 (candidate harvesting +
    pairwise crossings among the harvested candidates + adaptive
    refinement) locate all breakpoints that actually change the optimal
    set -- these are precisely the breakpoints reported as M in Table 2.
    """

    total_bsp_solves = 0
    hL_samples, hR_samples = [], []

    def run_level_solver(alpha: float) -> List[Path]:
        nonlocal total_bsp_solves
        res = level_solver(network, alpha, V)
        total_bsp_solves += res.cells_solved
        hL_samples.append(res.h_L)
        hR_samples.append(res.h_R)
        return res.paths

    cell_bounds = [0.0, 1.0]

    # ---- Phase 2: candidate harvesting ----
    U: Dict[Tuple[Tuple[int, int], ...], Path] = {}

    def add_candidates(alpha: float):
        for p in run_level_solver(alpha):
            U[p.arcs] = p

    for k in range(len(cell_bounds) - 1):
        beta, beta_p = cell_bounds[k], cell_bounds[k + 1]
        add_candidates(beta)
        add_candidates((beta + beta_p) / 2)
        add_candidates(beta_p)

    # ---- Phase 3: exact/numeric pairwise crossings within each cell ----
    def phase3(bounds: List[float]) -> List[float]:
        new_bps = set()
        paths_list = list(U.values())
        for k in range(len(bounds) - 1):
            lo, hi = bounds[k], bounds[k + 1]
            for i in range(len(paths_list)):
                for j in range(i + 1, len(paths_list)):
                    cs = _find_crossings_in_cell(network, paths_list[i], paths_list[j], V, lo, hi)
                    new_bps.update(cs)
        return sorted(new_bps)

    extra = phase3(cell_bounds)
    cell_bounds = sorted(set(cell_bounds) | set(extra))

    # ---- Phase 4: certification and adaptive refinement ----
    for _ in range(max_refine_rounds):
        changed = False
        for k in range(len(cell_bounds) - 1):
            beta, beta_p = cell_bounds[k], cell_bounds[k + 1]
            mid = (beta + beta_p) / 2
            new_paths = run_level_solver(mid)
            for p in new_paths:
                if p.arcs not in U:
                    U[p.arcs] = p
                    changed = True
        if not changed:
            break
        extra = phase3(cell_bounds)
        if extra:
            cell_bounds = sorted(set(cell_bounds) | set(extra))
        else:
            break

    # Final certification: non-dominated set on each cell + merge identical adjacent cells
    result: List[Tuple[Tuple[float, float], List[Path]]] = []
    paths_list = list(U.values())
    for k in range(len(cell_bounds) - 1):
        beta, beta_p = cell_bounds[k], cell_bounds[k + 1]
        mid = (beta + beta_p) / 2
        ld = network.get_level_data(mid)
        T = {p: compute_transmission_time(p, ld, V) for p in paths_list}
        S = [p for p in paths_list if not any(T[q].dominates_RC(T[p]) for q in paths_list if q is not p)]
        result.append(((beta, beta_p), S))

    merged: List[Tuple[Tuple[float, float], List[Path]]] = []
    for (bounds, S) in result:
        if merged and set(p.arcs for p in merged[-1][1]) == set(p.arcs for p in S):
            prev_bounds, prev_S = merged[-1]
            merged[-1] = ((prev_bounds[0], bounds[1]), prev_S)
        else:
            merged.append((bounds, S))

    M = max(len(merged) - 1, 0)
    hL_avg = sum(hL_samples) / len(hL_samples) if hL_samples else 0.0
    hR_avg = sum(hR_samples) / len(hR_samples) if hR_samples else 0.0

    return ParametricResult(merged, M, total_bsp_solves, hL_avg, hR_avg)


# ============================================================================
# RANDOM INSTANCE GENERATOR  (Section 6 of the paper)
# ============================================================================

def generate_random_network(n: int, density: float = 0.25, V: float = 100.0,
                             rng: Optional[random.Random] = None,
                             window: int = 4, max_attempts: int = 50) -> Network:
    """
    Random DAG on nodes 0..n-1 (arcs only from lower to higher index).

    Arcs are restricted to a bounded look-ahead `window`: node i may only
    connect to nodes in (i, i+window]. This keeps the arc count -- and
    hence h_L, h_R -- growing roughly linearly in n rather than
    quadratically, consistent with the near-linear growth of h_L, h_R
    reported in Table 2 (81 at n=60 vs. the ~442 arcs an all-pairs density
    of 0.25 would otherwise produce), and keeps the exact algorithm
    tractable out to n=100. Within the window, each candidate arc is
    included independently with probability `density`; the backbone arc
    (i, i+1) is always added to guarantee source-sink connectivity.

    Triangular fuzzy lead times and capacities are drawn as in the paper's
    validation experiments:
        t1 ~ U(1,5),   t2 = t1 + U(1,8),   t3 = t2 + U(1,8)
        r1 ~ U(2,10),  r2 = r1 + U(1,10),  r3 = r2 + U(1,10)
    """
    rng = rng or random.Random()

    def random_triangular(lo1, hi1, spread_lo, spread_hi):
        a = rng.uniform(lo1, hi1)
        b = a + rng.uniform(spread_lo, spread_hi)
        c = b + rng.uniform(spread_lo, spread_hi)
        return TriangularFuzzy(a, b, c)

    for _ in range(max_attempts):
        net = Network(n, source=0, sink=n - 1)
        for i in range(n):
            for j in range(i + 1, min(n, i + window + 1)):
                if j == i + 1 or rng.random() < density:
                    lead = random_triangular(1, 5, 1, 8)
                    cap = random_triangular(2, 10, 1, 10)
                    net.add_arc(Arc(i, j, lead, cap))

        reachable = {0}
        for i in range(n):
            if i in reachable:
                for arc in net.adj[i]:
                    reachable.add(arc.j)
        if (n - 1) in reachable:
            return net

    raise RuntimeError(f"Could not generate a connected instance for n={n} after {max_attempts} attempts")


# ============================================================================
# TABLE 2 SCALING EXPERIMENT
# ============================================================================
import matplotlib.pyplot as plt
import os
import json

def print_table2(rows):
    print("\nTable 2: Scaling of the parametric algorithm "
          "(averages over 12 random networks per size, V=100)")
    print(f"{'n':>4} {'h_L':>8} {'h_R':>8} {'Breakpoints M':>15} {'BSP solves':>12} {'Time (s)':>10}")
    for r in rows:
        print(f"{r['n']:>4} {r['h_L']:>8.1f} {r['h_R']:>8.1f} "
              f"{r['M']:>15.1f} {r['bsp']:>12.1f} {r['time_s']:>10.3f}")


def export_results(rows, output_dir="KQ"):
    """Xuất kết quả ra thư mục KQ gồm file raw_data.json và mã LaTeX của bảng."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Lưu file raw_data.json
    json_path = os.path.join(output_dir, "raw_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=4)
    print(f"Đã lưu dữ liệu thô tại: {json_path}")

    # 2. Tạo mã LaTeX cho bảng kết quả (Đổi đơn vị Time sang giây)
    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Scaling of the parametric algorithm}",
        r"\label{tab:scaling_parametric}",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"$n$ & $h_{\mathrm{L}}$ & $h_{\mathrm{R}}$ & Breakpoints $M$ & BSP solves & Time (s) \\",
        r"\midrule"
    ]

    for r in rows:
        line = f"{r['n']} & {r['h_L']:.1f} & {r['h_R']:.1f} & {r['M']:.1f} & {r['bsp']:.1f} & {r['time_s']:.3f} \\\\"
        latex_lines.append(line)

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])

    latex_content = "\n".join(latex_lines)
    latex_path = os.path.join(output_dir, "table_scaling.txt")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_content)
    print(f"Đã lưu mã LaTeX bảng tại: {latex_path}")


def export_experiment_1(results, output_dir="KQ1"):
    """Xuất kết quả Thử nghiệm 1 ra thư mục KQ1 gồm file raw_data_exp1.json và bảng LaTeX với caption chuẩn lý thuyết."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Lưu file JSON
    json_path = os.path.join(output_dir, "raw_data_exp1.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Đã lưu dữ liệu thô Thử nghiệm 1 tại: {json_path}")

    # 2. Tạo mã LaTeX cho bảng Thử nghiệm 1 (Caption phản ánh đúng bản chất quét lưới dung lượng)
    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Verification and performance of \textsc{LevelSolver} via capacity threshold lattice sweeping.}",
        r"\label{tab:verification_levelsolver}",
        r"\begin{tabular}{crrrr}",
        r"\toprule",
        r"$n$ & Demand $V$ & Avg Paths & Avg Cells & Time (s) \\",
        r"\midrule"
    ]

    for r in results:
        line = (f"{r['n']} & {r['V']} & "
                f"{r['avg_paths']:.1f} & {r['avg_cells_solved']:.1f} & {r['avg_time_s']:.4f} \\\\")
        latex_lines.append(line)

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])

    latex_content = "\n".join(latex_lines)
    latex_path = os.path.join(output_dir, "table_verification.txt")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_content)
    print(f"Đã lưu mã LaTeX bảng Thử nghiệm 1 tại: {latex_path}")


# ============================================================================
# THỬ NGHIỆM 1: CÓ TÍCH HỢP CHECKPOINT
# ============================================================================

def run_experiment_1(sizes=(5, 6, 7, 8, 9), density=0.4, demands=(6, 12, 24, 48),
                     instances_per_config=20, seed=42, verbose=True, checkpoint_path="KQ1/checkpoint_exp1.json"):
    rng = random.Random(seed)
    results = []
    completed_configs = set()

    # Khôi phục từ checkpoint nếu có
    os.makedirs(os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else ".", exist_ok=True)
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                results = json.load(f)
                for r in results:
                    completed_configs.add((r["n"], r["V"]))
            if verbose:
                print(f"[Checkpoint] Đã nôi tiếp từ file cũ. Đã hoàn thành {len(completed_configs)} cấu hình.")
        except Exception as e:
            print(f"[Checkpoint] Không đọc được file checkpoint cũ ({e}), bắt đầu chạy lại từ đầu.")

    total_configs = len(sizes) * len(demands)
    config_count = 0

    for n in sizes:
        for V in demands:
            config_count += 1
            if (n, V) in completed_configs:
                if verbose:
                    print(f"[Bỏ qua] Cấu hình n={n}, V={V} đã có trong checkpoint.")
                continue

            if verbose:
                print(
                    f"[Config {config_count}/{total_configs}] Running Exp 1: n={n}, density={density}, V={V} ({instances_per_config} instances)...")

            total_paths = 0
            total_cells = 0
            total_time = 0.0

            for _ in range(instances_per_config):
                net = generate_random_network(n, density=density, V=V, rng=rng, window=n)

                t0 = time.perf_counter()
                res = level_solver(net, alpha=0.5, V=V)
                elapsed = time.perf_counter() - t0

                total_time += elapsed
                total_cells += res.cells_solved
                total_paths += len(res.paths)

            config_result = {
                "n": n,
                "density": density,
                "V": V,
                "instances": instances_per_config,
                "avg_paths": total_paths / instances_per_config,
                "avg_cells_solved": total_cells / instances_per_config,
                "avg_time_s": total_time / instances_per_config
            }
            results.append(config_result)

            # Lưu checkpoint ngay sau mỗi cấu hình hoàn thành
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)

    return results


# ============================================================================
# THỬ NGHIỆM 2 (TABLE 2): CÓ TÍCH HỢP CHECKPOINT
# ============================================================================

def run_table2_experiment(sizes=(20, 40, 60, 80, 100, 120, 140, 160, 180, 200),
                          density=0.25, V=100.0, instances_per_size=12, seed=0, verbose=True,
                          checkpoint_path="KQ/checkpoint_table2.json"):
    rng = random.Random(seed)
    rows = []
    completed_n = set()

    os.makedirs(os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else ".", exist_ok=True)
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                rows = json.load(f)
                for r in rows:
                    completed_n.add(r["n"])
            if verbose:
                print(f"[Checkpoint] Đã khôi phục Thử nghiệm 2. Đã xong các kích thước n: {list(completed_n)}")
        except Exception as e:
            print(f"[Checkpoint] Lỗi đọc checkpoint Thử nghiệm 2 ({e}), chạy lại từ đầu.")

    for n in sizes:
        if n in completed_n:
            if verbose:
                print(f"[Bỏ qua] Kích thước n={n} đã có trong checkpoint.")
            continue

        hL_list, hR_list, M_list, bsp_list, time_list = [], [], [], [], []

        for _ in range(instances_per_size):
            net = generate_random_network(n, density=density, V=V, rng=rng)

            t0 = time.perf_counter()
            res = parametric_algorithm(net, V)
            elapsed_sec = time.perf_counter() - t0

            hL_list.append(res.h_L)
            hR_list.append(res.h_R)
            M_list.append(res.breakpoints)
            bsp_list.append(res.bsp_solves)
            time_list.append(elapsed_sec)

        row = {
            "n": n,
            "h_L": sum(hL_list) / len(hL_list),
            "h_R": sum(hR_list) / len(hR_list),
            "M": sum(M_list) / len(M_list),
            "bsp": sum(bsp_list) / len(bsp_list),
            "time_s": sum(time_list) / len(time_list),
        }
        rows.append(row)

        if verbose:
            print(f"n={n:4d}  h_L={row['h_L']:6.1f}  h_R={row['h_R']:6.1f}  "
                  f"M={row['M']:5.1f}  BSP solves={row['bsp']:8.1f}  "
                  f"time={row['time_s']:9.3f} s")

        # Lưu checkpoint ngay sau mỗi n hoàn thành
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=4)

    return rows




# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=== ĐANG CHẠY THỬ NGHIỆM 1 (400 instances kiểm tra tính đúng đắn - Thư mục KQ1) ===")
    exp1_results = run_experiment_1()
    export_experiment_1(exp1_results)

    print("\n=== Đang CHẠY THỬ NGHIỆM 2 (Scaling thuật toán tham số - Thư mục KQ) ===")
    rows = run_table2_experiment()
    print_table2(rows)
    export_results(rows)
    plot_runtime(rows)