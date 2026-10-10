"""
Experiment "Does capacity fuzziness change the recommended route?"  ->  Table 6 of the paper (tab:value).

Road-like r x r grids (road_grid, fuzzy-data scheme of the case study), r in {10, 15}, 20 instances each
(seed 5000 + 100 r + rep), random source-sink pair at Manhattan distance >= r, V in {600, 6000, 30000},
alpha in {0, 0.25, 0.5, 0.75}.  The alpha-optimal set of the full fuzzy model (F, level_label) is compared
with the routes recommended by
  (I) the interval model at level alpha with capacities fixed at their modal values, and
  (C) the crisp quickest path at the core values (alpha = 1),
both evaluated in the full model.

Usage (in the Code folder):
    python table6.py                              # paper setting -> Result_table6/table6.tex
    python table6.py --table-only                 # only rebuild the table from the stored JSON
    python table6.py --resume                     # continue an interrupted run
    python table6.py --V 6000                     # other demand(s)   -> Result_table6/table6_V....tex
    python table6.py --sizes 10 --reps 3   # smaller trial
    python table6.py --V 6000 --tag run2          # repeated trial with a label
Every option that differs from the paper setting is appended to the file names, so trials never
overwrite each other. A new run never deletes results: an existing JSON is first renamed *_backup_<date>.json. All files of this table go to Result_table6/:
    res_table6[_options].json   (raw results)      table6[_options].tex   (LaTeX table)
"""
import os, sys, json, argparse, time, random, statistics as st
from collections import defaultdict

from level_solver import Net, level_label, path_T, canon_set
from generators import road_grid

SIZES = (10, 15)
REPS = 20
VS = (600, 6000, 30000)
ALPHAS = (0.0, 0.25, 0.5, 0.75)
REVISION_COLOR = True        # \color{red} in the table (as in the paper: Table 6 only)
NAME = 'table6'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'Result_' + NAME)     # every run of this table is stored here
RES_FILE = os.path.join(OUT_DIR, f'res_{NAME}.json')
TAB_FILE = os.path.join(OUT_DIR, f'{NAME}.tex')


def _num(x):
    return int(x) if float(x).is_integer() else float(x)


def configure(argv=None):
    """Command-line options. Every option that differs from the paper setting is appended to the
    file names, so repeated trials with other parameters never overwrite each other."""
    global SIZES, REPS, VS, RES_FILE, TAB_FILE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--V', type=float, nargs='+', help='demands V (paper: 600 6000 30000)')
    ap.add_argument('--sizes', type=int, nargs='+', help='grid sizes r (paper: 10 15)')
    ap.add_argument('--reps', type=int, help='instances per size')
    ap.add_argument('--tag', default='', help='free label appended to the file names, e.g. run2')
    ap.add_argument('--table-only', action='store_true', help='only rebuild the table from the stored JSON')
    ap.add_argument('--resume', action='store_true', help='continue an interrupted run (keeps the instances already in the JSON)')
    
    a = ap.parse_args(argv)
    tag = ''
    if a.V and tuple(_num(v) for v in a.V) != tuple(VS):
        VS = tuple(_num(v) for v in a.V); tag += '_V' + '-'.join(str(v) for v in VS)
    if a.sizes and tuple(a.sizes) != tuple(SIZES):
        SIZES = tuple(a.sizes); tag += '_' + 'r' + '-'.join(str(s) for s in SIZES)
    if a.reps and a.reps != REPS:
        REPS = a.reps; tag += f'_reps{REPS}'
    if a.tag:
        tag += '_' + a.tag
    RES_FILE = os.path.join(OUT_DIR, f'res_{NAME}{tag}.json')
    TAB_FILE = os.path.join(OUT_DIR, f'{NAME}{tag}.tex')
    return a



def with_modal_capacity(net):
    m = Net(net.n, net.s, net.t)
    for (u, v, T, R) in net.arcs:
        m.add(u, v, T, (R[1], R[1], R[1]))
    return m


# ------------------------------------------------------------------ experiment
def run(previous=()):
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = list(previous)
    per = defaultdict(int)
    for x in rows:
        per[(x['r'], x['rep'])] += 1
    done = {k for k, c in per.items() if c == len(VS) * len(ALPHAS)}
    rows = [x for x in rows if (x['r'], x['rep']) in done]      # drop an incomplete instance
    for r in SIZES:
        for rep in range(REPS):
            if (r, rep) in done:
                continue
            net = road_grid(r, r, seed=5000 + 100 * r + rep)
            rng = random.Random(rep + 77 * r)
            while True:
                s = (rng.randrange(r), rng.randrange(r)); t = (rng.randrange(r), rng.randrange(r))
                if abs(s[0] - t[0]) + abs(s[1] - t[1]) >= r:
                    break
            net.s = s[0] * r + s[1]; net.t = t[0] * r + t[1]
            netI = with_modal_capacity(net); netI.s, netI.t = net.s, net.t
            for V in VS:
                coreC = [p for p, _ in level_label(net, 1.0, V)[0]]       # crisp model at the core
                for a in ALPHAS:
                    F = level_label(net, a, V)[0]
                    Fset = set(p for p, _ in F)
                    Fval = canon_set(net, Fset, a, V)
                    TRstar = min(v[0] for _, v in F)
                    TCstar = min(v[1] for _, v in F)
                    rec = {}
                    rec['I'] = [p for p, _ in level_label(netI, a, V)[0]]
                    rec['C'] = coreC
                    row = dict(r=r, rep=rep, V=V, alpha=a, nF=len(F))
                    for key, X in rec.items():
                        vals = [path_T(net, p, a, V) for p in X]
                        dom = [canon_set(net, [p], a, V) <= Fval for p in X]
                        row[f'{key}_n'] = len(X)
                        row[f'{key}_dominated_any'] = int(not all(dom))
                        row[f'{key}_dominated_all'] = int(not any(dom))
                        row[f'{key}_TRexcess'] = (min(v[1] for v in vals) - TRstar) / TRstar
                        row[f'{key}_TCexcess'] = (min(v[2] for v in vals) - TCstar) / TCstar
                    rows.append(row)
            print(f'grid {r}x{r}, instance {rep} done', flush=True)
            with open(RES_FILE, 'w') as f:
                json.dump(rows, f, indent=1, default=float)
    return rows


# ------------------------------------------------------------------ table
def fmt(x, d=1):
    return '--' if x is None else f'{x:.{d}f}'


def table(R):
    g = defaultdict(list)
    for r in R:
        g[(r['V'], r['alpha'])].append(r)
    lines = []
    for V in VS:
        for a in ALPHAS:
            rs = g[(V, a)]
            cells = [str(V) if a == 0.0 else '', f'{a:.2f}', fmt(st.mean(r['nF'] for r in rs), 2)]
            for k in ('I', 'C'):
                dom = [r for r in rs if r[f'{k}_dominated_any']]
                pct = 100 * len(dom) / len(rs)
                exc = [100 * r[f'{k}_TRexcess'] for r in dom]
                cells += [fmt(pct, 1), fmt(st.mean(exc), 2) if exc else '--', fmt(max(exc), 1) if exc else '--']
            lines.append(' & '.join(cells) + r' \\')
        lines.append(r'\hline' if V != VS[-1] else '')
    body = '\n'.join(l for l in lines if l)
    return r'''\begin{table}[H]
''' + (r'''\color{red}
''' if REVISION_COLOR else '') + r'''\centering
\caption{Routes recommended by models that ignore capacity fuzziness, evaluated in the fuzzy model (''' + str(len(SIZES) * REPS) + r''' road-like grids, $r\in\{''' + ','.join(str(x) for x in SIZES) + r'''\}$)}
\label{tab:value}
\footnotesize
\begin{tabular}{rrr rrr rrr}
\hline
 & & & \multicolumn{3}{c}{(I) interval lead times,} & \multicolumn{3}{c}{(C) crisp core values} \\
 & & & \multicolumn{3}{c}{modal capacities} & \multicolumn{3}{c}{} \\
\cline{4-6}\cline{7-9}
 & & & dominated & \multicolumn{2}{c}{excess of $\TR$} & dominated & \multicolumn{2}{c}{excess of $\TR$} \\
$V$ & $\al$ & $|\Pp^\ast_F(\al)|$ & (\%) & mean (\%) & max (\%) & (\%) & mean (\%) & max (\%) \\
\hline
''' + body + r'''
\hline
\end{tabular}
\par\smallskip
\parbox{0.95\textwidth}{\footnotesize ``Dominated'': at least one recommended route is strictly $\precrc$-dominated in the full fuzzy model at level $\al$. ``Excess'': relative excess of the smallest worst-case transmission time $\TR$ among the recommended routes over the smallest $\TR$ of the full model, over the dominated cases.}
\end{table}
'''


def summary(R):
    """Numbers quoted in the text after the table."""
    print('\nSummary (for the text):')
    for k, name in (('I', 'interval model, modal capacities'), ('C', 'crisp model')):
        pct = 100 * st.mean(r[f'{k}_dominated_any'] for r in R)
        byV = [100 * st.mean(r[f'{k}_dominated_any'] for r in R if r['V'] == V) for V in VS]
        exc = [100 * r[f'{k}_TRexcess'] for r in R if r[f'{k}_dominated_any']]
        print(f"  {name}: dominated in {pct:.1f}% of {len(R)} cases "
              f"(by V: {', '.join(f'{x:.1f}%' for x in byV)}); excess of T^R over the dominated cases: "
              f"mean {(st.mean(exc) if exc else 0.0):.1f}%, max {(max(exc) if exc else 0.0):.1f}%")


if __name__ == '__main__':
    args = configure()
    print(f'{NAME}: sizes={SIZES}, instances per size={REPS}, V={VS}')
    print(f'  results -> {RES_FILE}')
    if args.table_only:
        with open(RES_FILE) as f:
            R = json.load(f)
    else:
        previous = []
        if os.path.exists(RES_FILE):
            if args.resume:
                with open(RES_FILE) as f:
                    previous = json.load(f)
                print(f'  resuming: {len(previous)} result rows already in {RES_FILE}')
            else:
                backup = RES_FILE[:-5] + time.strftime('_backup_%Y%m%d_%H%M%S') + '.json'
                os.replace(RES_FILE, backup)
                print(f'  previous results kept as {backup}')
        R = run(previous)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = table(R)
    with open(TAB_FILE, 'w', encoding='utf-8') as f:
        f.write(out[0] if isinstance(out, tuple) else out)
    summary(R)
    print(f'\nTable written to {TAB_FILE}')
    
