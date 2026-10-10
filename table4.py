"""
Experiment "Exact partition versus sampling"  ->  Table 4 of the paper (tab:sampling).

Banded random DAGs (random_dag, density 0.25, window 4), V = 100,
n in {20, 40, 60, 80, 100, 150, 200}, 10 instances per size (seed 1000*n + r).
For every instance:
  * exact critical partition (Algorithm 2 of the paper, parametric_exact, K = 16);
  * sampling-based procedure of the first version (parametric_sampling), capped at 300 s;
  * union of the optimal sets at 11 and at 101 equally spaced levels (level_label, Algorithm 1 of the paper).

Usage (in the Code folder):
    python table4.py                              # paper setting -> Result_table4/table4.tex
    python table4.py --table-only                 # only rebuild the table from the stored JSON
    python table4.py --resume                     # continue an interrupted run
    python table4.py --check                # also re-check every level where sampling and exact disagree
    python table4.py --V 200                     # other demand(s)   -> Result_table4/table4_V....tex
    python table4.py --sizes 20 40 60 --reps 3   # smaller trial
    python table4.py --V 200 --tag run2          # repeated trial with a label
Every option that differs from the paper setting is appended to the file names, so trials never
overwrite each other. A new run never deletes results: an existing JSON is first renamed *_backup_<date>.json. All files of this table go to Result_table4/:
    res_table4[_options].json   (raw results)      table4[_options].tex   (LaTeX table)
"""
import os, sys, json, argparse, time, statistics as st
from collections import defaultdict
import numpy as np

from level_solver import level_label, canon_set
from parametric import parametric_exact, parametric_sampling
from generators import random_dag

SIZES = (20, 40, 60, 80, 100, 150, 200)
REPS = 10
VS = (100,)                   # demand V (one value)
V = 100.0
K = 16
TCAP = 300                    # time cap (s) of the sampling procedure
REVISION_COLOR = False        # \color{red} in the table (as in the paper: Table 6 only)
NAME = 'table4'

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
    ap.add_argument('--V', type=float, nargs='+', help='demand V (one value; paper: 100)')
    ap.add_argument('--sizes', type=int, nargs='+', help='numbers of nodes n (paper: 20 40 60 80 100 150 200)')
    ap.add_argument('--reps', type=int, help='instances per size')
    ap.add_argument('--tag', default='', help='free label appended to the file names, e.g. run2')
    ap.add_argument('--table-only', action='store_true', help='only rebuild the table from the stored JSON')
    ap.add_argument('--resume', action='store_true', help='continue an interrupted run (keeps the instances already in the JSON)')
    ap.add_argument('--check', action='store_true', help='re-check every level where sampling and exact disagree')
    a = ap.parse_args(argv)
    tag = ''
    if a.V and tuple(_num(v) for v in a.V) != tuple(VS):
        VS = tuple(_num(v) for v in a.V); tag += '_V' + '-'.join(str(v) for v in VS)
    if a.sizes and tuple(a.sizes) != tuple(SIZES):
        SIZES = tuple(a.sizes); tag += '_' + 'n' + '-'.join(str(s) for s in SIZES)
    if a.reps and a.reps != REPS:
        REPS = a.reps; tag += f'_reps{REPS}'
    if a.tag:
        tag += '_' + a.tag
    RES_FILE = os.path.join(OUT_DIR, f'res_{NAME}{tag}.json')
    TAB_FILE = os.path.join(OUT_DIR, f'{NAME}{tag}.tex')
    return a



def pieces_paths(part, minlen=1e-9):
    s = set()
    for lo, hi, S in part:
        if hi - lo > minlen:
            s |= set(S)
    return s


def set_at(part, a):
    for lo, hi, S in part:
        if lo - 1e-12 <= a <= hi + 1e-12:
            return S


def instance(n, r):
    return random_dag(n, 0.25, 4, seed=1000 * n + r)


# ------------------------------------------------------------------ experiment
def run(previous=()):
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = list(previous)
    done = {(x['n'], x['rep']) for x in rows}
    for n in SIZES:
        for r in range(REPS):
            if (n, r) in done:
                continue
            net = instance(n, r)
            t = time.time(); part, bps, stt_ = parametric_exact(net, V, K=K); tex = time.time() - t
            ex_paths = pieces_paths(part)
            t = time.time()
            try:
                old, calls, oldU = parametric_sampling(net, V, time_limit=TCAP); told = time.time() - t
                old_paths = pieces_paths(old)
                ag = np.linspace(0, 1, 2001)
                wrong = sum(1 for a in ag if set_at(part, a) != set_at(old, a)) / len(ag)
                row = dict(n=n, rep=r, m=net.m, M=stt_['M'], opt_paths=len(ex_paths), U=stt_['allU'],
                           labels=stt_['labels'], t_exact=tex, M_old=len(old) - 1, t_old=told,
                           old_missed=len(ex_paths - old_paths), old_wrong_frac=wrong,
                           old_level_calls=calls, old_timeout=0)
            except TimeoutError:
                row = dict(n=n, rep=r, m=net.m, M=stt_['M'], opt_paths=len(ex_paths), U=stt_['allU'],
                           labels=stt_['labels'], t_exact=tex, M_old=None, t_old=None, old_missed=None,
                           old_wrong_frac=None, old_level_calls=None, old_timeout=1)
            for k in (11, 101):
                gp = set()
                for a in np.linspace(0, 1, k):
                    gp |= set(p for p, _ in level_label(net, a, V)[0])
                missed_pieces = sum(1 for lo, hi, S in part
                                    if not any(lo - 1e-12 <= a <= hi + 1e-12 for a in np.linspace(0, 1, k)))
                row[f'grid{k}_missed'] = len(ex_paths - gp)
                row[f'grid{k}_missed_pieces'] = missed_pieces
            rows.append(row)
            print(f"n={n:3d} inst={r}  arcs={row['m']:3d}  M={row['M']:2d}  opt.paths={row['opt_paths']:2d}  "
                  f"t_exact={tex:7.2f}s  t_sampling={'timeout' if row['old_timeout'] else format(row['t_old'], '7.2f') + 's'}  "
                  f"missed: sampling={row['old_missed']} grid11={row['grid11_missed']} grid101={row['grid101_missed']}",
                  flush=True)
            with open(RES_FILE, 'w') as f:
                json.dump(rows, f, indent=1, default=float)
    return rows


# ------------------------------------------------------------------ table
def fmt(x, d=1):
    return '--' if x is None else f'{x:.{d}f}'


def table(R):
    g = defaultdict(list)
    for r in R:
        g[r['n']].append(r)
    lines = []
    tot = dict(old=0, inst=0, g11=0, g101=0, to=0)
    for n in sorted(g):
        rs = g[n]; ok = [r for r in rs if not r.get('old_timeout')]
        to = sum(r.get('old_timeout', 0) for r in rs)
        told = fmt(st.mean(r['t_old'] for r in ok), 2) if ok else '--'
        if to: told += f'$^{{\\dagger}}$'
        om = sum(r['old_missed'] for r in ok); oi = sum(1 for r in ok if r['old_missed'] > 0 or r['old_wrong_frac'] > 0)
        tot['old'] += om; tot['inst'] += oi; tot['g11'] += sum(r['grid11_missed'] for r in rs)
        tot['g101'] += sum(r['grid101_missed'] for r in rs); tot['to'] += to
        lines.append(' & '.join([str(n), fmt(st.mean(r['m'] for r in rs), 1), fmt(st.mean(r['M'] for r in rs), 1),
                                 fmt(st.mean(r['opt_paths'] for r in rs), 1), fmt(st.mean(r['t_exact'] for r in rs), 2),
                                 told, f'{om} ({oi})', str(sum(r['grid11_missed'] for r in rs)),
                                 str(sum(r['grid101_missed'] for r in rs))]) + r' \\')
    return r'''\begin{table}[H]
''' + (r'''\color{red}
''' if REVISION_COLOR else '') + r'''\centering
\caption{Exact critical partition versus sampling-based procedures (banded random DAGs, ''' + str(REPS) + r''' instances per size, $V=''' + f'{V:g}' + r'''$)}
\label{tab:sampling}
\footnotesize
\begin{tabular}{rrrrrrrrr}
\hline
 & & & & \multicolumn{2}{c}{Time (s)} & \multicolumn{3}{c}{Optimal paths missed (total)} \\
\cline{5-6}\cline{7-9}
$n$ & arcs & $M$ & opt.\ paths & exact & sampling & sampling (inst.) & 11 levels & 101 levels \\
\hline
''' + '\n'.join(lines) + r'''
\hline
\end{tabular}
\par\smallskip
\parbox{0.95\textwidth}{\footnotesize $M$: breakpoints of the exact partition; opt.\ paths: distinct paths optimal on a set of levels of positive length. ``sampling (inst.)'': optimal paths missed by the sampling-based procedure, and in parentheses the number of instances on which its reported partition differs from the exact one.''' + (r''' $^{\dagger}$Mean over the instances on which the sampling procedure finished within ''' + str(TCAP) + r'''\,s.''' if tot['to'] else '') + r'''}
\end{table}
''', tot


def summary(R):
    """Numbers quoted in the text after the table."""
    ok = [r for r in R if not r.get('old_timeout')]
    print('\nSummary (for the text):')
    print(f"  instances: {len(R)}; sampling missed {sum(r['old_missed'] for r in ok)} optimal paths on "
          f"{sum(1 for r in ok if r['old_missed'] > 0)} instances; wrong partition on "
          f"{sum(1 for r in ok if r['old_missed'] > 0 or r['old_wrong_frac'] > 0)} instances; "
          f"timeouts: {sum(r.get('old_timeout', 0) for r in R)}")
    print(f"  uniform grids missed {sum(r['grid11_missed'] for r in R)} paths (11 levels), "
          f"{sum(r['grid101_missed'] for r in R)} (101 levels)")
    print(f"  exact algorithm: at most {max(r['t_exact'] for r in R):.1f} s per instance; faster than sampling on "
          f"{sum(1 for r in R if r.get('old_timeout') or (r['t_old'] is not None and r['t_exact'] < r['t_old']))} "
          f"of {len(R)} instances")


def check(R):
    """Re-check every level at which the sampling procedure disagrees with the exact partition
    against the labelling solver (Algorithm 1 of the paper), which is exact at a single level."""
    print('\nIndependent check of the disagreements:')
    for r in R:
        if r.get('old_timeout') or not ((r.get('old_missed') or 0) > 0 or (r.get('old_wrong_frac') or 0) > 0):
            continue
        n, idx = r['n'], r.get('rep', [x for x in R if x['n'] == r['n']].index(r))
        net = instance(n, idx)
        part, bps, _ = parametric_exact(net, V, K=K)
        old, calls, U = parametric_sampling(net, V)
        bad = [a for a in np.linspace(0, 1, 2001) if set_at(part, a) != set_at(old, a)]
        agree_ex = agree_old = 0
        chk = bad[:: max(1, len(bad) // 20)]
        for a in chk:
            lv = canon_set(net, [p for p, _ in level_label(net, a, V)[0]], a, V)
            agree_ex += lv == canon_set(net, set_at(part, a), a, V)
            agree_old += lv == canon_set(net, set_at(old, a), a, V)
        missed = pieces_paths(part) - pieces_paths(old)
        mlen = [sum(hi - lo for lo, hi, S in part if p in S) for p in missed]
        print(f"  n={n} inst={idx}: {len(bad)} disagreeing levels, checked {len(chk)}: exact agrees {agree_ex}, "
              f"sampling agrees {agree_old}; missed paths optimal on intervals of length {[round(float(x), 4) for x in mlen]}")


if __name__ == '__main__':
    args = configure()
    V = float(VS[0])
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
    if args.check:
        check(R)
