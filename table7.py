"""
Experiment "Scaling on road-like networks"  ->  Table 7 of the paper (tab:scaling).

Road-like r x r grids (road_grid), source top-left, sink bottom-right, r in {10, 20, 30, 40},
10 instances per size (seed 9000 + 100 r + rep), V in {600, 6000}.
  * level alpha = 0.5: labelling solver (Algorithm 1 of the paper, level_label) and the lattice sweep
    (baseline level solver of Section 5.1, level_lattice) with a time limit of LATTICE_TCAP = 300 s per run;
    a run that exceeds the limit is reported as '--' with a footnote under the table. Once the lattice sweep
    exceeded the limit on every instance of a grid size, larger grids are not attempted (LATTICE_SKIP_LARGER).
    When it finishes, its optimal set is checked to be identical to that of the labelling solver;
  * exact critical partition (Algorithm 2 of the paper, parametric_exact, K = 16).

Usage (in the Code folder):
    python table7.py                              # paper setting -> Result_table7/table7.tex
    python table7.py --table-only                 # only rebuild the table from the stored JSON
    python table7.py --resume                     # continue an interrupted run
    python table7.py --V 6000                     # other demand(s)   -> Result_table7/table7_V....tex
    python table7.py --sizes 10 20 --reps 3   # smaller trial
    python table7.py --V 6000 --tag run2          # repeated trial with a label
Every option that differs from the paper setting is appended to the file names, so trials never
overwrite each other. A new run never deletes results: an existing JSON is first renamed *_backup_<date>.json. All files of this table go to Result_table7/:
    res_table7[_options].json   (raw results)      table7[_options].tex   (LaTeX table)
"""
import os, sys, json, argparse, time, statistics as st
from collections import defaultdict

from level_solver import level_label, level_lattice, canon_set
from parametric import parametric_exact
from generators import road_grid

SIZES = (10, 20, 30, 40)
REPS = 10
VS = (600, 6000)
K = 16
LATTICE_TCAP = 300            # time limit (s) of one lattice-sweep run; a run that exceeds it is reported as '--'
LATTICE_SKIP_LARGER = True    # once the lattice sweep exceeded the limit on EVERY instance of a grid size,
                              # it is not attempted on larger grids (they are reported as exceeding the limit too)
REVISION_COLOR = False        # \color{red} in the table (as in the paper: Table 6 only)
NAME = 'table7'

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
    ap.add_argument('--V', type=float, nargs='+', help='demands V (paper: 600 6000)')
    ap.add_argument('--sizes', type=int, nargs='+', help='grid sizes r (paper: 10 20 30 40)')
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



def pieces_paths(part, minlen=1e-9):
    s = set()
    for lo, hi, S in part:
        if hi - lo > minlen:
            s |= set(S)
    return s


# ------------------------------------------------------------------ experiment
def run(previous=()):
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = list(previous)
    done = {(x['r'], x['rep'], x['V']) for x in rows}
    # grid sizes on which the lattice sweep exceeded the limit on every instance (from earlier rows, e.g. --resume)
    lattice_dead = None
    for rr in sorted({x['r'] for x in rows}):
        xs = [x for x in rows if x['r'] == rr]
        if xs and all(x.get('lattice_timeout') for x in xs):
            lattice_dead = rr if lattice_dead is None else min(lattice_dead, rr)
    for r in SIZES:
        if LATTICE_SKIP_LARGER and lattice_dead is None:
            prev = [x for x in rows if x['r'] < r and x['r'] in SIZES]
            if prev:
                last = max(x['r'] for x in prev)
                xs = [x for x in prev if x['r'] == last]
                if all(x.get('lattice_timeout') for x in xs):
                    lattice_dead = last
        for rep in range(REPS):
            net = road_grid(r, r, seed=9000 + 100 * r + rep)
            for V in VS:
                if (r, rep, V) in done:
                    continue
                t = time.time(); L, nl = level_label(net, 0.5, V); tl = time.time() - t
                tla, cells, lto, lskip = None, None, 0, 0
                if LATTICE_SKIP_LARGER and lattice_dead is not None and r > lattice_dead:
                    lto, lskip = 1, 1            # not attempted: exceeded the limit on every smaller-grid instance
                else:
                    t = time.time()
                    try:
                        La, cells = level_lattice(net, 0.5, V, time_limit=LATTICE_TCAP); tla = time.time() - t
                        assert canon_set(net, [p for p, _ in La], 0.5, V) == canon_set(net, [p for p, _ in L], 0.5, V)
                    except TimeoutError:
                        tla, cells, lto = None, None, 1
                t = time.time(); part, bps, stt_ = parametric_exact(net, V, K=K); tp = time.time() - t
                row = dict(r=r, rep=rep, n=r * r, m=net.m, V=V, lvl_paths=len(L), lvl_labels=nl, t_label=tl,
                           t_lattice=tla, lattice_cells=cells, M=stt_['M'], opt_paths=len(pieces_paths(part)),
                           U=stt_['allU'], labels=stt_['labels'], t_param=tp,
                           lattice_timeout=lto, lattice_skipped=lskip)
                rows.append(row)
                print(f"grid {r}x{r} inst={rep} V={V}: Alg.1 (label) {tl:.3f}s" +
                      (f", lattice sweep {tla:.2f}s ({cells} cells)" if tla is not None else
                       (f", lattice sweep > {LATTICE_TCAP}s" + (" (not attempted)" if lskip else ""))) +
                      f", Alg.2 (partition) {tp:.2f}s (labels={row['labels']}, |U|={row['U']}, M={row['M']}, "
                      f"opt.paths={row['opt_paths']})", flush=True)
                with open(RES_FILE, 'w') as f:
                    json.dump(rows, f, indent=1, default=float)
    return rows


# ------------------------------------------------------------------ table
def fmt(x, d=1):
    return '--' if x is None else f'{x:.{d}f}'


def lattice_note(foot):
    """Footnote for the lattice-sweep columns."""
    note = ''
    if foot.get('timeout'):
        note += r' --: the lattice sweep did not finish within ' + str(LATTICE_TCAP) + r'\,s on any instance.'
    if foot.get('partial'):
        note += r' $^{\dagger}$Mean over the instances on which the lattice sweep finished within ' + str(LATTICE_TCAP) + r'\,s.'
    if foot.get('notrun'):
        note += r' --: lattice sweep not run.'
    return note


def table(R):
    g = defaultdict(list)
    for r in R:
        g[(r['r'], r['V'])].append(r)
    lines = []
    foot = {}
    for (r, V) in sorted(g):
        rs = g[(r, V)]
        fin = [x for x in rs if x['t_lattice'] is not None]
        if not fin:
            tla = cells = '--'
            if any(x.get('lattice_timeout') for x in rs):
                foot['timeout'] = True
            else:
                foot['notrun'] = True
        else:
            tla = fmt(st.mean(x['t_lattice'] for x in fin), 2)
            cells = fmt(st.mean(x['lattice_cells'] for x in fin), 0)
            if len(fin) < len(rs):
                tla += r'$^{\dagger}$'; cells += r'$^{\dagger}$'; foot['partial'] = True
        lines.append(' & '.join([f'{r}$\\times${r}', fmt(st.mean(x['m'] for x in rs), 0), str(V),
                                 fmt(st.mean(x['t_label'] for x in rs), 3), tla, cells,
                                 fmt(st.mean(x['labels'] for x in rs), 0), fmt(st.mean(x['U'] for x in rs), 1),
                                 fmt(st.mean(x['M'] for x in rs), 1), fmt(st.mean(x['opt_paths'] for x in rs), 1),
                                 fmt(st.mean(x['t_param'] for x in rs), 2), fmt(max(x['t_param'] for x in rs), 2)]) + r' \\')
    return r'''\begin{table}[H]
''' + (r'''\color{red}
''' if REVISION_COLOR else '') + r'''\centering
\caption{Scaling on road-like grids (''' + str(REPS) + r''' instances per size and $V$; averages)}
\label{tab:scaling}
\footnotesize
\begin{tabular}{lrr rrr rrrrrr}
\hline
 & & & \multicolumn{3}{c}{Level $\al=0.5$} & \multicolumn{6}{c}{Exact critical partition (Algorithm~\ref{alg:parametric}, $K=16$)} \\
\cline{4-6}\cline{7-12}
Grid & arcs & $V$ & Alg.~\ref{alg:label} (s) & Lattice (s) & cells & labels & $|U|$ & $M$ & opt.\ paths & time (s) & max (s) \\
\hline
''' + '\n'.join(lines) + r'''
\hline
\end{tabular}
\par\smallskip
\parbox{0.95\textwidth}{\footnotesize cells: lattice cells solved by the lattice sweep after removing duplicate arc sets; labels: labels created over all cells; $|U|$: number of distinct candidate paths over all cells.''' + lattice_note(foot) + r'''}
\end{table}
'''


def summary(R):
    """Numbers quoted in the text after the table."""
    print('\nSummary (for the text):')
    for r in SIZES:
        rs = [x for x in R if x['r'] == r and x['t_lattice'] is not None]
        if rs:
            ratio = st.mean(x['t_lattice'] for x in rs) / st.mean(x['t_label'] for x in rs)
            print(f"  {r}x{r}: labelling solver about {ratio:.0f} times faster than the lattice sweep "
                  f"({st.mean(x['lattice_cells'] for x in rs):.0f} cells)")
    for r in SIZES:
        xs = [x for x in R if x['r'] == r]
        to = sum(1 for x in xs if x.get('lattice_timeout'))
        if to:
            sk = sum(1 for x in xs if x.get('lattice_skipped'))
            print(f"  {r}x{r}: lattice sweep exceeded {LATTICE_TCAP} s on {to} of {len(xs)} runs"
                  + (f" ({sk} not attempted)" if sk else ""))
    big = [x for x in R if x['r'] == max(SIZES)]
    if big:
        print(f"  {max(SIZES)}x{max(SIZES)}: exact partition {st.mean(x['t_param'] for x in big):.1f} s on average "
              f"(at most {max(x['t_param'] for x in big):.1f} s)")
    byrv = defaultdict(list)
    for x in R:
        byrv[(x['r'], x['V'])].append(x['M'])
    print(f"  breakpoints: at most {max(st.mean(v) for v in byrv.values()):.1f} on average, "
          f"{max(x['M'] for x in R)} on any instance; distinct optimal routes at most {max(x['opt_paths'] for x in R)}")


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
    
