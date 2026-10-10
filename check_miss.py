"""Re-checks every level at which the sampling procedure disagrees with the exact partition
(Table 4 experiment), against the labelling solver (Algorithm 1), which is exact at a single level.
Usage (in the Code folder): python check_miss.py      (reads Result_table4/res_table4.json)"""
import os, json
import numpy as np
from level_solver import level_label, canon_set
from parametric import parametric_exact, parametric_sampling
from table4 import pieces_paths, set_at, instance, V, K

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Result_table4', 'res_table4.json')

if __name__ == '__main__':
    if not os.path.exists(RES):
        raise SystemExit(f'{RES} not found -- run python table4.py first')
    R = json.load(open(RES))
    for r in R:
        if r.get('old_timeout'):
            continue
        if (r.get('old_missed') or 0) > 0 or (r.get('old_wrong_frac') or 0) > 0:
            n = r['n']; idx = r['rep'] if 'rep' in r else [x for x in R if x['n'] == n].index(r)
            net = instance(n, idx)
            part, bps, st = parametric_exact(net, V, K=K)
            old, calls, U = parametric_sampling(net, V)
            bad = [a for a in np.linspace(0, 1, 2001) if set_at(part, a) != set_at(old, a)]
            agree_ex = agree_old = 0
            for a in bad[:: max(1, len(bad) // 20)]:
                lv = canon_set(net, [p for p, _ in level_label(net, a, V)[0]], a, V)
                agree_ex += lv == canon_set(net, set_at(part, a), a, V)
                agree_old += lv == canon_set(net, set_at(old, a), a, V)
            missed = pieces_paths(part) - pieces_paths(old)
            mlen = [round(float(sum(hi - lo for lo, hi, S in part if p in S)), 4) for p in missed]
            print('n', n, 'inst', idx, 'bad pts', len(bad), 'checked: exact agrees', agree_ex, 'old agrees', agree_old,
                  'missed path optimal measure', mlen, 'U_old', len(U))
