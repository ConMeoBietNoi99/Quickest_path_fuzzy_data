"""Detailed summary of the experiment of Table 4 (tab:sampling) (from code/summarize.py).
Usage (in the Code folder): python summarize.py"""
import json, statistics as st, sys
from collections import defaultdict
def E2(files):
    R=[]
    for f in files:
        try: R+=json.load(open(f))
        except FileNotFoundError: pass
    g=defaultdict(list)
    for r in R: g[r['n']].append(r)
    print('n reps m M opt U t_exact t_old(med) old_missed old_wrongfrac% timeouts grid11_missed grid11_pieces grid101_missed grid101_pieces')
    for n in sorted(g):
        rs=g[n]; ok=[r for r in rs if not r.get('old_timeout')]
        f=lambda k,rs=rs: st.mean(r[k] for r in rs)
        print(n,len(rs),round(f('m'),1),round(f('M'),1),round(f('opt_paths'),1),round(f('U'),1),round(f('t_exact'),2),
              round(st.mean(r['t_old'] for r in ok),2) if ok else '-', round(st.median(r['t_old'] for r in ok),2) if ok else '-',
              sum(r['old_missed'] for r in ok), round(100*st.mean(r['old_wrong_frac'] for r in ok),2) if ok else '-',
              sum(r.get('old_timeout',0) for r in rs),
              sum(r['grid11_missed'] for r in rs), sum(r['grid11_missed_pieces'] for r in rs),
              sum(r['grid101_missed'] for r in rs), sum(r['grid101_missed_pieces'] for r in rs),
              'inst_with_old_miss', sum(1 for r in ok if r['old_missed']>0), 'inst_wrong>0', sum(1 for r in ok if r['old_wrong_frac']>0))
import os
_D = os.path.dirname(os.path.abspath(__file__))
E2([os.path.join(_D, 'Result_table4', 'res_table4.json')])      # results of table4.py (n = 20 ... 200)
