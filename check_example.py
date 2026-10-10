"""Verifies Example 1 of the paper.  Usage: python check_example.py"""
from level_solver import *
from parametric import *
net = Net(4, 0, 3)
net.add(0,1,(1,2,7),(2,5,6))   # (1,2)  arc0
net.add(1,3,(1,3,8),(3,6,12))  # (2,4)  arc1
net.add(0,2,(2,4,5),(8,10,11)) # (1,3)  arc2
net.add(2,3,(3,4,5),(7,8,9))   # (3,4)  arc3
net.add(1,2,(1,2,3),(5,7,9))   # (2,3)  arc4
V=24
A=(0,1); B=(2,3); C=(0,4,3)
for a in (0,0.5,1):
    for nm,p in (('A',A),('B',B),('C',C)):
        print(a,nm,[round(x,4) for x in path_T(net,p,a,V)])
# table formulas
import numpy as np
for a in np.linspace(0,1,5):
    TRA=(-30*a*a+25*a+54)/(3*a+2); TRB=(-2*a*a-4*a+94)/(a+7); TRC=(-21*a*a+31*a+54)/(3*a+2)
    TCA=(21*a**3-163*a**2+236*a+396)/(2*(3*a+2)*(6-a)); TCB=(-a**3-13*a**2+93*a+1329)/(2*(9-a)*(a+7))
    TCC=(6*a**3-62*a**2+160*a+216)/((3*a+2)*(6-a))
    print(round(a,2),'table',[round(x,4) for x in (TRA,TRB,TRC,TCA,TCB,TCC)],
      'code',[round(path_T(net,p,a,V)[k],4) for k in (1,) for p in (A,B,C)]+[round(path_T(net,p,a,V)[2],4) for p in (A,B,C)])
print(np.roots([24,169,45,-190]))
print(np.roots([12,-91,-630,3769,3360,-4500]))
part,bps,st=parametric_exact(net,V,K=1)
for lo,hi,S in part: print(round(lo,6),round(hi,6),[{A:'A',B:'B',C:'C'}[p] for p in S])
print(st)
