# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 10:04:56 2018

@author: 顺叔公
"""

from optizimation import simann,mypso
def banana(x):
    x1 = x[0]
    x2 = x[1]
    return x1**4 - 2*x2*x1**2 + x2**2 + x1**2 - 2*x1 + 5

def con(x):
    x1 = x[0]
    x2 = x[1]
    return [-(x1 + 0.25)**2 + 0.75*x2]

lb = [-3, -1]
ub = [2, 6]

xopt, fopt = simann(banana, lb, ub, f_cons=con)
xx,yy=mypso(banana, lb, ub, f_cons=con)
