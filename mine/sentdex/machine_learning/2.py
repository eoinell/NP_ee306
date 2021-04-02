# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 16:13:05 2020

@author: Eoin
making a linear regression algorithm 
"""
from statistics import mean
import numpy as np

xs = np.array([1,2,3,4,5], dtype=np.float64)
ys = np.array([5,4,6,5,6], dtype=np.float64)

def best_fit_slope(xs,ys):
    m = ( ((mean(xs)*mean(ys)) - mean(xs*ys)) /
           (mean(xs)**2))
    return m

m = best_fit_slope(xs,ys)