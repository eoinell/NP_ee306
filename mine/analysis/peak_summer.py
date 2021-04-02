# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 13:44:51 2020

@author: Eoin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from mine.analysis.misc import split

PEAKS = [545.529,478.053,406.404,290.517]
WIDTHS = [70]*len(PEAKS)

class Summer:
    def __init__(self, spec, shifts, **kwargs):
        self.spec = spec
        self.shifts = shifts
        self.smoothed = gaussian_filter(self.spec, 4)
        self.stokes_sign = (-1, 1)[np.sum(shifts < 0) < np.sum(shifts > 0)]
            
        self.centres = [self.stokes_sign*p for p in PEAKS]
        self.widths = WIDTHS
    def Run(self, **kwargs):
        
        self._Run(**kwargs)
   
    def _Run(self, plot=False, **kwargs):
        self.peaks = []
        for centre, width in zip(self.centres, self.widths):
            limits = [centre - width / 2,
                          centre + width / 2]
            shifts, region = split(self.shifts, self.spec, *limits)
            smoothed = split(self.shifts, self.smoothed, *limits)[1]
            m = (smoothed[-1] - smoothed[0])/(shifts[-1] - shifts[0])
            line = (shifts-shifts[0])*m + smoothed[0]
            

            self.peaks.append([(region - line).sum(), centre, width])
            if plot:
                plt.plot(shifts, line, 'k--')
            
                plt.fill_between(shifts, line, region)
                plt.plot(shifts[0], smoothed[0], 'ko')
                plt.plot(shifts[-1], smoothed[-1], 'ko')
                # import pdb
                # pdb.set_trace()
       
    def plot_result(self):
        plt.figure()
        plt.plot(self.shifts, self.spec)
        self._Run(plot=True)
            
def summer(shifts, spec, centres, widths, plot=False):
    sums = []
    smoothed = gaussian_filter(spec, 4)
    for centre, width in zip(centres, widths):
        limits = [centre - width / 2,
                      centre + width / 2]
        region = split(shifts, spec, *limits)[1]
        split_shifts, split_smoothed = split(shifts, smoothed, *limits)
        m = (split_smoothed[-1] - split_smoothed[0])/(split_shifts[-1] - split_shifts[0])
        line = (split_shifts-split_shifts[0])*m + split_smoothed[0]
        sums.append((region - line).sum())  
        if plot:
            plt.plot(split_shifts, line, 'k--')
            
            plt.fill_between(split_shifts, line, region)
            plt.plot(split_shifts[0], split_smoothed[0], 'ko')
            plt.plot(split_shifts[-1], split_smoothed[-1], 'ko')
    return sums
                
