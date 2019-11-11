# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 10:30:06 2019

@author: Eoin Elliott
"""

import matplotlib.pyplot as plt
import os
import h5py
import numpy as np
import misc as ms
import  conversions as cnv
import time
import ipdb
from nplab.analysis import smoothing as sm
from scipy.interpolate import griddata
plt.rc('font',family='arial', size = 18)

def extract_SERS_intensity(particle, shifts):
    img = particle['image_before']
    plt.figure('particle image')
    plt.imshow(img)
    zscan = particle['z_scan']
    full_wl = zscan.attrs['wavelengths']
    zscan = ms.condenseZscan(zscan)
    zscan,wl = ms.truncate(zscan, full_wl, 466,950)
    zscan = sm.convex_smooth(zscan,20)[0]
    plt.figure('Z scan')
    plt.plot(wl, zscan)
    
    series = particle['SERS']
    places = particle['SERS'].attrs['places'] 
    linwidth = 40
    counter = -1
    S_intensities = []
    AS_intensities = []
    total_intensities = []
    plt.figure()
    for peak in [286, -286]:
        counter+=1
        for index, spec in enumerate(series):
            plt.plot(shifts, spec)
            around_peak_spec, around_peak_shifts = ms.truncate(spec, shifts, peak - linwidth, peak + linwidth)
            
            y = np.array([around_peak_shifts[3], around_peak_shifts[-3]])
            x = np.array([np.mean(around_peak_spec[0:5]), np.mean(around_peak_spec[-5:])])
                    
            line = np.polyfit(y,x,1)
            Sum = np.sum(around_peak_spec - np.polyval(line, around_peak_shifts))
            
            if counter == 0:
                S_intensities.append(Sum)
            elif counter == 1:
                AS_intensities.append(Sum)
    for spec in series: 
        total_intensities.append(np.sum(spec))
        
    
    return S_intensities, AS_intensities, total_intensities, places

def plot_grid(intensities, places, label = 'SERS Map'):
    
    side = int((len(intensities)**0.5))
    places = np.reshape(places, [side, side, 3])
    #xys = np.take(places, [0,1], axis = 2)
    #xys.reshape(xys.size/2, 2)
  
    x = places[0,:,0] # an x axis
    y = places[:,0,1] # a y axis
 
    xx, yy = np.meshgrid(x, y) #x and y coordinates to be evalueated at
    xy_evalled = [[X,Y] for X, Y in zip(np.ravel(xx), np.ravel(yy))]
    
    xs = places[:,:,0].flatten() # the coordinates of the actual measurements
    ys = places[:,:,1].flatten()

    xy = np.transpose(np.append([xs], [ys], axis = 0))
       
    zz = griddata(xy, intensities, xy_evalled,  method = 'linear')    
    x_cent= x- x[len(x)/2]
    y_cent = y- y[len(y)/2]
    
    plt.figure(label)
    plt.pcolormesh(x_cent,y_cent, np.reshape(zz, [side, side]))
    plt.colorbar()
    plt.figure()    
    plt.plot(intensities)
if __name__ == '__main__':
    plt.rc('font',family='arial', size = 18)
    plt.close('all')
    start = time.time()
    os.chdir(r'V:\ee306\2019.11.07 Grid SERS BPT')
    with h5py.File(ms.findH5File(os.getcwd()), mode = 'r') as File:   
        scan = File
        analysis_range = -980, 890 #cm-1
        notch = 220
        shifts = -cnv.wavelength_to_cm(scan['Grid_SERS_BPT_12']['SERS'].attrs['x_axis'][::-1], centre_wl = 785)
        AS_shifts = ms.truncate(shifts, shifts, -np.inf, 0)[0]
        S_shifts = ms.truncate(shifts, shifts, 0, np.inf)[0]
        AS_fit = np.loadtxt(r'C:\Users\Eoin Elliott\Documents\GitHub\NP_ee306\mine\calibration\Lab 5\785nm_anti_Stokes_polynomial.txt')
        S_fit = np.loadtxt(r'C:\Users\Eoin Elliott\Documents\GitHub\NP_ee306\mine\calibration\Lab 5\785nm_Stokes_polynomial.txt')
        calibrated_AS_shifts = np.polyval(AS_fit, AS_shifts)
        calibrated_S_shifts = np.polyval(S_fit, S_shifts)
        calibrated_shifts = np.append(calibrated_AS_shifts, calibrated_S_shifts)
        T_misc_shifts, T_misc = np.loadtxt(r'C:\Users\Eoin Elliott\Documents\GitHub\NP_ee306\mine\calibration\Lab 5\Transmission_function_100x_09_NA_600lmm_785nm_5050bs.txt')
       
       
        
        S, AS, T, places = extract_SERS_intensity(scan['Grid_SERS_BPT_0'], calibrated_shifts)
        ##### particle 5, 12 4-ish, 3-ish 
        plot_grid(S, places, label = 'Stokes, 289')
        plot_grid(AS, places, label = 'Anti-Stokes, 289')
        plot_grid(T, places, label = 'Total intensity')
        
    end = time.time()    
    print 'That took '+str(np.round(end - start))+ 'seconds'
    
    
    
    
    
    
    