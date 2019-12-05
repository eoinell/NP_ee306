# -*- coding: utf-8 -*-
"""
This is a very simple script that pops up a data browser for one file.
"""
import nplab
import nplab.datafile
import nplab.ui.hdf5_browser as browser
from nplab.utils.gui import get_qt_app
import matplotlib.pyplot as plt
import re
import os
import h5py
import numpy as np
from scipy import interpolate
plt.rc('font',family='arial')
#import PyQtGraph as pg


def smooth_spectra(counts_array, wavelength_array, points=9):
    i = 0
    s = 0
    
    smoothed_counts = np.zeros(len(counts_array)//points)
    smoothed_wavelengths = np.zeros(len(counts_array)//points)
    for s in range(len(counts_array)//points):
       
        
        smoothed_counts[s] = np.mean(counts_array[i:(i+points)])
        if points%2 == 0:
            smoothed_wavelengths[s] = wavelength_array[i+(points/2)]
        else:
            smoothed_wavelengths[s] = wavelength_array[i+(points/2) -(1/2)]  
        i += points
        s+=1
    return smoothed_counts, smoothed_wavelengths  


def deg_to_mT(deg):
    calfilepath = 'R:\ee306\Python\magcal.txt'
    magcal = np.loadtxt(fname = calfilepath)
    degs = magcal[:,0]
    fields = magcal[:,1]
    tck = interpolate.splrep(degs, fields)
    mT = interpolate.splev(deg,tck)
    return float(mT)
    
    
    
    

def split_spectrum(counts_array, wavelength_array, cutoff, lower_cutoff):
    i = -1 
    for wl in wavelength_array:
        i+=1
        #this just splits the array to the relevant area
        if wl>cutoff:
             break
    if lower_cutoff ==0:
        s = 0
    else:
            
        
        s = -1
        for wl in wavelength_array:
            s+=1
            #this just splits the array to the relevant area
            if wl>=lower_cutoff:
                break    
     
    wl_cut = wavelength_array[s:i] 
    spec_cut = counts_array[s:i]
    return spec_cut, wl_cut
colors = plt.cm.plasma_r(np.linspace(0,1,25))
if __name__ == "__main__":
    for filename in os.listdir('.'):
        
        if re.match('\d\d\d\d-[01]\d-[0123]\d.h5', filename):#Finds first instance of file with name in "yyyy-mm-dd.h5" format
            break
    #nplab.datafile.set_current("2019-02-11.h5", mode="r")
    
    #app = get_qt_app()
    #v = browser.HDF5ItemViewer()
    #v.show()
    df = h5py.File(filename, mode = "r")
    oe = df['OceanOpticsSpectrometer']
    spectra = np.zeros(25, dtype = object)
    
    smoothed_counts = np.zeros(25, dtype = object)
    wavelengths = oe['deg0_0'].attrs['wavelengths']
    bg = oe['deg0_0'].attrs['background']
    ref = oe['deg0_0'].attrs['reference']
    #smoothed_wavelengths = smooth_spectra(ref, wavelengths)[1]
    
    for i in range(len(oe)):
        deg = i*15 
        name = 'deg'+str(deg) +'_0'
        if deg ==195:
            spectra[i] = oe['deg180_0']    
        
        else:
            spectra[i]=oe[name]
        
        counts = (spectra[i]-bg)/(ref-bg)
        cnts, wl = split_spectrum(counts, wavelengths, 1000, 500)
        smoothed_counts,smoothed_wavelengths = smooth_spectra(cnts, wl)
        plt.figure(1)
        plt.plot(wl,cnts, label = str(deg), color = colors[i], alpha = 0.4)
        
        
        plt.figure(2)
        plt.plot(smoothed_wavelengths, smoothed_counts, label = str(deg), color = colors[i], alpha = 0.4)
        
        
    
    
    
    plt.figure(1)
    
    plt.ylabel('Intensity', fontname = 'arial', fontsize = '16')
    plt.xlabel('wavelenghts (nm)')
   # plt.legend()
    plt.tick_params(direction = 'in')
    #plt.ylim([0, 0.05])
    
    plt.figure(2)
    plt.ylabel('Intensity', fontname = 'arial', fontsize = '16')
    plt.xlabel('wavelenghts (nm)')
   # plt.legend()
    plt.tick_params(direction = 'in')
    #plt.ylim([0, 0.05])
    #plt.legend()
    plt.tick_params(direction = 'in')
    
    
    
    
    
