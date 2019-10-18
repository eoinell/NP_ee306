# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 11:10:25 2019

@author: Eoin Elliott
"""
import matplotlib.pyplot as plt
import re
import os
import h5py
import numpy as np
import scipy 
from scipy.optimize import curve_fit
from scipy.constants import pi, c, k,hbar
from scipy.optimize import curve_fit
from lmfit import Minimizer, Parameters#, report_fit
plt.rc('font',family='arial')

def wavelength_to_omega(wavelengths, centre_wl = 633): # wl in nm
    omega = 2*np.pi*scipy.constants.c*(1/(wavelengths*1e-9) - 1/(centre_wl*1e-9))
    return omega
def exponential(omega,A,T): # work in omega, simple exponential
    
        #return A*(np.exp((scipy.constants.hbar/scipy.constants.k)*omega/T) -1)**-1 +bg # x is omega
        return A*np.exp((-scipy.constants.hbar/scipy.constants.k)*omega/T)
def exponential2(omega,A,T): # work in omega, simple exponential
    
        return A*(np.exp((scipy.constants.hbar/scipy.constants.k)*omega/T) -1)**-1  # x is omega
        r#eturn A*np.exp((-scipy.constants.hbar/scipy.constants.k)*omega/T)   

omega_array = np.linspace(wavelength_to_omega(615), wavelength_to_omega(590), num = 50)
plt.plot(omega_array, exponential(omega_array,1000,300))
plt.plot(omega_array, exponential2(omega_array,1000,300))