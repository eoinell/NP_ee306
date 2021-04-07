# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:50:45 2019

@author: Eoin Elliott
"""
from __future__ import division
from past.utils import old_div
import numpy as np
import scipy.constants
      
    
def wl_to_omega(wavelengths, centre_wl = 633.): # wl in nm
    omega = 2*np.pi*scipy.constants.c*(1./(wavelengths*1e-9) - 1./(centre_wl*1e-9))
    return omega # Stokes omegas are negative

def omega_to_wl(omega, centre_wl = 633):
    wavelengths = old_div(1e9,(old_div(omega,(2.*np.pi*scipy.constants.c)) + 1./(centre_wl*1e-9)))
    return wavelengths # consistant signs
def OD_to_power(P0,OD):
    return P0*10.**(-OD)

def wl_to_cm(wavelengths, centre_wl = 633.):
    return (1./(centre_wl*1e-9) - 1./(wavelengths*1e-9))/100
    # Stokes cm are negative
    
def cm_to_wl(cm, centre_wl = 633):
    return 1e9/(cm*100 +1./(centre_wl*1e-9))

def simple_wl_to_omega(wavelengths):
    omega = old_div(2*np.pi*scipy.constants.c*1./(wavelengths*1e-9),1.0e-9)
    
    return omega

def simple_omega_to_wl(omega):
    wavelength =  old_div(1e9*(2.*np.pi*scipy.constants.c),omega)
    return wavelength

def cm_to_omega(cm):
    return 2*np.pi*scipy.constants.c*100.*cm