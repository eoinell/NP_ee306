# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 16:56:31 2021

@author: Eoin
"""
from . import conversions as cnv
import numpy as np
from scipy.constants import hbar, k

def stokesratio(Stokes_counts, antiStokes_counts, shift, laser_wavelength = 785.):#returns T, shift is in cm-1
    #omega is the raman shift in omega. 
    omega = cnv.cm_to_omega(shift)
    
    omega_AS = omega +cnv.simple_wavelength_to_omega(laser_wavelength)
    omega_S = cnv.simple_wavelength_to_omega(laser_wavelength)-omega
    logarg = ((np.array(Stokes_counts)/np.array(antiStokes_counts)))*((omega_AS/omega_S))**4
    T = (hbar*omega/(k*np.log(logarg)))
    return T 