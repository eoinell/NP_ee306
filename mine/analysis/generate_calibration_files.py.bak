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
from nplab.analysis import Adaptive_Polynomial as AP
from nplab.analysis import smoothing as sm
from nplab.analysis import Auto_Fit_Raman as AFR
import time
import ipdb
import scipy
from PIL import Image


plt.rc('font',family='arial', size = 18)


#for key in File.keys():
 #   if list(key)[1]==a:
       

def transmission_function():
  
    print r'C:\Users\Eoin Elliott\Documents\GitHub\NP_ee306\mine\calibration\Lab 5' 
    OO_wl = np.transpose(np.loadtxt(r'C:\Users\Eoin Elliott\Documents\GitHub\NP_ee306\mine\calibration\Lab 5\Ocean_Optics_halogen_and_xenon.txt' ))[0]
    OO_halogen = np.transpose(np.loadtxt(r'C:\Users\Eoin Elliott\Documents\GitHub\NP_ee306\mine\calibration\Lab 5\Ocean_Optics_halogen_and_xenon.txt' ))[1]
    OO_xenon = np.transpose(np.loadtxt(r'C:\Users\Eoin Elliott\Documents\GitHub\NP_ee306\mine\calibration\Lab 5\Ocean_Optics_halogen_and_xenon.txt' ))[2]
    xenon_reference = np.transpose(np.loadtxt(r'C:\Users\Eoin Elliott\Documents\GitHub\NP_ee306\mine\calibration\Lab 5\xenon_reference.txt'))[1]
    xenon_wl = np.transpose(np.loadtxt(r'C:\Users\Eoin Elliott\Documents\GitHub\NP_ee306\mine\calibration\Lab 5\xenon_reference.txt' ))[0]
    OO_halogen = sm.convex_smooth(OO_halogen,10, normalise = False)[0]
    OO_xenon = sm.convex_smooth(OO_xenon,10, normalise = False)[0]
    andor_wl = np.transpose(np.loadtxt(r'C:\Users\Eoin Elliott\Documents\GitHub\NP_ee306\mine\calibration\Lab 5\100x_09_NA_600lmm_785nm_5050bs.txt' ))[0][::-1]#reverses the wavelengths
    andor_halogen = np.transpose(np.loadtxt(r'C:\Users\Eoin Elliott\Documents\GitHub\NP_ee306\mine\calibration\Lab 5\100x_09_NA_600lmm_785nm_5050bs.txt' ))[1][::-1]
    
    andor_halogen = sm.convex_smooth(andor_halogen,10, normalise = True)[0]

    
    andor_wl = -cnv.wavelength_to_cm(andor_wl, centre_wl = 785)
    
    S_andor_halogen, S_andor_wl = ms.truncate(andor_halogen, andor_wl, 0, np.inf)
    AS_andor_halogen, AS_andor_wl = ms.truncate(andor_halogen, andor_wl, -np.inf, 0)
    andor_halogen = np.append(AS_andor_halogen, S_andor_halogen)
    
    S_andor_wl = np.polyval(S_fit, S_andor_wl)
    AS_andor_wl = np.polyval(AS_fit, AS_andor_wl)
    andor_cm = np.append(AS_andor_wl, S_andor_wl)
    andor_wl = cnv.cm_to_wavelength(-andor_cm, centre_wl = 785)

    OO_halogen = scipy.interpolate.interp1d(OO_wl, OO_halogen)(andor_wl)
    OO_xenon = scipy.interpolate.interp1d(OO_wl, OO_xenon)(andor_wl)
    xenon_reference = scipy.interpolate.interp1d(xenon_wl, xenon_reference)(andor_wl)
    T = OO_xenon*andor_halogen/OO_halogen*xenon_reference
    
    
    
    plt.figure('Transmission_functions')
    plt.plot(andor_wl, OO_xenon/float(max(OO_xenon)), label = 'OO_xenon')
    plt.plot(andor_wl, andor_halogen/float(max(andor_halogen)), label = 'andor_halogen')
    plt.plot(andor_wl, OO_halogen/float(max(OO_halogen)), label = 'OO_halogen')
    plt.plot(andor_wl, xenon_reference/float(max(xenon_reference)), label = 'xenon_reference')
    plt.plot(andor_wl, T/float(max(T)), label = 'transmission')
    plt.legend()
    
    
    plt.savefig(r'C:\Users\Eoin Elliott\Documents\GitHub\NP_ee306\mine\calibration\Lab 5\Transmission_function_100x_09_NA_600lmm_785nm_5050bs')
    T_tuple = np.append([andor_cm], [T], axis = 0)
    np.savetxt(r'C:\Users\Eoin Elliott\Documents\GitHub\NP_ee306\mine\calibration\Lab 5\Transmission_function_100x_09_NA_600lmm_785nm_5050bs.txt',T_tuple)
    return T/max(T), andor_cm

def IRF_OO():
    OO_xenon = np.transpose(np.loadtxt(r'C:\Users\Eoin Elliott\Documents\GitHub\NP_ee306\mine\calibration\Lab 5\Ocean_Optics_halogen_and_xenon.txt'))[2]
    OO_wl = np.transpose(np.loadtxt(r'C:\Users\Eoin Elliott\Documents\GitHub\NP_ee306\mine\calibration\Lab 5\Ocean_Optics_halogen_and_xenon.txt' ))[0]
    xenon_reference = np.transpose(np.loadtxt(r'C:\Users\Eoin Elliott\Documents\GitHub\NP_ee306\mine\calibration\Lab 5\xenon_reference.txt'))[1]
    xenon_wl = np.transpose(np.loadtxt(r'C:\Users\Eoin Elliott\Documents\GitHub\NP_ee306\mine\calibration\Lab 5\xenon_reference.txt' ))[0]
    
    plt.figure('entire xenon')
    plt.plot(xenon_wl, xenon_reference)
    
    plt.figure('Xenon and halogen')
    plt.plot(OO_wl, np.transpose(np.loadtxt(r'C:\Users\Eoin Elliott\Documents\GitHub\NP_ee306\mine\calibration\Lab 5\Ocean_Optics_halogen_and_xenon.txt'))[1], label = 'Halogen')
    plt.plot(OO_wl, np.transpose(np.loadtxt(r'C:\Users\Eoin Elliott\Documents\GitHub\NP_ee306\mine\calibration\Lab 5\Ocean_Optics_halogen_and_xenon.txt'))[2], label = 'Xenon')
    plt.legend()
    
    xenon_ref = scipy.interpolate.interp1d(xenon_wl, xenon_reference)(OO_wl)
    IRF_OO = (np.true_divide((OO_xenon), (xenon_ref)), OO_wl)
    
    plt.figure('xenon reference and OO_xenon')
    plt.plot(OO_wl, xenon_ref/float(max(xenon_ref)), label = 'xenon_ref')
    plt.plot(OO_wl, OO_xenon/float(max(OO_xenon)), label = 'OO_xenon')
    
    plt.figure('IRF')
    plt.plot(OO_wl, IRF_OO[0]/max(IRF_OO[0]))
    plt.legend()
    
    np.savetxt(r'C:\Users\Eoin Elliott\Documents\GitHub\NP_ee306\mine\calibration\Lab 5\IRF_OO.txt', IRF_OO)
    
    


def calibrate_BPT(counts, wavelengths, notch, analysis_range, Si_counts=None, Si_wl = None):
    BPT_lines = [1278.49,1078.08,1012.33,995.828,827.954,758.232,709.338,655.325,615.569,545.529,478.053,406.404,290.517]    
    Si_line = 520.7
    
    init_shifts = -cnv.wavelength_to_cm(wavelengths, centre_wl = 785.)
    counts, init_shifts = ms.truncate(counts, init_shifts, analysis_range[0], analysis_range[1])
    for line in BPT_lines:
        if line>max(init_shifts):
            BPT_lines.remove(line)
    S_portion, S_shifts = ms.truncate(counts,init_shifts, notch, np.inf)
    AS_portion, AS_shifts = ms.truncate(counts,init_shifts, -np.inf, -notch )
      
    S_bg = AP.Run(S_portion, 3, Auto_Remove = False )
    AS_bg = AP.Run(AS_portion, 3, Auto_Remove = False )

    full_bg = np.append(AS_bg,S_bg)
    notch_len = len(counts) - len(full_bg)
    full_bg = np.insert(full_bg,len(AS_portion),np.zeros(notch_len))
    subbed_counts = counts-full_bg
    
    S_values, S_errors =  AFR.Run(S_portion-S_bg, S_shifts,  Noise_Threshold = 1)
    AS_values, AS_errors = AFR.Run(AS_portion-AS_bg, -AS_shifts,  Noise_Threshold = 1)
   
    plt.figure('BPT calibration plot')  
    plt.plot(init_shifts, counts)
    #plt.plot(S_shifts, S_bg, color = 'saddlebrown')
    plt.fill_between(S_shifts, S_bg, 0,color = 'saddlebrown', alpha = 0.2)
    #plt.plot(AS_shifts, AS_bg, color = 'saddlebrown')
    plt.fill_between(AS_shifts, AS_bg, 0,color = 'saddlebrown', alpha = 0.2)
    plt.ylim(np.min(full_bg[0:100]-100), max(counts)+100)
    plt.fill_between([-notch, notch], 0, 1000000, color = 'olive', alpha = 0.1)
    
    
    S_peak_array = []
    S_peak_height = []
    S_peak_shift = []
    S_peak_width = []
    S_peak_area = []
    if S_values is not None:
        for peak_number in range(len(S_values)/3):
            if S_values[peak_number*3:peak_number*3 +3][2]<15:
                S_peak_array = S_values[peak_number*3:peak_number*3 +3]
                S_peak_height.append( S_peak_array[0])
                S_peak_shift.append( S_peak_array[1])
                S_peak_width.append( S_peak_array[2])
                
            #arrow_start = S_peak_height+ np.float(an.truncate(full_bg, shifts, S_peak_shift, S_peak_shift+10)[0][0]) +arrow_length+100
            #plt.arrow(S_peak_shift, arrow_start, 0, -arrow_length, width = 10, length_includes_head = True, head_width = 20, head_length = arrow_length/6) 
        for peak_number in range(len(S_peak_height)):    
            S_peak_area.append( 4*np.pi*S_peak_height[peak_number]*S_peak_width[peak_number])
            L_bg, L_input = ms.truncate(full_bg, init_shifts, S_peak_shift[peak_number]-5*S_peak_width[peak_number], S_peak_shift[peak_number]+5*S_peak_width[peak_number])
            L_output =  AFR.L(L_input, S_peak_height[peak_number], S_peak_shift[peak_number], S_peak_width[peak_number])        
        
            plt.figure('BPT calibration plot')
            plt.plot(L_input, L_output+L_bg, color = 'k')
            plt.fill_between(L_input, L_output+L_bg, L_bg, color = 'k', alpha = 0.2)
    
    AS_peak_array = []
    AS_peak_height = []
    AS_peak_shift = []
    AS_peak_width = []
    AS_peak_area = []
    if AS_values is not None:
        for peak_number in range(len(AS_values)/3):
            if AS_values[peak_number*3:peak_number*3 +3][2]<15:
                AS_peak_array=AS_values[peak_number*3:peak_number*3+3]
                AS_peak_height.append(AS_peak_array[0])
                AS_peak_shift.append(-AS_peak_array[1])
                AS_peak_width.append(AS_peak_array[2])
                
            
            #arrow_start = S_peak_height+ np.float(an.truncate(full_bg, shifts, S_peak_shift, S_peak_shift+10)[0][0]) +arrow_length+100
            #plt.arrow(S_peak_shift, arrow_start, 0, -arrow_length, width = 10, length_includes_head = True, head_width = 20, head_length = arrow_length/6) 
        for peak_number in range(len(AS_peak_height)):
            AS_peak_area.append(4*np.pi*AS_peak_height[peak_number]*AS_peak_width[peak_number])
            L_bg, L_input = ms.truncate(full_bg, init_shifts, AS_peak_shift[peak_number]-5*AS_peak_width[peak_number], AS_peak_shift[peak_number]+5*AS_peak_width[peak_number])
            L_output =  AFR.L(L_input, AS_peak_height[peak_number], AS_peak_shift[peak_number], AS_peak_width[peak_number])        
            
            plt.figure('BPT calibration plot')
            plt.plot(L_input, L_output+L_bg, color = 'k')
            plt.fill_between(L_input, L_output+L_bg, L_bg, color = 'k', alpha = 0.2)
            plt.savefig('Spectrum for calibration.png')
        if len(S_peak_shift)!=len(AS_peak_shift):
            print 'Peak\'s gone missing!'
       
        
        S_real_peaks = []
        for S_peak in S_peak_shift:
            closest = ms.find_closest(S_peak, np.array(BPT_lines))
            if closest[2]<50:
                S_real_peaks = np.append(S_real_peaks, closest[0])
            else:
                S_peak_shift.remove(S_peak)    
        S_real_peaks = np.append(S_real_peaks, 0)
        S_peak_shift = np.append(S_peak_shift, 0)
        if Si_counts is not None:
            S_wl = ms.get_peak_heights(Si_counts, Si_wl, Si_line, Range = 50, return_wavelength = True,inputcm=True)[2]
            Si_S_shift = -cnv.wavelength_to_cm(S_wl, centre_wl = 785.)
            
            S_real_peaks = np.append(S_real_peaks, 520.7)
            S_peak_shift = np.append(S_peak_shift, Si_S_shift)
        S_fit = np.polyfit(S_peak_shift, S_real_peaks, 3)
        
        
        AS_real_peaks = []
        for AS_peak in AS_peak_shift:
            closest = ms.find_closest(AS_peak, -np.array(BPT_lines))
            if closest[2]<50:
                AS_real_peaks = np.append(AS_real_peaks, closest[0])
            else:
                AS_peak_shift.remove(AS_peak)

        AS_real_peaks = np.append(AS_real_peaks, 0)
        AS_peak_shift = np.append(AS_peak_shift, 0)   
        if Si_counts is not None:
            AS_wl = ms.get_peak_heights(Si_counts, Si_wl, Si_line, Range = 50, return_wavelength = True, inputcm=True)[3]
            Si_AS_shift = cnv.wavelength_to_cm(AS_wl, centre_wl = 785.)
            AS_real_peaks = np.append(AS_real_peaks, -Si_line)
            AS_peak_shift = np.append(AS_peak_shift, Si_AS_shift)
            
        AS_fit = np.polyfit(AS_peak_shift, AS_real_peaks, 3)
        
        

        plt.figure()
        plt.plot(S_peak_shift, S_real_peaks, '+', color='g', markersize=15)
        plt.plot(AS_peak_shift, AS_real_peaks, '+', color='b', markersize=15)
        calibrated_S_shifts = np.polyval(S_fit,ms.truncate(init_shifts, init_shifts, 0, np.inf)[1])
        calibrated_AS_shifts = np.polyval(AS_fit, ms.truncate(init_shifts, init_shifts, -np.inf, 0)[1])
        plt.plot(ms.truncate(init_shifts, init_shifts, 0, np.inf)[1], calibrated_S_shifts, color='g', markersize=10)
        plt.plot(ms.truncate(init_shifts, init_shifts, -np.inf, 0)[1], calibrated_AS_shifts,color='b', markersize=10)
        plt.grid()
        plt.savefig('BPT calibration plot.png')
        
        calibrated_shifts = np.append(calibrated_AS_shifts, calibrated_S_shifts)
        np.savetxt(r'C:\Users\Eoin Elliott\Documents\GitHub\NP_ee306\mine\calibration\Lab 5\785nm_Stokes_polynomial.txt', S_fit)
        np.savetxt(r'C:\Users\Eoin Elliott\Documents\GitHub\NP_ee306\mine\calibration\Lab 5\785nm_anti_Stokes_polynomial.txt', AS_fit)
        return calibrated_shifts, S_fit, AS_fit
if __name__ == '__main__':
    start = time.time()
    #File = h5py.File(ms.findH5File(os.getcwd()), mode = 'r')   
    #scan = File['ParticleScannerScan_0']
    #Si_counts = File['Si_3']['Power_Series'][-1]
    #Si_wl = File['Si_3']['Power_Series'].attrs['wavelengths']  
    #scan_to_be_condensed = scan['Particle_1']['power_series_5']
    #BPT_spec = np.mean(scan_to_be_condensed, axis = 0) 
    #BPT_wl = scan['Particle_1']['power_series_6'].attrs['wavelengths']
    #analysis_range = -np.inf, np.inf#cm-1
    #notch = 222
    #calibrated_shifts, S_fit, AS_fit = calibrate_BPT(BPT_spec, BPT_wl, notch, analysis_range)
    #transmission_function()
    IRF_OO()
    #extract_temperature('BPT_1', T_NPoM = False, plot_intensities = True)
    end = time.time()
    print 'That took '+str(np.round(end - start))+ ' seconds'