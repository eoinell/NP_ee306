# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 10:30:06 2019

@author: Eoin Elliott
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from builtins import zip
from builtins import str
from builtins import range
from past.utils import old_div
import matplotlib.pyplot as plt
import os
import h5py
import numpy as np
from . import misc as ms
from . import  conversions as cnv
from nplab.analysis import smoothing as sm
import time
import ipdb
import scipy

from .peaks_and_bg_fitting import fullfit 

plt.rc('font',family='arial', size = 18)
          
def L(x,H,C,W): # height centre width
    	"""
    	Defines a lorentzian
    	"""
    	return old_div(H,(1.+((old_div((x-C),W))**2)))            

def fancy_log(x):
    x = np.array(x)
    return np.log(((x+1)**-1)**-1)

def display_spectra_and_image(particle):
     
    zscan_before = particle['z_scan_before']
    full_wl = zscan_before.attrs['wavelengths']
    zscan_before = ms.condenseZscan(zscan_before)
    zscan_before,wl = ms.truncate(zscan_before, full_wl, 466,950)
    zscan_before = sm.convex_smooth(zscan_before,20)[0]
    
    zscan_after = particle['z_scan_after']
    zscan_after = ms.condenseZscan(zscan_after)
    zscan_after = ms.truncate(zscan_after,full_wl, 466,950)[0]
    zscan_after = sm.convex_smooth(zscan_after, 20)[0]
    
    plt.figure('Z scan')
    plt.plot(wl, zscan_before, label = 'before')
    plt.plot(wl, zscan_after, label = 'after')
    plt.fill_between([cnv.cm_to_wavelength(-analysis_range[0], centre_wl = 785), cnv.cm_to_wavelength(-analysis_range[1], centre_wl = 785)], 0, 1000000, color = 'k', alpha = 0.1)
    plt.axvline(x=785.,color = 'r', ls = '--')
    plt.legend()
    plt.ylim([0,1.2*max(zscan_before)])
    img = particle['image_before']
    plt.figure('particle image')
    plt.imshow(img)
    #img.show()
    return zscan_before, zscan_after, wl         

def condense_power_series(particle, input_shifts, analysis_range):
    powers = particle['power_series_0'].attrs['powers'] # the target powers
    measured_powers = []
    condensed_power_series = [[] for power in powers]
    for index, power in enumerate(powers):        
        name = 'power_series_%d' % index
        kinscan = particle[name]
        measured_powers.append(kinscan.attrs['measured_power'])
        kinscan_truncd = [ms.truncate(spec, input_shifts, *analysis_range)[0] for spec in kinscan]    
        
        # I think here is where the filtering for transient behaviour will take place

        condensed_power_series[index] = np.mean(kinscan_truncd, axis = 0) 
    return condensed_power_series, powers, measured_powers


def fit(particle, input_shifts, analysis_range, T_misc = None, T_misc_shifts = None, T_NPoM = True, plot = False, plot_log = False):#shift in cm-1, previously calibrated
    print('analysing '+str(particle))
    power_series, powers, measured_powers = condense_power_series(particle, input_shifts, analysis_range) # turn the kinetic power series a single spectrum for each power
    shifts = ms.truncate(input_shifts, input_shifts, *analysis_range)[0]
    
    #---generate the normalisation factor 
    normfac = np.ones(len(shifts))
    if T_misc is not None:
        T_misc = scipy.interpolate.interp1d(T_misc_shifts, T_misc)(shifts)
        normfac *=T_misc
    if T_NPoM ==True:
        df_spec, dump, df_wl = display_spectra_and_image(particle)
        T_NPoM = np.sqrt(scipy.interpolate.interp1d(df_wl,df_spec)(cnv.cm_to_wavelength(-shifts, centre_wl = 785)))
        #T_NPoM = scipy.interpolate.interp1d(df_wl,df_spec)(cnv.cm_to_wavelength(-shifts, centre_wl = 785))
        #T_NPoM = (scipy.interpolate.interp1d(df_wl,df_spec)(cnv.cm_to_wavelength(-shifts, centre_wl = 785)))**2
        normfac*=T_NPoM
    normfac/=float(max(normfac))
    S_normfac = ms.truncate(normfac,shifts, notch, np.inf)[0]
    AS_normfac = ms.truncate(normfac,shifts, -np.inf, -notch )[0]
    color_fac = 0.5/max(powers) # used for choosing plotting colours
    previous_S_fit = None # starts out with no initial fit
    previous_AS_fit = None
   
    
    #--- initialise the parameters to be extracted
    n = len(powers)
    Speaks = [0]*n # Stokes peaks
    asymmSpeaks = [0]*n
    ASpeaks = [0]*n # Anti-Stokes peaks
    asymmASpeaks = [0]*n #peaks optimised asymetrically
    bgs = [0]*n # fitted backgrounds
    signals = [0]*n # spectra - backgrounds
    S_bg_ps = [0]*n # the polynomial coefficients of the Stokes background
    AS_bg_ps = [0]*n # the polynomial coefficients of the AS background. If an exponential is used, these are prefactor, Temperature and const. background
    # convert the powers into 2 arrays, up and down. may need another line when max power isn't repeated
    if plot == True: maxes = []
    if plot_log == True: logmaxes = []
    powers_flipped = np.append(powers[:old_div(len(powers),2) +1], powers[old_div(len(powers),2) +1:][::-1])
    indices = list(range(len(powers)))
    indices_flipped = np.append(indices[:old_div(len(indices),2) +1], indices[old_div(len(indices),2) +1:][::-1])
    # done so the lowest powers fits are done first
    counter = -1
    for index, power in zip(indices_flipped, powers_flipped):
        counter+=1
        if counter == old_div(len(powers),2) + 1:  
            previous_S_fit = None 
            previous_AS_fit = None
        raw_counts = power_series[index]
        print('peaks ' + str(old_div((counter)*100,(len(powers)))) +'% fitted')
        #---split the spectrum into Stokes and anti-Stokes
        S_portion, S_shifts = ms.truncate(raw_counts,shifts, notch, np.inf)
        AS_portion, AS_shifts = ms.truncate(raw_counts,shifts, -np.inf, -notch )
        
        #---fit the spectra
        kwargs = {'min_peak_spacing' : 6, 'noise_factor' : 0.01, 'maxwidth' : 40, 'minwidth' : 6}
        Sfit = fullfit(S_portion, S_shifts, order = 11, transmission = S_normfac) 
        Sfit.Run(initial_fit = previous_S_fit, **kwargs)
        S_bg = Sfit.bg
        S_signal = Sfit.signal
        S_peaks = Sfit.peaks_stack
        S_bg_p = Sfit.bg_p.tolist()
        S_bg_ps[index] = S_bg_p
        Speaks[index] = S_peaks
        asymmSpeaks[index] = Sfit.asymm_peaks_stack
        
        use_exponential = False
        ASfit = fullfit(AS_portion, AS_shifts,  order = 11, use_exponential = use_exponential, transmission = AS_normfac)
        ASfit.Run(initial_fit = previous_AS_fit, **kwargs)
        AS_bg = ASfit.bg
        AS_signal = ASfit.signal
        AS_peaks = ASfit.peaks_stack
        asymmASpeaks[index] = ASfit.asymm_peaks_stack
        ASpeaks[index] = AS_peaks
        AS_bg_p = ASfit.bg_p.tolist()
        AS_bg_ps[index] = AS_bg_p
        
        full_bg = np.append(AS_bg, S_bg)
        full_signal = np.append(AS_signal, S_signal)
        notch_len = len(raw_counts) - len(full_bg)
        full_bg = np.insert(full_bg,len(AS_portion),np.zeros(notch_len))
        full_signal = np.insert(full_signal,len(AS_portion),np.zeros(notch_len))
        bgs[index] = full_bg
        signals[index] = full_signal
        #---set up next fitting
        previous_S_fit = Sfit.peaks
        previous_AS_fit = ASfit.peaks
        
        
        #---plot the fits
        if plot == True:
            plt.figure('full fits')  
            plt.plot(shifts, raw_counts, color = plt.cm.plasma_r(power*color_fac), zorder = 1)
            plt.fill_between(S_shifts, S_bg, 0,color = 'saddlebrown', alpha = 0.1, linestyle = 'None')
            plt.fill_between(AS_shifts, AS_bg, 0,color = 'saddlebrown', alpha = 0.1, linestyle = 'None')
            allpeaks = np.append(Sfit.asymm_peaks_stack, ASfit.asymm_peaks_stack, axis = 0)
            plt.plot(shifts, full_bg+ Sfit.asymm_multi_L(shifts, *allpeaks)*normfac, '--', color = 'k', alpha = 0.5, zorder = 1)
            maxes.append(max(raw_counts))
            if counter == len(powers)-1: 
                plt.fill_between([-notch, notch], 0, 1000000, color = 'w', alpha = 1, zorder = 2, linestyle = 'None')
                plt.fill_between([-notch, notch], 0, 1000000, alpha = 0, zorder = 3, linestyle = 'None', hatch = '//')
                plt.ylim((np.min(full_bg[0:100]*0.9), max(maxes)*1.1))
        
        if plot_log == True and use_exponential == True:
            plt.figure('log fits')  
            fullcounts = old_div((raw_counts - AS_bg_p[-1]),normfac)
            plt.plot(shifts, fancy_log(fullcounts), color = plt.cm.plasma_r(power*color_fac), zorder = 1)
            plt.fill_between(S_shifts, fancy_log(S_bg - AS_bg_p[-1]), 0, color = 'saddlebrown', alpha = 0.1, linestyle = 'None')
            plt.fill_between(AS_shifts, fancy_log(old_div((AS_bg - AS_bg_p[-1]),AS_normfac)), 0, color = 'saddlebrown', alpha = 0.1, linestyle = 'None')
            allpeaks = np.append(Sfit.peaks, ASfit.peaks)
            plt.plot(shifts, fancy_log(full_bg - AS_bg_p[-1] + Sfit.asymm_multi_L(shifts, *allpeaks)), '--', color = 'k', alpha = 0.5, zorder = 1)
            logmaxes.append(max(fullcounts))
            if counter == len(powers)-1: 
                plt.fill_between([-notch, notch], -100, 100, color = 'w', alpha = 1, zorder = 2, linestyle = 'None')
                plt.fill_between([-notch, notch], -100, 100, alpha = 0, zorder = 3, linestyle = 'None', hatch = '//')
                plt.ylim(1, fancy_log(max(maxes))*1.1)
    
    parameter_tuple = (asymmSpeaks, asymmASpeaks, bgs,  AS_bg_ps, signals, powers, measured_powers, shifts) 
    return parameter_tuple
       
def sort_fits(signals, shifts, S_peaks_stacks, AS_peaks_stacks, AS_bg_ps, powers, bgs):
    #---sort the data by peak position, then Stokes/anti-Stokes/powers, then peak parameters (height, centre, width, area), then powers
    peak_labels = {}
    temperature = []
    amplitude = []
    for index, power in enumerate(powers):
        temperature.append(AS_bg_ps[index][1]) 
        amplitude.append(AS_bg_ps[index][0]) 
        for peak in S_peaks_stacks[index]: 
            matching_antiStokes_peak_index, stokes_residual = ms.find_closest(-peak[1], np.transpose(AS_peaks_stacks[index])[1])[1:]
            if stokes_residual<20:
                try:
                    position_label, dump, label_residual = ms.find_closest(peak[1], list(peak_labels.keys())) 
                except:
                    position_label = np.around(peak[1], decimals = 0)
                    label_residual = 0
                    empty_position = {position_label : [[[],[],[],[],[],[],[[],[]]],
                                                        [[],[],[],[],[],[],[[],[]]],
                                                        []]} # stokes, anti-stokes, powers
                    peak_labels.update(empty_position)
                if label_residual>10:
                    position_label = np.around(peak[1], decimals = 0)
                    
                    label_residual = 0
                    empty_position = {position_label : [[[],[],[],[],[],[],[[],[]]],
                                                        [[],[],[],[],[],[],[[],[]]],
                                                        []]}
                    peak_labels.update(empty_position)
                
                peak_labels[position_label][0][0].append(peak[0])
                peak_labels[position_label][0][1].append(peak[1])
                peak_labels[position_label][0][2].append(peak[2])
                peak_labels[position_label][0][3].append(peak[0]*peak[1])
                peak_labels[position_label][0][4].append(np.sum(ms.truncate(signals[index], shifts, peak[1] - peak[2], peak[1] + peak[2])[0]))
                linwidth = 40
                around_peak_spec, around_peak_shifts = ms.truncate(signals[index]+bgs[index], shifts, peak[1] - linwidth, peak[1] + linwidth)
                y = np.array([around_peak_shifts[3], around_peak_shifts[-3]])
                x = np.array([np.mean(around_peak_spec[0:5]), np.mean(around_peak_spec[-5:])])
                
                line = np.polyfit(y,x,1)
                Sum = np.sum(around_peak_spec - np.polyval(line, around_peak_shifts))
                peak_labels[position_label][0][5].append(Sum)
                peak_labels[position_label][0][6][0].append(around_peak_shifts)
                peak_labels[position_label][0][6][1].append(np.polyval(line, around_peak_shifts))
                
                aspeak = AS_peaks_stacks[index][matching_antiStokes_peak_index]
                peak_labels[position_label][1][0].append(aspeak[0])
                peak_labels[position_label][1][1].append(aspeak[1])
                peak_labels[position_label][1][2].append(aspeak[2])
                peak_labels[position_label][1][3].append(aspeak[0]*peak[1])
                peak_labels[position_label][1][4].append(np.sum(ms.truncate(signals[index], shifts, aspeak[1] - aspeak[2], aspeak[1] + aspeak[2])[0]))
                
                around_peak_spec, around_peak_shifts = ms.truncate(signals[index]+bgs[index], shifts, aspeak[1] - linwidth, aspeak[1] + linwidth)
                y = [around_peak_shifts[3], around_peak_shifts[-3]]
                x = [np.mean(around_peak_spec[0:5]), np.mean(around_peak_spec[-5:])]
                line = np.polyfit(y,x,1)
                Sum = np.sum(around_peak_spec - np.polyval(line, around_peak_shifts))
                peak_labels[position_label][1][5].append(Sum)
                peak_labels[position_label][1][6][0].append(around_peak_shifts)
                peak_labels[position_label][1][6][1].append(np.polyval(line, around_peak_shifts))
                
                peak_labels[position_label][2].append(power)
    
    return peak_labels, temperature, amplitude # the anti-stokes exponential temperature

def plot_fits(signals, shifts, sorted_fits, powers):
    colorfac = 0.5/max(powers)
    
    plt.figure('fitted spectra')
    
    for index, signal in enumerate(signals):
        plt.plot(shifts, signal, color = plt.cm.plasma_r(powers[index]*colorfac), alpha = 1, zorder = 1)
        biggest_shift = max(sorted_fits.keys())
    for peak_position, parameters in list(sorted_fits.items()):
        for index, power in enumerate(parameters[2]):
            Svalues = []
            ASvalues = []
            for parameter in parameters[0][0:3]:
                Svalues.append(parameter[index])
            for parameter in parameters[1][0:3]:
                ASvalues.append(parameter[index])
            shift_range = ms.truncate(shifts, shifts,Svalues[1]-Svalues[2], Svalues[1]+Svalues[2])[0]
            plt.fill_between(shift_range, np.zeros(len(shift_range)), L(shift_range, *Svalues), color = plt.cm.winter(peak_position/float(biggest_shift)), linestyle = 'None', alpha = 0.1, zorder = 1)# hatch = '/')
            shift_range = ms.truncate(shifts, shifts,ASvalues[1]-ASvalues[2], ASvalues[1]+ASvalues[2])[0]
            plt.fill_between(shift_range, np.zeros(len(shift_range)), L(shift_range, *ASvalues), color = plt.cm.winter(peak_position/float(biggest_shift)), linestyle = 'None', alpha = 0.1,zorder = 1)#, hatch = '/')
    
    upperlim= max(np.array(signals).flatten())
    plt.ylim((min(signals[-1])-0.1*upperlim, 1.1*upperlim))
    plt.fill_between([-notch, notch], 0, 1000000, alpha = 1, zorder = 2, linestyle = 'None', color = 'w')
    plt.fill_between([-notch, notch], 0, 1000000, alpha = 0, zorder = 3, linestyle = 'None', hatch = '//')

def plot_linear_bg_fits(signals, shifts, sorted_fits, powers, bgs):
    colorfac = 0.5/max(powers)
    
    plt.figure('fitted spectra linear bgs')
    
    for index, signal in enumerate(signals):
        plt.plot(shifts, signal+bgs[index], color = plt.cm.plasma_r(powers[index]*colorfac), alpha = 1, zorder = 1)
        biggest_shift = max(sorted_fits.keys())
    for peak_position, parameters in list(sorted_fits.items()):
        for index, power in enumerate(parameters[2]):
            plt.plot(parameters[0][6][0][index],parameters[0][6][1][index], color = 'k')
            plt.plot(parameters[1][6][0][index],parameters[1][6][1][index], color = 'k')
    
    upperlim= max(np.array(signals).flatten())
    plt.ylim((min(signals[-1])-0.1*upperlim, 1.1*upperlim))
    plt.fill_between([-notch, notch], 0, 1000000, alpha = 1, zorder = 2, linestyle = 'None', color = 'w')
    plt.fill_between([-notch, notch], 0, 1000000, alpha = 0, zorder = 3, linestyle = 'None', hatch = '//')

def draw_arrow(powers, normed_data):
        x = np.mean(powers[:2]) 
        y = np.mean(normed_data[:2])
        dx = (powers[1]-powers[0])*0.2
        dy =(normed_data[1]-normed_data[0])*0.2
        plt.arrow(x,y,dx,dy, color = 'k', width = 0.005)

def plot_power_dependence(sorted_data): # data should be for a given peak - goes Stokes/anti-Stokes, then peak parameters (height, width, area), then powers              
    for peak_label, data in list(sorted_data.items()):
        powers = data[2]
        if len(powers)>4:
            plt.figure(str(peak_label)+ ', power_dependence')
            
            
#            plt.plot(powers, data[0][0]/max(data[0][0]), '--o', label = 'Stokes heights')
#            draw_arrow(powers, data[0][0]/max(data[0][0]))
#            
#            plt.plot(powers, data[0][2]/max(data[0][2]), '--o', label = 'Stokes widths')
#            draw_arrow(powers, data[0][2]/max(data[0][2]))
#            
            plt.plot(powers, old_div(data[0][3],max(data[0][3])), '-o', color = 'r', label = 'Stokes areas')
            draw_arrow(powers, old_div(data[0][3],max(data[0][3])))
            
            plt.plot(powers, old_div(data[0][4],max(data[0][4])), '--o', color = 'r', label = 'Stokes summed counts')
            draw_arrow(powers, old_div(data[0][4],max(data[0][4])))
            
#            plt.plot(powers, data[1][0]/max(data[1][0]), '--o', label = 'Anti-Stokes heights')
#            draw_arrow(powers, data[1][0]/max(data[1][0]))
#            
#            plt.plot(powers, data[1][2]/max(data[1][2]), '--o', label = 'Anti-Stokes widths')
#            draw_arrow(powers, data[1][2]/max(data[1][2]))
#            
            plt.plot(powers, old_div(data[1][3],max(data[1][3])), '-o', color = 'b', label = 'Anti-Stokes areas')
            draw_arrow(powers, old_div(data[1][3],max(data[1][3])))
            
            plt.plot(powers, old_div(data[1][4],max(data[1][4])), '--o', color = 'b', label = 'Anti-Stokes summed counts')
            draw_arrow(powers, old_div(data[1][4],max(data[1][4])))
           
            plt.xlabel('Power on sample (mW)')
            plt.ylabel('normalised intensity')
            plt.legend()


def plot_temperature(sorted_data): #all peaks
    
    plt.figure('peak temperatures')
    
    peak_labels = [key for key in list(sorted_data.keys())]
    
    peak_labels.sort()
    plabels = []
    for label in peak_labels:
        if int(label)<620 and int(label)>250 and len(sorted_data[label][2])>5:
            plabels.append(label)
    colors = plt.cm.viridis(np.linspace(0,1,num = len(plabels), endpoint = True))
    for index, peak_label in enumerate(plabels):   
        Ts = ms.stokesratio(sorted_data[peak_label][0][3], sorted_data[peak_label][1][3],int(peak_label), laser_wavelength = 633)
        plt.plot(sorted_data[peak_label][2], Ts, color = colors[index], label = str(peak_label))
        draw_arrow(sorted_data[peak_label][2], Ts)
        
        summed_Ts = ms.stokesratio(sorted_data[peak_label][0][4], sorted_data[peak_label][1][4],int(peak_label), laser_wavelength = 633)
        plt.plot(sorted_data[peak_label][2], summed_Ts, '--', color = colors[index], label = str(peak_label) + ' summed')
        draw_arrow(sorted_data[peak_label][2], summed_Ts)
        silly_Ts = ms.stokesratio(sorted_data[peak_label][0][5], sorted_data[peak_label][1][5],int(peak_label), laser_wavelength = 633)
        plt.plot(sorted_data[peak_label][2], silly_Ts, linestyle = 'dotted', color = colors[index], label = str(peak_label) + ' linear')
        draw_arrow(sorted_data[peak_label][2], silly_Ts)
        #plt.plot(peak[2], ms.stokesratio(peak[0][3], peak[1][3], int(peak_label)), label = str(peak_label), laser_wavelength = 633)
    plt.legend()
    plt.xlabel('Power on sample (mW)')
    plt.ylabel('Temperature (K)')
def plot_AS_temperature(powers,Ts, new_figure = True):
    if new_figure == True: plt.figure('AS temperature')
    plt.plot(powers, Ts, label = 'exp')
    draw_arrow(powers, Ts)
    plt.xlabel('Power on sample (mW)')
    plt.ylabel('Temperature (K)')
    plt.legend()
    
def compare_powers(particle, input_shifts, analysis_range):
    power_series, powers, measured_powers = condense_power_series(particle, input_shifts, analysis_range)
    extracted_powers = ms.get_laser_power_from_leak(power_series, input_shifts)
    plt.figure('laser power from leak')
    plt.plot(powers, old_div(extracted_powers,max(extracted_powers)), label = 'laser leak')
    draw_arrow(powers, old_div(extracted_powers,max(extracted_powers)))
    
    corrected_measured_powers = np.append(measured_powers[1:], measured_powers[1])
    plt.plot(powers, old_div(corrected_measured_powers,max(corrected_measured_powers)), label = 'powermeter')
    draw_arrow(powers, old_div(corrected_measured_powers,max(corrected_measured_powers)))
    plt.ylabel('normalised power extracted from laser')
    plt.xlabel('target power (mW)')
    plt.legend()

def analyse_particle(particle, shifts, T_misc, T_misc_shifts, analysis_range):
    '''
    particle is the hdf5 group. Shifts should be calibrated.
    '''
    fit_kwargs = {'T_misc' : T_misc,
                  'T_misc_shifts' : T_misc_shifts,
                  'plot' : True,
                  'plot_log' : False}
    Speaks, ASpeaks, bgs, AS_bg_ps, signals, powers, measured_powers, shifts = fit(particle, shifts, analysis_range, **fit_kwargs)  
    sorted_fits, AS_temperature, A = sort_fits(signals, shifts, Speaks, ASpeaks, AS_bg_ps, powers, bgs)
    plot_fits(signals, shifts, sorted_fits, powers)
    plot_power_dependence(sorted_fits)
    plot_temperature(sorted_fits)
    plot_linear_bg_fits(signals, shifts, sorted_fits, powers, bgs)
    compare_powers(particle, shifts, analysis_range)  
    plot_AS_temperature(powers, AS_temperature)
    return sorted_fits, AS_temperature, signals

def produce_h5_summmary(scan, of, shifts, T_misc, T_misc_shifts, analysis_range):
    for particle_name, particle in list(scan.items()):
        particle_number = int(particle_name[9:])
        if particle_number not in np.arange(108,len(scan)+1).tolist():
            group = of.require_group(particle_name)
            failed_particles = []
            try: sorted_fits, AS_temperature, signals = analyse_particle(particle, shifts, T_misc, T_misc_shifts, analysis_range)
            except: failed_particles.append(str(particle.name))
    
            for peak_label, peak in list(sorted_fits.items()):
                peak_group = group.require_group(str(peak_label) + ' data')
                for n, SaSP in enumerate(peak):
                    if n==2:
                        peak_group.create_dataset('powers', data = SaSP)
                    elif n == 0:
                        peak_group.create_dataset('Stokes', data = SaSP)
                        
                    elif n == 1:
                       peak_group.create_dataset('Anti-Stokes', data = SaSP)
            group.create_dataset('AS_temperature', data = AS_temperature)
            group.create_dataset('signals', data = signals)
            for figlabel in plt.get_figlabels():
                fig = plt.figure(figlabel)
                fig.canvas.draw()
                im = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                im = im.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                group.create_dataset(figlabel, data = im)
                plt.close(figlabel)
    
if __name__ == '__main__':
    start = time.time()
    os.chdir(r'R:\ee306\Experimental data\2019.10.04 Particle track 4hr BPT lab 5 power series')
    with h5py.File(ms.findH5File(os.getcwd()), mode = 'r') as File:   
        scan = File['ParticleScannerScan_0']
        analysis_range = -980, 890 #cm-1
        notch = 220
        shifts = -cnv.wavelength_to_cm(scan['Particle_0']['power_series_0'].attrs['wavelengths'], centre_wl = 785)
        AS_shifts = ms.truncate(shifts, shifts, -np.inf, 0)[0]
        S_shifts = ms.truncate(shifts, shifts, 0, np.inf)[0]
        AS_fit = np.loadtxt(r'R:\ee306\Calibration\Lab 5\785nm_anti_Stokes_polynomial.txt')
        S_fit = np.loadtxt(r'R:\ee306\Calibration\Lab 5\785nm_Stokes_polynomial.txt')
        calibrated_AS_shifts = np.polyval(AS_fit, AS_shifts)
        calibrated_S_shifts = np.polyval(S_fit, S_shifts)
        calibrated_shifts = np.append(calibrated_AS_shifts, calibrated_S_shifts)
        [6,7,9,14,16,23,24,34,35,70,75,83]
        T_misc_shifts, T_misc = np.loadtxt(r'R:\ee306\Calibration\Lab 5\Transmission_function_100x_09_NA_600lmm_785nm_5050bs.txt')
        analyse_particle(scan['Particle_7'], calibrated_shifts, T_misc, T_misc_shifts, analysis_range)        
        
    end = time.time()
    print('That took '+str(np.round(end - start))+ 'seconds')
    
    
    
    
    
    
    