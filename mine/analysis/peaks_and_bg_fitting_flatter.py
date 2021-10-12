# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:50:45 2019

@author: Eoin Elliott -ee306

The fullfit class is the main thing here - sample use:
    from nplab.analysis.peaks_and_bg_fitting import fullfit
>>>ff = fullfit(
                 spec,
                 shifts,
                 lineshape = 'L',
                 order = 7,
                 transmission = None,
                 bg_function = 'poly',
                 vary_const_bg = True)
    #   initialise the object.
        spec, shifts are the spectrum and their shifts
        lineshape can be 'L' (Lorentzian) or 'G' (Gaussian). Gaussians are less broad
        The 'order' key-word argument is the order of the background polynomial. 3-7 works well.
        transmission is the instrument response function of the detector. It's necessary to include it here rather than in the raw data as most of the background is electronic, so dividing this by the IRF introduces unphysical features.
        This way, the IRF is only applied to the background-subtracted signal
        bg function determines what function to use for the background, default is polynomial. Feel free to add your own functions here!
        set vary_const_bg to False if the spectrum is already (electronic) background subtracted and you're using an exponential fit

>>>ff.Run() # this does the actual fitting.
                then the peaks are stored as
>>>ff.peaks # 2d list of parameters: height, x-position, and width. Plot with ff.multi_L(peaks)
>>>ff.peaks_stack has them in a 2d array so you can plot the peaks individually like so:
        for peak in ff.peaks_stack:
            plt.plot(ff.shifts, ff.L(ff.shifts, peak))
>>>ff.bg gives the background as a 1d array.
>>>ff.signal gives the background-subtracted spectrum, divided by the transmission
>>>ff.bg_p gives the polynomial coefficients, or in the case of exponential background Amplitude, Temperature and constant background

The fitting works as follows:
    Run(self,
            initial_fit=None,
            add_peaks = True,
            allow_asymmetry = False,
            minwidth = 8,
            maxwidth = 30,
            regions = 20,
            noise_factor = 0.01,
            min_peak_spacing = 5,
            comparison_thresh = 0.05,
            verbose = False):

        initial fit should be a 1D array of height, centre widths (or None).
        if add_peaks is True, then the code will try to add additional peaks. If you're happy with the peaks already, set = False and it will just optimize a background and the peaks
        allow_asymmetry, if true allows the peaks to become asymmetric as a final step. see optimize_asymm(self) for details
        minwidth  is the minimum  width a fitted peak can have.
        maxwidth is the maximum width a peak can have.
        regions works as in Iterative_Raman_Fitting - see add_new_peak for details
        noise_factor is the minimum height above the noise level a peak must have. It's not connected to anything physical however, just tune it to exclude/include lower S:N peaks
        min_peak_spacing is the minimum separation (in # of peak widths) a new peak must have from all existing peaks. Prevents multiple Lorentzians being fitted to the one peak.
        comparison_thresh  is the fractional difference allowed between fit optimisations for the peak to be considered fitted.

    initial_bg_poly() takes a guess at what the background is. The signal is (spectrum-bg)/transmission

    if initial_fit is included, the peak heights are then somewhat manually assigned by getting the maximum of the signal around the peak centre.
    The widths and centres of the peaks are then optimised for these heights.
    optionally, you can include the commented out optimize_peaks() here to optimise all the peak parameters together but I leave this to the end.

    add_new_peak() forcibly adds a peak to the signal similarly to in Iterative_Raman_fitting.



    If this new peak improves the fit, it adds another peak, repeats.

    Else, if it doesn't improve the fit (a sign that the right # of peaks have been added) the number of regions is multiplied by 5 to try more possible places to add a peak.
    It also now optimises the background with the peaks, as doing so before will make it cut through the un-fitted peaks

    If no new peak has been added (again, a good sign) the rest of the function matches each peak with its nearest neighbour (hopefully itself)
    from before the latest round of optimisation. If none of the peaks have moved by comparison_thresh*sqrt(height and width added in quadrature)
    the fit is considered optimised, and the number of peak adding regions is increased by 5x.

    one last optional round of optimisation is included at the very end

This script uses cm-1 for shifts. Using wavelengths is fine, but make sure and adjust the maxwidth, minwidth parameters accordingly

The time taken to fit increases a lot with spectrum size/length, so cutting out irrelevant regions such as the notch is important,
as is fitting the stokes and anti-stokes spectra separately.
"""
from numba import jit, njit, vectorize
import pdb
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize

from scipy.ndimage.filters import gaussian_filter
from scipy.signal import argrelextrema


def sm(spec, sigma=3):
    return gaussian_filter(spec, sigma)


DUMMY = False
PEAKWIDTH = 16


def split(spectrum, wavelengths, lower=-np.inf, upper=np.inf):
    ''' a more concise truncate'''

    if upper < lower:
        upper, lower = lower, upper
    condition = (lower <= wavelengths) & (
        wavelengths < upper)  # <= allows recombination
    return spectrum[condition], wavelengths[condition],


def closest_arg(value, arr):
    '''Taking an input value and array, it searches for the value and index in the array which is closest to the input value '''

    i = np.argmin(np.abs(arr-value))
    return i, abs(arr[i]-value)  # closest, its index and the residual


def reshape(_list, n):
    while len(_list) >= n:
        yield _list[:n]
        _list = _list[n:]


@jit()
def L(x, H, C, W):  # height centre width
    """
    Defines a lorentzian
    """
    return H/(1.+((x-C)/W)**2)


@jit()
def G(x, H, C, W):
    '''
    A Gaussian
    '''
    return H*np.exp(-((x-C)/W)**2)


# @njit()
def initial_bg_poly(spec, shifts, order):
    '''
    takes an inital guess at the background.
    takes the local minima of the smoothed spectrum, weighted by how far they are to other minima, and fits to these.
    weighting is to prioritise flatter portions of the spectrum (which would naturally have fewer minima)

    '''

    smoothed = sm(spec)
    bg_indices = argrelextrema(smoothed, np.less)[0]
    while len(bg_indices) < 3*order:
        bg_indices = np.append(bg_indices,
                               min(np.random.randint(0,
                                                     high=len(
                                                         spec),
                                                     size=(10,)),
                                   key=lambda i: spec[i]))
    bg_vals = smoothed[bg_indices]

    residuals = [closest_arg(index, np.setdiff1d(bg_indices, index))[
        1] for index in bg_indices]
    norm_fac = 5./max(residuals)
    extra_lens = norm_fac*np.array(residuals)
    for bg_index, extra_len in zip(bg_indices, extra_lens):
        extra_len = int(extra_len)
        if extra_len < 1:
            extra_len = 1
        for extra in np.arange(extra_len)+1:
            try:
                bg_vals.append(smoothed[bg_index+extra])
                bg_indices.append(bg_index+extra)
            except:
                pass
            try:
                bg_vals.append(smoothed[bg_index-extra])
                bg_indices.append(bg_index-extra)
            except:
                pass
    edges = np.arange(5)+1
    edges = np.append(edges, -edges).tolist()
    bg_indices = np.append(bg_indices, edges)
    bg_vals = np.append(bg_vals, smoothed[edges])

    bg_bound = (min(spec), max(spec))
    bg_bounds = []
    while len(bg_bounds) < len(bg_vals):
        bg_bounds.append(bg_bound)
    bg_p = np.polyfit(shifts[bg_indices], bg_vals, order)
    return bg_p


def run(spec,
        shifts,
        lineshape='L',
        bg_function='poly',
        order=7,
        transmission=None,
        initial_fit=None,
        add_peaks=True,
        minwidth=2.5,
        maxwidth=20,
        regions=10,
        noise_factor=0.1,
        min_peak_spacing=3.1,
        comparison_thresh=0.01,
        verbose=False
        ):

    spec = np.array(spec)
    shifts = np.array(shifts)
    transmission = np.ones(len(spec))
    if transmission is not None:
        transmission *= transmission
    # if lineshape == 'L':
    #     @njit()
    #     def line(H, C, W):
    #         return L(shifts, H, C, W)
    #     @njit()
    #     def multi_line(parameters):
    #         return np.sum(np.array([line(*peak) for peak in parameters]), axis=0)

    # if lineshape == 'G':

    @njit()
    def line(HCW):  # ndarray(height, centre width)
        return G(shifts, HCW[0], HCW[1], HCW[2])

    @njit()
    def multi_line(parameters):
        lol = np.empty((len(parameters), len(shifts)))
        for i, p in enumerate(parameters):
            lol[i] = line(p)
        return lol.sum(axis=0)

    if lineshape == 'L':
        maxwidth /= 2.
        minwidth /= 2.
    if lineshape == 'G':
        maxwidth /= 0.95
        minwidth /= 0.95

    width = PEAKWIDTH  # a guess for the peak width

    noise = np.sqrt(np.var(spec/gaussian_filter(spec, 40))).mean()

    noise_threshold = noise_factor*noise

    # number of regions the spectrum will be split into to add a new peak
    regions = regions
    if regions > len(spec):
        # can't be have more regions than points in spectrum
        regions = len(spec)//2

    # @njit()
    def add_new_peak(shifts, signal, current, peaks, regions, noise_threshold, width, bound):
        '''
        lifted from Iterative_Raman_Fitting
        '''
        # -----Calc. size of x_axis regions-------
        size = (shifts.max()-shifts.min())
        sectionsize = size/regions
        start = np.min(shifts)

        results = np.empty((regions, 3))
        loss_results = np.empty(regions)
        
        @njit()
        def loss(params):
            return np.abs(signal - current - line(params)).sum()

        for i in range(int(regions)):
            bounds = [(0, np.inf), (i*sectionsize+start, (i+1) *
                                    sectionsize+start), (0, size)]
            Centre = (i+np.random.rand())*sectionsize+start
            try:
                Height = max(split(current, shifts, i*sectionsize +
                                   start, (i+1)*sectionsize+start)[0])-min(current)
            except:
                Height = noise_threshold
            starting_params = np.array([Height, Centre, width])
            params = minimize(loss, starting_params, bounds=bounds).x

            results[i] = params
            loss_results[i] = loss(params)

        sorted_results = results[np.argsort(loss_results)]

        # test the top 50% of peak

        for peak_candidate in sorted_results:

            # has a height, minimum width - maximum width are within bounds
            if peak_candidate[0] > noise_threshold and maxwidth > peak_candidate[2] > minwidth:
                if peaks.size != 0:
                    peak_index, residual = closest_arg(
                        peak_candidate[1], np.transpose(peaks)[1])
                    # is far enough away from existing peaks
                    if residual > min_peak_spacing*peaks[peak_index][2]:
                        return peak_candidate

                else:  # If no existing peaks, accept it
                    return peak_candidate

        return np.zeros(3)

    # takes a guess at the background
    bg_p = initial_bg_poly(spec, shifts, order)

    def bg(p):
        return np.polyval(p, shifts)

    def signal(bg_p):
        return spec - bg(bg_p)

    _signal = signal(bg_p)
    if noise_threshold > _signal.max():
        noise_threshold = _signal.min()
        print('warning, noise threhsold above spectrum')
    height_bound = (noise_threshold, np.max(_signal))
    pos_bound = (np.min(shifts), np.max(shifts))
    width_bound = (minwidth, maxwidth)

    bound = (height_bound, pos_bound, width_bound)

    def optimize_peaks_and_bg(bg_p, peaks):
        '''
        optimizes the peaks and background in one procedure,
        allowing for better interplay of peaks and bg
        '''

           
        def loss(peaks_and_bg):
            fit = np.polyval(peaks_and_bg[:order+1], shifts)
            fit += multi_line(peaks_to_matrix(
                peaks_and_bg[order+1:]))*transmission
            residual = spec - fit
            above = residual[residual > 0]
            below = residual[residual < 0]
            # prioritises fitting the background to lower data points
            obj = (np.absolute(above)).sum() + (below**2).sum()
            return obj

        peaks_and_bg = np.append(bg_p, np.ravel(peaks))
        bounds = [(-np.inf, np.inf) for p in bg_p]
        if len(peaks):
            for _ in peaks:
                bounds.extend(bound)
        peaks_and_bg = minimize(
            loss, peaks_and_bg, bounds=bounds).x.tolist()

        bg_p = peaks_and_bg[:order+1]
        peaks = peaks_to_matrix(peaks_and_bg[order+1:])
        return bg_p, peaks

    def optimize_bg(bg_p, peakless):
        '''
        it's important to note that the parameter optimised isn't the polynomial coefficients bg_p ,
        but the points taken on the spectrum-peaks curve (bg_vals) at positions bg_indices, decided by initial_bg_poly().
        This is because it's easy to put the bounds of the minimum and maximum of the spectrum on these to improve optimisation time. (maybe)
        '''
        def bg_loss(bg_p):
            '''
            evaluates the fit of the background to spectrum-peaks
            '''

            fit = np.polyval(bg_p, shifts)
            residual = peakless - fit
            above = residual[residual > 0]
            below = residual[residual < 0]
            obj = np.sum(np.absolute(above))+10*np.sum(np.array(below))**4

            return obj
        return minimize(bg_loss, bg_p).x

    def optimize_peaks(peaks, signal):
        '''
        optimizes the height, centres and widths of all peaks
        '''
        bounds = bound*len(peaks)

        # @njit
        def peak_loss(peaks):
            '''
            evalutes difference between the fitted peaks and the signal (spectrum - background)
            '''
            fit = multi_line(peaks_to_matrix(peaks))
            obj = np.sum(np.square(signal - fit))
            return obj

        peaks = peaks_to_matrix(
            minimize(peak_loss, np.ravel(peaks), bounds=bounds).x)
        return peaks

    def optimize_centre_and_width(peaks, signal):
        '''
        optimizes the centres(positions) and widths of the peakss for a given heights.
        '''
        if len(peaks) < 2:
            return peaks
        heights = peaks[:, 0]
        centres_and_widths = peaks[:, 1:]
        width_bound = (minwidth, maxwidth)
        centre_and_width_bounds = []
        for centre, width in centres_and_widths:
            centre_and_width_bounds.extend(
                [(centre-width, centre+width), (width_bound)])  # height, position, width

        def multi_line_centres_and_widths(centres_and_widths):
            """
            Defines a sum of Lorentzians. Params goes Height1,Centre1, Width1,Height2.....
            """
            params = np.array([[h, c, w] for h, (c, w) in zip(
                heights, reshape(centres_and_widths, 2))])
            return multi_line(params)

        def loss_centres_and_widths(centres_and_widths):
            fit = multi_line_centres_and_widths(centres_and_widths)
            obj = np.sum(np.square(signal - fit))
            return obj
        centres_and_widths = minimize(loss_centres_and_widths, np.ravel(
            centres_and_widths, 'F'), bounds=centre_and_width_bounds).x
        peaks = np.array([[h, *cw] for h, cw in zip(
            heights, reshape(centres_and_widths, 2))])
        return peaks

    def optimize_heights(peaks, signal):
        '''
        crudely gets the maximum of the signal within the peak width as an estimate for the peak height
        '''
        if len(peaks) < 1:
            return peaks
        else:

            for index, peak in enumerate(peaks):

                peaks[index][0] = max(
                    split(signal, shifts, peak[1]-peak[2]/4., peak[1]+peak[2]/4.)[0])
        return peaks

    if initial_fit is not None:
        peaks = np.array(initial_fit)
        if not add_peaks:
            # if regions is bigger than the spectrum length, then no peaks are added
            regions = len(spec)+1
        # 2d array of peak parameters
        # bounds for the peak parameters

        # creates a bound for each peak
        optimize_heights()  # see functions for descriptions
        optimize_centre_and_width()
        optimize_peaks()

    def loss_function(bg_p, peaks):
        '''
        evaluates the overall (bg+peaks) fit to the spectrum
        '''

        fit = bg(bg_p) + multi_line(peaks)*transmission

        return np.sum(np.square(spec - fit))

    peaks = add_new_peak(shifts, _signal, np.zeros(len(_signal)), np.array(
        [[]]), regions, noise_threshold, width, bound)[np.newaxis, :]
    while regions <= len(spec):
        existing_loss_score = loss_function(
            bg_p, peaks)  # measure of fit 'goodness'

        new_peak = add_new_peak(shifts, signal(bg_p), multi_line(
            peaks), peaks, regions, noise_threshold, width, bound)  # adds a peak
        if any(new_peak):
            peaks = np.vstack((peaks, new_peak))

        else:
            regions *= 4
            continue
        bg_p, peaks = optimize_peaks_and_bg(bg_p, peaks)

        new_loss_score = loss_function(bg_p, peaks)
        # plt.figure()
        # plt.plot(shifts, bg(bg_p))
        # plt.plot(shifts, multi_line(peaks)+bg(bg_p))
        # plt.plot(shifts, spec)

        # ---Check to increase regions

        if new_loss_score >= existing_loss_score:  # if fit has worsened, delete last peak
            peaks = peaks[:-1]
            regions *= 4  # increase regions, as this is a sign fit is nearly finished

    # ---One last round of optimization for luck: can comment these in and out as you see fit.
#

    bg_p, peaks = optimize_peaks_and_bg(bg_p, peaks)
    bg_p = optimize_bg(bg_p, spec-multi_line(peaks))
    _signal = signal(bg_p)
    peaks = optimize_heights(peaks, _signal)
    peaks = optimize_centre_and_width(peaks, _signal)
    peaks = optimize_peaks(peaks, _signal)
    return bg_p, peaks


def peaks_to_matrix(peak_array):
    '''
    converts a 1d peak_array into a 2d one
    '''

    return np.reshape(peak_array, (len(peak_array)//3, 3))


def plot_result(spec, shifts, bg_p, peaks):
    '''
    plots the spectrum and the individual peaks, and their sums

    '''
    plt.figure()
    plt.plot(shifts, spec)
    bg = np.polyval(bg_p, shifts)
    plt.plot(shifts, bg, 'k')
    for peak in peaks:
        plt.plot(shifts, bg+G(shifts, *peak), '--')
    # plt.plot(shifts, bg + multi_line(peaks)
    #          * transmission, linestyle='--', color='k')


if __name__ == '__main__':
    from time import time
    from nplab.analysis.example_data import SERS_and_shifts

    spec = SERS_and_shifts[0]
    shifts = SERS_and_shifts[1]
    spec, shifts = split(spec, shifts, 190, np.inf)

    start = time()
    bg_p, peaks = run(spec, shifts, lineshape='G', order=9)
    print(f'TIME TAKEN: {time()-start}s')
    plot_result(spec, shifts, bg_p, peaks)
