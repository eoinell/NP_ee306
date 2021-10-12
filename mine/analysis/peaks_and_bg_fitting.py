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
    optionally, you can include the commented out self.optimize_peaks() here to optimise all the peak parameters together but I leave this to the end.
    
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


from numba import njit, vectorize
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
    return i, abs(arr[i]-value) # closest, its index and the residual

def reshape(_list, n):
    while len(_list) >= n:
        yield _list[:n]
        _list = _list[n:]

@njit()
def L(x, H, C, W):  # height centre width
    """
    Defines a lorentzian
    """
    return H/(1.+((x-C)/W)**2)

@njit()  
def G(x, H, C, W):
    '''
    A Gaussian
    '''
    return H*np.exp(-((x-C)/W)**2)

class FullFit():
    def __init__(self,
                 spec,
                 shifts,
                 lineshape='L',
                 bg_function='poly',
                 order=7,
                 transmission=None,
                 ):

        self.spec = np.array(spec)
        self.shifts = np.array(shifts)
        self.order = order
        self.peaks = []

        self.peak_bounds = []

        self.asymm_peaks = None
        self.transmission = np.ones(len(spec))
        if transmission is not None:
            self.transmission *= transmission
        if lineshape == 'L':
            self.line = lambda H, C, W: L(self.shifts, H, C, W)
        if lineshape == 'G':
            self.line = lambda H, C, W: G(self.shifts, H, C, W)
        self.lineshape = lineshape

    def multi_line(self, parameters):
        """
        returns a sum of Lorenzians/Gaussians. 
        """
        return np.sum(np.array([self.line(*peak) for peak in parameters]), axis=0)

    def add_new_peak(self):
        '''
        lifted from Iterative_Raman_Fitting
        '''
        # -----Calc. size of x_axis regions-------
        sectionsize = (self.shifts.max()--self.shifts.min())/self.regions
        start = np.min(self.shifts)

        results = []
        loss_results = []

        
        if not len(self.peaks):
            Current = np.array(self.shifts)*0
        else:
            Current = self.multi_line(self.peaks)

        def loss(params):
            # *self.multi_L(self.shifts,*self.peaks))# if this overlaps with another lorentzian it's biased against it
            return np.sum(np.abs(Current+self.line(*params)-self.signal))

        for i in range(int(self.regions)):
            bounds = [(0, np.inf), (i*sectionsize+start, (i+1) *
                                    sectionsize+start), (0, max(self.shifts)-min(self.shifts))]
            Centre = (i+np.random.rand())*sectionsize+start
            try:
                Height = max(split(self.signal, self.shifts, i*sectionsize +
                                      start, (i+1)*sectionsize+start)[0])-min(self.signal)
            except:
                Height = self.noise_threshold
            params = [Height, Centre, self.width]

            params = minimize(loss, params, bounds=bounds).x.tolist()

            results.append(params)
            loss_results.append(loss(params))

        sorted_indices = np.argsort(loss_results)

        self.peak_added = False
        i = -1
        # test the top 50% of peaks
        while self.peak_added == False and i < (len(loss_results) - 1):
            i += 1
            peak_candidate = results[sorted_indices[i]]
            # has a height, minimum width - maximum width are within bounds
            if peak_candidate[0] > self.noise_threshold and self.maxwidth > peak_candidate[2] > self.minwidth:
                if len(self.peaks) != 0:
                    peak_index, residual = closest_arg(peak_candidate[1], np.transpose(self.peaks)[1])
                    # is far enough away from existing peaks
                    if residual > self.min_peak_spacing*self.peaks[peak_index][2]:
                        self.peaks.append(peak_candidate)
                        self.peak_added = True

                else:  # If no existing peaks, accept it
                    self.peaks.append(peak_candidate)
                    self.peak_added = True
            else:
                # print(peak_candidate[0], self.noise_threshold, self.maxwidth, peak_candidate[2],self.minwidth)
                pass
        if self.peak_added:  # TODO store these values
            self.peak_bounds.append(self.bound)  # height, position, width
            if self.verbose:
                print('peak added')
        if not self.peak_added and self.verbose:
            print('no suitable peaks to add')
            
    def initial_bg_poly(self):
        '''
        takes an inital guess at the background.
        takes the local minima of the smoothed spectrum, weighted by how far they are to other minima, and fits to these.
        weighting is to prioritise flatter portions of the spectrum (which would naturally have fewer minima)

        '''

        smoothed = sm(self.spec)
        self.bg_indices = argrelextrema(smoothed, np.less)[0]
        while len(self.bg_indices) < 3*self.order:
            self.bg_indices = np.append(self.bg_indices,
                                        min(np.random.randint(0,
                                                              high=len(
                                                                  self.spec),
                                                              size=(10,)),
                                            key=lambda i: self.spec[i]))
        self.bg_vals = smoothed[self.bg_indices]

        residuals = [closest_arg(index, np.setdiff1d(self.bg_indices, index))[1] for index in self.bg_indices]
        norm_fac = 5./max(residuals)
        extra_lens = norm_fac*np.array(residuals)
        for bg_index, extra_len in zip(self.bg_indices, extra_lens):
            extra_len = int(extra_len)
            if extra_len < 1:
                extra_len = 1
            for extra in np.arange(extra_len)+1:
                try:
                    self.bg_vals.append(smoothed[bg_index+extra])
                    self.bg_indices.append(bg_index+extra)
                except:
                    pass
                try:
                    self.bg_vals.append(smoothed[bg_index-extra])
                    self.bg_indices.append(bg_index-extra)
                except:
                    pass
        edges = np.arange(5)+1
        edges = np.append(edges, -edges).tolist()
        self.bg_indices = np.append(self.bg_indices, edges)
        self.bg_vals = np.append(self.bg_vals, smoothed[edges])

    
        self.bg_bound = (min(self.spec), max(self.spec))
        self.bg_bounds = []
        while len(self.bg_bounds) < len(self.bg_vals):
            self.bg_bounds.append(self.bg_bound)
        self.bg_p = np.polyfit(
            self.shifts[self.bg_indices], self.bg_vals, self.order)
        self.bg = np.polyval(self.bg_p, self.shifts)

        self.signal = np.array(self.spec - self.bg)/self.transmission

    def bg_loss(self, bg_p):
        '''
        evaluates the fit of the background to spectrum-peaks
        '''

        fit = np.polyval(bg_p, self.shifts)
        residual = self.spec - self.peaks_evaluated - fit
        above = residual[residual > 0]
        below = residual[residual < 0]
        obj = np.sum(np.absolute(above))+10*np.sum(np.array(below))**4

        return obj

    def optimize_bg(self):
        '''
        it's important to note that the parameter optimised isn't the polynomial coefficients bg_p , 
        but the points taken on the spectrum-peaks curve (bg_vals) at positions bg_indices, decided by initial_bg_poly().
        This is because it's easy to put the bounds of the minimum and maximum of the spectrum on these to improve optimisation time. (maybe)
        '''
        if self.asymm_peaks_stack == None:
            self.peaks_evaluated = self.multi_line(
                self.shifts, self.peaks)*self.transmission
        else:
            self.peaks_evaluated = self.asymm_multi_line(
                self.shifts, self.asymm_peaks)*self.transmission

        if self.bg_type == 'poly':
            self.bg_p = minimize(self.bg_loss, self.bg_p).x.tolist()
            self.bg = np.polyval(self.bg_p, self.shifts)
            self.signal = (np.array(self.spec - self.bg)/
                                   self.transmission)
        elif self.bg_type == 'exponential' or self.bg_type == 'exponential2':

            self.bg_p = curve_fit(self.bg_function,
                                  self.shifts,
                                  self.spec -
                                  self.multi_line(
                                      self.shifts, self.peaks)*self.transmission,
                                  p0=self.bg_p,
                                  bounds=self.bg_bounds,
                                  maxfev=10000)[0]
            self.bg = self.bg_function(self.shifts, *self.bg_p)

    

    def peak_loss(self, peaks):
        '''
        evalutes difference between the fitted peaks and the signal (spectrum - background)
        '''
        fit = self.multi_line(self.peaks_to_matrix(peaks))
        obj = np.sum(np.square(self.signal - fit))
        return obj

    def optimize_peaks(self):
        '''
        optimizes the height, centres and widths of all peaks
        '''
        if len(self.peaks) < 2:
            return
        bounds = []
        for bound in self.peak_bounds:
            bounds.extend(bound)
        self.peaks = self.peaks_to_matrix(
            minimize(self.peak_loss, np.ravel(self.peaks), bounds=bounds).x.tolist())

    def optimize_centre_and_width(self):
        '''
        optimizes the centres(positions) and widths of the peakss for a given heights.
        '''
        if len(self.peaks) < 2:
            return
        heights = np.transpose(self.peaks)[0]
        centres_and_widths = np.transpose(self.peaks)[1:]
        # centres_and_widths = np.ravel(centres_and_widths)
        width_bound = (self.minwidth, self.maxwidth)
        centre_and_width_bounds = []
        for centre, width in zip(*centres_and_widths):
            centre_and_width_bounds.extend(
                [(centre-width, centre+width), (width_bound)])  # height, position, width
       
        def multi_line_centres_and_widths(centres_and_widths):
            """
            Defines a sum of Lorentzians. Params goes Height1,Centre1, Width1,Height2.....
            """
            params = [[h, c, w] for h, (c, w) in zip(
                heights, reshape(centres_and_widths, 2))]
            return self.multi_line(params)
        
        def loss_centres_and_widths(centres_and_widths):
            fit = multi_line_centres_and_widths(centres_and_widths)
            obj = np.sum(np.square(self.signal - fit))
            return obj
        centres_and_widths = minimize(loss_centres_and_widths, np.ravel(
            centres_and_widths, 'F'), bounds=centre_and_width_bounds).x
        self.peaks = [[h, c, w] for h, (c, w) in zip(
            heights, reshape(centres_and_widths, 2))]

    def optimize_heights(self):
        '''
        crudely gets the maximum of the signal within the peak width as an estimate for the peak height
        '''
        if len(self.peaks) < 1:
            return
        else:

            for index, peak in enumerate(self.peaks):
                try:
                    self.peaks[index][0] = max(
                        split(self.signal, self.shifts, peak[1]-peak[2]/4., peak[1]+peak[2]/4.)[0])
                except IndexError:
                    self.peaks[index][0] = self.signal[min(range(len(self.signal)), key=lambda i: abs(peak[1]-self.shifts[i]))]

    def loss_function(self):
        '''
        evaluates the overall (bg+peaks) fit to the spectrum
        '''

        fit = self.bg + self.multi_line(self.peaks)*self.transmission
        obj = np.sum(np.square(self.spec - fit))
        return obj

    @staticmethod
    def peaks_to_matrix(peak_array):
        '''
        converts a 1d peak_array into a 2d one
        '''

        return [peak for peak in reshape(peak_array, 3)]    
    
    def optimize_peaks_and_bg(self):
        '''
        optimizes the peaks and background in one procedure, 
        allowing for better interplay of peaks and bg
        '''
        def loss(peaks_and_bg):
            fit = np.polyval(peaks_and_bg[:self.order+1], self.shifts)
            fit += self.multi_line(self.peaks_to_matrix(
                peaks_and_bg[self.order+1:]))*self.transmission
            residual = self.spec - fit
            above = residual[residual > 0]
            below = residual[residual < 0]
            # prioritises fitting the background to lower data points
            obj = np.sum(np.absolute(above))+np.sum(np.array(below)**2)
            return obj
        
        peaks_and_bg = np.append(self.bg_p, np.ravel(self.peaks))
        bounds = [(-np.inf, np.inf) for p in self.bg_p]
        if len(self.peaks):
            for bound in self.peak_bounds:
                bounds.extend(bound)
        peaks_and_bg = minimize(
            loss, peaks_and_bg, bounds=bounds).x.tolist()

        self.bg_p = peaks_and_bg[:self.order+1]
        self.peaks = self.peaks_to_matrix(peaks_and_bg[self.order+1:])
        self.bg = np.polyval(self.bg_p, self.shifts)
       

    def dummy_run(self,
                 initial_fit=None,
                 add_peaks=True,
                 allow_asymmetry=False,
                 minwidth=8,
                 maxwidth=30,
                 regions=20, noise_factor=2,
                 min_peak_spacing=5,
                 comparison_thresh=0.05,
                 verbose=False):
        '''
        handy to test other scripts quickly
        '''

        self.initial_bg_poly()
        self.minwidth = minwidth
        self.maxwidth = maxwidth
#        if initial_fit is not None:
#            self.peaks = initial_fit
#            self.peaks_stack = np.reshape(self.peaks, [len(self.peaks)/3, 3])
#            self.optimize_heights
#            #self.optimize_centre_and_width()

        smoothed = sm(self.spec)
        maxima = argrelextrema(smoothed, np.greater)[0]
        heights = smoothed[maxima]
        maxima = maxima[np.argsort(heights)[-5:]]
        heights = smoothed[maxima]

        centres = self.shifts[maxima]
        widths = np.ones(len(maxima))*PEAKWIDTH

        self.peaks = np.transpose(np.stack([heights, centres, widths]))

        self.optimize_heights
        self.optimize_centre_and_width()
        print("I'm a dummy!")

    def run(self, *args, **kwargs):
        if DUMMY:
            self.dummy_run(*args, **kwargs)
        else:
            self._run(*args, **kwargs)

    def _run(self,
             initial_fit=None,
             add_peaks=True,
             allow_asymmetry=False,
             minwidth=2.5,
             maxwidth=20,
             regions=10,
             noise_factor=0.1,
             min_peak_spacing=3.1,
             comparison_thresh=0.01,
             verbose=False):
        '''
        described at the top
        '''
        if self.lineshape == 'L':
            self.maxwidth = maxwidth/2.
            self.minwidth = minwidth/2.
        if self.lineshape == 'G':
            self.maxwidth = maxwidth/0.95
            self.minwidth = minwidth/0.95
        self.verbose = verbose
        self.min_peak_spacing = min_peak_spacing
        self.width = PEAKWIDTH  # a guess for the peak width
        
        noise = np.sqrt(np.var(self.spec/gaussian_filter(self.spec, 40))).mean()

        self.noise_threshold = noise_factor*noise

        # number of regions the spectrum will be split into to add a new peak
        self.regions = regions
        if self.regions > len(self.spec):
            # can't be have more regions than points in spectrum
            self.regions = len(self.spec)//2

        self.initial_bg_poly()  # takes a guess at the background
        if self.noise_threshold>np.max(self.signal):
            self.noise_threshold=np.min(self.signal)
        height_bound = (self.noise_threshold, np.max(self.signal))
        
        pos_bound = (np.min(self.shifts), np.max(self.shifts))
        width_bound = (self.minwidth, self.maxwidth)

        self.bound = [height_bound, pos_bound, width_bound]

        if initial_fit is not None:
            self.peaks = deepcopy(initial_fit)
            if add_peaks == False:
                # if regions is bigger than the spectrum length, then no peaks are added
                self.regions = len(self.spec)+1
            self.peaks_stack = self.peaks_to_matrix(
                self.peaks)  # 2d array of peak parameters
            # bounds for the peak parameters
            height_bound = (self.noise_threshold, max(self.signal))
            pos_bound = (np.min(self.shifts), np.max(self.shifts))
            width_bound = (self.minwidth, self.maxwidth)
            self.peak_bounds = [self.bound for _ in self.peaks]
            # creates a bound for each peak
            self.optimize_heights()  # see functions for descriptions
            self.optimize_centre_and_width()

            self.optimize_peaks()
        while self.regions <= len(self.spec):
            if verbose == True:
                print('Region fraction: ', np.around(
                    self.regions/float(len(self.spec)), decimals=2))
            existing_loss_score = self.loss_function()  # measure of fit 'goodness'
            Old = self.peaks  # peaks before new one added
            self.add_new_peak()  # adds a peak
            if verbose == True:
                print('# of peaks:', len(self.peaks))
#            self.optimize_heights
#            self.optimize_centre_and_width()
            self.optimize_peaks_and_bg()  # optimizes
            new_loss_score = self.loss_function()

            # ---Check to increase regions
            if new_loss_score >= existing_loss_score:  # if fit has worsened, delete last peak
                if self.peak_added:
                    self.peaks = self.peaks[:-1]
                    self.peak_bounds = self.peak_bounds[:-1]
                    if verbose:
                        print('peak removed as it made the fit worse')
                self.regions *= 4  # increase regions, as this is a sign fit is nearly finished

            elif not self.peak_added:  # Otherwise, same number of peaks?
                #                self.optimize_bg()
                #                self.optimize_heights() # fails if no peaks
                #                self.optimize_centre_and_width()
                #                self.optimize_peaks()
                self.optimize_peaks_and_bg()
                New = self.peaks
                New_trnsp = np.transpose(New)
                residual = []
                for old_peak in Old:
                    # returns index of the new peak which matches it
                    new_peak = closest_arg(old_peak[1], New_trnsp[1])[0]
                    old_height = old_peak[0]
                    old_pos = old_peak[1]/self.width
                    new_height = New[new_peak][0]/old_height
                    # normalise the height and position parameters to add them into one comparison score
                    new_pos = New[new_peak][1]/self.width
                    # the difference between old and new for each peak
                    residual.append(np.linalg.norm(
                        np.array([1, old_pos])-np.array([new_height, new_pos])))
                comparison = np.array(residual) > comparison_thresh
                if type(comparison) == bool:  # happens if only 1 peak
                    if comparison == False:
                        self.regions *= 5
                else:
                    # if none of the peaks have changed by more than comparison_thresh fraction
                    if all(comparison) == False:
                        self.regions *= 5
                        if verbose:
                            print(
                                "peaks haven't changed significantly; regions increased")
            elif not len(self.peaks):  # if there wasn't a peak added, try harder
                self.regions *= 5

        # ---One last round of optimization for luck: can comment these in and out as you see fit.
#        self.optimize_bg()
#
        self.optimize_peaks_and_bg()
        self.optimize_heights()
        self.optimize_centre_and_width()
        self.optimize_peaks()
        
    def plot_result(self):
        '''
        plots the spectrum and the individual peaks, and their sums

        '''
        plt.figure()
        plt.plot(self.shifts, self.spec)
        plt.plot(self.shifts, self.bg)
        for peak in self.peaks:
            plt.plot(self.shifts, self.bg+self.line(*peak)*self.transmission)
        plt.plot(self.shifts, self.bg + self.multi_line(self.peaks)
                 * self.transmission, linestyle='--', color='k')


if __name__ == '__main__':
    from nplab.analysis.example_data import SERS_and_shifts
    spec = SERS_and_shifts[0]
    shifts = SERS_and_shifts[1]
    spec, shifts = split(spec, shifts, 190, np.inf)
    ff = FullFit(spec, shifts, lineshape='G', order=9)
    ff.run(verbose=True)
    ff.plot_result()
    plt.figure()
    plt.plot(ff.shifts, ff.signal)
    plt.plot(ff.shifts, ff.multi_line(ff.peaks))
