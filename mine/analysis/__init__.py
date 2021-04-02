# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 14:20:47 2019

@author: Eoin Elliott
"""
import numpy as np
from pathlib import Path
import h5py
from scipy.ndimage import gaussian_filter

class Spectrum(np.ndarray):
    def __new__(cls, wavelengths, spectrum, *args, **kwargs):
        
        assert len(wavelengths) == np.shape(spectrum)[-1]
        
        obj = np.asarray(spectrum).view(cls)
        obj.wavelengths = np.asarray(wavelengths)
        return obj
        
    def __array_finalize__(self, obj):
        
        if obj is None: return 
        if not obj.shape:
            return np.array(obj)
        self.wavelengths = getattr(obj, 'wavelengths', np.arange(obj.shape[-1]))
    
    @classmethod
    def from_h5(cls, group):
        if type(group) is h5py.Group:
            attrs = group[0].attrs
        else:
            attrs = group.attrs
        ref = attrs.get('reference', 1)
        bg = attrs.get('background', 0)            
        return cls(group.attrs['wavelengths'], (group[()]-bg)/(ref-bg))
       
    def norm(self):
        return self.__class__(self.wl, self/self.ravel().max())
    
    @property
    def wl(self):
        return self.wavelengths
    
    @wl.setter
    def wl(self, value):
        self.wavelengths = np.array(value)
    
    def split(self, lower=-np.inf, upper=np.inf):
        if upper<lower: upper, lower = lower, upper
        condition = (lower <= self.wl) & (self.wl < upper) # <= allows recombination
     
        return self.__class__(self.wl[condition], self.T[condition].T) 
    
    def squash(self):
        return self.__class__(self.wl, self.sum(axis=0))

class SERS(Spectrum):
    center_wavelength=632.8
    @property
    def shifts(self):
        return (1./(self.wl*1e-9) - 1./(self.centre_wavelength*1e-9))/100
    
    
    
def latest_scan(file):
    return file[max(file, key=lambda x: int(x.split('_')[-1]) 
            if x.startswith('ParticleScannerScan') else 0)]

def split(wavelengths, spectrum, lower=-np.inf, upper=np.inf):
    ''' a more concise truncate'''
    
    if upper<lower: upper, lower = lower, upper
    spectrum, wavelengths = map(np.array, [spectrum, wavelengths])
    condition = (lower <= wavelengths) & (wavelengths < upper) # <= allows recombination
    return wavelengths[condition], spectrum.T[condition].T     

def load_h5(location='.'):
    '''return the latest h5 in a given directory. If location is left blank,
    loads the latest file in the current directory.'''
    
    path = Path(location) 
    return h5py.File(path / max(f for f in path.iterdir() if f.suffix == '.h5'), 'r')

def condense_z_scan(h5group):
    '''
    

    Parameters
    ----------
    h5group : h55py.Group
        A spectrum with background, reference and wavelength attributes

    Returns
    -------
    tuple
        (ndarry of wavelengths, ndarray of referenced, bg subtracted spectrum)

    '''
    bg = h5group.attrs['background']
    ref = h5group.attrs['reference']
    spec = h5group[()]
    return h5group.attrs['wavelengths'], ((spec-bg)/(ref-bg)).max(axis=0)



def norm(array):
    '''divide a spectrum by it's highest value to compare spectra of different 
    intensity'''
    array = np.array(array)
    return array/array.ravel().max()        

def remove_cosmic_ray(spectrum, thresh=5, smooth=30): 
    g = gaussian_filter(spectrum, smooth)
    n = spectrum/g
    noise = np.sqrt(np.var(n))
    mean_noise = n.mean()
    spikes = np.arange(len(n))[n>mean_noise+(thresh*noise)]
    rays = set()
    for spike in spikes:
        for side in (-1, 1):
            step=0
            while 0<(coord := spike+(side*step))<len(n)-1:
                step +=1
                
                if n[coord] > mean_noise + noise:

                    rays.add(coord)
                else:
                    break
    
    rays = list(rays)
    rough = np.copy(spectrum)
    rough[rays] = gaussian_filter(rough, smooth)[rays]
    rough = gaussian_filter(rough, smooth)
    cleaned = np.copy(spectrum)
    cleaned[rays] = rough[rays]  
    return cleaned


def sweep_to_angles(group, prefix='', condenser=lambda x: x):
    angles = []
    data = []
    for name, value in group.items():
        if name.startswith(prefix): 
            angles.append(float(name.split('_')[-1]))
            data.append(condenser(value[()]))
            attrs = dict(value.attrs)
    return *map(np.array, zip(*sorted(zip(angles, np.array(data))))), attrs
    sort = np.argsort(angles)
    return np.array(angles)[sort], np.array(data)[sort], attrs
    

def flatten(data):
    return np.sum(data, axis=0)

def vis(arr):
    return float((arr.max()-arr.min())/(arr.max()+arr.min()))