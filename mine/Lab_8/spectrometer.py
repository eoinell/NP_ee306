# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:54:12 2020

@author: ee306
"""

from nplab.instrument.spectrometer.seabreeze import OceanOpticsSpectrometer
spec = OceanOpticsSpectrometer(0)
spec.show_gui(blocking = False)
