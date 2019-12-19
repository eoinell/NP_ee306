# -*- coding: utf-8 -*-
"""
Created on Thu Aug 01 16:38:56 2019

@author: Hera
"""
from __future__ import print_function


if __name__ == '__main__':
    import os 
    from nplab.instrument.spectrometer.shamdor import Shamdor 

    os.chdir(r'C:\Users\np-albali\Documents\ee306')       
    
    shamdor = Shamdor()
    shamdor.show_gui(blocking = False)