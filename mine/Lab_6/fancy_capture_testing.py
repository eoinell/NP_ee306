# -*- coding: utf-8 -*-
"""
Created on Thu Aug 01 16:38:56 2019

@author: ee306

"""
from __future__ import print_function

import time
import winsound
from nplab import datafile


def fancy_capture():
        '''
        Takes a spectrum on the shamdor, but turns off the white light and turns on the laser first, 
        then restores the instrument to its initial state
        '''
        start = time.time()        
        
        wutter_open = Lab.wutter.is_open()
        wrtime = time.time()
        lutter_closed = Lab.lutter.is_closed()
        lrtime =time.time()
        if wutter_open: Lab.wutter.close_shutter()
        wtime = time.time()
        if lutter_closed: Lab.lutter.toggle()         # toggle is more efficient than open/close
        ltime = time.time() 
#        time.sleep(0.1)
        Lab.shamdor.raw_image(update_latest_frame=True)
        ctime = time.time() 
        if lutter_closed: Lab.lutter.toggle()
        l2time = time.time()
        if wutter_open: Lab.wutter.open_shutter()
        w2time = time.time() 
        to_print = [start,wrtime, lrtime, wtime, ltime, ctime, l2time, w2time]
        for i,p in enumerate(to_print):
            if i!=0:
                print(p- to_print[i-1])
fancy_capture()

