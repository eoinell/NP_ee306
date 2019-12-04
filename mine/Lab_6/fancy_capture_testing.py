# -*- coding: utf-8 -*-
"""
Created on Thu Aug 01 16:38:56 2019

@author: ee306

"""
from __future__ import print_function

import time
import winsound
from nplab import datafile


def fancy_capture(Lab):
        '''
        Takes a spectrum on the shamdor, but turns off the white light and turns on the laser first, 
        then restores the instrument to its initial state
        '''
        wutter_open = Lab.wutter.is_open()
        lutter_closed = Lab.lutter.is_closed()
        
        if wutter_open: Lab.wutter.close_shutter()
        if lutter_closed: Lab.lutter.toggle() # toggle is more efficient than open/close
        
#        time.sleep(0.1)
        Lab.shamdor.raw_image(update_latest_frame=True)
        
        if wutter_open: Lab.wutter.open_shutter()
        if lutter_closed: Lab.lutter.toggle()



fancy_capture()
