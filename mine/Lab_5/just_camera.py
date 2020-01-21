# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 10:24:26 2019

@author: Hera
"""

if __name__ == '__main__': 
   
   from nplab.instrument.camera.lumenera import LumeneraCamera
   cam = LumeneraCamera(1)
   cam.show_gui(blocking = False)