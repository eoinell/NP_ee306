# -*- coding: utf-8 -*-
"""
Created on Thu Aug 01 16:38:56 2019

@author: Hera
"""
from __future__ import print_function


if __name__ == '__main__':
    import os

    from nplab.instrument.camera.lumenera import LumeneraCamera
    from nplab.instrument.camera.camera_with_location import CameraWithLocation
    from nplab.instrument.stage.prior import ProScan

    os.chdir(r'C:\Users\np-albali\Documents\ee306')       
    
    cam = LumeneraCamera(1)
    stage = ProScan("COM9")
    CWL = CameraWithLocation(cam, stage)
    CWL.show_gui(blocking = False)
