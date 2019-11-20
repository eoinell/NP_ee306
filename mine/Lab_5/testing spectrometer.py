# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:59:23 2019

@author: Hera
"""

from nplab.instrument.camera.lumenera import LumeneraCamera
from nplab.instrument.stage.prior import ProScan
from nplab.instrument.spectrometer.seabreeze import OceanOpticsSpectrometer
from nplab.instrument.camera.camera_with_location import CameraWithLocation
from nplab.experiment.gui import  run_function_modally
from nplab.utils.array_with_attrs import ArrayWithAttrs
from nplab.instrument.spectrometer.Triax.Trandor_Lab5 import Trandor
from particle_tracking_app.particle_tracking_wizard import TrackingWizard
import time


#stage = ProScan("COM3", hardware_version = 2)
#cam = LumeneraCamera(1)
#
#
#CWL = CameraWithLocation(cam,stage)
#CWL.show_gui(blocking =False)
#stage.show_gui(blocking = False)

spec = OceanOpticsSpectrometer(0)
spec.show_gui(blocking = False)
trandor=Trandor()
trandor.exposure = 30
def busy_work(update_progress = lambda p:p):
    start = time.time()        
    elapsed = 0    
    
    while elapsed <30:
        elapsed = time.time()-start
        update_progress(elapsed)
def run():
    run_function_modally(busy_work, progress_maximum = 30)

def tr():
    trandor.Capture()
