# -*- coding: utf-8 -*-
"""
Created on Thu Aug 01 16:38:56 2019

@author: Hera
"""
import sys
import numpy as np
import time
from qtpy import QtWidgets, uic
from nplab.utils.gui_generator import GuiGenerator
from nplab.ui.ui_tools import UiTools
from nplab.experiment.gui import run_function_modally
from nplab.instrument import Instrument
from nplab.instrument.spectrometer.shamdor import Shamdor
from nplab import datafile
from nplab.utils.notified_property import DumbNotifiedProperty

def thumb_focus():
        CWL.autofocus(use_thumbnail = True)        
    

def SERS():
    wutter.close_shutter()        
    lutter.open_shutter()
    time.sleep(0.2)    
    dump, to_save = shamdor.raw_snapshot()
    to_save.attrs['x_axis'] = shamdor.x_axis
    to_save.attrs['wavelengths'] = shamdor.x_axis
    wizard.particle_group.create_dataset('SERS', data = to_save, attrs = attrs)
    wutter.open_shutter()        
    lutter.close_shutter()
    return to_save 

if __name__ == '__main__':
    import os
    from nplab.instrument.spectrometer.seabreeze import OceanOpticsSpectrometer
    from nplab.instrument.camera.lumenera import LumeneraCamera
    from nplab.instrument.camera.camera_with_location import CameraWithLocation
    from nplab.instrument.spectrometer.spectrometer_aligner import SpectrometerAligner
    from nplab.instrument.stage.prior import ProScan
    from nplab.instrument.shutter.BX51_uniblitz import Uniblitz
    from nplab.instrument.spectrometer.shamrock import Shamrock
    from nplab.instrument.camera.Andor import Andor
    from nplab.instrument.shutter.thorlabs_sc10 import ThorLabsSC10
    from nplab.instrument.stage.Thorlabs_FW212C import FW212C   
    from particle_tracking_app.particle_tracking_wizard import TrackingWizard
    from mine.Lab_6.setup_gui import Lab

    os.chdir(r'C:\Users\np-albali\Documents')       
    spec = OceanOpticsSpectrometer(0) 
    lutter = ThorLabsSC10('COM1')
    lutter.set_mode(1)
    wutter = Uniblitz("COM7")
    cam = LumeneraCamera(1)
    stage = ProScan("COM9")
    CWL = CameraWithLocation(cam, stage)
    shamdor = Shamdor(Andor)
    filter_wheel = FW212C()
    alinger = SpectrometerAligner(spec,stage)
    
    #=================================================#
    
    track_dict = {'spectrometer':spec,
                  'alinger':alinger, 
                  'shamdor' : shamdor}
    File = datafile.current()
    wizard = TrackingWizard(CWL,track_dict,task_list = ['thumb_focus','CWL.thumb_image','alinger.z_scan', 'SERS'])
    wizard.data_file.show_gui(blocking = False)
    wizard.show(blocking = False)
    
    #=================================================#
    
    lab_dict = {'spectrometer' : spec,
                'laser_shutter' : lutter,
                'white_shutter' : wutter,
                'camera' : cam,
                'CWL' : CWL,
                'shamrock' : shamdor.shamrock,
                'shamdor' : shamdor,
                'power_wheel' : filter_wheel}
    lab = Lab(lab_dict)
    
    #=================================================#
    
    gui_equipment_dict = {'Lab' : lab,
                          'spectrometer' : spec,
                          'laser_shutter' : lutter,
                          'white_shutter' : wutter,
                          'Camera' : cam,
                          'CWL' : CWL,
                          'shamrock' : shamdor.shamrock,
                          'andor' : shamdor}
    
    gui = GuiGenerator(gui_equipment_dict, dock_settings_path = r'C:\Users\np-albali\Documents\GitHub\NP_ee306\mine\Lab_6\config.npy',
                       scripts_path = r'C:\Users\np-albali\Documents\GitHub\NP_ee306\mine\Lab_6')                                                 
    gui.show()
    
    #=================================================#
    
    
    
