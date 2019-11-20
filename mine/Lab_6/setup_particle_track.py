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
    equipment_dict = {'spectrometer':spec,
                      'alinger':alinger, 
                      'shamdor' : shamdor}

    def thumb_focus():
        CWL.autofocus(use_thumbnail = True)        
    

    def SERS():
        dump, to_save = shamdor.raw_snapshot()
        attrs = to_save.attrs
        attrs['wavelengths'] = attrs['x_axis']
        wizard.particle_group.create_dataset('SERS', data = to_save, attrs = attrs)
        return to_save 

    shamdor.show_gui(blocking = False)
    shamdor.shamrock.show_gui(blocking = False)
    CWL.show_gui(blocking = False)
    spec.show_gui(blocking = False)
    
    wizard = TrackingWizard(CWL,equipment_dict,task_list = ['thumb_focus','CWL.thumb_image','alinger.z_scan', 'SERS'])
    wizard.data_file.show_gui(blocking = False)
    wizard.show()
    
    if dump==dump:pass   
        
