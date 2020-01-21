# -*- coding: utf-8 -*-
"""
Created on Thu Aug 01 16:38:56 2019

@author: ee306
"""
import time
from nplab.utils.gui_generator import GuiGenerator
from nplab import datafile
from mine.Lab_6.setup_gui import Lab
from particle_tracking_app.particle_tracking_wizard import TrackingWizard


class PT_lab(Lab):
    def __init__(self, lab_dict):
        
        super(PT_lab, self).__init__(lab_dict)
        track_dict ={'spectrometer' : self.spec,
                     'alinger' : self.aligner
                     }
        self.wizard = TrackingWizard(self.CWL,track_dict, task_list = ['Lab.thumb_focus','CWL.thumb_image','alinger.z_scan', 'Lab.SERS'])
        self.wizard.show()
        
    def thumb_focus(self):
            self.CWL.autofocus(use_thumbnail = True)
    
    def SERS(self):
        '''Save returned'''
        self.wutter.close_shutter()        
        self.lutter.open_shutter()
        time.sleep(0.2)    
        to_save = self.shamdor.raw_snapshot()[-1:]
        to_save.attrs['x_axis'] = self.shamdor.x_axis
        to_save.attrs['wavelengths'] = self.shamdor.x_axis
        self.wutter.open_shutter()        
        self.lutter.close_shutter()
        return to_save 
       
if __name__ == '__main__':
    import os
    from nplab.instrument.spectrometer.seabreeze import OceanOpticsSpectrometer
    from nplab.instrument.camera.lumenera import LumeneraCamera
    from nplab.instrument.camera.camera_with_location import CameraWithLocation
    from nplab.instrument.spectrometer.spectrometer_aligner import SpectrometerAligner
    from nplab.instrument.stage.prior import ProScan
    from nplab.instrument.shutter.BX51_uniblitz import Uniblitz
    from nplab.instrument.spectrometer.shamdor import Shamdor
    from nplab.instrument.camera.Andor import Andor
    from nplab.instrument.shutter.thorlabs_sc10 import ThorLabsSC10
    from nplab.instrument.stage.Thorlabs_FW212C import FW212C   

    os.chdir(r'C:\Users\hera\Documents')       
    spec = OceanOpticsSpectrometer(0) 
    lutter = ThorLabsSC10('COM1')
    lutter.set_mode(1)
    wutter = Uniblitz("COM10")
    cam = LumeneraCamera(1)
    stage = ProScan("COM4")
    CWL = CameraWithLocation(cam, stage)
    shamdor = Shamdor(Andor)
    filter_wheel = FW212C()
    alinger = SpectrometerAligner(spec,stage)
    
    #=================================================#
    
    lab_dict = {'spectrometer' : spec,
                'laser_shutter' : lutter,
                'white_shutter' : wutter,
                'camera' : cam,
                'CWL' : CWL,
                'shamrock' : shamdor.shamrock,
                'shamdor' : shamdor,
                'power_wheel' : filter_wheel}
    lab = PT_lab(lab_dict)
    
    #=================================================#
    
    File = datafile.current()
    
    #=================================================#
    
    gui_equipment_dict = {'Lab' : lab,
                          'spectrometer' : spec,
                          'laser_shutter' : lutter,
                          'white_shutter' : wutter,
                          'Camera' : cam,
                          'CWL' : CWL,
                          'shamrock' : shamdor.shamrock,
                          'andor' : shamdor,
                          'power_wheel' : filter_wheel}
    
    gui = GuiGenerator(gui_equipment_dict, dock_settings_path = r'C:\Users\hera\Documents\GitHub\NP_ee306\mine\Lab_6\config.npy',
                       scripts_path = r'C:\Users\hera\Documents\GitHub\NP_ee306\mine\Lab_6')                                                 
    gui.show()
    
    #=================================================#
    
    
    
