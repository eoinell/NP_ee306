# -*- coding: utf-8 -*-
"""
Created on Thu Aug 01 16:38:56 2019

@author: Hera
"""
import sys

import numpy as np
from scipy import interpolate 
import time
from qtpy import QtWidgets, uic
from nplab.utils.gui_generator import GuiGenerator
from nplab.ui.ui_tools import UiTools
from nplab.experiment.gui import run_function_modally
from nplab.instrument import Instrument
from scipy.interpolate import UnivariateSpline
import winsound

class Lab:
    '''
    meta-instrument for all the equipment in Lab 6. Works analogously to CWL in many respects. 
    '''
    def __init__(self, equipment_dict, parent = None):     
        self.laser = '_785' 
        self.initiate_all(equipment_dict)
        #self.equipment_dict = {'Exp':self, 'spec':self.spec, 'lutter':self.lutter, 'wutter':self.wutter, 'pometer':self.pometer, 'CWL':self.CWL, 'shamdor':self.shamdor}         
                        
        #self.anglez = np.linspace(self.minangle , self.maxangle, num = 50, endpoint = True)
        self.lutter.close_shutter()
        self.wutter.open_shutter()         

    def initiate_all(self, ed):
        self.init_spec = False
        self.init_lutter = False
        self.init_cam = False
        self.init_CWL = False
        self.init_wutter = False
        self.init_shamrock = False
        self.init_andor = False
        self.init_aligner = False                
               
        self._initiate_spectrometer(ed['spec'])
        self._initiate_lutter(ed['laser shutter'])
        self._initiate_cam(ed['cam'])        
        self._initiate_CWL(ed['CWL'])
        self._initiate_wutter(ed['white shutter'])
        self._initiate_shamrock(ed['shamrock'])
        self._initiate_andor(ed['andor'])
        self._initiate_aligner()

    def _initiate_spectrometer(self, instrument):
        if self.init_spec is True:        
            print 'Spectrometer already initialised'
        else:
            self.spec = instrument
            self.init_spec = True
    def _initiate_lutter(self, instrument):
        if self.init_lutter is True:
            print 'Laser shutter already initialised'
        else:
            self.lutter = instrument
            self.lutter.set_mode(1)
            self.init_lutter = True        
 

    def _initiate_cam(self, instrument):
        if self.init_cam is True:
            print 'Camera already initalised'
        else:
            self.cam = instrument
            self.cam.exposure=800.
            self.cam.gain = 20.
            self.init_cam = True
    def _initiate_CWL(self, instrument):
        if self.init_CWL is True:
            print 'Camera with location already initialised'
        else:
            self.CWL = instrument
            self.CWL.load_calibration()
            self.init_CWL = True
    def _initiate_wutter(self, instrument):    
        if self.init_wutter is True :           
            print 'White light shutter already initialised'
        else:            
            self.wutter = instrument
            self.wutter.close_shutter()            
            self.wutter.open_shutter()
            self.init_wutter = True
    def _initiate_shamrock(self, instrument):
        if self.init_shamrock is True:
            'Print Shamrock already initialised'
        else:            
            self.shamrock = instrument             
            self.shamrock_centre_wl = 700            
            self.shamrock.HSSpeed=2
            self.shamrock.SetSlit(100)

#            self.shamdor.CoolerON()
            self.shamrock.center_wavelength = 700
#            self.shamrock.ShamrockSetPixelWidth(16)
#            self.shamrock.ShamrockSetNumberPixels(1600)
        
            #set the centre wavelengths up
#            self.shamrock.ShamrockGetWavelengthLimits()
#           
            self.init_shamrock = True
    def _initiate_andor(self, instrument):
        if self.init_andor is True:
            'Print Andor already initialised'
        else:            
            self.andor = instrument
 
            self.andor_exposure = 1               
            self.andor.HSSpeed=2
          
            self.andor.SetTemperature = -90
#            self.shamdor.CoolerON()
            self.andor.Exposure= 1

            self.andor.ReadMode=3
            self.andor.SingleTrack = (100, 30)
            self.andor.AcquisitionMode=3
#            self.shamdor.ShamrockSetPixelWidth(16)
#            self.shamdor.ShamrockSetNumberPixels(1600)
        
            #set the centre wavelengths up
#            self.shamdor.ShamrockGetWavelengthLimits()
#            self.spinbox_centrewl.setRange(self.shamdor.shamrock.wl_limits[0],
#                                           self.shamdor.shamrock.wl_limits[1])
            self.init_andor = True
    def _initiate_aligner(self):
        if self.init_aligner is True:                 
            'Spectrometer aligner already initiated'
        else:
            
            self.aligner = SpectrometerAligner(self.spec, self.CWL.stage)
            self.init_aligner = True
    def example(self):
        print "I'm an example"
        winsound.Beep(1000,500)
    def get_qt_ui(self):
        return Lab_gui(self)

class Lab_gui(QtWidgets.QWidget,UiTools):
    def __init__(self, lab):
        super(Lab_gui, self).__init__()
        uic.loadUi(r'C:\Users\np-albali\Documents\GitHub\NP_ee306\mine\Lab_6\setup_gui.ui', self)
        self.Lab = lab 
        self.SetupSignals()
    def SetupSignals(self): 
        self.example_pushButton.clicked.connect(self.Lab.example)
       
        
  
  
if __name__ == '__main__': 
    import os
    import visa
    from nplab.instrument.spectrometer.seabreeze import OceanOpticsSpectrometer
    from nplab.instrument.camera.lumenera import LumeneraCamera
    from nplab.instrument.camera.camera_with_location import CameraWithLocation
    from nplab.instrument.spectrometer.spectrometer_aligner import SpectrometerAligner
    from nplab.instrument.stage.prior import ProScan
    from nplab.instrument.shutter.BX51_uniblitz import Uniblitz
    from nplab.instrument.spectrometer.shamrock import Shamrock
    from nplab.instrument.camera.Andor import Andor
    from nplab import datafile
    from nplab.instrument.shutter.thorlabs_sc10 import ThorLabsSC10
    from particle_tracking_app.particle_tracking_wizard import TrackingWizard

    os.chdir(r'C:\Users\np-albali\Documents')    
    app = QtWidgets.QApplication(sys.argv)    
    rm= visa.ResourceManager()
    
    spec = OceanOpticsSpectrometer(0) 
    lutter = ThorLabsSC10('COM1')
    lutter.set_mode(1)

    wutter = Uniblitz("COM7")
    cam = LumeneraCamera(1)
    
    stage = ProScan("COM9")
    CWL = CameraWithLocation(cam, stage)
    
    shamrock = Shamrock()
    andor = Andor()
    
    equipment_dict = {'spec' : spec,
                    'laser shutter' : lutter,
                    'white shutter' : wutter,
                    'cam' : cam,
                    'CWL' : CWL,
                    'shamrock' : shamrock,
                    'andor' : andor}
                 
    File = datafile.current()
    lab = Lab(equipment_dict)
    equipment_dict['lab'] = lab    
    gui = GuiGenerator(equipment_dict, scripts_path= r'C:\Users\np-albali\Documents\GitHub\NP_ee306\mine\Lab_6',
                       dock_settings_path = r'C:\Users\np-albali\Documents\GitHub\NP_ee306\mine\Lab_6\config.npy')
                                                 
    gui.show()
        
        
