# -*- coding: utf-8 -*-
"""
Created on Thu Aug 01 16:38:56 2019

@author: Hera
"""
from __future__ import print_function
import sys
import numpy as np
import time
from qtpy import QtWidgets, uic
from nplab.utils.gui_generator import GuiGenerator
from nplab.ui.ui_tools import UiTools
from nplab.experiment.gui import run_function_modally
from nplab.instrument import Instrument
from nplab.instrument.spectrometer.shamdor import Shamdor
from nplab.instrument.spectrometer.spectrometer_aligner import SpectrometerAligner
from nplab import datafile
import winsound

class Lab(Instrument):
    '''
    meta-instrument for all the equipment in Lab 6. Works analogously to CWL in many respects.
    Takes care of data handling, use the create_dataset, create_group functions. 
    Keeps track of all their states. Functions which will be called by buttons should be put in here
    Each instrument should have its own gui though!
    '''
    def __init__(self, equipment_dict, parent = None):     
        self.initiate_all(equipment_dict)
        self.lutter.close_shutter()
        self.wutter.open_shutter()  
        self.steps = 5 
        self.laser = '_633'
        Instrument.__init__(self) 
    
    def initiate_all(self, ed):
        self.init_spec = False
        self.init_lutter = False
        self.init_cam = False
        self.init_CWL = False
        self.init_wutter = False
        self.init_shamrock = False
        self.init_shamdor = False
        self.init_aligner = False
        self.init_power_wheel = False                
               
        self._initiate_spectrometer(ed['spectrometer'])
        self._initiate_lutter(ed['laser_shutter'])
        self._initiate_cam(ed['camera'])        
        self._initiate_CWL(ed['CWL'])
        self._initiate_wutter(ed['white_shutter'])
        self._initiate_shamrock(ed['shamrock'])
        self._initiate_shamdor(ed['shamdor'])
        self._initiate_power_wheel(ed['power_wheel'])
        self._initiate_aligner()

    def _initiate_spectrometer(self, instrument):
        if self.init_spec is True:        
            print('Spectrometer already initialised')
        else:
            self.spec = instrument
            self.init_spec = True
            
    def _initiate_lutter(self, instrument):
        if self.init_lutter is True:
            print('Laser shutter already initialised')
        else:
            self.lutter = instrument
            self.lutter.set_mode(1)
            self.init_lutter = True        

    def _initiate_cam(self, instrument):
        if self.init_cam is True:
            print('Camera already initalised')
        else:
            self.cam = instrument
            self.cam.exposure=200.
            self.cam.gain = 20.
            self.init_cam = True
    def _initiate_CWL(self, instrument):
        if self.init_CWL is True:
            print('Camera with location already initialised')
        else:
            self.CWL = instrument
            try: self.CWL.load_calibration()
            except: print('Stage not XY calibrated')
            self.init_CWL = True
    def _initiate_wutter(self, instrument):    
        if self.init_wutter is True :           
            print('White light shutter already initialised')
        else:            
            self.wutter = instrument
            self.wutter.close_shutter()            
            self.wutter.open_shutter()
            self.init_wutter = True
    def _initiate_shamrock(self, instrument):
        
        if self.init_shamrock is True:
            print('Shamrock already initialised')
        else:            
            self.shamrock = instrument             
            self.shamrock_centre_wl = 650            
            self.shamrock.HSSpeed=2
            self.shamrock.SetSlit(100)
            self.shamrock.center_wavelength = 650
            self.shamrock.pixel_number = 1600
            self.shamrock.pixel_width = 16
            self.init_shamrock = True
    def _initiate_shamdor(self, instrument):
        if self.init_shamdor is True:
            print('shamdor already initialised')
        else:            
            self.shamdor = instrument
 
            self.shamdor_exposure = 1               
            self.shamdor.HSSpeed=2
          
            self.shamdor.SetTemperature = -90
            self.shamdor.cooler = True
            self.shamdor.Exposure= 1

            self.shamdor.ReadMode=3
            self.shamdor.SingleTrack = (100, 30)
            self.shamdor.AcquisitionMode = 3
   
            self.init_shamdor = True
    def _initiate_aligner(self):
        if self.init_aligner is True:                 
            print('Spectrometer aligner already initiated')
        else:
            
            self.aligner = SpectrometerAligner(self.spec, self.CWL.stage)
            self.init_aligner = True
    def _initiate_power_wheel(self, instrument):
        if self.init_power_wheel is True:
            print('Power Wheel already initialised')
        else:            
            self.power_wheel = instrument
            self.power_wheel.setPosition(1)
            
    def get_laser(self):
        return self.laser
    
    def set_laser(self, Laser):
        self.shamdor.laser = Laser
        
    laser = property(get_laser, set_laser)
    
    
    def fancy_capture(self):
        '''
        Takes a spectrum on the shamdor, but turns off the white light and turns on the laser first, 
        then restores the instrument to its initial state
        '''
        wutter_open = self.wutter.is_open()
        lutter_closed = self.lutter.is_closed()
        
        if wutter_open: self.wutter.close_shutter()
        if lutter_closed: self.lutter.toggle() # toggle is more efficient than open/close
        
#        time.sleep(0.1)
        self.shamdor.raw_image(update_latest_frame=True)
        
        if wutter_open: self.wutter.open_shutter()
        if lutter_closed: self.lutter.toggle()

    def example(self):
        print("I'm an example")
        winsound.Beep(100,500)
        group = self.create_data_group('example_group')
        group.create_dataset('example', data = [0,1,2,3])
    def modal_example(self, steps = None, update_progress = lambda p:p):
        '''
        update_progress is a function that simply returns its argument.
        run_function_modally uses this to tell how far along the function is.
        '''
        if steps == None: steps = self.steps
        else: steps = 5
        frequencies = np.linspace(100, 500, num = steps)
        for counter, frequency in enumerate(frequencies):
            winsound.Beep(int(frequency), 500)
            update_progress(counter)

    def get_qt_ui(self):
        return Lab_gui(self)

class Lab_gui(QtWidgets.QWidget,UiTools):
    def __init__(self, lab):
        super(Lab_gui, self).__init__()
        uic.loadUi(r'C:\Users\np-albali\Documents\GitHub\NP_ee306\mine\Lab_6\setup_gui.ui', self)
        self.Lab = lab
        self.group_name = 'particle_%d' 
        self.gui_current_group = None
        self.SetupSignals()
    def SetupSignals(self): 
        self.fancy_capture_pushButton.clicked.connect(self.Lab.fancy_capture)
        self.set_power_pushButton.clicked.connect(self.set_power_gui)        
        self.group_name_lineEdit.textChanged.connect(self.update_group_name)
        self.create_group_pushButton.clicked.connect(self.create_data_group_gui)
        self.use_created_group_checkBox.stateChanged.connect(self.update_use_current_group)
        self.example_pushButton.clicked.connect(self.Lab.example)
        self.modal_example_pushButton.clicked.connect(self.modal_example_gui)
        self.steps_spinBox.valueChanged.connect(self.update_steps)
        self.use_shifts_checkBox.stateChanged.connect(self.update_use_shifts)
        self._633_radioButton.clicked.connect(self.update_laser_gui)
        self._785_radioButton.clicked.connect(self.update_laser_gui)
    def set_power_gui(self):
        self.Lab.power_wheel.setPosition(self.power_wheel_spinBox.value())
    def update_group_name(self):
        self.group_name = self.group_name_lineEdit.text() 
    def create_data_group_gui(self):
        init_use_cur = datafile._use_current_group
        datafile._use_current_group = False
        self.gui_current_group = self.Lab.create_data_group(self.group_name)
        if self.use_created_group_checkBox.checkState(): 
            datafile._current_group = self.gui_current_group
        if init_use_cur: datafile._use_current_group = True
    def update_use_current_group(self):
        if self.use_created_group_checkBox.checkState():
            datafile._use_current_group = True
            try:datafile._current_group = self.gui_current_group
            except: print('No created group (yet)!')
        else:
            datafile._use_current_group = False  
    def update_use_shifts(self):
        self.Lab.shamdor.use_shifts = self.use_shifts_checkBox.checkState()
    def update_laser_gui(self):
        if self._633_radioButton.isChecked():     
            self.Lab.laser  = '_633'
        if self._785_radioButton.isChecked(): 
            self.Lab.laser = '_785'   
    def modal_example_gui(self):
        '''
        running a function modally produces a progress bar, and takes care of threading stuff for you to keep the GUI responsive
        see nplab for details.
        '''
        run_function_modally(self.Lab.modal_example, progress_maximum = self.Lab.steps+1)
    def update_steps(self):
        self.Lab.steps = self.steps_spinBox.value()

if __name__ == '__main__':
    import os
    from nplab.instrument.spectrometer.seabreeze import OceanOpticsSpectrometer
    from nplab.instrument.camera.lumenera import LumeneraCamera
    from nplab.instrument.camera.camera_with_location import CameraWithLocation
    from nplab.instrument.stage.prior import ProScan
    from nplab.instrument.shutter.BX51_uniblitz import Uniblitz
    from nplab.instrument.shutter.thorlabs_sc10 import ThorLabsSC10
    from nplab.instrument.stage.Thorlabs_FW212C import FW212C   

    os.chdir(r'C:\Users\np-albali\Documents')       
    spec = OceanOpticsSpectrometer(0) 
    lutter = ThorLabsSC10('COM1')
    lutter.set_mode(1)
    wutter = Uniblitz("COM7")
    cam = LumeneraCamera(1)
    stage = ProScan("COM9")
    CWL = CameraWithLocation(cam, stage)
    shamdor = Shamdor(use_shifts = False)
    filter_wheel = FW212C()
    equipment_dict = {'spectrometer' : spec,
                    'laser_shutter' : lutter,
                    'white_shutter' : wutter,
                    'camera' : cam,
                    'CWL' : CWL,
                    'shamrock' : shamdor.shamrock,
                    'shamdor' : shamdor,
                    'power_wheel' : filter_wheel}
                 
    File = datafile.current()
    lab = Lab(equipment_dict)
    gui_equipment_dict = {'Lab' : lab,
                          'spectrometer' : spec,
                          'laser_shutter' : lutter,
                          'white_shutter' : wutter,
                          'Camera' : cam,
                          'CWL' : CWL,
                          'shamrock' : shamdor.shamrock,
                          'andor' : shamdor,
                          'filter wheel' :  filter_wheel}
    
    gui = GuiGenerator(gui_equipment_dict, dock_settings_path = r'C:\Users\np-albali\Documents\GitHub\NP_ee306\mine\Lab_6\config.npy',
                       scripts_path = r'C:\Users\np-albali\Documents\GitHub\NP_ee306\mine\Lab_6')                                                 
    gui.show()
        
        
