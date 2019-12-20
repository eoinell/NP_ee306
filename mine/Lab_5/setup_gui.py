# -*- coding: utf-8 -*-
"""
Created on Thu Aug 01 16:38:56 2019

@author: ee306
"""
import sys

import numpy as np
import time
from qtpy import QtWidgets, uic
import matplotlib.pyplot as plt
from nplab.utils.gui_generator import GuiGenerator
from nplab.ui.ui_tools import UiTools
from scipy.interpolate import UnivariateSpline
from nplab.experiment.gui import run_function_modally
from nplab.instrument import Instrument
from nplab import datafile
from mine.Lab_5.power_control import PowerControl

 
def laser_merit(im):
    merit = 1
    im = np.sum(im, axis = 2)
    x_len, y_len = np.shape(im)
    xindices = np.arange(x_len/2 - 3, x_len/2 +3)
    x_slice = np.mean(np.take(im, xindices, axis = 0), axis = 0)
    spl = UnivariateSpline(range(len(x_slice)), x_slice - max(x_slice)/3)
    roots = spl.roots()
    try: merit = 1/(max(roots)-min(roots))
    except: merit = 0
    return merit
    
class Lab(Instrument):
    '''
    meta-instrument for all the equipment in Lab 5. Works analogously to CWL in many respects. 
    '''
    def __init__(self, equipment_dict, parent = None):       
        self.laser = '_785' 
        self.initiate_all(equipment_dict)        
        self.power_series_name = 'particle_%d'
        self.loop_down = False
        self.steps = 5
        self.max_nkin = 10        
        
        self.pc.update_power_calibration()
        self.lutter.close_shutter()
        self.wutter.open_shutter()  
        Instrument.__init__(self)

    def initiate_all(self, ed):
        self.init_spec = False
        self.init_lutter = False
        self.init_pometer = False
        self.init_cam = False
        self.init_CWL = False
        self.init_wutter = False
        self.init_trandor = False
        self.init_aligner = False                
        self.init_pc = False               
        
        self._initiate_spectrometer(ed['spec'])
        self._initiate_lutter(ed['lutter'])
        self._initiate_pometer(ed['pometer'])
        self._initiate_cam(ed['cam'])        
        self._initiate_CWL(ed['CWL'])
        self._initiate_wutter(ed['wutter'])
        self._initiate_trandor(ed['trandor'])
        self._initiate_pc(ed['pc'])
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

    def _initiate_pometer(self, instrument):
        if self.init_pometer:
            print 'Power meter already initialised'
        else:            
            self.pometer = instrument
            self.pometer.system.beeper.immediate()
            if self.laser == '_785': self.pometer.sense.correction.wavelength = 785   
            if self.laser == '_633': self.pometer.sense.correction.wavelength = 633              
            self.init_pometer = True
    def _initiate_cam(self, instrument):
        if self.init_cam:
            print 'Camera already initalised'
        else:
            self.cam = instrument
            self.cam.exposure=800.
            self.cam.gain = 20.
            self.init_cam = True
    def _initiate_CWL(self, instrument):
        if self.init_CWL:
            print 'Camera with location already initialised'
        else:            
            self.CWL = instrument
#            try: self.CWL.load_calibration()
#            except: print('stage xy calibration not found')
            self.init_CWL = True
    def _initiate_wutter(self, instrument):    
        if self.init_wutter:           
            print 'White light shutter already initialised'
        else:            
            self.wutter = instrument
            self.wutter.close_shutter()            
            self.wutter.open_shutter()
            self.init_wutter = True
    def _initiate_trandor(self, instrument):
        if self.init_trandor:
            print 'Triax and Andor already initialised'
        else:            
            self.trandor = instrument
            self.andor_gui = self.trandor.get_qt_ui()     
            self.trandor_exposure = 1        
            self.trandor_centre_wl = 785            
            self.trandor.HSSpeed=2
            self.trandor.Grating(1)
            self.trandor.triax.Slit(100)
            self.trandor.SetTemperature = -90
            self.trandor.CoolerON()
            self.trandor.Exposure=self.trandor_exposure
            self.trandor.Set_Center_Wavelength(785)
            self.trandor.ReadMode=3
            self.trandor.SingleTrack=(100,30)
            self.trandor.AcquisitionMode=3
            self.trandor.SetParameter('NKin', 1)
            self.init_trandor = True
    def _initiate_aligner(self):
        if self.init_aligner:                 
            print 'Spectrometer aligner already initialised'
        else:
            
            self.aligner = SpectrometerAligner(self.spec, self.CWL.stage)
            self.init_aligner = True
    
    def _initiate_pc(self, instrument):
        if self.init_pc is True:                 
            print 'power controller already initialised'
        else:
            self.pc = instrument
            self.init_pc = True
    
    def Power_Series(self,
                     tick_this_box = False,
                     focus_with_laser = True,
                     update_progress=lambda p:p):        
        self.pc.update_power_calibration()  # necessary if changed lasers      
        
        if not tick_this_box:
            self.create_data_group(self.power_series_name)
            group = datafile._current_group 
        else:
            group = self.wizard.particle_group             

        if self.maxpower is None: 
            maxpower = max(np.array(self.power_calibration['real_powers']))
            minpower = 0.2*maxpower
        else:
            maxpower = self.pc.maxpower
            minpower = self.pc.minpower    
        powers_up = np.linspace(minpower,maxpower, num = self.steps, endpoint = True)
        powers_down = []                    
        if self.loop_down: powers_down = powers_up[:-1][::-1]
        self.Powers = np.append(powers_up, powers_down)

        kinetic_fac =   self.max_nkin*float(minpower)

        attrs = self.trandor.metadata
        attrs['x_axis']=np.flipud(attrs['x_axis'])
        attrs['wavelengths'] = attrs['x_axis']
        attrs['AcquisitionTimes'] = self.trandor.AcquisitionTimings
        attrs['powers']=self.Powers
        
        if focus_with_laser: self.focus_with_laser() 
        self.lutter.close_shutter()
        self.wutter.open_shutter()
        group.create_dataset('image_before',data = self.CWL.thumb_image())        
        #self.aligner.optimise(0.003, max_steps=10, stepsize=0.5, npoints=3, dz=0.02,verbose=False)
        data = self.aligner.z_scan(dz =np.arange(-0.25,0.25,0.05))   
        group.create_dataset('z_scan_before', data = data, attrs = data.attrs)   
        self.wutter.close_shutter() 
        self.lutter.open_shutter()        
        time.sleep(0.2)    

        for index, Power in enumerate(self.Powers):
            if focus_with_laser: self.focus_with_laser()            
            attrs['power'] = Power           
            self.lutter.close_shutter()   
            self.wutter.open_shutter() 
            time.sleep(5)
            group.create_dataset('spectrum_before_%d', data = self.spec.read())            
            self.wutter.close_shutter()   
            self.lutter.open_shutter()            
            
            Captures = []            
            self.pc.power = Power             
            time.sleep(0.2)
            attrs['measured_power'] = self.read_pometer()  
            attrs['measured_power'] = self.read_pometer()  
            nkin = int(kinetic_fac/Power) 
            if nkin<1: nkin = 1
            self.trandor.SetParameter('NKin', nkin)
            Captures.append(self.trandor.capture()[0])
            self.lutter.close_shutter()   
            self.wutter.open_shutter()
            time.sleep(5)            
           

            To_Save = []            
            for i in Captures:
                To_Save+=np.reshape(i,[len(i)/1600,1600]).tolist()
                group.create_dataset('power_series_%d',data=To_Save,attrs=attrs)  
            update_progress(index)
        self.lutter.close_shutter()   
        self.wutter.open_shutter()
        time.sleep(35)     
        group.create_dataset('image_after',data = self.CWL.thumb_image())   
        data = self.aligner.z_scan(dz =np.arange(-0.25,0.25,0.05))        
        group.create_dataset('z_scan_after', data = data, attrs = data.attrs)   
        self.trandor.SetParameter('NKin' , 1)    
    def _launch_particle_track(self):
        particle_track_dict = {'aligner' : self.aligner,
                              'spectrometer' : self.spec}        
        self.wizard = TrackingWizard(self.CWL, particle_track_dict, task_list = ['Lab.Power_Series'])
        self.wizard.show()
    def focus_with_laser(self):
        initial_exp = self.CWL.camera.exposure
        initial_gain = self.CWL.camera.gain
        initial_param = self.pc.param 
        
        self.CWL.camera.exposure = 0.
        self.CWL.camera.gain = 1
        if self.laser == 785: self.rotate_to(self.minangle)
        if self.laser == 633: self.AOM.Power(1)
        self.wutter.close_shutter()
        self.lutter.open_shutter()
        time.sleep(1)
        self.CWL.autofocus(merit_function = laser_merit)
        self.CWL.camera.exposure = initial_exp
        self.CWL.camera.gain = initial_gain 
        self.pc.param = initial_param
    def get_qt_ui(self):
        return Lab_gui(self)

class Lab_gui(QtWidgets.QWidget,UiTools):
    def __init__(self, lab):
        super(Lab_gui, self).__init__()
        uic.loadUi(r'C:\Users\00\Documents\GitHub\NP_ee306\mine\Lab_5\setup_gui.ui', self)
        self.Lab = lab         
        self.SetupSignals()
        
    def SetupSignals(self):
        self.checkBox_633.stateChanged.connect(self._select_laser_633)
        self.checkBox_785.stateChanged.connect(self._select_laser_785)
        self.checkBox_785.setChecked(True)                   
        self.spinBox_steps.valueChanged.connect(self.update_steps)
        self.spinBox_max_nkin.valueChanged.connect(self.update_nkin)
        self.spinBox_max_nkin.setValue(self.Lab.max_nkin)
        self.checkBox_loop_down.stateChanged.connect(self.update_loop_down)
        self.pushButton_Power_Series.clicked.connect(self.Power_Series_gui)
        self.lineEdit_Power_Series_Name.textChanged.connect(self.update_power_series_name)        
        self.pushButton_particletrack.clicked.connect(self.Lab._launch_particle_track)
    def update_power_series_name(self):
        self.power_series_name = self.lineEdit_Power_Series_Name.text() 
    def _select_laser_633(self):
        if self.checkBox_633.isChecked() == True:        
            self.checkBox_785.setChecked(False)
            self.Lab.laser = '_633' 
            self.Lab.pometer.sense.correction.wavelength = 633
            self.Lab.pc.laser = self.Lab.laser
        else:
            self.checkBox_785.setChecked(True)
            self.Lab.laser = '_785'
            self.Lab.pometer.sense.correction.wavelength = 785
            self.Lab.pc.laser = self.Lab.laser
            
    def _select_laser_785(self):
        if self.checkBox_785.isChecked() == True:       
            self.checkBox_633.setChecked(False)
            self.Lab.laser = '_785' 
            self.Lab.pometer.sense.correction.wavelength = 785
            self.Lab.pc.laser = self.Lab.laser
        else:
            self.checkBox_633.setChecked(True)
            self.Lab.laser = '_633'
            self.Lab.pometer.sense.correction.wavelength = 633
            self.Lab.pc.laser = self.Lab.laser
        
    def set_trandor_centre_wl(self):
        self.Lab.trandor.Set_Center_Wavelength(self.doubleSpinBox_trandor_centre_wl.value())
    def set_slit(self):
        self.Lab.trandor.triax.Slit(self.doubleSpinBox_slit.value())
    
    def update_steps(self):
        self.Lab.steps = self.spinBox_steps.value()
    def update_nkin(self):
        self.Lab.max_nkin = self.spinBox_max_nkin.value()
    def update_loop_down(self):
        self.Lab.loow_down = self.checkBox_loop_down.isChecked()
    def Power_Series_gui(self):
        run_function_modally(self.Lab.Power_Series,  progress_maximum = self.Lab.steps if self.Lab.ramp == True else self.Lab.steps*2)


if __name__ == '__main__': 
    import os
    from nplab.instrument.spectrometer.seabreeze import OceanOpticsSpectrometer
    from nplab.instrument.camera.lumenera import LumeneraCamera
    from nplab.instrument.camera.camera_with_location import CameraWithLocation
    from nplab.instrument.spectrometer.spectrometer_aligner import SpectrometerAligner
    from nplab.instrument.stage.prior import ProScan
    from nplab.instrument.shutter.BX51_uniblitz import Uniblitz
    from nplab.instrument.electronics.ThorlabPM100_powermeter import ThorPM100
    from AOM import AOM as Aom
    from Rotation_Stage import Filter_Wheel
    from nplab.instrument.shutter.thorlabs_sc10 import ThorLabsSC10
    from nplab.instrument.spectrometer.Triax.Trandor_Lab5 import Trandor
    from particle_tracking_app.particle_tracking_wizard import TrackingWizard

    os.chdir(r'C:\Users\00\Documents\ee306')
    app = QtWidgets.QApplication(sys.argv)       
    #_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-    
    spec = OceanOpticsSpectrometer(0) 
    lutter = ThorLabsSC10('COM30')
    lutter.set_mode(1)
    FW= Filter_Wheel() 
    aom = Aom()
    pometer = ThorPM100(address = 'USB0::0x1313::0x807B::17121118::INSTR')
    wutter = Uniblitz("COM8")
    PC_785 = PowerControl(FW, wutter, lutter, pometer)
#    PC_633 = PowerControl(aom, wutter, lutter, pometer)
    cam = LumeneraCamera(1)
    stage = ProScan("COM32",hardware_version=2)
    CWL = CameraWithLocation(cam, stage)
    trandor=Trandor()
    #_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
    equipment_dict = {'spec' : spec,
                    'lutter' : lutter,
                    'FW' : FW,
                    'AOM' : aom,
                    'pometer' : pometer,
                    'wutter' : wutter,
                    'cam' : cam,
                    'CWL' : CWL,
                    'trandor' : trandor,
                    'pc' : PC_785}
    lab = Lab(equipment_dict)
  #_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
    gui_equipment_dict = {'Lab': lab,
                         'spec': spec, 
                         'cam': cam, 
                         'CWL': CWL, 
                         'andor': trandor,
                         'triax': trandor.triax,
                         'power_control' : PC_785,
                         'power_meter' : pometer,
                         'lutter' : lutter,
                         'wutter' : wutter}
    gui = GuiGenerator(gui_equipment_dict,
                       dock_settings_path = r'C:\Users\00\Documents\GitHub\NP_ee306\mine\Lab_5\config.npy',
                       scripts_path= r'C:\Users\00\Documents\GitHub\NP_ee306\mine\Lab_5')                           
    gui.show()
    #_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-     
        
    