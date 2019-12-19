# -*- coding: utf-8 -*-
"""
Created on Thu Aug 01 16:38:56 2019

@author: ee306
"""
import sys

import numpy as np
from scipy import interpolate 
import time
from qtpy import QtWidgets, uic
import matplotlib.pyplot as plt
from nplab.utils.gui_generator import GuiGenerator
from nplab.ui.ui_tools import UiTools
from scipy.interpolate import UnivariateSpline
from nplab.experiment.gui import run_function_modally
from nplab.instrument import Instrument
 
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

class PowerControl(Instrument):
    '''
    meta-instrument for all the equipment in Lab 5. Works analogously to CWL in many respects. 
    '''
    def __init__(self, power_controller, white_light_shutter, laser_shutter, parent = None):       
        self.pc = power_controller        
        if isinstance(self.pc, 'AOM'):
            self.laser = '_633'
            self._633 = True
            self._785 = False
            self.min_param = 0
            self.max_param = 1
            self.points = np.linspace(0,1,num = 50, endpoint = True) 
        elif isinstance(self.pc, 'Filter_Wheel'):
            self.laser = '_785'
            self._633 = False
            self._785 = True
            self.min_param = 260
            self.max_param = 500                  
            self.points = np.logspace(0,np.log10(self.maxangle-self.minangle),50)+self.minangle
        self.mid_param = (self.max_param - self.min_param)/2
        self.go_to(self.mid_param)
        
        self.maxpower = None 
        self.minpower = None
        self.update_power_calibration()
        self.wutter = white_light_shutter        
        self.lutter = laser_shutter        
        self.lutter.close_shutter()
        
        self.wutter.open_shutter()  
        super(PowerControl, self).__init__(self)
        self._initiate_pc()
  
    def _initiate_pc(self):
        if self._633:            
            self.pc.Switch_Mode()
            self.pc.Power(0.95)
        if self._785:
            self.go_to
    
    def _set_to_midpoint(self):
        if self.laser == '_785':
            self.rotate_to(self.midangle)
        if self.laser == '_633':
            self.AOM.Power(self.midvolt)
    def _set_to_maxpoint(self):
        if self.laser == '_785':
            self.rotate_to(self.minangle)
        if self.laser == '_633':
            self.AOM.Power(self.maxvolt)
    def _initiate_pometer(self, instrument):
        if self.init_pometer is True:
            print 'Power meter already initialised'
        else:            
            self.pometer = instrument
            self.pometer.system.beeper.immediate()
            if self._785: self.pometer.sense.correction.wavelength = 785   
            if self._633: self.pometer.sense.correction.wavelength = 633              
            self.init_pometer = True
    def read_pometer(self):
        Output=[]
        Fail=0
        while len(Output)<20:
            try:
                Output.append(self.pometer.read)
                Fail=0
            except:
                Fail+=1
            if Fail==10:
                raise Exception('Restart power meter')
        return np.median(Output)*1000 # mW
    
    @property
    def param(self):
        if self._785:
            p = int(self.pc.Stage.Get_Position().split(' ')[-1])
            return 0. if p>500 else np.round(p, decimals = 2)
        if self._633:
            return self.pc.Get_Power()               
    @param.setter
    def param(self,value):
        if self.__785:        
            self.pc.Stage.Rotate_To(value)     
            self.param = value
        if self._633:
            self.pc.Power(value) 
    
    def Calibrate_Power(self, update_progress=lambda p:p):# Power in mW, measured at maxpoint in FW
        #if you don't want to use a seperate power meter, set Measured_Power = False
        attrs = {}       
        attrs['Measured power at maxpoint'] = self.measured_power
        if self.laser == '_785':
            attrs['Angles']  = self.points  
            attrs['wavelengths'] = self.points   
    
        if self.laser == '_633':
            attrs['Voltages'] = self.points
            attrs['wavelengths'] = self.points

        powers = []
        
        self.wutter.close_shutter()    
        self.lutter.open_shutter() 
        if self.laser == '_785':
            for counter, angle in enumerate(self.anglez):          
                self.rotate_to(angle)
                time.sleep(1)
                powers = np.append(powers,self.read_pometer())
                update_progress(counter)
            group = self.create_data_group('Power_Calibration_785_%d', attrs = attrs)
        if self.laser == '_633':
            for counter, voltage in enumerate(self.voltagez):
                self.AOM.Power(voltage)                
                time.sleep(0.1)
                powers = np.append(powers,self.read_pometer())
                update_progress(counter)
            group = self.create_data_group('Power_Calibration_633_%d', attrs = attrs)
        group.create_dataset('powers',data=powers)
        if self.measured_power == False:
            group.create_dataset('real_powers',data=powers, attrs = attrs)
        else:
    
            group.create_dataset('real_powers',data=( powers*self.measured_power/max(powers)), attrs = attrs)
        self.lutter.close_shutter()
        self._set_to_midpoint()
        self.wutter.open_shutter()
        self.update_power_calibration()    
    def update_power_calibration(self):
        search_in = self.get_root_data_folder()        
        power_group = []        
        if self.laser == '_633':        
            try:
                for key in search_in.keys():
                    if key[0:21] == 'Power_Calibration_633':       
                        n = 0                
                        while n<50: 
                            n-=1
                                                    
                            if key[n] == '_':
                                break                   
    
                        power_group.append(int(key[n+1:]))
                self.power_calibration = search_in['Power_Calibration_633_'+str(max(power_group))]
            except:
                print 'Power calibration not found'
        if self.laser == '_785':
            try:
                for key in search_in.keys():
                    if key[0:21] == 'Power_Calibration_785':       
                        n = 0                
                        while n<50: 
                            n-=1
                                                    
                            if key[n] == '_':
                                break                   
    
                        power_group.append(int(key[n+1:]))
                self.power_calibration = search_in['Power_Calibration_785_'+str(max(power_group))]
            except:
                print 'Power calibration not found'
    def Power(self, power):
        if self.laser == '_785':
            self.rotate_to(self.Power_to_Angle(power))
        if self.laser == '_633':
            self.AOM.Power(self.Power_to_Voltage(power))
    def Power_to_Angle(self, power):
        angles = self.power_calibration.attrs['Angles']    
        real_powers = np.array(self.power_calibration['real_powers'])
        curve = interpolate.interp1d(real_powers, angles, kind = 'cubic') #  
        angle = curve(power)
        if min(self.anglez)<=angle<=max(self.anglez):        
            return angle
        elif np.absolute(angle-min(self.anglez))<3:
            return min(self.anglez)
        elif np.absolute(angle-max(self.anglez))<3:
            return max(self.anglez)
        else:
            print 'Error, angle of '+str(angle)+' outside allowed range'
    def Power_to_Voltage(self, power):
        voltages = self.power_calibration.attrs['Voltages']    
        try:
            real_powers = np.array(self.power_calibration['real_powers'])
            curve = interpolate.interp1d(real_powers, voltages, kind = 'cubic') #  
            voltage = curve(power)
            if -0.01<=voltage<=1:        
                return voltage
            if 1<voltage<1.1:
                return 1.
            else:
                print 'Error, voltage of '+str(voltage)+' outside allowed range'
        except:
            print 'Power Calibration not found'
    
    def get_qt_ui(self):
        return PowerControl_UI(self)

class PowerControl_UI(QtWidgets.QWidget,UiTools):
    def __init__(self, PC):
        super(Lab_gui, self).__init__()
        uic.loadUi(r'C:\Users\00\Documents\GitHub\NP_ee306\mine\Lab_5\setup_gui.ui', self)
        self.PC = PC         
        self.SetupSignals()
        
    def SetupSignals(self):
        self.pushButton_set_midpoint.clicked.connect(self.Lab._set_to_midpoint)
        self.pushButton_set_maxpoint.clicked.connect(self.Lab._set_to_maxpoint)
        self.pushButton_calibrate_power.clicked.connect(self.Calibrate_Power_gui)
        self.checkBox_633.stateChanged.connect(self._select_laser_633)
        self.checkBox_785.stateChanged.connect(self._select_laser_785)
        self.checkBox_785.setChecked(True)                
        self.doubleSpinBox_min_param.valueChanged.connect(self.update_min_max_params)
        self.doubleSpinBox_max_param.valueChanged.connect(self.update_min_max_params)
        if self.Lab.laser == '_785':
            self.doubleSpinBox_min_param.value = self.Lab.minangle
            self.doubleSpinBox_max_param.value = self.Lab.maxangle
        if self.Lab.laser == '_633':
            self.doubleSpinBox_min_param = self.Lab.minvolt 
            self.doubleSpinBox_max_param.value = self.Lab.maxvolt
            
        self.pushButton_set_param.clicked.connect(self.set_param)
        self.pushButton_lutter.clicked.connect(self.Lab.lutter.toggle)
        self.pushButton_wutter.clicked.connect(self.Lab.wutter.toggle)
        self.lineEdit_Power_Series_Name.textChanged.connect(self.update_power_series_name)
        self.doubleSpinBox_measured_power.valueChanged.connect(self.update_measured_power)        
        self.pushButton_particletrack.clicked.connect(self.Lab._launch_particle_track)
    def update_power_series_name(self):
        self.power_series_name = self.lineEdit_Power_Series_Name.text() 
    def _select_laser_633(self):
        if self.checkBox_633.isChecked() == True:        
            self.checkBox_785.setChecked(False)
            self.Lab.laser = '_633' 
            self.Lab.pometer.sense.correction.wavelength = 633
            self.anglez = np.linspace(self.minangle , self.maxangle, num = 50, endpoint = True)
        else:
            self.checkBox_785.setChecked(True)
            self.Lab.laser = '_785'
            self.Lab.pometer.sense.correction.wavelength = 785
            self.Lab.anglez = np.logspace(0,np.log10(self.maxangle - self.minangle),50)+self.minangle
    def _select_laser_785(self):
        if self.checkBox_785.isChecked() == True:       
            self.checkBox_633.setChecked(False)
            self.Lab.laser = '_785' 
            self.Lab.pometer.sense.correction.wavelength = 785
            self.Lab.anglez = np.logspace(0,np.log10(self.Lab.maxangle - self.Lab.minangle),50)+self.Lab.minangle
        else:
            self.checkBox_633.setChecked(True)
            self.Lab.laser = '_633'
            self.Lab.pometer.sense.correction.wavelength = 633
            self.Lab.anglez = np.linspace(self.Lab.minangle , self.Lab.maxangle, num = 50, endpoint = True)    
    def update_exposure(self):
        self.Lab.trandor_exposure = self.doubleSpinBox_exposure.value()
        self.Lab.trandor.Exposure = self.Lab.trandor_exposure        
    def set_trandor_centre_wl(self):
        self.Lab.trandor.Set_Center_Wavelength(self.doubleSpinBox_trandor_centre_wl.value())
    
    def set_slit(self):
        self.Lab.trandor.triax.Slit(self.doubleSpinBox_slit.value())
    def update_steps(self):
        self.Lab.steps = self.spinBox_steps.value()
    def update_nkin(self):
        self.Lab.max_nkin = self.spinBox_max_nkin.value()
    def update_ramp(self):
        self.Lab.ramp = self.checkBox_ramp.isChecked()
    def update_measured_power(self):
        self.Lab.measured_power = self.doubleSpinBox_measured_power.value()
    def update_min_max_params(self):
        if self.Lab.laser == '_785':
            self.Lab.minangle = self.doubleSpinBox_min_param.value()
            self.Lab.maxangle = self.doubleSpinBox_max_param.value()
            self.Lab.anglez = np.logspace(0,np.log10(self.Lab.maxangle-self.Lab.minangle),50)+self.Lab.minangle
            self.Lab.midangle = (self.Lab.maxangle - self.Lab.minangle)/2
        if self.laser == '_633':
            self.Lab.minvolt = self.doubleSpinBox_min_param.value()
            self.Lab.maxvolt = self.doubleSpinBox_max_param.value() 
            if self.maxvolt>1:
                print 'voltages over 1 not allowed!'
                self.Lab.maxvolt = 1
            self.Lab.voltagez = np.linspace(0,self.Lab.maxvolt,num = 40, endpoint = True)        
            self.Lab.midvolt = self.Lab.voltagez[len(self.Lab.voltagez)/2]
    def set_param(self):
        param = self.doubleSpinBox_set_input_param.value()
        if self.Lab.laser == '_785':
            self.Lab.rotate_to(param)
        elif self.Lab.laser == '_633':
            if param>1:
                print 'voltages >1 not allowed!'
                param = 1
            self.Lab.AOM.Power(param)
   
    def Calibrate_Power_gui(self):
        run_function_modally(self.Lab.Calibrate_Power, progress_maximum = len(self.Lab.anglez) if self.Lab.laser == '785' else len(self.Lab.voltagez))
    


if __name__ == '__main__': 
    import os
    import visa
    from nplab.instrument.spectrometer.seabreeze import OceanOpticsSpectrometer
    from nplab.instrument.camera.lumenera import LumeneraCamera
    from nplab.instrument.camera.camera_with_location import CameraWithLocation
    from nplab.instrument.spectrometer.spectrometer_aligner import SpectrometerAligner
    from nplab.instrument.stage.prior import ProScan
    from nplab.instrument.shutter.BX51_uniblitz import Uniblitz
    from nplab import datafile
    from ThorlabsPM100.ThorlabsPM100 import ThorlabsPM100
    import Rotation_Stage as RS
    import AOM
    from nplab.instrument.shutter.thorlabs_sc10 import ThorLabsSC10
    from nplab.instrument.spectrometer.Triax.Trandor_Lab5 import Trandor
    from particle_tracking_app.particle_tracking_wizard import TrackingWizard

    os.chdir(r'C:\Users\00\Documents\ee306')    
    app = QtWidgets.QApplication(sys.argv)    
    rm= visa.ResourceManager()
    
    spec = OceanOpticsSpectrometer(0) 
    lutter = ThorLabsSC10('COM30')
    FW = RS.Filter_Wheel() 
    lutter.set_mode(1)
    aom = AOM.AOM()
    aom.Switch_Mode()
    aom.Power(0.95)
    inst = rm.open_resource('USB0::0x1313::0x807B::17121118::INSTR',timeout=1)
    pometer = ThorlabsPM100(inst=inst)
    wutter = Uniblitz("COM8")
    cam = LumeneraCamera(1)
    
    stage = ProScan("COM32",hardware_version=2)
    CWL = CameraWithLocation(cam, stage)
    
    trandor=Trandor()
    
    equipment_dict = {'spec' : spec,
                    'lutter' : lutter,
                    'FW' : FW,
                    'AOM' : aom,
                    'pometer' : pometer,
                    'wutter' : wutter,
                    'cam' : cam,
                    'CWL' : CWL,
                    'trandor' : trandor}
    lab = Lab(equipment_dict)
    gui_equipment_dict = {'Lab': lab,
                         'spec': spec, 
                         'cam': cam, 
                         'CWL': CWL, 
                         'trandor': trandor}    
    
    File = datafile.current()
    gui = GuiGenerator(gui_equipment_dict,
                       dock_settings_path = r'C:\Users\00\Documents\GitHub\NP_ee306\mine\Lab_5\config.npy',
                       scripts_path= r'C:\Users\00\Documents\GitHub\NP_ee306\mine\Lab_5')
                                
    gui.show()
        
        
    