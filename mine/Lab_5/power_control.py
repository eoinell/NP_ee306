# -*- coding: utf-8 -*-
"""
Created on Thu Aug 01 16:38:56 2019

@author: ee306
"""
import sys
import os
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
from AOM import AOM as Aom
from Rotation_Stage import Filter_Wheel


class PowerControl(Instrument):
    '''
    Controls the power
    '''
    def __init__(self, power_controller, white_light_shutter, laser_shutter, power_meter):       
        self.pc = power_controller        
        if isinstance(self.pc, Aom):
            self.laser = '_633'
            self._633 = True
            self._785 = False
            self.min_param = 0
            self.max_param = 1
        elif isinstance(self.pc, Filter_Wheel):
            self.laser = '_785'
            self._633 = False
            self._785 = True
            self.min_param = 260
            self.max_param = 500                  
        else: raise ValueError, 'power_controller must be AOM or Filter Wheel'
        self.param = self.mid_param
        self.maxpower = None 
        self.minpower = None
        self.measured_power = None
        self.update_power_calibration()
        self.wutter = white_light_shutter        
        self.lutter = laser_shutter        
        self.lutter.close_shutter()
        
        self.wutter.open_shutter()  
        super(PowerControl, self).__init__()
        self._initiate_pc()
        self.pometer = power_meter
  
    def _initiate_pc(self):
        if self._633:            
            self.pc.Switch_Mode()
        self.param = self.mid_param
   
    def _set_to_midpoint(self):
        self.param = self.mid_param
    def _set_to_maxpoint(self):
       self.param = self.max_param
    def _initiate_pometer(self):
        if isinstance(self.pometer, 'Thorlabs_powermeter'):      
            self.pometer.system.beeper.immediate()
            if self._785: self.pometer.sense.correction.wavelength = 785   
            if self._633: self.pometer.sense.correction.wavelength = 633              
    
    @property
    def param(self):
        if self._785:
            p = int(self.pc.Stage.Get_Position().split(' ')[-1])
            return 0. if p>500 else np.round(p, decimals = 2)
        if self._633:
            return self.pc.Get_Power()               
    @param.setter
    def param(self,value):
        if self._785:        
            self.pc.Stage.Rotate_To(value)     
        if self._633:
            self.pc.Power(value) 
    @property
    def mid_param(self):
        return (self.max_param - self.min_param)/2

    @property
    def points(self):
        if self._785:        
            return np.logspace(0,np.log10(self.max_param-self.min_param),50)+self.min_param
        if self._633:
            return np.linspace(self.min_param,self.max_param,num = 50, endpoint = True) 
            
    def Calibrate_Power(self, update_progress=lambda p:p):
        attrs = {}       
        if self.measured_power is not None: attrs['Measured power at maxpoint'] = self.measured_power
        if self.laser == '_785':
            attrs['Angles']  = self.points  
    
        if self.laser == '_633':
            attrs['Voltages'] = self.points
        attrs['wavelengths'] = self.points

        powers = []
        
        self.wutter.close_shutter()    
        self.lutter.open_shutter() 
        
        for counter, point in enumerate(self.points):          
            self.param = point
            time.sleep(0.01)
            powers = np.append(powers,self.pometer.power)
            update_progress(counter)
        group = self.create_data_group('Power_Calibration{}_%d'.format(self.laser), attrs = attrs)
        group.create_dataset('measured_powers',data=powers)
        if self.measured_power is None:
            group.create_dataset('ref_powers',data=powers, attrs = attrs)
        else:
            group.create_dataset('ref_powers',data=( powers*self.measured_power/max(powers)), attrs = attrs)
        self.lutter.close_shutter()
        self._set_to_midpoint()
        self.wutter.open_shutter()
        self.update_power_calibration()    
    def update_power_calibration(self, laser = None):
        if laser is None:
           laser = self.laser 
        search_in = self.get_root_data_folder()
        try: 
            power_calibration_group = max([(int(name.split('_')[-1]), group)\
            for name, group in search_in.items() \
            if name.startswith('Power_Calibration') and (name.split('_')[-2] == laser[1:])])[1]
            self.power_calibration = {'ref_powers' : power_calibration_group['ref_powers']} 
            if self._785:
                self.power_calibration.update({'Angles' : power_calibration_group.attrs['Angles']})
                self.update_config('Angles'+self.laser, power_calibration_group.attrs['Angles'])
            if self._633:
                self.power_calibration.update({'Voltages' : power_calibration_group.attrs['Voltages']})
                self.update_config('Voltages'+self.laser, power_calibration_group.attrs['Voltages'])
            self.update_config('ref_powers'+self.laser, self.power_calibration['ref_powers'])
            
        except ValueError:
            if len(self.config_file)>0:            
                self.power_calibration = {'_'.join(n.split('_')[:-1]) : f for n,f in self.config_file.items() if n.endswith(self.laser)}
                print 'No power calibration in current file, using inaccurate configuration'
            else:
                print('No power calibration found')

    @property
    def power(self):
        return self.pometer.power
    @power.setter
    def power(self, value):
        if self._633:
            self.param = self.Power_to_Voltage(value)
        if self._785:
            self.param = self.Power_to_Angle(value)
    def Power_to_Angle(self, power):
        self.update_power_calibration()        
        angles = self.power_calibration['Angles']    
        powers = np.array(self.power_calibration['ref_powers'])
        curve = interpolate.interp1d(powers, angles, kind = 'cubic') #  
        angle = curve(power)
        if min(self.points)<=angle<=max(self.points):        
            return angle
        elif np.absolute(angle-min(self.params))<3:
            return min(self.points)
        elif np.absolute(angle-max(self.points))<3:
            return max(self.points)
        else:
            print 'Error, angle of '+str(angle)+' outside allowed range'
    def Power_to_Voltage(self, power):
        voltages = self.power_calibration['Voltages']    
        try:
            powers = np.array(self.power_calibration['ref_powers'])
            curve = interpolate.interp1d(powers, voltages, kind = 'cubic') #  
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
        super(PowerControl_UI, self).__init__()
        uic.loadUi(os.path.join(os.path.dirname(__file__), 'power_control.ui'), self)
        self.PC = PC         
        self.SetupSignals()
        
    def SetupSignals(self):
        self.pushButton_calibrate_power.clicked.connect(self.Calibrate_Power_gui)             
        self.doubleSpinBox_min_param.setValue(self.PC.min_param)
        self.doubleSpinBox_max_param.setValue(self.PC.max_param)
        self.doubleSpinBox_max_param.valueChanged.connect(self.update_min_max_params)  
        self.doubleSpinBox_min_param.valueChanged.connect(self.update_min_max_params)
        self.laser_textBrowser.setPlainText('Laser: '+self.PC.laser[1:])    
        self.pushButton_set_param.clicked.connect(self.set_param)
        self.doubleSpinBox_measured_power.valueChanged.connect(self.update_measured_power)
        self.pushButton_set_power.clicked.connect(self.set_power_gui)        
     
    def update_min_max_params(self):
        self.PC.min_param = self.doubleSpinBox_min_param.value()
        self.PC.max_param = self.doubleSpinBox_max_param.value()
        if self.PC.laser == '_633' and self.PC.max_param>1:
            print 'voltages over 1 not allowed!'
            self.PC.maxvolt = 1
    def update_measured_power(self):
        self.PC.measured_power = float(self.doubleSpinBox_measured_power.value())
    def set_param(self):
        if self.PC._633 and self.doubleSpinBox_set_input_param.value()>1:       
            self.PC.param = 1
            print('voltages over 1 not allowed')
        else:
            self.PC.param = self.doubleSpinBox_set_input_param.value()
    def set_power_gui(self):
        self.PC.power = float(self.doubleSpinBox_set_power.value())
    def Calibrate_Power_gui(self):
        run_function_modally(self.PC.Calibrate_Power, progress_maximum = len(self.PC.points))
    
if __name__ == '__main__': 

    from nplab.instrument.shutter.BX51_uniblitz import Uniblitz
    from mine.Lab_5.thorlabs_pm1000 import Thorlabs_powermeter
    from nplab.instrument.shutter.thorlabs_sc10 import ThorLabsSC10
    from nplab import datafile
    os.chdir(r'C:\Users\00\Documents\ee306')    
   
    lutter = ThorLabsSC10('COM30')
    FW = Filter_Wheel() 
    lutter.set_mode(1)
    aom = Aom()
    aom.Switch_Mode()
    aom.Power(0.95)
    pometer = Thorlabs_powermeter()
    wutter = Uniblitz("COM8")
    PC = PowerControl(FW, wutter, lutter, pometer)
    PC.show_gui(blocking = False)
    pometer.show_gui(blocking = False)
    wutter.show_gui(blocking = False)
    lutter.show_gui(blocking = False)    
    datafile.current().show_gui(blocking = False)
    PC2 = PowerControl(aom, wutter, lutter, pometer)
    PC2.show_gui(blocking = False)