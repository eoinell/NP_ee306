# -*- coding: utf-8 -*-
"""
Created on Thu Aug 01 16:38:56 2019

@author: Hera
"""
import sys

from nplab.instrument.spectrometer.seabreeze import OceanOpticsSpectrometer
from nplab.instrument.camera.lumenera import LumeneraCamera
from nplab.instrument.camera.camera_with_location import CameraWithLocation
from nplab.instrument.spectrometer.spectrometer_aligner import SpectrometerAligner
from nplab.instrument.stage.prior import ProScan
from nplab.instrument.shutter.BX51_uniblitz import Uniblitz
from nplab import datafile
from ThorlabsPM100.ThorlabsPM100 import ThorlabsPM100

from particle_tracking_app.particle_tracking_wizard import TrackingWizard
import numpy as np
from scipy import interpolate 
import time
import AOM
from nplab.instrument.spectrometer.Triax.Trandor_Lab5 import Trandor
import Rotation_Stage as RS
from nplab.instrument.shutter.thorlabs_sc10 import ThorLabsSC10
from qtpy import QtWidgets, uic
import matplotlib.pyplot as plt
from nplab.utils.gui_generator import GuiGenerator
from nplab.ui.ui_tools import UiTools
import visa
from scipy.interpolate import UnivariateSpline
rm= visa.ResourceManager()    

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

class Exp(QtWidgets.QWidget,UiTools):
    def __init__(self, equipment_dict, parent = None):
        super(Exp,self).__init__(parent)       
        self.laser = '_785'        
        
        self.initiate_all(equipment_dict)
        
        uic.loadUi(r'C:\Users\00\Documents\GitHub\Lab5 programming\Gui\setup_gui.ui', self)
        #self.equipment_dict = {'Exp':self, 'spec':self.spec, 'lutter':self.lutter, 'wutter':self.wutter, 'pometer':self.pometer, 'CWL':self.CWL, 'trandor':self.trandor}         
        self.minangle = 260
        self.maxangle = 500            
        self.SetupSignals()        
        self.File = datafile.current()
  
        #self.anglez = np.linspace(self.minangle , self.maxangle, num = 50, endpoint = True)
        self.anglez = np.logspace(0,np.log10(self.maxangle-self.minangle),50)+self.minangle
        self.midangle = (self.maxangle - self.minangle)/2
        self.rotate_to(self.midangle)
        
        self.voltagez = np.linspace(0,1,num = 60, endpoint = True)        
        self.midvolt = self.voltagez[len(self.voltagez)/2]
        self.maxvolt = self.voltagez[-1]
        self.maxpower = None 
        self.minpower = None
        self.power_series_name = 'particle_%d'
        self.ramp = False
        self.steps = 5
        self.max_nkin = 10        
        self.measured_power = False
        self.update_power_calibration()
        self.lutter.close_shutter()
        self.wutter.open_shutter()        
        self.equipment_dict = {'Exp':self, 'spec':self.spec, 'cam':self.cam, 'CWL':self.CWL, 'trandor':self.trandor}        
        self.gui = GuiGenerator(self.equipment_dict,
                                dock_settings_path = r'C:\Users\00\Documents\ee306\ee306.npy',
                                scripts_path=r'C:\Users\00\Documents\GitHub\NP_ee306\mine\Lab_5')
    def initiate_all(self, ed):
        self.init_spec = False
        self.init_lutter = False
        self.init_FW = False
        self.init_pometer = False
        self.init_cam = False
        self.init_CWL = False
        self.init_wutter = False
        self.init_trandor = False
        self.init_aligner = False                
        self.init_AOM = False               
        self._initiate_spectrometer(ed['spec'])
        self._initiate_lutter(ed['lutter'])
        self._initiate_FW(ed['FW'])
        self._initiate_pometer(ed['pometer'])
        self._initiate_cam(ed['cam'])        
        self._initiate_CWL(ed['CWL'])
        self._initiate_wutter(ed['wutter'])
        self._initiate_trandor(ed['trandor'])
        self._initiate_aligner()
        self._initiate_AOM(ed['AOM'])
    
    
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
            
    def _initiate_FW(self, instrument):
        if self.init_FW is True:            
            print 'Filter Wheel already initialised'
        else:            
            self.FW = instrument
               
            self.init_FW = True
   
    def _initiate_AOM(self, instrument):
        if self.init_AOM == True:
            print 'AOM already initialised'
        else:
            self.AOM = instrument
            self.AOM.Switch_Mode()
            self.AOM.Power(0.95)
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
            if self.laser == '_785': self.pometer.sense.correction.wavelength = 785   
            if self.laser == '_633': self.pometer.sense.correction.wavelength = 633              
            self.init_pometer = True
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
            self.init_CWL = True
    def _initiate_wutter(self, instrument):    
        if self.init_wutter is True :           
            print 'White light shutter already initialised'
        else:            
            self.wutter = instrument
            self.wutter.close_shutter()            
            self.wutter.open_shutter()
            self.init_wutter = True
    def _initiate_trandor(self, instrument):
        if self.init_trandor is True:
            'Print Triax and Andor already initialised'
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
        if self.init_aligner is True:                 
            'Spectrometer aligner already initiated'
        else:
            
            self.aligner = SpectrometerAligner(self.spec, self.CWL.stage)
            self.init_aligner = True
    
    def SetupSignals(self):
        self.pushButton_set_midpoint.clicked.connect(self._set_to_midpoint)
        self.pushButton_set_maxpoint.clicked.connect(self._set_to_maxpoint)
        self.pushButton_calibrate_power.clicked.connect(self.Calibrate_Power)
        self.checkBox_633.stateChanged.connect(self._select_laser_633)
        self.checkBox_785.stateChanged.connect(self._select_laser_785)
        self.checkBox_785.setChecked(True)        
        self.doubleSpinBox_trandor_centre_wl.valueChanged.connect(self.update_trandor_centre_wl)        
        self.pushButton_set_trandor_centre_wl.clicked.connect(self.set_trandor_centre_wl)        
        self.doubleSpinBox_exposure.valueChanged.connect(self.update_exposure)
        self.spinBox_steps.valueChanged.connect(self.update_steps)
        self.spinBox_max_nkin.valueChanged.connect(self.update_nkin)
        self.checkBox_ramp.stateChanged.connect(self.update_ramp)
        self.doubleSpinBox_minpower.valueChanged.connect(self.update_minpower)
        self.doubleSpinBox_minpower.valueChanged.connect(self.update_maxpower)          
        self.pushButton_Power_Series.clicked.connect(self._Power_Series)
        
        self.pushButton_lutter.clicked.connect(self.lutter.toggle)
        self.pushButton_wutter.clicked.connect(self.wutter.toggle)
        self.lineEdit_Power_Series_Name.textChanged.connect(self.update_power_series_name)
        self.doubleSpinBox_measured_power.valueChanged.connect(self.update_measured_power)        
        
        self.pushButton_particletrack.clicked.connect(self._launch_particle_track)
        
    def _select_laser_633(self):
        if self.checkBox_633.isChecked() == True:        
            self.checkBox_785.setChecked(False)
            self.laser = '_633' 
            self.pometer.sense.correction.wavelength = 633
            self.anglez = np.linspace(self.minangle , self.maxangle, num = 50, endpoint = True)
        else:
            self.checkBox_785.setChecked(True)
            self.laser = '_785'
            self.pometer.sense.correction.wavelength = 785
            self.anglez = np.logspace(0,np.log10(self.maxangle - self.minangle),50)+self.minangle
    
    def _select_laser_785(self):
        if self.checkBox_785.isChecked() == True:       
            self.checkBox_633.setChecked(False)
            self.laser = '_785' 
            self.pometer.sense.correction.wavelength = 785
            self.anglez = np.logspace(0,np.log10(self.maxangle - self.minangle),50)+self.minangle
        else:
            self.checkBox_633.setChecked(True)
            self.laser = '_633'
            self.pometer.sense.correction.wavelength = 633
            self.anglez = np.linspace(self.minangle , self.maxangle, num = 50, endpoint = True)
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
    def update_power_series_name(self):
        self.power_series_name = self.lineEdit_Power_Series_Name.text() 
   
    def rotate_to(self,angle):
        self.FW[0].Stage.Rotate_To(angle)     
        self.angle = angle
    def Calibrate_Power(self):# Power in mW, measured at maxpoint in FW
        #if you don't want to use a seperate power meter, set Measured_Power = False
        attrs = {}       
        attrs['Measured power at maxpoint'] = self.measured_power  
        if self.laser == '_785':
            attrs['Angles']  = self.anglez  
            attrs['wavelengths'] = self.anglez   
            attrs['maxpoint'] = self.minangle
        if self.laser == '_633':
            attrs['Voltages'] = self.voltagez
            attrs['wavelengths'] = self.voltagez
            attrs['maxpoint'] = self.maxvolt        
        powers = []
        
        self.wutter.close_shutter()    
        self.lutter.open_shutter() 
        if self.laser == '_785':
            for angle in self.anglez:
                print angle            
                self.rotate_to(angle)
                time.sleep(1)
                powers = np.append(powers,self.read_pometer())
            group = self.File.create_group('Power_Calibration_785_%d', attrs = attrs)
        if self.laser == '_633':
            for voltage in self.voltagez:
                self.AOM.Power(voltage)                
                time.sleep(0.1)
                powers = np.append(powers,self.read_pometer())
            group = self.File.create_group('Power_Calibration_633_%d', attrs = attrs)
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
        power_group = []        
        if self.laser == '_633':        
            try:
                for key in self.File.keys():
                    if key[0:21] == 'Power_Calibration_633':       
                        n = 0                
                        while n<50: 
                            n-=1
                                                    
                            if key[n] == '_':
                                break                   
    
                        power_group.append(int(key[n+1:]))
                self.power_calibration = self.File['Power_Calibration_633_'+str(max(power_group))]
            except:
                print 'Power calibration not found'
        if self.laser == '_785':
            try:
                for key in self.File.keys():
                    if key[0:21] == 'Power_Calibration_785':       
                        n = 0                
                        while n<50: 
                            n-=1
                                                    
                            if key[n] == '_':
                                break                   
    
                        power_group.append(int(key[n+1:]))
                self.power_calibration = self.File['Power_Calibration_785_'+str(max(power_group))]
            except:
                print 'Power calibration not found'
    def Power(power_frac, self):
        maxpower = max(np.array(self.power_calibration['real_powers']))
        self.rotate_to(self.Power_to_Angle(power_frac*maxpower))
        
    def update_exposure(self):
        self.trandor_exposure = self.doubleSpinBox_exposure.value()
        self.trandor.Exposure = self.trandor_exposure
    def update_trandor_centre_wl(self):
        self.trandor_centre_wl = self.doubleSpinBox_trandor_centre_wl.value()
        
    def set_trandor_centre_wl(self):
        self.trandor.Set_Center_Wavelength(self.trandor_centre_wl)
    
    def update_steps(self):
        self.steps = self.spinBox_steps.value()
    def update_nkin(self):
        self.max_nkin = self.spinBox_max_nkin.value()
    def update_ramp(self):
        self.ramp = self.checkBox_ramp.isChecked()
    def update_minpower(self):
        self.minpower = self.doubleSpinBox_minpower.value()
    def update_maxpower(self):
        self.maxpower = self.doubleSpinBox_maxpower.value()
    def update_measured_power(self):
        self.measured_power = self.doubleSpinBox_measured_power.value()
    def Power_to_Angle(self, power):
        """Helppppp """
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
    
    def _Power_Series(self):
        self.focus_with_laser()        
        self.update_power_calibration()  # necessary if changed lasers      
        datafile.set_current_group(self.File.create_group(self.power_series_name))
        group = datafile._current_group              
        #self.focus_with_laser()      
        
        if self.maxpower == None: 
            maxpower = max(np.array(self.power_calibration['real_powers']))
            minpower = 0.2*maxpower
        else:
            maxpower = self.maxpower
            minpower = self.minpower
        print minpower    
        powers_up = np.linspace(minpower,maxpower, num = self.steps, endpoint = True)
        powers_down = []                    
        if self.ramp == True: powers_down = powers_up[:-1][::-1]
        self.Powers = np.append(powers_up, powers_down)

        kinetic_fac =   self.max_nkin*float(minpower)

        attrs = self.trandor.metadata
        attrs['x_axis']=np.flipud(attrs['x_axis'])
        attrs['wavelengths'] = attrs['x_axis']
        attrs['AcquisitionTimes'] = self.trandor.AcquisitionTimings
        attrs['powers']=self.Powers
        attrs['Sleep']=.2
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
            self.focus_with_laser()            
            attrs['power'] = Power           
            self.lutter.close_shutter()   
            self.wutter.open_shutter() 
            time.sleep(5)
            group.create_dataset('spectrum_before_%d', data = self.spec.read())            
            self.wutter.close_shutter()   
            self.lutter.open_shutter()            
            
            Captures = []            
            if self.laser == '_785': self.rotate_to(self.Power_to_Angle(Power))  
            if self.laser == '_633': self.AOM.Power(self.Power_to_Voltage(Power))              
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
#            group.create_dataset('spectrum_after_%d', data = self.spec.read())            
#            self.wutter.close_shutter()   
#            self.lutter.open_shutter()            

            To_Save = []            
            for i in Captures:
                To_Save+=np.reshape(i,[len(i)/1600,1600]).tolist()
                group.create_dataset('power_series_%d',data=To_Save,attrs=attrs)  
        self.lutter.close_shutter()   
        self.wutter.open_shutter()
             
        group.create_dataset('image_after',data = self.CWL.thumb_image())   
        data = self.aligner.z_scan(dz =np.arange(-0.25,0.25,0.05))        
        group.create_dataset('z_scan_after', data = data, attrs = data.attrs)   
        self.trandor.SetParameter('NKin' , 1)
    
    def Power_Series(self, steps_in_power_series = 5, max_Power_Series_Number = 30, exposure = 2, centre_wl = 785., ramp = False ):
        #don't know what this does - probably the number of spectra in a kinetic spectrum
        #FW[0].Stage.Rotate_To(Angles[0])#FilterWheel        
         # number of spectra
        self.focus_with_laser()
        self.trandor.Exposure = exposure 
        self.trandor.Set_Center_Wavelength(centre_wl)
        
        maxpower = max(np.array(self.power_calibration['real_powers']))
        #minpower = np.array(File['Power Calibration']['real_powers'])[-1]
        powers_up = np.linspace(0.2*maxpower,maxpower, num = steps_in_power_series, endpoint = True)
        powers_down = []                  
        if ramp == True: powers_down = powers_up[:-1][::-1]
        self.Powers = np.append(powers_up, powers_down)
        self.Powers = np.append(self.Powers[:-1], self.Powers)
        kinetic_fac =   max_Power_Series_Number*float(0.2*maxpower)

        attrs = self.trandor.metadata
        attrs['x_axis']=np.flipud(attrs['x_axis'])
        attrs['wavelengths'] = attrs['x_axis']
        attrs['AcquisitionTimes'] = self.trandor.AcquisitionTimings
        attrs['powers']=self.Powers
        attrs['Sleep']=.2
        self.lutter.close_shutter()
        self.wutter.open_shutter()
        self.wizard.particle_group.create_dataset('image_before',data = self.CWL.thumb_image())        
        #self.aligner.optimise(0.003, max_steps=10, stepsize=0.5, npoints=3, dz=0.02,verbose=False)
        data = self.aligner.z_scan(dz =np.arange(-0.25,0.25,0.05))   
        self.wizard.particle_group.create_dataset('z_scan_before', data = data, attrs = data.attrs)   
        self.wutter.close_shutter() 
        self.lutter.open_shutter()        
        time.sleep(0.2)    

        for index, Power in enumerate(self.Powers):
            self.focus_with_laser()            
            self.lutter.close_shutter()
            self.wutter.open_shutter() 
            time.sleep(5)
            self.wizard.particle_group.create_dataset('spectrum_before_%d', data = self.spec.read())     
                        
            self.wutter.close_shutter()
            self.lutter.open_shutter()
            attrs['power'] = Power                   
            Captures = []            
            if self.laser == '_785': self.rotate_to(self.Power_to_Angle(Power))  
            if self.laser == '_633': self.AOM.Power(self.Power_to_Voltage(Power))
            time.sleep(0.2)            
            attrs['measured_power'] = self.read_pometer()  
            nkin = int(kinetic_fac/Power)   
            if nkin<1: nkin = 1
            self.trandor.SetParameter('NKin', nkin)       
            time.sleep(0.2)
            Captures.append(self.trandor.capture()[0])            
            To_Save = []            
            for i in Captures:
                To_Save+=np.reshape(i,[len(i)/1600,1600]).tolist()
                self.wizard.particle_group.create_dataset('power_series_%d',data=To_Save,attrs=attrs)  
            self.lutter.close_shutter()
            self.wutter.open_shutter()
            time.sleep(15)            
            self.wizard.particle_group.create_dataset('spectrum_after_%d', data = self.spec.read())
       
        self.wizard.particle_group.create_dataset('image_after',data = self.CWL.thumb_image())   
        data = self.aligner.z_scan(dz =np.arange(-0.25,0.25,0.05)) 
        self.lutter.close_shutter()
        self.wutter.open_shutter()
        self.wizard.particle_group.create_dataset('z_scan_after', data = data, attrs = data.attrs)    
    def _launch_particle_track(self):
        particle_track_dict = {'aligner' : self.aligner,
                              'spectrometer' : self.spec}        
        self.wizard = TrackingWizard(self.CWL, particle_track_dict, task_list = ['Exp.Power_Series'])
        self.wizard.show()
    
    def focus_with_laser(self):
        initial_exp = self.CWL.camera.exposure
        initial_gain = self.CWL.camera.gain
        initial_angle = self.angle        
        initial_voltage = self.AOM.Get_Power()   
        
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
        if self.laser == 785: self.rotate_to(initial_angle)
        if self.laser == 633: self.AOM.Power(initial_voltage)
    def get_qt_ui(self):
        return self

if __name__ == '__main__': 
    import os
    os.chdir(r'C:\Users\00\Documents\ee306')    
    app = QtWidgets.QApplication(sys.argv)    
    rm= visa.ResourceManager()
    
    spec = OceanOpticsSpectrometer(0) 
    lutter = ThorLabsSC10('COM30')
    FW=[RS.Filter_Wheel()] 
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
    File = datafile.current()
    equipment_dict = {'spec' : spec,
                      'lutter' : lutter,
                      'FW' : FW,
                      'AOM' : aom,
                      'pometer' : pometer,
                      'wutter' : wutter,
                      'cam' : cam,
                      'CWL' : CWL,
                      'trandor' : trandor}
    
   
    exp = Exp(equipment_dict)
    
    exp.show()
        
        
    