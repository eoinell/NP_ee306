
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt


def watts_to_area(mW, spot_diameter): #mW, um
    uW = mW*1000
    spot_area = ((spot_diameter/2)**2)*np.pi
    umm2 = uW/spot_area
    return umm2
    
deg_array = np.arange(15,375,15)
peak_array = np.zeros(24)
laser_line_array = np.zeros(24)
norm_peak_array = np.zeros(24)
shift_peak_array = np.zeros(24)
norm_fac = watts_to_area(13,30)

for i in range(24):
    
    
    name = "532_100x_1s_15deg_"+str(i+1)+".txt"
    
    array = np.loadtxt(fname = name)
    shift = array[:,0]
    cnts = array[:,1]
    
    

    laser_line_array[i] = np.max(cnts[400:600])
    peak_array[i] = np.max(cnts[600:])
    
    norm_peak_array[i] = peak_array[i]/laser_line_array[i] 
    print np.argmax(cnts[600:])+600
    shift_peak_array[i] = shift[np.argmax(cnts[400:600])+400]

        
        

    
plt.plot(deg_array,peak_array/norm_fac)

plt.legend(frameon = False)
plt.xlabel('angle (deg)', fontname = 'arial',fontsize = '16')
plt.ylabel('Counts/$\mu$m/$\mu$m$^2$', fontname = 'arial', fontsize = '16')
#plt.legend()
plt.tick_params(direction = 'in')
plt.figure()
plt.plot(deg_array,norm_peak_array/norm_fac)
#plt.legend(frameon = False)
plt.xlabel('angle (deg)', fontname = 'arial',fontsize = '16')
plt.ylabel('normalised intensity', fontname = 'arial', fontsize = '16')
#plt.legend()
plt.tick_params(direction = 'in')
plt.figure()
plt.plot(shift_peak_array)