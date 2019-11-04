# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:25:44 2019

Grid SERS

@author: Eoin Elliott
"""
from scipy.interpolate import griddata
import time
from nplab import datafile

def grid_SERS(ed, size, steps): # size in um
    
    lutter = ed['lutter']
    CWL = ed['CWL']
    stage = CWL.stage
    wutter = ed['wutter']

    trandor = ed['trandor']
    lutter.close_shutter()
    wutter.open_shutter()
 
    wutter.close_shutter()
    lutter.open_shutter()
    initial_position = stage.get_position() # array
    z = 0
    xs = np.linspace(-size/2., size/2., num = steps)
    ys = xs
#    xs += initial_position[0]
#    ys += initial_position[1]
    counter = -1
    places = []
    
    for y in ys:
        counter+=1
        if counter%2 ==0:
            for x in xs:
                places.append([x,y,z])
        else:
            for x in xs[::-1]:
                places.append([x,y,z])
    
    places = np.array(places)
#    plt.figure()
#    plt.plot(places[:,[0,1]])
    captures = []            
    for place in places:
        stage.move_rel(place)
                
        time.sleep(0.5)
        captures.append(trandor.capture()[0])
    
    
    File = datafile.current()
    
    attrs = trandor.metadata
    attrs['places'] = places
    group = File.create_group('Grid SERS')
    group.create_dataset('SERS', data = captures, attrs = attrs) 
    stage.move(initial_position)
    lutter.close_shutter()
    wutter.open_shutter()
    
    return np.sum(captures, axis = 1), places    

def plot_grid(intensities, places):
    
    side = int((len(intensities)**0.5))
    places = np.reshape(places, [side, side, 3])
    #xys = np.take(places, [0,1], axis = 2)
    #xys.reshape(xys.size/2, 2)
  
    x = places[0,:,0] # an ex axis
    y = places[:,0,1] # a y axis
 
    xx, yy = np.meshgrid(x, y) #x and y coordinates to be evalueated at
    xy_evalled = [[x,y] for x, y in zip(np.ravel(xx), np.ravel(yy))]
    
    xs = places[:,:,0].flatten() # the coordinates of the actual measurements
    ys = places[:,:,1].flatten()

    xy = np.transpose(np.append([xs], [ys], axis = 0))
       
    zz = griddata(xy, intensities, xy_evalled,  method = 'linear')    
   
    plt.figure()
    plt.pcolormesh(np.reshape(zz, [side, side]))
    plt.colorbar()
    plt.figure()    
    plt.plot(intensities)
    
    
grid_sers_dict = {'lutter' : Exp.lutter,
                  'CWL' : Exp.CWL,
                  'wutter' : Exp.wutter,
                  'trandor' : Exp.trandor}    

intensities, places, = grid_SERS(grid_sers_dict, 0.5, 2)

plot_grid(intensities, places)