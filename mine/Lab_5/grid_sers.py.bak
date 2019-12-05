# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:25:44 2019

Grid SERS

@author: Eoin Elliott
"""
from scipy.interpolate import griddata
import time
from nplab import datafile

def grid_SERS(ed, group, step_size, steps): # size in um
        
    lutter = ed['lutter']
    CWL = ed['CWL']
    stage = CWL.stage
    wutter = ed['wutter']
    aligner = ed['aligner']
    trandor = ed['trandor']
    trandor.triax.Slit(300)
    lutter.close_shutter()
    wutter.open_shutter()
    time.sleep(2)
    group.create_dataset('image_before',data = CWL.thumb_image()) 
    data = aligner.z_scan(dz =np.arange(-0.25,0.25,0.05))    
    group.create_dataset('z_scan', data = data)
    exp.focus_with_laser()    
    wutter.close_shutter()
    lutter.open_shutter()
    initial_position = stage.get_position() # array   
    z = initial_position[2]
    xs = np.linspace(0,(steps)*step_size, num = steps, endpoint = True) - (steps)*step_size/2
    ys = np.linspace(0,(steps)*step_size, num = steps, endpoint = True) - (steps)*step_size/2
    xs += initial_position[0]
    print xs
    ys += initial_position[1]
    counter = -1
    places = []
#    
    for y in ys:
        counter+=1
        if counter%2 ==0:
            for x in xs:
                places.append([x,y,z])
        else:
            for x in xs[::-1]:
                places.append([x,y,z])
    

#    
    places = np.asarray(places) 

    captures = []            
    trandor.capture()    
    for place in places:
        stage.move(place)
#        print place
        print place-initial_position       
        time.sleep(0.5)
        captures.append(trandor.capture()[0])
    
    
    
    attrs = trandor.metadata
    attrs['places'] = places
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
  
    x = places[0,:,0] # an x axis
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
    
    
grid_sers_dict = {'lutter' : exp.lutter,
                  'CWL' : exp.CWL,
                  'wutter' : exp.wutter,
                  'trandor' : exp.trandor,
                  'focus_with_laser' : exp.focus_with_laser,
                  'aligner' : exp.aligner}    

File = datafile.current()
grid_sers_group = File.create_group('Grid_SERS_BPT_%d')
#exp.Power(0.3)

intensities, places, = grid_SERS(grid_sers_dict, grid_sers_group, 0.1, 11)

plot_grid(intensities, places)