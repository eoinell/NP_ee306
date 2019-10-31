# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:25:44 2019

Grid SERS

@author: Eoin Elliott
"""
from scipy.interpolate import griddata


def grid_SERS(ed, size, steps): # size in um
    
    lutter = ed['lutter']
    cam = ed['cam']
    wutter = ed['wutter']
    stage = ed['stage']
    trandor = ed['trandor']
    
    
    
    wutter.close_shutter()
    initial_position = stage.get_position() # array
    z = initial_position[2]
    xs = np.linspace(-size/2., size/2., num = steps)
    ys = xs
    xs += initial_position[0]
    ys += initial_position[1]
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
    captures = []            
    for place in places:
        stage.move(place)
        time.sleep(0.5)
        captures.append(trandor.capture()[0])
    
    to_save = np.reshape(captures, [len(captures)/1600, 1600]).to_list()
    File = datafile.current()
    
    attrs = trandor.metadata
    attrs['places'] = places
    group = File.create_group('Grid SERS')
    group.create_dataset('SERS', data = to_save, attrs = attrs) 
    stage.move(initial_position)
    lutter.close_shutter()
    wutter.open_shutter()
    
    return np.sum(to_save, axis = 1), places    

def plot_grid(intensities, places):
   
    side = int((len(intensities)**0.5))
    places = np.reshape(places, [side, side, 3])
    xys = np.take(places, [0,1], axis = 2)
    xys.reshape(xys.size/2, 2)
    xs = places[0,:,0]
    ys = places[:,0,1]
    
    zz = griddata(xys, intensities, (ys, xs),  method = 'linear')
    plt.figure()
    plt.pcolormesh(zz)
    plt.colorbar()
    
    
    
    
intensities, places, = grid_SERS(grid_sers_dict, 0.5, 5)

plot_grid(intensities, places)