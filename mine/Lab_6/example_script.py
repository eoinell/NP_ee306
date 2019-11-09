# -*- coding: utf-8 -*-
"""
Created on Thu Aug 01 16:38:56 2019

@author: ee306

"""
desc = """This is an example script. It can be run through the 'scripts' dropdown of the main gui. 
The idea is that experiments that don't need to be gui-controlled be written into a script, either because they're once-off or 
only used by 1 person be written here.

It's also really good for writing/testing code as you don't have to re-initialise all the instruments every time you make a change!
Just save the script, and run again.
"""

print desc

lab.example()

lutter.close_shutter()
time.sleep(1)
winsound.Beep(100, 1000)
lutter.open_shutter()

"""
If you want the script to be more general to other labs (always good)
then make a function that takes an equipment dictionary argument. You'll have to redefine the dictionary though!
"""
def example_script_function(equipment_dictionary)
    lutter = ed['lutter']
    CWL = ed['CWL']
    stage = CWL.stage
    wutter = ed['wutter']
    aligner = ed['aligner']
    example = ed['example']
    wutter.close_shutter()
    winsound.Beep(200, 1000)
    wutter.open_shutter()
    example(steps = 6)


equipment_dictionary = {'lutter' : lutter,
                        'CWL' : CWL,
                        'wutter' : wutter,
                        'example' : lab.modal_example,
                        'aligner' : lab.aligner}    

example_script_function(equipment_dictionary)
