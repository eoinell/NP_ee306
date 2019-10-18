# -*- coding: utf-8 -*-
"""
Created on Sun May 26 11:38:08 2019

@author: np-albali
"""

import os
import nplab
import re
import h5py
from nplab import datafile as df
def findH5File(rootDir, mostRecent = False, nameFormat = 'date'):
    '''
    Finds either oldest or most recent .h5 file in a folder whose name contains a specified string
    '''

    os.chdir(rootDir)

    if mostRecent == True:
        n = -1

    else:
        n = 0

    if nameFormat == 'date':

        if mostRecent == True:
            print 'Searching for most recent instance of yyyy-mm-dd.h5 or similar...'

        else:
            print 'Searching for oldest instance of yyyy-mm-dd.h5 or similar...'

        h5File = sorted([i for i in os.listdir('.') if re.match('\d\d\d\d-[01]\d-[0123]\d', i[:10])
                         and (i.endswith('.h5') or i.endswith('.hdf5'))],
                        key = lambda i: os.path.getmtime(i))[n]

    else:

        if mostRecent == True:
            print 'Searching for most recent instance of %s.h5 or similar...' % nameFormat

        else:
            print 'Searching for oldest instance of %s.h5 or similar...' % nameFormat

        h5File = sorted([i for i in os.listdir('.') if i.startswith(nameFormat)
                         and (i.endswith('.h5') or i.endswith('.hdf5'))],
                        key = lambda i: os.path.getmtime(i))[n]

    print '\tH5 file %s found\n' % h5File

    return h5File
filename = findH5File(os.getcwd())
File = h5py.File(filename, 'r+')
del File['ParticleScannerScan_1']
File.close()