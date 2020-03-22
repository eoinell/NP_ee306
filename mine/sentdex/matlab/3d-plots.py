# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:01:33 2020

@author: Eoin Elliott
"""

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import random
import numpy as np

def f(x, y):
   return np.sin(np.sqrt(x ** 2 + y ** 2))
	
x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_wireframe(X, Y, Z, color='black')
ax.set_title('wireframe')
plt.show()
