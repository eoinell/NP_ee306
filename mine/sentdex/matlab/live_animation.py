# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 16:31:54 2020

@author: Eoin Elliott
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

import time
# Here, the only new import is the matplotlib.animation as animation. This is the module that will allow us to animate the figure after it has been shown.

# Next, we'll add some code that you should be familiar with if you're following this series:

style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
# Now we write the animation function:

def animate(i):
    graph_data = open('example.txt','r').read()
    lines = graph_data.split('\n')
    xs = []
    ys = []
    for line in lines:
        if len(line) > 1:
            x, y = line.split(',')
            xs.append(float(x))
            ys.append(float(y))
    ax1.clear()
    ax1.plot(xs, ys)
# What we're doing here is building the data and then plotting it. Note that we do not do plt.show() here. We read data from an example file, which has the contents of:
d = '''1,5
2,3
3,4
4,7
5,4
6,3
7,5
8,7
9,4
10,4'''
with open('example.txt', 'w+') as f:
    f.write(d)
    

# We open the above file, and then store each line, split by comma, into xs and ys, which we'll plot. Then:

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()