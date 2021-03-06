# -*- coding: utf-8 -*-
__author__ = 'Eoin Elliott'
"""
A utility for adding any number of graphs,
plotting any number of equations each, and varying
any number of parameters in these equations through
a gui.
"""

import numpy as np
import pyqtgraph as pg
from qtpy import QtGui, QtWidgets, QtCore

class GraphWidget(pg.PlotWidget):
    '''
    template for an interactive graph
    
    Input: 
        equation: should be a function of only 1 variable (x). 
            Parameters to be varied should be a Parameter object.
        xlim: interval over which the function will be plotted
        ylin: currently does nothing
        
    make use of xlabel and ylabel methods!
        
    '''
    def __init__(self, *args, 
                 xlim=(-10,10),
                 ylim=(0,100),
                 title='graph',
                 xlabel = 'X axis',
                 ylabel = 'Y axis'):
        super().__init__(title=title)
        self.equations = args
        self.xlim = xlim
        self.ylim = ylim
        self.title(title)
        self.xlabel(xlabel)
        self.ylabel(ylabel)
        self.hasLegend=False
        self.addLegend()
    @property
    def x(self):
        return np.linspace(*self.xlim, num=100)
    
    @property
    def ys(self):
        return [equation(self.x) for equation in self.equations]
        
    def update(self):
        def name(eq): # takes the name of the function the first time,
        #and returns none after to stop the legend from exploding
            if self.hasLegend: return None
            return str(eq).split(' ')[1]
            
        self.clearPlots() 
        for i,(y,eq) in enumerate(zip(self.ys, self.equations)): 
            self.plot(self.x, y, pen=(i,len(self.ys)),name=name(eq))

    def xlabel(self, label):
        self.setLabel('bottom', label)
    def ylabel(self, label):
        self.setLabel('left', label)
    def title(self, title):
        self._title = title
        self.setTitle(title)
    def export(self):
        print('x, y(s):',self.x, self.ys, sep='\n')


class GraphGroup(QtGui.QGroupBox):
    '''
    feed me GraphWidget objects and 
    I'll lay them out horizontally
    '''
    def __init__(self, graphs):
        super().__init__('Graphs')
        self.setLayout(QtWidgets.QHBoxLayout())
        self.graphs = graphs
        for g in graphs:
            self.layout().addWidget(g)
    def update_graphs(self):
        for g in self.graphs:
            g.update()
            g.hasLegend = True
    def export(self):
        for g in self.graphs:
            print('Graph: ', g._title)
            g.export()
            
            
class FloatMathMixin():
    
    def __add__(self, other):
        return float(self) + np.array(other)
    def __sub__(self, other):
        return float(self) - np.array(other)
    def __mul__(self, other):
        return float(self)*np.array(other)
    def __truediv__(self,other):
        return float(self)/np.array(other)
    def __pow__(self, other):
        return float(self)**np.array(other)
    
    def __radd__(self,other):
        return self.__add__(other)
    def __rsub__(self,other):
        return self.__sub__(other)
    def __rmul__(self,other):
        return self.__mul__(other)
    def __rtruediv__(self,other):
        return self.__truediv__(other)
    def __rpow__(self, other):
        return self.__pow__(other)
                
class Parameter(QtWidgets.QWidget, FloatMathMixin):
    '''
    Representation of a parameter to be varied in an equation.
    Takes its value from the Gui.
    Supports basic array math.
    
    Inputs:
        name: the label the paramter will have in the gui
        Default: it's initial value
        Min: minimum value allowed to be entered in the gui
        Max: maximum...
        
    '''
    
    param_changed = QtCore.Signal(int)
    def __init__(self, name, Default=1,Min=0,Max=100,units=None):
        super().__init__()
        self.name = name
        self.units = f' ({units})' if units is not None else ''
        self.setLayout(QtWidgets.QFormLayout())
        self.layout().addWidget(QtGui.QLabel(self.name+self.units))
        self.box = QtGui.QDoubleSpinBox()
        self.layout().addWidget(self.box)
        self.box.setValue(Default)
        self.box.valueChanged.connect(self.param_changed.emit)
    def __repr__(self):
        return str(self.box.value())
    def __str__(self):
        return f'Parameter {self.name}: {self.box.value()} {self.units}'
    def __int__(self):
        return int(self.box.value())
    def __float__(self):
        return float(self.box.value())
    
       
class ParameterWidget(QtGui.QGroupBox):
    '''
    feed me parameters and i'll add spinBoxes for them, and 
    emit a signal when they're changed to update the graphs. 
    '''
    param_changed = QtCore.Signal(int)
    def __init__(self, parameters):
        super().__init__('Parameter controls')
        self.parameters = parameters
        self.setLayout(QtWidgets.QHBoxLayout())
        for p in self.parameters:
            self.layout().addWidget(p)
            p.param_changed.connect(self.param_changed.emit)
    def export(self):
        for p in self.parameters:
            print(p)
            
class LivePlotWindow(QtWidgets.QMainWindow):
    '''Puts the graphing and parameter widgets together'''
    def __init__(self, graphing_group, parameter_widget):  
        super().__init__()
        QtGui.QApplication.setPalette(QtGui.QApplication.style().standardPalette())
        layout = QtWidgets.QVBoxLayout()
        self.resize(1500,1500)
        layout.addWidget(graphing_group)
        export_button = QtGui.QPushButton('Export values')
        export_button.clicked.connect(self.export)
        layout.addWidget(export_button)
        layout.addWidget(parameter_widget)

        self.graphing_group = graphing_group
        self.parameter_widget = parameter_widget
        self.setWindowTitle('Quasi-Normal Modes')
        # self.setWindowIcon(QtGui.QIcon('bessel.png'))
        self.setWindowIcon(QtGui.QIcon('maxwell.png'))
        self.widget = QtGui.QWidget()
        self.widget.setLayout(layout)
        self.setCentralWidget(self.widget)
        self.parameter_widget.param_changed.connect(self.update_graphs)
        self.update_graphs()
    
    def update_graphs(self):
        self.graphing_group.update_graphs()        
    
    def export(self):
        self.graphing_group.export()
        self.parameter_widget.export()

if __name__ == '__main__':
    #Initialize all parameters
    A = Parameter('A', 15, Min=0, Max=10, units= 'm')
    B =  Parameter('B', 6)
    C = Parameter('C',15)
    D = Parameter('D',6)
    parameter_widget = ParameterWidget([A,B,C,D])
    #define the equations for each plot
    def equation1(x):
        return A*x**2 + B*x**2 - C*x + D
    def equation2(x):
        return A*x**3 - (B/C)*x**2 + D
    def eq3(x):
        return A**np.sin(C*x)/D*x
    def eq4(x):
        return (np.sin(A*x)/B*x) +D
    
    graph1 = GraphWidget(equation1, equation2, title='1st')
    graph2 = GraphWidget(equation2, xlim=(-5,5), title='2nd')
    g3 = GraphWidget(eq3, title='etc,', xlabel = ':)')
    g4 = GraphWidget(eq4, title='etc.')
    graphs = GraphGroup([graph1,graph2, g3, g4])

    live_plot_window = LivePlotWindow(graphs, parameter_widget)
    live_plot_window.show() 
   

