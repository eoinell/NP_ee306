# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 12:23:58 2020

@author: Eoin Elliott
"""

from nplab.utils.notified_property import DumbNotifiedProperty, NotifiedProperty, register_for_property_changes


class A():
    dumb = DumbNotifiedProperty(True)
    def __init__(self):
        self._prop = 10
    def get_prop(self):
        return self._prop
    def set_prop(self, value):
        print('setting')
        self._prop = value
    prop = NotifiedProperty(get_prop, set_prop)
    def test(self):
        print(type(self.prop))
        print(type(self.__class__.prop))
    
class B():
    def __init__(self, a):    
        self.a = a
        register_for_property_changes(self.a, 'dumb', self.dumb_changed)
        register_for_property_changes(self.a, 'prop', self.prop_changed)
    def dumb_changed(self, new):
        print('dumb changed to ' + str(new))
    def prop_changed(self, new):
        print('prop changed to ' + str(new))
def here(new):
        print('here->',new)    
if __name__ == '__main__':
    a = A()
    b = B(a)
    a.dumb = False
    a.prop = 11
    # print('_______________')
    # a = A()
    # register_for_property_changes(a, 'dumb', here)
    # register_for_property_changes(a, 'prop', here)
    # a.dumb = False
    # a.prop = 11
    # b = B(a)
    # b.a.dumb = True
    # b.a.prop = 12
    # print(1, isinstance(a.dumb, NotifiedProperty))
    # print(2, isinstance(a.prop, NotifiedProperty))
    
    
    