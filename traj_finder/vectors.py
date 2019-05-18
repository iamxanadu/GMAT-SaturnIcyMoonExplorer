"""
Created on Apr 20, 2019

@author: David Mueller
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from mpl_toolkits.mplot3d import Axes3D
from quaternion import Matrix4x4
import numpy as np
import math
from matplotlib.projections.polar import PolarTransform
from matplotlib.ticker import MultipleLocator

FIG_WIDTH = 8.5
FIG_DPI = 127.33567
SAVE_DPI = 300

SAVE_OUT = True

PASTEL_RED = (.867, .467, .467, 1)
PASTEL_GREEN = (.467, .867, .467, 1)
PASTEL_BLUE = (.467, .467, .867, 1)

def fig_size(aspect):
    return (FIG_WIDTH, FIG_WIDTH/aspect)

def size_label(label):
    return "%s (actual size @ %.1f DPI)" % (label, FIG_DPI)

class Vector:
    QUIVER_ANGLES = 'xy'
    QUIVER_SCALE_UNITS = 'xy'
    QUIVER_SCALE = 1
    
    _quiver_list = []
    
    def __init__(self, u=0, v=0, w=0, r=None, theta=None, theta_deg=None, 
                 np_arr=None):
        assert theta is None or theta_deg is None
        assert not ((r is None) ^ (theta is None and theta_deg is None))
        
        if theta_deg is not None:
            theta = math.radians(theta_deg)
            
        if np_arr is not None:
            self.array = np_arr
            
        elif r is not None:
            self.array = r * np.array([0,0,0,math.cos(theta), math.sin(theta), w])
            
        else:
            self.array = np.array([0,0,0,u,v,w])
            
        # hatch accepts:
        #     [ ‘/’ | ‘\’ | ‘|’ | ‘-‘ | ‘+’ | ‘x’ | ‘o’ | ‘O’ | ‘.’ | ‘*’ ]
        self.plot_kwargs = {
            'color'      : 'black',
            #'linestyle'  : 'solid',
            'linewidth' : 1.0,
            }
        
#         'width'      : .005,
#             'facecolor'  : 'black',
#             'linestyle'  : 'solid',
#             'edgecolor'  : 'black',
#             'linewidth'  : 1.0, 
#             'hatch'      : None,}
        
        self.quiver_kwargs = {
            'angles'     : self.QUIVER_ANGLES, 
            'scale_units': self.QUIVER_SCALE_UNITS, 
            'scale'      : self.QUIVER_SCALE,
            'width'      : .0075,
            'facecolor'  : 'black',
            'edgecolor'  : 'black',
            'hatch'      : None,
            'pivot'      : 'tail',}
    
    @property
    def theta(self):
        return math.atan2(self.y, self.x)
    
    @property
    def r(self):
        return self.norm()
    
    @property
    def delta_theta(self):
        return self.tip_theta - self.p_theta
    
    @property
    def delta_r(self):
        return self.tip_r - self.p_r
    
    @property
    def tip_theta(self):
        return math.atan2(self.py+self.y, self.px+self.x)
    
    @property
    def tip_r(self):
        return math.sqrt((self.px + self.x)**2
                         + (self.py + self.y)**2
                         + (self.pz + self.z)**2)
    
    @property
    def p_theta(self):
        return math.atan2(self.py, self.px)
    
    @property
    def p_r(self):
        return math.sqrt(self.px**2+self.py**2+self.pz**2)
    
    @property
    def px(self):
        return self.array[0]
    
    @property
    def py(self):
        return self.array[1]
    
    @property
    def pz(self):
        return self.array[2]
    
    @property
    def x(self):
        return self.array[3]
    
    @property
    def y(self):
        return self.array[4]
    
    @property
    def z(self):
        return self.array[5]
            
    @property
    def value(self):
        return self.array[-3:]
    
    @property
    def pivot(self):
        return self.quiver_kwargs['pivot']
    
    @pivot.setter
    def pivot(self, value):
        self.quiver_kwargs['pivot'] = value
            
    def clone(self):
        return np.copy(self.array)
    
    @classmethod
    def from_np(cls, np_arr):
        return Vector(np_arr=np_arr)
    
    @classmethod
    def cross(cls, left, right):
        temp = np.cross(left.value, right.value)
        new_array = np.concatenate(([0,0,0], temp))
        return Vector(np_arr=new_array)
        
    def norm(self):
        return np.linalg.norm(self.value)
    
    def normalize(self):
        norm = self.norm()
        if norm == 0:
            raise RuntimeError("divide by zero")
        return Vector(np_arr=self.array/norm)
    
    def plot(self, label, as_line=False, **plot_kwargs):        
        if as_line:
            kwargs = {**self.plot_kwargs, **plot_kwargs}
            plt.plot([self.px, self.x], [self.py, self.y], **kwargs)
        else:
            kwargs = {**self.plot_kwargs, 
                      **self.quiver_kwargs, 
                      **plot_kwargs}
            q = plt.quiver(self.px, self.py, self.x, self.y, **kwargs)
            self._quiver_list.append({'Q': q, 'label': label})
            
    def plot_polar(self, label, as_line=False, **plot_kwargs):
        if as_line:
            kwargs = {**self.plot_kwargs, **plot_kwargs}
            plt.plot([self.p_theta, self.delta_theta], 
                     [self.tip_r, self.delta_r], **kwargs)
        else:
            kwargs = {**self.plot_kwargs, 
                      **self.quiver_kwargs, 
                      **plot_kwargs}
            q = plt.quiver(self.p_theta, self.p_r, 
                           self.delta_theta, self.delta_r, **kwargs)
            
            self._quiver_list.append({'Q': q, 'label': label})
            
    @classmethod
    def quiver_labels(cls, ax, center, U=10):
        y_span = .5
        x_center = center[0]
        y_center = center[1]
        
        y_start = y_center-.5*y_span
        y_stop =  y_center+.5*y_span
        
        x_pos = x_center
        y_pos = np.linspace(y_start, y_stop, len(cls._quiver_list))
        
        for i in range(len(cls._quiver_list)):
            ax.quiverkey(X=x_pos, Y=y_pos[i], U=U, **cls._quiver_list[i])
            
    
    def rotate_about(self, rot_vec, angle):
        m = Matrix4x4()
        m.from_axis_angle(rot_vec.normalize(), angle)
#         print("rot@(%.0f):\n" % math.degrees(angle), m.rotation_only())
        
        # not sure why the transpose on the rotation matrix is needed. Confirmed
        # that the matrix is indeed transposed from the value it should be. i.e.
        # this isn't a hack, but correcting some unknown issue in the creation
        # of the matrix
        temp = self.value @ m.rotation_only().transpose()
        
#         print("in: ", self.value)
#         print("out: ", temp)
        return Vector(np_arr=np.concatenate(([0,0,0], temp)))
    
    def translate(self, that):
        (x,y,z) = self.array[:3] + that.array[-3:]
        return Vector.from_np(np.array([x, y, z, *self.value]))
    
    def __pow__(self, that):
        if that != 2:
            raise TypeError("unsupported operand type for pow(): 'Vector' and 'int'")
        return self.norm()**that
    
    def __add__(self, that):
        return Vector.from_np(self.array + that.array)
    
    def __neg__(self):
        return Vector.from_np(-self.array)
    
    def __sub__(self, that):
        try:
            return Vector(np_arr=(self.array - that.array))
        except:
            return Vector(np_arr=(self.array - that))
        
    def __mul__(self, that):
        return Vector(np_arr=(self.array * that))
    
    def __rmul__(self, that):
        return Vector(np_arr=(that * self.array))
    
    def __iter__(self):
        return iter(self.array)
    
    def __str__(self):
        return str(self.array)
        


