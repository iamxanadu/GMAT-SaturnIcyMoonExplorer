"""
Created on Apr 8, 2019

@author: David Mueller
"""
import math

import astropy.units as u
import numpy as np

from poliastro.twobody.orbit import Orbit
from poliastro.bodies import Sun
from poliastro.iod import izzo
from poliastro.iod import vallado

class Transfer:

    def __init__(self, from_body, to_body):
        self.start_body = from_body
        self.end_body = to_body
        self.start_v = None
        self.end_v = None
        self.start_epoch = None
        self.end_epoch = None
        self.initial_orbit = None
        self.final_orbit = None
        
    def plot(self):
        self.initial_orbit.plot()
        
    @property
    def duration(self):
        return self.end_epoch - self.start_epoch
    '''
    def __str__(self):
        str(self.start_body) + " -> " + str(self.end_body)
        '''
        
class TransferPlanner:
    
    def __init__(self, central_body=Sun):
        self.central_body = central_body
        self.start_body = None
        self.end_body = None
        
    def target_orbit(self, epoch):
        return Orbit.from_body_ephem(self.end_body, epoch)
    
    def depart_orbit(self, epoch):
        return Orbit.from_body_ephem(self.start_body, epoch)
    
    @staticmethod
    def project_to_plane(vector, normal):
        from poliastro.util import norm
        
        unitless_vec = vector.value

        normal_proj = np.dot(unitless_vec, normal)*normal
        result_hat = unitless_vec - normal_proj
        result_hat = result_hat / norm(result_hat)
        return result_hat * norm(vector)
    
    @staticmethod
    def match_to_plane(start_v, flyby_body, time_j):
        from poliastro.core.util import cross
        from poliastro.util import norm
        
        J_orbit = Orbit.from_body_ephem(flyby_body, time_j)
        
        def project_to_plane(vec, normal_vec):
            normal_proj = np.dot(vec, normal_vec)*normal_vec
            return vec - normal_proj
        
        J_hat = cross(J_orbit.r, J_orbit.v)
        J_hat = (J_hat / (norm(J_orbit.r) * norm(J_orbit.v) / u.km**2 * u.s)).value
#         print("J_hat:      ", J_hat)
#         print("start_v:    ", start_v)
        start_v_hat = project_to_plane(start_v/u.km*u.s, J_hat)
        start_v_hat = start_v_hat / norm(start_v_hat)
#         print("start_v_hat:", start_v_hat)
        start_v = start_v_hat * norm(start_v)
        
        return start_v
    
    @staticmethod
    def match_flyby(v_depart, flyby_body, epoch, theta_inf, z_hat=None):
        from poliastro.util import norm
        from vectors import Vector
        x,y,z = v_depart / u.km * u.s
        v_d = Vector(x,y,z)

        body_orbit = Orbit.from_body_ephem(flyby_body, epoch)
        r_p = body_orbit.r
        x,y,z = body_orbit.v / u.km * u.s
        v_p = Vector(x,y,z)
        v_out = v_d-v_p
        v_inf = v_out.norm()
        
        if z_hat is None:
            h_vec = body_orbit.h_vec
            h_hat = h_vec / norm(h_vec)
            x,y,z = h_hat
            z_hat = -Vector(x,y,z)
            
        r_ref = v_out.rotate_about(z_hat, theta_inf)
        v_in = -v_out.rotate_about(z_hat, 2*theta_inf)
#         print("v_in:", v_in)
#         print("|v_in|:", v_in.norm())
        v_a = v_in + v_p
#         print("v_a:", v_a)
        v_a_final = np.array(v_a.value)
#         print("v_a_final:", v_a_final)
        
#         v_in_extend = v_in
        v_in = v_in.translate(-v_in)
        v_a = v_a.translate(-v_a)
        v_a = v_a.value * u.km / u.s
        
        return v_a
        
    def make_transfer(self, start_epoch=None, end_epoch=None, duration=None):
        assert (start_epoch is None) ^ (end_epoch is None) ^ (duration is None)
        
        if start_epoch is None:
            start_epoch = end_epoch - duration
        elif end_epoch is None:
            end_epoch = start_epoch + duration
        else:
            duration = end_epoch - start_epoch

        depart_orbit = self.depart_orbit(start_epoch)
        target_orbit = self.target_orbit(end_epoch)
    
        #compute transfer lambert trajectory
#         try:
        (start_v, end_v), = vallado.lambert(self.central_body.k, 
                                             depart_orbit.r, 
                                             target_orbit.r, 
                                             duration,
                                             numiter=100)
#         except RuntimeError:
#             (start_v, end_v), = izzo.lambert(self.central_body.k, 
#                                                   depart_orbit.r, 
#                                                   target_orbit.r, 
#                                                   duration,
#                                                   numiter=1000)
        
        xfer = Transfer(self.start_body, self.end_body)
        xfer.initial_orbit = Orbit.from_vectors(self.central_body, 
                                                depart_orbit.r, 
                                                start_v, 
                                                epoch=start_epoch)
        
        xfer.final_orbit = Orbit.from_vectors(self.central_body, 
                                              target_orbit.r, 
                                              end_v, 
                                              epoch=end_epoch)
        
        xfer.start_v = start_v
        xfer.end_v = end_v
        xfer.depart_orbit = depart_orbit
        xfer.target_orbit = target_orbit
        xfer.start_epoch = start_epoch
        xfer.end_epoch = end_epoch
        
        return xfer
        