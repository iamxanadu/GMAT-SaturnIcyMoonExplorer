"""
Created on Apr 8, 2019

@author: david
"""
import astropy.units as u
from poliastro.util import norm
from poliastro.twobody.orbit import Orbit
import math

class Boundary:
    
    def __init__(self, central_body, epoch, inbound_orbit=None, outbound_orbit=None):
        assert (inbound_orbit is None) ^ (outbound_orbit is None), \
            "inbound_orbit: %s, outbound_orbit: %s" % (inbound_orbit, outbound_orbit)
        
        self.central_body = central_body
        self.epoch = epoch
        
        # heliocentric orbit
        if inbound_orbit is not None:
            self.orbit = inbound_orbit
        else:
            self.orbit = outbound_orbit
        
    @property
    def relative_v(self):
        return self.orbit.v
        
    @property
    def parent_orbit(self):
        return Orbit.from_body_ephem(self.central_body, self.epoch)
        
    @staticmethod
    def from_transfers(inbound_xfer=None, outbound_xfer=None):
        assert (inbound_xfer is None) ^ (outbound_xfer is None), \
            "inbound_xfer: %s, outbound_xfer: %s" % (inbound_xfer, outbound_xfer)
        
        if inbound_xfer is not None:
            return Capture(central_body = inbound_xfer.end_body, 
                           epoch = inbound_xfer.end_epoch, 
                           helio_orbit = inbound_xfer.final_orbit)
        else:
            return Escape(central_body = outbound_xfer.start_body, 
                          epoch = outbound_xfer.start_epoch, 
                          helio_orbit = outbound_xfer.final_orbit)
            
        assert False, "shouldn't get here"
            
class Capture(Boundary):
    def __init__(self, central_body, epoch, helio_orbit):
        super().__init__(central_body, 
                         epoch, 
                         inbound_orbit=helio_orbit)
    
    @property
    def delta_v(self):
        # TODO improve this: capture can be much cheaper if we specify the
        # capture orbit
        return norm(self.parent_orbit.v - self.relative_v).to(u.km/u.s)
    
class Escape(Boundary):
    def __init__(self, central_body, epoch, helio_orbit):
        super().__init__(central_body, 
                         epoch, 
                         outbound_orbit=helio_orbit)
        
    @property
    def delta_v(self):
        return norm(self.parent_orbit.v - self.relative_v)

class Flyby:

    def __init__(self, central_body, inbound_v, outbound_v, epoch, outbound_basis=True):
        self.central_body = central_body
        self.inbound_v = inbound_v
        self.outbound_v = outbound_v
        self.outbound_basis = outbound_basis
        self.epoch = epoch
    
    @property
    def parent_orbit(self):
        return Orbit.from_body_ephem(self.central_body, self.epoch)
    
    @property
    def delta_v(self):
        return norm(self.outbound_v - self.inbound_v)
    
    @property
    def psi(self):
        """Turning angle, in radians"""
        return math.acos((self.v_i.dot(self.v_f)) / (norm(self.v_i)*norm(self.v_f)))
        
    @property
    def psi_deg(self):
        """Turning angle, in degrees"""
        return math.degrees(self.psi)
        
    @property
    def r_p(self):
        """Perifocal distance"""
        return self.central_body.k/(self.v_inf**2) \
            *((.5 - self.v_i.dot(self.v_f) / (2*norm(self.v_i)*norm(self.v_f)))**(-.5) - 1)
              
    @property
    def r_pr(self):
        """Ratio of perifocal distance to central-body radius"""
        return float(self.r_p / self.central_body.R)
    
    @property
    def v_i(self):
        """Central-body relative velocity at encounter"""
        return self.inbound_v - self.parent_orbit.v
    
    @property
    def v_f(self):
        """Central-body relative velocity at escape"""
        return self.outbound_v - self.parent_orbit.v
    
    @property
    def v_err(self):
        """
        Error in v_inf value of non-basis velocity vector
        
        Normally v_inf is assumed to be constant on the inbound vector and
        outbound vector, but the construction of this flyby doesn't require this
        to be the case.
        """
        if self.outbound_basis:
            return norm(self.outbound_v) - norm(self.inbound_v)
        else:
            return norm(self.inbound_v) - norm(self.outbound_v)
    
    @property
    def v_inf(self):
        """Central-body relative velocity magnitude"""
        """
        if self.outbound_basis:
            return norm(self.outbound_v - self.parent_orbit.v)
        else:
            return norm(self.inbound_v - self.parent_orbit.v)
        """
        return (norm(self.outbound_v - norm(self.parent_orbit.v))
                + norm(self.inbound_v) - norm(self.parent_orbit.v)) / 2
                
    @classmethod
    def from_transfers(cls, inbound_xfer, outbound_xfer):
        assert inbound_xfer.end_epoch == outbound_xfer.start_epoch
        assert inbound_xfer.end_body == outbound_xfer.start_body
        return Flyby(inbound_xfer.end_body,
                     inbound_xfer.end_v, 
                     outbound_xfer.start_v, 
                     inbound_xfer.end_epoch)
