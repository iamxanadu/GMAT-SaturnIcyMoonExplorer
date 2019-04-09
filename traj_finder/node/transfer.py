"""
Created on Apr 8, 2019

@author: David Mueller
"""
from poliastro.twobody.orbit import Orbit
from poliastro.bodies import Sun
from poliastro.iod import izzo

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
        (start_v, end_v), = izzo.lambert(self.central_body.k, 
                                             depart_orbit.r, 
                                             target_orbit.r, 
                                             duration)
        
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
        