#!/usr/bin/env python3
# coding: utf-8

# In[1]:

import numpy as np

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris

from poliastro.bodies import Sun, Venus, Earth, Jupiter, Saturn
from poliastro.core.util import cross
from poliastro.plotting import OrbitPlotter2D
from poliastro.util import norm

from scipy import optimize as opt
from node.transfer import TransferPlanner
from node.flyby import Boundary, Flyby
import random
import math
from poliastro.twobody.orbit import Orbit
from astropy.units.quantity import Quantity
from poliastro.twobody.propagation import cowell, kepler

TIME_SCALE="tdb"

def causality_constraint(before, after, delta=30):
    def _cmp(args):
        return args[after] - args[before] - delta
    return _cmp

def flyby_constraint(index, minimum, flyby_list):
    def _cmp(_args):
        try:
            return flyby_list[index].r_pr - minimum
        except IndexError:
            print("caught")
            return 1
    return _cmp
    
def make_z_hat(x_vec, y_vec):
    z_vec = cross(x_vec, y_vec)
    z_hat = z_vec / np.linalg.norm(z_vec)
    if z_hat[2] < 0:
        z_hat = -z_hat
        
def xfer_into_xfer(js_orbit, theta_inf_deg, target_body, origin_body, origin_epoch_guess):
    from vectors import Vector
    j_theta_inf = math.radians(theta_inf_deg)
    
    target_epoch = js_orbit.epoch
    
    origin_orbit = Orbit.from_body_ephem(origin_body, origin_epoch_guess)
    target_orbit = Orbit.from_body_ephem(target_body, target_epoch)
    
    x,y,z = cross(origin_orbit.r, target_orbit.r)
    
    # ensure z_hat always points "down"
    if z >= 0:
        z_hat = -Vector(x,y,z).normalize()
    else:
        z_hat = Vector(x,y,z).normalize()
#     print("z_hat:", z_hat)
    v_approach_oop = TransferPlanner.match_flyby(js_orbit.v, 
                                                 target_body, 
                                                 target_epoch, 
                                                 j_theta_inf,
                                                 z_hat)
    
    v_approach = TransferPlanner.project_to_plane(v_approach_oop, 
                                                  Quantity(z_hat.value))
    
#     print("v_approach:", v_approach)
#     print("js_orbit.r:", js_orbit.r)
#     print("target_epoch:", target_epoch)
    result = Orbit.from_vectors(Sun, 
                              js_orbit.r, 
                              v_approach, 
                              epoch=target_epoch)
#     print("result:", result)
    return result

def xfer_err(base_orbit, start_body):
    """
    Error in the closest approach to the starting body
    """
    def _inner(delta_t=None, back_prop_xfer=None):
#         print("delta_t:", delta_t*u.day)
        if back_prop_xfer is None:
            assert delta_t is not None
            back_prop_xfer = base_orbit.propagate(delta_t*u.day, method=kepler)
        start_orbit = Orbit.from_body_ephem(start_body, 
                                            back_prop_xfer.epoch)
#         print("t_f:", base_orbit.epoch)
#         print("t_0:", back_prop_xfer.epoch)
#         print("delta_r:", (norm(base_orbit.r) - norm(back_prop_xfer.r)).to(u.au))
#         print("r_f:", norm(base_orbit.r).to(u.au))
#         print("r_0:", norm(back_prop_xfer.r).to(u.au))
        r_err = back_prop_xfer.r - start_orbit.r
        return norm(r_err) / u.km
    return _inner

class RouteBuilder(object):
    
    def __init__(self, base_epoch_body, base_epoch):
        self.base_epoch_body = base_epoch_body
        self.base_epoch = base_epoch
        
    def find_flyby(self, x):
        V_epoch = self.base_epoch
#         print("trying:", x)
        js_delta_t = x[0]
        ej_delta_t0 = x[1]
        j_theta_inf_deg = x[2]
        ve_delta_t = x[3]
        e_theta_inf_deg = x[4]
        
#         E_epoch = J_orbit.epoch + ej_delta_t0*u.day
#         V_epoch = E_epoch + ve_delta_t*u.day
        E_epoch = V_epoch - ve_delta_t*u.day
        J_epoch = E_epoch - ej_delta_t0*u.day
        S_epoch = J_epoch - js_delta_t*u.day
#         print(V_epoch)
#         print(E_epoch)
#         print(J_epoch)
#         print(S_epoch)

        epoch_dict = {Venus: V_epoch,
                      Earth: E_epoch,
                      Jupiter: J_epoch,
                      Saturn: S_epoch}
        
        theta_inf_dict = {Earth: e_theta_inf_deg, Jupiter: j_theta_inf_deg}
        
        route = Route(epoch_dict, theta_inf_dict)

        V_orbit = Orbit.from_body_ephem(Venus, V_epoch)
        
        v_in_venus = route.departure_orbit(Venus).v - V_orbit.v
        c3_venus = norm(v_in_venus)**2
        
        err_weight = .0001
        c3_weight = 100
        f_term = err_weight * (1-route.xfer_error(Earth)/10000)**2
        g_term = err_weight * (1-route.xfer_error(Venus)/10000)**2
        c3_term = c3_weight * (c3_venus.value)**2
        
#         print("score terms: %10.0f %10.0f %9.0f" % (f_term, g_term, c3_term))
        score = f_term + g_term + c3_term
#         print("trying:", x, score)
        return score
    
    def opt_ej(self):
        ej_bounds = [(-365*6, -365*2), (-600,-300), (90.001,180), (-180,-14), (95,180)]
        x0 = [
            -700,  # JS delta t 
            -350,  # EJ delta t
            120,   # J theta_inf
            -90,   # VE delta t
            120]   # E theta_inf
        
        V_epoch = Time("2041-01-08", scale=TIME_SCALE)
        
        minimizer_kwargs = {
            'tol': 1e-3,
            'bounds': ej_bounds,
    #             'method': 'COBYLA'
        }
        opt_out = opt.basinhopping(self.find_flyby,
                                   x0=x0,
                                   niter=5,
                                   minimizer_kwargs=minimizer_kwargs,
                                   disp=True).x
                              
        (js_delta_t, ej_delta_t, j_angle, ve_delta_t, e_angle) = opt_out 
        
    #         e_epoch = J_orbit.epoch + ej_delta_t*u.day
    #         v_epoch = e_epoch + ve_delta_t*u.day
        
        E_epoch = V_epoch - ve_delta_t*u.day
        J_epoch = E_epoch - ej_delta_t*u.day
        S_epoch = J_epoch - js_delta_t*u.day
        
        return {'s_epoch': S_epoch, 'j_epoch': J_epoch, 'e_epoch': E_epoch,
                'v_epoch': V_epoch, 'j_angle': j_angle, 'e_angle': e_angle,
                'js_delta_t': js_delta_t, 'ej_delta_t': ej_delta_t,
                've_delta_t': ve_delta_t,
                'opt_out': opt_out}
        return (j_angle, E_epoch, e_angle, V_epoch, opt_out)
        
    def trajectory_calculator(self, route, plot_on=False, disp_on=False):
        ref_epoch = Time("2020-01-01", scale=TIME_SCALE)
        
        tmod = [(epoch - ref_epoch).to(u.day).value for epoch in route.values()]
    
        time_v = route[Venus]
        time_e = route[Earth]
        time_j = route[Jupiter]
        time_s = route[Saturn]
    
        nodes = list(route.keys())
        
        lower_bound = 0
        upper_bound = 40*365
        
        options = {'maxiter': 1,'disp': True}
        
        xfer_list = []
        flyby_list = []
        
        index_pairs = zip(range(0,len(route)-1), range(1,len(route)))
        constraint_list = [
            {'type': 'ineq', 'fun':causality_constraint(low_i, high_i)}
            for low_i, high_i in index_pairs]
        
        constraint_list.append({'type': 'ineq', 
                                'fun':flyby_constraint(0, 1.1, flyby_list)})
        constraint_list.append({'type': 'ineq', 
                                'fun':flyby_constraint(1, 4, flyby_list)})
        
        print("first pass")
    
        J_orbit = Orbit.from_body_ephem(Jupiter, time_j)
        
        results = self.opt_ej()
        print("results:", results)
        
        epoch_dict = {Venus: results['v_epoch'],
                      Earth: results['e_epoch'],
                      Jupiter: results['j_epoch'],
                      Saturn: results['s_epoch']}
        
        theta_inf_deg_dict = {Earth: results['e_angle'],
                              Jupiter: results['j_angle']}
        
        route = Route(epoch_dict, theta_inf_deg_dict)
        
        V_orbit = Orbit.from_body_ephem(Venus, results['v_epoch'])
        v_in_venus = route.departure_orbit(Venus).v - V_orbit.v
        c3_venus = norm(v_in_venus)**2
        print("C3 at venus:", c3_venus)
        
    #     js_xfer = xfer_list[2]
    #     ej_orbit = xfer_list[1]
    #     ve_orbit = xfer_list[0]         
    
        js_xfer = route.js_xfer
        ej_xfer = route.ej_xfer
        ve_xfer = route.ve_xfer
    
        xfers = route.xfer_dict
        encounters = route.encounter_dict
        
        ################## Plot ##################  
        if plot_on:
            op = OrbitPlotter2D()
            
            orbit_v = ve_xfer.initial_orbit
            orbit_e = ej_xfer.initial_orbit
            orbit_j = js_xfer.initial_orbit
            orbit_s = js_xfer.target_orbit 
    
            op.plot(orbit_v, label="Venus2 Orbit")
            op.plot(orbit_e, label="Earth2 Orbit")
            op.plot(orbit_j, label="Jupiter Orbit")
            op.plot(orbit_s, label="Saturn Orbit")
    
            op.plot(ve_xfer.orbit, label="V2->E")
            op.plot(ej_xfer.orbit, label="E->J")
            op.plot(js_xfer.orbit, label="J->S")
        
        print(tuple((k,v) for k,v in encounters.items()))
        orbits = tuple(v.parent_orbit for v in encounters.values())
        xfer_orbits = tuple(v.initial_orbit for v in xfers.values())
        delta_vs = tuple(v.delta_v for k,v in encounters.items() 
                   if k in {Earth, Jupiter, Saturn})
        times = tuple(v.epoch for v in encounters.values())
        
        #print("orbits: ", orbits)
        #print("delta_vs: ", delta_vs)
        #print("xfers.values(): ", xfers.values())
        #print("xfer orbits[0]: ", xfer_orbits[0].r_p.to(u.AU), ", ", xfer_orbits[0].r_a.to(u.AU))
        #print(xfer_orbits[0].epoch)
        
    #     ee = encounters[Earth]
    #     print(ee.parent_orbit.v.to(KPS))
    #     print(ee.inbound_v, "->", ee.v_i)
    #     print(ee.outbound_v, "->", ee.v_f)
    #     print(ee.v_err)
        
        print("C3_v:      %3s"    % encounters[Venus].C3)
        print()
        print("v_inf_e:   %3s"    % encounters[Earth].v_inf)
        print("psi_e:     %3.1f°" % encounters[Earth].psi_deg)
        print("r_p/R]_e:  %3.2f"  % encounters[Earth].r_pr)
        print("delta_v_e: %3s"  % encounters[Earth].delta_v)
        print()
        print("v_inf_j:   %3s"    % encounters[Jupiter].v_inf)
        print("psi_j:     %3.1f°" % encounters[Jupiter].psi_deg)
        print("r_p/R]_j:  %3.2f"  % encounters[Jupiter].r_pr)
        print("delta_v_j: %3s"    % encounters[Jupiter].delta_v)
        
        return (sum(delta_vs), orbits, xfer_orbits, delta_vs, times)
    
class Route(object):
    def __init__(self, epoch_dict, theta_inf_dict):
        V_epoch = epoch_dict[Venus] 
        E_epoch = epoch_dict[Earth] 
        J_epoch = epoch_dict[Jupiter] 
        S_epoch = epoch_dict[Saturn] 
        
        ej_delta_t0 = E_epoch - J_epoch
        j_theta_inf_deg = theta_inf_dict[Jupiter]
        ve_delta_t = V_epoch - E_epoch
        e_theta_inf_deg = theta_inf_dict[Earth]
        
        planner = TransferPlanner()
        planner.start_body = Jupiter
        planner.end_body = Saturn
        js_xfer = planner.make_transfer(start_epoch=J_epoch, 
                                        end_epoch=S_epoch)
        
        js_orbit = js_xfer.initial_orbit
        ej_orbit = xfer_into_xfer(js_orbit, 
                                 theta_inf_deg=j_theta_inf_deg, 
                                 target_body=Jupiter,
                                 origin_body=Earth, 
                                 origin_epoch_guess=E_epoch)
        try:
            back_ej_orbit = ej_orbit.propagate(ej_delta_t0, method=kepler, rtol=1e-5)
        except RuntimeError:
            back_ej_orbit = ej_orbit.propagate(ej_delta_t0, method=cowell, rtol=1e-5)

        ve_orbit = xfer_into_xfer(back_ej_orbit, 
                                 theta_inf_deg=e_theta_inf_deg,
                                 target_body=Earth, 
                                 origin_body=Venus, 
                                 origin_epoch_guess=V_epoch)

        try:
            back_ve_orbit = ve_orbit.propagate(ve_delta_t, method=kepler, rtol=1e-5)
        except RuntimeError:
            back_ve_orbit = ve_orbit.propagate(ve_delta_t, method=cowell, rtol=1e-5)
        
        self._js_xfer = js_xfer
        
        self._epochs = {Venus: V_epoch,
                        Earth: E_epoch,
                        Jupiter: J_epoch,
                        Saturn: S_epoch}
        
        self._departure_orbits = {Venus: back_ve_orbit,
                                  Earth: back_ej_orbit,
                                  Jupiter: js_xfer.initial_orbit}
    
    def departure_orbit(self, from_body):
        return self._departure_orbits[from_body]
    
    def body_epoch(self, at_body):
        return self._epochs[at_body]
    
    @property
    def js_orbit(self):
        return self.js_xfer.initial_orbit
    
    @property
    def js_xfer(self):
        return self._js_xfer
    
    @property
    def ej_xfer(self):
        try:
            return self._ej_xfer
        except AttributeError:
            pass  # fall through
        
        planner = TransferPlanner()
        planner.start_body = Earth
        planner.end_body = Jupiter
        self._ej_xfer = planner.make_transfer(
            start_epoch=self.body_epoch(Earth), 
            end_epoch=self.body_epoch(Jupiter))
        return self._ej_xfer
    
    @property
    def ve_xfer(self):
        try:
            return self._ve_xfer
        except AttributeError:
            pass  # fall through
        
        planner = TransferPlanner()
        planner.start_body = Venus
        planner.end_body = Earth
        self._ve_xfer = planner.make_transfer(
            start_epoch=self.body_epoch(Venus), 
            end_epoch=self.body_epoch(Earth))
        return self._ve_xfer
    
    @property
    def xfer_dict(self):
        return {(Venus, Earth):    self.ve_xfer,
                (Earth, Jupiter):  self.ej_xfer,
                (Jupiter, Saturn): self.js_xfer}
        
    @property
    def encounter_dict(self):
        return {Venus:   Boundary.from_transfers(None, self.ve_xfer),
                Earth:   Flyby.from_transfers(self.ve_xfer, self.ej_xfer),
                Jupiter: Flyby.from_transfers(self.ej_xfer, self.js_xfer),
                Saturn:  Boundary.from_transfers(self.js_xfer, None)}
    
    def xfer_error(self, start_body):
        """
        Error in the closest approach to the starting body
        """
        back_prop_xfer = self.departure_orbit(start_body)
        body_orbit = Orbit.from_body_ephem(start_body, 
                                           back_prop_xfer.epoch)
#         print("t_f:", base_orbit.epoch)
#         print("t_0:", back_prop_xfer.epoch)
#         print("delta_r:", (norm(base_orbit.r) - norm(back_prop_xfer.r)).to(u.au))
#         print("r_f:", norm(base_orbit.r).to(u.au))
#         print("r_0:", norm(back_prop_xfer.r).to(u.au))
        r_err = back_prop_xfer.r - body_orbit.r
        return norm(r_err) / u.km