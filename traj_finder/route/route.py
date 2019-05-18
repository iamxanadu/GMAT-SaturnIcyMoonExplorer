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
from enum import Enum
from poliastro.plotting.core import OrbitPlotter3D

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

class RouteBuilder(object):
    
    Arg = Enum('Arg', 'JS_DELTA_T EJ_DELTA_T J_THETA_INF' 
               + ' VE_DELTA_T E_THETA_INF')
    
    bodies = (Venus, Earth, Jupiter, Saturn)
    
    def __init__(self, base_epoch_body, base_epoch):
        self.base_epoch_body = base_epoch_body
        self.base_epoch = base_epoch
        self._epochs_from_x_func = RouteBuilder._gen_epochs_from_x_func(
            base_epoch_body, base_epoch)
        self._last_route = None
    
    @staticmethod
    def _gen_epochs_from_x_func(base_epoch_body, base_epoch):
        # x[0] = JS delta t
        # x[1] = EJ delta t
        # x[3] = VE delta t
        if base_epoch_body is Venus:
            def _inner(x):
                return {Venus:   base_epoch,
                        Earth:   base_epoch - x[3],                # v_epoch - x[3]
                        Jupiter: base_epoch - x[3] - x[1],         # e_epoch - x[1]
                        Saturn:  base_epoch - x[3] - x[1] - x[0]}  # j_epoch - x[0]
            
        elif base_epoch_body is Earth:
            def _inner(x):
                return {Earth:   base_epoch,
                        Venus:   base_epoch + x[3],         # e_epoch + x[3]
                        Jupiter: base_epoch - x[1],         # e_epoch - x[1]
                        Saturn:  base_epoch - x[1] - x[0]}  # j_epoch - x[0]
            
        elif base_epoch_body is Jupiter:
            def _inner(x):
                return {Jupiter: base_epoch,
                        Earth:   base_epoch + x[1],         # j_epoch + x[1]
                        Venus:   base_epoch + x[1] + x[3],  # e_epoch + x[3]
                        Saturn:  base_epoch - x[0]}         # j_epoch - x[0]
            
        elif base_epoch_body is Saturn:
            def _inner(x):
                return {Saturn:  base_epoch,
                        Jupiter: base_epoch + x[0],                # s_epoch + x[0]
                        Earth:   base_epoch + x[0] + x[1],         # j_epoch + x[1]
                        Venus:   base_epoch + x[0] + x[1] + x[3]}  # e_epoch + x[3]
        
        else:
            assert False, "unknown base epoch body: %s" % base_epoch_body
        
        return _inner
    
    def epoch_from_x(self, x):
        return self._epochs_from_x_func(x)
    
    def make_route(self, x):
        #         print("trying:", x)
        j_theta_inf_deg = x[2]
        e_theta_inf_deg = x[4]
        epoch_dict = self.epoch_from_x(x)
        
        theta_inf_dict = {Earth: e_theta_inf_deg, Jupiter: j_theta_inf_deg}
        
        route = Route(epoch_dict, theta_inf_dict)
        return route
        
    def find_flyby(self, x):
        route = self.make_route(x)
        
        V_orbit = route.body_orbit(Venus)
        
        v_in_venus = route.departure_orbit(Venus).v - V_orbit.v
        c3_venus = norm(v_in_venus)**2
        
#         err_weight = .0001
#         c3_weight = 100
#         f_term = err_weight * (1-route.xfer_error(Earth)/10000)**2
#         g_term = err_weight * (1-route.xfer_error(Venus)/10000)**2
#         c3_term = c3_weight * (c3_venus.value)**2
        c3_term = c3_venus.value
        
#         print("score terms: %10.0f %10.0f %9.0f" % (f_term, g_term, c3_term))
#         score = f_term + g_term + c3_term
        score = c3_term
#         print("score: %9.4f" % c3_term)
#         print("trying:", x, score)
        
        self._last_route = route
        return score
    
    def window_inner_contraint(self, start_body):
        radius = start_body.R
        def _inner(x):
            if self._last_route is None:
                self.find_flyby(x)
            
            # must be greater than 0
#             route = self.make_route(x)
#             return route.xfer_error(start_body)*u.km - start_body.R
            err = self._last_route.xfer_error(start_body)
            return err/radius - 1
        
        return _inner
    
    def window_outer_contraint(self, start_body):
        radius = start_body.R
        def _inner(x):
            # must be greater than 0
#             route = self.make_route(x)
#             return 100*start_body.R - route.xfer_error(start_body)*u.km
            err = self._last_route.xfer_error(start_body)
            
            return 1000 - err/radius
        
        return _inner
    
    def optimize(self, x0, bounds, minimizer_kwargs, basinhop=False):
        
        cons = ({'type': 'ineq', 'fun': self.window_inner_contraint(Venus)},
                {'type': 'ineq', 'fun': self.window_outer_contraint(Venus)},
                {'type': 'ineq', 'fun': self.window_inner_contraint(Earth)},
                {'type': 'ineq', 'fun': self.window_outer_contraint(Earth)})
        
        # Convert bounds into constraints if the COBYLA method is used, since
        # it can't use bounds
        if minimizer_kwargs.get('method') == 'COBYLA':
            
            # Sorry about the bat-shit insane lambdas
            # X[i] - MIN[i] > 0
            lower_bound = lambda i: {'type': 'ineq', 
                                     'fun': lambda x:  x[i] - bounds[i][0]}
            # MAX[i] - X[i] > 0
            upper_bound = lambda i: {'type': 'ineq', 
                                     'fun': lambda x:  bounds[i][1] - x[i]}
            
            cons += tuple(map(lower_bound, range(len(bounds))))
            cons += tuple(map(upper_bound, range(len(bounds))))
        
        else:
            minimizer_kwargs['bounds'] = bounds
            
        minimizer_kwargs['constraints'] = cons
        
        if basinhop:
            opt_out = opt.basinhopping(self.find_flyby,
                                       x0=x0,
                                       niter=30,
                                       minimizer_kwargs=minimizer_kwargs,
                                       disp=True)
        else:                                   
            opt_out = opt.minimize(self.find_flyby,
                                   x0=x0,
                                   **minimizer_kwargs,
                                   )
                              
        (js_delta_t, ej_delta_t, j_angle, ve_delta_t, e_angle) = opt_out.x
        
        epochs = self._last_route._epochs
        V_epoch = epochs[Venus]
        E_epoch = epochs[Earth]
        J_epoch = epochs[Jupiter]
        S_epoch = epochs[Saturn]
        
        return {'s_epoch': S_epoch, 'j_epoch': J_epoch, 'e_epoch': E_epoch,
                'v_epoch': V_epoch, 'j_angle': j_angle, 'e_angle': e_angle,
                'js_delta_t': js_delta_t, 'ej_delta_t': ej_delta_t,
                've_delta_t': ve_delta_t,
                'opt_out': opt_out,
                'v_inner_cons': self.window_inner_contraint(Venus)(opt_out.x).value,
                'v_outer_cons': self.window_outer_contraint(Venus)(opt_out.x).value,
                'e_inner_cons': self.window_inner_contraint(Earth)(opt_out.x).value,
                'e_outer_cons': self.window_outer_contraint(Earth)(opt_out.x).value}
        return (j_angle, E_epoch, e_angle, V_epoch, opt_out)
    
    @classmethod
    def route_to_x0(cls, route):
        """
        Convert from a dictionary of Body-epoch pairs to the state vector format
        used by `trajectory_calculator`.
        
        Parameters
        ----------
        route : dict(:Body: -> :Time:)
            dictionary mapping bodies in the route to the corresponding flyby
            epoch.
            
        Returns
        -------
        list(float)
            TODO
            [JS_DELTA_T, EJ_DELTA_T, J_THETA_INF, VE_DELTA_T, E_THETA_INF]
        
        """
        temp_dict = {
            cls.Arg.JS_DELTA_T:  route[Jupiter] - route[Saturn],
            cls.Arg.EJ_DELTA_T:  route[Earth] - route[Jupiter],
            cls.Arg.J_THETA_INF: route.get(cls.Arg.J_THETA_INF, 120),
            cls.Arg.VE_DELTA_T:  route[Venus] - route[Earth],
            cls.Arg.E_THETA_INF: route.get(cls.Arg.E_THETA_INF, 120)}
        
        # Convert to days and drop the units
        for k in (cls.Arg.JS_DELTA_T, cls.Arg.EJ_DELTA_T, cls.Arg.VE_DELTA_T):
            temp_dict[k] = temp_dict[k].to(u.day).value
        
        # Ensure the order is consistent, since dict doesn't guarantee it
        return [temp_dict[k] for k in cls.Arg]
    
    @classmethod
    def solution_to_route(cls, results):
        """
        Convert the output given by `trajectory_calculator` into a new `Route`
        object.
         
        """
        epoch_dict = {Venus: results['v_epoch'],
                      Earth: results['e_epoch'],
                      Jupiter: results['j_epoch'],
                      Saturn: results['s_epoch']}
        
        theta_inf_deg_dict = {Earth: results['e_angle'],
                              Jupiter: results['j_angle']}
        
        return Route(epoch_dict, theta_inf_deg_dict)
        
    def trajectory_calculator(self, x0, disp_on=False):
        """
        Attempt to find a trajectory near the given initial epochs in `x0`. No
        guarantee is made that the final solution will be near these epochs,
        except for the epoch corresponding to base_epoch_body in `__init__`.
        
        Parameters
        ----------
        x0 : list(float) 
            Initial solution guess. List elements must be in the order: 
            [JS_DELTA_T, EJ_DELTA_T, J_THETA_INF, VE_DELTA_T, E_THETA_INF]. The 
            variable `Route.bodies` can help cross-check the for the correct 
            ordering. JS, EJ, and VE refer to Jupiter-Saturn, Earth-Jupiter, and
            Venus-Earth, respectively, and each of these three elements is in
            units of days. The two THETA_INF variables are measured in 
            **degrees**.
        disp_on : bool, optional)
            True to display status updates of optimization convergence.
    
        Returns
        -------
        bool
            TODO
        
        """
        print("first pass")
        
        bounds_dict = {
            self.Arg.JS_DELTA_T:  (-365*6, -365*2),
            self.Arg.EJ_DELTA_T:  (-600, -300),
            self.Arg.J_THETA_INF: (90.001, 180),
            self.Arg.VE_DELTA_T:  (-180, -14),
            self.Arg.E_THETA_INF: (95, 180)}


        bounds = [bounds_dict[k] for k in self.Arg]
#         x0 = [
#             -700,  # JS delta t 
#             -350,  # EJ delta t
#             120,   # J theta_inf
#             -90,   # VE delta t
#             130]   # E theta_inf
        
        options = {
#             'tol': 1,
            'ftol': 1e0,
#             'catol': 10000,
            'disp': disp_on,
            'maxiter': 30,
        }
        
        minimizer_kwargs = {
            'tol': 1e0,
#             'method': 'COBYLA',
            'options': options,
        }
        results = self.optimize(x0, bounds, minimizer_kwargs, basinhop=False)
        
#         if results['opt_out'].success:
        return results
#         else:
#             print("results['opt_out']: ", results['opt_out'])
#             quit(0)
    
class Route(object):
    bodies = RouteBuilder.bodies
    
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
        
        self._body_orbits = {}
    
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
    def orbits(self):
        """
        Tuple of each body's orbit at the corresponding flyby epoch 
        """
        return tuple(self.body_orbit(k) for k in self.bodies)
    
    @property
    def epochs(self):
        """
        A tuple of epochs of the flybys at each body
        """
        return tuple(self.body_epoch(k) for k in self.bodies)
    
    @property
    def xfer_orbits(self):
        """ 
        Tuple of all transfer orbits from each body to the next.
        
        Provided for compatability with older top-level code
        """
        return tuple(v.initial_orbit for v in self.xfer_dict.values())
        
    @property
    def encounter_dict(self):
        return {Venus:   Boundary.from_transfers(None, self.ve_xfer),
                Earth:   Flyby.from_transfers(self.ve_xfer, self.ej_xfer),
                Jupiter: Flyby.from_transfers(self.ej_xfer, self.js_xfer),
                Saturn:  Boundary.from_transfers(self.js_xfer, None)}
        
    @property
    def delta_vs(self):
        return tuple(v.delta_v for k,v in self.encounter_dict.items() 
                     if k in {Earth, Jupiter, Saturn})
    
    def legacy_format(self):
        """
        Condense the info for this route down to the list format used by Dev's
        original code.
        """
        return (sum(self.delta_vs), 
                self.orbits, 
                self.xfer_orbits, 
                self.delta_vs, 
                self.times)
        
    def body_orbit(self, body):
        try:
            return self._body_orbits[body]
        except KeyError:
            assert body in {Venus, Earth, Jupiter, Saturn}
            # fall through
            
        self._body_orbits[body] = Orbit.from_body_ephem(body, 
                                                        self.body_epoch(body))
        return self._body_orbits[body]
    
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
        return norm(r_err)
    
    def printout(self, extra=False):
        V_orbit = self.body_orbit(Venus)
        v_in_venus = self.departure_orbit(Venus).v - V_orbit.v
        c3_venus = norm(v_in_venus)**2
        print("C3 at venus:", c3_venus)
    
        xfers = self.xfer_dict
        encounters = self.encounter_dict
        
        print(tuple((k,v) for k,v in encounters.items()))

        if extra:
            orbits = tuple(v.parent_orbit for v in encounters.values())
            xfer_orbits = tuple(v.initial_orbit for v in xfers.values())
            delta_vs = tuple(v.delta_v for k,v in encounters.items() 
                       if k in {Earth, Jupiter, Saturn})
        
            print("orbits: ", orbits)
            print("delta_vs: ", delta_vs)
            print("xfers.values(): ", xfers.values())
            print("xfer orbits[0]: ", xfer_orbits[0].r_p.to(u.AU), ", ", 
                  xfer_orbits[0].r_a.to(u.AU))
            print(xfer_orbits[0].epoch)
            
            ee = encounters[Earth]
            print(ee.parent_orbit.v.to(u.km/u.s))
            print(ee.inbound_v, "->", ee.v_i)
            print(ee.outbound_v, "->", ee.v_f)
            print(ee.v_err)
    
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

    def plot(self, use_3d=False):
        if use_3d:
            op = OrbitPlotter3D()
        else:
            op = OrbitPlotter2D()
            
        orbit_v = self.ve_xfer.initial_orbit
        orbit_e = self.ej_xfer.initial_orbit
        orbit_j = self.js_xfer.initial_orbit
        orbit_s = self.js_xfer.target_orbit 

        op.plot(orbit_v, label="Venus2 Orbit")
        op.plot(orbit_e, label="Earth2 Orbit")
        op.plot(orbit_j, label="Jupiter Orbit")
        op.plot(orbit_s, label="Saturn Orbit")

        op.plot(self.ve_xfer.orbit, label="V2->E")
        op.plot(self.ej_xfer.orbit, label="E->J")
        op.plot(self.js_xfer.orbit, label="J->S")