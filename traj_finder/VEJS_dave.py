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
 
solar_system_ephemeris.set("jpl")

KPS = u.km / u.s
TIME_SCALE="tdb"
dt = Time("2020-01-01", scale=TIME_SCALE) - Time("2021-01-01", scale=TIME_SCALE)
assert Time("2020-01-01", scale=TIME_SCALE) - Time("2021-01-01", scale=TIME_SCALE) < 0

#test = Orbit.from_body_ephem(Jupiter)
#test.propagate(-dt)

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
        
def angle_between(vec_a, vec_b):
    return math.acos(vec_a.dot(vec_b)/(norm(vec_a)*norm(vec_b)))

def xfer_err(base_xfer, start_body):
    """
    Error in the closest approach to the starting body
    """
    def _inner(delta_t=None, back_prop_xfer=None):
#         print("delta_t:", delta_t*u.day)
        if back_prop_xfer is None:
            assert delta_t is not None
            back_prop_xfer = base_xfer.propagate(delta_t*u.day, method=kepler)
        start_orbit = Orbit.from_body_ephem(start_body, 
                                            back_prop_xfer.epoch)
#         print("t_f:", base_xfer.epoch)
#         print("t_0:", back_prop_xfer.epoch)
#         print("delta_r:", (norm(base_xfer.r) - norm(back_prop_xfer.r)).to(u.au))
#         print("r_f:", norm(base_xfer.r).to(u.au))
#         print("r_0:", norm(back_prop_xfer.r).to(u.au))
        r_err = back_prop_xfer.r - start_orbit.r
        return norm(r_err) / u.km
    return _inner

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

def trajectory_calculator(route, plot_on=False, disp_on=False):
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
    
    def find_flyby(x, V_epoch):
        S_epoch_base = Time("2048-01-08", scale=TIME_SCALE)
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
        planner = TransferPlanner()
        planner.start_body = Jupiter
        planner.end_body = Saturn
        js_xfer = planner.make_transfer(start_epoch=J_epoch, 
                                        end_epoch=S_epoch)
        js_orbit = js_xfer.initial_orbit
#         print("js:", js_orbit)
        ej_orbit = xfer_into_xfer(js_orbit, 
                                 theta_inf_deg=j_theta_inf_deg, 
                                 target_body=Jupiter,
                                 origin_body=Earth, 
                                 origin_epoch_guess=E_epoch)
        try:
            back_ej_orbit = ej_orbit.propagate(ej_delta_t0*u.day, method=kepler, rtol=1e-5)
        except RuntimeError:
            back_ej_orbit = ej_orbit.propagate(ej_delta_t0*u.day, method=cowell, rtol=1e-5)
#         print("back_ej:", back_ej_orbit)
        ve_orbit = xfer_into_xfer(back_ej_orbit, 
                                 theta_inf_deg=e_theta_inf_deg,
                                 target_body=Earth, 
                                 origin_body=Venus, 
                                 origin_epoch_guess=V_epoch)
#         print("propagate:", ve_delta_t*u.day)
        kwargs = {'numiter':100000}
        try:
            back_ve_orbit = ve_orbit.propagate(ve_delta_t*u.day, method=kepler, rtol=1e-5)
        except RuntimeError:
            back_ve_orbit = ve_orbit.propagate(ve_delta_t*u.day, method=cowell, rtol=1e-5)
            #method=cowell
#         print("back_ve:", back_ve_orbit)

        f = xfer_err(ej_orbit, Earth)
        g = xfer_err(ve_orbit, Venus)
        
        V_orbit = Orbit.from_body_ephem(Venus, V_epoch)
        
        v_in_venus = back_ve_orbit.v - V_orbit.v
        c3_venus = norm(v_in_venus)**2
        
        err_weight = .0001
        c3_weight = 100
        f_term = err_weight * (1-f(back_prop_xfer=back_ej_orbit)/10000)**2
        g_term = err_weight * (1-g(back_prop_xfer=back_ve_orbit)/10000)**2
        c3_term = c3_weight * (c3_venus.value)**2
        
        print("score terms: %10.0f %10.0f %9.0f" % (f_term, g_term, c3_term))
        score = f_term + g_term + c3_term
#         print("trying:", x, score)
        return score

    J_orbit = Orbit.from_body_ephem(Jupiter, time_j)
    
    def opt_ej():
        ej_bounds = [(-365*6, -365*1), (-600,-300), (90.001,180), (-180,-14), (95,180)]
        x0 = [
            -700,  # JS delta t 
            -350,  # EJ delta t
            120,   # J theta_inf
            -90,   # VE delta t
            120]   # E theta_inf
        
        V_epoch = Time("2041-01-08", scale=TIME_SCALE)
        
        minimizer_kwargs = {
            'args': (V_epoch),
            'tol': 1e-3,
            'bounds': ej_bounds,
    #             'method': 'COBYLA'
        }
        opt_out = opt.basinhopping(find_flyby,
                                   x0=x0,
                                   niter=100,
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

    
    results = opt_ej()
    print("results:", results)
    
    planner = TransferPlanner()
    
    planner.start_body = Jupiter
    planner.end_body = Saturn
    start_epoch = results['j_epoch']
    end_epoch = results['s_epoch']
    
    js_xfer = planner.make_transfer(start_epoch=start_epoch, 
                                    end_epoch=end_epoch)
    
    js_orbit = js_xfer.initial_orbit
    ej_orbit = xfer_into_xfer(js_orbit, 
                                 theta_inf_deg=results['j_angle'], 
                                 target_body=Jupiter,
                                 origin_body=Earth, 
                                 origin_epoch_guess=results['e_epoch'])
    
    back_ej_orbit = ej_orbit.propagate(results['ej_delta_t']*u.day, 
                                       method=kepler)
    ve_orbit = xfer_into_xfer(back_ej_orbit, 
                                 theta_inf_deg=results['e_angle'], 
                                 target_body=Earth,
                                 origin_body=Venus, 
                                 origin_epoch_guess=results['v_epoch'])
    
    back_ve_orbit = ve_orbit.propagate(results['ve_delta_t']*u.day, 
                                       method=kepler)
    
    V_orbit = Orbit.from_body_ephem(Venus, results['v_epoch'])
    v_in_venus = back_ve_orbit.v - V_orbit.v
    c3_venus = norm(v_in_venus)**2
    print("C3 at venus:", c3_venus)
    
#     js_xfer = xfer_list[2]
#     ej_orbit = xfer_list[1]
#     ve_orbit = xfer_list[0]         

    planner.start_body = Earth
    planner.end_body = Jupiter
    ej_xfer = planner.make_transfer(start_epoch=results['e_epoch'], 
                                    end_epoch=results['j_epoch'])
    
    planner.start_body = Venus
    planner.end_body = Earth
    ve_xfer = planner.make_transfer(start_epoch=results['v_epoch'], 
                                    end_epoch=results['e_epoch'])

    xfers = {(Venus, Earth):    ve_xfer,
             (Earth, Jupiter):  ej_xfer,
             (Jupiter, Saturn): js_xfer}
    
    encounters = {Venus:   Boundary.from_transfers(None, ve_xfer),
                  Earth:   Flyby.from_transfers(ve_xfer, ej_xfer),
                  Jupiter: Flyby.from_transfers(ej_xfer, js_xfer),
                  Saturn:  Boundary.from_transfers(js_xfer, None)}
    
    ################## Plot ##################  
    if plot_on:
        op = OrbitPlotter2D()
        
        orbit_v = ve_orbit.initial_orbit
        orbit_e = ej_orbit.initial_orbit
        orbit_j = js_xfer.initial_orbit
        orbit_s = js_xfer.target_orbit 

        op.plot(orbit_v, label="Venus2 Orbit")
        op.plot(orbit_e, label="Earth2 Orbit")
        op.plot(orbit_j, label="Jupiter Orbit")
        op.plot(orbit_s, label="Saturn Orbit")

        op.plot(ve_orbit.orbit, label="V2->E")
        op.plot(ej_orbit.orbit, label="E->J")
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

def make_transfer(route, ref_epoch, xfer_list, flyby_list):
    planner = TransferPlanner()
    
    expected_delta_t = (7*u.year + 30*u.day).to(u.day).value;
    
    def err_calc(epoch_list):
        nonlocal xfer_list, flyby_list
        xfer_list.clear()
        flyby_list.clear()
        
        start_rev_range = range(len(route)-2, -1, -1) 
        end_rev_range = range(len(route)-1, 0, -1) 
        zipped = zip(start_rev_range, end_rev_range)
        
        for start, end in zipped:
            assert start == end-1
            planner.start_body = route[start]
            planner.end_body = route[end]
            
            start_epoch = ref_epoch + epoch_list[start]*u.day
            end_epoch = ref_epoch + epoch_list[end]*u.day
            
            if end_epoch < start_epoch:
                return float('Inf')
            
            xfer_list.append(planner.make_transfer(
                start_epoch=start_epoch,
                end_epoch=end_epoch))
            
        xfer_list.reverse()
        
        zipped = list(zip(xfer_list[:-1], xfer_list[1:]))
        for inbound, outbound in zipped:
            flyby_list.append(Flyby.from_transfers(inbound, outbound))
            
        terminus = Boundary.from_transfers(xfer_list[-1], None)
        
        t_total = max(epoch_list) - min(epoch_list)
        t_penalty = max(0, t_total - expected_delta_t)**2
        
        summation = sum([flyby.v_err**2 for flyby in flyby_list])
        
        #print("epoch: ", epoch_list)
        print("summation: ", summation)
        print("delta_v: ", terminus.delta_v**2)
        print("t penalty: ", t_penalty)
        
        return (100*summation + terminus.delta_v**2).value + t_penalty
    
    return err_calc

def print_times(times):
    print("Venus:   ",times[0])
    print("Earth:   ",times[1])
    print("Jupiter: ",times[2])
    print("Saturn:  ",times[3])
    print("**")
    print("V-E: ", (times[1]-times[0]).to(u.year))
    print("E-J: ", (times[2]-times[1]).to(u.year))
    print("J-S: ", (times[3]-times[2]).to(u.year))

def calc_max_deltav(v_inf,qratio=10, body=Earth):
    return 2*v_inf*(1+((v_inf)**2*(qratio*body.R)/(body.k)).to(u.m/u.m))**-1

if __name__ == "__main__":
    
    # earth perifocal = 1100 km + R_e
    # r_p / R = 1.17
    # delta v = 5.5 km/s
    cassini_route = {Venus:   Time("1999-06-24", scale=TIME_SCALE),
                     Earth:   Time("1999-08-18", scale=TIME_SCALE),
                     Jupiter: Time("2000-12-30", scale=TIME_SCALE),
                     Saturn:  Time("2004-07-01", scale=TIME_SCALE)}
        
    cassini_venus2 = Time("1999-06-24", scale=TIME_SCALE)
    cassini_earth2 = Time("1999-08-18", scale=TIME_SCALE)
    cassini_jupiter = Time("2000-12-30", scale=TIME_SCALE)
    cassini_saturn = Time("2004-07-01", scale=TIME_SCALE)
    
    t_guess_cassini = [
        cassini_venus2,
        cassini_earth2,
        cassini_jupiter,
        cassini_saturn]
    
    trial = {Venus:   Time("2027-11-24 12:46:13.205", scale=TIME_SCALE),
             Earth:   Time("2028-02-14 15:20:30.134", scale=TIME_SCALE),
             Jupiter: Time("2030-08-03 01:11:35.078", scale=TIME_SCALE),
             Saturn:  Time("2035-09-05 20:40:09.445", scale=TIME_SCALE)}
    
    sol = trajectory_calculator(cassini_route)

    print()
    print("Total deltav: ", sol[0])
    print("Delta vs: ",[x.value for x in sol[3]])
    print()
    
    print('Times:')
    print_times(sol[4])
    print()
    t_ve=sol[4][1]-sol[4][0]
    t_ej=sol[4][2]-sol[4][1]
    t_js=sol[4][3]-sol[4][2]
    
    trx_ve = sol[2][0]
    trx_ej = sol[2][1]
    trx_js = sol[2][2]
    
    trx_ve_half = trx_ve.propagate(t_ve/2)
    trx_ej_half = trx_ej.propagate(t_ej/2)
    trx_js_half = trx_js.propagate(t_js/2)
    
    op=OrbitPlotter2D()
    op.plot(sol[1][0],label='V')
    op.plot(sol[1][1],label='E')
    op.plot(sol[1][2],label='J')
    op.plot(sol[1][3],label='S')
    op.plot(trx_ve_half,label='VE')
    op.plot(trx_ej_half,label='EJ')
    op.plot(trx_js_half,label='JS')
    
    op.show()
    '''
    epoch = Time("2018-08-17 12:05:50", scale="tdb")

    plotter = StaticOrbitPlotter()
    plotter.plot(Orbit.from_body_ephem(Earth, epoch), label="Earth")
    plotter.plot(Orbit.from_body_ephem(Jupiter, epoch), label="Jupiter");
    '''
    
    '''
    print()
    print("Total deltav: ", sol[0])
    print("Delta vs: ",[x.value for x in sol[3]])
    print()
    #print("Earth FB constraint:   ", earthflybyConstraint(t_guess_cassini))
    #print("Jupiter FB constraint: ", jupiter_flyby_constraint(t_guess_cassini))
    '''