#!/usr/bin/env python3
# coding: utf-8

# In[1]:

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris

from poliastro.bodies import Sun, Venus, Earth, Jupiter, Saturn
from poliastro.plotting import OrbitPlotter2D

from scipy import optimize as opt
from node.transfer import TransferPlanner
from node.flyby import Boundary, Flyby
 
solar_system_ephemeris.set("jpl")

KPS = u.km / u.s
TIME_SCALE="tdb"

assert Time("2020-01-01", scale=TIME_SCALE) - Time("2021-01-01", scale=TIME_SCALE) < 0

def causality_constraint(before, after, delta=30):
    def _cmp(args):
        return args[after] - args[before] - delta
    return _cmp

def trajectory_calculator(t=None,plot_on=0,disp_on=False):
    ref_epoch = Time("2020-01-01", scale=TIME_SCALE)
    
    if t is None:
        t = [0, 0, 0, ref_epoch]
    
    tmod = [(epoch - ref_epoch).to(u.day).value for epoch in t]

    time_v = t[0]  
    time_e = t[1]
    time_j = t[2]
    time_s = t[3]

    route = [Venus, Earth, Jupiter, Saturn]
    
    lower_bound = 0
    upper_bound = 40*365
    
    bounds = [(-7496.0,-90), (-7466.0,-60), (-7436.0,-30), (-7406.0,0)]
    options = {'maxiter': 100,'disp': True}
    
    xfer_list = []
    flyby_list = []
    
    index_pairs = zip(range(0,len(route)-1), range(1,len(route)))
    constraint_list = [
        {'type': 'ineq', 'fun':causality_constraint(low_i, high_i)}
        for low_i, high_i in index_pairs]
    
    def make_transfer(route, ref_epoch, xfer_list, flyby_list):
        print("route: ", route)
        planner = TransferPlanner()
        
        expected_delta_t = (5*u.year + 30*u.day).to(u.day).value;
        
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
            
            print("summation: ", summation)
            print("delta_v: ", terminus.delta_v**2)
            print("t penalty: ", t_penalty)
            
            return (10*summation + terminus.delta_v**2).value + t_penalty
        
        return err_calc
    
    obj_func = make_transfer(route, ref_epoch, xfer_list, flyby_list)
    opt_sol = opt.minimize(obj_func,
                           x0=tmod,
                           options=options,
                           bounds=bounds,
                           constraints=constraint_list)
    print("opt_sol: ", opt_sol)
    
    js_xfer = xfer_list[2]
    ej_xfer = xfer_list[1]
    ve_xfer = xfer_list[0]              

    xfers = {(Venus, Earth):    ve_xfer,
             (Earth, Jupiter):  ej_xfer,
             (Jupiter, Saturn): js_xfer}
    
    encounters = {Venus:   Boundary.from_transfers(None, ve_xfer),
                  Earth:   flyby_list[0],
                  Jupiter: flyby_list[1],
                  Saturn:  Boundary.from_transfers(js_xfer, None)}
    
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
    
    # In[2]:
    cassini_venus2 = Time("1999-06-24", scale=TIME_SCALE)
    
    # earth perifocal = 1100 km + R_e
    # r_p / R = 1.17
    # delta v = 5.5 km/s
    cassini_earth2 = Time("1999-08-18", scale=TIME_SCALE)
    cassini_jupiter = Time("2000-12-30", scale=TIME_SCALE)
    cassini_saturn = Time("2004-07-01", scale=TIME_SCALE)
    
    #cassini transit
    cassini_js_time = cassini_saturn-cassini_jupiter
    cassini_ej_time = cassini_jupiter-cassini_earth2
    cassini_ve_time = cassini_earth2-cassini_venus2
    
    t_guess_cassini = [
        cassini_saturn,
        cassini_js_time,
        cassini_ej_time,
        cassini_ve_time]
    
    t_guess_cassini2 = [
        cassini_venus2,
        cassini_earth2,
        cassini_jupiter,
        cassini_saturn]
    
    trial_venus2 =  Time("2035-05-23 12:58:08.285", scale=TIME_SCALE)
    trial_earth2 =  Time("2036-07-01 13:48:36.679", scale=TIME_SCALE)
    trial_jupiter = Time("2039-02-26 12:23:01.441", scale=TIME_SCALE)
    trial_saturn =  Time("2046-02-26 06:23:01.441", scale=TIME_SCALE)
    
    trial = [trial_venus2,
             trial_earth2,
             trial_jupiter,
             trial_saturn]
    
    #sol = trajectory_calculator(t_guess_cassini)
    sol = trajectory_calculator(t_guess_cassini2)

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