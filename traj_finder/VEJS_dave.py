#!/usr/bin/env python3 -u
# coding: utf-8

# In[1]:
import csv
import datetime
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
from route.route import RouteBuilder
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

def angle_between(vec_a, vec_b):
    return math.acos(vec_a.dot(vec_b)/(norm(vec_a)*norm(vec_b)))

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
    
    trial = {Venus:   Time("2037-11-25 12:46:13.205", scale=TIME_SCALE),
             Earth:   Time("2038-02-10 15:20:30.134", scale=TIME_SCALE),
             Jupiter: Time("2040-12-08 01:11:35.078", scale=TIME_SCALE),
             Saturn:  Time("2046-01-04 20:40:09.445", scale=TIME_SCALE)}
    
    terminate_epoch = Time("2045-05-05", scale=TIME_SCALE)
    
#     base_epoch_body = Venus
#     base_epoch = Time("2041-01-08", scale=TIME_SCALE)
    base_epoch_body = Jupiter
    base_epoch = trial[base_epoch_body]
    route_builder = RouteBuilder(base_epoch_body, base_epoch)
    
    filename = datetime.datetime.now().strftime("optimal-route-%y-%m-%d-%H-%M.csv")
    fieldnames = ('v_epoch', 'e_epoch', 'j_epoch', 's_epoch', 
                       'e_angle', 'j_angle', 'v_inner_cons', 'v_outer_cons',
                       'e_inner_cons', 'e_outer_cons', 've_delta_t',
                       'ej_delta_t', 'js_delta_t')

    with open(filename, mode='w') as result_file:
        writer = csv.DictWriter(result_file, fieldnames=fieldnames)
        writer.writeheader()
        
        print("Jupiter at ", base_epoch)
        x0 = RouteBuilder.route_to_x0(trial)
        solution = route_builder.trajectory_calculator(x0)
        print("results:", solution)
            
        while trial[base_epoch_body] < terminate_epoch:
#         while not solution['opt_out'].success:
            for key in trial:
                trial[key] += 15 * u.day
                
            if trial[base_epoch_body] > terminate_epoch:
                break
            try:
                route_builder = RouteBuilder(base_epoch_body, trial[base_epoch_body])
                print("Jupiter at ", trial[base_epoch_body])
                x0 = RouteBuilder.route_to_x0(trial)
                solution = route_builder.trajectory_calculator(x0)
                print("results:", solution)
                
                row = {k:v for k,v in solution.items() if k in fieldnames}
                writer.writerow(row)
                result_file.flush()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print("exception caught: ", e)
                continue
        
    route = RouteBuilder.solution_to_route(solution)
    
    sol = route.legacy_format()
    
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