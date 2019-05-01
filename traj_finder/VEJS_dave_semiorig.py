#!/usr/bin/env python3
# coding: utf-8

# In[1]:

import numpy as np

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris

from poliastro.bodies import Sun,Venus, Earth, Jupiter, Saturn
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver
from poliastro.iod import izzo
from poliastro.plotting import OrbitPlotter2D
from poliastro.util import norm
import poliastro.twobody.propagation as Propagation

from scipy import optimize as opt
 
solar_system_ephemeris.set("jpl")

KPS = u.km / u.s

# In[2]:


cassini_venus2 = Time("1999-06-24")
cassini_earth2 = Time("1999-08-18")
cassini_jupiter = Time("2000-12-30")
cassini_saturn = Time("2004-07-01")

#cassini transit
cassini_js_time = cassini_saturn-cassini_jupiter
cassini_ej_time = cassini_jupiter-cassini_earth2
cassini_ve_time = cassini_earth2-cassini_venus2


# In[3]:


tmod=[0,0,0,0]
plot_on=0;

def trajectory_calculator(t=[0,0,0,0],plot_on=0,disp_on=False):
    TIME_SCALE="tdb"
    
    #start with cassini style assumptions
    cassini_venus2 = Time("1999-06-24")
    cassini_earth2 = Time("1999-08-18")
    cassini_jupiter = Time("2000-12-30")
    cassini_saturn = Time("2004-07-01")

    #cassini transit
    cassini_js_time = cassini_saturn-cassini_jupiter
    cassini_ej_time = cassini_jupiter-cassini_earth2
    cassini_ve_time = cassini_earth2-cassini_venus2


    tmod_s  = t[0] #arrival time in saturn relative to 2020
    tmod_js = t[1] #transit times
    tmod_ej = t[2] 
    tmod_ve = t[3]

    time_js = tmod_js*u.year  
    time_ej = tmod_ej*u.year
    time_ve = tmod_ve*u.year

    date_arrive_saturn = Time("2020-01-01", scale=TIME_SCALE)+tmod_s*u.year;


    #create target orbit
    o_sf = Orbit.from_body_ephem(Saturn, date_arrive_saturn)
    r_sf, v_sf = o_sf.rv()
    #o_sf.plot()
    if disp_on:
        print('computed saturn')

    ################## Jupiter ################## 

    #guess flyby date of Jupiter
    date_flyby_jupiter = date_arrive_saturn-time_js

    #construct j orbit
    o_j = Orbit.from_body_ephem(Jupiter, date_flyby_jupiter)
    #o_j.plot()

    #compute transfer lambert trajectory
    (v_jo, v_si), = izzo.lambert(Sun.k, o_j.r, o_sf.r, time_js)
    trx_js = Orbit.from_vectors(Sun, o_j.r, v_jo, epoch=date_flyby_jupiter)
    #trx_js.plot()
    if disp_on:
        print('computed J-S')
    ################## Earth ################## 
    #guess flyby date of Earth
    date_flyby_earth = date_arrive_saturn-time_js-time_ej

    #construct j orbit
    o_e2 = Orbit.from_body_ephem(Earth, date_flyby_earth)
    #o_e2.plot()

    #compute transfer lambert trajectory
    (v_eo, v_ji), = izzo.lambert(Sun.k, o_e2.r, o_j.r, time_ej)
    trx_ej = Orbit.from_vectors(Sun, o_e2.r, v_eo, epoch=date_flyby_earth)
    #trx_ej.plot()
    if disp_on:
        print('computed E-J')                            

    ################## Venus2 ################## 
    #guess flyby date of Venus2
    date_flyby_venus2 = date_arrive_saturn-time_js-time_ej-time_ve

    #construct j orbit
    o_v2 = Orbit.from_body_ephem(Venus, date_flyby_venus2)
    #o_v2.plot()

    #compute transfer lambert trajectory
    (v_v2o, v_ei), = izzo.lambert(Sun.k, o_v2.r, o_e2.r, time_ve)
    trx_v2e = Orbit.from_vectors(Sun, o_v2.r, v_v2o, epoch=date_flyby_venus2)
    #trx_v2e.plot()    
    if disp_on:
        print('computed V-E')                            

    ################## Sum delta v ##################                             
    delv_e = norm(v_eo-v_ei)
    delv_j = norm(v_jo-v_ji)
    delv_s = norm(v_si-v_sf)

    total_deltav=sum([delv_e,delv_j,delv_s]) 

    if disp_on:
        print('Total delta-v: ', total_deltav)

    ################## Plot ##################  

    if plot_on:
        op = OrbitPlotter2D()

        op.plot(o_v2,label="Venus2 Orbit")
        op.plot(o_e2,label="Earth2 Orbit")
        op.plot(o_j, label="Jupiter Orbit")
        op.plot(o_sf, label="Saturn Orbit")


        op.plot(trx_v2e, label="V2-E")
        op.plot(trx_ej, label="E-J")
        op.plot(trx_js, label="J-S")
        
    orbits = (o_v2,o_e2,o_j,o_sf)
    trajectories = (trx_v2e,trx_ej,trx_js)
    deltavs = (delv_e,delv_j,delv_s)
    times = (date_flyby_venus2,date_flyby_earth,date_flyby_jupiter,date_arrive_saturn)
    return (total_deltav,orbits,trajectories,deltavs,times)

def print_times(times):
    print("Venus:   ",times[0])
    print("Earth:   ",times[1])
    print("Jupiter: ",times[2])
    print("Saturn:  ",times[3])
    print("**")
    print("V-E: ", (times[1]-times[0]).to(u.year))
    print("E-J: ", (times[2]-times[1]).to(u.year))
    print("E-S: ", (times[3]-times[2]).to(u.year))
        


# In[4]:


# In[5]:
cassini_js_time.to(u.year).value


# In[6]:
t_guess_cassini = [
    -15.5,
    cassini_js_time.to(u.year).value,
    cassini_ej_time.to(u.year).value,
    cassini_ve_time.to(u.year).value]
print("cassini guess: ", t_guess_cassini)


# In[7]:

print("finding cassini trajectory...")
sol=trajectory_calculator(t=t_guess_cassini,plot_on=0,disp_on=False)
print("done")

print()
print("Total deltav: ", sol[0])
print("Delta vs: ",[x.value for x in sol[3]])
print()

print('Times:')
print_times(sol[4])
print()
quit()
# In[10]:
t_guess_cassini = [
    18,
    cassini_js_time.to(u.year).value,
    cassini_ej_time.to(u.year).value,
    cassini_ve_time.to(u.year).value]
# bounds on input vector, so min/max transit times for each leg: Sat Arrive, JS, EJ, VE, 
bounds = ((10,20), (0.01,5), (0.01,3), (0.01,1))
options = {'maxiter': 30,'disp': False}

def total_deltav_calculator(tmod):
    #print(tmod)
    sol=trajectory_calculator(tmod,plot_on=0,disp_on=False)
    #print(sol[0])
    return sol[0].to(KPS).value

opt_sol = opt.minimize(total_deltav_calculator,
                       x0=t_guess_cassini,
                       options=options,
                       bounds=bounds)

# In[13]:
sol=trajectory_calculator(t=opt_sol.x,plot_on=0,disp_on=False)

print("Tmod: ",opt_sol.x)
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
trx_js_half = trx_js.propagate(t_ej/2)


op=OrbitPlotter2D()
op.plot(sol[1][0],label='V')
op.plot(sol[1][1],label='E')
op.plot(sol[1][2],label='J')
op.plot(sol[1][3],label='S')
op.plot(trx_ve_half,label='VE')
op.plot(trx_ej_half,label='EJ')
op.plot(trx_js_half,label='JS')


# In[14]:
norm(sol[1][2].v).to(KPS)


# In[15]:


#tmod = opt_sol.x
tmod = [27.238877,4.798,1.2879,0.1313]
sol=trajectory_calculator(t=tmod,plot_on=0,disp_on=False)

#print("Tmod: ",opt_sol.x)
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

o_v = sol[1][0]  # orbit of venus at venus flyby
o_e = sol[1][1]  # orbit of earth at earth flyby
o_j = sol[1][2]  # orbit of jupiter at jupiter flyby
o_s = sol[1][3]  # orbit of saturn at saturn flyby/arrive
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

# In[26]:
trx_ve_half = trx_ve.propagate(t_ve/2)


# In[27]:
trx_ve_half.plot()

# In[29]:


print_times(sol[4])

# In[39]:
t_guess_cassini = [-15.5,cassini_js_time.to(u.year).value,cassini_ej_time.to(u.year).value,cassini_ve_time.to(u.year).value]

t_guess_online_traj_tool = [20, 7.93, 2.19,0.15]
t_guess_online_traj_tool = [12, 2, 2,0.15]


t_guess = opt_sol.x
t_guess = t_guess_cassini

#t_guess=t_guess_online_traj_tool

bounds = ((-20,-10),(0.01,10),(0.01,3),(0.01,1))

options = {'maxiter': 60,'disp': False}

def calc_max_deltav(v_inf,qratio=10, body=Earth):
    return 2*v_inf*(1+((v_inf)**2*(qratio*body.R)/(body.k)).to(u.m/u.m))**-1

calc_max_deltav(3.5*KPS)

def weighted_deltav_calculator(tmod):
        #print(tmod)
        sol = trajectory_calculator(tmod, plot_on=0, disp_on=False)
        #print(sol[3])
        
        # 10 is a weight
        # 10*V_inf at Saturn + sum of all other deltaVs (including saturn)
        return 10*sol[3][2].to(KPS).value + sum(sol[3]).to(KPS).value

def saturn_deltav_calculator(tmod):
    #print(tmod)
    sol=trajectory_calculator(tmod,plot_on=0,disp_on=False)
    #print(sol[3])
    
    
    
    return 5*sol[3][2].to(KPS).value + sum(sol[3]).to(KPS).value


def earthflybyConstraint(ts):
    
    sol = trajectory_calculator(t=ts,plot_on=0,disp_on=False)
    
    deltavearth = sol[3][0].to(KPS)
    v_earth = sol[1][1].v.to(KPS)
    v_sat_leaving_earth = sol[2][1].v
    v_inf = norm(v_sat_leaving_earth-v_earth).to(KPS)
    max_deltav = calc_max_deltav(v_inf,qratio=2, body=Earth)
    
    return float((max_deltav-deltavearth)/max_deltav)

def jupiter_flyby_constraint(ts):
    body = Jupiter
    q_ratio = 2
        
    sol = trajectory_calculator(t=ts, plot_on=0, disp_on=False)
    
    deltavjup = sol[3][1].to(KPS)
    v_jup = sol[1][2].v.to(KPS)
    v_sat_leaving_jup = sol[2][2].v
    v_inf = norm(v_sat_leaving_jup-v_jup).to(KPS)
    max_deltav = calc_max_deltav(v_inf,qratio=q_ratio, body=body)
    
    return float((max_deltav-deltavjup)/max_deltav)


earth_constraint ={'type':'ineq','fun':earthflybyConstraint}
jupiter_constraint ={'type':'ineq','fun':jupiter_flyby_constraint}

opt_sol = opt.minimize(weighted_deltav_calculator,
                       x0=t_guess,
                       options=options,
                       bounds=bounds,
                       constraints=(earth_constraint,jupiter_constraint))

opt_sol


# In[35]:




# In[ ]:


#sol = trajectory_calculator(t=opt_sol.x,plot_on=0,disp_on=1)
sol = trajectory_calculator(t=t_guess_cassini, plot_on=0, disp_on=False)

# In[ ]:


opt_sol.x


# In[ ]:


calc_max_deltav(9*KPS)


# In[ ]:


sol = trajectory_calculator(t=opt_sol.x, plot_on=0, disp_on=False)

print(opt_sol)

#sol = trajectory_calculator([ 21.82450945,   9.33104326,   2.39418937,   0.2])

#print("Tmod: ",opt_sol.x)
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

print()
print("Total deltav: ", sol[0])
print("Delta vs: ",[x.value for x in sol[3]])
print()
print("Earth FB constraint:   ", earthflybyConstraint(t_guess_cassini))
print("Jupiter FB constraint: ", jupiter_flyby_constraint(t_guess_cassini))
