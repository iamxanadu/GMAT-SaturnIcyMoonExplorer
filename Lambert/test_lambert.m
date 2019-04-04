clear all; close all; clc;

%dist in km
r1 = [3.30211*10^7, -1.03692*10^8, -3.30675*10^6];
r2 = [4.38478*10^8, 6.06692*10^8, -1.23639*10^7];
tf = 5*365;
m=1;

GM_central = 1.327*10^11; %km^3/s^2;

[V1, V2, extremal_distances, exitflag] = lambert(r1, r2, tf, m, GM_central)