"""
Created on Apr 22, 2019

@author: david
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from quaternion import Matrix4x4
import numpy as np
import math
from vectors import Vector

FIG_WIDTH = 6.5
FIG_DPI = 127.33567
SAVE_DPI = 300

SAVE_OUT = True

PASTEL_RED = (.867, .467, .467, 1)
PASTEL_GREEN = (.467, .867, .467, 1)
PASTEL_BLUE = (.467, .467, .867, 1)

def fig_size(aspect):
    return (FIG_WIDTH, FIG_WIDTH/aspect)

def size_label(label):
    return "%s (actual size @ %.1f DPI)" % (label, FIG_DPI)

nu = math.radians(60)
theta_d = math.radians(140)
theta_inf = math.radians(145)

v_d = Vector(r=5, theta=theta_d)

v_p = Vector(r=2.5, theta=(nu + math.pi/2))
v_out = v_d-v_p
v_inf = v_out.norm()

rot_axis = Vector.cross(v_out, v_p)
r_ref = v_out.rotate_about(rot_axis, theta_inf)
v_in = -v_out.rotate_about(rot_axis, 2*theta_inf)
v_a = v_in + v_p

v_in_extend = v_in
v_in = v_in.translate(-v_in)
v_a = v_a.translate(-v_a)

v_in_extend.plot_kwargs['linestyle'] = '--'
r_ref.plot_kwargs['linestyle'] = '-.'

v_a.quiver_kwargs['linestyle'] = 'dashed'
v_d.quiver_kwargs['linestyle'] = 'dashed'

v_in.quiver_kwargs['facecolor'] = PASTEL_BLUE
v_a.quiver_kwargs['facecolor'] = PASTEL_BLUE
v_d.quiver_kwargs['facecolor'] = PASTEL_RED
v_out.quiver_kwargs['facecolor'] = PASTEL_RED

X, Y, Z, U, V, W = zip(v_d, v_out, v_p, v_in, v_a)
#print(U)
#print(V)

title = r'Sample Flyby Solution: $\theta_d=%.0f%s, \theta_{\infty}=%.0f%s$' \
        % (math.degrees(theta_d), u'\u00b0', 
           math.degrees(theta_inf), u'\u00b0')
plain_title = 'flyby theta_d %.0f theta_inf %.0f' \
        % (math.degrees(theta_d), math.degrees(theta_inf))

if SAVE_OUT:
    fig_title = plain_title.replace(' ', '_') + '.png'
else:
    fig_title = size_label(plain_title)

fig = plt.figure(fig_title, 
                 frameon=False, 
                 figsize=fig_size(4/3), 
                 dpi=FIG_DPI)
plt.polar(aspect='equal')

#r_ref.plot_polar(as_line=True)
#v_in_extend.plot_polar(as_line=True)
v_out.plot_polar(r'$\vec{v}_{out}$')
v_d.plot_polar(r'$\vec{v}_d$')
v_p.plot_polar(r'$\vec{v}_p$')
v_a.plot_polar(r'$\vec{v}_a$')
v_in.plot_polar(r'$\vec{v}_{in}$')

ax = fig.gca()
Vector.quiver_labels(ax, (1.2, .5), U=100)

# x_min = int(min((*X, *U)) - 1.5)
# x_max = int(max((*X, *U)) + 1.5)
# y_min = int(min((*Y, *V)) - 1.5)
# y_max = int(max((*Y, *V)) + 1.5)
y_max = int(1.5*v_inf + 1.5)
 
#ax.set_aspect('equal')
ax.set_ylim([0, y_max])

left, right = ax.get_xlim()
ax.tick_params(axis='x', which='major', direction='inout', grid_alpha=.25)
ax.set_xticks(np.arange(left, right, step=math.radians(30)))
ax.set_title(title)

ax.annotate(r'$\hat{r}_{ref}$',
            xy=(r_ref.theta, y_max),  # theta, radius
            xytext=(r_ref.theta, 1.1*y_max),    # fraction, fraction
            )

ax.annotate(r'$\theta_d$',
            xy=(theta_d, y_max),  # theta, radius
            xytext=(theta_d, 1.1*y_max),    # fraction, fraction
            )

ax.axvline(v_in.theta, linestyle='--', color=PASTEL_BLUE, linewidth=1)
ax.axvline(v_out.theta, linestyle='--', color=PASTEL_RED, linewidth=1)
#ax.axvline(v_p.theta, linestyle='-.', color='k', linewidth=1)
ax.axvline(r_ref.theta, linestyle='-.', color='k', linewidth=1)

# plt.xticks([v_in.theta, r_ref.theta, v_p.theta, v_out.the
plt.yticks([v_inf], ['$v_\infty$'])

# ax.set_xlim([-math.radians(360), math.radians(360)])
# ax.set_xlim([x_min, x_max])
# ax.set_ylim([y_min, y_max])

if SAVE_OUT:
    plt.savefig(fig_title, dpi=SAVE_DPI, transparent=True)
    
plt.show()