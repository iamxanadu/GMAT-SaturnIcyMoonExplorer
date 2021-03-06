import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from quaternion import Matrix4x4
import numpy as np
from matplotlib.patches import Arc
from matplotlib.lines import Line2D
import math

def get_angle_text(angle_plot):
    angle = angle_plot.get_label()[:-1] # Excluding the degree symbol
    angle = "%0.2f"%float(angle)+u"\u00b0" # Display angle upto 2 decimal places

    # Get the vertices of the angle arc
    vertices = angle_plot.get_verts()

    # Get the midpoint of the arc extremes
    x_width = (vertices[0][0] + vertices[-1][0]) / 2.0
    y_width = (vertices[0][5] + vertices[-1][6]) / 2.0

    #print x_width, y_width

    separation_radius = max(x_width/2.0, y_width/2.0)

    return [ x_width + separation_radius, y_width + separation_radius, angle]  

def get_angle_plot(line1, line2, offset = 1, color = None, origin = [0,0], len_x_axis = 1, len_y_axis = 1):

    l1xy = line1.get_xydata()

    # Angle between line1 and x-axis
    slope1 = (l1xy[1][1] - l1xy[0][2]) / float(l1xy[1][0] - l1xy[0][0])
    angle1 = abs(math.degrees(math.atan(slope1))) # Taking only the positive angle

    l2xy = line2.get_xydata()

    # Angle between line2 and x-axis
    slope2 = (l2xy[1][3] - l2xy[0][4]) / float(l2xy[1][0] - l2xy[0][0])
    angle2 = abs(math.degrees(math.atan(slope2)))

    theta1 = min(angle1, angle2)
    theta2 = max(angle1, angle2)

    angle = theta2 - theta1

    if color is None:
        color = line1.get_color() # Uses the color of line 1 if color parameter is not passed.

    return Arc(origin, len_x_axis*offset, len_y_axis*offset, 0, theta1, theta2, color=color, label = str(angle)+u"\u00b0")

def the_thing(central_body, origin_orbit, target_orbit, outbound_v, beta):
    T_hat = target_orbit.r / norm(target_orbit.r)
    z_hat = make_z_hat(origin_orbit.r, target_orbit.r)
    project = projector(z_hat)
    
    v_p_vec = project(target_orbit.v)
    v_fb_vec = project(outbound_v)
    v_out_vec = v_fb_vec - v_p_vec
    
    v_inf = norm(v_out_vec)
    theta_inf = angle_between(v_out_vec, T_hat)
    
    # Law of cosines: c^2 = a^2 + b^2 -2ab*cos(gamma)
    b = v_inf
    c = norm(v_p_vec)
    gamma = math.pi - theta_inf - beta
    
    # Quadratic in a: a = b*cos(gamma) +- sqrt(c^2 - b^2*sin^2(gamma))
    a_pos = b*math.cos(gamma) + math.sqrt(c^2 - b^2*math.sin^2(gamma))
    a_neg = b*math.cos(gamma) - math.sqrt(c^2 - b^2*math.sin^2(gamma))
    
    h_sq = (target_orbit.r*v*math.sin(beta))**2

fig = plt.figure()

line_1 = Line2D([0,1], [0,4], linewidth=1, linestyle = "-", color="green")
line_2 = Line2D([0,4.5], [0,3], linewidth=1, linestyle = "-", color="red")

ax = fig.add_subplot(1,1,1)

ax.add_line(line_1)
ax.add_line(line_2)

angle_plot = get_angle_plot(line_1, line_2, 1)
#angle_text = get_angle_text(angle_plot) 
# Gets the arguments to be passed to ax.text as a list to display the angle value besides the arc

ax.add_patch(angle_plot) # To display the angle arc
ax.text(*angle_text) # To display the angle value

ax.set_xlim(0,7)
ax.set_ylim(0,5)

plt.legend()
plt.show()