import numpy as np
import matplotlib.pyplot as plt
import cv2
from math import pow, atan2, pi, sqrt, cos, sin
import json

def do_rom_splines(route):
    size = len(route)
    C = []

    for i in range(size - 3):
        c = rom_spline(route[i], route[i+1], route[i+2], route[i+3])

        # Ingoring NaN values
        if ~(np.isnan(np.sum(c))):
            C.extend(c)
    
    return C

def rom_spline(P0, P1, P2, P3, t: tuple = (0, 0.33, 0.66, 1), alpha: float = 0.5):

    P0, P1, P2, P3 = map(np.array, [P0, P1, P2, P3])

    t0 = 0
    t1 = traj(P0, P1, t0, alpha)
    t2 = traj(P1, P2, t1, alpha)
    t3 = traj(P2, P3, t2, alpha)

    t = np.linspace(t1, t2, 100)
    t = t.reshape(len(t), 1)

    A1 = (t1 - t)/(t1 - t0)*P0 + (t - t0)/(t1 - t0)*P1
    A2 = (t2 - t)/(t2 - t1)*P1 + (t - t1)/(t2 - t1)*P2
    A3 = (t3 - t)/(t3 - t2)*P2 + (t - t2)/(t3 - t2)*P3 

    B1 = A1*(t2 - t)/(t2 - t0) + A2*(t - t0)/(t2 - t0)
    B2 = A2*(t3 - t)/(t3 - t1) + A3*(t - t1)/(t3 - t1) 
            
    pt = B1*(t2 - t)/(t2 - t1) + B2*(t - t1)/(t2 - t1)
        
    return pt

def traj(Pi, Pj, t, alpha):

    xi, yi, thi = Pi
    xj, yj, thj = Pj

    return (pow((xj - xi),2) + pow((yj-yi),2) + pow((thj - thi),2))**alpha + t


def calculate_angles(route):
    # Calculating angle between two points
    angles = [atan2((y2 - y1), (x2 - x1)) for (x1, y1), (x2, y2) in zip(route[:-1], route[1:])]

    # Iterating over all angles and normalizing angles to [-pi, pi]
    for i in range(len(angles) - 1):
        angles[i] = min_theta(angles[i])
    
    return angles


def min_theta(angle):
    # Setting minimum angular difference 
    if (angle > pi):
        angle -= 2 * pi
    elif (angle <= -pi): 
        angle += 2 * pi

    return angle


def odometry_drift_simple(x, y, th, drift_constant_std: float = 0.05):
    """ 
    Simply adding a constant to the wheels to simulate drift 
    The drift is computed using a normal gaussian distribution with standard deviation from the constant drift_constant_std

    return:
        The fully drifted route 
    """

    x = np.asarray_chkfinite(x)
    y = np.asarray_chkfinite(y)
    th = np.asarray_chkfinite(th)

    x_noise = []
    y_noise = []
    th_noise = []

    noise = 0
    for i in range(len(x)):

        x_new = x[i] + noise
        y_new = y[i] + noise*0.1
        th_new = th[i] + noise*0.2

        x_noise.append(x_new)
        y_noise.append(y_new)
        th_noise.append(th_new)

        noise += np.random.normal(0.1, drift_constant_std)

    return x_noise, y_noise, th_noise
    

def reduce_dimensions(route):
    """ Reducing the size of splined route down to a third of the size

    Args:
        route ([np.array([x, y, theta])]): [the full robot route]

    Returns:
        [np.array]: [reduced robot path np.array([x, y, theta])]
    """    

    reduced = []

    for i in range(len(route[0,:])):
        if (i % 1.5 == 1):
            reduced.append(route[:,i])

    print('Reduced size of path from {} to {}'.format(len(route[1]), np.shape(reduced)[0]))

    return np.asarray_chkfinite(reduced)
 


def save_to_json(l, name):
    with open(name, 'w') as f:
        json.dump(l, f)

def load_from_json(name):
    with open(name, 'r') as f:
        return json.load(f)



#######################
# BELOW IS DEPRECATED #

def odometry_drift(x, y, th, std: float = 0.3):
    """
    Adding gaussian noise to odometry
    """
    error_x = np.random.normal(1, std)
    error_y = np.random.normal(1, std*0.5)
    error_th = np.random.normal(1, std*0.1)

    x = np.asarray_chkfinite(x)
    y = np.asarray_chkfinite(y)
    th = np.asarray_chkfinite(th)

    dist = []
    angle = []

    # Converting to polar coordinates and back again in order to add radial noise
    for i in range(len(x)):
        r = sqrt(pow(x[i],2) + pow(y[i],2)) # range
        a = atan2(x[i],y[i])    # angle
        
        dist.append(r * error_x)
        angle.append(a * error_y)
    
    x_noise = []
    y_noise = []
    th_noise = []
    
    for i in range(len(dist)):
        x1 = dist[i] * cos(angle[i])
        y1 = dist[i] * sin(angle[i])
        x_noise.append(x1)
        y_noise.append(y1)
        th_noise.append(th[i] * error_th)

    return x_noise, y_noise, th_noise
    

def relative_position(x, y, th):
    # Route should contain x, y, theta in cartesian absolute coordinates
    # Returns robot path in relative coordinates

    x_start = x[0]
    y_start = y[0]

    x_relative = []
    y_relative = []
    for i in len(x):

        x_relative.append(x_start - x[i])
        y_relative.append(y_start - y[i])

    x_relative, y_relative = x_relative * np.cos(-th) - y_relative * np.sin(-th), \
                             x_relative * np.sin(-th) + y_relative * np.cos(-th)

    return x_relative, y_relative

