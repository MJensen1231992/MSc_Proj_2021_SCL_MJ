import numpy as np
import matplotlib.pyplot as plt
from math import atan2, pi, cos, sin
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

def rom_spline(P0: float, P1: float, P2: float, P3: float, t: tuple = (0, 0.33, 0.66, 1), alpha: float = 0.5):

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
            
    return ( ( (xj-xi)**2 + (yj-yi)**2 + (thj - thi)**2 )**0.5 )**alpha + t


def calculate_angles(route):
    # Calculating angle between two points
    angles = [atan2((y2 - y1), (x2 - x1)) for (x1, y1), (x2, y2) in zip(route[:-1], route[1:])]

    # Iterating over all angles and normalizing angles to [-pi, pi]
    for i in range(len(angles) - 1):
        angles[i] = min_theta(angles[i])
    
    return angles


def min_theta(theta):
    # Setting minimum angular difference 
    if (theta > pi):
        theta -= 2 * pi
    elif (theta <= -pi): 
        theta += 2 * pi

    return theta


def deg2grad(theta):
    
    return pi*theta / 180.0

def rad2deg(theta):

    return 180.0 * theta / pi

def robot_heading(x, y, theta, length: float = 0.00001, width: float = 0.00001):
        """
        Method that plots the heading of every pose
        """
        x = x[1:-1]
        y = y[1:-1]
        theta = theta[1:-1]

        terminus_x = x + length * np.cos(theta)
        terminus_y = y + length * np.sin(theta)
        plt.plot([x, terminus_x], [y, terminus_y])

def add_GNSS_noise(x, y, std_gps_x: float = 0.0001, std_gps_y: float = 0.001):
    """Adding noise to GNSS(GPS) points from groundt truht

    Args:
        x ([np.array 1xN]): [x displacement]
        y ([np.array 1xN]): [y displacement]
        std_gps (float, optional): [standard deviation for GPS points]. Defaults to 0.01.
    """    
    noise_x_dir = np.random.normal(0, std_gps_x)
    noise_y_dir = np.random.normal(0, std_gps_y)
    x_noise = x + noise_x_dir
    y_noise = y + noise_y_dir
    return x_noise, y_noise

def odometry_drift_simple(x, y, th, drift_constant_std: float = 0.0000001):
    """ 
    Simply adding a constant to the wheels to simulate drift 
    The drift is computed using a normal gaussian distribution with standard deviation from the constant drift_constant_std

    return:
        The fully drifted route 
    """

    x, y, th = convert_to_np_array(x, y, th)

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

        noise += np.random.normal(0.000001, drift_constant_std)

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
        if (i % 5 == 0):
            reduced.append(route[:,i])

    print('Reduced size of path from {} to {}'.format(len(route[1]), np.shape(reduced)[0]))

    return np.asarray_chkfinite(reduced)
 

def convert_to_np_array(x, y, th):
    # Converting list to numpy array
        
    x_new = np.asarray_chkfinite(x)
    y_new = np.asarray_chkfinite(y)
    th_new = np.asarray_chkfinite(th)

    return x_new, y_new, th_new

def save_to_json(l, name):
    with open(name, 'w') as f:
        json.dump(l, f)

def load_from_json(name):
    with open(name, 'r') as f:
        return json.load(f)


#######################
# BELOW IS DEPRECATED #

def odometry_drift(x, y, th, std: float = 0.03):
    
    x, y, th = convert_to_np_array(x, y, th)
    xN = np.zeros(len(x)); yN = np.zeros(len(y)); thN = np.zeros(len(th))
    xN[0] = x[0]; yN[0] = y[0]; thN[0] = th[0]

    for i in range(1, len(x)):
        pcurr = (x[i-1], y[i-1], th[i-1])
        pnext = (x[i], y[i], th[i])
        
        Tcurr_world = np.array([[cos(pcurr[2]), -sin(pcurr[2]), pcurr[0]],
                                [sin(pcurr[2]),  cos(pcurr[2]), pcurr[0]],
                                [0, 0, 1]])
        Tnext_world = np.array([[cos(pnext[2]), -sin(pnext[2]), pnext[0]],
                                [sin(pnext[2]),  cos(pnext[2]), pnext[0]],
                                [0, 0, 1]])
        Tnext_curr = np.dot(np.linalg.inv(Tcurr_world), Tnext_world)

        del_x = Tnext_curr[0][2]
        del_y = Tnext_curr[1][2]
        del_th = atan2(Tnext_curr[1, 0], Tnext_curr[0, 0])

        # Adding noise
        if (i<5):
            x_noise = 0; y_noise = 0; th_noise = 0
        else:
            x_noise = np.random.normal(0, std)
            y_noise = np.random.normal(0, std)
            th_noise = np.random.normal(0, std)
        
        del_xn = del_x + x_noise
        del_yn = del_y + y_noise
        del_thn = del_th + th_noise

        # Converting to Tnext_curr'
        Tnext_curr_n = np.array([[cos(del_thn), -sin(del_thn), del_xn],
                                [sin(del_thn),  cos(del_thn), del_yn],
                                [0, 0, 1]])

        pcurr = (xN[i-1], yN[i-1], thN[i-1])
        Tcurr_wN = np.array([[cos(pcurr[2]), -sin(pcurr[2]), pcurr[0]],
                                [sin(pcurr[2]),  cos(pcurr[2]), pcurr[0]],
                                [0, 0, 1]])
        Tnext_wN = np.dot(Tcurr_wN, Tnext_curr_n)

        x2N = Tnext_wN[0][2]
        y2N = Tnext_wN[1][2]
        theta2N = atan2(Tnext_wN[1, 0], Tnext_wN[0, 0])

        xN[i] = x2N; yN[i] = y2N; thN[i] = theta2N

    return xN, yN, thN
    

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

