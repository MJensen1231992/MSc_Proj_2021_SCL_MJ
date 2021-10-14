import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from math import atan2, pi, cos, sin, sqrt, ceil
import json

def do_rom_splines(route):
    size = len(route)
    C = []

    for i in range(size - 3):
        diffs = np.linalg.norm(np.diff(route[i:i+3], axis=0), axis = 1)

        if ~(np.any(diffs) < 0.5):
            c = rom_spline(route[i], route[i+1], route[i+2], route[i+3])

            # Ingoring NaN values
            if ~(np.isnan(np.sum(c))):
                C.extend(c) 
    
    return C

def rom_spline(P0: float, P1: float, P2: float, P3: float, t: tuple = (0, 0.33, 0.66, 1), alpha: float = 0.5, N: int = None, d: float = 0.01):

    P0, P1, P2, P3 = map(np.array, [P0, P1, P2, P3])
    
    if N == None:
        dist = np.linalg.norm(P1 - P2)
        N = ceil(dist / d)

    if N > 40:
        N = 40
    elif N < 30:
        N = 30

    t0 = 0
    t1 = traj(P0, P1, t0, alpha)
    t2 = traj(P1, P2, t1, alpha)
    t3 = traj(P2, P3, t2, alpha)

    # t = [(t2 - t1) * n / (N - 1) + t1 for n in range(N)]
    
    t = np.linspace(t1, t2, N, dtype=np.float128)
    
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

    return ( ( sqrt((xj-xi)**2 + (yj-yi)**2 + (thj-thi)**2 )**0.5 ))**alpha + t


def calculate_angles(route):
    # Calculating angle between two points
    angles = [atan2((y2 - y1), (x2 - x1)) for (x1, y1), (x2, y2) in zip(route[:-1], route[1:])]

    # Iterating over all angles and normalizing angles to [-pi, pi]
    for i in range(len(angles) - 1):
        angles[i] = min_theta(angles[i])
    
    return angles

def calc_bearing(x1,y1,x2,y2):

    angle = atan2((y2 - y1), (x2 - x1))

    return min_theta(angle)

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

def robot_heading(x, y, theta, length: float = 0.1, width: float = 0.1):
        """
        Method that plots the heading of every pose
        """
        x = x[1:-1]
        y = y[1:-1]
        theta = theta[1:-1]

        terminus_x = x + length * np.cos(theta)
        terminus_y = y + length * np.sin(theta)
        plt.plot([x, terminus_x], [y, terminus_y])

def add_GNSS_noise(x, y, std_gps_x: float = 0.33, std_gps_y: float = 0.1):
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

# This function is depricated
def odometry_drift_simple(x, y, th, drift_constant_std: float = 0.33):
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

        noise += np.random.normal(0.33, drift_constant_std)

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

def save_to_json(l, name, indent: int = None):
    with open(name, 'w') as f:
        json.dump(l, f, indent=indent)

def load_from_json(name):
    with open(name, 'r') as f:
        return json.load(f)

def distance_traveled(x,y):
    """Computing euclidean distance traveled 

    Args:
        x (tuple): [row of x values]
        y (tuple): [row of y values]

    Returns:
        [float]: [distance traveled in meters]
    """    
    x = np.vstack(x)
    y = np.vstack(y)
    route = np.hstack((x,y))
    distance = sum([np.linalg.norm(pt1 - pt2) for pt1, pt2 in zip(route[:-1], route[1:])])

    return distance

def addNoise(x, y, th):

    """Takes in odometry values and adding noise in relative pose

    Returns:
        xN, yN, thN: The corresponding odometry values with added noise
    """    

    xN = np.zeros(len(x)); yN = np.zeros(len(y)); tN = np.zeros(len(th))
    xN[0] = x[0]; yN[0] = y[0]; tN[0] = th[0]

    for i in range(1, len(x)):
        # Get T2_1
        p1 = (x[i-1], y[i-1], th[i-1])
        p2 = (x[i], y[i], th[i])

        T1_w = vec2trans(p1)
        T2_w = vec2trans(p2)

        try:
            T2_1 = np.linalg.inv(T1_w) @ T2_w
        except:
            print(f"{T2_1} is not invertible.")
        del_x = T2_1[0][2]
        del_y = T2_1[1][2]
        del_th = atan2(T2_1[1, 0], T2_1[0, 0])

        # Add noise
        if(i<5):
            xNoise = 0; yNoise = 0; thNoise = 0
        else:
            xNoise = np.random.normal(0, 0.033); 
            yNoise = np.random.normal(0, 0.01); 
            thNoise = np.random.normal(0, 0.01)

        del_xN = del_x + xNoise; del_yN = del_y + yNoise; del_thetaN = del_th + thNoise

        # Convert to T2_1'
        T2_1N = np.array([[cos(del_thetaN), -sin(del_thetaN), del_xN], 
                         [sin(del_thetaN), cos(del_thetaN), del_yN], 
                         [0, 0, 1]])

        # Get T2_w' = T1_w' . T2_1'
        p1 = (xN[i-1], yN[i-1], tN[i-1])
        T1_wN = vec2trans(p1)

        T2_wN = T1_wN @ T2_1N

        # Get x2', y2', theta2'
        x2N = T2_wN[0][2]
        y2N = T2_wN[1][2]
        theta2N = atan2(T2_wN[1, 0], T2_wN[0, 0])

        xN[i] = x2N; yN[i] = y2N; tN[i] = theta2N

    return xN, yN, tN

def vec2trans(p):
    T = np.array([[cos(p[2]), -sin(p[2]), p[0]], 
                 [sin(p[2]),  cos(p[2]), p[1]], 
                 [0, 0, 1]])
    return T    

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

