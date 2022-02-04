import numpy as np
import matplotlib.pyplot as plt
from math import atan2, pi, cos, sin, sqrt, ceil
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning) 

# Local imports
from lib.helpers import *

def do_rom_splines(route: list):
    """Execute Catmull Rom spline algorithm over a sequence of points

    Args:
        route (list): list of ground truth points

    Returns:
        C: route
    """
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

def rom_spline(P0: float, P1: float, P2: float, P3: float, t: tuple=(0, 0.33, 0.66, 1), alpha: float=0.5, N: int = None, d: float=0.5):
    """The core math for making Catmull Rom splines. alpha of 0.5 results in centripetal parameterization

    Args:
        P0, P1, P2, P3 (float): Control point 1 to 4
        t (tuple, optional): knot sequence. Defaults to (0, 0.33, 0.66, 1).
        alpha (float, optional): parameterization value. Defaults to 0.5.
        N (int, optional): Amount of points in the interpolation, changes dynamically. Defaults to None.
        d (float, optional): variable to change N. Defaults to 0.5.

    Returns:
        pt: interpolation points 
    """
    P0, P1, P2, P3 = map(np.array, [P0, P1, P2, P3])
    
    if N == None:
        dist = np.linalg.norm(P1 - P2)
        N = ceil(dist / d)

    t0 = 0
    t1 = traj(P0, P1, t0, alpha)
    t2 = traj(P1, P2, t1, alpha)
    t3 = traj(P2, P3, t2, alpha)
    
    t = np.linspace(t1, t2, N, dtype=np.float64)
    
    t = t.reshape(len(t), 1)

    A1 = (t1 - t)/(t1 - t0)*P0 + (t - t0)/(t1 - t0)*P1
    A2 = (t2 - t)/(t2 - t1)*P1 + (t - t1)/(t2 - t1)*P2
    A3 = (t3 - t)/(t3 - t2)*P2 + (t - t2)/(t3 - t2)*P3 

    B1 = (t2 - t)/(t2 - t0)*A1 + (t - t0)/(t2 - t0)*A2
    B2 = (t3 - t)/(t3 - t1)*A2 + (t - t1)/(t3 - t1)*A3
            
    pt = (t2 - t)/(t2 - t1)*B1 + (t - t1)/(t2 - t1)*B2
        
    return pt

def traj(Pi, Pj, t, alpha):
    """
    Helper function to catmull rom splines
    """    
    xi, yi = Pi[0:2]
    xj, yj = Pj[0:2]

    return sqrt(((xj - xi)**2 + (yj - yi)**2))**alpha + t


def robot_heading(x, y, theta, color: str, length: float=1, alpha: float=1, constant: float=5):
        """
        Method that plots the heading of every pose
        """
        dx = np.cos(theta)*constant
        dy = np.sin(theta)*constant

        plt.quiver(x, y, dx, dy, color=color, angles='xy', scale_units='xy', scale=length, alpha=alpha)



def reduce_dimensions(route, descriptor: str='half'):
    """ Reducing the size of splined route down to a third of the size

    Args:
        route ([np.array([x, y, theta])]): [the full robot route]

    Returns:
        [np.array]: [reduced robot path np.array([x, y, theta])]
    """    

    reduction = \
    {
        'none': 1,
        'half': 2,
        'third': 3,
        'fourth': 4,
        'fifth': 5,
        'tenth': 10,
        'twentieth': 20,
        'thirtieth': 30
    }

    reduced = []

    for i in range(len(route[0,:])):
        if (i % reduction[descriptor] == 0):
            reduced.append(route[:,i])

    print('Reduced size of path from {} to {}'.format(len(route[1]), np.shape(reduced)[0]))

    return np.asarray_chkfinite(reduced)

def distance_traveled(x, y):
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

def add_bearing_noise(bearing, systematic_lm_noise, std_lm_bearing):
    """Adding noise to landmark bearing observations

    Returns:
        float: noisy bearing in radians
    """    
    
    noise = np.random.normal(systematic_lm_noise, std_lm_bearing)
    
    return bearing + noise

def add_landmark_noise(landmarks, std_lm_x: float=0.5, std_lm_y: float=0.5):
    """Adding noise to landmark positions from GIS data to robot simulator.

    Args:
        landmarks ([dict]): contains landmark type [key] and position [value]
        std_lm_x (float, optional): [gaussian sample for noise in x direction]. Defaults to 0.5.
        std_lm_y (float, optional): [gaussian sample for noise in y direction]. Defaults to 0.5.

    Returns:
        [dict]: contains noisy landmarks 
    """    
    new_landmarks = {}
    for key, landmark in landmarks.items():
        for pos in landmark:
            
            noise_x_dir = np.random.normal(0, std_lm_x)
            noise_y_dir = np.random.normal(0, std_lm_y)

            x_noise = pos[0] + noise_x_dir
            y_noise = pos[1] + noise_y_dir

            new_landmarks.setdefault(key, [])
            new_landmarks[key].append(([x_noise, y_noise]))
    
    return new_landmarks
    
def addNoise(x, y, th, std_x: float, std_y: float, std_th: float, mu: float):
    """Creating odometry path from ground truth poses

    Args:
        x, y, th (float): ground truth pose
        std_x (float): std for gaussian sample in x
        std_y (float): std for gaussian sample in y
        std_th (float): std for gaussian sample in th
        mu (float): bias in y-direction

    Returns:
        xN, yN, thN (float): odometry route
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
        if(i<0):
            xNoise = 0; yNoise = 0; thNoise = 0
        else:
            xNoise = np.random.normal(0, std_x)
            yNoise = np.random.normal(mu, std_y) 
            thNoise = np.random.normal(0, std_th)

        del_xN = del_x + xNoise; del_yN = del_y + yNoise; del_thetaN = wrap2pi(del_th + thNoise)
        

        # Convert to T2_1'
        T2_1N = np.array([[cos(del_thetaN), -sin(del_thetaN), del_xN], 
                          [sin(del_thetaN),  cos(del_thetaN), del_yN], 
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


def calc_bearing(x1, y1, x2, y2):
    """Creating angle

    Returns:
        float: angle between robot and landmark
    """    
    angle = atan2((y2 - y1), (x2 - x1))

    return angle

def calculate_angles(route):
    """Creating angles between consecutive robot poses

    Args:
        route (array): robot positions (x,y)

    Returns:
        angles: list of angles in radians in the interval (-pi, pi] 
    """    
    
    angles = [atan2((y2 - y1), (x2 - x1)) for (x1, y1), (x2, y2) in zip(route[:-1], route[1:])] 

    return angles

def _ready_data(data):
    data = np.vstack(data)

    datax = data[:,0]
    datay = data[:,1]

    dx = np.vstack([datax[0::2], datax[1::2]])
    dy = np.vstack([datay[0::2], datay[1::2]])

    d_dx = abs(dx[0] - dx[1])
    d_dy = abs(dy[0] - dy[1])

    return d_dx, d_dy

def plot_outliers_vs_normal(pose_pose_outlier, pose_landmark_outlier, pose_pose_inliers, pose_landmark_inliers):

    pp_dx, pp_dy = _ready_data(pose_pose_inliers)
    pl_dx, pl_dy = _ready_data(pose_landmark_inliers)

    ppo_dx, ppo_dy = _ready_data(pose_pose_outlier)
    plo_dx, plo_dy = _ready_data(pose_landmark_outlier)

    plt.figure()
    plt.scatter(pp_dx, pp_dy, marker='^', color='magenta', label='pose-pose inliers')
    plt.scatter(ppo_dx, ppo_dy, marker='x', color='red', label='pose-pose outliers')
    plt.legend()
    
    plt.figure()
    plt.scatter(pl_dx, pl_dy, marker='<', color='lime', label='pose-landmark inliers')
    plt.scatter(plo_dx, plo_dy, marker='o', color='blue', label='pose-landmark outliers')
    plt.legend()
    plt.show()



    