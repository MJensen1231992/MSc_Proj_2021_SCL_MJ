from matplotlib import markers
import numpy as np
import matplotlib.pyplot as plt
from math import atan2, pi, cos, sin, sqrt, ceil
from sklearn.metrics import mean_squared_error
import random
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning) 

# Local imports
from lib.helpers import *

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

def rom_spline(P0: float, P1: float, P2: float, P3: float, t: tuple=(0, 0.33, 0.66, 1), alpha: float=0.5, N: int = None, d: float=0.01):

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

    xi, yi = Pi[0:2]
    xj, yj = Pj[0:2]

    return sqrt(((xj - xi)**2 + (yj - yi)**2))**alpha + t


def robot_heading(x, y, theta, color: str, length: float=1, alpha=1):
        """
        Method that plots the heading of every pose
        """
        dx = np.cos(theta)
        dy = np.sin(theta)

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

def add_GNSS_noise(x, y, std_gps_x, std_gps_y):
    """
    Adding noise to GNSS(GPS) points from ground truth in UTM32

    Args:
        x ([np.array 1xN]): [x ground truth robot path]
        y ([np.array 1xN]): [y ground truth robot path]
        std_gps_x (float, optional): [standard deviation for GPS points in x direction]. 
        std_gps_y (float, optional): [standard deviation for GPS points in y direction]. 
    """    

    if (random.random() < 0.9):
        std_gps_x, std_gps_y = std_gps_x*2, std_gps_y*2
    else:
        std_gps_x, std_gps_y = 0.33, 0.1

    noise_x_dir = np.random.normal(0, std_gps_x)
    noise_y_dir = np.random.normal(0, std_gps_y)
    x_noise = x + noise_x_dir
    y_noise = y + noise_y_dir

    return x_noise, y_noise

def add_bearing_noise(bearing, systematic_lm_noise, std_lm_bearing):

    noise = np.random.normal(systematic_lm_noise, std_lm_bearing)
    
    return bearing + noise

def add_landmark_noise(landmarks, std_lm_x: float=0.5, std_lm_y: float=0.5):
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
    
def addNoise(x, y, th, std_x, std_y, std_th, mu):

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
        if(i<0):
            xNoise = 0; yNoise = 0; thNoise = 0
        else:
            xNoise = np.random.normal(0, std_x)
            yNoise = np.random.normal(mu, std_y) 
            thNoise = np.random.normal(0, std_th)

        del_xN = del_x + xNoise; del_yN = del_y + yNoise; del_thetaN = del_th + thNoise
        

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
    
    angle = atan2((y2 - y1), (x2 - x1))

    return angle

def calculate_angles(route):
    # Calculating angle between two points of a route
    angles = [atan2((y2 - y1), (x2 - x1)) for (x1, y1), (x2, y2) in zip(route[:-1], route[1:])]

    return angles

def RMSE(predicted, actual):
    return np.square(np.subtract(actual,predicted)).mean() 

def MAE(predicted, actual):
    return abs(np.subtract(actual, predicted)).mean()

def ATE(predicted, actual):

    xsum, ysum, ssum = 0, 0, 0
    pred1 = []
    act1 = []
    steps = []
    xsteps = []
    ysteps = []

    for i in range(len(actual[0][:])-1):

        pred1.append(sqrt((predicted[0][i+1] - predicted[0][i])**2 + (predicted[1][i+1] - predicted[1][i])**2))
        act1.append(sqrt((actual[0][i+1] - actual[0][i])**2 + (actual[1][i+1] - actual[1][i])**2))
        
        xsum += sqrt((predicted[0][i+1] - predicted[0][i])**2) - (sqrt((actual[0][i+1] - actual[0][i])**2))
        ysum += sqrt((predicted[1][i+1] - predicted[1][i])**2) - (sqrt((actual[1][i+1] - actual[1][i])**2))
        # ssum += xsum + ysum 

        # steps.append(ssum)
        xsteps.append(xsum)
        ysteps.append(ysum)
        
    # plt.plot(steps, label='total')
    plt.plot(xsteps, label='xtotal')
    plt.plot(ysteps, label='ytotal')
    plt.legend()

    return (mean_squared_error(act1,pred1)) 
    

def ALE(predicted, actual):

    pred_pose = []
    act_pose = []

    for landmark in predicted.values():
        for pose in landmark:
            pred_pose.append(pose)
    
    for landmark in actual.values():
        for pose in landmark:
            act_pose.append(pose)

    return (mean_squared_error(act_pose,pred_pose)) 

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



    