import random
import numpy as np
from math import pi, atan2, cos, sin, sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# @staticmethod
# def GNSS_reading(x_odo, y_odo, frequency: int=5, std_gps_x: float=0.33, std_gps_y: float=0.33):
#     x_gps, y_gps = [], []
#     gt_x_gps, gt_y_gps = [], []

#     for i in range(len(x_odo)):
#         if (i % frequency == 0):
            
#             x_gpsN, y_gpsN = add_GNSS_noise(x_odo[i], y_odo[i], std_gps_x, std_gps_y)

#             x_gps.append(x_gpsN); y_gps.append(y_gpsN)
#             gt_x_gps.append(x_odo[i]); gt_y_gps.append(y_odo[i])

#     # print('Added {} GPS readings'.format(len(x_gps)))
#     return x_gps, y_gps, gt_x_gps, gt_y_gps, len(x_gps)

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

def convert_to_np_array(x, y, th):
    # Converting list to numpy array
        
    x_new = np.asarray_chkfinite(x)
    y_new = np.asarray_chkfinite(y)
    th_new = np.asarray_chkfinite(th)

    return x_new, y_new, th_new

def odometry_drift_simple(x, y, th, drift_constant_std: float=0.33):
    """ 
    Adding a constant to the wheels to simulate drift 
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

    # Trying to impement Thrun noise model algorithm ://
def addNoisev2(x,y,th):

    xN = np.zeros(len(x)); yN = np.zeros(len(y)); tN = np.zeros(len(th))
    xN[0] = x[0]; yN[0] = y[0]; tN[0] = th[0]

    for i in range(1, len(x)):
        
        p1 = (x[i-1], y[i-1], th[i-1])
        p2 = (x[i], y[i], th[i])

        T1_w = vec2trans(p1)
        T2_w = vec2trans(p2)

        T2_1 = np.linalg.inv(T1_w) @ T2_w

        dx = T2_1[0][2]
        dy = T2_1[1][2]

        trans = sqrt(dx*dx + dy*dy)
        rot1 = atan2(T2_w[1][2] - T1_w[1][2], T2_w[0][2] - T1_w[0][2]) - atan2(T1_w[1][2], T1_w[0][2])
        rot2 = atan2(T2_w[1][2], T2_w[0][2]) - atan2(T1_w[1][2], T1_w[0][2]) - rot1

        a1 = 0.005
        a2 = 1.0*pi/180.0
        a3 = 0.005
        a4 = 0.001

        sd_rot1 = a1*abs(rot1) + a2*trans
        sd_rot2 = a1*abs(rot2) + a2*trans
        sd_trans = a3*trans + a4*(abs(rot1)) + abs(rot2)

        t = trans + np.random.normal(0, sd_trans*sd_trans)
        r1 = rot1 + np.random.normal(0, sd_rot1*sd_rot1)
        r2 = rot2 + np.random.normal(0, sd_rot2*sd_rot2)

        xT = t*cos(atan2(T1_w[1][2], T1_w[0][2]) + r1)
        yT = t*sin(atan2(T1_w[1][2], T1_w[0][2]) + r1)
        tT = atan2(T1_w[1][2], T1_w[0][2]) + r1 + r2

        p3 = (xT, yT, tT)
        TN = vec2trans(p3)

        p1 = (xN[i-1], yN[i-1], tN[i-1])
        TNN = vec2trans(p1)

        TNNN = TNN @ TN

        xN[i] = TNNN[0][2]
        yN[i] = TNNN[1][2]
        tN[i] = atan2(TNN[1, 0], TNNN[0, 0])
        

    return xN, yN, tN

class ICP:

    def del_miss(self, indeces, dist, max_dist, th_rate = 0.8):
        th_dist = max_dist * th_rate
        return np.array([indeces[0][np.where(dist.T[0] < th_dist)]])

    def is_converge(self, Tr, scale):
        delta_angle = 0.0001
        delta_scale = scale * 0.0001
        
        min_cos = 1 - delta_angle
        max_cos = 1 + delta_angle
        min_sin = -delta_angle
        max_sin = delta_angle
        min_move = -delta_scale
        max_move = delta_scale
        
        return min_cos < Tr[0, 0] and Tr[0, 0] < max_cos and \
               min_cos < Tr[1, 1] and Tr[1, 1] < max_cos and \
               min_sin < -Tr[1, 0] and -Tr[1, 0] < max_sin and \
               min_sin < Tr[0, 1] and Tr[0, 1] < max_sin and \
               min_move < Tr[0, 2] and Tr[0, 2] < max_move and \
               min_move < Tr[1, 2] and Tr[1, 2] < max_move


    def ICP(self, d1, d2, max_iterate = 100):

        src = np.array([d1.T], copy=True).astype(np.float32)
        dst = np.array([d2.T], copy=True).astype(np.float32)
        
        knn = cv2.ml.KNearest_create()
        responses = np.array(range(len(d2[0]))).astype(np.float32)
        knn.train(src[0], cv2.ml.ROW_SAMPLE, responses)
            
        Tr = np.array([[np.cos(0), -np.sin(0), 0],
                       [np.sin(0), np.cos(0),  0],
                       [0,         0,          1]])

        dst = cv2.transform(dst, Tr[0:2])
        max_dist = sys.maxsize
        
        scale_x = np.max(d1[0]) - np.min(d1[0])
        scale_y = np.max(d1[1]) - np.min(d1[1])
        scale = max(scale_x, scale_y)
        
        for _ in range(max_iterate):
            _, results, _, dist = knn.findNearest(dst[0], 1)
            
            indeces = results.astype(np.int32).T     
            indeces = self.del_miss(indeces, dist, max_dist)  
            
            T, _ = cv2.estimateAffinePartial2D(dst[0, indeces], src[0, indeces], True)

            max_dist = np.max(dist)
            dst = cv2.transform(dst, T)
            Tr = np.dot(np.vstack((T,[0,0,1])), Tr)
            
            if (self.is_converge(T, scale)):
                break
            
        return Tr[0:2]

class LandmarkAssociation:

    def __init__(self):
        self.bearing_only = BearingOnly()
        self.ICP = ICP()

        # Random guess
        self.cluster_interval = (3,10)
        self.L_thresh = 15
        
        self.landmark_hist = []
        self.clusters = {}

    def do_landmark_association(self, curr_pose, curr_landmark):

        if curr_landmark in self.landmark_hist:
            pass
        else:
            self.landmark_hist.append(curr_landmark)

        return
