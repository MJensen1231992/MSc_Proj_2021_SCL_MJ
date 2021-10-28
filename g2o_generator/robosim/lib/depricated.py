import random
import numpy as np
from math import pi, atan2, cos, sin

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
