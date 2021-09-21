import numpy as np
import matplotlib.pyplot as plt
import cv2
from math import pow, ceil, atan2, pi

def rom_spline(path, t: tuple = (0, 0.33, 0.66, 1), d: float = 0.01, alpha: float = 0.5):
    """
    
    """
    dt = np.dtype('int,int,float')
    path = np.array(path, dtype=dt)
    print(path[0][0])
    dist = np.linalg.norm(np.array([path[0][0],path[0][1]-path[1][0],path[1][1]]))
    N = ceil(dist/d)

    x = [int(p[0]) for p in path]
    y = [int(p[1]) for p in path]
    theta = [float(p[2]) for p in path]
        
    t0 = 0
    t1 = traj(x, y, theta, alpha, t0)
    t2 = traj(x, y, theta, alpha, t1)
    t3 = traj(x, y, theta, alpha, t2)


    T = [(t2 - t1) * n / (N - 1) + t1 for n in range(N)]

    points = []
    for t in T:    
        A1 = (t1 - t)/(t1 - t0)*path[0] + (t - t0)/(t1 - t0)*path[1]
        A2 = (t2 - t)/(t2 - t1)*path[1] + (t - t1)/(t2 - t1)*path[2]
        A3 = (t3 - t)/(t3 - t2)*path[2] + (t - t2)/(t3 - t2)*path[3] 

        B1 = A1*(t2 - t)/(t2 - t0) + A2*(t - t0)/(t2 - t0)
        B2 = A2*(t3 - t)/(t3 - t1) + A3*(t - t1)/(t3 - t1) 
            
        pt = B1*(t2 - t)/(t2 - t1) + B2*(t - t1)/(t2 - t1)
        points.append(pt)

    return points
    
def do_splines(route, angles):
        
    poses = [(pose[0], pose[1], theta) for pose, theta in zip(route, angles)]
    poses = [poses[0]] + poses + [poses[-1]]

    full_route = [rom_spline(poses[i:i+4]) for i in range(len(poses) - 3)]

    return poses, full_route

def traj(x, y, theta, alpha, ti):

        return pow((x[1] - x[0])**2 + (y[1] - y[0])**2 + (theta[1] - theta[0])**2, alpha) + ti

def calculate_angles(route):

    angles = [atan2((y2 - y1), (x2 - x1)) for (x1, y1), (x2, y2) in zip(route[:-1], route[1:])]
    angles = [angles[0]] + angles
    angles = [angle % (2 * pi) for angle in angles]

    for i in range(len(angles) - 1):
        angles[i + 1] = min_theta(angles[i], angles[i + 1])
    
    return angles


def min_theta(x, y):

    if 0 <= (x and y) <= 2*pi:
        a = (x - y) % 2*pi
        b = (y - x) % 2*pi
    else:
        print('Non normalized input')

    return -a if a < b else b

