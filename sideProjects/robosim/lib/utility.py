import numpy as np
import matplotlib.pyplot as plt
import cv2
from math import pow, ceil, atan2, pi

def do_rom_splines(route):
    size = len(route)
    C = []

    for i in range(size - 3):
        c = rom_spline(route[i], route[i+1], route[i+2], route[i+3])
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

# Old version    
def do_splines(route, angles):
        
    poses = [(pose[0], pose[1], theta) for pose, theta in zip(route, angles)]
    poses = [poses[0]] + poses + [poses[-1]]

    full_route = [rom_spline(poses[i:i+4]) for i in range(len(poses) - 3)]

    return poses, full_route


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

