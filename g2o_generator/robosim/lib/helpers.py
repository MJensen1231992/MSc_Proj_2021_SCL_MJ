import numpy as np
import json
from math import atan2, cos, sin, pi
import utm

def save_to_json(l, name, indent: int = None):
    with open(name, 'w') as f:
        json.dump(l, f, indent=indent)

def load_from_json(name):
    with open(name, 'r') as f:
        return json.load(f)


def from_utm(lat1, lat2, lon1, lon2):
    
    x1, y1, _, _ = utm.from_latlon(lat1, lon1)
    x2, y2, _, _ = utm.from_latlon(lat2, lon2)

    return np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
    

def to_utm(x1, x2, y1, y2):

    lon1, lat1 = utm.to_latlon(x1, y1, 32, 'U')
    lon2, lat2 = utm.to_latlon(x2, y2, 32, 'U')

    return np.linalg.norm(np.array([lon1, lat1]) - np.array([lon2, lat2]))

def min_theta(theta_i, theta_j):
    
    # Setting minimum angular difference 
    diff = theta_j - theta_i

    if diff > pi:
        diff -= 2 * pi
    elif diff <= -pi:
        diff += 2 * pi
    
    return diff

def wrap2pi(angle):

    if angle > np.pi:
        angle = angle-2*np.pi

    elif angle < -np.pi:
        angle = angle + 2*np.pi

    return angle

def vec2trans(p):
    T = np.array([[cos(p[2]), -sin(p[2]), p[0]], 
                 [sin(p[2]),  cos(p[2]), p[1]], 
                 [0, 0, 1]])
    return T

def trans2vec(T):

    x = T[0,2]
    y = T[1,2]
    theta = atan2(T[1,0],
                       T[0,0])
    vec = np.array([x,y,theta],dtype=np.float64)

    return vec

def deg2rad(theta):
    return pi*theta / 180.0

def rad2deg(theta):
    return 180.0 * theta / pi