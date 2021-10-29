import numpy as np
import json
from math import atan2, cos, sin, pi

def save_to_json(l, name, indent: int = None):
    with open(name, 'w') as f:
        json.dump(l, f, indent=indent)

def load_from_json(name):
    with open(name, 'r') as f:
        return json.load(f)

def min_theta(theta_i, theta_j):
    
    # Setting minimum angular difference 
    diff = theta_j - theta_i

    if diff > pi:
        diff -= 2 * pi
    elif diff <= -pi:
        diff += 2 * pi
    
    return diff

def vec2trans(p):
    T = np.array([[cos(p[2]), -sin(p[2]), p[0]], 
                 [sin(p[2]),  cos(p[2]), p[1]], 
                 [0, 0, 1]])
    return T

def deg2rad(theta):
    return pi*theta / 180.0

def rad2deg(theta):
    return 180.0 * theta / pi