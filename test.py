import numpy as np
import matplotlib.pyplot as plt
from math import atan2, pi
import random

def min_theta(theta_i, theta_j):
    
    # Setting minimum angular difference 
    diff = theta_j - theta_i

    if diff > pi:
        diff -= 2 * pi
    elif diff <= -pi:
        diff += 2 * pi
    else:
        return theta_j
    
    return diff


print(min_theta(1,3))