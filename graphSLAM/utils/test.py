

import numpy as np

info = np.array([[9.70681744, 0.0, 0.0],
    [0.000, 9.70681744, 0.0000],
    [0.000, 0.0000, 9.93812436]])

A = np.array([[-4.54377355e-01,  8.90809306e-01,  6.11364246e+01],
 [-8.90809306e-01, -4.54377355e-01,  2.06766511e+01],
 [ 1.10914510e-02 ,-1.08196586e-02, -1.0000]])

B = np.array([[ 0.07373752,  0.99727768],
 [-0.99727768,  0.07373752],
 [-0.06004772,  0.00390272]])


A= np.array([[-1.12659696],
 [ 0.25298198],
 [-0.01731634]])

B= np.array([[-1.12659696],
[ 0.25298198]])

info = [[163.81269212]]
print(np.shape(info))
print(np.dot(A,info))
#print(B)
