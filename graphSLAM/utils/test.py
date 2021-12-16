import numpy as np
from math import atan2

v1 = np.array([0, 0, 2, 20])
m1 = (v1[3] - v1[1])/(v1[2] - v1[0])
v2 = np.array([3, 0, 2, 20])
m2 = (v2[3] - v2[1])/(v2[2] - v2[0])
v3 = np.array([421.79,-1676.21,76.69,-162.38])
m3 = (v3[3] - v3[1])/(v3[2] - v3[0])
v4 = np.array([-3.42, 3.05, -1.44, 4.19])
m4 = (v4[3] - v4[1])/(v4[2] - v4[0])

th1 = np.arctan(m1)
th2 = np.arctan(m2)
th3 = np.arctan(m3)
th4 = np.arctan(m4)

parallel = min(np.pi - abs(m1-m2), abs(m1-m2))
parallel1 = abs(m2-m3)
parallel2 = abs(m2-m4)
print(parallel)
# print(parallel1)
# print(parallel2)



