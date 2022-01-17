import numpy as np
import pandas as pd

from math import atan2

import sys
sys.path.append('g2o_generator/robosim')
from lib.helpers import vec2trans


filename = 'g2o_generator/capra/data/bag_data.csv'
data = pd.read_csv(filename)

df = pd.DataFrame(data, columns=['Timestamp', 'x', 'y', 'th'])
x = df['x'].to_numpy(); y = df['y'].to_numpy(); th = df['th'].to_numpy()

H_odo = np.linalg.inv(np.array([[0.1, 0, 0],
                                [0, 0.1, 0],
                                [0, 0, 0.01]]))

g2o = open('g2o_generator/capra/data/capra.g2o', 'w')

# Odometry vertecies
for idx, (x_, y_, th_) in enumerate(zip(x, y, th)):
    if idx < 0:
        print('VERTEX odometry data is not in correct format')
    else:
        l = '{} {} {} {} {}'.format("VERTEX_SE2", idx, x_, y_, th_)
        g2o.write(l)
        g2o.write("\n")

# Odometry edges
for i in range(1, len(x)):
    p1 = (x[i-1], y[i-1], th[i-1])
    p2 = (x[i], y[i], th[i])

    T1_w = vec2trans(p1)
    T2_w = vec2trans(p2)
    T2_1 = np.linalg.inv(T1_w) @ T2_w

    del_x = str(T2_1[0][2])
    del_y = str(T2_1[1][2])
    del_th = str(atan2(T2_1[1, 0], T2_1[0, 0]))

    l = '{} {} {} {} {} {} {} {} {} {} {} {}'.format("EDGE_SE2", str(i-1), str(i), 
                                        del_x, del_y, del_th, 
                                        H_odo[0,0], H_odo[0,1], H_odo[0,2], H_odo[1,1], H_odo[1,2], H_odo[2,2])
    g2o.write(l)
    g2o.write("\n")