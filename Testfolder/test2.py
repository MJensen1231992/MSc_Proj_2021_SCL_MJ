from matplotlib import scale
import numpy as np
import itertools
from math import sin, cos, atan2
import matplotlib.pyplot as plt

filename = "noise_20211207-102548.g2o"

edges = []
nodes = []
lm = []

with open(filename, 'r') as file:
    for line in file:
        data = line.split()
        
        if data[0] == 'VERTEX_SE2':

            nodeId = int(data[1])
            pose = np.array(data[2:5],dtype=np.float64)
            nodes.append(pose)

        elif data[0] == 'VERTEX_XY':

            nodeId = int(data[1])
            landmark = np.array(data[2:4],dtype=np.float64)  
            lm.append(landmark)

        elif data[0] == 'EDGE_SE2_BEARING':

                nodeFrom = int(data[1])
                nodeTo = int(data[2])
                poseMeasurement = np.array(data[3], dtype=np.float64)
                edges.append(poseMeasurement)

_nodes = np.vstack(nodes)
_landmark = np.vstack(lm)
_edge = np.vstack(edges)

z1 = np.hstack((_nodes[1:66,0:2],_edge[1:66]))
new_z1 = np.add(_edge[1:66], _nodes[1:66,2][np.newaxis].T)

z2 = np.hstack((_nodes[66:89,0:2],_edge[66:89]))
new_z2 = np.add(_edge[66:89], _nodes[66:89,2][np.newaxis].T)

z = np.vstack((z1,z2))
new_z = np.vstack((new_z1, new_z2))

A1 = np.zeros((2*len(z[:,0]), 2))
np.set_printoptions(threshold=np.inf)

A2 = np.zeros((2*len(z[:,0]),len(new_z)))
b = np.zeros((2*len(z[:,0]),1))
k = 0

for i in range(2*len(z[:,0])-1):
    
    phi_mj = z[i,2] + new_z[i]
    c = -np.cos(phi_mj)
    s = -np.sin(phi_mj)

    if i > 0:
        i = i + 1

    A1[i,0] = 1
    A1[i+1,1] = 1
    A2[i,i-1] = c
    A2[i+1,i-1] = s
    
    b[i] = z[i,0]
    b[i+1] = z[i,1]


print(A1)
A = np.hstack((A1,A2))

x = np.linalg.pinv(A) @ b
print(x[0:2])
print(lm)




# print(_edge)
# print(_nodes[1:13,2][np.newaxis].T)

plt.scatter(_nodes[:,0], _nodes[:,1], color='blue')

# Bearings
plt.quiver(z[:,0], z[:,1], np.cos(new_z), np.sin(new_z), color='magenta', angles='xy', scale=1)
# plt.quiver(z1[:,0], z1[:,1], np.cos(new_z1), np.sin(new_z1), color='magenta', angles='xy', scale=1)
plt.scatter(_landmark[:,0], _landmark[:,1], color='green')
# Robot orientation
# plt.quiver(_nodes[:,0], _nodes[:,1], np.cos(_nodes[:,2]), np.sin(_nodes[:,2]), angles='xy', scale=20, color='lime')
plt.show()
