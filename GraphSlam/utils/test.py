import numpy as np


omega = np.zeros((3*3+2*3,3*10+2*3))
omega[0:3, 0:3] = 1.0 * np.identity(3)
observe = np.full((3,3), False)
# print(observe)
# print(omega)
G = np.identity(3)
G_extend = np.zeros((3, 6))

print(G_extend)
G_extend[:, 0:3] = -G
G_extend[:, 3:6] = np.identity(3)

print(G_extend)
