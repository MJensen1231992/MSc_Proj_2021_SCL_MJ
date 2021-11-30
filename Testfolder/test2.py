import numpy as np
import itertools
from math import sin, cos

# poses = np.array([[1,1,0, 3.14],[2,2,np.pi/5, 3.14/2],[3,4,np.pi/3, 3.14/1.5]])


# combinations = list(itertools.combinations(poses, 2))
# n_pairs = len(combinations)

# n_stack = []
# P_stack = []
# denom_stack = []

# for d in combinations:
#     # Orientation World RF
#     thi = d[0][2]
#     thj = d[1][2]
    
#     # Bearing Local RF
#     psii = d[0][3]
#     psij = d[1][3]
    
#     xi = d[0][0]
#     xj = d[1][0]

#     yi = d[0][1]
#     yj = d[1][1]

#     P_stack.append(np.array([[xi],[yi], [xj], [yj]]))

#     si = sin(thi+psii); ci = cos(thi+psii)
#     sj = sin(thj+psij); cj = cos(thj+psij)

#     denom_stack.append([1/(si*cj-sj*ci), 1/(si*cj-sj*ci)])

    

#     nom = np.array([[si*cj, -ci*cj, -sj*ci, ci*cj],
#                     [si*sj, -sj*ci, -si*sj, si*cj]])
#     n_stack.append(nom)

# rng = np.arange(n_pairs*2)
# A1 = np.zeros((n_pairs*2, n_pairs*2))
# A1[rng, rng] = list(itertools.chain(*denom_stack))
# A2 = np.vstack(n_stack)
# A = A1 @ A2
# P = np.vstack(P_stack)

# Psi = A @ P
# print(Psi)

# for Xr, z in combs:
#     print(Xr, z)

# lm_poses = np.random.default_rng().uniform(0,10,(4,2))

# print(lm_poses)
# print(lm_poses[0,1])
n_poses = 5
r = [[0,1],[1,1],[2,1],[3,1],[4,1],[5,1],[6,2],[6.5,3],[7,4],[6,5],[5,6]]
route = r[0:n_poses]

print(route)
