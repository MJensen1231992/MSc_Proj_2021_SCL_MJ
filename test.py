import numpy as np

x = np.ones((10,3))
print(np.shape(x))
reduced = []
for i in range(len(x[:,0])):
    if (i % 1.5 == 1):
            reduced.append(x[i,:])

print(np.shape(x))
print(np.shape(reduced))
