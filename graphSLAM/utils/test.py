from helper import wrap2pi
import numpy as np
import matplotlib.pyplot as plt

# a = np.array([ 1, 2, 3])
# b = [ 4, 5, 6]
# d = [ 5, 7, 9]


# c =np.linalg.norm(a)+np.linalg.norm(b)


# #print(np.linalg.norm(a1+b1))
# k = np.linalg.norm(a+b)

# k1 = np.linalg.norm(a1+b1)
# u = np.linalg.norm(a1)+np.linalg.norm(b1)
# k2 = np.linalg.norm(a2+b2)
# z = np.linalg.norm(a2)+np.linalg.norm(b2)
# print(c, u,z)
# print(k, k1, k2)

# print(u/c,z/u)
# print(k1/k, k2/k1)
# print(k/c, k1/u, k2/z)

# t = np.linspace(0,99,200)
# g = np.linspace(0,999,200)
# tl=np.linalg.norm(t)
# gl=np.linalg.norm(g)
# print(t)
# print(g)
# print(tl+gl)
# print(np.linalg.norm(sum(a)))
# print(np.linalg.norm([ 71.7565695,  137.72204542,   6.64420391]))


mean =[0,0]

# b = np.array([[ 0.,         0.,          1.52578446],
#  [-0.49545915,  1.31452831,  1.55420751],
#   [-0.49545915,  1.31452831,  1.55420751]])

# print(b[:, :2])

# x,y = np.random.multivariate_normal(mean,cov, 10000).T
# plt.hist2d(x,y,bins=30, cmap='Blues')
# plt.show()

# b = np.asfarray([2,3,40,10])
# Id = 2


# print(Id == np.any(b))

a1 = np.array([9, 1, 24])
b1 = np.array([5, 6, 7])
a2 = [3, 4, 5]
b2 = [6, 7, 8]

c = a1-b1
print(c)
print(np.var(c))
print(np.median(c))