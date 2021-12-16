from helper import wrap2pi
import numpy as np


# a = np.array([ 1, 2, 3])
# b = [ 4, 5, 6]
# d = [ 5, 7, 9]


# c =np.linalg.norm(a)+np.linalg.norm(b)
# a1 = [2, 3, 4]
# b1 = [5, 6, 7]
# a2 = [3, 4, 5]*1000
# b2= [6, 7, 8]*1000

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


headingA = -9.46
headingB = -41.48

measA = 162.18
measB = 194.35-360

print(measB)


A_bear =  headingA+measA
B_bear =  headingB+measB


print(np.abs(A_bear-B_bear)-360)

print(0.03*180/np.pi)