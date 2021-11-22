import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys

from numpy.random import *
from math import atan2

class BearingOnly:

    def __init__(self):
        pass

    def normalize_angle(self, angle):
        if angle < 0:
            angle += 2 * np.pi
            return angle
        else:
            return angle

    def line_intersection(self, X_vi, X_vj, th_vi, th_vj):
        
        phi_vi = X_vi[2]
        si = np.sin(phi_vi + th_vi) 
        ci = np.cos(phi_vi + th_vi)
            
        phi_vj = X_vj[2]
        sj = np.sin(phi_vj + th_vj) 
        cj = np.cos(phi_vj + th_vj)
        
        x_f = (X_vi[0]*si*cj - X_vj[0]*sj*ci + (X_vj[1]-X_vi[1])*ci*cj)/(si*cj - sj*ci)
        y_f = (X_vj[1]*si*cj - X_vi[1]*sj*ci + (X_vi[0]-X_vj[0])*si*sj)/(si*cj - sj*ci)

        return np.array([x_f, y_f])

class ICP:

    def del_miss(self, indeces, dist, max_dist, th_rate = 0.8):
        th_dist = max_dist * th_rate
        return np.array([indeces[0][np.where(dist.T[0] < th_dist)]])

    def is_converge(self, Tr, scale):
        delta_angle = 0.0001
        delta_scale = scale * 0.0001
        
        min_cos = 1 - delta_angle
        max_cos = 1 + delta_angle
        min_sin = -delta_angle
        max_sin = delta_angle
        min_move = -delta_scale
        max_move = delta_scale
        
        return min_cos < Tr[0, 0] and Tr[0, 0] < max_cos and \
               min_cos < Tr[1, 1] and Tr[1, 1] < max_cos and \
               min_sin < -Tr[1, 0] and -Tr[1, 0] < max_sin and \
               min_sin < Tr[0, 1] and Tr[0, 1] < max_sin and \
               min_move < Tr[0, 2] and Tr[0, 2] < max_move and \
               min_move < Tr[1, 2] and Tr[1, 2] < max_move


    def ICP(self, d1, d2, max_iterate = 100):

        src = np.array([d1.T], copy=True).astype(np.float32)
        dst = np.array([d2.T], copy=True).astype(np.float32)
        
        knn = cv2.ml.KNearest_create()
        responses = np.array(range(len(d2[0]))).astype(np.float32)
        knn.train(src[0], cv2.ml.ROW_SAMPLE, responses)
            
        Tr = np.array([[np.cos(0), -np.sin(0), 0],
                       [np.sin(0), np.cos(0),  0],
                       [0,         0,          1]])

        dst = cv2.transform(dst, Tr[0:2])
        max_dist = sys.maxsize
        
        scale_x = np.max(d1[0]) - np.min(d1[0])
        scale_y = np.max(d1[1]) - np.min(d1[1])
        scale = max(scale_x, scale_y)
        
        for _ in range(max_iterate):
            _, results, _, dist = knn.findNearest(dst[0], 1)
            
            indeces = results.astype(np.int32).T     
            indeces = self.del_miss(indeces, dist, max_dist)  
            
            T, _ = cv2.estimateAffinePartial2D(dst[0, indeces], src[0, indeces], True)

            max_dist = np.max(dist)
            dst = cv2.transform(dst, T)
            Tr = np.dot(np.vstack((T,[0,0,1])), Tr)
            
            if (self.is_converge(T, scale)):
                break
            
        return Tr[0:2]

class LandmarkAssociation:

    def __init__(self):
        self.bearing_only = BearingOnly()
        self.ICP = ICP()

        # Random guess
        self.cluster_interval = (3,10)
        self.L_thresh = 15
        
        self.landmark_hist = []
        self.clusters = {}

    def do_landmark_association(self, curr_pose, curr_landmark):

        if curr_landmark in self.landmark_hist:
            pass
        else:
            self.landmark_hist.append(curr_landmark)

        
        

        


        return

def main():


    import random

    # radius = 200
    # rangeX = (0, 2500)
    # rangeY = (0, 2500)
    # qty = 100  # or however many points you want

    # Generate a set of all points within 200 of the origin, to be used as offsets later
    # There's probably a more efficient way to do this.
    # deltas = set()
    # for x in range(-radius, radius+1):
    #     for y in range(-radius, radius+1):
    #         if x*x + y*y <= radius*radius:
    #             deltas.add((x,y))

    # randPoints = []
    # excluded = set()
    # i = 0
    # while i<qty:
    #     x = random.randrange(*rangeX)
    #     y = random.randrange(*rangeY)
    #     if (x,y) in excluded: continue
    #     randPoints.append((x,y))
    #     i += 1
    #     excluded.update((x+dx, y+dy) for (dx,dy) in deltas)

    # # Robot poses in world coordinates
    # # X_vi = [x_vi, y_vi, phi_vi]
    # X_vi = np.array([3.5, 3, np.deg2rad(12.65694)])
    # # X_vj = [x_vj, y_vj, phi_vj]
    # X_vj = np.array([3.66055, 3.03605, np.deg2rad(3.43883)])

    # # Relative bearing measurements to landmark:
    # b_i = np.deg2rad(63.5016)
    # b_j = np.deg2rad(76.59988)

    # # Landmark gt position: (4.03699, 5.17944)

    # plt.figure()
    # ax = plt.gca()
    # plt.scatter(X_vi[0], X_vi[1], color='red', label='R_i')
    # plt.scatter(X_vj[0], X_vj[1], color='darkred', label='R_j')

    # dx = np.cos(b_i+X_vi[2])
    # dy = np.sin(b_i+X_vi[2])
    # plt.quiver(X_vi[0], X_vi[1], dx, dy, color='blue', angles='xy', scale_units='xy', scale=0.165, label='Bearing')

    # dx = np.cos(b_j+X_vj[2])
    # dy = np.sin(b_j+X_vj[2])
    # plt.quiver(X_vj[0], X_vj[1], dx, dy, color='blue', angles='xy', scale_units='xy', scale=0.2, label='Bearing')
    # # ax.set_xlim([1.5, 4.5])
    # # ax.set_ylim([0.5, 7.5])
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # plt.figure()
    # # Computing the intersection of the two lines
    
    # X_f = BearingOnly.line_intersection(X_vi, X_vj, b_i, b_j)
    # print(f'The coordinate of the landmark is: ({X_f})')
    # print(f'Error in %: {((4.03699-X_f[0])/4.03699+(5.17944-X_f[1])/5.17944)/2*100}')

    # plt.scatter(X_f[0], X_f[1], color='green', label='L')
    # plt.scatter(X_vi[0], X_vi[1], color='red', label='R_i')
    # plt.scatter(X_vj[0], X_vj[1], color='darkred', label='R_j')
    # plt.plot([X_vi[0], X_f[0], X_vj[0], X_f[0]], [X_vi[1], X_f[1], X_vj[1], X_f[1]], label='Intersection', color='blue')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    main()
    
