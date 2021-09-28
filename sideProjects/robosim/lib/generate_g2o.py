import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
import scipy

import sys
sys.path.append('sideProjects/robosim')

# local import
from utility import *
from create_world import *

class g2o:
    def __init__(self, odometry_file, landmarks, MAX_RANGE: float = None):
        
        odometry = load_from_json(odometry_file)
        temp_x = np.asfarray(odometry[0]); temp_y = np.asfarray(odometry[1]); temp_th = np.asfarray(odometry[2])
        loaded_route = [[pose_x, pose_y, pose_th] for pose_x, pose_y, pose_th in zip(temp_x, temp_y, temp_th)]

        full_route = do_rom_splines(np.asfarray(loaded_route))
        temp_x1, temp_y1, temp_th1 = zip(*full_route)
        reduced_path = reduce_dimensions(np.array([temp_x1, temp_y1, temp_th1]))

        self.x, self.y, self.th = zip(*np.asarray_chkfinite(reduced_path, dtype=np.float64))
        self.xN, self.yN, self.thN = addNoise(self.x, self.y, self.th)

        with open(landmarks, 'r') as f:
            self.landmarks = np.genfromtxt(f, delimiter=',')

        g2o.loopClosureDistance(self.xN, self.yN)

        if True:
            plt.plot(self.x, self.y)
            plt.plot(self.xN, self.yN)

            for i in range(len(self.th)):
                x2 = 0.000001*cos(self.thN[i]) + self.xN[i]
                y2 = 0.000001*sin(self.thN[i]) + self.yN[i]
                plt.plot([self.xN[i], x2], [self.yN[i], y2], 'm->')

            # plt.xlim([min(self.x), max(self.x)])
            # plt.ylim([min(self.y), max(self.y)])

            plt.show()

    @staticmethod
    def vec2trans(p):
        T = np.array([[cos(p[2]), -sin(p[2]), p[0]], 
                      [sin(p[2]),  cos(p[2]), p[1]], 
                      [0, 0, 1]])
        return T

    def writeOdometry(self, x, y, theta, landmarks):
        g2oW = open('sideProjects/robosim/data/g2o/noise.g2o', 'w')

        # Landmark id and position 
        for idx, (x_l, y_l) in enumerate(zip(landmarks[:,0], landmarks[:,1])):
            if idx < 0:
                print('VERTEX odometry data is not in correct format')
                return
            else:
                l = '{} {} {} {}'.format("VERTEX_XY", idx, x_l, y_l)
                g2oW.write(l)
                g2oW.write("\n")

        # Odometry id and pose
        for idx, (x_odo, y_odo, th_odo) in enumerate(zip(x, y, theta)):
            if idx < 0:
                print('VERTEX odometry data is not in correct format')
                return
            else:
                l = '{} {} {} {} {}'.format("VERTEX_SE2", idx+len(landmarks[:,0]), x_odo, y_odo, th_odo)
                g2oW.write(l)
                g2oW.write("\n")
        
        H = "100.0 0.0 0.0 100.0 0.0 100.0"
        for i in range(1, len(x)):
            p1 = (x[i-1], y[i-1], theta[i-1])
            p2 = (x[i], y[i], theta[i])

            T1_w = g2o.vec2trans(p1)
            T2_w = g2o.vec2trans(p2)
            T2_1 = np.dot(np.linalg.inv(T1_w), T2_w)

            del_x = str(T2_1[0][2])
            del_y = str(T2_1[1][2])
            del_th = str(atan2(T2_1[1, 0], T2_1[0, 0]))

            l = '{} {} {} {} {} {} {}'.format(str(i - 1), str(i), del_x, del_y, del_th, H)
            g2oW.write(l)
            g2oW.write("\n")

        return g2oW

    # def writeLoop()

    @staticmethod
    def loopClosureDistance(x, y):

        points =  [[x, y] for x, y in zip(x, y)]
        distances = []
        loop_closure_points = []

        for i in range(len(x)):
            for j in range(len(x)):
                if  abs(i - j) > 4:
                    
                    d = np.linalg.norm(np.array(points[i], dtype=np.float64) - np.array(points[j], dtype=np.float64))
                    if d > 0.0029285251851053485/2:
                        distances.append(d)
                        # loop_closure_points.append(points[])
                
        # print(np.mean(distances))
        # print(min(distances))
        # print(max(distances))


                


    def addNoise(self, X, Y, THETA):
        """Takes in odometry values and adding noise in relative pose

        Returns:
            xN, yN, thN: The corresponding odometry values with added noise
        """    

        xN = np.zeros(len(X)); yN = np.zeros(len(Y)); tN = np.zeros(len(THETA))
        xN[0] = X[0]; yN[0] = Y[0]; tN[0] = THETA[0]

        for i in range(1, len(X)):
            # Get T2_1
            p1 = (X[i-1], Y[i-1], THETA[i-1])
            p2 = (X[i], Y[i], THETA[i])

            T1_w = g2o.vec2trans(p1)
            T2_w = g2o.vec2trans(p2)

            T2_1 = np.dot(np.linalg.inv(T1_w), T2_w)
            del_x = T2_1[0][2]
            del_y = T2_1[1][2]
            del_theta = atan2(T2_1[1, 0], T2_1[0, 0])

            # Add noise
            if(i<5):
                xNoise = 0; yNoise = 0; tNoise = 0
            else:
                xNoise = np.random.normal(0.0, 0.000035) 
                yNoise = np.random.normal(0.0, 0.000035) 
                tNoise = np.random.normal(0.0, 0.000035)

            del_xN = del_x + xNoise; del_yN = del_y + yNoise; del_thetaN = del_theta + tNoise

            # Convert to T2_1'
            T2_1N = np.array([[cos(del_thetaN), -sin(del_thetaN), del_xN], 
                              [sin(del_thetaN), cos(del_thetaN), del_yN], 
                              [0, 0, 1]])

            # Get T2_w' = T1_w' . T2_1'
            p1 = (xN[i-1], yN[i-1], tN[i-1])
            T1_wN = g2o.vec2trans(p1)
            T2_wN = np.dot(T1_wN, T2_1N)

            # Get x2', y2', theta2'
            x2N = T2_wN[0][2]
            y2N = T2_wN[1][2]
            theta2N = atan2(T2_wN[1, 0], T2_wN[0, 0])

            xN[i] = x2N; yN[i] = y2N; tN[i] = theta2N

        return xN, yN, tN
        
    

if __name__ == '__main__':

    odometry_file = './sideProjects/robosim/data/robopath/Aarhus_path1.json'
    landmark_file = 'sideProjects/GIS_Extraction/landmarks/landmarks_points.csv'
    genG2O = g2o(odometry_file, landmark_file)

    