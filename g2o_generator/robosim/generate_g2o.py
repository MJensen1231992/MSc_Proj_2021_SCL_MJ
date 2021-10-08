import numpy as np
import matplotlib.pyplot as plt
import utm
from math import *

import sys
sys.path.append('g2o_generator/GIS_Extraction')

# local import
from lib.utility import *
from create_world import *
import csv_reader as GIS

class g2o:
    def __init__(self, odometry_file, landmarks):

        odometry = load_from_json(odometry_file)
        temp_x = np.asfarray(odometry[0]); temp_y = np.asfarray(odometry[1]); temp_th = np.asfarray(odometry[2])
        loaded_route = [[pose_x, pose_y, pose_th] for pose_x, pose_y, pose_th in zip(temp_x, temp_y, temp_th)]

        full_route = do_rom_splines(np.asfarray(loaded_route))
        temp_x1, temp_y1, temp_th1 = zip(*full_route)
        reduced_path = reduce_dimensions(np.array([temp_x1, temp_y1, temp_th1]))

        self.x, self.y, self.th = zip(*np.asarray_chkfinite(reduced_path, dtype=np.float64))
        # self.x = self.x[1:-2]; self.y = self.y[1:-2]; self.th = self.th[1:-2]
        self.xN, self.yN, self.thN = addNoise(self.x, self.y, self.th)

        with open(landmarks, 'r') as f:
            self.landmarks = np.genfromtxt(f, delimiter=',')

        # For plotting polygons
        filenamePoints = 'g2o_generator/GIS_Extraction/data/aarhus_features.csv'
        filenamePoly = 'g2o_generator/GIS_Extraction/data/aarhus_polygons.csv'
        self.aarhus = GIS.read_csv(filenamePoints, filenamePoly)
        self.rowPoints, self.rowPoly = self.aarhus.read()

        # GPS on ground truth
        self.x_gps, self.y_gps = g2o.GNSS_reading(self.x, self.y)


        # Write g2o file
        if True:
            # Using noise route for g2o file
            g2oW = self.writeOdometry(self.xN, self.yN, self.thN, self.landmarks)
            self.do_loop_closure(self.xN, self.yN, self.thN, self.landmarks, g2oW)
            

        if True:

            # Scale of map
            miny = min(self.y)
            minx = min(self.x)
            maxy = max(self.y)
            maxx = max(self.x)             

            # print("Diagonal distance of map {:.2f} meters".format(np.linalg.norm(np.array([minx,miny]) - np.array([maxx, maxy]))))
            # print("Diagonal distance of map {:.10f} in utm".format(np.linalg.norm(np.array([minlon, minlat]) - np.array([maxlon, maxlat]))))

            # Plotting:
            self.aarhus.squeeze_polygons(self.rowPoly, plot=True)
            plt.plot(self.x, self.y, label='Ground truth', color='purple')
            plt.plot(self.xN, self.yN, label='Noise route', color='orange')
            plt.scatter(self.x[50], self.y[50], color='red')
            # plt.scatter(self.x_gps, self.y_gps, label='GPS points', color='darkblue')
            plt.scatter(self.landmarks[:,0], self.landmarks[:,1], color='green', marker='x', label='Landmarks')
            circle1 = plt.Circle((self.x[50], self.y[50]), 0.00013951, fill=False, label='Loop closure range', color='red')
            plt.gca().add_patch(circle1)
            plt.legend(loc='upper right')
            plt.ylabel("Latitude")
            plt.xlabel("Longitude")
            plt.show()

    @staticmethod
    def vec2trans(p):
        T = np.array([[cos(p[2]), -sin(p[2]), p[0]], 
                      [sin(p[2]),  cos(p[2]), p[1]], 
                      [0, 0, 1]])
        return T

    @staticmethod
    def GNSS_reading(x_odo, y_odo, std_gps_x: float = 0.33, std_gps_y: float = 0.33):
        # Add GNSS points
        x_gps = []
        y_gps = []
        for i in range(len(x_odo)):
            # Adding GPS point for every 5th point
            if (i % 5 == 0):
                # print('Adding GPS point {}'.format(i))
                x_gpsN, y_gpsN = add_GNSS_noise(x_odo[i], y_odo[i], std_gps_x, std_gps_y)
                x_gps.append(x_gpsN); y_gps.append(y_gpsN)

        print('Added {} GPS readings'.format(len(x_gps)))
        return x_gps, y_gps

    def writeOdometry(self, x, y, theta, landmarks):
        g2oW = open('g2o_generator/robosim/data/g2o/noise.g2o', 'w')

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
        
        H = "100.0 0.0 0.0 100.0 0.0 1000.0"
        for i in range(1, len(x)):
            p1 = (x[i-1], y[i-1], theta[i-1])
            p2 = (x[i], y[i], theta[i])

            T1_w = g2o.vec2trans(p1)
            T2_w = g2o.vec2trans(p2)
            T2_1 = np.linalg.inv(T1_w) @ T2_w

            del_x = str(T2_1[0][2])
            del_y = str(T2_1[1][2])
            del_th = str(atan2(T2_1[1, 0], T2_1[0, 0]))

            l = '{} {} {} {} {} {} {}'.format("EDGE_SE2", str((i - 1) + len(landmarks[:,0])), str(i + len(landmarks[:,0])), del_x, del_y, del_th, H)
            g2oW.write(l)
            g2oW.write("\n")

        return g2oW

    def ground_truth(self):
        g2oWG = open('g2o_generator/robosim/data/g2o/ground_truth.g2o', 'w')
        
        for i in range(len(self.x)):
            l = '{} {} {} {} {}'.format("VERTEX_SE2", str(i), self.x[i], self.y[i], self.th[i])
            g2oWG.write(l)
            g2oWG.write("\n")

    def do_loop_closure(self, x, y, th, landmarks, g2oW):       

        points =  [[x, y, th] for x, y, th in zip(x, y, th)]

        n_lc_odo = 0
        n_lc_lm = 0

        # just random set until i can calucalte something

        H = "100.0 0.0 0.0 100.0 0.0 1000.0" # Odometry
        H_XY = "100.0 0.0 100.0" # Landmark

        # Gotten from the mean distance of all points to have to idea of the distance to use in this scale
        # lc_range = 0.0003285251851053485 # Qualitative 
        lc_range = 10 # 10 meters


        for pose_id in range(len(x)):
            
            # Re initializing for every pose_id
            lc_check_odo = []
            check_pose_odo = 0

            # Odometry constraints
            for j in range(len(x)):
                if  j < pose_id and abs(pose_id - j) > 8:

                    d_odo = np.linalg.norm(np.array(points[pose_id][0:2], dtype=np.float64) - np.array(points[j][0:2], dtype=np.float64))
                    # print("d_odo: ",d_odo)
                    lc_check_odo.append(np.greater(lc_range, d_odo))

            # Checking if previous poses are withing range_lc
            for check1 in lc_check_odo:
                if check1 == True:

                    n_lc_odo += 1
                    edge_lc_odo = (check_pose_odo, pose_id+1)

                    lc_constraint_odo = self.generate_lc_constraint(np.array([[x[check_pose_odo]], [y[check_pose_odo]], [th[check_pose_odo]]]), 
                                                                    np.array([[x[pose_id]], [y[pose_id]], [th[pose_id]]]), 
                                                                    "odometry")

                    l1 = '{} {} {} {} {} {} {}'.format("EDGE_SE2", edge_lc_odo[0]+len(landmarks[:,0]), edge_lc_odo[1]+len(landmarks[:,0]), 
                                                lc_constraint_odo[0,0], lc_constraint_odo[1,0], lc_constraint_odo[2,0], H)
                    g2oW.write(l1)
                    g2oW.write('\n')

                check_pose_odo += 1

            # Re initialization
            check_pose_lm = 0
            lc_check_lm = []

            # Writing landmark constraints
            for j in range(len(landmarks[:,0])):

                d_landmark = np.linalg.norm(np.array(points[pose_id][0:2], dtype=np.float64) - np.array(self.landmarks[j,0:2], dtype=np.float64))
                # print("d_landmark: ",d_landmark)
                lc_check_lm.append(np.greater(lc_range, d_landmark))

            # Checking if landmark distance is within range_lc
            for check2 in lc_check_lm:
                if check2 == True:
                    
                    n_lc_lm += 1
                    edge_lc_lm = (pose_id+1, check_pose_lm)
                    
                    lc_constraint_lm = self.generate_lc_constraint(np.array([[x[pose_id]], [y[pose_id]], [th[pose_id]]]),
                                                                np.array([[landmarks[check_pose_lm,0]], [landmarks[check_pose_lm,1]]]),
                                                                "landmark")

                    l2 = '{} {} {} {} {} {}'.format("EDGE_SE2_XY", edge_lc_lm[0]+len(landmarks[:,0]), edge_lc_lm[1], 
                                                    lc_constraint_lm[0][0], lc_constraint_lm[0][1],
                                                    H_XY)
                    g2oW.write(l2)                   
                    g2oW.write('\n')

                check_pose_lm += 1

        print('Number of loop closure (odometry): {}'.format(n_lc_odo))
        print('Number of loop closure (landmarks): {}'.format(n_lc_lm))

    def generate_lc_constraint(self, x1, x2, descriptor: str = None):
        
        if descriptor == "odometry":
            delta_x = x2[0:2,0] - x1[0:2,0]
            R = np.transpose(np.matrix([[cos(x1[2]), -sin(x1[2])], [sin(x1[2]), cos(x1[2])]]))
            delta_x_l = R @ delta_x
            d_theta = min_theta(x2[2] - x1[2])

            return np.array([[delta_x_l[0,0]], [delta_x_l[0,1]], [d_theta[0]]])

        elif descriptor == "landmark":
            delta_x = x2[0:2,0] - x1[0:2,0]
            R = np.transpose(np.matrix([[cos(x1[2]), -sin(x1[2])], [sin(x1[2]), cos(x1[2])]]))
            delta_x_l = R @ delta_x

            return np.array([[delta_x_l[0,0], delta_x_l[0,1]]])

        else:
            return print("Specify a descriptor: (x1, x2, descriptor: str -> 'odometry' or 'landmark')")


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

            T2_1 = np.linalg.inv(T1_w) @ T2_w
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
            T2_wN = T1_wN @ T2_1N

            # Get x2', y2', theta2'
            x2N = T2_wN[0][2]
            y2N = T2_wN[1][2]
            theta2N = atan2(T2_wN[1, 0], T2_wN[0, 0])

            xN[i] = x2N; yN[i] = y2N; tN[i] = theta2N

        return xN, yN, tN
        
    def from_utm(self, lat1, lat2, lon1, lon2):
        
        x1, y1, _, _ = utm.from_latlon(lat1, lon1)
        x2, y2, _, _ = utm.from_latlon(lat2, lon2)

        return np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
    
    def to_utm(self, x1, x2, y1, y2):

        lon1, lat1 = utm.to_latlon(x1, y1, 32, 'U')
        lon2, lat2 = utm.to_latlon(x2, y2, 32, 'U')

        return np.linalg.norm(np.array([lon1, lat1]) - np.array([lon2, lat2]))


if __name__ == '__main__':

    odometry_file = './g2o_generator/robosim/data/robopath/Aarhus_path1.json'
    landmark_file = 'g2o_generator/GIS_Extraction/landmarks/landmarks_points.csv'
    genG2O = g2o(odometry_file, landmark_file)
    genG2O.ground_truth()


    