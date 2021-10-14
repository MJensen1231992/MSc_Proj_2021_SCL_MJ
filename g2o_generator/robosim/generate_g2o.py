import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import angle
import utm
from math import *

import sys
sys.path.append('g2o_generator/GIS_Extraction')

# local import
from lib.utility import *
import create_world as CW
import csv_reader as GIS

class g2o:
    def __init__(self, odometry_file, filenamePoints, filenamePoly, filelandmarks):

        self.pose_pose, self.pose_landmark = [], []

        # Loading and extracting save odometry route
        odometry = load_from_json(odometry_file)
        temp_x = np.asfarray(odometry[0]); temp_y = np.asfarray(odometry[1]); temp_th = np.asfarray(odometry[2])
        loaded_route = [[pose_x, pose_y, pose_th] for pose_x, pose_y, pose_th in zip(temp_x, temp_y, temp_th)]

        full_route = do_rom_splines(np.asfarray(loaded_route))
        temp_x1, temp_y1, temp_th1 = zip(*full_route)
        reduced_path = reduce_dimensions(np.array([temp_x1, temp_y1, temp_th1]))

        # Adding noise to odo route
        self.x, self.y, self.th = zip(*np.asarray_chkfinite(reduced_path, dtype=np.float64))
        self.xN, self.yN, self.thN = addNoise(self.x, self.y, self.th)
        print("Distance traveled: {:.0f} in meters".format(distance_traveled(self.x,self.y)))

        # GPS on ground truth
        self.x_gps, self.y_gps = g2o.GNSS_reading(self.x, self.y)

        
        self.aarhus = GIS.read_csv(filenamePoints, filenamePoly)
        _, self.rowPoly = self.aarhus.read()
        landmarks = load_from_json(filelandmarks)
        self.landmarks = landmark_noise(landmarks)

    
    def generate_g2o(self, plot: bool = True):

        # Write g2o file
        g2oW = self.writeOdometry(self.xN, self.yN, self.thN, self.landmarks)
        self.do_loop_closure(self.x, self.y, self.th, self.xN, self.yN, self.thN ,self.landmarks, g2oW)

        if plot:
            # Plotting:
            self.aarhus.plot_map()
            self.plot_constraints()
            plt.plot(self.x, self.y, label='Groundtruth')
            plt.plot(self.xN, self.yN, label='Noise route')
            # plt.scatter(np.asarray_chkfinite(self.x_gps), np.asarray_chkfinite(self.y_gps), marker='x', color='red',
            #                                     label='GPS points')
            # plt.scatter(self.x[50], self.y[50], color='red')
            # circle1 = plt.Circle((self.x[50], self.y[50]), 10, fill=False, label='Loop closure range', color='red')
            # plt.gca().add_patch(circle1)
            # plt.legend(loc='upper right')
            plt.ylabel("UTM32 Y")
            plt.xlabel("UTM32 X")
            plt.show()

    def writeOdometry(self, x, y, theta, landmarks):
        g2oW = open('g2o_generator/robosim/data/g2o/noise.g2o', 'w')
        self.n_landmarks = 0
        
        # Landmark id and position 
        idx = 0
        for _, landmark in landmarks.items():
            for pos in landmark:
                if idx < 0:
                    print('VERTEX odometry data is not in correct format')
                    return
                else:
                    l = '{} {} {} {}'.format("VERTEX_XY", idx, pos[0], pos[1])
                    g2oW.write(l)
                    g2oW.write("\n")
                    self.n_landmarks += 1
                    idx += 1
        
        # Odometry id and pose

        for idx, (x_odo, y_odo, th_odo) in enumerate(zip(x, y, theta)):
            if idx < 0:
                print('VERTEX odometry data is not in correct format')
                return
            else:
                l = '{} {} {} {} {}'.format("VERTEX_SE2", idx+self.n_landmarks, x_odo, y_odo, th_odo)
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

            # self.pose_pose.append([p1[0:2], p2[0:2]])

            l = '{} {} {} {} {} {} {}'.format("EDGE_SE2", str((i - 1) + self.n_landmarks), str(i + self.n_landmarks), del_x, del_y, del_th, H)
            g2oW.write(l)
            g2oW.write("\n")

        return g2oW

    def ground_truth(self):
        g2oWG = open('g2o_generator/robosim/data/g2o/ground_truth.g2o', 'w')
        
        for i in range(len(self.x)):
            l = '{} {} {} {} {}'.format("VERTEX_SE2", str(i), self.x[i], self.y[i], self.th[i])
            g2oWG.write(l)
            g2oWG.write("\n")

    def do_loop_closure(self, x, y, th, xN, yN, thN, landmarks, g2oW):       
       
        # Ground truth points
        points =  [[x, y, th] for x, y, th in zip(x, y, th)]

        # Noisy points
        pointsN =  [[xN, yN, thN] for xN, yN, thN in zip(xN, yN, thN)]

        # Keeping track of amount of loop closures
        n_lc_odo = 0
        n_lc_lm = 0

        # Information matrix: Odometry
        H = "100.0 0.0 0.0 100.0 0.0 1000.0" 
        
        # Information matrix: Landmarks
        H_XY = {
                '"tree"':           "50.0 0.0 0.0 50.0 0.0 1000.0",
                "traffic_signals":  "25.0 0.0 0.0 25.0 0.0 700.0",
                "bin":              "10.0 0.0 0.0 10.0 0.0 400.0",
                "bench":            "20.0 0.0 0.0 20.0 0.0 600.0",
                "fountain":         "40.0 0.0 0.0 40.0 0.0 950.0",
                "statue":           "35.0 0.0 0.0 35.0 0.0 900.0",
                "bump":             "25.0 0.0 0.0 25.0 0.0 700.0"
        }

        # Gotten from the mean distance of all points to have to idea of the distance to use in this scale
        lc_range = 10 # 10 meters
        
        for pose_id in range(len(x)):
            
            # Re initializing for every pose_id
            check_pose_odo = 0

            # Odometry constraints
            for j in range(len(x)):
                if  j < pose_id and abs(pose_id - j) > 8:

                    d_odo = np.linalg.norm(np.array(points[pose_id][0:2], dtype=np.float64) - np.array(points[j][0:2], dtype=np.float64))
                    if np.greater(lc_range, d_odo):

                        n_lc_odo += 1
                        edge_lc_odo = (pose_id+1, check_pose_odo)

                        # Saving pose_pose for visualization
                        self.pose_pose.append([pointsN[check_pose_odo][0:2], pointsN[pose_id][0:2]])

                        lc_constraint_odo = self.generate_lc_constraint(np.array([[xN[check_pose_odo]], [yN[check_pose_odo]], [thN[check_pose_odo]]]), 
                                                                        np.array([[xN[pose_id]], [yN[pose_id]], [thN[pose_id]]]))

                        l1 = '{} {} {} {} {} {} {}'.format("EDGE_SE2", edge_lc_odo[0]+self.n_landmarks, edge_lc_odo[1]+self.n_landmarks, 
                                                        lc_constraint_odo[0,0], lc_constraint_odo[1,0], lc_constraint_odo[2,0], 
                                                        H)
                        g2oW.write(l1)
                        g2oW.write('\n')

                check_pose_odo += 1

            # Re initialization
            check_pose_lm = 0

            # Writing landmark constraints
            for key, landmark in landmarks.items():
                for pos in landmark:
                    
                    # Checking for loop closure range
                    d_landmark = np.linalg.norm(np.array(points[pose_id][0:2], dtype=np.float64) - np.array(pos, dtype=np.float64))
                    if np.greater(lc_range, d_landmark):
                        
                        n_lc_lm += 1
                        edge_lc_lm = (pose_id+1, check_pose_lm)
                        
                        # Saving pose_landmark for visualization
                        self.pose_landmark.append([pointsN[pose_id][0:2], pos])
                        
                        # Bearing to landmark
                        bearing = calc_bearing(x[pose_id], y[pose_id], pos[0], pos[1])
                        lc_constraint_lm = self.generate_lc_constraint(np.array([[xN[pose_id]], [yN[pose_id]], [thN[pose_id]]]),
                                                                       np.array([[pos[0]], [pos[1]], [bearing]]))
 
                        l2 = '{} {} {} {} {} {} {}'.format("EDGE_SE2_XY", edge_lc_lm[0]+self.n_landmarks, edge_lc_lm[1], 
                                                        lc_constraint_lm[0,0], lc_constraint_lm[1,0], lc_constraint_lm[2,0],
                                                        H_XY[key])
                        g2oW.write(l2)                   
                        g2oW.write('\n')

                    check_pose_lm += 1

        print('Number of loop closure (odometry): {}'.format(n_lc_odo))
        print('Number of loop closure (landmarks): {}'.format(n_lc_lm))

    def generate_lc_constraint(self, x1, x2):
        """Current pose (x1) and next pose (x2), transforming to robot frame

        Args:
            x1 (numpy matrix): x,y,th of current pose
            x2 (numpy matrix): x,y,th of the next pose

        Returns:
            [numpy matrix]: [relative distance of the two poses x,y,th]
        """        
        delta_x = x2[0:2,0] - x1[0:2,0]
        R = np.transpose(np.matrix([[cos(x1[2]), -sin(x1[2])], [sin(x1[2]), cos(x1[2])]]))
        delta_x_l = R @ delta_x
        d_theta = min_theta(x2[2] - x1[2])

        return np.array([[delta_x_l[0,0]], [delta_x_l[0,1]], [d_theta[0]]])

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

    @staticmethod
    def vec2trans(p):
        T = np.array([[cos(p[2]), -sin(p[2]), p[0]], 
                      [sin(p[2]),  cos(p[2]), p[1]], 
                      [0, 0, 1]])
        return T

    def from_utm(self, lat1, lat2, lon1, lon2):
        
        x1, y1, _, _ = utm.from_latlon(lat1, lon1)
        x2, y2, _, _ = utm.from_latlon(lat2, lon2)

        return np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
    
    def to_utm(self, x1, x2, y1, y2):

        lon1, lat1 = utm.to_latlon(x1, y1, 32, 'U')
        lon2, lat2 = utm.to_latlon(x2, y2, 32, 'U')

        return np.linalg.norm(np.array([lon1, lat1]) - np.array([lon2, lat2]))

    def plot_constraints(self):
    
        # pose_landmark
        pose_landmark = np.vstack(self.pose_landmark)
        
        pose_landmarkx = pose_landmark[:,0]
        pose_landmarky = pose_landmark[:,1]
        
        plx = np.vstack([pose_landmarkx[0::2], pose_landmarkx[1::2]])
        ply = np.vstack([pose_landmarky[0::2], pose_landmarky[1::2]])
        
        plt.plot(plx, ply, color='black')


        # pose_pose
        pose_pose = np.vstack(self.pose_pose)
        
        pose_posex = pose_pose[:,0]
        pose_posey = pose_pose[:,1]
        
        ppx = np.vstack([pose_posex[0::2], pose_posex[1::2]])
        ppy = np.vstack([pose_posey[0::2], pose_posey[1::2]])
        
        plt.plot(ppx, ppy, color='red')
        

if __name__ == '__main__':

    filenamePoints = 'g2o_generator/GIS_Extraction/data/aarhus_features_v2.csv'
    filenamePoly = 'g2o_generator/GIS_Extraction/data/aarhus_polygons_v2.csv'
    filelandmarks = 'g2o_generator/GIS_Extraction/landmarks/landmarks_w_types.json'
    odometry_file = './g2o_generator/robosim/data/robopath/Aarhus_path1.json'
    genG2O = g2o(odometry_file, filenamePoints, filenamePoly, filelandmarks)
    genG2O.ground_truth()
    # genG2O.plot_constraints()
    genG2O.generate_g2o(plot=True)


    