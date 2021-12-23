import numpy as np
import matplotlib.pyplot as plt
import random
import time
import shapely.geometry
import utm
from math import *

import sys
sys.path.append('g2o_generator/GIS_Extraction')

# local import
from lib.utility import *
from lib.geometric_utility import p_intersection
import csv_reader as GIS

class g2o:
    def __init__(self, odometry_file, filenamePoints, filenamePoly, filelandmarks, lm_lc_range: float=15, odo_lc_range: float=2, fov: float=45):

        self.pose_pose, self.pose_landmark = [], []
        self.pose_pose_outliers, self.pose_landmark_outlier = [], []
        self.lm_lc_range = lm_lc_range
        self.odo_lc_range = odo_lc_range
        self.lm_lut = {}
        self.landmark_arrow = []

        # This will only get the bearing to landmarks and not 
        # initialize the g2o file with landmark locations
        self.bearing_only = True
        self.FOV = np.deg2rad(fov) # Camera field of view of the robot

        self.lm_thresh = 3
        self.lm_lc_prob = 0.93
        self.GNSS_freq = 5

        self.time_stamp = time.strftime("%Y%m%d-%H%M%S")

        # Corrupt the dataset with outliers
        self.corrupt_dataset = False
        self.n_outliers = 100
        self.outlier_type = {"random": 1,           # Connect any two randomly sampled nodes in the graph
                             "local": 2,            # Conenct random nodes that ar ein the vincinity of each other
                             "random_grouped": 3,   
                             "local_grouped": 4,
                             "none": -1}

        std_gnss_x = 0.7; std_gnss_y = 0.7
        std_odo_x = 0.1; std_odo_y = 0.2; std_odo_th = deg2rad(0.1); mu_bias = 0.8
        std_lm_x = 0.7; std_lm_y = 0.7; self.std_lm_th = deg2rad(1)

        # Information matrices: 
        # Odometry
        # GNSS
        # Landmarks(Specific for each landmark type)
        self.H_odo = np.linalg.inv(np.array([[std_odo_x**2, 0, 0],
                                             [0, std_odo_y**2, 0],
                                             [0, 0, 0.08**2]]))
        self.H_gnss = np.linalg.inv(np.array([[std_gnss_x**2, 0],
                                              [0, std_gnss_y**2]]))
        self.H_xy = np.linalg.inv(np.array([[std_lm_x**2, 0, 0],
                                            [0, std_lm_y**2, 0],
                                            [0, 0, 0.03**2]]))
        # self.H_xy = np.linalg.inv(np.array([[std_lm_x**2, 0, 0],
        #                                     [0, std_lm_y**2, 0],
        #                                     [0, 0, np.sin(self.std_lm_th**2)]]))

        # Loading and extracting save odometry route
        odometry = load_from_json(odometry_file)
        temp_x = np.asfarray(odometry[0]); temp_y = np.asfarray(odometry[1]); temp_th = np.asfarray(odometry[2])
        loaded_route = [[pose_x, pose_y, pose_th] for pose_x, pose_y, pose_th in zip(temp_x, temp_y, temp_th)]

        temp_x1, temp_y1, temp_th1 = zip(*loaded_route)
        reduced_path = reduce_dimensions(np.array([temp_x1, temp_y1, temp_th1]), 'half')

        # Adding noise to odo route
        self.x, self.y, self.th = zip(*np.asarray_chkfinite(reduced_path, dtype=np.float64))
        self.xN, self.yN, self.thN = addNoise(self.x, self.y, self.th, std_odo_x, std_odo_y, std_odo_th, mu_bias)
        

        print("Distance traveled: {:.0f} m".format(distance_traveled(self.x,self.y)))

        # GPS on ground truth
        self.x_gnss, self.y_gnss, self.gt_x_gnss, self.gt_y_gnss, self.n_gps = g2o.GNSS_reading(self.x, self.y, self.GNSS_freq, std_gnss_x, std_gnss_y)

        if self.bearing_only:
            self.n_gps = 0
        
        self.aarhus = GIS.read_csv(filenamePoints, filenamePoly)
        _, self.rowPoly = self.aarhus.read()
        self.cascaded_poly = self.aarhus.squeeze_polygons(self.rowPoly)

        loaded_landmarks = load_from_json(filelandmarks)
        self.gt_landmarks = loaded_landmarks
        self.landmarks = add_landmark_noise(loaded_landmarks, std_lm_x, std_lm_y)

    
    def generate_g2o(self, corruption_type: str, plot: bool=False, plot_outliers: bool=False, plot_constraints: bool=False, plot_robot_heading: bool=False):

        # Write g2o file
        g2oW = self.write_basics(self.x, self.y, self.th, self.xN, self.yN, self.thN, self.gt_landmarks, self.landmarks)
        self.ground_truth(self.x, self.y, self.th, self.gt_landmarks)
        self.do_loop_closure(self.x, self.y, self.th, self.xN, self.yN, self.thN, g2oW, corruption_type)
       
        lm_x, lm_y, lm_th = zip(*self.landmark_arrow)
        
        if plot:
            
            # print("RMSE: {:.2f} \t|\t MAE: {:.2f} \t|\t ATE: {:.6f} \t|\t ALE {:.2f}".format(\
            #                                 RMSE(np.array([self.xN, self.yN]), np.array([self.x, self.y])), \
            #                                 MAE(np.array([self.xN, self.yN]), np.array([self.x, self.y])), \
            #                                 ATE(np.array([self.xN, self.yN]), np.array([self.x, self.y])), \
            #                                 ALE(self.landmarks, self.gt_landmarks)))
            
            
            self.aarhus.plot_map(self.landmarks, show=False)
            if plot_constraints:
                self.plot_constraints()

            if plot_outliers:
                plt.plot(self.xN, self.yN, label='Noise route')
                self.plot_outliers()
                plot_outliers_vs_normal(self.pose_pose_outliers, self.pose_landmark_outlier, self.pose_pose, self.pose_landmark)

            if plot_robot_heading:
                # robot_heading(self.x, self.y, self.th, color="blue", length=1)
                robot_heading(self.xN, self.yN, self.thN, color="gray", length=1)
                robot_heading(lm_x, lm_y, lm_th, color="green", length=0.01, alpha=0.5)

            plt.plot(self.x, self.y, label='Groundtruth')
            plt.plot(self.xN, self.yN, label='Noise route')
            # plt.scatter(np.asarray_chkfinite(self.x_gnss), np.asarray_chkfinite(self.y_gnss), marker='x', color='red',
                                                # label='GPS points')
            # plt.scatter(self.x[50], self.y[50], color='red')
            # circle1 = plt.Circle((self.x[50], self.y[50]), 10, fill=False, label='Loop closure range', color='red')
            # plt.gca().add_patch(circle1)
            plt.legend(loc='upper right')
            # legend = plt.legend()
            # legend.remove()
            plt.ylabel("UTM32 Y")
            plt.xlabel("UTM32 X")
            plt.show()


    def write_basics(self, x, y, theta, xN, yN, thetaN, gt_landmarks, N_landmarks):

        g2oW = open('g2o_generator/robosim/data/g2o/noise_'+self.time_stamp+'.g2o', 'w')
        self.n_landmarks = 0

        self.xscale = self.x[0]
        self.yscale = self.y[0]
        self.th_scale = self.th[0]

        # If a landmark has been seen before, no not append. Also saving landmarks in a look up table
        lm_hist = []
        for pose_id in range(len(self.x)):

            curr_pose = np.array([[x[pose_id]], [y[pose_id]], [theta[pose_id]]])
                                                            # ground truth       noisy landmarks
            for (key, gt_landmark), (_, N_landmark)in zip(gt_landmarks.items(), gt_landmarks.items()):
                for (pos, posN) in zip(gt_landmark, N_landmark):

                    d_landmark = np.linalg.norm(curr_pose[0:2,:].T - np.array(pos, dtype=np.float64))
                    
                    if np.greater(self.lm_lc_range, d_landmark):
                        
                        bearing = calc_bearing(x[pose_id], y[pose_id], pos[0], pos[1])
                        other_pose = np.array([[pos[0]], [pos[1]], [bearing]])

                        lc_constraint_lm = self.generate_lc_constraint(curr_pose, other_pose)
                        
                        line = shapely.geometry.LineString([[x[pose_id], y[pose_id]], [pos[0], pos[1]]])
                        obstruction = p_intersection(line, self.cascaded_poly)

                        if -self.FOV <= lc_constraint_lm[2,0] <= self.FOV and obstruction:
                           
                            if pos in lm_hist:
                                continue

                            else:
                                l = '{} {} {} {}'.format("VERTEX_XY", self.n_landmarks, posN[0]-self.xscale, posN[1]-self.yscale)
                                g2oW.write(l)
                                g2oW.write("\n")
                                self.n_landmarks += 1
                                lm_hist.append(pos)
                                self.lm_lut[self.n_landmarks, key] = pos

        if not self.bearing_only:
            # GPS id and position (in UTM32)
            for idx, (x_gnss, y_gnss) in enumerate(zip(self.x_gnss, self.y_gnss)):
                if idx < 0:
                    return print('VERTEX GPS data is not in correct format')

                else:
                    l = '{} {} {} {}'.format("VERTEX_GPS", idx+self.n_landmarks, x_gnss-self.xscale, y_gnss-self.yscale)
                    g2oW.write(l)
                    g2oW.write("\n")

        # Odometry id and pose
        for idx, (x_odo, y_odo, th_odo) in enumerate(zip(xN, yN, thetaN)):
            if idx < 0:
                print('VERTEX odometry data is not in correct format')
                return
            else:
                l = '{} {} {} {} {}'.format("VERTEX_SE2", idx+self.n_landmarks+self.n_gps, x_odo-self.xscale, y_odo-self.yscale, th_odo)
                g2oW.write(l)
                g2oW.write("\n")

        # Odometry constraints for consecutive poses
        for i in range(1, len(x)):
            p1 = (xN[i-1], yN[i-1], thetaN[i-1])
            p2 = (xN[i], yN[i], thetaN[i])

            T1_w = g2o.vec2trans(p1)
            T2_w = g2o.vec2trans(p2)
            T2_1 = np.linalg.inv(T1_w) @ T2_w

            del_x = str(T2_1[0][2])
            del_y = str(T2_1[1][2])
            del_th = str(atan2(T2_1[1, 0], T2_1[0, 0]))

            l = '{} {} {} {} {} {} {} {} {} {} {} {}'.format("EDGE_SE2", str((i-1) + self.n_landmarks+self.n_gps), str(i+self.n_landmarks+self.n_gps), 
                                              del_x, del_y, del_th, 
                                              self.H_odo[0,0], self.H_odo[0,1], self.H_odo[0,2], self.H_odo[1,1], self.H_odo[1,2], self.H_odo[2,2])
            g2oW.write(l)
            g2oW.write("\n")

        if not self.bearing_only:
            # GPS constraints
            idx_gps = 0
            rob_gnss_hist = np.zeros(self.n_gps)
      
            for pose_id in range(len(self.x)-1):
                if pose_id % self.GNSS_freq == 0:
        
                    pos_gt = np.array([[x[pose_id]], [y[pose_id]], [theta[pose_id]]])
                    pos_gnss = np.array([[self.x_gnss[idx_gps]], [self.y_gnss[idx_gps]]])
                    delta_d = pos_gnss[0:2,0] - pos_gt[0:2,0]

                    # Distance from robot to GPS
                    rob_gnss_dist = np.linalg.norm(delta_d)
                    rob_gnss_hist[idx_gps] = rob_gnss_dist

                    rot_R = np.transpose(np.matrix([[cos(pos_gt[2]), -sin(pos_gt[2])], [sin(pos_gt[2]), cos(pos_gt[2])]]))
                    delta_d_l = rot_R @ delta_d
                    pos_gnss_rel = np.array([[delta_d_l[0,0], delta_d_l[0,1]]])
                
                    l = '{} {} {} {} {} {} {} {}'.format("EDGE_SE2_GPS", pose_id+1+self.n_landmarks+self.n_gps, idx_gps+self.n_landmarks, 
                                                    pos_gnss_rel[0][0], pos_gnss_rel[0][1], self.H_gnss[0,0], self.H_gnss[0,1], self.H_gnss[1,1])
                    g2oW.write(l)                   
                    g2oW.write('\n')
                    
                    idx_gps += 1

        # print("Mean error in GPS: {:.3} m".format(np.mean(rob_gnss_hist)))

        return g2oW


    def ground_truth(self, x, y, theta, landmarks):
        g2oWG = open('g2o_generator/robosim/data/g2o/ground_truth_'+self.time_stamp+'.g2o', 'w')
        n_landmarks = 0
        n_gps = 0

        # If a landmark has been seen before, no not append. Also saving landmarks in a look up table
        lm_hist = []
        for pose_id in range(len(self.x)):
            
            curr_pose = np.array([[x[pose_id]], [y[pose_id]], [theta[pose_id]]])

            for _, landmark in landmarks.items():
                for pos in landmark:
                    
                    d_landmark = np.linalg.norm(curr_pose[0:2,:].T - np.array(pos, dtype=np.float64))
                    
                    if np.greater(self.lm_lc_range, d_landmark):

                        bearing = calc_bearing(x[pose_id], y[pose_id], pos[0], pos[1])
                        other_pose = np.array([[pos[0]], [pos[1]], [bearing]])

                        lc_constraint_lm = self.generate_lc_constraint(curr_pose, other_pose)

                        line = shapely.geometry.LineString([[x[pose_id], y[pose_id]], [pos[0], pos[1]]])
                        obstruction = p_intersection(line, self.cascaded_poly)

                        if -self.FOV <= lc_constraint_lm[2,0] <= self.FOV and obstruction:

                            if pos in lm_hist:
                                continue

                            else:
                                l = '{} {} {} {}'.format("VERTEX_XY", n_landmarks, pos[0]-self.xscale, pos[1]-self.yscale)
                                g2oWG.write(l)
                                g2oWG.write("\n")
                                n_landmarks += 1
                                lm_hist.append(pos)

        if not self.bearing_only:
            # GPS id and position (in UTM32)
            for idx, (gt_x_gnss, gt_y_gnss) in enumerate(zip(self.gt_x_gnss, self.gt_y_gnss)):
                l = '{} {} {} {}'.format("VERTEX_GPS", str(idx+n_landmarks), gt_x_gnss-self.x[0], gt_y_gnss-self.y[0])
                n_gps += 1

                g2oWG.write(l)
                g2oWG.write("\n")

        for i in range(len(self.x)):
            l = '{} {} {} {} {}'.format("VERTEX_SE2", str(i+n_landmarks+n_gps), self.x[i]-self.x[0], self.y[i]-self.y[0], self.th[i])

            g2oWG.write(l)
            g2oWG.write("\n")

        for pose_id in range(len(self.x)-1):
            
            curr_pose = np.array([[x[pose_id]], [y[pose_id]], [theta[pose_id]]])
           
            # Writing landmark constraints
            for (lm_id, _), pos in self.lm_lut.items():
                
                # Checking for loop closure range
                d_landmark = np.linalg.norm(curr_pose[0:2,:].T - np.array(pos, dtype=np.float64))

                # Checking distance criteria, pose-landmarks
                if np.greater(self.lm_lc_range, d_landmark):

                    offset_lm = self.n_landmarks+self.n_gps
                    edge_lc_lm = (pose_id+1, lm_id-offset_lm-1)

                    # Bearing
                    bearing = calc_bearing(x[pose_id], y[pose_id], pos[0], pos[1])
                    other_pose = np.array([[pos[0]], [pos[1]], [bearing]]) 
                    lc_bearing = self.generate_lc_constraint(curr_pose, other_pose) 

                    # Shapely linestring for intersection
                    line = shapely.geometry.LineString([[x[pose_id], y[pose_id]], [pos[0], pos[1]]])
                    obstruction = p_intersection(line, self.cascaded_poly)

                    # Robot camera FOV AND calculating vision intersection with polygons
                    if -self.FOV <= lc_bearing[2,0] <= self.FOV and obstruction:
                        
                        # Adding noise to bearing
                        other_pose = np.array([[pos[0]], [pos[1]], [bearing]]) # World coordinate
                        lc_constraint_lm = self.generate_lc_constraint(curr_pose, other_pose) # Calc constraint to robot coordinate

                        # print("Writing bearing only wrt landmarks")
                        l = '{} {} {} {} {}'.format("EDGE_SE2_BEARING", 
                                                    edge_lc_lm[0]+offset_lm, edge_lc_lm[1]+offset_lm, 
                                                    lc_constraint_lm[2,0],
                                                    self.H_xy[2,2])
                        g2oWG.write(l)
                        g2oWG.write('\n')



    def do_loop_closure(self, x, y, th, xN, yN, thN, g2oW, ct):       
       
        # Ground truth points; Noisy points
        points =  [[x, y, th] for x, y, th in zip(x, y, th)]
        pointsN =  [[xN, yN, thN] for xN, yN, thN in zip(xN, yN, thN)]
        
        # Keeping track of amount of loop closures
        n_lc_odo = 0
        n_lc_lm = 0

        check_n_outliers = 0
        check_n_outliers_lm = 0
        offset_odo = self.n_landmarks+self.n_gps

        # Searching through poses looking for constraints
        path_size = len(x)
        for pose_id in range(path_size-1):
            
            curr_pose = np.array([[x[pose_id]], [y[pose_id]], [th[pose_id]]])

            rand_poseID = random.randint(0, path_size-1)
            random_pose = np.array([[x[rand_poseID]], [y[rand_poseID]], [th[rand_poseID]]])

            # Re initializing for every pose_id
            check_pose_odo = 0
            if not self.bearing_only:
                # Odometry constraints
                for j in range(path_size-1):
                    if  j < pose_id and abs(pose_id - j) > 8:
                        
                        # Checking for loop closure range pose-pose
                        d_odo = np.linalg.norm(curr_pose[0:2,:].T - np.array(points[j][0:2], dtype=np.float64))
                        
                        if np.greater(self.odo_lc_range, d_odo):
                            
                            n_lc_odo += 1

                            # Saving pose_pose for visualization
                            self.pose_pose.append([pointsN[check_pose_odo][0:2], pointsN[pose_id][0:2]])

                            # Computing the relative distance between two poses in close vincinity (Simulating ICP)
                            edge_lc_odo = (pose_id+1, check_pose_odo)
                            other_pose = np.array([[x[check_pose_odo]], [y[check_pose_odo]], [th[check_pose_odo]]])
                            lc_constraint_odo = self.generate_lc_constraint(curr_pose, other_pose)                

                            self.write_loop_constraints(g2oW, "EDGE_SE2", offset_odo, edge_lc_odo, lc_constraint_odo, self.H_odo)

                    check_pose_odo += 1

                # CORRUPTING DATA WITH OUTLIERS
                if self.corrupt_dataset and check_n_outliers <= self.n_outliers:
                
                    if self.outlier_type[ct] == 1 or self.outlier_type[ct] == 3:
                        cluster = 1 if self.outlier_type[ct] == 1 or self.outlier_type[ct] == 3 else 7 # http://www2.informatik.uni-freiburg.de/~spinello/agarwalICRA13.pdf page 4
                        
                        for _ in range(cluster):
                            # Random pose-pose loop closure (OUTLIER)
                            # Selecting random node from the dataset
                            while True:
                                rand_node = random.randint(0, path_size-1)
                                if rand_node != rand_poseID:
                                    break 

                            edge_lc_odo = (rand_poseID, rand_node)
                            
                            other_pose = np.array([[x[rand_node]], [y[rand_node]], [th[rand_node]]])
                            lc_constraint_odo = self.generate_lc_constraint(random_pose, other_pose)

                            self.pose_pose_outliers.append([pointsN[rand_poseID][0:2], pointsN[rand_node][0:2]])
                            self.write_loop_constraints(g2oW, "EDGE_SE2", offset_odo, edge_lc_odo, lc_constraint_odo, self.H_odo)
                        
                        check_n_outliers += 1

                    elif self.outlier_type[ct] == 2 or self.outlier_type[ct] == 4:
                        cluster = 1 if self.outlier_type[ct] == 1 or self.outlier_type[ct] == 3 else 7 # http://www2.informatik.uni-freiburg.de/~spinello/agarwalICRA13.pdf page 4

                        for _ in range(cluster):
                            # Local pose-pose loop closure (OUTLIER)
                            while True:
                                local_node = random.randint(-15,15) + rand_poseID
                                if 0 <= local_node <= path_size-1 and local_node != rand_poseID:
                                    break

                            edge_lc_odo = (rand_poseID, local_node)
                            vicinity_pose = np.array([[x[local_node]], [y[local_node]], [th[local_node]]])
                            lc_constraint_odo = self.generate_lc_constraint(random_pose, vicinity_pose)

                            self.pose_pose_outliers.append([pointsN[rand_poseID][0:2], pointsN[local_node][0:2]])
                            self.write_loop_constraints(g2oW, "EDGE_SE2", offset_odo, edge_lc_odo, lc_constraint_odo, self.H_odo)

                        check_n_outliers += 1
                    else:
                        pass

            # Writing landmark constraints
            for (lm_id, _), pos in self.lm_lut.items():

                # Checking for loop closure range
                d_landmark = np.linalg.norm(curr_pose[0:2,:].T - np.array(pos, dtype=np.float64))

                # Checking distance criteria, pose-landmarks
                if np.greater(self.lm_lc_range, d_landmark):

                    offset_lm = self.n_landmarks+self.n_gps
                    edge_lc_lm = (pose_id+1, lm_id-offset_lm-1)

                    # Bearing
                    bearing = calc_bearing(x[pose_id], y[pose_id], pos[0], pos[1])
                    other_pose = np.array([[pos[0]], [pos[1]], [bearing]])
                    lc_bearing = self.generate_lc_constraint(curr_pose, other_pose) 

                    # Shapely linestring for intersection
                    line = shapely.geometry.LineString([[x[pose_id], y[pose_id]], [pos[0], pos[1]]])
                    obstruction = p_intersection(line, self.cascaded_poly)

                    # Robot camera FOV AND calculating vision intersection with polygons
                    if -self.FOV <= lc_bearing[2,0] <= self.FOV and obstruction:
                        
                        # Adding noise to bearing
                        noisy_bearing = add_bearing_noise(bearing, systematic_lm_noise=deg2rad(0), std_lm_bearing=self.std_lm_th)
                        other_pose = np.array([[pos[0]], [pos[1]], [noisy_bearing]]) # World coordinate
                        lc_constraint_lm = self.generate_lc_constraint(curr_pose, other_pose) # Calc constraint to robot coordinate

                        # Saving pose_landmark for visualization
                        self.pose_landmark.append([points[pose_id][0:2], pos])
                        self.landmark_arrow.append([pointsN[pose_id][0], pointsN[pose_id][1], noisy_bearing])

                        if self.bearing_only:
                            # print("Writing bearing only wrt landmarks")
                            l = '{} {} {} {} {}'.format("EDGE_SE2_BEARING", 
                                                        edge_lc_lm[0]+offset_lm, edge_lc_lm[1]+offset_lm, 
                                                        lc_constraint_lm[2,0],
                                                        self.H_xy[2,2])
                            g2oW.write(l)                   
                            g2oW.write('\n')
                        else:
                            self.write_loop_constraints(g2oW, "EDGE_SE2_XY", offset_lm, edge_lc_lm, lc_constraint_lm, self.H_xy)

                        n_lc_lm += 1

                        # CORRUPTING LANDMARK DATA
                        if self.corrupt_dataset and check_n_outliers_lm <= self.n_outliers and random.random() < 0.1:
                            
                            cluster = 1 if self.outlier_type[ct] == 1 or self.outlier_type[ct] == 3 else 7

                            for _ in range(cluster):
                                rand_landmarkID = random.randint(0, path_size-1)
                                edge_lc_lm = (rand_landmarkID, lm_id-1)

                                self.pose_landmark_outlier.append([pointsN[rand_landmarkID][0:2], pos])

                                lc_constraint_lm = self.generate_lc_constraint(random_pose, other_pose)
                                self.write_loop_constraints(g2oW, "EDGE_SE2_XY", offset_lm, edge_lc_lm, lc_constraint_lm, self.H_xy)

                            check_n_outliers_lm += 1


        
        print('Odometry loop closure: {}\t outliers: {}'.format(n_lc_odo, len(self.pose_pose_outliers)))
        print('Landmark loop closure: {}\t outliers: {}'.format(n_lc_lm, len(self.pose_landmark_outlier)))

    def write_loop_constraints(self, g2o, edge_type, offset, edge_lc, lc_constraint, H):

        l = '{} {} {} {} {} {} {} {} {} {} {} {}'.format(edge_type, edge_lc[0]+offset, edge_lc[1]+offset, 
                                                         lc_constraint[0,0], lc_constraint[1,0], lc_constraint[2,0],
                                                         H[0,0], H[0,1], H[0,2], H[1,1], H[1,2], H[2,2])
        g2o.write(l)                   
        g2o.write('\n')


    @staticmethod
    def generate_lc_constraint(x1, x2):
        """Current pose (x1) and next pose (x2), transforming to robot frame and calculating delta x,y,th

        Args:
            x1 (numpy matrix): x,y,th of current pose
            x2 (numpy matrix): x,y,th of the next pose

        Returns:
            [numpy matrix]: [relative distance of the two poses x,y,th]
        """        
        delta_x = x2[0:2,0] - x1[0:2,0]
        R = np.transpose(np.matrix([[cos(x1[2]), -sin(x1[2])], [sin(x1[2]), cos(x1[2])]]))
        delta_x_l = R @ delta_x
        d_theta = min_theta(x1[2], x2[2])
        
        return np.array([[delta_x_l[0,0]], [delta_x_l[0,1]], [d_theta[0]]])


    @staticmethod
    def GNSS_reading(x_odo, y_odo, frequency: int=5, std_gps_x: float=0.33, std_gps_y: float=0.33):
        x_gps, y_gps = [], []
        gt_x_gps, gt_y_gps = [], []

        for i in range(len(x_odo)):
            if (i % frequency == 0):
                
                x_gpsN, y_gpsN = add_GNSS_noise(x_odo[i], y_odo[i], std_gps_x, std_gps_y)

                x_gps.append(x_gpsN); y_gps.append(y_gpsN)
                gt_x_gps.append(x_odo[i]); gt_y_gps.append(y_odo[i])

        print('Added {} GPS readings'.format(len(x_gps)))
        return x_gps, y_gps, gt_x_gps, gt_y_gps, len(x_gps)


    @staticmethod
    def vec2trans(p):
        T = np.array([[cos(p[2]), -sin(p[2]), p[0]], 
                      [sin(p[2]),  cos(p[2]), p[1]], 
                      [0, 0, 1]])
        return T


    @staticmethod
    def from_utm(lat1, lat2, lon1, lon2):
        
        x1, y1, _, _ = utm.from_latlon(lat1, lon1)
        x2, y2, _, _ = utm.from_latlon(lat2, lon2)

        return np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
    

    @staticmethod
    def to_utm(x1, x2, y1, y2):

        lon1, lat1 = utm.to_latlon(x1, y1, 32, 'U')
        lon2, lat2 = utm.to_latlon(x2, y2, 32, 'U')

        return np.linalg.norm(np.array([lon1, lat1]) - np.array([lon2, lat2]))

    def plot_outliers(self):
        
        # if len(self.pose_pose_outliers) > 1:
        #     pose_pose_outliers = np.vstack(self.pose_pose_outliers)
            
        #     pose_outliersx = pose_pose_outliers[:,0]
        #     pose_outliersy = pose_pose_outliers[:,1]
            
        #     pox = np.vstack([pose_outliersx[0::2], pose_outliersx[1::2]])
        #     poy = np.vstack([pose_outliersy[0::2], pose_outliersy[1::2]])
        
        #     plt.plot(pox, poy, color='gray', alpha=0.3)
        # else:
        #     print("No pose pose outlier loop closures")

        if len(self.pose_landmark_outlier) > 1:
            pose_landmark_outlier = np.vstack(self.pose_landmark_outlier)
            
            landmark_outliersx = pose_landmark_outlier[:,0]
            landmark_outliersy = pose_landmark_outlier[:,1]
            
            plx = np.vstack([landmark_outliersx[0::2], landmark_outliersx[1::2]])
            ply = np.vstack([landmark_outliersy[0::2], landmark_outliersy[1::2]])
        
            plt.plot(plx, ply, color='gray', alpha=0.3)
        else:
            print("No pose pose outlier loop closures")

    def plot_constraints(self):
        
        if len(self.pose_landmark) > 1:
            # pose_landmark
            pose_landmark = np.vstack(self.pose_landmark)
            
            pose_landmarkx = pose_landmark[:,0]
            pose_landmarky = pose_landmark[:,1]
            
            plx = np.vstack([pose_landmarkx[0::2], pose_landmarkx[1::2]])
            ply = np.vstack([pose_landmarky[0::2], pose_landmarky[1::2]])
            
            plt.plot(plx, ply, color='purple')
        else:
            print("No pose landmark loop closures")

        # if len(self.pose_pose) > 1:
        #     # pose_pose
        #     pose_pose = np.vstack(self.pose_pose)
            
        #     pose_posex = pose_pose[:,0]
        #     pose_posey = pose_pose[:,1]
            
        #     ppx = np.vstack([pose_posex[0::2], pose_posex[1::2]])
        #     ppy = np.vstack([pose_posey[0::2], pose_posey[1::2]])
            
        #     plt.plot(ppx, ppy, color='purple')
        # else:
        #     print("No pose pose loop closures")



if __name__ == '__main__':

    filenamePoints = 'g2o_generator/GIS_Extraction/data/aarhus_features_v2.csv'
    filenamePoly = 'g2o_generator/GIS_Extraction/data/aarhus_polygons_v2.csv'
    filelandmarks = 'g2o_generator/GIS_Extraction/landmarks/landmarks_w_types.json'
    odometry_file = './g2o_generator/robosim/data/robopath/Aarhus_path1.json'
    genG2O = g2o(odometry_file, filenamePoints, filenamePoly, filelandmarks, 15, 2)
    genG2O.generate_g2o(corruption_type='random', plot=True, plot_outliers=True, plot_constraints=True, plot_robot_heading=True)


    