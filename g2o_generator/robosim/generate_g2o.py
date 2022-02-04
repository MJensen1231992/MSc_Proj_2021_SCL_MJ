import numpy as np
import matplotlib.pyplot as plt
import random
import time
import shapely.geometry
from math import *

import sys
sys.path.append('g2o_generator/GIS_Extraction')

# local import
from lib.utility import *
from lib.helpers import *
from lib.geometric_utility import p_intersection
import csv_reader as GIS

class g2o:
    def __init__(self, odometry_file_gt, odometry_file_noise, filenamePoints, filenamePoly, filelandmarks,\
                landmark_only: bool, bearing_only: bool, pose_only: bool, corrupt: bool,\
                lm_lc_range: float=15, odo_lc_range: float=2, fov: float=45, n_outliers: float=50):

        self.pose_pose, self.pose_landmark = [], []
        self.pose_pose_outliers, self.pose_landmark_outlier = [], []
        self.lm_lc_range = lm_lc_range
        self.odo_lc_range = odo_lc_range
        self.lm_lut = {}
        self.landmark_arrow = []

        # This will only get the bearing to landmarks and not 
        # initialize the g2o file with landmark locations
        self.bearing_only = bearing_only
        self.FOV = np.deg2rad(fov) # Camera field of view of the robot

        # Activate or deactivate only pose-landmark constraints
        self.landmark_only = landmark_only

        # Activate or deactivate only pose-pose constraints
        self.pose_only = pose_only

        self.lm_thresh = 3
        self.lm_lc_prob = 0.93
        self.GNSS_freq = 5

        self.time_stamp = time.strftime("%Y%m%d-%H%M%S")

        # Corrupt the dataset with outliers
        self.corrupt_dataset = corrupt
        self.n_outliers = n_outliers
        self.outlier_type = {"random": 1,           # Connect any two randomly sampled nodes in the graph
                             "local": 2,            # Conenct random nodes that ar ein the vincinity of each other
                             "random_grouped": 3,   
                             "local_grouped": 4,
                             "none": -1}

        # gnss variance
        std_gnss_x = 0.7; std_gnss_y = 0.7
        # odo variance
        std_odo_x = 0.1; std_odo_y = 0.1; std_odo_th = deg2rad(1); mu_bias = 0.1
        # lm variance
        self.std_lm_x = 0.5; self.std_lm_y = 0.5; 
        # bearing variance
        self.std_lm_th = deg2rad(2)

        # Information matrices: 
        # Odometry
        self.H_odo = np.linalg.inv(np.array([[std_odo_x**2, 0, 0],
                                             [0, std_odo_y**2, 0],
                                             [0, 0, std_odo_th**2]]))
        # GNSS
        self.H_gnss = np.linalg.inv(np.array([[std_gnss_x**2, 0],
                                              [0, std_gnss_y**2]]))
        # Landmarks(Specific for each landmark type)
        self.H_xy = np.linalg.inv(np.array([[self.std_lm_x**2, 0, 0],
                                            [0, self.std_lm_y**2, 0],
                                            [0, 0, self.std_lm_th**2]]))
        # ICP
        self.H_icp = np.linalg.inv(np.array([[std_odo_x/10**2, 0, 0],
                                             [0, std_odo_y/10**2, 0],
                                             [0, 0, std_odo_th/10**2]]))

        # Loading and extracting save odometry route
        odometry = load_from_json(odometry_file_gt)
        temp_x = np.asfarray(odometry[0]); temp_y = np.asfarray(odometry[1]); temp_th = np.asfarray(odometry[2])
        loaded_route = [[pose_x, pose_y, pose_th] for pose_x, pose_y, pose_th in zip(temp_x, temp_y, temp_th)]

        temp_x1, temp_y1, temp_th1 = zip(*loaded_route)
        reduced_path = reduce_dimensions(np.array([temp_x1, temp_y1, temp_th1]), 'half')

        # Adding noise to odo route
        self.x, self.y, self.th = zip(*np.asarray_chkfinite(reduced_path, dtype=np.float64))

        load_noise = False# load_noise = True -> You load the same odometry route. False -> you load the same ground truth route and THEN adding noise
        if load_noise:
            odometry_noise = load_from_json(odometry_file_noise)
            temp_xN = np.asfarray(odometry_noise[0]); temp_yN = np.asfarray(odometry_noise[1]); temp_thN = np.asfarray(odometry_noise[2])
            loaded_route_noise = [[pose_x, pose_y, pose_th] for pose_x, pose_y, pose_th in zip(temp_xN, temp_yN, temp_thN)]
            temp_x2, temp_y2, temp_th2 = zip(*loaded_route_noise)
            reduced_path_noise = reduce_dimensions(np.array([temp_x2, temp_y2, temp_th2]), 'half')
            self.xN, self.yN, self.thN = zip(*np.asarray_chkfinite(np.array(reduced_path_noise), dtype=np.float64))

        else:
            self.xN, self.yN, self.thN = addNoise(self.x, self.y, self.th, std_odo_x, std_odo_y, std_odo_th, mu_bias)
           
            

        print("Distance traveled: {:.0f} m".format(distance_traveled(self.x, self.y)))
        
        self.aarhus = GIS.read_csv(filenamePoints, filenamePoly)
        _, self.rowPoly = self.aarhus.read()
        self.cascaded_poly = self.aarhus.squeeze_polygons(self.rowPoly)

        loaded_landmarks = load_from_json(filelandmarks)
        self.gt_landmarks = loaded_landmarks
        self.landmarks = add_landmark_noise(loaded_landmarks, 0, 0)

    
    def generate_g2o(self, corruption_type: str, plot: bool=False, plot_outliers: bool=False, plot_constraints: bool=False, plot_robot_heading: bool=False):
        
        # Write g2o file
        g2oW = self.write_basics(self.x, self.y, self.th, self.xN, self.yN, self.thN, self.gt_landmarks, self.landmarks)
        self.ground_truth(self.x, self.y, self.th, self.gt_landmarks)
        self.do_loop_closure(self.x, self.y, self.th, self.xN, self.yN, self.thN, g2oW, corruption_type)
        
        try:
            lm_x, lm_y, lm_th = zip(*self.landmark_arrow)
        except ValueError:
            lm_x = lm_y = lm_th = []
            pass
        
        if plot:    
            self.aarhus.plot_map(self.landmarks, show=False)
            if plot_constraints:
                self.plot_constraints()
                if self.bearing_only:
                    robot_heading(lm_x, lm_y, lm_th, color="blue", length=0.08, alpha=0.5, constant=3)

            if plot_outliers:
                plot_outliers_vs_normal(self.pose_pose_outliers, self.pose_landmark_outlier, self.pose_pose, self.pose_landmark)
                self.plot_outliers()

            if plot_robot_heading:
                robot_heading(self.x, self.y, self.th, color="green", length=1)
                robot_heading(self.xN, self.yN, self.thN, color="red", alpha=0.7, length=0.5)

            plt.plot(self.x, self.y, color='forestgreen', linestyle='dashed', label='Groundtruth', linewidth=2)
            plt.plot(self.xN, self.yN, color='firebrick', label='Odometry', linewidth=2)

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))

            plt.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=14, frameon=False)
            plt.ylabel("y (m)", fontsize=16); plt.xlabel("x (m)", fontsize=16)
            plt.xticks(fontsize=16); plt.yticks(fontsize=16)
            plt.ticklabel_format(useOffset=False)
            plt.xlim(574750, 575100); plt.ylim(6222350, 6222700)
            plt.tight_layout()
            # fig = plt.gcf()
            # fig.set_size_inches(14,8)
            # plt.savefig('results/simulated_data/low_density_interpolation.png')
            plt.show()


    def write_basics(self, x, y, theta, xN, yN, thetaN, gt_landmarks, N_landmarks):

        g2oW = open('graphSLAM/data/noise_'+self.time_stamp+'.g2o', 'w')
        self.n_landmarks = 0

        self.xscale = self.x[0]
        self.yscale = self.y[0]
        self.th_scale = self.th[0]
        

        if self.landmark_only or self.bearing_only:
            # If a landmark has been seen before, no not append. Also saving landmarks in a look up table
            self.lm_hist = []
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
                            
                                if pos in self.lm_hist:
                                    continue

                                else:
                                    self.n_landmarks += 1
                                    self.lm_hist.append(pos)
                                    self.lm_lut[self.n_landmarks, key] = pos

        # Odometry id and pose
        for idx, (x_odo, y_odo, th_odo) in enumerate(zip(xN, yN, thetaN)):
                l = '{} {} {} {} {}'.format("VERTEX_SE2", idx+self.n_landmarks, x_odo-self.xscale, y_odo-self.yscale, th_odo)
                g2oW.write(l)
                g2oW.write("\n")

        # Odometry constraints for consecutive poses
        for i in range(1, len(x)):
            p1 = (xN[i-1], yN[i-1], thetaN[i-1])
            p2 = (xN[i], yN[i], thetaN[i])

            T1_w = vec2trans(p1)
            T2_w = vec2trans(p2)
            T2_1 = np.linalg.inv(T1_w) @ T2_w

            del_x = (T2_1[0][2])
            del_y = (T2_1[1][2])
            del_th = (atan2(T2_1[1, 0], T2_1[0, 0]))

            l = '{} {} {} {} {} {} {} {} {} {} {} {}'.format("EDGE_SE2", str((i-1) + self.n_landmarks), str(i+self.n_landmarks), 
                                                            del_x, del_y, del_th, 
                                                            self.H_odo[0,0], self.H_odo[0,1], self.H_odo[0,2], self.H_odo[1,1], self.H_odo[1,2], self.H_odo[2,2])
            g2oW.write(l)
            g2oW.write("\n")

        return g2oW


    def ground_truth(self, x, y, theta, landmarks):
        g2oWG = open('graphSLAM/data/ground_truth_'+self.time_stamp+'.g2o', 'w')
        n_landmarks = 0

        if self.landmark_only or self.bearing_only:
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
        
        for i in range(len(self.x)):
            l = '{} {} {} {} {}'.format("VERTEX_SE2", str(i+n_landmarks), self.x[i]-self.x[0], self.y[i]-self.y[0], self.th[i])

            g2oWG.write(l)
            g2oWG.write("\n")

    def do_loop_closure(self, x, y, th, xN, yN, thN, g2oW, ct):       
       
        # Ground truth points; Noisy points
        points =  [[x, y, th] for x, y, th in zip(x, y, th)]
        pointsN =  [[xN, yN, thN] for xN, yN, thN in zip(xN, yN, thN)]
        
        # Keeping track of amount of loop closures
        n_lc_odo = 0
        n_lc_lm = 0

        check_n_outliers = 0
        check_n_outliers_lm = 0
        offset_odo = self.n_landmarks
        lm_id_check = []

        # Searching through poses looking for constraints
        path_size = len(x)
        for pose_id in range(path_size-1):
            
            curr_pose = np.array([[x[pose_id]], [y[pose_id]], [th[pose_id]]])

            rand_poseID = random.randint(0, path_size-1)
            random_pose = np.array([[xN[rand_poseID]], [yN[rand_poseID]], [thN[rand_poseID]]])

            # Re initializing for every pose_id
            check_pose_odo = 0
            if self.pose_only:
                # Odometry constraints
                for j in range(path_size-1):
                    if  j < pose_id and abs(pose_id-j) > 10:
                        
                        # Checking for loop closure range pose-pose
                        d_odo = np.linalg.norm(curr_pose[0:2,:].T - np.array(points[j][0:2], dtype=np.float64))
                        
                        if np.greater(self.odo_lc_range, d_odo):
                        
                            n_lc_odo += 1

                            # Saving pose_pose for visualization
                            self.pose_pose.append([pointsN[check_pose_odo][0:2], pointsN[pose_id][0:2]])
                            # self.pose_pose.append([points[check_pose_odo][0:2], points[pose_id][0:2]])

                            # Computing the relative distance between two poses in close vincinity (Simulating ICP)
                            edge_lc_odo = (pose_id+1, check_pose_odo)
                            other_pose = np.array([[x[check_pose_odo]], [y[check_pose_odo]], [th[check_pose_odo]]])
                            lc_constraint_odo = self.generate_lc_constraint(curr_pose, other_pose)                

                            self.write_loop_constraints(g2oW, "EDGE_SE2", offset_odo, edge_lc_odo, lc_constraint_odo, self.H_icp, descriptor='pose')

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
                            self.write_loop_constraints(g2oW, "EDGE_SE2", offset_odo, edge_lc_odo, lc_constraint_odo, self.H_icp, descriptor='pose')
                        
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
                            self.write_loop_constraints(g2oW, "EDGE_SE2", offset_odo, edge_lc_odo, lc_constraint_odo, self.H_odo, descriptor='pose')

                        check_n_outliers += 1
                    else:
                        pass
            
            if self.bearing_only or self.landmark_only:
                N_pose = np.array([[xN[pose_id]], [yN[pose_id]], [thN[pose_id]]])

                # Writing landmark constraints
                for (lm_id, _), pos in self.lm_lut.items():

                    # Checking for loop closure range
                    d_landmark = np.linalg.norm(curr_pose[0:2,:].T - np.array(pos, dtype=np.float64))
                    
                    # Checking distance criteria, pose-landmarks
                    if np.greater(self.lm_lc_range, d_landmark):
                        
                        offset_lm = self.n_landmarks
                        edge_lc_lm = (pose_id, lm_id-1)

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
                            lm_noise = [np.random.normal(0, self.std_lm_x), 
                                        np.random.normal(0, self.std_lm_y)]
                            
                            other_pose = np.array([[xN[pose_id]+d_landmark*np.cos(float(bearing))+lm_noise[0]],
                                                   [yN[pose_id]+d_landmark*np.sin(float(bearing))+lm_noise[1]], 
                                                   [noisy_bearing]]) 

                            lc_constraint_lm = self.generate_lc_constraint(N_pose, other_pose) 

                            if self.bearing_only:
                                self.write_loop_constraints(g2oW, "EDGE_SE2_BEARING", offset_lm, edge_lc_lm, lc_constraint_lm, self.H_xy, descriptor='bearing')
                                self.landmark_arrow.append([pointsN[pose_id][0], pointsN[pose_id][1], noisy_bearing])
                                n_lc_lm += 1

                            elif self.landmark_only:
                                self.write_loop_constraints(g2oW, "EDGE_SE2_XY", offset_lm, edge_lc_lm, lc_constraint_lm, self.H_xy, descriptor='landmark')
                                self.pose_landmark.append([pointsN[pose_id][0:2], [other_pose[0], other_pose[1]]])
                                n_lc_lm += 1

                                if lm_id in lm_id_check:
                                    continue
                                else:
                                    lm_id_check.append(lm_id)
                                    l = '{} {} {} {}'.format("VERTEX_XY", lm_id-1,\
                                                             float(other_pose[0,0]-self.xscale+lm_noise[0]),\
                                                             float(other_pose[1,0]-self.yscale+lm_noise[1]))
                                    g2oW.write(l)
                                    g2oW.write("\n")
                                
                            
                            # CORRUPTING LANDMARK DATA
                            if self.corrupt_dataset and check_n_outliers_lm <= self.n_outliers: #and random.random() < 0.1:

                                cluster = 1 if self.outlier_type[ct] == 1 or self.outlier_type[ct] == 3 else 7

                                for _ in range(cluster*2):
                                    rand_landmarkID = random.randint(0, path_size-1)
                                    edge_lc_lm = (rand_landmarkID, lm_id-1)

                                    self.pose_landmark_outlier.append([pointsN[rand_landmarkID][0:2], pos])
                                    lc_constraint_lm = self.generate_lc_constraint(random_pose, other_pose)

                                    if self.bearing_only:
                                        self.write_loop_constraints(g2oW, "EDGE_SE2_BEARING", offset_lm, edge_lc_lm, lc_constraint_lm, self.H_xy, descriptor='bearing')

                                    elif self.landmark_only:
                                        self.write_loop_constraints(g2oW, "EDGE_SE2_XY", offset_lm, edge_lc_lm, lc_constraint_lm, self.H_xy, descriptor='landmark')

                                check_n_outliers_lm += 1


        
        print('\nOdometry loop closure: {}\t outliers: {}'.format(n_lc_odo, len(self.pose_pose_outliers)))
        print('Landmark loop closure: {}\t outliers: {}'.format(n_lc_lm, len(self.pose_landmark_outlier)))

        print(f'\nCondition set to;\nPose-Pose:\t\t{self.pose_only}\nPose-Landmark:\t{self.landmark_only}\nPose-Bearing:\t{self.bearing_only}\n')

    def write_loop_constraints(self, g2o, edge_type, offset, edge_lc, lc_constraint, H, descriptor: str='pose'):
        """Function that writes to the g2o file constraint information. 

        Args:
            g2o: file object
            edge_type (str): type of constraint
            offset (int): ID offset
            edge_lc (int): IDs
            lc_constraint (matrix): contains relative constraint information
            H (matrix): information matrix
            descriptor (str, optional): The constraint type wanted. Defaults to 'pose'.
        """        
        if descriptor == 'pose':
            l = '{} {} {} {} {} {} {} {} {} {} {} {}'.format(edge_type, edge_lc[0]+offset, edge_lc[1]+offset, 
                                                            lc_constraint[0,0], lc_constraint[1,0], lc_constraint[2,0],
                                                            H[0,0], H[0,1], H[0,2], H[1,1], H[1,2], H[2,2])
        elif descriptor == 'landmark':
            l = '{} {} {} {} {} {} {} {}'.format(edge_type, edge_lc[0]+offset, edge_lc[1], 
                                                          lc_constraint[0,0], lc_constraint[1,0],
                                                          H[0,0], H[0,1], H[1,1])
        elif descriptor == 'bearing':
            l = '{} {} {} {} {}'.format("EDGE_SE2_BEARING", 
                                        edge_lc[0]+offset, edge_lc[1], 
                                        lc_constraint[2,0],
                                        self.H_xy[2,2])

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

    def plot_outliers(self):
        
        try: # pose_pose
            pose_pose_outliers = np.vstack(self.pose_pose_outliers)
            
            pose_outliersx = pose_pose_outliers[:,0]
            pose_outliersy = pose_pose_outliers[:,1]
            
            pox = np.vstack([pose_outliersx[0::2], pose_outliersx[1::2]])
            poy = np.vstack([pose_outliersy[0::2], pose_outliersy[1::2]])
        
            plt.plot(pox, poy, color='gray', alpha=0.8, label="Outlier pose constraints")
        except:
            pass

        try: # pose_landmark/bearing
            pose_landmark_outlier = np.vstack(self.pose_landmark_outlier)
            
            landmark_outliersx = pose_landmark_outlier[:,0]
            landmark_outliersy = pose_landmark_outlier[:,1]
            
            plx = np.vstack([landmark_outliersx[0::2], landmark_outliersx[1::2]])
            ply = np.vstack([landmark_outliersy[0::2], landmark_outliersy[1::2]])
        
            plt.plot(plx, ply, color='gray', alpha=0.8, label="Outlier landmark constraints")
        except:
            pass

    def plot_constraints(self):
        
        try: # pose_landmark/bearing
            pose_landmark = np.vstack(self.pose_landmark)
            
            pose_landmarkx = pose_landmark[:,0]
            pose_landmarky = pose_landmark[:,1]
            
            plx = np.vstack([pose_landmarkx[0::2], pose_landmarkx[1::2]])
            ply = np.vstack([pose_landmarky[0::2], pose_landmarky[1::2]])
            

            plt.plot(plx, ply, color='purple', label="Landmark observation")
        except:
            pass

        try: # pose_pose
            pose_pose = np.vstack(self.pose_pose)
            
            pose_posex = pose_pose[:,0]
            pose_posey = pose_pose[:,1]
            
            ppx = np.vstack([pose_posex[0::2], pose_posex[1::2]])
            ppy = np.vstack([pose_posey[0::2], pose_posey[1::2]])
            
            plt.plot(ppx, ppy, color='purple', label="Real pose constraints")
        except:
            pass


if __name__ == '__main__':

    filenamePoints = 'g2o_generator/GIS_Extraction/data/aarhus_features_v2.csv'
    filenamePoly = 'g2o_generator/GIS_Extraction/data/aarhus_polygons_v2.csv'
    filelandmarks = 'g2o_generator/GIS_Extraction/landmarks/landmarks_w_types.json'
    odometry_file = './g2o_generator/robosim/data/robopath/Aarhus_path1.json'
    genG2O = g2o(odometry_file, filenamePoints, filenamePoly, filelandmarks, 15, 2)
    genG2O.generate_g2o(corruption_type='random', plot=True, plot_outliers=True, plot_constraints=True, plot_robot_heading=True)


    