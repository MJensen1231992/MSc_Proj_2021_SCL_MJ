import numpy as np
import matplotlib.pyplot as plt
from descartes import PolygonPatch
from numpy.core.numeric import full
import sys
sys.path.append('g2o_generator/GIS_Extraction')

# local packages
import csv_reader as GIS
from lib.utility import *


class world:
    def __init__(self, OSM_polygons: str, OSM_features: str, landmarks: str, save_path: bool = False,
                 load_path: bool = False, path_name: str = 'Aarhus_path1.json', GNSS_frequency: int = 10):
        """
        csv_info takes in csv file for landmarks and csv file for polygons
        """

        self.load_path = load_path
        self.save_path = save_path
        self.path_name = path_name

        self.aarhus = GIS.read_csv(OSM_polygons, OSM_features)
        self.rowPoints, self.rowPoly = self.aarhus.read()
        self.poly_stack = self.aarhus.squeeze_polygons(self.rowPoly, plot=False)
        self.landmarks = load_from_json(landmarks)

        self.GNSS_frequency = GNSS_frequency

        if self.load_path:
            loaded_route = load_from_json('./g2o_generator/robosim/data/robopath/'+self.path_name)
            temp_x = np.asfarray(loaded_route[0]); temp_y = np.asfarray(loaded_route[1]); temp_th = np.asfarray(loaded_route[2])
            self.loaded_route = [[pose_x, pose_y, pose_th] for pose_x, pose_y, pose_th in zip(temp_x, temp_y, temp_th)]

            full_route = do_rom_splines(np.asfarray(self.loaded_route, dtype=np.float64))
            temp_x1, temp_y1, temp_th1 = zip(*full_route)
            reduced_path = reduce_dimensions(np.array([temp_x1, temp_y1, temp_th1]), 'fifth')
            self.x_odo, self.y_odo, self.th_odo = zip(*reduced_path)


    # Draw points using mouse for plt figures
    def get_points(self, event):
        x = float(event.xdata)
        y = float(event.ydata)
        self.route.append([x, y])


    def make_robot_path(self, plot_route: bool = True):
        """[summary]

        Args:
            plot_route (bool, optional): [description]. Defaults to True.
        """        

        # If we do not have a robot path saved in 'sideProjects/robosim/data' then set 'save_path=True'
        if self.save_path:
            
            self.route = []
            fig = plt.figure()

            for id, _ in enumerate(self.poly_stack):
                ax = fig.add_subplot()
                patch = PolygonPatch(self.poly_stack[id].buffer(0))
                ax.add_patch(patch)
                ax.plot()

            fig.canvas.mpl_connect('button_press_event', self.get_points)
            plt.show()

            # calculating angles between all points and concatenating
            # angles = calculate_angles(self.route)
            # print("ANGLES")
            # print(angles)
            # poses = [[pose[0], pose[1], theta] for pose, theta in zip(self.route, angles)]
            # poses = [poses[0]] + poses
            

            # x, y, th = zip(*np.asarray_chkfinite(poses[1:-1], dtype=np.float64)) ## 

            # Smoothening of route using splines

            full_route = do_rom_splines(self.route)
            full_route = [[pose[0], pose[1]] for pose in full_route]

            angles = calculate_angles(full_route)

            full_route_poses = [[pose[0], pose[1], theta] for pose, theta in zip(full_route, angles)]
            x_odo, y_odo, th_odo = zip(*full_route_poses)


            # plt.plot(x_odo, y_odo)
            # robot_heading(x_odo,y_odo,th_odo, color="blue")

            #reduced_path = reduce_dimensions(np.array([self.x_odo, self.y_odo, self.th_odo]), 'fifth')
            #self.x_odo, self.y_odo, self.th_odo = zip(*reduced_path)

            # Saving the reduced route to json file 
            
            json_path = np.array([x_odo, y_odo, th_odo])
            json_path1 = json_path.tolist()
            save_to_json(json_path1,'./g2o_generator/robosim/data/robopath/'+self.path_name)
    
        # Odometry drift
        #self.x_odo_noisy, self.y_odo_noisy, self.th_odo_noisy = odometry_drift_simple(self.x_odo, self.y_odo, self.th_odo)

        # Add GNSS points
        # self.GNSS_points = []
        # for i in range(len(self.x_odo)):
        #     if (i % self.GNSS_frequency == 0):
        #         # print('Adding GPS point {}'.format(i))
        #         x_gps, y_gps = add_GNSS_noise(self.x_odo[i], self.y_odo[i], std_gps_x=0.00001, std_gps_y=0.00001)
        #         self.GNSS_points.append([x_gps, y_gps])
        
        # if plot_route:
        #     self.plot_robot_route(show=True)

    #def plot_robot_route(self, show: bool=0):
            
            #self.aarhus.plot_map(self.landmarks)
            #plt.plot(self.x_odo, self.y_odo, label='Groundtruth')
            #plt.plot(self.x_odo_noisy, self.y_odo_noisy, label='Noise route')
            #plt.scatter(np.asarray_chkfinite(self.GNSS_points)[:,0], np.asarray_chkfinite(self.GNSS_points)[:,1], marker='x', color='red',
                                    #         label='GPS points')

            #plt.xlim([min(min(self.x_odo), min(self.x_odo_noisy)), max(max(self.x_odo), max(self.x_odo_noisy))])
            #plt.ylim([min(min(self.y_odo), min(self.y_odo_noisy)), max(max(self.y_odo), max(self.y_odo_noisy))])

            # For debugging purposes of angle calculations
            #if False:
                #px, py, pth = zip(*self.loaded_route)
                #px = np.asarray_chkfinite(px)
                #py = np.asarray_chkfinite(py)
               # pth = np.asarray_chkfinite(pth)
                
              #  robot_heading(px, py, pth)
            
           # plt.legend(loc="upper left")

            #if show:
             #   plt.show()


def main():

    filenamePoints = 'g2o_generator/GIS_Extraction/data/aarhus_features_v2.csv'
    filenamePoly = 'g2o_generator/GIS_Extraction/data/aarhus_polygons_v2.csv'
    landmarks = './g2o_generator/GIS_Extraction/landmarks/landmarks_w_types.json'

    show = world(filenamePoints, filenamePoly, landmarks, save_path=False, load_path=True)
    show.make_robot_path()
  

if __name__ == "__main__":
    main()

