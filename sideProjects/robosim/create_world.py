import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.core.numeric import full


# local packages
from lib.utility import *

class world:
    def __init__(self, background: str, landmarks: str, save_path: bool = False, load_path: bool = False, path_name: str = 'Aarhus_path1.json'):
       
        self.load_path = load_path
        self.save_path = save_path
        self.path_name = path_name
        
       # Loading map from GIS data as a binary image
        img = cv2.imread(background, cv2.IMREAD_GRAYSCALE)
        img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]
        #self.background = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)[1]

        # Masking outside box to be black
        mask = np.zeros_like(img)
        mask = cv2.rectangle(mask, (80,58), (576,427), (255, 255, 255), -1)
        img = cv2.bitwise_and(img, mask)
        self.crop_img = img[56:429, 78:578]

        if self.load_path:
            self.loaded_route = load_from_json('./sideProjects/robosim/data/robopath/'+self.path_name)
            self.x_odo, self.y_odo, self.th_odo = zip(*np.asarray_chkfinite(self.loaded_route))
            
            
        # Loading landmarks that was saved using GIS data
        with open(landmarks, 'r') as f:
            self.landmarks = np.genfromtxt(f, delimiter=',')


    def draw_points(self, event, x, y, flags, param):
        """ Draws circle when clicking (left click) on the image """

        if event == cv2.EVENT_LBUTTONDOWN:

            cv2.circle(self.grid_copy, (x,y), 5, (0, 255, 0), -1)
            cv2.imshow('Robot_route', self.grid_copy)
            self.route.append([x,y])

    @staticmethod
    def robot_heading(x, y, theta, length: float = 5, width: float = 0.05):
        """
        Method that plots the heading of every pose
        """
        x = x[1:-1]
        y = y[1:-1]
        theta = theta[1:-1]

        terminus_x = x + length * np.cos(theta)
        terminus_y = y + length * np.sin(theta)
        plt.plot([x, terminus_x], [y, terminus_y])


    def make_robot_path(self, plot_route: bool = True):
        self.grid = np.asarray(self.crop_img)
        self.grid_copy = self.grid.copy()
        self.route = []

        # If we do not have a robot path saved in 'sideProjects/robosim/data'
        if self.save_path:
            cv2.namedWindow('Robot_route')
            cv2.imshow('Robot_route',self.grid_copy)
            cv2.setMouseCallback('Robot_route', self.draw_points, self.grid_copy)

            cv2.waitKey(0)
            cv2.destroyAllWindows()


            # calculating angles between all points and concatenating
            angles = calculate_angles(self.route)
            poses = [[pose[0], pose[1], theta] for pose, theta in zip(self.route, angles)]
            poses = [poses[0]] + poses + [poses[-1]]

            # Smoothening of route using splines
            full_route = do_rom_splines(poses)
            self.x_odo, self.y_odo, self.th_odo = zip(*full_route)

            # Saving the reduced route to json file 
            json_path = reduce_dimensions(np.array([self.x_odo, self.y_odo, self.th_odo]))
            json_path1 = json_path.tolist()
            save_to_json(json_path1,'./sideProjects/robosim/data/robopath/'+self.path_name)
        

        # Odometry drift
        self.x_odo_noisy, self.y_odo_noisy, self.th_odo_noisy = odometry_drift_simple(self.x_odo, self.y_odo, self.th_odo)
        if plot_route:
            
            plt.plot(self.x_odo, self.y_odo, label='Original route')
            plt.plot(self.x_odo_noisy, self.y_odo_noisy, label='Noise route')

            if False:
                px, py, pth = zip(*poses)
                px = np.asarray_chkfinite(px)
                py = np.asarray_chkfinite(py)
                pth = np.asarray_chkfinite(pth)
                
                self.robot_heading(px, py, pth)
            
            
            plt.imshow(self.crop_img, cmap='gray')
            plt.legend(loc="upper left")
            plt.show()



def main():
    # print(cv2.__version__)
    map = 'sideProjects/GIS_Extraction/plots/GIS_map3.png'
    landmarks_file = 'sideProjects/GIS_Extraction/landmarks/landmarks_points.csv'

    show = world(map, landmarks_file, save_path=True, load_path=False)
    show.make_robot_path()
  

if __name__ == "__main__":
    main()

