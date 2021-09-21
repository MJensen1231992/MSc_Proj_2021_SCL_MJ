import numpy as np
import matplotlib as plt
import cv2
import matplotlib.pyplot as plt
import sys

from numpy.core.defchararray import count

# local packages
from lib.utility import do_splines, calculate_angles

class world:
    def __init__(self, background: str, landmarks: str):

        img = cv2.imread(background, cv2.IMREAD_GRAYSCALE)

        img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]
        #self.background = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)[1]

        # Masking outside box to be black
        mask = np.zeros_like(img)
        mask = cv2.rectangle(mask, (80,58), (576,427), (255, 255, 255), -1)
        img = cv2.bitwise_and(img, mask)
        self.crop_img = img[56:429, 78:578]

        with open(landmarks, 'r') as f:
            self.landmarks = np.genfromtxt(f, delimiter=',')

        # cv2.imshow('map', img)
        # cv2.imshow('cropped', self.crop_img)

    def draw_points(self, event, x, y, flags, param):
        """ Draws circle when using the mouse on the image """

        if event == cv2.EVENT_LBUTTONDBLCLK:

            cv2.circle(self.grid_copy, (x,y), 5, (0, 255, 0), -1)
            cv2.imshow('Robot_route', self.grid_copy)
            self.route.append((x,y))

    def make_robot_path(self, print_coordinates: bool = True):
        self.grid = np.asarray(self.crop_img)
        self.grid_copy = self.grid.copy()
        self.route = []

        cv2.namedWindow('Robot_route')
        cv2.imshow('Robot_route',self.grid_copy)
        cv2.setMouseCallback('Robot_route', self.draw_points, self.grid_copy)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # print(self.route)
        angles = calculate_angles(self.route)
        

        # poses, full_route = do_splines(self.route, angles)


        # if print_coordinates:
        #     print(full_route)




def main():
    # print(cv2.__version__)
    map = 'sideProjects/GIS_Extraction/plots/GIS_map3.png'
    landmarks_file = 'sideProjects/GIS_Extraction/landmarks/landmarks_points.csv'
    show = world(map, landmarks_file)
    show.make_robot_path()
    # while True:
    #         k = cv2.waitKey(0) & 0xFF
    #         print(k)
    #         if k == 27:
    #             cv2.destroyAllWindows()
    #             break

if __name__ == "__main__":
    main()

