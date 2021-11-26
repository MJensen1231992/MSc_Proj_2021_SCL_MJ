import numpy as np
from math import atan2, cos, sin
import matplotlib.pyplot as plt

import sys
sys.path.append('g2o_generator/robosim')

from lib.bearing_utils import least_squares, triangulation

class Test_LM:

    def __init__(self):
        # self.create_data()
        pass

        

    def create_data(self):

        self.route = []
        #                                               (n_landmarks, (x,y))
        lm_poses = np.random.default_rng().uniform(0,10,(5,2))
        # robot_poses = np.random.default_rng().uniform(0,1,(7,2))

        self.fig = plt.figure()
        self.axes = self.fig.add_subplot(111)

        self.axes.scatter(lm_poses[:,0], lm_poses[:,1])
        self.axes.set_ylim([0,10])
        self.axes.set_xlim([0,10])
        self.fig.canvas.mpl_connect('button_press_event', self.get_points)
        plt.show()

        robot_poses = np.asfarray(self.route)
        angles = [atan2((y2 - y1), (x2 - x1)) for (x1, y1), (x2, y2) in zip(robot_poses[:-1], robot_poses[1:])]
        route_poses = [[pose[0], pose[1], theta] for pose, theta in zip(robot_poses, angles)]
        b = []

        for i in range(len(lm_poses[:,0])):
            for x, y, _ in route_poses:
                e = np.deg2rad(np.random.normal(0, 5))#+np.deg2rad(2)
                # print(np.rad2deg(e))
                gt_bearing = atan2(lm_poses[i,1]-y,lm_poses[i,0]-x)+e
                b.append(gt_bearing)
            b_stack = np.vstack((np.array(b)))
                
        b_stack = np.reshape(b_stack, (len(lm_poses[:,0]), len(route_poses)))
        route = np.vstack(route_poses)

        plt.scatter(route[:,0],route[:,1], color='blue', label='robot poses')
        plt.quiver(route[:,0],route[:,1], np.cos(route[:,2]), np.sin(route[:,2]), angles='xy', scale=4, color='red', alpha=0.5)
        for i in range(len(b_stack[:,0])):
            plt.quiver(route[:,0],route[:,1], np.cos(b_stack[i,:]), np.sin(b_stack[i,:]), angles='xy', scale=1, color='magenta', alpha=0.5)
        plt.scatter(lm_poses[:,0],lm_poses[:,1], color='green', label='ground truth')
        plt.legend()
        # plt.grid()
        plt.show()
        return route, b_stack, lm_poses


    def get_points(self, event):

        x = float(event.xdata)
        y = float(event.ydata)
        self.axes.scatter(x,y, color='black')
        self.fig.canvas.draw()
        self.route.append([x, y])

def main():
    cd = Test_LM()
    Xr, z_list, lm = cd.create_data()

    ls = least_squares.LeastSquares()
    tri = triangulation.Triangulation()

    Xl_ls = []
    Xl_tri = []
    for i in range(len(z_list[:,0])):
        z = z_list[i,:]
        Xl_ls.append(ls.least_squares_klines(Xr, z))
        Xl_tri.append(tri.triangulation(Xr, z))

    dim = (len(lm[:,0]),2)

    Xl_ls = np.vstack(Xl_ls)
    Xl_ls = np.reshape(Xl_ls, dim)

    Xl_tri = np.vstack(Xl_tri)
    Xl_tri = np.reshape(Xl_tri, dim)

    # print(lm)
    # print(Xl_ls)
    # print(Xl_tri)
    ls_e = np.linalg.norm(Xl_ls - lm)
    tri_e = np.linalg.norm(Xl_tri - lm)
    print(f"Distance error: ls: {ls_e} \ttri: {tri_e}")

    plt.scatter(Xl_ls[:,0], Xl_ls[:,1], label='Least squares lm', color='red')
    plt.scatter(Xl_tri[:,0], Xl_tri[:,1], label='Triangualtion lm', color='orange')
    plt.scatter(lm[:,0], lm[:,1], label='Ground truth lm', color='green')
    plt.legend()
    plt.show()

        


if __name__ == "__main__":
    main()


