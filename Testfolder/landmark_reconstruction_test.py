import numpy as np
from math import atan2
import matplotlib.pyplot as plt
import time

import sys
sys.path.append('g2o_generator/robosim')

from lib.bearing_utils import least_squares, triangulation

class Test_LM:

    def __init__(self):
        # self.create_data()
        pass

        

    def create_data(self, n_poses, sigma):

        
        self.route = []
        ax = (0,11)
        #                                               (n_landmarks, (x,y))
        # lm_poses = np.random.default_rng().uniform(ax[0],ax[1],(10,2))
        lm_poses = np.matrix([[3,2]])
        lm_poses = np.array(lm_poses)
        r = [[0,1],[1,1],[2,1],[3,1],[4,1],[5,1],[6,2],[6.5,3],[7,4],[6,5],[5,6],[0,2],[0,3],[1,0]]
        self.route = r[0 : n_poses]
        # print(lm_poses)
        # print(lm_poses1)
        # lm_poses = np.matrix([[3,3],[5.6,7],[1,8]])
        # print(lm_poses)
        # robot_poses = np.random.default_rng().uniform(0,1,(7,2))
        # robot_poses = np.matrix([[],])

        # self.fig = plt.figure()
        # self.axes = self.fig.add_subplot(111)

        # self.axes.scatter(lm_poses[:,0], lm_poses[:,1])
        # self.axes.set_ylim([ax[0],ax[1]])
        # self.axes.set_xlim([ax[0],ax[1]])
        # self.fig.canvas.mpl_connect('button_press_event', self.get_points)
        # plt.show()
        # print(self.route)
        robot_poses = np.asfarray(self.route)
        angles = [atan2((y2 - y1), (x2 - x1)) for (x1, y1), (x2, y2) in zip(robot_poses[:-1], robot_poses[1:])]
        route_poses = [[pose[0], pose[1], theta] for pose, theta in zip(robot_poses, angles)]

        b = []

        for i in range(len(lm_poses[:,0])):
            for x, y, _ in route_poses:
                e = np.deg2rad(np.random.normal(0, sigma))#+np.deg2rad(2)
                # print(np.rad2deg(e))
                gt_bearing = atan2(lm_poses[i,1]-y,lm_poses[i,0]-x)+e
                b.append(gt_bearing)
            b_stack = np.vstack((np.array(b)))
                
        b_stack = np.reshape(b_stack, (len(lm_poses[:,0]), len(route_poses)))
        route = np.vstack(route_poses)

        if n_poses > 6 and sigma > 2.5:
            plt.figure(101)
            plt.scatter(route[:,0],route[:,1], color='blue', label='robot poses')
            # plt.quiver(route[:,0],route[:,1], np.cos(route[:,2]), np.sin(route[:,2]), angles='xy', scale=4, color='red', alpha=0.5)
            for i in range(len(b_stack[:,0])):
                plt.quiver(route[:,0],route[:,1], np.cos(b_stack[i,:]), np.sin(b_stack[i,:]), angles='xy', scale=1, color='magenta', alpha=0.5)
            # plt.scatter(lm_poses[:,0],lm_poses[:,1], color='green', label='ground truth')
            plt.legend()
            plt.grid()
            # plt.show()
        return route, b_stack, lm_poses.T


    def get_points(self, event):

        x = float(event.xdata)
        y = float(event.ydata)
        self.axes.scatter(x,y, color='black')
        self.fig.canvas.draw()
        self.route.append([x, y])

def main():

    # sigma = np.linspace(0, 8, 100)

    # ls_stat = []
    # tri_stat = []

    # time_ls = []
    # time_tri = []

    # ls = least_squares.LeastSquares()
    # tri = triangulation.Triangulation()
    # cd = Test_LM()

    # for s in sigma:
    #     Xr, z_list, lm = cd.create_data(12, s)
    #     Xl_ls = []
    #     Xl_tri = []
        
    #     for i in range(len(z_list[:,0])):
    #         z = z_list[i,:]

    #         _ls = ls.least_squares_klines(Xr, z)
    #         _tri = tri.triangulation(Xr, z)
           
    #         Xl_ls.append(_ls)
    #         Xl_tri.append(_tri)

    #     if s != 0:
    #         ls_e = np.linalg.norm(_ls - lm)
    #         tri_e = np.linalg.norm(_tri - lm)
    #         # print(f"ls: {ls_e}\ttri: {tri_e}")
    #         ls_stat.append(ls_e)
    #         tri_stat.append(tri_e)
        
    #     dim = (len(lm[:,0]),2)

        # Xl_ls = np.ndarray.flatten(np.array(Xl_ls))
        # print(Xl_ls)
        # Xl_ls = np.vstack(Xl_ls)
        # Xl_ls = np.reshape(Xl_ls, dim)

        # Xl_tri = np.ndarray.flatten(Xl_tri)
        # # Xl_tri = np.vstack(Xl_tri)
        # Xl_tri = np.reshape(Xl_tri, dim)

        # plt.figure()
        # plt.scatter(Xl_ls[:,0], Xl_ls[:,1], marker='^', label='Least squares lm', color='red')
        # plt.scatter(Xl_tri[:,0], Xl_tri[:,1], marker=',', label='Triangualtion lm', color='orange')
        # plt.scatter(lm[:,0], lm[:,1], label='Ground truth lm', color='green')
        # plt.legend()
        # plt.show()

        # print(lm)
        # print(Xl_ls)
        # print(Xl_tri)
        

        
    
    # plt.plot(sigma[1:], ls_stat, label='Error - Least squares', color='orange')
    # plt.plot(sigma[1:], tri_stat, label='Error - Triangulation', color='red', alpha=0.5)
    # plt.xlabel('Bearing variance [rad]')
    # plt.ylabel('Dist error [m]')
    # plt.grid()
    # plt.legend()
    # plt.show()
    
    #########################################

    sigma = np.linspace(0, 4, 50)
    n_poses = np.linspace(4, 10, 7)

    ls_stat = []
    tri_stat = []
    iter = 20

    time_ls = []
    time_tri = []

    ls = least_squares.LeastSquares()
    tri = triangulation.Triangulation()
    cd = Test_LM()

    for n in n_poses:
        for i in range(iter):
            for s in sigma:
                Xr, z_list, lm = cd.create_data(int(n), s)
                lm = np.array(lm)

                Xl_ls = []
                Xl_tri = []

                for i in range(len(z_list[:,0])):
                    z = z_list[i,:]

                    start_ls = time.time()
                    _ls = ls.least_squares_klines(Xr, z)
                    end_ls = time.time()

                    start_tri = time.time()
                    _tri = tri.triangulation(Xr, z)
                    end_tri = time.time()

                    Xl_ls.append(_ls.T)
                    Xl_tri.append(_tri.T)

                
                time_ls.append(end_ls-start_ls)
                time_tri.append(end_tri-start_tri)

                dim = (len(lm[:,0]),2)
                ls_e = np.linalg.norm(Xl_ls - lm.T)
                tri_e = np.linalg.norm(Xl_tri - lm.T)
                ls_stat.append([ls_e])
                tri_stat.append([tri_e])

                Xl_ls = np.vstack(Xl_ls)
                # Xl_ls = np.reshape(Xl_ls, dim)

                Xl_tri = np.vstack(Xl_tri)
                # Xl_tri = np.reshape(Xl_tri, dim)

                # print(lm)
                # print(Xl_ls)
                # print(Xl_tri)
                
                if n > 6 and s > 2.5:
                    print(f"Distance error: ls: {ls_e} \t \ttri: {tri_e}")
                    print(f"Amount of poses: {n}\nVariance: {s}")
                    plt.figure(101)
                    plt.scatter(Xl_ls[:,0], Xl_ls[:,1], marker='^', label='Least squares lm', color='red')
                    plt.scatter(Xl_tri[:,0], Xl_tri[:,1], marker=',', label='Triangualtion lm', color='orange')
                    plt.scatter(lm.T[:,0], lm.T[:,1], label='Ground truth lm', color='green')
                    plt.legend()
                    plt.show()
            
            ls_tot = np.vstack(ls_stat)
            tri_tot = np.vstack(tri_stat)
        
        # print(ls_tot)
        ls_tot = np.reshape(ls_tot, (iter,len(sigma)))
        tri_tot = np.reshape(tri_tot, (iter,len(sigma)))
        # print(ls_tot)

        ls_tot = np.mean(ls_tot, axis=0)
        tri_tot = np.mean(tri_tot, axis=0)

        title = f"n poses {int(n)-1}"

        plt.figure(int(n)-1)
        plt.plot(sigma, tri_tot, label='Error - Triangulation', color='orange')
        plt.plot(sigma, ls_tot, label='Error - Least squares', color='red', alpha=0.5)
        plt.xlabel('Bearing variance [rad]')
        plt.ylabel('Dist error [m]')
        plt.ylim([0,4])
        plt.title(title)
        plt.legend()

        ls_stat, tri_stat = [], []
        del ls_tot, tri_tot
        
    avg_time_ls = np.mean(np.array(time_ls))
    avg_time_tri = np.mean(np.array(time_tri))

    print(f"Avg ls time: {avg_time_ls} \tAvg tri time: {avg_time_tri}")

    plt.show()




if __name__ == "__main__":
    main()


