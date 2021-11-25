import numpy as np
from math import atan2, cos, sin
import matplotlib.pyplot as plt

class LsLines:

    def __init__(self):
        # cj is the confidence value of 
        self.cj = 5
        self.least_squares_line_intersection()

        

    def least_squares_line_intersection(self):

        self.route = []

        lm_pose = np.array([[5],[5]])
        # robot_poses = np.random.default_rng().uniform(0,1,(7,2))

        self.fig = plt.figure()
        self.axes = self.fig.add_subplot(111)

        self.axes.scatter(lm_pose[0,0], lm_pose[1,0])
        self.axes.set_ylim([0,10])
        self.axes.set_xlim([0,10])
        self.fig.canvas.mpl_connect('button_press_event', self.get_points)
        plt.show()

        robot_poses = np.asfarray(self.route)

        b = []
        n = []
        for x, y in robot_poses:
            e = np.deg2rad(np.random.normal(1, 5))#+np.deg2rad(2)
            # print(np.rad2deg(e))
            gt_bearing = atan2(lm_pose[1,0]-y,lm_pose[0,0]-x)+e
            b.append(gt_bearing)
            n.append([cos(gt_bearing), sin(gt_bearing)])


        n = np.array(n)
        R = np.zeros((2,2))
        q = np.zeros((2,1))
        eye = np.eye(2)

        for i in range(len(robot_poses)):

            nn = np.matrix([n[i,0], n[i,1]])
            a = np.matrix([robot_poses[i,0]+(n[i,0]*2),robot_poses[i,1]+(n[i,1]*2)])
            # a = np.matrix([robot_poses[i,0],robot_poses[i,1]])
            R += (nn.T @ nn - eye) * self.cj
            q += (nn.T @ nn - eye) @ a.T * self.cj

        p = np.linalg.inv(R) @ q

        e = np.linalg.norm(p-lm_pose)
        print(f"Error: {e: .2f}")
        print(f"Landmark ground truth pose: {lm_pose.T}\n Landmark guess pose: {p.T}")


        plt.scatter(robot_poses[:,0],robot_poses[:,1], color='blue', label='robot poses')
        plt.quiver(robot_poses[:,0],robot_poses[:,1], n[:,0], n[:,1], angles='xy', scale=2, color='red', alpha=0.5)
        plt.scatter(lm_pose[0],lm_pose[1], color='green', label='ground truth')
        plt.scatter(p[0], p[1], color='magenta', label='least squares guess')
        plt.legend()
        plt.show()

    def get_points(self, event):

        x = float(event.xdata)
        y = float(event.ydata)
        self.axes.scatter(x,y, color='black')
        self.fig.canvas.draw()
        self.route.append([x, y])

def main():
    ls = LsLines()

if __name__ == "__main__":
    main()


