from primitives import Point, Line

import numpy as np
from scipy.stats import chi2
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import sys

def get_ellipse(mean, w, v, color: tuple = 'red', p: float = 0.95):
    angle = np.arctan2(v[1,0], v[0,0])
    angle = np.rad2deg(angle)

    val = chi2(2).ppf(p)
    ellipse = Ellipse(mean, np.sqrt(val * w[0]) * 2, np.sqrt(val * w[1]) * 2, angle=angle, fill=False, color=color)

    return ellipse

class Landmark:
    def __init__(self, pose: list = None):
        self.poses = []

        if pose != None:
            self.poses = [pose]

    def add_pose(self, pose):
        self.poses += [pose]
        self.matrix = np.array([[pose.x, pose.y] for pose in self.poses])

    def get_pose(self):

        mean = np.average(self.matrix, axis=0)
        return Point(*mean)

    def get_covariance(self):

        return np.cov(self.matrix, rowvar=False)

    def plot(self, label_name: str = None, color_mean: tuple = None, color_samples: tuple = None, color_contour: tuple = None):
        mean = np.average(self.matrix, axis=0)

        if color_mean != None:
            label = None
            if label_name != None:
                label = f'{label_name} Mean'

            plt.plot(mean[0], mean[1], 'o', color=color_mean, label=label)

        if color_samples != None:
            label = None
            if label_name != None:
                label = f'{label_name} Samples'

            plt.plot(self.matrix[:,0], self.matrix[:,1], 'x', color=color_samples, label=label)


        if color_contour != None:
            ax = plt.gca()

            cov = self.get_covariance()
            w, v = np.linalg.eig(cov)

            ellipse = get_ellipse(mean, w, v, color=color_contour)
            ax.add_patch(ellipse)

    def distance_point(self, point, plot: bool = False, default_variance: float = 1.0) -> (float):
        u = np.array([point.x, point.y])
        v = np.average(self.matrix, axis=0)

        V = np.array([[default_variance, 0], [0, default_variance]])
        if len(self.poses) > 1:
            if np.linalg.cond(V) >= 1/sys.float_info.epsilon:
                V = np.cov(self.matrix, rowvar=False)

        Vi = np.linalg.inv(V)

        d = distance.mahalanobis(u, v, Vi)

        if np.isnan(d):
            print(f'ERROR: {d}\n v: {v}\n u: {u}\n V: {V}\n Vi: {Vi}')
            return 100

        return d

    def distance_line(self, line, plot: bool = False) -> (float, Point):
        mean = np.average(self.matrix, axis=0)
        cov = self.get_covariance()
        w, v = np.linalg.eig(cov)

        if w[0] == 0:
            y = line.get_y(mean[0])
            y_ = (y - mean[1]) / np.sqrt(w[1])

            line_ = Line(Point(0, y_), Point(1, y_))

            dist = y_

            p = Point(mean[0], y)
            p_ = Point(0, y_)

            pts = np.zeros(self.matrix.shape)
            pts[:,1] = (self.matrix[:,1] - mean[1]) / np.sqrt(w[1])

        elif w[1] == 0:
            x = line.get_x(mean[1])
            x_ = (x - mean[0]) / np.sqrt(w[0])

            line_ = Line(Point(x_, 0), Point(x_, 1))

            dist = x_

            p = Point(x, mean[1])
            p_ = Point(x_, 0)

            pts = np.zeros(self.matrix.shape)
            pts[:,0] = (self.matrix[:,0] - mean[0]) / np.sqrt(w[0])

        else:
            pt_a = np.dot(np.array([line.a.x, line.a.y]) - mean, v) / np.sqrt(w)
            pt_b = np.dot(np.array([line.b.x, line.b.y]) - mean, v) / np.sqrt(w)

            line_ = Line(Point(*pt_a), Point(*pt_b))
            dist, p_ = line_.get_distance(Point(0, 0))

            pts = np.dot(self.matrix - mean, v) / np.sqrt(w)
            p = Point(*np.array(np.dot([p_.x, p_.y] * np.sqrt(w), v.transpose()) + mean))

        if plot:
            mean_ = np.average(pts, axis=0)

            plt.figure()
            plt.title('Undistorted')
            ax = plt.gca()
            ax.set_aspect(1)

            plt.plot(self.matrix[:,0], self.matrix[:,1], 'x', color='orange', label='Sample points')
            plt.plot(mean[0], mean[1], 'o', color='red', label='Sample mean')
            
            if w[0] != 0 and w[1] != 0:
                ax.add_patch(get_ellipse(mean, w, v))

            plt.plot([line.a.x, line.b.x], [line.a.y, line.b.y], 'o', color='green')
            plt.axline([line.a.x, line.a.y], [line.b.x, line.b.y], color='green', label='line')

            plt.plot(p.x, p.y, 'o', color='blue', label='Closest point')
            plt.legend()

            plt.figure()
            plt.title('Distorted')
            ax = plt.gca()
            ax.set_aspect(1)

            plt.plot(pts[:,0], pts[:,1], 'x', color='orange', label='Sample points')
            plt.plot(mean_[0], mean_[1], 'o', color='red', label='Sample mean')

            if w[0] != 0 and w[1] != 0:
                ax.add_patch(get_ellipse(mean_, *np.linalg.eig(np.cov(pts, rowvar=False))))
                plt.plot([line_.a.x, line_.b.x], [line_.a.y, line_.b.y], 'o', color='green')
                
            plt.axline([line_.a.x, line_.a.y], [line_.b.x, line_.b.y], color='green', label='line')

            plt.plot(p_.x, p_.y, 'o', color='blue', label='Closest point')
            plt.legend()

        return dist, p

if __name__ == '__main__':
    import random

    positions = [
        np.array([5, 3]),
        np.array([5, 7]),
        np.array([8, 9]),
    ]

    covariances = [
        np.array([[0.4, -0.15], [-0.15, 0.07]]),
        np.array([[0.1, 0], [0, 0.6]]),
        np.array([[0.3, 0.1], [0.1, 0.3]]),
    ]

    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    colors_samples = [(0.75, 0.45, 0), (0, 0.75, 0.45), (0.45, 0, 0.75)]

    ns = [100, 100, 200]

    landmarks = []

    plt.figure(0)

    for n, pos, cov, color, color_samples in zip(ns, positions, covariances, colors, colors_samples):
        measurements = [pos for _ in range(n)]
        measurements += np.random.multivariate_normal([0, 0], cov, n)

        landmark = Landmark()

        for pose in measurements:
            landmark.add_pose(Point(*pose))

        landmarks.append(landmark)
        landmark.plot(color_mean=color, color_samples=color_samples, color_contour=color)

    line = Line(Point(0, 0), Point(1, 1))
    line.plot()

    ax = plt.gca()
    ax.set_aspect(1)

    closest_points = [landmark.distance_line(line, plot=False) for landmark in landmarks]

    plt.figure(0)

    for landmark, (d, p) in zip(landmarks, closest_points):
        mean = landmark.get_pose()
        plt.plot([mean.x, p.x], [mean.y, p.y], '--o', color='black')
        plt.plot(p.x, p.y, 's', color='black')
        plt.text(p.x, p.y, f'{d:.2f}')
    
    plt.legend()
    plt.show()
