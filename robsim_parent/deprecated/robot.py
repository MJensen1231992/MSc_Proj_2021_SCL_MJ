from primitives.point import Point

from math import sin, cos, log, exp, pi
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from random import normalvariate, choice
import time
import numpy as np

class SimpleRobot:
    def __init__(self, dt: float, v_min: float = None, v_max: float = None, omega_min: float = None, omega_max: float = None, logistic_max: bool = False, a_mean: float = 0, a_std: float = None, alpha_mean: float = 0, alpha_std: float = None):
        self.dt = dt
        self.v_min = v_min
        self.v_max = v_max
        self.omega_min = omega_min
        self.omega_max = omega_max
        self.logistic_max = logistic_max

        self.alpha_mean = alpha_mean
        self.alpha_std = alpha_std
        if alpha_std == None:
            self.alpha_std = 0.5 * (abs(self.v_min) + abs(self.v_max))

        self.a_mean = a_mean
        self.a_std = a_std
        if a_std == None:
            self.a_std = 0.5 * (abs(self.v_min) + abs(self.v_max))

    def sample_control(self):
        return normalvariate(self.a_mean, self.a_std), normalvariate(self.alpha_mean, self.alpha_std)

    def propagate(self, state, a, alpha, steps):
        x, y, theta, v, omega, *_ = state

        hist = []
        for _ in range(steps):
            x, y, theta = \
                v * cos(theta) * self.dt + x, \
                v * sin(theta) * self.dt + y, \
                omega * self.dt + theta, \
            
            if self.logistic_max:
                x0 = log((self.v_max-self.v_min)/(v-self.v_min) - 1)
                v = (self.v_max-self.v_min) / (1 + exp(x0 - a * self.dt)) + self.v_min
                
                x0 = log((self.omega_max-self.omega_min)/(omega-self.omega_min) - 1)
                omega = (self.omega_max-self.omega_min) / (1 + exp(x0 - alpha * self.dt)) + self.omega_min
            
            else:
                v += a * self.dt
                omega += alpha * self.dt

            if self.v_min != None:
                v = max(self.v_min, v)

            if self.v_max != None:
                v = min(self.v_max, v)

            if self.omega_min != None:
                omega = max(self.omega_min, omega)
            
            if self.omega_max != None:
                omega = min(self.omega_max, omega)

            hist.append((x, y, theta, v, omega))

        return hist

if __name__ == '__main__':
    dt = 0.001
    robot = SimpleRobot(dt=dt, v_min=-1, v_max=2, omega_min=-2*pi, omega_max=2*pi, logistic_max=True)

    step_length = 1000
    steps = 10
    N = step_length * steps

    def update_vlines(h, x, ymin=None, ymax=None):
        seg_old = h.get_segments()
        if ymin is None:
            ymin = seg_old[0][0, 1]
        if ymax is None:
            ymax = seg_old[0][1, 1]

        seg_new = [np.array([[x, ymin], [x, ymax]])]

        h.set_segments(seg_new)

    def update_point(i, hist, pt, lns):
        pt[0].set_data(hist[i][0], hist[i][1])

        for ln in lns:
            update_vlines(ln, i * dt)

        return pt

    while True:
        x = (0, 0, 0, 0, 0)

        hist = [x]
        hist_input = [[0,0]]

        for i in range(steps):
            a, alpha = robot.sample_control()
            hist_input.extend([[a, alpha] for _ in range(step_length)])
            hist.extend(robot.propagate(x, a, alpha, step_length))
            x = hist[-1]

        hist = [[h[0], h[1], h[2], h[3], h[4], hi[0], hi[1]] for h, hi in zip(hist, hist_input)]

        titles = ['x', 'y', 'θ', 'v', 'ω', 'a', 'α']
        fig, axs = plt.subplots(nrows=2, ncols=4, constrained_layout=True)

        ax = axs[0, 0]
        ax.set_title('route')
        ax.plot([x[0] for x in hist], [x[1] for x in hist], '-')
        pt = ax.plot(hist[0][0], hist[0][1], 'x', color='black')

        lns = [ax.vlines(0, 0, 1) for ax in axs[0, 1:]]

        ani = animation.FuncAnimation(fig, update_point, N, fargs=(hist, pt, lns), interval=dt * 1000, blit=True)
        
        swaps = [(s + 1) * step_length for s in range(steps - 1)]

        for n in range(7):
            i = (n + 1) % 4
            j = (n + 1) // 4
            ax = axs[j, i]
            ax.set_title(titles[n])
            ax.plot([i * robot.dt for i in range(N + 1)], [x[n] for x in hist])

            if n == 3 or n == 4:
                for s in swaps:
                    ax.axvline(s * dt, ls='--')

            if n == 3:
                ax.plot([i * dt for i in range(N + 1)], [robot.v_min for _ in range(N + 1)], '--')
                ax.plot([i * dt for i in range(N + 1)], [robot.v_max for _ in range(N + 1)], '--')

            if n == 4:
                ax.plot([i * dt for i in range(N + 1)], [robot.omega_min for _ in range(N + 1)], '--')
                ax.plot([i * dt for i in range(N + 1)], [robot.omega_max for _ in range(N + 1)], '--')

        plt.show()