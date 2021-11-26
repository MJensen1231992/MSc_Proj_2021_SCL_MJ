import numpy as np
import itertools
import matplotlib.pyplot as plt
from math import cos, sin


class Triangulation:

    def __init__(self):
        pass

    def normalize_angle(self, angle):
        if angle < 0:
            angle += 2 * np.pi
            return angle
        else:
            return angle

    def triangulation(self, Xr, z_list):

        z_list = np.matrix(z_list)
 
        pairs = np.concatenate((Xr,z_list.T), axis=1)
    
        combinations = list(itertools.combinations(pairs, 2))
        n_pairs = len(combinations)

        # n_stack = []
        # P_stack = []
        # denom_stack = []
        Xl = []
        counter = 0
        
        for i, d in enumerate(combinations):
            
            if i > 0:
                xl_old = xl_new

            # Orientation World RF
            thi = d[0][0,2]
            thj = d[1][0,2]
            
            # Bearing Local RF
            psii = d[0][0,3]
            psij = d[1][0,3]
            
            # Robot position i World RF
            xi = d[0][0,0]
            xj = d[1][0,0]

            # Robot position j World RF
            yi = d[0][0,1]
            yj = d[1][0,1]

            # P_stack.append(np.array([[xi],[yi], [xj], [yj]]))

            si = sin(psii); ci = cos(psii)
            sj = sin(psij); cj = cos(psij)

            xl = (xi*si*cj - xj*sj*ci + (yj-yi)*ci*cj)/(si*cj - sj*ci)
            yl = (yj*si*cj - yi*sj*ci + (xi-xj)*si*sj)/(si*cj - sj*ci)

            xl_new = np.array([xl,yl])

            if i > 0:
                distance = np.linalg.norm(np.array(xl_old) - xl_new)
                if distance < 20:
                    Xl.append([xl, yl])
                    # plt.scatter(xl,yl, color='cyan', alpha=0.5)
                    counter += 1
            elif i == 0:
                Xl.append([xl, yl])
                counter += 1

            

        Xl = np.vstack(Xl)
        Xl = np.reshape(Xl, (counter,2))
        
        Xl = Xl.mean(0)
        # plt.scatter(Xl[0], Xl[1], color='green')
        # plt.show()

        #     denom_stack.append([1/(si*cj-sj*ci), 1/(si*cj-sj*ci)])

        #     nom = np.array([[si*cj, -ci*cj, -sj*ci, ci*cj],
        #                     [si*sj, -sj*ci, -si*sj, si*cj]])
        #     n_stack.append(nom)

        # P = np.vstack(P_stack)

        # rng = np.arange(n_pairs*2)
        # A1 = np.zeros((n_pairs*2, n_pairs*2))
        # A1[rng, rng] = list(itertools.chain(*denom_stack))
        # A2 = np.vstack(n_stack)

        # A = A1 @ A2
        # Psi = A @ P

        # X = Psi[0::2]
        # Y = Psi[1::2]
        # Xl = np.array([X.mean(), Y.mean()])

        return Xl







    

def main():
    # import random

    # radius = 200
    # rangeX = (0, 2500)
    # rangeY = (0, 2500)
    # qty = 100  # or however many points you want

    # Generate a set of all points within 200 of the origin, to be used as offsets later
    # There's probably a more efficient way to do this.
    # deltas = set()
    # for x in range(-radius, radius+1):
    #     for y in range(-radius, radius+1):
    #         if x*x + y*y <= radius*radius:
    #             deltas.add((x,y))

    # randPoints = []
    # excluded = set()
    # i = 0
    # while i<qty:
    #     x = random.randrange(*rangeX)
    #     y = random.randrange(*rangeY)
    #     if (x,y) in excluded: continue
    #     randPoints.append((x,y))
    #     i += 1
    #     excluded.update((x+dx, y+dy) for (dx,dy) in deltas)

    # # Robot poses in world coordinates
    # # X_vi = [x_vi, y_vi, phi_vi]
    # X_vi = np.array([3.5, 3, np.deg2rad(12.65694)])
    # # X_vj = [x_vj, y_vj, phi_vj]
    # X_vj = np.array([3.66055, 3.03605, np.deg2rad(3.43883)])

    # # Relative bearing measurements to landmark:
    # b_i = np.deg2rad(63.5016)
    # b_j = np.deg2rad(76.59988)

    # # Landmark gt position: (4.03699, 5.17944)

    # plt.figure()
    # ax = plt.gca()
    # plt.scatter(X_vi[0], X_vi[1], color='red', label='R_i')
    # plt.scatter(X_vj[0], X_vj[1], color='darkred', label='R_j')

    # dx = np.cos(b_i+X_vi[2])
    # dy = np.sin(b_i+X_vi[2])
    # plt.quiver(X_vi[0], X_vi[1], dx, dy, color='blue', angles='xy', scale_units='xy', scale=0.165, label='Bearing')

    # dx = np.cos(b_j+X_vj[2])
    # dy = np.sin(b_j+X_vj[2])
    # plt.quiver(X_vj[0], X_vj[1], dx, dy, color='blue', angles='xy', scale_units='xy', scale=0.2, label='Bearing')
    # # ax.set_xlim([1.5, 4.5])
    # # ax.set_ylim([0.5, 7.5])
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # plt.figure()
    # # Computing the intersection of the two lines
    
    # X_f = BearingOnly.line_intersection(X_vi, X_vj, b_i, b_j)
    # print(f'The coordinate of the landmark is: ({X_f})')
    # print(f'Error in %: {((4.03699-X_f[0])/4.03699+(5.17944-X_f[1])/5.17944)/2*100}')

    # plt.scatter(X_f[0], X_f[1], color='green', label='L')
    # plt.scatter(X_vi[0], X_vi[1], color='red', label='R_i')
    # plt.scatter(X_vj[0], X_vj[1], color='darkred', label='R_j')
    # plt.plot([X_vi[0], X_f[0], X_vj[0], X_f[0]], [X_vi[1], X_f[1], X_vj[1], X_f[1]], label='Intersection', color='blue')
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    pass

if __name__ == "__main__":
    main()
    
