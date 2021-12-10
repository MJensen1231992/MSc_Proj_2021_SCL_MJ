import numpy as np
import itertools
from math import cos, sin
# from scipy.spatial import distance

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
        """Calculating the location of a landmark using robot poses and relative bearing measurements


        Args:
            Xr (Matrix nx3): All robot poses with bearing measurements to a specific landmark ID
            z_list (List 1xn): List of bearings associated with the poses. Each pose can only have one bearing

        Returns:
            Xl (Matrix 2x1): Landmark location [x,y]^T
        """        
        z_list = np.matrix(z_list)
 
        pairs = np.concatenate((Xr,z_list.T), axis=1)

        # Computing an array of all possible combinations of point pairs where i != i
        combinations = list(itertools.combinations(pairs, 2))
        n_pairs = len(combinations)

        xl_old = []
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

            si = sin(psii); ci = cos(psii)
            sj = sin(psij); cj = cos(psij)

            xl = (xi*si*cj - xj*sj*ci + (yj-yi)*ci*cj)/(si*cj - sj*ci)
            yl = (yj*si*cj - yi*sj*ci + (xi-xj)*si*sj)/(si*cj - sj*ci)

            xl_new = np.array([xl,yl])

            if i == 0:
                Xl.append([xl, yl])
                counter += 1
            elif i > 0:
                distance = np.linalg.norm(xl_new - xl_old)
                if distance < 5:
                    Xl.append([xl, yl])
                    counter += 1


        Xl = np.vstack(Xl)
        Xl = np.reshape(Xl, (counter,2))
        
        Xl = Xl.mean(0)

        return np.array([[Xl[0]], [Xl[1]]])

    

def main():
    pass

if __name__ == "__main__":
    main()
    
