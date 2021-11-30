import numpy as np

from lib.bearing_utils.triangulation import Triangulation


class LeastSquares:

    def __init__(self):
        self.cj = 1
        
    def least_squares_klines(self, Xr, z_list, bias: bool=False):
        """Reconstruction of the landmark pose based on a least square approach.
           Method utilies line intersections of k lines. 

        Args:
            Xr ([nx3 vector]): [n Robot poses in world frame]
            z_list ([list]): [list of bearing measurements to the landmark]

        Returns:
            [vector 2x1]: [reconstructed landmark pose [x,y]]       
        """            

        if bias:
            tri = Triangulation()
            l = 0.8
            s = np.matrix(tri.triangulation(Xr, z_list))
            

        R = np.zeros((2,2))
        q = np.zeros((2,1))
        eye = np.eye(2)
    

        for x, y, th, z in zip(Xr[:,0], Xr[:,1], Xr[:,2], z_list):
  
            n = np.matrix([np.cos(z),np.sin(z)])           
            a = np.matrix([x+(n[0,0]*2),y+(n[0,1]*2)])
            
            if bias:
                R += (n.T @ n - eye) + l * eye
                q += (n.T @ n - eye) @ a.T + l * s.T
            else:
                R += (n.T @ n - eye) * self.cj
                q += (n.T @ n - eye) @ a.T * self.cj
        
        try:
            Xl = np.linalg.inv(R) @ q
        except:
            Xl = np.linalg.pinv(R) @ q

        return Xl

