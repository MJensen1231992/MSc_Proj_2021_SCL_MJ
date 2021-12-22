import numpy as np

class LeastSquares:

    def __init__(self):
        self.cj = 1
        
    def least_squares_klines(self, Xr, z_list):
        """Reconstruction of the landmark pose based on a least square approach.
           Method utilies line intersections of k lines. 

        Args:
            Xr ([nx3 vector]): [n Robot poses in world frame]
            z_list ([list]): [list of bearing measurements to the landmark]

        Returns:
            [vector 2x1]: [reconstructed landmark pose [x,y]]       
        """            
            
        lambda_dist = 2

        R = np.zeros((2,2))
        q = np.zeros((2,1))
        eye = np.eye(2)
    

        for x, y, th, z in zip(Xr[:,0], Xr[:,1], Xr[:,2], z_list):

            n = np.matrix([np.cos(z+th),np.sin(z+th)])           
            a = np.matrix([x+(n[0,0]*lambda_dist),y+(n[0,1]*lambda_dist)])

            R += (n.T @ n - eye) * self.cj
            q += (n.T @ n - eye) @ a.T * self.cj
        
        try:
            Xl = np.linalg.inv(R) @ q
        except:
            Xl = np.linalg.pinv(R) @ q

        
        return Xl