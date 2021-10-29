#from os import read, readlink
import warnings
import numpy as np
from numpy.linalg import inv
from collections import namedtuple
import matplotlib.pyplot as plt
#import scipy 
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from run_slam import *
from helper import *
#from scipy.optimize import least_squares

def information_matrix(graph):

    #initialization of information matrix H and Bs
    # b is e.T*Omega*J
    # H is J.T*Omega*J

    H = np.zeros((len(graph.x), len(graph.x)))
    b = np.zeros(len(graph.x))
    b = np.expand_dims(b, axis=1)

    return H,b 

def linearize_solve(graph, needToAddPrior=True):
    
    H, b = information_matrix(graph)
    
    for edge in graph.edges:
        
        if edge.Type == 'P':

            fromIdx = graph.lut[edge.nodeFrom]
            toIdx = graph.lut[edge.nodeTo]

            x_i = graph.x[fromIdx:fromIdx+3]
            x_j = graph.x[toIdx:toIdx+3]
            z_ij = edge.poseMeasurement
            omega_ij = edge.information

            error, A, B = pose_pose_constraints(x_i, x_j, z_ij)
            
            b_i = np.dot(np.dot(A.T,omega_ij), error).reshape(3,1)
            b_j = np.dot(np.dot(B.T,omega_ij), error).reshape(3,1)
            H_ii = np.dot(np.dot(A.T,omega_ij), A) 
            H_ij = np.dot(np.dot(A.T,omega_ij), B) 
            H_ji = np.dot(np.dot(B.T,omega_ij), A) 
            H_jj = np.dot(np.dot(B.T,omega_ij), B) 

            
            if needToAddPrior:
                H[fromIdx:fromIdx + 3, fromIdx:fromIdx + 3] = H[fromIdx:fromIdx + 3, fromIdx:fromIdx + 3] + 1000 * np.eye(3)
                needToAddPrior = False

            #adding them to H and b in respective places
            H[fromIdx:fromIdx+3, fromIdx:fromIdx+3] += H_ii
            H[fromIdx:fromIdx+3, toIdx:toIdx+3] += H_ij
            H[toIdx:toIdx+3, fromIdx:fromIdx+3] += H_ji
            H[toIdx:toIdx+3, toIdx:toIdx+3] += H_jj

            b[fromIdx:fromIdx+3] += b_i
            b[toIdx:toIdx+3] += b_j
            
            #May need to add prior to fix initial location!
            

        elif edge.Type == 'L':

            fromIdx = graph.lut[edge.nodeFrom]
            toIdx = graph.lut[edge.nodeTo]

            x_l = graph.x[fromIdx:fromIdx+3]
            z_ij = edge.poseMeasurement
            omega_ij = edge.information

            if graph.withoutBearing: 
                l = graph.x[toIdx:toIdx+2]
                error , A, B = pose_landmark_constraints(x_l, l, z_ij)
                b_i = np.dot(np.dot(A.T,omega_ij), error).reshape(3,1)
                b_j = np.dot(np.dot(B.T,omega_ij), error).reshape(2,1)
                H_ii = np.dot(np.dot(A.T,omega_ij), A) 
                H_ij = np.dot(np.dot(A.T,omega_ij), B) 
                H_ji = np.dot(np.dot(B.T,omega_ij), A)  
                H_jj = np.dot(np.dot(B.T,omega_ij), B) 

                #adding them to H and b

                H[fromIdx:fromIdx+3, fromIdx:fromIdx+3] += H_ii
                H[fromIdx:fromIdx+3, toIdx:toIdx+2] += H_ij
                H[toIdx:toIdx+2, fromIdx:fromIdx+3] += H_ji
                H[toIdx:toIdx+2, toIdx:toIdx+2] += H_jj

                b[fromIdx:fromIdx+3] += b_i
                b[toIdx:toIdx+2] += b_j
            
            else:
                l = graph.x[toIdx:toIdx+2]
                error, A, B = pose_landmark_constraints_bearing(x_l, l, z_ij)
                b_i = np.dot(np.dot(A.T,omega_ij), error).reshape(3,1)
                b_j = np.dot(np.dot(B.T,omega_ij), error).reshape(2,1)
                H_ii = np.dot(np.dot(A.T,omega_ij), A) 
                H_ij = np.dot(np.dot(A.T,omega_ij), B) 
                H_ji = np.dot(np.dot(B.T,omega_ij), A)  
                H_jj = np.dot(np.dot(B.T,omega_ij), B) 

                #adding them to H and b

                H[fromIdx:fromIdx+3, fromIdx:fromIdx+3] += H_ii
                H[fromIdx:fromIdx+3, toIdx:toIdx+3] += H_ij
                H[toIdx:toIdx+3, fromIdx:fromIdx+3] += H_ji
                H[toIdx:toIdx+3, toIdx:toIdx+3] += H_jj

                b[fromIdx:fromIdx+3] += b_i
                b[toIdx:toIdx+3] += b_j

        elif edge.Type == 'G':
            
            fromIdx = graph.lut[edge.nodeFrom]
            toIdx = graph.lut[edge.nodeTo]
            x_g= graph.x[fromIdx:fromIdx+3]
            g = graph.x[toIdx:toIdx+2]
            z_ij = edge.poseMeasurement
            omega_ij = edge.information

            

            #print(f"l from linearize{g}")
            error , A, B = pose_gps_constraints(x_g, g, z_ij)
            b_i = np.dot(np.dot(A.T,omega_ij), error).reshape(3,1)
            b_j = np.dot(np.dot(B.T,omega_ij), error).reshape(2,1)
            H_ii = np.dot(np.dot(A.T,omega_ij), A) 
            H_ij = np.dot(np.dot(A.T,omega_ij), B) 
            H_ji = np.dot(np.dot(B.T,omega_ij), A)  
            H_jj = np.dot(np.dot(B.T,omega_ij), B) 

            #adding them to H and b

            H[fromIdx:fromIdx+3, fromIdx:fromIdx+3] += H_ii
            H[fromIdx:fromIdx+3, toIdx:toIdx+2] += H_ij
            H[toIdx:toIdx+2, fromIdx:fromIdx+3] += H_ji
            H[toIdx:toIdx+2, toIdx:toIdx+2] += H_jj

            b[fromIdx:fromIdx+3] += b_i
            b[toIdx:toIdx+2] += b_j
            
    is_pos_def(H)
    H_sparse = csr_matrix(H)

    sparse_dxstar = spsolve(H_sparse,-b)
    dxstar_squeeze = sparse_dxstar.squeeze()
    
    return dxstar_squeeze, H_sparse, H, sparse_dxstar

def pose_pose_constraints(xi,xj,zij):
    
    #Linearization of pose pose constraints

    #angles
    theta_i = xi[2]
    theta_j = xj[2]
    theta_ij = zij[2]

    #translation part
    t_i = xi[:2].reshape(2,1)
    t_j = xj[:2].reshape(2,1)
    t_ij = zij[:2].reshape(2,1)

    #rotational part
    R_i = vec2trans(xi)[:2, :2]
    R_ij = vec2trans(zij)[:2, :2]

    dR_i = np.array([[-np.sin(theta_i), -np.cos(theta_i)],
              [np.cos(theta_i), -np.sin(theta_i)]])

    #from appendix tutorial on graphbased slam.
    e_xy = np.dot(np.dot(R_ij.T, R_i.T), t_j-t_i)-np.dot(R_ij.T, t_ij)  
    e_ang = theta_j - theta_i - theta_ij 
    e_full = np.vstack((e_xy,e_ang))

    A_11 = np.dot(-R_ij.T,R_i.T)
    A_12 = np.dot(np.dot(R_ij.T, dR_i.T), t_j-t_i)
    A_21_22 = np.array([0,0,-1])
    A_ij = np.vstack((np.hstack((A_11,A_12)),A_21_22))

    B_11 = np.dot(R_ij.T,R_i.T)
    B_12 = np.zeros((2,1),dtype=np.float64)
    B_21_22 = np.array([0,0,1])
    B_ij = np.vstack((np.hstack((B_11,B_12)),B_21_22))
    
    return e_full, A_ij, B_ij

    
def pose_landmark_constraints_bearing(x,l,z):
    #Linearization of pose pose constraints

    #angles
    theta_i = x[2]
    theta_j = l[2]
    theta_ij = z[2]

    #translation part
    t_i = x[:2].reshape(2,1)
    t_j = l[:2].reshape(2,1)
    t_ij = z[:2].reshape(2,1)

    #rotational part
    R_i = vec2trans(x)[:2, :2]
    R_ij = vec2trans(z)[:2, :2]


    dR_i = np.array([[-np.sin(theta_i), -np.cos(theta_i)],
              [np.cos(theta_i), -np.sin(theta_i)]])

    #from appendix tutorial on graphbased slam.
    e_xy = np.dot(np.dot(R_ij.T, R_i.T), t_j-t_i)-np.dot(R_ij.T, t_ij)  
    e_ang = theta_j - theta_i - theta_ij 
    e_full = np.vstack((e_xy,e_ang))

    A_11 = np.dot(-R_ij.T,R_i.T)
    A_12 = np.dot(np.dot(R_ij.T, dR_i.T), t_j-t_i)
    A_21_22 = np.array([0,0,-1])
    A_ij = np.vstack((np.hstack((A_11,A_12)),A_21_22))


    B_11 = np.dot(R_ij.T,R_i.T)
    B_12 = np.zeros((2,1),dtype=np.float64)
    B_21_22 = np.array([0,0,1])
    B_ij = np.vstack((np.hstack((B_11,B_12)),B_21_22))
    
    return e_full, A_ij, B_ij

def pose_landmark_constraints(x, l, z):
    
    t_i = x[:2].reshape(2,1)#translation part robot translation
    x_l = l.reshape(2,1)#landmark pose
    z_il = z.reshape(2,1) 

    theta_i = x[2]

    R_i = vec2trans(x)[:2, :2] # rotational part
    dR_i = np.array([[-np.sin(theta_i), -np.cos(theta_i)],
                     [np.cos(theta_i), -np.sin(theta_i)]])

    e_full = np.dot(R_i.T, x_l-t_i) - z_il #landmarkslam freiburg pdf
    #bearing only
    #e_bearing = atan2((x_l[x]-ti[y],x_l[x]-ti[x]) - robot_orientation-z_il
    #Jacobian A, B
    A_21_22 = -R_i.T
    A_23 = np.dot(dR_i.T, x_l-t_i)
    A_ij = np.hstack((A_21_22, A_23))

    B_ij = R_i.T

    return e_full, A_ij, B_ij

def pose_gps_constraints(x, g, z):
    
    t_i = x[:2].reshape(2,1)#translation part robot translation
    x_g = g.reshape(2,1)#landmark pose
    z_il = z.reshape(2,1) 
    theta_i = x[2]

    R_i = vec2trans(x)[:2, :2] # rotational part
    dR_i = np.array([[-np.sin(theta_i), -np.cos(theta_i)],
                     [np.cos(theta_i), -np.sin(theta_i)]])

    e_full = np.dot(R_i.T, x_g-t_i) - z_il #landmarkslam freiburg pdf
    #bearing only
    #e_bearing = atan2((x_l[x]-ti[y],x_l[x]-ti[x]) - robot_orientation-z_il
    #Jacobian A, B
    
    A_21_22 = -R_i.T
    A_23 = np.dot(dR_i.T, x_g-t_i)
    A_ij = np.hstack((A_21_22, A_23))

    B_ij = R_i.T

    return e_full, A_ij, B_ij