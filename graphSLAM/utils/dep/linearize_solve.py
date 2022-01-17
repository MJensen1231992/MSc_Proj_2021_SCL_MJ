from types import new_class
import numpy as np
from numpy.linalg import matrix_rank
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve,lsqr
from scipy import linalg
from scipy.sparse.linalg import svds, eigs
from run_slam import *
from helper import *
from error import *
from math import isclose
import sympy
import pandas as pd

import seaborn as sns

def information_matrix(graph):

    #initialization of information matrix H and Bs
    # b is e.T*Omega*J
    # H is J.T*Omega*J

    H = np.zeros((len(graph.x), len(graph.x)))
    b = np.zeros(len(graph.x))
    b = np.expand_dims(b, axis=1)

    return H,b 

def linearize_solve(graph, lambdaH: float = 1.0, needToAddPrior=True, dcs=True):#iter: int = 1,
    
    phi = PHI
    
    H, b = information_matrix(graph)
    
    
    for edge in graph.edges:
        
        if edge.Type == 'P':
            
            fromIdx = graph.lut[edge.nodeFrom]
            toIdx = graph.lut[edge.nodeTo]

            if needToAddPrior: #May need to add prior to fix initial location!
                fixArray = np.array([1e8,1e8,1e8])
                H[fromIdx:fromIdx + 3, fromIdx:fromIdx + 3] = H[fromIdx:fromIdx + 3, fromIdx:fromIdx + 3]+ fixArray * np.eye(3)
                needToAddPrior = False
           
            x_i = graph.x[fromIdx:fromIdx+3]
            x_j = graph.x[toIdx:toIdx+3]
            z_ij = edge.poseMeasurement
            
            omega_ij = edge.information
            
            

            error , A, B = pose_pose_constraints(x_i, x_j, z_ij)

            if dcs:
                s_ij = dynamic_covariance_scaling(error, phi)
                omega_ij = (s_ij**2)*omega_ij
               
            
            # print(f"error: \n{error}\n")
            b_i, b_j, H_ii, H_ij, H_ji, H_jj = calc_gradient_hessian(A,B,omega_ij, error, edgetype = 'P')

            #Adding them to H and b in respective places
            H, b = build_gradient_hessian(b_i, b_j, H_ii, H_ij, H_ji, H_jj, H, b, fromIdx, toIdx, edgetype='P')
            
        
        elif edge.Type == 'L':

            fromIdx = graph.lut[edge.nodeFrom]
            toIdx = graph.lut[edge.nodeTo]
            
            
            x_i = graph.x[fromIdx:fromIdx+3] #robot pose
            x_j = graph.x[toIdx:toIdx+2] # landmark pose
            z_ij = edge.poseMeasurement # measurement
            omega_ij = edge.information

            
            if graph.withoutBearing: 

                error , A, B = pose_landmark_constraints(x_i, x_j, z_ij)
               
                if dcs:
                   s_ij = dynamic_covariance_scaling(error, phi)
                   omega_ij = (s_ij**2)*omega_ij

                b_i, b_j, H_ii, H_ij, H_ji, H_jj = calc_gradient_hessian(A, B, omega_ij, error, edgetype = 'L')

                #adding them to H and b
                H, b = build_gradient_hessian(b_i, b_j, H_ii, H_ij, H_ji, H_jj, H, b, fromIdx, toIdx, edgetype='L')

            else: 

                error , A, B = pose_landmark_bearing_constraints(x_i, x_j, z_ij)
                
                if dcs:
                   s_ij = dynamic_covariance_scaling(error, phi)
                   omega_ij = (s_ij**2)*omega_ij

                b_i, b_j, H_ii, H_ij, H_ji, H_jj = calc_gradient_hessian(A, B, omega_ij, error, edgetype = 'LB')

                #adding them to H and b
                H, b = build_gradient_hessian(b_i, b_j, H_ii, H_ij, H_ji, H_jj, H, b, fromIdx, toIdx, edgetype='LB')
        
    
        elif edge.Type == 'G':
            
            fromIdx = graph.lut[edge.nodeFrom]
            toIdx = graph.lut[edge.nodeTo]

            x_i= graph.x[fromIdx:fromIdx+3]
            x_j = graph.x[toIdx:toIdx+2]

            z_ij = edge.poseMeasurement
            omega_ij = edge.information

            error , A, B = pose_gps_constraints(x_i, x_j, z_ij)

            if dcs:
               s_ij = dynamic_covariance_scaling(error, phi)
               omega_ij = (s_ij**2)*omega_ij

            b_i, b_j, H_ii, H_ij, H_ji, H_jj = calc_gradient_hessian(A,B,omega_ij, error, edgetype = 'G')

            #adding them to H and b
            H, b = build_gradient_hessian(b_i, b_j, H_ii, H_ij, H_ji, H_jj,H,b, fromIdx,toIdx, edgetype='G')
        
        elif edge.Type == 'B':
            #print(f"from index: \n{fromIdx}\nto index: \n{toIdx}\n")
            fromIdx = graph.lut[edge.nodeFrom]
            toIdx = graph.lut[edge.nodeTo]

            x_i = graph.x[fromIdx:fromIdx+3] # robot pose
            x_j = graph.x[toIdx:toIdx+2] # x,y of landmark in noisy gis. comes from loader 

            z_ij = edge.poseMeasurement # brug meas til at regne x,y punkt ud fra least squares, eller initial gæt (a+t*n) måske r + cos og sin(h+theta)
            omega_ij = edge.information
            
            

            # if iter < 10:
            error, A, B = pose_landmark_bearing_only_constraints(x_i, x_j, z_ij)
            # else:
            #     error, A, B = pose_landmark_constraints(x_i, x_j, z_ij)
            
            if dcs:
               s_ij = dynamic_covariance_scaling(error, phi)
               omega_ij = (s_ij**2)*omega_ij
            
            b_i, b_j, H_ii, H_ij, H_ji, H_jj = calc_gradient_hessian(A, B, omega_ij, error, edgetype='B')
           
            H, b = build_gradient_hessian(b_i, b_j, H_ii, H_ij, H_ji, H_jj,H,b, fromIdx, toIdx, edgetype='B') 
            
    #print(f"iteration {iter}")
   
    H_damp = (H+lambdaH*np.eye(H.shape[0]))
    H_sparse = csr_matrix(H_damp)
    sparse_dxstar = spsolve(H_sparse,-b)
    dX = sparse_dxstar.squeeze()#
    
    
    return dX, H_sparse, H, sparse_dxstar
    

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
    #Error functions from linearization
    e_xy = np.dot(np.dot(R_ij.T, R_i.T), t_j-t_i)-np.dot(R_ij.T, t_ij)

    e_ang = wrap2pi(wrap2pi(theta_j-theta_i)-theta_ij)
    e_full = np.vstack((e_xy,e_ang))
    

    #Jacobian of e_ij wrt. x_i
    A_11 = np.dot(-R_ij.T,R_i.T)
    A_12 = np.dot(np.dot(R_ij.T, dR_i.T), t_j-t_i)
    A_21_22 = np.array([0,0,-1])
    A_ij = np.vstack((np.hstack((A_11,A_12)),A_21_22))


    #Jacobian of e_ij wrt. x_j
    B_11 = np.dot(R_ij.T,R_i.T)
    B_12 = np.zeros((2,1),dtype=np.float64)
    B_21_22 = np.array([0,0,1])
    B_ij = np.vstack((np.hstack((B_11,B_12)),B_21_22))
    

    return e_full, A_ij, B_ij

def pose_landmark_lsq_constraint():
    pass

def pose_landmark_triang_constraint():
    pass

def pose_landmark_bearing_only_constraints(x,l,z):
    #Will receive x,y position of landmark from least squares or triangulation

    z_bearing = z
    x_j = l.reshape(2,1) #landmark posegæt 
    t_i = x[:2].reshape(2,1) # robot translation(x,y)
    theta_i = x[2]
    
    r_l_trans = (x_j-t_i)#+theta_i))
    r_l_angle = math.atan2(r_l_trans[1],r_l_trans[0])
   
    e_full = wrap2pi(wrap2pi((r_l_angle-theta_i))-z_bearing)
    
    #print(f"Bearing:\n{z_bearing}\nlandmark:\n{x_j}\nrobot trans:\n{t_i}\nheading:\n{theta_i}\nrobot landmark trans:\n{r_l_trans}\nrobot landmark angle:\n{r_l_angle}\n")
    # if not isclose(etest, e_full, rel_tol = 1e-5, abs_tol=0.0):
    #print(f"e_full bearing: \n{e_full}\n e_test bearing: \n{etest}\n JK: \n{J_k}\nheading:\n{theta_i} landmark:\n{x_j}\n trans:\n{t_i}\n")
   
    # else:
    #     #print('hej')
   

    r = ((x_j[0]-t_i[0])**2+(x_j[1]-t_i[1])**2)# distance between poses squared, denominator of jacobians
    
    A_ij = np.hstack((-(t_i[1]-x_j[1])/r, (t_i[0]-x_j[0])/r, -1.0)).reshape(1,3)
    B_ij = np.hstack(((t_i[1]-x_j[1])/r,-(t_i[0]-x_j[0])/r)).reshape(1,2)

    #print(f"Aij:\n{A_ij}\nAnew:\n{J_r}\nBij:\n{B_ij}\nBnew:\n{J_l}")
    #  #testangle = wrap2pi(z-r_l_angle)
    # R_i = vec2trans(x)[:2,:2]
    # RL = np.dot(R_i.T, x_j-t_i)
    # z_hat = math.atan2(RL[1],RL[0])

    # etest = math.atan2(np.sin(wrap2pi(z_hat-z)),np.cos(wrap2pi(z_hat-z)))
    # RdotO = np.array([[0, -1], [1, 0 ]])
    # J_13 = np.dot(R_i.T,np.dot(RdotO,x_j))
    
    # J_r = np.hstack((-R_i.T,J_13))
    # J_l = R_i.T
    # J_k = np.dot(1/(np.dot(RL.T,RL)),np.array([-RL[1],RL[0]]).reshape(1,2))
    # J_r = J_k @ J_r
    # J_l = J_k @ J_l
    
    return e_full, A_ij, B_ij

def pose_landmark_bearing_constraints(x,l,z):
    
    angle_error = iterative_global_landmark_bearing_error(x,l,z)
    

    t_i = x[:2].reshape(2,1)#translation part robot translation
    x_j = l.reshape(2,1)#landmark pose
    z_il = z[:2].reshape(2,1) 
    theta_i = x[2]

    R_i = vec2trans(x)[:2, :2] # rotational part of robot pose
    dR_i = np.array([[-np.sin(theta_i), -np.cos(theta_i)],
                    [np.cos(theta_i), -np.sin(theta_i)]])

    e_xy = np.dot(R_i.T, x_j-t_i) - z_il #landmarkslam freiburg pdf
    e_full = np.vstack((e_xy,angle_error))

    r = ((t_i[0]-x_j[0])**2+(t_i[1]-x_j[1])**2)
    A_11 = -R_i.T
    A_12 = np.dot(dR_i.T, x_j-t_i)
    A_21_22 = np.array([-(t_i[1]-x_j[1])/r,(t_i[0]-x_j[0])/r,-1])
    #A_21_22 = A_21_22.reshape(1,3)
    A_ij = np.asfarray((np.vstack((np.hstack((A_11, A_12)),A_21_22))))

    B_11 = R_i.T
    B_12 = np.array([(t_i[1]-x_j[1])/r,-(t_i[0]-x_j[0])/r])
    B_12 = B_12.reshape(1,2)
    B_ij = np.vstack((B_11,B_12))
    

    return e_full, A_ij, B_ij

def pose_landmark_constraints(x, l, z):
    
    t_i = x[:2].reshape(2,1)#translation part robot translation
    x_l = l.reshape(2,1)#landmark pose
    z_il = z.reshape(2,1) 
    theta_i = x[2]

    R_i = vec2trans(x)[:2, :2] # rotational part of robot pose
    dR_i = np.array([[-np.sin(theta_i), -np.cos(theta_i)],
                    [np.cos(theta_i), -np.sin(theta_i)]])

    e_full = np.dot(R_i.T, x_l-t_i) - z_il #landmarkslam freiburg pdf
    
    #bearing only
    #e_bearing = atan2((x_l[x]-ti[y],x_l[x]-ti[x]) - robot_orientation-z_il
    #Jacobian A, B
    #Checked in maple ! nice
    A_21_22= -R_i.T
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

    e_full = np.dot(R_i.T, x_g-t_i) - z_il 

    #Jacobian A, B
    A_21_22 = -R_i.T
    A_23 = np.dot(dR_i.T, x_g-t_i)
    A_ij = np.hstack((A_21_22, A_23))

    B_ij = R_i.T

    return e_full, A_ij, B_ij

