from types import new_class
import numpy as np
from numpy.linalg import matrix_rank
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy import linalg
from scipy.sparse.linalg import svds, eigs
from run_slam import *
from helper import *
from error import *

def information_matrix(graph):

    #initialization of information matrix H and Bs
    # b is e.T*Omega*J
    # H is J.T*Omega*J

    H = np.zeros((len(graph.x), len(graph.x)))
    b = np.zeros(len(graph.x))
    b = np.expand_dims(b, axis=1)

    return H,b 

def linearize_solve(graph, lambdaH: float = 1.0, needToAddPrior=True, dcs=True):
    dcsp_arr = []
    dcsb_arr = []
    phi = 0.2
    H, b = information_matrix(graph)
    
    
    for edge in graph.edges:
        
        if edge.Type == 'P':
            #print('entered P')
            fromIdx = graph.lut[edge.nodeFrom]
            toIdx = graph.lut[edge.nodeTo]

            x_i = graph.x[fromIdx:fromIdx+3]
            x_j = graph.x[toIdx:toIdx+3]
            z_ij = edge.poseMeasurement
            omega_ij = edge.information

            error , A, B = pose_pose_constraints(x_i, x_j, z_ij)
            #iter_error = iterative_global_poseerror(x_i, x_j, z_ij) check floating point diff 

            #print(f"error:{error} and itererror{iter_error}")

            if dcs:
                s_ij = dynamic_covariance_scaling(error, phi)
                omega_ij = (s_ij**2)*omega_ij
               # dcsp_arr.append(s_ij)
            #print(f"poseError:\n{error}\npose A:\n{A}\npose B:\n{B}\n")
            #print(f"prior:{needToAddPrior}")
            if needToAddPrior: #May need to add prior to fix initial location!
                H[fromIdx:fromIdx + 3, fromIdx:fromIdx + 3] = H[fromIdx:fromIdx + 3, fromIdx:fromIdx + 3] + 10 * np.eye(3)
                needToAddPrior = False

            b_i, b_j, H_ii, H_ij, H_ji, H_jj = calc_gradient_hessian(A,B,omega_ij, error, edgetype = 'P')

            #Adding them to H and b in respective places
            H, b = build_gradient_hessian(b_i, b_j, H_ii, H_ij, H_ji, H_jj, H, b, fromIdx, toIdx, edgetype='P')
            
        
        elif edge.Type == 'L':
            #print('entered landmark xy')
            fromIdx = graph.lut[edge.nodeFrom]
            toIdx = graph.lut[edge.nodeTo]
            
            #x_i is robot pose
            x_i = graph.x[fromIdx:fromIdx+3]
            z_ij = edge.poseMeasurement
            omega_ij = edge.information

            
            if graph.withoutBearing: 

                x_j = graph.x[toIdx:toIdx+2]

                error , A, B = pose_landmark_constraints(x_i, x_j, z_ij)

                if dcs:
                   s_ij = dynamic_covariance_scaling(error, phi)
                   omega_ij = (s_ij**2)*omega_ij


                b_i, b_j, H_ii, H_ij, H_ji, H_jj = calc_gradient_hessian(A, B, omega_ij, error, edgetype = 'L')

                #adding them to H and b
                H, b = build_gradient_hessian(b_i, b_j, H_ii, H_ij, H_ji, H_jj, H, b, fromIdx, toIdx, edgetype='L')

            else: 

                x_j = graph.x[toIdx:toIdx+2]
                
                error , A, B = pose_landmark_bearing_constraints(x_i, x_j, z_ij)

                if dcs:
                   s_ij = dynamic_covariance_scaling(error, phi)
                   omega_ij = (s_ij**2)*omega_ij

                b_i, b_j, H_ii, H_ij, H_ji, H_jj = calc_gradient_hessian(A, B, omega_ij, error, edgetype = 'LB')

                #adding them to H and b
                H, b = build_gradient_hessian(b_i, b_j, H_ii, H_ij, H_ji, H_jj, H, b, fromIdx, toIdx, edgetype='LB')
        
         


        elif edge.Type == 'G':
            #print('entered gps')
            fromIdx = graph.lut[edge.nodeFrom]
            toIdx = graph.lut[edge.nodeTo]

            x_g= graph.x[fromIdx:fromIdx+3]
            g = graph.x[toIdx:toIdx+2]

            z_ij = edge.poseMeasurement
            omega_ij = edge.information

            error , A, B = pose_gps_constraints(x_g, g, z_ij)
            #print(f"x_g:\n{x_g}\nb:\n{g}\nmeas:\n{z_ij}\ninfo:\n{omega_ij}\ngpserror:\n{error}\ngps A:\n{A}\ngps b:\n{B}\n")
            if dcs:
               s_ij = dynamic_covariance_scaling(error, phi)
               omega_ij = (s_ij**2)*omega_ij
               #print(f"hej g{omega_ij}")

            b_i, b_j, H_ii, H_ij, H_ji, H_jj = calc_gradient_hessian(A,B,omega_ij, error, edgetype = 'G')

            #adding them to H and b
            H, b = build_gradient_hessian(b_i, b_j, H_ii, H_ij, H_ji, H_jj,H,b, fromIdx,toIdx, edgetype='G')
        
        elif edge.Type == 'B':
            
            fromIdx = graph.lut[edge.nodeFrom]
            toIdx = graph.lut[edge.nodeTo]

            x_b = graph.x[fromIdx:fromIdx+3] # robot pose
            b_g = graph.x[toIdx:toIdx+2] # x,y of landmark in noisy gis. 

            z_ij = edge.poseMeasurement # brug meas til at regne x,y punkt ud fra least squares, eller initial gæt (a+t*n) måske r + cos og sin(h+theta)
            omega_ij = edge.information
            #print(f"Robot pose from(linearize func):\n{x_b}\nBearing(linearize func)\n{z_ij}\n")
            

            #b_gx = np.array([xguess,yguess])#,dtype=np.float64)
            #print(f"Landmark guess: {b_g}\n")

            # print(f"shape GIS:\n{np.shape(b_g)}\n")
            # print(f"GIS\n{b_g}\n")
            # print(f"guess landmark\n{b_gx}\n")
            # print(f"landmark shape guess:\n{np.shape(b_gx)}\n")
            
            error, A, B = pose_landmark_bearing_only_constraints(x_b, b_g, z_ij)
            
            #print(f"error:\n{error}\n")
            # print(f"x_b:\n{x_b}\nb:\n{b_g}\nmeas:\n{z_ij}\nbearingerror:\n{error}\n\ninfo:\n{omega_ij}\nbearingerror:\n{error}\nbearing A:\n{A}\nbearing b:\n{B}\n")
            #print(f"bearingerror:\n{error}\nbearing A:\n{A}\nbearing b:\n{B}\ninfo:\n{omega_ij}\n")
            if dcs:
               s_ij = dynamic_covariance_scaling(error, phi)
               omega_ij = (s_ij**2)*omega_ij
               #dcsp_arr.append(s_ij)
               #print(f"hej b{omega_ij}")
            
            b_i, b_j, H_ii, H_ij, H_ji, H_jj = calc_gradient_hessian(A, B, omega_ij, error, edgetype='B')
            #print(f"H: \n{H_ij}\nbi: \n{b_i}\n bj: \n{b_j}")
            #adding them to H and b
            H, b = build_gradient_hessian(b_i, b_j, H_ii, H_ij, H_ji, H_jj,H,b, fromIdx, toIdx, edgetype='B') 
            
    
    H_damp = (H+lambdaH*np.eye(H.shape[0]))
    
    H_sparse = csr_matrix(H_damp)
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
    #Error functions from linearization
    e_xy = np.dot(np.dot(R_ij.T, R_i.T), t_j-t_i)-np.dot(R_ij.T, t_ij)
    e_ang = wrap2pi(wrap2pi(theta_j-theta_i)-theta_ij)
    #print(f"angle pose error:\n{e_ang}\n")
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
    #testangle = wrap2pi(z-r_l_angle)

    e_full = wrap2pi(wrap2pi((r_l_angle-theta_i))-z_bearing)
    #print(f"Bearing:\n{z_bearing}\nlandmark:\n{x_j}\nrobot trans:\n{t_i}\nheading:\n{theta_i}\nrobot landmark trans:\n{r_l_trans}\nrobot landmark angle:\n{r_l_angle}\n")
    #print(f"e_full bearing:\n{e_full}\n")
   # r = ((t_i[0]-x_j[0])**2+(t_i[1]-x_j[1])**2) # distance between poses squared, denominator of jacobians

    r = ((x_j[0]-t_i[0])**2+(x_j[1]-t_i[1])**2)
    A_ij = np.hstack((-(t_i[1]-x_j[1])/r, (t_i[0]-x_j[0])/r, -1.0)).reshape(1,3)
    B_ij = np.hstack(((t_i[1]-x_j[1])/r,-(t_i[0]-x_j[0])/r)).reshape(1,2)

    #A_ij = np.hstack(((x_j[1]-t_i[1])/r, (x_j[0]-t_i[0])/r, -1.0)).reshape(1,3)
    #B_ij = np.hstack(((t_i[1]-x_j[1])/r,(t_i[0]-x_j[0])/r)).reshape(1,2)
    #print(f"Aij:\n{A_ij}\nAdeans:\n{A_ijj}\nBij:\n{B_ij}\nBdeans:\n{B_ijj}\nrobotpose:\n{t_i}\nlandmark pose:\n{x_j}\n")
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

