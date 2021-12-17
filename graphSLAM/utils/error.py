import numpy as np
from numpy.linalg import inv
from utils.helper import vec2trans, trans2vec, wrap2pi
import math


def iterative_global_poseerror(xi,xj,zij):
    #maybe delete. same as 
    x_i_inv_trans = inv(vec2trans(xi))
    x_j_trans = vec2trans(xj)
    z_ij_inv_trans = inv(vec2trans(zij))
    #Error from cyrill posegraph video 34:00
    pose_error = trans2vec(np.dot(z_ij_inv_trans,np.dot(x_i_inv_trans,x_j_trans)))
    pose_error[2] = wrap2pi(pose_error[2])

    #print(f"x_j pose:\n{xj}\n x_j trans pose:\n{x_j_trans}\n")
    return pose_error

def iterative_global_landerror(x,l,zij):
    
    z = np.expand_dims(zij,axis=1)
    R_i = vec2trans(x)[:2,:2]
    x_j = l.reshape(2,1)
    t_i = x[:2].reshape(2,1)
    
    land_error = np.dot(R_i.T,(x_j-t_i))-z
    
    return land_error

def iterative_global_landmark_bearing_error(x,l,zij):
    #land_error = iterative_global_landerror(x,l,zij)
    #z = np.expand_dims(zij,axis=1) # measurement
    #R_i = vec2trans(x)[:2,:2]
    z_bearing = zij[2]
    x_j = l.reshape(2,1)
    t_i = x[:2].reshape(2,1)
    theta_i = x[2]
    #robot_landmark_angle
    r_l_trans = (x_j-t_i)
    #r_l_trans = R_i.T @ (x_j-t_i)
    r_l_angle = math.atan2(r_l_trans[1],r_l_trans[0])
    #print(f"Robot landmark trans is:\n{r_l_trans}\n from pose:{t_i} to {x_j}\nRobot landmark angle is:\n{r_l_angle}\n")#x_j for landmark is:\n{x_j}\nRobot translation is:\n{t_i}\n


    
    land_bearing_error= wrap2pi(wrap2pi(r_l_angle-theta_i)-z_bearing)
   # print(f"not wrapped RL angle - meas:\n{(r_l_angle-theta_i)-z_bearing}\n")
   # print(f"wrapped bearing error:\n{land_bearing_error}\n")
    return land_bearing_error

def iterative_global_landmark_bearing_only_error(x,l,zij):
   
    z_bearing = zij
    x_j = l.reshape(2,1) #gæt 
    t_i = x[:2].reshape(2,1) # robot translation(x,y)
    theta_i = x[2]# robot heading

    #robot_landmark_angle
    r_l_trans = (x_j-t_i)
    #r_l_trans = np.dot(R_i.T,(x_j-t_i))
    r_l_angle = math.atan2(r_l_trans[1],r_l_trans[0])
    #print(f"Robot landmark trans is:\n{r_l_trans}\n from pose:{t_i} to {x_j}\nRobot landmark angle is:\n{r_l_angle}\n")#x_j for landmark is:\n{x_j}\nRobot translation is:\n{t_i}\n
    
    land_bearing_error= wrap2pi(wrap2pi((r_l_angle-theta_i))-z_bearing)
    #print(f"not wrapped RL angle - meas:\n{(r_l_angle-theta_i)-z_bearing}\n")
    # print(f"wrapped bearing error:\n{land_bearing_error}\n")
    return land_bearing_error


def compute_global_error(graph, noBearing: bool = False):
    
    err_Full = 0 
    err_Land = 0
    err_Pose = 0
    err_GPS = 0
    err_bearing = 0

    for edge in graph.edges:

        if edge.Type == 'P':
            
            fromIdx = graph.lut[edge.nodeFrom]
            toIdx = graph.lut[edge.nodeTo]

            x1 = graph.x[fromIdx:fromIdx+3]
            x2 = graph.x[toIdx:toIdx+3]

            z_12 = edge.poseMeasurement

            x1_inv_trans = inv(vec2trans(x1))
            x2_trans = vec2trans(x2)

            z_12_inv = inv(vec2trans(z_12))

            #Error from cyrill posegraph video 34:00
            err_Full += np.linalg.norm(trans2vec(np.dot(z_12_inv, np.dot(x1_inv_trans, x2_trans))))
            err_Pose += trans2vec(np.abs(np.dot(z_12_inv, np.dot(x1_inv_trans, x2_trans))))
            
            
        elif edge.Type == 'L':

            fromIdx = graph.lut[edge.nodeFrom]
            toIdx = graph.lut[edge.nodeTo]

            x = graph.x[fromIdx:fromIdx+3]
            l = graph.x[toIdx:toIdx+2]
            z = edge.poseMeasurement

            #x_inv_trans = inv(vec2trans(x))[:2, :2]
            if noBearing:
                R_xi = vec2trans(x)[:2, :2]
                l = l.reshape(2,1)
                t_i = x[:2]
                z = np.expand_dims(z,axis=1)

                #err_Full += np.linalg.norm(np.dot(x_inv_trans, l) - z)
                #err_Land += np.dot(R_xi.T,(l-t_i))-z
                err_Full += np.linalg.norm(np.dot(R_xi.T,(l-t_i))-z)
                err_Land += np.dot(R_xi.T,(l-t_i))-z
            else: 

                from linearize_solve import pose_landmark_bearing_constraints
                err,_,_ = pose_landmark_bearing_constraints(x,l,z)
                err_Full += np.linalg.norm(err)
                err_Land += err
            
            
        elif edge.Type == 'G':

            fromIdx = graph.lut[edge.nodeFrom]
            toIdx = graph.lut[edge.nodeTo]

            x = graph.x[fromIdx:fromIdx+3]
            gpsEdge = graph.x[toIdx:toIdx+2]
            z = edge.poseMeasurement
            #info_12 = edge.information
    
            #x_inv_trans = inv(vec2trans(x))[:2, :2]
            gpsEdge = np.expand_dims(gpsEdge,axis=1)
            R_xi = vec2trans(x)[:2, :2]

            t_i = x[:2]
            z = np.expand_dims(z,axis=1)
           
            # err_Full += np.linalg.norm(np.dot(x_inv_trans,gpsEdge) - z)
            # err_GPS += (np.dot(x_inv_trans,gpsEdge) - z)
            err_Full += np.linalg.norm(np.dot(R_xi.T,(gpsEdge-t_i))-z)
            err_GPS += np.dot(R_xi.T,(gpsEdge-t_i))-z

        elif edge.Type =='B':

            fromIdx = graph.lut[edge.nodeFrom]
            toIdx = graph.lut[edge.nodeTo]

            x = graph.x[fromIdx:fromIdx+3]
            l = graph.x[toIdx:toIdx+2]
            z = edge.poseMeasurement
            z_bearing = z
            x_j = l.reshape(2,1) #gæt 
            t_i = x[:2].reshape(2,1) # robot translation(x,y)
            theta_i = x[2]# robot heading

            #robot_landmark_angle
            r_l_trans = (x_j-t_i)
            #r_l_trans = np.dot(R_i.T,(x_j-t_i))
            r_l_angle = math.atan2(r_l_trans[1],r_l_trans[0])
            #print(f"Robot landmark trans is:\n{r_l_trans}\n from pose:{t_i} to {x_j}\nRobot landmark angle is:\n{r_l_angle}\n")#x_j for landmark is:\n{x_j}\nRobot translation is:\n{t_i}\n
            err_Full += np.linalg.norm((wrap2pi(wrap2pi((r_l_angle-theta_i))-z_bearing)))
            err_bearing += np.abs(wrap2pi(wrap2pi((r_l_angle-theta_i))-z_bearing))
    
    return err_Full, err_Pose, err_bearing, err_Land, err_GPS



def dynamic_covariance_scaling(chi2, phi):
    '''
    chi2: prev error
    phi: 

    '''
    chi2=np.linalg.norm(chi2)
    #print(f"chi{chi2}")
    dcs = (2*phi)/(phi+chi2)
    s_ij = min(1, dcs)

    return s_ij


def calc_error_diff_slam(graph):
    err_opt_f = []
    err_diff = []
    diff = []
    e_pose = []
    e_land = []
    e_gps = []
    
    error_before, err_pose , err_land , err_gps = compute_global_error(graph)
    #print(f"error: {error_before}")
    err_opt_f.append(error_before)
    #e_pose.setdefault(i, [])
    #e_pose[i].append(err_pose)
    e_pose.append(err_pose)
    e_land.append(err_land)
    e_gps.append(err_gps)
    return err_opt_f