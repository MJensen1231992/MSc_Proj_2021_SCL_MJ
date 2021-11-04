import numpy as np
from numpy.linalg import inv
from helper import *

def iterative_global_poseerror(xi,xj,zij):

    x_i_inv_trans = inv(vec2trans(xi))
    x_j_trans = vec2trans(xj)
    z_ij_inv = inv(vec2trans(zij))

    #Error from cyrill posegraph video 34:00
    #pose_error = trans2vec(np.dot(z_ij_inv,np.dot(x_i_inv_trans,x_j_trans)))
    pose_error_reverse = trans2vec(np.dot(z_ij_inv,np.dot(x_i_inv_trans,x_j_trans)))
    #print(f"pose_error:\n{pose_error}\n pose_error_reverse:\n{pose_error_reverse}\n")
    return pose_error_reverse

def iterative_global_landerror(x,l,zij):
    
    #R_i(*x)
    x_inv_trans = inv(vec2trans(x))[:2, :2]
    landEdge = np.expand_dims(l,axis=1)
    z = np.expand_dims(zij,axis=1)
    

    R_xi = vec2trans(x)[:2,:2]
    x_j = l.reshape(2,1)
    t_i = x[:2].reshape(2,1)
    #print(f"Rotation of x_i:\n{R_xi}\n landmark x_j:\n{x_j}\n translation of robot x_i:\n{t_i}")
    
    land_error = np.dot(R_xi.T,(x_j-t_i))-z
    #land_errorree = (np.dot(x_inv_trans,landEdge) - z)
    #land_error_test = (x_inv_trans @ (landEdge-x[:2].reshape(2,1))) - z

    #print(f"land_error:\n{land_error}\n land_error_test:\n {land_error_test}\n land_error_v2(cyrill):\n {land_error_v2}\n")
    
    return land_error



def compute_global_error(graph):
    
    err_Full = 0 
    err_Land = 0
    err_Pose = 0
    err_GPS = 0

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
            err_Full += np.linalg.norm(trans2vec(np.dot(z_12_inv,np.dot(x2_trans,x1_inv_trans))))
            err_Pose += trans2vec(np.dot(z_12_inv,np.dot(x2_trans,x1_inv_trans)))
            
            
            

            
        elif edge.Type == 'L':

            fromIdx = graph.lut[edge.nodeFrom]
            toIdx = graph.lut[edge.nodeTo]

            x = graph.x[fromIdx:fromIdx+3]
            landEdge = graph.x[toIdx:toIdx+2]
            z = edge.poseMeasurement
            info_12 = edge.information
    
            x_inv_trans = inv(vec2trans(x))[:2, :2]
            landEdge = np.expand_dims(landEdge,axis=1)
            z = np.expand_dims(z,axis=1)

            #print(f"check errors, x_inv_trans = {x_inv_trans}\n landedge: {landEdge}\n meas z : {z}")
            err_Full += np.linalg.norm(np.dot(x_inv_trans,landEdge) - z)
            err_Land += (np.dot(x_inv_trans,landEdge) - z)

        elif edge.Type == 'G':

            fromIdx = graph.lut[edge.nodeFrom]
            toIdx = graph.lut[edge.nodeTo]

            x = graph.x[fromIdx:fromIdx+3]
            gpsEdge = graph.x[toIdx:toIdx+2]
            z = edge.poseMeasurement
            info_12 = edge.information
    
            x_inv_trans = inv(vec2trans(x))[:2, :2]
            gpsEdge = np.expand_dims(gpsEdge,axis=1)
            z = np.expand_dims(z,axis=1)
           
            err_Full += np.linalg.norm(np.dot(x_inv_trans,gpsEdge) - z)
            err_GPS += (np.dot(x_inv_trans,gpsEdge) - z)
    
    return err_Full, err_Pose, err_Land, err_GPS

def dynamic_covariance_scaling(chi2, phi):
    '''
    chi2: 

    '''
    chi2=np.linalg.norm(chi2)

    dcs = (2*phi)/(phi+chi2)
    s_ij = min(1, dcs)
    #print(f"Chi squared error:\n {chi2}\n")

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