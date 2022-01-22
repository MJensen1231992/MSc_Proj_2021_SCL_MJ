import numpy as np
from numpy.linalg import inv
from utils.helper import vec2trans, trans2vec, wrap2pi
import math

def compute_global_error(graph):
    
    err_Full = 0 
    err_Pose = 0
    err_Bearing = 0
    err_Land = 0
    err_GPS = 0
    err_Landnoabs = 0


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
            err_Full += np.linalg.norm(trans2vec(np.dot(z_12_inv,np.dot(x1_inv_trans, x2_trans))))
            err_Pose += trans2vec(np.abs(np.dot(z_12_inv,np.dot(x1_inv_trans,x2_trans))))   
        
        elif edge.Type =='B':

            fromIdx = graph.lut[edge.nodeFrom]
            toIdx = graph.lut[edge.nodeTo]

            x = graph.x[fromIdx:fromIdx+3]
            l = graph.x[toIdx:toIdx+2]
            z = edge.poseMeasurement
            z_bearing = z
            x_j = l.reshape(2,1)
            t_i = x[:2].reshape(2,1) # robot translation(x,y)
            theta_i = x[2]# robot heading

            #robot_landmark_angle
            r_l_trans = (x_j-t_i)
            r_l_angle = math.atan2(r_l_trans[1],r_l_trans[0])

            err_Full += np.linalg.norm((wrap2pi(wrap2pi((r_l_angle-theta_i))-z_bearing)))
            err_Bearing += np.abs(wrap2pi(wrap2pi((r_l_angle-theta_i))-z_bearing))
            # print(f"err full bearing:\n{err_Full}")
            # print(f"err bearing:\n{err_Bearing}")
           
            
        elif edge.Type == 'L':
            
            fromIdx = graph.lut[edge.nodeFrom]
            toIdx = graph.lut[edge.nodeTo]

            x_i = graph.x[fromIdx:fromIdx+3] #robot pose
            x_j = graph.x[toIdx:toIdx+2] # landmark pose

            z_ij = edge.poseMeasurement # measurement

            t_i = x_i[:2].reshape(2,1)#translation part robot translation
            x_l = x_j.reshape(2,1)#landmark pose
            z_il = z_ij.reshape(2,1)
            R_i = vec2trans(x_i)[:2, :2]

            err_Full +=  np.linalg.norm(np.dot(R_i.T, x_l-t_i) - z_il)
            err_Land +=  np.abs(np.dot(R_i.T, x_l-t_i) - z_il)
        
            
        elif edge.Type == 'G':
            
            fromIdx = graph.lut[edge.nodeFrom]
            toIdx = graph.lut[edge.nodeTo]

            x = graph.x[fromIdx:fromIdx+3]
            gpsEdge = graph.x[toIdx:toIdx+2]
            z = edge.poseMeasurement
            gpsEdge = np.expand_dims(gpsEdge,axis=1)
            R_xi = vec2trans(x)[:2, :2]

            t_i = x[:2]
            z = np.expand_dims(z,axis=1)
           
            err_Full += np.linalg.norm(np.dot(R_xi.T,(gpsEdge-t_i))-z)
            err_GPS += np.abs(np.dot(R_xi.T,(gpsEdge-t_i))-z)
        
        else:
            print('Error: Edge type not found')

    return err_Full, err_Pose, err_Bearing, err_Land, err_GPS



def dynamic_covariance_scaling(error, phi):

    chi2=np.linalg.norm(error)

    dcs = (2*phi)/(phi+chi2)
    s_ij = min(1, dcs)

    return s_ij



# def calc_error_diff_slam(graph):
#     err_opt_f = []
#     err_diff = []
#     diff = []
#     e_pose = []
#     e_land = []
#     e_gps = []
    
#     error_before, err_pose , err_land , err_gps = compute_global_error(graph)
#     #print(f"error: {error_before}")
#     err_opt_f.append(error_before)
#     #e_pose.setdefault(i, [])
#     #e_pose[i].append(err_pose)
#     e_pose.append(err_pose)
#     e_land.append(err_land)
#     e_gps.append(err_gps)
#     return err_opt_f