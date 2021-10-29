import numpy as np
from numpy.linalg import inv
from helper import *
def compute_error(graph):
    
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
            omega_12 = edge.information

            x1_inv_trans = inv(vec2trans(x1))
            x2_trans = vec2trans(x2)

            z_12_inv = inv(vec2trans(z_12))

            #Error from cyrill posegraph video 34:00
            err_Full += np.linalg.norm(trans2vec(np.dot(z_12_inv,np.dot(x2_trans,x1_inv_trans))))
            #err_Pose += np.linalg.norm(trans2vec(np.dot(z_12_inv,np.dot(x2_trans,x1_inv_trans))))
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
        
            err_Full += np.linalg.norm(np.dot(x_inv_trans,landEdge) - z)
            err_Land += np.linalg.norm(np.dot(x_inv_trans,landEdge) - z)

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
            err_GPS += np.linalg.norm(np.dot(x_inv_trans,gpsEdge) - z)
    
    return err_Full, err_Pose, err_Land, err_GPS