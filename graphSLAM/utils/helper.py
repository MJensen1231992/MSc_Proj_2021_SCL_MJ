import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def get_poses_landmarks(graph):

    poses = []
    landmarks = []
    gps = []
    
    for nodeId in graph.nodes:
        
        offset = graph.lut[nodeId] # checking whether 2 or 3 next lines are needed. if pose or landmark
        
        if graph.nodeTypes[nodeId] == 'VSE2':
            pose = graph.x[offset:offset+3]
            poses.append(pose)
            
        if graph.nodeTypes[nodeId] == 'VXY':
            landmark = graph.x[offset:offset+2]
            landmarks.append(landmark)

        if graph.nodeTypes[nodeId] == 'VGPS':
            gp = graph.x[offset:offset+2]
            gps.append(gp)

    return poses, landmarks, gps
     
    
def vec2trans(pose):

    c = np.cos([pose[2]])
    s = np.sin([pose[2]])
    T_mat = np.array([[c, -s, pose[0]],
                      [s, c, pose[1]],
                      [0,0,1]],dtype=np.float64)

    return T_mat

def trans2vec(T):

    x = T[0,2]
    y = T[1,2]
    theta = math.atan2(T[1,0],
                       T[0,0])
    vec = np.array([x,y,theta],dtype=np.float64)

    return vec

def is_pos_def(infoH):
    if np.allclose(infoH, infoH.T):
        try:
            np.linalg.cholesky(infoH)
            print("Matrix is positive definite")
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

def from_uppertri_to_full(arr, n):
    
    triu0 = np.triu_indices(n,0)
    triu1 = np.triu_indices(n,1)
    tril1 = np.tril_indices(n,-1)

    mat = np.zeros((n,n), dtype=np.float64)

    mat[triu0] = arr[:]
    mat[tril1] = mat[triu1]

    return mat

def vision_length_check(graph):
    
    ks= []
    for edge in graph.edges:
        if edge.Type == 'L':
            x = (np.abs(edge.poseMeasurement[0]))
            y = (np.abs(edge.poseMeasurement[1]))
            ks.append(np.sqrt([x**2+y**2]))
    print(max(ks))

def wrap2pi(angle):

    if angle > np.pi:
        angle = angle-2*np.pi
        
    elif angle < -np.pi:
        angle = angle + 2*np.pi
    
    return angle

def RMSE(predicted, actual):
    return np.square(np.subtract(actual,predicted)).mean() 


def calc_gradient_hessian(A,B,information,error, edgetype: str):
    
    if edgetype == 'P':
        b_i = np.dot(np.dot(A.T,information), error)
        b_j = np.dot(np.dot(B.T,information), error)
        
    else:
        
        b_i = np.dot(np.dot(A.T,information), error)
        b_j = np.dot(np.dot(B.T,information), error)


    H_ii = np.dot(np.dot(A.T,information), A) 
    H_ij = np.dot(np.dot(A.T,information), B) 
    H_ji = np.dot(np.dot(B.T,information), A) 
    H_jj = np.dot(np.dot(B.T,information), B)  
    
    return b_i, b_j, H_ii, H_ij, H_ji, H_jj

def build_gradient_hessian(b_i, b_j, H_ii, H_ij, H_ji, H_jj,H,b,fromIdx,toIdx, edgetype: str):
    
    if edgetype=='P':

        H[fromIdx:fromIdx+3, fromIdx:fromIdx+3] += H_ii
        H[fromIdx:fromIdx+3, toIdx:toIdx+3] += H_ij
        H[toIdx:toIdx+3, fromIdx:fromIdx+3] += H_ji
        H[toIdx:toIdx+3, toIdx:toIdx+3] += H_jj

        b[fromIdx:fromIdx+3] += b_i
        b[toIdx:toIdx+3] += b_j

    else:

        H[fromIdx:fromIdx+3, fromIdx:fromIdx+3] += H_ii
        H[fromIdx:fromIdx+3, toIdx:toIdx+2] += H_ij
        H[toIdx:toIdx+2, fromIdx:fromIdx+3] += H_ji
        H[toIdx:toIdx+2, toIdx:toIdx+2] += H_jj
        
        b[fromIdx:fromIdx+3] += b_i
        b[toIdx:toIdx+2] += b_j

    return H, b

def printPrincipalDiagonal(mat, n):
    print("Principal Diagonal: ", end = "")
 
    for i in range(n):
        for j in range(n):
 
            # Condition for principal diagonal
            if (i == j):
                print(mat[i][j], end = ", ")
    print()

def colors():
    CB91_Blue = "#2CBDFE"
    CB91_Green = "#47DBCD"
    CB91_Pink = "#F3A0F2"
    CB91_Purple = "#9D2EC5"
    CB91_Violet = "#661D98"
    CB91_Amber = "#F5B14C"
    color_list = [
        CB91_Blue,
        CB91_Pink,
        CB91_Green,
        CB91_Amber,
        CB91_Purple,
        CB91_Violet,
    ]

    return color_list