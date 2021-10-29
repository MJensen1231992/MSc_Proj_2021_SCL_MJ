import numpy as np
from numpy.linalg import inv

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
    theta = np.arctan2(T[1,0],
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