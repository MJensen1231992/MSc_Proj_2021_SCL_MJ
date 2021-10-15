from os import read, readlink
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import squeeze
from numpy.linalg import cholesky, inv, norm
from numpy import diff, float64
import scipy 
from scipy.sparse import csr_matrix
from scipy.sparse.csc import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import lsqr
# from sksparse.cholmod import cholesky
# from scikits.sparse.cholmod import cholesky
from scipy.optimize import least_squares

class Graph: 
    def __init__(self, x, nodes, edges,lut):
        self.x = x
        self.nodes = nodes
        self.edges = edges
        self.lut = lut

def load_g2o_graph(filename):
    
    Edge = namedtuple(
        'Edge', ['Type', 'nodeFrom', 'nodeTo', 'poseMeasurement', 'information' ] # g2o format of files.
    )
    
    edges = []
    nodes = {}
    with open(filename, 'r') as file:
        for line in file:
            data = line.split() # splits the columns

            if data[0] == 'VERTEX_SE2':
                nodeId = int(data[1])
                pose = np.array(data[2:5],dtype=np.float64)
                nodes[nodeId] = pose

            elif data[0] == 'VERTEX_XY':
                nodeId = int(data[1])
                landmark = np.array(data[2:4],dtype=np.float64)
                nodes[nodeId] = landmark

            elif data[0] == 'EDGE_SE2':
                Type = 'P' # pose type
                nodeFrom = int(data[1])
                nodeTo = int(data[2])
                poseMeasurement = np.array(data[3:6], dtype=np.float64)
                upperTriangle = np.array(data[6:12], dtype=np.float64)
                information = np.array([[upperTriangle[0], upperTriangle[1], upperTriangle[2]],
                                        [upperTriangle[1], upperTriangle[3], upperTriangle[4]],
                                        [upperTriangle[2], upperTriangle[4], upperTriangle[5]]])# upper tri
                edge = Edge(Type, nodeFrom,nodeTo, poseMeasurement, information)
                edges.append(edge)

            elif data[0] == 'EDGE_SE2_XY':
                Type = 'L' #landmark type
                nodeFrom = int(data[1])
                nodeTo = int(data[2])
                poseMeasurement = np.array(data[3:5],dtype=np.float64)
                upperTriangle = np.array(data[5:8],dtype=np.float64)
                information = np.array([[upperTriangle[0], upperTriangle[1]],
                                        [upperTriangle[1], upperTriangle[2]]])
                edge = Edge(Type, nodeFrom,nodeTo, poseMeasurement, information)
                edges.append(edge)
            else: 
                print("error, edge/vertex not defined")
    
    lut = {}
    x = []
    offset = 0
    for nodeId in nodes:
        lut.update({nodeId: offset})
        offset = offset + len(nodes[nodeId])
        x.append(nodes[nodeId])
    x = np.concatenate(x, axis=0)

    # collect nodes, edges and lookup in graph structure
    graph = Graph(x, nodes, edges, lut)
    print('Loaded graph with {} nodes and {} edges'.format(
        len(graph.nodes), len(graph.edges)))
    
    return graph

def vec2trans(pose):

    c = np.cos([pose[2]])
    s = np.sin([pose[2]])
    T_mat = np.array([[c, -s, pose[0]], [s,c, pose[1]], [0,0,1]],dtype=np.float64)
    return T_mat

def trans2vec(T):

    x = T[0,2]
    y = T[1,2]
    theta = np.arctan2(T[1,0],T[0,0])

    vec = np.array([x,y,theta],dtype=np.float64)
    return vec

def graph_plot(graph, animate = False, poseEdgesPlot = True, landmarkEdgesPlot = False):

    #init plt figure
    plt.figure(1)
    plt.clf()

    poses, landmarks = get_poses_landmarks(graph)
    
    
    #plot poses and landmarks if exits
    if len(poses) > 0:
        poses = np.stack(poses, axis=0) # axis = 0 turns into integers/slices and not tuple
        plt.plot(poses[:,0], poses[:,1], 'bo')
    
    
    if len(landmarks) > 0:
        landmarks = np.stack(landmarks, axis=0)
        plt.plot(landmarks[:,0], landmarks[:,1], 'r*')
        

    poseEdgesFrom = []
    poseEdgesTo = []

    landmarkEdgesFrom = []
    landmarkEdgesTo = []

    for edge in graph.edges:
        fromIdx = graph.lut[edge.nodeFrom]
        toIdx = graph.lut[edge.nodeTo]

        if edge.Type == 'P':
            poseEdgesFrom.append(graph.x[fromIdx:fromIdx+3])
            poseEdgesTo.append(graph.x[toIdx:toIdx+3])

        elif edge.Type == 'L':
            landmarkEdgesFrom.append(graph.x[fromIdx:fromIdx+3])
            landmarkEdgesTo.append(graph.x[toIdx:toIdx+2])

    if len(poses) > 0:
        poseEdgesFrom = np.stack(poseEdgesFrom, axis = 0)
        poseEdgesTo = np.stack(poseEdgesTo, axis = 0)
        
        poseZip = zip(poseEdgesFrom, poseEdgesTo)
        poseEdges = np.vstack(poseZip)

        poseEdgesX = poseEdges[:,0]
        poseEdgesY = poseEdges[:,1]

        poseEdgeX_corr = np.vstack([poseEdgesX[0::2], poseEdgesX[1::2]])
        poseEdgeY_corr = np.vstack([poseEdgesY[0::2], poseEdgesY[1::2]])

        if poseEdgesPlot == True:
            plt.plot(poseEdgeX_corr,poseEdgeY_corr,'r--',label = 'poseEdges')
    
    if len(landmarks) > 0:
        landmarkEdgesFrom = np.stack(landmarkEdgesFrom, axis = 0)
        landmarkEdgesTo = np.stack(landmarkEdgesTo, axis = 0)

        # Zip landmark_from(x,y) with corresponding landmark_to(x,y)
        landmarkZip = zip(landmarkEdgesFrom[:,0:2], landmarkEdgesTo)
        
        landmarkEdges = np.vstack(landmarkZip)
        
        landmarkEdgesX = landmarkEdges[:,0]
        landmarkEdgesY = landmarkEdges[:,1]

        # # Use every 2nd x and y coordinate so correct correlation
        landmarkEdgeX_corr = np.vstack([landmarkEdgesX[0::2], landmarkEdgesX[1::2]])
        landmarkEdgeY_corr = np.vstack([landmarkEdgesY[0::2], landmarkEdgesY[1::2]])

        if landmarkEdgesPlot == True:
            plt.plot(landmarkEdgeX_corr,landmarkEdgeY_corr,'g--', label = 'landEdges')

    
    if animate == True:
        plt.draw()
        plt.pause(1)
    else:
        plt.show()
    
    return



def get_poses_landmarks(graph):
    poses = []
    landmarks = []


    for nodeId in graph.nodes:
        dim = len(graph.nodes[nodeId])
        offset = graph.lut[nodeId] # checking whether 2 or 3 next lines are needed. if pose or landmark

        if dim == 3:
            pose = graph.x[offset:offset+3]
            poses.append(pose)
            
        if dim == 2:
            landmark = graph.x[offset:offset+2]
            landmarks.append(landmark)
    
    return poses, landmarks


def compute_error(graph):
    
    err_Full = 0 
    err_Land = 0
    err_Pose = 0


    for i, edge in enumerate(graph.edges):

        
            
        if edge.Type == 'P':
            
            fromIdx = graph.lut[edge.nodeFrom]
            toIdx = graph.lut[edge.nodeTo]
            
            x1 = graph.x[fromIdx:fromIdx+3]
            x2 = graph.x[toIdx:toIdx+3]

            z_12 = edge.poseMeasurement
            omega_12 = edge.information

            x1_inv_trans = inv(vec2trans(x1))#[:2, :2] 
            x2_trans = vec2trans(x2)#[:2, :2]

            z_12_inv = inv(vec2trans(z_12))#[:2, :2]
            
            # print(f" inverse pose from {x1_inv_trans}")
            # print(f" pose to {x2_trans}")
            # print(f" Z pose measurement inverse {z_12_inv}")

            #err_Full += np.linalg.norm(trans2vec(np.dot(np.dot(z_12_inv,x1_inv_trans), x2_trans)))
            #err_Pose += np.linalg.norm(trans2vec(np.dot(np.dot(z_12_inv,x1_inv_trans), x2_trans)))
            err_Full += np.linalg.norm(trans2vec(np.dot(z_12_inv,np.dot(x2_trans,x1_inv_trans))))
            err_Pose += np.linalg.norm(trans2vec(np.dot(z_12_inv,np.dot(x2_trans,x1_inv_trans))))

        elif edge.Type == 'L':

            fromIdx = graph.lut[edge.nodeFrom]
            toIdx = graph.lut[edge.nodeTo]

            x = graph.x[fromIdx:fromIdx+3]
            landEdge = graph.x[toIdx:toIdx+2]

            z = edge.poseMeasurement
            info_12 = edge.information

            #aligning shapes
            x_inv_trans = inv(vec2trans(x))[:2, :2]
            landEdge = np.expand_dims(landEdge,axis=1)
            z = np.expand_dims(z,axis=1)

            # print(f" inverse poseland from {x_inv_trans}")
            # print(f" landedgeTo {landEdge}")
            # print(f" Z land measurement inverse {z}")
            
            err_Full += np.linalg.norm(np.dot(x_inv_trans,landEdge)-z)
            err_Land += np.linalg.norm(np.dot(x_inv_trans,landEdge)-z)

        

    return err_Full, err_Pose, err_Land

            


def information_matrix(graph):

    #initialization of information matrix H and Bs
    # b is e.T*Omega*J
    # H is J.T*Omega*J

    H = np.zeros((len(graph.x), len(graph.x)))
    b = np.zeros(len(graph.x))
    b = np.expand_dims(b, axis=1)

    return H,b 

def from_to_idx(graph):
    edge = 0
    fromIdx = graph.lut[edge.nodeFrom]
    toIdx = graph.lut[edge.nodeTo]

    return fromIdx, toIdx

def linearize_solve(graph, needToAddPrior=False):

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

            #contributions to pose pose H and b

            b_i = np.dot(np.dot(A.T,omega_ij), error).reshape(3,1)
            b_j = np.dot(np.dot(B.T,omega_ij), error).reshape(3,1)
            H_ii = np.dot(np.dot(A.T,omega_ij), A) 
            H_ij = np.dot(np.dot(A.T,omega_ij), B) 
            H_ji = np.dot(np.dot(B.T,omega_ij), A) 
            H_jj = np.dot(np.dot(B.T,omega_ij), B) 

            #adding them to H and b in respective places

            H[fromIdx:fromIdx+3, fromIdx:fromIdx+3] += H_ii
            H[fromIdx:fromIdx+3, toIdx:toIdx+3] += H_ij
            H[toIdx:toIdx+3, fromIdx:fromIdx+3] += H_ji
            H[toIdx:toIdx+3, toIdx:toIdx+3] += H_jj

            b[fromIdx:fromIdx+3] += b_i
            b[toIdx:toIdx+3] += b_j

            #May need to add prior to fix initial location!
            if needToAddPrior:
                H[fromIdx:fromIdx + 3, fromIdx:fromIdx + 3] = H[fromIdx:fromIdx + 3, fromIdx:fromIdx + 3] + 1000 * np.eye(3)
                needToAddPrior = False

        elif edge.Type == 'L':

            fromIdx = graph.lut[edge.nodeFrom]
            toIdx = graph.lut[edge.nodeTo]

            x_i = graph.x[fromIdx:fromIdx+3]
            l = graph.x[toIdx:toIdx+2]
            z_ij = edge.poseMeasurement
            omega_ij = edge.information

            error, A, B = pose_landmark_constraints(x_i, l, z_ij)

            #contributions to pose pose H and b

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
            
          
    
    H_sparse = csr_matrix(H)
    
    
    sparse_dxstar = spsolve(H_sparse,-b)
    dxstar_squeeze = sparse_dxstar.squeeze()
    
    return dxstar_squeeze
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
    #e_xy = np.dot(R_ij.T,np.dot(R_i.T, t_j-t_i)-np.dot(R_ij.T, t_ij))

    e_ang = theta_j - theta_i - theta_ij 

    e_full = np.vstack((e_xy,e_ang))

    #Get Jacobians Aij and Bij, diff on errors respective to xi and xj

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

def graph_slam_run_algorithm(graph, numIter):

    tol = 1e-4 # If error difference is smaller than tolerance it breaks.
    norm_dX_all = []
    
    for i in range(numIter):

        dX = linearize_solve(graph, needToAddPrior=True)
        #print(f"dx pre is {dX}\n")

        graph.x += dX

        graph_plot(graph, animate=True)
        
        norm_dX = np.linalg.norm(dX)

        print(f"|dx| for step {i} : {norm_dX}\n")
        norm_dX_all.append(norm_dX)

        #err_opt , err_opt_pose, err_opt_land = compute_error(graph)
        #print(f"error after is = {err_opt}")
        #print(f"Full error is {err_opt}\nPose error is {err_opt_pose} \nLandmark error is {err_opt_land}")

        if i >=1 and np.abs(norm_dX_all[i]-norm_dX_all[i-1]) < tol:
            
            break
    #err_opt , err_opt_pose, err_opt_land = compute_error(graph)        
    return norm_dX_all


def sparsity(graph):
    pass

def optimize_lsq(graph):
    pass


if __name__ == '__main__' :
    
    file = 'graphSLAM/data/INTEL.g2o'
    

    # files = ['graphSLAM/data/noise.g2o','graphSLAM/data/sim_pose_landmark.g2o','graphSLAM/data/dlr.g2o']
    # for file in files:
    #      graph = load_g2o_graph(file)
    #      graph_plot(graph)
    #      dx_squeezed, H_sparse = linearize_solve(graph, needToAddPrior=True) 
    #      print(H_sparse)
    #      plt.spy(H_sparse)
    #      plt.show()

    graph = load_g2o_graph(file)

    graph_plot(graph, landmarkEdgesPlot=True)
    #squeezed = linearize_solve(graph, needToAddPrior=True) 
    
    # #print(squeezed)
    # err_Full, err_Pose, err_Land = compute_error(graph)
    # print(f"Full error is {err_Full}\nPose error is {err_Pose} \nLandmark error is {err_Land}")

    
    #dx = graph_slam_run_algorithm(graph,100)