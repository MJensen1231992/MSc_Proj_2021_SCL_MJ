#from os import read, readlink
from os import error
import warnings
import numpy as np
from numpy.linalg import inv
from collections import namedtuple
import matplotlib.pyplot as plt
#import scipy 
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
#from scipy.optimize import least_squares
from linearize_solve import *
from helper import *
from error import *
from graph_plot import *
from g2o_loader import load_g2o_graph

warnings.filterwarnings('ignore')

class Graph: 
    def __init__(self, x, nodes, edges,lut, nodeTypes, withoutBearing=True):
        self.x = x
        self.nodes = nodes
        self.edges = edges
        self.lut = lut
        self.withoutBearing = withoutBearing
        self.nodeTypes = nodeTypes

#function to go from and to vector representation and homogeneous transformation

class G2O:
    def __init__(self, filename):
        
        self.graph = load_g2o_graph(filename,noBearing=True)
        self.error = compute_global_error(self.graph)


def graph_slam_run_algorithm(graph, numIter):

    tol = 1e-4 # If error difference is smaller than tolerance it breaks.
    norm_dX_all = []
    
    err_opt_f = []
    err_diff = []
    diff = []
    e_pose = []
    e_land = []
    e_gps = []
    graph_plot(graph, animate=True,landmarkEdgesPlot=True)

    for i in range(numIter):

        error_before, err_pose , err_land , err_gps = compute_global_error(graph)
        #print(f"error: {error_before}")
        err_opt_f.append(error_before)
        #e_pose.setdefault(i, [])
        #e_pose[i].append(err_pose)
        e_pose.append(err_pose)
        e_land.append(err_land)
        e_gps.append(err_gps)

        if i>0:
            err_diff = err_opt_f[i-1]-err_opt_f[i]
            print(f"error diff : {err_diff}")
            if err_diff < 0:
                print("Error is larger than previous iteration")

        diff = np.append(diff,err_diff)
        
        dX, _, _, _ = linearize_solve(graph, needToAddPrior=True)
        
        graph.x += dX

        graph_plot(graph, animate=True)
        
        norm_dX = np.linalg.norm(dX)

        print(f"|dx| for step {i} : {norm_dX}\n")
        norm_dX_all.append(norm_dX)
        # error_full, _ , _ = compute_error(graph)
        # print(f"error after = {error_full}")
        if i >=1 and np.abs(norm_dX_all[i]-norm_dX_all[i-1]) < tol: 
            break

    #print(f"error diff array: {diff}")

    plot_errors(e_pose,e_land,e_gps)
    return norm_dX_all

if __name__ == '__main__' :
    
    file = 'graphSLAM/data/dlr.g2o'
    
    # files = ['graphSLAM/data/noise.g2o','graphSLAM/data/sim_pose_landmark.g2o','graphSLAM/data/dlr.g2o']
    # for file in files:
    #      graph = load_g2o_graph(file)
    #      graph_plot(graph)
    #      dx_squeezed, H_sparse = linearize_solve(graph, needToAddPrior=True) 
    #      print(H_sparse)
    #      plt.spy(H_sparse)
    #      plt.show()

    # graph = load_g2o_graph(file)

    # graph_plot(graph)#, landmarkEdgesPlot=True, gpsEdgesPlot=True)
    # squeezed, H_s, H_norm, sparse_dx = linearize_solve(graph, needToAddPrior=True) 
    #err_Full, err_Pose, err_Land,_ = compute_error(graph)
    #plt.spy(H_s)
    #plt.show()
    
    noise = G2O('graphSLAM/data/noise_20211101-154600.g2o')
    ground = G2O('graphSLAM/data/ground_truth_20211101-154600.g2o')
    
    n_graph = noise.graph
    g_graph = ground.graph
    
    plot_ground_together_noise(g_graph,n_graph)
    
    
   # plt.show()
    
    # kerror,_,_,_ = SPL_SLAM.error
    # print(kerror)
    dx = graph_slam_run_algorithm(n_graph,10)