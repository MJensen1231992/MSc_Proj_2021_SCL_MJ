import warnings
import copy
import sys
from utils.linearize import *
from utils.helper import *
from utils.error import *
from utils.graph_plot import *
from utils.g2o_loader import load_g2o_graph
from utils.slam_iterate import *

sys.path.append
warnings.filterwarnings('ignore')


class Graph: 
    def __init__(self, x, nodes, edges,lut, nodeTypes, descriptor):
        self.x = x
        self.nodes = nodes
        self.edges = edges
        self.lut = lut
        self.nodeTypes = nodeTypes
        self.descriptor = descriptor 

class G2O:
    def __init__(self, filename, gt,):
        
        self.graph = load_g2o_graph(filename,gt,descriptor=False)
        self.error = compute_global_error(self.graph)

if __name__ == '__main__' :

    noise = G2O('graphSLAM/data/external/M3500.g2o', gt=False)
    n_graph = noise.graph
    
    ground = G2O('graphSLAM/data/ground_truth_20220119-104232.g2o', gt=True)
    g_graph = ground.graph

    pre_graph = copy.deepcopy(n_graph)
    # poses_per_landmark(n_graph)
    # statistics_plot(n_graph)
    # landmark_ba(n_graph,g_graph,pre_graph)
    plot_ground_together_noise(n_graph,g_graph,pre_graph)
    # plot_map(n_graph,g_graph)
    _,_,_, dcs_array = linearize_solve(n_graph)
    dcs_arrayplot(dcs_array)
    dx = graph_slam_run_algorithm(n_graph, 30, g_graph, pre_graph)


    
