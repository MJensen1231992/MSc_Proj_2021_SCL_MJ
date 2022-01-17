import warnings
import copy

from utils.linearize import *
from utils.helper import *
from utils.error import *
from utils.graph_plot import *
from utils.g2o_loader import load_g2o_graph
from utils.slam_iterate import *
import sys
sys.path.append
warnings.filterwarnings('ignore')



class Graph: 
    def __init__(self, x, nodes, edges,lut, nodeTypes, withoutBearing=False):
        self.x = x
        self.nodes = nodes
        self.edges = edges
        self.lut = lut
        self.withoutBearing = withoutBearing
        self.nodeTypes = nodeTypes

class G2O:
    def __init__(self, filename, gt):
        
        self.graph = load_g2o_graph(filename,gt, noBearing=False)
        self.error = compute_global_error(self.graph)
        


if __name__ == '__main__' :

    noise = G2O('graphSLAM/data/noise_20220117-125333.g2o', gt=False)
    n_graph = noise.graph
    
    
    ground = G2O('graphSLAM/data/ground_truth_20220117-125333.g2o', gt=True)
    g_graph = ground.graph

    pre_graph = copy.deepcopy(n_graph)
    # poses_per_landmark(n_graph)
    # statistics_plot(n_graph)
    # landmark_ba(n_graph,g_graph,pre_graph)
    # print(n_graph.nodes.items())
    # plot_ground_together_noise(n_graph,g_graph,pre_graph)
    # plot_map(n_graph,g_graph)

    # _, Hs , H, _ = linearize_solve(n_graph)
    # plt.spy(Hs)
    # plt.show()
    # hnul = np.nonzero(H)
    # print(np.shape(hnul))
    dx = graph_slam_run_algorithm(n_graph,30, g_graph, pre_graph)


    
