import warnings
import copy
import sys
sys.path.append('graphSLAM')
from utils.linearize import *
from utils.helper import *
from utils.error import *
from utils.graph_plot import *
from utils.g2o_loader import load_g2o_graph
from utils.slam_iterate import *


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
    def __init__(self, filename, gt):
        
        self.graph = load_g2o_graph(filename,gt,descriptor=False)
        self.error = compute_global_error(self.graph)

if __name__ == '__main__' :

   
    # noise = G2O('graphSLAM/newdata/noise_20220128-102337.g2o', gt=False)#120 10 outlier
    # noise = G2O('graphSLAM/newdata/noise_20220128-102519.g2o', gt=False)#120 50 outlier
    # noise = G2O('graphSLAM/newdata/noise_20220128-102748.g2o', gt=False)#120 100 outlier
    # noise = G2O('graphSLAM/newdata/noise_20220128-122414.g2o', gt=False)#120 1000outlier
    # noise = G2O('graphSLAM/newdata/noise_20220128-103719.g2o', gt=False)#120secondv
    # noise = G2O('graphSLAM/newdata/noise_20220127-192419.g2o', gt=False)#120

    noise = G2O('graphSLAM/data/external/victoriaPark.g2o', gt=False)#120
    n_graph = noise.graph
  
    # ground = G2O('graphSLAM/newdata/ground_truth_20220127-192419.g2o', gt=True)#120
    # ground = G2O('graphSLAM/newdata/ground_truth_20220128-102337.g2o', gt=True)#120 10 outlier
    # ground = G2O('graphSLAM/newdata/ground_truth_20220128-102519.g2o', gt=True)#120 50 outlier
    # ground = G2O('graphSLAM/newdata/ground_truth_20220128-102748.g2o', gt=True)#120 100 outlier
    # ground = G2O('graphSLAM/newdata/ground_truth_20220128-122414.g2o', gt=True)#120 1000 outlier
    # ground = G2O('graphSLAM/newdata/ground_truth_20220127-204733.g2o', gt=True)#120n2
    ground = G2O('graphSLAM/data/external/M3500a.g2o', gt=True)#120
    g_graph = ground.graph

    pre_graph = copy.deepcopy(n_graph)
    # poses_per_landmark(pre_graph, pre = True)
    # plot_map(n_graph, g_graph, post=False)
    # plot_ground_together_noise(n_graph,g_graph,pre_graph)
    dx = graph_slam_run_algorithm(n_graph,25, g_graph, pre_graph)

#route 1 no out
  # noise = G2O('graphSLAM/newdata/noise_20220127-192419.g2o', gt=False)#120
    # noise = G2O('graphSLAM/newdata/noise_20220127-193438.g2o', gt=False)#30
    # noise = G2O('graphSLAM/newdata/noise_20220127-193543.g2o', gt=False)#60
    # noise = G2O('graphSLAM/newdata/noise_20220127-193634.g2o', gt=False)#90
    # noise = G2O('graphSLAM/newdata/noise_20220127-193734.g2o', gt=False)#150'
    # noise = G2O('graphSLAM/newdata/noise_20220127-200956.g2o', gt=False)#180
    
    

    # ground = G2O('graphSLAM/newdata/ground_truth_20220127-192419.g2o', gt=True)#120
    # ground = G2O('graphSLAM/newdata/ground_truth_20220127-193438.g2o', gt=True)#30
    # ground = G2O('graphSLAM/newdata/ground_truth_20220127-193543.g2o', gt=True)#60
    # ground = G2O('graphSLAM/newdata/ground_truth_20220127-193634.g2o', gt=True)#90
    # ground = G2O('graphSLAM/newdata/ground_truth_20220127-193734.g2o', gt=True)#150
    # ground = G2O('graphSLAM/newdata/ground_truth_20220127-200956.g2o', gt=True)#180

#route 2 no out

 # noise = G2O('graphSLAM/newdata/noise_20220127-204733.g2o', gt=False)#120n2
    # noise = G2O('graphSLAM/newdata/noise_20220127-224957.g2o', gt=False)#30n2
    # noise = G2O('graphSLAM/newdata/noise_20220127-225117.g2o', gt=False)#60n2
    # noise = G2O('graphSLAM/newdata/noise_20220127-224841.g2o', gt=False)#90n2
    # noise = G2O('graphSLAM/newdata/noise_20220127-225410.g2o', gt=False)#120secondv
    # noise = G2O('graphSLAM/newdata/noise_20220127-225210.g2o', gt=False)#150n2
    # noise = G2O('graphSLAM/newdata/noise_20220127-225304.g2o', gt=False)#180n2
    
     # ground = G2O('graphSLAM/newdata/ground_truth_20220127-204733.g2o', gt=True)#120n2
    # ground = G2O('graphSLAM/newdata/ground_truth_20220127-224957.g2o', gt=True)#30n2
    # ground = G2O('graphSLAM/newdata/ground_truth_20220127-225117.g2o', gt=True)#60n2
    # ground = G2O('graphSLAM/newdata/ground_truth_20220127-224841.g2o', gt=True)#90n2
    # ground = G2O('graphSLAM/newdata/ground_truth_20220127-225410.g2o', gt=True)#120n2v2
    # ground = G2O('graphSLAM/newdata/ground_truth_20220127-225210.g2o', gt=True)#150n2
    # ground = G2O('graphSLAM/newdata/ground_truth_20220127-225304.g2o', gt=True)#180n2
    # ground = G2O('graphSLAM/newdata/ground_truth_20220127-225644.g2o', gt=True)#120n2